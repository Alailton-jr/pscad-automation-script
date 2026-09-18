# import numpy as np
from time import perf_counter
import pandas as pd
from pathlib import Path
import re
import mhi.pscad
import numpy as np
from tqdm import tqdm
import shutil

class Config:
    # --- Project / PSCAD configuration ---
    SEARCH_PATH = False

    # Name of the *base* project (original)
    PROJECT_NAME = 'Base'
    PROJECT_PATH = r''

    # List of PSCAD projects that will be run in parallel.
    # Each must be an identical copy of the base model and open in PSCAD.
    # Example: ['Base', 'Base_2', 'Base_3', 'Base_4']
    N_PROJECTS = 5
    PROJECT_NAMES = [f'Base{x+1}' for x in range(5)]

    CANVAS = 'Main'
    COMPILER = 'gf46'

    # --- Output / database configuration ---
    OUTPUT_DIR = None               # kept for backward compatibility (base project)
    SAVES_OUTPUT = None
    OUTPUT_FORMAT = 'parquet'       # or 'pkl', 'parquet', 'mat', 'npy'
    OUTPUT_FORMAT_COLUMNS = True
    RECORDING_START_TIME = 1.0      # seconds

    # --- Monte Carlo sweep control ---
    CASES_PER_BATCH = 50            # "waves" per batch (each wave runs len(PROJECT_NAMES) cases in parallel)
    MAX_BATCHES = 40                # maximum number of batches
    SIM_SET_NAME = "MCSet"          # PSCAD simulation set name

    # --- Database chunking configuration ---
    DB_MAX_CHUNK_MB = 500           # target max size per parquet chunk
    DB_BASENAME = 'simulation_database'


# You can adjust N_PROJECTS here if you want more/less workers
Config.PROJECT_NAMES = [f'Base{x+1}' for x in range(8)]

# Ensure we have copies of the base project file for each worker
for project in Config.PROJECT_NAMES:
    if not Path(project + '.pscx').exists():
        shutil.copy(Config.PROJECT_NAME + '.pscx', project + '.pscx')

# --- Optional: GUI-based project selection ---
if Config.SEARCH_PATH:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select PSCAD Project File",
        filetypes=[("PSCAD Project Files", "*.pscx")],
        initialdir='.'
    )
    if file_path:
        file_path = Path(file_path).resolve()
        Config.PROJECT_PATH = file_path.parent
        Config.PROJECT_NAME = file_path.stem
        Config.CANVAS = 'Main'
        # By default, use only the selected project
        Config.PROJECT_NAMES = [Config.PROJECT_NAME]

# Normalize path / name
if isinstance(Config.PROJECT_PATH, str):
    Config.PROJECT_PATH = Path(Config.PROJECT_PATH)

if '.' in Config.PROJECT_NAME:
    Config.PROJECT_NAME = Config.PROJECT_NAME.split('.')[0]

# Base project output / save directories
Config.OUTPUT_DIR = Config.PROJECT_PATH / f'{Config.PROJECT_NAME}.{Config.COMPILER}'
Config.SAVES_OUTPUT = Config.PROJECT_PATH / f'{Config.PROJECT_NAME}_saves'
Config.SAVES_OUTPUT.mkdir(parents=True, exist_ok=True)

# --- Global buffer for chunked saving ---
DB_BUFFER = []          # list of per-case DataFrames
DB_BUFFER_BYTES = 0     # approximate size in bytes of buffered rows


def _get_next_chunk_index() -> int:
    """
    Find the next chunk index based on existing files
    simulation_database_XXXXX.parquet in Config.SAVES_OUTPUT.
    """
    pattern = f'{Config.DB_BASENAME}_*.parquet'
    existing = list(Config.SAVES_OUTPUT.glob(pattern))
    if not existing:
        return 1

    max_idx = 0
    for p in existing:
        m = re.search(r'_(\d+)\.parquet$', p.name)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1


def flush_db_buffer():
    """
    Flush the in-memory buffer to a new parquet file chunk.
    Creates a new file every time it's called (no appending).
    """
    global DB_BUFFER, DB_BUFFER_BYTES
    if not DB_BUFFER:
        return

    big_df = pd.concat(DB_BUFFER, ignore_index=True)
    chunk_idx = _get_next_chunk_index()
    db_path = Config.SAVES_OUTPUT / f'{Config.DB_BASENAME}_{chunk_idx:05d}.parquet'
    big_df.to_parquet(db_path, index=False)

    # Reset buffer
    DB_BUFFER = []
    DB_BUFFER_BYTES = 0


def append_row_to_db(row_df: pd.DataFrame):
    """
    Append a single-row DataFrame to the in-memory buffer.
    When the buffer reaches the configured size (in MB),
    flush it to disk as a new parquet chunk file.
    """
    global DB_BUFFER, DB_BUFFER_BYTES

    DB_BUFFER.append(row_df)

    # Approximate memory footprint of this row
    row_bytes = int(row_df.memory_usage(deep=True).sum())
    DB_BUFFER_BYTES += row_bytes

    max_bytes = int(Config.DB_MAX_CHUNK_MB * 1024 * 1024)
    if DB_BUFFER_BYTES >= max_bytes:
        flush_db_buffer()


def get_output_dir_for_project(project_name: str) -> Path:
    """Return the PSCAD output directory for a given project."""
    return Config.PROJECT_PATH / f'{project_name}.{Config.COMPILER}'


def get_all_sim_names(path: Path):
    save_info = pd.read_csv(path, sep=r'\s+', header=None)
    save_names = []
    for idx in range(len(save_info)):
        column_num = save_info.iloc[idx].str.contains('Desc').idxmax()
        name = save_info.iloc[idx][column_num]
        name = re.findall(r'"(.*?)"', name)[0]
        save_names.append(name)
    return save_names


def save_data(project_name, sim_names, save_names, idx_case, case_params):
    """
    Save simulation data for a *specific project* to a chunked parquet database.

    - project_name: PSCAD project whose .out files we read
    - sim_names: mapping {variable_desc -> index} from .inf
    - save_names: list of variable descriptions to save
    - idx_case: case identifier (e.g. 'Case_0')
    - case_params: dict of Monte Carlo parameters
    """
    output_array = []
    time_data = None

    output_dir = get_output_dir_for_project(project_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pick up .out files for THIS project only
    output_files = list(output_dir.glob(f'{project_name}*.out'))
    if not output_files:
        raise FileNotFoundError(
            f'No .out files found for project "{project_name}" in "{output_dir}". '
            'Ensure the project is compiled and simulation ran successfully.'
        )

    if len(sim_names) > 10:
        # Multiple .out files with up to 10 channels each (time + 9 variables)
        output_data_list = [
            pd.read_csv(file, sep=r'\s+', header=None).to_numpy()
            for file in output_files
        ]
        time_data = output_data_list[0][:, 0]
        for name in save_names:
            pos = sim_names[name]
            idx_file = pos // 10
            idx_col = (pos % 10) + 1  # +1 for the time column at index 0
            output_array.append(output_data_list[idx_file][:, idx_col])
    else:
        # Single .out file
        output_file = output_files[0]
        output_data = pd.read_csv(output_file, sep=r'\s+', header=None).to_numpy()
        time_data = output_data[:, 0]
        for name in save_names:
            pos = sim_names[name] + 1  # +1 for time column
            output_array.append(output_data[:, pos])

    output_array = np.array(output_array).T

    # Filter to times >= RECORDING_START_TIME
    time_mask = time_data >= Config.RECORDING_START_TIME
    output_array = output_array[time_mask, :]

    # Build row dictionary
    row_data = {
        'project_name': project_name,
        'case_id': idx_case,
    }

    # Case parameters
    for param_name, param_value in case_params.items():
        row_data[f'case_{param_name}'] = param_value

    # Simulation results (store series as lists)
    for i, var_name in enumerate(save_names):
        row_data[f'result_{var_name}'] = [output_array[:, i].tolist()]

    new_row_df = pd.DataFrame(row_data)

    # --- NEW: chunked saving logic (no re-reading big file) ---
    append_row_to_db(new_row_df)


def load_cases_csv_antigo(path):
    """
    Load cases from a CSV file.
    The CSV file should have columns for each parameter.
    """
    cases_df = pd.read_csv(path, sep=',', header=None)
    cases = {}
    for idx, row in cases_df.iterrows():
        for i in range(0, len(row), 2):
            param_name = row[i]
            param_value = row[i + 1]
            if param_name not in cases:
                cases[param_name] = []
            cases[param_name].append(param_value)
    return pd.DataFrame(cases)


def load_variables_to_save_antigo(path):
    """
    Load variables to save from a CSV file.
    The CSV file should have columns for each variable to save.
    """
    variables_df = pd.read_csv(path, sep=',', header=None)
    variables = variables_df.iloc[:, 0].tolist()
    return variables


def get_sim_variables_names():
    """
    Parse the .inf file of the *base project* to map variable descriptions to positions.

    This mapping is assumed to be identical across all project copies.
    """
    sim_variables_path: Path = Config.OUTPUT_DIR / f'{Config.PROJECT_NAME}.inf'
    if not sim_variables_path.exists():
        raise FileNotFoundError(
            f'INF file "{sim_variables_path}" not found. '
            'Compile/run the base project at least once to generate the .inf file.'
        )

    sim_variables_txt = sim_variables_path.read_text(encoding='utf-8')
    pattern = r'PGB\((\d+)\).*?Desc="([^"]+)"'
    results = re.findall(pattern, sim_variables_txt)
    parsed = {desc: int(idx) - 1 for idx, desc in results}
    return parsed


def get_case(n_turbines=5, R_target=0.97):
    var_farm = 1.0 / 12.0
    var_eps_max = var_farm * (1.0 / (R_target**2) - 1.0)
    delta_max = np.sqrt(3.0 * var_eps_max)
    delta = delta_max
    P_farm = np.random.uniform(0.0, 1.0)
    eps = np.random.uniform(-delta, delta, size=n_turbines)
    scenario = P_farm + eps
    scenario = np.clip(scenario, 0.001, 1.0)

    resistances = [0.001, 5, 10, 25, 40, 50]
    fault_types = [1, 4, 8, 11]
    locals_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

    case = {
        'pot1': float(scenario[0].round(4)),
        'pot2': float(scenario[1].round(4)),
        'pot3': float(scenario[2].round(4)),
        'pot4': float(scenario[3].round(4)),
        'pot5': float(scenario[4].round(4)),
        'resistencia': float(np.random.choice(resistances)),
        'tipo': int(np.random.choice(fault_types)),
        'local': int(np.random.choice(locals_))
    }
    return case


if __name__ == '__main__':
    # variables_to_save = load_variables_to_save_antigo(Config.MEASUREMENTS_PATH)
    variables_to_save = ['Vbus:1', 'Vbus:2', 'Vbus:3',
                         'Ibus:1', 'Ibus:2', 'Ibus:3']

    # Map variable descriptions to indices (from base project .inf)
    sim_variables = get_sim_variables_names()

    for var in variables_to_save:
        if var not in sim_variables:
            raise ValueError(
                f'Variable "{var}" not found in simulation variables. '
                f'Available variables: {list(sim_variables.keys())}'
            )

    # Connect to PSCAD
    pscad = mhi.pscad.connect()
    if not pscad:
        raise RuntimeError('Failed to connect to PSCAD. Please ensure it is running.')

    # Prepare worker projects (one per parallel PSCAD task)
    worker_projects = []
    for project_name in Config.PROJECT_NAMES:
        project = pscad.project(project_name)
        if not project:
            raise RuntimeError(
                f'Project "{project_name}" not found. '
                'Make sure it is open in PSCAD and the name matches.'
            )
        canvas = project.canvas(Config.CANVAS)
        if not canvas:
            raise RuntimeError(
                f'Canvas "{Config.CANVAS}" not found in project "{project_name}". '
                'Check the canvas name.'
            )

        worker_projects.append({
            'name': project_name,
            'project': project,
            'canvas': canvas
        })

    # --- Configure simulation set with all worker projects ---
    sim_sets = pscad.simulation_sets()
    if Config.SIM_SET_NAME not in sim_sets:
        pscad.create_simulation_set(Config.SIM_SET_NAME)

    sim_set = pscad.simulation_set(Config.SIM_SET_NAME)
    sim_tasks = sim_set.list_tasks()

    for wp in worker_projects:
        if wp['name'] not in sim_tasks:
            sim_set.add_tasks(wp['name'])

    # --- Parallel Monte Carlo execution ---
    num_cases = 0
    t0 = perf_counter()

    for batch_idx in range(1, Config.MAX_BATCHES + 1):
        print(f'--- Starting batch {batch_idx} ---')

        # Each "step" here is a *wave* of parallel simulations (one per project)
        for _ in tqdm(range(Config.CASES_PER_BATCH), desc=f'Batch {batch_idx}'):
            # Build cases for each worker project
            wave_cases = []  # list of dicts: {project_name, canvas, case_id, case_params}

            for wp in worker_projects:
                case_params = get_case()
                case_id = f'Case_{num_cases}'
                num_cases += 1

                # Apply parameters to constants in THIS project's canvas
                for var_name, value in case_params.items():
                    try:
                        obj = wp['canvas'].find('master:const', var_name)
                    except Exception as e:
                        raise RuntimeError(
                            f'Error finding object "master:const" with name "{var_name}" '
                            f'in canvas "{Config.CANVAS}" of project "{wp["name"]}": {e}'
                        )
                    if obj:
                        obj.parameters(Value=f'{value}')
                    else:
                        raise ValueError(
                            f'Object "master:const" with name "{var_name}" not found '
                            f'in canvas "{Config.CANVAS}" of project "{wp["name"]}".'
                        )

                # Clean old outputs for THIS project only
                output_dir = get_output_dir_for_project(wp['name'])
                if output_dir.exists():
                    for file in output_dir.glob(f'{wp["name"]}*.out'):
                        file.unlink()

                wave_cases.append({
                    'project_name': wp['name'],
                    'case_id': case_id,
                    'case_params': case_params,
                })

            # Run all tasks in the simulation set in parallel
            pscad.run_simulation_sets(Config.SIM_SET_NAME)

            # Collect and save results for each worker project in this wave
            for wc in wave_cases:
                save_data(
                    project_name=wc['project_name'],
                    sim_names=sim_variables,
                    save_names=variables_to_save,
                    idx_case=wc['case_id'],
                    case_params=wc['case_params']
                )

    t1 = perf_counter()
    print(f'Simulation completed in {t1 - t0:.2f} seconds.')
    if num_cases > 0:
        print(f'Time per case: {(t1 - t0) / num_cases:.2f} seconds (avg over {num_cases} cases).')

    # Ensure any remaining buffered rows are written to disk
    flush_db_buffer()
