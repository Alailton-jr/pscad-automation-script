# import numpy as np
import pandas as pd
from pathlib import Path
import re
import mhi.pscad
import numpy as np
from scipy.io import savemat

class Config:
    SEARCH_PATH = False
    PROJECT_NAME = 'Modelo_LOOP_FP1_PEN025'
    PROJECT_PATH = r''
    CANVAS = 'Main'
    CASE_PATH = 'case-mock.csv'
    MEASUREMENTS_PATH = 'measuruements-mock.csv'
    COMPILER = 'gf46'
    OUTPUT_DIR = None
    SAVES_OUTPUT = None
    OUTPUT_FORMAT = 'csv'  # or 'pkl', 'parquet', 'mat', 'npy'
    OUTPUT_FORMAT_COLUMNS = True  # Whether to include column names in the output

if Config.SEARCH_PATH:
    # Open the search file dialog and look for files that ends with .pscx
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()  # Hide the root window
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
        
if isinstance(Config.PROJECT_PATH, str):
    Config.PROJECT_PATH = Path(Config.PROJECT_PATH)

if '.' in Config.PROJECT_NAME:
    Config.PROJECT_NAME = Config.PROJECT_NAME.split('.')[0]

Config.OUTPUT_DIR = Config.PROJECT_PATH / f'{Config.PROJECT_NAME}.{Config.COMPILER}'
Config.SAVES_OUTPUT = Config.PROJECT_PATH / f'{Config.PROJECT_NAME}_saves'
Config.SAVES_OUTPUT.mkdir(parents=True, exist_ok=True)

def get_all_sim_names(path):
    import re
    save_info = pd.read_csv(path, sep=r'\s+', header=None)
    save_names = []
    for idx in range(len(save_info)):
        column_num = save_info.iloc[idx].str.contains('Desc').idxmax()
        name = save_info.iloc[idx][column_num]
        name = re.findall(r'"(.*?)"', name)[0]
        save_names.append(name)
    return save_names

def save_data(sim_names, save_names, idx_case):
    output_array = []
    if len(sim_names) > 10:
        output_files = list(Config.OUTPUT_DIR.glob(f'{Config.PROJECT_NAME}*.out'))
        output_files = [pd.read_csv(file, sep=r'\s+', header=None).to_numpy() for file in output_files]
        for name in save_names:
            pos = sim_names[name]
            idx_file = pos//10
            idx_col = (pos % 10)+1 #there's the time col
            output_array.append(output_files[idx_file][:, idx_col])
    else:
        output_file = list(Config.OUTPUT_DIR.glob(f'{Config.PROJECT_NAME}*.out'))[0]
        output_array = pd.read_csv(output_file, sep='\s+', header=None).to_numpy()
        for name in save_names:
            pos = sim_names[name] + 1 
            output_array.append(output_file[:, idx_col])
    output_array = np.array(output_array).T
    if Config.OUTPUT_FORMAT_COLUMNS:
            output_df = pd.DataFrame(output_array, columns=save_names)
    else:
        output_df = pd.DataFrame(output_array)
    if Config.OUTPUT_FORMAT == 'csv':
        output_df.to_csv(Config.SAVES_OUTPUT / f'{idx_case}.csv', index=False)
    elif Config.OUTPUT_FORMAT == 'parquet':
        output_df.to_parquet(Config.SAVES_OUTPUT / f'{idx_case}.parquet', index=False)
    elif Config.OUTPUT_FORMAT == 'mat':
        savemat(Config.SAVES_OUTPUT / f'{idx_case}.mat', {f'{idx_case}': output_array})
    elif Config.OUTPUT_FORMAT == 'pkl':
        output_df.to_pickle(Config.SAVES_OUTPUT / f'{idx_case}.pkl')
    elif Config.OUTPUT_FORMAT == 'npy':
        np.save(Config.SAVES_OUTPUT / f'{idx_case}.npy', output_array)
    return   
    
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
            param_value = row[i+1]
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
    sim_variables_path:Path = Config.OUTPUT_DIR / f'{Config.PROJECT_NAME}.inf'
    sim_variables_txt = sim_variables_path.read_text(encoding='utf-8')
    pattern = r'PGB\((\d+)\).*?Desc="([^"]+)"'
    results = re.findall(pattern, sim_variables_txt)
    parsed = {desc: int(idx)-1 for idx, desc in results}
    return parsed

if __name__ == '__main__':
    cases = load_cases_csv_antigo(Config.CASE_PATH)
    variables_to_save = load_variables_to_save_antigo(Config.MEASUREMENTS_PATH)
    
    sim_variables = get_sim_variables_names()
    for var in variables_to_save:
        if var not in sim_variables:
            raise ValueError(f'Variable "{var}" not found in simulation variables. Available variables: {list(sim_variables.keys())}')
        
    pscad = mhi.pscad.connect()
    if not pscad:
        raise RuntimeError('Failed to connect to PSCAD. Please ensure it is running.')
    
    project = pscad.project(Config.PROJECT_NAME)
    canvas = project.canvas(Config.CANVAS)
    
    if not project:
        raise RuntimeError(f'Project "{Config.PROJECT_NAME}" not found. Please check the project name and path.')
    if not canvas:
        raise RuntimeError(f'Canvas "{Config.CANVAS}" not found in project "{Config.PROJECT_NAME}". Please check the canvas name.')
    
    for case in range(len(cases)):
        row = cases.iloc[case]
        print(f'Running case {case+1}/{len(cases)}: {row.to_dict()}')
        for col in cases.columns:
            obj = canvas.find('master:const', col)
            if obj:
                obj.parameters(Value=f'{case[col].values[0]}')
            else:
                raise ValueError(f'Object "master:const" with name "{col}" not found in canvas "{Config.CANVAS}".')
        
        data_path = list(Config.OUTPUT_DIR.glob(f'{Config.PROJECT_NAME}*.out'))
        for file in data_path:
            file.unlink()
        project.run()
        save_data(sim_variables, variables_to_save, f'Case_{case+1}')