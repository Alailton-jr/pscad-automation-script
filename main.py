# -*- coding: utf-8 -*-
"""
PSCAD Automation Script

This script automates running multiple simulation cases in PSCAD,
extracting specified measurements, and saving them to a desired format.

It is designed to be configured via the `Config` class and can be
executed directly.

Dependencies:
- pandas
- numpy
- scipy
- mhi.pscad (proprietary library for PSCAD interaction)

To run:
1. Adjust the parameters in the `Config` class below.
2. Ensure PSCAD is running.
3. Execute the script from your terminal: `python your_script_name.py`
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import mhi.pscad
import numpy as np
import pandas as pd
from scipy.io import savemat

# --- Configuration ---
class Config:
    """
    Configuration settings for the PSCAD automation task.
    
    Modify these values to match your project setup.
    """
    # Set to True to use a file dialog to select the PSCAD project file.
    # If False, PROJECT_PATH and PROJECT_NAME must be set manually.
    USE_FILE_DIALOG_SELECTOR = False

    # --- Project Details (Required if USE_FILE_DIALOG_SELECTOR is False) ---
    PROJECT_NAME: str = 'Modelo_LOOP_FP1_PEN025'
    PROJECT_PATH: Path = Path(r'')  # Example: Path(r'C:\Users\YourUser\Documents\PSCAD')
    CANVAS: str = 'Main'
    
    # --- Input Files ---
    # CSV file defining the simulation cases. Each row is a case, each column a parameter.
    CASES_FILE: Path = Path('case-mock.csv')
    # CSV file listing the names of the variables to save after each run.
    MEASUREMENTS_FILE: Path = Path('measurements-mock.csv')

    # --- PSCAD & Output Settings ---
    COMPILER: str = 'gf46'
    # Supported formats: 'csv', 'parquet', 'mat', 'pkl', 'npy'
    OUTPUT_FORMAT: str = 'parquet'
    # If True, the output file will include a header with column names.
    INCLUDE_HEADER_IN_OUTPUT: bool = True
    # The name of the component type in PSCAD to modify (e.g., a constant).
    COMPONENT_TYPE_TO_MODIFY: str = 'master:const'


# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class PSCADAutomation:
    """
    Handles the end-to-end process of running PSCAD simulations.
    """
    def __init__(self, config: Config):
        """
        Initializes the automation process with the given configuration.

        Args:
            config: An instance of the Config class containing all settings.
        """
        self.config = config
        self._setup_paths()

        # Initialize PSCAD connection and project object to None.
        # They will be connected in the `initialize_pscad` method.
        self.pscad = None
        self.project = None
        self.canvas = None

    def _setup_paths(self) -> None:
        """
        Validates and constructs all necessary paths from the configuration.
        """
        if self.config.USE_FILE_DIALOG_SELECTOR:
            self._select_project_with_dialog()

        # Ensure project name is clean (no file extension)
        if '.' in self.config.PROJECT_NAME:
            self.config.PROJECT_NAME = self.config.PROJECT_NAME.split('.')[0]
        
        if not self.config.PROJECT_PATH or not self.config.PROJECT_PATH.is_dir():
            raise FileNotFoundError(f"Project path does not exist: {self.config.PROJECT_PATH}")

        # Define output directories based on the project path
        self.output_dir = self.config.PROJECT_PATH / f'{self.config.PROJECT_NAME}.{self.config.COMPILER}'
        self.saves_dir = self.config.PROJECT_PATH / f'{self.config.PROJECT_NAME}_saves'
        
        # Create the directory for saved results if it doesn't exist
        self.saves_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Results will be saved in: {self.saves_dir}")

    def _select_project_with_dialog(self) -> None:
        """
        Opens a file dialog for the user to select a .pscx project file.
        Updates the configuration with the selected file's details.
        """
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError:
            logging.error("tkinter is required for the file dialog. Please install it.")
            raise

        root = tk.Tk()
        root.withdraw()  # Hide the main tkinter window
        
        file_path_str = filedialog.askopenfilename(
            title="Select PSCAD Project File",
            filetypes=[("PSCAD Project Files", "*.pscx")],
            initialdir='.'
        )
        
        if not file_path_str:
            raise FileNotFoundError("No project file was selected.")
            
        file_path = Path(file_path_str).resolve()
        self.config.PROJECT_PATH = file_path.parent
        self.config.PROJECT_NAME = file_path.stem
        logging.info(f"Selected project '{self.config.PROJECT_NAME}' in '{self.config.PROJECT_PATH}'")

    def initialize_pscad(self) -> None:
        """
        Connects to PSCAD, loads the project, and gets the canvas.
        """
        logging.info("Connecting to PSCAD...")
        self.pscad = mhi.pscad.connect()
        if not self.pscad:
            raise ConnectionError('Failed to connect to PSCAD. Please ensure it is running.')

        logging.info(f"Loading project: {self.config.PROJECT_NAME}")
        self.project = self.pscad.project(self.config.PROJECT_NAME)
        if not self.project:
            raise FileNotFoundError(f'Project "{self.config.PROJECT_NAME}" not found. Check project name and path.')

        logging.info(f"Accessing canvas: {self.config.CANVAS}")
        self.canvas = self.project.canvas(self.config.CANVAS)
        if not self.canvas:
            raise ValueError(f'Canvas "{self.config.CANVAS}" not found in project.')

    def load_cases(self) -> pd.DataFrame:
        """
        Loads simulation cases from a CSV file using the original method.
        The CSV file should have no header. Each row contains pairs of
        parameter names and values (e.g., 'Param1', 10, 'Param2', 20).
        """
        logging.info(f"Loading cases from: {self.config.CASES_FILE}")
        if not self.config.CASES_FILE.exists():
            raise FileNotFoundError(f"Cases file not found: {self.config.CASES_FILE}")

        cases_df = pd.read_csv(self.config.CASES_FILE, sep=',', header=None)
        cases = {}
        for _, row in cases_df.iterrows():
            # Drop NaN values that might result from uneven row lengths
            row = row.dropna()
            for i in range(0, len(row), 2):
                param_name = row.iloc[i]
                param_value = row.iloc[i+1]
                if param_name not in cases:
                    cases[param_name] = []
                cases[param_name].append(param_value)
        
        final_cases_df = pd.DataFrame(cases)
        logging.info(f"Found {len(final_cases_df)} cases to run.")
        return final_cases_df

    def load_variables_to_save(self) -> List[str]:
        """
        Loads the list of variable names to save from a CSV file using the original method.
        The file should contain one variable name per row in the first column, with no header.
        """
        logging.info(f"Loading variables to save from: {self.config.MEASUREMENTS_FILE}")
        if not self.config.MEASUREMENTS_FILE.exists():
            raise FileNotFoundError(f"Measurements file not found: {self.config.MEASUREMENTS_FILE}")

        variables_df = pd.read_csv(self.config.MEASUREMENTS_FILE, header=None)
        variables = variables_df.iloc[:, 0].tolist()
        logging.info(f"Found {len(variables)} variables to save: {variables}")
        return variables
    
    def get_simulation_variable_map(self) -> Dict[str, int]:
        """
        Parses the .inf file to get a map of all available simulation 
        output variables and their corresponding column indices.
        """
        inf_file = self.output_dir / f'{self.config.PROJECT_NAME}.inf'
        if not inf_file.exists():
            # Attempt to run once to generate the file if it's missing
            logging.warning(f".inf file not found at {inf_file}. Running simulation once to generate it.")
            self.project.run()
            if not inf_file.exists():
                 raise FileNotFoundError(f"Could not find or generate .inf file at: {inf_file}")

        logging.info(f"Reading simulation variable map from: {inf_file}")
        inf_text = inf_file.read_text(encoding='utf-8')
        
        # Regex to find variable names and their indices
        pattern = r'PGB\((\d+)\).*?Desc="([^"]+)"'
        matches = re.findall(pattern, inf_text)
        
        # Create a dictionary mapping: {variable_name: index}
        # The index is decremented by 1 to be 0-based.
        variable_map = {desc: int(idx) - 1 for idx, desc in matches}
        return variable_map

    def _read_simulation_output(self, variable_map: Dict[str, int], variables_to_save: List[str]) -> pd.DataFrame:
        """
        Reads the PSCAD .out file(s) and extracts the required data using the original logic.
        This logic handles cases with single or multiple .out files based on the total
        number of available simulation variables.

        Args:
            variable_map: A dictionary mapping all available variable names to their indices.
            variables_to_save: A list of variable names to extract.

        Returns:
            A pandas DataFrame containing the extracted data, including a 'Time' column.
        """
        output_data_columns = []
        output_files_paths = sorted(list(self.output_dir.glob(f'{self.config.PROJECT_NAME}*.out')))
        if not output_files_paths:
            raise FileNotFoundError(f"No output .out files found in {self.output_dir}")

        logging.info(f"Reading data from {len(output_files_paths)} output file(s).")
        
        # The logic for extracting variable data is split based on the total number of variables.
        if len(variable_map) > 10:
            # Multiple output files are expected. Read all into memory.
            output_files_data = [pd.read_csv(file, sep=r'\s+', header=None).to_numpy() for file in output_files_paths]
            for name in variables_to_save:
                pos = variable_map[name]
                idx_file = pos // 10
                idx_col = (pos % 10) + 1  # +1 to account for the time column in each file
                output_data_columns.append(output_files_data[idx_file][:, idx_col])
        else:
            # A single output file is expected.
            output_file_data = pd.read_csv(output_files_paths[0], sep=r'\s+', header=None).to_numpy()
            for name in variables_to_save:
                pos = variable_map[name] + 1 # +1 to account for the time column
                output_data_columns.append(output_file_data[:, pos])

        # Transpose the list of columns to get a 2D array (rows=time, cols=vars)
        output_array = np.array(output_data_columns).T
        
        # Create the final DataFrame with the requested variables
        result_df = pd.DataFrame(output_array, columns=variables_to_save)
        
        # Get the time column from the first output file and add it to the DataFrame
        time_data = pd.read_csv(output_files_paths[0], sep=r'\s+', header=None, usecols=[0]).squeeze()
        result_df.insert(0, 'Time', time_data)
            
        return result_df

    def _save_results(self, data: pd.DataFrame, case_name: str) -> None:
        """
        Saves the given DataFrame to a file in the configured format.

        Args:
            data: The DataFrame to save.
            case_name: The name for the output file (e.g., 'Case_1').
        """
        output_path = self.saves_dir / f"{case_name}.{self.config.OUTPUT_FORMAT}"
        logging.info(f"Saving results for '{case_name}' to {output_path}")

        output_format = self.config.OUTPUT_FORMAT.lower()
        
        # For formats that don't natively support headers, we convert to numpy
        if output_format in ['mat', 'npy']:
            output_array = data.to_numpy()
            if output_format == 'mat':
                savemat(output_path, {case_name: output_array})
            else: # npy
                np.save(output_path, output_array)
        else: # csv, parquet, pkl
            # Determine whether to include the header
            include_header = self.config.INCLUDE_HEADER_IN_OUTPUT
            if output_format == 'csv':
                data.to_csv(output_path, index=False, header=include_header)
            elif output_format == 'parquet':
                data.to_parquet(output_path, index=False)
            elif output_format == 'pkl':
                data.to_pickle(output_path)
            else:
                logging.error(f"Unsupported output format: '{output_format}'")

    def run_automation(self) -> None:
        """
        Executes the main automation workflow.
        """
        self.initialize_pscad()
        
        cases_df = self.load_cases()
        variables_to_save = self.load_variables_to_save()
        
        # Get the map of all available simulation variables
        sim_variable_map = self.get_simulation_variable_map()
        
        # Validate that all requested variables are available in the simulation
        for var in variables_to_save:
            if var not in sim_variable_map:
                raise ValueError(
                    f'Variable "{var}" not found in simulation. Available variables: '
                    f'{list(sim_variable_map.keys())}'
                )

        # --- Main Simulation Loop ---
        for idx, case_row in cases_df.iterrows():
            case_name = f'Case_{idx + 1}'
            logging.info(f"--- Running {case_name}/{len(cases_df)} ---")
            logging.info(f"Parameters: {case_row.to_dict()}")

            # Set parameters in PSCAD canvas for the current case
            for param_name, param_value in case_row.items():
                component = self.canvas.find(self.config.COMPONENT_TYPE_TO_MODIFY, param_name)
                if component:
                    component.parameters(Value=str(param_value))
                else:
                    raise ValueError(
                        f'Component "{param_name}" of type "{self.config.COMPONENT_TYPE_TO_MODIFY}" '
                        f'not found in canvas "{self.config.CANVAS}".'
                    )
            
            # Clean up old output files before running
            for file in self.output_dir.glob(f'{self.config.PROJECT_NAME}*.out'):
                file.unlink()

            # Run the simulation
            logging.info("Starting PSCAD simulation run...")
            self.project.run()
            logging.info("Simulation finished.")

            # Read output and save the results
            results_df = self._read_simulation_output(sim_variable_map, variables_to_save)
            self._save_results(results_df, case_name)
            
        logging.info("--- Automation complete. All cases have been processed. ---")

# --- Main Execution Block ---
if __name__ == '__main__':
    try:
        # 1. Initialize the configuration
        config = Config()
        
        # 2. Create an instance of the automation tool
        automation_runner = PSCADAutomation(config)
        
        # 3. Run the entire process
        automation_runner.run_automation()
        
    except (FileNotFoundError, ConnectionError, ValueError) as e:
        logging.error(f"An error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
