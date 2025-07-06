# PSCAD Simulation Automation (`main.py`)

This Python script provides a robust framework for automating PSCAD simulations. It is designed to run a series of simulation cases defined in a CSV file, modify component parameters for each case, execute the simulation, and save specified output variables to a chosen format.

## 🧩 Requirements

Install required libraries using:

```bash
pip install -r requirements.txt
````

---

## ⚙️ Configuration

All script settings are managed within the `Config` class in `main.py`.

### Key Configuration Options

| Option                            | Description                                                                     |
| --------------------------------- | ------------------------------------------------------------------------------- |
| `USE_FILE_DIALOG_SELECTOR` (bool) | If `True`, opens a dialog to select the `.pscx` file. If `False`, set manually. |
| `PROJECT_NAME` (str)              | The name of your PSCAD project (e.g., `'MyProject'`).                           |
| `PROJECT_PATH` (Path)             | Absolute path to the directory containing your `.pscx` file.                    |
| `CANVAS` (str)                    | Name of the canvas where target components are located (e.g., `'Main'`).        |
| `CASES_FILE` (Path)               | Path to the CSV file defining the simulation cases.                             |
| `MEASUREMENTS_FILE` (Path)        | Path to the CSV file listing output variables to save.                          |
| `COMPILER` (str)                  | Fortran compiler version used by PSCAD (e.g., `'gf46'`).                        |
| `OUTPUT_FORMAT` (str)             | Output format: `'csv'`, `'parquet'`, `'mat'`, `'pkl'`, `'npy'`.                 |
| `INCLUDE_HEADER_IN_OUTPUT` (bool) | If `True`, includes header row with variable names.                             |
| `COMPONENT_TYPE_TO_MODIFY` (str)  | PSCAD component type to be modified (e.g., `'master:const'`).                   |

---

## 📁 Input File Formats

### 1. Cases File (`case-mock.csv`)

Defines the parameters for each simulation case.
**Format**: CSV without a header. Each row contains parameter-value pairs.

**Example**:

```
Fault_Res,0.1,Fault_Time,0.5
Fault_Res,0.2,Fault_Time,0.5
Fault_Res,0.1,Fault_Time,0.6
```

* Row 1: `Fault_Res = 0.1`, `Fault_Time = 0.5`
* Row 2: `Fault_Res = 0.2`, `Fault_Time = 0.5`

### 2. Measurements File (`measurements-mock.csv`)

Lists output variables to save.
**Format**: CSV without a header. One variable name per row.

**Example**:

```
V_RMS_A
I_RMS_A
Frequency
```

> Variable names must match the **Desc** field in PSCAD output channel configuration.

---

## ▶️ How to Use

1. **Configure the Script**
   Open `main.py` and adjust the `Config` class to match your project.

2. **Prepare Input Files**
   Ensure `CASES_FILE` and `MEASUREMENTS_FILE` are correctly formatted and available.

3. **Run PSCAD**
   Open and keep the PSCAD application running.

4. **Execute the Script**
   From terminal:

   ```bash
   python main.py
   ```

5. **Monitor the Output**
   Console logs will show the simulation progress, case numbers, status, and saved files.

---

## 📤 Output

* A folder named `{PROJECT_NAME}_saves` will be created inside your `PROJECT_PATH`.
* For each case in `CASES_FILE`, a corresponding output file (e.g., `Case_1.csv`, `Case_2.csv`, etc.) will be saved.
* Each output file includes a `Time` column followed by columns from `MEASUREMENTS_FILE`.

---
