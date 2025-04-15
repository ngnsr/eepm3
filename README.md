# eepm3
### Installation

#### Clone this repository or download the project files (main.py, read.py, analyze.py, display.py) to a local directory.
Set Up a Virtual Environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
#### Install Dependencies: Install the required Python packages using pip:
```bash
pip install numpy scipy
```
#### Prepare Input Files:
Create matrix.txt and vector.txt in the project directory.
Ensure matrix.txt contains a square matrix (same number of rows and columns) with space-separated numbers.
Ensure vector.txt contains a single row of numbers matching the matrix’s dimension.

### Usage

Run the Program: Execute the main script, optionally specifying the input files as command-line arguments:
```bash
python main.py matrix.txt vector.txt
```

If no arguments are provided, the program defaults to matrix.txt and vector.txt.
