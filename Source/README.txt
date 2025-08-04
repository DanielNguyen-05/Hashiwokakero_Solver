# Hashiwokakero Solver

A comprehensive implementation of multiple algorithms to solve Hashiwokakero (Bridges) puzzles with performance comparison and memory tracking.

## Overview

This project implements four different algorithms to solve Hashiwokakero puzzles:
1. **PySAT** - Using SAT solver with Glucose3
2. **Brute Force** - Exhaustive search through all possible assignments
3. **DPLL Backtracking** - Davis-Putnam-Logemann-Loveland algorithm
4. **A* Search** - Informed search with admissible heuristics

Each algorithm includes memory tracking and performance metrics for comparison.

## Requirements

### Python Version
- Python 3.7 or higher

### Create the virtual environment and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Dependencies
Install the required packages using pip:

```bash
pip install --upgrade pip
pip install -r Source/requirements.txt
```

### Required Files
Make sure your project directory contains these files:
- `hashiwokakero_solver.py` - Main solver implementation
- `island_map.py` - Island map representation
- `cnf_generator.py` - CNF formula generation
- `solution_checker.py` - Solution validation
- `file_solver.py` - Input/output file handling
- `main.py` - Main execution script

## Input Format

Input files should be text files with the following format:
- Each line represents a row of the grid
- Numbers (1-8) represent islands with their required bridge count
- Zeros (0) represent empty cells

Example input file (`input-03.txt`):
```
1 0 2
0 0 0
0 0 1
```

## Output

### Console Output
The program displays:
1. Input grid visualization
2. Algorithm execution progress
3. Performance comparison table showing:
   - Success/Failure status
   - Execution time
   - Peak memory usage
   - Memory increase (Δ)
4. Best performers summary

Example output:
```
Input Grid:
[1, 0, 2]
[0, 0, 0]
[0, 0, 1]

============================================================
SOLVING WITH DIFFERENT METHODS
============================================================

--- Testing PySAT ---
Status: SUCCESS
Time: 0.0021s
Peak Memory: 0.03 MB
Memory Increase: 0.16 MB

--- Testing Brute Force ---
[timeout] Brute-force failed
Status: FAILED
Time: 10.0000s
Peak Memory: 0.03 MB
Memory Increase: -6.73 MB

--- Testing DPLL Backtracking ---
Status: SUCCESS
Time: 0.0040s
Peak Memory: 0.04 MB
Memory Increase: 0.55 MB

--- Testing A* Search ---
Status: SUCCESS
Time: 0.0003s
Peak Memory: 0.00 MB
Memory Increase: 0.06 MB

================================================================================
ALGORITHM PERFORMANCE COMPARISON
================================================================================
Algorithm            Status     Time (s)     Peak Memory (MB)   Memory Δ (MB)  
--------------------------------------------------------------------------------
PySAT                SUCCESS    0.0021       0.03               0.16           
Brute Force          FAILED     10.0000      0.03               -6.73          
DPLL Backtracking    SUCCESS    0.0040       0.04               0.55           
A* Search            SUCCESS    0.0003       0.00               0.06           

==================================================
BEST PERFORMERS
==================================================
Fastest: A* Search (0.0003s)
Lowest Memory: A* Search (0.00 MB)

============================================================
SAVING OUTPUTS
============================================================
✓ Saved output for PySAT (PySat)
✗ No valid solution from Brute Force
✓ Saved output for DPLL Backtracking (Backtracking)
✓ Saved output for A* Search (AStar)
```

### Output Files
Successful solutions are saved as (if available):
- `output-03-AStar.txt`
- `output-03-BruteForce.txt`
- `output-03-Backtracking.txt`
- `output-03-PySat.txt`

## Configuration

### Timeout Settings
You can adjust timeout values in `hashiwokakero_solver.py`:

```python
# In each solve method, change the timeout value:
timeout = 10  # seconds
```

### Memory Tracking
Memory tracking is enabled by default. To disable, remove the `@_track_memory` decorator from methods in `hashiwokakero_solver.py`.

## Algorithm Details

### 1. PySAT Solver
- Uses Glucose3 SAT solver
- Converts puzzle constraints to CNF
- Generally fastest for most puzzles
- Best for complex puzzles

### 2. Brute Force
- Tries all possible variable assignments
- Guaranteed to find solution if one exists
- Very slow for large puzzles
- Good for verification on small puzzles

### 3. DPLL Backtracking
- Implements unit propagation and backtracking
- More efficient than brute force
- Good balance of speed and reliability

### 4. A* Search
- Uses informed search with heuristics
- Prioritizes connections involving constrained islands
- Memory efficient
- Good for medium-sized puzzles

## File Structure

```
Hashiwokakero_Solver/
├── .venv/        # Virtual environment (in .gitignore)
├── Docs/
│   ├── Artificial Intelligence - A Modern Approach (3rd Edition).pdf
│   └── Report.pdf
├── Source/
│   ├── Inputs/
│   │   ├── input-03.txt
│   │   ├── input-04.txt
│   │   └── ...
│   ├── Outputs/    
│   │   ├── AStar/
│   │   │   └── output-03.txt
│   │   ├── Backtracking/
│   │   │   └── output-03.txt
│   │   ├── Bruteforce/
│   │   │   └── output-03.txt
│   │   └── PySat/
│   │       └── output-03.txt
│   ├── cnf_generator.py
│   ├── file_solver.py
│   ├── hashiwokakero_solver.py
│   ├── island_map.py
│   ├── main.py
│   ├── README.txt
│   ├── requirements.txt
│   └── solution_checker.py
├── .gitignore
├── Hashi_CNF.cnf   
└── Problem.pdf

```

## Contact

For questions, bug reports, or suggestions, feel free to contact me via email:

📧 **lnkhoi23@clc.fitus.edu.vn**
📧 **nhdang231@clc.fitus.edu.vn**
📧 **vnbtram23@clc.fitus.edu.vn**
📧 **pqthinh23@clc.fitus.edu.vn**


> Note: Please describe the issue clearly and, if possible, include the input file and the algorithm used.
