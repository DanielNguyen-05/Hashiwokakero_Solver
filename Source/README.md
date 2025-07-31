# 🧠 Hashiwokakero Puzzle Solver

A multi-algorithm solver for the classic **Hashiwokakero (Bridges/Hashi)** puzzle.

## 🚀 Features

This project supports solving puzzles using various algorithms:

- 🧩 **CNF (SAT-based)** — Uses PySAT for logical constraint solving  
- 🧠 **A\*** — Heuristic-based search with bridge-count deficit  
- 🔍 **Backtracking** — Systematic constraint-based search  
- 💥 **Brute Force** — Exhaustive solution search (for small grids)

---

## 📦 Installation

1. Ensure **Python 3.7+** is installed  

2. Create `.venv` environemt and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install required packages via:

```bash
pip install -r Source/requirements.txt
```

4. Run the program:

```bash
python Source/main.py
```
