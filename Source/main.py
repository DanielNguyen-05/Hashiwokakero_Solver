from helper import load_input, upload_output
from hashiwokakero_solver import HashiwokakeroSolver
from island_map import IslandMap
from cnf_generator import CNFGenerator
from solution_checker import SolutionChecker
import time
import os


"""
def main():
    grid = load_input("input-10.txt")
    for row in grid:
        print(row)
    
    solver = HashiwokakeroSolver(grid)
    result_grid, elapsed_time, success = solver.solve_with_pysat()

    if success: upload_output(result_grid, "output-10.txt", method="PySat")
"""

"""
def main():
    input_dir = "Inputs"
    method = "PySat"

    for fname in os.listdir(input_dir):
        if fname.startswith("input-") and fname.endswith(".txt"):
            grid = load_input(fname)
            solver = HashiwokakeroSolver(grid)

            start = time.time()
            result_grid, elapsed_time, success = solver.solve_with_pysat()
            end = time.time()

            if success:
                output_name = fname.replace("input-", "output-")
                upload_output(result_grid, output_name, method)
                print(f"Solve {fname} within {elapsed_time:.4f}s")
            else:
                print(f"Can not solve {fname} within {end - start:.4f}s")
"""

def main():
    grid = load_input("input-16.txt")
    for row in grid:
        print(row)
    
    solver = HashiwokakeroSolver(grid)
    result_grid, elapsed_time, success = solver.solve_with_backtracking()

    if success: upload_output(result_grid, "output-16.txt", method="Backtracking")
    else: print("[Fail] Solver failed")


if __name__ == "__main__":
    main()