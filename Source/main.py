from helper import load_input, upload_output
from hashiwokakero_solver import HashiwokakeroSolver
from island_map import IslandMap
from cnf_generator import CNFGenerator
from solution_checker import SolutionChecker

def main():
    grid = load_input("input-10.txt")
    for row in grid:
        print(row)
    
    solver = HashiwokakeroSolver(grid)
    result_grid, elapsed_time, success = solver.solve_with_pysat()

    if success: upload_output(result_grid, "output-10.txt", method="PySat")

if __name__ == "__main__":
    main()