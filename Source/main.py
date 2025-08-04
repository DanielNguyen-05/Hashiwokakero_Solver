from file_solver import load_input, upload_output
from hashiwokakero_solver import HashiwokakeroSolver
from island_map import IslandMap
from cnf_generator import CNFGenerator
from solution_checker import SolutionChecker
import time
import os

def main():
    print ("="*60)
    print("HASHIWOKAKERO SOLVER")
    print ("="*60)
    problem = input("Input the problem number (e.g., 03): ").strip()
    grid = load_input(f"input-{problem}.txt")
    for row in grid:
        print(row)
    
    print("\n" + "="*60)
    print("SOLVING WITH DIFFERENT METHODS")
    print("="*60)
    
    solver = HashiwokakeroSolver(grid)
    results = solver.compare_algorithms(verbose=True)
    solver.print_performance_summary(results)
    
    print("\n" + "="*60)
    print("SAVING OUTPUTS")
    print("="*60)
    
    method_mapping = {
        'A* Search': 'AStar',
        'Brute Force': 'BruteForce', 
        'DPLL Backtracking': 'Backtracking',
        'PySAT': 'PySat'
    }
    
    for method_name, result in results.items():
        if result['success'] and result['grid'] is not None:
            method_code = method_mapping.get(method_name, method_name.replace(' ', ''))
            try:
                upload_output(result['grid'], f"output-{problem}.txt", method=method_code)
                print(f"✓ Saved output for {method_name} ({method_code})")
            except Exception as e:
                print(f"✗ Failed to save output for {method_name}: {e}")
        else:
            print(f"✗ No valid solution from {method_name}")

if __name__ == "__main__":
    main()
