from file_solver import load_input, upload_output
from hashiwokakero_solver import HashiwokakeroSolver
from island_map import IslandMap
from cnf_generator import CNFGenerator
from solution_checker import SolutionChecker
import time
import os

def main():
    grid = load_input("input-16.txt")
    print("Input Grid:")
    for row in grid:
        print(row)
    
    print("\n" + "="*60)
    print("SOLVING WITH DIFFERENT METHODS")
    print("="*60)
    
    # Create single solver instance for comparison
    solver = HashiwokakeroSolver(grid)
    
    # Use the built-in comparison method
    results = solver.compare_algorithms(verbose=True)
    
    # Print performance summary
    solver.print_performance_summary(results)
    
    # Save outputs for successful solutions
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
                upload_output(result['grid'], "output-16.txt", method=method_code)
                print(f"✓ Saved output for {method_name} ({method_code})")
            except Exception as e:
                print(f"✗ Failed to save output for {method_name}: {e}")
        else:
            print(f"✗ No valid solution from {method_name}")

if __name__ == "__main__":
    main()
