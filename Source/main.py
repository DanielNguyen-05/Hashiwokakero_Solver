from file_solver import load_input, upload_output
from hashiwokakero_solver import HashiwokakeroSolver
from island_map import IslandMap
from cnf_generator import CNFGenerator
from solution_checker import SolutionChecker
import time
import os

def main():
    grid = load_input("input-03.txt")
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
                upload_output(result['grid'], "output-03.txt", method=method_code)
                print(f"✓ Saved output for {method_name} ({method_code})")
            except Exception as e:
                print(f"✗ Failed to save output for {method_name}: {e}")
        else:
            print(f"✗ No valid solution from {method_name}")

def main_batch_processing():
    """Alternative main function for batch processing multiple input files"""
    input_dir = "Inputs"
    
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' not found!")
        return
    
    input_files = [f for f in os.listdir(input_dir) 
                   if f.startswith("input-") and f.endswith(".txt")]
    
    if not input_files:
        print(f"No input files found in '{input_dir}'!")
        return
    
    print(f"Found {len(input_files)} input files to process")
    print("="*80)
    
    overall_results = {}
    
    for fname in sorted(input_files):
        print(f"\nProcessing {fname}...")
        print("-" * 50)
        
        try:
            grid = load_input(fname)
            solver = HashiwokakeroSolver(grid)
            
            # Run comparison for this file
            results = solver.compare_algorithms(verbose=False)
            overall_results[fname] = results
            
            # Quick summary for this file
            successful_methods = [name for name, result in results.items() if result['success']]
            print(f"✓ Successful methods: {', '.join(successful_methods) if successful_methods else 'None'}")
            
            # Save outputs for successful solutions
            method_mapping = {
                'A* Search': 'AStar',
                'Brute Force': 'BruteForce', 
                'DPLL Backtracking': 'Backtracking',
                'PySAT': 'PySat'
            }
            
            for method_name, result in results.items():
                if result['success'] and result['grid'] is not None:
                    method_code = method_mapping.get(method_name, method_name.replace(' ', ''))
                    output_name = fname.replace("input-", "output-")
                    try:
                        upload_output(result['grid'], output_name, method=method_code)
                    except Exception as e:
                        print(f"  ✗ Failed to save {method_name}: {e}")
                        
        except Exception as e:
            print(f"✗ Error processing {fname}: {e}")
            overall_results[fname] = {"error": str(e)}
    
    # Print overall summary
    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    
    method_names = ['A* Search', 'Brute Force', 'DPLL Backtracking', 'PySAT']
    success_counts = {method: 0 for method in method_names}
    total_files = len([r for r in overall_results.values() if "error" not in r])
    
    for fname, file_results in overall_results.items():
        if "error" not in file_results:
            for method in method_names:
                if method in file_results and file_results[method]['success']:
                    success_counts[method] += 1
    
    print(f"{'Method':<20} {'Success Rate':<15} {'Avg Time (s)':<15}")
    print("-" * 50)
    
    for method in method_names:
        if total_files > 0:
            success_rate = f"{success_counts[method]}/{total_files}"
            
            # Calculate average time for successful runs
            times = []
            for file_results in overall_results.values():
                if ("error" not in file_results and 
                    method in file_results and 
                    file_results[method]['success']):
                    times.append(file_results[method]['time'])
            
            avg_time = f"{sum(times)/len(times):.4f}" if times else "N/A"
            print(f"{method:<20} {success_rate:<15} {avg_time:<15}")

def main_single_method():
    """Alternative main function for testing a single method"""
    grid = load_input("input-03.txt")
    solver = HashiwokakeroSolver(grid)
    
    print("Testing PySAT solver...")
    result = solver.solve_with_pysat()
    
    if len(result) == 4:
        grid_result, elapsed_time, success, memory_info = result
        print(f"Success: {success}")
        print(f"Time: {elapsed_time:.4f}s")
        print(f"Peak Memory: {memory_info['peak_memory_mb']:.2f} MB")
        
        if success:
            upload_output(grid_result, "output-03.txt", method="PySat")
    else:
        grid_result, elapsed_time, success = result
        print(f"Success: {success}")
        print(f"Time: {elapsed_time:.4f}s")

if __name__ == "__main__":
    # Choose which main function to run:
    
    # Option 1: Single file with full comparison (recommended)
    main()
    
    # Option 2: Batch processing multiple files
    # main_batch_processing()
    
    # Option 3: Test single method
    # main_single_method()