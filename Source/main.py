#!/usr/bin/env python3
"""
Hashiwokakero Solver - Main Program
CSC14003 - Introduction to Artificial Intelligence
Project 02: Logic

This script provides a command-line interface to run the Hashiwokakero solver,
execute test suites, and perform performance analysis.
"""
import os
import sys
import argparse

# Import helper modules
from helper_01 import HashiwokakeroSolver, load_input
from helper_02 import (
    run_comprehensive_testing_and_analysis,
    analyze_cnf_for_grid,
    solve_and_display_puzzle,
    input_custom_grid
)

def print_banner():
    """Prints the project banner."""
    print("=" * 70)
    print("   HASHIWOKAKERO SOLVER - PROJECT 02")
    print("   CSC14003 - Introduction to Artificial Intelligence")
    print("   Multiple Algorithm Implementation & Analysis")
    print("=" * 70)
    print()

def show_project_info():
    """Displays information about the project and its structure."""
    print("\n[ PROJECT INFORMATION ]")
    print("-" * 40)
    print("This project implements a Hashiwokakero puzzle solver using four different")
    print("algorithms: PySAT (CNF-based), A* Search, Backtracking, and Brute Force.")
    
    print("\nProject Requirements Checklist:")
    print("  ✓ CNF formulation for puzzle rules")
    print("  ✓ Automatic CNF generation")
    print("  ✓ PySAT library integration for solving")
    print("  ✓ A* search algorithm (implemented from scratch)")
    print("  ✓ Backtracking algorithm (implemented from scratch)")
    print("  ✓ Brute-force algorithm for comparison")
    print("  ✓ Comprehensive testing suite (10+ test cases)")
    print("  ✓ Performance analysis and reporting")

    print("\nFile Structure:")
    print("  - Source/main.py: This main script.")
    print("  - Source/helper_01.py: Core solver logic (HashiwokakeroSolver class).")
    print("  - Source/helper_02.py: Test generation, analysis, and plotting functions.")
    print("  - Source/requirements.txt: Python dependencies.")
    print("  - Source/README.txt: Instructions on how to run.")
    print("  - Source/Inputs/: Directory for input puzzle files.")
    print("  - Source/Outputs/: Directory for generated solution files.")
    print()

def interactive_solver():
    """Handles the interactive solver menu."""
    print("\n[ INTERACTIVE SOLVER ]")
    print("-" * 40)
    
    while True:
        print("\nInteractive Solver Options:")
        print("  1. Load puzzle from file")
        print("  2. Enter custom grid manually")
        print("  3. Use a predefined example")
        print("  4. Return to main menu")
        
        choice = input("Enter your choice (1-4): ").strip()
        grid = None
        description = ""

        if choice == '1':
            filename = input("Enter filename (e.g., Inputs/input-01.txt): ").strip()
            try:
                grid = load_input(filename)
                description = f"File: {filename}"
            except Exception as e:
                print(f"Error: Could not load file: {e}", file=sys.stderr)
        
        elif choice == '2':
            grid = input_custom_grid()
            if grid:
                description = "Custom Grid"

        elif choice == '3':
            grid = [[2, 0, 3], [0, 0, 0], [1, 0, 2]]
            description = "Simple 3x3 Example"
        
        elif choice == '4':
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

        if grid:
            solve_and_display_puzzle(grid, description)

def main():
    """Main program entry point."""
    parser = argparse.ArgumentParser(description='Hashiwokakero Solver - CSC14003 Project 02')
    parser.add_argument('--file', type=str, help='Solve a single puzzle file directly.')
    parser.add_argument('--test', action='store_true', help='Run the comprehensive test suite and exit.')
    
    args = parser.parse_args()
    
    print_banner()

    if args.file:
        try:
            grid = load_input(args.file)
            solve_and_display_puzzle(grid, f"File: {args.file}")
        except Exception as e:
            print(f"Error processing file '{args.file}': {e}", file=sys.stderr)
        return

    if args.test:
        run_comprehensive_testing_and_analysis()
        return

    # Main menu loop
    while True:
        print("MAIN MENU")
        print("-" * 20)
        print("  1. Interactive Solver")
        print("  2. Run Comprehensive Test Suite")
        print("  3. Analyze CNF Formulation for a Puzzle")
        print("  4. Project Information")
        print("  5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            interactive_solver()
        elif choice == '2':
            run_comprehensive_testing_and_analysis()
        elif choice == '3':
            filename = input("Enter puzzle file for CNF analysis (e.g., Inputs/input-01.txt): ").strip()
            try:
                grid = load_input(filename)
                analyze_cnf_for_grid(grid)
            except Exception as e:
                print(f"Error: Could not load file: {e}", file=sys.stderr)
        elif choice == '4':
            show_project_info()
        elif choice == '5':
            print("\nThank you for using the Hashiwokakero solver!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")
        
        print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    # Ensure Inputs directory exists
    if not os.path.isdir('Inputs'):
        print("Warning: 'Inputs' directory not found. Some features may not work.", file=sys.stderr)
        print("Consider running the test suite once to generate input files.", file=sys.stderr)
    main()