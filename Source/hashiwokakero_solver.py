import time
import copy
import tracemalloc
import psutil
import os
from pysat.solvers import Glucose3
from itertools import product
from island_map import IslandMap
from cnf_generator import CNFGenerator
from solution_checker import SolutionChecker
import heapq
from collections import defaultdict

class HashiwokakeroSolver:
    def __init__(self, grid):
        self.map = IslandMap(grid) 
        self.checker = SolutionChecker(self.map.islands, self.map.connections, self.map.grid.shape) 

    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  

    def _track_memory(func):
        """Decorator to track memory usage for solving methods"""
        def wrapper(self, *args, **kwargs):
            tracemalloc.start()
            initial_memory = self._get_memory_usage()
            
            result = func(self, *args, **kwargs)
            
            current, peak = tracemalloc.get_traced_memory()
            final_memory = self._get_memory_usage()
            tracemalloc.stop()
            
            if len(result) == 3:
                grid, elapsed_time, success = result
                memory_info = {
                    'peak_memory_mb': peak / 1024 / 1024,
                    'current_memory_mb': current / 1024 / 1024,
                    'memory_increase_mb': final_memory - initial_memory,
                    'initial_memory_mb': initial_memory,
                    'final_memory_mb': final_memory
                }
                return grid, elapsed_time, success, memory_info
            else:
                return result
        return wrapper

    # Solve with SAT
    @_track_memory
    def solve_with_pysat(self):
        start = time.time()
        timeout = 10
        
        gen = CNFGenerator(self.map.islands, self.map.connections)
        cnf, conn_vars = gen.generate()

        cnf.to_file("Hashi_CNF.cnf")

        unique_clauses = list({tuple(sorted(clause)) for clause in cnf.clauses})
        with Glucose3(bootstrap_with=[list(c) for c in unique_clauses]) as solver:
            for model in solver.enum_models():
                if time.time() - start > timeout:
                    return None, time.time() - start, False
                
                assignment = [0] * len(self.map.connections)
                for i, (v1, v2) in conn_vars.items():
                    if model[v1 - 1] > 0:
                        assignment[i] = 1
                    elif model[v2 - 1] > 0:
                        assignment[i] = 2

                if self.checker.is_valid(assignment):
                    return self.checker.to_grid(assignment), time.time() - start, True
        return None, time.time() - start, False
    
    # Solve with BruteForce
    @_track_memory
    def solve_with_bruteforce(self):
        start = time.time()
        timeout = 10

        gen = CNFGenerator(self.map.islands, self.map.connections)
        cnf, conn_vars = gen.generate()
        num_vars = max(abs(lit) for clause in cnf for lit in clause)

        cnf.to_file("Hashi_CNF.cnf")

        for bits in product([False, True], repeat=num_vars):
            if time.time() - start > timeout:
                print("[timeout] Brute-force failed")
                return None, time.time() - start, False

            assignment_dict = {i + 1: bits[i] for i in range(num_vars)}

            if all(any((lit > 0 and assignment_dict[lit]) or (lit < 0 and not assignment_dict[-lit]) for lit in clause) for clause in cnf):
                bridge_assignment = [0] * len(self.map.connections)
                for i, (v1, v2) in conn_vars.items():
                    if assignment_dict.get(v1, False):
                        bridge_assignment[i] = 1
                    elif assignment_dict.get(v2, False):
                        bridge_assignment[i] = 2

                if self.checker.is_valid(bridge_assignment):
                    return self.checker.to_grid(bridge_assignment), time.time() - start, True

        return None, time.time() - start, False
    
    # Solve with DPLL Backtracking
    @_track_memory
    def solve_with_backtracking(self):
        start = time.time()
        timeout = 10

        gen = CNFGenerator(self.map.islands, self.map.connections)
        cnf, conn_vars = gen.generate()
        clauses = [list(c) for c in cnf.clauses]

        cnf.to_file("Hashi_CNF.cnf")

        def unit_propagate(clauses, assignment):
            updated = True
            while updated:
                updated = False
                for clause in clauses:
                    if any(
                        (lit > 0 and assignment.get(abs(lit)) is True) or
                        (lit < 0 and assignment.get(abs(lit)) is False)
                        for lit in clause
                    ):
                        continue

                    unassigned_lits = [lit for lit in clause if abs(lit) not in assignment]
                    if not unassigned_lits:
                        return None, None

                    if len(unassigned_lits) == 1:
                        unit = unassigned_lits[0]
                        var = abs(unit)
                        val = unit > 0
                        if var in assignment and assignment[var] != val:
                            return None, None  
                        assignment[var] = val
                        updated = True
                        break  

            simplified_clauses = []
            for clause in clauses:
                if any(
                    (lit > 0 and assignment.get(abs(lit)) is True) or
                    (lit < 0 and assignment.get(abs(lit)) is False)
                    for lit in clause
                ):
                    continue 
                simplified = [lit for lit in clause if abs(lit) not in assignment]
                simplified_clauses.append(simplified)

            return simplified_clauses, assignment

        # DPLL function
        def dpll(clauses, assignment):
            if time.time() - start > timeout:
                print("[timeout] DPLL failed")
                return None

            new_clauses, new_assignment = unit_propagate(clauses, assignment)
            if new_clauses is None:
                return None

            if all(any((lit > 0 and new_assignment.get(abs(lit), False)) or
                    (lit < 0 and not new_assignment.get(abs(lit), False))
                    for lit in clause)
                for clause in new_clauses):
                return new_assignment

            all_vars = set(abs(lit) for clause in new_clauses for lit in clause)
            unassigned = list(all_vars - set(new_assignment.keys()))
            if not unassigned:
                return None

            chosen = unassigned[0]
            for value in [True, False]:
                trial = new_assignment.copy()
                trial[chosen] = value
                result = dpll(new_clauses, trial)
                if result is not None:
                    return result
            return None

        model = dpll(clauses, {})
        if model is None:
            return None, time.time() - start, False

        assignment = [0] * len(self.map.connections)
        for i, (v1, v2) in conn_vars.items():
            if model.get(v1, False):
                assignment[i] = 1
            elif model.get(v2, False):
                assignment[i] = 2

        if self.checker.is_valid(assignment):
            return self.checker.to_grid(assignment), time.time() - start, True
        return None, time.time() - start, False

    @_track_memory
    def solve_with_AStar(self):
        start = time.time()
        timeout = 2
        
        gen = CNFGenerator(self.map.islands, self.map.connections)
        cnf, conn_vars = gen.generate()
        clauses = [list(c) for c in cnf.clauses]
        num_vars = gen.var_counter - 1

        cnf.to_file("Hashi_CNF.cnf")

        def is_clause_satisfied(clause, assignment):
            for literal in clause:
                var = abs(literal)
                val = assignment.get(var, None)
                if val is not None:
                    if (literal > 0 and val) or (literal < 0 and not val):
                        return True
            return False

        def is_cnf_satisfied(cnf_clauses, assignment):
            return all(is_clause_satisfied(clause, assignment) for clause in cnf_clauses)

        def has_conflict(cnf_clauses, assignment):
            """Check if current assignment creates unsolvable conflicts"""
            for clause in cnf_clauses:
                satisfied = False
                has_unassigned = False
                
                for literal in clause:
                    var = abs(literal)
                    if var not in assignment:
                        has_unassigned = True
                    elif (literal > 0 and assignment[var]) or (literal < 0 and not assignment[var]):
                        satisfied = True
                        break
                
                if not satisfied and not has_unassigned:
                    return True
            return False

        def heuristic(assignment, total_vars):
            """Admissible heuristic: number of unassigned variables"""
            # This is admissible because we need at least 1 step per unassigned variable
            # This is consistent because h decreases by exactly 1 when we assign a variable
            return total_vars - len(assignment)

        def get_best_variable(assignment, total_vars):
            """Choose variable using Most Constraining Variable heuristic"""
            unassigned = []
            for var in range(1, total_vars + 1):
                if var not in assignment:
                    unassigned.append(var)
            
            if not unassigned:
                return None
            
            var_counts = {}
            for var in unassigned:
                var_counts[var] = 0
            
            for clause in clauses:
                if not is_clause_satisfied(clause, assignment):
                    for literal in clause:
                        var = abs(literal)
                        if var in unassigned:
                            var_counts[var] += 1
            
            best_var = unassigned[0]
            best_count = var_counts[best_var]
            
            for var in unassigned:
                if var_counts[var] > best_count:
                    best_var = var
                    best_count = var_counts[var]
            
            return best_var

        def insert_sorted(open_set, node):
            """Insert node in open set sorted by f = g + h"""
            f = node[0]
            i = 0
            while i < len(open_set) and open_set[i][0] < f:
                i += 1
            open_set.insert(i, node)

        # A* search without external libraries
        initial_assignment = {}
        h_score = heuristic(initial_assignment, num_vars)
        open_set = [(h_score, 0, initial_assignment)]  # (f_score, g_score, assignment)
        visited = set()
        model = None

        while open_set:
            if time.time() - start > timeout:
                print("[timeout] A* failed")
                return None, time.time() - start, False

            f_score, g_score, assignment = open_set.pop(0) 
            state_key = frozenset(assignment.items())

            if state_key in visited:
                continue
            visited.add(state_key)

            if has_conflict(clauses, assignment):
                continue

            if len(assignment) == num_vars:
                if is_cnf_satisfied(clauses, assignment):
                    model = assignment
                    break
                continue

            var = get_best_variable(assignment, num_vars)
            if var is None:
                continue

            for val in [True, False]:
                new_assignment = assignment.copy()
                new_assignment[var] = val
                
                new_state_key = frozenset(new_assignment.items())
                if new_state_key in visited:
                    continue

                if has_conflict(clauses, new_assignment):
                    continue

                h = heuristic(new_assignment, num_vars)
                g = g_score + 1
                f = g + h
                
                insert_sorted(open_set, (f, g, new_assignment))

        if model:
            bridge_assignment = [0] * len(self.map.connections)
            for i, (v1, v2) in conn_vars.items():
                if model.get(v1, False):
                    bridge_assignment[i] = 1
                elif model.get(v2, False):
                    bridge_assignment[i] = 2
            
            if self.checker.is_valid(bridge_assignment):
                return self.checker.to_grid(bridge_assignment), time.time() - start, True

        return None, time.time() - start, False

    def compare_algorithms(self, verbose=True):
        """
        Compare all algorithms and return performance metrics
        """
        algorithms = {
            'PySAT': self.solve_with_pysat,
            'Brute Force': self.solve_with_bruteforce,
            'DPLL Backtracking': self.solve_with_backtracking,
            'A* Search': self.solve_with_AStar
        }
        
        results = {}
        
        for name, method in algorithms.items():
            if verbose:
                print(f"\n--- Testing {name} ---")
            
            try:
                result = method()
                if len(result) == 4:
                    grid, time_taken, success, memory_info = result
                    results[name] = {
                        'success': success,
                        'time': time_taken,
                        'memory': memory_info,
                        'grid': grid
                    }
                    
                    if verbose:
                        status = "SUCCESS" if success else "FAILED"
                        print(f"Status: {status}")
                        print(f"Time: {time_taken:.4f}s")
                        print(f"Peak Memory: {memory_info['peak_memory_mb']:.2f} MB")
                        print(f"Memory Increase: {memory_info['memory_increase_mb']:.2f} MB")
                else:
                    grid, time_taken, success = result
                    results[name] = {
                        'success': success,
                        'time': time_taken,
                        'memory': {'error': 'Memory tracking failed'},
                        'grid': grid
                    }
                    
            except Exception as e:
                results[name] = {
                    'success': False,
                    'time': 0,
                    'memory': {'error': str(e)},
                    'grid': None
                }
                if verbose:
                    print(f"ERROR: {str(e)}")
        
        return results

    def print_performance_summary(self, results):
        """
        Print a formatted summary of algorithm performance
        """
        print("\n" + "="*80)
        print("ALGORITHM PERFORMANCE COMPARISON")
        print("="*80)
        print(f"{'Algorithm':<20} {'Status':<10} {'Time (s)':<12} {'Peak Memory (MB)':<18} {'Memory Δ (MB)':<15}")
        print("-"*80)
        
        for name, result in results.items():
            status = "SUCCESS" if result['success'] else "FAILED"
            time_str = f"{result['time']:.4f}"
            
            if 'error' in result['memory']:
                memory_str = "ERROR"
                delta_str = "ERROR"
            else:
                memory_str = f"{result['memory']['peak_memory_mb']:.2f}"
                delta_str = f"{result['memory']['memory_increase_mb']:.2f}"
            
            print(f"{name:<20} {status:<10} {time_str:<12} {memory_str:<18} {delta_str:<15}")
        
        successful_results = {k: v for k, v in results.items() if v['success']}
        
        if successful_results:
            print("\n" + "="*50)
            print("BEST PERFORMERS")
            print("="*50)
            
            fastest = min(successful_results.items(), key=lambda x: x[1]['time'])
            print(f"Fastest: {fastest[0]} ({fastest[1]['time']:.4f}s)")
            
            memory_results = {k: v for k, v in successful_results.items() 
                            if 'error' not in v['memory']}
            if memory_results:
                lowest_memory = min(memory_results.items(), 
                                  key=lambda x: x[1]['memory']['peak_memory_mb'])
                print(f"Lowest Memory: {lowest_memory[0]} ({lowest_memory[1]['memory']['peak_memory_mb']:.2f} MB)")