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
        self.map = IslandMap(grid) # The map
        self.checker = SolutionChecker(self.map.islands, self.map.connections, self.map.grid.shape) # Solution checker

    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB

    def _track_memory(func):
        """Decorator to track memory usage for solving methods"""
        def wrapper(self, *args, **kwargs):
            # Start memory tracking
            tracemalloc.start()
            initial_memory = self._get_memory_usage()
            
            # Execute the function
            result = func(self, *args, **kwargs)
            
            # Get memory statistics
            current, peak = tracemalloc.get_traced_memory()
            final_memory = self._get_memory_usage()
            tracemalloc.stop()
            
            # Unpack original result
            if len(result) == 3:
                grid, elapsed_time, success = result
                # Return extended result with memory info
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

        # Assign false or true for each logical variables
        for bits in product([False, True], repeat=num_vars):
            # Check whether it is time exceed or not
            if time.time() - start > timeout:
                print("[timeout] Brute-force failed")
                return None, time.time() - start, False

            assignment_dict = {i + 1: bits[i] for i in range(num_vars)}

            # Check whether it satisfies CNF or not
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

        # Function for unit propagate
        def unit_propagate(clauses, assignment):
            updated = True
            while updated:
                updated = False
                for clause in clauses:
                    # Clause is already satisfied
                    if any(
                        (lit > 0 and assignment.get(abs(lit)) is True) or
                        (lit < 0 and assignment.get(abs(lit)) is False)
                        for lit in clause
                    ):
                        continue

                    unassigned_lits = [lit for lit in clause if abs(lit) not in assignment]
                    # Conflict when there is no literal left
                    if not unassigned_lits:
                        return None, None

                    # Unit clause
                    if len(unassigned_lits) == 1:
                        unit = unassigned_lits[0]
                        var = abs(unit)
                        val = unit > 0
                        if var in assignment and assignment[var] != val:
                            return None, None  
                        assignment[var] = val
                        updated = True
                        break  

            # Simplify for new assignment
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

            # Check whether all clauses are satisfied under the current assignment
            if all(any((lit > 0 and new_assignment.get(abs(lit), False)) or
                    (lit < 0 and not new_assignment.get(abs(lit), False))
                    for lit in clause)
                for clause in new_clauses):
                return new_assignment

            # Get unassigned variables
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
        """
        Solve Hashiwokakero puzzle using A* search algorithm with improved heuristics.
        """
        start = time.time()
        timeout = 10
        
        def calculate_improved_heuristic(assignment):
            """
            More sophisticated admissible heuristic combining multiple factors.
            """
            current_degrees = defaultdict(int)
            unassigned_connections = 0
            
            for i, count in enumerate(assignment):
                if count == -1:
                    unassigned_connections += 1
                elif count > 0:
                    p1, p2 = self.map.connections[i]
                    current_degrees[p1] += count
                    current_degrees[p2] += count
            
            total_deficit = 0
            
            for pos, required_degree in self.map.islands:
                current = current_degrees[pos]
                if current > required_degree:
                    return float('inf')  # Invalid state
                
                deficit = required_degree - current
                total_deficit += deficit
                
                # Check if it's impossible to satisfy this island's requirement
                max_possible = current
                for i, conn in enumerate(self.map.connections):
                    if assignment[i] == -1 and pos in conn:
                        max_possible += 2  # Maximum bridges per connection
                
                if max_possible < required_degree:
                    return float('inf')
            
            # Primary heuristic: minimum bridges needed
            bridge_heuristic = (total_deficit + 1) // 2
            
            # Secondary heuristic: encourage progress (fewer unassigned connections)
            progress_heuristic = unassigned_connections * 0.1
            
            return bridge_heuristic + progress_heuristic
        
        def calculate_connection_priority(assignment):
            """
            Smart connection selection: prioritize connections involving constrained islands.
            """
            island_constraints = {}
            for pos, required_degree in self.map.islands:
                current = sum(assignment[i] for i, conn in enumerate(self.map.connections) 
                            if assignment[i] > 0 and pos in conn)
                remaining = required_degree - current
                max_possible = sum(2 for i, conn in enumerate(self.map.connections) 
                                if assignment[i] == -1 and pos in conn)
                
                # Constraint ratio: how tight the constraint is
                constraint_ratio = remaining / max(max_possible, 1)
                island_constraints[pos] = constraint_ratio
            
            # Find unassigned connection with highest constraint
            best_conn = None
            best_priority = -1
            
            for i, val in enumerate(assignment):
                if val == -1:
                    p1, p2 = self.map.connections[i]
                    priority = island_constraints.get(p1, 0) + island_constraints.get(p2, 0)
                    if priority > best_priority:
                        best_priority = priority
                        best_conn = i
            
            return best_conn if best_conn is not None else next(
                (i for i, val in enumerate(assignment) if val == -1), None
            )
        
        def bridges_cross(conn1, conn2):
            """Optimized bridge crossing check."""
            (r1a, c1a), (r1b, c1b) = conn1
            (r2a, c2a), (r2b, c2b) = conn2
            
            # Quick rejection: if connections share an endpoint
            if (r1a, c1a) in [(r2a, c2a), (r2b, c2b)] or (r1b, c1b) in [(r2a, c2a), (r2b, c2b)]:
                return False
            
            h1, h2 = r1a == r1b, r2a == r2b
            if h1 == h2:  # Both horizontal or both vertical
                return False
            
            if h1:  # conn1 horizontal, conn2 vertical
                return (min(c1a, c1b) < c2a < max(c1a, c1b)) and (min(r2a, r2b) < r1a < max(r2a, r2b))
            else:  # conn1 vertical, conn2 horizontal
                return (min(r1a, r1b) < r2a < max(r1a, r1b)) and (min(c2a, c2b) < c1a < max(c2a, c2b))
        
        def is_valid_partial_assignment(assignment, new_conn_idx, new_bridges):
            """
            Optimized validity check - only check new bridge against existing ones.
            """
            if new_bridges == 0:
                return True
            
            new_conn = self.map.connections[new_conn_idx]
            for i, bridges in enumerate(assignment):
                if i != new_conn_idx and bridges > 0:
                    if bridges_cross(new_conn, self.map.connections[i]):
                        return False
            return True
        
        def get_valid_bridge_counts(assignment, conn_idx):
            """
            Return valid bridge counts for a connection based on island constraints.
            """
            valid_counts = []
            
            for bridges in [0, 1, 2]:
                # Check if this assignment would violate island degree constraints
                temp_assignment = list(assignment)
                temp_assignment[conn_idx] = bridges
                
                if is_valid_partial_assignment(assignment, conn_idx, bridges):
                    # Quick degree check
                    degrees = defaultdict(int)
                    for i, count in enumerate(temp_assignment):
                        if count > 0:
                            conn_p1, conn_p2 = self.map.connections[i]
                            degrees[conn_p1] += count
                            degrees[conn_p2] += count
                    
                    valid = True
                    for pos, required in self.map.islands:
                        if degrees[pos] > required:
                            valid = False
                            break
                    
                    if valid:
                        valid_counts.append(bridges)
            
            return valid_counts
        
        # Initialize search with better state management
        num_connections = len(self.map.connections)
        initial_state = tuple([-1] * num_connections)
        
        # Use a more sophisticated priority queue entry: (f_score, tie_breaker, g_score, state)
        open_set = []
        tie_breaker = 0
        heapq.heappush(open_set, (calculate_improved_heuristic(initial_state), tie_breaker, 0, initial_state))
        
        # Track best g_score for each state to avoid redundant exploration
        g_scores = {initial_state: 0}
        visited = set()
        
        while open_set:
            if time.time() - start > timeout:
                return None, time.time() - start, False
            
            f_score, _, g_score, current_state = heapq.heappop(open_set)
            
            if current_state in visited:
                continue
            visited.add(current_state)
            
            # Smart connection selection
            next_conn = calculate_connection_priority(current_state)
            
            # Goal test: if no more connections to assign
            if next_conn is None:
                assignment = list(current_state)
                if self.checker.is_valid(assignment):
                    return self.checker.to_grid(assignment), time.time() - start, True
                continue
            
            # Get valid bridge counts for this connection
            valid_bridge_counts = get_valid_bridge_counts(current_state, next_conn)
            
            for bridges in valid_bridge_counts:
                new_state = list(current_state)
                new_state[next_conn] = bridges
                new_state_tuple = tuple(new_state)
                
                if new_state_tuple in visited:
                    continue
                
                # Improved cost function: actual number of bridges placed
                new_g_score = g_score + bridges
                
                # Skip if we've found a better path to this state
                if new_state_tuple in g_scores and g_scores[new_state_tuple] <= new_g_score:
                    continue
                
                g_scores[new_state_tuple] = new_g_score
                
                h_score = calculate_improved_heuristic(new_state)
                if h_score == float('inf'):
                    continue
                
                new_f_score = new_g_score + h_score
                tie_breaker += 1
                
                heapq.heappush(open_set, (new_f_score, tie_breaker, new_g_score, new_state_tuple))
        
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
        
        # Find best performers
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