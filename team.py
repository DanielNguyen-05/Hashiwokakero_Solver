import numpy as np
import time
import heapq
from collections import defaultdict, deque
from itertools import combinations, product
from pysat.solvers import Glucose3
from pysat.formula import CNF
import os

class HashiwokakeroSolver:
    def __init__(self, grid):
        self.grid = np.array(grid)
        self.rows, self.cols = self.grid.shape
        self.islands = self._find_islands()
        self.variables = {}
        self.var_counter = 1
        
    def _find_islands(self):
        """Find all islands (non-zero cells) in the grid"""
        islands = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] > 0:
                    islands.append((i, j, self.grid[i][j]))
        return islands
    
    def _get_variable_id(self, var_name):
        """Get or create variable ID for CNF encoding"""
        if var_name not in self.variables:
            self.variables[var_name] = self.var_counter
            self.var_counter += 1
        return self.variables[var_name]
    
    def _find_possible_connections(self):
        """Find all possible bridge connections between islands"""
        connections = []
        
        for i, (r1, c1, val1) in enumerate(self.islands):
            for j, (r2, c2, val2) in enumerate(self.islands):
                if i >= j:  # Avoid duplicates
                    continue
                
                # Check if islands are in same row or column
                if r1 == r2:  # Horizontal connection
                    if self._is_path_clear(r1, c1, r2, c2, 'horizontal'):
                        connections.append(((r1, c1), (r2, c2), 'horizontal'))
                elif c1 == c2:  # Vertical connection
                    if self._is_path_clear(r1, c1, r2, c2, 'vertical'):
                        connections.append(((r1, c1), (r2, c2), 'vertical'))
        
        return connections
    
    def _is_path_clear(self, r1, c1, r2, c2, direction):
        """Check if path between two islands is clear"""
        if direction == 'horizontal':
            start_col, end_col = min(c1, c2), max(c1, c2)
            for c in range(start_col + 1, end_col):
                if self.grid[r1][c] != 0:
                    return False
        else:  # vertical
            start_row, end_row = min(r1, r2), max(r1, r2)
            for r in range(start_row + 1, end_row):
                if self.grid[r][c1] != 0:
                    return False
        
        return True
    
    def generate_cnf(self):
        """Generate CNF constraints for the Hashiwokakero puzzle"""
        cnf = CNF()
        self.connections = self._find_possible_connections()
        
        # Variables: bridge_i_j_k where i,j are island indices and k is bridge count (1 or 2)
        bridge_vars = {}
        for conn_idx, (island1, island2, direction) in enumerate(self.connections):
            # One bridge variable
            var_1 = self._get_variable_id(f"bridge_{conn_idx}_1")
            bridge_vars[(conn_idx, 1)] = var_1
            
            # Two bridges variable
            var_2 = self._get_variable_id(f"bridge_{conn_idx}_2")
            bridge_vars[(conn_idx, 2)] = var_2
            
            # Constraint: at most one of {1 bridge, 2 bridges} can be true
            # But both can be false (no bridge)
            cnf.append([-var_1, -var_2])  # Not both
        
        # Island degree constraints
        for island_idx, (r, c, required_bridges) in enumerate(self.islands):
            island_bridge_vars = []
            
            # Find all connections involving this island
            for conn_idx, (island1, island2, direction) in enumerate(self.connections):
                if island1 == (r, c) or island2 == (r, c):
                    # Add variables for 1 and 2 bridges with weights
                    island_bridge_vars.append((bridge_vars[(conn_idx, 1)], 1))
                    island_bridge_vars.append((bridge_vars[(conn_idx, 2)], 2))
            
            # Generate exactly-k constraints using combination method
            self._add_exactly_k_constraint(cnf, island_bridge_vars, required_bridges)
        
        # Bridge crossing constraints
        self._add_crossing_constraints(cnf, bridge_vars)
        
        return cnf, bridge_vars
    
    def _add_exactly_k_constraint(self, cnf, weighted_vars, k):
        """Add constraint that weighted sum of variables equals k"""
        if not weighted_vars:
            return
        
        # For small k, we can enumerate all valid combinations
        if k <= 8:  # Maximum island value
            valid_combinations = []
            
            # Generate all possible assignments
            for assignment in product([0, 1], repeat=len(weighted_vars)):
                total = sum(weight * val for (var, weight), val in zip(weighted_vars, assignment))
                if total == k:
                    valid_combinations.append(assignment)
            
            if not valid_combinations:
                # Unsatisfiable
                cnf.append([])
                return
            
            # Convert to CNF clauses
            # At least one valid combination must be true
            for assignment in valid_combinations:
                clause = []
                for i, val in enumerate(assignment):
                    var, weight = weighted_vars[i]
                    if val == 1:
                        clause.append(var)
                    else:
                        clause.append(-var)
                # This creates a large clause, we need to handle it differently
            
            # Better approach: use auxiliary variables for cardinality constraints
            self._add_cardinality_constraint(cnf, weighted_vars, k)
    
    def _add_cardinality_constraint(self, cnf, weighted_vars, k):
        """Add cardinality constraint using auxiliary variables"""
        # Simplified version: for each possible sum value, create constraints
        vars_only = [var for var, weight in weighted_vars]
        weights = [weight for var, weight in weighted_vars]
        
        # Create auxiliary variables for partial sums
        n = len(weighted_vars)
        max_sum = sum(weights)
        
        # aux[i][s] = true if first i variables sum to s
        aux_vars = {}
        
        for i in range(n + 1):
            for s in range(max_sum + 1):
                aux_vars[(i, s)] = self._get_variable_id(f"aux_{i}_{s}")
        
        # Base case: aux[0][0] = true, aux[0][s] = false for s > 0
        cnf.append([aux_vars[(0, 0)]])
        for s in range(1, max_sum + 1):
            cnf.append([-aux_vars[(0, s)]])
        
        # Recursive case: aux[i][s] = (aux[i-1][s] ∧ ¬x[i]) ∨ (aux[i-1][s-w[i]] ∧ x[i])
        for i in range(1, n + 1):
            var_i = vars_only[i-1]
            weight_i = weights[i-1]
            
            for s in range(max_sum + 1):
                aux_curr = aux_vars[(i, s)]
                
                # Case 1: don't take variable i
                aux_prev_same = aux_vars[(i-1, s)]
                cnf.append([-aux_curr, aux_prev_same, var_i])
                cnf.append([-aux_prev_same, -var_i, aux_curr])
                
                # Case 2: take variable i
                if s >= weight_i:
                    aux_prev_diff = aux_vars[(i-1, s - weight_i)]
                    cnf.append([-aux_curr, aux_prev_diff, -var_i])
                    cnf.append([-aux_prev_diff, var_i, aux_curr])
                else:
                    # Can't take variable i if it would exceed sum
                    cnf.append([-aux_curr, -var_i])
        
        # Final constraint: aux[n][k] must be true
        cnf.append([aux_vars[(n, k)]])
    
    def _add_crossing_constraints(self, cnf, bridge_vars):
        """Add constraints to prevent bridge crossings"""
        for i, (island1_a, island2_a, dir_a) in enumerate(self.connections):
            for j, (island1_b, island2_b, dir_b) in enumerate(self.connections):
                if i >= j:
                    continue
                
                if self._bridges_cross(island1_a, island2_a, island1_b, island2_b):
                    # These bridges cannot both exist
                    for k1 in [1, 2]:
                        for k2 in [1, 2]:
                            var_a = bridge_vars[(i, k1)]
                            var_b = bridge_vars[(j, k2)]
                            cnf.append([-var_a, -var_b])
    
    def _bridges_cross(self, island1_a, island2_a, island1_b, island2_b):
        """Check if two bridges would cross each other"""
        r1_a, c1_a = island1_a
        r2_a, c2_a = island2_a
        r1_b, c1_b = island1_b
        r2_b, c2_b = island2_b
        
        # Check if one is horizontal and other is vertical
        is_horizontal_a = r1_a == r2_a
        is_horizontal_b = r1_b == r2_b
        
        if is_horizontal_a == is_horizontal_b:
            return False  # Both horizontal or both vertical - no crossing
        
        if is_horizontal_a:  # A is horizontal, B is vertical
            # Check if they intersect
            min_col_a, max_col_a = min(c1_a, c2_a), max(c1_a, c2_a)
            min_row_b, max_row_b = min(r1_b, r2_b), max(r1_b, r2_b)
            
            return (min_col_a < c1_b < max_col_a and min_row_b < r1_a < max_row_b)
        else:  # A is vertical, B is horizontal
            min_row_a, max_row_a = min(r1_a, r2_a), max(r1_a, r2_a)
            min_col_b, max_col_b = min(c1_b, c2_b), max(c1_b, c2_b)
            
            return (min_row_a < r1_b < max_row_a and min_col_b < c1_a < max_col_b)
    
    def solve_with_pysat(self):
        """Solve using PySAT library"""
        start_time = time.time()
        
        cnf, bridge_vars = self.generate_cnf()
        
        solver = Glucose3()
        for clause in cnf.clauses:
            solver.add_clause(clause)
        
        if solver.solve():
            model = solver.get_model()
            solution = self._extract_solution(model, bridge_vars)
            end_time = time.time()
            return solution, end_time - start_time, True
        else:
            end_time = time.time()
            return None, end_time - start_time, False
    
    def solve_with_astar(self):
        """Solve using A* search algorithm"""
        start_time = time.time()
        
        initial_state = self._create_initial_state()
        goal_state = self._astar_search(initial_state)
        
        end_time = time.time()
        
        if goal_state:
            solution = self._state_to_solution(goal_state)
            return solution, end_time - start_time, True
        else:
            return None, end_time - start_time, False
    
    def _create_initial_state(self):
        """Create initial state for A* search"""
        # State: dictionary mapping connection indices to bridge counts (0, 1, or 2)
        connections = self._find_possible_connections()
        return {i: 0 for i in range(len(connections))}
    
    def _astar_search(self, initial_state):
        """A* search implementation"""
        open_set = [(self._heuristic(initial_state), 0, initial_state)]
        closed_set = set()
        g_scores = {str(initial_state): 0}
        
        while open_set:
            f_score, g_score, current_state = heapq.heappop(open_set)
            
            if str(current_state) in closed_set:
                continue
            
            closed_set.add(str(current_state))
            
            if self._is_goal_state(current_state):
                return current_state
            
            for next_state in self._get_neighbors(current_state):
                next_state_str = str(next_state)
                
                if next_state_str in closed_set:
                    continue
                
                tentative_g = g_score + 1
                
                if next_state_str not in g_scores or tentative_g < g_scores[next_state_str]:
                    g_scores[next_state_str] = tentative_g
                    f_score = tentative_g + self._heuristic(next_state)
                    heapq.heappush(open_set, (f_score, tentative_g, next_state))
        
        return None
    
    def _heuristic(self, state):
        """Heuristic function for A* search"""
        # Calculate how far each island is from its required bridge count
        total_deficit = 0
        
        for island_idx, (r, c, required) in enumerate(self.islands):
            current_bridges = 0
            
            for conn_idx, bridge_count in state.items():
                conn = self.connections[conn_idx]
                if (r, c) in [conn[0], conn[1]]:
                    current_bridges += bridge_count
            
            deficit = abs(required - current_bridges)
            total_deficit += deficit
        
        return total_deficit
    
    def _is_goal_state(self, state):
        """Check if state satisfies all constraints"""
        # Check island bridge counts
        for island_idx, (r, c, required) in enumerate(self.islands):
            current_bridges = 0
            
            for conn_idx, bridge_count in state.items():
                conn = self.connections[conn_idx]
                if (r, c) in [conn[0], conn[1]]:
                    current_bridges += bridge_count
            
            if current_bridges != required:
                return False
        
        # Check connectivity (simplified - could be more thorough)
        return self._is_connected(state)
    
    def _is_connected(self, state):
        """Check if all islands are connected"""
        # Build graph of connected islands
        graph = defaultdict(list)
        
        for conn_idx, bridge_count in state.items():
            if bridge_count > 0:
                conn = self.connections[conn_idx]
                island1, island2 = conn[0], conn[1]
                graph[island1].append(island2)
                graph[island2].append(island1)
        
        if not graph:
            return len(self.islands) <= 1
        
        # BFS to check connectivity
        start_island = (self.islands[0][0], self.islands[0][1])
        visited = set()
        queue = deque([start_island])
        visited.add(start_island)
        
        while queue:
            current = queue.popleft()
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == len(self.islands)
    
    def _get_neighbors(self, state):
        """Get neighboring states for A* search"""
        neighbors = []
        
        for conn_idx in range(len(self.connections)):
            current_bridges = state[conn_idx]
            
            # Try increasing bridge count
            if current_bridges < 2:
                new_state = state.copy()
                new_state[conn_idx] = current_bridges + 1
                
                if self._is_valid_state(new_state):
                    neighbors.append(new_state)
            
            # Try decreasing bridge count
            if current_bridges > 0:
                new_state = state.copy()
                new_state[conn_idx] = current_bridges - 1
                
                if self._is_valid_state(new_state):
                    neighbors.append(new_state)
        
        return neighbors
    
    def _is_valid_state(self, state):
        """Check if state is valid (doesn't violate constraints)"""
        # Check that no island exceeds its required bridge count
        for island_idx, (r, c, required) in enumerate(self.islands):
            current_bridges = 0
            
            for conn_idx, bridge_count in state.items():
                conn = self.connections[conn_idx]
                if (r, c) in [conn[0], conn[1]]:
                    current_bridges += bridge_count
            
            if current_bridges > required:
                return False
        
        # Check for bridge crossings
        active_connections = [(i, count) for i, count in state.items() if count > 0]
        
        for i, (conn_idx_a, count_a) in enumerate(active_connections):
            for j, (conn_idx_b, count_b) in enumerate(active_connections):
                if i >= j:
                    continue
                
                conn_a = self.connections[conn_idx_a]
                conn_b = self.connections[conn_idx_b]
                
                if self._bridges_cross(conn_a[0], conn_a[1], conn_b[0], conn_b[1]):
                    return False
        
        return True
    
    def solve_with_backtracking(self):
        """Solve using backtracking algorithm"""
        start_time = time.time()
        
        connections = self._find_possible_connections()
        assignment = [0] * len(connections)  # 0, 1, or 2 bridges for each connection
        
        result = self._backtrack(assignment, 0, connections)
        
        end_time = time.time()
        
        if result:
            solution = {i: assignment[i] for i in range(len(connections))}
            return self._state_to_solution(solution), end_time - start_time, True
        else:
            return None, end_time - start_time, False
    
    def _backtrack(self, assignment, conn_idx, connections):
        """Backtracking recursive function"""
        if conn_idx == len(connections):
            return self._is_complete_solution(assignment, connections)
        
        # Try 0, 1, 2 bridges for current connection
        for bridges in [0, 1, 2]:
            assignment[conn_idx] = bridges
            
            if self._is_partial_assignment_valid(assignment, conn_idx, connections):
                if self._backtrack(assignment, conn_idx + 1, connections):
                    return True
            
            assignment[conn_idx] = 0  # Backtrack
        
        return False
    
    def _is_partial_assignment_valid(self, assignment, up_to_idx, connections):
        """Check if partial assignment is valid"""
        # Check island constraints don't exceed requirements
        island_bridges = defaultdict(int)
        
        for i in range(up_to_idx + 1):
            if assignment[i] > 0:
                conn = connections[i]
                island_bridges[conn[0]] += assignment[i]
                island_bridges[conn[1]] += assignment[i]
        
        for island_pos, bridges in island_bridges.items():
            required = None
            for r, c, req in self.islands:
                if (r, c) == island_pos:
                    required = req
                    break
            
            if required is not None and bridges > required:
                return False
        
        # Check crossings
        active_connections = [(i, assignment[i]) for i in range(up_to_idx + 1) if assignment[i] > 0]
        
        for i, (conn_idx_a, count_a) in enumerate(active_connections):
            for j, (conn_idx_b, count_b) in enumerate(active_connections):
                if i >= j:
                    continue
                
                conn_a = connections[conn_idx_a]
                conn_b = connections[conn_idx_b]
                
                if self._bridges_cross(conn_a[0], conn_a[1], conn_b[0], conn_b[1]):
                    return False
        
        return True
    
    def _is_complete_solution(self, assignment, connections):
        """Check if complete assignment is a valid solution"""
        state = {i: assignment[i] for i in range(len(connections))}
        return self._is_goal_state(state)
    
    def solve_with_bruteforce(self):
        """Solve using brute force algorithm"""
        start_time = time.time()
        
        connections = self._find_possible_connections()
        n_connections = len(connections)
        
        # Try all possible combinations (3^n possibilities)
        for assignment in product([0, 1, 2], repeat=n_connections):
            state = {i: assignment[i] for i in range(n_connections)}
            
            if self._is_goal_state(state):
                end_time = time.time()
                solution = self._state_to_solution(state)
                return solution, end_time - start_time, True
        
        end_time = time.time()
        return None, end_time - start_time, False
    
    def _extract_solution(self, model, bridge_vars):
        """Extract solution from SAT model"""
        solution = {}
        
        for (conn_idx, bridge_count), var_id in bridge_vars.items():
            if var_id in model and model[var_id - 1] > 0:  # PySAT uses 1-based indexing
                solution[conn_idx] = bridge_count
        
        # Fill in missing connections with 0 bridges
        for i in range(len(self.connections)):
            if i not in solution:
                solution[i] = 0
        
        return self._state_to_solution(solution)
    
    def _state_to_solution(self, state):
        """Convert state to solution grid format"""
        solution_grid = [['0' if self.grid[i][j] == 0 else str(self.grid[i][j]) 
                         for j in range(self.cols)] for i in range(self.rows)]
        
        for conn_idx, bridge_count in state.items():
            if bridge_count > 0:
                conn = self.connections[conn_idx]
                island1, island2, direction = conn[0], conn[1], conn[2]
                
                r1, c1 = island1
                r2, c2 = island2
                
                if direction == 'horizontal':
                    start_col, end_col = min(c1, c2), max(c1, c2)
                    bridge_char = '=' if bridge_count == 2 else '-'
                    
                    for c in range(start_col + 1, end_col):
                        solution_grid[r1][c] = bridge_char
                
                else:  # vertical
                    start_row, end_row = min(r1, r2), max(r1, r2)
                    bridge_char = '$' if bridge_count == 2 else '|'
                    
                    for r in range(start_row + 1, end_row):
                        solution_grid[r][c1] = bridge_char
        
        return solution_grid

def load_input(filename):
    """Load input from file"""
    grid = []
    with open(filename, 'r') as f:
        for line in f:
            row = [int(x.strip()) for x in line.strip().split(',')]
            grid.append(row)
    return grid

def save_output(solution, filename):
    """Save solution to file"""
    with open(filename, 'w') as f:
        for row in solution:
            f.write('[' + ', '.join(f'"{cell}"' for cell in row) + ']\n')

def main():
    """Main function to test all algorithms"""
    # Test with example input
    # test_grid = [
    #     [0, 2, 0, 5, 0, 0, 2],
    #     [0, 0, 0, 0, 0, 0, 0],
    #     [4, 0, 2, 0, 2, 0, 4],
    #     [0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 0, 5, 0, 2, 0],
    #     [0, 0, 0, 0, 0, 0, 0],
    #     [4, 0, 0, 0, 0, 0, 3]
    # ]

    input_filename = 'input.txt'
    output_filename = 'output.txt'

    test_grid = load_input(input_filename)
    
    solver = HashiwokakeroSolver(test_grid)
    
    print("Testing Hashiwokakero Solver")
    print("=" * 40)
    
    # Test PySAT
    print("1. PySAT Solver:")
    try:
        solution, time_taken, success = solver.solve_with_pysat()
        if success:
            print(f"   Success! Time: {time_taken:.4f}s")
            for row in solution:
                print(f"   {row}")
            save_output(solution, output_filename)
        else:
            print(f"   Failed to find solution. Time: {time_taken:.4f}s")
    except Exception as e:
        print(f"   Error: {e}")
    
    print()
    
    # Test A*
    print("2. A* Solver:")
    try:
        solution, time_taken, success = solver.solve_with_astar()
        if success:
            print(f"   Success! Time: {time_taken:.4f}s")
            for row in solution:
                print(f"   {row}")
            save_output(solution, output_filename)
            
        else:
            print(f"   Failed to find solution. Time: {time_taken:.4f}s")
    except Exception as e:
        print(f"   Error: {e}")
    
    print()
    
    # Test Backtracking
    print("3. Backtracking Solver:")
    try:
        solution, time_taken, success = solver.solve_with_backtracking()
        if success:
            print(f"   Success! Time: {time_taken:.4f}s")
            for row in solution:
                print(f"   {row}")
            save_output(solution, output_filename)
        else:
            print(f"   Failed to find solution. Time: {time_taken:.4f}s")
    except Exception as e:
        print(f"   Error: {e}")
    
    print()
    
    # Test Brute Force (only for small grids)
    if len(solver.connections) <= 10:  # Limit brute force to small problems
        print("4. Brute Force Solver:")
        try:
            solution, time_taken, success = solver.solve_with_bruteforce()
            if success:
                print(f"   Success! Time: {time_taken:.4f}s")
                for row in solution:
                    print(f"   {row}")
                save_output(solution, output_filename)
            else:
                print(f"   Failed to find solution. Time: {time_taken:.4f}s")
        except Exception as e:
            print(f"   Error: {e}")
    else:
        print("4. Brute Force Solver: Skipped (too many connections)")

if __name__ == "__main__":
    main()