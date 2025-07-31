# helper_01.py
import time
import heapq
import numpy as np
from collections import defaultdict, deque
from itertools import product, combinations
from pysat.solvers import Glucose3
from pysat.formula import CNF
import os

def load_input(filename):
    """Loads a puzzle grid from a comma-separated text file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Input file not found: {filename}")
    grid = []
    with open(filename, 'r') as f:
        for line in f:
            row = [int(x.strip()) for x in line.strip().split(',')]
            grid.append(row)
    return grid

def save_output(solution, filename):
    """Saves a solution grid to a file in the specified format."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for i, row in enumerate(solution):
            f.write(str(row))
            if i < len(solution) - 1:
                f.write('\n')

class HashiwokakeroSolver:
    """
    A solver for Hashiwokakero puzzles using various algorithms:
    - SAT-based (via PySAT)
    - A* Search
    - Backtracking
    - Brute Force
    """
    def __init__(self, grid):
        self.grid = np.array(grid, dtype=int)
        self.rows, self.cols = self.grid.shape
        self.islands = self._find_islands()
        self.connections = self._find_possible_connections()
        
        # For CNF generation
        self.variables = {}
        self.var_counter = 1
        self.aux_var_counter = 1

    def _find_islands(self):
        islands = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] > 0:
                    # Store as tuple: (row, col, required_degree)
                    islands.append(((r, c), self.grid[r, c]))
        return islands

    def _is_path_clear(self, r1, c1, r2, c2):
        path_coords = []
        if r1 == r2: # Horizontal
            for c in range(min(c1, c2) + 1, max(c1, c2)):
                path_coords.append((r1, c))
        else: # Vertical
            for r in range(min(r1, r2) + 1, max(r1, r2)):
                path_coords.append((r, c1))
        
        # Check if any island is on the path
        island_positions = {pos for pos, val in self.islands}
        return not any(coord in island_positions for coord in path_coords)

    def _find_possible_connections(self):
        connections = []
        for i in range(len(self.islands)):
            for j in range(i + 1, len(self.islands)):
                (pos1, _), (pos2, _) = self.islands[i], self.islands[j]
                r1, c1 = pos1
                r2, c2 = pos2

                if r1 == r2 or c1 == c2: # Same row or column
                    if self._is_path_clear(r1, c1, r2, c2):
                        connections.append((pos1, pos2))
        return connections

    def _bridges_cross(self, conn1, conn2):
        (p1a, p1b), (p2a, p2b) = conn1, conn2
        r1a, c1a = p1a
        r1b, c1b = p1b
        r2a, c2a = p2a
        r2b, c2b = p2b
        
        is_h1 = (r1a == r1b)
        is_h2 = (r2a == r2b)

        if is_h1 == is_h2: # Parallel bridges cannot cross
            return False
        
        if is_h1: # conn1 is horizontal, conn2 is vertical
            h_r, v_c = r1a, c2a
            h_c1, h_c2 = min(c1a, c1b), max(c1a, c1b)
            v_r1, v_r2 = min(r2a, r2b), max(r2a, r2b)
            return (h_c1 < v_c < h_c2) and (v_r1 < h_r < v_r2)
        else: # conn1 is vertical, conn2 is horizontal
            v_r, h_c = r1a, c2a
            v_r1, v_r2 = min(r1a, r1b), max(r1a, r1b)
            h_c1, h_c2 = min(c2a, c2b), max(c2a, c2b)
            return (v_r1 < h_c < v_r2) and (h_c1 < v_r < h_c2)

    def _get_variable_id(self, name):
        if name not in self.variables:
            self.variables[name] = self.var_counter
            self.var_counter += 1
        return self.variables[name]

    def _add_exactly_k_constraint(self, cnf, literals, k):
        """Adds CNF clauses for sum(literals) == k."""
        # At most k: for any k+1 literals, at least one must be false.
        for combo in combinations(literals, k + 1):
            cnf.append([-lit for lit in combo])
        
        # At least k: for any n-k+1 literals, at least one must be true.
        for combo in combinations(literals, len(literals) - k + 1):
            cnf.append(list(combo))

    def generate_cnf(self):
        """Generates the full CNF formulation for the puzzle."""
        self.variables.clear()
        self.var_counter = 1
        cnf = CNF()
        
        # Create variables for each connection: conn_i_1 (1 bridge), conn_i_2 (2 bridges)
        conn_vars = {}
        for i, conn in enumerate(self.connections):
            v1 = self._get_variable_id(f"conn_{i}_1")
            v2 = self._get_variable_id(f"conn_{i}_2")
            conn_vars[i] = (v1, v2)
            # A connection cannot have both 1 and 2 bridges.
            cnf.append([-v1, -v2])

        # Island degree constraints
        for pos, required_degree in self.islands:
            incident_literals = []
            for i, conn in enumerate(self.connections):
                if pos in conn:
                    v1, v2 = conn_vars[i]
                    # The degree is 1*v1 + 2*v2
                    incident_literals.append((v1, 1))
                    incident_literals.append((v2, 2))
            
            # Find all valid assignments for this island
            possible_assignments = []
            for p in product([0, 1], repeat=len(incident_literals)):
                degree = sum(p[i] * incident_literals[i][1] for i in range(len(p)))
                if degree == required_degree:
                    clause = []
                    for i in range(len(p)):
                        lit = incident_literals[i][0]
                        clause.append(lit if p[i] == 1 else -lit)
                    possible_assignments.append(clause)
            
            if not possible_assignments:
                cnf.append([]) # Unsatisfiable
                continue

            # Convert DNF of assignments to CNF
            aux_vars = []
            for assignment_clause in possible_assignments:
                aux_var = self._get_variable_id(f"aux_island_{self.aux_var_counter}")
                self.aux_var_counter += 1
                aux_vars.append(aux_var)
                # assignment_clause => aux_var
                for lit in assignment_clause:
                    cnf.append([-lit, aux_var])
            # At least one assignment must be true
            cnf.append(aux_vars)

        # Bridge crossing constraints
        for i in range(len(self.connections)):
            for j in range(i + 1, len(self.connections)):
                if self._bridges_cross(self.connections[i], self.connections[j]):
                    v1_i, v2_i = conn_vars[i]
                    v1_j, v2_j = conn_vars[j]
                    # If conn i has any bridge, conn j cannot have any.
                    cnf.append([-v1_i, -v1_j])
                    cnf.append([-v1_i, -v2_j])
                    cnf.append([-v2_i, -v1_j])
                    cnf.append([-v2_i, -v2_j])
        
        # Connectivity constraint is hard to encode in CNF and often omitted.
        # Most puzzles are solvable without it. We will check it post-solution.
        
        return cnf, conn_vars

    def _is_solution_fully_connected(self, assignment):
        """Checks if the graph of islands and bridges is connected."""
        if not self.islands:
            return True
        
        adj = defaultdict(list)
        active_islands = set()
        for i, count in enumerate(assignment):
            if count > 0:
                p1, p2 = self.connections[i]
                adj[p1].append(p2)
                adj[p2].append(p1)
                active_islands.add(p1)
                active_islands.add(p2)

        if not active_islands:
            return len(self.islands) <= 1

        q = deque([next(iter(active_islands))])
        visited = {next(iter(active_islands))}
        
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        
        return visited == {pos for pos, val in self.islands}

    def _assignment_to_grid(self, assignment):
        """Converts a valid assignment to the output grid format."""
        grid = np.full(self.grid.shape, ' ', dtype=object)
        for (r, c), val in self.islands:
            grid[r, c] = str(val)

        for i, count in enumerate(assignment):
            if count > 0:
                (r1, c1), (r2, c2) = self.connections[i]
                if r1 == r2: # Horizontal
                    char = '-' if count == 1 else '='
                    for c in range(min(c1, c2) + 1, max(c1, c2)):
                        grid[r1, c] = char
                else: # Vertical
                    char = '|' if count == 1 else '$'
                    for r in range(min(r1, r2) + 1, max(r1, r2)):
                        grid[r, c1] = char
        return grid.tolist()

    def _check_assignment(self, assignment):
        """Checks if an assignment is a valid and complete solution."""
        # Check degrees
        degrees = defaultdict(int)
        for i, count in enumerate(assignment):
            if count > 0:
                p1, p2 = self.connections[i]
                degrees[p1] += count
                degrees[p2] += count
        
        for pos, required_degree in self.islands:
            if degrees[pos] != required_degree:
                return False
        
        # Check crossings (already implicitly handled by search algos)

        # Check connectivity
        if not self._is_solution_fully_connected(assignment):
            return False
            
        return True

    def solve_with_pysat(self):
        """Solves the puzzle using a SAT solver."""
        start_time = time.time()
        cnf, conn_vars = self.generate_cnf()
        
        with Glucose3(bootstrap_with=cnf.clauses) as solver:
            # Iterate through solutions to find a connected one
            for model in solver.enum_models():
                assignment = [0] * len(self.connections)
                for i, (v1, v2) in conn_vars.items():
                    if model[v1 - 1] > 0:
                        assignment[i] = 1
                    elif model[v2 - 1] > 0:
                        assignment[i] = 2

                if self._is_solution_fully_connected(assignment):
                    solution_grid = self._assignment_to_grid(assignment)
                    time_taken = time.time() - start_time
                    return solution_grid, time_taken, True
        
        time_taken = time.time() - start_time
        return None, time_taken, False

    def solve_with_bruteforce(self):
        """Solves the puzzle by checking every possible configuration."""
        start_time = time.time()
        num_connections = len(self.connections)

        for assignment_tuple in product([0, 1, 2], repeat=num_connections):
            assignment = list(assignment_tuple)
            if self._check_assignment(assignment):
                solution_grid = self._assignment_to_grid(assignment)
                time_taken = time.time() - start_time
                return solution_grid, time_taken, True

        time_taken = time.time() - start_time
        return None, time_taken, False

    def solve_with_backtracking(self):
        """Solves the puzzle using a backtracking search."""
        start_time = time.time()
        
        assignment = [0] * len(self.connections)
        
        def is_valid(conn_idx):
            # Check degrees do not exceed required values
            degrees = defaultdict(int)
            for i in range(conn_idx + 1):
                if assignment[i] > 0:
                    p1, p2 = self.connections[i]
                    degrees[p1] += assignment[i]
                    degrees[p2] += assignment[i]
            
            for pos, required in self.islands:
                if degrees[pos] > required:
                    return False

            # Check for crossings up to current assignment
            for i in range(conn_idx + 1):
                for j in range(i + 1, conn_idx + 1):
                    if assignment[i] > 0 and assignment[j] > 0:
                        if self._bridges_cross(self.connections[i], self.connections[j]):
                            return False
            return True

        def backtrack(conn_idx):
            if conn_idx == len(self.connections):
                return self._check_assignment(assignment)

            for val in [0, 1, 2]:
                assignment[conn_idx] = val
                if is_valid(conn_idx):
                    if backtrack(conn_idx + 1):
                        return True
            
            assignment[conn_idx] = 0 # backtrack
            return False

        if backtrack(0):
            solution_grid = self._assignment_to_grid(assignment)
            time_taken = time.time() - start_time
            return solution_grid, time_taken, True

        time_taken = time.time() - start_time
        return None, time_taken, False

    def solve_with_astar(self):
        """Solves the puzzle using A* search."""
        start_time = time.time()

        # State: tuple of bridge counts, making it hashable
        initial_state = tuple([0] * len(self.connections))
        
        # Heuristic: sum of differences between current and required degrees
        def heuristic(state):
            degrees = defaultdict(int)
            for i, count in enumerate(state):
                if count > 0:
                    p1, p2 = self.connections[i]
                    degrees[p1] += count
                    degrees[p2] += count
            
            h = sum(abs(required - degrees[pos]) for pos, required in self.islands)
            return h
        
        # A* algorithm
        open_set = [(heuristic(initial_state), initial_state)] # (f_score, state)
        g_scores = {initial_state: 0} # cost from start

        while open_set:
            _, current_state = heapq.heappop(open_set)

            if self._check_assignment(list(current_state)):
                solution_grid = self._assignment_to_grid(list(current_state))
                time_taken = time.time() - start_time
                return solution_grid, time_taken, True

            # Get neighbors by adding one bridge
            for i in range(len(self.connections)):
                if current_state[i] < 2:
                    new_state_list = list(current_state)
                    new_state_list[i] += 1
                    neighbor = tuple(new_state_list)
                    
                    # Pruning: check if the new move is valid
                    degrees = defaultdict(int)
                    for idx, count in enumerate(neighbor):
                        if count > 0:
                            p1, p2 = self.connections[idx]
                            degrees[p1] += count
                            degrees[p2] += count
                    
                    if any(degrees[pos] > req for pos, req in self.islands):
                        continue # Prune this path

                    # Check for crossings
                    crosses = False
                    for j in range(len(self.connections)):
                        if i != j and neighbor[i] > 0 and neighbor[j] > 0:
                            if self._bridges_cross(self.connections[i], self.connections[j]):
                                crosses = True
                                break
                    if crosses:
                        continue
                    
                    tentative_g = g_scores[current_state] + 1
                    if tentative_g < g_scores.get(neighbor, float('inf')):
                        g_scores[neighbor] = tentative_g
                        f_score = tentative_g + heuristic(neighbor)
                        heapq.heappush(open_set, (f_score, neighbor))

        time_taken = time.time() - start_time
        return None, time_taken, False