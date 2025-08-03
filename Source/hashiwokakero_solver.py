import time
import copy
from pysat.solvers import Glucose3
from itertools import product
from island_map import IslandMap
from cnf_generator import CNFGenerator
from solution_checker import SolutionChecker

class HashiwokakeroSolver:
    def __init__(self, grid):
        self.map = IslandMap(grid) # The map
        self.checker = SolutionChecker(self.map.islands, self.map.connections, self.map.grid.shape) # Solution checker

    # Solve with SAT
    def solve_with_pysat(self):
        start = time.time()
        gen = CNFGenerator(self.map.islands, self.map.connections)
        cnf, conn_vars = gen.generate()

        cnf.to_file("Hashi_CNF.cnf")

        unique_clauses = list({tuple(sorted(clause)) for clause in cnf.clauses})
        with Glucose3(bootstrap_with=[list(c) for c in unique_clauses]) as solver:
            for model in solver.enum_models():
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
                return None, time.time() - start, False

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
        
    
    
