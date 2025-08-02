import time
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
    
    
