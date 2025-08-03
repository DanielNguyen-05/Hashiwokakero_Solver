from pysat.formula import CNF

class CNFGenerator:
    def __init__(self, islands, connections):
        self.islands = islands
        self.connections = connections
        self.var_counter = 1
        self.variables = {}

    def _get_var(self, name):
        if name not in self.variables:
            self.variables[name] = self.var_counter
            self.var_counter += 1
        return self.variables[name]
    
    def _next_var(self):
        var = self.var_counter
        self.var_counter += 1
        return var

    def _bridges_cross(self, conn1, conn2):
        (r1a, c1a), (r1b, c1b) = conn1
        (r2a, c2a), (r2b, c2b) = conn2
        h1, h2 = r1a == r1b, r2a == r2b
        if h1 == h2:
            return False
        if h1:
            return (min(c1a, c1b) < c2a < max(c1a, c1b)) and (min(r2a, r2b) < r1a < max(r2a, r2b))
        else:
            return (min(r1a, r1b) < r2a < max(r1a, r1b)) and (min(c2a, c2b) < c1a < max(c2a, c2b))

    # Encode sum constraint
    def encode_sum(self, literals, weights, target_sum):
        def new_auxiliary_var():
            return self._next_var()

        clauses = []
        n = len(literals)
        if n == 0:
            return [[1], [-1]] if target_sum != 0 else []

        sum_vars = [{} for _ in range(n + 1)]
        sum_vars[0][0] = new_auxiliary_var()
        clauses.append([sum_vars[0][0]])  

        for i in range(1, n + 1):
            lit = literals[i - 1]
            weight = weights[i - 1]
            prev = sum_vars[i - 1]
            curr = {}

            if weight == 0:
                for t in prev:
                    aux_prev = prev[t]
                    if t not in curr:
                        curr[t] = new_auxiliary_var()
                    aux_curr = curr[t]
                    clauses.append([-aux_prev, aux_curr])
            else:
                for t in prev:
                    aux_prev = prev[t]

                    if t not in curr:
                        curr[t] = new_auxiliary_var()
                    aux_no_add = curr[t]
                    clauses.append([-aux_prev, lit, aux_no_add])

                    new_t = t + weight
                    if new_t not in curr:
                        curr[new_t] = new_auxiliary_var()
                    aux_add = curr[new_t]
                    clauses.append([-aux_prev, -lit, aux_add])

            sums = sorted(curr.keys())
            for idx in range(len(sums)):
                for jdx in range(idx + 1, len(sums)):
                    clauses.append([-curr[sums[idx]], -curr[sums[jdx]]])

            sum_vars[i] = curr

        if target_sum not in sum_vars[n]:
            return [[1], [-1]]  

        clauses.append([sum_vars[n][target_sum]])
        return clauses

    # Generate CNF 
    def generate(self):
        cnf = CNF()
        conn_vars = {}

        # Condition 1: Between two islands only 0 bridge or 1 bridge or 2 bridges can be built
        for i, conn in enumerate(self.connections):
            v1 = self._get_var(f"conn_{i}_1")
            v2 = self._get_var(f"conn_{i}_2")
            conn_vars[i] = (v1, v2)
            cnf.append([-v1, -v2])

        # Condition 2: Degree Constraint
        for pos, degree in self.islands:
            lits = []
            weights = []
            for i, conn in enumerate(self.connections):
                if pos in conn:
                    v1, v2 = conn_vars[i]
                    lits.extend([v1, v2])
                    weights.extend([1, 2])
            if lits:
                clauses = self.encode_sum(lits, weights, degree)
                cnf.extend(clauses)

        # Condition 3: Bridges Crossed Constraint
        for i in range(len(self.connections)):
            for j in range(i + 1, len(self.connections)):
                if self._bridges_cross(self.connections[i], self.connections[j]):
                    for x in conn_vars[i]:
                        for y in conn_vars[j]:
                            cnf.append([-x, -y])
        return cnf, conn_vars
