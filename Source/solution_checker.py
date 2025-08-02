from collections import deque, defaultdict
import numpy as np
from island_map import IslandMap

class SolutionChecker:
    def __init__(self, islands, connections, grid_shape):
        self.islands = islands # The position of the islands
        self.connections = connections # The connections of the islands
        self.grid_shape = grid_shape # The shape of the map

    # Check if the solution is fully connected or not
    def _check_connected(self, assignment):
        adj = defaultdict(list)
        active = set()
        for i, count in enumerate(assignment):
            if count > 0:
                p1, p2 = self.connections[i]
                adj[p1].append(p2)
                adj[p2].append(p1)
                active.update([p1, p2])
        if not active:
            return len(self.islands) <= 1
        visited = {next(iter(active))}
        q = deque(visited)
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        return visited == {pos for pos, _ in self.islands}

    # Check if the solution is valid or not
    def is_valid(self, assignment):
        deg = defaultdict(int)
        for i, count in enumerate(assignment):
            if count > 0:
                p1, p2 = self.connections[i]
                deg[p1] += count
                deg[p2] += count
        for pos, required in self.islands:
            if deg[pos] != required:
                return False
        return self._check_connected(assignment)

    # Convert the solution to grid
    def to_grid(self, assignment):
        grid = np.full(self.grid_shape, '0', dtype=object)
        for (r, c), val in self.islands:
            grid[r, c] = str(val)
        for i, count in enumerate(assignment):
            if count == 0:
                continue
            (r1, c1), (r2, c2) = self.connections[i]
            char = '-' if r1 == r2 and count == 1 else '=' if r1 == r2 else '|' if count == 1 else '$'
            if r1 == r2:
                for c in range(min(c1, c2) + 1, max(c1, c2)):
                    grid[r1, c] = char
            else:
                for r in range(min(r1, r2) + 1, max(r1, r2)):
                    grid[r, c1] = char
        return grid.tolist()
