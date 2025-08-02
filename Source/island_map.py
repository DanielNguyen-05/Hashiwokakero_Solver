from itertools import product
import numpy as np

class IslandMap:
    def __init__(self, grid):
        self.grid = np.array(grid, dtype=int) # The grid from map
        self.rows, self.cols = self.grid.shape # The size of the map
        self.islands = self._find_islands() # The islands positions from the map
        self.connections = self._find_possible_connections() # The possible connections from the map

    # Find all of the islands from the map
    def _find_islands(self):
        return [((r, c), self.grid[r, c])
                for r in range(self.rows)
                for c in range(self.cols)
                if self.grid[r, c] > 0]

    # Check if the path between two islands is clear
    def _is_path_clear(self, r1, c1, r2, c2):
        path = [(r1, c) for c in range(min(c1, c2)+1, max(c1, c2))] if r1 == r2 else \
               [(r, c1) for r in range(min(r1, r2)+1, max(r1, r2))]
        island_positions = {pos for pos, _ in self.islands}
        return not any(p in island_positions for p in path)

    # Find all possible connections
    def _find_possible_connections(self):
        conns = []
        for i in range(len(self.islands)):
            for j in range(i + 1, len(self.islands)):
                (p1, _), (p2, _) = self.islands[i], self.islands[j]
                if (p1[0] == p2[0] or p1[1] == p2[1]) and self._is_path_clear(*p1, *p2):
                    conns.append((p1, p2))
        return conns
