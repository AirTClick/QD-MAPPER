from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .mapf_utils import Coord, Grid, get_neighbors, is_valid_coord


@dataclass
class DistTable:
    """Distance table for computing shortest path distances using BFS.

    This class lazily evaluates distances from a goal position to any target
    position on the grid using breadth-first search (BFS). Distances are
    cached for efficient repeated queries.

    Attributes:
        grid: 2D boolean array where True indicates free space.
        goal: Goal position (y, x) coordinates.
        Q: Queue for BFS traversal (lazy distance evaluation).
        table: Distance matrix storing computed distances.
    """
    grid: Grid
    goal: Coord
    Q: deque[Coord] = field(init=False)  # lazy distance evaluation
    table: np.ndarray = field(init=False)  # distance matrix

    def __post_init__(self) -> None:
        """Initialize distance table with goal position."""
        self.Q = deque([self.goal])
        self.table = np.full(self.grid.shape, self.grid.size, dtype=int)
        self.table[self.goal] = 0

    def get(self, target: Coord) -> int:
        """Get shortest path distance from goal to target.

        Uses lazy BFS evaluation to compute distance on demand. Previously
        computed distances are cached in the table.

        Args:
            target: Target position (y, x) coordinates.

        Returns:
            Shortest path distance from goal to target. Returns grid.size
            if target is invalid or unreachable.
        """
        # check valid input
        if not is_valid_coord(self.grid, target):
            return self.grid.size

        # distance has been known
        if int(self.table[target]) < self.table.size:
            return int(self.table[target])

        # BFS with lazy evaluation
        while len(self.Q) > 0:
            u = self.Q.popleft()
            d = int(self.table[u])
            for v in get_neighbors(self.grid, u):
                if d + 1 < self.table[v]:
                    self.table[v] = d + 1
                    self.Q.append(v)
            if u == target:
                return d

        return self.grid.size
