"""Priority Inheritance with Backtracking (PIBT) algorithm for MAPF."""

from __future__ import annotations
import numpy as np

from .dist_table import DistTable
from .mapf_utils import Config, Configs, Coord, Grid, get_neighbors


class PIBT:
    """Priority Inheritance with Backtracking algorithm for MAPF.

    PIBT is an iterative algorithm that computes collision-free paths for
    multiple agents quickly, even with hundreds of agents or more. It uses
    priority inheritance and backtracking to resolve conflicts efficiently.

    The algorithm is sub-optimal but provides acceptable solutions almost
    immediately. It maintains distance tables for each agent to their goal
    and uses these for informed decision making. Priorities are dynamically
    updated based on progress toward goals.

    Completeness Guarantee:
        All agents are guaranteed to reach their destinations within a finite
        time when all pairs of adjacent vertices belong to a simple cycle 
        (i.e., biconnected). This property holds regardless of the number 
        of agents.

    Attributes:
        grid: 2D boolean array where True indicates free space.
        starts: Initial positions of all agents.
        goals: Goal positions of all agents.
        N: Number of agents.
        dist_tables: Distance tables for each agent to their goal.
        NIL: Sentinel value representing unassigned agent.
        NIL_COORD: Sentinel value representing unassigned coordinate.
        occupied_now: Current occupation status of each grid cell.
        occupied_nxt: Next timestep occupation status of each grid cell.
        rng: Random number generator for tie-breaking.

    Example:
        >>> grid = get_grid("map.map")
        >>> starts, goals = get_scenario("scenario.scen", N=100)
        >>> pibt = PIBT(grid, starts, goals, seed=42)
        >>> solution = pibt.run(max_timestep=1000)
        >>> print(f"Solution length: {len(solution)}")

    References:
        Okumura, K., Machida, M., Défago, X., & Tamura, Y. (2022).
        Priority inheritance with backtracking for iterative multi-agent
        path finding. Artificial Intelligence Journal.
        https://kei18.github.io/pibt2/

    Note:
        PIBT serves as a core component in LaCAM (AAAI-23), which uses
        PIBT to quickly obtain initial solutions for eventually optimal
        multi-agent pathfinding. See https://kei18.github.io/lacam-project/
    """

    def __init__(self, grid: Grid, starts: Config, goals: Config, seed: int = 0) -> None:
        """Initialize PIBT solver.

        Args:
            grid: 2D boolean array where True indicates free space.
            starts: Initial positions of all agents (y, x).
            goals: Goal positions of all agents (y, x).
            seed: Random seed for tie-breaking (default: 0).
        """
        self.grid = grid
        self.starts = starts
        self.goals = goals
        self.N = len(self.starts)

        # distance table
        self.dist_tables = [DistTable(grid, goal) for goal in goals]

        # cache
        self.NIL = self.N  # meaning \bot
        self.NIL_COORD: Coord = self.grid.shape  # meaning \bot
        self.occupied_now = np.full(grid.shape, self.NIL, dtype=int)
        self.occupied_nxt = np.full(grid.shape, self.NIL, dtype=int)

        # used for tie-breaking
        self.rng = np.random.default_rng(seed)

    def funcPIBT(self, Q_from: Config, Q_to: Config, i: int) -> bool:
        """Core PIBT function for single agent planning with priority inheritance.

        Attempts to assign a collision-free next position for agent i. If
        another agent j occupies the desired position, recursively invokes
        PIBT for agent j (priority inheritance). Backtracks if no valid
        position is found.

        Args:
            Q_from: Current configuration (positions at current timestep).
            Q_to: Next configuration being constructed (modified in-place).
            i: Agent index to plan for.

        Returns:
            True if successfully assigned a position to agent i, False otherwise.
        """
        # true -> valid, false -> invalid

        # get candidate next vertices
        C = [Q_from[i]] + get_neighbors(self.grid, Q_from[i])
        self.rng.shuffle(C)  # tie-breaking, randomize
        C = sorted(C, key=lambda u: self.dist_tables[i].get(u))

        # vertex assignment
        for v in C:
            # avoid vertex collision
            if self.occupied_nxt[v] != self.NIL:
                continue

            j = self.occupied_now[v]

            # avoid edge collision
            if j != self.NIL and Q_to[j] == Q_from[i]:
                continue

            # reserve next location
            Q_to[i] = v
            self.occupied_nxt[v] = i

            # priority inheritance (j != i due to the second condition)
            if (
                j != self.NIL
                and (Q_to[j] == self.NIL_COORD)
                and (not self.funcPIBT(Q_from, Q_to, j))
            ):
                continue

            return True

        # failed to secure node
        Q_to[i] = Q_from[i]
        self.occupied_nxt[Q_from[i]] = i
        return False

    def step(self, Q_from: Config, priorities: list[float]) -> Config:
        """Compute next configuration for all agents.

        Executes one timestep of PIBT by calling funcPIBT for all agents
        in priority order.

        Args:
            Q_from: Current configuration (positions at current timestep).
            priorities: Priority values for each agent (higher = earlier planning).

        Returns:
            Next configuration with updated positions for all agents.
        """
        # setup
        N = len(Q_from)
        Q_to: Config = []
        for i, v in enumerate(Q_from):
            Q_to.append(self.NIL_COORD)
            self.occupied_now[v] = i

        # perform PIBT
        A = sorted(list(range(N)), key=lambda i: priorities[i], reverse=True)
        for i in A:
            if Q_to[i] == self.NIL_COORD:
                self.funcPIBT(Q_from, Q_to, i)

        # cleanup
        for q_from, q_to in zip(Q_from, Q_to):
            self.occupied_now[q_from] = self.NIL
            self.occupied_nxt[q_to] = self.NIL

        return Q_to

    def run(self, max_timestep: int = 1000) -> Configs:
        """Run PIBT algorithm until all agents reach goals or timeout.

        Iteratively computes collision-free paths for all agents using PIBT.
        Priorities are dynamically updated: incremented when an agent hasn't
        reached its goal, reset otherwise.

        Args:
            max_timestep: Maximum number of timesteps to run (default: 1000).

        Returns:
            Sequence of configurations from start to goal. Each configuration
            is a list of agent positions (y, x) at that timestep.

        Example:
            >>> pibt = PIBT(grid, starts, goals)
            >>> solution = pibt.run(max_timestep=500)
            >>> print(f"Solved in {len(solution)} timesteps")
        """
        # define priorities
        priorities: list[float] = []
        for i in range(self.N):
            priorities.append(self.dist_tables[i].get(self.starts[i]) / self.grid.size)

        # main loop, generate sequence of configurations
        configs = [self.starts]
        while len(configs) <= max_timestep:
            # obtain new configuration
            Q = self.step(configs[-1], priorities)
            configs.append(Q)

            # update priorities & goal check
            flg_fin = True
            for i in range(self.N):
                if Q[i] != self.goals[i]:
                    flg_fin = False
                    priorities[i] += 1
                else:
                    priorities[i] -= np.floor(priorities[i])
            if flg_fin:
                break  # goal

        return configs
