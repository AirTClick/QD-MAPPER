"""Multi-Agent Path Finding (MAPF) utility functions.

This module provides utilities for loading MAPF problem instances from
standard benchmark files and validating MAPF solutions.
"""

from __future__ import annotations
import os
import re
try:  # Python >= 3.10
    from typing import TypeAlias
except ImportError:  # Python 3.9 (QD-MAPPER container)
    from typing_extensions import TypeAlias

import numpy as np

# y, x
Grid: TypeAlias = np.ndarray
Coord: TypeAlias = tuple[int, int]
Config: TypeAlias = list[Coord]
Configs: TypeAlias = list[Config]


def get_grid(map_file: str) -> Grid:
    """Load grid map from a MAPF benchmark file.

    Parses a .map file from the MAPF benchmarks (movingai.com format) and
    returns a 2D boolean array representing the grid.

    Args:
        map_file: Path to the .map file.

    Returns:
        2D boolean array where True indicates free space and False indicates
        an obstacle. Shape is (height, width) with indexing grid[y, x].

    Raises:
        AssertionError: If the map file format is invalid.
    """
    width, height = 0, 0
    with open(map_file, "r") as f:
        # retrieve map size
        for row in f:
            # get width
            res = re.match(r"width\s(\d+)", row)
            if res:
                width = int(res.group(1))

            # get height
            res = re.match(r"height\s(\d+)", row)
            if res:
                height = int(res.group(1))

            if width > 0 and height > 0:
                break

        # retrieve map
        grid = np.zeros((height, width), dtype=bool)
        y = 0
        for row in f:
            row = row.strip()
            if len(row) == width and row != "map":
                grid[y] = [s == "." for s in row]
                y += 1

    # simple error check
    assert y == height, f"map format seems strange, check {map_file}"

    # grid[y, x] -> True: available, False: obstacle
    return grid


def get_scenario(scen_file: str, N: int | None = None) -> tuple[Config, Config]:
    """Load start and goal configurations from a MAPF scenario file.

    Parses a .scen file from the MAPF benchmarks (movingai.com format) and
    extracts start and goal positions for agents.

    Args:
        scen_file: Path to the .scen file.
        N: Maximum number of agents to load. If None, loads all agents.

    Returns:
        A tuple (starts, goals) where each is a list of (y, x) coordinates.
    """
    with open(scen_file, "r") as f:
        starts, goals = [], []
        for row in f:
            res = re.match(
                r"\d+\t.+\.map\t\d+\t\d+\t(\d+)\t(\d+)\t(\d+)\t(\d+)\t.+", row
            )
            if res:
                x_s, y_s, x_g, y_g = [int(res.group(k)) for k in range(1, 5)]
                starts.append((y_s, x_s))  # align with grid
                goals.append((y_g, x_g))

                # check the number of agents
                if (N is not None) and len(starts) >= N:
                    break

    return starts, goals


def is_valid_coord(grid: Grid, coord: Coord) -> bool:
    """Check if a coordinate is valid and free on the grid.

    Args:
        grid: 2D boolean array representing the map.
        coord: Position (y, x) to check.

    Returns:
        True if coordinate is within bounds and not an obstacle, False otherwise.
    """
    y, x = coord
    if y < 0 or y >= grid.shape[0] or x < 0 or x >= grid.shape[1] or not grid[coord]:
        return False
    return True


def get_neighbors(grid: Grid, coord: Coord) -> list[Coord]:
    """Get valid neighboring coordinates (4-connected grid).

    Args:
        grid: 2D boolean array representing the map.
        coord: Center position (y, x).

    Returns:
        List of valid neighboring coordinates in 4 directions (left, right,
        up, down). Empty list if coord is invalid.
    """
    # coord: y, x
    neigh: list[Coord] = []

    # check valid input
    if not is_valid_coord(grid, coord):
        return neigh

    y, x = coord

    if x > 0 and grid[y, x - 1]:
        neigh.append((y, x - 1))

    if x < grid.shape[1] - 1 and grid[y, x + 1]:
        neigh.append((y, x + 1))

    if y > 0 and grid[y - 1, x]:
        neigh.append((y - 1, x))

    if y < grid.shape[0] - 1 and grid[y + 1, x]:
        neigh.append((y + 1, x))

    return neigh


def save_configs_for_visualizer(configs: Configs, filename: str) -> None:
    """Save solution configurations for visualization.

    Exports the solution in a format compatible with mapf-visualizer tool.

    Args:
        configs: List of configurations, where each configuration is a list
            of agent positions (y, x) at a timestep.
        filename: Output file path.

    Example:
        >>> configs = [[(0, 0), (1, 1)], [(0, 1), (1, 2)]]
        >>> save_configs_for_visualizer(configs, "output.txt")
    """
    dirname = os.path.dirname(filename)
    if len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)
    with open(filename, "w") as f:
        for t, config in enumerate(configs):
            row = f"{t}:" + "".join([f"({x},{y})," for (y, x) in config]) + "\n"
            f.write(row)


def validate_mapf_solution(
    grid: Grid,
    starts: Config,
    goals: Config,
    solution: Configs,
) -> None:
    """Validate a MAPF solution for correctness.

    Checks that the solution:
    - Starts at the specified start positions
    - Ends at the specified goal positions
    - Has valid transitions (agents move to adjacent cells or stay)
    - Has no vertex collisions (two agents at same position)
    - Has no edge collisions (two agents swap positions)

    Args:
        grid: 2D boolean array representing the map.
        starts: Initial positions of all agents.
        goals: Goal positions of all agents.
        solution: Sequence of configurations over time.

    Raises:
        AssertionError: If the solution violates any MAPF constraint.
    """
    # starts
    assert all(
        [u == v for (u, v) in zip(starts, solution[0])]
    ), "invalid solution, check starts"

    # goals
    assert all(
        [u == v for (u, v) in zip(goals, solution[-1])]
    ), "invalid solution, check goals"

    T = len(solution)
    N = len(starts)

    for t in range(T):
        for i in range(N):
            v_i_now = solution[t][i]
            v_i_pre = solution[max(t - 1, 0)][i]

            # check continuity
            assert v_i_now in [v_i_pre] + get_neighbors(
                grid, v_i_pre
            ), "invalid solution, check connectivity"

            # check collision
            for j in range(i + 1, N):
                v_j_now = solution[t][j]
                v_j_pre = solution[max(t - 1, 0)][j]
                assert not (v_i_now == v_j_now), "invalid solution, vertex collision"
                assert not (
                    v_i_now == v_j_pre and v_i_pre == v_j_now
                ), "invalid solution, edge collision"


def is_valid_mapf_solution(
    grid: Grid,
    starts: Config,
    goals: Config,
    solution: Configs,
) -> bool:
    """Check if a MAPF solution is valid.

    Wrapper around validate_mapf_solution that returns a boolean instead
    of raising exceptions.

    Args:
        grid: 2D boolean array representing the map.
        starts: Initial positions of all agents.
        goals: Goal positions of all agents.
        solution: Sequence of configurations over time.

    Returns:
        True if solution is valid, False otherwise.

    Example:
        >>> grid = get_grid("map.map")
        >>> starts, goals = get_scenario("scenario.scen", N=10)
        >>> pibt = PIBT(grid, starts, goals)
        >>> solution = pibt.run()
        >>> is_valid = is_valid_mapf_solution(grid, starts, goals, solution)
    """
    try:
        validate_mapf_solution(grid, starts, goals, solution)
        return True
    except Exception as e:
        print(e)
        return False
