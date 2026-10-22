"""Helpers shared by the warehouse_sim_* adapters."""
import json

import numpy as np


def load_instance(map_json, agent_num, eval_id):
    """Reads QD-MAPPER's map JSON the same way PBS/src/Instance.cpp does.

    Free cells are '.' and every other character is an obstacle. ``start`` and
    ``goal`` hold one list of linear cell indices (row * n_col + col) per
    evaluation, and the first ``agent_num`` agents of list ``eval_id`` are used.
    A JSON with ``num_of_agent`` (maps exported by tile_usage.py) holds a single
    flat list instead.

    Returns:
        grid: bool array, True = free cell.
        starts, goals: lists of (row, col) tuples.
    """
    data = json.loads(map_json) if isinstance(map_json, str) else map_json
    n_col = int(data["n_col"])
    grid = np.array([[c == "." for c in row] for row in data["layout"]],
                    dtype=bool)
    if "num_of_agent" in data:
        starts, goals = data["start"], data["goal"]
        agent_num = int(data["num_of_agent"])
    else:
        starts, goals = data["start"][eval_id], data["goal"][eval_id]
    starts = [divmod(int(s), n_col) for s in starts[:agent_num]]
    goals = [divmod(int(g), n_col) for g in goals[:agent_num]]
    return grid, starts, goals


def mapf_metrics(configs, goals):
    """Success rate, sum of costs and makespan of a run.

    Uses POGEMA's definitions for on_target='nothing' (NonDisappearISRMetric and
    SumOfCostsAndMakespanMetric), which is how warehouse_sim_LTF is scored, so
    PIBT and LTF are compared on equal terms.

    Args:
        configs: configs[t][i] is agent i's cell after t steps; configs[0] are
            the starts.
        goals: goal cell of each agent.

    Returns:
        success_rate: fraction of agents on their goal at the end.
        sum_of_cost: sum over agents of the step at which the agent last
            arrived at its goal (at least 1), or the number of steps run if it
            is not on its goal at the end.
        makespan: the largest such cost.
    """
    n_steps = len(configs) - 1
    costs = []
    n_on_goal = 0
    for i, goal in enumerate(goals):
        if configs[n_steps][i] == goal:
            n_on_goal += 1
            t = n_steps
            while t > 1 and configs[t - 1][i] == goal:
                t -= 1
            costs.append(t)
        else:
            costs.append(n_steps)
    return n_on_goal / len(goals), int(sum(costs)), int(max(costs))
