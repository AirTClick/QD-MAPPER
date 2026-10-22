"""PIBT for QD-MAPPER, using the vendored pypibt (mapf_py/pypibt).

Plays the role of warehouse_sim_PBS for obj = "PIBT" and "PIBT_LTF": it is called
by WarehouseModule.evaluate_PIBT and returns the keys that process_eval_result
reads (success_rate, sum_of_cost, make_span_PIBT).
"""
import json
import time

from mapf_py.common import load_instance, mapf_metrics
from mapf_py.pypibt import PIBT


def run(**kwargs) -> str:
    """Runs PIBT on one MAPF instance of a repaired map.

    Args (keyword):
        map: repaired map JSON string.
        agentNum: number of agents.
        eval_id: which start/goal list of the map to use.
        seed: random seed for PIBT's tie-breaking.
        max_makespan: maximum number of timesteps (M in the paper).

    Returns:
        JSON string with success_rate, sum_of_cost, make_span_PIBT, runtime (CPU
        seconds), solution_found (all agents on their goals) and num_agents.
    """
    grid, starts, goals = load_instance(kwargs["map"], kwargs["agentNum"],
                                        kwargs.get("eval_id", 0))
    start_time = time.process_time()
    configs = PIBT(grid, starts, goals, seed=int(kwargs.get("seed", 0))).run(
        max_timestep=int(kwargs.get("max_makespan", 1000)))
    runtime = time.process_time() - start_time

    success_rate, sum_of_cost, makespan = mapf_metrics(configs, goals)
    return json.dumps({
        "success_rate": success_rate,
        "sum_of_cost": sum_of_cost,
        "make_span_PIBT": makespan,
        "runtime": runtime,
        "solution_found": success_rate == 1.0,
        "num_agents": len(starts),
    })
