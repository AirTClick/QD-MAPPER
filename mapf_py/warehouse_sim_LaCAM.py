"""LaCAM for QD-MAPPER, using the vendored pylacam `pibt` branch (mapf_py/pycam).

Plays the role of warehouse_sim_PBS for obj = "LaCAM". As for PBS and EECBS, the
objective is the CPU runtime with time limit T = WarehouseEECBSConfig.cutoffTime
(20 s in the paper). LaCAM stops at its first solution (flg_star=False, no
anytime refinement, which would otherwise always use the whole time limit), and
an instance it does not solve within T counts as runtime T.
"""
import json
import time

from mapf_py.common import load_instance, mapf_metrics
from mapf_py.pycam import LaCAM
from mapf_py.pycam.mapf_utils import Config


def run(**kwargs) -> str:
    """Runs LaCAM on one MAPF instance of a repaired map.

    Args (keyword):
        map: repaired map JSON string.
        agentNum: number of agents.
        eval_id: which start/goal list of the map to use.
        seed: random seed for LaCAM's tie-breaking.
        cutoffTime: time limit in seconds.

    Returns:
        JSON string with runtime (CPU seconds, cutoffTime if unsolved),
        solution_found, solution_cost (sum of costs, None if unsolved),
        makespan (None if unsolved) and num_agents.
    """
    grid, starts, goals = load_instance(kwargs["map"], kwargs["agentNum"],
                                        kwargs.get("eval_id", 0))
    cutoff_time = float(kwargs.get("cutoffTime", 20))

    start_time = time.process_time()
    solution = LaCAM().solve(grid=grid,
                             starts=Config(list(starts)),
                             goals=Config(list(goals)),
                             time_limit_ms=int(cutoff_time * 1000),
                             flg_star=False,
                             seed=int(kwargs.get("seed", 0)),
                             verbose=0)
    runtime = time.process_time() - start_time

    solution_found = len(solution) > 0
    sum_of_cost, makespan = None, None
    if solution_found:
        _, sum_of_cost, makespan = mapf_metrics(solution, goals)
        runtime = min(runtime, cutoff_time)
    else:
        runtime = cutoff_time
    return json.dumps({
        "runtime": runtime,
        "solution_found": solution_found,
        "solution_cost": sum_of_cost,
        "makespan": makespan,
        "num_agents": len(starts),
    })
