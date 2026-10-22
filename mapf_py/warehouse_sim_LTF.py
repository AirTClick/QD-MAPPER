"""Learn-to-Follow (Follower) for QD-MAPPER, using the vendored inference code
in mapf_py/learn_to_follow.

Plays the role of warehouse_sim_PBS for obj = "LTF" and "PIBT_LTF": run_follower()
is called by WarehouseModule.evaluate_LTF and returns the keys that
process_eval_result reads (success_rate, sum_of_cost, make_span_LTF).

LTF is run as one-shot MAPF, as in the paper: POGEMA with on_target='nothing'
(agents keep a single goal; the episode ends when all agents are on their goals
or after max_makespan steps) plus StopAtFirstGoal, which holds every agent that
has reached its goal in place. The scores use POGEMA's metrics for that mode
(ISR, SoC, makespan).

The heavy imports (torch model, sample-factory, POGEMA) happen on the first call,
so importing this module costs nothing for PBS/EECBS experiments. The model is
loaded once per Dask worker process and reused.
"""
import json
import logging
import os
import time

from mapf_py.common import load_instance

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(REPO_ROOT, "env_search_PBS", "LTF", "model",
                               "follower")

_MODELS = {}  # (path_to_weights, device) -> (FollowerInferenceConfig, FollowerInference)


def _load_model(path_to_weights, device):
    key = (path_to_weights, device)
    if key not in _MODELS:
        from mapf_py import add_deps_to_path
        add_deps_to_path()
        from sample_factory.utils.utils import log as sample_factory_log
        from mapf_py.learn_to_follow.follower.inference import (
            FollowerInference, FollowerInferenceConfig)
        sample_factory_log.setLevel(logging.WARNING)
        config = FollowerInferenceConfig(path_to_weights=path_to_weights,
                                         device=device)
        _MODELS[key] = (config, FollowerInference(config))
    return _MODELS[key]


def build_planner():
    """Compiles LTF's C++ planner (cppimport) if needed. Used by
    scripts/install_mapf_py_deps.sh so the Dask workers don't all compile it."""
    from mapf_py import add_deps_to_path
    add_deps_to_path()
    import mapf_py.learn_to_follow.follower.planning  # noqa: F401  (triggers cppimport)


def run_follower(**kwargs) -> str:
    """Runs Learn-to-Follow on one MAPF instance of a repaired map.

    Args (keyword):
        map: repaired map JSON string.
        agentNum: number of agents.
        eval_id: which start/goal list of the map to use.
        seed: random seed (POGEMA and the policy's action sampling).
        max_makespan: maximum number of timesteps (M in the paper).
        device: "cpu" or "cuda" for the policy network.
        path_to_weights: Follower model folder (default: the weights shipped in
            env_search_PBS/LTF/model/follower).

    Returns:
        JSON string with success_rate, sum_of_cost, make_span_LTF, runtime (CPU
        seconds), solution_found (all agents on their goals) and num_agents.
    """
    grid, starts, goals = load_instance(kwargs["map"], kwargs["agentNum"],
                                        kwargs.get("eval_id", 0))
    algo_config, algo = _load_model(
        kwargs.get("path_to_weights", DEFAULT_WEIGHTS),
        kwargs.get("device", "cpu"))

    from mapf_py.learn_to_follow.env.create_env import create_env_base
    from mapf_py.learn_to_follow.follower.preprocessing import follower_preprocessor
    from mapf_py.learn_to_follow.follower.training_config import (
        DecMAPFConfig, Environment)
    from mapf_py.learn_to_follow.stop_at_first_goal import StopAtFirstGoal
    from mapf_py.learn_to_follow.utils.eval_utils import run_episode

    seed = int(kwargs.get("seed", 0)) % (2**31 - 1)
    grid_config = DecMAPFConfig(
        map=(~grid).astype(int).tolist(),  # POGEMA: 1 = obstacle
        size=max(grid.shape),
        num_agents=len(starts),
        agents_xy=[list(p) for p in starts],
        targets_xy=[list(p) for p in goals],
        on_target="nothing",
        max_episode_steps=int(kwargs.get("max_makespan", 512)),
        seed=seed)
    env = create_env_base(Environment(use_maps=False, grid_config=grid_config))
    env = StopAtFirstGoal(follower_preprocessor(env, algo_config))

    algo_config.seed = seed  # algo.reset_states() seeds torch with it
    start_time = time.process_time()
    metrics = run_episode(env, algo)
    runtime = time.process_time() - start_time

    return json.dumps({
        "success_rate": float(metrics["ISR"]),
        "sum_of_cost": int(metrics["SoC"]),
        "make_span_LTF": int(metrics["makespan"]),
        "runtime": runtime,
        "solution_found": bool(metrics["CSR"]),
        "num_agents": len(starts),
    })
