import time
import random
import logging
import pathlib

import traceback
import numpy as np

from typing import List
from env_search_PBS.warehouseEECBS.warehouse_result import WarehouseResult
from env_search_PBS.utils.worker_state import get_warehouse_module
from env_search_PBS.warehouseEECBS.module import WarehouseModule

logger = logging.getLogger(__name__)


def repair_warehouse(
    map: np.ndarray,
    parent_map: np.ndarray,
    sim_seed: int,
    repair_seed: int,
    # w_mode: bool,
    # min_n_shelf: int,
    min_n_obstacles: int,
    max_n_obstacles: int,
    # agentNum: int,
    n_evals: int,
):
    start = time.time()

    logger.info("seeding global randomness")
    np.random.seed(sim_seed // np.int32(4))
    random.seed(sim_seed // np.int32(2))

    logger.info("repair warehouse with seed %d", repair_seed)
    warehouse_module = get_warehouse_module()

    try:
        (
            map_json,
            map_np_unrepaired,
            map_comp_unrepaired,
            map_np_repaired,
            num_agent,
        ) = warehouse_module.repair(
            map=map,
            parent_map=parent_map,
            sim_seed=sim_seed,
            repair_seed=repair_seed,
            # w_mode=w_mode,
            # min_n_shelf=min_n_shelf,
            min_n_obstacles = min_n_obstacles,
            max_n_obstacles=max_n_obstacles,
            # agentNum=agentNum,
            n_evals = n_evals,
        )
    except TimeoutError as e:
        logger.warning(f"repair failed")
        logger.info(f"The map was {map}")
        (
            map_json,
            map_np_unrepaired,
            map_comp_unrepaired,
            map_np_repaired,
            num_agent,
        ) = [None] * 5

    logger.info("repair_warehouse done after %f sec", time.time() - start)

    return map_json, map_np_unrepaired, map_comp_unrepaired, map_np_repaired, num_agent



def run_warehouse_PIBT(
    map_json: str,
    eval_logdir: pathlib.Path,
    sim_seed: int,
    agentNum: int,
    map_id: int,
    eval_id: int,
) -> WarehouseResult:

    if map_json is None:
        logger.info("Evaluating failed layout. Skipping")
        result = {"success": False}
        return result

    start = time.time()

    logger.info("seeding global randomness")
    np.random.seed(sim_seed // np.int32(4))
    random.seed(sim_seed // np.int32(2))

    logger.info("run warehouse PIBT with seed %d", sim_seed)
    warehouse_module = get_warehouse_module()

    try:
        result_PIBT = warehouse_module.evaluate_PIBT(
            map_json=map_json,
            eval_logdir=eval_logdir,
            sim_seed=sim_seed,
            agentNum=agentNum,
            map_id=map_id,
            eval_id=eval_id, 
        )
        result_PIBT["success"] = True

    except TimeoutError as e:
        layout = map_json["layout"]
        logger.warning(f"evaluate failed")
        logger.info(f"The map was {layout}")
        result = {"success": False}

    logger.info("run_warehouse_PIBT done after %f sec", time.time() - start)

    return result_PIBT

def run_warehouse_LTF(
    map_json: str,
    eval_logdir: pathlib.Path,
    sim_seed: int,
    agentNum: int,
    map_id: int,
    eval_id: int,
) -> WarehouseResult:

    if map_json is None:
        logger.info("Evaluating failed layout. Skipping")
        result = {"success": False}
        return result

    start = time.time()

    logger.info("seeding global randomness")
    np.random.seed(sim_seed // np.int32(4))
    random.seed(sim_seed // np.int32(2))

    logger.info("run warehouse LTF with seed %d", sim_seed)
    warehouse_module = get_warehouse_module()

    try:
        result_LTF = warehouse_module.evaluate_LTF(
            map_json=map_json,
            eval_logdir=eval_logdir,
            sim_seed=sim_seed,
            agentNum=agentNum,
            map_id=map_id,
            eval_id=eval_id, 
        )
        result_LTF["success"] = True

    except TimeoutError as e:
        layout = map_json["layout"]
        logger.warning(f"evaluate failed")
        logger.info(f"The map was {layout}")
        result = {"success": False}

    logger.info("run_warehouse_LTF done after %f sec", time.time() - start)

    return result_LTF


def run_warehouse_LaCAM(
    map_json: str,
    eval_logdir: pathlib.Path,
    sim_seed: int,
    agentNum: int,
    map_id: int,
    eval_id: int,
) -> WarehouseResult:

    if map_json is None:
        logger.info("Evaluating failed layout. Skipping")
        result = {"success": False}
        return result

    start = time.time()

    logger.info("seeding global randomness")
    np.random.seed(sim_seed // np.int32(4))
    random.seed(sim_seed // np.int32(2))

    logger.info("run warehouse LaCAM with seed %d", sim_seed)
    warehouse_module = get_warehouse_module()

    result_LaCAM = warehouse_module.evaluate_LaCAM(
        map_json=map_json,
        eval_logdir=eval_logdir,
        sim_seed=sim_seed,
        agentNum=agentNum,
        map_id=map_id,
        eval_id=eval_id,
    )
    result_LaCAM["success"] = True

    logger.info("run_warehouse_LaCAM done after %f sec", time.time() - start)

    return result_LaCAM


# Run with two algorithms at a time
def run_warehouse_PBS(
    map_json: str,
    eval_logdir: pathlib.Path,
    sim_seed: int,
    agentNum: int,
    map_id: int,
    eval_id: int,
) -> WarehouseResult:

    if map_json is None:
        logger.info("Evaluating failed layout. Skipping")
        result = {"success": False}
        return result

    start = time.time()

    logger.info("seeding global randomness")
    np.random.seed(sim_seed // np.int32(4))
    random.seed(sim_seed // np.int32(2))

    logger.info("run warehouse PBS with seed %d", sim_seed)
    warehouse_module = get_warehouse_module()

    try:
        result_PBS = warehouse_module.evaluate_PBS(
            map_json=map_json,
            eval_logdir=eval_logdir,
            sim_seed=sim_seed,
            agentNum=agentNum,
            map_id=map_id,
            eval_id=eval_id, 
        )
        result_PBS["success"] = True

    except TimeoutError as e:
        layout = map_json["layout"]
        logger.warning(f"evaluate failed")
        logger.info(f"The map was {layout}")
        result = {"success": False}

    logger.info("run_warehouse_PBS done after %f sec", time.time() - start)

    return result_PBS


def run_warehouse_EECBS(
    map_json: str,
    eval_logdir: pathlib.Path,
    sim_seed: int,
    agentNum: int,
    map_id: int,
    eval_id: int,
    suboptimality_bound: float,
) -> WarehouseResult:
    """
    Repair map and run simulation

    Args:
        map (np.ndarray): input map in integer format
        parent_map (np.ndarray): parent solution of the map. Will be None if
                                    current sol is the initial population.
        eval_logdir (str): log dir of simulation
        n_evals (int): number of evaluations
        sim_seed (int): random seed for simulation. Should be different for
                        each solution
        repair_seed (int): random seed for repairing. Should be the same as
                            master seed
        w_mode (bool): whether to run with w_mode, which replace 'r' with
                        'w' in generated map layouts, where 'w' is a
                        workstation. Under w_mode, robots will start from
                        endpoints and their tasks will alternate between
                        endpoints and workstations.
        n_endpt (int): number of endpoint around each obstacle
        min_n_shelf (int): min number of shelves
        max_n_shelf (int): max number of shelves
        agentNum (int): number of drives
        map_id (int): id of the current map to be evaluated. The id
                      is only unique to each batch, NOT to the all the
                      solutions. The id can make sure that each simulation
                      gets a different log directory.
    """
    """Grabs the warehouse module and evaluates level n_evals times."""

    if map_json is None:
        logger.info("Evaluating failed layout. Skipping")
        result = {"success": False}
        return result

    start = time.time()

    logger.info("seeding global randomness")
    np.random.seed(sim_seed // np.int32(4))
    random.seed(sim_seed // np.int32(2))

    logger.info("run warehouse EECBS with seed %d", sim_seed)
    warehouse_module = get_warehouse_module()

    try:
        result_EECBS = warehouse_module.evaluate_EECBS(
            map_json=map_json,
            eval_logdir=eval_logdir,
            sim_seed=sim_seed,
            agentNum=agentNum,
            map_id=map_id,
            eval_id=eval_id,
            suboptimality_bound= suboptimality_bound
        )
        result_EECBS["success"] = True
    except TimeoutError as e:
        layout = map_json["layout"]
        logger.warning(f"evaluate failed")
        logger.info(f"The map was {layout}")
        result = {"success": False}

    logger.info("run_warehouse_EECBS done after %f sec", time.time() - start)

    return result_EECBS

def process_warehouse_eval_result(
    curr_result_json_EECBS: List[dict],
    curr_result_json_PBS: List[dict],
    curr_result_json_PIBT: List[dict],
    curr_result_json_LTF: List[dict],
    n_evals: int,
    map_np_unrepaired,
    map_comp_unrepaired,
    map_np_repaired,
    # w_mode: bool,
    max_n_shelf: int,
    map_id: int,
    obj: str,
):
    start = time.time()

    warehouse_module = get_warehouse_module()

    results = warehouse_module.process_eval_result(
        curr_result_json_EECBS=curr_result_json_EECBS,
        curr_result_json_PBS = curr_result_json_PBS,
        curr_result_json_PIBT = curr_result_json_PIBT,
        curr_result_json_LTF = curr_result_json_LTF,
        n_evals=n_evals,
        map_np_unrepaired=map_np_unrepaired,
        map_comp_unrepaired=map_comp_unrepaired,
        map_np_repaired=map_np_repaired,
        # w_mode=w_mode,
        max_n_shelf=max_n_shelf,
        map_id=map_id,
        obj=obj,
    )
    logger.info("process_warehouse_eval_result done after %f sec",
                time.time() - start)

    return results