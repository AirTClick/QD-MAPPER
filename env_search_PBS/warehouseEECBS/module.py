"""WarehouseConfig and WarehouseModule.

Usage:
    # Run as a script to demo the WarehouseModule.
    python env_search_PBS/warehouse/module.py
"""

import os
import gin
import copy
import json
import time
import fire
import logging
import pathlib
import warnings
import warehouse_sim_PBS  # type: ignore # ignore pylance warning
import numpy as np
import multiprocessing
import scipy
import time

from scipy.stats import entropy
from typing import List
from dataclasses import dataclass
from itertools import repeat, product
from typing import Collection, Optional
from queue import Queue
from typing import Collection
from env_search_PBS import LOG_DIR
from env_search_PBS.utils.logging import setup_logging
from env_search_PBS.warehouseEECBS.milp_repair import repair_env
from env_search_PBS.warehouseEECBS.warehouse_result import (WarehouseResult,
                                                   WarehouseMetadata)
from env_search_PBS.utils import (eecbs_obj_types, MIN_SCORE,
                              eecbs_env_number2str,
                              kiva_env_str2number, format_env_str,
                              read_in_kiva_map, flip_tiles)

# from env_search_PBS.test_entropy import maze_dist

logger = logging.getLogger(__name__)


@gin.configurable
@dataclass
class WarehouseEECBSConfig:
    """
    Config warehouse simulation

    Args:
        measure_names (list[str]): list of names of measures
        aggregation_type (str): aggregation over `n_evals` results
        scenario (str): scenario (SORTING, KIVA, ONLINE, BEE)
        task (str): input task file

        cutoffTime (int): cutoff time (seconds)
        screen (int): screen option (0: none; 1: results; 2:all)
        solver (str): solver (LRA, PBS, WHCA, ECBS)
        id (bool): independence detection
        single_agent_solver (str): single-agent solver (ASTAR, SIPP)
        lazyP (bool): use lazy priority
        simulation_time (int): run simulation
        simulation_window (int): call the planner every simulation_window
                                 timesteps
        travel_time_window (int): consider the traffic jams within the
                                  given window
        planning_window (int): the planner outputs plans with first
                                     planning_window timesteps collision-free
        potential_function (str): potential function (NONE, SOC, IC)
        potential_threshold (int): potential threshold
        rotation (bool): consider rotation
        robust (int): k-robust (for now, only work for PBS)
        CAT (bool): use conflict-avoidance table
        hold_endpoints (bool): Hold endpoints from Ma et al, AAMAS 2017
        dummy_paths (bool): Find dummy paths from Liu et al, AAMAS 2019
        prioritize_start (bool): Prioritize waiting at start locations
        suboptimal_bound (int): Suboptimal bound for ECBS
        log (bool): save the search trees (and the priority trees)
        test (bool): whether under testing mode.
        use_warm_up (bool): if True, will use the warm-up procedure. In
                            particular, for the initial population, the solution
                            returned from hamming distance objective will be
                            used. For mutated solutions, the solution of the
                            parent will be used.
        save_result (bool): Whether to allow C++ save the result of simulation
        save_solver (bool): Whether to allow C++ save the result of solver
        save_heuristics_table (bool): Whether to allow C++ save the result of
                                      heuristics table
        stop_at_traffic_jam (bool): whether stop the simulation at traffic jam
        obj_type (str): type of objective
                        ("throughput", "throughput_plus_n_shelf")
    """
    # Measures.
    measure_names: Collection[str] = gin.REQUIRED

    # Results.
    aggregation_type: str = gin.REQUIRED,

    # Simulation
    cutoffTime: int = gin.REQUIRED,
    screen: int = gin.REQUIRED,
    stats: bool = gin.REQUIRED,
    # id: bool = gin.REQUIRED,
    sipp: bool = gin.REQUIRED,
    obj_type : str = gin.REQUIRED,
    use_warm_up : bool = gin.REQUIRED,
    hamming_only : bool = gin.REQUIRED,

    # tile_usage_room: np.ndarray = gin.REQUIRED, 
    # obj_name : str = gin.REQUIRED,


class WarehouseModule:
    def __init__(self, config: WarehouseEECBSConfig):
        self.config = config

    def repair(
        self,
        map: np.ndarray,
        parent_map: np.ndarray,
        repair_seed: int,
        sim_seed: int,
        # w_mode: bool,
        min_n_obstacles: int,
        max_n_obstacles: int,
        agentNum: int,
    ):
        map_np_unrepaired = copy.deepcopy(map)

        # Create json string for the map
        # if self.config.scenario == "KIVA":
        # if self.config.obj_type == "throughput_plus_n_obstacle":
        #      assert max_n_obstacle == min_n_obstacle
        #     ADDITION_BLOCK_WIDTH = KIVA_WORKSTATION_BLOCK_WIDTH if w_mode \
        #                         else KIVA_ROBOT_BLOCK_WIDTH
        #     ADDITION_BLOCK_HEIGHT = 0 if w_mode else KIVA_ROBOT_BLOCK_HEIGHT
        n_row, n_col = map.shape
        map_comp_unrepaired = copy.deepcopy(map)
            # d = [(0, 1), (0, -1), (1, 0), (-1, 0)]

            # # Keep only max_obs_ratio of obstacles
            # n_max_obs = round(n_row * n_col * max_obs_ratio)
            # n_curr_obs = round(np.sum(map))
            # n_rm_obs = n_curr_obs - n_max_obs
            # if n_rm_obs > 0:
            #     all_obs_idx = np.transpose(
            #         np.nonzero(map == kiva_obj_types.index("@")))
            #     to_change = random.sample(list(all_obs_idx), k=n_rm_obs)
            #     for i, j in to_change:
            #         map[i, j] = kiva_obj_types.index(".")

            # map = put_endpoints(map, n_endpt=n_endpt)

            # Stack left and right additional blocks
            # l_block, r_block = get_additional_h_blocks(ADDITION_BLOCK_WIDTH,
            #                                            n_row, w_mode)
            # assert l_block.shape == r_block.shape == (n_row,
            #                                           ADDITION_BLOCK_WIDTH)
            # map_comp_unrepaired = np.hstack((l_block, map, r_block))
            # n_col_comp = n_col + 2 * ADDITION_BLOCK_WIDTH

            # # Stack top and bottom additional blocks (At this point, we assume
            # # that left and right blocks are stacked)
            # n_row_comp = n_row
            # if ADDITION_BLOCK_HEIGHT > 0:
            #     t_block, b_block = \
            #         get_additional_v_blocks(ADDITION_BLOCK_HEIGHT,
            #                                 n_col_comp, w_mode)
            #     map_comp_unrepaired = np.vstack(
            #         (t_block, map_comp_unrepaired, b_block))
            #     n_row_comp += 2 * ADDITION_BLOCK_HEIGHT

            # Repair environment here
        format_env = format_env_str(eecbs_env_number2str(map))

        logger.info(f"Repairing generated environment:\n{format_env}")

        # Limit n_obstacle? number of obstacles constraints
        # limit_n_obstacle = True
        # if self.config.obj_type == "throughput_plus_n_obstacle":
        #     limit_n_obstacle = False
        # Warm start schema
        warm_up_sols = None
        if self.config.use_warm_up:
            if parent_map is not None:
                parent_env_str = format_env_str(
                eecbs_env_number2str(parent_map))
                logger.info(f"Parent warm up solution:\n{parent_env_str}")
                warm_up_sols = [parent_map]
            # Get the solution from hamming distance objective
            hamming_repaired_env = repair_env(
                map,
                agentNum,
                add_movement=False,
                min_n_obstacles=min_n_obstacles,
                max_n_obstacles=max_n_obstacles,
                seed=repair_seed,
                # w_mode=w_mode,
                warm_envs_np=warm_up_sols,
                # limit_n_obstacle=limit_n_obstacle,
            )
            hamming_warm_env_str = format_env_str(
                eecbs_env_number2str(hamming_repaired_env))
            logger.info(
                f"Hamming warm up solution:\n{hamming_warm_env_str}")

            if parent_map is None:
                warm_up_sols = [hamming_repaired_env]
            else:
                warm_up_sols = [hamming_repaired_env, parent_map]

        # If hamming only, we just use hamming_repaired_env as the result
        # env
        if self.config.hamming_only:
            map_np_repaired = hamming_repaired_env
        else:
            map_np_repaired = repair_env(
                map,
                agentNum,
                add_movement=True,
                warm_envs_np=warm_up_sols,
                min_n_obstacles=min_n_obstacles,
                max_n_obstacles=max_n_obstacles,
                seed=repair_seed,
                # w_mode=w_mode,
                # limit_n_obstacle=limit_n_obstacle,
            )

        # Convert map layout to str format
        map_str_repaired = eecbs_env_number2str(map_np_repaired)

        format_env = format_env_str(map_str_repaired)
        logger.info(f"\nRepaired result:\n{format_env}")

        # Create json string to map layout
        map_json = json.dumps({
            "name":
            f"sol-seed={sim_seed}",
            "weight":
            False,
            "n_row":
            n_row,
            "n_col":
            n_col,
            # "n_endpoint":
            # sum(row.count('e') for row in map_str_repaired),
            # "n_agent_loc":
            # sum(row.count('r') for row in map_str_repaired),
            # "maxtime": ?
            # self.config.simulation_time, 
            "layout":
            map_str_repaired,
        })

        # else:
        #     NotImplementedError("Other warehouse types not supported yet.")

        return map_json, map_np_unrepaired, map_comp_unrepaired, map_np_repaired


    def evaluate(
        self,
        map_json: str,
        eval_logdir: pathlib.Path,
        sim_seed: int,
        agentNum: int,
        map_id: int,
        eval_id: int,
    ):
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
        output = str(eval_logdir / f"id_{map_id}-sim_{eval_id}-seed={sim_seed}")

        # We need to construct kwargs manually because some parameters
        # must NOT be passed in in order to use the default values
        # defined on the C++ side.
        # It is very dumb but works.

        kwargs = {
            "map" : map_json,
            "output" : output,
            "agentNum" : agentNum,
            "seed" : sim_seed,
            "cutoffTime": self.config.cutoffTime,
            "screen":self.config.screen,
            "stats": self.config.stats,
            # "id": self.config.id,
            # "highLevelSolver":self.config.highLevelSolver,
            # "lowLevelSolver":self.config.lowLevelSolver,
            # "inadmissibleH": self.config.inadmissibleH,
            # "suboptimality": self.config.suboptimality,
            # "heuristics": self.config.heuristics,
            # "prioritizingConflicts":self.config.prioritizingConflicts,
            # "bypass": self.config.bypass,
            # "disjointSplitting": self.config.disjointSplitting,
            # "rectangleReasoning": self.config.rectangleReasoning,
            # "corridorReasoning": self.config.corridorReasoning,
            # "targetReasoning": self.config.targetReasoning,
            "sipp": self.config.sipp,
            # "restart":self.config.restart,
            "force_new_logdir": True,
            
        }

        # For some of the parameters, we do not want to pass them in here
        # to the use the default value defined on the C++ side.
        # planning_window = self.config.planning_window
        # if planning_window is not None:
        #     kwargs["planning_window"] = planning_window

        logger.info(f"Map evaluate: {map_id} \n{map_json}")

        one_sim_result_jsonstr = warehouse_sim_PBS.run(**kwargs)

        result_json = json.loads(one_sim_result_jsonstr)
        
        return result_json


    def process_eval_result(
        self,
        curr_result_json: List[dict],
        n_evals: int,
        map_np_unrepaired: np.ndarray,
        map_comp_unrepaired: np.ndarray,
        map_np_repaired: np.ndarray,
        # w_mode: bool,
        max_n_shelf: int,
        map_id: int,
    ):
        """
        Process the evaluation result

        Args:
            results_json (List[dict]): result json of all simulations of 1 map.

        """
        # Collect the results
        keys = curr_result_json[0].keys()
        collected_results = {key: [] for key in keys}
        
        # print("Repaired shape: ",np.shape(map_np_repaired))
        # print("Comp unrepaired shape: ", np.shape(map_comp_unrepaired))
        # print("Unrepaired shape: ",np.shape(map_np_unrepaired))
        # print(collected_results)
        for result_json in curr_result_json:
            for key in keys:
                collected_results[key].append(result_json[key])

        # print(collected_results)
        # Calculate n_shelf and n_endpoint
        # Note: we use the number of searchable block (aka the portion of
        # the layout in the middle) as the totol number of tiles
        tile_ele, tile_cnt = np.unique(map_np_repaired, return_counts=True)
        tile_cnt_dict = dict(zip(tile_ele, tile_cnt))
        n_obstacles = tile_cnt_dict[eecbs_obj_types.index("@")]
        logger.info(f"Number of obstalces: {n_obstacles}")
        # n_endpoint = tile_cnt_dict[kiva_obj_types.index("e")]

        # # Get average length of all tasks
        # all_task_len_mean = collected_results.get("avg_task_len")
        # # all_task_len_mean = calc_path_len_mean(map_np_repaired, w_mode)
        # all_task_len_mean = all_task_len_mean[0]

        # logger.info(
        #     f"Map ID {map_id}: Average length of all possible tasks: {all_task_len_mean}")

        # Calculate number of connected shelf components
        format_env = format_env_str(eecbs_env_number2str(map_np_repaired))

        logger.info(f"Repaired Map: {map_id}: \n{format_env}")

        # Calculate layout entropy
        start_entropy = time.time()
        norm_pattern_dist, entropy = calc_layout_entropy(map_np_repaired)
        maze_ref_dist = [0.20222222222322223, 0.018888888889888888, 0.004444444445444445, 0.014444444445444444, 0.011111111112111112, 1e-12, 0.011111111112111112, 0.09722222222322223, 0.004444444445444445, 0.014444444445444444, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.004444444445444445, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.004444444445444445, 1e-12, 0.010000000001, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.005555555556555556, 1e-12, 1e-12, 1e-12, 0.005555555556555556, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.007777777778777778, 1e-12, 0.0033333333343333335, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.08111111111211111, 0.004444444445444445, 0.001111111112111111, 1e-12, 0.010555555556555556, 1e-12, 1e-12, 1e-12, 0.009444444445444445, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.007777777778777778, 0.05388888888988889, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.014444444445444444, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.011111111112111112, 0.0033333333343333335, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0016666666676666668, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0033333333343333335, 1e-12, 0.03222222222322222, 0.007222222223222222, 1e-12, 1e-12, 0.010555555556555556, 0.0038888888898888888, 0.004444444445444445, 1e-12, 0.002777777778777778, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.007777777778777778, 1e-12, 0.002777777778777778, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0033333333343333335, 1e-12, 0.0005555555565555556, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.007777777778777778, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.011666666667666668, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.011666666667666668, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.010000000001, 1e-12, 1e-12, 1e-12, 0.07055555555655556, 1e-12, 1e-12, 0.012222222223222223, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0066666666676666675, 1e-12, 1e-12, 1e-12, 0.005555555556555556, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.012777777778777779, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.006111111112111112, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.11611111111211112, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.010000000001, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0038888888898888888, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.017777777778777776, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12]
        room_ref_dist = [0.07111111111211112, 1e-12, 1e-12, 0.0066666666676666675, 1e-12, 0.018888888889888888, 0.022222222223222222, 0.014444444445444444, 1e-12, 0.015555555556555556, 1e-12, 0.004444444445444445, 1e-12, 0.007777777778777778, 1e-12, 0.0066666666676666675, 1e-12, 1e-12, 0.016666666667666665, 0.0066666666676666675, 1e-12, 1e-12, 0.002222222223222222, 0.00888888888988889, 0.00888888888988889, 0.0033333333343333335, 0.002222222223222222, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.018888888889888888, 0.0033333333343333335, 0.002222222223222222, 0.015555555556555556, 0.020000000001, 0.002222222223222222, 1e-12, 1e-12, 0.001111111112111111, 1e-12, 1e-12, 1e-12, 0.025555555556555556, 1e-12, 1e-12, 1e-12, 0.002222222223222222, 1e-12, 1e-12, 1e-12, 0.015555555556555556, 0.0033333333343333335, 0.0066666666676666675, 1e-12, 0.007777777778777778, 1e-12, 1e-12, 1e-12, 1e-12, 0.02111111111211111, 1e-12, 0.004444444445444445, 1e-12, 0.0033333333343333335, 1e-12, 0.0033333333343333335, 0.015555555556555556, 0.010000000001, 1e-12, 0.007777777778777778, 1e-12, 0.005555555556555556, 1e-12, 0.011111111112111112, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.004444444445444445, 0.007777777778777778, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0033333333343333335, 0.010000000001, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0033333333343333335, 0.014444444445444444, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.022222222223222222, 0.0033333333343333335, 1e-12, 1e-12, 1e-12, 0.007777777778777778, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.016666666667666665, 1e-12, 0.014444444445444444, 0.004444444445444445, 1e-12, 1e-12, 0.001111111112111111, 0.017777777778777776, 0.002222222223222222, 1e-12, 0.00888888888988889, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.002222222223222222, 1e-12, 0.001111111112111111, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.005555555556555556, 1e-12, 0.02111111111211111, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.014444444445444444, 0.005555555556555556, 0.0033333333343333335, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.004444444445444445, 0.0066666666676666675, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.004444444445444445, 1e-12, 0.0066666666676666675, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.023333333334333333, 0.001111111112111111, 0.005555555556555556, 0.007777777778777778, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.018888888889888888, 1e-12, 1e-12, 1e-12, 0.010000000001, 0.002222222223222222, 0.011111111112111112, 0.013333333334333335, 0.0033333333343333335, 1e-12, 1e-12, 1e-12, 0.002222222223222222, 1e-12, 1e-12, 1e-12, 0.004444444445444445, 1e-12, 1e-12, 1e-12, 0.00888888888988889, 1e-12, 1e-12, 1e-12, 0.005555555556555556, 1e-12, 1e-12, 1e-12, 0.023333333334333333, 1e-12, 1e-12, 1e-12, 0.018888888889888888, 0.004444444445444445, 1e-12, 1e-12, 0.005555555556555556, 1e-12, 1e-12, 1e-12, 0.004444444445444445, 0.00888888888988889, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0033333333343333335, 1e-12, 1e-12, 1e-12, 0.005555555556555556, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.023333333334333333, 1e-12, 0.004444444445444445, 1e-12, 0.007777777778777778, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.002222222223222222, 1e-12, 0.0033333333343333335, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.005555555556555556, 1e-12, 1e-12, 1e-12, 0.005555555556555556, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.014444444445444444, 0.005555555556555556, 0.0066666666676666675, 1e-12, 0.005555555556555556, 1e-12, 1e-12, 1e-12, 0.012222222223222223, 0.010000000001, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.014444444445444444, 1e-12, 0.013333333334333335, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.014444444445444444, 1e-12, 1e-12, 1e-12, 0.017777777778777776, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12]
        all_ref_dist = [0.38101899721926447, 0.012272519620151006, 0.010799579348309859, 0.004841418459963748, 0.01211036479282215, 0.003367234322640998, 0.004672003981369541, 0.024975235770048104, 0.011021869417740169, 0.0046469457458492664, 0.0024169663261690527, 0.001133364996363774, 0.0026462898895044215, 0.0008978673370924531, 0.0007524933858972114, 0.0005625664347245856, 0.010336929236361965, 0.002643078454988285, 0.004629599476314925, 0.0008992921284622882, 0.0026532103047293347, 0.0005880317534933849, 0.0008999027533350747, 0.0007400773468172196, 0.0050757627166975825, 0.000900468146735803, 0.002719768415863062, 0.000479408373345478, 0.0008088065686097429, 0.00021663613642302734, 0.00029251193080075534, 0.0004271434073821607, 0.010986407943646494, 0.0026568062067579665, 0.002399326052066332, 0.0007613587544206301, 0.004546237873311555, 0.0008947011340483751, 0.0011418911288467557, 0.0005590609956400706, 0.0034628310388361266, 0.0011166067359661895, 0.0006646086356880159, 0.00036967682213214376, 0.0010975190547576046, 0.0007120564498771287, 0.00037243594192769747, 0.00031022005211156343, 0.0049704186182739, 0.0008081054807928398, 0.002434900604840152, 0.00029646968460585296, 0.0008963973142505598, 0.00021842277956932854, 0.00047251057385659363, 0.0004294275967211027, 0.025495782166230566, 0.0024090055870867994, 0.0016476694494021982, 0.00017135943289271093, 0.002554221228129837, 0.00030339009983076644, 0.00016794445675231246, 0.0003725942520799014, 0.012116538888758103, 0.003390754688111292, 0.0026656037280732977, 0.0008996765959747835, 0.002647284981889703, 0.0008858809969970147, 0.0008854512980124613, 0.000451025624628921, 0.004522129498704504, 0.020990795396436144, 0.0007573331534074451, 0.0005625438189885564, 0.0008760883832964018, 0.00044087115915184206, 0.00027756292928550106, 0.0028834158817698397, 0.0026612162752836467, 0.0005906099474007056, 0.000799895968614266, 0.00021858108972153245, 0.0007941289559268381, 0.00013856661565047379, 0.000294027185114707, 0.00015982540751785512, 0.0008965782401387927,
                        0.0007346269544341995, 0.00029579121252497906, 0.0004291562078887532, 0.00029373318054632834, 0.0001605943425428455, 0.00012203451261318045, 0.000415111835814664, 0.0026440057001654795, 0.0008807019934463442, 0.0007657009757382229, 0.00028335255770895807, 0.0008794807437007712, 0.0003543207373683651, 0.00047196779619189454, 0.0002180835435288916, 0.0010945563933377886, 0.0008870343995345003, 0.00023741999783379695, 0.0001595766344215347, 0.0009414704761566139, 0.0007404844300657439, 0.00015261098772456294, 0.00014772598874227108, 0.0008022932366333537, 0.00018655720750428568, 0.00024578782016457473, 7.126218522778569e-05, 0.000299635887649931, 5.3033901988307665e-05, 0.0001679896882243707, 0.00010290159993253725, 0.002578917611873646, 0.00214958047909666, 0.00010032340602521654, 0.00015222652021206775, 0.0008491078102136508, 0.00022163421408546486, 7.54687121292037e-05, 0.00033971097189354775, 0.010769884886903612, 0.0026521021336639076, 0.0034171020205852275, 0.0010420200385421218, 0.0026639075478711127, 0.0008890472000410928, 0.0010246285375357215, 0.0008665671584281385, 0.0024053870693221385, 0.0007517696823442794, 0.0006608318077711514, 0.00037241332619166834, 0.0007656557442661646, 0.0002883280196353665, 0.00023215053133900988, 0.0001582649217318452, 0.004561458263659159, 0.0007973630061790035, 0.02025234638361302, 0.002035959021286315, 0.0008082637909450438, 0.0001833910044602076, 0.0022436393252418074, 0.0021793654034470227, 0.002438383428188638, 0.0002946604257235226, 0.0013695863591880273, 0.00016878123898539024, 0.00025212022625273084, 7.295836542997037e-05, 0.00010109234105020693, 0.0001519551313797182, 0.0024033290373434877, 0.0007659497488345432, 0.0006759617351746387, 0.00023515842423088405, 0.0007533075523942602, 0.00028629260339274484, 0.00037184793279094016, 0.0001577221440671461, 0.0006761426610628718, 0.0002331908551963498, 1e-12, 1e-12, 0.00023016034656844653, 6.725919995062984e-05, 1e-12, 1e-12, 0.00276576882294631, 0.0002475518475748468, 0.0014169437104330235, 9.765474917377931e-05, 0.0002979397074477463, 7.058371314691182e-05, 0.0001698441785787593, 0.00014867584965549448, 0.0016566478966057625, 0.00010050433191344958, 0.0014355338454489677, 0.00017538503390589593, 0.00010168035018696428, 2.9287379157722142e-05, 0.00018418255522122711, 0.00030974512165495173, 0.004737928851894453, 0.0008951986802410159, 0.0010379492060568786, 0.000772033381826379, 0.0008867177792300925, 0.00035590383889040416, 0.0009256846924082818, 0.0010190198350004975, 0.0011448537902665717, 0.000557387431173915, 0.0003769590891335233, 0.000305764752113825,
                        0.00046235610837951465, 0.000212022526273085, 0.00015220390447603861, 0.00014878892833564014, 0.0008963520827785015, 0.00021616120596641564, 0.0021752493394897213, 0.0003703552942130176, 0.00029936449881758147, 5.473008219049235e-05, 0.0008032430975465771, 0.00030551597901750456, 0.0004748852261396522, 0.00042974421702551054, 0.00017059049786772055, 0.0003731370297446005, 0.00016491394812440915, 0.0001039419237898772, 7.490331872847547e-05, 0.0003363638429612366, 0.0007562475980780469, 0.00029015989425372593, 0.0002359725907279327, 6.793767203150371e-05, 0.0002762964480678698, 0.00010792229333100391, 0.0001503267983856209, 4.812628726998666e-05, 0.00037404165918576563, 0.00015885293086860257, 1e-12, 1e-12, 0.00014885677554372752, 4.937015275158876e-05, 1e-12, 1e-12, 0.00029124544958312413, 7.237035629321302e-05, 0.00010079833648182825, 3.064432331946989e-05, 0.00012504240550505462, 2.2412195404866907e-05, 7.320713852629079e-05, 4.785489843763711e-05, 0.0001701607988831671, 0.00014786168315844586, 0.00017995341258378, 0.00030999389475127215, 7.381776339907727e-05, 4.851075478248185e-05, 0.00013248298265863806, 0.0007319582975827623, 0.012181084199385237, 0.0026479182224985187, 0.0026616912057402585, 0.0008780559523309361, 0.0034057263053625755, 0.0009008752299843272, 0.0009033855766835606, 0.0004485152779296877, 0.00265671574381385, 0.0008841395853227717, 0.0007624216940139992, 0.0004658163159919714, 0.0008874867142550829, 0.0003560169175705498, 0.00029009204704563853, 0.00021618382170244475, 0.002656580049397675, 0.000792184002628333, 0.0008007101351113147, 0.0003011737576999118, 0.0005801840930912771, 0.00013454101463728882, 0.00021530180799730873, 0.00016093357858328245, 0.000809530272162675, 0.0002978040130315716, 0.0002485921714321867, 0.00016256191157737975, 0.00018397901359696497, 5.208404107508424e-05, 7.28226710137956e-05, 0.00010459778013472193, 0.004656421739245471, 0.000886989168062442, 0.0007492367199090169, 0.000276680915580365, 0.021738516861031208, 0.0004518850225980279, 0.000572110275328878, 0.002764818962033087, 0.0011309224968726282, 0.0010842210019724767, 0.0002327159247397381, 0.00014958047909665966, 0.0009400456847867789, 0.0007745889599976705, 0.000159802791781826, 0.0001457358039717077, 0.0009060316177989687, 0.00029778139729554244, 0.0002882149409552208, 0.0001246353222565303, 0.0007538955615310175, 0.0001600515648781464, 0.00042933713377698625, 0.0004171020205852274, 0.0023780220287268924, 0.0007378836204223941, 0.00010127326693843996, 7.408915223142682e-05, 0.002147522447118009, 0.00020415225013494808, 0.00014867584965549448,
                        0.0003374946297626931, 0.003382635638876835, 0.0008930954167903069, 0.0008851346777080535, 0.0003609245322888708, 0.0008976864112042201, 0.0005194608418530656, 0.00035520275107350113, 0.0008336612625057557, 0.0008893412046094715, 0.00044684171346353214, 0.0002888707973000656, 0.00021297238718630843, 0.0003540041170639573, 0.0006621209047248118, 0.00010875907556408168, 0.0015512359509739918, 0.0005853178651698894, 0.00014037587453280412, 0.00018250899075507158, 5.18126522427347e-05, 0.000134744556261551, 2.3723908094556392e-05, 5.316959640448244e-05, 2.9151684741547367e-05, 0.00021851324251344505, 0.00015930524558918516, 7.334283294246556e-05, 0.00010319560450091593, 5.1676957826559926e-05, 2.910645326948911e-05, 2.234434819677952e-05, 7.015401416235836e-05, 0.0008921455558770835, 0.0003538458069117534, 0.0002807743638016374, 0.0001076735202346835, 0.00045837573883838795, 0.0007079177701837981, 0.00021342470190689102, 0.0013732727241607752, 0.0007034172387140014, 0.0007452789661039193, 6.73044314226881e-05, 4.9935546152316984e-05, 0.0007282945483460434, 0.002003211435516136, 4.9890314680258723e-05, 0.0006311373463649049, 0.00021826446941712463, 5.188049945082209e-05, 7.203112025277608e-05, 2.229911672472126e-05, 0.0001650270268045548, 2.935522636580953e-05, 0.00010391930805384806, 7.083248624323223e-05, 0.00032568921555548773, 0.00021772169175242554, 3.211434616136328e-05, 4.740258371705453e-05, 0.00023920664098009815, 7.702919791521361e-05, 4.7945361381753626e-05, 0.0002454033526520795, 0.004799036570645159, 0.000884524052835267, 0.0010531695964044825, 0.0010513829532581812, 0.0008987041193255309, 0.00035551937137790894, 0.0007476536183869778, 0.0010625777425926001, 0.0007628513929985526, 0.00027475857801788903, 0.000231472059258136, 0.00014946740041651402, 0.0002882149409552208, 0.00010934708470083904, 6.658072786975597e-05, 4.961892584790918e-05, 0.0008959223837939481, 0.0002949544302919013, 0.0020546848507184343, 0.0006872696031892032, 0.00021975710799504715, 5.314698066845331e-05, 0.00037551168202765904, 0.0002940724165867653, 0.00029529366633233823, 0.00012187620246097654, 9.978062836051744e-05, 7.499378167259199e-05, 7.436054106377637e-05, 2.2231269516633872e-05, 3.007892991874166e-05, 4.767397254940408e-05, 0.001147386752701834, 0.0004716511758874867, 0.00037372503888135787, 0.00014935432173636835, 0.0005722459697450528, 0.00021199991053705587,
                        0.0003076418582042427, 0.00015075649737017436, 0.0003701065211166972, 0.00014840446082314495, 1e-12, 1e-12, 0.00015751860244288396, 5.097587000965692e-05, 1e-12, 1e-12, 0.00047891082715283714, 0.00016387362426706923, 0.00016973109989861366, 7.52425547689124e-05, 0.00042770880078288895, 0.00010305991008474116, 0.0003679354104579008, 0.0003393038886450234, 0.00016830630852877852, 7.327498573437817e-05, 0.00017882262578232353, 0.0001342017785968519, 0.00015247529330838816, 4.715381062073411e-05, 0.000310491440943913, 0.0007334057046886266, 0.02569952733211699, 0.00044564307945398827, 0.0008750028279670036, 0.0010752877862409706, 0.0004388357429092204, 0.0008330732533689983, 0.001012958817744691, 0.003466087704824321, 0.0005582468291430219, 0.0027924101599886243, 0.00015876246792448606, 0.00014840446082314495, 0.00021704321967155166, 0.0013947576733884478, 4.952846290379266e-05, 0.0008991112025740553, 0.0007419544529076374, 0.0001612954303597485, 0.002202727458765113, 0.0002864509135449488, 0.00016226790700900106, 2.9400457837867786e-05, 0.00033111699220247866, 0.00012764321514840445, 0.00042854558301596673, 0.00041669493733670305, 0.00014648212326066898, 0.00033794694448327565, 0.00010256236389210032, 7.234774055718388e-05, 4.7900129909695366e-05, 0.0002458104359006038, 0.0005609607174665174, 0.00021412578972379402, 0.0001567044359458353, 5.027478219275392e-05, 0.0029073885619607166, 0.001602076125567474, 0.00015116358061869868, 0.0008596241274671959, 0.000306646765818961, 0.00014532872072318338, 1e-12, 1e-12, 0.00014768075727021281, 0.0005820159677096366, 1e-12, 1e-12, 0.0004286360459600832, 0.00010369315069355678, 0.00014982925219298007, 4.80810557979284e-05, 0.00041434290078967364, 6.866137558443585e-05, 0.0003401406708781012, 0.00024614967194104076, 0.00036687247086453174, 0.0003351425932156637, 0.00031259470439462197, 0.00072734468743282, 0.00033758509270680957, 0.00024612705620501163, 0.0007280231595136939, 0.02379489789095183]
        maze_kl_div = scipy.stats.entropy(norm_pattern_dist, np.array(maze_ref_dist), base = len(maze_ref_dist))
        room_kl_div = scipy.stats.entropy(norm_pattern_dist, np.array(room_ref_dist), base = len(room_ref_dist)) 
        all_kl_div = scipy.stats.entropy(norm_pattern_dist, np.array(all_ref_dist), base = len(all_ref_dist))  

        logger.info(f"Map ID {map_id}: Layout entropy: {entropy}, maze_KL_div: {maze_kl_div}, room_kl_div: {room_kl_div}")
        logger.info("cal_entropy done after %f sec", time.time() - start_entropy)



        # start_components = time.time()
        # n_shelf_components = calc_num_shelf_components(map_np_repaired)
        # logger.info(
        #     f"Map ID {map_id}: Number of connected shelf components: {n_shelf_components}")
        # logger.info("cal_shelf_comp done after %f sec", time.time() - start_components)

        #Connectivity of the Graph (need for heatmap)
        start_connectivity = time.time() 
        std_BC,connectivity_graph = cal_connectivity(map_np_repaired)
        logger.info("cal_shelf_comp done after %f sec", time.time() - start_connectivity) 
        #average degrees of all the grids in the Graph 
        start_degree = time.time()  
        avg_degrees = cal_avg_degrees(map_np_repaired)
        logger.info("cal_degree done after %f sec", time.time() - start_degree) 

        # Post process result if necessary
        tile_usage = np.array(collected_results.get("tile_usage"))

        # print("Tile_usage shape: ", tile_usage.shape)
        tile_usage = tile_usage.reshape(n_evals, *map_np_repaired.shape)
        # tasks_finished_timestep = [np.array(x) for x in collected_results.get("tasks_finished_timestep")]

        # print("run_time in json: ",collected_results.get("runtime"))
        run_time = np.array(collected_results.get("runtime"))
        avg_run_time = np.mean(run_time)
        # print("shape of run_time: ", run_time.shape)

        # print("solution_cost in json: ", collected_results.get("solution_cost"))
        solution_cost = np.array(collected_results.get("solution_cost")) 

        Suboptimality = np.array(collected_results.get("suboptimality"))

        # print("suboptimality_list: ", Suboptimality)

        average_suboptimality = np.mean(Suboptimality)
        # Get objective based on type
        # objs = entropy
        objs = -avg_run_time

        # Get start and goal locations
        start_locations = np.array(collected_results.get("start_locations"))
        goal_locations = np.array(collected_results.get("goal_locations"))
        


        # if self.config.obj_type == "throughput":
        #     objs = np.array(collected_results.get("throughput"))
        # elif self.config.obj_type == "throughput_plus_n_shelf":
        #     objs = np.array(collected_results.get("throughput")) - \
        #         (max_n_shelf - n_shelf)**2 * 0.5
        # else:
        #     return ValueError(
        #         f"Object type {self.config.obj_type} not supported")
        


        # Create WarehouseResult object using the mean of n_eval simulations
        # For tile_usage, num_wait, and finished_task_len, the mean is not taken
        metadata = WarehouseMetadata(
            objs=objs,
            # throughput=collected_results.get("throughput"),
            solution_cost=solution_cost,
            avg_run_time=avg_run_time,
            map_int_unrepaired=map_comp_unrepaired,
            map_int=map_np_repaired,
            map_int_raw=map_np_unrepaired,
            map_str=eecbs_env_number2str(map_np_repaired),
            n_obstacles=n_obstacles,
            # n_endpoint=n_endpoint,

            tile_usage=tile_usage,
            tile_usage_mean=np.mean(collected_results.get("tile_usage_mean")),
            tile_usage_std=np.mean(collected_results.get("tile_usage_std")),
            start_locations = start_locations,
            goal_locations = goal_locations,

            # num_wait=collected_results.get("num_wait"),
            # num_wait_mean=np.mean(collected_results.get("num_wait_mean")),
            # num_wait_std=np.mean(collected_results.get("num_wait_std")),
            # finished_task_len=collected_results.get("finished_task_len"),
            # finished_len_mean=np.mean(
            #     collected_results.get("finished_len_mean")),
            # finished_len_std=np.mean(collected_results.get("finished_len_std")),
            # all_task_len_mean=None,                                                     #not used in process_eval_result
            # tasks_finished_timestep=tasks_finished_timestep,
            # n_shelf_components=n_shelf_components,
            average_suboptimality = average_suboptimality,
            std_BC = std_BC,
            avg_degrees = avg_degrees,
            layout_entropy=entropy,
            connectivity_graph = connectivity_graph,
            maze_kl_div = maze_kl_div,
            room_kl_div = room_kl_div,
            all_kl_div = all_kl_div, 
        )
        result = WarehouseResult.from_raw(
            warehouse_metadata=metadata,
            opts={
                "aggregation": self.config.aggregation_type,
                "measure_names": self.config.measure_names,
            },
        )

        return result



    def actual_qd_score(self, objs):
        """Calculates QD score of the given objectives.

        Scores are normalized to be non-negative by subtracting a constant min
        score.

        Args:
            objs: List of objective values.
        """
        objs = np.array(objs)
        objs -= MIN_SCORE
        if np.any(objs < 0):
            warnings.warn("Some objective values are still negative.")
        return np.sum(objs)


logger = logging.getLogger(__name__)
d = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# RHCR Def
# def has_endpoint_around(env_np, i, j, n_endpt=2):
#     endpoint_cnt = 0
#     n_row, n_col = env_np.shape
#     for dx, dy in d:
#         n_i, n_j = i + dx, j + dy
#         if 0 <= n_i < n_row and 0 <= n_j < n_col:
#             if env_np[n_i, n_j] == kiva_obj_types.index("e"):
#                 endpoint_cnt += 1
#                 if endpoint_cnt >= n_endpt:
#                     return True
#     return False


# def put_endpoints(map, n_endpt=2):
#     # Use a new order of putting endpoints everytime
#     cur_d = copy.deepcopy(d)
#     # np.random.shuffle(cur_d)

#     # Put endpoint around the obstacles
#     n_row, n_col = map.shape
#     for i in range(n_row):
#         for j in range(n_col):
#             if map[i, j] == kiva_obj_types.index("@"):
#                 for dx, dy in cur_d:
#                     n_i, n_j = i + dx, j + dy
#                     # if in range and the tile is empty space, add endpoint
#                     if 0 <= n_i < n_row and \
#                         0 <= n_j < n_col and \
#                         map[n_i, n_j] == kiva_obj_types.index(".") and \
#                         not has_endpoint_around(map, i, j, n_endpt=n_endpt):
#                         map[n_i, n_j] = kiva_obj_types.index("e")
#     return map


# def reaches_goal(loc, goal_locs):
#     """
#     A `loc` reaches goal if any loc in `goal_locs` is around `loc`.
#     """
#     x, y = loc
#     around_goal = False
#     goals = []
#     for goal_loc in goal_locs:
#         for dx, dy in d:
#             n_x = x + dx
#             n_y = y + dy
#             if tuple([n_x, n_y]) == goal_loc:
#                 goals.append(goal_loc)
#                 around_goal = True
#     return around_goal, goals


# def BFS_path_len(start_loc, goal_locs, env_np, w_mode):
#     """
#     Find shortest path from start_loc to all goal_locs
#     """
#     # Set goal loc as none-blocking tile, otherwise it cannot be reached.
#     # env_np[goal_locs] = kiva_obj_types.index(".")
#     result_path_len = {}
#     n_goals = len(goal_locs)
#     q = Queue()
#     q.put(start_loc)
#     seen = set()
#     m, n = env_np.shape
#     dist_matrix = np.full((m, n), np.inf)
#     dist_matrix[start_loc] = 0
#     block_idxs = [
#         eecbs_obj_types.index("@"),
#         # kiva_obj_types.index("e"),
#         eecbs_obj_types.index("w"),
#         eecbs_obj_types.index("r"),
#     ]
#     if not w_mode:
#         block_idxs.append(eecbs_obj_types.index("e"))
#     while not q.empty():
#         curr = q.get()
#         x, y = curr
#         around_goal, goals = reaches_goal(curr, goal_locs)
#         if around_goal:
#             shortest = dist_matrix[x, y] + 1
#             for goal_reached in goals:
#                 result_path_len[goal_reached] = shortest
#                 goal_locs.remove(goal_reached)

#             # print(f"Found goal {goal_reached}")
#             # print(f"Remaining number of goals {len(goal_locs)}")

#             # All goals found?
#             if len(goal_locs) == 0:
#                 assert len(result_path_len) == n_goals
#                 return result_path_len

#         seen.add(curr)
#         for dx, dy in d:
#             n_x = x + dx
#             n_y = y + dy
#             if n_x < m and n_x >= 0 and \
#                n_y < n and n_y >= 0 and \
#                env_np[n_x,n_y] not in block_idxs and\
#                (n_x, n_y) not in seen:
#                 q.put((n_x, n_y))
#                 dist_matrix[n_x, n_y] = dist_matrix[x, y] + 1
#     raise ValueError(f"Start loc: {start_loc}. Remaining goal: {goal_locs}")


# def calc_path_len_mean(repaired_env, w_mode):
#     if w_mode:
#         start_locs = np.where(repaired_env == eecbs_obj_types.index("w"))
#     else:
#         start_locs = np.where(repaired_env == eecbs_obj_types.index("e"))

#     start_locs = np.stack(start_locs, axis=1)
#     end_locs = np.where(repaired_env == eecbs_obj_types.index("e"))
#     end_locs = np.stack(end_locs, axis=1)

#     path_length_table = {}
#     for start_loc in start_locs:
#         path_length_table[tuple(start_loc)] = BFS_path_len(
#                 tuple(start_loc),
#                 [tuple(end_loc) for end_loc in end_locs],
#                 copy.deepcopy(repaired_env),
#                 w_mode,
#             )

#     all_path_length = []
#     for start_loc in start_locs:
#         for end_loc in end_locs:
#             if tuple(start_loc) != tuple(end_loc):
#                 all_path_length.append(
#                     path_length_table[tuple(start_loc)][tuple(end_loc)])

#     return np.mean(all_path_length)


def BFS_shelf_component(start_loc, env_np, env_visited):
    """
    Find all shelves that are connected to the shelf at start_loc.
    """
    # We must start searching from shelf
    assert env_np[start_loc] == eecbs_obj_types.index("@")

    q = Queue()
    q.put(start_loc)
    seen = set()
    m, n = env_np.shape
    block_idxs = [
        eecbs_obj_types.index("."),
    ]
    while not q.empty():
        curr = q.get()
        x, y = curr
        env_visited[x,y] = True
        seen.add(curr)
        for dx, dy in d:
            n_x = x + dx
            n_y = y + dy
            if n_x < m and n_x >= 0 and \
               n_y < n and n_y >= 0 and \
               env_np[n_x,n_y] not in block_idxs and\
               (n_x, n_y) not in seen:
                q.put((n_x, n_y))


def calc_num_shelf_components(repaired_env):
    env_visited = np.zeros(repaired_env.shape, dtype=bool)
    n_row, n_col = repaired_env.shape
    n_shelf_components = 0
    for i in range(n_row):
        for j in range(n_col):
            if repaired_env[i,j] == eecbs_obj_types.index("@") and\
                not env_visited[i,j]:
                n_shelf_components += 1
                BFS_shelf_component((i, j), repaired_env, env_visited)
    return n_shelf_components

def cal_connectivity(repaired_env):
    n_row, n_col = repaired_env.shape
    total_grid = n_row * n_col
    num_of_obstacle = 0
    total_grids = n_row * n_col
    distance_matrix = np.zeros((total_grid,total_grid), dtype=int) 
    connectivity_graph = np.zeros(repaired_env.shape, dtype=int)

    #build the distance matrix for graphs
    for i in range(n_row):
        for j in range(n_col):
            if repaired_env[i,j] == eecbs_obj_types.index("@"):
                num_of_obstacle += 1
            else: 
                if i == 0 and j == 0:
                    if repaired_env[i+1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i+1)*n_col+j] = 1
                    if repaired_env[i,j+1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j+1] = 1
                elif i == n_row-1 and j == 0:
                    if repaired_env[i-1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i-1)*n_col+j] = 1
                    if repaired_env[i,j+1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j+1] = 1
                elif i == 0 and j == n_col - 1:
                    if repaired_env[i+1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i+1)*n_col+j] = 1
                    if repaired_env[i,j-1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j-1] = 1
                elif i == n_row-1 and j == n_col - 1:
                    if repaired_env[i-1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i-1)*n_col+j] = 1
                    if repaired_env[i,j-1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j-1] = 1
                #upper and down edge 
                elif i == 0:
                    if repaired_env[i+1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i+1)*n_col+j] = 1
                    if repaired_env[i,j-1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j-1] = 1
                    if repaired_env[i,j+1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j+1] = 1
                elif i == n_row -1:
                    if repaired_env[i-1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i-1)*n_col+j] = 1
                    if repaired_env[i,j-1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j-1] = 1
                    if repaired_env[i,j+1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j+1] = 1
                #left and right edge  
                elif j == 0:
                    if repaired_env[i+1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i+1)*n_col+j] = 1
                    if repaired_env[i-1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i-1)*n_col+j] = 1
                    if repaired_env[i,j+1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j+1] = 1
                elif j == n_col -1:
                    if repaired_env[i-1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i-1)*n_col+j] = 1
                    if repaired_env[i+1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i+1)*n_col+j] = 1
                    if repaired_env[i,j-1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j-1] = 1
                #normal cases
                else:
                    if repaired_env[i-1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i-1)*n_col+j] = 1
                    if repaired_env[i+1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i+1)*n_col+j] = 1
                    if repaired_env[i,j-1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j-1] = 1
                    if repaired_env[i,j+1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j+1] = 1

    distance_graph = scipy.sparse.csr_matrix(distance_matrix)

    #The matrix of distances between graph nodes. dist_matrix[i,j] gives the shortest distance from point i to point j along the graph.
    dist_matrix, predecessors= scipy.sparse.csgraph.floyd_warshall(csgraph=distance_graph, directed=False, return_predecessors=True)


    for i in range(predecessors.shape[0]):
        for j in range(predecessors.shape[1]):
            if predecessors[i,j] != -9999:
                # print("Processor: ", predecessors[i,j])
                result_row = predecessors[i,j] // n_col
                # print("Row: ", result_row)
                result_col = predecessors[i,j] % n_col
                # print("Col: ", result_col)
                connectivity_graph[result_row,result_col] += 1

    normalized_graph = connectivity_graph/(np.sum(connectivity_graph))

    # print("Conectivity graph: ", connectivity_graph)
    # print("Normalized graph: ", normalized_graph)
    
    # print("Sum: ", np.sum(connectivity_graph))
    # print("Mean: ", np.mean(connectivity_graph))
    connectivity_graph_nozero = normalized_graph[normalized_graph != 0]
    # sqr_diff= np.sum(np.square(normalized_graph-1))

    # total_obstacle_diff = 1 * num_of_obstacle

    # sq_sum_diff = sqr_diff - total_obstacle_diff

    # connectivity_std = np.sqrt(sq_sum_diff / (total_grids - num_of_obstacle))

    connectivity_std = np.std(connectivity_graph_nozero)
    before_norm = connectivity_std * (np.sum(connectivity_graph))

    # print("Connectivity_Std: ",connectivity_std)
    # print("Connectivity_Std before norm: ",before_norm)

    return connectivity_std,connectivity_graph





def cal_avg_degrees(repaired_env):
    n_row, n_col = repaired_env.shape
    sum_degree = 0
    num_of_obstacle = 0
    total_grids = n_row * n_col 
    for i in range(n_row):
        for j in range(n_col):
            #four edge cases in the corner
            if repaired_env[i,j] == eecbs_obj_types.index("@"):
                num_of_obstacle += 1 
            else:
                if i == 0 and j == 0:
                    if repaired_env[i+1,j] != eecbs_obj_types.index("@"): sum_degree += 1
                    if repaired_env[i,j+1] != eecbs_obj_types.index("@"): sum_degree += 1
                elif i == n_row-1 and j == 0:
                    if repaired_env[i-1,j] != eecbs_obj_types.index("@"): sum_degree += 1
                    if repaired_env[i,j+1] != eecbs_obj_types.index("@"): sum_degree += 1 
                elif i == 0 and j == n_col - 1:
                    if repaired_env[i+1,j] != eecbs_obj_types.index("@"): sum_degree += 1
                    if repaired_env[i,j-1] != eecbs_obj_types.index("@"): sum_degree += 1  
                elif i == n_row-1 and j == n_col - 1:
                    if repaired_env[i-1,j] != eecbs_obj_types.index("@"): sum_degree += 1
                    if repaired_env[i,j-1] != eecbs_obj_types.index("@"): sum_degree += 1  
                #upper and down edge 
                elif i == 0:
                    if repaired_env[i+1,j] != eecbs_obj_types.index("@"): sum_degree += 1
                    if repaired_env[i,j-1] != eecbs_obj_types.index("@"): sum_degree += 1 
                    if repaired_env[i,j+1] != eecbs_obj_types.index("@"): sum_degree += 1   
                elif i == n_row -1:
                    if repaired_env[i-1,j] != eecbs_obj_types.index("@"): sum_degree += 1
                    if repaired_env[i,j-1] != eecbs_obj_types.index("@"): sum_degree += 1 
                    if repaired_env[i,j+1] != eecbs_obj_types.index("@"): sum_degree += 1   
                #left and right edge  
                elif j == 0:
                    if repaired_env[i+1,j] != eecbs_obj_types.index("@"): sum_degree += 1
                    if repaired_env[i-1,j] != eecbs_obj_types.index("@"): sum_degree += 1 
                    if repaired_env[i,j+1] != eecbs_obj_types.index("@"): sum_degree += 1   
                elif j == n_col -1:
                    if repaired_env[i-1,j] != eecbs_obj_types.index("@"): sum_degree += 1
                    if repaired_env[i+1,j] != eecbs_obj_types.index("@"): sum_degree += 1 
                    if repaired_env[i,j-1] != eecbs_obj_types.index("@"): sum_degree += 1    
                #normal cases
                else:
                    if repaired_env[i-1,j] != eecbs_obj_types.index("@"): sum_degree += 1
                    if repaired_env[i+1,j] != eecbs_obj_types.index("@"): sum_degree += 1 
                    if repaired_env[i,j-1] != eecbs_obj_types.index("@"): sum_degree += 1 
                    if repaired_env[i,j+1] != eecbs_obj_types.index("@"): sum_degree += 1 
        
    return (sum_degree/(total_grids-num_of_obstacle))


# def get_additional_h_blocks(ADDITION_BLOCK_WIDTH, n_row, w_mode):
#     """
#     Generate additional blocks to horizontally stack to the map on the left and
#     right side
#     """

#     if w_mode:
#         # In 'w' mode, horizontally stack the workstations
#         # The workstation locations are fixed as the following:
#         # 1. Stack workstations on the border of the generated map,
#         #    meaning that there is no columns on the left/right side of the
#         #    left/right workstations.
#         # 2. The first row and last row has no workstations.
#         # 3. For the rest of the rows, starting from the second row, put
#         # workstations for every three rows, meaning that there are at least
#         # two empty cells between each pair of workstations.
#         # 4. The left and right side of workstation blocks are symmetrical
#         l_block = []
#         r_block = []
#         for i in range(n_row):
#             curr_l_row = None
#             curr_r_row = None
#             if i == 0 or i == n_row - 1 or (i - 1) % 3 != 0:
#                 curr_l_row = [
#                     kiva_obj_types.index(".")
#                     for _ in range(ADDITION_BLOCK_WIDTH)
#                 ]
#                 curr_r_row = copy.deepcopy(curr_l_row)
#             elif (i - 1) % 3 == 0:
#                 curr_l_row = [
#                     kiva_obj_types.index("w"),
#                     kiva_obj_types.index(".")
#                 ]
#                 curr_r_row = [
#                     kiva_obj_types.index("."),
#                     kiva_obj_types.index("w")
#                 ]
#             l_block.append(curr_l_row)
#             r_block.append(curr_r_row)
#         r_block = np.array(r_block)
#         l_block = np.array(l_block)

#     else:
        # In 'r' mode, horizontally stack the robot start locations
        # The robot start locations are fixed as the following:
        # 1. Stack robot location blocks on either sides of the generated map
        # 2. On each side, the length of the block is 4
        # 3. The top and bottom rows and the left and right columns have no
        #    robots
        # 4. Starting from the 2nd row, there are 2 robots in the middle column
        # 5. There are at most 3 sequential rows of robots
        # 6. For every 3 rows, append a row of empty space
    #     r_block = []
    #     n_robot_row = 0
    #     for i in range(n_row):
    #         curr_row = None
    #         if i == 0 or i == n_row - 1:
    #             curr_row = [
    #                 kiva_obj_types.index(".")
    #                 for _ in range(ADDITION_BLOCK_WIDTH)
    #             ]
    #         elif n_robot_row < 3:
    #             curr_row = [
    #                 kiva_obj_types.index("."),
    #                 kiva_obj_types.index("r"),
    #                 kiva_obj_types.index("r"),
    #                 kiva_obj_types.index("."),
    #             ]
    #             n_robot_row += 1
    #         elif n_robot_row >= 3:
    #             curr_row = [
    #                 kiva_obj_types.index(".")
    #                 for _ in range(ADDITION_BLOCK_WIDTH)
    #             ]
    #             n_robot_row = 0
    #         r_block.append(curr_row)

    #     # Under 'r' mode, left and right blocks are the same
    #     r_block = np.array(r_block)
    #     l_block = copy.deepcopy(r_block)

    # return l_block, r_block


# def get_additional_v_blocks(ADDITION_BLOCK_HEIGHT, n_col_comp, w_mode):
#     """
#     Generate additional blocks to vertically stack to the map on the top and
#     bottom
#     """
#     # Only applicable for r mode
#     assert not w_mode
#     # We only want even # of cols to make the map symmetrical
#     assert n_col_comp % 2 == 0
#     t_block = []
#     b_block = []
#     # For r mode, we need to append additional on top and bottom of the map
#     for i in range(ADDITION_BLOCK_HEIGHT):
#         # We add 'r' to everywhere except for:
#         # 1) the first and block column of each row
#         # 2) the 2 columns in the middle of each row
#         # 2) the first row for t_block and last row for b_block
#         cont_r_length = (n_col_comp - 4) // 2

#         if i == 0 or i == ADDITION_BLOCK_HEIGHT - 1:
#             t_block.append([
#                 kiva_obj_types.index(".") for _ in range(n_col_comp)
#             ])
#         else:
#             t_block.append([
#                 kiva_obj_types.index("."),
#                 *[kiva_obj_types.index("r") for _ in range(cont_r_length)],
#                 kiva_obj_types.index("."),
#                 kiva_obj_types.index("."),
#                 *[kiva_obj_types.index("r") for _ in range(cont_r_length)],
#                 kiva_obj_types.index("."),
#             ])

#     t_block = np.array(t_block)
#     b_block = copy.deepcopy(t_block)
#     assert t_block.shape == b_block.shape == (ADDITION_BLOCK_HEIGHT, n_col_comp)
#     return t_block, b_block

# def get_comp_map(
#     map,
#     seed,
#     w_mode,
#     n_endpt,
#     lvl_height,
# ):
#     """
#     Helper function that repair one map using hamming for EM-ME inner
#     loop.
#     """
#     np.random.seed(seed // np.int32(4))

#     # Put endpoints in raw maps and repair using hamming distance obj
#     ADDITION_BLOCK_WIDTH = KIVA_WORKSTATION_BLOCK_WIDTH if w_mode \
#                             else KIVA_ROBOT_BLOCK_WIDTH
#     # map = put_endpoints(map, n_endpt=n_endpt)
#     l_block, r_block = get_additional_h_blocks(ADDITION_BLOCK_WIDTH, lvl_height,
#                                              w_mode)
#     map_comp = np.hstack((l_block, map, r_block))

#     # Same as MILP, in the surrogate model, we replace 'w' with 'r' under
#     # w_mode to use 'r' internally.
#     if w_mode:
#         map_comp = flip_tiles(
#             map_comp,
#             'w',
#             'r',
#         )
#     return map_comp

def calc_layout_entropy(map_np_repaired):
    """
    Calculate entropy of the storage area of the layout.

    We first formulate the layout as a tile pattern distribution by following
    Lucas, Simon M. M. and Vanessa Volz. “Tile pattern KL-divergence for
    analysing and evolving game levels.” Proceedings of the Genetic and
    Evolutionary Computation Conference (2019).

    Then we calculate the entropy.
    """
    # Separate the storage area
    h, w = map_np_repaired.shape
    storage_area = map_np_repaired
    # if w_mode:
    #     storage_area = map_np_repaired[:,
    #         KIVA_WORKSTATION_BLOCK_WIDTH:w-KIVA_WORKSTATION_BLOCK_WIDTH]
    # else:
    #     storage_area = map_np_repaired[
    #         KIVA_ROBOT_BLOCK_WIDTH:h-KIVA_ROBOT_BLOCK_WIDTH,
    #         KIVA_ROBOT_BLOCK_WIDTH:w-KIVA_ROBOT_BLOCK_WIDTH]

    # Generate list of patterns (we use 2 x 2)
    storage_obj_types = eecbs_obj_types[:2]
    tile_patterns = {
        # for 2 * 2 window
        # "".join(x): 0 for x in product(storage_obj_types, repeat=4) 
        "".join(x): 0 for x in product(storage_obj_types, repeat=9)
    }

    s_h, s_w = storage_area.shape
    # Iterate over 2x2 blocks
    for i in range(s_h - 2):
        for j in range(s_w - 2):
            # take the constant as a parameter to test different window sizes
            curr_block = storage_area[i:i+3,j:j+3]
            curr_pattern = "".join(eecbs_env_number2str(curr_block))
            tile_patterns[curr_pattern] += 1
    pattern_dist = list(tile_patterns.values())

    np_pattern_dist = np.array(pattern_dist,dtype = int)

    sum_pattern = np.sum(np_pattern_dist)
    norm_pattern_dist = np_pattern_dist/sum_pattern

    epsilon = 1e-12
    norm_pattern_dist += epsilon

    # Use number of patterns as the base to bound the entropy to [0, 1]
    return norm_pattern_dist,entropy(pattern_dist, base=len(pattern_dist))


def single_simulation(seed, agent_num, kwargs, results_dir):
    kwargs["seed"] = int(seed)
    output_dir = os.path.join(results_dir,
        f"sim-agent_num={agent_num}-seed={seed}")
    # kwargs["output"] = output_dir
    kwargs["agentNum"] = agent_num

    # breakpoint()
    # for i in kwargs:
    #     print(i)
    print(kwargs["stats"])
    
    result_jsonstr = warehouse_sim_PBS.run(**kwargs)
    
    result_json = json.loads(result_jsonstr)

    solution_cost = result_json["runtime"]

    print("runtime: ",solution_cost)

    # print("Tile_usage_shape: ", len(result_json["tile_usage"]))

    # if result_json["throughput"] > max_obj:
    #     max_obj = result_json["throughput"]

    # if result_json["throughput"] < min_obj:
    #     min_obj = result_json["throughput"]

    # Write result to logdir
    # Load and then dump to format the json
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(output_dir, f"result.json"), "w") as f:
        f.write(json.dumps(result_json, indent=4))

    # Write kwargs to logdir
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        f.write(json.dumps(kwargs, indent=4))

    # # Increment number of agents if necessary
    # if mode == "inc_agents":
    #     kwargs["agentNum"] += 1

    return solution_cost

# def test_calc_path_len_mean(map_filepath):
#     # Read in map
#     with open(map_filepath, "r") as f:
#         raw_env_json = json.load(f)
#     repaired_env_str = raw_env_json["layout"]
#     repaired_env = kiva_env_str2number(repaired_env_str)

#     avg_len = calc_path_len_mean(repaired_env, True)
#     print(f"Average length (naive BFS): {avg_len}")


def test_calc_num_shelf_components(map_filepath):
    with open(map_filepath, "r") as f:
        raw_env_json = json.load(f)
    repaired_env_str = raw_env_json["layout"]
    repaired_env = kiva_env_str2number(repaired_env_str)
    n_shelf_components = calc_num_shelf_components(repaired_env)
    print(f"Number of connected shelf components: {n_shelf_components}")

def main(
    warehouse_config,
    map_filepath,
    agent_num=10,
    seed=0,
    n_evals=10,
    n_sim=2, # Run `inc_agents` `n_sim`` times
    mode="constant",
):
    """
    For testing purposes. Graph a map and run one simulation.

    Args:
        mode: "constant", "inc_agents", or "inc_timesteps".
              "constant" will run `n_eval` simulations with the same
              `agent_num`.
              "increase" will run `n_eval` simulations with an inc_agents
              number of `agent_num`.
    """
    setup_logging(on_worker=False)

    gin.parse_config_file(warehouse_config)

    # Read in map
    with open(map_filepath, "r") as f:
        raw_env_json = json.load(f)

    # Create log dir
    map_name = raw_env_json["name"]
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    base_log_dir = time_str + "_" + map_name
    log_dir = os.path.join(LOG_DIR, base_log_dir)
    results_dir = os.path.join(log_dir, "results")
    os.mkdir(log_dir)
    os.mkdir(results_dir)

    # Write map file to logdir
    with open(os.path.join(log_dir, "map.json"), "w") as f:
        f.write(json.dumps(raw_env_json, indent=4))


    # Construct kwargs
    kwargs = {
        "map" : json.dumps(raw_env_json),
        # "output" : log_dir,
        "agentNum" : agent_num,
        "cutoffTime": gin.query_parameter("WarehouseEECBSConfig.cutoffTime"),
        "screen":gin.query_parameter("WarehouseEECBSConfig.screen"),
        "stats": gin.query_parameter("WarehouseEECBSConfig.stats"),
        # "id": gin.query_parameter("WarehouseEECBSConfig.id"),
        # "highLevelSolver":gin.query_parameter("WarehouseEECBSConfig.highLevelSolver"),
        # "lowLevelSolver":gin.query_parameter("WarehouseEECBSConfig.lowLevelSolver"),
        # "inadmissibleH": gin.query_parameter("WarehouseEECBSConfig.inadmissibleH"),
        # "suboptimality": gin.query_parameter("WarehouseEECBSConfig.suboptimality"),
        # "heuristics": gin.query_parameter("WarehouseEECBSConfig.heuristics"),
        # "prioritizingConflicts":gin.query_parameter("WarehouseEECBSConfig.prioritizingConflicts"),
        # "bypass": gin.query_parameter("WarehouseEECBSConfig.bypass"),
        # "disjointSplitting": gin.query_parameter("WarehouseEECBSConfig.disjointSplitting"),
        # "rectangleReasoning": gin.query_parameter("WarehouseEECBSConfig.rectangleReasoning"),
        # "corridorReasoning": gin.query_parameter("WarehouseEECBSConfig.corridorReasoning"),
        # "targetReasoning": gin.query_parameter("WarehouseEECBSConfig.targetReasoning"),
        "sipp": gin.query_parameter("WarehouseEECBSConfig.sipp"),
        # "restart":gin.query_parameter("WarehouseEECBSConfig.restart"),
        "force_new_logdir": False,
    }

    # For some of the parameters, we do not want to pass them in here
    # to the use the default value defined on the C++ side.
    # try:
    #     planning_window = gin.query_parameter("WarehouseConfig.planning_window")
    #     if planning_window is not None:
    #         kwargs["planning_window"] = planning_window
    # except ValueError:
    #     pass
    n_success = 0
    n_workers = 32
    pool = multiprocessing.Pool(n_workers)
    if mode == "inc_agents":
        seeds = []
        agent_nums = []
        for i in range(n_sim):
            seeds.extend(repeat(seed + i, n_evals))
            agent_nums.extend([agent_num + j for j in range(n_evals)])

        solution_cost = pool.starmap(
            single_simulation,
            zip(seeds,
                agent_nums,
                repeat(kwargs, n_evals * n_sim),
                repeat(results_dir, n_evals * n_sim)),
        )
    elif mode == "constant":
        agent_nums = [agent_num for _ in range(n_evals)]
        seeds = np.random.choice(np.arange(10000), size=n_evals, replace=False)

        solution_cost = pool.starmap(
            single_simulation,
            zip(seeds,
                agent_nums,
                repeat(kwargs, n_evals),
                repeat(results_dir, n_evals)),
        )

    n_failure = np.array(solution_cost) >= 10
    n_success = 200 - np.sum(n_failure)
    succ_rate = n_success/200
    
    avg_obj = np.mean(solution_cost)
    std_obj = np.std(solution_cost)
    max_obj = np.max(solution_cost)
    min_obj = np.min(solution_cost)

    n_evals = n_evals * n_sim if mode == "inc_agents" else n_evals

    print(f"Number of success simulations: {n_success}")
    print(f"Success rate: {succ_rate}")
    

    print(f"Average Runtime over {n_evals} simulations: {avg_obj}")
    print(f"Std Runtime over {n_evals} simulations: {std_obj}")
    print(f"Max Runtime over {n_evals} simulations: {max_obj}")
    print(f"Min Runtime over {n_evals} simulations: {min_obj}")



if __name__ == "__main__":
    fire.Fire(main)
