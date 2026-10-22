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
import warehouse_sim_EECBS  # type: ignore # ignore pylance warning
from mapf_py import warehouse_sim_PIBT, warehouse_sim_LaCAM  # Python PIBT / LaCAM, see mapf_py/
import numpy as np
import multiprocessing
import scipy
import time
import itertools
import networkx as nx

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
                              eecbs_env_number2str, eecbs_env_str2number,
                              kiva_env_str2number, format_env_str,
                              read_in_kiva_map, flip_tiles)
from env_search_PBS.node_hashing import maze_hashing
from mapf_py.warehouse_sim_LTF import run_follower  # Python Learn-to-Follow, see mapf_py/
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
        max_makespan (int): max timesteps of PIBT and LTF (M in the paper)
        ltf_device (str): device of the LTF policy network ("cpu" or "cuda")
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
    highLevelSolver: str = gin.REQUIRED,
    lowLevelSolver: bool = gin.REQUIRED,
    inadmissibleH: str = gin.REQUIRED,
    suboptimality: float = gin.REQUIRED,
    heuristics: str = gin.REQUIRED,
    prioritizingConflicts: bool = gin.REQUIRED,
    bypass: bool = gin.REQUIRED,
    disjointSplitting: int = gin.REQUIRED,
    rectangleReasoning: bool = gin.REQUIRED,
    corridorReasoning: bool = gin.REQUIRED,
    targetReasoning: bool = gin.REQUIRED,
    sipp: bool = gin.REQUIRED,
    restart:int = gin.REQUIRED,
    obj_type : str = gin.REQUIRED,
    use_warm_up : bool = gin.REQUIRED,
    hamming_only : bool = gin.REQUIRED,
    obj: str = gin.REQUIRED,

    # Python MAPF solvers in mapf_py/ (PIBT, LaCAM, LTF)
    max_makespan: int = 1000
    ltf_device: str = "cpu"

    # tile_usage_room: np.ndarray = gin.REQUIRED, 
    # obj_name : str = gin.REQUIRED,


def regularized_success_rate(success_rates, sums_of_cost, C):
    """r(x) in the paper: the sum over the N_e instances of the regularized
    success rate of Eq. (1),

        RSR = SR            if SR < 1
        RSR = SR * C - SoC  if SR = 1

    where C is a constant larger than any sum of costs, so a map whose agents
    all reach their goals always scores higher than one where some do not.
    """
    return float(sum(sr if sr < 1 else sr * C - soc
                     for sr, soc in zip(success_rates, sums_of_cost)))


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
        # agentNum: int,
        n_evals: int,
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
                # agentNum,
                add_movement=False,
                min_n_obstacles=min_n_obstacles,
                max_n_obstacles=max_n_obstacles,
                seed=repair_seed,
                # w_mode=w_mode,
                warm_envs_np=warm_up_sols,
                # limit_n_obstacle=limit_n_obstacle,
            )
            # If the repair is failed (which happens very rarely), we
            # return None and remember the unrepaired layout.
            if hamming_repaired_env is None:
                failed_unrepaired_env = format_env_str(
                eecbs_env_number2str(map_comp_unrepaired))
                logger.info(f"Hamming repair failed! The layout is:\n{failed_unrepaired_env}")
                return None, map_np_unrepaired, map_comp_unrepaired, None, None
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
                # agentNum,
                add_movement=True,
                warm_envs_np=warm_up_sols,
                min_n_obstacles=min_n_obstacles,
                max_n_obstacles=max_n_obstacles,
                seed=repair_seed,
                # w_mode=w_mode,
                # limit_n_obstacle=limit_n_obstacle,
            )
            if map_np_repaired is None:
                failed_unrepaired_env = format_env_str(eecbs_env_number2str(map_comp_unrepaired))
                logger.info(f"Repair failed! The layout is:\n{failed_unrepaired_env}")
                return None, map_np_unrepaired, map_comp_unrepaired, None, None
        

        # Create start and goal locations at this point with num of agents 
        #and constraint on the distance between start and goal locations 
        #and distance between start locations
        start_locations, goal_locations = [] ,[]
        num_agent = []
        for i in range(0, n_evals):
            start, goal, agent_number = calc_start_goal_locations(map_np_repaired, agent_ratio=0.1, parts = 5)
            # start, goal = calc_start_goal_locations(map_np_repaired, agentNum, parts = 5)
            num_agent.append(agent_number)
            start_locations.append(start)
            goal_locations.append(goal)
    
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
            "start":
            start_locations,
            "goal":
            goal_locations,
        })

        # else:
        #     NotImplementedError("Other warehouse types not supported yet.")

        return map_json, map_np_unrepaired, map_comp_unrepaired, map_np_repaired, num_agent[0]

    # evaluate with EECBS 
    def evaluate_EECBS(
        self,
        map_json: str,
        eval_logdir: pathlib.Path,
        sim_seed: int,
        agentNum: int,
        map_id: int,
        eval_id: int, 
        suboptimality_bound: float,
    ):
        output = str(eval_logdir / f"id_{map_id}-sim_{eval_id}-seed={sim_seed}")

        kwargs = {
            "map" : map_json,
            "output" : output,
            "agentNum" : agentNum,
            "seed" : sim_seed,
            "cutoffTime": self.config.cutoffTime,
            "screen":self.config.screen,
            "stats": self.config.stats,
            # "id": self.config.id,
            "highLevelSolver":self.config.highLevelSolver,
            "lowLevelSolver":self.config.lowLevelSolver,
            "inadmissibleH": self.config.inadmissibleH,
            "suboptimality": self.config.suboptimality,
            "heuristics": self.config.heuristics,
            "prioritizingConflicts":self.config.prioritizingConflicts,
            "bypass": self.config.bypass,
            "disjointSplitting": self.config.disjointSplitting,
            "rectangleReasoning": self.config.rectangleReasoning,
            "corridorReasoning": self.config.corridorReasoning,
            "targetReasoning": self.config.targetReasoning,
            "sipp": self.config.sipp,
            "restart":self.config.restart,
            "force_new_logdir": True,
            "eval_id": eval_id,
            
        }

        logger.info(f"EECBS Map evaluate: {map_id}")

        one_sim_result_jsonstr = warehouse_sim_EECBS.run(**kwargs)

        result_json = json.loads(one_sim_result_jsonstr)
        
        return result_json

    # evaluate with PBS
    def evaluate_PBS(
        self,
        map_json: str,
        eval_logdir: pathlib.Path,
        sim_seed: int,
        agentNum: int,
        map_id: int,
        eval_id: int, 
    ):
        output = str(eval_logdir / f"id_{map_id}-sim_{eval_id}-seed={sim_seed}")

        kwargs = {
            "map" : map_json,
            "output" : output,
            "agentNum" : agentNum,
            "seed" : sim_seed,
            "cutoffTime": self.config.cutoffTime,
            "screen":self.config.screen,
            "stats": self.config.stats,
            "sipp": self.config.sipp,
            "force_new_logdir": True,
            "eval_id": eval_id,
            
        }

        logger.info(f"PBS Map evaluate: {map_id}")

        one_sim_result_jsonstr = warehouse_sim_PBS.run(**kwargs)

        result_json = json.loads(one_sim_result_jsonstr)
        
        return result_json 

    def evaluate_PIBT(
        self,
        map_json: str,
        eval_logdir: pathlib.Path,
        sim_seed: int,
        agentNum: int,
        map_id: int,
        eval_id: int, 
    ):
        output = str(eval_logdir / f"id_{map_id}-sim_{eval_id}-seed={sim_seed}")

        kwargs = {
            "map" : map_json,
            "agentNum" : agentNum,
            "eval_id": eval_id,
            "seed": sim_seed,
            "max_makespan": self.config.max_makespan,
        }

        logger.info(f"PIBT Map evaluate: {map_id}")

        one_sim_result_jsonstr = warehouse_sim_PIBT.run(**kwargs)

        result_json = json.loads(one_sim_result_jsonstr)
        
        return result_json 
    
    def evaluate_LTF(
        self,
        map_json: str,
        eval_logdir: pathlib.Path,
        sim_seed: int,
        agentNum: int,
        map_id: int,
        eval_id: int, 
    ):
        output = str(eval_logdir / f"id_{map_id}-sim_{eval_id}-seed={sim_seed}")

        kwargs = {
            "map" : map_json,
            "agentNum" : agentNum,
            "eval_id": eval_id,
            "seed": sim_seed,
            "max_makespan": self.config.max_makespan,
            "device": self.config.ltf_device,
        }

        logger.info(f"ltf Map evaluate: {map_id}")

        one_sim_result_jsonstr = run_follower(**kwargs)

        result_json = json.loads(one_sim_result_jsonstr)
        
        return result_json 

    def evaluate_LaCAM(
        self,
        map_json: str,
        eval_logdir: pathlib.Path,
        sim_seed: int,
        agentNum: int,
        map_id: int,
        eval_id: int,
    ):
        kwargs = {
            "map" : map_json,
            "agentNum" : agentNum,
            "eval_id": eval_id,
            "seed": sim_seed,
            "cutoffTime": self.config.cutoffTime,
        }

        logger.info(f"LaCAM Map evaluate: {map_id}")

        one_sim_result_jsonstr = warehouse_sim_LaCAM.run(**kwargs)

        result_json = json.loads(one_sim_result_jsonstr)

        return result_json

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
            "eval_id": eval_id,
            
        }

        # For some of the parameters, we do not want to pass them in here
        # to the use the default value defined on the C++ side.
        # planning_window = self.config.planning_window
        # if planning_window is not None:
        #     kwargs["planning_window"] = planning_window

        # logger.info(f"Map evaluate: {map_id} \n{map_json}")
        logger.info(f"Map evaluate: {map_id}")

        one_sim_result_jsonstr = warehouse_sim_PBS.run(**kwargs)

        result_json = json.loads(one_sim_result_jsonstr)
        
        return result_json


    def process_eval_result(
        self,
        curr_result_json_EECBS: List[dict],
        curr_result_json_PBS: List[dict],
        curr_result_json_PIBT: List[dict],
        curr_result_json_LTF: List[dict],
        n_evals: int,
        map_np_unrepaired: np.ndarray,
        map_comp_unrepaired: np.ndarray,
        map_np_repaired: np.ndarray,
        # w_mode: bool,
        max_n_shelf: int,
        map_id: int,
        obj: str,
    ):
        """
        Process the evaluation result

        Args:
            results_json (List[dict]): result json of all simulations of 1 map.

        """

        # Defaults for the PIBT / LTF / PIBT_LTF / LaCAM branches, which do not
        # set them (the other branches overwrite them).
        start_locations, goal_locations = [], []
        avg_run_time_LaCAM = None
        avg_success_rate_LaCAM = None
        avg_sum_of_cost_LaCAM = None

        # Deal with failed layout.
        # For now, failure only happens during MILP repair, so it failure
        # happens, all simulation json results would contain
        # {"success": False}.
        if obj == "PIBT":
            if not curr_result_json_PIBT[0]["success"]:
                logger.info(
                f"Map ID {map_id} failed.")

                metadata = WarehouseMetadata(
                    map_int_unrepaired=map_comp_unrepaired,
                    map_int_raw=map_np_unrepaired,
                )
                result = WarehouseResult.from_raw(
                    warehouse_metadata=metadata,
                    opts={
                        "aggregation": self.config.aggregation_type,
                        "measure_names": self.config.measure_names,
                    },
                )
                result.failed = True
                return result

            keys_PIBT = curr_result_json_PIBT[0].keys()

            collected_results_PIBT = {key: [] for key in keys_PIBT}

            for result_json_PIBT in curr_result_json_PIBT:
                for key in keys_PIBT:
                    collected_results_PIBT[key].append(result_json_PIBT[key])

            success_rate_PIBT = np.array(collected_results_PIBT.get("success_rate"))
            avg_success_rate_PIBT = np.mean(success_rate_PIBT)

            sum_of_cost_PIBT = np.array(collected_results_PIBT.get("sum_of_cost"))
            avg_sum_of_cost_PIBT = np.mean(sum_of_cost_PIBT)

            make_span_PIBT = np.array(collected_results_PIBT.get("make_span_PIBT"))
            avg_make_span_PIBT = np.mean(make_span_PIBT)

            # Paper Eq. (1): minimise r_PIBT; the archive maximises, hence -r
            actual_obj = -regularized_success_rate(
                success_rate_PIBT, sum_of_cost_PIBT,
                self.config.max_makespan * 150 * 150)


            lower_runtime_alg = "PIBT"

            avg_success_rate_LTF=0.0
            avg_sum_of_cost_LTF = 0.0 
            avg_make_span_LTF = 0.0 

            solution_cost_PBS=[0]
            avg_run_time_PBS = 0.0
            tile_usage_PBS=np.zeros(100)
            tile_usage_mean_PBS=0.0
            tile_usage_std_PBS=0.0
            average_suboptimality_PBS= 0.0

            solution_cost_EECBS=[]
            avg_run_time_EECBS = 0.0
            tile_usage_EECBS=np.zeros(100)
            tile_usage_mean_EECBS=0.0
            tile_usage_std_EECBS=0.0
            average_suboptimality_EECBS= 0.0

            binary_obj = 0
            times_no_solution_EECBS = np.zeros(5) 
            times_no_solution_PBS = np.zeros(5) 

        if obj == "LTF":
            if not curr_result_json_LTF[0]["success"]:
                logger.info(
                f"Map ID {map_id} failed.")

                metadata = WarehouseMetadata(
                    map_int_unrepaired=map_comp_unrepaired,
                    map_int_raw=map_np_unrepaired,
                )
                result = WarehouseResult.from_raw(
                    warehouse_metadata=metadata,
                    opts={
                        "aggregation": self.config.aggregation_type,
                        "measure_names": self.config.measure_names,
                    },
                )
                result.failed = True
                return result

            keys_LTF = curr_result_json_LTF[0].keys()

            collected_results_LTF = {key: [] for key in keys_LTF}

            for result_json_LTF in curr_result_json_LTF:
                for key in keys_LTF:
                    collected_results_LTF[key].append(result_json_LTF[key])

            success_rate_LTF = np.array(collected_results_LTF.get("success_rate"))
            avg_success_rate_LTF = np.mean(success_rate_LTF)

            sum_of_cost_LTF = np.array(collected_results_LTF.get("sum_of_cost"))
            avg_sum_of_cost_LTF = np.mean(sum_of_cost_LTF)

            make_span_LTF = np.array(collected_results_LTF.get("make_span_LTF"))
            avg_make_span_LTF = np.mean(make_span_LTF)

            # Paper Eq. (1): minimise r_LTF; the archive maximises, hence -r
            actual_obj = -regularized_success_rate(
                success_rate_LTF, sum_of_cost_LTF,
                self.config.max_makespan * 150 * 150)


            lower_runtime_alg = "LTF"

            avg_success_rate_PIBT = 0.0
            avg_sum_of_cost_PIBT = 0.0 
            avg_make_span_PIBT = 0.0 


            solution_cost_PBS=[0]
            avg_run_time_PBS = 0.0
            tile_usage_PBS=np.zeros(100)
            tile_usage_mean_PBS=0.0
            tile_usage_std_PBS=0.0
            average_suboptimality_PBS= 0.0

            solution_cost_EECBS=[]
            avg_run_time_EECBS = 0.0
            tile_usage_EECBS=np.zeros(100)
            tile_usage_mean_EECBS=0.0
            tile_usage_std_EECBS=0.0
            average_suboptimality_EECBS= 0.0

            binary_obj = 0
            times_no_solution_EECBS = np.zeros(5) 
            times_no_solution_PBS = np.zeros(5) 
        
            
        if obj == "LaCAM":
            # warehouse_manager puts the LaCAM results in all four result lists
            curr_result_json_LaCAM = curr_result_json_PBS
            if not curr_result_json_LaCAM[0]["success"]:
                logger.info(
                f"Map ID {map_id} failed.")

                metadata = WarehouseMetadata(
                    map_int_unrepaired=map_comp_unrepaired,
                    map_int_raw=map_np_unrepaired,
                )
                result = WarehouseResult.from_raw(
                    warehouse_metadata=metadata,
                    opts={
                        "aggregation": self.config.aggregation_type,
                        "measure_names": self.config.measure_names,
                    },
                )
                result.failed = True
                return result

            # Same objective as PBS / EECBS: mean CPU runtime, where an
            # unsolved instance counts as cutoffTime.
            avg_run_time_LaCAM = np.mean(
                [r["runtime"] for r in curr_result_json_LaCAM])
            avg_success_rate_LaCAM = np.mean(
                [r["solution_found"] for r in curr_result_json_LaCAM])
            solved_costs = [r["solution_cost"] for r in curr_result_json_LaCAM
                            if r["solution_found"]]
            avg_sum_of_cost_LaCAM = np.mean(solved_costs) if solved_costs else None

            actual_obj = avg_run_time_LaCAM

            lower_runtime_alg = "LaCAM"

            avg_success_rate_PIBT = 0.0
            avg_sum_of_cost_PIBT = 0.0
            avg_make_span_PIBT = 0.0
            avg_success_rate_LTF = 0.0
            avg_sum_of_cost_LTF = 0.0
            avg_make_span_LTF = 0.0

            solution_cost_PBS=[0]
            avg_run_time_PBS = 0.0
            tile_usage_PBS=np.zeros(100)
            tile_usage_mean_PBS=0.0
            tile_usage_std_PBS=0.0
            average_suboptimality_PBS= 0.0

            solution_cost_EECBS=[]
            avg_run_time_EECBS = 0.0
            tile_usage_EECBS=np.zeros(100)
            tile_usage_mean_EECBS=0.0
            tile_usage_std_EECBS=0.0
            average_suboptimality_EECBS= 0.0

            binary_obj = 0
            times_no_solution_EECBS = np.zeros(5)
            times_no_solution_PBS = np.zeros(5)

        if obj == "EECBS":
            if not curr_result_json_EECBS[0]["success"]:
                logger.info(
                f"Map ID {map_id} failed.")

                metadata = WarehouseMetadata(
                    map_int_unrepaired=map_comp_unrepaired,
                    map_int_raw=map_np_unrepaired,
                )
                result = WarehouseResult.from_raw(
                    warehouse_metadata=metadata,
                    opts={
                        "aggregation": self.config.aggregation_type,
                        "measure_names": self.config.measure_names,
                    },
                )
                result.failed = True
                return result

            keys_EECBS = curr_result_json_EECBS[0].keys()

            collected_results_EECBS = {key: [] for key in keys_EECBS}

            for result_json_EECBS in curr_result_json_EECBS:
                for key in keys_EECBS:
                    collected_results_EECBS[key].append(result_json_EECBS[key])
            
            # Get start and goal locations
            start_locations = np.array(collected_results_EECBS.get("start_locations"))
            goal_locations = np.array(collected_results_EECBS.get("goal_locations"))

            # ----------------------------NEED specific for each algorithm (EECBS here)-------------------------------------
            # Post process result if necessary
            tile_usage_EECBS = np.array(collected_results_EECBS.get("tile_usage"))

            # print("Tile_usage shape: ", tile_usage.shape)
            tile_usage_EECBS = tile_usage_EECBS.reshape(n_evals, *map_np_repaired.shape)
            tile_usage_mean_EECBS=np.mean(collected_results_EECBS.get("tile_usage_mean"))
            tile_usage_std_EECBS=np.mean(collected_results_EECBS.get("tile_usage_std"))
            # tasks_finished_timestep = [np.array(x) for x in collected_results.get("tasks_finished_timestep")]

            # print("run_time in json: ",collected_results.get("runtime"))
            run_time_EECBS = np.array(collected_results_EECBS.get("runtime"))
            avg_run_time_EECBS = np.mean(run_time_EECBS)
            # print("shape of run_time: ", run_time.shape)

            # print("solution_cost in json: ", collected_results.get("solution_cost"))
            solution_cost_EECBS = np.array(collected_results_EECBS.get("solution_cost")) 

            Suboptimality_EECBS = np.array(collected_results_EECBS.get("suboptimality"))

            times_no_solution_EECBS = np.array(collected_results_EECBS.get("times_no_solution")) 
            times_no_solution_PBS = np.zeros(5)
            # print("suboptimality_list: ", Suboptimality)

            average_suboptimality_EECBS = np.mean(Suboptimality_EECBS)
            # Get objective based on type
            # objs = entropy
            objs_EECBS = avg_run_time_EECBS

            actual_obj = objs_EECBS

            #set for fit purpose
            lower_runtime_alg = "EECBS"
            solution_cost_PBS=[0]
            avg_run_time_PBS = 0.0
            tile_usage_PBS=np.zeros(100)
            tile_usage_mean_PBS=0.0
            tile_usage_std_PBS=0.0
            average_suboptimality_PBS= 0.0

            avg_success_rate_PIBT = 0.0
            avg_sum_of_cost_PIBT = 0.0
            avg_make_span_PIBT = 0.0

            avg_success_rate_LTF=0.0
            avg_sum_of_cost_LTF = 0.0 
            avg_make_span_LTF = 0.0 

            binary_obj = 0


            # --------------------------------------------------------------------------------------------------


        if obj == "PBS":
            if not curr_result_json_PBS[0]["success"]:
                logger.info(
                f"Map ID {map_id} failed.")

                metadata = WarehouseMetadata(
                    map_int_unrepaired=map_comp_unrepaired,
                    map_int_raw=map_np_unrepaired,
                )
                result = WarehouseResult.from_raw(
                    warehouse_metadata=metadata,
                    opts={
                        "aggregation": self.config.aggregation_type,
                        "measure_names": self.config.measure_names,
                    },
                )
                result.failed = True
                return result 
           
            keys_PBS = curr_result_json_PBS[0].keys()

            collected_results_PBS = {key: [] for key in keys_PBS} 

            for result_json_PBS in curr_result_json_PBS:
                for key in keys_PBS:
                    collected_results_PBS[key].append(result_json_PBS[key])
            # Get start and goal locations
            start_locations = np.array(collected_results_PBS.get("start_locations"))
            goal_locations = np.array(collected_results_PBS.get("goal_locations"))


            # ----------------------------NEED specific for each algorithm (PBS here)-------------------------------------
            # Post process result if necessary
            tile_usage_PBS = np.array(collected_results_PBS.get("tile_usage"))

            # print("Tile_usage shape: ", tile_usage.shape)
            tile_usage_PBS = tile_usage_PBS.reshape(n_evals, *map_np_repaired.shape)
            tile_usage_mean_PBS=np.mean(collected_results_PBS.get("tile_usage_mean"))
            tile_usage_std_PBS=np.mean(collected_results_PBS.get("tile_usage_std"))
            # tasks_finished_timestep = [np.array(x) for x in collected_results.get("tasks_finished_timestep")]

            # print("run_time in json: ",collected_results.get("runtime"))
            run_time_PBS = np.array(collected_results_PBS.get("runtime"))
            avg_run_time_PBS = np.mean(run_time_PBS)
            # print("shape of run_time: ", run_time.shape)

            # print("solution_cost in json: ", collected_results.get("solution_cost"))
            solution_cost_PBS = np.array(collected_results_PBS.get("solution_cost")) 

            Suboptimality_PBS = np.array(collected_results_PBS.get("suboptimality"))

            times_no_solution_PBS = np.array(collected_results_PBS.get("times_no_solution")) 
            times_no_solution_EECBS = np.zeros(5)
            # print("suboptimality_list: ", Suboptimality)

            average_suboptimality_PBS = np.mean(Suboptimality_PBS)

            # Get objective based on type
            # objs = entropy
            objs_PBS = avg_run_time_PBS

            actual_obj = objs_PBS


            # binary objective to see if PBS can find solution in the given map
            binary_obj = 0
            for solution_cost in solution_cost_PBS:
                if solution_cost >= 0:
                    binary_obj = 1
                else:
                    binary_obj = 0


            #set for fit purpose
            lower_runtime_alg = "PBS"
            solution_cost_EECBS=[]
            avg_run_time_EECBS = 0.0
            tile_usage_EECBS=np.zeros(100)
            tile_usage_mean_EECBS=0.0
            tile_usage_std_EECBS=0.0
            average_suboptimality_EECBS= 0.0

            avg_success_rate_LTF=0.0
            avg_sum_of_cost_LTF = 0.0 
            avg_make_span_LTF = 0.0 

            avg_success_rate_PIBT = 0.0
            avg_sum_of_cost_PIBT = 0.0
            avg_make_span_PIBT = 0.0

        if obj == "PBS_EECBS":
            if not curr_result_json_EECBS[0]["success"]:
                logger.info(
                f"Map ID {map_id} failed.")

                metadata = WarehouseMetadata(
                    map_int_unrepaired=map_comp_unrepaired,
                    map_int_raw=map_np_unrepaired,
                )
                result = WarehouseResult.from_raw(
                    warehouse_metadata=metadata,
                    opts={
                        "aggregation": self.config.aggregation_type,
                        "measure_names": self.config.measure_names,
                    },
                )
                result.failed = True
                return result
            # Collect the results
            keys_EECBS = curr_result_json_EECBS[0].keys()
            keys_PBS = curr_result_json_PBS[0].keys()

            collected_results_EECBS = {key: [] for key in keys_EECBS}
            collected_results_PBS = {key: [] for key in keys_PBS} 

            for result_json_EECBS in curr_result_json_EECBS:
                for key in keys_EECBS:
                    collected_results_EECBS[key].append(result_json_EECBS[key])

            for result_json_PBS in curr_result_json_PBS:
                for key in keys_PBS:
                    collected_results_PBS[key].append(result_json_PBS[key])
            
            # Get start and goal locations
            start_locations = np.array(collected_results_EECBS.get("start_locations"))
            goal_locations = np.array(collected_results_EECBS.get("goal_locations"))

            # ----------------------------NEED specific for each algorithm (EECBS here)-------------------------------------
            # Post process result if necessary
            tile_usage_EECBS = np.array(collected_results_EECBS.get("tile_usage"))

            # print("Tile_usage shape: ", tile_usage.shape)
            tile_usage_EECBS = tile_usage_EECBS.reshape(n_evals, *map_np_repaired.shape)
            tile_usage_mean_EECBS=np.mean(collected_results_EECBS.get("tile_usage_mean"))
            tile_usage_std_EECBS=np.mean(collected_results_EECBS.get("tile_usage_std"))
            # tasks_finished_timestep = [np.array(x) for x in collected_results.get("tasks_finished_timestep")]

            # print("run_time in json: ",collected_results.get("runtime"))
            run_time_EECBS = np.array(collected_results_EECBS.get("runtime"))
            avg_run_time_EECBS = np.mean(run_time_EECBS)
            # print("shape of run_time: ", run_time.shape)

            # print("solution_cost in json: ", collected_results.get("solution_cost"))
            solution_cost_EECBS = np.array(collected_results_EECBS.get("solution_cost")) 

            Suboptimality_EECBS = np.array(collected_results_EECBS.get("suboptimality"))

            # print("suboptimality_list: ", Suboptimality)

            average_suboptimality_EECBS = np.mean(Suboptimality_EECBS)
            # Get objective based on type
            # objs = entropy
            objs_EECBS = avg_run_time_EECBS

            # --------------------------------------------------------------------------------------------------

            # ----------------------------NEED specific for each algorithm (PBS here)-------------------------------------
            # Post process result if necessary
            tile_usage_PBS = np.array(collected_results_PBS.get("tile_usage"))

            # print("Tile_usage shape: ", tile_usage.shape)
            tile_usage_PBS = tile_usage_PBS.reshape(n_evals, *map_np_repaired.shape)
            tile_usage_mean_PBS=np.mean(collected_results_PBS.get("tile_usage_mean"))
            tile_usage_std_PBS=np.mean(collected_results_PBS.get("tile_usage_std"))
            # tasks_finished_timestep = [np.array(x) for x in collected_results.get("tasks_finished_timestep")]

            # print("run_time in json: ",collected_results.get("runtime"))
            run_time_PBS = np.array(collected_results_PBS.get("runtime"))
            avg_run_time_PBS = np.mean(run_time_PBS)
            # print("shape of run_time: ", run_time.shape)

            # print("solution_cost in json: ", collected_results.get("solution_cost"))
            solution_cost_PBS = np.array(collected_results_PBS.get("solution_cost")) 

            Suboptimality_PBS = np.array(collected_results_PBS.get("suboptimality"))

            # print("suboptimality_list: ", Suboptimality)

            average_suboptimality_PBS = np.mean(Suboptimality_PBS)

            # Get objective based on type
            # objs = entropy
            objs_PBS = avg_run_time_PBS

            # --------------------------------------------------------------------------------------------------

            # For two algorithms experiment, the objective is set to the absolute value of the difference between runtime of two algorithms

            objs = avg_run_time_PBS - avg_run_time_EECBS

            # Add an attribute to show which one got low runtime

            if objs >= 0:
                lower_runtime_alg = "EECBS"
            else:
                lower_runtime_alg = "PBS"
            
            abs_objs = abs(objs)

            actual_obj = abs_objs


            times_no_solution_EECBS = np.array(collected_results_EECBS.get("times_no_solution")) 
            times_no_solution_PBS = np.array(collected_results_PBS.get("times_no_solution")) 

            avg_success_rate_LTF=0.0
            avg_sum_of_cost_LTF = 0.0 
            avg_make_span_LTF = 0.0 

            avg_success_rate_PIBT = 0.0
            avg_sum_of_cost_PIBT = 0.0
            avg_make_span_PIBT = 0.0

            binary_obj = 0
        
        if obj == "PIBT_LTF":
            if not curr_result_json_PIBT[0]["success"]:
                logger.info(
                f"Map ID {map_id} failed.")

                metadata = WarehouseMetadata(
                    map_int_unrepaired=map_comp_unrepaired,
                    map_int_raw=map_np_unrepaired,
                )
                result = WarehouseResult.from_raw(
                    warehouse_metadata=metadata,
                    opts={
                        "aggregation": self.config.aggregation_type,
                        "measure_names": self.config.measure_names,
                    },
                )
                result.failed = True
                return result

            #PIBT
            keys_PIBT = curr_result_json_PIBT[0].keys()

            collected_results_PIBT = {key: [] for key in keys_PIBT}

            for result_json_PIBT in curr_result_json_PIBT:
                for key in keys_PIBT:
                    collected_results_PIBT[key].append(result_json_PIBT[key])

            success_rate_PIBT = np.array(collected_results_PIBT.get("success_rate"))
            avg_success_rate_PIBT = np.mean(success_rate_PIBT)

            sum_of_cost_PIBT = np.array(collected_results_PIBT.get("sum_of_cost"))
            avg_sum_of_cost_PIBT = np.mean(sum_of_cost_PIBT)

            make_span_PIBT = np.array(collected_results_PIBT.get("make_span_PIBT"))
            avg_make_span_PIBT = np.mean(make_span_PIBT)

            # Paper Eq. (1); both algorithms use the same C (same max_makespan)
            actual_obj_PIBT = -regularized_success_rate(
                success_rate_PIBT, sum_of_cost_PIBT,
                self.config.max_makespan * 150 * 150)

            #LTF
            keys_LTF = curr_result_json_LTF[0].keys()

            collected_results_LTF = {key: [] for key in keys_LTF}

            for result_json_LTF in curr_result_json_LTF:
                for key in keys_LTF:
                    collected_results_LTF[key].append(result_json_LTF[key])

            success_rate_LTF = np.array(collected_results_LTF.get("success_rate"))
            avg_success_rate_LTF = np.mean(success_rate_LTF)

            sum_of_cost_LTF = np.array(collected_results_LTF.get("sum_of_cost"))
            avg_sum_of_cost_LTF = np.mean(sum_of_cost_LTF)

            make_span_LTF = np.array(collected_results_LTF.get("make_span_LTF"))
            avg_make_span_LTF = np.mean(make_span_LTF)

            actual_obj_LTF = -regularized_success_rate(
                success_rate_LTF, sum_of_cost_LTF,
                self.config.max_makespan * 150 * 150)

            objs = actual_obj_LTF - actual_obj_PIBT

            if objs >= 0:
                lower_runtime_alg = "PIBT"
            else:
                lower_runtime_alg = "LTF"

            actual_obj = abs(objs)

            # Keep LTF's statistics in the metadata (needed for the paper's
            # Fig. 11 heatmap of SR_PIBT - SR_LTF); they used to be reset here.
            # avg_success_rate_LTF=0.0
            # avg_sum_of_cost_LTF = 0.0
            # avg_make_span_LTF = 0.0

            solution_cost_PBS=[0]
            avg_run_time_PBS = 0.0
            tile_usage_PBS=np.zeros(100)
            tile_usage_mean_PBS=0.0
            tile_usage_std_PBS=0.0
            average_suboptimality_PBS= 0.0

            solution_cost_EECBS=[]
            avg_run_time_EECBS = 0.0
            tile_usage_EECBS=np.zeros(100)
            tile_usage_mean_EECBS=0.0
            tile_usage_std_EECBS=0.0
            average_suboptimality_EECBS= 0.0

            binary_obj = 0
            times_no_solution_EECBS = np.zeros(5) 
            times_no_solution_PBS = np.zeros(5) 

        if obj == "NONE":
            if map_np_repaired is None:
                logger.info(
                f"Map ID {map_id} failed.")

                metadata = WarehouseMetadata(
                    map_int_unrepaired=map_comp_unrepaired,
                    map_int_raw=map_np_unrepaired,
                )
                result = WarehouseResult.from_raw(
                    warehouse_metadata=metadata,
                    opts={
                        "aggregation": self.config.aggregation_type,
                        "measure_names": self.config.measure_names,
                    },
                )
                result.failed = True
                return result
            
            start_locations = []
            goal_locations = []
            avg_success_rate_LTF=0.0
            avg_sum_of_cost_LTF = 0.0 
            avg_make_span_LTF = 0.0 

            avg_success_rate_PIBT = 0.0
            avg_sum_of_cost_PIBT = 0.0
            avg_make_span_PIBT = 0.0

            avg_success_rate_LTF=0.0
            avg_sum_of_cost_LTF = 0.0 
            avg_make_span_LTF = 0.0 

            solution_cost_PBS=[0]
            avg_run_time_PBS = 0.0
            tile_usage_PBS=np.zeros(100)
            tile_usage_mean_PBS=0.0
            tile_usage_std_PBS=0.0
            average_suboptimality_PBS= 0.0

            solution_cost_EECBS=[]
            avg_run_time_EECBS = 0.0
            tile_usage_EECBS=np.zeros(100)
            tile_usage_mean_EECBS=0.0
            tile_usage_std_EECBS=0.0
            average_suboptimality_EECBS= 0.0

            binary_obj = 0
            times_no_solution_EECBS = np.zeros(5) 
            times_no_solution_PBS = np.zeros(5) 
            lower_runtime_alg = "PBS"
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
        # all_ref_dist = [0.38101899721926447, 0.012272519620151006, 0.010799579348309859, 0.004841418459963748, 0.01211036479282215, 0.003367234322640998, 0.004672003981369541, 0.024975235770048104, 0.011021869417740169, 0.0046469457458492664, 0.0024169663261690527, 0.001133364996363774, 0.0026462898895044215, 0.0008978673370924531, 0.0007524933858972114, 0.0005625664347245856, 0.010336929236361965, 0.002643078454988285, 0.004629599476314925, 0.0008992921284622882, 0.0026532103047293347, 0.0005880317534933849, 0.0008999027533350747, 0.0007400773468172196, 0.0050757627166975825, 0.000900468146735803, 0.002719768415863062, 0.000479408373345478, 0.0008088065686097429, 0.00021663613642302734, 0.00029251193080075534, 0.0004271434073821607, 0.010986407943646494, 0.0026568062067579665, 0.002399326052066332, 0.0007613587544206301, 0.004546237873311555, 0.0008947011340483751, 0.0011418911288467557, 0.0005590609956400706, 0.0034628310388361266, 0.0011166067359661895, 0.0006646086356880159, 0.00036967682213214376, 0.0010975190547576046, 0.0007120564498771287, 0.00037243594192769747, 0.00031022005211156343, 0.0049704186182739, 0.0008081054807928398, 0.002434900604840152, 0.00029646968460585296, 0.0008963973142505598, 0.00021842277956932854, 0.00047251057385659363, 0.0004294275967211027, 0.025495782166230566, 0.0024090055870867994, 0.0016476694494021982, 0.00017135943289271093, 0.002554221228129837, 0.00030339009983076644, 0.00016794445675231246, 0.0003725942520799014, 0.012116538888758103, 0.003390754688111292, 0.0026656037280732977, 0.0008996765959747835, 0.002647284981889703, 0.0008858809969970147, 0.0008854512980124613, 0.000451025624628921, 0.004522129498704504, 0.020990795396436144, 0.0007573331534074451, 0.0005625438189885564, 0.0008760883832964018, 0.00044087115915184206, 0.00027756292928550106, 0.0028834158817698397, 0.0026612162752836467, 0.0005906099474007056, 0.000799895968614266, 0.00021858108972153245, 0.0007941289559268381, 0.00013856661565047379, 0.000294027185114707, 0.00015982540751785512, 0.0008965782401387927,
        #                 0.0007346269544341995, 0.00029579121252497906, 0.0004291562078887532, 0.00029373318054632834, 0.0001605943425428455, 0.00012203451261318045, 0.000415111835814664, 0.0026440057001654795, 0.0008807019934463442, 0.0007657009757382229, 0.00028335255770895807, 0.0008794807437007712, 0.0003543207373683651, 0.00047196779619189454, 0.0002180835435288916, 0.0010945563933377886, 0.0008870343995345003, 0.00023741999783379695, 0.0001595766344215347, 0.0009414704761566139, 0.0007404844300657439, 0.00015261098772456294, 0.00014772598874227108, 0.0008022932366333537, 0.00018655720750428568, 0.00024578782016457473, 7.126218522778569e-05, 0.000299635887649931, 5.3033901988307665e-05, 0.0001679896882243707, 0.00010290159993253725, 0.002578917611873646, 0.00214958047909666, 0.00010032340602521654, 0.00015222652021206775, 0.0008491078102136508, 0.00022163421408546486, 7.54687121292037e-05, 0.00033971097189354775, 0.010769884886903612, 0.0026521021336639076, 0.0034171020205852275, 0.0010420200385421218, 0.0026639075478711127, 0.0008890472000410928, 0.0010246285375357215, 0.0008665671584281385, 0.0024053870693221385, 0.0007517696823442794, 0.0006608318077711514, 0.00037241332619166834, 0.0007656557442661646, 0.0002883280196353665, 0.00023215053133900988, 0.0001582649217318452, 0.004561458263659159, 0.0007973630061790035, 0.02025234638361302, 0.002035959021286315, 0.0008082637909450438, 0.0001833910044602076, 0.0022436393252418074, 0.0021793654034470227, 0.002438383428188638, 0.0002946604257235226, 0.0013695863591880273, 0.00016878123898539024, 0.00025212022625273084, 7.295836542997037e-05, 0.00010109234105020693, 0.0001519551313797182, 0.0024033290373434877, 0.0007659497488345432, 0.0006759617351746387, 0.00023515842423088405, 0.0007533075523942602, 0.00028629260339274484, 0.00037184793279094016, 0.0001577221440671461, 0.0006761426610628718, 0.0002331908551963498, 1e-12, 1e-12, 0.00023016034656844653, 6.725919995062984e-05, 1e-12, 1e-12, 0.00276576882294631, 0.0002475518475748468, 0.0014169437104330235, 9.765474917377931e-05, 0.0002979397074477463, 7.058371314691182e-05, 0.0001698441785787593, 0.00014867584965549448, 0.0016566478966057625, 0.00010050433191344958, 0.0014355338454489677, 0.00017538503390589593, 0.00010168035018696428, 2.9287379157722142e-05, 0.00018418255522122711, 0.00030974512165495173, 0.004737928851894453, 0.0008951986802410159, 0.0010379492060568786, 0.000772033381826379, 0.0008867177792300925, 0.00035590383889040416, 0.0009256846924082818, 0.0010190198350004975, 0.0011448537902665717, 0.000557387431173915, 0.0003769590891335233, 0.000305764752113825,
        #                 0.00046235610837951465, 0.000212022526273085, 0.00015220390447603861, 0.00014878892833564014, 0.0008963520827785015, 0.00021616120596641564, 0.0021752493394897213, 0.0003703552942130176, 0.00029936449881758147, 5.473008219049235e-05, 0.0008032430975465771, 0.00030551597901750456, 0.0004748852261396522, 0.00042974421702551054, 0.00017059049786772055, 0.0003731370297446005, 0.00016491394812440915, 0.0001039419237898772, 7.490331872847547e-05, 0.0003363638429612366, 0.0007562475980780469, 0.00029015989425372593, 0.0002359725907279327, 6.793767203150371e-05, 0.0002762964480678698, 0.00010792229333100391, 0.0001503267983856209, 4.812628726998666e-05, 0.00037404165918576563, 0.00015885293086860257, 1e-12, 1e-12, 0.00014885677554372752, 4.937015275158876e-05, 1e-12, 1e-12, 0.00029124544958312413, 7.237035629321302e-05, 0.00010079833648182825, 3.064432331946989e-05, 0.00012504240550505462, 2.2412195404866907e-05, 7.320713852629079e-05, 4.785489843763711e-05, 0.0001701607988831671, 0.00014786168315844586, 0.00017995341258378, 0.00030999389475127215, 7.381776339907727e-05, 4.851075478248185e-05, 0.00013248298265863806, 0.0007319582975827623, 0.012181084199385237, 0.0026479182224985187, 0.0026616912057402585, 0.0008780559523309361, 0.0034057263053625755, 0.0009008752299843272, 0.0009033855766835606, 0.0004485152779296877, 0.00265671574381385, 0.0008841395853227717, 0.0007624216940139992, 0.0004658163159919714, 0.0008874867142550829, 0.0003560169175705498, 0.00029009204704563853, 0.00021618382170244475, 0.002656580049397675, 0.000792184002628333, 0.0008007101351113147, 0.0003011737576999118, 0.0005801840930912771, 0.00013454101463728882, 0.00021530180799730873, 0.00016093357858328245, 0.000809530272162675, 0.0002978040130315716, 0.0002485921714321867, 0.00016256191157737975, 0.00018397901359696497, 5.208404107508424e-05, 7.28226710137956e-05, 0.00010459778013472193, 0.004656421739245471, 0.000886989168062442, 0.0007492367199090169, 0.000276680915580365, 0.021738516861031208, 0.0004518850225980279, 0.000572110275328878, 0.002764818962033087, 0.0011309224968726282, 0.0010842210019724767, 0.0002327159247397381, 0.00014958047909665966, 0.0009400456847867789, 0.0007745889599976705, 0.000159802791781826, 0.0001457358039717077, 0.0009060316177989687, 0.00029778139729554244, 0.0002882149409552208, 0.0001246353222565303, 0.0007538955615310175, 0.0001600515648781464, 0.00042933713377698625, 0.0004171020205852274, 0.0023780220287268924, 0.0007378836204223941, 0.00010127326693843996, 7.408915223142682e-05, 0.002147522447118009, 0.00020415225013494808, 0.00014867584965549448,
        #                 0.0003374946297626931, 0.003382635638876835, 0.0008930954167903069, 0.0008851346777080535, 0.0003609245322888708, 0.0008976864112042201, 0.0005194608418530656, 0.00035520275107350113, 0.0008336612625057557, 0.0008893412046094715, 0.00044684171346353214, 0.0002888707973000656, 0.00021297238718630843, 0.0003540041170639573, 0.0006621209047248118, 0.00010875907556408168, 0.0015512359509739918, 0.0005853178651698894, 0.00014037587453280412, 0.00018250899075507158, 5.18126522427347e-05, 0.000134744556261551, 2.3723908094556392e-05, 5.316959640448244e-05, 2.9151684741547367e-05, 0.00021851324251344505, 0.00015930524558918516, 7.334283294246556e-05, 0.00010319560450091593, 5.1676957826559926e-05, 2.910645326948911e-05, 2.234434819677952e-05, 7.015401416235836e-05, 0.0008921455558770835, 0.0003538458069117534, 0.0002807743638016374, 0.0001076735202346835, 0.00045837573883838795, 0.0007079177701837981, 0.00021342470190689102, 0.0013732727241607752, 0.0007034172387140014, 0.0007452789661039193, 6.73044314226881e-05, 4.9935546152316984e-05, 0.0007282945483460434, 0.002003211435516136, 4.9890314680258723e-05, 0.0006311373463649049, 0.00021826446941712463, 5.188049945082209e-05, 7.203112025277608e-05, 2.229911672472126e-05, 0.0001650270268045548, 2.935522636580953e-05, 0.00010391930805384806, 7.083248624323223e-05, 0.00032568921555548773, 0.00021772169175242554, 3.211434616136328e-05, 4.740258371705453e-05, 0.00023920664098009815, 7.702919791521361e-05, 4.7945361381753626e-05, 0.0002454033526520795, 0.004799036570645159, 0.000884524052835267, 0.0010531695964044825, 0.0010513829532581812, 0.0008987041193255309, 0.00035551937137790894, 0.0007476536183869778, 0.0010625777425926001, 0.0007628513929985526, 0.00027475857801788903, 0.000231472059258136, 0.00014946740041651402, 0.0002882149409552208, 0.00010934708470083904, 6.658072786975597e-05, 4.961892584790918e-05, 0.0008959223837939481, 0.0002949544302919013, 0.0020546848507184343, 0.0006872696031892032, 0.00021975710799504715, 5.314698066845331e-05, 0.00037551168202765904, 0.0002940724165867653, 0.00029529366633233823, 0.00012187620246097654, 9.978062836051744e-05, 7.499378167259199e-05, 7.436054106377637e-05, 2.2231269516633872e-05, 3.007892991874166e-05, 4.767397254940408e-05, 0.001147386752701834, 0.0004716511758874867, 0.00037372503888135787, 0.00014935432173636835, 0.0005722459697450528, 0.00021199991053705587,
        #                 0.0003076418582042427, 0.00015075649737017436, 0.0003701065211166972, 0.00014840446082314495, 1e-12, 1e-12, 0.00015751860244288396, 5.097587000965692e-05, 1e-12, 1e-12, 0.00047891082715283714, 0.00016387362426706923, 0.00016973109989861366, 7.52425547689124e-05, 0.00042770880078288895, 0.00010305991008474116, 0.0003679354104579008, 0.0003393038886450234, 0.00016830630852877852, 7.327498573437817e-05, 0.00017882262578232353, 0.0001342017785968519, 0.00015247529330838816, 4.715381062073411e-05, 0.000310491440943913, 0.0007334057046886266, 0.02569952733211699, 0.00044564307945398827, 0.0008750028279670036, 0.0010752877862409706, 0.0004388357429092204, 0.0008330732533689983, 0.001012958817744691, 0.003466087704824321, 0.0005582468291430219, 0.0027924101599886243, 0.00015876246792448606, 0.00014840446082314495, 0.00021704321967155166, 0.0013947576733884478, 4.952846290379266e-05, 0.0008991112025740553, 0.0007419544529076374, 0.0001612954303597485, 0.002202727458765113, 0.0002864509135449488, 0.00016226790700900106, 2.9400457837867786e-05, 0.00033111699220247866, 0.00012764321514840445, 0.00042854558301596673, 0.00041669493733670305, 0.00014648212326066898, 0.00033794694448327565, 0.00010256236389210032, 7.234774055718388e-05, 4.7900129909695366e-05, 0.0002458104359006038, 0.0005609607174665174, 0.00021412578972379402, 0.0001567044359458353, 5.027478219275392e-05, 0.0029073885619607166, 0.001602076125567474, 0.00015116358061869868, 0.0008596241274671959, 0.000306646765818961, 0.00014532872072318338, 1e-12, 1e-12, 0.00014768075727021281, 0.0005820159677096366, 1e-12, 1e-12, 0.0004286360459600832, 0.00010369315069355678, 0.00014982925219298007, 4.80810557979284e-05, 0.00041434290078967364, 6.866137558443585e-05, 0.0003401406708781012, 0.00024614967194104076, 0.00036687247086453174, 0.0003351425932156637, 0.00031259470439462197, 0.00072734468743282, 0.00033758509270680957, 0.00024612705620501163, 0.0007280231595136939, 0.02379489789095183]
        # only one maze map
        all_ref_dist = [1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.012795078816840062, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.017154940408535563, 1e-12, 0.019850057671126874, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0014109957718573625, 0.003771626298577855, 1e-12, 1e-12, 0.003483275664206459, 0.00582083814017724, 1e-12, 1e-12, 0.01600538254617493, 1e-12, 0.016505190312418684, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.06092272203098847, 0.013986928105575164, 0.01362168396870473, 1e-12, 0.01490965013556363, 0.0035793925423302578, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0032718185323341023, 0.007366397540407921, 1e-12, 1e-12, 0.009988465975625145, 0.009815455595002307, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.015205690120184929, 0.00932718185413341, 1e-12, 1e-12, 0.012545174933718185, 0.0024067666292199156, 1e-12, 1e-12, 1e-12, 1e-12, 0.00046905036624413684, 0.002364475202845444, 1e-12, 1e-12, 0.00196078431472549, 0.006847366398539409, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.011707035756478662, 1e-12, 0.033533256440830835, 0.00960399846312995, 1e-12, 1e-12, 0.01091118800561361, 0.00837370242314533, 0.01655901576416801, 1e-12, 0.01043444829011957, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.020742022300115723, 1e-12, 0.010776624376240293, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.013771626298577856, 1e-12, 0.003971549405075356, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.002364475202845444, 0.007058823530411765, 1e-12, 1e-12, 0.00946559015863168, 0.01461361015094233, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.009926951173625914, 0.004621299501192234, 1e-12, 1e-12, 0.011607074203229912, 0.004083044983698963, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.004036908882199539, 0.012648981162091889, 1e-12, 1e-12, 0.008223760093272204, 0.010403690889119955, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.01350249903983122, 0.010346020762245675, 1e-12, 1e-12, 0.009154171473510574, 0.0022145328729723185, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0027450980402156863, 1e-12, 0.01140330642160746, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.008523644753018455, 1e-12, 0.025198000769935024, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.009511726260131104, 1e-12, 0.021484044599231446, 0.005567089581930412, 0.010049980777624376, 1e-12, 1e-12, 0.009438677432757017, 0.032364475202845445, 1e-12, 0.009777008843752788, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.00397923875532526, 0.0024259900048446753, 1e-12, 1e-12, 0.0027835447914652056, 0.000588235295117647, 1e-12, 1e-12, 1e-12, 1e-12, 0.002379853903345252, 0.011956939639600539, 1e-12, 1e-12, 0.0066320645915421, 0.015586312957555172, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.009530949635755863, 0.009300269128258747, 1e-12, 1e-12, 0.004605920800692427, 0.0037793156488277586, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0075816993474052295, 0.01562475970880469, 1e-12, 0.01174163783260323, 0.014452133795694348, 0.058569780854517875, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.02191849288835102, 1e-12, 0.014286812765321415, 1e-12, 1e-12, 0.008442906575394465, 0.003467896963706651, 1e-12, 1e-12, 0.00439446366882007, 0.001507112649981161, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.025801614764552477, 1e-12, 0.013733179547328335, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.008823529412764706, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12]
        maze_all_ref = [0.46792836088783835, 0.005544470076611944, 0.0014197103688072536, 0.004124759708804691, 0.005077213893092785, 1e-12, 0.0037242086387034475, 0.04668582596537268, 0.001934960913469563, 0.003609509164142381, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0035372292717932847, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.00476643598715917, 1e-12, 0.005490772780700116, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0018012943749587337, 1e-12, 1e-12, 1e-12, 0.003275919519134051, 1e-12, 1e-12, 1e-12, 0.00021792900266602588, 0.0006209150336797386, 1e-12, 1e-12, 0.0005724721271053441, 0.001001922338562476, 1e-12, 1e-12, 0.00451038062383737, 1e-12, 0.004682750225272716, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.048279764194259, 0.005688965783391388, 0.003708573626528643, 1e-12, 0.006109381008304883, 0.0005995130088175061, 1e-12, 1e-12, 0.0051014353463799825, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.003221517366115981, 0.03543963860153825, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.005413110343176086, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0005701012441116238, 0.0012386261704220173, 1e-12, 1e-12, 0.0016601307199542485, 0.0016452646428044343, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.00621760861307228, 0.003246828144021915, 1e-12, 1e-12, 0.0020453030895556837, 0.00042566961525092914, 1e-12, 1e-12, 0.0013169293871335384, 1e-12, 8.317313953646033e-05, 0.0004011277723699859, 1e-12, 1e-12, 0.0003408945287457388, 0.0011689093948228887, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0033382673340770215, 1e-12, 0.03348090478121274, 0.004663270537972959, 1e-12, 1e-12, 0.005225169807484686, 0.0033633218003079586, 0.004696719211560041, 1e-12, 0.0029300269137258745, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.005624951942560939, 1e-12, 0.0030572856603617838, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.003767845701371652, 1e-12, 0.0011214276570297322, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0037845059602464436, 1e-12, 0.00039023452618262204, 0.0011817249785727284, 1e-12, 1e-12, 0.0016029732164299629, 0.002429450212457132, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.005048763297168141, 0.0007734204803028323, 1e-12, 1e-12, 0.0019249006802259387, 0.0006569909019355376, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.005285531206946431, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.003537998206818275, 1e-12, 1e-12, 1e-12, 0.03670889401612239, 1e-12, 1e-12, 0.005077406126849033, 0.0006695501740103806, 0.0020532487514805845, 1e-12, 1e-12, 0.001371523773907856, 0.0017424067676282198, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.005640202487223248, 0.0017310008980908624, 1e-12, 1e-12, 0.003236319365347046, 0.00037184416350160193, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0004873125730876586, 1e-12, 0.0019229142647447135, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0014353453809820582, 1e-12, 0.004121171345354736, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.001568947841574138, 1e-12, 0.0036193130857110085, 0.0009785338982190182, 0.001662245291272972, 1e-12, 1e-12, 0.0016047033202361912, 0.005475265924362809, 1e-12, 0.0016361015004232988, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0006600666420354992, 0.00041131616145110856, 1e-12, 1e-12, 0.0004698193012691273, 9.489939866756375e-05, 1e-12, 1e-12, 0.004033769064180829, 1e-12, 0.0004200307584009996, 0.001950083302294374, 1e-12, 1e-12, 0.0011041906968861975, 0.0025548506994493144, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.004697424068666282, 0.0015925925935925925, 1e-12, 1e-12, 0.000804562348814943, 0.0006263616567734205, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.04784557221681443, 1e-12, 0.001224080482865949, 0.002599513008817506, 1e-12, 0.001925925926925926, 0.002420863771344739, 0.009624247085454698, 1e-12, 0.005063821608074203, 1e-12, 1e-12, 1e-12, 0.0036787773943102653, 1e-12, 0.002401127772369986, 1e-12, 1e-12, 0.0033514673853393567, 0.0005974625154175317, 1e-12, 1e-12, 0.0007276047683971549, 0.00024259900138446752, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.005380750994207741, 0.004267781623452903, 1e-12, 0.002282647700602717, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0014963475596312956, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12]
        room_all_ref = [0.6932400999625532, 1e-12, 1e-12, 0.0031203383324109957, 1e-12, 0.0030556516734336794, 0.0030149942339873126, 0.032371683968704726, 1e-12, 0.0030582468291430217, 1e-12, 4.507881684006152e-05, 1e-12, 5.219146582122261e-05, 1e-12, 0.00038533256539830836, 6.728191468665898e-07, 1e-12, 0.0030695886207616303, 4.62322193775471e-05, 1e-12, 1e-12, 4.978854386812765e-05, 0.0003806228383702422, 0.003133025760323337, 5.1903115186851215e-05, 4.527105059630911e-05, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.003133025760323337, 5.622837470242215e-05, 5.738177723990773e-05, 0.00037620146197654746, 0.0030715109583241062, 5.0653595771241834e-05, 1e-12, 1e-12, 5.594002406805075e-05, 1e-12, 1e-12, 1e-12, 0.0030308535188777394, 1e-12, 4.911572572126105e-05, 1e-12, 5.690119284928874e-05, 1e-12, 1.922347562475971e-07, 1e-12, 0.032349865437370624, 0.00036553248850480584, 0.000367358709189158, 9.611697812379855e-07, 0.0003639946184548251, 1e-12, 5.767022687427912e-07, 0.00015926566805113417, 1e-12, 0.0030754517503271818, 1e-12, 5.815071226489812e-05, 1e-12, 5.7766244752402924e-05, 1e-12, 0.0004129181094198385, 0.003116878124798539, 0.03230296040084621, 1e-12, 0.0003665897741641676, 1e-12, 0.00038360246159207997, 1e-12, 0.0032280853527877737, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 4.113802483698578e-05, 0.00037053056616724334, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 4.2195310496347565e-05, 0.000393598616916955, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.00032420223091157247, 0.003279123415071511, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1.0092273202998847e-05, 1e-12, 1e-12, 0.003088619762630142, 5.171088143060362e-05, 1e-12, 1e-12, 5.728566036178393e-05, 0.0004137831613229527, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0031299500202233758, 1e-12, 0.032285563245905806, 0.0003350634381395617, 1e-12, 1e-12, 0.0003651480209923106, 0.0032196270675128795, 3.6043830296424456e-05, 1e-12, 0.00034467512595194155, 5.767022687427912e-07, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 4.3156479277585546e-05, 1e-12, 0.00037053056616724334, 1e-12, 1e-12, 1e-12, 1.249520415609381e-06, 1e-12, 0.0003211264908116109, 1e-12, 0.0032618223770092274, 8.650520031141869e-06, 1e-12, 1e-12, 7.785468128027681e-06, 8.650529031141868e-07, 0.0031968473673975395, 4.767397254940408e-05, 4.306036239946175e-05, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 5.4498270896193775e-05, 0.0003715878518266051, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 4.988466074625145e-05, 1e-12, 0.0003476547491737793, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1.4417541718569782e-06, 0.00016032295371049595, 1e-12, 1e-12, 1e-12, 9.900039446751248e-06, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1.0572866593617839e-06, 1e-12, 8.266052518646674e-06, 3.8446851249519415e-07, 1e-12, 1e-12, 1e-12, 1.922347562475971e-07, 1e-12, 1e-12, 1e-12, 1e-12, 0.0031515763178012303, 6.468665997731642e-05, 6.478277685544021e-05, 0.00040916955117301036, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0031955017311038063, 1e-12, 1e-12, 1e-12, 0.033650038447751246, 0.00042156862845098037, 0.0004211841609384852, 0.0032277970021534026, 5.007689450249904e-05, 1e-12, 1e-12, 1e-12, 0.0004277201086509035, 1e-12, 1e-12, 1e-12, 4.62322193775471e-05, 1e-12, 1e-12, 1e-12, 0.0004323337188008458, 1e-12, 1e-12, 1e-12, 0.000315840062514802, 1e-12, 1e-12, 1e-12, 0.003277104960630911, 1e-12, 1e-12, 1.1053441984236832e-05, 0.0031234140725109574, 5.017301138062284e-05, 1e-12, 1e-12, 5.978469919300269e-05, 1e-12, 1e-12, 1e-12, 5.680507597116494e-05, 0.0003951364869669358, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 5.824682914302192e-05, 1e-12, 1e-12, 1e-12, 0.00043723567958515953, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.003083621684967705, 1e-12, 5.171088143060362e-05, 1e-12, 5.459438777431757e-05, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 5.1038063283737024e-05, 1e-12, 0.00037898885144213763, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 5.863129665551711e-05, 1e-12, 1e-12, 1e-12, 0.00043810073148827373, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 8.650529031141868e-07, 1e-12, 1e-12, 1e-12, 0.00015138408404498268, 9.80392256862745e-06, 5.767022687427912e-07, 1e-12, 8.650520031141869e-06, 1e-12, 1e-12, 1e-12, 5.767022687427912e-07, 1e-12, 0.03372625913210342, 0.00037783544890465203, 0.0003737985400234525, 1e-12, 0.000367358709189158, 1e-12, 1e-12, 1e-12, 0.0003865820848139177, 0.0033591887745486352, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.00039004229242637446, 1e-12, 0.003340157632680123, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1.1630143252979622e-05, 1e-12, 1e-12, 1e-12, 1e-12, 0.00038860053925451746, 1e-12, 1e-12, 1e-12, 0.0033498654373706267, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1.0188390081122644e-05, 1e-12, 0.00014686659077316417, 1.057285759361784e-05, 8.650529031141868e-07, 1.922347562475971e-07, 1.1534026374855824e-05, 1e-12, 9.611787812379855e-08, 0.008346789697270665]

        maze_kl_div = scipy.stats.entropy(norm_pattern_dist, np.array(maze_ref_dist), base = len(maze_ref_dist))
        room_kl_div = scipy.stats.entropy(norm_pattern_dist, np.array(room_ref_dist), base = len(room_ref_dist)) 

        all_kl_div = scipy.stats.entropy(norm_pattern_dist, np.array(all_ref_dist), base = len(all_ref_dist))  
        maze_all_kl_div = scipy.stats.entropy(norm_pattern_dist, np.array(maze_all_ref), base = len(maze_all_ref))
        room_all_kl_div = scipy.stats.entropy(norm_pattern_dist, np.array(room_all_ref), base = len(room_all_ref))


        # cal lambda2
        lambda2 = cal_lambda2(map_np_repaired)

        # calcualte KL of node hashing distance
        # compute the KL divergience between the map with all maze maps
        # import hash lists precomputed
        node_hashing_map = node_hashing(map_np_repaired)
        base_len = len(node_hashing_map)
        logger.info(f"Hashing Success {map_id}") 
        result_entropy_list = []

        maze_hash = maze_hashing()
        #only one maze node hashing
        maze_list = [2e-12, 2e-12, 0.04662358569219844, 0.40948417270229215, 0.04116852698355961, 0.0027237146299497372, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 0.04662358569219844, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 0.07334574391178936, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 0.0036049164213452405, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 0.0016365176145916487, 0.27144448428914203, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 0.026787008569875422, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 0.005477947068856895, 0.010868155429211207, 2e-12, 8.773870850258256e-05, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 0.016231660704977773, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 0.04116852698355961, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 0.0027237146299497372, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12, 2e-12]
        abs_diff_hash = scipy.stats.entropy(node_hashing_map,maze_list,base = base_len)  


        # all maze maps node hashing
        # for i in range(len(maze_hash)):
        #     entropy_result = scipy.stats.entropy(node_hashing_map,maze_hash[i],base = base_len) 
        #     result_entropy_list.append(entropy_result)
        
        # abs_diff_hash = abs(sum(result_entropy_list)/len(maze_hash))
        


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

        # if self.config.obj_type == "throughput":
        #     objs = np.array(collected_results.get("throughput"))
        # elif self.config.obj_type == "throughput_plus_n_shelf":
        #     objs = np.array(collected_results.get("throughput")) - \
        #         (max_n_shelf - n_shelf)**2 * 0.5
        # else:
        #     return ValueError(
        #         f"Object type {self.config.obj_type} not supported")
        
        # logger.info("actual_obj %f", actual_obj)

        hamming_dist, sim_score = cal_similarity_score(map_comp_unrepaired, map_np_repaired)
        logger.info(f"Map ID {map_id}: Hamming dist: {hamming_dist}")
        logger.info(f"Map ID {map_id}: similarity score: {sim_score}")



        # Create WarehouseResult object using the mean of n_eval simulations
        # For tile_usage, num_wait, and finished_task_len, the mean is not taken
        metadata = WarehouseMetadata(
            objs= actual_obj,
            # objs= sim_score,
            # objs = objs,  # for extracting maps where one algorithm works well on
            # throughput=collected_results.get("throughput"),

            # attributes specific for two algorithm experiments
            lower_runtime_alg = lower_runtime_alg,
            solution_cost_EECBS=solution_cost_EECBS,
            solution_cost_PBS=solution_cost_PBS,

            avg_run_time_EECBS=avg_run_time_EECBS,
            avg_run_time_PBS = avg_run_time_PBS,

            tile_usage_EECBS=tile_usage_EECBS,
            tile_usage_mean_EECBS=tile_usage_mean_EECBS,
            tile_usage_std_EECBS=tile_usage_std_EECBS,
            average_suboptimality_EECBS = average_suboptimality_EECBS,

            tile_usage_PBS=tile_usage_PBS,
            tile_usage_mean_PBS=tile_usage_mean_PBS,
            tile_usage_std_PBS=tile_usage_std_PBS,
            average_suboptimality_PBS= average_suboptimality_PBS, 

            # attributes shared by maps 
            map_int_unrepaired=map_comp_unrepaired,
            map_int=map_np_repaired,
            map_int_raw=map_np_unrepaired,
            map_str=eecbs_env_number2str(map_np_repaired),
            n_obstacles=n_obstacles,
            # n_endpoint=n_endpoint,
            start_locations = start_locations,
            goal_locations = goal_locations,
            abs_diff_hash = abs_diff_hash,

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
            std_BC = std_BC,
            avg_degrees = avg_degrees,
            layout_entropy=entropy,
            connectivity_graph = connectivity_graph,
            maze_kl_div = maze_kl_div,
            room_kl_div = room_kl_div,
            maze_all_kl_div = maze_all_kl_div, 
            room_all_kl_div = room_all_kl_div, 
            all_kl_div = maze_all_kl_div, 
            lambda2 = lambda2, 

            avg_success_rate_PIBT = avg_success_rate_PIBT,
            avg_sum_of_cost_PIBT = avg_sum_of_cost_PIBT,
            avg_make_span_PIBT = avg_make_span_PIBT,

            avg_success_rate_LTF=avg_success_rate_LTF,
            avg_sum_of_cost_LTF = avg_sum_of_cost_LTF,
            avg_make_span_LTF = avg_make_span_LTF,

            #binary_obj for PBS
            binary_obj = binary_obj,

            times_no_solution_EECBS = times_no_solution_EECBS,
            times_no_solution_PBS = times_no_solution_PBS, 
            avg_run_time_LaCAM = avg_run_time_LaCAM,
            avg_success_rate_LaCAM = avg_success_rate_LaCAM,
            avg_sum_of_cost_LaCAM = avg_sum_of_cost_LaCAM,


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

def cal_similarity_score(map_np_unrepaired, map_np_repaired):
    # Calculate hamming distance
    assert map_np_unrepaired.shape == map_np_repaired.shape
    hamming_dist = (map_np_unrepaired != map_np_repaired).sum()
    n_tiles_storage = np.prod(map_np_unrepaired.shape)
    sim_score = 1 - hamming_dist/n_tiles_storage
    return hamming_dist, sim_score

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

def split_list(lst, parts=5):
    length = len(lst)
    part_size = length // parts
    remainder = length % parts
    result = []
    start = 0
    for i in range(parts):
        end = start + part_size + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end
    return result

# def calc_start_goal_locations(repaired_env, num_of_agents, parts):
def calc_start_goal_locations(repaired_env, agent_ratio, parts):
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
    # print("distance_graph: " ,distance_graph)

    # The matrix of distances between graph nodes. dist_matrix[i,j] gives the shortest distance from point i to point j along the graph.
    dist_matrix, predecessors= scipy.sparse.csgraph.floyd_warshall(csgraph=distance_graph, directed=False, return_predecessors=True)
    
    # take inf to 0 and get indices of all index pairs with distance greater than half of the longest paths 
    dist_matrix[np.isinf(dist_matrix)] = 0
    less_restrict_indices = np.where(dist_matrix > (np.max(dist_matrix)/8)) 
    indices = np.where(dist_matrix > (np.max(dist_matrix)/4))

    total_emtpy = total_grid - num_of_obstacle
    num_of_agents = int(total_emtpy * agent_ratio)
    if num_of_agents % 5 !=0:
        remainder = num_of_agents % 5
        num_of_agents += 5 - remainder
    # print(num_of_agents) 
    assert(len(indices[0])>num_of_agents)

    # try to split start and goals into "parts" lists to evenly distribute the start and goal locations
    # parts = 5
    result_start = split_list(indices[0])
    result_goal = split_list(indices[1])

    start_locations = []
    goal_locations = []

    interval = np.arange(num_of_agents//parts,num_of_agents+(num_of_agents//parts),num_of_agents//parts)
    # print(interval)

    # select num_of_agents start and goal locations 
    j = 0
    k = 0
    count = 1
    # print("Agents: ",num_of_agents)
    while j < num_of_agents:
        if j <= interval[k]:
            temp_start = result_start[k]
            temp_goal = result_goal[k]
        else:
            k += 1
            # print("K: " ,k)
            # print("START: " ,len(result_start))
            # print("GOAL: ", len(result_goal))
            temp_start = result_start[k]
            temp_goal = result_goal[k]
        
        if count % 3 == 0:
            if k < parts-1:
                temp_start = result_start[k+1]
                temp_goal = result_goal[k+1]
            else:
                temp_start = result_start[0]
                temp_goal = result_goal[0] 
        
        if count > 15:
            temp_start = less_restrict_indices[0]
            temp_goal = less_restrict_indices[1] 

        random_index = np.random.randint(0,len(temp_start)-1,size = 1)
        index_0 = temp_start[random_index]
        index_1 = temp_goal[random_index]
        if (index_0 not in start_locations) and (index_1 not in goal_locations):
            start_locations.append(int(index_0))
            goal_locations.append(int(index_1))
            j += 1
            # print(f"{j}th index made")
        else:
            count += 1
        # print("J: ", j)
    
    return start_locations,goal_locations,num_of_agents
    
# function to check duplicates in generated start and goal locations
def has_duplicates(lst):
    seen = set()
    for num in lst:
        if num in seen:
            return True
        seen.add(num)
    return False

# switch position of "." and "@" to do node hashing
def env_str2number(env_str):
    env_np = []
    obj_types = "@."
    for row_str in env_str:
        # print(row_str)
        row_np = [obj_types.index(tile) for tile in row_str]
        env_np.append(row_np)
    return np.array(env_np, dtype=int)

# doing the node hashing 
def node_connection(map_int,map_hash):
    n_row, n_col = map_int.shape
    for i in range(n_row):
        for j in range(n_col):
            if map_int[i][j] == 0:
                continue
            else:
                if i == 0 and j == 0:
                    number = str(map_int[i,j+1] * 10 + map_int[i+1,j])
                    result_number = int(number.replace("0", ""))
                    if (i,j) not in map_hash:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                    else:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                elif i == n_row-1 and j == 0:
                    number = str(map_int[i,j+1] * 10 + map_int[i-1,j])
                    result_number = int(number.replace("0", ""))
                    if (i,j) not in map_hash:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                    else:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                elif i == 0 and j == n_col - 1:
                    number = str(map_int[i,j-1] * 10 + map_int[i+1,j])
                    result_number = int(number.replace("0", ""))
                    if (i,j) not in map_hash:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                    else:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                elif i == n_row-1 and j == n_col - 1:
                    number = str(map_int[i,j-1] * 10 + map_int[i-1,j])
                    result_number = int(number.replace("0", ""))
                    if (i,j) not in map_hash:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                    else:
                        map_hash[(i,j)] = (map_int[i][j],result_number)

                #upper and down edge 
                elif i == 0:
                    number = str(map_int[i,j-1] * 100 + map_int[i,j+1] * 10 + map_int[i+1,j])
                    result_number = int(number.replace("0", ""))
                    if (i,j) not in map_hash:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                    else:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
            
                elif i == n_row -1:
                    number = str(map_int[i,j-1] * 100 + map_int[i,j+1] * 10 + map_int[i-1,j])
                    result_number = int(number.replace("0", ""))
                    if (i,j) not in map_hash:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                    else:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                #left and right edge  
                elif j == 0:
                    number = str(map_int[i,j+1] * 100 + map_int[i+1,j] * 10 + map_int[i-1,j])
                    result_number = int(number.replace("0", ""))
                    if (i,j) not in map_hash:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                    else:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                    
                elif j == n_col -1:
                    number = str(map_int[i,j-1] * 100 + map_int[i+1,j] * 10 + map_int[i-1,j])
                    result_number = int(number.replace("0", ""))
                    if (i,j) not in map_hash:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                    else:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                    
                #normal cases
                else:
                    number = str(map_int[i,j-1] * 1000 + map_int[i,j+1] * 100 + map_int[i+1,j] * 10 + map_int[i-1,j])
                    result_number = int(number.replace("0", ""))
                    if (i,j) not in map_hash:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                    else:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
    
    return map_hash

# add to the graphlet vector 
def add_to_vector(map,glv):
    n_row, n_col = map.shape
    for i in range(n_row):
        for j in range(n_col):
            if map[i][j] not in glv:
                glv[map[i][j]] = 1
            else:
                glv[map[i][j]] += 1 
    
    return glv

# normalize the graphlet vector
def normalize(glv,sum_value):
    for key in glv.keys():
        glv[key] /= sum_value     
    return glv
# sort an integer
def sort_int(num):
    # Convert the integer to a string
    num_str = str(num)
    
    # Convert the string to a list of characters
    num_list = list(num_str)
    
    # Sort the list of characters
    num_list.sort()
    
    # Join the sorted list back into a string
    sorted_num_str = ''.join(num_list)
    
    # Convert the string back to an integer
    sorted_num = int(sorted_num_str)
    
    return sorted_num
    
def node_hashing(map_np_repaired):
    # open the json file and convert from string to int
    # with open(map_path, "r") as map_json:
    #     map_data = json.load(map_json)

    # map_string = json_file["layout"]
    map_str = eecbs_env_number2str(map_np_repaired)
    map_int = env_str2number(map_str)

    # directionary for collecting the index and value after connection check
    map_hash = dict()

    n_row, n_col = map_int.shape

    map_hash = node_connection(map_int,map_hash)
    
    # create graphlet vector and add initial map information 
    glv = dict()

    glv = add_to_vector(map_int,glv) 

    # first hash table
    hash_table_a = dict()

    hash_table_a[(1,1)] = 2
    hash_table_a[(1,11)] = 3
    hash_table_a[(1,111)] = 4
    hash_table_a[(1,1111)] = 5

    # create the map after one iteration of hashing

    map_one_iter = copy.deepcopy(map_int)

    for key in map_hash:
        x_axis = key[0]
        y_axis = key[1]

        map_one_iter[x_axis][y_axis] = hash_table_a[map_hash[key]]

    # add first iteration hash information to the graphlet vector

    glv = add_to_vector(map_one_iter,glv) 

    # do the second node connection 

    map_hash = node_connection(map_one_iter,map_hash)

    # create the second hash table

    last_hash = [2,3,4,5]
    new_hash = []

    # Collect all combinations from length 1 to 4
    all_combinations = []
    for length in range(1, 5):  # Lengths from 1 to 4
        combinations = itertools.combinations_with_replacement(last_hash, length)
        all_combinations.extend(combinations)
    

    all_combinations = list(all_combinations)

    # Normalize combinations
    normalized_combinations = {tuple(sorted(combination)) for combination in all_combinations}

    # Convert back to a list of tuples
    unique_combinations = list(normalized_combinations)

    # Print all combinations
    for combination in unique_combinations:
        # Join the tuple into a string for a better readable format
        new_hash_value = ''.join(map(str, combination)) 
        new_hash.append(int(new_hash_value))


    num = 6 
    new_hash_table = dict()
    for i in last_hash:
        for j in new_hash:
            new_hash_table[(i,j)] = num
            num += 1
    

    # create the map after the second iteration of hashing

    map_second_iter = copy.deepcopy(map_one_iter)

    for key in map_hash:
        x_axis = key[0]
        y_axis = key[1]

        value_0 = map_hash[key][0]
        value_1 = map_hash[key][1]
        value_1 = sort_int(value_1)
        value = (value_0,value_1)

        map_second_iter[x_axis][y_axis] = new_hash_table[value]

    # add second iteration hash information to the graphlet vector

    glv = add_to_vector(map_second_iter,glv) 
    glv[0] /= 3

    # delete the first two positions (since it only represent number of obstacles and empty space)
    glv[0] = 0 
    glv[1] = 0

    # do normalization to the graphlet vector
    sum_value = 0

    for value in glv.values():
        sum_value += value

    glv_normalized = copy.deepcopy(glv)

    glv_normalized = normalize(glv_normalized,sum_value)
    
    final_node_array = np.zeros(281)

    for key in glv_normalized.keys():
        final_node_array[key] = glv_normalized[key]
    epsilon=1e-12

    for i in range(0,len(final_node_array)):
        final_node_array[i] += epsilon
    # for key in glv.keys():
    #     final_node_array[key] = glv[key]

    return final_node_array.tolist()

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

def cal_lambda2(map_np_repaired):
    dim_x, dim_y = map_np_repaired.shape

    G = nx.Graph()
    movements = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    for i in range(dim_x):
        for j in range(dim_y):
            if map_np_repaired[i][j] != 1:
                G.add_node((i, j))
    
    for i in range(dim_x):
        for j in range(dim_y):
            if map_np_repaired[i][j] != 1:
                # check neighbors
                for mi, mj in movements:
                    if -1 < mi + i < dim_x and -1 < mj + j < dim_y and map_np_repaired[mi + i, mj + j] != 1:
                        G.add_edge((i, j), (mi + i, mj + j))


    eigen = nx.normalized_laplacian_spectrum(G)
    # print('lambda2 is: %f' % eigen[1])

    return (eigen[1])

# def single_simulation(seed, agent_num, kwargs, results_dir, eval_id, alg, obs, std_bc):
def single_simulation(seed, agent_num, kwargs, results_dir, eval_id, alg):
    kwargs["seed"] = int(seed)
    output_dir = os.path.join(results_dir,
        f"sim-agent_num={agent_num}-seed={seed}")
    # kwargs["output"] = output_dir
    kwargs["agentNum"] = agent_num

    # breakpoint()
    # for i in kwargs:
    #     print(i)
    print(kwargs["stats"])
    # print(kwargs)
    #for eval_id
    kwargs["eval_id"] = eval_id
    if alg == "PBS":
        result_jsonstr = warehouse_sim_PBS.run(**kwargs)
    elif alg == "EECBS":
        result_jsonstr = warehouse_sim_EECBS.run(**kwargs)
    
    result_json = json.loads(result_jsonstr)

    solution_cost = result_json["runtime"]
    real_solution_cost = result_json["solution_cost"]

    # print("runtime: ",solution_cost)
    # print("solution_cost: ", real_solution_cost)

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

    # return obs, std_bc, real_solution_cost,solution_cost
    return real_solution_cost,solution_cost

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
    alg = "PBS",
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

    if mode == "maps":
        file_list = [os.path.join(map_filepath, f) for f in os.listdir(map_filepath) if os.path.isfile(os.path.join(map_filepath, f))]
        n_evals = len(file_list)
        # print(n_evals)
        # print(file_list)
        kwargs_list = []
        agent_list = []
        obs_list = []
        std_bc_list = []

        time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
        base_log_dir = time_str + "_"
        log_dir = os.path.join(LOG_DIR, base_log_dir)
        results_dir = os.path.join(log_dir, "results")
        os.mkdir(log_dir)
        os.mkdir(results_dir)
        
        # Read in map
        for file in file_list:
            with open(file, "r") as f:
                raw_env_json = json.load(f)
            print(raw_env_json["name"])
            map_read = eecbs_env_str2number(raw_env_json['layout']) 
            # print(raw_env_json["num_of_agent"])
            agent_list.append(raw_env_json["num_of_agent"])
            obs_list.append(raw_env_json["num_of_obstacles"])
            std_bc_list.append(raw_env_json["std_bc"])
            #used for generating instances
            start_location, goal_location = [], []
            start_location.append(raw_env_json['start'])
            goal_location.append(raw_env_json['goal'])
            # for i in range(0, n_evals):
            #     start_, goal_ = calc_start_goal_locations(map_read, agent_num, parts = 5)
            #     start_location.append(start_)
            #     goal_location.append(goal_)
            #     # print(i)
            raw_env_json['start'] = start_location
            raw_env_json['goal'] = goal_location 
            
            # print(len(raw_env_json['start']))

            # Create log dir
            # map_name = raw_env_json["name"]
            # time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
            # base_log_dir = time_str + "_" + map_name
            # log_dir = os.path.join(LOG_DIR, base_log_dir)
            # results_dir = os.path.join(log_dir, "results")
            # os.mkdir(log_dir)
            # os.mkdir(results_dir)

            # # Write map file to logdir
            # with open(os.path.join(log_dir, "map.json"), "w") as f:
            #     f.write(json.dumps(raw_env_json, indent=4))


            # Construct kwargs
            # PBS Config
            if alg == "PBS":
                kwargs = {
                    "map" : json.dumps(raw_env_json),
                    "agentNum" : raw_env_json["num_of_agent"],
                    "cutoffTime": gin.query_parameter("WarehouseEECBSConfig.cutoffTime"),
                    "screen":gin.query_parameter("WarehouseEECBSConfig.screen"),
                    "stats": gin.query_parameter("WarehouseEECBSConfig.stats"),
                    "sipp": gin.query_parameter("WarehouseEECBSConfig.sipp"),
                    "force_new_logdir": False,
                }
            elif alg == "EECBS":
                # EECBS Config
                kwargs = {
                    "map" : json.dumps(raw_env_json),
                    # "output" : log_dir,
                    "agentNum" : raw_env_json["num_of_agent"],
                    "cutoffTime": gin.query_parameter("WarehouseEECBSConfig.cutoffTime"),
                    "screen":gin.query_parameter("WarehouseEECBSConfig.screen"),
                    "stats": gin.query_parameter("WarehouseEECBSConfig.stats"),
                    # "id": gin.query_parameter("WarehouseEECBSConfig.id"),
                    "highLevelSolver":gin.query_parameter("WarehouseEECBSConfig.highLevelSolver"),
                    "lowLevelSolver":gin.query_parameter("WarehouseEECBSConfig.lowLevelSolver"),
                    "inadmissibleH": gin.query_parameter("WarehouseEECBSConfig.inadmissibleH"),
                    "suboptimality": gin.query_parameter("WarehouseEECBSConfig.suboptimality"),
                    "heuristics": gin.query_parameter("WarehouseEECBSConfig.heuristics"),
                    "prioritizingConflicts":gin.query_parameter("WarehouseEECBSConfig.prioritizingConflicts"),
                    "bypass": gin.query_parameter("WarehouseEECBSConfig.bypass"),
                    "disjointSplitting": gin.query_parameter("WarehouseEECBSConfig.disjointSplitting"),
                    "rectangleReasoning": gin.query_parameter("WarehouseEECBSConfig.rectangleReasoning"),
                    "corridorReasoning": gin.query_parameter("WarehouseEECBSConfig.corridorReasoning"),
                    "targetReasoning": gin.query_parameter("WarehouseEECBSConfig.targetReasoning"),
                    "sipp": gin.query_parameter("WarehouseEECBSConfig.sipp"),
                    "restart":gin.query_parameter("WarehouseEECBSConfig.restart"),
                    "force_new_logdir": False,
                }
            kwargs_list.append(kwargs)
        assert(len(kwargs_list) == n_evals)
    else:
        with open(map_filepath, "r") as f:
            raw_env_json = json.load(f)
        map_read = eecbs_env_str2number(raw_env_json['layout']) 
        start_location, goal_location, agent_numbers = [], [], []
        for i in range(0, n_evals):
            start_, goal_, agent_number = calc_start_goal_locations(map_read, agent_ratio=0.1, parts = 5)
            start_location.append(start_)
            goal_location.append(goal_)
            agent_numbers.append(agent_number)
            print(i)
        raw_env_json['start'] = start_location
        raw_env_json['goal'] = goal_location 
        print(len(raw_env_json['start']))

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
        
        if alg == "PBS":
            kwargs = {
                "map" : json.dumps(raw_env_json),
                "agentNum" : agent_numbers[0],
                "cutoffTime": gin.query_parameter("WarehouseEECBSConfig.cutoffTime"),
                "screen":gin.query_parameter("WarehouseEECBSConfig.screen"),
                "stats": gin.query_parameter("WarehouseEECBSConfig.stats"),
                "sipp": gin.query_parameter("WarehouseEECBSConfig.sipp"),
                "force_new_logdir": False,
            }
        elif alg == "EECBS":

            # EECBS Config
            kwargs = {
                "map" : json.dumps(raw_env_json),
                # "output" : log_dir,
                "agentNum" : agent_numbers[0],
                "cutoffTime": gin.query_parameter("WarehouseEECBSConfig.cutoffTime"),
                "screen":gin.query_parameter("WarehouseEECBSConfig.screen"),
                "stats": gin.query_parameter("WarehouseEECBSConfig.stats"),
                # "id": gin.query_parameter("WarehouseEECBSConfig.id"),
                "highLevelSolver":gin.query_parameter("WarehouseEECBSConfig.highLevelSolver"),
                "lowLevelSolver":gin.query_parameter("WarehouseEECBSConfig.lowLevelSolver"),
                "inadmissibleH": gin.query_parameter("WarehouseEECBSConfig.inadmissibleH"),
                "suboptimality": gin.query_parameter("WarehouseEECBSConfig.suboptimality"),
                "heuristics": gin.query_parameter("WarehouseEECBSConfig.heuristics"),
                "prioritizingConflicts":gin.query_parameter("WarehouseEECBSConfig.prioritizingConflicts"),
                "bypass": gin.query_parameter("WarehouseEECBSConfig.bypass"),
                "disjointSplitting": gin.query_parameter("WarehouseEECBSConfig.disjointSplitting"),
                "rectangleReasoning": gin.query_parameter("WarehouseEECBSConfig.rectangleReasoning"),
                "corridorReasoning": gin.query_parameter("WarehouseEECBSConfig.corridorReasoning"),
                "targetReasoning": gin.query_parameter("WarehouseEECBSConfig.targetReasoning"),
                "sipp": gin.query_parameter("WarehouseEECBSConfig.sipp"),
                "restart":gin.query_parameter("WarehouseEECBSConfig.restart"),
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
        eval_id_list = [i for i in range(n_evals)]
        alg_list = [alg for _ in range(n_evals)]  

        real_solution_cost,solution_cost = pool.starmap(
            single_simulation,
            zip(seeds,
                agent_nums,
                repeat(kwargs, n_evals * n_sim),
                repeat(results_dir, n_evals * n_sim),
                eval_id_list,
                alg_list),
        )
    elif mode == "constant":
        agent_nums = [agent_num for _ in range(n_evals)]
        seeds = np.random.choice(np.arange(10000), size=n_evals, replace=False)
        eval_id_list = [i for i in range(n_evals)] 
        alg_list = [alg for _ in range(n_evals)] 

        solution_cost = pool.starmap(
            single_simulation,
            zip(seeds,
                agent_nums,
                repeat(kwargs, n_evals),
                repeat(results_dir, n_evals),
                eval_id_list,
                alg_list),
        )
    elif mode == "maps":
        seeds = np.random.choice(np.arange(10000), size=n_evals, replace=False)
        eval_id_list = [0 for i in range(n_evals)] 
        alg_list = [alg for _ in range(n_evals)] 
        solution_cost = pool.starmap(
            single_simulation,
            zip(seeds,
                agent_list,
                kwargs_list,
                repeat(results_dir, n_evals),
                eval_id_list,
                alg_list,
                obs_list,
                std_bc_list),
        )

    # print("Solution_cost: " ,solution_cost)
    if mode == "maps":
        real_obj = []
        runtime_ = [] 
        obs = []
        std_bc = []
        for sc in solution_cost: 
            obs.append(sc[0])
            std_bc.append(sc[1])
            real_obj.append(sc[2])
            runtime_.append(sc[3])
        
        filename = "output.json"

        data = {
            "obs" : obs,
            "std_bc" : std_bc,
            "real_obj": real_obj,
            "runtime" : runtime_,
        }
    # Write the data to the JSON file
    # with open(filename, 'w') as json_file:
    #     json.dump(data, json_file, indent=4) 

        with open(os.path.join(log_dir, filename), "w") as f:
            f.write(json.dumps(data, indent=4))
    else:
        real_obj = []
        runtime_ = [] 
        for sc in solution_cost: 
            real_obj.append(sc[0])
            runtime_.append(sc[1])
        print("Runtime: ", runtime_) 
        n_failure = np.array(runtime_) >= 20
        n_success = 200 - np.sum(n_failure)
        succ_rate = n_success/200 

        avg_obj = np.mean(runtime_)
        std_obj = np.std(runtime_)
        max_obj = np.max(runtime_)
        min_obj = np.min(runtime_)

        real_avg_obj = np.mean(real_obj)
        real_std_obj = np.std(real_obj)
        real_max_obj = np.max(real_obj)
        real_min_obj = np.min(real_obj)


        n_evals = n_evals * n_sim if mode == "inc_agents" else n_evals

        print(f"Number of success simulations: {n_success}")
        print(f"Success rate: {succ_rate}")
        

        print(f"Average Runtime over {n_evals} simulations: {avg_obj}")
        print(f"Std Runtime over {n_evals} simulations: {std_obj}")
        print(f"Max Runtime over {n_evals} simulations: {max_obj}")
        print(f"Min Runtime over {n_evals} simulations: {min_obj}")


        print(f"Average Solution_cost over {n_evals} simulations: {real_avg_obj}")
        print(f"Std Solution_cost over {n_evals} simulations: {real_std_obj}")
        print(f"Max Solution_cost over {n_evals} simulations: {real_max_obj}")
        print(f"Min Solution_cost over {n_evals} simulations: {real_min_obj}")

    # print("Runtime: ", runtime_) 
    # n_failure = np.array(runtime_) >= 20
    # n_success = 200 - np.sum(n_failure)
    # succ_rate = n_success/200 

    # avg_obj = np.mean(runtime_)
    # std_obj = np.std(runtime_)
    # max_obj = np.max(runtime_)
    # min_obj = np.min(runtime_)

    # real_avg_obj = np.mean(real_obj)
    # real_std_obj = np.std(real_obj)
    # real_max_obj = np.max(real_obj)
    # real_min_obj = np.min(real_obj)


    # n_evals = n_evals * n_sim if mode == "inc_agents" else n_evals

    # print(f"Number of success simulations: {n_success}")
    # print(f"Success rate: {succ_rate}")
    

    # print(f"Average Runtime over {n_evals} simulations: {avg_obj}")
    # print(f"Std Runtime over {n_evals} simulations: {std_obj}")
    # print(f"Max Runtime over {n_evals} simulations: {max_obj}")
    # print(f"Min Runtime over {n_evals} simulations: {min_obj}")


    # print(f"Average Solution_cost over {n_evals} simulations: {real_avg_obj}")
    # print(f"Std Solution_cost over {n_evals} simulations: {real_std_obj}")
    # print(f"Max Solution_cost over {n_evals} simulations: {real_max_obj}")
    # print(f"Min Solution_cost over {n_evals} simulations: {real_min_obj}")


if __name__ == "__main__":
    fire.Fire(main)
