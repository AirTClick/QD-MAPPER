#!/bin/bash

readarray -t files < <(find "../logs/PBS_differ_start_goal_python/generated_maps/" -type f)
readarray -t gins < <(find "../config/test/" -type f)

gin="../config/test/pure_simulation_EECBS_0.gin"

for file in "${files[@]}"; do
        echo "$file"
        python3 warehouseEECBS/module.py "$gin" "$file" --agent_num 150 --n_evals 1 --n_sim 1 --mode constant
        # echo "$output" 
    done  
done 