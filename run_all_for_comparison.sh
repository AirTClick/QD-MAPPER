#!/bin/bash

readarray -t files < <(find "logs/PBS_python_start_goal/generated_maps/" -type f)
readarray -t gins < <(find "config/test/" -type f)

output_file="output_10.txt"

for file in "${files[@]}"; do
    for gin in "${gins[@]}"; do 
        # Check if the current item is a file
        echo "$file"
        echo "$gin"
        output=$(bash scripts/run_single_sim_EECBS.sh "$gin" "$file" 150 200 constant 1)
        echo "$output" 
    done  
done > "$output_file"


