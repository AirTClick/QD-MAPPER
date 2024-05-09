#!/bin/bash

print_header() {
  echo
  echo "------------- $1 -------------"
}

# Prints "=" across an entire line.
print_thick_line() {
  printf "%0.s=" $(seq 1 `tput cols`)
  echo
}

readarray -t files < <(find "../logs/PBS_differ_start_goal_python/generated_maps/" -type f)
readarray -t gins < <(find "../config/test/" -type f)

for file in "${files[@]}"; do
    for gin in "${gins[@]}"; do 
        # Check if the current item is a file
        echo "$file"
        echo "$gin"
        python3 warehouseEECBS/module.py "$gin" "$file" --agent_num 150 --n_evals 1 --n_sim 1 --mode constant
        # echo "$output" 
    done  
done 