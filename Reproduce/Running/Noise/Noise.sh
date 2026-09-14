#!/bin/bash

project="***"

declare -A sys_dists
sys_dists["H2_Noise"]="1 2 3"

system_order=(H2_Noise)

max_jobs=10
current_jobs=0
for system in "${system_order[@]}"; do
    mkdir -p $project/Output/$system
    mkdir -p $project/Data/$system
    mkdir -p $project/Data/$system/Chkfile
    mkdir -p $project/Data/$system/Energy
    mkdir -p $project/Data/$system/Res
    mkdir -p $project/Data/$system/qEOM
    mkdir -p $project/Data/$system/Dipole
    mkdir -p $project/Data/$system/Spin_Square
    read -r -a dist_array <<< "${sys_dists[$system]}"
    for dist in "${dist_array[@]}"; do
        mkdir -p "$project/Output/$system"
        nohup python -u $project/Code/Running/Noise/$system.py $dist > $project/Output/$system/Noise_$dist.out 2>&1 &
        ((current_jobs++))
        if (( current_jobs >= max_jobs )); then
            wait -n
            ((current_jobs--))
        fi  
    done
done

wait 
echo "All Calculation Done"