#!/bin/bash

project=""

declare -A sys_dists
sys_dists["H2_Noise"]="0 1 2 3"

max_jobs=8
current_jobs=0

for system in "${!sys_dists[@]}"; do
    
    mkdir -p "$project/Output/$system"
    mkdir -p "$project/Data/$system"
    mkdir -p "$project/Data/$system/Chkfile"
    mkdir -p "$project/Data/$system/Density"
    mkdir -p "$project/Data/$system/Energy"
    mkdir -p "$project/Data/$system/Res"
    mkdir -p "$project/Data/$system/qEOM"
    mkdir -p "$project/Data/$system/Dipole"

    read -r -a dist_array <<< "${sys_dists[$system]}"

    for dist in "${dist_array[@]}"; do
        
        mkdir -p "$project/Output/$system"
        mkdir -p "$project/Data/$system/Noise"

        nohup python -u $project/Code/Linear_Response/Noise/$system.py $dist > $project/Output/$system/output_Noise_$dist 2>&1 &
        
        ((current_jobs++))
        
        if (( current_jobs >= max_jobs )); then
            wait -n
            ((current_jobs--))
        fi
        
    done
done

wait 