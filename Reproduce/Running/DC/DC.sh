#!/bin/bash

project="***"

declare -A sys_dists
sys_dists["H2"]="0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0"
sys_dists["LiH"]="1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0"
sys_dists["H2O"]="0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6"

system_order=(H2 LiH H2O)

max_jobs=32
current_jobs=0
for system in "${system_order[@]}"; do
    mkdir -p $project/Output/$system
    mkdir -p $project/Data/$system
    mkdir -p $project/Data/$system/Chkfile
    mkdir -p $project/Data/$system/Density
    mkdir -p $project/Data/$system/Energy
    mkdir -p $project/Data/$system/Res
    mkdir -p $project/Data/$system/qEOM
    mkdir -p $project/Data/$system/Dipole
    mkdir -p $project/Data/$system/Spin_Square
    mkdir -p $project/Data/$system/VAC
    mkdir -p $project/Data/$system/Hermitian_Contamination
    read -r -a dist_array <<< "${sys_dists[$system]}"
    for dist in "${dist_array[@]}"; do
        mkdir -p "$project/Output/$system"
        nohup python -u $project/Code/Running/DC/$system.py $dist > $project/Output/$system/DC_$dist.out 2>&1 &
        ((current_jobs++))
        if (( current_jobs >= max_jobs )); then
            wait -n
            ((current_jobs--))
        fi  
    done
done

wait 
echo "All Calculation Done"