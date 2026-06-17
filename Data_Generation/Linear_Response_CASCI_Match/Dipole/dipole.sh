#!/bin/bash

project=""

declare -A sys_dists
#sys_dists["H2O"]="0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6"
sys_dists["H2_Large"]="0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0"
max_jobs=8
current_jobs=0

for system in "${!sys_dists[@]}"; do
    
    read -r -a dist_array <<< "${sys_dists[$system]}"

    for dist in "${dist_array[@]}"; do
        
        mkdir -p "$project/Output/$system"

        nohup python -u $project/Code/Linear_Response_CASCI_Match/Dipole/$system.py $dist > $project/Output/$system/casci_match_output_dipole_$dist 2>&1 &
        
        ((current_jobs++))
        
        if (( current_jobs >= max_jobs )); then
            wait -n
            ((current_jobs--))
        fi
        
    done
done

wait 