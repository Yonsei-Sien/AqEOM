project=
system=H2O

Dists=(0.5 0.6) # 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6

mkdir $project/Output/$system
mkdir $project/Data/$system
mkdir $project/Data/$system/Probability_Distribution
mkdir $project/Data/$system/Chkfile
mkdir $project/Data/$system/Density
mkdir $project/Data/$system/Energy
mkdir $project/Data/$system/Res
mkdir $project/Data/$system/qEOM
mkdir $project/Data/$system/Dipole
mkdir $project/Data/$system/VAC

for dist in "${Dists[@]}"; do
    nohup python -u $project/Code/Running/$system.py $dist > $project/Output/$system/output_$dist &
done
