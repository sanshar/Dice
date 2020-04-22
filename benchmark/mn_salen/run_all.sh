#!/bin/bash

#
# User settings
#
N_PROCS=28        # Number of MPI processors to use
N_REPEATS=10      # Number of times to repeat the calculation
TEST_TYPE="large" # Either "small" or "large"

# 
# Command shortcuts (DO NOT EDIT)
#

WRAPPED_DICE="/usr/bin/time -v mpirun -np $N_PROCS ../../Dice input_$TEST_TYPE.dat"

#
# Loop To create data
#

# Cleanup old output
rm -f benchmark_raw.out

for ((i=1; i<=$N_REPEATS; i++))
do 
    echo "Running iter: $i"
    $WRAPPED_DICE 1> output.dat 2>> benchmark_raw.out
done

#
# Analyze data
#

# Requires NumPy and Pandas
python analyze_benchmark.py > benchmark.out

