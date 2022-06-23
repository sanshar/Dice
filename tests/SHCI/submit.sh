#!/bin/bash

#SBATCH --job-name runTests
#SBATCH --nodes 1
#SBATCH --exclusive
#SBATCH --time 00:50:00
#SBATCH --export=NONE
##SBATCH --qos blanca-sh

export OMP_NUM_THREADS=1

#module purge
module load intel
module load mkl
module load python

export I_MPI_FABRICS=tcp
export TMP="/rc_scratch/jasm3285/" 
export TEMP="/rc_scratch/jasm3285/"
export TMPDIR="/rc_scratch/jasm3285/"
export PATH=/curc/slurm/blanca/current/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jasm3285/projects/boost_1_57_0/stage_blanca/lib
export PYTHONPATH=$PYTHONPATH:/home/jasm3285/projects
export LD_PRELOAD=$MKLROOT/lib/intel64/libmkl_avx.so:$MKLROOT/lib/intel64/libmkl_core.so:$MKLROOT/lib/intel64/libmkl_sequential.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/curc/sw/gcc/6.1.0/lib64

./runTests.sh



exit
