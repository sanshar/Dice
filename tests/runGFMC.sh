#!/bin/bash

printf "\n\nRunning Tests for GFMC\n"
printf "======================================================\n"

MPICOMMAND="mpirun -np 4"
VMCPATH="../../../bin/VMC vmc.json"
GFMCPATH="../../../bin/GFMC gfmc.json"
here=`pwd`
tol=1.0e-7
clean=1

cd $here/GFMC/hubbard_18_tilt/
../../clean.sh
printf "...running hubbard_18_tilt uhf\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../../testEnergy.py 'vmc' $tol
printf "...running hubbard_18_tilt gfmc\n"
$MPICOMMAND $GFMCPATH > gfmc.out
python2 ../../testEnergy.py 'gfmc' $tol
if [ $clean == 1 ]
then    
    ../../clean.sh
fi

cd $here
