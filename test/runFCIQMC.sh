#!/bin/bash

printf "\n\nRunning Tests for FCIQMC\n"
printf "======================================================\n"

MPICOMMAND="mpirun -np 4"
FCIQMCPATH="../../../bin/FCIQMC fciqmc.json"
here=`pwd`
tol=1.0e-7
clean=1

cd $here/FCIQMC/He2
../../clean.sh
printf "...running FCIQMC/He2\n"
$MPICOMMAND $FCIQMCPATH > fciqmc.out
python2 ../../testEnergy.py 'fciqmc' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

cd $here/FCIQMC/Ne_plateau
../../clean.sh
printf "...running FCIQMC/Ne_plateau\n"
$MPICOMMAND $FCIQMCPATH > fciqmc.out
python2 ../../testEnergy.py 'fciqmc' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

cd $here/FCIQMC/Ne_initiator
../../clean.sh
printf "...running FCIQMC/Ne_initiator\n"
$MPICOMMAND $FCIQMCPATH > fciqmc.out
python2 ../../testEnergy.py 'fciqmc' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

cd $here
