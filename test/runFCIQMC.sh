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
    ../../clean.sh
fi

cd $here/FCIQMC/He2_hb_uniform
../../clean.sh
printf "...running FCIQMC/He2_hb_uniform\n"
$MPICOMMAND $FCIQMCPATH > fciqmc.out
python2 ../../testEnergy.py 'fciqmc' $tol
if [ $clean == 1 ]
then
    ../../clean.sh
fi

cd $here/FCIQMC/Ne_plateau
../../clean.sh
printf "...running FCIQMC/Ne_plateau\n"
$MPICOMMAND $FCIQMCPATH > fciqmc.out
python2 ../../testEnergy.py 'fciqmc' $tol
if [ $clean == 1 ]
then
    ../../clean.sh
fi

cd $here/FCIQMC/Ne_initiator
../../clean.sh
printf "...running FCIQMC/Ne_initiator\n"
$MPICOMMAND $FCIQMCPATH > fciqmc.out
python2 ../../testEnergy.py 'fciqmc' $tol
if [ $clean == 1 ]
then
    ../../clean.sh
fi

cd $here/FCIQMC/Ne_initiator_replica
../../clean.sh
printf "...running FCIQMC/Ne_initiator_replica\n"
$MPICOMMAND $FCIQMCPATH > fciqmc.out
python2 ../../testEnergy.py 'fciqmc_replica' $tol
if [ $clean == 1 ]
then
    ../../clean.sh
fi

cd $here/FCIQMC/Ne_initiator_en2
../../clean.sh
printf "...running FCIQMC/Ne_initiator_en2\n"
$MPICOMMAND $FCIQMCPATH > fciqmc.out
python2 ../../testEnergy.py 'fciqmc_replica' $tol
if [ $clean == 1 ]
then
    ../../clean.sh
fi

cd $here/FCIQMC/Ne_initiator_en2_ss
../../clean.sh
printf "...running FCIQMC/Ne_initiator_en2_ss\n"
$MPICOMMAND $FCIQMCPATH > fciqmc.out
python2 ../../testEnergy.py 'fciqmc_replica' $tol
if [ $clean == 1 ]
then
    ../../clean.sh
fi

cd $here/FCIQMC/water_vdz_hb
../../clean.sh
printf "...running FCIQMC/water_vdz_hb\n"
$MPICOMMAND $FCIQMCPATH > fciqmc.out
python2 ../../testEnergy.py 'fciqmc' $tol
if [ $clean == 1 ]
then
    ../../clean.sh
fi

cd $here/FCIQMC/N2_fixed_node
../../clean.sh
printf "...running FCIQMC/N2_fixed_node\n"
$MPICOMMAND $FCIQMCPATH > fciqmc.out
python2 ../../testEnergy.py 'fciqmc_trial' $tol
if [ $clean == 1 ]
then
    ../../clean.sh
fi

cd $here/FCIQMC/N2_is_ss
../../clean.sh
printf "...running FCIQMC/N2_is_ss\n"
$MPICOMMAND $FCIQMCPATH > fciqmc.out
python2 ../../testEnergy.py 'fciqmc_trial' $tol
if [ $clean == 1 ]
then
    ../../clean.sh
fi

cd $here/FCIQMC/H10_free_prop
../../clean.sh
printf "...running FCIQMC/H10_free_prop\n"
$MPICOMMAND $FCIQMCPATH > fciqmc.out
python2 ../../testEnergy.py 'fciqmc_trial' $tol
if [ $clean == 1 ]
then
    ../../clean.sh
fi

cd $here/FCIQMC/H10_partial_node
../../clean.sh
printf "...running FCIQMC/H10_partial_node\n"
$MPICOMMAND $FCIQMCPATH > fciqmc.out
python2 ../../testEnergy.py 'fciqmc_trial' $tol
if [ $clean == 1 ]
then
    ../../clean.sh
fi

cd $here
