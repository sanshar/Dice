#!/bin/bash

printf "\n\nRunning Tests for AFQMC\n"
printf "======================================================\n"

MPICOMMAND="mpirun -np 4"
DQMCPATH="../../../bin/DQMC afqmc.json"
here=`pwd`
tol=1.0e-7
clean=1

cd $here/DQMC/rhf_rhf
../../clean.sh
printf "...running DQMC/rhf_rhf\n"
$MPICOMMAND $DQMCPATH > afqmc.out
python2 ../../testEnergy.py 'afqmc' $tol
if [ $clean == 1 ]
then
    ../../clean.sh
fi

cd $here/DQMC/rhf_uhf
../../clean.sh
printf "...running DQMC/rhf_uhf\n"
$MPICOMMAND $DQMCPATH > afqmc.out
python2 ../../testEnergy.py 'afqmc' $tol
if [ $clean == 1 ]
then
    ../../clean.sh
fi

cd $here/DQMC/uhf_rhf
../../clean.sh
printf "...running DQMC/uhf_rhf\n"
$MPICOMMAND $DQMCPATH > afqmc.out
python2 ../../testEnergy.py 'afqmc' $tol
if [ $clean == 1 ]
then
    ../../clean.sh
fi

cd $here/DQMC/uhf_uhf
../../clean.sh
printf "...running DQMC/uhf_uhf\n"
$MPICOMMAND $DQMCPATH > afqmc.out
python2 ../../testEnergy.py 'afqmc' $tol
if [ $clean == 1 ]
then
    ../../clean.sh
fi

cd $here/DQMC/multislater_rhf
../../clean.sh
printf "...running DQMC/multislater_rhf\n"
$MPICOMMAND $DQMCPATH > afqmc.out
python2 ../../testEnergy.py 'afqmc' $tol
if [ $clean == 1 ]
then
    ../../clean.sh
fi

cd $here/DQMC/multislater_uhf
../../clean.sh
printf "...running DQMC/multislater_uhf\n"
$MPICOMMAND $DQMCPATH > afqmc.out
python2 ../../testEnergy.py 'afqmc' $tol
if [ $clean == 1 ]
then
    ../../clean.sh
fi

cd $here/DQMC/ghf_ghf_soc
../../clean.sh
printf "...running DQMC/ghf_ghf_soc\n"
$MPICOMMAND $DQMCPATH > afqmc.out
python2 ../../testEnergy.py 'afqmc' $tol
if [ $clean == 1 ]
then
    ../../clean.sh
fi

cd $here/DQMC/uhf_uhf_ui
../../clean.sh
printf "...running DQMC/uhf_uhf_ui\n"
$MPICOMMAND $DQMCPATH > afqmc.out
python2 ../../testEnergy.py 'afqmc' $tol
if [ $clean == 1 ]
then
    ../../clean.sh
fi

cd $here/DQMC/multislater_uhf_ui
../../clean.sh
printf "...running DQMC/multislater_uhf_ui\n"
$MPICOMMAND $DQMCPATH > afqmc.out
python2 ../../testEnergy.py 'afqmc' $tol
if [ $clean == 1 ]
then
    ../../clean.sh
fi

cd $here/DQMC/ghf_ghf_gi
../../clean.sh
printf "...running DQMC/ghf_ghf_gi\n"
$MPICOMMAND $DQMCPATH > afqmc.out
python2 ../../testEnergy.py 'afqmc' $tol
if [ $clean == 1 ]
then
    ../../clean.sh
fi

cd $here
