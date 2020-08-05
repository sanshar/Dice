#!/bin/bash

printf "\n\nRunning Tests for NEVPT2\n"
printf "======================================================\n"

MPICOMMAND="mpirun -np 4"
NEVPTPATH="../../../../bin/VMC nevpt.json"
here=`pwd`
tol=1.0e-7
clean=1

# SC-NEVPT2 tests

cd $here/NEVPT2/n2_vdz/stoch
../../../clean.sh
printf "...running NEVPT2/n2_vdz/stoch\n"
$MPICOMMAND $NEVPTPATH > nevpt.out
python2 ../../../testEnergy.py 'nevpt' $tol
if [ $clean == 1 ]
then
    ../../../clean.sh
fi

cd $here/NEVPT2/n2_vdz/continue_norms
../../../clean_wo_bkp.sh
printf "...running NEVPT2/n2_vdz/continue_norms\n"
$MPICOMMAND $NEVPTPATH > nevpt.out
python2 ../../../testEnergy.py 'nevpt' $tol
if [ $clean == 1 ]
then
    ../../../clean_wo_bkp.sh
fi

cd $here/NEVPT2/n2_vdz/write_exact_energies
../../../clean.sh
printf "...running NEVPT2/n2_vdz/write_exact_energies\n"
$NEVPTPATH > nevpt.out
python2 ../../../testEnergy.py 'nevpt' $tol
if [ $clean == 1 ]
then
    ../../../clean.sh
fi

cd $here/NEVPT2/n2_vdz/read_exact_energies
../../../clean_wo_bkp.sh
printf "...running NEVPT2/n2_vdz/read_exact_energies\n"
$NEVPTPATH > nevpt.out
python2 ../../../testEnergy.py 'nevpt' $tol
if [ $clean == 1 ]
then
    ../../../clean_wo_bkp.sh
fi

cd $here/NEVPT2/h4_631g/determ
../../../clean.sh
printf "...running NEVPT2/h4_631g/determ\n"
$MPICOMMAND $NEVPTPATH > nevpt.out
python2 ../../../testEnergy.py 'nevpt' $tol
if [ $clean == 1 ]
then
    ../../../clean.sh
fi

cd $here/NEVPT2/polyacetylene/stoch
../../../clean.sh
printf "...running NEVPT2/polyacetylene/stoch\n"
$MPICOMMAND $NEVPTPATH > nevpt.out
python2 ../../../testEnergy.py 'nevpt' $tol
if [ $clean == 1 ]
then
    ../../../clean.sh
fi

# SC-NEVPT2 single perturber

cd $here/NEVPT2/n2_vdz/single_perturber
../../../clean.sh
printf "...running NEVPT2/n2_vdz/single_perturber\n"
$NEVPTPATH > nevpt.out
python2 ../../../testEnergy.py 'single_perturber' $tol
if [ $clean == 1 ]
then
    ../../../clean.sh
fi

cd $here
