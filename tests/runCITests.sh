#!/bin/bash
# This script controls the SHCI crontab test jobs.

printf "\n\nRunning Tests for SHCI/SHCISCF\n"
printf "======================================================\n"

MPICOMMAND="mpirun -np 2"
MPIFLAGS="--allow-run-as-root"
HCIPATH="../../build/Dice"
here=`pwd`

# O2 SHCI tests.
cd $here/o2_omp1_stoc
printf "...running o2_omp1_stoc\n"
$MPICOMMAND $HCIPATH > output.dat $MPIFLAGS
python ../test_energy.py 1  1.0e-5
python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8

cd $here/o2_omp1_det
printf "...running o2_omp1_det\n"
$MPICOMMAND $HCIPATH > output.dat $MPIFLAGS
python ../test_energy.py 1  1.0e-6
#python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8

cd $here/o2_omp1_det_trev
printf "...running o2_omp1_det_trev\n"
$MPICOMMAND $HCIPATH > output.dat $MPIFLAGS
python ../test_energy.py 1  1.0e-6
#python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8

cd $here/o2_omp1_det_trev_direct
printf "...running o2_omp1_det_trev_direct\n"
$MPICOMMAND $HCIPATH > output.dat $MPIFLAGS
python ../test_energy.py 1  1.0e-6

cd $here/o2_omp1_trev_direct
printf "...running o2_omp1_trev_direct\n"
$MPICOMMAND $HCIPATH > output.dat $MPIFLAGS
python ../test_energy.py 1  1.0e-6
python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8

cd $here/o2_omp1_det_direct
printf "...running o2_omp1_det_direct\n"
$MPICOMMAND $HCIPATH > output.dat $MPIFLAGS
python ../test_energy.py 1  1.0e-6

cd $here/restart
printf "...running restart test\n"
$MPICOMMAND $HCIPATH input2.dat > output2.dat $MPIFLAGS
$MPICOMMAND $HCIPATH input3.dat > output3.dat $MPIFLAGS
python $here/test_energy.py 1 5e-5

cd $here/restart_trev
printf "...running restart test\n"
$MPICOMMAND $HCIPATH input2.dat > output2.dat $MPIFLAGS
$MPICOMMAND $HCIPATH input3.dat > output3.dat $MPIFLAGS
python $here/test_energy.py 1 5e-5

cd $here/fullrestart
printf "...running full restart test\n"
$MPICOMMAND $HCIPATH input2.dat > output2.dat $MPIFLAGS
$MPICOMMAND $HCIPATH input3.dat > output3.dat $MPIFLAGS
python $here/test_energy.py 1 5e-5

## Clean up
cd $here
./clean.sh
