#!/bin/bash
# This script controls the SHCI crontab test jobs.

printf "\n\nRunning Tests for SHCI/SHCISCF\n"
printf "======================================================\n"

MPICOMMAND="mpirun -np 28"
HCIPATH="../../Dice"
here=`pwd`

# O2 SHCI tests.
cd $here/o2_omp1_stoc
printf "...running o2_omp1_stoc\n"
export OMP_NUM_THREADS=1
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py 1  1.0e-5
python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8

cd $here/o2_omp1_det
printf "...running o2_omp1_det\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py 1  1.0e-6
#python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8

cd $here/o2_omp1_det_trev
printf "...running o2_omp1_det_trev\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py 1  1.0e-6
#python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8

cd $here/o2_omp1_det_trev_direct
printf "...running o2_omp1_det_trev_direct\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py 1  1.0e-6

cd $here/o2_omp1_trev_direct
printf "...running o2_omp1_trev_direct\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py 1  1.0e-6
python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8

cd $here/o2_omp1_det_direct
printf "...running o2_omp1_det_direct\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py 1  1.0e-6
#python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8

cd $here/o2_omp4_det
printf "...running o2_omp4_det\n"
export OMP_NUM_THREADS=4
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py 1  1.0e-6
#python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8

cd $here/o2_omp4_seed
printf "...running o2_omp4_seed\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py 1  5.0e-6
python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8

cd $here/restart
printf "...running restart test\n"
$MPICOMMAND $HCIPATH input2.dat > output2.dat
$MPICOMMAND $HCIPATH input3.dat > output3.dat
python $here/test_energy.py 1 5e-5

cd $here/restart_trev
printf "...running restart test\n"
$MPICOMMAND $HCIPATH input2.dat > output2.dat
$MPICOMMAND $HCIPATH input3.dat > output3.dat
python $here/test_energy.py 1 5e-5

cd $here/fullrestart
printf "...running full restart test\n"
$MPICOMMAND $HCIPATH input2.dat > output2.dat
$MPICOMMAND $HCIPATH input3.dat > output3.dat
python $here/test_energy.py 1 5e-5

## PT RDM Test
#cd $here/c2_pt_rdm
#printf "...running c2_pt_rdm\n"
#$MPICOMMAND $HCIPATH > output.dat
#python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8

# Mn(salen) tests.
cd $here/mn_salen_stoc
export OMP_NUM_THREADS=1
printf "...running mn_salen_stoc\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py 1  8.0e-5

cd $here/mn_salen_seed
printf "...running mn_salen_seed\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py 1  8.0e-5

# Butadiene tests.
#cd $here/bd_stoc
#printf "...running bd_stoc\n"
#export OMP_NUM_THREADS=7
#$MPICOMMAND $HCIPATH > output.dat
#python ../test_energy.py 1  1.0e-6


## Clean up
cd $here
./clean.sh
