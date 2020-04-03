#!/bin/bash
# This script controls the SHCI crontab test jobs.

printf "\n\nRunning Tests for SHCI/SHCISCF\n"
printf "======================================================\n"

MPICOMMAND="mpirun -np 28"
HCIPATH="../../Dice"
here=`pwd`

## Clean up
cd $here
./clean.sh

# O2 SHCI tests.
cd $here/o2_omp1_stoc
printf "...running o2_omp1_stoc\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py  1.0e-5
python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8

cd $here/o2_omp1_det
printf "...running o2_omp1_det\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py  1.0e-6

cd $here/o2_omp1_det_trev
printf "...running o2_omp1_det_trev\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py  1.0e-6

cd $here/o2_omp1_det_trev_direct
printf "...running o2_omp1_det_trev_direct\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py  1.0e-6

cd $here/o2_omp1_trev_direct
printf "...running o2_omp1_trev_direct\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py  1.0e-6
python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8

cd $here/o2_omp1_det_direct
printf "...running o2_omp1_det_direct\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py  1.0e-6
#python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8

# cd $here/o2_omp4_det
# printf "...running o2_omp4_det\n"
# # export OMP_NUM_THREADS=4
# $MPICOMMAND $HCIPATH > output.dat
# python ../test_energy.py  1.0e-6
#python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8

# cd $here/o2_omp4_seed
# printf "...running o2_omp4_seed\n"
# $MPICOMMAND $HCIPATH > output.dat
# python ../test_energy.py  5.0e-6
# python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8

cd $here/restart
printf "...running restart test\n"
$MPICOMMAND $HCIPATH input2.dat > output2.dat
$MPICOMMAND $HCIPATH input3.dat > output3.dat
python $here/test_energy.py 5e-5

cd $here/restart_trev
printf "...running restart test\n"
$MPICOMMAND $HCIPATH input2.dat > output2.dat
$MPICOMMAND $HCIPATH input3.dat > output3.dat
python $here/test_energy.py 5e-5

cd $here/fullrestart
printf "...running full restart test\n"
$MPICOMMAND $HCIPATH input2.dat > output2.dat
$MPICOMMAND $HCIPATH input3.dat > output3.dat
python $here/test_energy.py 5e-5

cd $here/ref_det
printf "...running reference determinant tests\n"
$MPICOMMAND $HCIPATH input1.dat > output1.dat
python $here/test_energy.py 5e-5 e_spin_0.e
$MPICOMMAND $HCIPATH input2.dat > output2.dat
python $here/test_energy.py 5e-5 e_spin_2.e

# Mn(salen) tests.
cd $here/mn_salen_stoc
printf "...running mn_salen_stoc\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py 8.0e-5

cd $here/mn_salen_seed
printf "...running mn_salen_seed\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py 8.0e-5

# Cr2 Tests
cd $here/cr2_dinfh_rdm
printf "...running cr2_dinfh_rdm\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py 8.0e-5
python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-6

cd $here/cr2_dinfh_trev_rdm
printf "...running cr2_dinfh_trev_rdm\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py 8.0e-5
python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-6

# Butadiene tests.
#cd $here/bd_stoc
#printf "...running bd_stoc\n"
#export OMP_NUM_THREADS=7
#$MPICOMMAND $HCIPATH > output.dat
#python ../test_energy.py 1  1.0e-6

# O2 SHCI tests.
cd $here/n2_RAS
printf "...running n2_RAS\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py  1.0e-5

## Clean up
cd $here
./clean.sh
