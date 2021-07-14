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
cd $here/o2_stoc
printf "...running o2_stoc\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py  1.0e-5
python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8

cd $here/o2_det
printf "...running o2_det\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py  1.0e-6

cd $here/o2_det_trev
printf "...running o2_det_trev\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py  1.0e-6

cd $here/o2_det_trev_direct
printf "...running o2_det_trev_direct\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py  1.0e-6


cd $here/o2_det_direct
printf "...running o2_det_direct\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py  1.0e-6
#python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8

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
$MPICOMMAND $HCIPATH input3.dat > output3.dat
python $here/test_energy.py 5e-5 e_spin_1.e
$MPICOMMAND $HCIPATH input4.dat > output4.dat
python $here/test_energy.py 5e-5 e_spin_3.e

cd $here/spin1rdm
printf "...running spin1RDM tests\n"
$HCIPATH input.dat > output.dat
python $here/test_spin1RDM.py spin1RDM.0.0.txt spatial1RDM.0.0.txt 1e-8
$MPICOMMAND $HCIPATH input.dat > output.dat
python $here/test_spin1RDM.py spin1RDM.0.0.txt spatial1RDM.0.0.txt 1e-8

cd $here/spin2rdm
printf "...running spin2RDM tests\n"
printf "WARNING: PySCF must be installed (or in PYTHONPATH) for these tests to run.\n"
printf "WARNING: The spin2RDM tests are EXTENSIVE. If 1-3 tests fail, that's usually OK.\n"
printf "WARNING: We test RDMs from high-lying excited states so tests can sometimes fail\n"
printf "WARNING: because they are slightly larger than our selected tolerances.\n"
printf "WARNING: See Dice/tests/spin2RDM/ for specific details.\n\n"
python test_spin2rdm.py

cd $here/H2He_rdm
printf "...running H2He RDM Test\n"
$HCIPATH input.dat > output.dat
python $here/test_onepdm.py spatial1RDM.0.0.txt pyscf_1RDM.npy 1.e-9
python $here/test_twopdm.py spatialRDM.0.0.txt pyscf_2RDM.npy 1.e-9
python $here/test_threepdm.py spatial3RDM.0.0.txt pyscf_3RDM.npy 1.e-9
python $here/test_spin1RDM.py spin1RDM.0.0.txt spatial1RDM.0.0.txt 1e-9
python $here/test_rdm_nelec.py ../integrals/H2H2_FCIDUMP_nosym spatial1RDM.0.0.txt spatialRDM.0.0.txt spatial3RDM.0.0.txt 5e-8

cd $here/h2o_trev_direct_rdm
printf "...running H2O Trev Direct RDM Test\n"
$HCIPATH test_input.dat > output.dat
python $here/test_twopdm.py spatialRDM.0.0.txt pyscf_2RDM.npy 2.e-8

cd $here/h2co_rdm
printf "...running H2CO RDM Test\n"
$HCIPATH test_input.dat > output.dat
python $here/test_onepdm.py spatial1RDM.0.0.txt pyscf_1RDM.npy 1e-8
python $here/test_twopdm.py spatialRDM.0.0.txt pyscf_2RDM.npy 1e-9
python $here/test_threepdm.py spatial3RDM.0.0.txt pyscf_3RDM.npy 1e-10
python $here/test_rdm_nelec.py ../integrals/h2co_FCIDUMP_nosymm spatial1RDM.0.0.txt spatialRDM.0.0.txt spatial3RDM.0.0.txt 1e-11

cd $here/lowest_energy_det
printf "...running test for finding the lowest energy determinants\n"
printf "        Running tests on Spin=0"
bash run_spin=0_tests.sh "$MPICOMMAND" $HCIPATH
printf "        Running tests on Spin=1"
bash run_spin=1_tests.sh "$MPICOMMAND" $HCIPATH
printf "        Running tests on Spin=2"
bash run_spin=2_tests.sh "$MPICOMMAND" $HCIPATH
printf "        Running tests on Spin=3"
bash run_spin=3_tests.sh "$MPICOMMAND" $HCIPATH
printf "        Running tests on Spin=4"
bash run_spin=4_tests.sh "$MPICOMMAND" $HCIPATH

#
# Larger/Longer Tests
#

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
python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-9

cd $here/cr2_dinfh_trev_rdm
printf "...running cr2_dinfh_trev_rdm\n"
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py 8.0e-5
python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-9



## Clean up
cd $here
./clean.sh
