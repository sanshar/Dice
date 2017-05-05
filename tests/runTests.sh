#!/bin/bash
# This script controls the SHCI crontab test jobs.

printf "\n\nRunning Tests for SHCI/SHCISCF\n" 
printf "======================================================\n"

MPICOMMAND="mpirun -np 4"
HCIPATH="../../SHCI"

# O2 SHCI tests.
cd o2_omp1_stoc
printf "...running o2_omp1_stoc\n"
export OMP_NUM_THREADS=1
$HCIPATH > output.dat
python ../test_energy.py 1  1.0e-5 


cd ../o2_omp1_det
printf "...running o2_omp1_det\n" 
$HCIPATH > output.dat
python ../test_energy.py 1  1.0e-6
python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8

cd ../o2_omp4_det
printf "...running o2_omp4_det\n" 
export OMP_NUM_THREADS=4
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py 1  1.0e-6
python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8

cd ../o2_omp4_seed
printf "...running o2_omp4_seed\n" 
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py 1  5.0e-6

# PT RDM Test
cd ../c2_pt_rdm
printf "...running c2_pt_rdm\n" 
$MPICOMMAND $HCIPATH > output.dat
python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8

# Mn(salen) tests.
cd ../mn_salen_stoc
printf "...running mn_salen_stoc\n" 
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py 1  8.0e-5 

cd ../mn_salen_seed
printf "...running mn_salen_seed\n" 
$MPICOMMAND $HCIPATH > output.dat
python ../test_energy.py 1  8.0e-5 

# Butadiene tests.
#cd ../bd_stoc
#printf "...running bd_stoc\n" 
#export OMP_NUM_THREADS=7
#$MPICOMMAND $HCIPATH > output.dat
#python ../test_energy.py 1  1.0e-6 





## Pyscf/SHCI (SHCISCF) tests. (6 x unittests)
#cd shciscf_c2
#printf "...running test_c2.py\n" >> ../shciTest.log



## Clean up
./clean.sh
