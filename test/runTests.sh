#!/bin/bash

printf "\n\nRunning Tests for VMC/GFMC/NEVPT2/FCIQMC\n"
printf "======================================================\n"

MPICOMMAND="mpirun -np 4"
VMCPATH="../../bin/VMC vmc.json"
CIPATH="../../bin/VMC ci.json"
LANCZOSPATH="../../bin/VMC lanczos.dat"
GFMCPATH="../../bin/GFMC gfmc.json"
NEVPTPATH="../../../../bin/VMC nevpt.json"
NEVPTPRINTPATH="../../../../bin/VMC nevpt_print.json"
NEVPTREADPATH="../../../../bin/VMC nevpt_read.json"
FCIQMCPATH="../../../bin/FCIQMC fciqmc.json"
here=`pwd`
tol=1.0e-7
clean=1

# VMC tests

cd $here/hubbard_1x10
../clean.sh
printf "...running hubbard_1x10\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
#printf "...running hubbard_1x10 lanczos\n"
#$MPICOMMAND $LANCZOSPATH > lanczos.out
#python2 ../testEnergy.py 'lanczos' $tol
if [ $clean == 1 ]
then    
    ../clean.sh
fi

cd $here/hubbard_1x10ghf
../clean.sh
printf "...running hubbard_1x10 ghf\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then    
    ../clean.sh
fi

cd $here/hubbard_1x10agp
../clean.sh
printf "...running hubbard_1x10 agp\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then    
    ../clean.sh
fi

cd $here/hubbard_1x14
../clean.sh
printf "...running hubbard_1x14\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
printf "...running hubbard_1x14 ci\n"
$MPICOMMAND $CIPATH > ci.out
python2 ../testEnergy.py 'ci' $tol
if [ $clean == 1 ]
then    
    ../clean.sh
fi

cd $here/hubbard_1x22
../clean.sh
printf "...running hubbard_1x22\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then    
    ../clean.sh
fi

cd $here/hubbard_1x50
../clean.sh
printf "...running hubbard_1x50\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then    
    ../clean.sh
fi

cd $here/hubbard_1x6
../clean.sh
printf "...running hubbard_1x6\n"
$VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then    
    ../clean.sh
fi

cd $here/hubbard_18_tilt/
../clean.sh
printf "...running hubbard_18_tilt uhf\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
printf "...running hubbard_18_tilt gfmc\n"
$MPICOMMAND $GFMCPATH > gfmc.out
python2 ../testEnergy.py 'gfmc' $tol
if [ $clean == 1 ]
then    
    ../clean.sh
fi

#cd $here/h6/
#../clean.sh
#printf "...running h6\n"
#$MPICOMMAND $VMCPATH > vmc.out
#python2 ../testEnergy.py 'vmc' $tol
#if [ $clean == 1 ]
#then    
#    ../clean.sh
#fi

cd $here/h4_ghf_complex/
../clean.sh
printf "...running h4 ghf complex\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then    
    ../clean.sh
fi

cd $here/h4_pfaffian_complex/
../clean.sh
printf "...running h4 pfaffian complex\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then    
    ../clean.sh
fi

#cd $here/h10sr/
#../clean.sh
#printf "...running h10 sr\n"
#$MPICOMMAND $VMCPATH > vmc.out
#python2 ../testEnergy.py 'vmc' $tol
#if [ $clean == 1 ]
#then    
#    ../clean.sh
#fi

cd $here/h10pfaff/
../clean.sh
printf "...running h10 pfaffian\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then    
    ../clean.sh
fi

cd $here/h20/
../clean.sh
printf "...running h20\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then    
    ../clean.sh
fi

cd $here/h20ghf/
../clean.sh
printf "...running h20 ghf\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then    
    ../clean.sh
fi

cd $here/c2/
../clean.sh
printf "...running c2\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then    
    ../clean.sh
fi

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
printf "...running NEVPT2/n2_vdz/continue_norms PRINT\n"
$MPICOMMAND $NEVPTPRINTPATH > nevpt_print.out
python2 ../../../testEnergy.py 'nevpt_print' $tol
if [ $clean == 1 ]
then
    ../../../clean_wo_bkp.sh
fi

cd $here/NEVPT2/n2_vdz/continue_norms
../../../clean_wo_bkp.sh
printf "...running NEVPT2/n2_vdz/continue_norms READ\n"
$MPICOMMAND $NEVPTREADPATH > nevpt_read.out
python2 ../../../testEnergy.py 'nevpt_read' $tol
if [ $clean == 1 ]
then
    ../../../clean.sh
fi

cd $here/NEVPT2/n2_vdz/exact_energies
../../../clean.sh
printf "...running NEVPT2/n2_vdz/exact_energies PRINT\n"
$NEVPTPRINTPATH > nevpt_print.out
python2 ../../../testEnergy.py 'nevpt_print' $tol
if [ $clean == 1 ]
then
    ../../../clean_wo_bkp.sh
fi

cd $here/NEVPT2/n2_vdz/exact_energies
../../../clean_wo_bkp.sh
printf "...running NEVPT2/n2_vdz/exact_energies READ\n"
$NEVPTREADPATH > nevpt_read.out
python2 ../../../testEnergy.py 'nevpt_read' $tol
if [ $clean == 1 ]
then
    ../../../clean.sh
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

#cd $here/FCIQMC/N2_fixed_node
#../../clean.sh
#printf "...running FCIQMC/N2_fixed_node\n"
#$MPICOMMAND $FCIQMCPATH > fciqmc.out
#python2 ../../testEnergy.py 'fciqmc_trial' $tol
#if [ $clean == 1 ]
#then
#    ../../clean.sh
#fi
#
#cd $here/FCIQMC/N2_is_ss
#../../clean.sh
#printf "...running FCIQMC/N2_is_ss\n"
#$MPICOMMAND $FCIQMCPATH > fciqmc.out
#python2 ../../testEnergy.py 'fciqmc_trial' $tol
#if [ $clean == 1 ]
#then
#    ../../clean.sh
#fi
#
#cd $here/FCIQMC/H10_free_prop
#../../clean.sh
#printf "...running FCIQMC/H10_free_prop\n"
#$MPICOMMAND $FCIQMCPATH > fciqmc.out
#python2 ../../testEnergy.py 'fciqmc_trial' $tol
#if [ $clean == 1 ]
#then
#    ../../clean.sh
#fi
#
#cd $here/FCIQMC/H10_partial_node
#../../clean.sh
#printf "...running FCIQMC/H10_partial_node\n"
#$MPICOMMAND $FCIQMCPATH > fciqmc.out
#python2 ../../testEnergy.py 'fciqmc_trial' $tol
#if [ $clean == 1 ]
#then
#    ../../clean.sh
#fi

cd $here

./runDQMC.sh

cd $here
