#!/bin/bash

printf "\n\nRunning Tests for VMC/GFMC/NEVPT2/FCIQMC/DQMC\n"
printf "======================================================\n"

here=`pwd`

./runVMC.sh
cd $here

./runGFMC.sh
cd $here

./runNEVPT.sh
cd $here

./runFCIQMC.sh
cd $here

./runDQMC.sh
cd $here

