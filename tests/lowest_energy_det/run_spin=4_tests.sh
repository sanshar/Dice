#!/bin/bash

#
# Commandline Args
#
MPICOMMAND=$1
HCIPATH=$2
SPIN=4
FCIDUMP_PATH="c4_FCIDUMP_d2h_spin=${SPIN}"

if [ "$#" -ne 2 ]; then
    echo "Wrong number of commandline args"
    echo "Use the following:"
    echo "        bash $0 MPICOMMAND HCIPATH"
    exit
# else
#     echo "Commandline Args"
#     echo "================"
#     echo "MPICOMMAND = " $MPICOMMAND
#     echo "HCIPATH = " $HCIPATH
#     echo ""
fi



# Setup
mkdir -p _inputs _outputs _energies

# Sanity check to make "known solutions" don't have spin contamination
# grep "S^2" _make_integrals.out | tail -n 8

# Run tests
for IRREP in Ag B3u B2u B1g B1u B2g B3g Au
do 

F_INPUT=_inputs/${IRREP}.dat

echo "
#system
nocc 8
0 2 4 6 1 3 5 7
end
orbitals ../integrals/${FCIDUMP_PATH}
nroots 1
pointGroup d2h
irrep ${IRREP}
spin ${SPIN}
searchForLowestEnergyDet

#variational
schedule 
0	0
3	0
end
davidsonTol 5e-05
dE 1e-08
maxiter 9

#pt
nPTiter 0
epsilon2 1e-07
epsilon2Large 1000
targetError 0.0001
sampleN 200

# misc
noio

" > ${F_INPUT}

# printf "Running test on ${IRREP}\n"
$MPICOMMAND $HCIPATH ${F_INPUT} > _outputs/${IRREP}.out
mv shci.e _energies/${IRREP}_energy.e

done

python compare_energies.py pyscf_energies_spin=$SPIN.csv 1e-6 

rm -rf _inputs _energies _outputs
rm -f *.dat *.bkp *.txt rm tmp*