# Lowest Energy Determinants Tests

This directory contains test for finding the lowest energy determinant given spin,
irrep, and point group. The "trusted" data from PySCF is already generated, but if 
you want to tinker with how the data is generated modify `make_integrals.py`. All 
tests are on the C4 molecule with D2h symmetry. For each spin level we target each
of the 8 possible irreps and compare to the exact results from PySCF.