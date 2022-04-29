# QMC methods for *ab initio* electronic structure theory

This repository contains code for performing VMC, GFMC, DMC, FCIQMC, stochastic MRCI and SC-NEVPT2, and AFQMC calculations with a focus on *ab initio* systems. Many parts of the code are under development, and documentation and examples for various methods will be added soon! 

## Compiling the DQMC binary 
The DQMC (determinantal QMC) part of the code can perform free projection and phaseless AFQMC (auxiliary field QMC). Compilation is similar to the [Dice](https://github.com/sanshar/Dice/) code. It has the following dependencies:
 
1. Boost (known to work with versions up to 1.70)
2. HDF5

The header-only [Eigen linear algebra library](https://gitlab.com/libeigen/eigen) is included in this repository. Intel or GCC (recommended) compilers can be used. The compiler as well as paths for dependencies are specified at the top of the Makefile:
```
USE_MPI = yes
USE_INTEL = no
ONLY_DQMC = no
HAS_AVX2 = yes

BOOST=${BOOST_ROOT}
HDF5=${CURC_HDF5_ROOT}
```

Note that the Boost libraries (mpi and serialization) have to be compiled with the same compiler used for compiling DQMC. More information about compiling Boost can be found in the [Dice documentation](https://sanshar.github.io/Dice/). Since we use MPI for parallelization, MPI compilers should be used (we have tested with Intel MPI). HAS_AVX2 should be set to no if your processor does not support avx2. After modifying the Makefile, DQMC can be compiled using
```
make bin/DQMC -j
```
Tests can be run using the runDQMC.sh script in the "tests" directory. Examples of phaseless AFQMC calculations are in the "examples" directory along with output files. The python scripts used in these examples require python 3.6 or newer. They also require pyscf and pandas packages. 


## Compiling other parts of the code

Other parts of the code have the following additional dependencies:

1. Libigl
2. Sparsehash
3. More Boost libraries (program_options, system, filesystem) 

Everything on the master branch can be compiled with
```
make -j
```
to generate various binaries for the other methods mentioned above. Tests for most of the code are in the "tests" directory and can be run with the runTests.sh script.
