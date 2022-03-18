# QMC methods for *ab inito* electronic structure theory

This repository contains code for performing VMC, GFMC, DMC, FCIQMC, stochastic MRCI and SC-NEVPT2, and AFQMC calculations with a focus on *ab initio* systems. Many parts of the code are under development, and documentation and examples for various methods will be added soon! 

## Compiling the DQMC binary 
The DQMC (determinantal QMC) part of the code can perform free projection and phaseless AFQMC (auxiliary field QMC). Compilation is similar to the [Dice](https://github.com/sanshar/Dice/) code. It has following dependencies:

1. Eigen 
2. Boost (known to work with versions up to 1.70)
3. HDF5

Intel or GCC compilers can be used. The compiler as well as paths for dependencies are specified at the top of the Makefile:
```
USE_MPI = yes
USE_INTEL = yes
ONLY_DQMC = no
USE_AVX2 = yes

EIGEN=/projects/ilsa8974/apps/eigen/
BOOST=/projects/anma2640/boost_1_66_0/
HDF5=/curc/sw/hdf5/1.10.1/impi/17.3/intel/17.4/
```

Note that the Boost libraries (mpi and serializtion) have to be compiled with the same compiler used for compiling DQMC. More information about compiling Boost can be found in the [Dice documentation](https://sanshar.github.io/Dice/). Since we use MPI for parallelization, MPI compilers should be used. HAS_AVX2 should be set to no if your processor does not support avx2. After modifying the Makefile, DQMC can be compiled using
```
make bin/DQMC -j
```
 
There are two phaseless AFQMC examples in the "examples" directory along with output files that can be used to make sure that the code is working correctly. The python scripts used in these examples require python 3.6 or newer. They also require pyscf and pandas packages. Tests to be added soon!


## Compiling other parts of the code

Other parts of the code have following additional dependencies:

1. Libigl
2. Sparsehash
3. More Boost libraries (program_options, system, filesystem) 

Everything on the master branch can be compiled with
```
make -j
```
to generate various binaries for the other methods mentioned above. Tests for most of the code are in the "test" directory and can be run with the runTests.sh script.
