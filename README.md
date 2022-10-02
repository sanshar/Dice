# Dice
<div><img src="https://github.com/sanshar/Dice/blob/master/docs/images/dice_lateral.png" height="360px"/></div>

*Dice* contains code for performing SHCI, VMC, GFMC, DMC, FCIQMC, stochastic MRCI and SC-NEVPT2, and AFQMC calculations with a focus on *ab initio* systems. Many parts of the code are under development, and documentation and examples for various methods will be added soon! 

Prerequisites
------------

*Dice* requires:

* [Boost](http://www.boost.org/) (when compiling the Boost library make sure that you use the same compiler as you do for *Dice*)

An example of download and compilation commands for the `NN` version of Boost can be:

```
  wget https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_NN_0.tar.gz
  tar -xf boost_1_NN_0.tar.gz
  cd boost_1_NN_0
  ./bootstrap.sh
  echo "using mpi ;" >> project-config.jam
  ./b2 -j6 --target=shared,static
```

* [HDF5](http://www.hdfgroup.org/downloads/hdf5)


The header-only [Eigen linear algebra library](https://gitlab.com/libeigen/eigen) is included in this repository. 

* About compiler requirements:
    - GNU: g++ 6.1 or newer (recommended)
    - Intel: icpc 20.0 or newer
    - In any case: the C++0x/C++11 standards must be supported.

The compiler as well as paths for dependencies are specified at the top of the Makefile:

```
USE_INTEL = no
HAS_AVX2 = yes

BOOST=${BOOST_ROOT}
EIGEN=./eigen/
HDF5=${CURC_HDF5_ROOT}
MKL=${MKLROOT}

INCLUDE_MKL = -I$(MKL)/include
LIB_MKL = -L$(MKL)/lib/intel64/ -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core

INCLUDE_BOOST = -I$(BOOST)/include -I$(BOOST)
LIB_BOOST = -L$(BOOST)/lib -L$(BOOST)/stage/lib

INCLUDE_HDF5 = -I$(HDF5)/include
LIB_HDF5 = -L$(HDF5)/lib -lhdf5
```

Note that the Boost libraries have to be compiled with the same compiler used for compiling Dice. More information about compiling Boost can be found in the [SHCI documentation](https://sanshar.github.io/Dice/). Since we use MPI for parallelization, MPI compilers should be used (we have tested with Intel MPI). After modifying the Makefile, everything can be compiled using

```
make -j
```

To compile only a part of the code, say DQMC, one can use

```
make DQMC -j
```

Testing
-------

Tests can be run using the runTests.sh script in the "tests" directory. SHCI tests are in the SHCI subdirectory at the moment. Again for testing specific parts of the code, say DQMC, one can use the runDQMC.sh script instead. Examples are in the "examples" directory along with output files. The python scripts used in these examples require python 3.6 or newer. They also require pyscf and pandas packages. 


