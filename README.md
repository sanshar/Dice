<div><img src="https://github.com/sanshar/Dice/docs/images/dice_orange.png" height="80px"/></div>

DICE software
=============

*Dice* implements the semistochastic heat bath configuration interaction (SHCI) algorithm for *ab initio* Hamiltonian of a quantum chemical system. Unlike full configuration interaction (FCI), SHCI can be used to treat active spaces containing 30-100 orbitals. SHCI is able to accomplish this by taking advantage of the fact that although the full Hilbert space may be enormous, only a small fraction of the determinants in the space have appreciable coefficients. Compared to other methods in its class SHCI is often not only orders of magnitude faster, it also does not suffer from a serious memory bottleneck that plauges such methods. The resulting algorithm as implemented in Dice allows us to treat difficult benchmark systems such as the Chromium dimer and Mn-Salen (a challenging bioinorganic cluster) at a cost that is often an order of magnitude faster than either density matrix renormalization group (DMRG) or full configuration interaction quantum Monte Carlo (FCIQMC). Thus if you are interested in performing multireference calculations with active space containing several tens to hundreds of orbitals, SHCI might be an ideal choice for you.

* Available with the `PySCF <https://github.com/sunqm/pyscf/blob/master/README.md>`_ package.

* Download the latest version of Dice: :download:`Dice.tar.gz <images/Dice.tar.gz>`

Prerequisites
------------

SHCI requires:

* `Boost <http://www.boost.org/>`_ (when compiling the Boost library make sure that you use the same compiler as you do for SHCI)

* `Eigen <http://eigen.tuxfamily.org/dox/>`_ (Eigen consists of header files and does not have to be compiled)

* About compilers:
 GNU: g++ 4.8 or newer

 Intel: icpc 14.0.1 or newer

 In any case: the C++0x/C++11 standards must be supported.

Compilation
-------

Edit the `Makefile` in the main directory and change the paths to your Eigen and Boost libraries. The user can choose whether to use gcc or intel by setting the `USE_INTEL` variable accordingly, and whether or not to compile with MPI by setting the `USE_MPI` variable. All the lines in the `Makefile` that need to be edited are shown below:

```
  USE_MPI = yes
  USE_INTEL = yes
  EIGEN=/path_to/eigen
  BOOST=/path_to/boost_1_NN_0
```


Testing
-------
One can test the code using the `runTests.sh` script in `/path_to/Dice/tests/`:

```
  cd /path_to/Dice/tests/
  ./runTests.sh
```

Before running this script, edit the `MPICOMMAND` variable to the appropriate number of processors you wish to run the tests onto.

Note:

  If your system has limited memory or slow processing power, you may wish to comment out the tests for Mn(salen) in the runTests.sh script because they require a large amount of processing power and memory.

#Citing DICE
#-----------
#
#This paper should be cited for calculations performed using the DICE software: #paper
#
