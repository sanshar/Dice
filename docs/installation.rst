Installation
************
Source Code
-----------
Available with the <a href="https://github.com/sunqm/pyscf/blob/master/README.md">PySCF</a> package.
Download the latest version of SHCI: ???LINK???

Compile
-------
SHCI requires `Boost <http://www.boost.org/>`_ and `Eigen <http://eigen.tuxfamily.org/dox/>`_ libraries. When compiling the Boost library make sure that you use the same compiler as you do for SHCI. Eigen consists of header files and does not have to be compiled. When choosing your compiler, either GNU or Intel, C++0x/C++11 standards must be supported.

* GNU g++ 4.8 or newer required.
* Intel icpc 14.0.1 or newer required.

Before you can compile SHCI you must edit the Makefile in the main directory and change the paths in the FLAGS, DFLAGS, and LFLAGS variables to point to for your Eigen and Boost libraries.
                        
Finally, the user must edit the lines shown below before compiling:

::

  CXX = mpiicpc
  CC = mpiicpc
  FLAGS = -std=c++11 -qopenmp -O2 -I/PATH_TO/eigen -I/PATH_TO/boost_1_57_0/
  LFLAGS = -L/PATH_TO/boost_1_57_0_lib -lboost_serialization -lboost_mpi
