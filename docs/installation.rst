Installation
************
Source Code
-----------
* Available with the `PySCF <https://github.com/sunqm/pyscf/blob/master/README.md>`_ package.

* Download the latest version of Dice: :download:`Dice.tar.gz <images/Dice.tar.gz>`

Compile
-------
SHCI requires `Boost <http://www.boost.org/>`_ and `Eigen <http://eigen.tuxfamily.org/dox/>`_ libraries. When compiling the Boost library make sure that you use the same compiler as you do for SHCI. Eigen consists of header files and does not have to be compiled. When choosing your compiler, either GNU or Intel, C++0x/C++11 standards must be supported.

* GNU g++ 4.8 or newer required.
* Intel icpc 14.0.1 or newer required.

Before you can compile SHCI you must edit the Makefile in the main directory and change the paths in the FLAGS, DFLAGS, and LFLAGS variables to point to for your Eigen and Boost libraries.

Finally, the user must edit the lines shown below before compiling:

::

  USE_MPI = yes
  USE_INTEL = yes

  FLAGS = -std=c++11 -g -O3 -I/path_to/eigen -I/home/path_to/boost_1_57_0/
  DFLAGS = -std=c++11 -g -O3 -I/path_to/apps/eigen -I/path_to/boost_1_57_0/ -DComplex


Testing
-------
Once compiling is complete you should test the code using the "/path_to/Dice/tests/runTests.sh" script. Before running this script, edit the MPICOMMAND variable to show the appropriate number of processors you wish to run in parallel.

.. note::

  If your system has limited memory or slow processing power, you may wish to comment out the tests for Mn(salen) in the runTests.sh script because they require a large amount of processing power and memory.
