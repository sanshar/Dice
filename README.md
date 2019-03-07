[![Codacy Badge](https://api.codacy.com/project/badge/Grade/49d4e0485f3448da95b7ae0711bbcf2a)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=sanshar/Dice&amp;utm_campaign=Badge_Grade)
[![CodeFactor](https://www.codefactor.io/repository/github/sanshar/dice/badge/cmake)](https://www.codefactor.io/repository/github/sanshar/dice/overview/cmake)

<div><img src="https://github.com/sanshar/Dice/blob/master/docs/images/dice_lateral.png" height="360px"/></div>

*Dice* implements the semistochastic heat bath configuration interaction (SHCI) algorithm for *ab initio* Hamiltonians of quantum chemical systems.

Unlike full configuration interaction (FCI), SHCI can be used to treat active spaces containing 30 to 100 orbitals.
SHCI is able to accomplish this by taking advantage of the fact that although the full Hilbert space may be enormous,
only a small fraction of the determinants in the space have appreciable coefficients.

Compared to other methods in its class, SHCI is often not only orders of magnitude faster,
it also does not suffer from serious memory bottlenecks that plagues these methods.
The resulting algorithm as implemented in *Dice* allows us to treat difficult benchmark systems
such as the Chromium dimer and Mn-Salen (a challenging bioinorganic cluster) at a cost that is often
an order of magnitude faster than density matrix renormalization group (DMRG) or full configuration interaction quantum Monte Carlo (FCIQMC).

Thus if you are interested in performing multireference calculations with active space containing several tens to hundreds of orbitals,
*Dice* might be an ideal choice for you.


* *Dice* is available with the [PySCF](https://github.com/sunqm/pyscf/blob/master/README.md) package.

* The latest version of *Dice* is also downloadable as a tar archive: [Dice.tar.gz](images/Dice.tar.gz)

Prerequisites
------------

*Dice* requires:

* [Boost](http://www.boost.org/) (when compiling the Boost library make sure that you use the same compiler as you do for *Dice*)

An example of download and compilation commands for the `NN` version of Boost can be:

```bash
wget https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_NN_0.tar.gz
tar -xf boost_1_NN_0.tar.gz
cd boost_1_NN_0
./bootstrap.sh
echo "using mpi ;" >> project-config.jam
./b2 -j6 --target=shared,static
```

**NOTE** | Set the ENV variable `BOOST_ROOT` to the path where the boost `include` and `lib` directories are installed.
-|-

* [CMake]() (Build configuration system)

CMake is used to control the software compilation process using simple platform and compiler independent configuration files, and generate native makefiles and workspaces that can be used in the compiler environment of your choice.

You can download it in a variety of ways, we think that the package manager `pip` is the easiest:

```bash
pip install cmake
```

<!-- * [Eigen](http://eigen.tuxfamily.org/dox/) (Eigen consists of header files and does not have to be compiled but can be installed)
This is automatically downloaded during the installation of *Dice*. -->


* Compiler requirements:
    - GNU: g++ 4.8 or newer
    - Intel: icpc 14.0.1 or newer
    - The compiler must support the C++14 standard.
    - The compiler must support OpenMP


Compilation
-------

```bash
export BOOST_ROOT=<path_to_boost>
cd build
cmake ..
make -j
```


Testing
-------

To build the unit-tests you can add the CMake command line option `-DENABLE_TESTS=ON`. The modified build procedure:

```bash
export BOOST_ROOT=<path_to_boost>
cd build
cmake .. -DENABLE_TESTS=ON
make -j
make test
```

There are two sets of smoke tests (tests that evaluate the final result of a calculation): first, the serial tests (`Dice/tests/test_serial.py`) is a short series of tests that are small enough to be run on any machine; second, the mpi tests (`Dice/tests/test_mpi.py`) are larger and might not be able to run on every configuration.

We suggest running the unit-tests with `pytest`, which can be installed using [`pip`](https://docs.pytest.org/en/latest/getting-started.html#install-pytest) or [`conda`](https://anaconda.org/anaconda/pytest).

To run the tests:

```bash
cd tests
pytest -v test_serial.py
pytest -v test_mpi.py
```
