# Unit Testing

(This README is a draft)

## With CMake/CTest

Currently Dice uses Catch2 and CMake/CTest to handle unit tests.

```bash
$ pwd
/home/user/apps/Dice
$ cd build
$ cmake ..
$ make -j Dice
$ make unit1
$ make test
Running tests...
Test project /home/jets/apps/test/Dice/build
    Start 1: Symmetry
1/1 Test #1: Symmetry .........................   Passed    0.01 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) =   0.01 sec
```

To view a more detailed breakdown of the tests go to `$(DICE_PATH)/build/Testing/Temporary/LastTest.log`.

## Without CMake/CTest

If you aren't a big fan of CMake, you can compile the unit tests without CMake as shown below:

```bash
$ pwd
/home/user/apps/Dice
$ cd unittests
$ mpicxx -std=c++11 -g -O3 -I$(DICE_PATH)/External/Catch -I$(BOOST_PATH) -I$(EIGEN_PATH) -L$(BOOST_LIB_PATH) -o unit-symmetry unit-symmetry.cpp $(DICE_PATH)/build/CMakeFiles/Dice.dir/Dice/symmetry.cpp.o -lboost_mpi -lboost_serialization  && ./unit-symmetry --success
```
