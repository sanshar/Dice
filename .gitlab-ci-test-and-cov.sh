# Build unit tests
cd build
make unit1
make test

# Test coverage
mkdir coverage
cd coverage
 gcov ../../Dice/symmetry.cpp --object-file ../unittests/CMakeFiles/unit1.dir/__/Dice/ | grep -B 1 "symmetry.cpp.gcov"

# Functional Tests
cd ../../tests/o2_omp1_det
../../build/Dice
mpirun -n 2 --allow-run-as-root ../../build/Dice
