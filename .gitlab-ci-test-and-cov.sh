# Build unit tests
cd build
cmake ..
make -j all
make test

# Test coverage
mkdir coverage
cd coverage
gcov ../../Dice/symmetry.cpp --object-file ../unittests/CMakeFiles/test-symmetry.dir/__/Dice/ | grep -B 1 "symmetry.cpp.gcov"
gcov ../../Dice/Determinants.cpp --object-file ../unittests/CMakeFiles/test-Determinants.dir/__/Dice/ | grep -B 1 "Determinants.cpp.gcov"

# Functional Tests
cd ../../tests/o2_omp1_det
../../build/Dice
mpirun -n 2 --allow-run-as-root ../../build/Dice
