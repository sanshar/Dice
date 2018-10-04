# Build unit tests
cd build
export CXX=mpicxx
cmake ..
make Dice
make test-Determinants
make test-integral
make test-symmetry
make test

# Test coverage
mkdir coverage
cd coverage
gcov ../../Dice/symmetry.cpp --object-file ../unittests/CMakeFiles/test-symmetry.dir/__/Dice/ | grep -B 1 "symmetry.cpp.gcov"
gcov ../../Dice/Determinants.cpp --object-file ../unittests/CMakeFiles/test-Determinants.dir/__/Dice/ | grep -B 1 "Determinants.cpp.gcov"
gcov ../../Dice/integral.cpp --object-file ../unittests/CMakeFiles/test-integral.dir/__/Dice/ | grep -B 1 "integral.cpp.gcov"

# Functional Tests
# cd ../../tests/
# ./runCITests.sh
