# Install pytest
python -m pip install -U pytest

# Build unit tests
mkdir build
cd build
export CXX=mpicxx
cmake .. -DENABLE_TESTS=On
make Dice
make unit-Determinants
make unit-integral
make unit-symmetry
make test

# Test coverage
mkdir coverage
cd coverage
gcov ../../Dice/symmetry.cpp --object-file ../unittests/CMakeFiles/unit-symmetry.dir/__/Dice/ | grep -B 1 "symmetry.cpp.gcov"
gcov ../../Dice/Determinants.cpp --object-file ../unittests/CMakeFiles/unit-Determinants.dir/__/Dice/ | grep -B 1 "Determinants.cpp.gcov"
gcov ../../Dice/integral.cpp --object-file ../unittests/CMakeFiles/unit-integral.dir/__/Dice/ | grep -B 1 "integral.cpp.gcov"

# Functional Tests
cd ../../tests/
pytest -v test_serial.py
