# Make Symmetry Library
include_directories(${EXTERNAL_INCLUDE_DIR}/Catch/)
include_directories(${DICE_SOURCE_DIR})

# Set compiler flags for gcov
set(CMAKE_CXX_FLAGS "-g -O0 -Wall -fprofile-arcs -ftest-coverage")
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1) # *.cpp.o -> *.o

# Compile and link
add_executable(unit1 unit-symmetry.cpp ../Dice/symmetry.cpp)
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  target_link_libraries(unit1 -L/${Boost_LIBRARY_DIRS} -lboost_serialization -lboost_mpi)
ELSE()
  target_link_libraries(unit1 -L/${Boost_LIBRARY_DIRS} -lboost_serialization -lboost_mpi -lrt)
ENDIF()
add_test(NAME Symmetry COMMAND unit1 --success)

# $ pwd
# build
# $ mkdir coverage
# $ cd coverage
# $ gcov ../../Dice/symmetry.cpp --object-file ../unittests/CMakeFiles/unit1.dir/__/Dice/ | grep -B 1 "symmetry.cpp.gcov"
# Lines executed:49.54% of 216
# /Users/jets/apps/Dice/Dice/symmetry.cpp:creating 'symmetry.cpp.gcov'
