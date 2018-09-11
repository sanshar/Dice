# Print test message
message(STATUS "\n-- Setting up tests ...\n-- ")

# Make Symmetry Library
include_directories(${EXTERNAL_INCLUDE_DIR}/Catch/)
include_directories(${DICE_SOURCE_DIR})

# Set compiler flags for gcov
set(CMAKE_CXX_FLAGS "-std=c++11 -g -O0 -w -fprofile-arcs -ftest-coverage -DSERIAL")
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1) # *.cpp.o -> *.o

# Gather all test sources in the unittests directory
file(GLOB TEST_SOURCES *.cpp)

# Iterate through each .cpp file found in unittests and create and executable
# and test.
foreach(file ${TEST_SOURCES})
  get_filename_component(file_basename ${file} NAME_WE)

  # Convert unit-file1.cpp to test-file1 (creates test name)
  string(REGEX REPLACE "unit-([^$]+)" "test-\\1" testcase ${file_basename})
  message(STATUS ${testcase})

  # Convert unit-file1.cpp to ../Dice/file.cpp (creates path to source file)
  string(REGEX REPLACE "unit-([^$]+)" "../Dice/\\1.cpp" src_file ${file_basename})
  message(STATUS "Found source file for ${testcase}: ${src_file}")

  # Add executable for test (including source file)
  # TODO currently build SHCIshm.cpp into all test (rather than keep a list of
  # files where it should included)
  add_executable(${testcase} ${file} ${src_file} ${DICE_SOURCE_DIR}/SHCIshm.cpp)

  # Add linker flags (in case Boost is used in test)
  IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_link_libraries(${testcase} -L/${Boost_LIBRARY_DIRS} -lboost_serialization -lboost_mpi)
  ELSE()
    target_link_libraries(${testcase} -L/${Boost_LIBRARY_DIRS} -lboost_serialization -lboost_mpi -lrt)
  ENDIF()

  # Add test so executable is picked up by CMake/CTest
  # Adding Catch2 specific flag "--success" which prints out more detailed test
  # results to /build/Testing/Temporary/LastTest.log
  add_test(NAME ${testcase} COMMAND ${testcase} --success)

endforeach()
