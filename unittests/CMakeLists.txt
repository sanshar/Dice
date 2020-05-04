list(APPEND CMAKE_MODULE_PATH "${PROJ_DIR}/cmake")
include(doctest)

# Adding individual tests (for devs) takes two steps:
# 1) Add `add_executable()` for test along with all sources
# 2) Add the name of the test executable to ${TEST_LIST}

# Under the hood this does three things:
# A) Compiles the tests
# B) Links the test to the proper libraries
# C) Adds the tests so running `make test` will include the test

#
# Step 1)
#
add_executable(test_determinants 
    ${DICE_SOURCE_DIR}/unittests/test_determinants.cpp
    ${DICE_SOURCE_DIR}/Determinants.cpp
    ${DICE_SOURCE_DIR}/staticVariables.cpp 
    ${DICE_SOURCE_DIR}/integral.cpp
    )

add_executable(test_halfdeterminants 
    ${DICE_SOURCE_DIR}/unittests/test_halfdeterminants.cpp
    ${DICE_SOURCE_DIR}/Determinants.cpp
    ${DICE_SOURCE_DIR}/staticVariables.cpp 
    ${DICE_SOURCE_DIR}/integral.cpp
    )

#
# Step 2)
#

# This is used to link tests so make sure you update it when you add tests!
set(TEST_LIST test_determinants test_halfdeterminants)

#
# DO NOT EDIT (Steps B and C)
#

# Link all tests
foreach(X IN LISTS TEST_LIST)
    target_link_libraries(${X} -L/${Boost_LIBRARY_DIRS} -lboost_serialization -lboost_mpi -lrt)    
    doctest_discover_tests(${X})
endforeach()


# Create shortcut to compile all tests at once
# Need CMake 3.15
# See https://stackoverflow.com/a/56490428/9371695
# ADD_CUSTOM_TARGET(all_tests
#     COMMAND ${CMAKE_COMMAND} --build . --target "${TEST_LIST}"
# )