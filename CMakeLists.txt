cmake_minimum_required (VERSION 3.0)
message(STATUS "\n-- Finding compilers ...\n-- ")

###############
### Project ###
###############

project (Dice CXX)
set (EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-g -O3")
option(SERIAL "SERIAL" OFF)
option(COMPLEX "COMPLEX" OFF)
option(ENABLE_TESTS "Enable unit tests." OFF)


#####################
### Version Flags ###
#####################
message(STATUS "\n-- Finding git version ...\n-- ")

# Get the current working branch
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the latest abbreviated commit hash of the working branch
execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "GIT BRANCH: ${GIT_BRANCH}")
message(STATUS "GIT COMMIT: ${GIT_COMMIT_HASH}")

# Add them to definition flags
add_definitions(-Dgit_branch="${GIT_BRANCH}")
add_definitions(-Dgit_commit="${GIT_COMMIT_HASH}")

###################
### Dependecies ###
###################

message(STATUS "\n-- Finding dependencies ...\n-- ")

# For some external project macros
include(ExternalProject)



### OpenMP #####################################################################
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()



### Boost ######################################################################
set( BOOST_ROOT "$ENV{BOOST_ROOT}" )
IF( NOT BOOST_ROOT )
  message(STATUS
  "ENV variable BOOST_ROOT not found. Using find_package to locate Boost.")
ENDIF()
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
message( STATUS "Using Boost inlcude directory: ${Boost_INCLUDE_DIRS}")
message( STATUS "Using Boost library directory: ${Boost_LIBRARY_DIRS}")



### Eigen ######################################################################
ExternalProject_Add(
  eigen_project
  SOURCE_DIR "${EXTERNAL_DIR}/eigen"
  URL "https://github.com/eigenteam/eigen-git-mirror/archive/3.3.5.tar.gz"
  URL_HASH MD5=305eed19157335016bde9b4d57ee3d35
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)
ExternalProject_Get_Property(eigen_project SOURCE_DIR)
set( EIGEN3_INCLUDE_DIR ${EXTERNAL_DIR}/eigen )
include_directories(${EIGEN3_INCLUDE_DIR})
message( STATUS "Using Eigen include directory: ${EIGEN3_INCLUDE_DIR}")



### Doctest #####################################################################
if (NOT EXISTS "${EXTERNAL_DIR}/doctest/doctest.h")
  file(DOWNLOAD
    "https://github.com/onqtam/doctest/blob/master/doctest/doctest.h"
    "${EXTERNAL_DIR}/doctest/doctest.h")
endif()
set(DOCTEST_INCLUDE_DIR ${EXTERNAL_DIR}/doctest)
include_directories(${DOCTEST_INCLUDE_DIR})


###############
### Sources ###
###############

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(DICE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/) # TODO
set(DICE_SOURCES
  ${DICE_SOURCE_DIR}/SHCI.cpp
  ${DICE_SOURCE_DIR}/SHCIbasics.cpp
  ${DICE_SOURCE_DIR}/Determinants.cpp
  ${DICE_SOURCE_DIR}/integral.cpp
  ${DICE_SOURCE_DIR}/input.cpp
  ${DICE_SOURCE_DIR}/Davidson.cpp
  ${DICE_SOURCE_DIR}/SHCIgetdeterminants.cpp
  ${DICE_SOURCE_DIR}/SHCIsampledeterminants.cpp
  ${DICE_SOURCE_DIR}/SHCIrdm.cpp
  ${DICE_SOURCE_DIR}/SHCISortMpiUtils.cpp
  ${DICE_SOURCE_DIR}/SHCImakeHamiltonian.cpp
  ${DICE_SOURCE_DIR}/SHCIshm.cpp
  ${DICE_SOURCE_DIR}/LCC.cpp # TODO problem with Serial
  ${DICE_SOURCE_DIR}/symmetry.cpp
  ${DICE_SOURCE_DIR}/staticVariables.cpp
)

IF(${COMPLEX})
  set(ZDICE_SOURCES
      ${DICE_SOURCES}
      ${DICE_SOURCE_DIR}/SOChelper.cpp
      ${DICE_SOURCE_DIR}/new_anglib.cpp
  )
ENDIF()



#################################
### Compilitation and Linking ###
#################################
message(STATUS "Compile Flags: ${CMAKE_CXX_FLAGS}")

# Executable
add_executable(Dice ${DICE_SOURCES})
add_dependencies(Dice eigen_project)

IF(${COMPLEX})
  add_executable(ZDice ${ZDICE_SOURCES})
  add_dependencies(ZDice eigen_project)
ENDIF()


# Add compile definitions
IF(${SERIAL})
  set_target_properties(Dice PROPERTIES COMPILE_FLAGS "-DSERIAL")
ENDIF()

IF(${COMPLEX})
  set_target_properties(ZDice PROPERTIES COMPILE_FLAGS "-DComplex")
  message(STATUS "Building ZDice for complex numbers!")
ENDIF()

# Do not include -lrt on Mac OSX builds
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  target_link_libraries(Dice -L/${Boost_LIBRARY_DIRS} -lboost_serialization -lboost_mpi)
  IF(${COMPLEX})
    target_link_libraries(ZDice -L/${Boost_LIBRARY_DIRS} -lboost_serialization -lboost_mpi)
  ENDIF()
ELSE() # Linux Systems
  target_link_libraries(Dice -L/${Boost_LIBRARY_DIRS} -lboost_serialization -lboost_mpi -lrt)
  IF(${COMPLEX})
    target_link_libraries(ZDice -L/${Boost_LIBRARY_DIRS} -lboost_serialization -lboost_mpi -lrt)
  ENDIF()
ENDIF()



###############
### Testing ###
###############

include(CTest)
enable_testing()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(doctest)

# Adding individual tests takes three steps:
# 1) Compile with all source files
# 2) Add eigen as dependency
# 3) Link against boost and other libs
add_executable(test_determinants ${DICE_SOURCE_DIR}/unittests/test_determinants.cpp ${DICE_SOURCE_DIR}/Determinants.cpp ${DICE_SOURCE_DIR}/staticVariables.cpp ${DICE_SOURCE_DIR}/integral.cpp)
# add_dependencies(test_determinants eigen_project)
target_link_libraries(test_determinants -L/${Boost_LIBRARY_DIRS} -lboost_serialization -lboost_mpi -lrt)


doctest_discover_tests(test_determinants)