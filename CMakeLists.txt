cmake_minimum_required (VERSION 3.0)

################
### Projects ###
################

project (Dice)


######################
### Compiler Flags ###
######################

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "-g")

IF ( CMAKE_BUILD_TYPE MATCHES Debug )
  message(WARNING "Using debug mode so no optimization will be used.")
  set( CMAKE_VERBOSE_MAKEFILE on )
ELSE()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
ENDIF()


###################
### Dependecies ###
###################

# Get MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# Get Boost
set( BOOST_ROOT "$ENV{BOOST_ROOT}" )
IF( NOT BOOST_ROOT )
  message(STATUS "ENV variable BOOST_ROOT not found. Using find_package to locate Boost.")
ENDIF()
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
message( STATUS ${Boost_INCLUDE_DIRS})
message( STATUS ${Boost_LIBRARY_DIRS})

# Get Eigen
SET( EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}" )
IF( NOT EIGEN3_INCLUDE_DIR )
  message(STATUS "ENV variable EIGEN3_INCLUDE_DIR not found. Using find_package to locate Eigen.")
  find_package(Eigen3 REQUIRED)
ENDIF()
include_directories(${EIGEN3_INCLUDE_DIR})
message( STATUS ${EIGEN3_INCLUDE_DIR})


###############
### Sources ###
###############

set(DICE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Dice/) # TODO
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
  ${DICE_SOURCE_DIR}/LCC.cpp
  ${DICE_SOURCE_DIR}/symmetry.cpp
)

set(ZDICE2_SOURCES ${DICE_SOURCES}
  ${DICE_SOURCE_DIR}/SOChelper.cpp
  ${DICE_SOURCE_DIR}/new_anglib.cpp
)


#####################
### Compilitation ###
#####################

add_executable(Dice ${DICE_SOURCES})
add_executable(ZDice2 ${ZDICE2_SOURCES})

###############
### Linking ###
###############

# Do not include -lrt on Mac OSX builds
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  target_link_libraries(Dice -L/${Boost_LIBRARY_DIRS} -lboost_serialization -lboost_mpi)
  target_link_libraries(ZDice2 -L/${Boost_LIBRARY_DIRS} -lboost_serialization -lboost_mpi)
ELSE()
  target_link_libraries(Dice -L/${Boost_LIBRARY_DIRS} -lboost_serialization -lboost_mpi -lrt)
  target_link_libraries(ZDice2 -L/${Boost_LIBRARY_DIRS} -lboost_serialization -lboost_mpi -lrt)
ENDIF()
