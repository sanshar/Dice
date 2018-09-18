cmake_minimum_required (VERSION 3.0)
message(STATUS "\n-- Finding compilers ...\n-- ")

###############
### Project ###
###############

project (Dice)
set (EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/External)

######################
### Compiler Flags ###
######################

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-std=c++11 -g -O3")

IF ( CMAKE_BUILD_TYPE MATCHES Debug )
  message(WARNING "Using debug mode so no optimization will be used.")
  set( CMAKE_VERBOSE_MAKEFILE on )
ENDIF()


###################
### Dependecies ###
###################
message(STATUS "\n-- Finding dependencies ...\n-- ")

# Get MPI
# find_package(MPI REQUIRED)
# include_directories(${MPI_INCLUDE_PATH})

# Get Boost
# set( BOOST_ROOT "$ENV{BOOST_ROOT}" )
# IF( NOT BOOST_ROOT )
#   message(STATUS "ENV variable BOOST_ROOT not found. Using find_package to locate Boost.")
# ENDIF()
# find_package(Boost REQUIRED)

# For some external project macros
include(ExternalProject)

# Download boost from git and build regex module
set( Boost_url "http://sourceforge.net/projects/boost/files/boost/1.59.0/boost_1_59_0.tar.gz")
set( Boost_md5 51528a0e3b33d9e10aaa311d9eb451e3 )
set( Boost_Bootstrap_Command ./bootstrap.sh )
set( Boost_b2_Command ./b2 )

ExternalProject_Add(boost
  # SOURCE_DIR "${CMAKE_BINARY_DIR}/External/boost"
  BUILD_IN_SOURCE 1
  URL ${Boost_url}
  URL_MD5 ${Boost_md5}
  UPDATE_COMMAND ${Boost_Bootstrap_Command} --prefix=${EXTERNAL_DIR}/boost/lib --with-libraries=serialization,mpi
  CONFIGURE_COMMAND echo "using mpi ;\n" >> project-config.jam
  BUILD_COMMAND ${Boost_b2_Command} install --prefix=${EXTERNAL_DIR}/boost --with-serialization --with-mpi
  INSTALL_COMMAND ""
)
set(BOOST_ROOT ${EXTERNAL_DIR}/boost)
set( Boost_INCLUDE_DIRS ${EXTERNAL_DIR}/boost )
set( Boost_LIBRARY_DIRS ${EXTERNAL_DIR}/boost/lib )
# find_package(Boost REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})
message( STATUS "Using Boost inlcude directory: ${Boost_INCLUDE_DIRS}")
message( STATUS "Using Boost library directory: ${Boost_LIBRARY_DIRS}")

# Get Eigen
SET( EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}" )
IF( NOT EIGEN3_INCLUDE_DIR )
  message(STATUS "ENV variable EIGEN3_INCLUDE_DIR not found. Using attached version of Eigen.")
  SET( EIGEN3_INCLUDE_DIR ${EXTERNAL_DIR}/eigen )
ENDIF()
include_directories(${EIGEN3_INCLUDE_DIR})
message( STATUS "Using Eigen include directory: ${EIGEN3_INCLUDE_DIR}")

# Get OpenMP Flag
find_package(OpenMP)

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

# add_library(dice_lib SHARED ${DICE_LIB_SOURCES})

# set(ZDICE2_SOURCES ${DICE_SOURCES}
#   ${DICE_SOURCE_DIR}/SOChelper.cpp
#   ${DICE_SOURCE_DIR}/new_anglib.cpp
# )


#####################
### Compilitation ###
#####################

add_executable(Dice ${DICE_SOURCES})
add_dependencies(Dice boost)
# add_executable(DiceSerial ${DICE_SOURCES})
# set_target_properties(DiceSerial PROPERTIES COMPILE_FLAGS "-DSERIAL")
# add_executable(ZDice2 ${ZDICE2_SOURCES})


###############
### Linking ###
###############

# Do not include -lrt on Mac OSX builds
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  target_link_libraries(Dice -L/${Boost_LIBRARY_DIRS} -lboost_serialization -lboost_mpi)
  # target_link_libraries(ZDice2 -L/${Boost_LIBRARY_DIRS} -lboost_serialization -lboost_mpi)
ELSE()
  target_link_libraries(Dice -L/${Boost_LIBRARY_DIRS} -lboost_serialization -lboost_mpi -lrt)
  # target_link_libraries(ZDice2 -L/${Boost_LIBRARY_DIRS} -lboost_serialization -lboost_mpi -lrt)
ENDIF()

###############
### Testing ###
###############

include(CTest) #adds option BUILD_TESTING (default ON)
enable_testing()
set(UNIT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/unittests)
add_subdirectory(${UNIT_DIR})
message(STATUS ${UNIT_DIR})
