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
set(CMAKE_CXX_FLAGS "-g -O3")

IF ( CMAKE_BUILD_TYPE MATCHES Debug )
  message(WARNING "Using debug mode so no optimization will be used.")
  set( CMAKE_VERBOSE_MAKEFILE on )
ENDIF()

# Get OpenMP Flag
find_package(OpenMP)



###################
### Dependecies ###
###################
message(STATUS "\n-- Finding dependencies ...\n-- ")

# For some external project macros
include(ExternalProject)

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


### Catch2 #####################################################################
if (NOT EXISTS "${EXTERNAL_DIR}/Catch2/catch2/catch.hpp")
  file(DOWNLOAD
    "https://github.com/catchorg/Catch2/releases/download/v2.4.0/catch.hpp"
    "${EXTERNAL_DIR}/Catch2/catch2/catch.hpp")
endif()
set(CATCH2_INCLUDE_DIR ${EXTERNAL_DIR}/Catch2/catch2)
add_library(Catch2 INTERFACE)
target_include_directories(Catch2 INTERFACE ${CATCH2_INCLUDE_DIR})



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
add_dependencies(Dice eigen_project)
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
