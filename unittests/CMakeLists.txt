# Make Symmetry Library
include_directories(${EXTERNAL_INCLUDE_DIR}/Catch/)
include_directories(${DICE_SOURCE_DIR})
add_library(symmetry SHARED ${DICE_SOURCE_DIR}/symmetry.cpp ${DICE_SOURCE_DIR}/symmetry.h)
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  target_link_libraries(symmetry -L/${Boost_LIBRARY_DIRS} -lboost_serialization -lboost_mpi)
ELSE()
  target_link_libraries(symmetry -L/${Boost_LIBRARY_DIRS} -lboost_serialization -lboost_mpi -lrt)
ENDIF()

add_executable(unit1 unit-symmetry.cpp)
# target_link_libraries(unit1 dice -L/${Boost_LIBRARY_DIRS} -lboost_serialization -lboost_mpi -lrt)
target_link_libraries(unit1 symmetry)
add_test(NAME Symmetry COMMAND unit1 --success)
