# Make Symmetry Library
include_directories(${EXTERNAL_INCLUDE_DIR}/Catch/)
add_library(symmetry-lib STATIC ${DICE_SOURCE_DIR}/symmetry.cpp ${DICE_SOURCE_DIR}/symmetry.h)
target_link_libraries(symmetry-lib -L/${Boost_LIBRARY_DIRS} -lboost_serialization -lboost_mpi -lrt)

add_executable(unit1 unit-symmetry.cpp)
target_link_libraries(unit1 symmetry-lib)
add_test(NAME tests COMMAND unit1)
