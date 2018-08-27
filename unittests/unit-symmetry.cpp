// Unit tests for the symmetry class

// Let Catch provide main():
#define CATCH_CONFIG_MAIN
#include "../Dice/symmetry.h"
#include <catch.hpp>
#include <string>

TEST_CASE("All products of the same irrep should be Ag (1)", "[symmetry]") {
  symmetry sym((std::string) "d2h");

  for (int i = 1; i < 9; i++) {
    REQUIRE(sym.getProduct(i, i));
  }
}

/*
mpicxx -std=c++11 -g -O3 -I/Users/jets/apps/Dice/External/Catch
-I/Users/jets/apps/boost_1_67_0/ -I/Users/jets/apps/eigen
-L/Users/jets/apps/boost_1_67_0/stage/lib -o unit-symmetry unit-symmetry.cpp
/Users/jets/apps/Dice/build/CMakeFiles/Dice.dir/Dice/symmetry.cpp.o -lboost_mpi
-lboost_serialization  && ./unit-symmetry --success

mpicxx -std=c++11 -g -O3 -I/home/jets/apps/test/Dice/External/Catch
-I/home/jets/apps/boost_1_67_0/ -I/home/jets/apps/eigen
-L/home/jets/apps/boost_1_67_0/stage/lib -o unit-symmetry unit-symmetry.cpp
/home/jets/apps/test/Dice/build/CMakeFiles/Dice.dir/Dice/symmetry.cpp.o
-lboost_mpi -lboost_serialization  && ./unit-symmetry --success
*/
