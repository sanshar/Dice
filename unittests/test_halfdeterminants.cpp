#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include "Determinants.h"

HalfDet SetUpHD(std::vector<int> closed) {
  HalfDet::norbs = 10;

  HalfDet hd;
  for (auto c : closed) {
    hd.setocc(c, true);
  }
  return hd;
}

TEST_CASE("HalfDets: Basics") {
  std::vector<int> closed = {0, 1, 2};
  auto ha = SetUpHD(closed);
  std::cout << ha << std::endl;

  for (auto c : closed) {
    REQUIRE(ha.getocc(c) == true);
  }
}

// TEST_CASE("HalfDets: OpenClosed") {}