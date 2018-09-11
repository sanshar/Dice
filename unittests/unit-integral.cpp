// Unit tests for the symmetry class

// Let Catch provide main():
#define CATCH_CONFIG_MAIN
#include "SHCIshm.h"
#include "integral.h"
#include <catch.hpp>
#include <string>

TEST_CASE("*Ints", "[integrals]") {
  initSHM();
  int nsorbs = 10;
  int norbs = nsorbs / 2;

  SECTION("oneInt") {
    oneInt I1;
    I1.store.clear();
    I1.store.resize(norbs * norbs, 0.0);
    I1.norbs = norbs;

    I1(2, 3) = 5.0;

    REQUIRE(I1.store.at(2 * norbs + 3) == 5.0);
  }

  SECTION("twoInt") {
    twoInt I2;
    I2.ksym = false;
    I2.norbs = norbs;
    long npair = norbs * (norbs + 1) / 2;
    size_t I2memory = npair * (npair + 1) / 2; // memory in bytes

    int2Segment.truncate((I2memory) * sizeof(double));
    regionInt2 = boost::interprocess::mapped_region{
        int2Segment, boost::interprocess::read_write};
    memset(regionInt2.get_address(), 0., (I2memory) * sizeof(double));

    I2.store = static_cast<double *>(regionInt2.get_address());

    // Test setting integral values
    int i = 0, j = 2, k = 1, l = 3;
    I2(i, j, k, l) = 99.0;
    int I = i / 2;
    int J = j / 2;
    int K = k / 2;
    int L = l / 2;
    int IJ = max(I, J) * (max(I, J) + 1) / 2 + min(I, J);
    int KL = max(K, L) * (max(K, L) + 1) / 2 + min(K, L);
    int A = max(IJ, KL), B = min(IJ, KL);
    int ind = A * (A + 1) / 2 + B;
    REQUIRE(I2.store[ind] == 99.0);

    // Test invalid indices
    REQUIRE(I2(0, 1, 2, 3) == 0);
  }

  // SECTION("twoIntHeatBath") {
  //
  // }
}
