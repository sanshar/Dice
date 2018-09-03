// Unit tests for the symmetry class

// Let Catch provide main():
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <string>
#include "Determinants.h"

// Defining static variables
int norbs = 10;  // Spin orbs
int nalpha = 3;
int nbeta = 2;
int nelec = nalpha + nbeta;
int Determinant::norbs = norbs;  // spin orbitals
int HalfDet::norbs = norbs;      // spin orbitals
int Determinant::EffDetLen = norbs / 64 + 1;
char Determinant::Trev = 0;  // Time reversal
Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> Determinant::LexicalOrder;

TEST_CASE("Simple Determinants Function", "[determinants]") {
  SECTION("Orbital Occupancy Properties") {
    // Initialize
    Determinant det;
    det.setocc(0, true);
    det.setocc(1, true);
    det.setocc(2, true);
    det.setocc(3, true);
    det.setocc(4, true);

    // Test
    REQUIRE(det.Noccupied() == nelec);
    REQUIRE(det.Nalpha() == nalpha);
    REQUIRE(det.Nbeta() == nbeta);

    // TODO
    // getOpenClosed
  }

  SECTION("Comparison of two determinants") {
    // Initialize dets
    Determinant det1;
    det1.setocc(0, true);
    det1.setocc(1, true);
    det1.setocc(2, true);
    det1.setocc(3, true);
    det1.setocc(4, true);

    Determinant det2;
    det2.setocc(0, true);
    det2.setocc(1, true);
    det2.setocc(2, true);
    det2.setocc(5, true);
    det2.setocc(6, true);

    // Test connection and excitation distance
    REQUIRE(det1.connected(det2) == true);
    REQUIRE(det1.ExcitationDistance(det2) == 2);

    // Test overloaded operators
    det2.setocc(5, false);
    det2.setocc(6, false);
    det2.setocc(3, true);
    det2.setocc(4, true);
    REQUIRE(det1 == det2);
    std::cout << "det1 " << det1 << std::endl;
    std::cout << "det2 " << det2 << std::endl;
  }
}
