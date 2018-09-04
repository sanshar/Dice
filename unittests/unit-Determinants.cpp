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
  // Initialize dets
  // Determinant::initLexicalOrder(nelec);
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

  SECTION("Orbital Occupancy Properties") {
    // Test as is
    REQUIRE(det1.Noccupied() == nelec);
    REQUIRE(det1.Nalpha() == nalpha);
    REQUIRE(det1.Nbeta() == nbeta);
    REQUIRE(det1.hasUnpairedElectrons() == true);
    REQUIRE(det1.numUnpairedElectrons() == 1);

    // Test Open/Closed
    // std::vector<int> open;
    // std::vector<int> closed;
    // std::vector<int> open_comp = {5, 6, 7, 8, 9};
    // std::vector<int> closed_comp = {0, 1, 2, 3, 4};
    // det1.getOpenClosed(open, closed);
    // REQUIRE(open == open_comp);
    // REQUIRE(closed == closed_comp);
  }

  SECTION("Flipping all spins") {
    // Test after flipping alpha and beta
    det1.flipAlphaBeta();
    REQUIRE(det1.Noccupied() == nelec);
    REQUIRE(det1.Nalpha() == nbeta);
    REQUIRE(det1.Nbeta() == nalpha);
    REQUIRE(det1.hasUnpairedElectrons() == true);
    REQUIRE(det1.numUnpairedElectrons() == 1);
  }

  SECTION("Comparison of two determinants") {
    // Test connection and excitation distance
    REQUIRE(det1.connected(det2) == true);
    REQUIRE(det1.ExcitationDistance(det2) == 2);

    // Test overloaded operators
    det2.setocc(5, false);
    det2.setocc(6, false);
    det2.setocc(3, true);
    det2.setocc(4, true);
    REQUIRE(det1 == det2);
  }
}
