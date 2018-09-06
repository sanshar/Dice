// Unit tests for the symmetry class

// Let Catch provide main():
#define CATCH_CONFIG_MAIN
#include "Determinants.h"
#include <catch.hpp>
#include <string>

// // Defining static variables
// int norbs = 10; // Spin orbs
// int nalpha = 3;
// int nbeta = 2;
// int nelec = nalpha + nbeta;
// int Determinant::norbs = norbs; // spin orbitals
// int HalfDet::norbs = norbs;     // spin orbitals
// int Determinant::EffDetLen = norbs / 64 + 1;
// char Determinant::Trev = 0; // Time reversal
// Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>
// Determinant::LexicalOrder;
int HalfDet::norbs = 1;     // spin orbitals
int Determinant::norbs = 1; // spin orbitals
int Determinant::EffDetLen = 1;
char Determinant::Trev = 0; // Time reversal
Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> Determinant::LexicalOrder;

TEST_CASE("Simple Determinants Function", "[determinants]") {

  int norbs = 10; // Spin orbs
  int nalpha = 3;
  int nbeta = 2;
  int nelec = nalpha + nbeta;
  Determinant::norbs = norbs; // spin orbitals
  HalfDet::norbs = norbs;     // spin orbitals
  Determinant::EffDetLen = norbs / 64 + 1;
  Determinant::Trev = 0; // Time reversal
  // Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>
  // Determinant::LexicalOrder;

  // Initialize dets
  Determinant::initLexicalOrder(nelec);
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
    // REQUIRE(det1.numUnpairedElectrons() == 1); // TODO

    // Test Open/Closed
    std::vector<int> open(norbs - nelec, 0);
    std::vector<int> closed(nelec, 0);
    std::vector<int> open_comp = {5, 6, 7, 8, 9};
    std::vector<int> closed_comp = {0, 1, 2, 3, 4};
    det1.getOpenClosed(open, closed);
    REQUIRE(open == open_comp);
    REQUIRE(closed == closed_comp);
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

  SECTION("Comparison of occupations") {
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

  SECTION("Parity") {
    // Test parity of single excitation
    double sgn = 1.0;
    det1.parity(3, 5, sgn);
    REQUIRE(sgn == -1.0);

    // Double excitation
    sgn = 1.0;
    int i = 3, j = 4, a = 5, b = 6;
    det1.parity(i, j, a, b, sgn);
    REQUIRE(sgn == 1.0);

    // Triple and quadruple excitation
    int c0 = 1, c1 = 2, c2 = 3, c3 = 4, d0 = 5, d1 = 6, d2 = 7, d3 = 8;
    sgn = 1.0;
    det1.parity(c0, c1, c2, d0, d1, d2, sgn);
    REQUIRE(sgn == 1.0);

    sgn = 1.0;
    det1.parity(c0, c1, c2, c3, d0, d1, d2, d3, sgn);
    REQUIRE(sgn == 1.0);
  }
}
