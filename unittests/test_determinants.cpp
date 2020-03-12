#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
// #define EIGEN_USE_MKL_ALL
#include <doctest.h>
#include <array>

#include "Determinants.h"

/**
 * @brief Initializes the determinant member variables for a given set of
 * parameters consistent with HF, i.e. lowest orbitals populated first.
 *
 * @param norbs Number of SPATIAL orbitals.
 * @param nalpha Number of alpha orbitals.
 * @param nbeta Number of beta orbitals.
 * @return Determinant HF determinant.
 */
Determinant HFDeterminantSetup(int norbs, int nalpha, int nbeta) {
  Determinant::EffDetLen = (norbs) / 64 + 1;
  Determinant::norbs = norbs;
  Determinant::n_spinorbs = norbs * 2;
  Determinant::nalpha = nalpha;
  Determinant::nbeta = nbeta;

  Determinant det;

  for (int i = 0; i < Determinant::n_spinorbs; i++) {
    if (i < nalpha) {
      det.setocc(2 * i, true);
    }
    if (i < nbeta) {
      det.setocc(2 * i + 1, true);
    }
  }

  return det;
}

/**
 * @brief Generates a vector of example HF determinants. If you want to add a
 * test case, add it here. The tuples represent norbs (spatial orbs), nalpha,
 * and nbeta respectively.
 *
 * @return std::vector<std::array<int, 3>>
 */
std::vector<std::array<int, 3>> HFDetParams() {
  std::vector<std::array<int, 3>> Dets;

  // Add new configurations here
  Dets.push_back(std::array<int, 3>{8, 6, 6});
  Dets.push_back(std::array<int, 3>{8, 4, 4});
  Dets.push_back(std::array<int, 3>{10, 6, 4});
  Dets.push_back(std::array<int, 3>{11, 5, 4});
  return Dets;
}

TEST_CASE("Determinants: Basics") {
  int norbs, nalpha, nbeta;

  auto hf_det_params = HFDetParams();
  for (auto det_p : hf_det_params) {
    norbs = det_p[0], nalpha = det_p[1], nbeta = det_p[2];
    auto det = HFDeterminantSetup(norbs, nalpha, nbeta);
    std::cout << det << std::endl;  // JETS: for debugging

    REQUIRE(det.Nalpha() == nalpha);
    REQUIRE(det.Nbeta() == nbeta);
    REQUIRE(det.Noccupied() == nalpha + nbeta);
    REQUIRE(det.hasUnpairedElectrons() == !(nalpha == nbeta));
    REQUIRE(det.numUnpairedElectrons() == nalpha - nbeta);
    // REQUIRE(det.parityOfFlipAlphaBeta() == 1.);
  }
}