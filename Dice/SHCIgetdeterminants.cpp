/*
  Developed by Sandeep Sharma with contributions from James E. T. Smith and Adam
  A. Holmes, 2017 Copyright (c) 2017, Sandeep Sharma

  This file is part of DICE.

  This program is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or (at your option) any later
  version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with
  this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <fstream>
#include <map>
#include <tuple>
#include <vector>
#include "Dice/Davidson.h"
#include "Dice/Utils/Determinants.h"
#include "Dice/Utils/input.h"
#include "Dice/Utils/integral.h"
#include "Dice/Hmult.h"
#include "boost/format.hpp"
#include "math.h"

#include "Dice/SHCIgetdeterminants.h"  // Keep separate or clang-tidy will rearrange

using namespace std;
using namespace Eigen;
using namespace boost;

//=============================================================================
void SHCIgetdeterminants::getDeterminantsDeterministicPT(
    Determinant& d, double epsilon, CItype ci1, CItype ci2, oneInt& int1,
    twoInt& int2, twoIntHeatBathSHM& I2hb, vector<int>& irreps, double coreE,
    double E0, std::vector<Determinant>& dets, std::vector<CItype>& numerator,
    std::vector<double>& energy, schedule& schd, int Nmc, int nelec) {
  //-----------------------------------------------------------------------------
  /*!
  BM_description

  :Inputs:

      Determinant& d:
          The reference |D_i>
      double epsilon:
          The criterion for chosing new determinants (understood as epsilon/c_i)
      CItype ci1:
          The reference CI coefficient c_i
      CItype ci2:
          The reference CI coefficient c_i (unused)
      oneInt& int1:
          One-electron tensor of the Hamiltonian
      twoInt& int2:
          Two-electron tensor of the Hamiltonian
      twoIntHeatBathSHM& I2hb:
          The sorted two-electron integrals to choose the bi-excited
  determinants vector<int>& irreps: Irrep of the orbitals double coreE: The core
  energy double E0: The current variational energy std::vector<Determinant>&
  dets: The determinants' determinant std::vector<CItype>& numerator: The
  determinants' numerator std::vector<double>& energy: The determinants' energy
      schedule& schd:
          The schedule
      int Nmc:
          BM_description
      int nelec:
          Number of electrons
  */
  //-----------------------------------------------------------------------------

  // initialize variables
  int norbs = d.norbs;
  int nclosed = nelec;
  int nopen = norbs - nclosed;
  vector<int> closed(nelec, 0);
  vector<int> open(norbs - nelec, 0);
  d.getOpenClosed(open, closed);
  // d.getRepArray(detArray);
  double Energyd = d.Energy(int1, int2, coreE);

  // mono-excited determinants
  for (int ia = 0; ia < nopen * nclosed; ia++) {
    int i = ia / nopen, a = ia % nopen;
    // CItype integral = d.Hij_1Excite(closed[i],open[a],int1,int2);
    CItype integral =
        Hij_1Excite(open[a], closed[i], int1, int2, &closed[0], nclosed);

    // sgn
    if (closed[i] % 2 != open[a] % 2) {
      double sgn = 1.0;
      d.parity(min(open[a], closed[i]), max(open[a], closed[i]), sgn);
      integral = int1(open[a], closed[i]) * sgn;
    }

    // generate determinant if integral is above the criterion
    if (fabs(integral) > epsilon) {
      dets.push_back(d);
      Determinant& di = *dets.rbegin();
      di.setocc(open[a], true);
      di.setocc(closed[i], false);

      // numerator and energy
      numerator.push_back(integral * ci1);
#ifndef Complex
      double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i,
                                       open[a], Energyd);
#else
      double E = di.Energy(int1, int2, coreE);
#endif
      energy.push_back(E);
    }
  }  // ia

  // bi-excitated determinants
  //#pragma omp parallel for schedule(dynamic)
  if (fabs(int2.maxEntry) < epsilon) return;
  // for all pairs of closed
  for (int ij = 0; ij < nclosed * nclosed; ij++) {
    int i = ij / nclosed, j = ij % nclosed;
    if (i <= j) continue;
    int I = closed[i] / 2, J = closed[j] / 2;
    int X = max(I, J), Y = min(I, J);

    int pairIndex = X * (X + 1) / 2 + Y;
    size_t start = closed[i] % 2 == closed[j] % 2
                       ? I2hb.startingIndicesSameSpin[pairIndex]
                       : I2hb.startingIndicesOppositeSpin[pairIndex];
    size_t end = closed[i] % 2 == closed[j] % 2
                     ? I2hb.startingIndicesSameSpin[pairIndex + 1]
                     : I2hb.startingIndicesOppositeSpin[pairIndex + 1];
    float* integrals = closed[i] % 2 == closed[j] % 2
                           ? I2hb.sameSpinIntegrals
                           : I2hb.oppositeSpinIntegrals;
    short* orbIndices = closed[i] % 2 == closed[j] % 2 ? I2hb.sameSpinPairs
                                                       : I2hb.oppositeSpinPairs;

    // for all HCI integrals
    for (size_t index = start; index < end; index++) {
      // if we are going below the criterion, break
      if (fabs(integrals[index]) < epsilon) break;

      // otherwise: generate the determinant corresponding to the current
      // excitation
      int a = 2 * orbIndices[2 * index] + closed[i] % 2,
          b = 2 * orbIndices[2 * index + 1] + closed[j] % 2;
      if (!(d.getocc(a) || d.getocc(b))) {
        dets.push_back(d);
        Determinant& di = *dets.rbegin();
        di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i], false),
            di.setocc(closed[j], false);

        // sgn
        double sgn = 1.0;
        di.parity(a, b, closed[i], closed[j], sgn);

        // numerator and energy
        numerator.push_back(integrals[index] * sgn * ci1);
        double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i,
                                         a, j, b, Energyd);
        energy.push_back(E);
      }
    }  // heatbath integrals
  }    // ij
  return;
}  // end SHCIgetdeterminants::getDeterminantsDeterministicPT

//=============================================================================
void SHCIgetdeterminants::getDeterminantsDeterministicPTKeepRefDets(
    Determinant det, int det_ind, double epsilon, CItype ci, oneInt& int1,
    twoInt& int2, twoIntHeatBathSHM& I2hb, vector<int>& irreps, double coreE,
    double E0, std::vector<Determinant>& dets, std::vector<CItype>& numerator,
    std::vector<double>& energy, std::vector<int>& var_indices,
    std::vector<size_t>& orbDifference, schedule& schd, int nelec) {
  //-----------------------------------------------------------------------------
  /*!
  Similar to SHCIgetdeterminants::getDeterminantsDeterministicPT,
  but also keeps track of the reference dets each connected det came from

  :Inputs:

      Determinant det:
          The reference |D_i>
      int det_ind:
          BM_description
      double epsilon:
          The criterion for chosing new determinants (understood as epsilon/c_i)
      CItype ci:
          The reference CI coefficient c_i
      oneInt& int1:
          One-electron tensor of the Hamiltonian
      twoInt& int2:
          Two-electron tensor of the Hamiltonian
      twoIntHeatBathSHM& I2hb:
          The sorted two-electron integrals to choose the bi-excited
  determinants vector<int>& irreps: Irrep of the orbitals double coreE: The core
  energy double E0: The current variational energy std::vector<Determinant>&
  dets: The determinants' determinant std::vector<CItype>& numerator: The
  determinants' numerator std::vector<double>& energy: The determinants' energy
      std::vector<int>& var_indices:
          BM_description
      std::vector<size_t>& orbDifference:
          The determinants' orbital differences
      schedule& schd:
          The schedule
      int nelec:
          Number of electrons
  */
  //-----------------------------------------------------------------------------

  // initialize variables
  int norbs = det.norbs;
  int nclosed = nelec;
  int nopen = norbs - nclosed;
  vector<int> closed(nelec, 0);
  vector<int> open(norbs - nelec, 0);
  det.getOpenClosed(open, closed);
  // d.getRepArray(detArray);
  double Energyd = det.Energy(int1, int2, coreE);
  size_t orbDiff;
  std::vector<int> var_indices_vec;
  std::vector<size_t> orbDiff_vec;

  // mono-excited determinants
  for (int ia = 0; ia < nopen * nclosed; ia++) {
    int i = ia / nopen, a = ia % nopen;
    // if (open[a]/2 > schd.nvirt+nclosed/2) continue; //dont occupy above a
    // certain orbital
    if (irreps[closed[i] / 2] != irreps[open[a] / 2]) continue;
    CItype integral =
        Hij_1Excite(open[a], closed[i], int1, int2, &closed[0], nclosed);

    // generate determinant if integral is above the criterion
    if (fabs(integral) > epsilon) {
      dets.push_back(det);
      Determinant& di = *dets.rbegin();
      di.setocc(open[a], true);
      di.setocc(closed[i], false);

      // numerator and energy
      numerator.push_back(integral * ci);
      double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i,
                                       open[a], Energyd);
      energy.push_back(E);

      // ...
      var_indices.push_back(det_ind);
      size_t A = open[a], N = norbs, I = closed[i];
      orbDiff = A * N + I;  // a = creation, i = annihilation
      orbDifference.push_back(orbDiff);
    }
  }  // ia

  // bi-excitated determinants
  //#pragma omp parallel for schedule(dynamic)
  if (fabs(int2.maxEntry) < epsilon) return;
  // for all pairs of closed
  for (int ij = 0; ij < nclosed * nclosed; ij++) {
    int i = ij / nclosed, j = ij % nclosed;
    if (i <= j) continue;
    int I = closed[i] / 2, J = closed[j] / 2;
    int X = max(I, J), Y = min(I, J);

    int pairIndex = X * (X + 1) / 2 + Y;
    size_t start = closed[i] % 2 == closed[j] % 2
                       ? I2hb.startingIndicesSameSpin[pairIndex]
                       : I2hb.startingIndicesOppositeSpin[pairIndex];
    size_t end = closed[i] % 2 == closed[j] % 2
                     ? I2hb.startingIndicesSameSpin[pairIndex + 1]
                     : I2hb.startingIndicesOppositeSpin[pairIndex + 1];
    float* integrals = closed[i] % 2 == closed[j] % 2
                           ? I2hb.sameSpinIntegrals
                           : I2hb.oppositeSpinIntegrals;
    short* orbIndices = closed[i] % 2 == closed[j] % 2 ? I2hb.sameSpinPairs
                                                       : I2hb.oppositeSpinPairs;

    // for all HCI integrals
    for (size_t index = start; index < end; index++) {
      // if we are going below the criterion, break
      if (fabs(integrals[index]) < epsilon) break;

      // otherwise: generate the determinant corresponding to the current
      // excitation
      int a = 2 * orbIndices[2 * index] + closed[i] % 2,
          b = 2 * orbIndices[2 * index + 1] + closed[j] % 2;
      if (!(det.getocc(a) || det.getocc(b))) {
        dets.push_back(det);
        Determinant& di = *dets.rbegin();
        di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i], false),
            di.setocc(closed[j], false);

        // sgn
        double sgn = 1.0;
        di.parity(a, b, closed[i], closed[j], sgn);

        // numerator and energy
        numerator.push_back(integrals[index] * sgn * ci);
        double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i,
                                         a, j, b, Energyd);
        energy.push_back(E);

        // ...
        var_indices.push_back(det_ind);
        size_t A = a, B = b, N = norbs, I = closed[i], J = closed[j];
        orbDiff = A * N * N * N + I * N * N + B * N + J;  // i>j and a>b??
        orbDifference.push_back(orbDiff);
      }
    }  // heatbath integrals
  }    // ij
  return;
}  // end SHCIgetdeterminants::getDeterminantsDeterministicPTKeepRefDets

//=============================================================================
void SHCIgetdeterminants::getDeterminantsDeterministicPTWithSOC(
    Determinant det, int det_ind, double epsilon1, CItype ci1, double epsilon2,
    CItype ci2, oneInt& int1, twoInt& int2, twoIntHeatBathSHM& I2hb,
    vector<int>& irreps, double coreE, std::vector<Determinant>& dets,
    std::vector<CItype>& numerator1, std::vector<CItype>& numerator2,
    std::vector<double>& energy, schedule& schd, int nelec) {
  //-----------------------------------------------------------------------------
  /*!
  Similar to SHCIgetdeterminants::getDeterminantsDeterministicPT,
  but with SOC modifications

  :Inputs:

      Determinant det:
          The reference |D_i>
      int det_ind:
          BM_description
      double epsilon1:
          The criterion for chosing new determinants (understood as epsilon/c_i)
      CItype ci1:
          The reference CI coefficient c_i
      double epsilon2:
          The criterion for chosing new determinants (understood as epsilon/c_i)
      CItype ci2:
          The reference CI coefficient c_i
      oneInt& int1:
          One-electron tensor of the Hamiltonian
      twoInt& int2:
          Two-electron tensor of the Hamiltonian
      twoIntHeatBathSHM& I2hb:
          The sorted two-electron integrals to choose the bi-excited
  determinants vector<int>& irreps: Irrep of the orbitals double coreE: The core
  energy std::vector<Determinant>& dets: The determinants' determinant
      std::vector<CItype>& numerator1:
          The determinants' numerator
      std::vector<CItype>& numerator2:
          The determinants' numerator
      std::vector<double>& energy:
          The determinants' energy
      schedule& schd:
          The schedule
      int nelec:
          Number of electrons
  */
  //-----------------------------------------------------------------------------

  // initialize variables
  int norbs = det.norbs;
  int nclosed = nelec;
  int nopen = norbs - nclosed;
  vector<int> closed(nelec, 0);
  vector<int> open(norbs - nelec, 0);
  det.getOpenClosed(open, closed);
  double Energyd = det.Energy(int1, int2, coreE);
  size_t orbDiff;
  std::vector<int> var_indices_vec;
  std::vector<size_t> orbDiff_vec;

  // mono-excited determinants
  for (int ia = 0; ia < nopen * nclosed; ia++) {
    int i = ia / nopen, a = ia % nopen;
    CItype integral =
        Hij_1Excite(open[a], closed[i], int1, int2, &closed[0], nclosed);

    // sgn
    if (closed[i] % 2 != open[a] % 2) {
      double sgn = 1.0;
      det.parity(min(open[a], closed[i]), max(open[a], closed[i]), sgn);
      integral = int1(open[a], closed[i]) * sgn;
    }

    // generate determinant if integral is above the criterion
    if (fabs(integral) > epsilon1 || fabs(integral) > epsilon2) {
      dets.push_back(det);
      Determinant& di = *dets.rbegin();
      di.setocc(open[a], true);
      di.setocc(closed[i], false);

      // numerator and energy
      if (fabs(integral) > epsilon1)
        numerator1.push_back(integral * ci1);
      else
        numerator1.push_back(0.0);
      if (fabs(integral) > epsilon2)
        numerator2.push_back(integral * ci2);
      else
        numerator2.push_back(0.0);
      double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i,
                                       open[a], Energyd);
      // double E = Energyd - int1(closed[i], closed[i]) +
      // int1(open[a],open[a]);
      if (closed[i] % 2 != open[a] % 2) E = di.Energy(int1, int2, coreE);
      energy.push_back(E);
    }
  }  // ia

  // bi-excitated determinants
  //#pragma omp parallel for schedule(dynamic)
  if (fabs(int2.maxEntry) < epsilon1 && fabs(int2.maxEntry) < epsilon2) return;
  // for all pairs of closed
  for (int ij = 0; ij < nclosed * nclosed; ij++) {
    int i = ij / nclosed, j = ij % nclosed;
    if (i <= j) continue;
    int I = closed[i] / 2, J = closed[j] / 2;
    int X = max(I, J), Y = min(I, J);

    int pairIndex = X * (X + 1) / 2 + Y;
    size_t start = closed[i] % 2 == closed[j] % 2
                       ? I2hb.startingIndicesSameSpin[pairIndex]
                       : I2hb.startingIndicesOppositeSpin[pairIndex];
    size_t end = closed[i] % 2 == closed[j] % 2
                     ? I2hb.startingIndicesSameSpin[pairIndex + 1]
                     : I2hb.startingIndicesOppositeSpin[pairIndex + 1];
    float* integrals = closed[i] % 2 == closed[j] % 2
                           ? I2hb.sameSpinIntegrals
                           : I2hb.oppositeSpinIntegrals;
    short* orbIndices = closed[i] % 2 == closed[j] % 2 ? I2hb.sameSpinPairs
                                                       : I2hb.oppositeSpinPairs;

    // for all HCI integrals
    for (size_t index = start; index < end; index++) {
      // if we are going below the criterion, break
      if (fabs(integrals[index]) < epsilon1 &&
          fabs(integrals[index]) < epsilon2)
        break;

      // otherwise: generate the determinant corresponding to the current
      // excitation
      int a = 2 * orbIndices[2 * index] + closed[i] % 2,
          b = 2 * orbIndices[2 * index + 1] + closed[j] % 2;
      if (!(det.getocc(a) || det.getocc(b))) {
        dets.push_back(det);
        Determinant& di = *dets.rbegin();
        di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i], false),
            di.setocc(closed[j], false);

        // sgn
        double sgn = 1.0;
        di.parity(a, b, closed[i], closed[j], sgn);

        // numerator and energy
        if (fabs(integrals[index]) > epsilon1)
          numerator1.push_back(integrals[index] * sgn * ci1);
        else
          numerator1.push_back(0.0);
        if (fabs(integrals[index]) > epsilon2)
          numerator2.push_back(integrals[index] * sgn * ci2);
        else
          numerator2.push_back(0.0);
        double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i,
                                         a, j, b, Energyd);
        energy.push_back(E);
      }
    }  // heatbath integrals
  }    // ij
  return;
}  // end SHCIgetdeterminants::getDeterminantsDeterministicPTWithSOC

//=============================================================================
void SHCIgetdeterminants::getDeterminantsVariational(
    Determinant& d, double epsilon, CItype ci1, CItype ci2, oneInt& int1,
    twoInt& int2, twoIntHeatBathSHM& I2hb, vector<int>& irreps, double coreE,
    double E0, std::vector<Determinant>& dets, schedule& schd, int Nmc,
    int nelec) {
  //-----------------------------------------------------------------------------
  /*!
  Make the int represenation of open and closed orbitals of determinant
  this helps to speed up the energy calculation

  :Inputs:

      Determinant& d:
          The reference |D_i>
      double epsilon:
          The criterion for chosing new determinants (understood as epsilon/c_i)
      CItype ci1:
          The reference CI coefficient c_i
      CItype ci2:
          The reference CI coefficient c_i
      oneInt& int1:
          One-electron tensor of the Hamiltonian
      twoInt& int2:
          Two-electron tensor of the Hamiltonian
      twoIntHeatBathSHM& I2hb:
          The sorted two-electron integrals to choose the bi-excited
  determinants vector<int>& irreps: Irrep of the orbitals double coreE: The core
  energy double E0: The current variational energy std::vector<Determinant>&
  dets: The determinants' determinant schedule& schd: The schedule int Nmc:
          BM_description
      int nelec:
          Number of electrons
  */
  //-----------------------------------------------------------------------------

  // initialize variables
  int norbs = d.norbs;
  int nclosed = nelec;
  int nopen = norbs - nclosed;
  vector<int> closed(nelec, 0);
  vector<int> open(norbs - nelec, 0);
  d.getOpenClosed(open, closed);

  // mono-excited determinants
  for (int ia = 0; ia < nopen * nclosed; ia++) {
    int i = ia / nopen, a = ia % nopen;
    if (closed[i] / 2 < schd.ncore || open[a] / 2 >= schd.ncore + schd.nact)
      continue;
      // if we are doing SOC calculation then breaking spin and point group
      // symmetry is allowed
#ifndef Complex
    if (closed[i] % 2 != open[a] % 2 ||
        irreps[closed[i] / 2] != irreps[open[a] / 2])
      continue;
#endif
    CItype integral =
        Hij_1Excite(open[a], closed[i], int1, int2, &closed[0], nclosed);

    if (closed[i] % 2 != open[a] % 2) {
      integral = int1(open[a], closed[i]) * schd.socmultiplier;
    }

    // generate determinant if integral is above the criterion
    if (fabs(integral) > epsilon) {
      dets.push_back(d);
      Determinant& di = *dets.rbegin();
      di.setocc(open[a], true);
      di.setocc(closed[i], false);
      // if (Determinant::Trev != 0) di.makeStandard();
    }
  }  // ia

  // bi-excitated determinants
  if (fabs(int2.maxEntry) < epsilon) return;
  // for all pairs of closed
  for (int ij = 0; ij < nclosed * nclosed; ij++) {
    int i = ij / nclosed, j = ij % nclosed;
    if (i <= j) continue;
    int I = closed[i] / 2, J = closed[j] / 2;
    int X = max(I, J), Y = min(I, J);

    if (closed[i] / 2 < schd.ncore || closed[j] / 2 < schd.ncore) continue;

    int pairIndex = X * (X + 1) / 2 + Y;
    size_t start = closed[i] % 2 == closed[j] % 2
                       ? I2hb.startingIndicesSameSpin[pairIndex]
                       : I2hb.startingIndicesOppositeSpin[pairIndex];
    size_t end = closed[i] % 2 == closed[j] % 2
                     ? I2hb.startingIndicesSameSpin[pairIndex + 1]
                     : I2hb.startingIndicesOppositeSpin[pairIndex + 1];
    float* integrals = closed[i] % 2 == closed[j] % 2
                           ? I2hb.sameSpinIntegrals
                           : I2hb.oppositeSpinIntegrals;
    short* orbIndices = closed[i] % 2 == closed[j] % 2 ? I2hb.sameSpinPairs
                                                       : I2hb.oppositeSpinPairs;

    // for all HCI integrals
    for (size_t index = start; index < end; index++) {
      // if we are going below the criterion, break
      if (fabs(integrals[index]) < epsilon) break;

      // otherwise: generate the determinant corresponding to the current
      // excitation
      int a = 2 * orbIndices[2 * index] + closed[i] % 2,
          b = 2 * orbIndices[2 * index + 1] + closed[j] % 2;
      if (a / 2 >= schd.ncore + schd.nact || b / 2 >= schd.ncore + schd.nact)
        continue;
      if (!(d.getocc(a) || d.getocc(b))) {
        dets.push_back(d);
        Determinant& di = *dets.rbegin();
        di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i], false),
            di.setocc(closed[j], false);
        // if (Determinant::Trev != 0) di.makeStandard();
      }
    }  // heatbath integrals
  }    // ij
  return;
}  // end SHCIgetdeterminants::getDeterminantsVariational

//=============================================================================
void SHCIgetdeterminants::getDeterminantsVariationalApprox(
    Determinant& d, double epsilon, CItype ci1, CItype ci2, oneInt& int1,
    twoInt& int2, twoIntHeatBathSHM& I2hb, vector<int>& irreps, double coreE,
    double E0, std::vector<Determinant>& dets, schedule& schd, int Nmc,
    int nelec, Determinant* SortedDets, int SortedDetsSize) {
  //-----------------------------------------------------------------------------
  /*!
  Make the int represenation of open and closed orbitals of determinant
  this helps to speed up the energy calculation

  :Inputs:

      Determinant& d:
          The reference |D_i>
      double epsilon:
          The criterion for chosing new determinants (understood as epsilon/c_i)
      CItype ci1:
          The reference CI coefficient c_i
      CItype ci2:
          The reference CI coefficient c_i
      oneInt& int1:
          One-electron tensor of the Hamiltonian
      twoInt& int2:
          Two-electron tensor of the Hamiltonian
      twoIntHeatBathSHM& I2hb:
          The sorted two-electron integrals to choose the bi-excited
  determinants vector<int>& irreps: Irrep of the orbitals double coreE: The core
  energy double E0: The current variational energy std::vector<Determinant>&
  dets: The determinants' determinant schedule& schd: The schedule int Nmc:
          BM_description
      int nelec:
          Number of electrons
      Determinant* SortedDets:
          The sorted list of determinants
      int SortedDetsSize:
          The number of unique determinants
  */
  //-----------------------------------------------------------------------------

  // initialize variables
  int norbs = d.norbs;
  int nclosed = nelec;
  int nopen = norbs - nclosed;
  vector<int> closed(nelec, 0);
  vector<int> open(norbs - nelec, 0);
  d.getOpenClosed(open, closed);
  int unpairedElecs = schd.enforceSeniority ? d.numUnpairedElectrons() : 0;

  // mono-excited determinants
  for (int ia = 0; ia < nopen * nclosed; ia++) {
    int i = ia / nopen, a = ia % nopen;
    if (closed[i] / 2 < schd.ncore || open[a] / 2 >= schd.ncore + schd.nact)
      continue;
    CItype integral = I2hb.Singles(
        open[a], closed[i]);  // Hij_1Excite(open[a],closed[i],int1,int2,
                              // &closed[0], nclosed);

    if (fabs(integral) > epsilon)
      if (closed[i] % 2 == open[a] % 2)
        integral =
            Hij_1Excite(open[a], closed[i], int1, int2, &closed[0], nclosed);

    // generate determinant if integral is above the criterion
    // if (fabs(integral/(E0-Energyd)) > epsilon ) {
    if (fabs(integral) > epsilon) {
      Determinant di = d;
      di.setocc(open[a], true);
      di.setocc(closed[i], false);

      ////if (schd.enforceSeniority && di.numUnpairedElectrons() >
      /// schd.maxSeniority) continue;
      // if (schd.enforceSenioExc){
      //  if (di.ExcitationDistance(schd.HF) > schd.maxExcitation &&
      //      di.numUnpairedElectrons()      > schd.maxSeniority){
      //    continue;
      //  }
      //} else if (schd.enforceExcitation && di.ExcitationDistance(schd.HF) >
      // schd.maxExcitation){
      //  continue;
      //} else if (schd.enforceSeniority  && di.numUnpairedElectrons()      >
      // schd.maxSeniority) {
      //  continue;
      //}

      if (!binary_search(SortedDets, SortedDets + SortedDetsSize, di))
        dets.push_back(di);
#ifdef Complex
      Determinant detcpy = di;
      detcpy.flipAlphaBeta();
      if (!binary_search(SortedDets, SortedDets + SortedDetsSize, detcpy))
        dets.push_back(detcpy);
#endif
    }
  }  // ia

  // bi-excitated determinants
  if (fabs(int2.maxEntry) < epsilon) return;
  // for all pairs of closed
  for (int ij = 0; ij < nclosed * nclosed; ij++) {
    int i = ij / nclosed, j = ij % nclosed;
    if (i <= j) continue;
    int I = closed[i] / 2, J = closed[j] / 2;
    int X = max(I, J), Y = min(I, J);

    if (closed[i] / 2 < schd.ncore || closed[j] / 2 < schd.ncore) continue;

    int pairIndex = X * (X + 1) / 2 + Y;
    size_t start = closed[i] % 2 == closed[j] % 2
                       ? I2hb.startingIndicesSameSpin[pairIndex]
                       : I2hb.startingIndicesOppositeSpin[pairIndex];
    size_t end = closed[i] % 2 == closed[j] % 2
                     ? I2hb.startingIndicesSameSpin[pairIndex + 1]
                     : I2hb.startingIndicesOppositeSpin[pairIndex + 1];
    float* integrals = closed[i] % 2 == closed[j] % 2
                           ? I2hb.sameSpinIntegrals
                           : I2hb.oppositeSpinIntegrals;
    short* orbIndices = closed[i] % 2 == closed[j] % 2 ? I2hb.sameSpinPairs
                                                       : I2hb.oppositeSpinPairs;

    // for all HCI integrals
    for (size_t index = start; index < end; index++) {
      // if we are going below the criterion, break
      if (fabs(integrals[index]) < epsilon) break;

      // otherwise: generate the determinant corresponding to the current
      // excitation
      int a = 2 * orbIndices[2 * index] + closed[i] % 2,
          b = 2 * orbIndices[2 * index + 1] + closed[j] % 2;
      // double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i,
      // a, j, b, Energyd); if (abs(integrals[index]/(E0-Energyd)) <epsilon)
      // continue;
      if (a / 2 >= schd.ncore + schd.nact || b / 2 >= schd.ncore + schd.nact)
        continue;
      if (!(d.getocc(a) || d.getocc(b))) {
        Determinant di = d;
        di.setocc(a, true);
        di.setocc(b, true);
        di.setocc(closed[i], false);
        di.setocc(closed[j], false);

        ////if (schd.enforceSeniority && di.numUnpairedElectrons() >
        /// schd.maxSeniority) continue;
        // if (schd.enforceSenioExc){
        //  if (!(di.ExcitationDistance(schd.HF) <= schd.maxExcitation ||
        //        di.numUnpairedElectrons()      <= schd.maxSeniority))
        //        continue;
        //} else if (schd.enforceExcitation && di.ExcitationDistance(schd.HF) >
        // schd.maxExcitation){
        //  continue;
        //} else if (schd.enforceSeniority  && di.numUnpairedElectrons()      >
        // schd.maxSeniority) {
        //  continue;
        //}

        if (!binary_search(SortedDets, SortedDets + SortedDetsSize, di))
          dets.push_back(di);
#ifdef Complex
        Determinant detcpy = di;
        detcpy.flipAlphaBeta();
        if (!binary_search(SortedDets, SortedDets + SortedDetsSize, detcpy))
          dets.push_back(detcpy);
#endif

        // if (Determinant::Trev != 0) di.makeStandard();
      }
    }  // heatbath integrals
  }    // ij
  return;
}  // end SHCIgetdeterminants::getDeterminantsVariationalApprox

//=============================================================================
void SHCIgetdeterminants::getDeterminantsStochastic(
    Determinant& d, double epsilon, CItype ci1, CItype ci2, oneInt& int1,
    twoInt& int2, twoIntHeatBathSHM& I2hb, vector<int>& irreps, double coreE,
    double E0, std::vector<Determinant>& dets, std::vector<CItype>& numerator1,
    vector<double>& numerator2, std::vector<double>& energy, schedule& schd,
    int Nmc, int nelec) {
  //-----------------------------------------------------------------------------
  /*!
  BM_description

  :Inputs:

      Determinant& d:
          The reference |D_i>
      double epsilon:
          The criterion for chosing new determinants (understood as epsilon/c_i)
      CItype ci1:
          The reference CI coefficient c_i
      CItype ci2:
          The reference CI coefficient c_i
      oneInt& int1:
          One-electron tensor of the Hamiltonian
      twoInt& int2:
          Two-electron tensor of the Hamiltonian
      twoIntHeatBathSHM& I2hb:
          The sorted two-electron integrals to choose the bi-excited
  determinants vector<int>& irreps: Irrep of the orbitals double coreE: The core
  energy double E0: The current variational energy std::vector<Determinant>&
  dets: The determinants' determinant std::vector<CItype>& numerator1: The
  determinants' numerator vector<double>& numerator2: The determinants'
  numerator std::vector<double>& energy: The determinants' energy schedule&
  schd: The schedule int Nmc: BM_description int nelec: Number of electrons
  */
  //-----------------------------------------------------------------------------

  // initialize variables
  int norbs = d.norbs;
  int nclosed = nelec;
  int nopen = norbs - nclosed;
  vector<int> closed(nelec, 0);
  vector<int> open(norbs - nelec, 0);
  d.getOpenClosed(open, closed);
  // d.getRepArray(detArray);
  double Energyd = d.Energy(int1, int2, coreE);
  double Nmcd = 1. * Nmc;

  // mono-excited determinants
  for (int ia = 0; ia < nopen * nclosed; ia++) {
    int i = ia / nopen, a = ia % nopen;
    // if (open[a]/2 > schd.nvirt+nclosed/2) continue; //dont occupy above a
    // certain orbital
#ifndef Complex
    if (closed[i] % 2 != open[a] % 2 ||
        irreps[closed[i] / 2] != irreps[open[a] / 2])
      continue;
#endif
    CItype integral =
        Hij_1Excite(open[a], closed[i], int1, int2, &closed[0], nclosed);

    // generate determinant if integral is above the criterion
    if (fabs(integral) > epsilon) {
      dets.push_back(d);
      Determinant& di = *dets.rbegin();
      di.setocc(open[a], true);
      di.setocc(closed[i], false);

      // numerator and energy
      numerator1.push_back(integral * ci1);
#ifndef Complex
      numerator2.push_back(integral * integral * ci1 *
                           (ci1 * Nmcd / (Nmcd - 1) - ci2));
#else
      numerator2.push_back(
          (integral * integral * ci1 * (ci1 * Nmcd / (Nmcd - 1) - ci2)).real());
#endif
#ifndef Complex
      double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i,
                                       open[a], Energyd);
#else
      double E = di.Energy(int1, int2, coreE);
#endif
      energy.push_back(E);
    }
  }  // ia

  // bi-excitated determinants
  //#pragma omp parallel for schedule(dynamic)
  if (fabs(int2.maxEntry) < epsilon) return;
  // for all pairs of closed
  for (int ij = 0; ij < nclosed * nclosed; ij++) {
    int i = ij / nclosed, j = ij % nclosed;
    if (i <= j) continue;
    int I = closed[i] / 2, J = closed[j] / 2;
    int X = max(I, J), Y = min(I, J);

    int pairIndex = X * (X + 1) / 2 + Y;
    size_t start = closed[i] % 2 == closed[j] % 2
                       ? I2hb.startingIndicesSameSpin[pairIndex]
                       : I2hb.startingIndicesOppositeSpin[pairIndex];
    size_t end = closed[i] % 2 == closed[j] % 2
                     ? I2hb.startingIndicesSameSpin[pairIndex + 1]
                     : I2hb.startingIndicesOppositeSpin[pairIndex + 1];
    float* integrals = closed[i] % 2 == closed[j] % 2
                           ? I2hb.sameSpinIntegrals
                           : I2hb.oppositeSpinIntegrals;
    short* orbIndices = closed[i] % 2 == closed[j] % 2 ? I2hb.sameSpinPairs
                                                       : I2hb.oppositeSpinPairs;

    // for all HCI integrals
    for (size_t index = start; index < end; index++) {
      // if we are going below the criterion, break
      if (fabs(integrals[index]) < epsilon) break;

      // otherwise: generate the determinant corresponding to the current
      // excitation
      int a = 2 * orbIndices[2 * index] + closed[i] % 2,
          b = 2 * orbIndices[2 * index + 1] + closed[j] % 2;
      if (!(d.getocc(a) || d.getocc(b))) {
        dets.push_back(d);
        Determinant& di = *dets.rbegin();
        di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i], false),
            di.setocc(closed[j], false);

        // sgn
        double sgn = 1.0;
        di.parity(a, b, closed[i], closed[j], sgn);

        // numerator and energy
        numerator1.push_back(integrals[index] * sgn * ci1);
#ifndef Complex
        numerator2.push_back(integrals[index] * integrals[index] * ci1 *
                             (ci1 * Nmcd / (Nmcd - 1) - ci2));
#else
        numerator2.push_back((integrals[index] * integrals[index] * 1.0 * ci1 *
                              (ci1 * Nmcd / (Nmcd - 1) - ci2))
                                 .real());
#endif
        double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i,
                                         a, j, b, Energyd);
        energy.push_back(E);
      }
    }  // heatbath integrals
  }    // ij
  return;
}  // end SHCIgetdeterminants::getDeterminantsStochastic

//=============================================================================
void SHCIgetdeterminants::getDeterminantsStochastic2Epsilon(
    Determinant& d, double epsilon, double epsilonLarge, CItype ci1, CItype ci2,
    oneInt& int1, twoInt& int2, twoIntHeatBathSHM& I2hb, vector<int>& irreps,
    double coreE, double E0, std::vector<Determinant>& dets,
    std::vector<CItype>& numerator1A, vector<CItype>& numerator2A,
    vector<char>& present, std::vector<double>& energy, schedule& schd, int Nmc,
    int nelec) {
  //-----------------------------------------------------------------------------
  /*!
  BM_description

  :Inputs:

      Determinant& d:
          The reference |D_i>
      double epsilon:
          The criterion for chosing new determinants (understood as epsilon/c_i)
      double epsilonLarge:
          The criterion for chosing new determinants (understood as epsilon/c_i)
      CItype ci1:
          The reference CI coefficient c_i
      CItype ci2:
          The reference CI coefficient c_i
      oneInt& int1:
          One-electron tensor of the Hamiltonian
      twoInt& int2:
          Two-electron tensor of the Hamiltonian
      twoIntHeatBathSHM& I2hb:
          The sorted two-electron integrals to choose the bi-excited
  determinants vector<int>& irreps: Irrep of the orbitals double coreE: The core
  energy double E0: The current variational energy std::vector<Determinant>&
  dets: The determinants' determinant std::vector<CItype>& numerator1A: The
  determinants' numerator vector<CItype>& numerator2A: The determinants'
  numerator vector<char>& present: BM_description std::vector<double>& energy:
          The determinants' energy
      schedule& schd:
          The schedule
      int Nmc:
          BM_description
      int nelec:
          Number of electrons
  */
  //-----------------------------------------------------------------------------

  // initialize variables
  int norbs = d.norbs;
  int nclosed = nelec;
  int nopen = norbs - nclosed;
  vector<int> closed(nelec, 0);
  vector<int> open(norbs - nelec, 0);
  d.getOpenClosed(open, closed);
  // d.getRepArray(detArray);
  double Energyd = d.Energy(int1, int2, coreE);
  double Nmcd = 1. * Nmc;

  // mono-excited determinants
  for (int ia = 0; ia < nopen * nclosed; ia++) {
    int i = ia / nopen, a = ia % nopen;
    CItype integral =
        Hij_1Excite(open[a], closed[i], int1, int2, &closed[0], nclosed);

    // sgn
    if (closed[i] % 2 != open[a] % 2) {
      double sgn = 1.0;
      d.parity(min(open[a], closed[i]), max(open[a], closed[i]), sgn);
      integral = int1(open[a], closed[i]) * sgn;
    }

    // generate determinant if integral is above the criterion
    if (fabs(integral) > epsilon) {
      dets.push_back(d);
      Determinant& di = *dets.rbegin();
      di.setocc(open[a], true);
      di.setocc(closed[i], false);

      // numerator and energy
      numerator1A.push_back(integral * ci1);
#ifndef Complex
      numerator2A.push_back(integral * integral * ci1 *
                            (ci1 * Nmcd / (Nmcd - 1) - ci2));
#else
      numerator2A.push_back(pow(abs(integral * ci1), 2) * Nmcd / (Nmcd - 1) *
                            (1. - abs(ci2) / abs(ci1)));
      // numerator2A.push_back( (integral*integral*ci1 *(ci1*Nmcd/(Nmcd-1)-
      // ci2)).real() );
#endif
#ifndef Complex
      double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i,
                                       open[a], Energyd);
#else
      double E = di.Energy(int1, int2, coreE);
#endif
      energy.push_back(E);

      // ...
      if (fabs(integral) > epsilonLarge)
        present.push_back(true);
      else
        present.push_back(false);
    }
  }  // ia

  // bi-excitated determinants
  //#pragma omp parallel for schedule(dynamic)
  if (fabs(int2.maxEntry) < epsilon) return;
  // for all pairs of closed
  for (int ij = 0; ij < nclosed * nclosed; ij++) {
    int i = ij / nclosed, j = ij % nclosed;
    if (i <= j) continue;
    int I = closed[i] / 2, J = closed[j] / 2;
    int X = max(I, J), Y = min(I, J);

    int pairIndex = X * (X + 1) / 2 + Y;
    size_t start = closed[i] % 2 == closed[j] % 2
                       ? I2hb.startingIndicesSameSpin[pairIndex]
                       : I2hb.startingIndicesOppositeSpin[pairIndex];
    size_t end = closed[i] % 2 == closed[j] % 2
                     ? I2hb.startingIndicesSameSpin[pairIndex + 1]
                     : I2hb.startingIndicesOppositeSpin[pairIndex + 1];
    float* integrals = closed[i] % 2 == closed[j] % 2
                           ? I2hb.sameSpinIntegrals
                           : I2hb.oppositeSpinIntegrals;
    short* orbIndices = closed[i] % 2 == closed[j] % 2 ? I2hb.sameSpinPairs
                                                       : I2hb.oppositeSpinPairs;

    // for all HCI integrals
    for (size_t index = start; index < end; index++) {
      // if we are going below the criterion, break
      if (fabs(integrals[index]) < epsilon) break;

      // otherwise: generate the determinant corresponding to the current
      // excitation
      int a = 2 * orbIndices[2 * index] + closed[i] % 2,
          b = 2 * orbIndices[2 * index + 1] + closed[j] % 2;
      if (!(d.getocc(a) || d.getocc(b))) {
        dets.push_back(d);
        Determinant& di = *dets.rbegin();
        di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i], false),
            di.setocc(closed[j], false);

        // sgn
        double sgn = 1.0;
        di.parity(a, b, closed[i], closed[j], sgn);

        // numerator and energy
        numerator1A.push_back(integrals[index] * sgn * ci1);
#ifndef Complex
        numerator2A.push_back(integrals[index] * integrals[index] * ci1 *
                              (ci1 * Nmcd / (Nmcd - 1) - ci2));
#else
        numerator2A.push_back(pow(abs(integrals[index] * 1.0 * ci1), 2) * Nmcd /
                              (Nmcd - 1) * (1. - abs(ci2) / abs(ci1)));
        // numerator2A.push_back(
        // (integrals[index]*integrals[index]*ci1*(ci1*Nmcd/(Nmcd-1)-
        // ci2)).real());
#endif
        // numerator2A.push_back(
        // fabs(integrals[index]*integrals[index]*ci1*(ci1*Nmc/(Nmc-1)- ci2)));
        double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i,
                                         a, j, b, Energyd);
        energy.push_back(E);

        // ...
        if (fabs(integrals[index]) > epsilonLarge)
          present.push_back(true);
        else
          present.push_back(false);
      }
    }  // heatbath integrals
  }    // ij
  return;
}  // end SHCIgetdeterminants::getDeterminantsStochastic2Epsilon
