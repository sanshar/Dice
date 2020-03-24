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
#include "Determinants.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include "input.h"
#include "integral.h"
#include "workingArray.h"

using namespace std;
using namespace Eigen;

BigDeterminant::BigDeterminant(const Determinant& d) {
  int norbs = Determinant::norbs;
  occupation.resize(2 * norbs, 0);
  for (int i = 0; i < 2 * norbs; i++)
    if (d.getocc(i)) occupation[i] = 1;
}

const char& BigDeterminant::operator[](int j) const { return occupation[j]; }

char& BigDeterminant::operator[](int j) { return occupation[j]; }

Determinant::Determinant() {
  for (int i = 0; i < DetLen; i++) {
    reprA[i] = 0;
    reprB[i] = 0;
  }
}

Determinant::Determinant(const Determinant& d) {
  for (int i = 0; i < DetLen; i++) {
    reprA[i] = d.reprA[i];
    reprB[i] = d.reprB[i];
  }
}

void Determinant::operator=(const Determinant& d) {
  for (int i = 0; i < DetLen; i++) {
    reprA[i] = d.reprA[i];
    reprB[i] = d.reprB[i];
  }
}

void Determinant::getOpenClosed(std::vector<int>& open,
                                std::vector<int>& closed) const {
  for (int i = 0; i < norbs; i++) {
    if (getoccA(i))
      closed.push_back(2 * i);
    else
      open.push_back(2 * i);
    if (getoccB(i))
      closed.push_back(2 * i + 1);
    else
      open.push_back(2 * i + 1);
  }
}

void Determinant::getOpenClosed(bool sz, std::vector<int>& open,
                                std::vector<int>& closed) const {
  for (int i = 0; i < norbs; i++) {
    if (sz == 0) {
      if (getoccA(i))
        closed.push_back(i);
      else
        open.push_back(i);
    } else {
      if (getoccB(i))
        closed.push_back(i);
      else
        open.push_back(i);
    }
  }
}

void Determinant::getOpenClosedAlphaBeta(std::vector<int>& openAlpha,
                                         std::vector<int>& closedAlpha,
                                         std::vector<int>& openBeta,
                                         std::vector<int>& closedBeta) const {
  for (int i = 0; i < norbs; i++) {
    if (getoccA(i))
      closedAlpha.push_back(i);
    else
      openAlpha.push_back(i);
    if (getoccB(i))
      closedBeta.push_back(i);
    else
      openBeta.push_back(i);
  }
}

void Determinant::getClosedAlphaBeta(std::vector<int>& closedAlpha,
                                     std::vector<int>& closedBeta) const {
  for (int i = 0; i < norbs; i++) {
    if (getoccA(i)) closedAlpha.push_back(i);
    if (getoccB(i)) closedBeta.push_back(i);
  }
}

void Determinant::getAlphaBeta(std::vector<int>& alpha,
                               std::vector<int>& beta) const {
  for (int i = 0; i < 64 * EffDetLen; i++) {
    if (getoccA(i)) alpha.push_back(i);
    if (getoccB(i)) beta.push_back(i);
  }
}

void Determinant::getClosed(bool sz, std::vector<int>& closed) const {
  for (int i = 0; i < norbs; i++) {
    if (sz == 0) {
      if (getoccA(i)) closed.push_back(i);
    } else {
      if (getoccB(i)) closed.push_back(i);
    }
  }
}

int Determinant::getNbetaBefore(int i) const {
  int occ = 0;
  for (int n = 0; n < i / 64; n++) {
    occ += CountNonZeroBits(reprB[n]);
  }
  long one = 1;
  long mask = (one << (i % 64)) - one;
  long result = (reprB[i / 64] & mask);
  occ += CountNonZeroBits(result);
  return occ;
}

int Determinant::getNalphaBefore(int i) const {
  int occ = 0;
  for (int n = 0; n < i / 64; n++) {
    occ += CountNonZeroBits(reprA[n]);
  }
  long one = 1;
  long mask = (one << (i % 64)) - one;
  long result = (reprA[i / 64] & mask);
  occ += CountNonZeroBits(result);
  return occ;
}

double Determinant::parityA(const int& a, const int& i) const {
  double parity = 1.0;
  int occ = getNalphaBefore(i);
  occ += getNalphaBefore(a);

  parity *= (occ % 2 == 0) ? 1. : -1.;
  if (i < a) parity *= -1.;

  return parity;
}

double Determinant::parity(const int& a, const int& i, const bool& sz) const {
  if (sz == 0)
    return parityA(a, i);
  else
    return parityB(a, i);
}

double Determinant::parityB(const int& a, const int& i) const {
  double parity = 1.0;
  int occ = getNbetaBefore(i);
  occ += getNbetaBefore(a);

  parity *= (occ % 2 == 0) ? 1. : -1.;
  if (i < a) parity *= -1.;

  return parity;
}

double Determinant::parityA(const vector<int>& aArray,
                            const vector<int>& iArray) const {
  double p = 1.;
  Determinant dcopy = *this;
  for (int i = 0; i < iArray.size(); i++) {
    p *= dcopy.parityA(aArray[i], iArray[i]);

    dcopy.setoccA(iArray[i], false);
    dcopy.setoccA(aArray[i], true);
  }
  return p;
}

double Determinant::parityB(const vector<int>& aArray,
                            const vector<int>& iArray) const {
  double p = 1.;
  Determinant dcopy = *this;
  for (int i = 0; i < iArray.size(); i++) {
    p *= dcopy.parityB(aArray[i], iArray[i]);

    dcopy.setoccB(iArray[i], false);
    dcopy.setoccB(aArray[i], true);
  }
  return p;
}

double Determinant::parity(const vector<int>& aArray, const vector<int>& iArray,
                           bool sz) const {
  if (sz == 0)
    return parityA(aArray, iArray);
  else
    return parityB(aArray, iArray);
}

int Determinant::Noccupied() const {
  int nelec = 0;
  for (int i = 0; i < DetLen; i++) {
    nelec += CountNonZeroBits(reprA[i]);
    nelec += CountNonZeroBits(reprB[i]);
  }
  return nelec;
}

int Determinant::Nalpha() const {
  int nelec = 0;
  for (int i = 0; i < DetLen; i++) {
    nelec += CountNonZeroBits(reprA[i]);
  }
  return nelec;
}

int Determinant::Nbeta() const {
  int nelec = 0;
  for (int i = 0; i < DetLen; i++) {
    nelec += CountNonZeroBits(reprB[i]);
  }
  return nelec;
}

// Is the excitation between *this and d less than equal to 2.
bool Determinant::connected(const Determinant& d) const {
  int ndiff = 0;
  long u;

  for (int i = 0; i < DetLen; i++) {
    ndiff += CountNonZeroBits(reprA[i] ^ d.reprA[i]);
    ndiff += CountNonZeroBits(reprB[i] ^ d.reprB[i]);
  }
  return ndiff <= 4;
  // return true;
}

// Get the number of electrons that need to be excited to get determinant d from
// *this determinant e.g. single excitation will return 1
int Determinant::ExcitationDistance(const Determinant& d) const {
  int ndiff = 0;
  for (int i = 0; i < DetLen; i++) {
    ndiff += CountNonZeroBits(reprA[i] ^ d.reprA[i]);
    ndiff += CountNonZeroBits(reprB[i] ^ d.reprB[i]);
  }
  return ndiff / 2;
}

// the comparison between determinants is performed
bool Determinant::operator<(const Determinant& d) const {
  for (int i = DetLen - 1; i >= 0; i--) {
    if (reprA[i] < d.reprA[i])
      return true;
    else if (reprA[i] > d.reprA[i])
      return false;
    if (reprB[i] < d.reprB[i])
      return true;
    else if (reprB[i] > d.reprB[i])
      return false;
  }
  return false;
}

// check if the determinants are equal
bool Determinant::operator==(const Determinant& d) const {
  for (int i = DetLen - 1; i >= 0; i--) {
    if (reprA[i] != d.reprA[i]) return false;
    if (reprB[i] != d.reprB[i]) return false;
  }
  return true;
}

// set the occupation of the ith orbital
void Determinant::setoccA(int i, bool occ) {
  long Integer = i / 64, bit = i % 64, one = 1;
  if (occ)
    reprA[Integer] |= one << bit;
  else
    reprA[Integer] &= ~(one << bit);
}

// set the occupation of the ith orbital
void Determinant::setoccB(int i, bool occ) {
  long Integer = i / 64, bit = i % 64, one = 1;
  if (occ)
    reprB[Integer] |= one << bit;
  else
    reprB[Integer] &= ~(one << bit);
}

void Determinant::setocc(int i, bool occ) {
  if (i % 2 == 0)
    return setoccA(i / 2, occ);
  else
    return setoccB(i / 2, occ);
}

void Determinant::setocc(int i, bool sz, bool occ) {
  if (sz == 0)
    return setoccA(i, occ);
  else
    return setoccB(i, occ);
}

bool Determinant::getocc(int i) const {
  if (i % 2 == 0)
    return getoccA(i / 2);
  else
    return getoccB(i / 2);
}

bool Determinant::getocc(int i, bool sz) const {
  if (sz == 0)
    return getoccA(i);
  else
    return getoccB(i);
}

// get the occupation of the ith orbital
bool Determinant::getoccA(int i) const {
  // asser(i<norbs);
  long Integer = i / 64, bit = i % 64, reprBit = reprA[Integer];
  if ((reprBit >> bit & 1) == 0)
    return false;
  else
    return true;
}

bool Determinant::getoccB(int i) const {
  // asser(i<norbs);
  long Integer = i / 64, bit = i % 64, reprBit = reprB[Integer];
  if ((reprBit >> bit & 1) == 0)
    return false;
  else
    return true;
}

// Prints the determinant
ostream& operator<<(ostream& os, const Determinant& d) {
  for (int i = 0; i < Determinant::norbs; i++) {
    if (d.getoccA(i) == false && d.getoccB(i) == false)
      os << 0 << " ";
    else if (d.getoccA(i) == true && d.getoccB(i) == false)
      os << "a"
         << " ";
    else if (d.getoccA(i) == false && d.getoccB(i) == true)
      os << "b"
         << " ";
    else if (d.getoccA(i) == true && d.getoccB(i) == true)
      os << 2 << " ";
    if ((i + 1) % 5 == 0) os << "  ";
  }
  return os;
}

//=============================================================================
double Determinant::Energy(const oneInt& I1, const twoInt& I2,
                           const double& coreE) const {
  double energy = 0.0;
  size_t one = 1;
  vector<int> closed;
  for (int i = 0; i < DetLen; i++) {
    long reprBit = reprA[i];
    while (reprBit != 0) {
      int pos = __builtin_ffsl(reprBit);
      closed.push_back(2 * (i * 64 + pos - 1));
      reprBit &= ~(one << (pos - 1));
    }

    reprBit = reprB[i];
    while (reprBit != 0) {
      int pos = __builtin_ffsl(reprBit);
      closed.push_back(2 * (i * 64 + pos - 1) + 1);
      reprBit &= ~(one << (pos - 1));
    }
  }

  for (int i = 0; i < closed.size(); i++) {
    int I = closed.at(i);
#ifdef Complex
    energy += I1(I, I).real();
#else
    energy += I1(I, I);
#endif
    for (int j = i + 1; j < closed.size(); j++) {
      int J = closed.at(j);
      energy += I2.Direct(I / 2, J / 2);
      if ((I % 2) == (J % 2)) {
        energy -= I2.Exchange(I / 2, J / 2);
      }
    }
  }

  return energy + coreE;
}

//=============================================================================
double Determinant::parityAA(const int& i, const int& j, const int& a,
                             const int& b) const {
  double sgn = 1.0;
  Determinant dcopy = *this;
  sgn *= dcopy.parityA(a, i);
  dcopy.setoccA(i, false);
  dcopy.setoccA(a, true);
  sgn *= dcopy.parityA(b, j);
  return sgn;
}

double Determinant::parityBB(const int& i, const int& j, const int& a,
                             const int& b) const {
  double sgn = 1.0;
  Determinant dcopy = *this;
  sgn = dcopy.parityB(a, i);
  dcopy.setoccB(i, false);
  dcopy.setoccB(a, true);
  sgn *= dcopy.parityB(b, j);
  return sgn;
}

//=============================================================================
CItype Determinant::Hij_2ExciteAA(const int& a, const int& i, const int& b,
                                  const int& j, const oneInt& I1,
                                  const twoInt& I2) const {
  double sgn = parityAA(i, j, a, b);
  return sgn *
         (I2(2 * a, 2 * i, 2 * b, 2 * j) - I2(2 * a, 2 * j, 2 * b, 2 * i));
}

CItype Determinant::Hij_2ExciteBB(const int& a, const int& i, const int& b,
                                  const int& j, const oneInt& I1,
                                  const twoInt& I2) const {
  double sgn = parityBB(i, j, a, b);
  return sgn * (I2(2 * a + 1, 2 * i + 1, 2 * b + 1, 2 * j + 1) -
                I2(2 * a + 1, 2 * j + 1, 2 * b + 1, 2 * i + 1));
}

CItype Determinant::Hij_2ExciteAB(const int& a, const int& i, const int& b,
                                  const int& j, const oneInt& I1,
                                  const twoInt& I2) const {
  double sgn = parityA(a, i);
  sgn *= parityB(b, j);
  return sgn * I2(2 * a, 2 * i, 2 * b + 1, 2 * j + 1);
}

CItype Determinant::Hij_1ExciteScreened(const int& a, const int& i,
                                        const twoIntHeatBathSHM& I2hb,
                                        const double& TINY,
                                        bool doparity) const {
  double tia = I1(a, i);
  int X = max(i / 2, a / 2), Y = min(i / 2, a / 2);
  int pairIndex = X * (X + 1) / 2 + Y;
  size_t start = I2hb.startingIndicesSingleIntegrals[pairIndex];
  size_t end = I2hb.startingIndicesSingleIntegrals[pairIndex + 1];
  float* integrals = I2hb.singleIntegrals;
  short* orbIndices = I2hb.singleIntegralsPairs;
  for (size_t index = start; index < end; index++) {
    if (fabs(integrals[index]) < TINY) break;
    int j = orbIndices[2 * index];
    if (i % 2 == 1 && j % 2 == 1)
      j--;
    else if (i % 2 == 1 && j % 2 == 0)
      j++;

    if (getocc(j)) tia += integrals[index];
  }

  double sgn = 1.0;
  int A = a / 2, I = i / 2;
  if (doparity && i % 2 == 0)
    sgn *= parityA(A, I);
  else if (doparity && i % 2 == 1)
    sgn *= parityB(A, I);
  return tia * sgn;
}

//=============================================================================
CItype Determinant::Hij_1ExciteA(const int& a, const int& i, const oneInt& I1,
                                 const twoInt& I2, bool doparity) const {
  double sgn = 1.0;
  if (doparity) sgn *= parityA(a, i);

  CItype energy = I1(2 * a, 2 * i);
  // if (schd.Hamiltonian == HUBBARD) return energy * sgn; // From VMC DO NOT
  // DELETE

  long one = 1;
  for (int I = 0; I < DetLen; I++) {
    long reprBit = reprA[I];
    while (reprBit != 0) {
      int pos = __builtin_ffsl(reprBit);
      int j = I * 64 + pos - 1;
      energy +=
          (I2(2 * a, 2 * i, 2 * j, 2 * j) - I2(2 * a, 2 * j, 2 * j, 2 * i));
      reprBit &= ~(one << (pos - 1));
    }
    reprBit = reprB[I];
    while (reprBit != 0) {
      int pos = __builtin_ffsl(reprBit);
      int j = I * 64 + pos - 1;
      energy += (I2(2 * a, 2 * i, 2 * j + 1, 2 * j + 1));
      reprBit &= ~(one << (pos - 1));
    }
  }
  energy *= sgn;
  return energy;
}

CItype Determinant::Hij_1ExciteB(const int& a, const int& i, const oneInt& I1,
                                 const twoInt& I2, bool doparity) const {
  double sgn = 1.0;
  if (doparity) sgn *= parityB(a, i);

  CItype energy = I1(2 * a + 1, 2 * i + 1);
  // if (schd.Hamiltonian == HUBBARD) return energy * sgn; // From VMC DO NOT
  // DELETE

  long one = 1;
  for (int I = 0; I < DetLen; I++) {
    long reprBit = reprA[I];
    while (reprBit != 0) {
      int pos = __builtin_ffsl(reprBit);
      int j = I * 64 + pos - 1;
      energy += (I2(2 * a + 1, 2 * i + 1, 2 * j, 2 * j));
      reprBit &= ~(one << (pos - 1));
    }
    reprBit = reprB[I];
    while (reprBit != 0) {
      int pos = __builtin_ffsl(reprBit);
      int j = I * 64 + pos - 1;
      energy += (I2(2 * a + 1, 2 * i + 1, 2 * j + 1, 2 * j + 1) -
                 I2(2 * a + 1, 2 * j + 1, 2 * j + 1, 2 * i + 1));
      reprBit &= ~(one << (pos - 1));
    }
  }
  energy *= sgn;
  return energy;
}

//=============================================================================
CItype Hij(const Determinant& bra, const Determinant& ket, const oneInt& I1,
           const twoInt& I2, const double& coreE) {
  int cre[200], des[200], ncrea = 0, ncreb = 0, ndesa = 0, ndesb = 0;
  long u, b, k, one = 1;
  cre[0] = -1;
  cre[1] = -1;
  des[0] = -1;
  des[1] = -1;

  for (int i = 0; i < Determinant::EffDetLen; i++) {
    u = bra.reprA[i] ^ ket.reprA[i];
    b = u & bra.reprA[i];  // the cre bits
    k = u & ket.reprA[i];  // the des bits

    while (b != 0) {
      int pos = __builtin_ffsl(b);
      cre[ncrea + ncreb] = 2 * (pos - 1 + i * 64);
      ncrea++;
      b &= ~(one << (pos - 1));
    }
    while (k != 0) {
      int pos = __builtin_ffsl(k);
      des[ndesa + ndesb] = 2 * (pos - 1 + i * 64);
      ndesa++;
      k &= ~(one << (pos - 1));
    }

    u = bra.reprB[i] ^ ket.reprB[i];
    b = u & bra.reprB[i];  // the cre bits
    k = u & ket.reprB[i];  // the des bits

    while (b != 0) {
      int pos = __builtin_ffsl(b);
      cre[ncrea + ncreb] = 2 * (pos - 1 + i * 64) + 1;
      ncreb++;
      b &= ~(one << (pos - 1));
    }
    while (k != 0) {
      int pos = __builtin_ffsl(k);
      des[ndesa + ndesb] = 2 * (pos - 1 + i * 64) + 1;
      ndesb++;
      k &= ~(one << (pos - 1));
    }
  }

  if (ncrea + ncreb == 0) {
    cout << bra << endl;
    cout << ket << endl;
    cout << "Use the function for energy" << endl;
    exit(0);
  } else if (ncrea == 1 && ncreb == 0) {
    int c0 = cre[0] / 2, d0 = des[0] / 2;
    return ket.Hij_1ExciteA(c0, d0, I1, I2);
  } else if (ncrea == 0 && ncreb == 1) {
    int c0 = cre[0] / 2, d0 = des[0] / 2;
    return ket.Hij_1ExciteB(c0, d0, I1, I2);
  } else if (ncrea == 0 && ncreb == 2) {
    int c0 = cre[0] / 2, d0 = des[0] / 2;
    int c1 = cre[1] / 2, d1 = des[1] / 2;
    return ket.Hij_2ExciteBB(c0, d0, c1, d1, I1, I2);
  } else if (ncrea == 2 && ncreb == 0) {
    int c0 = cre[0] / 2, d0 = des[0] / 2;
    int c1 = cre[1] / 2, d1 = des[1] / 2;
    return ket.Hij_2ExciteAA(c0, d0, c1, d1, I1, I2);
  } else if (ncrea == 1 && ncreb == 1) {
    int c0 = cre[0] / 2, d0 = des[0] / 2;
    int c1 = cre[1] / 2, d1 = des[1] / 2;
    if (cre[0] % 2 == 0)
      return ket.Hij_2ExciteAB(c0, d0, c1, d1, I1, I2);
    else
      return ket.Hij_2ExciteAB(c1, d1, c0, d0, I1, I2);
  } else {
    return 0.;
  }
}

void getDifferenceInOccupation(const Determinant& bra, const Determinant& ket,
                               vector<int>& creA, vector<int>& desA,
                               vector<int>& creB, vector<int>& desB) {
  std::fill(creA.begin(), creA.end(), -1);
  std::fill(desA.begin(), desA.end(), -1);
  std::fill(creB.begin(), creB.end(), -1);
  std::fill(desB.begin(), desB.end(), -1);

  int ncre = 0, ndes = 0;
  long u, b, k, one = 1;

  for (int i = 0; i < DetLen; i++) {
    u = bra.reprA[i] ^ ket.reprA[i];
    b = u & bra.reprA[i];  // the cre bits
    k = u & ket.reprA[i];  // the des bits

    while (b != 0) {
      int pos = __builtin_ffsl(b);
      creA[ncre] = pos - 1 + i * 64;
      ncre++;
      b &= ~(one << (pos - 1));
    }
    while (k != 0) {
      int pos = __builtin_ffsl(k);
      desA[ndes] = pos - 1 + i * 64;
      ndes++;
      k &= ~(one << (pos - 1));
    }
  }

  ncre = 0;
  ndes = 0;
  for (int i = 0; i < DetLen; i++) {
    u = bra.reprB[i] ^ ket.reprB[i];
    b = u & bra.reprB[i];  // the cre bits
    k = u & ket.reprB[i];  // the des bits

    while (b != 0) {
      int pos = __builtin_ffsl(b);
      creB[ncre] = pos - 1 + i * 64;
      ncre++;
      b &= ~(one << (pos - 1));
    }
    while (k != 0) {
      int pos = __builtin_ffsl(k);
      desB[ndes] = pos - 1 + i * 64;
      ndes++;
      k &= ~(one << (pos - 1));
    }
  }
}

void getDifferenceInOccupation(const Determinant& bra, const Determinant& ket,
                               vector<int>& cre, vector<int>& des, bool sz) {
  std::fill(cre.begin(), cre.end(), -1);
  std::fill(des.begin(), des.end(), -1);

  int ncre = 0, ndes = 0;
  long u, b, k, one = 1;

  if (sz == 0) {
    for (int i = 0; i < DetLen; i++) {
      u = bra.reprA[i] ^ ket.reprA[i];
      b = u & bra.reprA[i];  // the cre bits
      k = u & ket.reprA[i];  // the des bits

      while (b != 0) {
        int pos = __builtin_ffsl(b);
        cre[ncre] = pos - 1 + i * 64;
        ncre++;
        b &= ~(one << (pos - 1));
      }
      while (k != 0) {
        int pos = __builtin_ffsl(k);
        des[ndes] = pos - 1 + i * 64;
        ndes++;
        k &= ~(one << (pos - 1));
      }
    }
  }

  else {
    for (int i = 0; i < DetLen; i++) {
      u = bra.reprB[i] ^ ket.reprB[i];
      b = u & bra.reprB[i];  // the cre bits
      k = u & ket.reprB[i];  // the des bits

      while (b != 0) {
        int pos = __builtin_ffsl(b);
        cre[ncre] = pos - 1 + i * 64;
        ncre++;
        b &= ~(one << (pos - 1));
      }
      while (k != 0) {
        int pos = __builtin_ffsl(k);
        des[ndes] = pos - 1 + i * 64;
        ndes++;
        k &= ~(one << (pos - 1));
      }
    }
  }
}

void getDifferenceInOccupation(const Determinant& bra, const Determinant& ket,
                               int& I, int& A) {
  I = -1;
  A = -1;
  long u, b, k, one = 1;

  for (int i = 0; i < DetLen; i++) {
    u = bra.reprA[i] ^ ket.reprA[i];
    b = u & bra.reprA[i];  // the cre bits
    k = u & ket.reprA[i];  // the des bits

    while (b != 0) {
      int pos = __builtin_ffsl(b);
      I = 2 * (pos - 1 + i * 64);
      b &= ~(one << (pos - 1));
    }
    while (k != 0) {
      int pos = __builtin_ffsl(k);
      A = 2 * (pos - 1 + i * 64);
      k &= ~(one << (pos - 1));
    }
  }

  for (int i = 0; i < DetLen; i++) {
    u = bra.reprB[i] ^ ket.reprB[i];
    b = u & bra.reprB[i];  // the cre bits
    k = u & ket.reprB[i];  // the des bits

    while (b != 0) {
      int pos = __builtin_ffsl(b);
      I = 2 * (pos - 1 + i * 64) + 1;
      b &= ~(one << (pos - 1));
    }
    while (k != 0) {
      int pos = __builtin_ffsl(k);
      A = 2 * (pos - 1 + i * 64) + 1;
      k &= ~(one << (pos - 1));
    }
  }
}

void getDifferenceInOccupation(const Determinant& bra, const Determinant& ket,
                               int& I, int& J, int& A, int& B) {
  I = -1;
  A = -1;
  J = -1;
  B = -1;
  long u, b, k, one = 1;

  for (int i = 0; i < DetLen; i++) {
    u = bra.reprA[i] ^ ket.reprA[i];
    b = u & bra.reprA[i];  // the cre bits
    k = u & ket.reprA[i];  // the des bits

    while (b != 0) {
      int pos = __builtin_ffsl(b);

      if (I == -1)
        I = 2 * (pos - 1 + i * 64);
      else
        J = 2 * (pos - 1 + i * 64);

      b &= ~(one << (pos - 1));
    }
    while (k != 0) {
      int pos = __builtin_ffsl(k);
      if (A == -1)
        A = 2 * (pos - 1 + i * 64);
      else
        B = 2 * (pos - 1 + i * 64);
      // A = 2 * (pos - 1 + i * 64);
      k &= ~(one << (pos - 1));
    }
  }

  for (int i = 0; i < DetLen; i++) {
    u = bra.reprB[i] ^ ket.reprB[i];
    b = u & bra.reprB[i];  // the cre bits
    k = u & ket.reprB[i];  // the des bits

    while (b != 0) {
      int pos = __builtin_ffsl(b);

      if (I == -1)
        I = 2 * (pos - 1 + i * 64) + 1;
      else
        J = 2 * (pos - 1 + i * 64) + 1;

      b &= ~(one << (pos - 1));
    }
    while (k != 0) {
      int pos = __builtin_ffsl(k);
      if (A == -1)
        A = 2 * (pos - 1 + i * 64) + 1;
      else
        B = 2 * (pos - 1 + i * 64) + 1;

      k &= ~(one << (pos - 1));
    }
  }
}

double getParityForDiceToAlphaBeta(const Determinant& det) {
  double parity = 1.0;
  int nalpha = det.Nalpha();
  int norbs = Determinant::norbs;
  for (int i = 0; i < norbs; i++) {
    if (det.getoccB(norbs - 1 - i)) {
      int nAlphaAfteri = nalpha - det.getNalphaBefore(norbs - 1 - i);
      if (det.getoccA(norbs - 1 - i)) nAlphaAfteri--;
      if (nAlphaAfteri % 2 == 1) parity *= -1;
    }
  }
  return parity;
}

void generateScreenedSingleDoubleExcitation(const Determinant& d,
                                            const double& THRESH,
                                            const double& TINY,
                                            workingArray& work, bool doparity) {
  int norbs = Determinant::norbs;
  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);

  for (int i = 0; i < closed.size(); i++) {
    for (int a = 0; a < open.size(); a++) {
      if (closed[i] % 2 == open[a] % 2) {
        // if (closed[i] % 2 == open[a] % 2 &&
        // abs(I2hb.Singles(closed[i], open[a])) > THRESH)
        //{
        int I = closed[i] / 2, A = open[a] / 2;

        const double tia =
            d.Hij_1ExciteScreened(open[a], closed[i], I2hb, TINY, doparity);

        if (abs(tia) > THRESH) {
          work.appendValue(0., closed[i] * 2 * norbs + open[a], 0, tia);
        }
      }
    }
  }

  int nclosed = closed.size();
  for (int i = 0; i < nclosed; i++) {
    for (int j = 0; j < i; j++) {
      const float* integrals;
      const short* orbIndices;
      size_t numIntegrals;
      I2hb.getIntegralArray(closed[i], closed[j], integrals, orbIndices,
                            numIntegrals);
      size_t numLargeIntegrals =
          std::lower_bound(
              integrals, integrals + numIntegrals, THRESH,
              [](const float& x, float val) { return fabs(x) > val; }) -
          integrals;

      // for all HCI integrals
      for (size_t index = 0; index < numLargeIntegrals; index++) {
        // if we are going below the criterion, break
        // if (fabs(integrals[index]) < THRESH)
        //  break;

        // otherwise: generate the determinant corresponding to the current
        // excitation
        int a = 2 * orbIndices[2 * index] + closed[i] % 2,
            b = 2 * orbIndices[2 * index + 1] + closed[j] % 2;

        if (!(d.getocc(a) || d.getocc(b))) {
          work.appendValue(0.0, closed[i] * 2 * norbs + a,
                           closed[j] * 2 * norbs + b, integrals[index]);
        }
      }
    }
  }
}

void generateAllScreenedSingleExcitation(const Determinant& d,
                                         const double& THRESH,
                                         const double& TINY, workingArray& work,
                                         bool doparity) {
  int norbs = Determinant::norbs;
  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);

  for (int i = 0; i < closed.size(); i++) {
    for (int a = 0; a < open.size(); a++) {
      if (closed[i] % 2 == open[a] % 2) {
        // if (closed[i] % 2 == open[a] % 2 &&
        // abs(I2hb.Singles(closed[i], open[a])) > THRESH)
        //{
        int I = closed[i] / 2, A = open[a] / 2;

        const double tia =
            d.Hij_1ExciteScreened(open[a], closed[i], I2hb, TINY, doparity);

        if (abs(tia) > THRESH) {
          work.appendValue(0., closed[i] * 2 * norbs + open[a], 0, tia);
        }
      }
    }
  }
}

void generateAllScreenedDoubleExcitation(const Determinant& d,
                                         const double& THRESH,
                                         const double& TINY, workingArray& work,
                                         bool doparity) {
  int norbs = Determinant::norbs;
  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);

  int nclosed = closed.size();
  for (int i = 0; i < nclosed; i++) {
    for (int j = 0; j < i; j++) {
      const float* integrals;
      const short* orbIndices;
      size_t numIntegrals;
      I2hb.getIntegralArray(closed[i], closed[j], integrals, orbIndices,
                            numIntegrals);
      size_t numLargeIntegrals =
          std::lower_bound(
              integrals, integrals + numIntegrals, THRESH,
              [](const float& x, float val) { return fabs(x) > val; }) -
          integrals;

      // for all HCI integrals
      for (size_t index = 0; index < numLargeIntegrals; index++) {
        // if we are going below the criterion, break
        // if (fabs(integrals[index]) < THRESH)
        //  break;

        // otherwise: generate the determinant corresponding to the current
        // excitation
        int a = 2 * orbIndices[2 * index] + closed[i] % 2,
            b = 2 * orbIndices[2 * index + 1] + closed[j] % 2;

        if (!(d.getocc(a) || d.getocc(b))) {
          work.appendValue(0.0, closed[i] * 2 * norbs + a,
                           closed[j] * 2 * norbs + b, integrals[index]);
        }
      }
    }
  }
}

bool applyExcitation(int a, int b, int k, int l, Determinant& dcopy) {
  bool valid = true;

  if (dcopy.getocc(l) == true)
    dcopy.setocc(l, false);
  else
    return false;

  if (dcopy.getocc(b) == false)
    dcopy.setocc(b, true);
  else
    return false;

  if (dcopy.getocc(k) == true)
    dcopy.setocc(k, false);
  else
    return false;

  if (dcopy.getocc(a) == false)
    dcopy.setocc(a, true);
  else
    return false;

  return valid;
}

// generate all the alpha or beta strings
void comb(int N, int K, vector<vector<int>>& combinations) {
  std::vector<int> bitmask(K, 1);
  bitmask.resize(N, 0);  // N-K trailing 0's

  // print integers and permute bitmask
  // int index = 0; // JETS: unused-variable
  do {
    vector<int> comb;
    for (int i = 0; i < N; ++i)  // [0..N-1] integers
    {
      if (bitmask[i] == 1) comb.push_back(i);
    }
    combinations.push_back(comb);
  } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
}

void generateAllDeterminants(vector<Determinant>& allDets, int norbs,
                             int nalpha, int nbeta) {
  vector<vector<int>> alphaDets, betaDets;
  comb(norbs, nalpha, alphaDets);
  comb(norbs, nbeta, betaDets);

  for (size_t a = 0; a < alphaDets.size(); a++)
    for (size_t b = 0; b < betaDets.size(); b++) {
      Determinant d;
      for (size_t i = 0; i < alphaDets[a].size(); i++)
        d.setoccA(alphaDets[a][i], true);
      for (size_t i = 0; i < betaDets[b].size(); i++)
        d.setoccB(betaDets[b][i], true);
      allDets.push_back(d);
    }

  alphaDets.clear();
  betaDets.clear();
}

//
// From Dice
//

void updateHijForTReversal(CItype& hij, Determinant& dj, Determinant& dk,
                           oneInt& I1, twoInt& I2, double& coreE,
                           size_t& orbDiff) {
  /*!
     with treversal symmetry dj and dk will find each other multiple times
     we prune this possibility as follows
     ->  we only look for connection is dj is positive (starndard form)
     -> even it is positive there still might be two connections to dk
       -> so this updates hij

     :Arguments:

      CItype& hij:
          Hamiltonian matrix element, modifed in function.
      Determinant& dj:
          Determinant j.
      Determinant& dk:
          Determinant k.
      oneInt& I1:
          One body integrals.
      twoInt& I2:
          Two body integrals.
      double& coreE:
          Core energy.
      size_t& orbDiff:
          Different number of orbitals between determinants j and k.
   */
  if (Determinant::Trev != 0 && !dj.hasUnpairedElectrons() &&
      dk.hasUnpairedElectrons()) {
    Determinant detcpy = dk;

    detcpy.flipAlphaBeta();
    if (!detcpy.connected(dj)) return;
    double parity = dk.parityOfFlipAlphaBeta();
    CItype hijCopy =
        Hij(dj, detcpy, I1, I2, coreE, orbDiff);  // JETS: from old Dice
    // CItype hijCopy = Hij(dj, detcpy, I1, I2, coreE);
    hij = (hij + parity * Determinant::Trev * hijCopy) / pow(2., 0.5);
  } else if (Determinant::Trev != 0 && dj.hasUnpairedElectrons() &&
             !dk.hasUnpairedElectrons()) {
    Determinant detcpy = dj;

    detcpy.flipAlphaBeta();
    if (!detcpy.connected(dk)) return;
    double parity = dj.parityOfFlipAlphaBeta();
    CItype hijCopy =
        Hij(detcpy, dk, I1, I2, coreE, orbDiff);  // JETS: from old Dice
    // CItype hijCopy = Hij(detcpy, dk, I1, I2, coreE);
    hij = (hij + parity * Determinant::Trev * hijCopy) / pow(2., 0.5);
  } else if (Determinant::Trev != 0 && dj.hasUnpairedElectrons() &&
             dk.hasUnpairedElectrons()) {
    Determinant detcpyk = dk;

    detcpyk.flipAlphaBeta();
    if (!detcpyk.connected(dj)) return;
    double parityk = dk.parityOfFlipAlphaBeta();
    CItype hijCopy1 =
        Hij(dj, detcpyk, I1, I2, coreE, orbDiff);  // JETS: from old Dice
    // CItype hijCopy1 = Hij(dj, detcpyk, I1, I2, coreE);
    // CItype hijCopy2, hijCopy3; // JETS: unused-variable
    hij = hij + Determinant::Trev * parityk * hijCopy1;
  }
}

void Determinant::initLexicalOrder(int nelec) {
  LexicalOrder.setZero(norbs - nelec + 1, nelec);
  Matrix<size_t, Dynamic, Dynamic> NodeWts(norbs - nelec + 2, nelec + 1);
  NodeWts(0, 0) = 1;
  for (int i = 0; i < nelec + 1; i++) NodeWts(0, i) = 1;
  for (int i = 0; i < norbs - nelec + 2; i++) NodeWts(i, 0) = 1;

  for (int i = 1; i < norbs - nelec + 2; i++)
    for (int j = 1; j < nelec + 1; j++)
      NodeWts(i, j) = NodeWts(i - 1, j) + NodeWts(i, j - 1);

  for (int i = 0; i < norbs - nelec + 1; i++) {
    for (int j = 0; j < nelec; j++) {
      LexicalOrder(i, j) = NodeWts(i, j + 1) - NodeWts(i, j);
    }
  }
}

//
// JETS: From Old Dice
//
CItype Determinant::Hij_2Excite(int& i, int& j, int& a, int& b, oneInt& I1,
                                twoInt& I2) {
  /*!
  Calculate the hamiltonian matrix element connecting determinants connected by
  :math:`\Gamma = a^\dagger_i a^\dagger_j a_b a_a`, i.e. double excitation.

  :Arguments:

      int& i:
          Creation operator index.
      int& j:
          Creation operator index.
      int& a:
          Destruction operator index.
      int& b:
          Destruction operator index.
      oneInt& I1:
          One body integrals.
      twoInt& I2:
          Two body integrals.

  */
  double sgn = 1.0;
  int I = min(i, j), J = max(i, j), A = min(a, b), B = max(a, b);
  parity(min(I, A), max(I, A), sgn);
  parity(min(J, B), max(J, B), sgn);
  if (A > J || B < I) sgn *= -1.;
  return sgn * (I2(A, I, B, J) - I2(A, J, B, I));
}

//=============================================================================
CItype Hij_1Excite(int a, int i, oneInt& I1, twoInt& I2, int* closed,
                   int& nclosed) {
  /*!
  Calculate the hamiltonian matrix element connecting determinants connected by
  :math:`\Gamma = a^\dagger_i a^\dagger_j a_b a_a`, i.e. double excitation.

  :Arguments:

      int& i:
          Creation operator index.
      int& j:
          Creation operator index.
      int& a:
          Destruction operator index.
      int& b:
          Destruction operator index.
      oneInt& I1:
          One body integrals.
      twoInt& I2:
          Two body integrals.

  */
  // int a = cre[0], i = des[0];
  double sgn = 1.0;

  CItype energy = I1(a, i);
  for (int j = 0; j < nclosed; j++) {
    if (closed[j] > min(i, a) && closed[j] < max(i, a)) sgn *= -1.;
    energy += (I2(a, i, closed[j], closed[j]) - I2(a, closed[j], closed[j], i));
  }

  return energy * sgn;
}

//=============================================================================
CItype Determinant::Hij_1Excite(int a, int i, oneInt& I1, twoInt& I2) {
  /*!
  Calculate the hamiltonian matrix element connecting determinants connected by
  :math:`\Gamma = a^\dagger_a a_i`, i.e. single excitation.

  :Arguments:

      int& a:
          Creation operator index.
      int& i:
          Destruction operator index.
      oneInt& I1:
          One body integrals.
      twoInt& I2:
          Two body integrals.

  */
  double sgn = 1.0;
  parity(min(a, i), max(a, i), sgn);

  CItype energy = I1(a, i);
  long one = 1;
  for (int I = 0; I < EffDetLen; I++) {
    long reprBit = reprA[I];
    while (reprBit != 0) {
      int pos = __builtin_ffsl(reprBit);
      int j = 2 * (I * 64 + pos - 1);  // JETS: convert to spinorbs
      energy += (I2(a, i, j, j) - I2(a, j, j, i));
      reprBit &= ~(one << (pos - 1));
    }

    reprBit = reprB[I];
    while (reprBit != 0) {
      int pos = __builtin_ffsl(reprBit);
      int j = 2 * (I * 64 + pos - 1) + 1;  // JETS: convert to spinorbs
      energy += (I2(a, i, j, j) - I2(a, j, j, i));
      reprBit &= ~(one << (pos - 1));
    }
  }
  energy *= sgn;
  return energy;
}

//=============================================================================
void getOrbDiff(Determinant& bra, Determinant& ket, size_t& orbDiff) {
  /*!
  Calculates the number of orbitals with differing occuations between bra and
  ket.

  :Arguments:

      Determinant& bra:
          Determinant in bra.
      Determinant& ket:
          Determinant in ket.
      size_t& orbDiff:
          Number of orbitals with differing occupations. Changed in this
  function.
  */
  int cre[2], des[2], ncre = 0, ndes = 0;
  long ua, ba, ka;
  long ub, bb, kb;
  cre[0] = -1;
  cre[1] = -1;
  des[0] = -1;
  des[1] = -1;

  for (int i = 0; i < Determinant::EffDetLen; i++) {
    // Alpha excitations
    ua = bra.reprA[i] ^ ket.reprA[i];
    ba = ua & bra.reprA[i];  // the cre bits
    ka = ua & ket.reprA[i];  // the des bits
    GetLadderOps(ba, cre, ncre, i, false);
    GetLadderOps(ka, des, ndes, i, false);

    // Beta excitations
    ub = bra.reprB[i] ^ ket.reprB[i];
    bb = ub & bra.reprB[i];  // the cre bits
    kb = ub & ket.reprB[i];  // the des bits
    GetLadderOps(bb, cre, ncre, i, true);
    GetLadderOps(kb, des, ndes, i, true);
  }

  size_t N = Determinant::n_spinorbs;
  if (ncre == 0) {
    orbDiff = 0;
  } else if (ncre == 1) {
    size_t c0 = cre[0], d0 = des[0];
    orbDiff = c0 * N + d0;
  } else if (ncre == 2) {
    size_t c0 = cre[0], c1 = cre[1], d1 = des[1], d0 = des[0];
    orbDiff = c1 * N * N * N + d1 * N * N + c0 * N + d0;
  } else {
    std::cout << "Different by more than 2 exitations." << std::endl;
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief GetLadderOps determines the positions of non-zero bits and keeps track
 * of their total. TODO this is still a bit messy and shoudl really be part of a
 * larger function used in getOrbDiff() and Hij().
 *
 * @param bits A long representing the bits.
 * @param ladder_ops An empty c style array of the ladder operator indices.
 * @param n_ops The number of operators, this is modified in the function and
 * must be a reference.
 * @param ith_rep The position in the total number of irreps.
 * @param beta Whether or not the bits represent a beta sting.
 */
void GetLadderOps(long bits, int ladder_ops[], int& n_ops, int ith_rep,
                  bool beta) {
  long one = 1;
  while (bits != 0) {
    int pos = __builtin_ffsl(bits);
    ladder_ops[n_ops] =
        2 * (pos - 1 + ith_rep * 64) + beta;  // JETS: convert to spinorbs
    n_ops++;
    bits &= ~(one << (pos - 1));
  }
}

/**
 * @brief Calculates the hamiltonian matrix element connecting the two
 * determinants bra and ket.
 *
 * @param bra The bra determinant.
 * @param ket The ket determinant.
 * @param I1 The one-body integrals.
 * @param I2 The two-body integrals.
 * @param coreE The core energy in Ha.
 * @param orbDiff Reference for orital difference, modifies in function.
 * @return CItype The Hamiltonian element.
 */
CItype Hij(Determinant& bra, Determinant& ket, oneInt& I1, twoInt& I2,
           double& coreE, size_t& orbDiff) {
  int cre[2], des[2], ncre = 0, ndes = 0;  // JETS: changed size of cre

  long ua, ba, ka;
  long ub, bb, kb;
  cre[0] = -1;
  cre[1] = -1;
  des[0] = -1;
  des[1] = -1;

  for (int i = 0; i < Determinant::EffDetLen; i++) {
    // Alpha excitations
    ua = bra.reprA[i] ^ ket.reprA[i];
    ba = ua & bra.reprA[i];  // the cre bits
    ka = ua & ket.reprA[i];  // the des bits
    GetLadderOps(ba, cre, ncre, i, false);
    GetLadderOps(ka, des, ndes, i, false);

    // Beta excitations
    ub = bra.reprB[i] ^ ket.reprB[i];
    bb = ub & bra.reprB[i];  // the cre bits
    kb = ub & ket.reprB[i];  // the des bits
    GetLadderOps(bb, cre, ncre, i, true);
    GetLadderOps(kb, des, ndes, i, true);
  }

  size_t N = Determinant::n_spinorbs;
  if (ncre == 0) {
    cout << bra << endl;
    cout << ket << endl;
    std::cout << "Use the function for energy!" << std::endl;
    exit(EXIT_FAILURE);
  } else if (ncre == 1) {
    size_t c0 = cre[0], d0 = des[0];
    orbDiff = c0 * N + d0;
    return ket.Hij_1Excite(cre[0], des[0], I1, I2);
  } else if (ncre == 2) {
    size_t c0 = cre[0], c1 = cre[1], d1 = des[1], d0 = des[0];
    orbDiff = c1 * N * N * N + d1 * N * N + c0 * N + d0;
    return ket.Hij_2Excite(des[0], des[1], cre[0], cre[1], I1, I2);
  } else {
    // std::cout << std::endl
    //           << ncre << std::endl
    //           << bra << std::endl
    //           << ket << std::endl; // JETS: rm
    return 0.;
  }
}

double EnergyAfterExcitation(vector<int>& closed, int& nclosed, oneInt& I1,
                             twoInt& I2, double& coreE, int i, int A,
                             double Energyd) {
  /*!
     Calculates the new energy of a determinant after single excitation.

     .. note:: Assumes that the spin of i and a orbitals is the same

     :Arguments:

     vector<int>& closed:
         Occupied orbitals in a vector.
     int& nclosed:
         Number of occupied orbitals.
     oneInt& I1:
         One body integrals.
     twoInt& I2:
         Two body integrals.
     double& coreE:
         Core energy.
     int i:
         Orbital index for destruction operator.
     int A:
         Orbital index for creation operator.
     double Energyd:
         Old determinant energy.

     :Returns:

      double E:
          Energy after excitation.
   */

  double E = Energyd;
#ifdef Complex
  E += -I1(closed[i], closed[i]).real() + I1(A, A).real();
#else
  E += -I1(closed[i], closed[i]) + I1(A, A);
#endif

  for (int I = 0; I < nclosed; I++) {
    if (I == i) continue;
    E = E - I2.Direct(closed[I] / 2, closed[i] / 2) +
        I2.Direct(closed[I] / 2, A / 2);
    if ((closed[I] % 2) == (closed[i] % 2))
      E = E + I2.Exchange(closed[I] / 2, closed[i] / 2) -
          I2.Exchange(closed[I] / 2, A / 2);
  }
  return E;
}

// Assumes that the spin of i and a orbitals is the same
// and the spins of j and b orbitals is the same
//=============================================================================
double EnergyAfterExcitation(vector<int>& closed, int& nclosed, oneInt& I1,
                             twoInt& I2, double& coreE, int i, int A, int j,
                             int B, double Energyd) {
  /*!
     Calculates the new energy of a determinant after double excitation. i -> A
     and j -> B.

     .. note:: Assumes that the spin of each orbital pair (i-A and j-B) is the
     same.

     :Arguments:

     vector<int>& closed:
         Occupied orbitals in a vector.
     int& nclosed:
         Number of occupied orbitals.
     oneInt& I1:
         One body integrals.
     twoInt& I2:
         Two body integrals.
     double& coreE:
         Core energy.
     int i:
         Orbital index for destruction operator.
     int j:
         Orbital index for destruction operator.
     int A:
         Orbital index for creation operator.
     int B:
         Orbital index for creation operator.
     double Energyd:
         Old determinant energy.

     :Returns:

      double E:
          Energy after excitation.
   */

#ifdef Complex
  double E = Energyd - (I1(closed[i], closed[i]) - I1(A, A) +
                        I1(closed[j], closed[j]) - I1(B, B))
                           .real();
#else
  double E = Energyd - I1(closed[i], closed[i]) + I1(A, A) -
             I1(closed[j], closed[j]) + I1(B, B);
#endif

  for (int I = 0; I < nclosed; I++) {
    if (I == i) continue;
    E = E - I2.Direct(closed[I] / 2, closed[i] / 2) +
        I2.Direct(closed[I] / 2, A / 2);
    if ((closed[I] % 2) == (closed[i] % 2))
      E = E + I2.Exchange(closed[I] / 2, closed[i] / 2) -
          I2.Exchange(closed[I] / 2, A / 2);
  }

  for (int I = 0; I < nclosed; I++) {
    if (I == i || I == j) continue;
    E = E - I2.Direct(closed[I] / 2, closed[j] / 2) +
        I2.Direct(closed[I] / 2, B / 2);
    if ((closed[I] % 2) == (closed[j] % 2))
      E = E + I2.Exchange(closed[I] / 2, closed[j] / 2) -
          I2.Exchange(closed[I] / 2, B / 2);
  }

  E = E - I2.Direct(A / 2, closed[j] / 2) + I2.Direct(A / 2, B / 2);
  if ((closed[i] % 2) == (closed[j] % 2))
    E = E + I2.Exchange(A / 2, closed[j] / 2) - I2.Exchange(A / 2, B / 2);

  return E;
}

/**
 * @brief Compute the parity of the determinant exciting from start -> end. Used
 * mainly in SHCI.
 *
 * @param start Destruction operator IN SPIN ORBITAL BASIS.
 * @param end  Creation operator IN SPIN ORBITAL BASIS.
 * @param parity Parity of excitation, modified in function.
 */
void Determinant::parity(const int& start, const int& end, double& parity) {
  int a_start = (start % 2 == 0) ? start / 2 : start / 2 + 1;
  int a_end = (end % 2 == 0) ? end / 2 : end / 2 + 1;
  int b_start = start / 2;
  int b_end = end / 2;
  int n_elec = getNalphaBefore(a_start) + getNbetaBefore(b_start) +
               getNalphaBefore(a_end) + getNbetaBefore(b_end);
  parity *= (n_elec % 2 == 0) ? 1. : -1.;

  if (getocc(start)) {
    parity *= -1;
  }
  return;
}

/**
 * @brief Calculate the parity of a double excitation. The indices should be
 * given in spin orbital basis. The total string of creation and annihilation
 * operators is :math:`a^\dagger_a a^\dagger_b a_j a_i`.
 *
 * @param i First destruction operator.
 * @param j Second destruction operator.
 * @param a Creation operator paired with i.
 * @param b Creation operator paired with j.
 * @param sgn Parity of excitation, modified by function.
 */
void Determinant::parity(int& i, int& j, int& a, int& b, double& sgn) {
  parity(min(i, a), max(i, a), sgn);
  setocc(i, false);
  setocc(a, true);
  parity(min(j, b), max(j, b), sgn);
  setocc(i, true);
  setocc(a, false);
  return;
}

// Gamma = c0 c1 c2 d0 d1 d2
// d2 -> c0   d1 -> c1   d0 -> c2
// Always set true last so if there are duplicates the last operation doesn't
// depopulate determinants.
void Determinant::parity(int& c0, int& c1, int& c2, int& d0, int& d1, int& d2,
                         double& sgn) {
  parity(min(d2, c0), max(d2, c0), sgn);
  setocc(d2, false);
  setocc(c0, true);
  parity(min(d1, c1), max(d1, c1), sgn);
  setocc(d1, false);
  setocc(c1, true);
  parity(min(d0, c2), max(d0, c2), sgn);
  setocc(c1, false);
  setocc(d1, true);
  setocc(c0, false);
  setocc(d2, true);
  return;
}

// Gamma = c0 c1 c2 c3 d0 d1 d2 d3
// Do NOT use with matching c and d pairs.
void Determinant::parity(int& c0, int& c1, int& c2, int& c3, int& d0, int& d1,
                         int& d2, int& d3, double& sgn) {
  parity(min(d3, c0), max(d3, c0), sgn);
  setocc(d3, false);
  setocc(c0, true);

  parity(min(d2, c1), max(d2, c1), sgn);
  setocc(d2, false);
  setocc(c1, true);

  parity(min(d1, c2), max(d1, c2), sgn);
  setocc(d1, false);
  setocc(c2, true);
  parity(min(d0, c3), max(d0, c3), sgn);

  setocc(c2, false);
  setocc(d1, true);
  setocc(c1, false);
  setocc(d2, true);
  setocc(c0, false);
  setocc(d3, true);
  return;
}

size_t hash_value(Determinant const& d) {
  std::size_t seed = 0;
  boost::hash_combine(seed, d.reprA[0] * 2654435761);
  boost::hash_combine(seed, d.reprB[0] * 2654435761);
  // JETS: The code below was commented out, but seems more general than the
  // code above.
  // for (int i = 0; i < DetLen; i++) {
  //  boost::hash_combine(seed, d.reprA[i]);
  //  boost::hash_combine(seed, d.reprB[i]);
  //}
  return seed;
}