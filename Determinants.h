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
#ifndef Determinants_HEADER_H
#define Determinants_HEADER_H

#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <boost/functional/hash.hpp>
#include <boost/serialization/serialization.hpp>

#include "global.h"

class oneInt;
class twoInt;
class twoIntHeatBathSHM;
class workingArray;

using namespace std;

inline int CountNonZeroBits(long x) {
  x = (x & 0x5555555555555555ULL) + ((x >> 1) & 0x5555555555555555ULL);
  x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
  x = (x & 0x0F0F0F0F0F0F0F0FULL) + ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL);
  return (x * 0x0101010101010101ULL) >> 56;
}

inline int BitCount(long x) {  // This should be redundant now -JETS
  x = (x & 0x5555555555555555ULL) + ((x >> 1) & 0x5555555555555555ULL);
  x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
  x = (x & 0x0F0F0F0F0F0F0F0FULL) + ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL);
  return (x * 0x0101010101010101ULL) >> 56;
}

class HalfDet {  // TODO clean this up
 private:
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    for (int i = 0; i < DetLen; i++) ar& repr[i];
  }

 public:
  long repr[DetLen];
  static int norbs;  // JETS: spin orbs + redundant with Determinant::n_spinorbs
  HalfDet() {
    for (int i = 0; i < DetLen; i++) repr[i] = 0;
  }

  // the comparison between determinants is performed
  bool operator<(const HalfDet& d) const {
    for (int i = DetLen - 1; i >= 0; i--) {
      if (repr[i] < d.repr[i])
        return true;
      else if (repr[i] > d.repr[i])
        return false;
    }
    return false;
  }

  bool operator==(const HalfDet& d) const {
    for (int i = DetLen - 1; i >= 0; i--)
      if (repr[i] != d.repr[i]) return false;
    return true;
  }

  int ExcitationDistance(const HalfDet& d) const {
    int ndiff = 0;
    for (int i = 0; i < DetLen; i++) {
      ndiff += BitCount(repr[i] ^ d.repr[i]);
    }
    return ndiff / 2;
  }

  // set the occupation of the ith orbital
  void setocc(int i, bool occ) {
    // assert(i< norbs);
    long Integer = i / 64, bit = i % 64, one = 1;
    if (occ)
      repr[Integer] |= one << bit;
    else
      repr[Integer] &= ~(one << bit);
  }

  // get the occupation of the ith orbital
  bool getocc(int i) const {
    // assert(i< norbs);
    long Integer = i / 64, bit = i % 64, reprBit = repr[Integer];
    if ((reprBit >> bit & 1) == 0)
      return false;
    else
      return true;
  }

  int getClosed(vector<int>& closed) {
    int cindex = 0;
    for (int i = 0; i < 64 * DetLen; i++) {
      if (getocc(i)) {
        closed.at(cindex) = i;
        cindex++;
      }
    }
    return cindex;
  }

  int getOpenClosed(vector<int>& open, vector<int>& closed) {
    int cindex = 0;
    int oindex = 0;
    for (int i = 0; i < 64 * DetLen; i++) {
      if (getocc(i)) {
        closed.at(cindex) = i;
        cindex++;
      } else {
        open.at(oindex) = i;
        oindex++;
      }
    }
    return cindex;
  }

  friend ostream& operator<<(ostream& os, const HalfDet& d) {
    char det[norbs / 2];
    d.getRepArray(det);
    for (int i = 0; i < norbs / 2; i++) os << (int)(det[i]) << " ";
    return os;
  }

  void getRepArray(char* repArray) const {
    for (int i = 0; i < norbs / 2; i++) {
      if (getocc(i))
        repArray[i] = 1;
      else
        repArray[i] = 0;
    }
  }
};

/**
 * This is the occupation number representation of a Determinants
 * with alpha, beta strings
 */
class Determinant {
 private:
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    for (int i = 0; i < DetLen; i++) ar& reprA[i] & reprB[i];
  }

 public:
  // 0th position of 0th long is the first position
  // 63rd position of the last long is the last position
  long reprA[DetLen], reprB[DetLen];
  static int EffDetLen;
  static char Trev;
  static int norbs;
  static int n_spinorbs;
  static int nalpha, nbeta;
  static Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> LexicalOrder;

  // Constructors
  Determinant();
  Determinant(const Determinant& d);
  void operator=(const Determinant& d);

  // mutates the Determinant
  void setoccA(int i, bool occ);
  void setoccB(int i, bool occ);
  void setocc(int i, bool occ);  // i is the spin orbital index
  void setocc(
      int i, bool sz,
      bool occ);  // i is the spatial orbital index, sz=0 for alpha, =1 for beta

  bool getocc(int i) const;           // i is the spin orbital index
  bool getocc(int i, bool sz) const;  // i is the spatial orbital index
  bool getoccA(int i) const;
  bool getoccB(int i) const;
  void getOpenClosed(std::vector<int>& open,
                     std::vector<int>& closed) const;  // gets spin orbitals
  void getOpenClosed(bool sz, std::vector<int>& open,
                     std::vector<int>& closed) const;  // gets spatial orbitals
  void getOpenClosedAlphaBeta(std::vector<int>& openAlpha,
                              std::vector<int>& closedAlpha,
                              std::vector<int>& openBeta,
                              std::vector<int>& closedBeta) const;
  void getClosedAlphaBeta(std::vector<int>& closedAlpha,
                          std::vector<int>& closedBeta) const;
  void getAlphaBeta(std::vector<int>& alpha, std::vector<int>& beta) const;
  void getClosed(bool sz, std::vector<int>& closed) const;
  int getNbetaBefore(int i) const;
  int getNalphaBefore(int i) const;
  int Noccupied() const;
  int Nalpha() const;
  int Nbeta() const;

  //
  // From Dice
  //
  void flipAlphaBeta() {
    long tmp;
    for (int i = 0; i < DetLen; i++) {
      tmp = reprA[i];
      reprA[i] = reprB[i];
      reprB[i] = tmp;
    }
  }

  /**
   * @brief Check whether flipping the the bit representation (in the older Dice
   * style of a,b,a,b..) makes the numerical value of the representation greater
   * or smaller. If it's larger, then we're already in the standard
   * representation.
   *
   * @return true
   * @return false
   */
  bool isStandard() {
    if (!hasUnpairedElectrons()) return true;
    for (int i = EffDetLen - 1; i >= 0; i--) {
      if (reprA[i] > reprB[i]) {
        return false;

      } else if (reprA[i] < reprB[i]) {
        return true;
      }
    }
    cout << "Error finding standard for determinant " << *this << endl;
    cout << hasUnpairedElectrons() << endl;
    exit(EXIT_FAILURE);
  }

  bool hasUnpairedElectrons() {
    for (int i = 0; i < DetLen; i++) {
      if (reprA[i] != reprB[i]) {
        return true;
      }
    }
    return false;
  }

  void makeStandard() {
    if (!isStandard()) flipAlphaBeta();
  }

  // I think these are obsolete now since we're using the boost hash functions
  static void initLexicalOrder(int nelec);

  size_t getHash() {
    size_t seed = 0;
    boost::hash_combine(seed, reprA[0] * 2654435761);
    boost::hash_combine(seed, reprB[0] * 2654435761);
    return seed;
  }

  double parityOfFlipAlphaBeta() {
    long tmp;
    int n_paired = 0;
    for (int i = 0; i < DetLen; i++) {
      if (reprA[i] != reprB[i]) {
        tmp = reprA[i] & reprB[i];
        n_paired += CountNonZeroBits(tmp);
      }
    }

    double parity = n_paired % 2 == 0 ? 1.0 : -1.0;

    if (Nbeta() % 2 == 0) {
      return parity;
    } else {
      return -1. * parity;
    }
  }

  CItype Hij_1Excite(int a, int i, oneInt& I1, twoInt& I2);

  CItype Hij_2Excite(int& i, int& j, int& a, int& b, oneInt& I1, twoInt& I2);

  int numUnpairedElectrons() {
    // unsigned long even = 0x5555555555555555, odd = 0xAAAAAAAAAAAAAAAA;
    int unpairedElecs = 0;
    unsigned long unpaired;
    for (int i = EffDetLen - 1; i >= 0; i--) {
      // unsigned long unpaired = (((repr[i] & even) << 1) ^ (repr[i] & odd));
      unpaired = reprA[i] ^ reprB[i];
      unpairedElecs += CountNonZeroBits(unpaired);
    }
    return unpairedElecs;
  }

  void parity(const int& start, const int& end, double& parity);
  void parity(int& i, int& j, int& a, int& b, double& sgn);
  void parity(int& c0, int& c1, int& c2, int& d0, int& d1, int& d2,
              double& sgn);
  void parity(int& c0, int& c1, int& c2, int& c3, int& d0, int& d1, int& d2,
              int& d3, double& sgn);

  // Get HalfDet with just the alpha string
  HalfDet getAlpha() const {
    HalfDet d;
    for (int i = 0; i < EffDetLen; i++)
      for (int j = 0; j < 64; j++) {
        d.setocc(i * 64 + j, getocc(i * 64 + j * 2));
      }
    return d;
  }

  // get HalfDet with just the beta string
  HalfDet getBeta() const {
    HalfDet d;
    for (int i = 0; i < EffDetLen; i++)
      for (int j = 0; j < 64; j++)
        d.setocc(i * 64 + j, getocc(i * 64 + j * 2 + 1));
    return d;
  }
  //
  // end From Dice
  //

  double parityA(const int& a, const int& i) const;
  double parityB(const int& a, const int& i) const;
  double parity(const int& a, const int& i, const bool& sz) const;
  double parityA(const vector<int>& aArray, const vector<int>& iArray) const;
  double parityB(const vector<int>& aArray, const vector<int>& iArray) const;
  double parity(const vector<int>& aArray, const vector<int>& iArray,
                bool sz) const;
  double parityAA(const int& i, const int& j, const int& a, const int& b) const;
  double parityBB(const int& i, const int& j, const int& a, const int& b) const;

  double Energy(const oneInt& I1, const twoInt& I2, const double& coreE) const;
  CItype Hij_1ExciteScreened(const int& a, const int& i,
                             const twoIntHeatBathSHM& Ishm, const double& TINY,
                             bool doparity = true) const;
  CItype Hij_1ExciteA(const int& a, const int& i, const oneInt& I1,
                      const twoInt& I2, bool doparity = true) const;
  CItype Hij_1ExciteB(const int& a, const int& i, const oneInt& I1,
                      const twoInt& I2, bool doparity = true) const;
  CItype Hij_2ExciteAA(const int& a, const int& i, const int& b, const int& j,
                       const oneInt& I1, const twoInt& I2) const;
  CItype Hij_2ExciteBB(const int& a, const int& i, const int& b, const int& j,
                       const oneInt& I1, const twoInt& I2) const;
  CItype Hij_2ExciteAB(const int& a, const int& i, const int& b, const int& j,
                       const oneInt& I1, const twoInt& I2) const;

  bool connected(const Determinant& d) const;
  int ExcitationDistance(const Determinant& d) const;

  // operators
  bool operator<(const Determinant& d) const;
  bool operator==(const Determinant& d) const;
  friend ostream& operator<<(ostream& os, const Determinant& d);
};

// instead of storing memory in bits it uses 1 integer per bit
// so it is clearly very exepnsive. This is only used during computations
class BigDeterminant {
 public:
  vector<char> occupation;
  BigDeterminant(const Determinant& d);
  BigDeterminant(const BigDeterminant& d) : occupation(d.occupation){};
  const char& operator[](int j) const;
  char& operator[](int j);
};

// note some of i, j, k, l might be repeats
// and some its possible that the determinant might get killed
// the return value tell us whether the determinant is killed
bool applyExcitation(int a, int b, int k, int l, Determinant& dcopy);

CItype Hij(const Determinant& bra, const Determinant& ket, const oneInt& I1,
           const twoInt& I2, const double& coreE);

void getDifferenceInOccupation(const Determinant& bra, const Determinant& ket,
                               vector<int>& creA, vector<int>& desA,
                               vector<int>& creB, vector<int>& desB);
void getDifferenceInOccupation(const Determinant& bra, const Determinant& ket,
                               vector<int>& cre, vector<int>& des, bool sz);
void getDifferenceInOccupation(const Determinant& bra, const Determinant& ket,
                               int& I, int& A);
void getDifferenceInOccupation(const Determinant& bra, const Determinant& ket,
                               int& I, int& J, int& A, int& B);

double getParityForDiceToAlphaBeta(const Determinant& det);

void generateScreenedSingleDoubleExcitation(const Determinant& d,
                                            const double& THRESH,
                                            const double& TINY,
                                            workingArray& work, bool doparity);

void generateAllScreenedDoubleExcitation(const Determinant& det,
                                         const double& screen,
                                         const double& TINY, workingArray& work,
                                         bool doparity = false);

void generateAllScreenedSingleExcitation(const Determinant& det,
                                         const double& screen,
                                         const double& TINY, workingArray& work,
                                         bool doparity = false);

void comb(int N, int K, vector<vector<int>>& combinations);

void generateAllDeterminants(vector<Determinant>& allDets, int norbs,
                             int nalpha, int nbeta);

// From Dice
void updateHijForTReversal(CItype& hij, Determinant& dj, Determinant& dk,
                           oneInt& I1, twoInt& I2, double& coreE,
                           size_t& orbDiff);

// This is used to store just the alpha or the beta sub string of the entire
// determinant

size_t hash_value(Determinant const& d);
void GetLadderOps(long bits, int ladder_ops[], int& n_ops, int ith_rep,
                  bool beta);

CItype Hij_1Excite(int a, int i, oneInt& I1, twoInt& I2, int* closed,
                   int& nclosed);
double EnergyAfterExcitation(vector<int>& closed, int& nclosed, oneInt& I1,
                             twoInt& I2, double& coreE, int i, int A,
                             double Energyd);
double EnergyAfterExcitation(vector<int>& closed, int& nclosed, oneInt& I1,
                             twoInt& I2, double& coreE, int i, int A, int j,
                             int B, double Energyd);

void getOrbDiff(Determinant& bra, Determinant& ket, size_t& orbDiff);
CItype Hij(Determinant& bra, Determinant& ket, oneInt& I1, twoInt& I2,
           double& coreE, size_t& orbDiff);
#endif
