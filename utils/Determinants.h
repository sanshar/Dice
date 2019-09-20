/*
  Developed by Sandeep Sharma with contributions from James E. T. Smith and Adam A. Holmes, 2017
  Copyright (c) 2017, Sandeep Sharma

  This file is part of DICE.

  This program is free software: you can redistribute it and/or modify it under the terms
  of the GNU General Public License as published by the Free Software Foundation,
  either version 3 of the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with this program.
  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef Determinants_HEADER_H
#define Determinants_HEADER_H

#include "global.h"
#include <iostream>
#include <vector>
#include <boost/serialization/serialization.hpp>
#include <boost/functional/hash.hpp>
#include <Eigen/Dense>


class oneInt;
class twoInt;
class twoIntHeatBathSHM;
class workingArray;

using namespace std;

inline int CountNonZeroBits (long x)
{
  x = (x & 0x5555555555555555ULL) + ((x >> 1) & 0x5555555555555555ULL);
  x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
  x = (x & 0x0F0F0F0F0F0F0F0FULL) + ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL);
  return (x * 0x0101010101010101ULL) >> 56;
}

// Just the determinant bit string, stored in a contiguous array
typedef std::array<long, 2*DetLen> simpleDet;

/**
* This is the occupation number representation of a Determinants
* with alpha, beta strings
*/
class Determinant {

 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    for (int i=0; i<DetLen; i++)
      ar & reprA[i] & reprB[i];
  }

 public:
  // 0th position of 0th long is the first position
  // 63rd position of the last long is the last position
  long reprA[DetLen], reprB[DetLen];
  static int EffDetLen;
  static char Trev;
  static int norbs;
  static int nalpha, nbeta;
  static Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> LexicalOrder;


  //Constructors
  Determinant();
  Determinant(const Determinant& d);

  Determinant(const simpleDet combined) {
    for (int i=0; i<DetLen; i++) {
      reprA[i] = combined[i];
    }
    for (int i=0; i<DetLen; i++) {
      reprB[i] = combined[i+DetLen];
    }
  }

  void operator=(const Determinant& d);

  //mutates the Determinant
  void setoccA(int i, bool occ);
  void setoccB(int i, bool occ);
  void setocc(int i, bool occ) ; //i is the spin orbital index
  void setocc(int i, bool sz, bool occ) ; //i is the spatial orbital index, sz=0 for alpha, =1 for beta
  
  bool getocc(int i) const ;//i is the spin orbital index
  bool getocc(int i, bool sz) const ;//i is the spatial orbital index 
  bool getoccA(int i) const ;
  bool getoccB(int i) const;
  void getOpenClosed( std::vector<int>& open, std::vector<int>& closed) const;//gets spin orbitals
  void getOpenClosed( bool sz, std::vector<int>& open, std::vector<int>& closed) const;//gets spatial orbitals
  void getOpenClosedAlphaBeta( std::vector<int>& openAlpha,
                               std::vector<int>& closedAlpha,
                               std::vector<int>& openBeta,
                               std::vector<int>& closedBeta ) const;
  void getClosedAlphaBeta( std::vector<int>& closedAlpha,
                           std::vector<int>& closedBeta ) const;
  void getAlphaBeta(std::vector<int>& alpha, std::vector<int>& beta) const;
  void getClosed(bool sz, std::vector<int>& closed) const;
  int getNbetaBefore(int i) const;
  int getNalphaBefore(int i) const;
  int Noccupied() const;
  int Nalpha() const;
  int Nbeta() const;

  
  double parityA(const int& a, const int& i) const;
  double parityB(const int& a, const int& i) const;
  double parity(const int& a, const int& i, const bool& sz) const;
  double parityA(const vector<int>& aArray, const vector<int>& iArray) const ;
  double parityB(const vector<int>& aArray, const vector<int>& iArray) const ;
  double parity(const vector<int>& aArray, const vector<int>& iArray, bool sz) const ;
  double parityAA(const int& i, const int& j, const int& a, const int& b) const ;
  double parityBB(const int& i, const int& j, const int& a, const int& b) const ;

  
  double Energy(const oneInt& I1, const twoInt& I2, const double& coreE) const ;
  CItype Hij_1ExciteScreened(const int& a, const int& i, const twoIntHeatBathSHM& Ishm,
                             const double& TINY, bool doparity = true) const;
  CItype Hij_1ExciteA(const int& a, const int& i, const oneInt&I1,  const twoInt& I2,
                      bool doparity=true) const ;
  CItype Hij_1ExciteB(const int& a, const int& i, const oneInt&I1, const twoInt& I2,
                      bool doparity=true) const ;
  CItype Hij_2ExciteAA(const int& a, const int& i, const int& b, const int& j,
                       const oneInt&I1, const twoInt& I2) const ;
  CItype Hij_2ExciteBB(const int& a, const int& i, const int& b, const int& j,
                       const oneInt&I1, const twoInt& I2) const ;
  CItype Hij_2ExciteAB(const int& a, const int& i, const int& b, const int& j,
                       const oneInt&I1, const twoInt& I2) const ;



  bool connected(const Determinant& d) const;
  int ExcitationDistance(const Determinant& d) const;

  //operators
  bool operator<(const Determinant& d) const;
  bool operator==(const Determinant& d) const;
  friend ostream& operator<<(ostream& os, const Determinant& d);
  friend size_t hash_value(Determinant const& d);


  // Get unique processor label for this determinant
  int getProc() const {
    std::size_t h_tot = 0;
    for (int i=0; i<DetLen; i++) {
      boost::hash_combine(h_tot, reprA[i]);
      boost::hash_combine(h_tot, reprB[i]);
    }
    return h_tot % commsize;
  }

  // Return simplified version of determinant
  simpleDet getSimpleDet() const {
    simpleDet combined;
    for (int i=0; i<DetLen; i++) {
      combined[i] = reprA[i];
    }
    for (int i=0; i<DetLen; i++) {
      combined[i+DetLen] = reprB[i];
    }
    return combined;
  }

};

//instead of storing memory in bits it uses 1 integer per bit
//so it is clearly very exepnsive. This is only used during computations
class BigDeterminant {  
 public:
  vector<char> occupation;
  BigDeterminant(const Determinant& d);
  BigDeterminant(const BigDeterminant& d) : occupation(d.occupation){};
  const char& operator[] (int j) const ;
  char& operator[] (int j) ;
};


//note some of i, j, k, l might be repeats
//and some its possible that the determinant might get killed
//the return value tell us whether the determinant is killed
bool applyExcitation(int a, int b, int k, int l, Determinant& dcopy);

CItype Hij(const Determinant& bra, const Determinant& ket,
           const oneInt& I1, const twoInt& I2, const double& coreE);

void getDifferenceInOccupation(const Determinant &bra, const Determinant &ket,
                               vector<int> &creA, vector<int> &desA,
                               vector<int> &creB, vector<int> &desB);
void getDifferenceInOccupation(const Determinant &bra, const Determinant &ket,
                               vector<int> &cre, vector<int> &des,
                               bool sz);
void getDifferenceInOccupation(const Determinant &bra, const Determinant &ket, int &I, int &A);
void getDifferenceInOccupation(const Determinant &bra, const Determinant &ket,
                               int &I, int &J, int& A, int& B);

double getParityForDiceToAlphaBeta(const Determinant& det);

//---Generate all screened excitations---------------------------

void generateAllScreenedSingleExcitation(const Determinant& det,
                                         const double& screen,
                                         const double& TINY,
                                         workingArray& work,
                                         bool doparity = false);

void generateAllScreenedDoubleExcitation(const Determinant& det,
                                         const double& screen,
                                         const double& TINY,
                                         workingArray& work,
                                         bool doparity = false);

//---Generate all screened excitations in the FOIS---------------

void generateAllScreenedDoubleExcitationsFOIS(const Determinant& det,
                                         const double& screen,
                                         const double& TINY,
                                         workingArray& work,
                                         bool doparity = false);

void generateAllScreenedSingleExcitationsDyall(const Determinant& det,
                                         const Determinant& detAct,
                                         const double& screen,
                                         const double& TINY,
                                         workingArray& work,
                                         bool doparity = false);

void generateAllScreenedDoubleExcitationsDyall(const Determinant& det,
                                         const double& screen,
                                         const double& TINY,
                                         workingArray& work,
                                         bool doparity = false);


//---Generate all screened excitations into the CAS-------------------

//---From excitation class 0 (the CAS itself) into the CAS------------
void generateAllScreenedSingleExcitationsCAS_0h0p(const Determinant& det,
                                         const double& screen,
                                         const double& TINY,
                                         workingArray& work,
                                         bool doparity = false);

//---From excitation class 0 (the CAS itself) into the CAS------------
void generateAllScreenedDoubleExcitationsCAS_0h0p(const Determinant& det,
                                         const double& screen,
                                         workingArray& work);

//---From excitation class 1 (0 holes in core, 1 particle in virtuals) into the CAS
void generateAllScreenedSingleExcitationsCAS_0h1p(const Determinant& det,
                                         const double& screen,
                                         const double& TINY,
                                         workingArray& work,
                                         const int& i,
                                         bool doparity = false);

//---From excitation class 1 (0 holes in core, 1 particle in virtuals) into the CAS
void generateAllScreenedDoubleExcitationsCAS_0h1p(const Determinant& det,
                                         const double& screen,
                                         workingArray& work,
                                         const int& i);

//---From excitation class 2 (0 holes in core, 2 particles in virtuals) into the CAS
void generateAllScreenedExcitationsCAS_0h2p(const Determinant& det,
                                         const double& screen,
                                         workingArray& work,
                                         const int& iExc, const int& jExc);

//---From excitation class 3 (1 hole in core, 0 particles in virtuals) into the CAS
void generateAllScreenedSingleExcitationsCAS_1h0p(const Determinant& det,
                                         const double& screen,
                                         const double& TINY,
                                         workingArray& work,
                                         const int& a,
                                         bool doparity = false);

//---From excitation class 3 (1 hole in core, 0 particles in virtuals) into the CAS
void generateAllScreenedDoubleExcitationsCAS_1h0p(const Determinant& det,
                                         const double& screen,
                                         workingArray& work,
                                         const int& a);

//---From excitation class 4 (1 hole in core, 1 particle in virtuals) into the CAS
void generateAllScreenedSingleExcitationsCAS_1h1p(const Determinant& det,
                                         const double& screen,
                                         const double& TINY,
                                         workingArray& work,
                                         const int& i, const int& a,
                                         bool doparity = false);

//---From excitation class 4 (1 hole in core, 1 particle in virtuals) into the CAS
void generateAllScreenedDoubleExcitationsCAS_1h1p(const Determinant& det,
                                         const double& screen,
                                         workingArray& work,
                                         const int& i, const int& a);

//---From excitation class 5 (1 hole in core, 2 particles in virtuals) into the CAS
void generateAllScreenedExcitationsCAS_1h2p(const Determinant& det,
                                         const double& screen,
                                         workingArray& work,
                                         const int& iExc, const int& jExc,
                                         const int& a);

//---From excitation class 6 (2 holes in core, 0 particles in virtuals) into the CAS
void generateAllScreenedExcitationsCAS_2h0p(const Determinant& det,
                                         const double& screen,
                                         workingArray& work,
                                         const int& aExc, const int& bExc);

//---From excitation class 7 (2 holes in core, 1 particle in virtuals) into the CAS
void generateAllScreenedExcitationsCAS_2h1p(const Determinant& det,
                                         const double& screen,
                                         workingArray& work,
                                         const int& iExc,
                                         const int& aExc, const int& bExc);

//---From excitation class 8 (2 holes in core, 2 particles in virtuals) into the CAS
void generateAllScreenedExcitationsCAS_2h2p(const double& screen,
                                         workingArray& work,
                                         const int& iExc, const int& jExc,
                                         const int& aExc, const int& bExc);

void comb(int N, int K, vector<vector<int>> &combinations);

void generateAllDeterminants(vector<Determinant>& allDets, int norbs, int nalpha, int nbeta);
void generateAllDeterminantsFOIS(vector<Determinant>& allDets, int norbs, int nalpha, int nbeta);

template<> struct hash<Determinant>
{
  std::size_t operator()(Determinant const& d) const noexcept
  {
    std::size_t h_tot = 0;
    for (int i=0; i<DetLen; i++) {
      boost::hash_combine(h_tot, d.reprA[i]);
      boost::hash_combine(h_tot, d.reprB[i]);
    }
    return h_tot;
  }
};

#endif
