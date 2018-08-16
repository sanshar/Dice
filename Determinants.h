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
#include <Eigen/Dense>


class oneInt;
class twoInt;
class twoIntHeatBathSHM;

using namespace std;

inline int BitCount (long x)
{
  x = (x & 0x5555555555555555ULL) + ((x >> 1) & 0x5555555555555555ULL);
  x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
  x = (x & 0x0F0F0F0F0F0F0F0FULL) + ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL);
  return (x * 0x0101010101010101ULL) >> 56;
}




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

  Determinant() {
    for (int i=0; i<DetLen; i++) {
      reprA[i] = 0;
      reprB[i] = 0;
    }
  }

  Determinant(const Determinant& d) {
    for (int i=0; i<DetLen; i++) {
      reprA[i] = d.reprA[i];
      reprB[i] = d.reprB[i];
    }
  }

  void operator=(const Determinant& d) {
    for (int i=0; i<DetLen; i++) {
      reprA[i] = d.reprA[i];
      reprB[i] = d.reprB[i];
    }
  }

  void getOpenClosed( std::vector<int>& open, std::vector<int>& closed) {
    for (int i=0; i<norbs; i++) {
      if ( getoccA(i)) closed.push_back(2*i);
      else open.push_back(2*i);
      if ( getoccB(i)) closed.push_back(2*i+1);
      else open.push_back(2*i+1);
    }
  }

  void getOpenClosedAlphaBeta( std::vector<int>& openAlpha,
                               std::vector<int>& closedAlpha,
                               std::vector<int>& openBeta,
                               std::vector<int>& closedBeta
                             ) {
    for (int i=0; i<norbs; i++) {
      if ( getoccA(i)) closedAlpha.push_back(i);
      else openAlpha.push_back(i);
      if ( getoccB(i)) closedBeta.push_back(i);
      else openBeta.push_back(i);
    }
  }

  void getAlphaBeta(std::vector<int>& alpha, std::vector<int>& beta) {
    for (int i=0; i<64*EffDetLen; i++) {
      if (getoccA(i)) alpha.push_back(i);
      if (getoccB(i)) beta .push_back(i);
    }
  }

  double Energy(oneInt& I1, twoInt& I2, double& coreE);

  int getNbetaBefore(int i) {
    int occ = 0;
    for (int n = 0; n < i/64; n++) {
      occ += BitCount(reprB[n]);
    }
    long one = 1; long mask = ( one << (i%64) ) - one;
    long result = (reprB[i/64] & mask ) ;
    occ += BitCount(result);
    return occ;
  }

  int getNalphaBefore(int i) {
    int occ = 0;
    for (int n = 0; n < i/64; n++) {
      occ += BitCount(reprA[n]);
    }
    long one = 1; long mask = ( one << (i%64) ) - one;
    long result = (reprA[i/64] & mask ) ;
    occ += BitCount(result);
    return occ;
  }

  void parityA(int& a, int& i, double& parity) {
    int occ = getNalphaBefore(i);
    setoccA(i, false);
    occ += getNalphaBefore(a);
    setoccA(i, true);
    parity *= (occ%2==0) ? 1.: -1.;

    return;
  }

  void parityB(int& a, int& i, double& parity) {
    int occ = getNbetaBefore(i);
    setoccB(i, false);
    occ += getNbetaBefore(a);
    setoccB(i, true);
    parity *= (occ%2==0) ? 1.: -1.;

    return;
  }

  double parityA(vector<int>& aArray, vector<int>& iArray)
  {
    double p = 1.;
    Determinant dcopy = *this;
    for (int i = 0; i < iArray.size(); i++)
    {
      dcopy.parityA(aArray[i], iArray[i], p);

      dcopy.setoccA(iArray[i], false);
      dcopy.setoccA(aArray[i], true);
    }
    return p;
  }

  double parityB(vector<int>& aArray, vector<int>& iArray)
  {
    double p = 1.;
    Determinant dcopy = *this;
    for (int i = 0; i < iArray.size(); i++)
    {
      dcopy.parityB(aArray[i], iArray[i], p);

      dcopy.setoccB(iArray[i], false);
      dcopy.setoccB(aArray[i], true);
    }
    return p;
  }

  void parityAA(int& i, int& j, int& a, int& b, double& sgn);
  void parityBB(int& i, int& j, int& a, int& b, double& sgn);

  CItype Hij_1ExciteA(int& a, int& i, oneInt&I1, twoInt& I2, bool doparity=true);
  CItype Hij_1ExciteB(int& a, int& i, oneInt&I1, twoInt& I2, bool doparity=true);

  CItype Hij_2ExciteAA(int& a, int& i, int& b, int& j, oneInt&I1, twoInt& I2);
  CItype Hij_2ExciteBB(int& a, int& i, int& b, int& j, oneInt&I1, twoInt& I2);
  CItype Hij_2ExciteAB(int& a, int& i, int& b, int& j, oneInt&I1, twoInt& I2);


  int Noccupied() const {
    int nelec = 0;
    for (int i=0; i<DetLen; i++) {
      nelec += BitCount(reprA[i]);
      nelec += BitCount(reprB[i]);
    }
    return nelec;
  }

  int Nalpha() const {
    int nelec = 0;
    for (int i=0; i<DetLen; i++) {
      nelec += BitCount(reprA[i]);
    }
    return nelec;
  }

  int Nbeta() const {
    int nelec = 0;
    for (int i=0; i<DetLen; i++) {
      nelec += BitCount(reprB[i]);
    }
    return nelec;
  }

  //Is the excitation between *this and d less than equal to 2.
  bool connected(const Determinant& d) const {
    int ndiff = 0; long u;

    for (int i=0; i<DetLen; i++) {
      ndiff += BitCount(reprA[i] ^ d.reprA[i]);
      ndiff += BitCount(reprB[i] ^ d.reprB[i]);
    }
    return ndiff<=4;
      //return true;
  }


  //Get the number of electrons that need to be excited to get determinant d from *this determinant
  //e.g. single excitation will return 1
  int ExcitationDistance(const Determinant& d) const {
    int ndiff = 0;
    for (int i=0; i<DetLen; i++) {
      ndiff += BitCount(reprA[i] ^ d.reprA[i]);
      ndiff += BitCount(reprB[i] ^ d.reprB[i]);
    }
    return ndiff/2;
  }


  //the comparison between determinants is performed
  bool operator<(const Determinant& d) const {
    for (int i=DetLen-1; i>=0 ; i--) {
      if (reprA[i] < d.reprA[i]) return true;
      else if (reprA[i] > d.reprA[i]) return false;
      if (reprB[i] < d.reprB[i]) return true;
      else if (reprB[i] > d.reprB[i]) return false;
    }
    return false;
  }

  //check if the determinants are equal
  bool operator==(const Determinant& d) const {
    for (int i=DetLen-1; i>=0 ; i--) {
      if (reprA[i] != d.reprA[i]) return false;
      if (reprB[i] != d.reprB[i]) return false;
    }
    return true;
  }

  //set the occupation of the ith orbital
  void setoccA(int i, bool occ) {
    long Integer = i/64, bit = i%64, one=1;
    if (occ)
      reprA[Integer] |= one << bit;
    else
      reprA[Integer] &= ~(one<<bit);
  }

  //set the occupation of the ith orbital
  void setoccB(int i, bool occ) {
    long Integer = i/64, bit = i%64, one=1;
    if (occ)
      reprB[Integer] |= one << bit;
    else
      reprB[Integer] &= ~(one<<bit);
  }

  void setocc(int i, bool occ)  {
    if (i%2 == 0) return setoccA(i/2, occ);
    else return setoccB(i/2, occ);
  }

  bool getocc(int i) const {
    if (i%2 == 0) return getoccA(i/2);
    else return getoccB(i/2);
  }

  //get the occupation of the ith orbital
  bool getoccA(int i) const {
    //asser(i<norbs);
    long Integer = i/64, bit = i%64, reprBit = reprA[Integer];
    if(( reprBit>>bit & 1) == 0)
      return false;
    else
      return true;
  }

  bool getoccB(int i) const {
    //asser(i<norbs);
    long Integer = i/64, bit = i%64, reprBit = reprB[Integer];
    if(( reprBit>>bit & 1) == 0)
      return false;
    else
      return true;
  }


  //Prints the determinant
  friend ostream& operator<<(ostream& os, const Determinant& d) {
    for (int i=0; i<norbs; i++) {
      if (d.getoccA(i)==false && d.getoccB(i) == false)
	os<<0<<" ";
      else if (d.getoccA(i)==true && d.getoccB(i) == false)
	os<<"a"<<" ";
      else if (d.getoccA(i)==false && d.getoccB(i) == true)
	os<<"b"<<" ";
      else if (d.getoccA(i)==true && d.getoccB(i) == true)
	os<<2<<" ";
      if ( (i+1)%5 == 0)
	os <<"  ";
    }
    return os;
  }

// Calculate the number of singles reachable from the current determinant
//given that the screening is used
  int numberPossibleSingles(double& screen, oneInt& I1, twoInt& I2,
			    twoIntHeatBathSHM& I2hb);


};


CItype Hij(Determinant& bra, Determinant& ket, oneInt& I1, twoInt& I2, double& coreE);

void sampleSingleDoubleExcitation(Determinant& d,  oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb,
				  int nterms,
				  vector<int>& Isingle, vector<int>& Asingle,
				  vector<int>& Idouble, vector<int>& Adouble,
				  vector<int>& Jdouble, vector<int>& Bdouble,
				  vector<double>& psingle, vector<double>& pdouble);

void getOrbDiff(Determinant &bra, Determinant &ket, vector<int> &creA, vector<int> &desA,
                vector<int> &creB, vector<int> &desB);

double getParityForDiceToAlphaBeta(Determinant& det);
#endif
