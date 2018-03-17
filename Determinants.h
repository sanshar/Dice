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

using namespace std;

inline int BitCount (long x)
{
  x = (x & 0x5555555555555555ULL) + ((x >> 1) & 0x5555555555555555ULL);
  x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
  x = (x & 0x0F0F0F0F0F0F0F0FULL) + ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL);
  return (x * 0x0101010101010101ULL) >> 56;

  //unsigned int u2=u>>32, u1=u;

  //return __builtin_popcount(u2)+__builtin_popcount(u);
  /*
  u1 = u1
    - ((u1 >> 1) & 033333333333)
    - ((u1 >> 2) & 011111111111);


  u2 = u2
    - ((u2 >> 1) & 033333333333)
    - ((u2 >> 2) & 011111111111);

  return (((u1 + (u1 >> 3))
	   & 030707070707) % 63) +
    (((u2 + (u2 >> 3))
      & 030707070707) % 63);
  */
}


//This is used to store just the alpha or the beta sub string of the entire determinant
class HalfDet {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    for (int i=0; i<DetLen/2; i++)
      ar & repr[i];
  }
 public:
  long repr[DetLen/2];
  static int norbs;
  HalfDet() {
    for (int i=0; i<DetLen/2; i++)
      repr[i] = 0;
  }

  //the comparison between determinants is performed
  bool operator<(const HalfDet& d) const {
    for (int i=DetLen/2-1; i>=0 ; i--) {
      if (repr[i] < d.repr[i]) return true;
      else if (repr[i] > d.repr[i]) return false;
    }
    return false;
  }

  bool operator==(const HalfDet& d) const {
    for (int i=DetLen/2-1; i>=0 ; i--)
      if (repr[i] != d.repr[i]) return false;
    return true;
  }

  int ExcitationDistance(const HalfDet& d) const {
    int ndiff = 0; 
    for (int i=0; i<DetLen/2; i++) {
      ndiff += BitCount(repr[i] ^ d.repr[i]);
    }
    return ndiff/2;
  }

  //set the occupation of the ith orbital
  void setocc(int i, bool occ) {
    //assert(i< norbs);
    long Integer = i/64, bit = i%64, one=1;
    if (occ)
      repr[Integer] |= one << bit;
    else
      repr[Integer] &= ~(one<<bit);
  }

  //get the occupation of the ith orbital
  bool getocc(int i) const {
    //assert(i< norbs);
    long Integer = i/64, bit = i%64, reprBit = repr[Integer];
    if(( reprBit>>bit & 1) == 0)
      return false;
    else
      return true;
  }

  int getClosed(vector<int>& closed){
    int cindex = 0;
    for (int i=0; i<32*DetLen; i++) {
      if (getocc(i)) {closed.at(cindex) = i; cindex++;}
    }
    return cindex;
  }

  int getOpenClosed(vector<int>& open, vector<int>& closed){
    int cindex = 0;
    int oindex = 0;
    for (int i=0; i<32*DetLen; i++) {
      if (getocc(i)) {closed.at(cindex) = i; cindex++;}
      else {open.at(oindex) = i; oindex++;}
    }
    return cindex;
  }

  friend ostream& operator<<(ostream& os, const HalfDet& d) {
    char det[norbs/2];
    d.getRepArray(det);
    for (int i=0; i<norbs/2; i++)
      os<<(int)(det[i])<<" ";
    return os;
  }

  void getRepArray(char* repArray) const {
    for (int i=0; i<norbs/2; i++) {
      if (getocc(i)) repArray[i] = 1;
      else repArray[i] = 0;
    }
  }

};

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

  void getAlphaBeta(vector<int>& alpha, vector<int>& beta) {
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

  void parityAA(int& i, int& j, int& a, int& b, double& sgn);
  void parityBB(int& i, int& j, int& a, int& b, double& sgn);

  CItype Hij_1ExciteA(int& a, int& i, oneInt&I1, twoInt& I2);
  CItype Hij_1ExciteB(int& a, int& i, oneInt&I1, twoInt& I2);

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


};


CItype Hij(Determinant& bra, Determinant& ket, oneInt& I1, twoInt& I2, double& coreE);

#endif
