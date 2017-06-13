/*
Developed by Sandeep Sharma with contributions from James E. Smith and Adam A. Homes, 2017
Copyright (c) 2017, Sandeep Sharma

This file is part of DICE.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
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

inline int DeterminantToIntegral(int i) {
  return (i < 64*(DetLen/2)) ? 2*i : 2*(i - 64*(DetLen/2))+1;
}

inline int IntegralToDeterminant(int i) {
  return i/2 + (DetLen/2)*64*(i%2);
}

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


//to the outside world 0,1,2,3... are 0a,0b,1a,1b orbitals (integral convention)
//but internally we store alpha beta strings (determinant convention)
class Determinant {

 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    for (int i=0; i<DetLen; i++)
      ar & repr[i];
  }

 public:
  // 0th position of 0th long is the first position
  // 63rd position of the last long is the last position
  long repr[DetLen];
  static char Trev;
  static int norbs;
  static int EffDetLen;
  static Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> LexicalOrder;

  Determinant() {
    for (int i=0; i<DetLen; i++)
      repr[i] = 0;
  }

  Determinant(const Determinant& d) {
    for (int i=0; i<DetLen; i++)
      repr[i] = d.repr[i];
  }

  void operator=(const Determinant& d) {
    for (int i=0; i<DetLen; i++)
      repr[i] = d.repr[i];
  }

  double Energy(oneInt& I1, twoInt& I2, double& coreE);
  static void initLexicalOrder(int nelec);

  //i and a are given with integral convention
  void parity(int& i, int& j, int& a, int& b, double& sgn) ;


  //i and a are given with integral convention
  void parity(const int& s, const int& e, double& parity) {
    int sD = IntegralToDeterminant(s), eD = IntegralToDeterminant(e);
    int start = min(sD, eD), end = max(sD, eD);
    long one = 1;
    long mask = (one<< (start%64))-one;
    long result = repr[start/64]&mask;
    int nonZeroBits = -BitCount(result);

    for (int i=start/64; i<end/64; i++) {
      nonZeroBits += BitCount(repr[i]);
    }
    mask = (one<< (end%64) )-one;

    result = repr[end/64] & mask;
    nonZeroBits += BitCount(result);


    parity *= (-2.*(nonZeroBits%2)+1);
    if (getoccDeterminant(start)) parity *= -1.;

    return;
  }

  //i and a are given with integral convention
  CItype Hij_1Excite(int& i, int& a, oneInt&I1, twoInt& I2);

  //i, j, a, b are given with integral convention
  CItype Hij_2Excite(int& i, int& j, int& a, int& b, oneInt&I1, twoInt& I2);


  size_t getLexicalOrder() {
    size_t order = 0;
    int pnelec = 0;
    long one = 1;
    for(int i=0; i<DetLen; i++) {
      long reprBit = repr[i];
      while (reprBit != 0) {
	int pos = __builtin_ffsl(reprBit);
	order += LexicalOrder(i*64+pos-1-pnelec, pnelec);
	pnelec++;
	//reprBit = reprBit
	reprBit &= ~(one<<(pos-1));
      }
    }
    return order;
  }

  size_t getHash() {
    return getLexicalOrder();
  }

  
  //in standard form alpha string > beta string
  bool isStandard() {
    if (!hasUnpairedElectrons()) return true;
  
    for (int i=DetLen/2-1; i>=0; i--) {
      if (repr[i] < repr[i+DetLen/2]) return false;
      else if (repr[i] > repr[i+DetLen/2]) return true;
    }
    cout << "Error finding standard for determinant "<<*this<<endl;
    cout << hasUnpairedElectrons()<<endl;
    exit(0);
  }

  bool hasUnpairedElectrons() {
    for (int i=DetLen/2-1; i>=0; i--) 
      if (repr[i] != repr[i+DetLen/2]) return true;
    return false;
  }

  void flipAlphaBeta() {
    long temp;
    for (int i=DetLen/2-1; i>=0; i--) {
      temp = repr[i];
      repr[i] = repr[i+DetLen/2];
      repr[i+DetLen/2] = temp;
    }
  }


  void makeStandard() {
    if (!isStandard())
      flipAlphaBeta();
  }

  int Noccupied() const {
    int nelec = 0;
    for (int i=0; i<DetLen; i++) {
      nelec += BitCount(repr[i]);
    }
    return nelec;

  }

  int Nalpha() const {
    int nalpha = 0;
    for (int i=0; i<DetLen/2; i++) {
      nalpha += BitCount(repr[i]);
    }
    return nalpha;
  }

  int Nbeta() const {
    int nbeta = 0;
    for (int i=DetLen/2; i<DetLen; i++) {
      nbeta += BitCount(repr[i]);
    }
    return nbeta;
  }

  //Is the excitation between *this and d less than equal to 2.
  bool connected(const Determinant& d) const {
    int ndiff = 0; long u;

    for (int i=0; i<DetLen; i++) {
      ndiff += BitCount(repr[i] ^ d.repr[i]);
    }
    return ndiff<=4;
      //return true;
  }

  bool connectedToFlipAlphaBeta(const Determinant& d) const {
    int ndiff=0;
    for (int i=0; i<DetLen/2; i++) {
      ndiff += BitCount(repr[i] ^ d.repr[i+DetLen/2]);
    }
    for (int i=0; i<DetLen/2; i++) {
      ndiff += BitCount(repr[i+DetLen/2] ^ d.repr[i]);
    }
    return ndiff<=4;
  }

  //Get the number of electrons that need to be excited to get determinant d from *this determinant
  //e.g. single excitation will return 1
  int ExcitationDistance(const Determinant& d) const {
    int ndiff = 0; 
    for (int i=0; i<DetLen; i++) {
      ndiff += BitCount(repr[i] ^ d.repr[i]);
    }
    return ndiff/2;
  }

  //Get HalfDet with just the alpha string
  HalfDet getAlpha() const {
    HalfDet d;
    for (int i=0; i<DetLen/2; i++)
      d.repr[i] = repr[i];
    return d;
  }


  //get HalfDet with just the beta string
  HalfDet getBeta() const {
    HalfDet d;
    for (int i=0; i<DetLen/2; i++)
      d.repr[i] = repr[i+DetLen/2];
    return d;
  }

  /******* CHANGE THIS BACK
  //the comparison between determinants is performed
  bool operator<(const Determinant& d) const {
    for (int i=DetLen-1; i>=0 ; i--) {
      if (repr[i] < d.repr[i]) return true;
      else if (repr[i] > d.repr[i]) return false;
    }
    return false;
  }

  //check if the determinants are equal
  bool operator==(const Determinant& d) const {
    for (int i=DetLen-1; i>=0 ; i--)
      if (repr[i] != d.repr[i]) return false;
    return true;
  }
  */
  bool operator<(const Determinant& d) const {
    for (int i=norbs-1; i>=0; i--) {
      if (getocc(i) && !d.getocc(i)) return false;
      else if (!getocc(i) && d.getocc(i)) return true;
    }
    return false;
  }
  bool operator==(const Determinant& d) const {
    for (int i=norbs-1; i>=0; i--) {
      if (getocc(i) && !d.getocc(i)) return false;
      else if (!getocc(i) && d.getocc(i)) return false;
    }
    return true;
  }

  //set the occupation of the ith orbital
  void setocc(int I, bool occ) {
    int i = IntegralToDeterminant(I);
    long Integer = i/64, bit = i%64, one=1;
    if (occ)
      repr[Integer] |= one << bit;
    else
      repr[Integer] &= ~(one<<bit);
  }


  bool getoccDeterminant(int i) const {
    long Integer = i/64, bit = i%64, reprBit = repr[Integer];
    if(( reprBit>>bit & 1) == 0)
      return false;
    else
      return true;
  }


  //get the occupation of the ith orbital
  bool getocc(int I) const {
    int i = IntegralToDeterminant(I);
    long Integer = i/64, bit = i%64, reprBit = repr[Integer];
    if(( reprBit>>bit & 1) == 0)
      return false;
    else
      return true;
  }

  void getRepArray(char* repArray) const {
    for (int i=0; i<norbs; i++) {
      if (getocc(i)) repArray[i] = 1;
      else repArray[i] = 0;
    }
  }

  //Prints the determinant
  friend ostream& operator<<(ostream& os, const Determinant& d) {
    char det[norbs];
    d.getRepArray(det);
    for (int i=0; i<norbs/2; i++) {
      if (det[2*i]==false && det[2*i+1] == false)
	os<<0<<" ";
      else if (det[2*i]==true && det[2*i+1] == false)
	os<<"a"<<" ";
      else if (det[2*i]==false && det[2*i+1] == true)
	os<<"b"<<" ";
      else if (det[2*i]==true && det[2*i+1] == true)
	os<<2<<" ";
      if ( (i+1)%5 == 0)
	os <<"  ";
    }
    return os;
  }


  //returns integer array containing the closed and open orbital indices
  int getOpenClosed(unsigned short* open, unsigned short* closed) const {
    int oindex=0,cindex=0;
    for (int i=0; i<norbs; i++) {
      if (getocc(i)) {closed[cindex] = i; cindex++;} 
      else {open[oindex] = i; oindex++;}
    }
    return cindex;
  }

  //returns integer array containing the closed and open orbital indices
  void getOpenClosed(vector<int>& open, vector<int>& closed) const {
    int oindex=0,cindex=0;
    for (int i=0; i<norbs; i++) {
      if (getocc(i)) {closed[cindex] = i; cindex++;} 
      else {open[oindex] = i; oindex++;}
    }
  }

  //returns integer array containing the closed and open orbital indices
  int getOpenClosed(int* open, int* closed) const {
    int oindex=0,cindex=0;
    for (int i=0; i<norbs; i++) {
      if (getocc(i)) {closed[cindex] = i; cindex++;} 
      else {open[oindex] = i; oindex++;}
    }
    return cindex;
  }

};


double EnergyAfterExcitation(vector<int>& closed, int& nclosed, oneInt& I1, twoInt& I2, double& coreE,
			     int i, int A, double Energyd) ;
double EnergyAfterExcitation(vector<int>& closed, int& nclosed, oneInt& I1, twoInt& I2, double& coreE,
			     int i, int A, int j, int B, double Energyd) ;
CItype Hij(Determinant& bra, Determinant& ket, oneInt& I1, twoInt& I2, double& coreE, size_t& orbDiff);

CItype Hij_1Excite(int i, int a, oneInt& I1, twoInt& I2, int* closed, int& nclosed);

void updateHijForTReversal(CItype& hij, Determinant& dk, Determinant& dj,
			   oneInt& I1,
			   twoInt& I2, 
			   double& coreE);

#endif
