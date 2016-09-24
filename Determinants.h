#ifndef Determinants_HEADER_H
#define Determinants_HEADER_H

#include "global.h"
#include <iostream>
#include <vector>
#include <boost/serialization/serialization.hpp>

using namespace std;
inline int BitCount (long& u)
{
  if (u==0) return 0;
  unsigned int u2=u>>32, u1=u;
  
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
}

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
      if (getocc(i)) {closed[cindex] = i; cindex++;}
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
      ar & repr[i];
  }

 public:
  // 0th position of 0th long is the first position
  // 63rd position of the last long is the last position
  long repr[DetLen];
  static int norbs;
  static int EffDetLen;

  Determinant() {
    for (int i=0; i<DetLen; i++)
      repr[i] = 0;
  }


  bool connected(const Determinant& d) const {
    int ndiff = 0; long u;
    for (int i=0; i<EffDetLen; i++) {
      u = repr[i] ^ d.repr[i];
      ndiff += BitCount(u);
      if (ndiff > 4) return false;
    }
    return true;
  }

  int ExcitationDistance(const Determinant& d) const {
    int ndiff = 0; long u;
    for (int i=0; i<EffDetLen; i++) {
      u = repr[i] ^ d.repr[i];
      ndiff += BitCount(u);
    }
    return ndiff/2;
  }

  HalfDet getAlpha() const {
    HalfDet d;
    for (int i=0; i<EffDetLen; i++)
      for (int j=0; j<32; j++) {
	d.setocc(i*32+j, getocc(i*64+j*2));
      }
    return d;
  }

  HalfDet getBeta() const {
    HalfDet d;
    for (int i=0; i<EffDetLen; i++)
      for (int j=0; j<32; j++)
	d.setocc(i*32+j, getocc(i*64+j*2+1));
    return d;
  }


  //the excitation array should contain at least  
  void ExactExcitation(const Determinant& d, unsigned short* excitation) const {
    int ncre=0, ndes=0;
    long u,b,k,one=1;
    for (int i=0;i<EffDetLen;i++) {
      u = d.repr[i] ^ repr[i];
      b = u & d.repr[i]; //the cre bits
      k = u & repr[i]; //the des bits
      for (int j=0;j<64;j++) {
	if (b == 0) break;
	if (b & one) {
	  if (ncre==4) {
	    *excitation=-1;
	    return;
	  }
	  *(excitation+ncre) = i*64+j; 
	  ncre++; 
	}
	b=b>>1;
      }
      for (int j=0;j<64;j++) {
	if (k == 0) break;
	if (k & one) { if (ndes==4) {*excitation=-1;return;};*(excitation+4+ndes) = i*64+j; ndes++;}
	k=k >> 1;
      }
      
    }
  }

  //the comparison between determinants is performed
  bool operator<(const Determinant& d) const {
    for (int i=EffDetLen-1; i>=0 ; i--) {
      if (repr[i] < d.repr[i]) return true;
      else if (repr[i] > d.repr[i]) return false;
    }
    return false;
  }

  bool operator==(const Determinant& d) const {
    for (int i=EffDetLen-1; i>=0 ; i--) 
      if (repr[i] != d.repr[i]) return false;
    return true;    
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
    //asser(i<norbs);
    long Integer = i/64, bit = i%64, reprBit = repr[Integer];
    if(( reprBit>>bit & 1) == 0)
      return false;
    else
      return true;
  }

  //the represenation where each char represents an orbital
  //have to be very careful that the repArray is properly allocated with enough space
  //before it is passed to this function
  void getRepArray(char* repArray) const {
    for (int i=0; i<norbs; i++) {
      if (getocc(i)) repArray[i] = 1;
      else repArray[i] = 0;
    }
  }

  friend ostream& operator<<(ostream& os, const Determinant& d) {
    char det[norbs];
    d.getRepArray(det);
    for (int i=0; i<norbs; i++)
      os<<(int)(det[i])<<" ";
    return os;
  }

  int getOpenClosed(unsigned short* open, unsigned short* closed) const {
    int oindex=0,cindex=0;
    for (int i=0; i<norbs; i++) {
      if (getocc(i)) {closed[cindex] = i; cindex++;}
      else {open[oindex] = i; oindex++;}
    }
    return cindex;
  }

  void getOpenClosed(vector<int>& open, vector<int>& closed) const {
    int oindex=0,cindex=0;
    for (int i=0; i<norbs; i++) {
      if (getocc(i)) {closed.at(cindex) = i; cindex++;}
      else {open.at(oindex) = i; oindex++;}
    }
  }
  int getOpenClosed(int* open, int* closed) const {
    int oindex=0,cindex=0;
    for (int i=0; i<norbs; i++) {
      if (getocc(i)) {closed[cindex] = i; cindex++;}
      else {open[oindex] = i; oindex++;}
    }
    return cindex;
  }

};


#endif
