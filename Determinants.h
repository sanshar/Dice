#ifndef Determinants_HEADER_H
#define Determinants_HEADER_H

#include "global.h"

int BitCount (long& u);

class Determinant {
 public:
  // 0th position of 0th long is the first position
  // 63rd position of the last long is the last position
  long repr[DetLen];
  static int norbs;

  Determinant() {
    for (int i=0; i<DetLen; i++)
      repr[i] = 0;
  }

  bool connected(const Determinant& d) const {
    int ndiff = 0; long u;
    for (int i=0; i<DetLen; i++) {
      u = repr[i] ^ d.repr[i];
      ndiff += BitCount(u);
    }
    if (ndiff > 4) return false;
    return true;
  }

  //the comparison between determinants is performed
  bool operator<(const Determinant& d) const {
    for (int i=DetLen-1; i>=0 ; i--) {
      if (repr[i] < d.repr[i]) return true;
      else if (repr[i] > d.repr[i]) return false;
    }
    return false;
  }

  bool operator==(const Determinant& d) const {
    for (int i=DetLen-1; i>=0 ; i--) 
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

  int getOpenClosed(unsigned short* open, unsigned short* closed) const {
    int oindex=0,cindex=0;
    for (int i=0; i<norbs; i++) {
      if (getocc(i)) {closed[cindex] = i; cindex++;}
      else {open[oindex] = i; oindex++;}
    }
    return cindex;
  }

};


#endif
