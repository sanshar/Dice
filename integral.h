#ifndef INTEGRAL_HEADER_H
#define INTEGRAL_HEADER_H
#include <vector>
#include <string>
#include <iostream>
using namespace std;

class oneInt {
 public:
  std::vector<double> store;
  double zero ;
 oneInt() :zero(0.0) {}
  double& operator()(int i, int j) {
    zero = 0.0;
    if (!((i%2 == j%2))) {
      return zero;
    }
    int I = i/2; int J=j/2;
    int A = max(I,J), B = min(I,J);
    return store.at(A*(A+1)/2 +B);
  }
};
  
class twoInt {
 public:
  std::vector<double> store;
  double zero ;
 twoInt() :zero(0.0) {}
  double& operator()(int i, int j, int k, int l) {
    zero = 0.0;
    if (!((i%2 == j%2) && (k%2 == l%2))) {
      return zero;
    }
    int I=i/2;int J=j/2;int K=k/2;int L=l/2;
    
    int IJ = max(I,J)*(max(I,J)+1)/2 + min(I,J);
    int KL = max(K,L)*(max(K,L)+1)/2 + min(K,L);
    int A = max(IJ,KL), B = min(IJ,KL);
    return store[A*(A+1)/2+B];
  }
  
};
  
void readIntegrals(string fcidump, twoInt& I2, oneInt& I1, int& nelec, int& norbs, double& coreE);

#endif
