#include <iostream>
#include <algorithm>
//#include <stdio.h>
#include "integral.h"
#include "Hmult.h"
#include <Eigen/Dense>
#include <Eigen/Core>
using namespace std;
using namespace Eigen;

double parity(char* d, int& sizeA, int& i) {
  double sgn = 1.;
  for (int j=0; i<sizeA; j++) {
    if (j >= i)
      break;
    if (d[j] != 0)
      sgn *= -1;
  }
  return sgn;
}

double Energy(char* ket, int& sizeA, oneInt& I1, twoInt& I2, double& coreE) {
  double energy = 0.0;

  for (int i=0; i<sizeA; i++) {
    if (ket[i]) {
      energy += I1( i,i);
      for (int j=i; j<sizeA; j++) {
	if (ket[j]){
	  energy += I2(i,i,j,j) - I2(i,j,i,j);
	}
      }
    }
  }
  return energy+(coreE);
}

double Hij_1Excite(int i, int a, oneInt& I1, twoInt& I2, char* ket, int& sizeA) {
  //int a = cre[0], i = des[0];
  double sgn = parity(ket,sizeA, a)*parity(ket,sizeA,i);
  if (a > i)
    sgn*=-1;
  
  double energy = I1(a,i);
  for (int j=0; j<sizeA; j++)
    if (ket[j] != 0)
      energy += (I2(a,i,j,j) - I2(a,j,i,j));
  return energy*sgn;
}

double Hij_2Excite(int i, int j, int a, int b, twoInt& I2, char* ket, int& sizeA) {
    double sgn = parity(ket,sizeA,a)*parity(ket,sizeA,i)*parity(ket,sizeA,b)*parity(ket,sizeA,j);
    if (b > j) sgn*=-1 ;
    if (i > j) sgn*=-1 ;
    if (i > b) sgn*=-1 ;
    if (a > j) sgn*=-1 ;
    if (a > b) sgn*=-1 ;
    if (a > i) sgn*=-1 ;
    return sgn*(I2(a,i,b,j) - I2(a,j,b,i));
}

double Hij(char* bra, char* ket, int& sizeA, oneInt& I1, twoInt& I2, double& coreE) {
  int cre[2], des[2];cre[0]=-1;cre[1]=-1;des[0]=-1;des[1]=-1;
  int ncre = 0, ndes = 0;

  int ndiff = 0;
  for (int i=0; i<sizeA; i++) 
    if (bra[i] != ket[i]) ndiff++;
  if (ndiff >4) return 0.0;

  for (int i=0; i<sizeA; i++) {
    if (bra[i] && !ket[i]) {
      if (ncre == 2) return 0.0;
      cre[ncre] = i;
      ncre++;
    }
    else if (bra[i]==0 && ket[i] !=0){
      if (ndes == 2) return 0.0;
      des[ndes] = i;
      ndes++;
    }
  }

  double energy = 0.0;
  if (ncre == 0 && ndes == 0) {
    return Energy(ket, sizeA, I1, I2, coreE);
  }
  else if (ncre ==1 && ndes == 1) {
    return Hij_1Excite(des[0], cre[0], I1, I2, ket, sizeA);
  }
  else if (ncre==2 && ndes == 2){
    int a=cre[0], b=cre[1], i=des[0], j=des[1];
    return Hij_2Excite(i,j,a,b,I2, ket, sizeA);
  }
  return energy;

}




