#include <iostream>
#include <algorithm>
#include "integral.h"
#include "Hmult.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include "Determinants.h"
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

/*
int BitCount (long& u)
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

int BitCountbkp (long& u)
{
  unsigned int uCount=0 ;
  
  for(; u; u&=(u-1))
    uCount++;
  
  return uCount ;
}
*/
double Energy(vector<int>& occ, int& sizeA, oneInt& I1, twoInt& I2, double& coreE) {
  double energy = 0.0;
  for (int i=0; i<sizeA; i++) {
    int I = occ.at(i);
    energy += I1(I,I);
    for (int j=i+1; j<sizeA; j++) {
      int  J = occ.at(j);
      energy += I2.Direct(I/2,J/2);
      if ( (I%2) == (J%2) )
	energy -= I2.Exchange(I/2,J/2);
    }
  }
  return energy+coreE;
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


double Hij(Determinant& bra, Determinant& ket, int& sizeA, oneInt& I1, twoInt& I2, double& coreE) {  
  int cre[2],des[2],ncre=0,ndes=0; long u,b,k,one=1;
  for (int i=0;i<DetLen;i++) {
    u = bra.repr[i] ^ ket.repr[i];
    b = u & bra.repr[i]; //the cre bits
    k = u & ket.repr[i]; //the des bits
    for (int j=0;j<64;j++) {
      if (b == 0) break;
      if (b>>1 & 1) {cre[ncre] = j; ncre++;}
    }
    for (int j=0;j<64;j++) {
      if (k == 0) break;
      if (k>>1 & 1) {des[ndes] = j; ndes++;}
    }
    
  }
  ///NOT YET IMPLEMENTED
  /*
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
  */
}




