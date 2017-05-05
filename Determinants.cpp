#include <iostream>
#include <algorithm>
#include "integral.h"
#include "Hmult.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include "Determinants.h"
using namespace std;
using namespace Eigen;

//Assumes that the spin of i and a orbitals is the same
double EnergyAfterExcitation(vector<int>& closed, int& nclosed, oneInt& I1, twoInt& I2, double& coreE,
			     int i, int A, double Energyd) {

  double E = Energyd;
#ifdef Complex
  E += - I1(closed[i], closed[i]).real() + I1(A, A).real();
#else
  E += - I1(closed[i], closed[i]) + I1(A, A);
#endif

  for (int I = 0; I<nclosed; I++) {
    if (I == i) continue;
    E = E - I2.Direct(closed[I]/2, closed[i]/2) + I2.Direct(closed[I]/2, A/2);
    if ( (closed[I]%2) == (closed[i]%2) )
      E = E + I2.Exchange(closed[I]/2, closed[i]/2) - I2.Exchange(closed[I]/2, A/2);
  }
  return E;
}

//Assumes that the spin of i and a orbitals is the same
//and the spins of j and b orbitals is the same
double EnergyAfterExcitation(vector<int>& closed, int& nclosed, oneInt& I1, twoInt& I2, double& coreE,
			     int i, int A, int j, int B, double Energyd) {

#ifdef Complex
  double E = Energyd - (I1(closed[i], closed[i]) - I1(A, A)+ I1(closed[j], closed[j]) - I1(B, B)).real();
#else
  double E = Energyd - I1(closed[i], closed[i]) + I1(A, A)- I1(closed[j], closed[j]) + I1(B, B);
#endif

  for (int I = 0; I<nclosed; I++) {
    if (I == i) continue;
    E = E - I2.Direct(closed[I]/2, closed[i]/2) + I2.Direct(closed[I]/2, A/2);
    if ( (closed[I]%2) == (closed[i]%2) )
      E = E + I2.Exchange(closed[I]/2, closed[i]/2) - I2.Exchange(closed[I]/2, A/2);
  }

  for (int I=0; I<nclosed; I++) {
    if (I == i || I == j) continue;
    E = E - I2.Direct(closed[I]/2, closed[j]/2) + I2.Direct(closed[I]/2, B/2);
    if ( (closed[I]%2) == (closed[j]%2) )
      E = E + I2.Exchange(closed[I]/2, closed[j]/2) - I2.Exchange(closed[I]/2, B/2);
  }

  E = E - I2.Direct(A/2, closed[j]/2) + I2.Direct(A/2, B/2);
  if ( (closed[i]%2) == (closed[j]%2) )
    E = E + I2.Exchange(A/2, closed[j]/2) - I2.Exchange(A/2, B/2);
  

  return E;
}

double Determinant::Energy(oneInt& I1, twoInt&I2, double& coreE) {
  double energy = 0.0;
  size_t one = 1;
  vector<int> closed;
  for(int i=0; i<EffDetLen; i++) {
    long reprBit = repr[i];
    while (reprBit != 0) {
      int pos = __builtin_ffsl(reprBit);
      closed.push_back(i*64+pos-1);
      reprBit &= ~(one<<(pos-1));
    }
  }

  for (int i=0; i<closed.size(); i++) {
    int I = closed.at(i);
#ifdef Complex
    energy += I1(I,I).real();
#else
    energy += I1(I,I);
#endif
    for (int j=i+1; j<closed.size(); j++) {
      int  J = closed.at(j);
      energy += I2.Direct(I/2,J/2);
      if ( (I%2) == (J%2) ) {
	energy -= I2.Exchange(I/2, J/2);
      }
    }
  }

  return energy+coreE;
}


void Determinant::initLexicalOrder(int nelec) {
  LexicalOrder.setZero(norbs-nelec+1, nelec);
  Matrix<size_t, Dynamic, Dynamic> NodeWts(norbs-nelec+2, nelec+1);
  NodeWts(0,0) = 1;
  for (int i=0; i<nelec+1; i++)
    NodeWts(0,i) = 1;
  for (int i=0; i<norbs-nelec+2; i++)
    NodeWts(i,0) = 1;

  for (int i=1; i<norbs-nelec+2; i++)
    for (int j=1; j<nelec+1; j++)
      NodeWts(i,j) = NodeWts(i-1, j) + NodeWts(i, j-1);

  for (int i=0; i<norbs-nelec+1; i++) {
    for (int j=0; j<nelec; j++) {
      LexicalOrder(i,j) = NodeWts(i,j+1)-NodeWts(i,j);
    }
  }
}

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


//i->a and j->b
void Determinant::parity(int& i, int& j, int& a, int& b, double& sgn) {
  parity(min(i, a), max(i,a), sgn); 
  setocc(i, false); setocc(a,true);
  parity(min(j, b), max(j,b), sgn);
  setocc(i, true); setocc(a, false);
  return;
}

CItype Determinant::Hij_2Excite(int& i, int& j, int& a, int& b, oneInt&I1, twoInt& I2) {

  double sgn = 1.0;
  int I = min(i,j), J= max(i,j), A= min(a,b), B = max(a,b);
  parity(min(I, A), max(I,A), sgn); 
  parity(min(J, B), max(J,B), sgn);
  if(A>J || B<I) sgn *= -1.;
  return sgn*(I2(A,I,B,J) - I2(A,J,B,I));
}


CItype Hij_1Excite(int a, int i, oneInt& I1, twoInt& I2, int* closed, int& nclosed) {
  //int a = cre[0], i = des[0];
  double sgn=1.0;
  
  CItype energy = I1(a,i);
  for (int j=0; j<nclosed; j++) {
    if (closed[j]>min(i,a)&& closed[j] <max(i,a))
      sgn*=-1.;
    energy += (I2(a,i,closed[j],closed[j]) - I2(a,closed[j],closed[j], i));
  }

  return energy*sgn;
}


CItype Determinant::Hij_1Excite(int& a, int& i, oneInt&I1, twoInt& I2) {
  double sgn = 1.0;
  parity(min(a,i), max(a,i), sgn); 
  
  CItype energy = I1(a,i);
  long one = 1;
  for (int I=0; I<EffDetLen; I++) {

    long reprBit = repr[I];
    while (reprBit != 0) {
      int pos = __builtin_ffsl(reprBit);
      int j = I*64+pos-1;
      energy += (I2(a,i,j,j) - I2(a,j,j,i));
      reprBit &= ~(one<<(pos-1));
    }

  }
  return energy*sgn;
}

CItype Hij(Determinant& bra, Determinant& ket, oneInt& I1, twoInt& I2, double& coreE, size_t& orbDiff) {  
  int cre[2],des[2],ncre=0,ndes=0; long u,b,k,one=1;
  cre[0]=-1;cre[1]=-1;des[0]=-1;des[1]=-1;

  for (int i=0;i<Determinant::EffDetLen;i++) {
    u = bra.repr[i] ^ ket.repr[i];
    b = u & bra.repr[i]; //the cre bits
    k = u & ket.repr[i]; //the des bits

    while(b != 0) {
      int pos = __builtin_ffsl(b);
      cre[ncre] = pos-1+i*64;
      ncre++;
      b &= ~(one<<(pos-1));
    }
    while(k != 0) {
      int pos = __builtin_ffsl(k);
      des[ndes] = pos-1+i*64;
      ndes++;
      k &= ~(one<<(pos-1));
    }
  }

  if (ncre == 0) {
    cout <<"Use the function for energy"<<endl;
    exit(0);
  }
  else if (ncre ==1 ) {
    size_t c0=cre[0], N=bra.norbs, d0 = des[0];
    orbDiff = c0*N+d0;
    //orbDiff = cre[0]*bra.norbs+des[0];
    return ket.Hij_1Excite(cre[0], des[0], I1, I2);
  }
  else if (ncre == 2) {
    size_t c0=cre[0], c1=cre[1], d1=des[1],N=bra.norbs, d0 = des[0];
    orbDiff = c1*N*N*N+d1*N*N+c0*N+d0;
    //orbDiff = cre[1]*bra.norbs*bra.norbs*bra.norbs+des[1]*bra.norbs*bra.norbs+cre[0]*bra.norbs+des[0];
    return ket.Hij_2Excite(des[0], des[1], cre[0], cre[1], I1, I2);
  }
  else {
    cout << "Should not be here"<<endl;
    return 0.;
  }
}




