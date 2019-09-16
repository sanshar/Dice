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
#include <iostream>
#include <algorithm>
#include "integral.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include "Determinants.h"
#include "input.h"
#include "workingArray.h"

using namespace std;
using namespace Eigen;

BigDeterminant::BigDeterminant(const Determinant& d) {
  int norbs = Determinant::norbs;
  occupation.resize(2*norbs, 0);
  for (int i=0; i<2*norbs; i++)
    if (d.getocc(i)) occupation[i] = 1;
}

const char& BigDeterminant::operator[] (int j) const
{
  return occupation[j];
}

char& BigDeterminant::operator[] (int j)
{
  return occupation[j];
}

Determinant::Determinant() {
  for (int i=0; i<DetLen; i++) {
    reprA[i] = 0;
    reprB[i] = 0;
  }
}

Determinant::Determinant(const Determinant& d) {
  for (int i=0; i<DetLen; i++) {
    reprA[i] = d.reprA[i];
    reprB[i] = d.reprB[i];
  }
}

void Determinant::operator=(const Determinant& d) {
  for (int i=0; i<DetLen; i++) {
    reprA[i] = d.reprA[i];
    reprB[i] = d.reprB[i];
  }
}

void Determinant::getOpenClosed( std::vector<int>& open, std::vector<int>& closed) const {
  for (int i=0; i<norbs; i++) {
    if ( getoccA(i)) closed.push_back(2*i);
    else open.push_back(2*i);
    if ( getoccB(i)) closed.push_back(2*i+1);
    else open.push_back(2*i+1);
  }
}

void Determinant::getOpenClosed( bool sz, std::vector<int>& open, std::vector<int>& closed) const {
  
  for (int i=0; i<norbs; i++) 
  {
    if (sz==0)
    {
      if ( getoccA(i)) closed.push_back(i);
      else open.push_back(i);
    }
    else
    {
      if ( getoccB(i)) closed.push_back(i);
      else open.push_back(i);
    }
  }

}

void Determinant::getOpenClosedAlphaBeta( std::vector<int>& openAlpha,
                                          std::vector<int>& closedAlpha,
                                          std::vector<int>& openBeta,
                                          std::vector<int>& closedBeta
                                          ) const {
  for (int i=0; i<norbs; i++) {
    if ( getoccA(i)) closedAlpha.push_back(i);
    else openAlpha.push_back(i);
    if ( getoccB(i)) closedBeta.push_back(i);
    else openBeta.push_back(i);
  }
}

void Determinant::getClosedAlphaBeta( std::vector<int>& closedAlpha,
                                      std::vector<int>& closedBeta ) const 
{
  for (int i=0; i<norbs; i++) {
    if ( getoccA(i)) closedAlpha.push_back(i);
    if ( getoccB(i)) closedBeta.push_back(i);
  }
}

void Determinant::getAlphaBeta(std::vector<int>& alpha, std::vector<int>& beta) const {
  for (int i=0; i<64*EffDetLen; i++) {
    if (getoccA(i)) alpha.push_back(i);
    if (getoccB(i)) beta .push_back(i);
  }
}

void Determinant::getClosed( bool sz, std::vector<int>& closed) const {
  
  for (int i=0; i<norbs; i++) 
  {
    if (sz==0)
    {
      if ( getoccA(i)) closed.push_back(i);
    }
    else
    {
      if ( getoccB(i)) closed.push_back(i);
    }
  }

}

int Determinant::getNbetaBefore(int i) const {
  int occ = 0;
  for (int n = 0; n < i/64; n++) {
    occ += CountNonZeroBits(reprB[n]);
  }
  long one = 1; long mask = ( one << (i%64) ) - one;
  long result = (reprB[i/64] & mask ) ;
  occ += CountNonZeroBits(result);
  return occ;
}

int Determinant::getNalphaBefore(int i) const {
  int occ = 0;
  for (int n = 0; n < i/64; n++) {
    occ += CountNonZeroBits(reprA[n]);
  }
  long one = 1; long mask = ( one << (i%64) ) - one;
  long result = (reprA[i/64] & mask ) ;
  occ += CountNonZeroBits(result);
  return occ;
}

double Determinant::parityA(const int& a, const int& i) const {
  double parity = 1.0;
  int occ = getNalphaBefore(i);
  occ += getNalphaBefore(a);

  parity *= (occ%2==0) ? 1.: -1.;
  if (i < a) parity *= -1.;
    
  return parity;
}

double Determinant::parity(const int& a, const int& i, const bool& sz) const {
  if (sz == 0) return parityA(a, i);
  else return parityB(a, i);
}

double Determinant::parityB(const int& a, const int& i) const {
  double parity = 1.0;
  int occ = getNbetaBefore(i);
  occ += getNbetaBefore(a);

  parity *= (occ%2==0) ? 1.: -1.;
  if (i < a) parity *= -1.;

  return parity;
}

double Determinant::parityA(const vector<int>& aArray, const vector<int>& iArray) const
{
  double p = 1.;
  Determinant dcopy = *this;
  for (int i = 0; i < iArray.size(); i++)
  {
    p *= dcopy.parityA(aArray[i], iArray[i]);

    dcopy.setoccA(iArray[i], false);
    dcopy.setoccA(aArray[i], true);
  }
  return p;
}

double Determinant::parityB(const vector<int>& aArray, const vector<int>& iArray) const
{
  double p = 1.;
  Determinant dcopy = *this;
  for (int i = 0; i < iArray.size(); i++)
  {
    p *= dcopy.parityB(aArray[i], iArray[i]);

    dcopy.setoccB(iArray[i], false);
    dcopy.setoccB(aArray[i], true);
  }
  return p;
}

double Determinant::parity(const vector<int>& aArray, const vector<int>& iArray, bool sz) const
{
  if (sz==0) return parityA(aArray, iArray);
  else return parityB(aArray, iArray);
}

int Determinant::Noccupied() const {
  int nelec = 0;
  for (int i=0; i<DetLen; i++) {
    nelec += CountNonZeroBits(reprA[i]);
    nelec += CountNonZeroBits(reprB[i]);
  }
  return nelec;
}

int Determinant::Nalpha() const {
  int nelec = 0;
  for (int i=0; i<DetLen; i++) {
    nelec += CountNonZeroBits(reprA[i]);
  }
  return nelec;
}

int Determinant::Nbeta() const {
  int nelec = 0;
  for (int i=0; i<DetLen; i++) {
    nelec += CountNonZeroBits(reprB[i]);
  }
  return nelec;
}

//Is the excitation between *this and d less than equal to 2.
bool Determinant::connected(const Determinant& d) const {
  int ndiff = 0; long u;

  for (int i=0; i<DetLen; i++) {
    ndiff += CountNonZeroBits(reprA[i] ^ d.reprA[i]);
    ndiff += CountNonZeroBits(reprB[i] ^ d.reprB[i]);
  }
  return ndiff<=4;
  //return true;
}


//Get the number of electrons that need to be excited to get determinant d from *this determinant
//e.g. single excitation will return 1
int Determinant::ExcitationDistance(const Determinant& d) const {
  int ndiff = 0;
  for (int i=0; i<DetLen; i++) {
    ndiff += CountNonZeroBits(reprA[i] ^ d.reprA[i]);
    ndiff += CountNonZeroBits(reprB[i] ^ d.reprB[i]);
  }
  return ndiff/2;
}


//the comparison between determinants is performed
bool Determinant::operator<(const Determinant& d) const {
  for (int i=DetLen-1; i>=0 ; i--) {
    if (reprA[i] < d.reprA[i]) return true;
    else if (reprA[i] > d.reprA[i]) return false;
    if (reprB[i] < d.reprB[i]) return true;
    else if (reprB[i] > d.reprB[i]) return false;
  }
  return false;
}

//check if the determinants are equal
bool Determinant::operator==(const Determinant& d) const {
  for (int i=DetLen-1; i>=0 ; i--) {
    if (reprA[i] != d.reprA[i]) return false;
    if (reprB[i] != d.reprB[i]) return false;
  }
  return true;
}

//set the occupation of the ith orbital
void Determinant::setoccA(int i, bool occ) {
  long Integer = i/64, bit = i%64, one=1;
  if (occ)
    reprA[Integer] |= one << bit;
  else
    reprA[Integer] &= ~(one<<bit);
}

//set the occupation of the ith orbital
void Determinant::setoccB(int i, bool occ) {
  long Integer = i/64, bit = i%64, one=1;
  if (occ)
    reprB[Integer] |= one << bit;
  else
    reprB[Integer] &= ~(one<<bit);
}

void Determinant::setocc(int i, bool occ)  {
  if (i%2 == 0) return setoccA(i/2, occ);
  else return setoccB(i/2, occ);
}

void Determinant::setocc(int i, bool sz, bool occ)  {
  if (sz == 0) return setoccA(i, occ);
  else return setoccB(i, occ);
}

bool Determinant::getocc(int i) const {
  if (i%2 == 0) return getoccA(i/2);
  else return getoccB(i/2);
}

bool Determinant::getocc(int i, bool sz) const {
  if (sz == 0) return getoccA(i);
  else return getoccB(i);
}

//get the occupation of the ith orbital
bool Determinant::getoccA(int i) const {
  //asser(i<norbs);
  long Integer = i/64, bit = i%64, reprBit = reprA[Integer];
  if(( reprBit>>bit & 1) == 0)
    return false;
  else
    return true;
}

bool Determinant::getoccB(int i) const {
  //asser(i<norbs);
  long Integer = i/64, bit = i%64, reprBit = reprB[Integer];
  if(( reprBit>>bit & 1) == 0)
    return false;
  else
    return true;
}


//Prints the determinant
ostream& operator<<(ostream& os, const Determinant& d) {
  for (int i=0; i<Determinant::norbs; i++) {
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

size_t hash_value(Determinant const& d) {
  std::size_t seed = 0;
  //boost::hash_combine(seed, d.reprA[0]);
  //boost::hash_combine(seed, d.reprB[0]);
  boost::hash_combine(seed, d.reprA[0] * 2654435761);
  boost::hash_combine(seed, d.reprB[0] * 2654435761);
  //for (int i = 0; i < DetLen; i++) {
  //  boost::hash_combine(seed, d.reprA[i]);
  //  boost::hash_combine(seed, d.reprB[i]);
  //}
  return seed;
}

//=============================================================================
double Determinant::Energy(const oneInt& I1, const twoInt&I2, const double& coreE) const {
  double energy = 0.0;
  size_t one = 1;
  vector<int> closed;
  for(int i=0; i<DetLen; i++) {
    long reprBit = reprA[i];
    while (reprBit != 0) {
      int pos = __builtin_ffsl(reprBit);
      closed.push_back( 2*(i*64+pos-1));
      reprBit &= ~(one<<(pos-1));
    }

    reprBit = reprB[i];
    while (reprBit != 0) {
      int pos = __builtin_ffsl(reprBit);
      closed.push_back( 2*(i*64+pos-1)+1);
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
      int J = closed.at(j);
      energy += I2.Direct(I/2,J/2);
      if ( (I%2) == (J%2) ) {
        energy -= I2.Exchange(I/2, J/2);
      }
    }
  }

  return energy+coreE;
}





//=============================================================================
double Determinant::parityAA(const int& i, const int& j, const int& a, const int& b) const {
  double sgn = 1.0;
  Determinant dcopy = *this;
  sgn *= dcopy.parityA(a, i);
  dcopy.setoccA(i, false); dcopy.setoccA(a,true);
  sgn *= dcopy.parityA(b, j);
  return sgn;
}

double Determinant::parityBB(const int& i, const int& j, const int& a, const int& b) const {
  double sgn = 1.0;
  Determinant dcopy = *this;
  sgn = dcopy.parityB(a, i);
  dcopy.setoccB(i, false); dcopy.setoccB(a,true);
  sgn *= dcopy.parityB(b, j);
  return sgn;
}



//=============================================================================
CItype Determinant::Hij_2ExciteAA(const int& a, const int& i, const int& b,
                                  const int& j, const oneInt&I1, const twoInt& I2) const
{

  double sgn = parityAA(i, j, a, b);
  return sgn*(I2(2*a,2*i,2*b,2*j) - I2(2*a,2*j,2*b,2*i));
}

CItype Determinant::Hij_2ExciteBB(const int& a, const int& i, const int& b,
                                  const int& j, const oneInt&I1, const twoInt& I2) const
{
  double sgn = parityBB(i, j, a, b);
  return sgn*(I2(2*a+1, 2*i+1, 2*b+1, 2*j+1) - I2(2*a+1, 2*j+1, 2*b+1, 2*i+1 ));
}

CItype Determinant::Hij_2ExciteAB(const int& a, const int& i, const int& b,
                                  const int& j, const oneInt&I1, const twoInt& I2) const {

  double sgn = parityA(a, i);
  sgn *= parityB(b,j);
  return sgn*I2(2*a,2*i,2*b+1,2*j+1);
}


CItype Determinant::Hij_1ExciteScreened(const int& a, const int& i,
                                        const twoIntHeatBathSHM& I2hb, const double& TINY,
                                        bool doparity) const {

  double tia = I1(a, i);
  int X = max(i/2, a/2), Y = min(i/2, a/2);
  int pairIndex = X * (X + 1) / 2 + Y;
  size_t start = I2hb.startingIndicesSingleIntegrals[pairIndex];
  size_t end = I2hb.startingIndicesSingleIntegrals[pairIndex + 1];
  float *integrals = I2hb.singleIntegrals;
  short *orbIndices = I2hb.singleIntegralsPairs;
  for (size_t index = start; index < end; index++)
  {
    if (fabs(integrals[index]) < TINY)
      break;
    int j = orbIndices[2 * index];
    if (i % 2 == 1 && j % 2 == 1)
      j--;
    else if (i % 2 == 1 && j % 2 == 0)
      j++;
    
    if (getocc(j) )
      tia += integrals[index];
  }

  double sgn = 1.0;
  int A = a/2, I = i/2;
  if (doparity && i%2 == 0) sgn *= parityA(A, I);
  else if (doparity && i%2 == 1) sgn *= parityB(A, I);
  return tia*sgn;
}


//=============================================================================
CItype Determinant::Hij_1ExciteA(const int& a, const int& i, const oneInt&I1,
                                 const twoInt& I2, bool doparity) const {
  double sgn = 1.0;
  if (doparity) sgn *= parityA(a, i);

  CItype energy = I1(2*a, 2*i);
  if (schd.Hamiltonian == HUBBARD) return energy*sgn;

  long one = 1;
  for (int I=0; I<DetLen; I++) {
    long reprBit = reprA[I];
    while (reprBit != 0) {
      int pos = __builtin_ffsl(reprBit);
      int j = I*64+pos-1;
      energy += (I2(2*a, 2*i, 2*j, 2*j) - I2(2*a, 2*j, 2*j, 2*i));
      reprBit &= ~(one<<(pos-1));
    }
    reprBit = reprB[I];
    while (reprBit != 0) {
      int pos = __builtin_ffsl(reprBit);
      int j = I*64+pos-1;
      energy += (I2(2*a, 2*i, 2*j+1, 2*j+1));
      reprBit &= ~(one<<(pos-1));
    }

  }
  energy *= sgn;
  return energy;
}

CItype Determinant::Hij_1ExciteB(const int& a, const int& i, const oneInt&I1,
                                 const twoInt& I2, bool doparity)  const {
  double sgn = 1.0;
  if (doparity) sgn *= parityB(a, i);

  CItype energy = I1(2*a+1, 2*i+1);
  if (schd.Hamiltonian == HUBBARD) return energy*sgn;

  long one = 1;
  for (int I=0; I<DetLen; I++) {
    long reprBit = reprA[I];
    while (reprBit != 0) {
      int pos = __builtin_ffsl(reprBit);
      int j = I*64+pos-1;
      energy += (I2(2*a+1, 2*i+1, 2*j, 2*j));
      reprBit &= ~(one<<(pos-1));
    }
    reprBit = reprB[I];
    while (reprBit != 0) {
      int pos = __builtin_ffsl(reprBit);
      int j = I*64+pos-1;
      energy += (I2(2*a+1, 2*i+1, 2*j+1, 2*j+1) - I2(2*a+1, 2*j+1, 2*j+1, 2*i+1));
      reprBit &= ~(one<<(pos-1));
    }

  }
  energy *= sgn;
  return energy;
}



//=============================================================================
CItype Hij(const Determinant& bra, const Determinant& ket, const oneInt& I1,
           const twoInt& I2, const double& coreE)  {
  int cre[200],des[200],ncrea=0,ncreb=0,ndesa=0,ndesb=0;
  long u,b,k,one=1;
  cre[0]=-1; cre[1]=-1; des[0]=-1; des[1]=-1;

  for (int i=0; i<Determinant::EffDetLen; i++) {
    u = bra.reprA[i] ^ ket.reprA[i];
    b = u & bra.reprA[i]; //the cre bits
    k = u & ket.reprA[i]; //the des bits

    while(b != 0) {
      int pos = __builtin_ffsl(b);
      cre[ncrea+ncreb] = 2*(pos-1+i*64);
      ncrea++;
      b &= ~(one<<(pos-1));
    }
    while(k != 0) {
      int pos = __builtin_ffsl(k);
      des[ndesa+ndesb] = 2*(pos-1+i*64);
      ndesa++;
      k &= ~(one<<(pos-1));
    }

    u = bra.reprB[i] ^ ket.reprB[i];
    b = u & bra.reprB[i]; //the cre bits
    k = u & ket.reprB[i]; //the des bits

    while(b != 0) {
      int pos = __builtin_ffsl(b);
      cre[ncrea+ncreb] = 2*(pos-1+i*64)+1;
      ncreb++;
      b &= ~(one<<(pos-1));
    }
    while(k != 0) {
      int pos = __builtin_ffsl(k);
      des[ndesa+ndesb] = 2*(pos-1+i*64)+1;
      ndesb++;
      k &= ~(one<<(pos-1));
    }

  }

  if (ncrea+ncreb == 0) {
    cout << bra<<endl;
    cout << ket<<endl;
    cout <<"Use the function for energy"<<endl;
    exit(0);
  }
  else if (ncrea == 1 && ncreb == 0) {
    int c0=cre[0]/2, d0 = des[0]/2;
    return ket.Hij_1ExciteA(c0, d0, I1, I2);
  }
  else if (ncrea == 0 && ncreb == 1) {
    int c0=cre[0]/2, d0 = des[0]/2;
    return ket.Hij_1ExciteB(c0, d0, I1, I2);
  }
  else if (ncrea == 0 && ncreb == 2) {
    int c0=cre[0]/2, d0 = des[0]/2;
    int c1=cre[1]/2, d1 = des[1]/2;
    return ket.Hij_2ExciteBB(c0, d0, c1, d1, I1, I2);
  }
  else if (ncrea == 2 && ncreb == 0) {
    int c0=cre[0]/2, d0 = des[0]/2;
    int c1=cre[1]/2, d1 = des[1]/2;
    return ket.Hij_2ExciteAA(c0, d0, c1, d1, I1, I2);
  }
  else if (ncrea == 1 && ncreb == 1) {
    int c0=cre[0]/2, d0 = des[0]/2;
    int c1=cre[1]/2, d1 = des[1]/2;
    if (cre[0]%2 == 0)
      return ket.Hij_2ExciteAB(c0, d0, c1, d1, I1, I2);
    else
      return ket.Hij_2ExciteAB(c1, d1, c0, d0, I1, I2);
  }
  else {
    return 0.;
  }
}


void getDifferenceInOccupation(const Determinant &bra, const Determinant &ket,
                               vector<int>& creA, vector<int>& desA,
                               vector<int>& creB, vector<int>& desB)
{
  std::fill(creA.begin(), creA.end(), -1);
  std::fill(desA.begin(), desA.end(), -1);
  std::fill(creB.begin(), creB.end(), -1);
  std::fill(desB.begin(), desB.end(), -1);

  int ncre = 0, ndes = 0;
  long u, b, k, one = 1;

  for (int i = 0; i < DetLen; i++)
  {
    u = bra.reprA[i] ^ ket.reprA[i];
    b = u & bra.reprA[i]; //the cre bits
    k = u & ket.reprA[i]; //the des bits

    while (b != 0)
    {
      int pos = __builtin_ffsl(b);
      creA[ncre] = pos - 1 + i * 64;
      ncre++;
      b &= ~(one << (pos - 1));
    }
    while (k != 0)
    {
      int pos = __builtin_ffsl(k);
      desA[ndes] = pos - 1 + i * 64;
      ndes++;
      k &= ~(one << (pos - 1));
    }
  }


  ncre = 0; ndes = 0;
  for (int i = 0; i < DetLen; i++)
  {
    u = bra.reprB[i] ^ ket.reprB[i];
    b = u & bra.reprB[i]; //the cre bits
    k = u & ket.reprB[i]; //the des bits

    while (b != 0)
    {
      int pos = __builtin_ffsl(b);
      creB[ncre] = pos - 1 + i * 64;
      ncre++;
      b &= ~(one << (pos - 1));
    }
    while (k != 0)
    {
      int pos = __builtin_ffsl(k);
      desB[ndes] = pos - 1 + i * 64;
      ndes++;
      k &= ~(one << (pos - 1));
    }
  }
}

void getDifferenceInOccupation(const Determinant &bra, const Determinant &ket,
                               vector<int>& cre, vector<int>& des,
                               bool sz)
{
  std::fill(cre.begin(), cre.end(), -1);
  std::fill(des.begin(), des.end(), -1);

  int ncre = 0, ndes = 0;
  long u, b, k, one = 1;

  if (sz == 0)
  {
    for (int i = 0; i < DetLen; i++)
    {
      u = bra.reprA[i] ^ ket.reprA[i];
      b = u & bra.reprA[i]; //the cre bits
      k = u & ket.reprA[i]; //the des bits

      while (b != 0)
      {
        int pos = __builtin_ffsl(b);
        cre[ncre] = pos - 1 + i * 64;
        ncre++;
        b &= ~(one << (pos - 1));
      }
      while (k != 0)
      {
        int pos = __builtin_ffsl(k);
        des[ndes] = pos - 1 + i * 64;
        ndes++;
        k &= ~(one << (pos - 1));
      }
    }
  }

  else
  {
    for (int i = 0; i < DetLen; i++)
    {
      u = bra.reprB[i] ^ ket.reprB[i];
      b = u & bra.reprB[i]; //the cre bits
      k = u & ket.reprB[i]; //the des bits

      while (b != 0)
      {
        int pos = __builtin_ffsl(b);
        cre[ncre] = pos - 1 + i * 64;
        ncre++;
        b &= ~(one << (pos - 1));
      }
      while (k != 0)
      {
        int pos = __builtin_ffsl(k);
        des[ndes] = pos - 1 + i * 64;
        ndes++;
        k &= ~(one << (pos - 1));
      }
    }
  }
}

void getDifferenceInOccupation(const Determinant &bra, const Determinant &ket, int &I, int &A)
{
  I = -1; A = -1;
  long u, b, k, one = 1;
  
  for (int i = 0; i < DetLen; i++)
  {
    u = bra.reprA[i] ^ ket.reprA[i];
    b = u & bra.reprA[i]; //the cre bits
    k = u & ket.reprA[i]; //the des bits
    
    while (b != 0)
    {
      int pos = __builtin_ffsl(b);
      I = 2*(pos - 1 + i * 64);
      b &= ~(one << (pos - 1));
    }
    while (k != 0)
    {
      int pos = __builtin_ffsl(k);
      A = 2 * (pos - 1 + i * 64);
      k &= ~(one << (pos - 1));
    }
  }

  for (int i = 0; i < DetLen; i++)
  {
    u = bra.reprB[i] ^ ket.reprB[i];
    b = u & bra.reprB[i]; //the cre bits
    k = u & ket.reprB[i]; //the des bits
    
    while (b != 0)
    {
      int pos = __builtin_ffsl(b);
      I = 2 * (pos - 1 + i * 64) + 1;
      b &= ~(one << (pos - 1));
    }
    while (k != 0)
    {
      int pos = __builtin_ffsl(k);
      A = 2 * (pos - 1 + i * 64) + 1;
      k &= ~(one << (pos - 1));
    }
  }
}


void getDifferenceInOccupation(const Determinant &bra, const Determinant &ket, int &I, int &J,
                               int& A, int& B)
{
  I = -1; A = -1; J = -1; B = -1;
  long u, b, k, one = 1;
  
  for (int i = 0; i < DetLen; i++)
  {
    u = bra.reprA[i] ^ ket.reprA[i];
    b = u & bra.reprA[i]; //the cre bits
    k = u & ket.reprA[i]; //the des bits
    
    while (b != 0)
    {
      int pos = __builtin_ffsl(b);

      if (I == -1)
	I = 2*(pos - 1 + i * 64);
      else
	J = 2*(pos - 1 + i * 64);
	
      b &= ~(one << (pos - 1));
    }
    while (k != 0)
    {
      int pos = __builtin_ffsl(k);
      if (A == -1)
	A = 2*(pos - 1 + i * 64);
      else
	B = 2*(pos - 1 + i * 64);
      //A = 2 * (pos - 1 + i * 64);
      k &= ~(one << (pos - 1));
    }
  }

  for (int i = 0; i < DetLen; i++)
  {
    u = bra.reprB[i] ^ ket.reprB[i];
    b = u & bra.reprB[i]; //the cre bits
    k = u & ket.reprB[i]; //the des bits
    
    while (b != 0)
    {
      int pos = __builtin_ffsl(b);

      if (I == -1)
	I = 2*(pos - 1 + i * 64) + 1;
      else
	J = 2*(pos - 1 + i * 64) + 1;

      b &= ~(one << (pos - 1));
    }
    while (k != 0)
    {
      int pos = __builtin_ffsl(k);
      if (A == -1)
	A = 2*(pos - 1 + i * 64) + 1;
      else
	B = 2*(pos - 1 + i * 64) + 1;

      k &= ~(one << (pos - 1));
    }
  }
}


double getParityForDiceToAlphaBeta(const Determinant& det) 
{
  double parity = 1.0;
  int nalpha = det.Nalpha();
  int norbs = Determinant::norbs;
  for (int i=0; i<norbs; i++) 
  {
    if (det.getoccB(norbs-1-i))
    {
      int nAlphaAfteri = nalpha - det.getNalphaBefore(norbs-1-i);
      if (det.getoccA(norbs-1-i)) nAlphaAfteri--;
      if (nAlphaAfteri%2 == 1) parity *= -1;
    }
  }
  return parity;
}


void generateAllScreenedSingleExcitation(const Determinant& d,
                                         const double& THRESH,
                                         const double& TINY,
                                         workingArray& work,
                                         bool doparity) {
  int norbs = Determinant::norbs;
  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);

  //schd.active = number of active spatial orbitals, assumed to be contiguous and at the beginning
  auto ub_1 = upper_bound(open.begin(), open.end(), 2*schd.numActive - 1);
  int indAct = distance(open.begin(), ub_1);

  //auto ub_2 = upper_bound(closed.begin(), closed.end(), 2*schd.nciCore - 1);
  //int indCore = distance(closed.begin(), ub_2);

  for (int i = 0; i < closed.size(); i++) {
    for (int a = 0; a < indAct; a++) {
      if (closed[i] % 2 == open[a] % 2 &&
          abs(I2hb.Singles(closed[i], open[a])) > THRESH)
      {
        int I = closed[i] / 2, A = open[a] / 2;

        const double tia = d.Hij_1ExciteScreened(open[a], closed[i], I2hb,
                                                 TINY, doparity);
        
        if (abs(tia) > THRESH) {
          work.appendValue(0., closed[i]*2*norbs+open[a], 0, tia);
        }
      }
    }
  }

}

void generateAllScreenedDoubleExcitation(const Determinant& d,
                                         const double& THRESH,
                                         const double& TINY,
                                         workingArray& work,
                                         bool doparity) {
  int norbs = Determinant::norbs;
  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);

  auto ub = upper_bound(closed.begin(), closed.end(), 2*schd.nciCore - 1);
  int indCore = distance(closed.begin(), ub);

  int nclosed = closed.size();
  for (int i = 0; i<nclosed; i++) {
    for (int j = 0; j<i; j++) {

      const float *integrals; const short* orbIndices;
      size_t numIntegrals;
      I2hb.getIntegralArray(closed[i], closed[j], integrals, orbIndices, numIntegrals);
      size_t numLargeIntegrals = std::lower_bound(integrals, integrals + numIntegrals, THRESH, [](const float &x, float val){ return fabs(x) > val; }) - integrals;

      // for all HCI integrals
      for (size_t index = 0; index < numLargeIntegrals; index++)
      {
        // if we are going below the criterion, break
        //if (fabs(integrals[index]) < THRESH)
        //  break;

        // otherwise: generate the determinant corresponding to the current excitation
        int a = 2 * orbIndices[2 * index] + closed[i] % 2,
            b = 2 * orbIndices[2 * index + 1] + closed[j] % 2;

        if ((!(d.getocc(a) || d.getocc(b))) && (a < 2*schd.numActive) && (b < 2*schd.numActive)) {//uncomment for VMC active space calculations
        //if (!(d.getocc(a) || d.getocc(b))) {
          //cout << "a   " << a << "  b  " << b << endl;
          work.appendValue(0.0, closed[i] * 2 * norbs + a,
                           closed[j] * 2 * norbs + b, integrals[index]);
        }
      }
    }
  }

}

void generateAllScreenedDoubleExcitationsFOIS(const Determinant& d,
                                         const double& THRESH,
                                         const double& TINY,
                                         workingArray& work,
                                         bool doparity) {
  int norbs = Determinant::norbs;
  int first_virtual = schd.nciCore + schd.nciAct;

  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);

  auto ub = upper_bound(closed.begin(), closed.end(), 2*schd.nciCore - 1);
  int indCore = distance(closed.begin(), ub);

  int nclosed = closed.size();
  for (int i = indCore; i<nclosed; i++) {
    for (int j = indCore; j<i; j++) {

      const float *integrals; const short* orbIndices;
      size_t numIntegrals;
      I2hb.getIntegralArray(closed[i], closed[j], integrals, orbIndices, numIntegrals);
      size_t numLargeIntegrals = std::lower_bound(integrals, integrals + numIntegrals, THRESH, [](const float &x, float val){ return fabs(x) > val; }) - integrals;

      // for all HCI integrals
      for (size_t index = 0; index < numLargeIntegrals; index++)
      {
        // if we are going below the criterion, break
        //if (fabs(integrals[index]) < THRESH)
        //  break;

        // otherwise: generate the determinant corresponding to the current excitation
        int a = 2 * orbIndices[2 * index] + closed[i] % 2,
            b = 2 * orbIndices[2 * index + 1] + closed[j] % 2;

        if (a >= 2*first_virtual && b >= 2*first_virtual && closed[i] < 2*first_virtual && closed[j] < 2*first_virtual) continue;

        //if ((!(d.getocc(a) || d.getocc(b))) && (a < 2*schd.numActive) && (b < 2*schd.numActive)) {//uncomment for VMC active space calculations
        if (!(d.getocc(a) || d.getocc(b))) {
          //cout << "a   " << a << "  b  " << b << endl;
          work.appendValue(0.0, closed[i] * 2 * norbs + a,
                           closed[j] * 2 * norbs + b, integrals[index]);
        }
      }
    }
  }

}

void generateAllScreenedSingleExcitationsDyall(const Determinant& det,
                                         const Determinant& detAct,
                                         const double& THRESH,
                                         const double& TINY,
                                         workingArray& work,
                                         bool doparity) {
  int norbs = Determinant::norbs;
  int first_virtual = schd.nciCore + schd.nciAct;

  vector<int> closed;
  vector<int> open;
  det.getOpenClosed(open, closed);

  //schd.active = number of active spatial orbitals, assumed to be contiguous and at the beginning

  for (int i = 0; i < closed.size(); i++) {
    for (int a = 0; a < open.size(); a++) {
      if (closed[i] % 2 == open[a] % 2 &&
          abs(I2hb.Singles(closed[i], open[a])) > THRESH)
      {
        int I = closed[i] / 2, A = open[a] / 2;

        const double tia = det.Hij_1ExciteScreened(open[a], closed[i], I2hb,
                                                 TINY, doparity);

        double tiaD = 0.;
        if (I < first_virtual && A < first_virtual)
          tiaD = detAct.Hij_1ExciteScreened(open[a], closed[i], I2hb,
                                                 TINY, doparity);

        if (abs(tia) > THRESH) {
          work.appendValue(tiaD, closed[i]*2*norbs+open[a], 0, tia);//jailbreaking overlapRatio for Dyall ham element (tiaD)
        }
      }
    }
  }

}

void generateAllScreenedDoubleExcitationsDyall(const Determinant& det,
                                         const double& THRESH,
                                         const double& TINY,
                                         workingArray& work,
                                         bool doparity) {
  int norbs = Determinant::norbs;
  int first_virtual = schd.nciCore + schd.nciAct;

  vector<int> closed;
  vector<int> open;
  det.getOpenClosed(open, closed);

  for (int i=0; i<closed.size(); i++) {
    for (int j = 0; j<i; j++) {

      const float *integrals; const short* orbIndices;
      size_t numIntegrals;
      I2hb.getIntegralArray(closed[i], closed[j], integrals, orbIndices, numIntegrals);
      size_t numLargeIntegrals = std::lower_bound(integrals, integrals + numIntegrals, THRESH, [](const float &x, float val){ return fabs(x) > val; }) - integrals;

      // for all HCI integrals
      for (size_t index = 0; index < numLargeIntegrals; index++)
      {
        // if we are going below the criterion, break
        //if (fabs(integrals[index]) < THRESH)
        //  break;

        // otherwise: generate the determinant corresponding to the current excitation
        int a = 2 * orbIndices[2 * index] + closed[i] % 2,
            b = 2 * orbIndices[2 * index + 1] + closed[j] % 2;

        double flag = 0.;
        if (closed[i] < 2*first_virtual && a < 2*first_virtual && b < 2*first_virtual) flag = 1.0; //Dyall excitation, (note j < i, so no j condition)
        //if ((!(d.getocc(a) || d.getocc(b))) && (a < 2*schd.numActive) && (b < 2*schd.numActive)) {//uncomment for VMC active space calculations
        if (!(det.getocc(a) || det.getocc(b))) {
          //cout << "a   " << a << "  b  " << b << endl;
          work.appendValue(flag, closed[i] * 2 * norbs + a,
                           closed[j] * 2 * norbs + b, integrals[index]);
        }
      }
    }
  }

}

//---Generate all screened excitations into the CAS-------------------

//---From excitation class 0 (the CAS itself) into the CAS------------
void generateAllScreenedSingleExcitationsCAS_0h0p(const Determinant& d,
                                                  const double& THRESH,
                                                  const double& TINY,
                                                  workingArray& work,
                                                  bool doparity) {
  int norbs = Determinant::norbs;
  int first_virtual = schd.nciCore + schd.nciAct;

  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);

  auto ub_1 = upper_bound(open.begin(), open.end(), 2*first_virtual - 1);
  int indAct = distance(open.begin(), ub_1);

  auto ub_2 = upper_bound(closed.begin(), closed.end(), 2*schd.nciCore - 1);
  int indCore = distance(closed.begin(), ub_2);

  for (int i = indCore; i < closed.size(); i++) {
    for (int a = 0; a < indAct; a++) {
      if (closed[i] % 2 == open[a] % 2 &&
          abs(I2hb.Singles(closed[i], open[a])) > THRESH)
      {
        const double tia = d.Hij_1ExciteScreened(open[a], closed[i], I2hb,
                                                 TINY, doparity);

        if (abs(tia) > THRESH) {
          work.appendValue(0., closed[i]*2*norbs+open[a], 0, tia);
        }
      }
    }
  }
}

//---From excitation class 0 (the CAS itself) into the CAS------------
void generateAllScreenedDoubleExcitationsCAS_0h0p(const Determinant& d,
                                                  const double& THRESH,
                                                  workingArray& work) {
  int norbs = Determinant::norbs;
  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);

  auto ub = upper_bound(closed.begin(), closed.end(), 2*schd.nciCore - 1);
  int indCore = distance(closed.begin(), ub);

  int nclosed = closed.size();
  for (int i = indCore; i<nclosed; i++) {
    for (int j = indCore; j<i; j++) {

      const float *integrals; const short* orbIndices;
      size_t numIntegrals;
      I2hbCAS.getIntegralArrayCAS(closed[i], closed[j], integrals, orbIndices, numIntegrals);
      size_t numLargeIntegrals = std::lower_bound(integrals, integrals + numIntegrals, THRESH, [](const float &x, float val){ return fabs(x) > val; }) - integrals;

      // for all HCI integrals
      for (size_t index = 0; index < numLargeIntegrals; index++)
      {
        // if we are going below the criterion, break
        //if (fabs(integrals[index]) < THRESH)
        //  break;

        // otherwise: generate the determinant corresponding to the current excitation
        int a = 2 * orbIndices[2 * index] + closed[i] % 2;
        int b = 2 * orbIndices[2 * index + 1] + closed[j] % 2;

        //if ((!(d.getocc(a) || d.getocc(b))) && (a < 2*schd.numActive) && (b < 2*schd.numActive)) {//uncomment for VMC active space calculations
        if (!(d.getocc(a) || d.getocc(b))) {
          //cout << "a   " << a << "  b  " << b << endl;
          work.appendValue(0.0, closed[i] * 2 * norbs + a,
                           closed[j] * 2 * norbs + b, integrals[index]);
        }
      }
    }
  }
}

//---From excitation class 1 (0 holes in core, 1 particle in virtuals) into the CAS------------
void generateAllScreenedSingleExcitationsCAS_0h1p(const Determinant& d,
                                                  const double& THRESH,
                                                  const double& TINY,
                                                  workingArray& work,
                                                  const int& i,
                                                  bool doparity) {
  int norbs = Determinant::norbs;
  int first_virtual = schd.nciCore + schd.nciAct;

  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);

  //schd.active = number of active spatial orbitals, assumed to be contiguous and at the beginning
  //auto ub = upper_bound(open.begin(), open.end(), 2*schd.numActive - 1);
  auto ub = upper_bound(open.begin(), open.end(), 2*first_virtual - 1);
  int indAct = distance(open.begin(), ub);

  for (int a = 0; a < indAct; a++) {
    if (i % 2 == open[a] % 2 &&
        abs(I2hb.Singles(i, open[a])) > THRESH)
    {
      const double tia = d.Hij_1ExciteScreened(open[a], i, I2hb,
                                               TINY, doparity);

      if (abs(tia) > THRESH) {
        work.appendValue(0., i*2*norbs+open[a], 0, tia);
      }
    }
  }
}

//---From excitation class 1 (0 holes in core, 1 particle in virtuals) into the CAS------------
void generateAllScreenedDoubleExcitationsCAS_0h1p(const Determinant& d,
                                                  const double& THRESH,
                                                  workingArray& work,
                                                  const int& i) {
  int norbs = Determinant::norbs;
  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);

  auto ub = upper_bound(closed.begin(), closed.end(), 2*schd.nciCore - 1);
  int indCore = distance(closed.begin(), ub);

  int nclosed = closed.size();
  for (int n=indCore; n < nclosed-1; n++) {
    int j = closed[n];

    const float *integrals; const short* orbIndices;
    size_t numIntegrals;
    I2hb.getIntegralArrayCAS(i, j, integrals, orbIndices, numIntegrals);
    size_t numLargeIntegrals = std::lower_bound(integrals, integrals + numIntegrals, THRESH, [](const float &x, float val){ return fabs(x) > val; }) - integrals;

    // for all HCI integrals
    for (size_t index = 0; index < numLargeIntegrals; index++)
    {
      // if we are going below the criterion, break
      //if (fabs(integrals[index]) < THRESH)
      //  break;

      // otherwise: generate the determinant corresponding to the current excitation
      int a = 2 * orbIndices[2 * index] + i % 2;
      int b = 2 * orbIndices[2 * index + 1] + j % 2;

      //if ((!(d.getocc(a) || d.getocc(b))) && (a < 2*schd.numActive) && (b < 2*schd.numActive)) {//uncomment for VMC active space calculations
      if (!(d.getocc(a) || d.getocc(b))) {
        //cout << "a   " << a << "  b  " << b << endl;
        work.appendValue(0.0, i * 2 * norbs + a, j * 2 * norbs + b, integrals[index]);
      }
    }
  }
}

//---From excitation class 2 (0 holes in core, 2 particles in virtuals) into the CAS------------
void generateAllScreenedExcitationsCAS_0h2p(const Determinant& d,
                                            const double& THRESH,
                                            workingArray& work,
                                            const int& iExc, const int& jExc) {
  int norbs = Determinant::norbs;

  int i = max(iExc, jExc), j = min(iExc, jExc);
  const float *integrals; const short* orbIndices;
  size_t numIntegrals;
  I2hbCAS.getIntegralArrayCAS(i, j, integrals, orbIndices, numIntegrals);
  size_t numLargeIntegrals = std::lower_bound(integrals, integrals + numIntegrals, THRESH, [](const float &x, float val){ return fabs(x) > val; }) - integrals;

  // for all HCI integrals
  for (size_t index = 0; index < numLargeIntegrals; index++)
  {
    // if we are going below the criterion, break
    //if (fabs(integrals[index]) < THRESH)
    //  break;

    // otherwise: generate the determinant corresponding to the current excitation
    int a = 2 * orbIndices[2 * index] + i % 2;
    int b = 2 * orbIndices[2 * index + 1] + j % 2;

    //if ((!(d.getocc(a) || d.getocc(b))) && (a < 2*schd.numActive) && (b < 2*schd.numActive)) {//uncomment for VMC active space calculations
    if (!(d.getocc(a) || d.getocc(b))) {
      //cout << "a   " << a << "  b  " << b << endl;
      work.appendValue(0.0, i * 2 * norbs + a, j * 2 * norbs + b, integrals[index]);
    }
  }
}

//---From excitation class 3 (1 hole in core, 0 particles in virtuals) into the CAS------------
void generateAllScreenedSingleExcitationsCAS_1h0p(const Determinant& d,
                                                  const double& THRESH,
                                                  const double& TINY,
                                                  workingArray& work,
                                                  const int& a,
                                                  bool doparity) {
  int norbs = Determinant::norbs;

  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);

  auto ub = upper_bound(closed.begin(), closed.end(), 2*schd.nciCore - 1);
  int indCore = distance(closed.begin(), ub);

  for (int i = indCore; i < closed.size(); i++) {
    if (a % 2 == closed[i] % 2 &&
        abs(I2hb.Singles(closed[i], a)) > THRESH)
    {
      const double tia = d.Hij_1ExciteScreened(closed[i], a, I2hb, TINY, doparity);

      if (abs(tia) > THRESH) {
        work.appendValue(0., closed[i]*2*norbs+a, 0, tia);
      }
    }
  }
}

//---From excitation class 3 (1 hole in core, 0 particles in virtuals) into the CAS------------
void generateAllScreenedDoubleExcitationsCAS_1h0p(const Determinant& d,
                                                  const double& THRESH,
                                                  workingArray& work,
                                                  const int& a) {
  int norbs = Determinant::norbs;
  int first_virtual = schd.nciCore + schd.nciAct;

  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);

  auto ub = upper_bound(open.begin(), open.end(), 2*first_virtual - 1);
  int indAct = distance(open.begin(), ub);

  int nclosed = closed.size();
  for (int n = 1; n < indAct; n++) {
    int b = open[n];

    const float *integrals; const short* orbIndices;
    size_t numIntegrals;
    //I2hb.getIntegralArrayCAS(i, j, integrals, orbIndices, numIntegrals);
    I2hb.getIntegralArray(a, b, integrals, orbIndices, numIntegrals);
    size_t numLargeIntegrals = std::lower_bound(integrals, integrals + numIntegrals, THRESH, [](const float &x, float val){ return fabs(x) > val; }) - integrals;

    // for all HCI integrals
    for (size_t index = 0; index < numLargeIntegrals; index++)
    {
      // generate the determinant corresponding to the current excitation
      int i = 2 * orbIndices[2 * index] + a % 2;
      int j = 2 * orbIndices[2 * index + 1] + b % 2;

      if (d.getocc(i) && d.getocc(j)) {
        //cout << "a   " << a << "  b  " << b << endl;
        work.appendValue(0.0, i * 2 * norbs + a, j * 2 * norbs + b, integrals[index]);
      }
    }
  }
}

//---From excitation class 4 (1 hole in core, 1 particle in virtuals) into the CAS------------
void generateAllScreenedSingleExcitationsCAS_1h1p(const Determinant& d,
                                                  const double& THRESH,
                                                  const double& TINY,
                                                  workingArray& work,
                                                  const int& i, const int& a,
                                                  bool doparity) {
  int norbs = Determinant::norbs;

  if (i%2 == a%2 && abs(I2hb.Singles(i, a)) > THRESH)
  {
    const double tia = d.Hij_1ExciteScreened(i, a, I2hb, TINY, doparity);

    if (abs(tia) > THRESH) {
      work.appendValue(0., i*2*norbs+a, 0, tia);
    }
  }
}

//---From excitation class 4 (1 hole in core, 1 particle in virtuals) into the CAS------------
void generateAllScreenedDoubleExcitationsCAS_1h1p(const Determinant& d,
                                                  const double& THRESH,
                                                  workingArray& work,
                                                  const int& i, const int& a) {
  int norbs = Determinant::norbs;
  int max_act_ind = 2*(schd.nciCore + schd.nciAct) - 1;

  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);

  auto ub = upper_bound(closed.begin(), closed.end(), 2*schd.nciCore - 1);
  int indCore = distance(closed.begin(), ub);

  int nclosed = closed.size();
  for (int n=indCore; n < nclosed-1; n++) {
    int j = closed[n];

    const float *integrals; const short* orbIndices;
    size_t numIntegrals;
    //I2hb.getIntegralArrayCAS(i, j, integrals, orbIndices, numIntegrals);
    I2hb.getIntegralArray(i, j, integrals, orbIndices, numIntegrals);
    size_t numLargeIntegrals = std::lower_bound(integrals, integrals + numIntegrals, THRESH, [](const float &x, float val){ return fabs(x) > val; }) - integrals;

    // for all HCI integrals
    for (size_t index = 0; index < numLargeIntegrals; index++)
    {
      // otherwise: generate the determinant corresponding to the current excitation
      int a_new = 2 * orbIndices[2 * index] + i % 2;
      int b_new = 2 * orbIndices[2 * index + 1] + j % 2;

      if (a_new > max_act_ind || b_new > max_act_ind) continue;

      if (a_new == a || b_new == a) {
        if (!(d.getocc(a_new) || d.getocc(b_new))) {
          //cout << i << "  " << j << "  " << a_new << "  " << b_new << "  " << integrals[index] << endl;
          work.appendValue(0.0, i * 2 * norbs + a_new, j * 2 * norbs + b_new, integrals[index]);
        }
      }

    }
  }
}

//---From excitation class 4 (1 hole in core, 1 particle in virtuals) into the CAS------------
// An alternative implementation of this function for testing purposes
// TODO: Remove when all debugging has been done
//void generateAllScreenedDoubleExcitationsCAS_1h1p(const Determinant& d,
//                                                  const double& THRESH,
//                                                  workingArray& work,
//                                                  const int& iExc, const int& aExc) {
//  int norbs = Determinant::norbs;
//  int first_virtual = schd.nciCore + schd.nciAct;
//
//  vector<int> closed;
//  vector<int> open;
//  d.getOpenClosed(open, closed);
//
//  double integral;
//  int i, j, a, b, b_temp;
//
//  auto ub_1 = upper_bound(open.begin(), open.end(), 2*first_virtual - 1);
//  int indAct = distance(open.begin(), ub_1);
//
//  auto ub_2 = upper_bound(closed.begin(), closed.end(), 2*schd.nciCore - 1);
//  int indCore = distance(closed.begin(), ub_2);
//
//  for (int n = indCore; n < closed.size(); n++) {
//    for (int m = 0; m < indAct; m++) {
//
//      i = max(iExc, closed[n]), j = min(iExc, closed[n]);
//      b_temp = open[m];
//
//      if (i%2 == aExc%2 && j%2 == b_temp%2) {
//        a = aExc;
//        b = b_temp;
//      }
//      else if (i%2 == b_temp%2 && j%2 == aExc%2) {
//        a = b_temp;
//        b = aExc;
//      }
//      else {
//        continue;
//      }
//
//      if (i%2 == j%2) {
//        // same spin
//        integral = I2(i, a, j, b) - I2(i, b, j, a);
//      }
//      else {
//        // opposite spin
//        integral = I2(i, a, j, b);
//      }
//
//      if (fabs(integral) > THRESH) {
//        //cout << i << "  " << j << "  " << a << "  " << b << "  " << integral << endl;
//        work.appendValue(0.0, i * 2 * norbs + a, j * 2 * norbs + b, integral);
//      }
//
//    }
//  }
//}

//---From excitation class 5 (1 hole in core, 2 particles in virtuals) into the CAS------------
void generateAllScreenedExcitationsCAS_1h2p(const Determinant& d,
                                            const double& THRESH,
                                            workingArray& work,
                                            const int& iExc, const int& jExc,
                                            const int& a) {
  int norbs = Determinant::norbs;
  int max_act_ind = 2*(schd.nciCore + schd.nciAct) - 1;

  int i = max(iExc, jExc), j = min(iExc, jExc);
  const float *integrals; const short* orbIndices;
  size_t numIntegrals;
  I2hbCAS.getIntegralArray(i, j, integrals, orbIndices, numIntegrals);
  size_t numLargeIntegrals = std::lower_bound(integrals, integrals + numIntegrals, THRESH, [](const float &x, float val){ return fabs(x) > val; }) - integrals;

  // for all HCI integrals
  for (size_t index = 0; index < numLargeIntegrals; index++)
  {
    // otherwise: generate the determinant corresponding to the current excitation
    int a_new = 2 * orbIndices[2 * index] + i % 2;
    int b_new = 2 * orbIndices[2 * index + 1] + j % 2;

    if (a_new > max_act_ind || b_new > max_act_ind) continue;

    if (a_new == a || b_new == a) {
      if (!(d.getocc(a_new) || d.getocc(b_new))) {
        //cout << "a   " << a << "  b  " << b << endl;
        work.appendValue(0.0, i * 2 * norbs + a_new, j * 2 * norbs + b_new, integrals[index]);
      }
    }
  }
}

//---From excitation class 5 (1 hole in core, 2 particles in virtuals) into the CAS------------
// An alternative implementation of this function for testing purposes
// TODO: Remove when all debugging has been done
//void generateAllScreenedExcitationsCAS_1h2p(const Determinant& d,
//                                            const double& THRESH,
//                                            workingArray& work,
//                                            const int& iExc, const int& jExc,
//                                            const int& aExc) {
//  int norbs = Determinant::norbs;
//  int first_virtual = schd.nciCore + schd.nciAct;
//
//  vector<int> closed;
//  vector<int> open;
//  d.getOpenClosed(open, closed);
//
//  double integral;
//  int a, b, b_temp;
//
//  int i = max(iExc, jExc), j = min(iExc, jExc);
//
//  auto ub = upper_bound(open.begin(), open.end(), 2*first_virtual - 1);
//  int indAct = distance(open.begin(), ub);
//
//  for (int m = 0; m < indAct; m++) {
//    b_temp = open[m];
//
//    if (i%2 == aExc%2 && j%2 == b_temp%2) {
//      a = aExc;
//      b = b_temp;
//    }
//    else if (i%2 == b_temp%2 && j%2 == aExc%2) {
//      a = b_temp;
//      b = aExc;
//    }
//    else {
//      continue;
//    }
//
//    if (i%2 == j%2) {
//      // same spin
//      integral = I2(i, a, j, b) - I2(i, b, j, a);
//    }
//    else {
//      // opposite spin
//      integral = I2(i, a, j, b);
//    }
//
//    if (fabs(integral) > THRESH) {
//      //cout << i << "  " << j << "  " << a << "  " << b << "  " << integral << endl;
//      work.appendValue(0.0, i * 2 * norbs + a, j * 2 * norbs + b, integral);
//    }
//  }
//}

//---From excitation class 6 (2 holes in core, 0 particles in virtuals) into the CAS------------
void generateAllScreenedExcitationsCAS_2h0p(const Determinant& d,
                                            const double& THRESH,
                                            workingArray& work,
                                            const int& aExc, const int& bExc) {
  int norbs = Determinant::norbs;

  int a = max(aExc, bExc), b = min(aExc, bExc);
  const float *integrals; const short* orbIndices;
  size_t numIntegrals;
  I2hbCAS.getIntegralArrayCAS(a, b, integrals, orbIndices, numIntegrals);
  size_t numLargeIntegrals = std::lower_bound(integrals, integrals + numIntegrals, THRESH, [](const float &x, float val){ return fabs(x) > val; }) - integrals;

  // for all HCI integrals
  for (size_t index = 0; index < numLargeIntegrals; index++)
  {
    // otherwise: generate the determinant corresponding to the current excitation
    int i = 2 * orbIndices[2 * index] + a % 2;
    int j = 2 * orbIndices[2 * index + 1] + b % 2;

    if (d.getocc(i) && d.getocc(j)) {
      //cout << "a   " << a << "  b  " << b << endl;
      work.appendValue(0.0, i * 2 * norbs + a, j * 2 * norbs + b, integrals[index]);
    }
  }
}

//---From excitation class 8 (2 holes in core, 2 particles in virtuals) into the CAS------------
void generateAllScreenedExcitationsCAS_2h2p(const Determinant& d,
                                            const double& THRESH,
                                            workingArray& work,
                                            const int& iExc, const int& jExc,
                                            const int& aExc, const int& bExc) {
  int norbs = Determinant::norbs;

  int i = max(iExc, jExc), j = min(iExc, jExc);

  int a, b;

  if (i%2 == aExc%2 && j%2 == bExc%2) {
    a = aExc;
    b = bExc;
  }
  else if (i%2 == bExc%2 && j%2 == aExc%2) {
    a = bExc;
    b = aExc;
  }
  else {
    return;
  }

  double integral;

  if (i%2 == j%2) {
    // same spin
    integral = I2(i, a, j, b) - I2(i, b, j, a);
  }
  else {
    // opposite spin
    integral = I2(i, a, j, b);
  }

  if (fabs(integral) > THRESH) {
    work.appendValue(0.0, i * 2 * norbs + a, j * 2 * norbs + b, integral);
  }
}

bool applyExcitation(int a, int b, int k, int l, Determinant& dcopy) {
  bool valid = true;

  if (dcopy.getocc(l) == true)
    dcopy.setocc(l, false);
  else
    return false;

  if (dcopy.getocc(b) == false)
    dcopy.setocc(b, true);
  else
    return false;
  
  if (dcopy.getocc(k) == true)
    dcopy.setocc(k, false);
  else
    return false;

  if (dcopy.getocc(a) == false)
    dcopy.setocc(a, true);
  else
    return false;
  
  return valid;
}

//generate all the alpha or beta strings
void comb(int N, int K, vector<vector<int>> &combinations)
{
  std::vector<int> bitmask(K, 1);
  bitmask.resize(N, 0); // N-K trailing 0's

  // print integers and permute bitmask
  int index = 0;
  do
  {
    vector<int> comb;
    for (int i = 0; i < N; ++i) // [0..N-1] integers
    {
      if (bitmask[i] == 1)
        comb.push_back(i);
    }
    combinations.push_back(comb);
  } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
}

void generateAllDeterminants(vector<Determinant>& allDets, int norbs, int nalpha, int nbeta) {
  vector<vector<int>> alphaDets, betaDets;
  comb(norbs, nalpha, alphaDets);
  comb(norbs, nbeta, betaDets);
  
  for (int a = 0; a < alphaDets.size(); a++)
    for (int b = 0; b < betaDets.size(); b++)
    {
      Determinant d;
      for (int i = 0; i < alphaDets[a].size(); i++)
        d.setoccA(alphaDets[a][i], true);
      for (int i = 0; i < betaDets[b].size(); i++)
        d.setoccB(betaDets[b][i], true);
      allDets.push_back(d);
    }

  alphaDets.clear();
  betaDets.clear();
}
