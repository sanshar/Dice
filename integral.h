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
#ifndef INTEGRAL_HEADER_H
#define INTEGRAL_HEADER_H
#include <vector>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <map>
#include <utility>
#include "iowrapper.h"
#include <boost/interprocess/managed_shared_memory.hpp>

using namespace std;
using namespace Eigen;



bool myfn(double i, double j);



class compAbs {
  public:
    bool operator()(const float& a, const float& b) const { return fabs(a) < fabs(b); }
    bool operator()(const complex<double>& a, const complex<double>& b) const { return std::abs(a) < std::abs(b); }
};



class oneInt {
  private:
    friend class boost::serialization::access;
    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){ ar & store & norbs; }

  public:
    std::vector<CItype> store;
    int norbs;
    //I explicitly store all elements of the matrix
    //so for normal operator if i and j dont have the same spin
    //then it will just return zero. If we have SOC and
    // i and j have different spin then it can be a complex number.
    inline CItype& operator()(int i, int j) { return store.at(i*norbs+j); }
};



class twoInt {
  private:
    friend class boost::serialization::access;
    template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
        ar & maxEntry \
           & Direct \
           & Exchange \
           & zero     \
           & norbs   \
           & ksym;
      }

  public:
    //double* store;
    CItype* store;
    double maxEntry;
    MatrixXd Direct, Exchange;
    //double zero ;
    CItype zero;
    size_t norbs;
    bool ksym;
    twoInt() :zero(0.0),maxEntry(100.) {}
    //inline double& operator()(int i, int j, int k, int l) {
    inline CItype& operator()(int i, int j, int k, int l) {
      zero = complex<double>(0.0,0.0);
      if (!((i%2 == j%2) && (k%2 == l%2))) return zero;
      int I=i/2;int J=j/2;int K=k/2;int L=l/2;

      if(!ksym) {
        int IJ = max(I,J)*(max(I,J)+1)/2 + min(I,J);
        int KL = max(K,L)*(max(K,L)+1)/2 + min(K,L);
        int A = max(IJ,KL), B = min(IJ,KL);
        return store[A*(A+1)/2+B];
      } else {
        int IJ = I*norbs+J, KL = K*norbs+L;
        int A = max(IJ,KL), B = min(IJ,KL);
        return store[A*(A+1)/2+B];
      }
    }
};



class twoIntHeatBath {
  public:
    //i,j,a,b are spatial orbitals
    //first pair is i,j (i>j)
    //the map contains a list of integral which are equal to (ia|jb) where a,b are the second pair
    //if the integrals are for the same spin you have to separately store (ia|jb) and (ib|ja)
    //for opposite spin you just have to store for a>b because the integral is (ia|jb) - (ib|ja)
    //now this class is made by just considering integrals that are smaller than threshhold
    //std::map<std::pair<short,short>, std::multimap<float, std::pair<short,short>, compAbs > > sameSpin;
    std::map<std::pair<short,short>, std::multimap<complex<double>, std::pair<short,short>, compAbs > > sameSpin;
    //std::map<std::pair<short,short>, std::multimap<float, std::pair<short,short>, compAbs > > oppositeSpin;
    std::map<std::pair<short,short>, std::multimap<complex<double>, std::pair<short,short>, compAbs > > oppositeSpin;
    MatrixXd Singles;
 
    double epsilon;
    double zero ;
    twoIntHeatBath(double epsilon_) :zero(0.0),epsilon(fabs(epsilon_)) {}
 
    //the orbs contain all orbitals used to make the ij pair above
    //typically these can be all orbitals of the problem or just the active space ones
    //ab will typically contain all orbitals(norbs)
    void constructClass(std::vector<int>& orbs, twoInt& I2, oneInt& I1, int norbs) {
      for (int i=0; i<orbs.size(); i++)
        for (int j=0;j<=i;j++) {
          std::pair<short,short> IJ=make_pair(i,j);
          //sameSpin[IJ]=std::map<double, std::pair<int,int> >();
          //oppositeSpin[IJ]=std::map<double, std::pair<int,int> >();
          for (int a=0; a<norbs; a++)
            for (int b=0; b<norbs; b++) {
              if (abs(I2(i, a, j, b)) > epsilon) {
              //opposite spin
              //if (fabs(I2(2*i, 2*a, 2*j, 2*b)) > epsilon)
              if (abs(I2(i,a,j,b)) > epsilon)
                oppositeSpin[IJ].insert(pair<complex<double>, std::pair<short,short> >(I2(i,a,j,b), make_pair(a,b)));
                //oppositeSpin[IJ].insert(pair<float, std::pair<short,short> >(I2(2*i, 2*a, 2*j, 2*b), make_pair(a,b)));
              //samespin
              //if (a>=b && fabs(I2(2*i,2*a,2*j,2*b) - I2(2*i,2*b,2*j,2*a)) > epsilon) {
                //sameSpin[IJ].insert(pair<float, std::pair<short,short> >( I2(2*i,2*a,2*j,2*b) - I2(2*i,2*b,2*j,2*a), make_pair(a,b)));
                //sameSpin[IJ][fabs(I2(2*i,2*a,2*j,2*b) - I2(2*i,2*b,2*j,2*a))] = make_pair<int,int>(a,b);
              }
          }
      } // ij
 
      //Singles = MatrixXd::Zero(2*norbs, 2*norbs);
      Singles = MatrixXd::Zero(norbs, norbs);
      for (int i=0; i<2*norbs; i++)
        for (int a=0; a<2*norbs; a++) {
          Singles(i,a) = std::abs(I1(i,a));
          for (int j=0; j<2*norbs; j++) {
            //if (fabs(Singles(i,a)) < fabs(I2(i,a,j,j) - I2(i, j, j, a)))
            if (abs(Singles(i,a)) < abs(I2(i,a,j,j) - I2(i,j,j,a)))
              Singles(i,a) = std::abs(I2(i,a,j,j) - I2(i,j,j,a));
          }
        } 
    } // end constructClass
};



class twoIntHeatBathSHM {
  public:
    float* sameSpinIntegrals;
    float* oppositeSpinIntegrals;
    size_t* startingIndicesSameSpin;
    size_t* startingIndicesOppositeSpin;
    short* sameSpinPairs;
    short* oppositeSpinPairs;
    double* singleExcitation;
    MatrixXd Singles;
 
    double epsilon;
    twoIntHeatBathSHM(double epsilon_) : epsilon(fabs(epsilon_)) {}
 
    void constructClass(int norbs, twoIntHeatBath& I2) ;
};


#ifdef Complex
void readSOCIntegrals(oneInt& I1soc, int norbs, string fileprefix);

void readGTensorIntegrals(vector<oneInt>& I1soc, int norbs, string fileprefix);
#endif

int readNorbs(string fcidump);

void readIntegrals(
        string fcidump,
        twoInt& I2, oneInt& I1,
        int& nelec, int& norbs, double& coreE,
        std::vector<int>& irrep);

#endif
