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
#include "global.h"

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
    std::vector<double> store;
    int norbs;
    //I explicitly store all elements of the matrix
    //so for normal operator if i and j dont have the same spin
    //then it will just return zero. If we have SOC and
    // i and j have different spin then it can be a complex number.
    inline double& operator()(int i, int j) { return store.at(i*norbs+j); }
    inline double operator()(int i, int j) const { return store.at(i*norbs+j); }
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
           & npair   \
           & ksym;
      }

  public:
    double* store;
    double maxEntry;
    MatrixXd Direct, Exchange;
    double zero ;
    size_t norbs;
    size_t npair;
    size_t inner;
    size_t virt;
    size_t nii;
    size_t niv;
    size_t nvv;
    size_t niiii;
    size_t niiiv;
    size_t niviv;
    size_t niivv;
    bool ksym;
    twoInt() :zero(0.0),maxEntry(100.) {}
    inline double& operator()(int i, int j, int k, int l) {
      zero = 0.0;
      if (!((i%2 == j%2) && (k%2 == l%2))) return zero;
      int I=i/2;int J=j/2;int K=k/2;int L=l/2;

      if(!ksym) {
        //unsigned int IJ = max(I,J)*(max(I,J)+1)/2 + min(I,J);
        //unsigned int KL = max(K,L)*(max(K,L)+1)/2 + min(K,L);
        //unsigned int A = max(IJ,KL), B = min(IJ,KL);
        //return store[A*(A+1)/2+B];
        //unsigned int IJ = min(I,J) * norbs -(min(I,J) * (min(I,J) - 1)) / 2 + max(I,J) - min(I,J);
        //unsigned int KL = min(K,L) * norbs -(min(K,L)* (min(K,L) - 1)) / 2 + max(K,L) - min(K,L);
        size_t IJ, KL;
        bool iv1 = false, iv2 = false;
        if (I < inner && J < inner) IJ = max(I,J)*(max(I,J)+1)/2 + min(I,J);
        else if (I >= inner && J >= inner) IJ = inner*(inner+1)/2 + inner*virt + max(I-inner,J-inner)*(max(I-inner,J-inner)+1)/2 + min(I-inner,J-inner);
        else {IJ = inner*(inner+1)/2 + inner*(max(I,J)-inner) + min(I,J); iv1 = true;}
        if (K < inner && L < inner) KL = max(K,L)*(max(K,L)+1)/2 + min(K,L);
        else if (K >= inner && L >= inner) KL = inner*(inner+1)/2 + inner*virt + max(K-inner,L-inner)*(max(K-inner,L-inner)+1)/2 + min(K-inner,L-inner);
        else {KL = inner*(inner+1)/2 + inner*(max(K,L)-inner) + min(K,L); iv2 = true;}
        size_t ind;
        if (iv1 && iv2) ind = niiii + niiiv + niivv + max(IJ-nii,KL-nii)*(max(IJ-nii,KL-nii)+1)/2 + min(IJ-nii,KL-nii);
        else ind = min(IJ,KL) * npair - (min(IJ,KL) * (min(IJ,KL) - 1)) / 2 + max(IJ,KL) - min(IJ,KL);
        //cout << I << "  " << J << "  " << K << "  " << L << "    " << IJ << "  " << KL << "    " << ind << endl;
        return store[ind];
      } else {
        unsigned int IJ = I*norbs+J, KL = K*norbs+L;
        unsigned int A = max(IJ,KL), B = min(IJ,KL);
        return store[A*(A+1)/2+B];
      }
    }

  inline double operator()(int i, int j, int k, int l) const {
    double zero = 0.0;
    if (!((i%2 == j%2) && (k%2 == l%2))) return zero;
    int I=i/2;int J=j/2;int K=k/2;int L=l/2;
    
    if(!ksym) {
      //unsigned int IJ = max(I,J)*(max(I,J)+1)/2 + min(I,J);
      //unsigned int KL = max(K,L)*(max(K,L)+1)/2 + min(K,L);
      //unsigned int A = max(IJ,KL), B = min(IJ,KL);
      //return store[A*(A+1)/2+B];
      //unsigned int IJ = min(I,J) * norbs -(min(I,J) * (min(I,J) - 1)) / 2 + max(I,J) - min(I,J);
      //unsigned int KL = min(K,L) * norbs -(min(K,L)* (min(K,L) - 1)) / 2 + max(K,L) - min(K,L);
      size_t IJ, KL;
      bool iv1 = false, iv2 = false;
      if (I < inner && J < inner) IJ = max(I,J)*(max(I,J)+1)/2 + min(I,J);
      else if (I >= inner && J >= inner) IJ = inner*(inner+1)/2 + inner*virt + max(I-inner,J-inner)*(max(I-inner,J-inner)+1)/2 + min(I-inner,J-inner);
      else {IJ = inner*(inner+1)/2 + inner*(max(I,J)-inner) + min(I,J); iv1 = true;}
      if (K < inner && L < inner) KL = max(K,L)*(max(K,L)+1)/2 + min(K,L);
      else if (K >= inner && L >= inner) KL = inner*(inner+1)/2 + inner*virt + max(K-inner,L-inner)*(max(K-inner,L-inner)+1)/2 + min(K-inner,L-inner);
      else {KL = inner*(inner+1)/2 + inner*(max(K,L)-inner) + min(K,L); iv2 = true;}
      size_t ind;
      if (iv1 && iv2) ind = niiii + niiiv + niivv + max(IJ-nii,KL-nii)*(max(IJ-nii,KL-nii)+1)/2 + min(IJ-nii,KL-nii);
      else ind = min(IJ,KL) * npair - (min(IJ,KL) * (min(IJ,KL) - 1)) / 2 + max(IJ,KL) - min(IJ,KL);
      return store[ind];
    } else {
      unsigned int IJ = I*norbs+J, KL = K*norbs+L;
      unsigned int A = max(IJ,KL), B = min(IJ,KL);
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
    std::map<std::pair<short,short>, std::multimap<float, std::pair<short,short>, compAbs > > sameSpin;
    std::map<std::pair<short,short>, std::multimap<float, std::pair<short,short>, compAbs > > oppositeSpin;
    std::map<std::pair<short,short>, std::multimap<float, short, compAbs > > singleIntegrals;

    MatrixXd Singles;
 
    double epsilon;
    double zero ;
    twoIntHeatBath(double epsilon_) :zero(0.0),epsilon(fabs(epsilon_)) {}
 
    //the orbs contain all orbitals used to make the ij pair above
    //typically these can be all orbitals of the problem or just the active space ones
    //ab will typically contain all orbitals(norbs)
    void constructClass(std::vector<int>& orbs, twoInt& I2, oneInt& I1, int ncore, int nact, bool cas=false) {
      int first_virtual = ncore + nact;

      int s = nact;
      if (cas) s = orbs.size();
      for (int i=0; i<s; i++) {
        for (int j=0;j<=i;j++) {
          std::pair<short,short> IJ=make_pair(i,j);

          int start = 0;
          int end = orbs.size();
          if (cas) {start = ncore; end = first_virtual;}
	      for (int a=start; a<end; a++) {
	        if (fabs(I2(2*i, 2*j, 2*a, 2*a) - I2(2*i, 2*a, 2*a, 2*j)) > epsilon) 
	          singleIntegrals[IJ].insert(pair<float, short>(I2(2*i, 2*j, 2*a, 2*a) - I2(2*i, 2*a, 2*a, 2*j), 2*a));
	        
	        if (fabs(I2(2*i, 2*j, 2*a+1, 2*a+1) ) > epsilon)
	          singleIntegrals[IJ].insert(pair<float, short>(I2(2*i, 2*j, 2*a+1, 2*a+1), 2*a+1));
	      }
        }
      }

      s = orbs.size();
      for (int i=0; i<s; i++) {
        for (int j=0;j<=i;j++) {
          std::pair<short,short> IJ=make_pair(i,j);
          //sameSpin[IJ]=std::map<double, std::pair<int,int> >();
          //oppositeSpin[IJ]=std::map<double, std::pair<int,int> >();
          for (int a=ncore; a<first_virtual; a++) {
            for (int b=ncore; b<first_virtual; b++) {
              //opposite spin
              if (fabs(I2(2*i, 2*a, 2*j, 2*b)) > epsilon)
                oppositeSpin[IJ].insert(pair<float, std::pair<short,short> >(I2(2*i, 2*a, 2*j, 2*b), make_pair(a,b)));
              //samespin
              if (a>=b && fabs(I2(2*i,2*a,2*j,2*b) - I2(2*i,2*b,2*j,2*a)) > epsilon) {
                sameSpin[IJ].insert(pair<float, std::pair<short,short> >( I2(2*i,2*a,2*j,2*b) - I2(2*i,2*b,2*j,2*a), make_pair(a,b)));
                //sameSpin[IJ][fabs(I2(2*i,2*a,2*j,2*b) - I2(2*i,2*b,2*j,2*a))] = make_pair<int,int>(a,b);
              }
            }
          }
        } 
      } // ij  
      
      Singles = MatrixXd::Zero(2*nact, 2*nact);
      if (!cas) {
        for (int i=0; i<2*nact; i++) {
          for (int a=0; a<2*nact; a++) {
            Singles(i,a) = std::abs(I1(i,a));
            for (int j=0; j<2*orbs.size(); j++) {
              //if (fabs(Singles(i,a)) < fabs(I2(i,a,j,j) - I2(i, j, j, a)))
	          Singles(i,a) += std::abs(I2(i,a,j,j) - I2(i, j, j, a));
            }
          }
        }
      }
    } // end constructClass
};



class twoIntHeatBathSHM {
  public:
    float* sameSpinIntegrals;
    float* oppositeSpinIntegrals;
    float* singleIntegrals;

    size_t* startingIndicesSameSpin;
    size_t* startingIndicesOppositeSpin;
    size_t* startingIndicesSingleIntegrals;

    short* sameSpinPairs;
    short* oppositeSpinPairs;
    short* singleIntegralsPairs;

    MatrixXd Singles;

    //for each pair i,j it has sum_{a>b} abs((ai|bj)-(aj|bi)) if they are same spin
    //and sum_{ab} abs(ai|bj) if they are opposite spins
    MatrixXd sameSpinPairExcitations;
    MatrixXd oppositeSpinPairExcitations;

    double epsilon;
    twoIntHeatBathSHM(double epsilon_) : epsilon(fabs(epsilon_)) {}
 
    void constructClass(int norbs, twoIntHeatBath& I2, bool cas) ;

  void getIntegralArray(int i, int j, const float* &integrals,
                        const short* &orbIndices, size_t& numIntegrals) const ;
  
  void getIntegralArrayCAS(int i, int j, const float* &integrals,
                        const short* &orbIndices, size_t& numIntegrals) const ;
};


#ifdef Complex
void readSOCIntegrals(oneInt& I1soc, int norbs, string fileprefix);

void readGTensorIntegrals(vector<oneInt>& I1soc, int norbs, string fileprefix);
#endif

int readNorbs(string fcidump);


void readIntegralsAndInitializeDeterminantStaticVariables(string fcidump);

void readIntegralsHDF5AndInitializeDeterminantStaticVariables(string fcidump);

void readDQMCIntegralsRG(string fcidump, int& norbs, int& nalpha, int& nbeta, double& ecore, Eigen::MatrixXd& h1, Eigen::MatrixXd& h1Mod, std::vector<Eigen::Map<Eigen::MatrixXd>>& chol, std::vector<Eigen::Map<Eigen::MatrixXd>>& cholMat, bool ghf=false);
void readDQMCIntegralsU(string fcidump, int& norbs, int& nalpha, int& nbeta, double& ecore, std::array<Eigen::MatrixXd, 2>& h1, std::array<Eigen::MatrixXd, 2>& h1Mod, std::vector<std::array<Eigen::Map<Eigen::MatrixXd>, 2>>& chol);
void readDQMCIntegralsSOC(string fcidump, int& norbs, int& nelec, double& ecore, Eigen::MatrixXcd& h1, Eigen::MatrixXcd& h1Mod, std::vector<Eigen::Map<Eigen::MatrixXd>>& chol);

#endif
