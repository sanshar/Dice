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
    double* store;
    double maxEntry;
    MatrixXd Direct, Exchange;
    double zero ;
    size_t norbs;
    bool ksym;
    twoInt() :zero(0.0),maxEntry(100.) {}
    inline double& operator()(int i, int j, int k, int l) {
      zero = 0.0;
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



#ifdef Complex
void readSOCIntegrals(oneInt& I1soc, int norbs, string fileprefix);

void readGTensorIntegrals(vector<oneInt>& I1soc, int norbs, string fileprefix);
#endif

int readNorbs(string fcidump);

void readIntegrals(
        string fcidump,
        twoInt& I2, oneInt& I1,
        int& nalpha, int& nbeta, int& norbs, double& coreE,
        std::vector<int>& irrep);

#endif
