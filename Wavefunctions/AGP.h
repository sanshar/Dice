/*
  Developed by Sandeep Sharma
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
#ifndef AGP_HEADER_H
#define AGP_HEADER_H
#include <vector>
#include <set>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>


class oneInt;
class twoInt;
class twoIntHeatBathSHM;
class Determinant;
class workingArray;

using namespace Eigen;

/**
* This is the wavefunction, it is a linear combination of
* slater determinants made of Hforbs
*/
class AGP {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
      ar & pairMat;
  }

  
 public:
  
   MatrixXcd pairMat;        //pairing matrix, F_pq

   /**
    * constructor
    */
   AGP();


  //variables are ordered as:
  //cicoeffs of the reference multidet expansion, followed by hforbs (row major)
  //in case of uhf all alpha first followed by beta
  void getVariables(Eigen::VectorBlock<VectorXd> &v) const;
  long getNumVariables() const;
  void updateVariables(const Eigen::VectorBlock<VectorXd> &v);
  void printVariables() const;
  const MatrixXcd& getPairMat() const { return pairMat;}
  string getfileName() const {return "AGP";};
  
};


#endif
