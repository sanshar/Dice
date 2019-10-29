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
#ifndef BFSlater_HEADER_H
#define BFSlater_HEADER_H
#include <vector>
#include <set>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <Eigen/Dense>
#include "Determinants.h" 

class oneInt;
class twoInt;
class twoIntHeatBathSHM;
class workingArray;
enum HartreeFock {Restricted, UnRestricted, Generalized};


/**
 * This is the wavefunction, it is a linear combination of
 * slater determinants made of Hforbs
 */
class BFSlater {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & hftype
      & det  
      & HforbsA
      & HforbsB
      & bf;
  }
  
  
 public:
  
  HartreeFock hftype;                       //r/u/ghf
  Determinant det;
  Eigen::MatrixXcd HforbsA, HforbsB;        //mo coeffs, HforbsA=HforbsB for r/ghf
  Eigen::MatrixXd bf;  //backflow varibales \eta_ij, where i and j are spatial orbitals, assumed to be the same for up and down spins
  //read mo coeffs fomr hf.txt
  void initDet();
  void initHforbs();
  void initBf();
  
  
  BFSlater();

  //variables are ordered as:
  //cicoeffs of the reference multidet expansion, followed by hforbs (row major): real and complex parts alternating
  //in case of uhf all alpha first followed by beta
  void getVariables(Eigen::VectorBlock<Eigen::VectorXd> &v) const;
  long getNumVariables() const;
  void updateVariables(const Eigen::VectorBlock<Eigen::VectorXd> &v);
  void printVariables() const;
  const Determinant &getDeterminant() const { return det; }
  //int getNumOfDets() const;
  //const std::vector<double> &getciExpansion() const { return ciExpansion; }
  const Eigen::MatrixXcd& getHforbsA() const { return HforbsA;}
  const Eigen::MatrixXcd& getHforbsB() const  {return HforbsB;}
  const Eigen::MatrixXcd& getHforbs(bool sz = 0) const { if(sz == 0) return HforbsA; else return HforbsB;}
  const Eigen::MatrixXd& getBf() const  {return bf;}
  string getfileName() const {return "BFSlater";};

};


#endif
