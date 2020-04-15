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
#ifndef MultiSlater_HEADER_H
#define MultiSlater_HEADER_H
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


/**
 * This is the wavefunction, it is a linear combination of
 * slater determinants made of Hforbs
 */
class MultiSlater {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & ref
        & ciParity
        & ciCoeffs
        & numDets
        & Hforbs;
  }
  
  
 public:
 
  std::vector<int> ref;                                      // reference determinant occupations
  std::vector<std::array<Eigen::VectorXi, 2>> ciExcitations; // ci expansion excitations
  std::vector<int> ciParity;                                 // parity factors for the ci exctiations
  std::vector<double> ciCoeffs;                              // ci coeffs
  size_t numDets;                                            // ci expansion size
  Eigen::MatrixXcd Hforbs;                                   // mo coeffs, assuming ghf for now (TODO: crappy notation, not changing now for uniformity) 

  //read mo coeffs from hf.txt
  void initHforbs();

  //initialize the ci expansion by reading determinants generated from Dice
  void initCiExpansion();
 
  //constructor
  MultiSlater();

  //variables are ordered as:
  //cicoeffs of the reference multidet expansion, followed by hforbs (row major): real and complex parts alternating
  void getVariables(Eigen::VectorBlock<Eigen::VectorXd> &v) const;
  size_t getNumVariables() const;
  void updateVariables(const Eigen::VectorBlock<Eigen::VectorXd> &v);
  void printVariables() const;
  size_t getNumOfDets() const { return numDets; }
  const Eigen::MatrixXcd& getHforbs() const { return Hforbs; }
  string getfileName() const { return "MultiSlater"; }

};


#endif
