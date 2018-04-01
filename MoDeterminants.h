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
#ifndef MoDeterminants_HEADER_H
#define MoDeterminants_HEADER_H
#include <Eigen/Dense>
#include <vector>
#include "iowrapper.h"

class oneInt;
class twoInt;
class Determinant;

//Stores the Determinant in the form of a matrix
//Each column is are the coefficients of the particular MO
//the number of rows is equal to the number of AO (should be orthogonal) 
class MoDeterminant {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & AlphaOrbitals & BetaOrbitals;
  }


  double Overlap(std::vector<int>& alpha, std::vector<int>& beta);
  void HamAndOvlp(std::vector<int>& alpha, std::vector<int>& beta, 
		  double& ovlp, double& ham,
		  oneInt& I1, twoInt& I2, double& coreE);


 public:
  Eigen::MatrixXd AlphaOrbitals; 
  Eigen::MatrixXd BetaOrbitals; 
  static int norbs;
  static int nalpha;
  static int nbeta;

  MoDeterminant(Eigen::MatrixXd& a, Eigen::MatrixXd& b) : AlphaOrbitals(a), BetaOrbitals(b){};

  double OverlapA(Determinant& d, int i, int a,
		  Eigen::MatrixXd& alphainv, Eigen::MatrixXd &betainv,
		  bool doparity = true);
  double OverlapB(Determinant& d, int i, int a,
		  Eigen::MatrixXd& alphainv, Eigen::MatrixXd &betainv,
		  bool doparity = true);
  double OverlapAA(Determinant& d, int i, int j, int a, int b,
		   Eigen::MatrixXd& alphainv, Eigen::MatrixXd &betainv,
		   bool doparity = true);
  double OverlapBB(Determinant& d, int i, int j, int a, int b,
		   Eigen::MatrixXd& alphainv, Eigen::MatrixXd &betainv,
		   bool doparity = true);
  double OverlapAB(Determinant& d, int i, int j, int a, int b,
		   Eigen::MatrixXd& alphainv, Eigen::MatrixXd &betainv,
		   bool doparity = true);
  void getDetMatrix(Determinant& d, Eigen::MatrixXd& alpha, Eigen::MatrixXd &beta);
  double Overlap(MoDeterminant& m);

  void HamAndOvlp(Determinant& d, 
		  double& ovlp, double& ham,
		  oneInt& I1, twoInt& I2, double& coreE);
  double Overlap(Determinant& d);
    
};

void getunoccupiedOrbs(std::vector<int>& alpha, std::vector<int>& alphaOpen, int& norbs);

#endif
