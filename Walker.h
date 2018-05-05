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
#ifndef Walker_HEADER_H
#define Walker_HEADER_H

#include "Determinants.h"
#include <boost/serialization/serialization.hpp>
#include <Eigen/Dense>
class Wfn;
class CPSSlater;
class oneInt;

class Walker {

 public:
  Determinant d;
  Eigen::MatrixXd alphainv; 
  Eigen::MatrixXd betainv;
  double alphaDet;
  double betaDet;

 Walker(Determinant& pd) : d(pd) {};
  void initUsingWave(Wfn& w, bool check=false) ;

  //these are not absolute orbital indices, but instead the
  //ith occupied and ath unoccupied
  void updateA(int i, int a, CPSSlater& w);
  void updateB(int i, int a, CPSSlater& w);
  double getDetFactorA(int i, int a, CPSSlater& w);
  double getDetFactorB(int i, int a, CPSSlater& w);
  bool makeMove(CPSSlater& w);
  bool makeMovePropPsi(CPSSlater& w);
  bool makeCleverMove(CPSSlater& w);

  void genAllMoves(CPSSlater& w, vector<Determinant>& dout, 
		   vector<double>& prob, vector<size_t>& alphaExcitation,
		   vector<size_t>& betaExcitation);
  void genAllMoves2(CPSSlater& w, vector<Determinant>& dout, 
		   vector<double>& prob, vector<size_t>& alphaExcitation,
		   vector<size_t>& betaExcitation);

};

#endif
