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
#ifndef HFWalkerHelper_HEADER_H
#define HFWalkerHelper_HEADER_H

#include <array>

enum HartreeFock;

using namespace Eigen;

class Slater;
class Determinant;

//used to efficiently get overlaps when using a slater determinant reference
//in all arrays, first object referes to alpha spins and the second to beta
//in case of ghf, second object of thetaInv is empty, that of thetaDet is 1 
class HFWalkerHelper
{

public:
  HartreeFock hftype;                           //hftype same as that in slater
  array<MatrixXd, 2> thetaInv;          //inverse of the theta matrix
  vector<array<double, 2>> thetaDet;    //determinant of the theta matrix, vector for multidet
  array<vector<int>, 2> openOrbs;       //set of open orbitals in the walker
  array<vector<int>, 2> closedOrbs;     //set of closed orbitals in the walker
  array<vector<int>, 2> closedOrbsRef;  //set of closed orbitals in the reference (zeroth det)
  vector<array<MatrixXd, 2>> rTable;    //table used for efficiently, vector for multidet
  
  /**
   * constructor
   */
  HFWalkerHelper(const Slater &w, const Determinant &d);
  HFWalkerHelper() {};

  //fills open closed orbs 
  void fillOpenClosedOrbs(const Determinant &d);

  //makes rtable using inverse
  void makeTable(const Slater &w, const MatrixXd& inv, const Eigen::Map<VectorXi>& colClosed, int detIndex, bool sz);
  
  //used for multidet calculations, to be used only after inv is calculated
  void calcOtherDetsTables(const Slater& w, bool sz);

  //initializes inverse, dets and tables 
  void initInvDetsTables(const Slater &w);
  
  //concatenates v1 and v2 and adds norbs to v2
  void concatenateGhf(const vector<int>& v1, const vector<int>& v2, vector<int>& result) const;
  
  //makes rtable using inverse
  void makeTableGhf(const Slater &w, const Eigen::Map<VectorXi>& colTheta);
  
  //initializes inverse, dets and tables 
  void initInvDetsTablesGhf(const Slater &w);

  //updates helpers for excitation in the det given by cre and des
  void excitationUpdate(const Slater &w, vector<int>& cre, vector<int>& des, bool sz, double parity, const Determinant& excitedDet);
  
  void excitationUpdateGhf(const Slater &w, vector<int>& cre, vector<int>& des, bool sz, double parity, const Determinant& excitedDet);
  
  //gets relative indices used in the tables relI (i th occupied) and relA (a th unoccupied) for excitation i -> a with spin sz
  void getRelIndices(int i, int &relI, int a, int &relA, bool sz) const; 

};


#endif
