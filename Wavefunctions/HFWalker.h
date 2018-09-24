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
#include <array>

using namespace Eigen;

class Slater;
//enum class HartreeFock {Restricted, UnRestricted, Generalized};

//used to efficiently get overlaps when using a slater determinant reference
//in all arrays, first object referes to alpha spins and the second to beta
//in case of ghf, second object of thetaInv is empty, that of thetaDet is 1 
class HFWalkerHelper
{

public:
  int hftype;                           //hftype same as that in slater
  array<MatrixXd, 2> thetaInv;          //inverse of the theta matrix
  vector<array<double, 2>> thetaDet;    //determinant of the theta matrix, vector for multidet
  array<vector<int>, 2> openOrbs;       //set of open orbitals in the walker
  array<vector<int>, 2> closedOrbs;     //set of closed orbitals in the walker
  array<vector<int>, 2> closedOrbsRef;  //set of closed orbitals in the reference (zeroth det)
  vector<array<MatrixXd, 2>> rTable;    //table used for efficiently, vector for multidet
  
  /**
   * constructor
   */
  HFWalkerHelper(Slater &w, Determinant &d);
  HFWalkerHelper() {};

  //fills open closed orbs 
  void fillOpenClosedOrbs(Determinant &d);

  //makes rtable using inverse
  void makeTable(Slater &w, MatrixXd& inv, Eigen::Map<VectorXi>& colClosed, int detIndex, bool sz);
  
  //used for multidet calculations, to be used only after inv is calculated
  void calcOtherDetsTables(Slater& w, bool sz);

  //initializes inverse, dets and tables 
  void initInvDetsTables(Slater &w);
  
  //concatenates v1 and v2 and adds norbs to v2
  void concatenateGhf(vector<int>& v1, vector<int>& v2, vector<int>& result);
  
  //makes rtable using inverse
  void makeTableGhf(Slater &w, Eigen::Map<VectorXi>& colTheta);
  
  //initializes inverse, dets and tables 
  void initInvDetsTablesGhf(Slater &w);

  //updates helpers for excitation in the det given by cre and des
  void excitationUpdate(Slater &w, vector<int>& cre, vector<int> des, bool sz, double parity, Determinant& excitedDet);
  
  void excitationUpdateGhf(Slater &w, vector<int>& cre, vector<int> des, bool sz, double parity, Determinant& excitedDet);
  
  //gets relative indices used in the tables relI (i th occupied) and relA (a th unoccupied) for excitation i -> a with spin sz
  void getRelIndices(int i, int &relI, int a, int &relA, bool sz); 

};

/**
* Is essentially a single determinant used in the VMC/DMC simulation
* At each step in VMC one need to be able to calculate the following
* quantities
* a. The local energy = <walker|H|Psi>/<walker|Psi>
* b. The gradient     = <walker|H|Psi_t>/<walker/Psi>
* c. The update       = <walker'|Psi>/<walker|Psi>
*
* To calculate these efficiently the walker uses the HFWalkerHelper class
*
**/

class HFWalker
{

public:
  Determinant d; //The current determinant
  HFWalkerHelper helper;  

  /**
   * constructors
   */
  //HFWalker(Determinant &pd);
  HFWalker() {};

  HFWalker(Slater &w); 
    
  HFWalker(Slater &w, Determinant &pd);

  /**
   * reads dets from 'BesetDeterminant.txt'
   */
  void readBestDeterminant(Determinant& d);

  /**
   * makes det based on mo coeffs 
   */
  void guessBestDeterminant(Determinant& d, Eigen::MatrixXd& HforbsA, Eigen::MatrixXd& HforbsB); 

  void initDet(MatrixXd& HforbsA, MatrixXd& HforbsB); 
  
  Determinant &getDet() { return d; }
  
  //overlap with i th det in the wavefunction
  double getIndividualDetOverlap(int i) { return helper.thetaDet[i][0] * helper.thetaDet[i][1]; }
  
  //get overlap with the reference
  double getDetOverlap(Slater &w);
  
  //returns < m | Psi0 >/< d | Psi0 >, where m is obtained by exciting the walker with 
  // spin orbital excitations i->a, j->b
  double getDetFactor(int i, int a, Slater &w); 
  double getDetFactor(int I, int J, int A, int B, Slater &w); 
  
  //i, j, a, b are spatial orbitals
  double getDetFactor(int i, int a, bool sz, Slater &w);
  double getDetFactor(int i, int j, int a, int b, bool sz1, bool sz2, Slater &w);

  //updates det and helpers afterspatial orb excitations i->a, j->b with spin sz
  void update(int i, int a, bool sz, Slater &w);
  void update(int i, int j, int a, int b, bool sz, Slater &w);
 
  //ex1 and ex2 are spin related indices
  void updateWalker(Slater& w, int ex1, int ex2);
  void exciteWalker(Slater& w, int excite1, int excite2, int norbs);

  bool operator<(const HFWalker &w) const { return d < w.d; }
  bool operator==(const HFWalker &w) const { return d == w.d; }

  //calc ovelap wih gradient wrt orbitals in slater
  void OverlapWithGradient(Slater &w, Eigen::VectorXd &grad, double detovlp);
  void OverlapWithGradientGhf(Slater &w, Eigen::VectorXd &grad, double detovlp);
};

#endif
