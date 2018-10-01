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

enum HartreeFock;

using namespace Eigen;

class Slater;

//used to efficiently get overlaps when using a slater determinant reference
//in all arrays, first object referes to alpha spins and the second to beta
//in case of ghf, second object of thetaInv is empty, that of thetaDet is 1,
//that of rTable is the same as first
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

  HFWalker(const Slater &w); 
    
  HFWalker(const Slater &w, const Determinant &pd);

  /**
   * reads dets from 'BesetDeterminant.txt'
   */
  void readBestDeterminant(Determinant& d) const;

  /**
   * makes det based on mo coeffs 
   */
  void guessBestDeterminant(Determinant& d, const Eigen::MatrixXd& HforbsA, const Eigen::MatrixXd& HforbsB) const; 

  void initDet(const MatrixXd& HforbsA, const MatrixXd& HforbsB); 
  
  const Determinant &getDet() const { return d; }
  
  //overlap with i th det in the wavefunction
  double getIndividualDetOverlap(int i) const { return helper.thetaDet[i][0] * helper.thetaDet[i][1]; }
  
  //get overlap with the reference
  double getDetOverlap(const Slater &w) const;
  
  //returns < m | Psi0 >/< d | Psi0 >, where m is obtained by exciting the walker with 
  // spin orbital excitations i->a, j->b
  double getDetFactor(int i, int a, const Slater &w) const; 
  double getDetFactor(int I, int J, int A, int B, const Slater &w) const; 
  
  //i, j, a, b are spatial orbitals
  double getDetFactor(int i, int a, bool sz, const Slater &w) const;
  double getDetFactor(int i, int j, int a, int b, bool sz1, bool sz2, const Slater &w) const;

  //updates det and helpers afterspatial orb excitations i->a, j->b with spin sz
  void update(int i, int a, bool sz, const Slater &w);
  void update(int i, int j, int a, int b, bool sz, const Slater &w);
 
  //ex1 and ex2 are spin related indices
  void updateWalker(const Slater& w, int ex1, int ex2);
  void exciteWalker(const Slater& w, int excite1, int excite2, int norbs);

  bool operator<(const HFWalker &w) const { return d < w.d; }
  bool operator==(const HFWalker &w) const { return d == w.d; }

  //calc ovelap wih gradient wrt orbitals in slater
  void OverlapWithGradient(const Slater &w, Eigen::VectorXd &grad, double detovlp) const;
  void OverlapWithGradientGhf(const Slater &w, Eigen::VectorXd &grad, double detovlp) const;
};

#endif
