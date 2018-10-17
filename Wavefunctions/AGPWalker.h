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
#ifndef AGPWalker_HEADER_H
#define AGPWalker_HEADER_H

#include "Determinants.h"
#include <array>


using namespace Eigen;

class AGP;

class AGPWalkerHelper
{

public:
  MatrixXd thetaInv;                    //inverse of the theta matrix
  double thetaDet;                      //determinant of the theta matrix
  array<vector<int>, 2> openOrbs;       //set of open orbitals in the walker
  array<vector<int>, 2> closedOrbs;     //set of closed orbitals in the walker
  array<MatrixXd, 3> rTable;            //table used for efficiently
  
  /**
   * constructor
   */
  AGPWalkerHelper(const AGP &w, const Determinant &d);
  AGPWalkerHelper() {};

  //fills open closed orbs 
  void fillOpenClosedOrbs(const Determinant &d);

  //makes rtable using inverse
  void makeTables(const AGP &w);
  
  //initializes inverse, dets and tables 
  void initInvDetsTables(const AGP &w);
  
  //updates helpers for excitation in the det given by cre and des
  void excitationUpdate(const AGP &w, vector<int>& cre, vector<int>& des, bool sz, double parity, const Determinant& excitedDet);
  
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
* To calculate these efficiently the walker uses the AGPWalkerHelper class
*
**/

class AGPWalker
{

public:
  Determinant d; //The current determinant
  AGPWalkerHelper helper;  

  /**
   * constructors
   */
  //AGPWalker(Determinant &pd);
  AGPWalker() {};

  AGPWalker(const AGP &w); 
    
  AGPWalker(const AGP &w, const Determinant &pd);

  /**
   * reads dets from 'BesetDeterminant.txt'
   */
  void readBestDeterminant(Determinant& d) const;

  /**
   * makes det based on pairMat
   */
  void guessBestDeterminant(Determinant& d, const Eigen::MatrixXd& pairMat) const; 

  void initDet(const MatrixXd& pairMat); 
  
  const Determinant &getDet() const { return d; }
  
  //get overlap with the reference
  double getDetOverlap(const AGP &w) const;
  
  //returns < m | Psi0 >/< d | Psi0 >, where m is obtained by exciting the walker with 
  // spin orbital excitations i->a, j->b
  double getDetFactor(int i, int a, const AGP &w) const; 
  double getDetFactor(int I, int J, int A, int B, const AGP &w) const; 
  
  //i, j, a, b are spatial orbitals
  double getDetFactor(int i, int a, bool sz, const AGP &w) const;
  double getDetFactor(int i, int j, int a, int b, bool sz1, bool sz2, const AGP &w) const;

  //updates det and helpers afterspatial orb excitations i->a, j->b with spin sz
  void update(int i, int a, bool sz, const AGP &w);
  void update(int i, int j, int a, int b, bool sz, const AGP &w);
 
  //ex1 and ex2 are spin related indices
  void updateWalker(const AGP& w, int ex1, int ex2);
  void exciteWalker(const AGP& w, int excite1, int excite2, int norbs);

  bool operator<(const AGPWalker &w) const { return d < w.d; }
  bool operator==(const AGPWalker &w) const { return d == w.d; }
  friend ostream& operator<<(ostream& os, const AGPWalker& walk);

  //calc ovelap wih gradient wrt orbitals in slater
  void OverlapWithGradient(const AGP &w, Eigen::VectorBlock<VectorXd> &grad, double detovlp) const;
};

#endif
