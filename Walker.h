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

/**
* Is essentially a single determinant used in the VMC/DMC simulation
* At each step in VMC one need to be able to calculate the following
* quantities
* a. The local energy = <walker|H|Psi>/<walker|Psi>
* b. The gradient     = <walker|H|Psi_t>/<walker/Psi>
* c. The update       = <walker'|Psi>/<walker|Psi>
*
* To do these steps efficiently the walker stores the inverse of the
* determinant matrix, the alphainv and betainv and also the Determinant
* alphaDet and betaDet
*/

class Walker {

 public:
  Determinant     d;                      //The current determinant
  Eigen::MatrixXd alphainv;               //The inverse of the beta determinant
  Eigen::MatrixXd betainv;                //The inverse of the beta determinant
  vector<double>  alphaDet;               //The alpha determinant
  vector<double>  betaDet;                //The beta determinant
  vector<int>     AlphaOpen;              //The set of open alpha orbitals
  vector<int>     AlphaClosed;            //The set of closed alpha orbitals
  vector<int>     BetaOpen;               //The set of open beta orbitals
  vector<int>     BetaClosed;             //The set of closed alpha orbitals
  vector<Eigen::MatrixXd> AlphaTable;     //This is the table used for efficiently
  vector<Eigen::MatrixXd> BetaTable ;     //calculation of local energy, gradient, update

  // The constructor
  Walker(Determinant& pd) : d(pd) {};

  //Use the wavefunction to initialize the alphainv, betainv, alphaDet
  //and betadet
  void initUsingWave(CPSSlater& w, bool check=false) ;

  double getDetOverlap(CPSSlater& w);

  //these are not absolute orbital indices, but instead the
  //ith occupied and ath unoccupied
  void   updateA(int i, int a, CPSSlater& w);
  void   updateB(int i, int a, CPSSlater& w);
  double getDetFactorA(int i, int a, CPSSlater& w, bool doparity=true);
  double getDetFactorB(int i, int a, CPSSlater& w, bool doparity=true);
  double getDetFactorA(int i, int j, int a, int b, CPSSlater& w, bool doparity=true);
  double getDetFactorB(int i, int j, int a, int b, CPSSlater& w, bool doparity=true);
  double getDetFactorAB(int i, int j, int a, int b, CPSSlater& w, bool doparity=true);
  double getDetFactorA(vector<int>& i, vector<int>& a, CPSSlater& w, bool doparity=true);
  double getDetFactorB(vector<int>& i, vector<int>& a, CPSSlater& w, bool doparity=true);

  /**
   * This takes an inverse and determinant of a matrix formed by a subset of
   * columns and rows of Hforbs
   * and generates the new inverse and determinant 
   * by replacing cols with incides des with those with indices des
   * RowVec is the set of row indices that are common to both in the 
   * incoming and outgoing matrices. ColIn ais the column indices
   * of the incoming matrix. 
   */
  void calculateInverseDeterminant(Eigen::MatrixXd &inverseIn, double &detValueIn,
                                   Eigen::MatrixXd &inverseOut, double &detValueOut,
                                   vector<int> &cre, vector<int> &des,
                                   Eigen::Map<Eigen::VectorXi> &RowVec,
                                   vector<int> &ColIn);

  bool   makeMove(CPSSlater& w);
  void   exciteWalker(CPSSlater& w, int excite1, int excite2, int norbs);


  //ThESE ARE DEPRECATED
  bool   makeMovePropPsi(CPSSlater& w);
  bool   makeCleverMove(CPSSlater& w);
  void   genAllMoves(CPSSlater& w, vector<Determinant>& dout,
		     vector<double>& prob, vector<size_t>& alphaExcitation,
		     vector<size_t>& betaExcitation);
  void   genAllMoves2(CPSSlater& w, vector<Determinant>& dout,
		     vector<double>& prob, vector<size_t>& alphaExcitation,
		     vector<size_t>& betaExcitation);

};

#endif
