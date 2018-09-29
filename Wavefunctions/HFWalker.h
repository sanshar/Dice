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
#include "HFWalkerHelper.h"
#include <array>

using namespace Eigen;

class Slater;
class CPSSlater;

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

struct HFWalker
{

  Determinant d; //The current determinant
  HFWalkerHelper helper;  //allinformation about the slaterDeterminant

  HFWalker() {};

  HFWalker(const Slater &w); 
    
  HFWalker(const Slater &w, const Determinant &pd);

  //ex1 and ex2 are spin related indices
  void updateWalker(const Slater& w, int ex1, int ex2);
  void exciteWalker(const Slater& w, int excite1, int excite2, int norbs);


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
 
  bool operator<(const HFWalker &w) const { return d < w.d; }
  bool operator==(const HFWalker &w) const { return d == w.d; }

  //calc ovelap wih gradient wrt orbitals in slater
  void OverlapWithGradient(const Slater &w, Eigen::VectorXd &grad, double detovlp) const;
  void OverlapWithGradientGhf(const Slater &w, Eigen::VectorXd &grad, double detovlp) const;
};

#endif
