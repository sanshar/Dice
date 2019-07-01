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
#ifndef SimpleWalker_HEADER_H
#define SimpleWalker_HEADER_H

#include "Determinants.h"
#include <boost/serialization/serialization.hpp>
#include <Eigen/Dense>
#include "input.h"
#include <unordered_set>
#include <iterator>
#include "integral.h"

using namespace Eigen;

/**
* Is essentially a single determinant used in the VMC/DMC simulation
* At each step in VMC one need to be able to calculate the following
* quantities
* a. The local energy = <walker|H|Psi>/<walker|Psi>
*
*/

class SimpleWalker
{

public:
  Determinant d;                      //The current determinant
  unordered_set<int> excitedOrbs;     //spin orbital indices of excited electrons (in virtual orbitals) in d 
  std::array<VectorXd, 2> energyIntermediates; 

  // The constructor
  SimpleWalker(Determinant &pd) : d(pd) {};
  SimpleWalker(const Determinant &corr, const Determinant &ref, Determinant &pd) : d(pd) {};
  SimpleWalker(const SimpleWalker &w): d(w.d), excitedOrbs(w.excitedOrbs) {};
  SimpleWalker(){};

  Determinant getDet() { return d; }

  void updateA(int i, int a);

  void updateA(int i, int j, int a, int b);

  void updateB(int i, int a);

  void updateB(int i, int j, int a, int b);

  void update(int i, int a, bool sz, const Determinant &ref, const Determinant &corr) { return; };//to be defined for metropolis
  
  void updateEnergyIntermediate(const oneInt& I1, const twoInt& I2, int I, int A);
  
  void updateEnergyIntermediate(const oneInt& I1, const twoInt& I2, int I, int A, int J, int B);

  void updateWalker(const Determinant &ref, const Determinant &corr, int ex1, int ex2, bool updateIntermediate = true);
  
  void exciteWalker(const Determinant &ref, const Determinant &corr, int excite1, int excite2, int norbs);
  
  bool operator<(const SimpleWalker &w) const
  {
    return d < w.d;
  }

  bool operator==(const SimpleWalker &w) const
  {
    return d == w.d;
  }

  friend ostream& operator<<(ostream& os, const SimpleWalker& w) {
    os << w.d << endl;
    //os << "excited Orbs   " << w.excitedOrbs << endl;
    os << "excitedOrbs   ";
    copy(w.excitedOrbs.begin(), w.excitedOrbs.end(), ostream_iterator<int>(os, " "));
    os << "energyIntermediates\n";
    os << "up\n";
    os << w.energyIntermediates[0] << endl;
    os << "down\n";
    os << w.energyIntermediates[1] << endl;
    return os;
  }
  //template <typename Wfn>
  //void exciteTo(Wfn& w, Determinant& dcopy) {
  //  d = dcopy;
  //}
  
};

#endif
