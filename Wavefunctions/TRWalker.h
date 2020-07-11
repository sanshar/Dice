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
#ifndef TRWalker_HEADER_H
#define TRWalker_HEADER_H

#include "Walker.h"

using namespace Eigen;

/**
 Vector of Jastrow-Slater walkers to work with the TRWavefunction
*
*/

class TRWalker
{

public:
  array<Walker<Jastrow, Slater>, 2> walkerPair;
  Determinant d;
  array<double, 2> overlaps;
  double totalOverlap;

  // constructors
  // default
  TRWalker(){};
 
  // the following constructors are used by the wave function initWalker function
  // for deterministic
  TRWalker(Jastrow &corr, const Slater &ref, Determinant &pd) 
  {
    d = pd;
    walkerPair[0] = Walker<Jastrow, Slater>(corr, ref, pd);
    Determinant dcopy = pd;
    dcopy.flipAlphaBeta();
    walkerPair[1] = Walker<Jastrow, Slater>(corr, ref, dcopy);
    overlaps[0] = corr.Overlap(walkerPair[0].d) * walkerPair[0].getDetOverlap(ref); 
    overlaps[1] = corr.Overlap(walkerPair[1].d) * walkerPair[1].getDetOverlap(ref);
    totalOverlap = overlaps[0] + overlaps[1];
  };
 

  TRWalker(Jastrow &corr, const Slater &ref) 
  {
    walkerPair[0] = Walker<Jastrow, Slater>(corr, ref);
    Determinant dcopy = walkerPair[0].d;
    d = walkerPair[0].d;
    dcopy.flipAlphaBeta();
    walkerPair[1] = Walker<Jastrow, Slater>(corr, ref, dcopy);
    overlaps[0] = corr.Overlap(walkerPair[0].d) * walkerPair[0].getDetOverlap(ref); 
    overlaps[1] = corr.Overlap(walkerPair[1].d) * walkerPair[1].getDetOverlap(ref);
    totalOverlap = overlaps[0] + overlaps[1];
  };
  
  // this is used for storing bestDet
  Determinant getDet() { return walkerPair[0].getDet(); }

  // used during sampling
  void updateWalker(const Slater &ref, Jastrow &corr, int ex1, int ex2, bool doparity = true) {
    walkerPair[0].updateWalker(ref, corr, ex1, ex2, doparity);
    d = walkerPair[0].d;
    int norbs = Determinant::norbs;
    // for the flipped determinant, flip the excitations
    if (ex1%2 == 0) ex1 += 2*norbs + 1;
    else ex1 -= (2*norbs + 1);
    if (ex2 != 0) {
      if (ex2%2 == 0) ex2 += 2*norbs + 1;
      else ex2 -= (2*norbs + 1);
    }
    walkerPair[1].updateWalker(ref, corr, ex1, ex2, doparity);
    overlaps[0] = corr.Overlap(walkerPair[0].d) * walkerPair[0].getDetOverlap(ref); 
    overlaps[1] = corr.Overlap(walkerPair[1].d) * walkerPair[1].getDetOverlap(ref);
    totalOverlap = overlaps[0] + overlaps[1];
  }
 
  //to be defined for metropolis
  void update(int i, int a, bool sz, const Slater &ref, const Jastrow &corr) { return; };
  
  // used for debugging
  friend ostream& operator<<(ostream& os, const TRWalker& w) {
    os << "Walker 0" << endl << endl;
    os << w.walkerPair[0] << endl;
    os << "Walker 1" << endl << endl;
    os << w.walkerPair[1] << endl;
    return os;
  }
  
};

#endif
