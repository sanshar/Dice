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
#ifndef PermutedTRWalker_HEADER_H
#define PermutedTRWalker_HEADER_H

#include "TRWalker.h"

using namespace Eigen;

/**
 Vector of Jastrow-Slater walkers to work with the PermutedWavefunction
*
*/

class PermutedTRWalker
{

public:
  vector<TRWalker> walkerVec;
  // stored in both the wave function and here
  MatrixXd permutations;

  // constructors
  // default
  PermutedTRWalker(){};
 
  // the following constructors are used by the wave function initWalker function
  // for deterministic
  PermutedTRWalker(Jastrow &corr, const Slater &ref, Determinant &pd, MatrixXd &pPermutations) 
  {
    walkerVec.push_back(TRWalker(corr, ref, pd));
    vector<int> occA, occB;
    pd.getClosedAlphaBeta(occA, occB);
    permutations = pPermutations;
    for (int i = 0; i < permutations.rows(); i++) {
      Determinant dcopy;
      for (int j = 0; j < occA.size(); j++) {
        dcopy.setoccA(permutations(i, occA[j]), true);
      }
      for (int j = 0; j < occB.size(); j++) {
        dcopy.setoccB(permutations(i, occB[j]), true);
      }
      walkerVec.push_back(TRWalker(corr, ref, dcopy));
    }
  };
 
  PermutedTRWalker(Jastrow &corr, const Slater &ref, MatrixXd &pPermutations) 
  {
    walkerVec.push_back(TRWalker(corr, ref));
    vector<int> occA, occB;
    walkerVec[0].d.getClosedAlphaBeta(occA, occB);
    permutations = pPermutations;
    for (int i = 0; i < permutations.rows(); i++) {
      Determinant dcopy;
      for (int j = 0; j < occA.size(); j++) {
        dcopy.setoccA(permutations(i, occA[j]), true);
      }
      for (int j = 0; j < occB.size(); j++) {
        dcopy.setoccB(permutations(i, occB[j]), true);
      }
      walkerVec.push_back(TRWalker(corr, ref, dcopy));
    }
  };
  
  // this is used for storing bestDet
  Determinant getDet() { return walkerVec[0].getDet(); }

  // used during sampling
  void updateWalker(const Slater &ref, Jastrow &corr, int ex1, int ex2) {
    walkerVec[0].updateWalker(ref, corr, ex1, ex2);
    int norbs = Determinant::norbs;
    // calculate corresponding excitaitons for the permuted determinants
    int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I; 
    int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
    int i = I / 2, a = A / 2;
    int j = J / 2, b = B / 2;
    int sz1 = I%2, sz2 = J%2;
    for (int n = 0; n < permutations.rows(); n++) {
      int ex1p = 0, ex2p = 0;
      int ip = permutations(n, i);
      int ap = permutations(n, a);
      ex1p = (2 * ip + sz1) * 2 * norbs + (2 * ap + sz1);
      if (ex2 != 0) { 
        int jp = permutations(n, j);
        int bp = permutations(n, b);
        ex2p = (2 * jp + sz2) * 2 * norbs + (2 * bp + sz2);
      }
      walkerVec[n+1].updateWalker(ref, corr, ex1p, ex2p);
    }
  }
 
  //to be defined for metropolis
  void update(int i, int a, bool sz, const Slater &ref, const Jastrow &corr) { return; };
  
  // used for debugging
  friend ostream& operator<<(ostream& os, const PermutedTRWalker& w) {
    for (int i = 0; i < w.walkerVec.size(); i++) {
      os << "Walker " << i << endl << endl;
      os << w.walkerVec[i] << endl;
    }
    return os;
  }
  
};

#endif
