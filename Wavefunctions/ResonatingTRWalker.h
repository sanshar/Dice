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
#ifndef ResonatingTRWalker_HEADER_H
#define ResonatingTRWalker_HEADER_H

#include "TRWalker.h"

using namespace Eigen;

/**
 Vector of Jastrow-Slater walkers to work with the ResonatingTRWavefunction
*
*/

class ResonatingTRWalker
{

public:
  vector<TRWalker> walkerVec;

  // constructors
  // default
  ResonatingTRWalker(){};
 
  // the following constructors are used by the wave function initWalker function
  // for deterministic
  ResonatingTRWalker(vector<Jastrow> &corr, const vector<Slater> &ref, Determinant &pd) 
  {
    for (int i = 0; i < corr.size(); i++) {
      walkerVec.push_back(TRWalker(corr[i], ref[i], pd));
    }
  };
 

  ResonatingTRWalker(vector<Jastrow> &corr, const vector<Slater> &ref) 
  {
    for (int i = 0; i < corr.size(); i++) {
      walkerVec.push_back(TRWalker(corr[i], ref[i]));
    }
  };
  
  // this is used for storing bestDet
  Determinant getDet() { return walkerVec[0].getDet(); }

  // used during sampling
  void updateWalker(const vector<Slater> &ref, vector<Jastrow> &corr, int ex1, int ex2) {
    for (int i = 0; i < walkerVec.size(); i++) {
      walkerVec[i].updateWalker(ref[i], corr[i], ex1, ex2);
    }
  }
 
  //to be defined for metropolis
  void update(int i, int a, bool sz, const vector<Slater> &ref, const vector<Jastrow> &corr) { return; };
  
  // used for debugging
  friend ostream& operator<<(ostream& os, const ResonatingTRWalker& w) {
    for (int i = 0; i < w.walkerVec.size(); i++) {
      os << "Walker " << i << endl << endl;
      os << w.walkerVec[i] << endl;
    }
    return os;
  }
  
};

#endif
