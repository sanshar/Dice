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
#ifndef JastrowMultiSlaterWalker_HEADER_H
#define JastrowMultiSlaterWalker_HEADER_H

#include "WalkerHelper.h"

using namespace Eigen;

/**
 Vector of Jastrow-Slater walkers to work with the ResonatingWavefunction
*
*/

class JastrowMultiSlaterWalker
{

public:
  Determinant d;
  Walker<Jastrow, MultiSlater> walker;
  MatrixXcd intermediate, s;
  double updateTime;

  // constructors
  // default
  JastrowMultiSlaterWalker(){};
 
  // the following constructors are used by the wave function initWalker function
  // for deterministic
  JastrowMultiSlaterWalker(Jastrow &corr, const MultiSlater &ref, Determinant &pd) 
  {
    d = pd;
    updateTime = 0.;
    walker = Walker<Jastrow, MultiSlater> (corr, ref, pd);
  };
 

  JastrowMultiSlaterWalker(Jastrow &corr, const MultiSlater &ref) 
  {
    walker = Walker<Jastrow, MultiSlater> (corr, ref);
    d = walker.d;
    updateTime = 0.;
  };
  
  // this is used for storing bestDet
  Determinant getDet() { return d; }
  
  double getIndividualDetOverlap(int i) const
  {
    return walker.getIndividualDetOverlap(i);
  }

  double getDetOverlap(const MultiSlater &ref) const
  {
    return walker.getDetOverlap(ref);
  }

  // these det ratio functions return < m | phi_0 > / < n | phi_0 > with complex projection
  double getDetFactor(int i, int a, const MultiSlater &ref) const 
  {
    if (i % 2 == 0)
      return getDetFactor(i / 2, a / 2, 0, ref);
    else
      return getDetFactor(i / 2, a / 2, 1, ref);
  }

  double getDetFactor(int I, int J, int A, int B, const MultiSlater &ref) const 
  {
    if (I % 2 == J % 2 && I % 2 == 0)
      return getDetFactor(I / 2, J / 2, A / 2, B / 2, 0, 0, ref);
    else if (I % 2 == J % 2 && I % 2 == 1)
      return getDetFactor(I / 2, J / 2, A / 2, B / 2, 1, 1, ref);
    else if (I % 2 != J % 2 && I % 2 == 0)
      return getDetFactor(I / 2, J / 2, A / 2, B / 2, 0, 1, ref);
    else
      return getDetFactor(I / 2, J / 2, A / 2, B / 2, 1, 0, ref);
  }
  
  double getDetFactor(int i, int a, bool sz, const MultiSlater &ref) const
  {
    int tableIndexi, tableIndexa;
    walker.refHelper.getRelIndices(i, tableIndexi, a, tableIndexa, sz);
    return (walker.refHelper.rt(tableIndexa, tableIndexi) * walker.refHelper.refOverlap).real() / walker.refHelper.refOverlap.real();
  }
  
  double getDetFactor(int i, int j, int a, int b, bool sz1, bool sz2, const MultiSlater &ref) const
  {
    int tableIndexi, tableIndexa, tableIndexj, tableIndexb;
    walker.refHelper.getRelIndices(i, tableIndexi, a, tableIndexa, sz1);
    walker.refHelper.getRelIndices(j, tableIndexj, b, tableIndexb, sz2);
    complex<double> sliceDet = walker.refHelper.rt(tableIndexa, tableIndexi) * walker.refHelper.rt(tableIndexb, tableIndexj)
                             - walker.refHelper.rt(tableIndexa, tableIndexj) * walker.refHelper.rt(tableIndexb, tableIndexi);
    return (sliceDet * walker.refHelper.refOverlap).real() / walker.refHelper.refOverlap.real();
  }

  // used during sampling
  void updateWalker(const MultiSlater &ref, Jastrow &corr, int ex1, int ex2) {
    double init = getTime();
    walker.updateWalker(ref, corr, ex1, ex2);
    updateTime += (getTime() - init);
    d = walker.d;
  }
  
  void OverlapWithGradient(const MultiSlater &ref, Eigen::VectorBlock<VectorXd> &grad) const
  {
    if (schd.optimizeCiCoeffs) {
      for (int i = 0; i < ref.numDets; i++) grad[i] += walker.refHelper.ciOverlaps[i] / (walker.refHelper.refOverlap.real());
    }
    // orb gradient to be implemented
  }
 
  //to be defined for metropolis
  void update(int i, int a, bool sz, const MultiSlater &ref, const Jastrow &corr) { return; };
  
  friend ostream& operator<<(ostream& os, const JastrowMultiSlaterWalker& w) {
    os << w.d << endl << endl;
    os << "t\n" << w.walker.refHelper.t << endl << endl;
    os << "rt\n" << w.walker.refHelper.rt << endl << endl;
    os << "tc\n" << w.walker.refHelper.tc << endl << endl;
    os << "rtc_b\n" << w.walker.refHelper.rtc_b << endl << endl;
    os << "totalOverlap\n" << w.walker.refHelper.totalOverlap << endl << endl;
    return os;
  }
  
  
};

#endif
