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

  complex<double> getDetFactorComplex(int i, int a, const MultiSlater &ref) const 
  {
    if (i % 2 == 0)
      return getDetFactorComplex(i / 2, a / 2, 0, ref);
    else
      return getDetFactorComplex(i / 2, a / 2, 1, ref);
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
  
  complex<double> getDetFactorComplex(int i, int a, bool sz, const MultiSlater &ref) const
  {
    int tableIndexi, tableIndexa;
    walker.refHelper.getRelIndices(i, tableIndexi, a, tableIndexa, sz);
    return walker.refHelper.rt(tableIndexa, tableIndexi);
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
    
    // from Filippi's paper, 10.1021/acs.jctc.7b00648, section 3.2
    if (schd.optimizeOrbs) {
      int norbs = Determinant::norbs, nclosed = walker.refHelper.closedOrbs.size(), nvirt = 2*norbs - nclosed;
      // building the y matrix
      MatrixXcd yMat = MatrixXcd::Zero(2*norbs, nclosed);
      int count4 = 0;
      for (int i = 1; i < ref.numDets; i++) {
        int rank = ref.ciExcitations[i][0].size();
        if (rank == 1) { 
          yMat(ref.ciExcitations[i][1][0], ref.ciExcitations[i][0][0]) += ref.ciCoeffs[i] * ref.ciParity[i];
        }
        else if (rank == 2) {
          yMat(ref.ciExcitations[i][1][0], ref.ciExcitations[i][0][0]) += ref.ciCoeffs[i] * ref.ciParity[i] * walker.refHelper.tc(ref.ciExcitations[i][0][1], ref.ciExcitations[i][1][1]);
          yMat(ref.ciExcitations[i][1][1], ref.ciExcitations[i][0][1]) += ref.ciCoeffs[i] * ref.ciParity[i] * walker.refHelper.tc(ref.ciExcitations[i][0][0], ref.ciExcitations[i][1][0]);
          yMat(ref.ciExcitations[i][1][0], ref.ciExcitations[i][0][1]) -= ref.ciCoeffs[i] * ref.ciParity[i] * walker.refHelper.tc(ref.ciExcitations[i][0][1], ref.ciExcitations[i][1][0]);
          yMat(ref.ciExcitations[i][1][1], ref.ciExcitations[i][0][0]) -= ref.ciCoeffs[i] * ref.ciParity[i] * walker.refHelper.tc(ref.ciExcitations[i][0][0], ref.ciExcitations[i][1][1]);
        }
        else if (rank == 3) {
          Matrix3cd tcSlice;
          for  (int mu = 0; mu < 3; mu++) 
            for (int nu = 0; nu < 3; nu++) 
              tcSlice(mu, nu) = walker.refHelper.tc(ref.ciExcitations[i][0][mu], ref.ciExcitations[i][1][nu]);
          auto inv = tcSlice.inverse();
          for  (int mu = 0; mu < 3; mu++) 
            for (int nu = 0; nu < 3; nu++) 
              yMat(ref.ciExcitations[i][1][mu], ref.ciExcitations[i][0][nu]) += ref.ciCoeffs[i] * walker.refHelper.ciOverlapRatios[i] * inv(mu, nu);
        }
        else if (rank == 4) {
          auto inv = walker.refHelper.tcSlice[count4].inverse();
          for  (int mu = 0; mu < 4; mu++) 
            for (int nu = 0; nu < 4; nu++) 
              yMat(ref.ciExcitations[i][1][mu], ref.ciExcitations[i][0][nu]) += ref.ciCoeffs[i] * walker.refHelper.ciOverlapRatios[i] * inv(mu, nu);
          count4++;
        }
        else {
          MatrixXcd tcSlice = MatrixXcd::Zero(rank, rank);
          for  (int mu = 0; mu < rank; mu++) 
            for (int nu = 0; nu < rank; nu++) 
              tcSlice(mu, nu) = walker.refHelper.tc(ref.ciExcitations[i][0][mu], ref.ciExcitations[i][1][nu]);
          auto inv = tcSlice.inverse();
          for  (int mu = 0; mu < rank; mu++) 
            for (int nu = 0; nu < rank; nu++) 
              yMat(ref.ciExcitations[i][1][mu], ref.ciExcitations[i][0][nu]) += ref.ciCoeffs[i] * walker.refHelper.ciOverlapRatios[i] * inv(mu, nu);
        }
      }
      yMat *= walker.refHelper.refOverlap / walker.refHelper.totalComplexOverlap;
      
      MatrixXcd yt = yMat * walker.refHelper.t * walker.refHelper.totalComplexOverlap;
      MatrixXcd t_tcyt = (walker.refHelper.t * walker.refHelper.totalComplexOverlap - walker.refHelper.tc * yt);
      
      // iterating over orbitals
      for (int i = 0; i < nclosed; i++) {
        for (int j = 0; j < nclosed; j++) {
          grad[ref.numDets + 4 * walker.refHelper.closedOrbs[i] * norbs + 2 * ref.ref[j]] = t_tcyt(j, i).real() / (walker.refHelper.refOverlap.real());
          grad[ref.numDets + 4 * walker.refHelper.closedOrbs[i] * norbs + 2 * ref.ref[j] + 1] = -t_tcyt(j, i).imag() / (walker.refHelper.refOverlap.real());
        }
      }
      
      
      std::vector<int> all(2*norbs);
      std::iota(all.begin(), all.end(), 0);
      std::vector<int> virt(nvirt);
      std::set_difference(all.begin(), all.end(), ref.ref.begin(), ref.ref.end(), virt.begin());
      
      
      for (int i = 0; i < nclosed; i++) {
        for (int j = 0; j < nvirt; j++) {
          grad[ref.numDets + 4 * walker.refHelper.closedOrbs[i] * norbs + 2 * virt[j]] = yt(virt[j], i).real() / (walker.refHelper.refOverlap.real());
          grad[ref.numDets + 4 * walker.refHelper.closedOrbs[i] * norbs + 2 * virt[j] + 1] = -yt(virt[j], i).imag() / (walker.refHelper.refOverlap.real());
        }
      }
    }
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
