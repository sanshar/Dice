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
#ifndef NNBSWalker_HEADER_H
#define NNBSWalker_HEADER_H

#include "input.h"
#include "Determinants.h"
#include "global.h"
#include "fnn.h"


using namespace Eigen;

/**
 Vector of Jastrow-Slater walkers to work with the PermutedWavefunction
*
*/

class NNBSWalker
{

public:
  Determinant det;
  VectorXd occ;         // same as det but a vector of +1 and -1 instead of bits
  MatrixXd occSlice;    // occupied slice of coefficient matrix, for wave function evaluation
  MatrixXd occSliceInv; // for wave function derivatives  

  // constructors
  // default
  NNBSWalker(){};
 
  // the following constructors are used by the wave function initWalker function
  // for deterministic
  NNBSWalker(MatrixXd& moCoeffs, Determinant &pd) 
  {
    det = pd;
    int norbs = Determinant::norbs;
    int nelec = Determinant::nalpha + Determinant::nbeta;
    occ = VectorXd::Zero(2*norbs);
    for (int i = 0; i < norbs; i++) {
      occ[i] = pd.getoccA(i) ?  1. : -1.;
      occ[norbs + i] = pd.getoccB(i) ? 1. : -1.;
    }
    vector<int> openOrbs, closedOrbs;
    vector<int> closedBeta, openBeta;
    det.getOpenClosedAlphaBeta(openOrbs, closedOrbs, openBeta, closedBeta);
    for (int& c_i : closedBeta) c_i += Determinant::norbs;
    closedOrbs.insert(closedOrbs.end(), closedBeta.begin(), closedBeta.end());
    Eigen::Map<VectorXi> occRows(&closedOrbs[0], closedOrbs.size());
    VectorXi occColumns = VectorXi::LinSpaced(nelec, 0, nelec - 1);
    occSlice = moCoeffs(occRows, occColumns);
  };
 
  NNBSWalker(MatrixXd& moCoeffs) 
  {
    int norbs = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta = Determinant::nbeta;
    int nelec = nalpha + nbeta;
    char file[5000];
    sprintf(file, "BestDeterminant.txt");
    ifstream ofile(file);
    if (ofile) {// read bestdet from prev run
      if (commrank == 0) {
        std::ifstream ifs(file, std::ios::binary);
        boost::archive::binary_iarchive load(ifs);
        load >> det;
      }
#ifndef SERIAL
    MPI_Bcast(&det.reprA, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&det.reprB, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    }
    else {// guess or read given

      det = Determinant();
      if (boost::iequals(schd.determinantFile, "")) {
        // choose alpha occupations randomly
        std::vector<int> bitmask(nalpha, 1);
        bitmask.resize(norbs, 0); // N-K trailing 0's
        vector<int> comb;
        random_shuffle(bitmask.begin(), bitmask.end());
        for (int i = 0; i < norbs; i++) {
          if (bitmask[i] == 1) det.setoccA(i, true);
        }

        // fill beta, trying to avoid occupied alphas as much as possible
        int nbetaFilled = 0;
        // first pass, only fill empty orbs
        for (int i = 0; i < norbs; i++) {
          if (nbetaFilled == nbeta) break;
          if (bitmask[i] == 0) { // empty
            det.setoccB(i, true);
            nbetaFilled++;
          }
        }
        
        // if betas leftover, fill sequentially
        if (nbetaFilled < nbeta) {
          for (int i = 0; i < norbs; i++) {
            if (nbetaFilled == nbeta) break;
            if (bitmask[i] == 1) {// alpha occupied
              det.setoccB(i, true);
              nbetaFilled++;
            }
          }
        }
      }
      else if (boost::iequals(schd.determinantFile, "bestDet")) {
        std::vector<Determinant> dets;
        std::vector<double> ci;
        readDeterminants(schd.determinantFile, dets, ci);
        det = dets[0];
      }
    }
    occ = VectorXd::Zero(2*norbs);
    for (int i = 0; i < norbs; i++) {
      occ[i] = det.getoccA(i) ?  1. : -1.;
      occ[norbs + i] = det.getoccB(i) ? 1. : -1.;
    }
    
    vector<int> openOrbs, closedOrbs;
    vector<int> closedBeta, openBeta;
    det.getOpenClosedAlphaBeta(openOrbs, closedOrbs, openBeta, closedBeta);
    for (int& c_i : closedBeta) c_i += Determinant::norbs;
    closedOrbs.insert(closedOrbs.end(), closedBeta.begin(), closedBeta.end());
    Eigen::Map<VectorXi> occRows(&closedOrbs[0], closedOrbs.size());
    VectorXi occColumns = VectorXi::LinSpaced(nelec, 0, nelec - 1);
    occSlice = moCoeffs(occRows, occColumns);
  };
  
  // this is used for storing bestDet
  Determinant getDet() { return det; }

  // used during sampling
  void updateWalker(const MatrixXd &moCoeffs, const fnn &fnnb, int ex1, int ex2, bool doparity = true) 
  {
    int norbs = Determinant::norbs;
    int nelec = Determinant::nalpha + Determinant::nbeta;
    int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I; 
    int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
    int i = I / 2, a = A / 2;
    int j = J / 2, b = B / 2;
    int sz1 = I%2, sz2 = J%2;
    det.setocc(i, sz1, false);
    det.setocc(a, sz1, true);
    occ[sz1 * norbs + i] = -1.;
    occ[sz1 * norbs + a] = 1.;
    if (ex2 != 0) {
      det.setocc(j, sz2, false);
      det.setocc(b, sz2, true);
      occ[sz2 * norbs + j] = -1.;
      occ[sz2 * norbs + b] = 1.;
    }
    
    vector<int> openOrbs, closedOrbs;
    vector<int> closedBeta, openBeta;
    det.getOpenClosedAlphaBeta(openOrbs, closedOrbs, openBeta, closedBeta);
    for (int& c_i : closedBeta) c_i += Determinant::norbs;
    closedOrbs.insert(closedOrbs.end(), closedBeta.begin(), closedBeta.end());
    Eigen::Map<VectorXi> occRows(&closedOrbs[0], closedOrbs.size());
    VectorXi occColumns = VectorXi::LinSpaced(nelec, 0, nelec - 1);
    occSlice = moCoeffs(occRows, occColumns);
  }
 
  //to be defined for metropolis
  void update(int i, int a, bool sz, const MatrixXd &ref, const fnn &corr) { return; };
  
  // used for debugging
  friend ostream& operator<<(ostream& os, const NNBSWalker& w) {
    os << "det  " << w.det << endl;
    os << "occ\n" << w.occ << endl << endl;
    os << "occSlice\n" << w.occSlice << endl << endl;
    return os;
  }
  
};

#endif
