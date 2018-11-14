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
#ifndef HFWalker_HEADER_H
#define HFWalker_HEADER_H

#include "Determinants.h"
#include "WalkerHelper.h"
#include <array>
#include "igl/slice.h"
#include "igl/slice_into.h"
#include "Slater.h"
#include "AGP.h"
#include "Pfaffian.h"

using namespace Eigen;

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

template<typename Corr, typename Reference>
struct Walker { };

template<typename Corr>
struct Walker<Corr, Slater> {

  Determinant d;
  WalkerHelper<Corr> corrHelper;
  WalkerHelper<Slater> refHelper;

  Walker() {};
  
  Walker(const Corr &corr, const Slater &ref) 
  {
    initDet(ref.getHforbsA(), ref.getHforbsB());
    refHelper = WalkerHelper<Slater>(ref, d);
    corrHelper = WalkerHelper<Corr>(corr, d);
  }

  Walker(const Corr &corr, const Slater &ref, const Determinant &pd) : d(pd), refHelper(ref, pd), corrHelper(corr, pd) {}; 

  Determinant& getDet() {return d;}
  void readBestDeterminant(Determinant& d) const 
  {
    if (commrank == 0) {
      char file[5000];
      sprintf(file, "BestDeterminant.txt");
      std::ifstream ifs(file, std::ios::binary);
      boost::archive::binary_iarchive load(ifs);
      load >> d;
    }
#ifndef SERIAL
    MPI_Bcast(&d.reprA, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d.reprB, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  }

  /**
   * makes det based on mo coeffs 
   */
  void guessBestDeterminant(Determinant& d, const Eigen::MatrixXd& HforbsA, const Eigen::MatrixXd& HforbsB) const 
  {
    int norbs = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta = Determinant::nbeta;

    d = Determinant();
    for (int i = 0; i < nalpha; i++) {
      int bestorb = 0;
      double maxovlp = 0;
      for (int j = 0; j < norbs; j++) {
        if (abs(HforbsA(i, j)) > maxovlp && !d.getoccA(j)) {
          maxovlp = abs(HforbsA(i, j));
          bestorb = j;
        }
      }
      d.setoccA(bestorb, true);
    }
    for (int i = 0; i < nbeta; i++) {
      int bestorb = 0;
      double maxovlp = 0;
      for (int j = 0; j < norbs; j++) {
        if (schd.hf == "rhf" || schd.hf == "uhf") {
          if (abs(HforbsB(i, j)) > maxovlp && !d.getoccB(j)) {
            bestorb = j;
            maxovlp = abs(HforbsB(i, j));
          }
        }
        else {
          if (abs(HforbsB(i+norbs, j)) > maxovlp && !d.getoccB(j)) {
            bestorb = j;
            maxovlp = abs(HforbsB(i+norbs, j));
          }
        }
      }
      d.setoccB(bestorb, true);
    }
  }

  void initDet(const MatrixXd& HforbsA, const MatrixXd& HforbsB) 
  {
    bool readDeterminant = false;
    char file[5000];
    sprintf(file, "BestDeterminant.txt");

    {
      ifstream ofile(file);
      if (ofile)
        readDeterminant = true;
    }
    if (readDeterminant)
      readBestDeterminant(d);
    else
      guessBestDeterminant(d, HforbsA, HforbsB);
  }

  double getIndividualDetOverlap(int i) const
  {
    return refHelper.thetaDet[i][0] * refHelper.thetaDet[i][1];
  }

  double getDetOverlap(const Slater &ref) const
  {
    double ovlp = 0.0;
    for (int i = 0; i < refHelper.thetaDet.size(); i++) {
      ovlp += ref.getciExpansion()[i] * refHelper.thetaDet[i][0] * refHelper.thetaDet[i][1];
    }
    return ovlp;
  }

  double getDetFactor(int i, int a, const Slater &ref) const 
  {
    if (i % 2 == 0)
      return getDetFactor(i / 2, a / 2, 0, ref);
    else                                   
      return getDetFactor(i / 2, a / 2, 1, ref);
  }

  double getDetFactor(int I, int J, int A, int B, const Slater &ref) const 
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

  double getDetFactor(int i, int a, bool sz, const Slater &ref) const
  {
    int tableIndexi, tableIndexa;
    refHelper.getRelIndices(i, tableIndexi, a, tableIndexa, sz); 

    double detFactorNum = 0.0;
    double detFactorDen = 0.0;
    for (int j = 0; j < ref.getDeterminants().size(); j++)
    {
      double factor = refHelper.rTable[j][sz](tableIndexa, tableIndexi);
      detFactorNum += ref.getciExpansion()[j] * factor * refHelper.thetaDet[j][0] * refHelper.thetaDet[j][1];
      detFactorDen += ref.getciExpansion()[j] * refHelper.thetaDet[j][0] * refHelper.thetaDet[j][1];
    }
    return detFactorNum / detFactorDen;
  }

  double getDetFactor(int i, int j, int a, int b, bool sz1, bool sz2, const Slater &ref) const
  {
    int tableIndexi, tableIndexa, tableIndexj, tableIndexb;
    refHelper.getRelIndices(i, tableIndexi, a, tableIndexa, sz1); 
    refHelper.getRelIndices(j, tableIndexj, b, tableIndexb, sz2) ;

    double detFactorNum = 0.0;
    double detFactorDen = 0.0;
    for (int j = 0; j < ref.getDeterminants().size(); j++)
    {
      double factor;
      if (sz1 == sz2 || refHelper.hftype == 2)
        factor = refHelper.rTable[j][sz1](tableIndexa, tableIndexi) * refHelper.rTable[j][sz1](tableIndexb, tableIndexj) 
            - refHelper.rTable[j][sz1](tableIndexb, tableIndexi) *refHelper.rTable[j][sz1](tableIndexa, tableIndexj);
      else
        factor = refHelper.rTable[j][sz1](tableIndexa, tableIndexi) * refHelper.rTable[j][sz2](tableIndexb, tableIndexj);
      detFactorNum += ref.getciExpansion()[j] * factor * refHelper.thetaDet[j][0] * refHelper.thetaDet[j][1];
      detFactorDen += ref.getciExpansion()[j] * refHelper.thetaDet[j][0] * refHelper.thetaDet[j][1];
    }
    return detFactorNum / detFactorDen;
  }

  void update(int i, int a, bool sz, const Slater &ref, const Corr &corr, bool doparity = true)
  {
    double p = 1.0;
    if (doparity) p *= d.parity(a, i, sz);
    d.setocc(i, sz, false);
    d.setocc(a, sz, true);
    if (refHelper.hftype == Generalized) {
      int norbs = Determinant::norbs;
      vector<int> cre{ a + sz * norbs }, des{ i + sz * norbs };
      refHelper.excitationUpdateGhf(ref, cre, des, sz, p, d);
    }
    else
    {
      vector<int> cre{ a }, des{ i };
      refHelper.excitationUpdate(ref, cre, des, sz, p, d);
    }

    corrHelper.updateHelper(corr, d, i, a, sz);
  }

  void update(int i, int j, int a, int b, bool sz, const Slater &ref, const Corr& corr, bool doparity = true)
  {
    double p = 1.0;
    Determinant dcopy = d;
    if (doparity) p *= d.parity(a, i, sz);
    d.setocc(i, sz, false);
    d.setocc(a, sz, true);
    if (doparity) p *= d.parity(b, j, sz);
    d.setocc(j, sz, false);
    d.setocc(b, sz, true);
    if (refHelper.hftype == Generalized) {
      int norbs = Determinant::norbs;
      vector<int> cre{ a + sz * norbs, b + sz * norbs }, des{ i + sz * norbs, j + sz * norbs };
      refHelper.excitationUpdateGhf(ref, cre, des, sz, p, d);
    }
    else {
      vector<int> cre{ a, b }, des{ i, j };
      refHelper.excitationUpdate(ref, cre, des, sz, p, d);
    }
    corrHelper.updateHelper(corr, d, i, j, a, b, sz);
  }

  void updateWalker(const Slater &ref, const Corr& corr, int ex1, int ex2, bool doparity = true)
  {
    int norbs = Determinant::norbs;
    int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
    int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
    if (I % 2 == J % 2 && ex2 != 0) {
      if (I % 2 == 1) {
        update(I / 2, J / 2, A / 2, B / 2, 1, ref, corr, doparity);
      }
      else {
        update(I / 2, J / 2, A / 2, B / 2, 0, ref, corr, doparity);
      }
    }
    else {
      if (I % 2 == 0)
        update(I / 2, A / 2, 0, ref, corr, doparity);
      else
        update(I / 2, A / 2, 1, ref, corr, doparity);

      if (ex2 != 0) {
        if (J % 2 == 1) {
          update(J / 2, B / 2, 1, ref, corr, doparity);
        }
        else {
          update(J / 2, B / 2, 0, ref, corr, doparity);
        }
      }
    }
  }

  void exciteWalker(const Slater &ref, const Corr& corr, int excite1, int excite2, int norbs)
  {
    int I1 = excite1 / (2 * norbs), A1 = excite1 % (2 * norbs);

    if (I1 % 2 == 0)
      update(I1 / 2, A1 / 2, 0, ref, corr);
    else
      update(I1 / 2, A1 / 2, 1, ref, corr);

    if (excite2 != 0) {
      int I2 = excite2 / (2 * norbs), A2 = excite2 % (2 * norbs);
      if (I2 % 2 == 0)
        update(I2 / 2, A2 / 2, 0, ref, corr);
      else
        update(I2 / 2, A2 / 2, 1, ref, corr);
    }
  }

  void OverlapWithOrbGradient(const Slater &ref, Eigen::VectorXd &grad, double detovlp) const
  {
    int norbs = Determinant::norbs;
    Determinant walkerDet = d;

    //K and L are relative row and col indices
    int KA = 0, KB = 0;
    for (int k = 0; k < norbs; k++) { //walker indices on the row
      if (walkerDet.getoccA(k)) {
        for (int det = 0; det < ref.getDeterminants().size(); det++) {
          Determinant refDet = ref.getDeterminants()[det];
          int L = 0;
          for (int l = 0; l < norbs; l++) {
            if (refDet.getoccA(l)) {
              grad(k * norbs + l) += ref.getciExpansion()[det] * refHelper.thetaInv[0](L, KA) * refHelper.thetaDet[det][0] * refHelper.thetaDet[det][1] / detovlp;
              L++;
            }
          }
        }
        KA++;
      }
      if (walkerDet.getoccB(k)) {
        for (int det = 0; det < ref.getDeterminants().size(); det++) {
          Determinant refDet = ref.getDeterminants()[det];
          int L = 0;
          for (int l = 0; l < norbs; l++) {
            if (refDet.getoccB(l)) {
              if (refHelper.hftype == UnRestricted)
                grad(norbs * norbs + k * norbs + l) += ref.getciExpansion()[det] * refHelper.thetaInv[1](L, KB) * refHelper.thetaDet[det][0] * refHelper.thetaDet[det][1] / detovlp;
              else
                grad(k * norbs + l) += ref.getciExpansion()[det] * refHelper.thetaInv[1](L, KB) * refHelper.thetaDet[det][0] * refHelper.thetaDet[det][1] / detovlp;
              L++;
            }
          }
        }
        KB++;
      }
    }
  }

  void OverlapWithOrbGradientGhf(const Slater &ref, Eigen::VectorXd &grad, double detovlp) const
  {
    int norbs = Determinant::norbs;
    Determinant walkerDet = d;
    Determinant refDet = ref.getDeterminants()[0];

    //K and L are relative row and col indices
    int K = 0;
    for (int k = 0; k < norbs; k++) { //walker indices on the row
      if (walkerDet.getoccA(k)) {
        int L = 0;
        for (int l = 0; l < norbs; l++) {
          if (refDet.getoccA(l)) {
            grad(2 * k * norbs + l) += refHelper.thetaInv[0](L, K) * refHelper.thetaDet[0][0] / detovlp;
            L++;
          }
        }
        for (int l = 0; l < norbs; l++) {
          if (refDet.getoccB(l)) {
            grad(2 * k * norbs + norbs + l) += refHelper.thetaInv[0](L, K) * refHelper.thetaDet[0][0] / detovlp;
            //grad(w.getNumJastrowVariables() + w.getciExpansion().size() + k*norbs+l) += walk.alphainv(L, KA);
            L++;
          }
        }
        K++;
      }
    }
    for (int k = 0; k < norbs; k++) { //walker indices on the row
      if (walkerDet.getoccB(k)) {
        int L = 0;
        for (int l = 0; l < norbs; l++) {
          if (refDet.getoccA(l)) {
            grad(2 * norbs * norbs +  2 * k * norbs + l) += refHelper.thetaDet[0][0] * refHelper.thetaInv[0](L, K) / detovlp;
            L++;
          } 
        }
        for (int l = 0; l < norbs; l++) {
          if (refDet.getoccB(l)) {
            grad(2 * norbs * norbs +  2 * k * norbs + norbs + l) += refHelper.thetaDet[0][0] * refHelper.thetaInv[0](L, K) / detovlp;
            L++;
          }
        }
        K++;
      }
    }
  }

  void OverlapWithGradient(const Slater &ref, Eigen::VectorBlock<VectorXd> &grad) const
  {
    double detovlp = getDetOverlap(ref);
    for (int i = 0; i < ref.ciExpansion.size(); i++)
      grad[i] += getIndividualDetOverlap(i) / detovlp;
    if (ref.determinants.size() <= 1 && schd.optimizeOrbs) {
      //if (hftype == UnRestricted)
      VectorXd gradOrbitals;
      if (ref.hftype == UnRestricted) {
        gradOrbitals = VectorXd::Zero(2 * ref.HforbsA.rows() * ref.HforbsA.rows());
        OverlapWithOrbGradient(ref, gradOrbitals, detovlp);
      }
      else {
        gradOrbitals = VectorXd::Zero(ref.HforbsA.rows() * ref.HforbsA.rows());
        if (ref.hftype == Restricted) OverlapWithOrbGradient(ref, gradOrbitals, detovlp);
        else OverlapWithOrbGradientGhf(ref, gradOrbitals, detovlp);
      }
      for (int i = 0; i < gradOrbitals.size(); i++)
        grad[ref.ciExpansion.size() + i] += gradOrbitals[i];
    }
  }

  friend ostream& operator<<(ostream& os, const Walker<Corr, Slater>& w) {
    os << w.d << endl << endl;
    os << "alphaTable\n" << w.refHelper.rTable[0][0] << endl << endl;
    os << "betaTable\n" << w.refHelper.rTable[0][1] << endl << endl;
    os << "dets\n" << w.refHelper.thetaDet[0][0] << "  " << w.refHelper.thetaDet[0][1] << endl << endl;
    os << "alphaInv\n" << w.refHelper.thetaInv[0] << endl << endl;
    os << "betaInv\n" << w.refHelper.thetaInv[1] << endl << endl;
    return os;
  }

};

template<typename Corr>
struct Walker<Corr, AGP> {

  Determinant d;
  WalkerHelper<Corr> corrHelper;
  WalkerHelper<AGP> refHelper;

  Walker() {};
  
  Walker(const Corr &corr, const AGP &ref) 
  {
    initDet(ref.getPairMat());
    refHelper = WalkerHelper<AGP>(ref, d);
    corrHelper = WalkerHelper<Corr>(corr, d);
  }

  Walker(const Corr &corr, const AGP &ref, const Determinant &pd) : d(pd), refHelper(ref, pd), corrHelper(corr, pd) {}; 

  Determinant& getDet() {return d;}
  void readBestDeterminant(Determinant& d) const 
  {
    if (commrank == 0) {
      char file[5000];
      sprintf(file, "BestDeterminant.txt");
      std::ifstream ifs(file, std::ios::binary);
      boost::archive::binary_iarchive load(ifs);
      load >> d;
    }
#ifndef SERIAL
  MPI_Bcast(&d.reprA, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&d.reprB, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  }
  
  void guessBestDeterminant(Determinant& d, const Eigen::MatrixXd& pairMat) const 
  {
    int norbs = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta = Determinant::nbeta;
    for (int i = 0; i < nalpha; i++)
      d.setoccA(i, true);
    for (int i = 0; i < nbeta; i++)
      d.setoccB(i, true);
  }
  
  void initDet(const MatrixXd& pairMat) 
  {
    bool readDeterminant = false;
    char file[5000];
    sprintf(file, "BestDeterminant.txt");
  
    {
      ifstream ofile(file);
      if (ofile)
        readDeterminant = true;
    }
    if (readDeterminant)
      readBestDeterminant(d);
    else
      guessBestDeterminant(d, pairMat);
  }
  
  double getDetOverlap(const AGP &ref) const
  {
    return refHelper.thetaDet;
  }
  
  double getDetFactor(int i, int a, const AGP &ref) const 
  {
    if (i % 2 == 0)
      return getDetFactor(i / 2, a / 2, 0, ref);
    else                                   
      return getDetFactor(i / 2, a / 2, 1, ref);
  }
  
  double getDetFactor(int I, int J, int A, int B, const AGP &ref) const 
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
  
  double getDetFactor(int i, int a, bool sz, const AGP &ref) const
  {
    int tableIndexi, tableIndexa;
    refHelper.getRelIndices(i, tableIndexi, a, tableIndexa, sz); 
    return refHelper.rTable[sz](tableIndexa, tableIndexi);
  }
  
  double getDetFactor(int i, int j, int a, int b, bool sz1, bool sz2, const AGP &ref) const
  {
    int tableIndexi, tableIndexa, tableIndexj, tableIndexb;
    refHelper.getRelIndices(i, tableIndexi, a, tableIndexa, sz1); 
    refHelper.getRelIndices(j, tableIndexj, b, tableIndexb, sz2) ;
  
    double factor;
    if (sz1 == sz2)
      factor = refHelper.rTable[sz1](tableIndexa, tableIndexi) * refHelper.rTable[sz1](tableIndexb, tableIndexj) 
          - refHelper.rTable[sz1](tableIndexb, tableIndexi) *refHelper.rTable[sz1](tableIndexa, tableIndexj);
    else
      if (sz1 == 0) {
        factor = refHelper.rTable[sz1](tableIndexa, tableIndexi) * refHelper.rTable[sz2](tableIndexb, tableIndexj) 
        + refHelper.thetaInv(tableIndexj, tableIndexi) * refHelper.rTable[2](tableIndexa, tableIndexb);
      }
      else {
        factor = refHelper.rTable[sz1](tableIndexa, tableIndexi) * refHelper.rTable[sz2](tableIndexb, tableIndexj) 
        + refHelper.thetaInv(tableIndexi, tableIndexj) * refHelper.rTable[2](tableIndexb, tableIndexa);
      }
    return factor;
  }
  
  void update(int i, int a, bool sz, const AGP &ref, const Corr &corr, bool doparity = true)
  {
    double p = 1.0;
    if (doparity) p *= d.parity(a, i, sz);
    d.setocc(i, sz, false);
    d.setocc(a, sz, true);
    vector<int> cre{ a }, des{ i };
    refHelper.excitationUpdate(ref, cre, des, sz, p, d);
    corrHelper.updateHelper(corr, d, i, a, sz);
  }
  
  void update(int i, int j, int a, int b, bool sz, const AGP &ref, const Corr &corr, bool doparity = true)
  {
    double p = 1.0;
    Determinant dcopy = d;
    if (doparity) p *= d.parity(a, i, sz);
    d.setocc(i, sz, false);
    d.setocc(a, sz, true);
    if (doparity) p *= d.parity(b, j, sz);
    d.setocc(j, sz, false);
    d.setocc(b, sz, true);
    vector<int> cre{ a, b }, des{ i, j };
    refHelper.excitationUpdate(ref, cre, des, sz, p, d);
    corrHelper.updateHelper(corr, d, i, j, a, b, sz);
  }
  
  void updateWalker(const AGP& ref, const Corr &corr, int ex1, int ex2, bool doparity = true)
  {
    int norbs = Determinant::norbs;
    int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
    int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
    if (I % 2 == J % 2 && ex2 != 0) {
      if (I % 2 == 1) {
        update(I / 2, J / 2, A / 2, B / 2, 1, ref, corr, doparity);
      }
      else {
        update(I / 2, J / 2, A / 2, B / 2, 0, ref, corr, doparity);
      }
    }
    else {
      if (I % 2 == 0)
        update(I / 2, A / 2, 0, ref, corr, doparity);
      else
        update(I / 2, A / 2, 1, ref, corr, doparity);
  
      if (ex2 != 0) {
        if (J % 2 == 1) {
          update(J / 2, B / 2, 1, ref, corr, doparity);
        }
        else {
          update(J / 2, B / 2, 0, ref, corr, doparity);
        }
      }
    }
  }
  
  void exciteWalker(const AGP& ref, const Corr &corr, int excite1, int excite2, int norbs)
  {
    int I1 = excite1 / (2 * norbs), A1 = excite1 % (2 * norbs);
  
    if (I1 % 2 == 0)
      update(I1 / 2, A1 / 2, 0, ref, corr);
    else
      update(I1 / 2, A1 / 2, 1, ref, corr);
  
    if (excite2 != 0) {
      int I2 = excite2 / (2 * norbs), A2 = excite2 % (2 * norbs);
      if (I2 % 2 == 0)
        update(I2 / 2, A2 / 2, 0, ref, corr);
      else
        update(I2 / 2, A2 / 2, 1, ref, corr);
    }
  }
  
  void OverlapWithGradient(const AGP &ref, Eigen::VectorBlock<VectorXd> &grad) const
  {
    int norbs = Determinant::norbs;
    Determinant walkerDet = d;
  
    //K and L are relative row and col indices
    int K = 0;
    for (int k = 0; k < norbs; k++) { //walker indices on the row
      if (walkerDet.getoccA(k)) {
        int L = 0;
        for (int l = 0; l < norbs; l++) {
          if (walkerDet.getoccB(l)) {
            grad(k * norbs + l) += refHelper.thetaInv(L, K) ;
            L++;
          }
        }
        K++;
      }
    }
  }
  
  friend ostream& operator<<(ostream& os, const Walker<Corr, AGP>& w) {
    os << w.d << endl << endl;
    os << "alphaTable\n" << w.refHelper.rTable[0] << endl << endl;
    os << "betaTable\n" << w.refHelper.rTable[1] << endl << endl;
    os << "thirdTable\n" << w.refHelper.rTable[2] << endl << endl;
    os << "dets\n" << w.refHelper.thetaDet << endl << endl;
    os << "thetaInv\n" << w.refHelper.thetaInv << endl << endl;
    return os;
  }

};

template<typename Corr>
struct Walker<Corr, Pfaffian> {

  Determinant d;
  WalkerHelper<Corr> corrHelper;
  WalkerHelper<Pfaffian> refHelper;

  Walker() {};
  
  Walker(const Corr &corr, const Pfaffian &ref) 
  {
    initDet(ref.getPairMat());
    refHelper = WalkerHelper<Pfaffian>(ref, d);
    corrHelper = WalkerHelper<Corr>(corr, d);
  }

  Walker(const Corr &corr, const Pfaffian &ref, const Determinant &pd) : d(pd), refHelper(ref, pd), corrHelper(corr, pd) {}; 
  
  Determinant& getDet() {return d;}
  void readBestDeterminant(Determinant& d) const 
  {
    if (commrank == 0) {
      char file[5000];
      sprintf(file, "BestDeterminant.txt");
      std::ifstream ifs(file, std::ios::binary);
      boost::archive::binary_iarchive load(ifs);
      load >> d;
    }
#ifndef SERIAL
  MPI_Bcast(&d.reprA, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&d.reprB, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  }
  
  void guessBestDeterminant(Determinant& d, const Eigen::MatrixXd& pairMat) const 
  {
    int norbs = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta = Determinant::nbeta;
    for (int i = 0; i < nalpha; i++)
      d.setoccA(i, true);
    for (int i = 0; i < nbeta; i++)
      d.setoccB(i, true);
  }
  
  void initDet(const MatrixXd& pairMat) 
  {
    bool readDeterminant = false;
    char file[5000];
    sprintf(file, "BestDeterminant.txt");
  
    {
      ifstream ofile(file);
      if (ofile)
        readDeterminant = true;
    }
    if (readDeterminant)
      readBestDeterminant(d);
    else
      guessBestDeterminant(d, pairMat);
  }
  
  double getDetOverlap(const Pfaffian &ref) const
  {
    return refHelper.thetaPfaff;
  }
  
  double getDetFactor(int i, int a, const Pfaffian &ref) const 
  {
    if (i % 2 == 0)
      return getDetFactor(i / 2, a / 2, 0, ref);
    else                                   
      return getDetFactor(i / 2, a / 2, 1, ref);
  }
  
  double getDetFactor(int I, int J, int A, int B, const Pfaffian &ref) const 
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
  
  double getDetFactor(int i, int a, bool sz, const Pfaffian &ref) const
  {
    int nopen = refHelper.openOrbs[0].size() + refHelper.openOrbs[1].size();
    int tableIndexi, tableIndexa;
    refHelper.getRelIndices(i, tableIndexi, a, tableIndexa, sz); 
    //return refHelper.rTable[0](tableIndexi * nopen + tableIndexa, tableIndexi);
    return refHelper.fMat.row(tableIndexi * nopen + tableIndexa) * refHelper.thetaInv.col(tableIndexi);
  }
  
  double getDetFactor(int i, int j, int a, int b, bool sz1, bool sz2, const Pfaffian &ref) const
  {
    int norbs = Determinant::norbs;
    int nopen = refHelper.openOrbs[0].size() + refHelper.openOrbs[1].size();
    int tableIndexi, tableIndexa, tableIndexj, tableIndexb;
    refHelper.getRelIndices(i, tableIndexi, a, tableIndexa, sz1); 
    refHelper.getRelIndices(j, tableIndexj, b, tableIndexb, sz2);
    //cout << "nopen  " << nopen << endl;
    //cout << "sz1  " << sz1 << "  ti  " << tableIndexi << "  ta  " << tableIndexa  << endl;
    //cout << "sz2  " << sz2 << "  tj  " << tableIndexj << "  tb  " << tableIndexb  << endl;
    double summand1, summand2, crossTerm;
    if (tableIndexi < tableIndexj) {
      crossTerm = (ref.getPairMat())(b + sz2 * norbs, a + sz1 * norbs);
      //summand1 = refHelper.rTable[0](tableIndexi * nopen + tableIndexa, tableIndexi) * refHelper.rTable[0](tableIndexj * nopen + tableIndexb, tableIndexj) 
      //    - refHelper.rTable[0](tableIndexj * nopen + tableIndexb, tableIndexi) * refHelper.rTable[0](tableIndexi * nopen + tableIndexa, tableIndexj);
      //summand2 = refHelper.thetaInv(tableIndexi, tableIndexj) * (refHelper.rTable[1](tableIndexi * nopen + tableIndexa, tableIndexj * nopen + tableIndexb) + crossTerm);
      double term1 = refHelper.fMat.row(tableIndexi * nopen + tableIndexa) * refHelper.thetaInv.col(tableIndexi);
      double term2 = refHelper.fMat.row(tableIndexj * nopen + tableIndexb) * refHelper.thetaInv.col(tableIndexj);
      double term3 = refHelper.fMat.row(tableIndexj * nopen + tableIndexb) * refHelper.thetaInv.col(tableIndexi);
      double term4 = refHelper.fMat.row(tableIndexi * nopen + tableIndexa) * refHelper.thetaInv.col(tableIndexj);
      summand1 = term1 * term2 - term3 * term4;
      double term5 = refHelper.fMat.row(tableIndexi * nopen + tableIndexa) * refHelper.thetaInv * (-refHelper.fMat.transpose().col(tableIndexj * nopen + tableIndexb));
      summand2 = refHelper.thetaInv(tableIndexi, tableIndexj) * (term5 + crossTerm);
    }
    else { 
      crossTerm = (ref.getPairMat())(a + sz1 * norbs, b + sz2 * norbs);
      //summand1 = refHelper.rTable[0](tableIndexj * nopen + tableIndexb, tableIndexj) * refHelper.rTable[0](tableIndexi * nopen + tableIndexa, tableIndexi) 
      //    - refHelper.rTable[0](tableIndexi * nopen + tableIndexa, tableIndexj) * refHelper.rTable[0](tableIndexj * nopen + tableIndexb, tableIndexi);
      //summand2 = refHelper.thetaInv(tableIndexj, tableIndexi) * (refHelper.rTable[1](tableIndexj * nopen + tableIndexb, tableIndexi * nopen + tableIndexa) + crossTerm);
      double term1 = refHelper.fMat.row(tableIndexj * nopen + tableIndexb) * refHelper.thetaInv.col(tableIndexj);
      double term2 = refHelper.fMat.row(tableIndexi * nopen + tableIndexa) * refHelper.thetaInv.col(tableIndexi);
      double term3 = refHelper.fMat.row(tableIndexi * nopen + tableIndexa) * refHelper.thetaInv.col(tableIndexj);
      double term4 = refHelper.fMat.row(tableIndexj * nopen + tableIndexb) * refHelper.thetaInv.col(tableIndexi);
      summand1 = term1 * term2 - term3 * term4;
      double term5 = refHelper.fMat.row(tableIndexj * nopen + tableIndexb) * refHelper.thetaInv * (-refHelper.fMat.transpose().col(tableIndexi * nopen + tableIndexa));
      summand2 = refHelper.thetaInv(tableIndexj, tableIndexi) * (term5 + crossTerm);
    }
    //cout << "double   " << crossTerm << "   " << summand1 << "  " << summand2 << endl; 
    return summand1 + summand2;
  }
  
  void update(int i, int a, bool sz, const Pfaffian &ref, const Corr &corr, bool doparity = true)
  {
    double p = 1.0;
    p *= d.parity(a, i, sz);
    d.setocc(i, sz, false);
    d.setocc(a, sz, true);
    refHelper.excitationUpdate(ref, i, a, sz, p, d);
    corrHelper.updateHelper(corr, d, i, a, sz);
  }
  
  void updateWalker(const Pfaffian& ref, const Corr &corr, int ex1, int ex2, bool doparity = true)
  {
    int norbs = Determinant::norbs;
    int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
    int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
    
    if (I % 2 == 0)
      update(I / 2, A / 2, 0, ref, corr, doparity);
    else
      update(I / 2, A / 2, 1, ref, corr, doparity);
  
    if (ex2 != 0) {
      if (J % 2 == 1) 
        update(J / 2, B / 2, 1, ref, corr, doparity);
      else 
        update(J / 2, B / 2, 0, ref, corr, doparity);
    }
  }
  
  void exciteWalker(const Pfaffian& ref, const Corr &corr, int excite1, int excite2, int norbs)
  {
    int I1 = excite1 / (2 * norbs), A1 = excite1 % (2 * norbs);
  
    if (I1 % 2 == 0)
      update(I1 / 2, A1 / 2, 0, ref, corr);
    else
      update(I1 / 2, A1 / 2, 1, ref, corr);
  
    if (excite2 != 0) {
      int I2 = excite2 / (2 * norbs), A2 = excite2 % (2 * norbs);
      if (I2 % 2 == 0)
        update(I2 / 2, A2 / 2, 0, ref, corr);
      else
        update(I2 / 2, A2 / 2, 1, ref, corr);
    }
  }
  
  void OverlapWithGradient(const Pfaffian &ref, Eigen::VectorBlock<VectorXd> &grad) const
  {
    int norbs = Determinant::norbs;
    Determinant walkerDet = d;
  
    //K and L are relative row and col indices
    int K = 0;
    for (int k = 0; k < norbs; k++) { //walker indices on the row
      if (walkerDet.getoccA(k)) {
        int L = 0;
        for (int l = 0; l < norbs; l++) {
          if (walkerDet.getoccA(l)) {
            grad(2 * k * norbs + l) += refHelper.thetaInv(L, K) / 2;
            L++;
          }
        }
        for (int l = 0; l < norbs; l++) {
          if (walkerDet.getoccB(l)) {
            grad(2 * k * norbs + norbs + l) += refHelper.thetaInv(L, K) / 2;
            L++;
          }
        }
        K++;
      }
    }
    for (int k = 0; k < norbs; k++) { //walker indices on the row
      if (walkerDet.getoccB(k)) {
        int L = 0;
        for (int l = 0; l < norbs; l++) {
          if (walkerDet.getoccA(l)) {
            grad(2 * norbs * norbs + 2 * k * norbs + l) += refHelper.thetaInv(L, K) / 2;
            L++;
          }
        }
        for (int l = 0; l < norbs; l++) {
          if (walkerDet.getoccB(l)) {
            grad(2 * norbs * norbs + 2 * k * norbs + norbs + l) += refHelper.thetaInv(L, K) / 2;
            L++;
          }
        }
        K++;
      }
    }
  }
  
  friend ostream& operator<<(ostream& os, const Walker<Corr, Pfaffian>& w) {
    os << w.d << endl << endl;
    os << "fMat\n" << w.refHelper.fMat << endl << endl;
    os << "fThetaInv\n" << w.refHelper.rTable[0] << endl << endl;
    os << "fThetaInvf\n" << w.refHelper.rTable[1] << endl << endl;
    os << "pfaff\n" << w.refHelper.thetaPfaff << endl << endl;
    os << "thetaInv\n" << w.refHelper.thetaInv << endl << endl;
    return os;
  }

};

#endif
