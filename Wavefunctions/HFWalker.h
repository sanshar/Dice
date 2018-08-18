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
#include "igl/slice.h"
#include "igl/slice_into.h"
#include "input.h"

using namespace Eigen;

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

class HFWalker
{

public:
  Determinant d;                      //The current determinant
  Eigen::MatrixXd alphainv;           //The inverse of the beta determinant
  Eigen::MatrixXd betainv;            //The inverse of the beta determinant
  vector<double> alphaDet;            //The alpha determinant
  vector<double> betaDet;             //The beta determinant
  vector<int> AlphaOpen;              //The set of open alpha orbitals
  vector<int> AlphaClosed;            //The set of closed alpha orbitals
  vector<int> BetaOpen;               //The set of open beta orbitals
  vector<int> BetaClosed;             //The set of closed alpha orbitals
  vector<Eigen::MatrixXd> AlphaTable; //This is the table used for efficiently
  vector<Eigen::MatrixXd> BetaTable;  //calculation of local energy, gradient, update

  // The constructor
  HFWalker(Determinant &pd) : d(pd){};
  HFWalker(){};

  //Use the wavefunction to initialize the alphainv, betainv, alphaDet
  //and betadet
  template <typename Wfn>
  void initUsingWave(Wfn &w, bool check = false)
  {

    AlphaTable.resize(w.getDeterminants().size());
    BetaTable.resize(w.getDeterminants().size());
    alphaDet.resize(w.getDeterminants().size());
    betaDet.resize(w.getDeterminants().size());

    AlphaOpen.clear();
    AlphaClosed.clear();
    BetaOpen.clear();
    BetaClosed.clear();
    d.getOpenClosedAlphaBeta(AlphaOpen, AlphaClosed, BetaOpen, BetaClosed);

    Eigen::Map<VectorXi> RowAlpha(&AlphaClosed[0], AlphaClosed.size());
    Eigen::Map<VectorXi> RowBeta(&BetaClosed[0], BetaClosed.size());
    Eigen::Map<VectorXi> RowAlphaOpen(&AlphaOpen[0], AlphaOpen.size());
    Eigen::Map<VectorXi> RowBetaOpen(&BetaOpen[0], BetaOpen.size());
    int norbs = Determinant::norbs;
    int nalpha = AlphaClosed.size();
    int nbeta = BetaClosed.size();

    vector<int> creA(nalpha, -1), desA(nalpha, -1), creB(nbeta, -1), desB(nbeta, -1);

    vector<int> alphaRef0, betaRef0;
    w.getDeterminants()[0].getAlphaBeta(alphaRef0, betaRef0);

    for (int i = 0; i < w.getDeterminants().size(); i++)
    {
      MatrixXd alpha, beta;
      MatrixXd alphainvCurrent, betainvCurrent;

      //Generate the alpha and beta strings for the wavefunction determinant
      vector<int> alphaRef, betaRef;
      w.getDeterminants()[i].getAlphaBeta(alphaRef, betaRef);
      Eigen::Map<VectorXi> ColAlpha(&alphaRef[0], alphaRef.size());
      Eigen::Map<VectorXi> ColBeta(&betaRef[0], betaRef.size());

      if (i == 0)
      //if (true)
      {
        igl::slice(w.getHforbsA(), RowAlpha, ColAlpha, alpha); //alpha = Hforbs(Row, Col)
        Eigen::FullPivLU<MatrixXd> lua(alpha);
        if (lua.isInvertible() || !check)
        {
          alphainv = lua.inverse();
          alphainvCurrent = alphainv;
          alphaDet[i] = lua.determinant();
        }
        else
        {
          cout << RowAlpha << endl
               << endl;
          cout << ColAlpha << endl
               << endl;
          cout << alpha << endl
               << endl;
          cout << lua.determinant() << endl;
          cout << lua.inverse() << endl;
          cout << "overlap with alpha determinant " << d << " not invertible" << endl;
          exit(0);
        }

        igl::slice(w.getHforbsB(), RowBeta, ColBeta, beta); //beta = Hforbs(Row, Col)
        Eigen::FullPivLU<MatrixXd> lub(beta);
        if (lub.isInvertible() || !check)
        {
          betainv = lub.inverse();
          betainvCurrent = betainv;
          betaDet[i] = lub.determinant();
        }
        else
        {
          cout << "overlap with beta determinant " << d << " not invertible" << endl;
          exit(0);
        }
      }
      else
      {
        getOrbDiff(w.getDeterminants()[i], w.getDeterminants()[0], creA, desA, creB, desB);
        double alphaParity = w.getDeterminants()[0].parityA(creA, desA);
        double betaParity = w.getDeterminants()[0].parityB(creB, desB);
        calculateInverseDeterminantWithColumnChange(alphainv, alphaDet[0], alphainvCurrent, alphaDet[i], creA, desA, RowAlpha, alphaRef0, w.getHforbsA());
        calculateInverseDeterminantWithColumnChange(betainv, betaDet[0], betainvCurrent, betaDet[i], creB, desB, RowBeta, betaRef0, w.getHforbsB());
        alphaDet[i] *= alphaParity;
        betaDet[i] *= betaParity;
      }
      //cout << alphainvCurrent<<endl<<endl;
      //cout << betainvCurrent<<endl<<endl;

      AlphaTable[i] = MatrixXd::Zero(AlphaOpen.size(), AlphaClosed.size()); //k-N x N
      MatrixXd HfopenAlpha;
      igl::slice(w.getHforbsA(), RowAlphaOpen, ColAlpha, HfopenAlpha);
      AlphaTable[i] = HfopenAlpha * alphainvCurrent;

      BetaTable[i] = MatrixXd::Zero(BetaOpen.size(), BetaClosed.size()); //k-N x N
      MatrixXd HfopenBeta;
      igl::slice(w.getHforbsB(), RowBetaOpen, ColBeta, HfopenBeta);
      BetaTable[i] = HfopenBeta * betainvCurrent;
    }
    //exit(0);
  }

  template <typename Wfn>
  double getDetOverlap(Wfn &w)
  {
    double ovlp = 0.0;
    for (int i = 0; i < alphaDet.size(); i++)
    {
      ovlp += w.getciExpansion()[i] * alphaDet[i] * betaDet[i];
    }
    return ovlp;
  }

  template <typename Wfn>
  void updateWalker(Wfn &w, int ex1, int ex2)
  {
    int norbs = Determinant::norbs;

    int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
    int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;

    if (I % 2 == J % 2 && ex2 != 0)
    {
      if (I % 2 == 1)
      {
        updateB(I / 2, J / 2, A / 2, B / 2, w);
      }

      else
      {
        updateA(I / 2, J / 2, A / 2, B / 2, w);
      }
    }
    else
    {
      if (I % 2 == 0)
        updateA(I / 2, A / 2, w);
      else
        updateB(I / 2, A / 2, w);

      if (ex2 != 0)
      {
        if (J % 2 == 1)
        {
          updateB(J / 2, B / 2, w);
        }
        else
        {
          updateA(J / 2, B / 2, w);
        }
      }
    }
  }

  Determinant &getDet() { return d; }

  //these are not absolute orbital indices, but instead the
  //ith occupied and ath unoccupied

  template <typename Wfn>
  double getDetFactorA(int i, int a, Wfn &w, bool doparity)
  {
    int tableIndexi = std::lower_bound(AlphaClosed.begin(), AlphaClosed.end(), i) - AlphaClosed.begin();
    int tableIndexa = std::lower_bound(AlphaOpen.begin(), AlphaOpen.end(), a) - AlphaOpen.begin();

    double p = 1.;
    if (doparity)
      d.parityA(a, i, p);

    double detFactorNum = 0.0;
    double detFactorDen = 0.0;
    for (int i = 0; i < w.getDeterminants().size(); i++)
    {
      double factor = AlphaTable[i](tableIndexa, tableIndexi);
      detFactorNum += w.getciExpansion()[i] * factor * alphaDet[i] * betaDet[i];
      detFactorDen += w.getciExpansion()[i] * alphaDet[i] * betaDet[i];
    }

    return p * detFactorNum / detFactorDen;
  }

  template <typename Wfn>
  double getDetFactorB(int i, int a, Wfn &w, bool doparity)
  {
    int tableIndexi = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), i) - BetaClosed.begin();
    int tableIndexa = std::lower_bound(BetaOpen.begin(), BetaOpen.end(), a) - BetaOpen.begin();

    double p = 1.;
    if (doparity)
      d.parityB(a, i, p);

    double detFactorNum = 0.0;
    double detFactorDen = 0.0;
    for (int i = 0; i < w.getDeterminants().size(); i++)
    {
      double factor = BetaTable[i](tableIndexa, tableIndexi);
      detFactorNum += w.getciExpansion()[i] * factor * alphaDet[i] * betaDet[i];
      detFactorDen += w.getciExpansion()[i] * alphaDet[i] * betaDet[i];
    }

    return p * detFactorNum / detFactorDen;
  }

  template <typename Wfn>
  double getDetFactorA(int i, int j, int a, int b, Wfn &w, bool doparity)
  {
    int tableIndexi = std::lower_bound(AlphaClosed.begin(), AlphaClosed.end(), i) - AlphaClosed.begin();
    int tableIndexa = std::lower_bound(AlphaOpen.begin(), AlphaOpen.end(), a) - AlphaOpen.begin();
    int tableIndexj = std::lower_bound(AlphaClosed.begin(), AlphaClosed.end(), j) - AlphaClosed.begin();
    int tableIndexb = std::lower_bound(AlphaOpen.begin(), AlphaOpen.end(), b) - AlphaOpen.begin();

    double detFactorNum = 0.0;
    double detFactorDen = 0.0;
    for (int i = 0; i < w.getDeterminants().size(); i++)
    {
      double factor = (AlphaTable[i](tableIndexa, tableIndexi) * AlphaTable[i](tableIndexb, tableIndexj) - AlphaTable[i](tableIndexb, tableIndexi) * AlphaTable[i](tableIndexa, tableIndexj));
      detFactorNum += w.getciExpansion()[i] * factor * alphaDet[i] * betaDet[i];
      detFactorDen += w.getciExpansion()[i] * alphaDet[i] * betaDet[i];
    }
    return detFactorNum / detFactorDen;
  }

  template <typename Wfn>
  double getDetFactorB(int i, int j, int a, int b, Wfn &w, bool doparity)
  {
    int tableIndexi = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), i) - BetaClosed.begin();
    int tableIndexa = std::lower_bound(BetaOpen.begin(), BetaOpen.end(), a) - BetaOpen.begin();
    int tableIndexj = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), j) - BetaClosed.begin();
    int tableIndexb = std::lower_bound(BetaOpen.begin(), BetaOpen.end(), b) - BetaOpen.begin();

    double detFactorNum = 0.0;
    double detFactorDen = 0.0;
    for (int i = 0; i < w.getDeterminants().size(); i++)
    {
      double factor = (BetaTable[i](tableIndexa, tableIndexi) * BetaTable[i](tableIndexb, tableIndexj) - BetaTable[i](tableIndexb, tableIndexi) * BetaTable[i](tableIndexa, tableIndexj));
      detFactorNum += w.getciExpansion()[i] * factor * alphaDet[i] * betaDet[i];
      detFactorDen += w.getciExpansion()[i] * alphaDet[i] * betaDet[i];
    }
    return detFactorNum / detFactorDen;
  }

  template <typename Wfn>
  double getDetFactorAB(int i, int j, int a, int b, Wfn &w, bool doparity)
  {
    int tableIndexi = std::lower_bound(AlphaClosed.begin(), AlphaClosed.end(), i) - AlphaClosed.begin();
    int tableIndexa = std::lower_bound(AlphaOpen.begin(), AlphaOpen.end(), a) - AlphaOpen.begin();
    int tableIndexj = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), j) - BetaClosed.begin();
    int tableIndexb = std::lower_bound(BetaOpen.begin(), BetaOpen.end(), b) - BetaOpen.begin();

    double detFactorNum = 0.0;
    double detFactorDen = 0.0;
    for (int I = 0; I < w.getDeterminants().size(); I++)
    {
      double factor = AlphaTable[I](tableIndexa, tableIndexi) * BetaTable[I](tableIndexb, tableIndexj);
      detFactorNum += w.getciExpansion()[I] * factor * alphaDet[I] * betaDet[I];
      detFactorDen += w.getciExpansion()[I] * alphaDet[I] * betaDet[I];
    }
    return detFactorNum / detFactorDen;
  }

  template <typename Wfn>
  double getDetFactorA(vector<int> &iArray, vector<int> &aArray, Wfn &w, bool doparity)
  {
    MatrixXd localDet = MatrixXd::Zero(aArray.size(), iArray.size());
    for (int i = 0; i < iArray.size(); i++)
    {
      int tableIndexi = std::lower_bound(AlphaClosed.begin(), AlphaClosed.end(), iArray[i]) - AlphaClosed.begin();
      for (int a = 0; a < aArray.size(); a++)
      {
        int tableIndexa = std::lower_bound(AlphaOpen.begin(), AlphaOpen.end(), aArray[a]) - AlphaOpen.begin();
        localDet(i, a) = AlphaTable[0](tableIndexi, tableIndexa);
      }
    }

    double p = 1.;
    Determinant dcopy = d;
    for (int i = 0; i < iArray.size(); i++)
    {
      if (doparity)
        dcopy.parityA(aArray[i], iArray[i], p);

      dcopy.setoccA(iArray[i], false);
      dcopy.setoccA(aArray[i], true);
    }

    double cpsFactor = 1.0;

    return p * cpsFactor * localDet.determinant();
  }

  template <typename Wfn>
  double getDetFactorB(vector<int> &iArray, vector<int> &aArray, Wfn &w, bool doparity)
  {
    MatrixXd localDet = MatrixXd::Zero(aArray.size(), iArray.size());
    for (int i = 0; i < iArray.size(); i++)
    {
      int tableIndexi = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), iArray[i]) - BetaClosed.begin();
      for (int a = 0; a < aArray.size(); a++)
      {
        int tableIndexa = std::lower_bound(BetaOpen.begin(), BetaOpen.end(), aArray[a]) - BetaOpen.begin();
        localDet(i, a) = BetaTable[0](tableIndexi, tableIndexa);
      }
    }

    double p = 1.;
    Determinant dcopy = d;
    for (int i = 0; i < iArray.size(); i++)
    {
      if (doparity)
        dcopy.parityB(aArray[i], iArray[i], p);

      dcopy.setoccB(iArray[i], false);
      dcopy.setoccB(aArray[i], true);
    }

    double cpsFactor = 1.0;

    return p * cpsFactor * localDet.determinant();
  }

  template <typename Wfn>
  void updateA(int i, int a, Wfn &w)
  {

    double p = 1.0;
    d.parityA(a, i, p);

    int tableIndexi = std::lower_bound(AlphaClosed.begin(), AlphaClosed.end(), i) - AlphaClosed.begin();
    int tableIndexa = std::lower_bound(AlphaOpen.begin(), AlphaOpen.end(), a) - AlphaOpen.begin();

    int norbs = Determinant::norbs;
    int nalpha = AlphaClosed.size();
    int nbeta = BetaClosed.size();
    vector<int> creA(nalpha, -1), desA(nalpha, -1), creB(nbeta, -1), desB(nbeta, -1);

    vector<int> alphaRef0, betaRef0;
    w.getDeterminants()[0].getAlphaBeta(alphaRef0, betaRef0);
    vector<int> cre(1, a), des(1, i);
    MatrixXd alphainvOld = alphainv;
    double alphaDetOld = alphaDet[0];
    Eigen::Map<Eigen::VectorXi> ColVec(&alphaRef0[0], alphaRef0.size());
    calculateInverseDeterminantWithRowChange(alphainvOld, alphaDetOld, alphainv,
                                             alphaDet[0], cre, des, ColVec, AlphaClosed, w.getHforbsA());
    alphaDet[0] *= p;

    d.setoccA(i, false);
    d.setoccA(a, true);
    AlphaOpen.clear();
    AlphaClosed.clear();
    BetaOpen.clear();
    BetaClosed.clear();
    d.getOpenClosedAlphaBeta(AlphaOpen, AlphaClosed, BetaOpen, BetaClosed);
    Eigen::Map<VectorXi> RowAlpha(&AlphaClosed[0], AlphaClosed.size());
    Eigen::Map<VectorXi> RowBeta(&BetaClosed[0], BetaClosed.size());

    Eigen::Map<VectorXi> RowAlphaOpen(&AlphaOpen[0], AlphaOpen.size());
    MatrixXd HfopenAlpha;
    igl::slice(w.getHforbsA(), RowAlphaOpen, ColVec, HfopenAlpha);
    AlphaTable[0] = HfopenAlpha * alphainv;

    for (int x = 1; x < w.getDeterminants().size(); x++)
    {
      MatrixXd alphainvCurrent, betainvCurrent;

      vector<int> alphaRef, betaRef;
      w.getDeterminants()[x].getAlphaBeta(alphaRef, betaRef);

      getOrbDiff(w.getDeterminants()[x], w.getDeterminants()[0], creA, desA, creB, desB);
      double alphaParity = w.getDeterminants()[0].parityA(creA, desA);
      calculateInverseDeterminantWithColumnChange(alphainv, alphaDet[0], alphainvCurrent, alphaDet[x], creA, desA, RowAlpha, alphaRef0, w.getHforbsA());
      alphaDet[x] *= alphaParity;

      MatrixXd HfopenAlpha;
      Eigen::Map<VectorXi> ColAlpha(&alphaRef[0], alphaRef.size());
      igl::slice(w.getHforbsA(), RowAlphaOpen, ColAlpha, HfopenAlpha);
      AlphaTable[x] = HfopenAlpha * alphainvCurrent;
    }
  }

  template <typename Wfn>
  void updateA(int i, int j, int a, int b, Wfn &w)
  {

    double p = 1.0;
    Determinant dcopy = d;
    dcopy.parityA(a, i, p);
    dcopy.setoccA(i, false);
    dcopy.setoccA(a, true);
    dcopy.parityA(b, j, p);
    dcopy.setoccA(j, false);
    dcopy.setoccA(b, true);

    int norbs = Determinant::norbs;
    int nalpha = AlphaClosed.size();
    int nbeta = BetaClosed.size();
    vector<int> creA(nalpha, -1), desA(nalpha, -1), creB(nbeta, -1), desB(nbeta, -1);

    vector<int> alphaRef0, betaRef0;
    w.getDeterminants()[0].getAlphaBeta(alphaRef0, betaRef0);
    vector<int> cre(2, a), des(2, i);
    cre[1] = b;
    des[1] = j;
    MatrixXd alphainvOld = alphainv;
    double alphaDetOld = alphaDet[0];
    Eigen::Map<Eigen::VectorXi> ColVec(&alphaRef0[0], alphaRef0.size());
    calculateInverseDeterminantWithRowChange(alphainvOld, alphaDetOld, alphainv,
                                             alphaDet[0], cre, des, ColVec, AlphaClosed, w.getHforbsA());
    alphaDet[0] *= p;

    d = dcopy;
    AlphaOpen.clear();
    AlphaClosed.clear();
    BetaOpen.clear();
    BetaClosed.clear();
    d.getOpenClosedAlphaBeta(AlphaOpen, AlphaClosed, BetaOpen, BetaClosed);
    Eigen::Map<VectorXi> RowAlpha(&AlphaClosed[0], AlphaClosed.size());
    Eigen::Map<VectorXi> RowBeta(&BetaClosed[0], BetaClosed.size());

    Eigen::Map<VectorXi> RowAlphaOpen(&AlphaOpen[0], AlphaOpen.size());
    MatrixXd HfopenAlpha;
    igl::slice(w.getHforbsA(), RowAlphaOpen, ColVec, HfopenAlpha);
    AlphaTable[0] = HfopenAlpha * alphainv;

    for (int x = 1; x < w.getDeterminants().size(); x++)
    {
      MatrixXd alphainvCurrent, betainvCurrent;

      vector<int> alphaRef, betaRef;
      w.getDeterminants()[x].getAlphaBeta(alphaRef, betaRef);

      getOrbDiff(w.getDeterminants()[x], w.getDeterminants()[0], creA, desA, creB, desB);
      double alphaParity = w.getDeterminants()[0].parityA(creA, desA);
      calculateInverseDeterminantWithColumnChange(alphainv, alphaDet[0], alphainvCurrent, alphaDet[x], creA, desA, RowAlpha, alphaRef0, w.getHforbsA());
      alphaDet[x] *= alphaParity;

      MatrixXd HfopenAlpha;
      Eigen::Map<VectorXi> ColAlpha(&alphaRef[0], alphaRef.size());
      igl::slice(w.getHforbsA(), RowAlphaOpen, ColAlpha, HfopenAlpha);
      AlphaTable[x] = HfopenAlpha * alphainvCurrent;
    }
  }

  template <typename Wfn>
  void updateB(int i, int a, Wfn &w)
  {

    double p = 1.0;
    d.parityB(a, i, p);

    int tableIndexi = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), i) - BetaClosed.begin();
    int tableIndexa = std::lower_bound(BetaOpen.begin(), BetaOpen.end(), a) - BetaOpen.begin();

    int norbs = Determinant::norbs;
    int nalpha = AlphaClosed.size();
    int nbeta = BetaClosed.size();
    vector<int> creA(nalpha, -1), desA(nalpha, -1), creB(nbeta, -1), desB(nbeta, -1);

    vector<int> alphaRef0, betaRef0;
    w.getDeterminants()[0].getAlphaBeta(alphaRef0, betaRef0);
    vector<int> cre(1, a), des(1, i);
    MatrixXd betainvOld = betainv;
    double betaDetOld = betaDet[0];
    Eigen::Map<Eigen::VectorXi> ColVec(&betaRef0[0], betaRef0.size());
    calculateInverseDeterminantWithRowChange(betainvOld, betaDetOld, betainv,
                                             betaDet[0], cre, des, ColVec, BetaClosed, w.getHforbsB());
    betaDet[0] *= p;

    d.setoccB(i, false);
    d.setoccB(a, true);
    AlphaOpen.clear();
    AlphaClosed.clear();
    BetaOpen.clear();
    BetaClosed.clear();
    d.getOpenClosedAlphaBeta(AlphaOpen, AlphaClosed, BetaOpen, BetaClosed);
    Eigen::Map<VectorXi> RowAlpha(&AlphaClosed[0], AlphaClosed.size());
    Eigen::Map<VectorXi> RowBeta(&BetaClosed[0], BetaClosed.size());

    Eigen::Map<VectorXi> RowBetaOpen(&BetaOpen[0], BetaOpen.size());
    MatrixXd HfopenBeta;
    igl::slice(w.getHforbsB(), RowBetaOpen, ColVec, HfopenBeta);
    BetaTable[0] = HfopenBeta * betainv;

    for (int x = 1; x < w.getDeterminants().size(); x++)
    {
      MatrixXd betainvCurrent;

      vector<int> alphaRef, betaRef;
      w.getDeterminants()[x].getAlphaBeta(alphaRef, betaRef);

      getOrbDiff(w.getDeterminants()[x], w.getDeterminants()[0], creA, desA, creB, desB);
      double betaParity = w.getDeterminants()[0].parityB(creB, desB);
      calculateInverseDeterminantWithColumnChange(betainv, betaDet[0], betainvCurrent, betaDet[x], creB, desB, RowBeta, betaRef0, w.getHforbsB());
      betaDet[x] *= betaParity;

      MatrixXd HfopenBeta;
      Eigen::Map<VectorXi> ColBeta(&betaRef[0], betaRef.size());
      igl::slice(w.getHforbsB(), RowBetaOpen, ColBeta, HfopenBeta);
      BetaTable[x] = HfopenBeta * betainvCurrent;
    }
  }

  template <typename Wfn>
  void updateB(int i, int j, int a, int b, Wfn &w)
  {

    double p = 1.0;
    Determinant dcopy = d;
    dcopy.parityB(a, i, p);
    dcopy.setoccB(i, false);
    dcopy.setoccB(a, true);
    dcopy.parityB(b, j, p);
    dcopy.setoccB(j, false);
    dcopy.setoccB(b, true);

    int tableIndexi = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), i) - BetaClosed.begin();
    int tableIndexa = std::lower_bound(BetaOpen.begin(), BetaOpen.end(), a) - BetaOpen.begin();

    int norbs = Determinant::norbs;
    int nalpha = AlphaClosed.size();
    int nbeta = BetaClosed.size();
    vector<int> creA(nalpha, -1), desA(nalpha, -1), creB(nbeta, -1), desB(nbeta, -1);

    vector<int> alphaRef0, betaRef0;
    w.getDeterminants()[0].getAlphaBeta(alphaRef0, betaRef0);
    vector<int> cre(2, a), des(2, i);
    cre[1] = b;
    des[1] = j;
    MatrixXd betainvOld = betainv;
    double betaDetOld = betaDet[0];
    Eigen::Map<Eigen::VectorXi> ColVec(&betaRef0[0], betaRef0.size());
    calculateInverseDeterminantWithRowChange(betainvOld, betaDetOld, betainv,
                                             betaDet[0], cre, des, ColVec, BetaClosed, w.getHforbsB());
    betaDet[0] *= p;

    d = dcopy;
    AlphaOpen.clear();
    AlphaClosed.clear();
    BetaOpen.clear();
    BetaClosed.clear();
    d.getOpenClosedAlphaBeta(AlphaOpen, AlphaClosed, BetaOpen, BetaClosed);
    Eigen::Map<VectorXi> RowAlpha(&AlphaClosed[0], AlphaClosed.size());
    Eigen::Map<VectorXi> RowBeta(&BetaClosed[0], BetaClosed.size());

    Eigen::Map<VectorXi> RowBetaOpen(&BetaOpen[0], BetaOpen.size());
    MatrixXd HfopenBeta;
    igl::slice(w.getHforbsB(), RowBetaOpen, ColVec, HfopenBeta);
    BetaTable[0] = HfopenBeta * betainv;

    for (int x = 1; x < w.getDeterminants().size(); x++)
    {
      MatrixXd betainvCurrent;

      vector<int> alphaRef, betaRef;
      w.getDeterminants()[x].getAlphaBeta(alphaRef, betaRef);

      getOrbDiff(w.getDeterminants()[x], w.getDeterminants()[0], creA, desA, creB, desB);
      double betaParity = w.getDeterminants()[0].parityB(creB, desB);
      calculateInverseDeterminantWithColumnChange(betainv, betaDet[0], betainvCurrent, betaDet[x], creB, desB, RowBeta, betaRef0, w.getHforbsB());
      betaDet[x] *= betaParity;

      MatrixXd HfopenBeta;
      Eigen::Map<VectorXi> ColBeta(&betaRef[0], betaRef.size());
      igl::slice(w.getHforbsB(), RowBetaOpen, ColBeta, HfopenBeta);
      BetaTable[x] = HfopenBeta * betainvCurrent;
    }
  }

  bool operator<(const HFWalker &w) const
  {
    return d < w.d;
  }

  bool operator==(const HFWalker &w) const
  {
    return d == w.d;
  }
  /**
   * This takes an inverse and determinant of a matrix formed by a subset of
   * columns and rows of Hforbs
   * and generates the new inverse and determinant 
   * by replacing cols with incides des with those with indices cre
   * RowVec is the set of row indices that are common to both in the 
   * incoming and outgoing matrices. ColIn are the column indices
   * of the incoming matrix. 
   */
  void calculateInverseDeterminantWithColumnChange(Eigen::MatrixXd &inverseIn, double &detValueIn,
                                                   Eigen::MatrixXd &inverseOut, double &detValueOut,
                                                   vector<int> &cre, vector<int> &des,
                                                   Eigen::Map<Eigen::VectorXi> &RowVec,
                                                   vector<int> &ColIn, Eigen::MatrixXd &Hforbs);

  /**
   * This takes an inverse and determinant of a matrix formed by a subset of
   * columns and rows of Hforbs
   * and generates the new inverse and determinant 
   * by replacing rows with incides des with those with indices des
   * ColVec is the set of col indices that are common to both in the 
   * incoming and outgoing matrices. RowIn are the column indices
   * of the incoming matrix. 
   */
  void calculateInverseDeterminantWithRowChange(Eigen::MatrixXd &inverseIn, double &detValueIn,
                                                Eigen::MatrixXd &inverseOut, double &detValueOut,
                                                vector<int> &cre, vector<int> &des,
                                                Eigen::Map<Eigen::VectorXi> &ColVec,
                                                vector<int> &RowIn, Eigen::MatrixXd &Hforbs);
  
  //template <typename Wfn>
  //bool makeMove(Wfn &w);

  template <typename Wfn>
  void exciteWalker(Wfn &w, int excite1, int excite2, int norbs)
  {
    int I1 = excite1 / (2 * norbs), A1 = excite1 % (2 * norbs);

    if (I1 % 2 == 0)
      updateA(I1 / 2, A1 / 2, w);
    else
      updateB(I1 / 2, A1 / 2, w);

    if (excite2 != 0)
    {
      int I2 = excite2 / (2 * norbs), A2 = excite2 % (2 * norbs);
      if (I2 % 2 == 0)
        updateA(I2 / 2, A2 / 2, w);
      else
        updateB(I2 / 2, A2 / 2, w);
    }
  }
  template <typename Wfn>
  void OverlapWithGradient(Wfn &w, Eigen::VectorXd &grad, double detovlp)
  {
    int numJastrowVariables = w.getNumJastrowVariables();
    int norbs = Determinant::norbs;
    HFWalker &walk = *this;

    int KA = 0, KB = 0;
    for (int k = 0; k < norbs; k++)
    { //walker indices on the row
      if (walk.d.getoccA(k))
      {

        for (int det = 0; det < w.getDeterminants().size(); det++)
        {
          Determinant ddet = w.getDeterminants()[det];
          int L = 0;
          for (int l = 0; l < norbs; l++)
          {
            if (ddet.getoccA(l))
            {
              grad(numJastrowVariables + w.getciExpansion().size() + k * norbs + l) += w.getciExpansion()[det] * walk.alphainv(L, KA) * walk.alphaDet[det] * walk.betaDet[det] / detovlp;
              //grad(w.getNumJastrowVariables() + w.getciExpansion().size() + k*norbs+l) += walk.alphainv(L, KA);
              L++;
            }
          }
        }
        KA++;
      }
      if (walk.d.getoccB(k))
      {

        for (int det = 0; det < w.getDeterminants().size(); det++)
        {
          Determinant ddet = w.getDeterminants()[det];
          int L = 0;
          for (int l = 0; l < norbs; l++)
          {
            if (ddet.getoccB(l))
            {
              if (schd.uhf)
                grad(numJastrowVariables + w.getciExpansion().size() + norbs * norbs + k * norbs + l) += w.getciExpansion()[det] * walk.alphaDet[det] * walk.betaDet[det] * walk.betainv(L, KB) / detovlp;
              else
                grad(numJastrowVariables + w.getciExpansion().size() + k * norbs + l) += w.getciExpansion()[det] * walk.alphaDet[det] * walk.betaDet[det] * walk.betainv(L, KB) / detovlp;
              //grad(w.getNumJastrowVariables() + w.getciExpansion().size() + k*norbs+l) += walk.betainv(L, KB);
              L++;
            }
          }
        }
        KB++;
      }
    }
  }
};

#endif
