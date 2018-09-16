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
#ifndef GHFWalker_HEADER_H
#define GHFWalker_HEADER_H

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
* theta matrix, and also the Determinant theta matrix
* 
*/

//only works with one GHF determinant (w.getDeterminants().Size() = 1)
//alpha and beta combined to make a contiguous GHF orbital index in the GHF determinant
//need to implement a separate GHF det

class GHFWalker
{

public:
  Determinant d;                      //The current determinant
  Eigen::MatrixXd GHFInv;           //The inverse of the ghf slice 
  vector<double> GHFDet;            //The determinant of the slice
  vector<int> AlphaOpen;              //The set of open alpha orbitals
  vector<int> AlphaClosed;            //The set of closed alpha orbitals
  vector<int> BetaOpen;               //The set of open beta orbitals
  vector<int> BetaClosed;             //The set of closed alpha orbitals
  vector<Eigen::MatrixXd> AlphaTable; //This is the table used for efficiently
  vector<Eigen::MatrixXd> BetaTable; //This is the table used for efficiently

  // The constructor
  GHFWalker(Determinant &pd) : d(pd){};
  GHFWalker(){};

  //Use the wavefunction to initialize the alphainv, betainv, alphaDet
  //and betadet
  template <typename Wfn>
  void initUsingWave(Wfn &w, bool check = false)
  {
    int norbs = Determinant::norbs;

    AlphaTable.resize(w.getDeterminants().size());
    BetaTable.resize(w.getDeterminants().size());
    GHFDet.resize(w.getDeterminants().size());

    AlphaOpen.clear();
    AlphaClosed.clear();
    BetaOpen.clear();
    BetaClosed.clear();
    d.getOpenClosedAlphaBeta(AlphaOpen, AlphaClosed, BetaOpen, BetaClosed);

    vector<int> RowThetaVec = AlphaClosed;
    RowThetaVec.insert(RowThetaVec.end(), BetaClosed.begin(), BetaClosed.end());    
    for (int j = AlphaClosed.size(); j < AlphaClosed.size()+BetaClosed.size(); j++)
        RowThetaVec[j] += norbs;
    Eigen::Map<VectorXi> RowTheta(&RowThetaVec[0], RowThetaVec.size());
    //cout << d << endl << endl;
    //cout << RowAlpha << endl << endl;
    //cout << RowBeta << endl << endl; 
    //RowTheta << RowAlpha, RowBeta;
    int nalpha = AlphaClosed.size();
    int nbeta = BetaClosed.size();



    //expected to execute for i=0 only
    for (int i = 0; i < w.getDeterminants().size(); i++)
    {
      MatrixXd theta;
      MatrixXd GHFInvCurrent;

      //Generate the alpha and beta strings for the wavefunction determinant
      vector<int> alphaRef, betaRef;
      w.getDeterminants()[i].getAlphaBeta(alphaRef, betaRef);
      for (int j = 0; j < betaRef.size(); j++)
          betaRef[j] += norbs;
      alphaRef.insert(alphaRef.end(), betaRef.begin(), betaRef.end());    
      Eigen::Map<VectorXi> ColTheta(&alphaRef[0], alphaRef.size());

      if (i == 0)
      //if (true)
      {
        //cout << w.getGHFOrbs() << endl << endl;
        igl::slice(w.getGHFOrbs(), RowTheta, ColTheta, theta); 
        Eigen::FullPivLU<MatrixXd> lua(theta);
        if (lua.isInvertible() || !check)
        {
          GHFInv = lua.inverse();
          GHFInvCurrent = GHFInv;
          GHFDet[i] = lua.determinant();
        }
        else
        {
          cout << RowTheta << endl
               << endl;
          cout << ColTheta << endl
               << endl;
          cout << theta << endl
               << endl;
          cout << lua.determinant() << endl;
          cout << lua.inverse() << endl;
          cout << "overlap with theta determinant " << d << " not invertible" << endl;
          exit(0);
        }

      }
      //cout << alphainvCurrent<<endl<<endl;
      //cout << betainvCurrent<<endl<<endl;

      AlphaTable[i] = MatrixXd::Zero(AlphaOpen.size(), AlphaClosed.size()); //k-N x N
      MatrixXd GHFOpenAlpha, ThetaInvAlphaSlice;
      VectorXi colSliceIndices = VectorXi::LinSpaced(AlphaClosed.size(), 0, AlphaClosed.size()-1);
      VectorXi rowSliceIndices = VectorXi::LinSpaced(RowThetaVec.size(), 0, RowThetaVec.size()-1);
      Eigen::Map<VectorXi> RowAlphaOpen(&AlphaOpen[0], AlphaOpen.size());
      //cout << RowAlphaOpen << endl << endl;
      igl::slice(w.getGHFOrbs(), RowAlphaOpen, ColTheta, GHFOpenAlpha);
      //cout << theta << endl << endl;
      //cout << GHFInvCurrent << endl << endl;
      //cout << GHFOpenAlpha << endl << endl;
      //cout << RowTheta << endl << endl;
      //cout << colSliceIndices << endl << endl;
      igl::slice(GHFInvCurrent, rowSliceIndices, colSliceIndices, ThetaInvAlphaSlice);
      //cout << ThetaInvAlphaSlice << endl << endl;
      AlphaTable[i] = GHFOpenAlpha * ThetaInvAlphaSlice;

      BetaTable[i] = MatrixXd::Zero(BetaOpen.size(), BetaClosed.size()); //k-N x N
      MatrixXd GHFOpenBeta, ThetaInvBetaSlice;
      colSliceIndices = VectorXi::LinSpaced(BetaClosed.size(), AlphaClosed.size(), BetaClosed.size()+AlphaClosed.size()-1);
      Eigen::VectorXi RowBetaOpen = VectorXi::Zero(BetaOpen.size());
      for (int j = 0; j < BetaOpen.size(); j++)
          RowBetaOpen[j] = BetaOpen[j]+norbs;
      igl::slice(w.getGHFOrbs(), RowBetaOpen, ColTheta, GHFOpenBeta);
      igl::slice(GHFInvCurrent, rowSliceIndices, colSliceIndices, ThetaInvBetaSlice);
      BetaTable[i] = GHFOpenBeta * ThetaInvBetaSlice;
      //cout << theta << endl << endl;
      //cout << GHFInvCurrent << endl << endl;
      //cout << GHFOpenBeta << endl << endl;
      //cout << rowSliceIndices << endl << endl;
      //cout << colSliceIndices << endl << endl;
      //cout << ThetaInvBetaSlice << endl << endl;
    }
    //exit(0);
  }

  template <typename Wfn>
  double getDetOverlap(Wfn &w)
  {
    double ovlp = 0.0;
    for (int i = 0; i < GHFDet.size(); i++)
    {
      ovlp += w.getciExpansion()[i] * GHFDet[i];
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
      p *= d.parityA(a, i);

    double detFactorNum = 0.0;
    double detFactorDen = 0.0;
    for (int i = 0; i < w.getDeterminants().size(); i++)
    {
      double factor = AlphaTable[i](tableIndexa, tableIndexi);
      detFactorNum += w.getciExpansion()[i] * factor * GHFDet[i];
      detFactorDen += w.getciExpansion()[i] * GHFDet[i];
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
      p *= d.parityB(a, i);

    double detFactorNum = 0.0;
    double detFactorDen = 0.0;
    for (int i = 0; i < w.getDeterminants().size(); i++)
    {
      double factor = BetaTable[i](tableIndexa, tableIndexi);
      detFactorNum += w.getciExpansion()[i] * factor * GHFDet[i];
      detFactorDen += w.getciExpansion()[i] * GHFDet[i];
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
      detFactorNum += w.getciExpansion()[i] * factor * GHFDet[i];
      detFactorDen += w.getciExpansion()[i] * GHFDet[i];
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
      detFactorNum += w.getciExpansion()[i] * factor * GHFDet[i];
      detFactorDen += w.getciExpansion()[i] * GHFDet[i];
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
      detFactorNum += w.getciExpansion()[I] * factor * GHFDet[I];
      detFactorDen += w.getciExpansion()[I] * GHFDet[I];
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
        p *= dcopy.parityA(aArray[i], iArray[i]);

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
        p *= dcopy.parityB(aArray[i], iArray[i]);

      dcopy.setoccB(iArray[i], false);
      dcopy.setoccB(aArray[i], true);
    }

    double cpsFactor = 1.0;

    return p * cpsFactor * localDet.determinant();
  }

  template <typename Wfn>
  void makeTables(Wfn &w)
  {
    int norbs = Determinant::norbs;
    vector<int> alphaRef0, betaRef0;
    w.getDeterminants()[0].getAlphaBeta(alphaRef0, betaRef0);
    for (int j = 0; j < betaRef0.size(); j++)
        betaRef0[j] += norbs;
    alphaRef0.insert(alphaRef0.end(), betaRef0.begin(), betaRef0.end());    
    Eigen::Map<VectorXi> ColTheta(&alphaRef0[0], alphaRef0.size());
  
    vector<int> RowThetaVec = AlphaClosed;
    RowThetaVec.insert(RowThetaVec.end(), BetaClosed.begin(), BetaClosed.end());    
    for (int j = AlphaClosed.size(); j < AlphaClosed.size()+BetaClosed.size(); j++)
        RowThetaVec[j] += norbs;
    Eigen::Map<VectorXi> RowTheta(&RowThetaVec[0], RowThetaVec.size());

    MatrixXd GHFOpenAlpha, ThetaInvAlphaSlice;
    VectorXi colSliceIndices = VectorXi::LinSpaced(AlphaClosed.size(), 0, AlphaClosed.size()-1);
    VectorXi rowSliceIndices = VectorXi::LinSpaced(RowThetaVec.size(), 0, RowThetaVec.size()-1);
    Eigen::Map<VectorXi> RowAlphaOpen(&AlphaOpen[0], AlphaOpen.size());
    igl::slice(w.getGHFOrbs(), RowAlphaOpen, ColTheta, GHFOpenAlpha);
    igl::slice(GHFInv, rowSliceIndices, colSliceIndices, ThetaInvAlphaSlice);
    AlphaTable[0] = GHFOpenAlpha * ThetaInvAlphaSlice;

    MatrixXd GHFOpenBeta, ThetaInvBetaSlice;
    colSliceIndices = VectorXi::LinSpaced(BetaClosed.size(), AlphaClosed.size(), BetaClosed.size()+AlphaClosed.size()-1);
    Eigen::VectorXi RowBetaOpen = VectorXi::Zero(BetaOpen.size());
    for (int j = 0; j < BetaOpen.size(); j++)
        RowBetaOpen[j] = BetaOpen[j]+norbs;
    igl::slice(w.getGHFOrbs(), RowBetaOpen, ColTheta, GHFOpenBeta);
    igl::slice(GHFInv, rowSliceIndices, colSliceIndices, ThetaInvBetaSlice);
    BetaTable[0] = GHFOpenBeta * ThetaInvBetaSlice;
 
  }

  template <typename Wfn>
  void updateA(int i, int a, Wfn &w)
  {

    double p = 1.0;
    p *= d.parityA(a, i);

    int tableIndexi = std::lower_bound(AlphaClosed.begin(), AlphaClosed.end(), i) - AlphaClosed.begin();
    int tableIndexa = std::lower_bound(AlphaOpen.begin(), AlphaOpen.end(), a) - AlphaOpen.begin();

    int norbs = Determinant::norbs;
    int nalpha = AlphaClosed.size();
    int nbeta = BetaClosed.size();
    vector<int> creA(nalpha, -1), desA(nalpha, -1), creB(nbeta, -1), desB(nbeta, -1);

    vector<int> alphaRef0, betaRef0;
    w.getDeterminants()[0].getAlphaBeta(alphaRef0, betaRef0);
    for (int j = 0; j < betaRef0.size(); j++)
        betaRef0[j] += norbs;
    alphaRef0.insert(alphaRef0.end(), betaRef0.begin(), betaRef0.end());    
    Eigen::Map<VectorXi> ColTheta(&alphaRef0[0], alphaRef0.size());
    vector<int> cre(1, a), des(1, i);
    MatrixXd GHFInvOld = GHFInv;
    double GHFDetOld = GHFDet[0];
    vector<int> RowIn(AlphaClosed.size()+BetaClosed.size(),0);
    for(int x=0; x<AlphaClosed.size(); x++)
        RowIn[x] = AlphaClosed[x];
    for(int x=0; x<BetaClosed.size(); x++)
        RowIn[x+AlphaClosed.size()] = norbs + BetaClosed[x];
    calculateInverseDeterminantWithRowChange(GHFInvOld, GHFDetOld, GHFInv,
                                             GHFDet[0], cre, des, ColTheta, RowIn, w.getGHFOrbs());
    GHFDet[0] *= p;

    d.setoccA(i, false);
    d.setoccA(a, true);
    AlphaOpen.clear();
    AlphaClosed.clear();
    BetaOpen.clear();
    BetaClosed.clear();
    d.getOpenClosedAlphaBeta(AlphaOpen, AlphaClosed, BetaOpen, BetaClosed);
    
    makeTables(w);

  }

  template <typename Wfn>
  void updateA(int i, int j, int a, int b, Wfn &w)
  {

    double p = 1.0;
    Determinant dcopy = d;
    p *= dcopy.parityA(a, i);
    dcopy.setoccA(i, false);
    dcopy.setoccA(a, true);
    p *= dcopy.parityA(b, j);
    dcopy.setoccA(j, false);
    dcopy.setoccA(b, true);

    int norbs = Determinant::norbs;
    int nalpha = AlphaClosed.size();
    int nbeta = BetaClosed.size();
    vector<int> creA(nalpha, -1), desA(nalpha, -1), creB(nbeta, -1), desB(nbeta, -1);

    vector<int> alphaRef0, betaRef0;
    w.getDeterminants()[0].getAlphaBeta(alphaRef0, betaRef0);
    for (int j = 0; j < betaRef0.size(); j++)
        betaRef0[j] += norbs;
    alphaRef0.insert(alphaRef0.end(), betaRef0.begin(), betaRef0.end());    
    Eigen::Map<VectorXi> ColTheta(&alphaRef0[0], alphaRef0.size());
    vector<int> cre(2, a), des(2, i);
    cre[1] = b;
    des[1] = j;
    MatrixXd GHFInvOld = GHFInv;
    double GHFDetOld = GHFDet[0];
    vector<int> RowIn(AlphaClosed.size()+BetaClosed.size(),0);
    for(int x=0; x<AlphaClosed.size(); x++)
        RowIn[x] = AlphaClosed[x];
    for(int x=0; x<BetaClosed.size(); x++)
        RowIn[x+AlphaClosed.size()] = norbs + BetaClosed[x];
    calculateInverseDeterminantWithRowChange(GHFInvOld, GHFDetOld, GHFInv,
                                             GHFDet[0], cre, des, ColTheta, RowIn, w.getGHFOrbs());
    GHFDet[0] *= p;

    d = dcopy;
    AlphaOpen.clear();
    AlphaClosed.clear();
    BetaOpen.clear();
    BetaClosed.clear();
    d.getOpenClosedAlphaBeta(AlphaOpen, AlphaClosed, BetaOpen, BetaClosed);
    
    makeTables(w);

  }

  template <typename Wfn>
  void updateB(int i, int a, Wfn &w)
  {

    double p = d.parityB(a, i);

    int tableIndexi = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), i) - BetaClosed.begin();
    int tableIndexa = std::lower_bound(BetaOpen.begin(), BetaOpen.end(), a) - BetaOpen.begin();

    int norbs = Determinant::norbs;
    int nalpha = AlphaClosed.size();
    int nbeta = BetaClosed.size();
    vector<int> creA(nalpha, -1), desA(nalpha, -1), creB(nbeta, -1), desB(nbeta, -1);

    vector<int> alphaRef0, betaRef0;
    w.getDeterminants()[0].getAlphaBeta(alphaRef0, betaRef0);
    for (int j = 0; j < betaRef0.size(); j++)
        betaRef0[j] += norbs;
    alphaRef0.insert(alphaRef0.end(), betaRef0.begin(), betaRef0.end());    
    Eigen::Map<VectorXi> ColTheta(&alphaRef0[0], alphaRef0.size());
    vector<int> cre(1, a+norbs), des(1, i+norbs);
    MatrixXd GHFInvOld = GHFInv;
    double GHFDetOld = GHFDet[0];
    vector<int> RowIn(AlphaClosed.size()+BetaClosed.size(),0);
    for(int x=0; x<AlphaClosed.size(); x++)
        RowIn[x] = AlphaClosed[x];
    for(int x=0; x<BetaClosed.size(); x++)
        RowIn[x+AlphaClosed.size()] = norbs + BetaClosed[x];
    calculateInverseDeterminantWithRowChange(GHFInvOld, GHFDetOld, GHFInv,
                                             GHFDet[0], cre, des, ColTheta, RowIn, w.getGHFOrbs());
    GHFDet[0] *= p;
    
    d.setoccB(i, false);
    d.setoccB(a, true);
    AlphaOpen.clear();
    AlphaClosed.clear();
    BetaOpen.clear();
    BetaClosed.clear();
    d.getOpenClosedAlphaBeta(AlphaOpen, AlphaClosed, BetaOpen, BetaClosed);

    makeTables(w);

  }

  template <typename Wfn>
  void updateB(int i, int j, int a, int b, Wfn &w)
  {

    double p = 1.0;
    Determinant dcopy = d;
    p *= dcopy.parityB(a, i);
    dcopy.setoccB(i, false);
    dcopy.setoccB(a, true);
    p *= dcopy.parityB(b, j);
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
    for (int j = 0; j < betaRef0.size(); j++)
        betaRef0[j] += norbs;
    alphaRef0.insert(alphaRef0.end(), betaRef0.begin(), betaRef0.end());    
    Eigen::Map<VectorXi> ColTheta(&alphaRef0[0], alphaRef0.size());
    vector<int> cre(2, a+norbs), des(2, i+norbs);
    cre[1] = b+norbs;
    des[1] = j+norbs;
    MatrixXd GHFInvOld = GHFInv;
    double GHFDetOld = GHFDet[0];
    vector<int> RowIn(AlphaClosed.size()+BetaClosed.size(),0);
    for(int x=0; x<AlphaClosed.size(); x++)
        RowIn[x] = AlphaClosed[x];
    for(int x=0; x<BetaClosed.size(); x++)
        RowIn[x+AlphaClosed.size()] = norbs + BetaClosed[x];
    calculateInverseDeterminantWithRowChange(GHFInvOld, GHFDetOld, GHFInv,
                                             GHFDet[0], cre, des, ColTheta, RowIn, w.getGHFOrbs());
    GHFDet[0] *= p;

    d = dcopy;
    AlphaOpen.clear();
    AlphaClosed.clear();
    BetaOpen.clear();
    BetaClosed.clear();
    d.getOpenClosedAlphaBeta(AlphaOpen, AlphaClosed, BetaOpen, BetaClosed);

    makeTables(w);

  }

  bool operator<(const GHFWalker &w) const
  {
    return d < w.d;
  }

  bool operator==(const GHFWalker &w) const
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
    GHFWalker &walk = *this;

    int K = 0;
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
              grad(numJastrowVariables + w.getciExpansion().size() + 2*k*norbs + l) += w.getciExpansion()[det] * walk.GHFInv(L, K) * walk.GHFDet[det] / detovlp;
              //grad(w.getNumJastrowVariables() + w.getciExpansion().size() + k*norbs+l) += walk.alphainv(L, KA);
              L++;
            }
          }
          for (int l = 0; l < norbs; l++)
          {
            if (ddet.getoccB(l))
            {
              grad(numJastrowVariables + w.getciExpansion().size() + 2*k*norbs+ norbs + l) += w.getciExpansion()[det] * walk.GHFInv(L, K) * walk.GHFDet[det] / detovlp;
              //grad(w.getNumJastrowVariables() + w.getciExpansion().size() + k*norbs+l) += walk.alphainv(L, KA);
              L++;
            }
          }
        }
        K++;
      }
    }
    for (int k = 0; k < norbs; k++)
    { //walker indices on the row
      if (walk.d.getoccB(k))
      {

        for (int det = 0; det < w.getDeterminants().size(); det++)
        {
          Determinant ddet = w.getDeterminants()[det];
          int L = 0;
          for (int l = 0; l < norbs; l++)
          {
            if (ddet.getoccA(l))
            {
              grad(numJastrowVariables + w.getciExpansion().size() + 2*norbs*norbs +  2*k*norbs + l) += w.getciExpansion()[det] * walk.GHFDet[det] * walk.GHFInv(L, K) / detovlp;
              L++;
            }
          }
          for (int l = 0; l < norbs; l++)
          {
            if (ddet.getoccB(l))
            {
              grad(numJastrowVariables + w.getciExpansion().size() + 2*norbs*norbs +  2*k*norbs + norbs + l) += w.getciExpansion()[det] * walk.GHFDet[det] * walk.GHFInv(L, K) / detovlp;
              L++;
            }
          }
        }
        K++;
      }
    }
  }
};

#endif
