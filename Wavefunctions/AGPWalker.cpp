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

#include <boost/archive/binary_iarchive.hpp>

#include "AGPWalker.h"
#include "AGP.h"
#include "ShermanMorrisonWoodbury.h"
#include "input.h"
#include "global.h"


using namespace Eigen;

AGPWalkerHelper::AGPWalkerHelper(const AGP &w, const Determinant &d) 
{
  //fill the spin strings for the walker and the zeroth reference det
  fillOpenClosedOrbs(d);
  initInvDetsTables(w);
}

void AGPWalkerHelper::fillOpenClosedOrbs(const Determinant &d)
{
  openOrbs[0].clear();
  openOrbs[1].clear();
  closedOrbs[0].clear();
  closedOrbs[1].clear();
  d.getOpenClosedAlphaBeta(openOrbs[0], closedOrbs[0], openOrbs[1], closedOrbs[1]);
}

void AGPWalkerHelper::makeTables(const AGP &w)
{
  Map<VectorXi> rowOpen(&openOrbs[0][0], openOrbs[0].size());
  Map<VectorXi> colClosed(&closedOrbs[1][0], closedOrbs[1].size());
  MatrixXd openThetaAlpha; 
  igl::slice(w.getPairMat(), rowOpen, colClosed, openThetaAlpha);
  rTable[0] = openThetaAlpha * thetaInv; 
  
  Map<VectorXi> rowClosed(&closedOrbs[0][0], closedOrbs[0].size());
  Map<VectorXi> colOpen(&openOrbs[1][0], openOrbs[1].size());
  MatrixXd openThetaBeta; 
  igl::slice(w.getPairMat(), rowClosed, colOpen, openThetaBeta);
  MatrixXd betaTableTranspose = thetaInv * openThetaBeta; 
  rTable[1] = betaTableTranspose.transpose(); 

  MatrixXd openTheta;
  igl::slice(w.getPairMat(), rowOpen, colOpen, openTheta);
  MatrixXd rtc = rTable[0] * openThetaBeta;
  rTable[2] = openTheta - rtc;
}

void AGPWalkerHelper::initInvDetsTables(const AGP &w)
{
  Eigen::Map<VectorXi> rowClosed(&closedOrbs[0][0], closedOrbs[0].size());
  Eigen::Map<VectorXi> colClosed(&closedOrbs[1][0], closedOrbs[1].size());
  MatrixXd theta;
  igl::slice(w.getPairMat(), rowClosed, colClosed, theta); 
  Eigen::FullPivLU<MatrixXd> lua(theta);
  if (lua.isInvertible()) {
    thetaInv = lua.inverse();
    thetaDet = lua.determinant();
  }
  else {
    cout << "pairMat\n" << w.getPairMat() << endl << endl;
    cout << "theta\n" << theta << endl << endl;
    cout << "rowClosed\n" << rowClosed << endl << endl;
    cout << "colClosed\n" << colClosed << endl << endl;
    cout << " overlap with determinant not invertible" << endl;
    exit(0);
  }
  makeTables(w);
}

void AGPWalkerHelper::excitationUpdate(const AGP &w, vector<int>& cre, vector<int>& des, bool sz, double parity, const Determinant& excitedDet)
{
  MatrixXd invOld = thetaInv;
  double detOld = thetaDet;
  if (sz == 0) {
    Eigen::Map<Eigen::VectorXi> colClosed(&closedOrbs[1][0], closedOrbs[1].size());
    calculateInverseDeterminantWithRowChange(invOld, detOld, thetaInv, thetaDet, cre, des, colClosed, closedOrbs[0], w.getPairMat());
  }
  if (sz == 1) {
    Eigen::Map<Eigen::VectorXi> rowClosed(&closedOrbs[0][0], closedOrbs[0].size());
    calculateInverseDeterminantWithColumnChange(invOld, detOld, thetaInv, thetaDet, cre, des, rowClosed, closedOrbs[1], w.getPairMat());
  }
  thetaDet *= parity;
  fillOpenClosedOrbs(excitedDet);
  makeTables(w);
}

void AGPWalkerHelper::getRelIndices(int i, int &relI, int a, int &relA, bool sz) const 
{
  relI = std::search_n(closedOrbs[sz].begin(), closedOrbs[sz].end(), 1, i) - closedOrbs[sz].begin();
  relA = std::search_n(openOrbs[sz].begin(), openOrbs[sz].end(), 1, a) - openOrbs[sz].begin();
}

AGPWalker::AGPWalker(const AGP &w) 
{
  initDet(w.getPairMat());
  helper = AGPWalkerHelper(w, d);
}

AGPWalker::AGPWalker(const AGP &w, const Determinant &pd) : d(pd), helper(w, pd) {}; 

void AGPWalker::readBestDeterminant(Determinant& d) const 
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

void AGPWalker::guessBestDeterminant(Determinant& d, const Eigen::MatrixXd& pairMat) const 
{
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;
  for (int i = 0; i < nalpha; i++)
    d.setoccA(i, true);
  for (int i = 0; i < nbeta; i++)
    d.setoccB(i, true);
}

void AGPWalker::initDet(const MatrixXd& pairMat) 
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

double AGPWalker::getDetOverlap(const AGP &w) const
{
  return helper.thetaDet;
}

double AGPWalker::getDetFactor(int i, int a, const AGP &w) const 
{
  if (i % 2 == 0)
    return getDetFactor(i / 2, a / 2, 0, w);
  else                                   
    return getDetFactor(i / 2, a / 2, 1, w);
}

double AGPWalker::getDetFactor(int I, int J, int A, int B, const AGP &w) const 
{
  if (I % 2 == J % 2 && I % 2 == 0)
    return getDetFactor(I / 2, J / 2, A / 2, B / 2, 0, 0, w);
  else if (I % 2 == J % 2 && I % 2 == 1)                  
    return getDetFactor(I / 2, J / 2, A / 2, B / 2, 1, 1, w);
  else if (I % 2 != J % 2 && I % 2 == 0)                  
    return getDetFactor(I / 2, J / 2, A / 2, B / 2, 0, 1, w);
  else                                                    
    return getDetFactor(I / 2, J / 2, A / 2, B / 2, 1, 0, w);
}

double AGPWalker::getDetFactor(int i, int a, bool sz, const AGP &w) const
{
  int tableIndexi, tableIndexa;
  helper.getRelIndices(i, tableIndexi, a, tableIndexa, sz); 
  return helper.rTable[sz](tableIndexa, tableIndexi);
}

double AGPWalker::getDetFactor(int i, int j, int a, int b, bool sz1, bool sz2, const AGP &w) const
{
  int tableIndexi, tableIndexa, tableIndexj, tableIndexb;
  helper.getRelIndices(i, tableIndexi, a, tableIndexa, sz1); 
  helper.getRelIndices(j, tableIndexj, b, tableIndexb, sz2) ;

  double factor;
  if (sz1 == sz2)
    factor = helper.rTable[sz1](tableIndexa, tableIndexi) * helper.rTable[sz1](tableIndexb, tableIndexj) 
        - helper.rTable[sz1](tableIndexb, tableIndexi) *helper.rTable[sz1](tableIndexa, tableIndexj);
  else
    if (sz1 == 0) {
      factor = helper.rTable[sz1](tableIndexa, tableIndexi) * helper.rTable[sz2](tableIndexb, tableIndexj) 
      + helper.thetaInv(tableIndexj, tableIndexi) * helper.rTable[2](tableIndexa, tableIndexb);
    }
    else {
      factor = helper.rTable[sz1](tableIndexa, tableIndexi) * helper.rTable[sz2](tableIndexb, tableIndexj) 
      + helper.thetaInv(tableIndexi, tableIndexj) * helper.rTable[2](tableIndexb, tableIndexa);
    }
  return factor;
}

void AGPWalker::update(int i, int a, bool sz, const AGP &w)
{
  double p = 1.0;
  p *= d.parity(a, i, sz);
  d.setocc(i, sz, false);
  d.setocc(a, sz, true);
  vector<int> cre{ a }, des{ i };
  helper.excitationUpdate(w, cre, des, sz, p, d);
}

void AGPWalker::update(int i, int j, int a, int b, bool sz, const AGP &w)
{
  double p = 1.0;
  Determinant dcopy = d;
  p *= d.parity(a, i, sz);
  d.setocc(i, sz, false);
  d.setocc(a, sz, true);
  p *= d.parity(b, j, sz);
  d.setocc(j, sz, false);
  d.setocc(b, sz, true);
  vector<int> cre{ a, b }, des{ i, j };
  helper.excitationUpdate(w, cre, des, sz, p, d);
}

void AGPWalker::updateWalker(const AGP& w, int ex1, int ex2)
{
  int norbs = Determinant::norbs;
  int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
  int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
  if (I % 2 == J % 2 && ex2 != 0) {
    if (I % 2 == 1) {
      update(I / 2, J / 2, A / 2, B / 2, 1, w);
    }
    else {
      update(I / 2, J / 2, A / 2, B / 2, 0, w);
    }
  }
  else {
    if (I % 2 == 0)
      update(I / 2, A / 2, 0, w);
    else
      update(I / 2, A / 2, 1, w);

    if (ex2 != 0) {
      if (J % 2 == 1) {
        update(J / 2, B / 2, 1, w);
      }
      else {
        update(J / 2, B / 2, 0, w);
      }
    }
  }
}

void AGPWalker::exciteWalker(const AGP& w, int excite1, int excite2, int norbs)
{
  int I1 = excite1 / (2 * norbs), A1 = excite1 % (2 * norbs);

  if (I1 % 2 == 0)
    update(I1 / 2, A1 / 2, 0, w);
  else
    update(I1 / 2, A1 / 2, 1, w);

  if (excite2 != 0) {
    int I2 = excite2 / (2 * norbs), A2 = excite2 % (2 * norbs);
    if (I2 % 2 == 0)
      update(I2 / 2, A2 / 2, 0, w);
    else
      update(I2 / 2, A2 / 2, 1, w);
  }
}

void AGPWalker::OverlapWithGradient(const AGP &w, Eigen::VectorBlock<VectorXd> &grad, double detovlp) const
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
          grad(k * norbs + l) += helper.thetaInv(L, K) ;
          L++;
        }
      }
      K++;
    }
  }
}

ostream& operator<<(ostream& os, const AGPWalker& walk) {
  cout << walk.d << endl << endl;
  cout << "alphaTable\n" << walk.helper.rTable[0] << endl << endl;
  cout << "betaTable\n" << walk.helper.rTable[1] << endl << endl;
  cout << "thirdTable\n" << walk.helper.rTable[2] << endl << endl;
  cout << "dets\n" << walk.helper.thetaDet << endl << endl;
  cout << "thetaInv\n" << walk.helper.thetaInv << endl << endl;
}
