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

#include "PfaffianWalker.h"
#include "Pfaffian.h"
#include "ShermanMorrisonWoodbury.h"
#include "input.h"
#include "global.h"


using namespace Eigen;

PfaffianWalkerHelper::PfaffianWalkerHelper(const Pfaffian &w, const Determinant &d) 
{
  //fill the spin strings for the walker and the zeroth reference det
  //cout << "det    " << d << endl << endl;
  fillOpenClosedOrbs(d);
  initInvDetsTables(w);
}

void PfaffianWalkerHelper::fillOpenClosedOrbs(const Determinant &d)
{
  openOrbs[0].clear();
  openOrbs[1].clear();
  closedOrbs[0].clear();
  closedOrbs[1].clear();
  d.getOpenClosedAlphaBeta(openOrbs[0], closedOrbs[0], openOrbs[1], closedOrbs[1]);
}

void PfaffianWalkerHelper::makeTables(const Pfaffian &w)
{
  int norbs = Determinant::norbs;
  int nopen = openOrbs[0].size() + openOrbs[1].size();
  int nclosed = closedOrbs[0].size() + closedOrbs[1].size();
  Map<VectorXi> openAlpha(&openOrbs[0][0], openOrbs[0].size());
  Map<VectorXi> openBeta(&openOrbs[1][0], openOrbs[1].size());
  //openBeta = (openBeta.array() + norbs).matrix();
  VectorXi open(nopen);
  open << openAlpha, (openBeta.array() + norbs).matrix();
  //cout << "closed   " << closedOrbs[0][0] << "  " << closedOrbs[1][0] << endl << endl;
  //cout << "open   " << openOrbs[0][0] << "  " << openOrbs[1][0] << endl << endl;
  Map<VectorXi> closedAlpha(&closedOrbs[0][0], closedOrbs[0].size());
  Map<VectorXi> closedBeta(&closedOrbs[1][0], closedOrbs[1].size());
  //cout << "closedBeta before\n" << closedBeta << endl << endl;
  //closedBeta = (closedBeta.array() + norbs).matrix();
  //cout << "closedBeta after\n" << closedBeta << endl << endl;
  VectorXi closed(nclosed);
  closed << closedAlpha, (closedBeta.array() + norbs).matrix();
   
  fMat = MatrixXd::Zero(open.size() * closed.size(), closed.size());
  //cout << "open\n" << open << endl << endl; 
  //cout << "closed\n" << closed << endl << endl; 
  for (int i = 0; i < closed.size(); i++) {
    for (int a = 0; a < open.size(); a++) {
      MatrixXd fRow;
      VectorXi rowSlice(1);
      rowSlice[0] = open[a];
      VectorXi colSlice = closed;
      colSlice(i) = open[a];
      igl::slice(w.getPairMat(), rowSlice, colSlice, fRow);
      fMat.block(i * open.size() + a, 0, 1, closed.size()) = fRow;
      //cout << i << "   " << a << endl;
      //cout << "rowSlice\n" << rowSlice << endl << endl; 
      //cout << "colSlice\n" << colSlice << endl << endl; 
      //cout << "fRow\n" << fRow << endl << endl;
    }
  }
  
  rTable[0] = fMat * thetaInv; 
  rTable[1] = - rTable[0] * fMat.transpose(); 
}

void PfaffianWalkerHelper::initInvDetsTables(const Pfaffian &w)
{
  int norbs = Determinant::norbs;
  int nclosed = closedOrbs[0].size() + closedOrbs[1].size();
  Map<VectorXi> closedAlpha(&closedOrbs[0][0], closedOrbs[0].size());
  Map<VectorXi> closedBeta(&closedOrbs[1][0], closedOrbs[1].size());
  //closedBeta = (closedBeta.array() + norbs).matrix();
  VectorXi closed(nclosed);
  closed << closedAlpha, (closedBeta.array() + norbs).matrix();
  MatrixXd theta;
  igl::slice(w.getPairMat(), closed, closed, theta); 
  Eigen::FullPivLU<MatrixXd> lua(theta);
  if (lua.isInvertible()) {
    thetaInv = lua.inverse();
    thetaPfaff = calcPfaffian(theta);
  }
  else {
    cout << "pairMat\n" << w.getPairMat() << endl << endl;
    cout << "theta\n" << theta << endl << endl;
    cout << "colClosed\n" << closed << endl << endl;
    cout << " overlap with determinant not invertible" << endl;
    exit(0);
  }
  makeTables(w);
}

void PfaffianWalkerHelper::excitationUpdate(const Pfaffian &w, int i, int a, bool sz, double parity, const Determinant& excitedDet)
{
  int tableIndexi, tableIndexa;
  getRelIndices(i, tableIndexi, a, tableIndexa, sz); 
  int norbs = Determinant::norbs;
  int nopen = openOrbs[0].size() + openOrbs[1].size();
  int nclosed = closedOrbs[0].size() + closedOrbs[1].size();
  Map<VectorXi> closedAlpha(&closedOrbs[0][0], closedOrbs[0].size());
  Map<VectorXi> closedBeta(&closedOrbs[1][0], closedOrbs[1].size());
  VectorXi closed(nclosed);
  closed << closedAlpha, (closedBeta.array() + norbs).matrix();
  
  Matrix2d cInv;
  cInv << 0, -1,
         1, 0;
  MatrixXd bMat = MatrixXd::Zero(nclosed, 2);
  bMat(tableIndexi, 1) = 1;
  VectorXi colSlice(1);
  colSlice[0] = i + sz * norbs;
  MatrixXd thetaSlice;
  igl::slice(w.getPairMat(), closed, colSlice, thetaSlice);
  bMat.block(0, 0, nclosed, 1) = - fMat.transpose().block(0, tableIndexi * nopen + tableIndexa, nclosed, 1) - thetaSlice;
  MatrixXd invOld = thetaInv;
  MatrixXd intermediate = (cInv + bMat.transpose() * invOld * bMat).inverse();
  
  thetaInv = invOld - invOld * bMat * intermediate * bMat.transpose() * invOld; 
  thetaPfaff = rTable[0](tableIndexi * nopen + tableIndexa, tableIndexi);
  thetaPfaff *= parity;
  fillOpenClosedOrbs(excitedDet);
  makeTables(w);
}

void PfaffianWalkerHelper::getRelIndices(int i, int &relI, int a, int &relA, bool sz) const 
{
  int factor = 0;
  if (sz != 0) factor = 1;
  relI = std::search_n(closedOrbs[sz].begin(), closedOrbs[sz].end(), 1, i) - closedOrbs[sz].begin() + factor * closedOrbs[0].size();
  relA = std::search_n(openOrbs[sz].begin(), openOrbs[sz].end(), 1, a) - openOrbs[sz].begin() + factor * openOrbs[0].size();
}

PfaffianWalker::PfaffianWalker(const Pfaffian &w) 
{
  initDet(w.getPairMat());
  helper = PfaffianWalkerHelper(w, d);
}

PfaffianWalker::PfaffianWalker(const Pfaffian &w, const Determinant &pd) : d(pd), helper(w, pd) {}; 

void PfaffianWalker::readBestDeterminant(Determinant& d) const 
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

void PfaffianWalker::guessBestDeterminant(Determinant& d, const Eigen::MatrixXd& pairMat) const 
{
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;
  for (int i = 0; i < nalpha; i++)
    d.setoccA(i, true);
  for (int i = 0; i < nbeta; i++)
    d.setoccB(i, true);
}

void PfaffianWalker::initDet(const MatrixXd& pairMat) 
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

double PfaffianWalker::getDetOverlap(const Pfaffian &w) const
{
  return helper.thetaPfaff;
}

double PfaffianWalker::getDetFactor(int i, int a, const Pfaffian &w) const 
{
  if (i % 2 == 0)
    return getDetFactor(i / 2, a / 2, 0, w);
  else                                   
    return getDetFactor(i / 2, a / 2, 1, w);
}

double PfaffianWalker::getDetFactor(int I, int J, int A, int B, const Pfaffian &w) const 
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

double PfaffianWalker::getDetFactor(int i, int a, bool sz, const Pfaffian &w) const
{
  int nopen = helper.openOrbs[0].size() + helper.openOrbs[1].size();
  int tableIndexi, tableIndexa;
  helper.getRelIndices(i, tableIndexi, a, tableIndexa, sz); 
  return helper.rTable[0](tableIndexi * nopen + tableIndexa, tableIndexi);
}

double PfaffianWalker::getDetFactor(int i, int j, int a, int b, bool sz1, bool sz2, const Pfaffian &w) const
{
  int norbs = Determinant::norbs;
  int nopen = helper.openOrbs[0].size() + helper.openOrbs[1].size();
  int tableIndexi, tableIndexa, tableIndexj, tableIndexb;
  helper.getRelIndices(i, tableIndexi, a, tableIndexa, sz1); 
  helper.getRelIndices(j, tableIndexj, b, tableIndexb, sz2) ;
  //cout << "nopen  " << nopen << endl;
  //cout << "sz1  " << sz1 << "  ti  " << tableIndexi << "  ta  " << tableIndexa  << endl;
  //cout << "sz2  " << sz2 << "  tj  " << tableIndexj << "  tb  " << tableIndexb  << endl;
  double summand1, summand2, crossTerm;
  if (tableIndexi < tableIndexj) {
    crossTerm = (w.getPairMat())(b + sz2 * norbs, a + sz1 * norbs);
    summand1 = helper.rTable[0](tableIndexi * nopen + tableIndexa, tableIndexi) * helper.rTable[0](tableIndexj * nopen + tableIndexb, tableIndexj) 
        - helper.rTable[0](tableIndexj * nopen + tableIndexb, tableIndexi) * helper.rTable[0](tableIndexi * nopen + tableIndexa, tableIndexj);
    summand2 = helper.thetaInv(tableIndexi, tableIndexj) * (helper.rTable[1](tableIndexi * nopen + tableIndexa, tableIndexj * nopen + tableIndexb) + crossTerm);
  }
  else { 
    crossTerm = (w.getPairMat())(a + sz1 * norbs, b + sz2 * norbs);
    summand1 = helper.rTable[0](tableIndexj * nopen + tableIndexb, tableIndexj) * helper.rTable[0](tableIndexi * nopen + tableIndexa, tableIndexi) 
        - helper.rTable[0](tableIndexi * nopen + tableIndexa, tableIndexj) * helper.rTable[0](tableIndexj * nopen + tableIndexb, tableIndexi);
    summand2 = helper.thetaInv(tableIndexj, tableIndexi) * (helper.rTable[1](tableIndexj * nopen + tableIndexb, tableIndexi * nopen + tableIndexa) + crossTerm);
  }
  //cout << "double   " << crossTerm << "   " << summand1 << "  " << summand2 << endl; 
  return summand1 + summand2;
}

void PfaffianWalker::update(int i, int a, bool sz, const Pfaffian &w)
{
  double p = 1.0;
  p *= d.parity(a, i, sz);
  d.setocc(i, sz, false);
  d.setocc(a, sz, true);
  helper.excitationUpdate(w, i, a, sz, p, d);
}

void PfaffianWalker::updateWalker(const Pfaffian& w, int ex1, int ex2)
{
  int norbs = Determinant::norbs;
  int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
  int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
  
  if (I % 2 == 0)
    update(I / 2, A / 2, 0, w);
  else
    update(I / 2, A / 2, 1, w);

  if (ex2 != 0) {
    if (J % 2 == 1) 
      update(J / 2, B / 2, 1, w);
    else 
      update(J / 2, B / 2, 0, w);
  }
}

void PfaffianWalker::exciteWalker(const Pfaffian& w, int excite1, int excite2, int norbs)
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

void PfaffianWalker::OverlapWithGradient(const Pfaffian &w, Eigen::VectorBlock<VectorXd> &grad, double detovlp) const
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
          grad(2 * k * norbs + l) += helper.thetaInv(L, K) / 2;
          L++;
        }
      }
      for (int l = 0; l < norbs; l++) {
        if (walkerDet.getoccB(l)) {
          grad(2 * k * norbs + norbs + l) += helper.thetaInv(L, K) / 2;
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
          grad(2 * norbs * norbs + 2 * k * norbs + l) += helper.thetaInv(L, K) / 2;
          L++;
        }
      }
      for (int l = 0; l < norbs; l++) {
        if (walkerDet.getoccB(l)) {
          grad(2 * norbs * norbs + 2 * k * norbs + norbs + l) += helper.thetaInv(L, K) / 2;
          L++;
        }
      }
      K++;
    }
  }
}

ostream& operator<<(ostream& os, const PfaffianWalker& walk) {
  cout << walk.d << endl << endl;
  cout << "fMat\n" << walk.helper.fMat << endl << endl;
  cout << "fThetaInv\n" << walk.helper.rTable[0] << endl << endl;
  cout << "fThetaInvf\n" << walk.helper.rTable[1] << endl << endl;
  cout << "pfaff\n" << walk.helper.thetaPfaff << endl << endl;
  cout << "thetaInv\n" << walk.helper.thetaInv << endl << endl;
}
