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

#include "HFWalker.h"
#include "Slater.h"
#include "ShermanMorrisonWoodbury.h"
#include "input.h"
#include "global.h"

enum HartreeFock {Restricted, UnRestricted, Generalized};

using namespace Eigen;

HFWalkerHelper::HFWalkerHelper(const Slater &w, const Determinant &d) 
{
  hftype = w.hftype;
 
  //fill the spin strings for the walker and the zeroth reference det
  fillOpenClosedOrbs(d);
  closedOrbsRef[0].clear();
  closedOrbsRef[1].clear();
  w.getDeterminants()[0].getClosedAlphaBeta(closedOrbsRef[0], closedOrbsRef[1]);
  
  rTable.resize(w.getNumOfDets());
  thetaDet.resize(w.getNumOfDets());
  
  if (hftype == Generalized) {
    initInvDetsTablesGhf(w);
  }
  else {
    initInvDetsTables(w);
  }
}

void HFWalkerHelper::fillOpenClosedOrbs(const Determinant &d)
{
  openOrbs[0].clear();
  openOrbs[1].clear();
  closedOrbs[0].clear();
  closedOrbs[1].clear();
  d.getOpenClosedAlphaBeta(openOrbs[0], closedOrbs[0], openOrbs[1], closedOrbs[1]);
}

void HFWalkerHelper::makeTable(const Slater &w, const MatrixXd& inv, const Eigen::Map<VectorXi>& colClosed, int detIndex, bool sz)
{
  Map<VectorXi> rowOpen(&openOrbs[sz][0], openOrbs[sz].size());
  rTable[detIndex][sz] = MatrixXd::Zero(openOrbs[sz].size(), closedOrbs[sz].size()); 
  MatrixXd HfopenTheta;
  igl::slice(w.getHforbs(sz), rowOpen, colClosed, HfopenTheta);
  rTable[detIndex][sz] = HfopenTheta * inv;
}

void HFWalkerHelper::calcOtherDetsTables(const Slater& w, bool sz)
{
  Eigen::Map<VectorXi> rowClosed(&closedOrbs[sz][0], closedOrbs[sz].size());
  vector<int> cre(closedOrbs[sz].size(), -1), des(closedOrbs[sz].size(), -1);
  for (int x = 1; x < w.getNumOfDets(); x++) {
    MatrixXd invCurrent;
    vector<int> ref;
    w.getDeterminants()[x].getClosed(sz, ref);
    Eigen::Map<VectorXi> colClosed(&ref[0], ref.size());
    getDifferenceInOccupation(w.getDeterminants()[x], w.getDeterminants()[0], cre, des, sz);
    double parity = w.getDeterminants()[0].parity(cre, des, sz);
    calculateInverseDeterminantWithColumnChange(thetaInv[sz], thetaDet[0][sz], invCurrent, thetaDet[x][sz], cre, des, rowClosed, closedOrbsRef[sz], w.getHforbs(sz));
    thetaDet[x][sz] *= parity;
    makeTable(w, invCurrent, colClosed, x, sz);
  }
}

//commenting out calcotherdetstables, uncomment for multidet
void HFWalkerHelper::initInvDetsTables(const Slater &w)
{
  for (int sz = 0; sz < 2; sz++) {
    Eigen::Map<VectorXi> rowClosed(&closedOrbs[sz][0], closedOrbs[sz].size());
    Eigen::Map<VectorXi> colClosed(&closedOrbsRef[sz][0], closedOrbsRef[sz].size());
    MatrixXd theta;
    igl::slice(w.getHforbs(sz), rowClosed, colClosed, theta); 
    Eigen::FullPivLU<MatrixXd> lua(theta);
    if (lua.isInvertible()) {
      thetaInv[sz] = lua.inverse();
      thetaDet[0][sz] = lua.determinant();
    }
    else {
      cout << sz << " overlap with determinant not invertible" << endl;
      exit(0);
    }
    makeTable(w, thetaInv[sz], colClosed, 0, sz);
    //calcOtherDetsTables(w, sz);
  }
}

void HFWalkerHelper::concatenateGhf(const vector<int>& v1, const vector<int>& v2, vector<int>& result) const
{
  int norbs = Determinant::norbs;
  result.clear();
  result = v1;
  result.insert(result.end(), v2.begin(), v2.end());    
  for (int j = v1.size(); j < v1.size() + v2.size(); j++)
      result[j] += norbs;
}

void HFWalkerHelper::makeTableGhf(const Slater &w, const Eigen::Map<VectorXi>& colTheta)
{
  int norbs = Determinant::norbs;
  
  rTable[0][0] = MatrixXd::Zero(openOrbs[0].size(), closedOrbs[0].size()); 
  MatrixXd ghfOpenAlpha; 
  Eigen::Map<VectorXi> rowAlphaOpen(&openOrbs[0][0], openOrbs[0].size());
  igl::slice(w.getHforbs(), rowAlphaOpen, colTheta, ghfOpenAlpha);
  rTable[0][0] = ghfOpenAlpha * thetaInv[0].block(0, 0, ghfOpenAlpha.cols(), closedOrbs[0].size());

  rTable[0][1] = MatrixXd::Zero(openOrbs[1].size(), closedOrbs[1].size()); 
  MatrixXd ghfOpenBeta;
  Eigen::VectorXi rowBetaOpen = VectorXi::Zero(openOrbs[1].size());
  for (int j = 0; j < openOrbs[1].size(); j++)
    rowBetaOpen[j] = openOrbs[1][j] + norbs;
  igl::slice(w.getHforbs(), rowBetaOpen, colTheta, ghfOpenBeta);
  rTable[0][1] = ghfOpenBeta * thetaInv[0].block(0, closedOrbs[0].size(), ghfOpenBeta.cols(), closedOrbs[1].size());
}

void HFWalkerHelper::initInvDetsTablesGhf(const Slater &w)
{
  vector<int> workingVec0, workingVec1;
  concatenateGhf(closedOrbs[0], closedOrbs[1], workingVec0);
  Eigen::Map<VectorXi> rowTheta(&workingVec0[0], workingVec0.size());
  concatenateGhf(closedOrbsRef[0], closedOrbsRef[1], workingVec1);
  Eigen::Map<VectorXi> colTheta(&workingVec1[0], workingVec1.size());
  
  MatrixXd theta;
  igl::slice(w.getHforbs(), rowTheta, colTheta, theta); 
  Eigen::FullPivLU<MatrixXd> lua(theta);
  if (lua.isInvertible()) {
    thetaInv[0] = lua.inverse();
    thetaDet[0][0] = lua.determinant();
  }
  else {
    Eigen::Map<VectorXi> v1(&closedOrbs[0][0], closedOrbs[0].size());
    Eigen::Map<VectorXi> v2(&closedOrbs[1][0], closedOrbs[1].size());
    cout << "alphaClosed\n" << v1 << endl << endl;
    cout << "betaClosed\n" << v2 << endl << endl;
    cout << "col\n" << colTheta << endl << endl;
    cout << theta << endl << endl;
    cout << "overlap with theta determinant not invertible" << endl;
    exit(0);
  }
  thetaDet[0][1] = 1.;
  makeTableGhf(w, colTheta);
}

//commenting out calcotherdetstables, uncomment for multidet
void HFWalkerHelper::excitationUpdate(const Slater &w, vector<int>& cre, vector<int>& des, bool sz, double parity, const Determinant& excitedDet)
{
  MatrixXd invOld = thetaInv[sz];
  double detOld = thetaDet[0][sz];
  Eigen::Map<Eigen::VectorXi> colClosed(&closedOrbsRef[sz][0], closedOrbsRef[sz].size());
  //cout << "colClosed\n" << colClosed << endl << endl;
  //cout << "cre " << cre[0] << " des " << des[0] << endl << endl;
  calculateInverseDeterminantWithRowChange(invOld, detOld, thetaInv[sz], thetaDet[0][sz], cre, des, colClosed, closedOrbs[sz], w.getHforbs(sz));
  thetaDet[0][sz] *= parity;
  fillOpenClosedOrbs(excitedDet);
  makeTable(w, thetaInv[sz], colClosed, 0, sz);
  //calcOtherDetsTables(w, sz);
}

void HFWalkerHelper::excitationUpdateGhf(const Slater &w, vector<int>& cre, vector<int>& des, bool sz, double parity, const Determinant& excitedDet)
{
  vector<int> colVec;
  concatenateGhf(closedOrbsRef[0], closedOrbsRef[1], colVec);
  Eigen::Map<VectorXi> colTheta(&colVec[0], colVec.size());
  vector<int> rowIn;
  concatenateGhf(closedOrbs[0], closedOrbs[1], rowIn);
  MatrixXd invOld = thetaInv[0];
  double detOld = thetaDet[0][0];
  calculateInverseDeterminantWithRowChange(invOld, detOld, thetaInv[0], thetaDet[0][0], cre, des, colTheta, rowIn, w.getHforbs());
  thetaDet[0][0] *= parity;
  fillOpenClosedOrbs(excitedDet);
  makeTableGhf(w, colTheta);
}

void HFWalkerHelper::getRelIndices(int i, int &relI, int a, int &relA, bool sz) const 
{
  //relI = std::lower_bound(closedOrbs[sz].begin(), closedOrbs[sz].end(), i) - closedOrbs[sz].begin();
  //relA = std::lower_bound(openOrbs[sz].begin(), openOrbs[sz].end(), a) - openOrbs[sz].begin();
  relI = std::search_n(closedOrbs[sz].begin(), closedOrbs[sz].end(), 1, i) - closedOrbs[sz].begin();
  relA = std::search_n(openOrbs[sz].begin(), openOrbs[sz].end(), 1, a) - openOrbs[sz].begin();
}

//HFWalker::HFWalker(Determinant &pd) : d(pd){};

HFWalker::HFWalker(const Slater &w) 
{
  initDet(w.getHforbsA(), w.getHforbsB());
  helper = HFWalkerHelper(w, d);
}

HFWalker::HFWalker(const Slater &w, const Determinant &pd) : d(pd), helper(w, pd) {}; 

void HFWalker::readBestDeterminant(Determinant& d) const 
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
void HFWalker::guessBestDeterminant(Determinant& d, const Eigen::MatrixXd& HforbsA, const Eigen::MatrixXd& HforbsB) const 
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

void HFWalker::initDet(const MatrixXd& HforbsA, const MatrixXd& HforbsB) 
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

double HFWalker::getDetOverlap(const Slater &w) const
{
  double ovlp = 0.0;
  for (int i = 0; i < helper.thetaDet.size(); i++) {
    ovlp += w.getciExpansion()[i] * helper.thetaDet[i][0] * helper.thetaDet[i][1];
  }
  return ovlp;
}

double HFWalker::getDetFactor(int i, int a, const Slater &w) const 
{
  if (i % 2 == 0)
    return getDetFactor(i / 2, a / 2, 0, w);
  else                                   
    return getDetFactor(i / 2, a / 2, 1, w);
}

double HFWalker::getDetFactor(int I, int J, int A, int B, const Slater &w) const 
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

double HFWalker::getDetFactor(int i, int a, bool sz, const Slater &w) const
{
  int tableIndexi, tableIndexa;
  helper.getRelIndices(i, tableIndexi, a, tableIndexa, sz); 

  double detFactorNum = 0.0;
  double detFactorDen = 0.0;
  for (int j = 0; j < w.getDeterminants().size(); j++)
  {
    double factor = helper.rTable[j][sz](tableIndexa, tableIndexi);
    detFactorNum += w.getciExpansion()[j] * factor * helper.thetaDet[j][0] * helper.thetaDet[j][1];
    detFactorDen += w.getciExpansion()[j] * helper.thetaDet[j][0] * helper.thetaDet[j][1];
  }
  return detFactorNum / detFactorDen;
}

double HFWalker::getDetFactor(int i, int j, int a, int b, bool sz1, bool sz2, const Slater &w) const
{
  int tableIndexi, tableIndexa, tableIndexj, tableIndexb;
  helper.getRelIndices(i, tableIndexi, a, tableIndexa, sz1); 
  helper.getRelIndices(j, tableIndexj, b, tableIndexb, sz2) ;

  double detFactorNum = 0.0;
  double detFactorDen = 0.0;
  for (int j = 0; j < w.getDeterminants().size(); j++)
  {
    double factor;
    if (sz1 == sz2)
      factor = helper.rTable[j][sz1](tableIndexa, tableIndexi) * helper.rTable[j][sz1](tableIndexb, tableIndexj) 
          - helper.rTable[j][sz1](tableIndexb, tableIndexi) *helper.rTable[j][sz1](tableIndexa, tableIndexj);
    else
      factor = helper.rTable[j][sz1](tableIndexa, tableIndexi) * helper.rTable[j][sz2](tableIndexb, tableIndexj);
    detFactorNum += w.getciExpansion()[j] * factor * helper.thetaDet[j][0] * helper.thetaDet[j][1];
    detFactorDen += w.getciExpansion()[j] * helper.thetaDet[j][0] * helper.thetaDet[j][1];
  }
  return detFactorNum / detFactorDen;
}

void HFWalker::update(int i, int a, bool sz, const Slater &w)
{
  double p = 1.0;
  p *= d.parity(a, i, sz);
  d.setocc(i, sz, false);
  d.setocc(a, sz, true);
  if (helper.hftype == Generalized) {
    int norbs = Determinant::norbs;
    vector<int> cre{ a + sz * norbs }, des{ i + sz * norbs };
    helper.excitationUpdateGhf(w, cre, des, sz, p, d);
  }
  else
  {
    vector<int> cre{ a }, des{ i };
    helper.excitationUpdate(w, cre, des, sz, p, d);
  }
}

void HFWalker::update(int i, int j, int a, int b, bool sz, const Slater &w)
{
  double p = 1.0;
  Determinant dcopy = d;
  p *= d.parity(a, i, sz);
  d.setocc(i, sz, false);
  d.setocc(a, sz, true);
  p *= d.parity(b, j, sz);
  d.setocc(j, sz, false);
  d.setocc(b, sz, true);
  if (helper.hftype == Generalized) {
    int norbs = Determinant::norbs;
    vector<int> cre{ a + sz * norbs, b + sz * norbs }, des{ i + sz * norbs, j + sz * norbs };
    helper.excitationUpdateGhf(w, cre, des, sz, p, d);
  }
  else {
    vector<int> cre{ a, b }, des{ i, j };
    helper.excitationUpdate(w, cre, des, sz, p, d);
  }
}

void HFWalker::updateWalker(const Slater& w, int ex1, int ex2)
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

void HFWalker::exciteWalker(const Slater& w, int excite1, int excite2, int norbs)
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

void HFWalker::OverlapWithGradient(const Slater &w, Eigen::VectorXd &grad, double detovlp) const
{
  int norbs = Determinant::norbs;
  Determinant walkerDet = d;

  //K and L are relative row and col indices
  int KA = 0, KB = 0;
  for (int k = 0; k < norbs; k++) { //walker indices on the row
    if (walkerDet.getoccA(k)) {
      for (int det = 0; det < w.getDeterminants().size(); det++) {
        Determinant refDet = w.getDeterminants()[det];
        int L = 0;
        for (int l = 0; l < norbs; l++) {
          if (refDet.getoccA(l)) {
            grad(k * norbs + l) += w.getciExpansion()[det] * helper.thetaInv[0](L, KA) * helper.thetaDet[det][0] * helper.thetaDet[det][1] / detovlp;
            L++;
          }
        }
      }
      KA++;
    }
    if (walkerDet.getoccB(k)) {
      for (int det = 0; det < w.getDeterminants().size(); det++) {
        Determinant refDet = w.getDeterminants()[det];
        int L = 0;
        for (int l = 0; l < norbs; l++) {
          if (refDet.getoccB(l)) {
            if (helper.hftype == UnRestricted)
              grad(norbs * norbs + k * norbs + l) += w.getciExpansion()[det] * helper.thetaInv[1](L, KB) * helper.thetaDet[det][0] * helper.thetaDet[det][1] / detovlp;
            else
              grad(k * norbs + l) += w.getciExpansion()[det] * helper.thetaInv[1](L, KB) * helper.thetaDet[det][0] * helper.thetaDet[det][1] / detovlp;
            L++;
          }
        }
      }
      KB++;
    }
  }
}

void HFWalker::OverlapWithGradientGhf(const Slater &w, Eigen::VectorXd &grad, double detovlp) const
{
  int norbs = Determinant::norbs;
  Determinant walkerDet = d;
  Determinant refDet = w.getDeterminants()[0];

  //K and L are relative row and col indices
  int K = 0;
  for (int k = 0; k < norbs; k++) { //walker indices on the row
    if (walkerDet.getoccA(k)) {
      int L = 0;
      for (int l = 0; l < norbs; l++) {
        if (refDet.getoccA(l)) {
          grad(2 * k * norbs + l) += helper.thetaInv[0](L, K) * helper.thetaDet[0][0] / detovlp;
          L++;
        }
      }
      for (int l = 0; l < norbs; l++) {
        if (refDet.getoccB(l)) {
          grad(2 * k * norbs + norbs + l) += helper.thetaInv[0](L, K) * helper.thetaDet[0][0] / detovlp;
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
          grad(2 * norbs * norbs +  2 * k * norbs + l) += helper.thetaDet[0][0] * helper.thetaInv[0](L, K) / detovlp;
          L++;
        }
      }
      for (int l = 0; l < norbs; l++) {
        if (refDet.getoccB(l)) {
          grad(2 * norbs * norbs +  2 * k * norbs + norbs + l) += helper.thetaDet[0][0] * helper.thetaInv[0](L, K) / detovlp;
          L++;
        }
      }
      K++;
    }
  }
}

ostream& operator<<(ostream& os, const HFWalker& walk) {
  cout << walk.d << endl << endl;
  cout << "alphaTable\n" << walk.helper.rTable[0][0] << endl << endl;
  cout << "betaTable\n" << walk.helper.rTable[0][1] << endl << endl;
  cout << "dets\n" << walk.helper.thetaDet[0][0] << "  " << walk.helper.thetaDet[0][1] << endl << endl;
  cout << "alphaInv\n" << walk.helper.thetaInv[0] << endl << endl;
  cout << "betaInv\n" << walk.helper.thetaInv[1] << endl << endl;
}

/*bool HFWalker::makeMove(CPSSlater &w)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));

  int norbs = Determinant::norbs,
      nalpha = Determinant::nalpha,
      nbeta = Determinant::nbeta;

  //pick a random occupied orbital
  int i = floor(random() * (nalpha + nbeta));
  if (i < nalpha)
  {
    int a = floor(random() * (norbs - nalpha));
    int I = AlphaClosed[i];
    int A = AlphaOpen[a];
    double detfactor = getDetFactorA(I, A, w);
    //cout << i<<"   "<<a<<"   "<<detfactor<<endl;
    if (pow(detfactor, 2) > random())
    {
      updateA(I, A, w);
      return true;
    }
  }
  else
  {
    i = i - nalpha;
    int a = floor(random() * (norbs - nbeta));
    int I = BetaClosed[i];
    int A = BetaOpen[a];
    double detfactor = getDetFactorB(I, A, w);
    //cout << i<<"   "<<a<<"   "<<detfactor<<endl;
    if (pow(detfactor, 2) > random())
    {
      updateB(I, A, w);
      return true;
    }
  }

  return false;
}

bool HFWalker::makeMovePropPsi(CPSSlater &w)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));

  int norbs = Determinant::norbs,
      nalpha = Determinant::nalpha,
      nbeta = Determinant::nbeta;

  //pick a random occupied orbital
  int i = floor(random() * (nalpha + nbeta));
  if (i < nalpha)
  {
    int a = floor(random() * (norbs - nalpha));
    int I = AlphaClosed[i];
    int A = AlphaOpen[a];
    double detfactor = getDetFactorA(I, A, w);

    if (abs(detfactor) > random())
    {
      updateA(I, A, w);
      return true;
    }
  }
  else
  {
    i = i - nalpha;
    int a = floor(random() * (norbs - nbeta));
    int I = BetaClosed[i];
    int A = BetaOpen[a];
    double detfactor = getDetFactorB(I, A, w);

    if (abs(detfactor) > random())
    {
      updateB(I, A, w);
      return true;
    }
  }

  return false;
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
      p *= dcopy.parityB(aArray[i], iArray[i]);

    dcopy.setoccB(iArray[i], false);
    dcopy.setoccB(aArray[i], true);
  }

  double cpsFactor = 1.0;

  return p * cpsFactor * localDet.determinant();
}
*/
