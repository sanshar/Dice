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
  Map<VectorXi> rowOpen(&openOrbs[0][0], openOrbs[0].size());
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
