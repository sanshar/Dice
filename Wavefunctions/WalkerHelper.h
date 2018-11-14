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
#ifndef WalkerHelper_HEADER_H
#define WalkerHelper_HEADER_H

#include "Determinants.h"
#include "igl/slice.h"
#include "igl/slice_into.h"
#include "ShermanMorrisonWoodbury.h"
#include "Slater.h"
#include "AGP.h"
#include "Pfaffian.h"
#include "CPS.h"
#include "Jastrow.h"

template<typename Reference>
class WalkerHelper {
};

template<>
class WalkerHelper<Slater>
{

 public:
  HartreeFock hftype;                           //hftype same as that in slater
  array<MatrixXd, 2> thetaInv;          //inverse of the theta matrix
  vector<array<double, 2>> thetaDet;    //determinant of the theta matrix, vector for multidet
  array<vector<int>, 2> openOrbs;       //set of open orbitals in the walker
  array<vector<int>, 2> closedOrbs;     //set of closed orbitals in the walker
  array<vector<int>, 2> closedOrbsRef;  //set of closed orbitals in the reference (zeroth det)
  vector<array<MatrixXd, 2>> rTable;    //table used for efficiently, vector for multidet

  WalkerHelper() {};
  
  WalkerHelper(const Slater &w, const Determinant &d) 
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

  void fillOpenClosedOrbs(const Determinant &d)
  {
    openOrbs[0].clear();
    openOrbs[1].clear();
    closedOrbs[0].clear();
    closedOrbs[1].clear();
    d.getOpenClosedAlphaBeta(openOrbs[0], closedOrbs[0], openOrbs[1], closedOrbs[1]);
    Map<VectorXi> rowOpen(&openOrbs[0][0], openOrbs[0].size());
  }

  void makeTable(const Slater &w, const MatrixXd& inv, const Eigen::Map<VectorXi>& colClosed, int detIndex, bool sz)
  {
    Map<VectorXi> rowOpen(&openOrbs[sz][0], openOrbs[sz].size());
    rTable[detIndex][sz] = MatrixXd::Zero(openOrbs[sz].size(), closedOrbs[sz].size()); 
    MatrixXd HfopenTheta;
    igl::slice(w.getHforbs(sz), rowOpen, colClosed, HfopenTheta);
    rTable[detIndex][sz] = HfopenTheta * inv;
  }

  void calcOtherDetsTables(const Slater& w, bool sz)
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
  void initInvDetsTables(const Slater &w)
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

  void concatenateGhf(const vector<int>& v1, const vector<int>& v2, vector<int>& result) const
  {
    int norbs = Determinant::norbs;
    result.clear();
    result = v1;
    result.insert(result.end(), v2.begin(), v2.end());    
    for (int j = v1.size(); j < v1.size() + v2.size(); j++)
      result[j] += norbs;
  }

  void makeTableGhf(const Slater &w, const Eigen::Map<VectorXi>& colTheta)
  {
    rTable[0][0] = MatrixXd::Zero(openOrbs[0].size() + openOrbs[1].size(), closedOrbs[0].size() + closedOrbs[1].size()); 
    MatrixXd ghfOpen;
    vector<int> rowVec;
    concatenateGhf(openOrbs[0], openOrbs[1], rowVec);
    Eigen::Map<VectorXi> rowOpen(&rowVec[0], rowVec.size());
    igl::slice(w.getHforbs(), rowOpen, colTheta, ghfOpen);
    rTable[0][0] = ghfOpen * thetaInv[0];
    rTable[0][1] = rTable[0][0];
  }

  void initInvDetsTablesGhf(const Slater &w)
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
  void excitationUpdate(const Slater &w, vector<int>& cre, vector<int>& des, bool sz, double parity, const Determinant& excitedDet)
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

  void excitationUpdateGhf(const Slater &w, vector<int>& cre, vector<int>& des, bool sz, double parity, const Determinant& excitedDet)
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

  void getRelIndices(int i, int &relI, int a, int &relA, bool sz) const 
  {
    //relI = std::lower_bound(closedOrbs[sz].begin(), closedOrbs[sz].end(), i) - closedOrbs[sz].begin();
    //relA = std::lower_bound(openOrbs[sz].begin(), openOrbs[sz].end(), a) - openOrbs[sz].begin();
    int factor = 0;
    if (hftype == 2 && sz != 0) factor = 1;
    relI = std::search_n(closedOrbs[sz].begin(), closedOrbs[sz].end(), 1, i) - closedOrbs[sz].begin() + factor * closedOrbs[0].size();
    relA = std::search_n(openOrbs[sz].begin(), openOrbs[sz].end(), 1, a) - openOrbs[sz].begin() + factor * openOrbs[0].size();
  }

};

template<>
class WalkerHelper<AGP>
{
  public:
    MatrixXd thetaInv;                    //inverse of the theta matrix
    double thetaDet;                      //determinant of the theta matrix
    array<vector<int>, 2> openOrbs;       //set of open orbitals in the walker
    array<vector<int>, 2> closedOrbs;     //set of closed orbitals in the walker
    array<MatrixXd, 3> rTable;            //table used for efficiently

    WalkerHelper() {};
    
    WalkerHelper(const AGP &w, const Determinant &d) 
    {
      //fill the spin strings for the walker and the zeroth reference det
      fillOpenClosedOrbs(d);
      initInvDetsTables(w);
    }
    
    void fillOpenClosedOrbs(const Determinant &d)
    {
      openOrbs[0].clear();
      openOrbs[1].clear();
      closedOrbs[0].clear();
      closedOrbs[1].clear();
      d.getOpenClosedAlphaBeta(openOrbs[0], closedOrbs[0], openOrbs[1], closedOrbs[1]);
    }
    
    void makeTables(const AGP &w)
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
    
    void initInvDetsTables(const AGP &w)
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
    
    void excitationUpdate(const AGP &w, vector<int>& cre, vector<int>& des, bool sz, double parity, const Determinant& excitedDet)
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
    
    void getRelIndices(int i, int &relI, int a, int &relA, bool sz) const 
    {
      relI = std::search_n(closedOrbs[sz].begin(), closedOrbs[sz].end(), 1, i) - closedOrbs[sz].begin();
      relA = std::search_n(openOrbs[sz].begin(), openOrbs[sz].end(), 1, a) - openOrbs[sz].begin();
    }

};

template<>
class WalkerHelper<Pfaffian>
{
  public:
    MatrixXd thetaInv;                    //inverse of the theta matrix
    double thetaPfaff;                      //determinant of the theta matrix
    array<vector<int>, 2> openOrbs;       //set of open orbitals in the walker
    array<vector<int>, 2> closedOrbs;     //set of closed orbitals in the walker
    //array<MatrixXd, 2> rTable;            //table used for efficiently
    MatrixXd fMat;

    WalkerHelper() {};
    
    WalkerHelper(const Pfaffian &w, const Determinant &d) 
    {
      fillOpenClosedOrbs(d);
      int nopen = openOrbs[0].size() + openOrbs[1].size();
      int nclosed = closedOrbs[0].size() + closedOrbs[1].size();
      fMat = MatrixXd::Zero(nopen * nclosed, nclosed);
      //rTable[0] = MatrixXd::Zero(nopen * nclosed, nclosed);
      //rTable[1] = MatrixXd::Zero(nopen * nclosed, nopen * nclosed);
      initInvDetsTables(w);
    }
    
    void fillOpenClosedOrbs(const Determinant &d)
    {
      openOrbs[0].clear();
      openOrbs[1].clear();
      closedOrbs[0].clear();
      closedOrbs[1].clear();
      d.getOpenClosedAlphaBeta(openOrbs[0], closedOrbs[0], openOrbs[1], closedOrbs[1]);
    }
    
    void makeTables(const Pfaffian &w)
    {
      int norbs = Determinant::norbs;
      int nopen = openOrbs[0].size() + openOrbs[1].size();
      int nclosed = closedOrbs[0].size() + closedOrbs[1].size();
      Map<VectorXi> openAlpha(&openOrbs[0][0], openOrbs[0].size());
      Map<VectorXi> openBeta(&openOrbs[1][0], openOrbs[1].size());
      VectorXi open(nopen);
      open << openAlpha, (openBeta.array() + norbs).matrix();
      Map<VectorXi> closedAlpha(&closedOrbs[0][0], closedOrbs[0].size());
      Map<VectorXi> closedBeta(&closedOrbs[1][0], closedOrbs[1].size());
      VectorXi closed(nclosed);
      closed << closedAlpha, (closedBeta.array() + norbs).matrix();
       
      MatrixXd fRow;
      VectorXi rowSlice(1), colSlice(nclosed);
      for (int i = 0; i < closed.size(); i++) {
        for (int a = 0; a < open.size(); a++) {
          colSlice = closed;
          rowSlice[0] = open[a];
          colSlice[i] = open[a];
          igl::slice(w.getPairMat(), rowSlice, colSlice, fRow);
          fMat.block(i * open.size() + a, 0, 1, closed.size()) = fRow;
        }
      }
      
      //rTable[0] = fMat * thetaInv; 
      //rTable[1] = - rTable[0] * fMat.transpose(); 
    }
    
    void initInvDetsTables(const Pfaffian &w)
    {
      int norbs = Determinant::norbs;
      int nclosed = closedOrbs[0].size() + closedOrbs[1].size();
      Map<VectorXi> closedAlpha(&closedOrbs[0][0], closedOrbs[0].size());
      Map<VectorXi> closedBeta(&closedOrbs[1][0], closedOrbs[1].size());
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
    
    void excitationUpdate(const Pfaffian &w, int i, int a, bool sz, double parity, const Determinant& excitedDet)
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
      MatrixXd shuffledThetaInv = invOld - invOld * bMat * intermediate * bMat.transpose() * invOld; 
      
      closed[tableIndexi] = a + sz * norbs; 
      std::vector<int> order(closed.size());
      auto rcopy = closed;
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&rcopy](size_t i1, size_t i2) { return rcopy[i1] < rcopy[i2]; });
      Eigen::Map<Eigen::VectorXi> orderVec(&order[0], order.size());
      igl::slice(shuffledThetaInv, orderVec, orderVec, thetaInv);
      
      //thetaPfaff = thetaPfaff * rTable[0](tableIndexi * nopen + tableIndexa, tableIndexi);
      double pfaffRatio = fMat.row(tableIndexi * nopen + tableIndexa) * invOld.col(tableIndexi);
      thetaPfaff = thetaPfaff * pfaffRatio;
      thetaPfaff *= parity;
      fillOpenClosedOrbs(excitedDet);
      makeTables(w);
    }
    
    void getRelIndices(int i, int &relI, int a, int &relA, bool sz) const 
    {
      int factor = 0;
      if (sz != 0) factor = 1;
      relI = std::search_n(closedOrbs[sz].begin(), closedOrbs[sz].end(), 1, i) - closedOrbs[sz].begin() + factor * closedOrbs[0].size();
      relA = std::search_n(openOrbs[sz].begin(), openOrbs[sz].end(), 1, a) - openOrbs[sz].begin() + factor * openOrbs[0].size();
    }

};

template<>
class WalkerHelper<CPS>
{
 public:
  std::vector<double> intermediateForEachSpinOrb;
  std::vector<int> commonCorrelators;
  std::vector<std::vector<int> > twoSitesToCorrelator;

  WalkerHelper() {};
  WalkerHelper(const CPS& cps, const Determinant& d) {
    int norbs = Determinant::norbs;
    intermediateForEachSpinOrb.resize(norbs*2);
    updateHelper(cps, d);

    if (cps.twoSiteOrSmaller) {
      vector<int> initial(norbs, -1);
      twoSitesToCorrelator.resize(norbs, initial);
    
      int corrIndex = 0;
      for (const auto& corr : cps.cpsArray) {
        if (corr.asites.size() == 2) {
          int site1 = corr.asites[0], site2 = corr.asites[1];
          twoSitesToCorrelator[site1][site2] = corrIndex;
          twoSitesToCorrelator[site2][site1] = corrIndex;
        }
        corrIndex++;
      }

    }
  }

  void updateHelper(const CPS& cps, const Determinant& d) {
    int norbs = Determinant::norbs;

    for (int i=0; i<2*norbs; i++) {
      intermediateForEachSpinOrb[i] = 1.0;
    
      Determinant dcopy1 = d, dcopy2 = d;
      dcopy1.setocc(i, true);  //make sure this is occupied
      dcopy2.setocc(i, false); //make sure this is unoccupied
    
      const vector<int>& cpsContainingi = cps.mapFromOrbitalToCorrelator[i/2];
      for (const auto& j : cpsContainingi) {
        intermediateForEachSpinOrb[i] *= cps.cpsArray[j].OverlapRatio(dcopy1, dcopy2);
      }
    }
  }

  double OverlapRatio(int i, int a, const CPS& cps, const Determinant &dcopy, const Determinant &d) const
  {
    vector<int>& common = const_cast<vector<int>&>(commonCorrelators);
    common.resize(0);

    if (cps.twoSiteOrSmaller) {
      common.push_back(twoSitesToCorrelator[i/2][a/2]);
      if (common[0] == -1) common.resize(0);
    }
    else {
      set_intersection(cps.mapFromOrbitalToCorrelator[i/2].begin(),
                       cps.mapFromOrbitalToCorrelator[i/2].end(),
                       cps.mapFromOrbitalToCorrelator[a/2].begin(),
                       cps.mapFromOrbitalToCorrelator[a/2].end(),
                       back_inserter(common));
      sort(common.begin(), common.end() );
    }
  
    double ovlp = intermediateForEachSpinOrb[a]/intermediateForEachSpinOrb[i];
  
    Determinant dcopy1 = d, dcopy2 = d;
    dcopy1.setocc(i, false); dcopy2.setocc(a, true);

    int previ = -1;
    for (const auto& j : common) {
      ovlp *= cps.cpsArray[j].OverlapRatio(dcopy,d);
      ovlp *= cps.cpsArray[j].OverlapRatio(d, dcopy2);
      ovlp *= cps.cpsArray[j].OverlapRatio(d, dcopy1);
    }
    return ovlp;
  }


  double OverlapRatio(int i, int j, int a, int b, const CPS& cps, const Determinant &dcopy, const Determinant &d) const
  {
    if (!(cps.twoSiteOrSmaller && i/2!=j/2 && i/2!=a/2 && i/2!= b/2 && j/2 != a/2 && j/2 != b/2 && a/2!=b/2))
      return cps.OverlapRatio(i/2, j/2, a/2, b/2, dcopy, d);

    vector<int>& common = const_cast<vector<int>&>(commonCorrelators);
    common.resize(0);
  

    double ovlp = intermediateForEachSpinOrb[a]*intermediateForEachSpinOrb[b]
        /intermediateForEachSpinOrb[i]/intermediateForEachSpinOrb[j];

    Determinant dcopy1 = d, dcopy2 = d;

    {//i j
      dcopy1.setocc(i, false); dcopy2.setocc(j, false);

      int index = twoSitesToCorrelator[i/2][j/2];
      if (index != -1) {
        ovlp *= cps.cpsArray[index].OverlapRatio(dcopy, d);
        ovlp *= cps.cpsArray[index].OverlapRatio(d, dcopy2);
        ovlp *= cps.cpsArray[index].OverlapRatio(d, dcopy1);
      }
      dcopy1.setocc(i, true); dcopy2.setocc(j, true);
    }
    {//i a
      dcopy1.setocc(i, false); dcopy2.setocc(a, true);

      int index = twoSitesToCorrelator[i/2][a/2];
      if (index != -1) {
        ovlp *= cps.cpsArray[index].OverlapRatio(dcopy, d);
        ovlp *= cps.cpsArray[index].OverlapRatio(d, dcopy2);
        ovlp *= cps.cpsArray[index].OverlapRatio(d, dcopy1);
      }
      dcopy1.setocc(i, true); dcopy2.setocc(a, false);
    }
    {//i b
      dcopy1.setocc(i, false); dcopy2.setocc(b, true);

      int index = twoSitesToCorrelator[i/2][b/2];
      if (index != -1) {
      ovlp *= cps.cpsArray[index].OverlapRatio(dcopy, d);
      ovlp *= cps.cpsArray[index].OverlapRatio(d, dcopy2);
      ovlp *= cps.cpsArray[index].OverlapRatio(d, dcopy1);
      }
      dcopy1.setocc(i, true); dcopy2.setocc(b, false);
    }
    {//j a
      dcopy1.setocc(j, false); dcopy2.setocc(a, true);

      int index = twoSitesToCorrelator[j/2][a/2];
      if (index != -1) {
      ovlp *= cps.cpsArray[index].OverlapRatio(dcopy, d);
      ovlp *= cps.cpsArray[index].OverlapRatio(d, dcopy2);
      ovlp *= cps.cpsArray[index].OverlapRatio(d, dcopy1);
      }
      dcopy1.setocc(j, true); dcopy2.setocc(a, false);
    }
    {//j b
      dcopy1.setocc(j, false); dcopy2.setocc(b, true);

      int index = twoSitesToCorrelator[j/2][b/2];
      if (index != -1) {
      ovlp *= cps.cpsArray[index].OverlapRatio(dcopy, d);
      ovlp *= cps.cpsArray[index].OverlapRatio(d, dcopy2);
      ovlp *= cps.cpsArray[index].OverlapRatio(d, dcopy1);
      }
      dcopy1.setocc(j, true); dcopy2.setocc(b, false);
    }
    {//a b
      dcopy1.setocc(a, true); dcopy2.setocc(b, true);

      int index = twoSitesToCorrelator[a/2][b/2];
      if (index != -1) {
      ovlp *= cps.cpsArray[index].OverlapRatio(dcopy, d);
      ovlp *= cps.cpsArray[index].OverlapRatio(d, dcopy2);
      ovlp *= cps.cpsArray[index].OverlapRatio(d, dcopy1);
      }
      dcopy1.setocc(a, false); dcopy2.setocc(b, false);
    }
  
    return ovlp;
  }

};

template<>
class WalkerHelper<Jastrow>
{
 public:
  std::vector<double> intermediateForEachSpinOrb;

  WalkerHelper() {};
  WalkerHelper(const Jastrow& cps, const Determinant& d) {
    int norbs = Determinant::norbs;
    intermediateForEachSpinOrb.resize(norbs*2);
    updateHelper(cps, d);
  }

  void updateHelper(const Jastrow& cps, const Determinant& d) {
    int norbs = Determinant::norbs;

    vector<int> closed;
    vector<int> open;
    d.getOpenClosed(open, closed);
    
    
    for (int i=0; i<2*norbs; i++) {
      intermediateForEachSpinOrb[i] = cps(i,i);
      for (int j=0; j<closed.size(); j++)
        if (closed[j] != i)
          intermediateForEachSpinOrb[i] *= cps(i, closed[j]);
    }
  }

  double OverlapRatio(int i, int a, const Jastrow& cps,
                      const Determinant &dcopy, const Determinant &d) const
  {
    return intermediateForEachSpinOrb[a]/intermediateForEachSpinOrb[i]/cps(i,a);
  }
  
  double OverlapRatio(int i, int j, int a, int b, const Jastrow& cps,
                      const Determinant &dcopy, const Determinant &d) const
  {
    return intermediateForEachSpinOrb[a]*intermediateForEachSpinOrb[b]*cps(a,b)*cps(i,j)
        /intermediateForEachSpinOrb[i]/intermediateForEachSpinOrb[j]/
        cps(i,a)/cps(j,a)/cps(i,b)/cps(j,b);
  }
};  

#endif
