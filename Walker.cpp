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
#include "Walker.h"
#include "Wfn.h"
#include "integral.h"
#include "global.h"
#include "input.h"
#include <algorithm>
#include "igl/slice.h"
#include "igl/slice_into.h"
using namespace Eigen;


bool Walker::makeMove(CPSSlater& w) {
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));

  int norbs = Determinant::norbs,
      nalpha = MoDeterminant::nalpha,
      nbeta = MoDeterminant::nbeta;

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
    int a = floor( random()*(norbs-nbeta));
    int I = BetaClosed[i];
    int A = BetaOpen[a];
    double detfactor = getDetFactorB(I, A, w);
    //cout << i<<"   "<<a<<"   "<<detfactor<<endl;
    if ( pow(detfactor, 2) > random() ) {
      updateB(I, A, w);
      return true;
    }

  }

  return false;
}


bool Walker::makeMovePropPsi(CPSSlater& w) {
  auto random = std::bind(std::uniform_real_distribution<double>(0,1),
			  std::ref(generator));

  int norbs = MoDeterminant::norbs,
    nalpha = MoDeterminant::nalpha,
    nbeta  = MoDeterminant::nbeta;

  //pick a random occupied orbital
  int i = floor( random()*(nalpha+nbeta) );
  if (i < nalpha) {
    int a = floor(random()* (norbs-nalpha) );
    int I = AlphaClosed[i];
    int A = AlphaOpen[a];
    double detfactor = getDetFactorA(I, A, w);

    if ( abs(detfactor) > random() ) {
      updateA(I, A, w);
      return true;
    }

  }
  else {
    i = i - nalpha;
    int a = floor( random()*(norbs-nbeta));
    int I = BetaClosed[i];
    int A = BetaOpen[a];
    double detfactor = getDetFactorB(I, A, w);

    if ( abs(detfactor) > random() ) {
      updateB(I, A, w);
      return true;
    }

  }

  return false;
}

double Walker::getDetOverlap(CPSSlater &w)
{
  double ovlp = 0.0;
  for (int i = 0; i < alphaDet.size(); i++)
    ovlp += w.ciExpansion[i] * alphaDet[i] * betaDet[i];
  return ovlp;
}


void Walker::initUsingWave(CPSSlater& w, bool check) {

  AlphaTable.resize(w.determinants.size());
  BetaTable.resize(w.determinants.size());
  alphaDet.resize(w.determinants.size());
  betaDet.resize(w.determinants.size());
  
  d.getOpenClosedAlphaBeta(AlphaOpen, AlphaClosed, BetaOpen , BetaClosed );

  Eigen::Map<VectorXi> RowAlpha(&AlphaClosed[0], AlphaClosed.size());
  Eigen::Map<VectorXi> RowBeta (&BetaClosed[0] , BetaClosed.size());
  Eigen::Map<VectorXi> RowAlphaOpen(&AlphaOpen[0], AlphaOpen.size());
  Eigen::Map<VectorXi> RowBetaOpen (&BetaOpen[0] , BetaOpen.size());
  int norbs  = Determinant::norbs;
  int nalpha = AlphaClosed.size();
  int nbeta  = BetaClosed.size();


  for (int i=0; i<w.determinants.size(); i++) 
  {
    MatrixXd alpha, beta;

    //Generate the alpha and beta strings for the wavefunction determinant
    vector<int> alphaRef, betaRef;
    w.determinants[i].getAlphaBeta(alphaRef, betaRef);
    Eigen::Map<VectorXi> ColAlpha(&alphaRef[0], alphaRef.size());
    Eigen::Map<VectorXi> ColBeta (&betaRef[0],  betaRef.size());

    igl::slice(Hforbs, RowAlpha, ColAlpha, alpha); //alpha = Hforbs(Row, Col)
    Eigen::FullPivLU<MatrixXd> lua(alpha);
    if (lua.isInvertible() || !check)
    {
      alphainv = lua.inverse();
      alphaDet[i] = lua.determinant();
    }
    else
    {
      cout << "overlap with alpha determinant " << d << " not invertible" << endl;
      exit(0);
    }

    igl::slice(Hforbs, RowBeta, ColBeta, beta); //beta = Hforbs(Row, Col)
    Eigen::FullPivLU<MatrixXd> lub(beta);
    if (lub.isInvertible() || !check)
    {
      betainv = lub.inverse();
      betaDet[i] = lub.determinant();
    }
    else
    {
      cout << "overlap with beta determinant " << d << " not invertible" << endl;
      exit(0);
    }

    AlphaTable[i] = MatrixXd::Zero(AlphaOpen.size(), AlphaClosed.size()); //k-N x N  
    MatrixXd HfopenAlpha;
    igl::slice(Hforbs, RowAlphaOpen, ColAlpha, HfopenAlpha);
    AlphaTable[i] = HfopenAlpha*alphainv;

    BetaTable[i] = MatrixXd::Zero(BetaOpen.size(), BetaClosed.size()); //k-N x N  
    MatrixXd HfopenBeta;
    igl::slice(Hforbs, RowBetaOpen, ColBeta, HfopenBeta);
    BetaTable[i] = HfopenBeta*betainv;
  }
}

void   Walker::exciteWalker(CPSSlater& w, int excite1, int excite2, int norbs)
{
  int I1 = excite1/(2*norbs), A1= excite1%(2*norbs);


  if (I1%2 == 0) updateA(I1/2, A1/2, w);
  else           updateB(I1/2, A1/2, w);

  if (excite2 != 0) {
    int I2 = excite2/(2*norbs), A2= excite2%(2*norbs);
    if (I2%2 == 0) updateA(I2/2, A2/2, w);
    else           updateB(I2/2, A2/2, w);
  }


}

double Walker::getDetFactorA(int i, int a, CPSSlater &w, bool doparity)
{
  int tableIndexi = std::lower_bound(AlphaClosed.begin(), AlphaClosed.end(), i) - AlphaClosed.begin();
  int tableIndexa = std::lower_bound(AlphaOpen.begin(), AlphaOpen.end(), a) - AlphaOpen.begin();

  double p = 1.;
  if (doparity)
    d.parityA(a, i, p);

  double detFactorNum = 0.0;
  double detFactorDen = 0.0;
  for (int i = 0; i < w.determinants.size(); i++)
  {
    double factor = AlphaTable[i](tableIndexa, tableIndexi);
    detFactorNum += w.ciExpansion[i] * factor * alphaDet[i] * betaDet[i];
    detFactorDen += w.ciExpansion[i] * alphaDet[i] * betaDet[i];
  }

  return p * detFactorNum / detFactorDen;
}

double Walker::getDetFactorB(int i, int a, CPSSlater &w, bool doparity)
{
  int tableIndexi = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), i) - BetaClosed.begin();
  int tableIndexa = std::lower_bound(BetaOpen.begin(), BetaOpen.end(), a) - BetaOpen.begin();

  double p = 1.;
  if (doparity)
    d.parityB(a, i, p);

  double detFactorNum = 0.0;
  double detFactorDen = 0.0;
  for (int i = 0; i < w.determinants.size(); i++)
  {
    double factor = BetaTable[i](tableIndexa, tableIndexi);
    detFactorNum += w.ciExpansion[i] * factor * alphaDet[i] * betaDet[i];
    detFactorDen += w.ciExpansion[i] * alphaDet[i] * betaDet[i];
  }

  return p * detFactorNum / detFactorDen;
}

double Walker::getDetFactorA(int i, int j, int a, int b, CPSSlater &w, bool doparity)
{
  int tableIndexi = std::lower_bound(AlphaClosed.begin(), AlphaClosed.end(), i) - AlphaClosed.begin();
  int tableIndexa = std::lower_bound(AlphaOpen.begin(), AlphaOpen.end(), a) - AlphaOpen.begin();
  int tableIndexj = std::lower_bound(AlphaClosed.begin(), AlphaClosed.end(), j) - AlphaClosed.begin();
  int tableIndexb = std::lower_bound(AlphaOpen.begin(), AlphaOpen.end(), b) - AlphaOpen.begin();

  double detFactorNum = 0.0;
  double detFactorDen = 0.0;
  for (int i = 0; i < w.determinants.size(); i++)
  {
    double factor = (AlphaTable[i](tableIndexa, tableIndexi) * AlphaTable[i](tableIndexb, tableIndexj) - AlphaTable[i](tableIndexb, tableIndexi) * AlphaTable[i](tableIndexa, tableIndexj));
    detFactorNum += w.ciExpansion[i] * factor * alphaDet[i] * betaDet[i];
    detFactorDen += w.ciExpansion[i] * alphaDet[i] * betaDet[i];
  }
  return detFactorNum / detFactorDen;
}

double Walker::getDetFactorB(int i, int j, int a, int b, CPSSlater &w, bool doparity)
{
  int tableIndexi = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), i) - BetaClosed.begin();
  int tableIndexa = std::lower_bound(BetaOpen.begin(), BetaOpen.end(), a) - BetaOpen.begin();
  int tableIndexj = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), j) - BetaClosed.begin();
  int tableIndexb = std::lower_bound(BetaOpen.begin(), BetaOpen.end(), b) - BetaOpen.begin();

  double detFactorNum = 0.0;
  double detFactorDen = 0.0;
  for (int i = 0; i < w.determinants.size(); i++)
  {
    double factor = (BetaTable[i](tableIndexa, tableIndexi) * BetaTable[i](tableIndexb, tableIndexj) - BetaTable[i](tableIndexb, tableIndexi) * BetaTable[i](tableIndexa, tableIndexj));
    detFactorNum += w.ciExpansion[i] * factor * alphaDet[i] * betaDet[i];
    detFactorDen += w.ciExpansion[i] * alphaDet[i] * betaDet[i];
  }
  return detFactorNum / detFactorDen;
}

double Walker::getDetFactorAB(int i, int j, int a, int b, CPSSlater &w, bool doparity)
{
  int tableIndexi = std::lower_bound(AlphaClosed.begin(), AlphaClosed.end(), i) - AlphaClosed.begin();
  int tableIndexa = std::lower_bound(AlphaOpen.begin(), AlphaOpen.end(), a) - AlphaOpen.begin();
  int tableIndexj = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), j) - BetaClosed.begin();
  int tableIndexb = std::lower_bound(BetaOpen.begin(), BetaOpen.end(), b) - BetaOpen.begin();

  double detFactorNum = 0.0;
  double detFactorDen = 0.0;
  for (int i = 0; i < w.determinants.size(); i++)
  {
    double factor = AlphaTable[i](tableIndexa, tableIndexi) * BetaTable[i](tableIndexb, tableIndexj);
    detFactorNum += w.ciExpansion[i] * factor * alphaDet[i] * betaDet[i];
    detFactorDen += w.ciExpansion[i] * alphaDet[i] * betaDet[i];
  }
  return detFactorNum / detFactorDen;
}

double Walker::getDetFactorA(vector<int>& iArray, vector<int>& aArray, CPSSlater &w, bool doparity)
{
  MatrixXd localDet = MatrixXd::Zero(aArray.size(), iArray.size());
  for (int i=0; i<iArray.size(); i++)
  {
    int tableIndexi = std::lower_bound(AlphaClosed.begin(), AlphaClosed.end(), iArray[i]) - AlphaClosed.begin();
    for (int a=0; a<aArray.size(); a++)
    {
      int tableIndexa = std::lower_bound(AlphaOpen  .begin(), AlphaOpen  .end(), aArray[a]) - AlphaOpen.begin();
      localDet(i,a) = AlphaTable[0](tableIndexi, tableIndexa);
    }
  }

  double p = 1.;
  Determinant dcopy = d;
  for (int i=0; i<iArray.size(); i++) 
  {
    if (doparity)
      dcopy.parityA(aArray[i], iArray[i], p);

    dcopy.setoccA(iArray[i], false);
    dcopy.setoccA(aArray[i], true);
  }

  double cpsFactor = 1.0;

  return p * cpsFactor * localDet.determinant() ;
}

double Walker::getDetFactorB(vector<int>& iArray, vector<int>& aArray, CPSSlater &w, bool doparity)
{
  MatrixXd localDet = MatrixXd::Zero(aArray.size(), iArray.size());
  for (int i=0; i<iArray.size(); i++)
  {
    int tableIndexi = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), iArray[i]) - BetaClosed.begin();
    for (int a=0; a<aArray.size(); a++)
    {
      int tableIndexa = std::lower_bound(BetaOpen  .begin(), BetaOpen  .end(), aArray[a]) - BetaOpen.begin();
      localDet(i,a) = BetaTable[0](tableIndexi, tableIndexa);
    }
  }

  double p = 1.;
  Determinant dcopy = d;
  for (int i=0; i<iArray.size(); i++) 
  {
    if (doparity)
      dcopy.parityB(aArray[i], iArray[i], p);

    dcopy.setoccB(iArray[i], false);
    dcopy.setoccB(aArray[i], true);
  }

  double cpsFactor = 1.0;

  return p * cpsFactor * localDet.determinant() ;
}


void Walker::updateA(int i, int a, CPSSlater& w) {

  double p = 1.0;
  d.parityA(a, i, p);

  int tableIndexi = std::lower_bound(AlphaClosed.begin(), AlphaClosed.end(), i) - AlphaClosed.begin();
  int tableIndexa = std::lower_bound(AlphaOpen  .begin(), AlphaOpen  .end(), a) - AlphaOpen.begin();

  int norbs = Determinant::norbs;
  int nalpha = AlphaClosed.size();
  int nbeta = BetaClosed.size();

  for (int i=0; i<w.determinants.size(); i++)
  {
    //Generate the alpha and beta strings for the wavefunction determinant
    vector<int> alphaRef, betaRef;
    w.determinants[i].getAlphaBeta(alphaRef, betaRef);
    Eigen::Map<VectorXi> ColAlpha(&alphaRef[0], alphaRef.size());

    double alphaDetFactor = AlphaTable[i](tableIndexa, tableIndexi);
    alphaDet[i] *= alphaDetFactor*p;

    Eigen::MatrixXd Hfnarrow;
    igl::slice(Hforbs, VectorXi::LinSpaced(norbs, 0, norbs + 1), ColAlpha, Hfnarrow); //Hfnarrow = Hforbs(:, R)

    MatrixXd vtAinv = (Hfnarrow.row(a) - Hfnarrow.row(i)) * alphainv;

    MatrixXd alphainvWrongOrder = alphainv - (alphainv.col(tableIndexi) * vtAinv) / alphaDetFactor;

    AlphaClosed[tableIndexi] = a;
    AlphaOpen[tableIndexa]   = i;

    std::vector<int> order(AlphaClosed.size()), acopy = AlphaClosed;
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&acopy](size_t i1, size_t i2) { return acopy[i1] < acopy[i2]; });
    Eigen::Map<VectorXi> orderVec(&order[0], order.size());
    igl::slice(alphainvWrongOrder, VectorXi::LinSpaced(nalpha, 0, nalpha + 1), orderVec, alphainv);

    d.setoccA(i, false);
    d.setoccA(a, true );
    std::sort(AlphaClosed.begin(), AlphaClosed.end());
    std::sort(AlphaOpen  .begin(), AlphaOpen  .end());


    Eigen::Map<VectorXi> RowAlphaOpen(&AlphaOpen[0], AlphaOpen.size());
    MatrixXd HfopenAlpha;
    igl::slice(Hforbs, RowAlphaOpen, ColAlpha, HfopenAlpha);
    AlphaTable[i] = HfopenAlpha*alphainv;


  }

}


void Walker::updateB(int i, int a, CPSSlater& w) {

  double p = 1.0;
  d.parityB(a, i, p);

  int tableIndexi = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), i) - BetaClosed.begin();
  int tableIndexa = std::lower_bound(BetaOpen  .begin(), BetaOpen  .end(), a) - BetaOpen.begin();

  int norbs = Determinant::norbs;
  int nalpha = AlphaClosed.size();
  int nbeta = BetaClosed.size();

  for (int i=0; i<w.determinants.size(); i++)
  {
    //Generate the alpha and beta strings for the wavefunction determinant
    vector<int> alphaRef, betaRef;
    w.determinants[i].getAlphaBeta(alphaRef, betaRef);
    Eigen::Map<VectorXi> ColBeta(&betaRef[0], betaRef.size());

    double betaDetFactor = BetaTable[i](tableIndexa, tableIndexi);
    betaDet[i] *= betaDetFactor*p;

    Eigen::MatrixXd Hfnarrow;
    igl::slice(Hforbs, VectorXi::LinSpaced(norbs, 0, norbs + 1), ColBeta, Hfnarrow); //Hfnarrow = Hforbs(:, R)

    MatrixXd vtAinv = (Hfnarrow.row(a) - Hfnarrow.row(i)) * betainv;

    MatrixXd betainvWrongOrder = betainv - (betainv.col(tableIndexi) * vtAinv) / betaDetFactor;

    BetaClosed[tableIndexi] = a;
    BetaOpen[tableIndexa]   = i;

    std::vector<int> order(BetaClosed.size()), bcopy = BetaClosed;
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&bcopy](size_t i1, size_t i2) { return bcopy[i1] < bcopy[i2]; });
    Eigen::Map<VectorXi> orderVec(&order[0], order.size());
    igl::slice(betainvWrongOrder, VectorXi::LinSpaced(nbeta, 0, nbeta + 1), orderVec, betainv);

    d.setoccB(i, false);
    d.setoccB(a, true );
    std::sort(BetaClosed.begin(), BetaClosed.end());
    std::sort(BetaOpen  .begin(), BetaOpen  .end());


    Eigen::Map<VectorXi> RowBetaOpen(&BetaOpen[0], BetaOpen.size());
    MatrixXd HfopenBeta;
    igl::slice(Hforbs, RowBetaOpen, ColBeta, HfopenBeta);
    BetaTable[i] = HfopenBeta*betainv;


  }
}
