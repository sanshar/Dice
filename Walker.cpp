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



void Walker::initUsingWave(CPSSlater& w, bool check) {

  MatrixXd alpha, beta;
  w.getDetMatrix(d, alpha, beta);

  MatrixXd alphainv, betainv; 
  //Sometimes the determinant is so poor that
  //the inverse is illconditioned, so it is useful
  //to test this
  Eigen::FullPivLU<MatrixXd> lua(alpha);
  if (lua.isInvertible() || !check) {
    alphainv = lua.inverse();
    alphaDet = lua.determinant();
  }
  else {
    cout << "overlap with alpha determinant "<< d <<" no invertible"<<endl;
    cout << "rank of the matrix: "<<lua.rank()<<endl;
    EigenSolver<MatrixXx> eigensolver(alpha);
    cout << eigensolver.eigenvalues()<<endl;
    exit(0);
  }

  Eigen::FullPivLU<MatrixXd> lub(beta);
  if (lub.isInvertible() || !check) {
    betainv = lub.inverse();
    betaDet = lub.determinant();
  }
  else {
    cout << "overlap with beta determinant "<< d <<" no invertible"<<endl;
    cout << "rank of the matrix: "<<lub.rank()<<endl;
    EigenSolver<MatrixXx> eigensolver(beta);
    cout << eigensolver.eigenvalues()<<endl;
    exit(0);
  }


  d.getOpenClosedAlphaBeta(AlphaOpen, AlphaClosed,
                           BetaOpen , BetaClosed );

  //Generate the alpha and beta strings for the wavefunction determinant
  vector<int> alphaRef, betaRef;
  w.determinants[0].getAlphaBeta(alphaRef, betaRef);
  Eigen::Map<VectorXi> RowAlpha(&alphaRef[0], alphaRef.size());
  Eigen::Map<VectorXi> RowBeta (&betaRef[0] , betaRef.size());
  int norbs = Determinant::norbs;
  int nalpha = AlphaClosed.size();
  int nbeta = BetaClosed.size();

  //slice  
  alphaGamma = MatrixXd::Zero(norbs, nalpha);
  betaGamma = MatrixXd::Zero(norbs, nbeta);
  igl::slice_into(alphainv, RowAlpha, 1, alphaGamma); //alphaGamma(R,:) = alphainv;
  igl::slice_into(betainv , RowBeta , 1, betaGamma );

  AlphaTable = MatrixXd::Zero(AlphaOpen.size(), AlphaClosed.size());
  
  for (int i=0; i<AlphaClosed.size(); i++)
  {
    for (int a=0; a<AlphaOpen.size(); a++)
    {
      AlphaTable(a, i) = (Hforbs.row(AlphaOpen[a])*alphaGamma.col(i))(0);
    }
  }  

  BetaTable = MatrixXd::Zero(BetaOpen.size(), BetaClosed.size());
  for (int i=0; i<BetaClosed.size(); i++)
  {
    for (int a=0; a<BetaOpen.size(); a++)
    {
      BetaTable(a, i) = (Hforbs.row(BetaOpen[a])*betaGamma.col(i))(0);
    }
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
  int tableIndexa = std::lower_bound(AlphaOpen  .begin(), AlphaOpen  .end(), a) - AlphaOpen.begin();

  double p = 1.;
  if(doparity) d.parityA(a, i, p);

  return p * AlphaTable(tableIndexa, tableIndexi) ;
}

double Walker::getDetFactorB(int i, int a, CPSSlater &w, bool doparity)
{
  int tableIndexi = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), i) - BetaClosed.begin();
  int tableIndexa = std::lower_bound(BetaOpen  .begin(), BetaOpen  .end(), a) - BetaOpen.begin();

  double p = 1.;
  if(doparity) d.parityB(a, i, p);

  return p * BetaTable(tableIndexa, tableIndexi) ;
}

double Walker::getDetFactorA(int i, int j, int a, int b, CPSSlater &w, bool doparity)
{
  int tableIndexi = std::lower_bound(AlphaClosed.begin(), AlphaClosed.end(), i) - AlphaClosed.begin();
  int tableIndexa = std::lower_bound(AlphaOpen  .begin(), AlphaOpen  .end(), a) - AlphaOpen.begin();
  int tableIndexj = std::lower_bound(AlphaClosed.begin(), AlphaClosed.end(), j) - AlphaClosed.begin();
  int tableIndexb = std::lower_bound(AlphaOpen  .begin(), AlphaOpen  .end(), b) - AlphaOpen.begin();

  return (AlphaTable(tableIndexa, tableIndexi) * AlphaTable(tableIndexb, tableIndexj) - AlphaTable(tableIndexb, tableIndexi)*AlphaTable(tableIndexa, tableIndexj));
}

double Walker::getDetFactorB(int i, int j, int a, int b, CPSSlater &w, bool doparity)
{
  int tableIndexi = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), i) - BetaClosed.begin();
  int tableIndexa = std::lower_bound(BetaOpen  .begin(), BetaOpen  .end(), a) - BetaOpen.begin();
  int tableIndexj = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), j) - BetaClosed.begin();
  int tableIndexb = std::lower_bound(BetaOpen  .begin(), BetaOpen  .end(), b) - BetaOpen.begin();

 
  return (BetaTable(tableIndexa, tableIndexi) * BetaTable(tableIndexb, tableIndexj) - BetaTable(tableIndexb, tableIndexi)*BetaTable(tableIndexa, tableIndexj)) ;
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
      localDet(i,a) = AlphaTable(tableIndexi, tableIndexa);
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
      localDet(i,a) = BetaTable(tableIndexi, tableIndexa);
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

 //Generate the alpha and beta strings for the wavefunction determinant
  vector<int> alphaRef, betaRef;
  w.determinants[0].getAlphaBeta(alphaRef, betaRef);
  Eigen::Map<VectorXi> RowAlpha(&alphaRef[0], alphaRef.size());
  Eigen::Map<VectorXi> RowBeta (&betaRef[0] , betaRef.size());
  int norbs = Determinant::norbs;
  int nalpha = AlphaClosed.size();
  int nbeta = BetaClosed.size();


  double alphaDetFactor = Hforbs.row(a) * alphaGamma.col(i);
  alphaDet *= alphaDetFactor*p;

  Eigen::MatrixXd Hfnarrow, alphainv;
  igl::slice(alphaGamma, RowAlpha, 1, alphainv); //alphainv = alphaGamma(R,:);
  igl::slice(Hforbs, VectorXi::LinSpaced(norbs+1,0,norbs), RowAlpha, Hfnarrow); //Hfnarrow = Hforbs(:, R)
  MatrixXd vtAinv = (Hfnarrow.row(a) - Hfnarrow.row(i)) *
                    alphainv;

  MatrixXd alphainvWrongOrder = alphainv - (alphainv.col(tableIndexi) * vtAinv) / alphaDetFactor;

  AlphaClosed[tableIndexi] = a;
  AlphaOpen  [tableIndexa] = i;

  std::vector<int> order(AlphaClosed.size()), acopy = AlphaClosed;
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&acopy](size_t i1, size_t i2) { return acopy[i1] < acopy[i2]; });

  for (int i = 0; i < order.size(); i++)
  {
    alphainv.col(i) = alphainvWrongOrder.col(order[i]);
  }

  d.setoccA(i, false);
  d.setoccA(a, true );
  std::sort(AlphaClosed.begin(), AlphaClosed.end());
  std::sort(AlphaOpen  .begin(), AlphaOpen  .end());

  igl::slice_into(alphainv, RowAlpha, 1, alphaGamma);
}


void Walker::updateB(int i, int a, CPSSlater& w) {
  double p = 1.0;
  d.parityB(a, i, p);

  int tableIndexi = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), i) - BetaClosed.begin();
  int tableIndexa = std::lower_bound(BetaOpen  .begin(), BetaOpen  .end(), a) - BetaOpen.begin();

 //Generate the alpha and beta strings for the wavefunction determinant
  vector<int> alphaRef, betaRef;
  w.determinants[0].getAlphaBeta(alphaRef, betaRef);
  Eigen::Map<VectorXi> RowBeta (&betaRef[0] , betaRef.size());
  int norbs = Determinant::norbs;
  int nbeta = BetaClosed.size();


  double betaDetFactor = Hforbs.row(a) * betaGamma.col(i);
  betaDet *= betaDetFactor*p;

  Eigen::MatrixXd Hfnarrow, betainv;
  igl::slice(betaGamma, RowBeta, 1, betainv); //alphainv = alphaGamma(R,:);
  igl::slice(Hforbs, VectorXi::LinSpaced(norbs+1,0,norbs), RowBeta, Hfnarrow); //Hfnarrow = Hforbs(:, R)
  MatrixXd vtAinv = (Hfnarrow.row(a) - Hfnarrow.row(i)) *
                    betainv;

  MatrixXd betainvWrongOrder = betainv - (betainv.col(tableIndexi) * vtAinv) / betaDetFactor;

  BetaClosed[tableIndexi] = a;
  BetaOpen  [tableIndexa] = i;

  std::vector<int> order(BetaClosed.size()), bcopy = BetaClosed;
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&bcopy](size_t i1, size_t i2) { return bcopy[i1] < bcopy[i2]; });

  for (int i = 0; i < order.size(); i++)
  {
    betainv.col(i) = betainvWrongOrder.col(order[i]);
  }

  d.setoccB(i, false);
  d.setoccB(a, true );
  std::sort(BetaClosed.begin(), BetaClosed.end());
  std::sort(BetaOpen  .begin(), BetaOpen  .end());

  igl::slice_into(betainv, RowBeta, 1, betaGamma);
}
