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
#include "GHFWalker.h"
#include "global.h"
#include <algorithm>

using namespace Eigen;



void GHFWalker::calculateInverseDeterminantWithColumnChange(MatrixXd &inverseIn, double &detValueIn,
                                                                  MatrixXd &inverseOut, double &detValueOut,
                                                                  vector<int> &cre, vector<int> &des,
                                                                  Eigen::Map<Eigen::VectorXi> &RowVec,
                                                                  vector<int> &ColIn, MatrixXd &Hforbs)
{
  int ncre = 0, ndes = 0;
  for (int i = 0; i < cre.size(); i++)
    if (cre[i] != -1)
      ncre++;
  for (int i = 0; i < des.size(); i++)
    if (des[i] != -1)
      ndes++;
  if (ncre == 0)
  {
    inverseOut = inverseIn;
    detValueOut = detValueIn;
    return;
  }

  Eigen::Map<VectorXi> ColCre(&cre[0], ncre);
  Eigen::Map<VectorXi> ColDes(&des[0], ndes);

  MatrixXd newCol, oldCol;
  igl::slice(Hforbs, RowVec, ColCre, newCol);
  igl::slice(Hforbs, RowVec, ColDes, oldCol);
  newCol = newCol - oldCol;

  MatrixXd vT = MatrixXd::Zero(ncre, ColIn.size());
  vector<int> ColOutWrong = ColIn;
  for (int i = 0; i < ndes; i++)
  {
    int index = std::lower_bound(ColIn.begin(), ColIn.end(), des[i]) - ColIn.begin();
    vT(i, index) = 1.0;
    ColOutWrong[index] = cre[i];
  }

  //igl::slice(inverseIn, ColCre, 1, vTinverseIn);
  MatrixXd vTinverseIn = vT * inverseIn;

  MatrixXd Id = MatrixXd::Identity(ncre, ncre);
  MatrixXd detFactor = Id + vTinverseIn * newCol;
  MatrixXd detFactorInv, inverseOutWrong;

  Eigen::FullPivLU<MatrixXd> lub(detFactor);
  if (lub.isInvertible())
  {
    detFactorInv = lub.inverse();
    inverseOutWrong = inverseIn - ((inverseIn * newCol) * detFactorInv) * (vTinverseIn);
    detValueOut = detValueIn * detFactor.determinant();
  }
  else
  {
    MatrixXd originalOrbs;
    Eigen::Map<VectorXi> Col(&ColIn[0], ColIn.size());
    igl::slice(Hforbs, RowVec, Col, originalOrbs);
    MatrixXd newOrbs = originalOrbs + newCol * vT;
    inverseOutWrong = newOrbs.inverse();
    detValueOut = newOrbs.determinant();
  }

  //now we need to reorder the inverse to correct the order of rows
  std::vector<int> order(ColOutWrong.size()), ccopy = ColOutWrong;
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&ccopy](size_t i1, size_t i2) { return ccopy[i1] < ccopy[i2]; });
  Eigen::Map<VectorXi> orderVec(&order[0], order.size());
  igl::slice(inverseOutWrong, orderVec, 1, inverseOut);
}

void GHFWalker::calculateInverseDeterminantWithRowChange(MatrixXd &inverseIn, double &detValueIn,
                                                               MatrixXd &inverseOut, double &detValueOut,
                                                               vector<int> &cre, vector<int> &des,
                                                               Eigen::Map<Eigen::VectorXi> &ColVec,
                                                               vector<int> &RowIn, MatrixXd &Hforbs)
{
  int ncre = 0, ndes = 0;
  for (int i = 0; i < cre.size(); i++)
    if (cre[i] != -1)
      ncre++;
  for (int i = 0; i < des.size(); i++)
    if (des[i] != -1)
      ndes++;
  if (ncre == 0)
  {
    inverseOut = inverseIn;
    detValueOut = detValueIn;
    return;
  }

  Eigen::Map<VectorXi> RowCre(&cre[0], ncre);
  Eigen::Map<VectorXi> RowDes(&des[0], ndes);

  MatrixXd newRow, oldRow;
  igl::slice(Hforbs, RowCre, ColVec, newRow);
  igl::slice(Hforbs, RowDes, ColVec, oldRow);
  newRow = newRow - oldRow;

  MatrixXd U = MatrixXd::Zero(ColVec.rows(), ncre);
  vector<int> RowOutWrong = RowIn;
  for (int i = 0; i < ndes; i++)
  {
    int index = std::lower_bound(RowIn.begin(), RowIn.end(), des[i]) - RowIn.begin();
    U(index, i) = 1.0;
    RowOutWrong[index] = cre[i];
  }
  //igl::slice(inverseIn, VectorXi::LinSpaced(RowIn.size(), 0, RowIn.size() + 1), RowDes, inverseInU);
  MatrixXd inverseInU = inverseIn * U;
  MatrixXd Id = MatrixXd::Identity(ncre, ncre);
  MatrixXd detFactor = Id + newRow * inverseInU;
  MatrixXd detFactorInv, inverseOutWrong;

  Eigen::FullPivLU<MatrixXd> lub(detFactor);
  if (lub.isInvertible())
  {
    detFactorInv = lub.inverse();
    inverseOutWrong = inverseIn - ((inverseInU)*detFactorInv) * (newRow * inverseIn);
    detValueOut = detValueIn * detFactor.determinant();
  }
  else
  {
    MatrixXd originalOrbs;
    Eigen::Map<VectorXi> Row(&RowIn[0], RowIn.size());
    igl::slice(Hforbs, Row, ColVec, originalOrbs);
    MatrixXd newOrbs = originalOrbs + U * newRow;
    inverseOutWrong = newOrbs.inverse();
    detValueOut = newOrbs.determinant();
  }

  //now we need to reorder the inverse to correct the order of rows
  std::vector<int> order(RowOutWrong.size()), rcopy = RowOutWrong;
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&rcopy](size_t i1, size_t i2) { return rcopy[i1] < rcopy[i2]; });
  Eigen::Map<VectorXi> orderVec(&order[0], order.size());
  igl::slice(inverseOutWrong, VectorXi::LinSpaced(ColVec.rows(), 0, ColVec.rows()), orderVec, inverseOut);
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
*/
