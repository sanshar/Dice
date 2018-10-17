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

#include <algorithm>
#include <numeric>
#include "ShermanMorrisonWoodbury.h"

/**
 * This takes an inverse and determinant of a matrix formed by a subset of
 * columns and rows of Hforbs
 * and generates the new inverse and determinant 
 * by replacing cols with incides des with those with indices cre
 * RowVec is the set of row indices that are common to both in the 
 * incoming and outgoing matrices. ColIn are the column indices
 * of the incoming matrix. 
 */
void calculateInverseDeterminantWithColumnChange(const Eigen::MatrixXd &inverseIn, const double &detValueIn,
                                                                  Eigen::MatrixXd &inverseOut, double &detValueOut,
                                                                  std::vector<int>& cre, std::vector<int>& des,
                                                                  const Eigen::Map<Eigen::VectorXi> &RowVec,
                                                                  std::vector<int> &ColIn, const Eigen::MatrixXd &Hforbs)
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

  Eigen::Map<Eigen::VectorXi> ColCre(&cre[0], ncre);
  Eigen::Map<Eigen::VectorXi> ColDes(&des[0], ndes);

  Eigen::MatrixXd newCol, oldCol;
  igl::slice(Hforbs, RowVec, ColCre, newCol);
  igl::slice(Hforbs, RowVec, ColDes, oldCol);
  newCol = newCol - oldCol;

  Eigen::MatrixXd vT = Eigen::MatrixXd::Zero(ncre, ColIn.size());
  std::vector<int> ColOutWrong = ColIn;
  for (int i = 0; i < ndes; i++)
  {
    int index = std::lower_bound(ColIn.begin(), ColIn.end(), des[i]) - ColIn.begin();
    vT(i, index) = 1.0;
    ColOutWrong[index] = cre[i];
  }

  //igl::slice(inverseIn, ColCre, 1, vTinverseIn);
  Eigen::MatrixXd vTinverseIn = vT * inverseIn;

  Eigen::MatrixXd Id = Eigen::MatrixXd::Identity(ncre, ncre);
  Eigen::MatrixXd detFactor = Id + vTinverseIn * newCol;
  Eigen::MatrixXd detFactorInv, inverseOutWrong;

  Eigen::FullPivLU<Eigen::MatrixXd> lub(detFactor);
  if (lub.isInvertible())
  {
    detFactorInv = lub.inverse();
    inverseOutWrong = inverseIn - ((inverseIn * newCol) * detFactorInv) * (vTinverseIn);
    detValueOut = detValueIn * detFactor.determinant();
  }
  else
  {
    Eigen::MatrixXd originalOrbs;
    Eigen::Map<Eigen::VectorXi> Col(&ColIn[0], ColIn.size());
    igl::slice(Hforbs, RowVec, Col, originalOrbs);
    Eigen::MatrixXd newOrbs = originalOrbs + newCol * vT;
    inverseOutWrong = newOrbs.inverse();
    detValueOut = newOrbs.determinant();
  }

  //now we need to reorder the inverse to correct the order of rows
  std::vector<int> order(ColOutWrong.size()), ccopy = ColOutWrong;
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&ccopy](size_t i1, size_t i2) { return ccopy[i1] < ccopy[i2]; });
  Eigen::Map<Eigen::VectorXi> orderVec(&order[0], order.size());
  igl::slice(inverseOutWrong, orderVec, 1, inverseOut);
}

/**
 * This takes an inverse and determinant of a matrix formed by a subset of
 * columns and rows of Hforbs
 * and generates the new inverse and determinant 
 * by replacing rows with incides des with those with indices des
 * ColVec is the set of col indices that are common to both in the 
 * incoming and outgoing matrices. RowIn are the column indices
 * of the incoming matrix. 
 */
void calculateInverseDeterminantWithRowChange(const Eigen::MatrixXd &inverseIn, const double &detValueIn,
                                                               Eigen::MatrixXd &inverseOut, double &detValueOut,
                                                               std::vector<int>& cre, std::vector<int>& des,
                                                               const Eigen::Map<Eigen::VectorXi> &ColVec,
                                                               std::vector<int> &RowIn, const Eigen::MatrixXd &Hforbs)
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

  Eigen::Map<Eigen::VectorXi> RowCre(&cre[0], ncre);
  Eigen::Map<Eigen::VectorXi> RowDes(&des[0], ndes);

  Eigen::MatrixXd newRow, oldRow;
  igl::slice(Hforbs, RowCre, ColVec, newRow);
  igl::slice(Hforbs, RowDes, ColVec, oldRow);
  newRow = newRow - oldRow;

  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(ColVec.rows(), ncre);
  std::vector<int> RowOutWrong = RowIn;
  for (int i = 0; i < ndes; i++)
  {
    int index = std::lower_bound(RowIn.begin(), RowIn.end(), des[i]) - RowIn.begin();
    U(index, i) = 1.0;
    RowOutWrong[index] = cre[i];
  }
  //igl::slice(inverseIn, VectorXi::LinSpaced(RowIn.size(), 0, RowIn.size() + 1), RowDes, inverseInU);
  Eigen::MatrixXd inverseInU = inverseIn * U;
  Eigen::MatrixXd Id = Eigen::MatrixXd::Identity(ncre, ncre);
  Eigen::MatrixXd detFactor = Id + newRow * inverseInU;
  Eigen::MatrixXd detFactorInv, inverseOutWrong;

  Eigen::FullPivLU<Eigen::MatrixXd> lub(detFactor);
  if (lub.isInvertible())
  {
    detFactorInv = lub.inverse();
    inverseOutWrong = inverseIn - ((inverseInU)*detFactorInv) * (newRow * inverseIn);
    detValueOut = detValueIn * detFactor.determinant();
  }
  else
  {
    Eigen::MatrixXd originalOrbs;
    Eigen::Map<Eigen::VectorXi> Row(&RowIn[0], RowIn.size());
    igl::slice(Hforbs, Row, ColVec, originalOrbs);
    Eigen::MatrixXd newOrbs = originalOrbs + U * newRow;
    inverseOutWrong = newOrbs.inverse();
    detValueOut = newOrbs.determinant();
  }

  //now we need to reorder the inverse to correct the order of rows
  std::vector<int> order(RowOutWrong.size()), rcopy = RowOutWrong;
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&rcopy](size_t i1, size_t i2) { return rcopy[i1] < rcopy[i2]; });
  Eigen::Map<Eigen::VectorXi> orderVec(&order[0], order.size());
  igl::slice(inverseOutWrong, Eigen::VectorXi::LinSpaced(ColVec.rows(), 0, ColVec.rows() - 1), orderVec, inverseOut);
}

