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
#include <Eigen/Eigenvalues>
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
void calculateInverseDeterminantWithColumnChange(const Eigen::MatrixXcd &inverseIn, const std::complex<double> &detValueIn,
                                                                  Eigen::MatrixXcd &inverseOut, std::complex<double> &detValueOut,
                                                                  std::vector<int>& cre, std::vector<int>& des,
                                                                  const Eigen::Map<Eigen::VectorXi> &RowVec,
                                                                  std::vector<int> &ColIn, const Eigen::MatrixXcd &Hforbs)
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

  Eigen::MatrixXcd newCol, oldCol;
  igl::slice(Hforbs, RowVec, ColCre, newCol);
  igl::slice(Hforbs, RowVec, ColDes, oldCol);
  newCol = newCol - oldCol;

  Eigen::MatrixXcd vT = Eigen::MatrixXcd::Zero(ncre, ColIn.size());
  std::vector<int> ColOutWrong = ColIn;
  for (int i = 0; i < ndes; i++)
  {
    int index = std::lower_bound(ColIn.begin(), ColIn.end(), des[i]) - ColIn.begin();
    vT(i, index) = 1.0;
    ColOutWrong[index] = cre[i];
  }

  //igl::slice(inverseIn, ColCre, 1, vTinverseIn);
  Eigen::MatrixXcd vTinverseIn = vT * inverseIn;

  Eigen::MatrixXcd Id = Eigen::MatrixXcd::Identity(ncre, ncre);
  Eigen::MatrixXcd detFactor = Id + vTinverseIn * newCol;
  Eigen::MatrixXcd detFactorInv, inverseOutWrong;

  Eigen::FullPivLU<Eigen::MatrixXcd> lub(detFactor);
  if (lub.isInvertible())
  {
    detFactorInv = lub.inverse();
    inverseOutWrong = inverseIn - ((inverseIn * newCol) * detFactorInv) * (vTinverseIn);
    detValueOut = detValueIn * detFactor.determinant();
  }
  else
  {
    Eigen::MatrixXcd originalOrbs;
    Eigen::Map<Eigen::VectorXi> Col(&ColIn[0], ColIn.size());
    igl::slice(Hforbs, RowVec, Col, originalOrbs);
    Eigen::MatrixXcd newOrbs = originalOrbs + newCol * vT;
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
void calculateInverseDeterminantWithRowChange(const Eigen::MatrixXcd &inverseIn, const std::complex<double> &detValueIn,
                                                               Eigen::MatrixXcd &inverseOut, std::complex<double> &detValueOut,
                                                               std::vector<int>& cre, std::vector<int>& des,
                                                               const Eigen::Map<Eigen::VectorXi> &ColVec,
                                                               std::vector<int> &RowIn, const Eigen::MatrixXcd &Hforbs)
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

  Eigen::MatrixXcd newRow, oldRow;
  igl::slice(Hforbs, RowCre, ColVec, newRow);
  igl::slice(Hforbs, RowDes, ColVec, oldRow);
  newRow = newRow - oldRow;

  Eigen::MatrixXcd U = Eigen::MatrixXd::Zero(ColVec.rows(), ncre);
  std::vector<int> RowOutWrong = RowIn;
  for (int i = 0; i < ndes; i++)
  {
    int index = std::lower_bound(RowIn.begin(), RowIn.end(), des[i]) - RowIn.begin();
    U(index, i) = 1.0;
    RowOutWrong[index] = cre[i];
  }
  //igl::slice(inverseIn, VectorXi::LinSpaced(RowIn.size(), 0, RowIn.size() + 1), RowDes, inverseInU);
  Eigen::MatrixXcd inverseInU = inverseIn * U;
  Eigen::MatrixXcd Id = Eigen::MatrixXd::Identity(ncre, ncre);
  Eigen::MatrixXcd detFactor = Id + newRow * inverseInU;
  Eigen::MatrixXcd detFactorInv, inverseOutWrong;

  Eigen::FullPivLU<Eigen::MatrixXcd> lub(detFactor);
  if (lub.isInvertible())
  {
    detFactorInv = lub.inverse();
    inverseOutWrong = inverseIn - ((inverseInU)*detFactorInv) * (newRow * inverseIn);
    detValueOut = detValueIn * detFactor.determinant();
  }
  else
  {
    Eigen::MatrixXcd originalOrbs;
    Eigen::Map<Eigen::VectorXi> Row(&RowIn[0], RowIn.size());
    igl::slice(Hforbs, Row, ColVec, originalOrbs);
    Eigen::MatrixXcd newOrbs = originalOrbs + U * newRow;
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


double calcPfaffianH(const Eigen::MatrixXd &mat)
{
  //if (mat.rows() % 2 == 1) return 0.;
  Eigen::HessenbergDecomposition<Eigen::MatrixXd> hd(mat);
  Eigen::MatrixXd triDiag = hd.matrixH();
  double pfaffian = 1.;
  int i = 0;
  while (i < mat.rows() - 1) {
    pfaffian *= triDiag(i, i+1);
    i++; i++;
  }
  return pfaffian;
}

std::complex<double> calcPfaffian(const Eigen::MatrixXcd &mat)
{
  Eigen::MatrixXcd matCopy = mat;
  int size = mat.rows();
  std::complex<double> pfaffian = 1.;
  int i = 0;
  while (i < size-1) {
    int currentSize = size-i;
    Eigen::VectorXd colNorm = matCopy.col(i).tail(currentSize-1).cwiseAbs();
    Eigen::VectorXd::Index maxIndex;
    colNorm.maxCoeff(&maxIndex);
    int ip = i+1+maxIndex;
    //pivot if necessary
    if (ip != i+1) {
      matCopy.block(i,i,currentSize,currentSize).row(1).swap(matCopy.block(i,i,currentSize, currentSize).row(ip-i));
      matCopy.block(i,i,currentSize,currentSize).col(1).swap(matCopy.block(i,i,currentSize, currentSize).col(ip-i));
      pfaffian *= -1;
    }
    //gauss elimination
    if (matCopy(i,i+1) != 0.) {
      pfaffian *= matCopy(i,i+1);
      Eigen::VectorXcd tau = matCopy.row(i).tail(currentSize-2);
      tau /= matCopy(i,i+1);
      if (i+2 < size) {
        matCopy.block(i+2,i+2,currentSize-2,currentSize-2) += tau * matCopy.col(i+1).tail(currentSize-2).transpose(); 
        matCopy.block(i+2,i+2,currentSize-2,currentSize-2) -= matCopy.col(i+1).tail(currentSize-2) * tau.transpose(); 
      }
    }
    else return 0.;
    i++; i++;
  }
  return pfaffian;
}
