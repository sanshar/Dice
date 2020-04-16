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
#ifndef SMW_HEADER_H
#define SMW_HEADER_H

#include <Eigen/Dense>
#include "igl/slice.h"
#include "igl/slice_into.h"


/**
 * This takes an inverse and determinant of a matrix formed by a subset of
 * columns and rows of Hforbs
 * and generates the new inverse and determinant 
 * by replacing cols with incides des with those with indices cre
 * RowVec is the set of row indices that are common to both in the 
 * incoming and outgoing matrices. ColIn are the column indices
 * of the incoming matrix. 
 */
void calculateInverseDeterminantWithColumnChange(const Eigen::MatrixXcd &inverseIn, const std::complex<double> &detValueIn, const Eigen::MatrixXcd &tableIn,
                                                                  Eigen::MatrixXcd &inverseOut, std::complex<double> &detValueOut, Eigen::MatrixXcd &tableOut,
                                                                  std::vector<int>& cre, std::vector<int>& des,
                                                                  const Eigen::Map<Eigen::VectorXi> &RowVec,
                                                                  std::vector<int> &ColIn, const Eigen::MatrixXcd &Hforbs);

/**
 * This takes an inverse and determinant of a matrix formed by a subset of
 * columns and rows of Hforbs
 * and generates the new inverse and determinant 
 * by replacing rows with incides des with those with indices des
 * ColVec is the set of col indices that are common to both in the 
 * incoming and outgoing matrices. RowIn are the column indices
 * of the incoming matrix. 
 */
void calculateInverseDeterminantWithRowChange(const Eigen::MatrixXcd &inverseIn, const std::complex<double> &detValueIn, const Eigen::MatrixXcd &tableIn,
                                                               Eigen::MatrixXcd &inverseOut, std::complex<double> &detValueOut, Eigen::MatrixXcd &tableOut,
                                                               std::vector<int>& cre, std::vector<int>& des,
                                                               const Eigen::Map<Eigen::VectorXi> &ColVec,
                                                               std::vector<int> &RowIn, const Eigen::MatrixXcd &Hforbs, const bool updateTable);


// determinant function specialized for small matrices
inline std::complex<double> calcDet(const Eigen::MatrixXcd &detSlice) {
  if (detSlice.rows() == 5) return detSlice(0, 0) * detSlice(1, 1) * detSlice(2, 2) * detSlice(3, 3) * detSlice(4, 4)
             - detSlice(0, 0) * detSlice(1, 1) * detSlice(2, 2) * detSlice(3, 4) * detSlice(4, 3)
             - detSlice(0, 0) * detSlice(1, 1) * detSlice(2, 3) * detSlice(3, 2) * detSlice(4, 4)
             + detSlice(0, 0) * detSlice(1, 1) * detSlice(2, 3) * detSlice(3, 4) * detSlice(4, 2)
             + detSlice(0, 0) * detSlice(1, 1) * detSlice(2, 4) * detSlice(3, 2) * detSlice(4, 3)
             - detSlice(0, 0) * detSlice(1, 1) * detSlice(2, 4) * detSlice(3, 3) * detSlice(4, 2)
             - detSlice(0, 0) * detSlice(1, 2) * detSlice(2, 1) * detSlice(3, 3) * detSlice(4, 4)
             + detSlice(0, 0) * detSlice(1, 2) * detSlice(2, 1) * detSlice(3, 4) * detSlice(4, 3)
             + detSlice(0, 0) * detSlice(1, 2) * detSlice(2, 3) * detSlice(3, 1) * detSlice(4, 4)
             - detSlice(0, 0) * detSlice(1, 2) * detSlice(2, 3) * detSlice(3, 4) * detSlice(4, 1)
             - detSlice(0, 0) * detSlice(1, 2) * detSlice(2, 4) * detSlice(3, 1) * detSlice(4, 3)
             + detSlice(0, 0) * detSlice(1, 2) * detSlice(2, 4) * detSlice(3, 3) * detSlice(4, 1)
             + detSlice(0, 0) * detSlice(1, 3) * detSlice(2, 1) * detSlice(3, 2) * detSlice(4, 4)
             - detSlice(0, 0) * detSlice(1, 3) * detSlice(2, 1) * detSlice(3, 4) * detSlice(4, 2)
             - detSlice(0, 0) * detSlice(1, 3) * detSlice(2, 2) * detSlice(3, 1) * detSlice(4, 4)
             + detSlice(0, 0) * detSlice(1, 3) * detSlice(2, 2) * detSlice(3, 4) * detSlice(4, 1)
             + detSlice(0, 0) * detSlice(1, 3) * detSlice(2, 4) * detSlice(3, 1) * detSlice(4, 2)
             - detSlice(0, 0) * detSlice(1, 3) * detSlice(2, 4) * detSlice(3, 2) * detSlice(4, 1)
             - detSlice(0, 0) * detSlice(1, 4) * detSlice(2, 1) * detSlice(3, 2) * detSlice(4, 3)
             + detSlice(0, 0) * detSlice(1, 4) * detSlice(2, 1) * detSlice(3, 3) * detSlice(4, 2)
             + detSlice(0, 0) * detSlice(1, 4) * detSlice(2, 2) * detSlice(3, 1) * detSlice(4, 3)
             - detSlice(0, 0) * detSlice(1, 4) * detSlice(2, 2) * detSlice(3, 3) * detSlice(4, 1)
             - detSlice(0, 0) * detSlice(1, 4) * detSlice(2, 3) * detSlice(3, 1) * detSlice(4, 2)
             + detSlice(0, 0) * detSlice(1, 4) * detSlice(2, 3) * detSlice(3, 2) * detSlice(4, 1)
             - detSlice(0, 1) * detSlice(1, 0) * detSlice(2, 2) * detSlice(3, 3) * detSlice(4, 4)
             + detSlice(0, 1) * detSlice(1, 0) * detSlice(2, 2) * detSlice(3, 4) * detSlice(4, 3)
             + detSlice(0, 1) * detSlice(1, 0) * detSlice(2, 3) * detSlice(3, 2) * detSlice(4, 4)
             - detSlice(0, 1) * detSlice(1, 0) * detSlice(2, 3) * detSlice(3, 4) * detSlice(4, 2)
             - detSlice(0, 1) * detSlice(1, 0) * detSlice(2, 4) * detSlice(3, 2) * detSlice(4, 3)
             + detSlice(0, 1) * detSlice(1, 0) * detSlice(2, 4) * detSlice(3, 3) * detSlice(4, 2)
             + detSlice(0, 1) * detSlice(1, 2) * detSlice(2, 0) * detSlice(3, 3) * detSlice(4, 4)
             - detSlice(0, 1) * detSlice(1, 2) * detSlice(2, 0) * detSlice(3, 4) * detSlice(4, 3)
             - detSlice(0, 1) * detSlice(1, 2) * detSlice(2, 3) * detSlice(3, 0) * detSlice(4, 4)
             + detSlice(0, 1) * detSlice(1, 2) * detSlice(2, 3) * detSlice(3, 4) * detSlice(4, 0)
             + detSlice(0, 1) * detSlice(1, 2) * detSlice(2, 4) * detSlice(3, 0) * detSlice(4, 3)
             - detSlice(0, 1) * detSlice(1, 2) * detSlice(2, 4) * detSlice(3, 3) * detSlice(4, 0)
             - detSlice(0, 1) * detSlice(1, 3) * detSlice(2, 0) * detSlice(3, 2) * detSlice(4, 4)
             + detSlice(0, 1) * detSlice(1, 3) * detSlice(2, 0) * detSlice(3, 4) * detSlice(4, 2)
             + detSlice(0, 1) * detSlice(1, 3) * detSlice(2, 2) * detSlice(3, 0) * detSlice(4, 4)
             - detSlice(0, 1) * detSlice(1, 3) * detSlice(2, 2) * detSlice(3, 4) * detSlice(4, 0)
             - detSlice(0, 1) * detSlice(1, 3) * detSlice(2, 4) * detSlice(3, 0) * detSlice(4, 2)
             + detSlice(0, 1) * detSlice(1, 3) * detSlice(2, 4) * detSlice(3, 2) * detSlice(4, 0)
             + detSlice(0, 1) * detSlice(1, 4) * detSlice(2, 0) * detSlice(3, 2) * detSlice(4, 3)
             - detSlice(0, 1) * detSlice(1, 4) * detSlice(2, 0) * detSlice(3, 3) * detSlice(4, 2)
             - detSlice(0, 1) * detSlice(1, 4) * detSlice(2, 2) * detSlice(3, 0) * detSlice(4, 3)
             + detSlice(0, 1) * detSlice(1, 4) * detSlice(2, 2) * detSlice(3, 3) * detSlice(4, 0)
             + detSlice(0, 1) * detSlice(1, 4) * detSlice(2, 3) * detSlice(3, 0) * detSlice(4, 2)
             - detSlice(0, 1) * detSlice(1, 4) * detSlice(2, 3) * detSlice(3, 2) * detSlice(4, 0)
             + detSlice(0, 2) * detSlice(1, 0) * detSlice(2, 1) * detSlice(3, 3) * detSlice(4, 4)
             - detSlice(0, 2) * detSlice(1, 0) * detSlice(2, 1) * detSlice(3, 4) * detSlice(4, 3)
             - detSlice(0, 2) * detSlice(1, 0) * detSlice(2, 3) * detSlice(3, 1) * detSlice(4, 4)
             + detSlice(0, 2) * detSlice(1, 0) * detSlice(2, 3) * detSlice(3, 4) * detSlice(4, 1)
             + detSlice(0, 2) * detSlice(1, 0) * detSlice(2, 4) * detSlice(3, 1) * detSlice(4, 3)
             - detSlice(0, 2) * detSlice(1, 0) * detSlice(2, 4) * detSlice(3, 3) * detSlice(4, 1)
             - detSlice(0, 2) * detSlice(1, 1) * detSlice(2, 0) * detSlice(3, 3) * detSlice(4, 4)
             + detSlice(0, 2) * detSlice(1, 1) * detSlice(2, 0) * detSlice(3, 4) * detSlice(4, 3)
             + detSlice(0, 2) * detSlice(1, 1) * detSlice(2, 3) * detSlice(3, 0) * detSlice(4, 4)
             - detSlice(0, 2) * detSlice(1, 1) * detSlice(2, 3) * detSlice(3, 4) * detSlice(4, 0)
             - detSlice(0, 2) * detSlice(1, 1) * detSlice(2, 4) * detSlice(3, 0) * detSlice(4, 3)
             + detSlice(0, 2) * detSlice(1, 1) * detSlice(2, 4) * detSlice(3, 3) * detSlice(4, 0)
             + detSlice(0, 2) * detSlice(1, 3) * detSlice(2, 0) * detSlice(3, 1) * detSlice(4, 4)
             - detSlice(0, 2) * detSlice(1, 3) * detSlice(2, 0) * detSlice(3, 4) * detSlice(4, 1)
             - detSlice(0, 2) * detSlice(1, 3) * detSlice(2, 1) * detSlice(3, 0) * detSlice(4, 4)
             + detSlice(0, 2) * detSlice(1, 3) * detSlice(2, 1) * detSlice(3, 4) * detSlice(4, 0)
             + detSlice(0, 2) * detSlice(1, 3) * detSlice(2, 4) * detSlice(3, 0) * detSlice(4, 1)
             - detSlice(0, 2) * detSlice(1, 3) * detSlice(2, 4) * detSlice(3, 1) * detSlice(4, 0)
             - detSlice(0, 2) * detSlice(1, 4) * detSlice(2, 0) * detSlice(3, 1) * detSlice(4, 3)
             + detSlice(0, 2) * detSlice(1, 4) * detSlice(2, 0) * detSlice(3, 3) * detSlice(4, 1)
             + detSlice(0, 2) * detSlice(1, 4) * detSlice(2, 1) * detSlice(3, 0) * detSlice(4, 3)
             - detSlice(0, 2) * detSlice(1, 4) * detSlice(2, 1) * detSlice(3, 3) * detSlice(4, 0)
             - detSlice(0, 2) * detSlice(1, 4) * detSlice(2, 3) * detSlice(3, 0) * detSlice(4, 1)
             + detSlice(0, 2) * detSlice(1, 4) * detSlice(2, 3) * detSlice(3, 1) * detSlice(4, 0)
             - detSlice(0, 3) * detSlice(1, 0) * detSlice(2, 1) * detSlice(3, 2) * detSlice(4, 4)
             + detSlice(0, 3) * detSlice(1, 0) * detSlice(2, 1) * detSlice(3, 4) * detSlice(4, 2)
             + detSlice(0, 3) * detSlice(1, 0) * detSlice(2, 2) * detSlice(3, 1) * detSlice(4, 4)
             - detSlice(0, 3) * detSlice(1, 0) * detSlice(2, 2) * detSlice(3, 4) * detSlice(4, 1)
             - detSlice(0, 3) * detSlice(1, 0) * detSlice(2, 4) * detSlice(3, 1) * detSlice(4, 2)
             + detSlice(0, 3) * detSlice(1, 0) * detSlice(2, 4) * detSlice(3, 2) * detSlice(4, 1)
             + detSlice(0, 3) * detSlice(1, 1) * detSlice(2, 0) * detSlice(3, 2) * detSlice(4, 4)
             - detSlice(0, 3) * detSlice(1, 1) * detSlice(2, 0) * detSlice(3, 4) * detSlice(4, 2)
             - detSlice(0, 3) * detSlice(1, 1) * detSlice(2, 2) * detSlice(3, 0) * detSlice(4, 4)
             + detSlice(0, 3) * detSlice(1, 1) * detSlice(2, 2) * detSlice(3, 4) * detSlice(4, 0)
             + detSlice(0, 3) * detSlice(1, 1) * detSlice(2, 4) * detSlice(3, 0) * detSlice(4, 2)
             - detSlice(0, 3) * detSlice(1, 1) * detSlice(2, 4) * detSlice(3, 2) * detSlice(4, 0)
             - detSlice(0, 3) * detSlice(1, 2) * detSlice(2, 0) * detSlice(3, 1) * detSlice(4, 4)
             + detSlice(0, 3) * detSlice(1, 2) * detSlice(2, 0) * detSlice(3, 4) * detSlice(4, 1)
             + detSlice(0, 3) * detSlice(1, 2) * detSlice(2, 1) * detSlice(3, 0) * detSlice(4, 4)
             - detSlice(0, 3) * detSlice(1, 2) * detSlice(2, 1) * detSlice(3, 4) * detSlice(4, 0)
             - detSlice(0, 3) * detSlice(1, 2) * detSlice(2, 4) * detSlice(3, 0) * detSlice(4, 1)
             + detSlice(0, 3) * detSlice(1, 2) * detSlice(2, 4) * detSlice(3, 1) * detSlice(4, 0)
             + detSlice(0, 3) * detSlice(1, 4) * detSlice(2, 0) * detSlice(3, 1) * detSlice(4, 2)
             - detSlice(0, 3) * detSlice(1, 4) * detSlice(2, 0) * detSlice(3, 2) * detSlice(4, 1)
             - detSlice(0, 3) * detSlice(1, 4) * detSlice(2, 1) * detSlice(3, 0) * detSlice(4, 2)
             + detSlice(0, 3) * detSlice(1, 4) * detSlice(2, 1) * detSlice(3, 2) * detSlice(4, 0)
             + detSlice(0, 3) * detSlice(1, 4) * detSlice(2, 2) * detSlice(3, 0) * detSlice(4, 1)
             - detSlice(0, 3) * detSlice(1, 4) * detSlice(2, 2) * detSlice(3, 1) * detSlice(4, 0)
             + detSlice(0, 4) * detSlice(1, 0) * detSlice(2, 1) * detSlice(3, 2) * detSlice(4, 3)
             - detSlice(0, 4) * detSlice(1, 0) * detSlice(2, 1) * detSlice(3, 3) * detSlice(4, 2)
             - detSlice(0, 4) * detSlice(1, 0) * detSlice(2, 2) * detSlice(3, 1) * detSlice(4, 3)
             + detSlice(0, 4) * detSlice(1, 0) * detSlice(2, 2) * detSlice(3, 3) * detSlice(4, 1)
             + detSlice(0, 4) * detSlice(1, 0) * detSlice(2, 3) * detSlice(3, 1) * detSlice(4, 2)
             - detSlice(0, 4) * detSlice(1, 0) * detSlice(2, 3) * detSlice(3, 2) * detSlice(4, 1)
             - detSlice(0, 4) * detSlice(1, 1) * detSlice(2, 0) * detSlice(3, 2) * detSlice(4, 3)
             + detSlice(0, 4) * detSlice(1, 1) * detSlice(2, 0) * detSlice(3, 3) * detSlice(4, 2)
             + detSlice(0, 4) * detSlice(1, 1) * detSlice(2, 2) * detSlice(3, 0) * detSlice(4, 3)
             - detSlice(0, 4) * detSlice(1, 1) * detSlice(2, 2) * detSlice(3, 3) * detSlice(4, 0)
             - detSlice(0, 4) * detSlice(1, 1) * detSlice(2, 3) * detSlice(3, 0) * detSlice(4, 2)
             + detSlice(0, 4) * detSlice(1, 1) * detSlice(2, 3) * detSlice(3, 2) * detSlice(4, 0)
             + detSlice(0, 4) * detSlice(1, 2) * detSlice(2, 0) * detSlice(3, 1) * detSlice(4, 3)
             - detSlice(0, 4) * detSlice(1, 2) * detSlice(2, 0) * detSlice(3, 3) * detSlice(4, 1)
             - detSlice(0, 4) * detSlice(1, 2) * detSlice(2, 1) * detSlice(3, 0) * detSlice(4, 3)
             + detSlice(0, 4) * detSlice(1, 2) * detSlice(2, 1) * detSlice(3, 3) * detSlice(4, 0)
             + detSlice(0, 4) * detSlice(1, 2) * detSlice(2, 3) * detSlice(3, 0) * detSlice(4, 1)
             - detSlice(0, 4) * detSlice(1, 2) * detSlice(2, 3) * detSlice(3, 1) * detSlice(4, 0)
             - detSlice(0, 4) * detSlice(1, 3) * detSlice(2, 0) * detSlice(3, 1) * detSlice(4, 2)
             + detSlice(0, 4) * detSlice(1, 3) * detSlice(2, 0) * detSlice(3, 2) * detSlice(4, 1)
             + detSlice(0, 4) * detSlice(1, 3) * detSlice(2, 1) * detSlice(3, 0) * detSlice(4, 2)
             - detSlice(0, 4) * detSlice(1, 3) * detSlice(2, 1) * detSlice(3, 2) * detSlice(4, 0)
             - detSlice(0, 4) * detSlice(1, 3) * detSlice(2, 2) * detSlice(3, 0) * detSlice(4, 1)
             + detSlice(0, 4) * detSlice(1, 3) * detSlice(2, 2) * detSlice(3, 1) * detSlice(4, 0);
  else return detSlice.determinant();
}

//pfaffian of a real matrix using Hessenberg decomposition
double calcPfaffianH(const Eigen::MatrixXd &mat); 
//pfaffian of a complex matrix using Parlett-Reid algorithm
std::complex<double> calcPfaffian(const Eigen::MatrixXcd &mat); 


#endif
