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
#include "Davidson.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>
#include <iostream>
#include "boost/format.hpp"
#include "iowrapper.h"
#include "global.h"

using namespace Eigen;
using namespace std;
using namespace boost;

void SolveEigen(MatrixXd& A, VectorXd& b, VectorXd& x) {
  x = A.colPivHouseholderQr().solve(b);

}

void GeneralizedEigen(MatrixXd& Hamiltonian, MatrixXd& Overlap, VectorXcd& eigenvalues, MatrixXcd& eigenvectors, VectorXd& betas) {

  GeneralizedEigenSolver<MatrixXd> ges(Hamiltonian, Overlap);
  eigenvalues = ges.eigenvalues();
  eigenvectors = ges.eigenvectors();
  betas = ges.betas();
} 

void SelfAdjointEigen(MatrixXd& Overlap, VectorXd& eigenvalues, MatrixXd& eigenvectors) {

  SelfAdjointEigenSolver<MatrixXd> oes(Overlap);
  eigenvalues = oes.eigenvalues();
  eigenvectors = oes.eigenvectors();
} 

