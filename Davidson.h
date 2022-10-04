/*
  Developed by Sandeep Sharma with contributions from James E. T. Smith and Adam
  A. Holmes, 2017 Copyright (c) 2017, Sandeep Sharma

  This file is part of DICE.

  This program is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or (at your option) any later
  version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with
  this program. If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef DAVIDSON_HEADER_H
#define DAVIDSON_HEADER_H
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "global.h"

class Hmult2;
class HmultDirect;
using namespace Eigen;
using namespace std;

void precondition(MatrixXx& r, MatrixXx& diag, double& e);

vector<double> davidson(Hmult2& H, vector<MatrixXx>& x0, MatrixXx& diag,
                        int maxCopies, double tol, int& numIter, bool print);

vector<double> davidsonDirect(HmultDirect& Hdirect, vector<MatrixXx>& x0,
                              MatrixXx& diag, int maxCopies, double tol,
                              int& numIter, bool print);

//(H0-E0)*x0 = b   and proj is used to keep the solution orthogonal to projc 
//E0 can be complex, in which case a normailized system (H0-E0)^{\dagger}(H0-E0)*x0=(H0-E0)^{\dagger}*b is solved using cg
double LinearSolver(Hmult2& H, CItype E0, MatrixXx& x0, MatrixXx& b, vector<CItype*>& p, double tol, bool print);

//(H0-E0)*x0 = b   and proj is used to keep the solution orthogonal to projc 
//E0 can be complex, in which case a normailized system (H0-E0)^{\dagger}(H0-E0)*x0=(H0-E0)^{\dagger}*b is solved using cg
double LinearSolver(Hmult2& H, MatrixXx& diag, CItype E0, MatrixXx& x0, MatrixXx& b, vector<CItype*>& p, double tol, bool print);

//(H0-E0)*x0 = b   and proj is used to keep the solution orthogonal to projc 
//solving using eigen with CG same as above
double LinearSolver(MatrixXx& H, CItype E0, MatrixXx& x0, MatrixXx& b, vector<CItype*>& p, double tol, bool print);

#endif
