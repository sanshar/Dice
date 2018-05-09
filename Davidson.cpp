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
#include "Hmult.h"
#include <Eigen/Dense>
#include <iostream>
#include <iostream>
#include "boost/format.hpp"
#include "iowrapper.h"
#include "global.h"

using namespace Eigen;
using namespace std;
using namespace boost;

double LinearSolver(Hmult2& H, CItype E0, MatrixXx& x0, MatrixXx& b, vector<CItype*>& proj, double tol, bool print) {

  for (int i=0; i<proj.size(); i++) {
    CItype dotProduct = 0.0, norm=0.0;
    for (int j=0; j<b.rows(); j++) {
      dotProduct += proj[i][j]*b(j,0);
      norm += proj[i][j]*proj[i][j];
    }
    for (int j=0; j<b.rows(); j++)
      b(j,0) = b(j,0) - dotProduct*proj[i][j]/norm;
  }

  x0.setZero(b.rows(),1);
  MatrixXx r = 1.*b, p = 1.*b;
  double rsold = r.squaredNorm();

  if (fabs(r.norm()) < tol) return 0.0;

  int iter = 0;
  while (true) {
    MatrixXx Ap = 0.*p;
    H(&p(0,0), &Ap(0,0)); ///REPLACE THIS WITH SOMETHING
#ifndef SERIAL
    MPI_Allreduce(MPI_IN_PLACE, &Ap(0,0), Ap.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    Ap = Ap - E0*p; //H0-E0
    CItype alpha = rsold/(p.adjoint()*Ap)(0,0);
    x0 += alpha*p;
    r -= alpha*Ap;

    for (int i=0; i<proj.size(); i++) {
      CItype dotProduct = 0.0, norm=0.0;
      for (int j=0; j<b.rows(); j++) {
        dotProduct += proj[i][j]*r(j,0);
        norm += proj[i][j]*proj[i][j];
      }
      for (int j=0; j<r.rows(); j++)
        r(j,0) = r(j,0) - dotProduct*proj[i][j]/norm;
    }

    //r = r - ((proj[i].adjoint()*r)(0,0))*proj[i]/((proj[i].adjoint()*proj[i])(0,0));
    //r = r- ((proj.adjoint()*r)(0,0))*proj/((proj.adjoint()*proj)(0,0));

    double rsnew = r.squaredNorm();
    CItype ept = -(x0.adjoint()*r + x0.adjoint()*b)(0,0);
    if (commrank==0)
      cout <<"#"<< iter<<" "<<ept<<"  "<<rsnew<<std::endl;
    if (r.norm() < tol || iter > 200) {
      p.setZero(p.rows(),1);
      H(&x0(0,0), &p(0,0)); ///REPLACE THIS WITH SOMETHING
      p -=b;
      return abs(ept);
    }

    p = r +(rsnew/rsold)*p;
    rsold = rsnew;
    iter++;
  }

}

