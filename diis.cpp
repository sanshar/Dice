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
#include "diis.h"
#include <iostream>
#include "global.h"

using namespace Eigen;
using namespace std;

DIIS::DIIS(int pmaxDim, int pvectorDim) : maxDim(pmaxDim), vectorDim(pvectorDim) {
  
  prevVectors  = MatrixXd::Zero(pvectorDim, pmaxDim);
  errorVectors = MatrixXd::Zero(pvectorDim, pmaxDim);
  diisMatrix   = MatrixXd::Zero(pmaxDim+1, pmaxDim+1);
  iter = 0;
  for (int i=0; i<maxDim; i++) {
    diisMatrix(i, maxDim) = 1.0;
    diisMatrix(maxDim, i) = 1.0;
  }
}

void DIIS::init(int pmaxDim, int pvectorDim) {
  maxDim = pmaxDim; vectorDim = pvectorDim;
  
  prevVectors  = MatrixXd::Zero(pvectorDim, pmaxDim);
  errorVectors = MatrixXd::Zero(pvectorDim, pmaxDim);
  diisMatrix   = MatrixXd::Zero(pmaxDim+1, pmaxDim+1);
  iter = 0;
  for (int i=0; i<maxDim; i++) {
    diisMatrix(i, maxDim) = -1.0;
    diisMatrix(maxDim, i) = 1.0;
  }
}


void DIIS::update(VectorXd& newV, VectorXd& errorV) {

  prevVectors .col(iter%maxDim)  = newV;
  errorVectors.col(iter%maxDim)  = errorV;

  int col = iter%maxDim;
  for (int i=0; i<maxDim; i++) {
    diisMatrix(i   , col) = errorV.transpose()*errorVectors.col(i);
    diisMatrix(col, i   ) = diisMatrix(i, col);
  }
  iter++;
  
  if (iter < maxDim) {
    VectorXd b = VectorXd::Zero(iter+1);
    b[iter] = 1.0;
    MatrixXd localdiis = diisMatrix.block(0,0,iter+1, iter+1);
    for (int i=0; i<iter; i++) {
      localdiis(i, iter) = -1.0;
      localdiis(iter, i) = 1.0;
    }
    VectorXd x = localdiis.colPivHouseholderQr().solve(b);
    newV = prevVectors.block(0,0,vectorDim,iter)*x.head(iter); 
    //+ errorVectors.block(0,0,vectorDim,iter)*x.head(iter);

  }
  else {
    VectorXd b = VectorXd::Zero(maxDim+1);
    b[maxDim] = 1.0;
    VectorXd x = diisMatrix.colPivHouseholderQr().solve(b);
    newV = prevVectors*x.head(maxDim);// + errorVectors*x.head(maxDim);
    
    //prevVectors.col((iter-1)%maxDim) = 1.* newV;
  }
}
