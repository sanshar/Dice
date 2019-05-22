/*
  Developed by Sandeep Sharma
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
#include "RBM.h"
#include "Correlator.h"
#include "Determinants.h"
#include <boost/container/static_vector.hpp>
#include <fstream>
#include "input.h"

using namespace Eigen;

RBM::RBM () {    
  int norbs = Determinant::norbs;
  numHidden = schd.hidden;
  wMat = MatrixXd::Random(numHidden, 2*norbs) / 50;
  bVec = VectorXd::Zero(numHidden);
  aVec = VectorXd::Zero(2*norbs);
  
  bool readRBM = false;
  char file[5000];
  sprintf(file, "RBM.txt");
  ifstream ofile(file);
  if (ofile)
    readRBM = true;
  if (readRBM) {
    for (int i = 0; i < wMat.rows(); i++) {
      for (int j = 0; j < wMat.cols(); j++){
        ofile >> wMat(i, j);
      }
    }
  }
};

double RBM::Overlap(const Determinant &d) const
{
  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);

  double ovlp = 1.0;
  VectorXd wn = VectorXd::Zero(numHidden);
  for (int j = 0; j < closed.size(); j++) {
    ovlp *= exp(aVec(closed[j]));
    wn += wMat.col(closed[j]);
  }
  ovlp *= cosh((bVec + wn).array()).prod();
  return ovlp;
}


double RBM::OverlapRatio (const Determinant &d1, const Determinant &d2) const {
  return Overlap(d1)/Overlap(d2);
}


double RBM::OverlapRatio(int i, int a, const Determinant &dcopy, const Determinant &d) const
{
  return OverlapRatio(dcopy, d);
}

double RBM::OverlapRatio(int i, int j, int a, int b, const Determinant &dcopy, const Determinant &d) const
{
  return OverlapRatio(dcopy, d);
}

//assuming bwn corresponds to d
void RBM::OverlapWithGradient(const Determinant& d, 
                              Eigen::VectorBlock<VectorXd>& grad,
                              const double& ovlp) const {
  int norbs = Determinant::norbs;
  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);
  ArrayXd tanhbwn = tanh(bwn.array());
  if (schd.optimizeCps) {
    grad.segment(numHidden * 2 * norbs, numHidden) = tanhbwn.matrix(); //b derivatives
    for (int j = 0; j < closed.size(); j++) {
      grad(numHidden * 2 * norbs + numHidden + closed[j]) = 1; //a derivatives
      grad.segment(numHidden * closed[j], numHidden) += tanhbwn.matrix();//w derivatives
    }
  }
}

long RBM::getNumVariables() const
{
  return wMat.rows() * wMat.cols() + bVec.size() + aVec.size();
}


void RBM::getVariables(Eigen::VectorBlock<VectorXd> &v) const
{
  int numVars = 0;
  for (int j = 0; j < wMat.cols(); j++) {
    for (int i = 0; i < wMat.rows(); i++) {
      v[numVars] = wMat(i,j);
      numVars++;
    }
  }

  for (int i = 0; i < bVec.size(); i++) {
    v[numVars] = bVec(i);
    numVars++;
  }

  for (int i = 0; i < aVec.size(); i++) {
    v[numVars] = aVec(i);
    numVars++;
  }
}

void RBM::updateVariables(const Eigen::VectorBlock<VectorXd> &v)
{
  int numVars = 0;
  for (int j = 0; j < wMat.cols(); j++) {
    for (int i = 0; i < wMat.rows(); i++) {
      wMat(i,j) = v[numVars];
      numVars++;
    }
  }

  for (int i = 0; i < bVec.size(); i++) {
    bVec(i) = v[numVars];
    numVars++;
  }

  for (int i = 0; i < aVec.size(); i++) {
    aVec(i) = v[numVars];
    numVars++;
  }
}

void RBM::printVariables() const
{
  cout << "RBM\n";
  cout << "wMat\n" << wMat << endl << endl;
  cout << "bVec\n" << bVec << endl << endl;
  cout << "aVec\n" << aVec << endl << endl;
}
