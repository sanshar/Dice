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
#include "SJastrow.h"
#include "Correlator.h"
#include "Determinants.h"
#include <boost/container/static_vector.hpp>
#include <fstream>
#include "input.h"

using namespace Eigen;

SJastrow::SJastrow () {    
  int norbs = Determinant::norbs;
  SpinCorrelator = MatrixXd::Constant(norbs, norbs, 0.);
/*
  if (schd.optimizeCps)
    SpinCorrelator += 0.01*MatrixXd::Random(2*norbs, 2*norbs);
*/
  bool readSJastrow = false;
  char file[5000];
  sprintf(file, "SJastrow.txt");
  ifstream ofile(file);
  if (ofile)
    readSJastrow = true;
  if (readSJastrow) {
    for (int i = 0; i < SpinCorrelator.rows(); i++) {
      for (int j = 0; j < SpinCorrelator.rows(); j++){
        ofile >> SpinCorrelator(i, j);
      }
    }
  }
};


double SJastrow::Overlap(const Determinant &d) const
{
  int norbs = Determinant::norbs;
  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);

  VectorXi occ = VectorXi::Zero(norbs);
  for (int i = 0; i < closed.size(); i++) occ[closed[i]/2]++;


  double exponent = 0.;
  for (int i = 0; i < norbs; i++) {
    if (occ[i] > 0) {
      for (int j = 0; j <= i; j++) {
        exponent += SpinCorrelator(i, j) * occ[i] * occ[j];
      }
    }
  }

  return exp(exponent);
}


double SJastrow::OverlapRatio (const Determinant &d1, const Determinant &d2) const {
  return Overlap(d1)/Overlap(d2);
}


double SJastrow::OverlapRatio(int i, int a, const Determinant &dcopy, const Determinant &d) const
{
  return OverlapRatio(dcopy, d);
}

double SJastrow::OverlapRatio(int i, int j, int a, int b, const Determinant &dcopy, const Determinant &d) const
{
  return OverlapRatio(dcopy, d);
}


void SJastrow::OverlapWithGradient(const Determinant& d, 
                              Eigen::VectorBlock<VectorXd>& grad,
                              const double& ovlp) const {
  int norbs = Determinant::norbs;
  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);

  VectorXi occ = VectorXi::Zero(norbs);
  for (int i = 0; i < closed.size(); i++) occ[closed[i]/2]++;

  if (schd.optimizeCps) {
    for (int i = 0; i < norbs; i++) {
      if (occ[i] > 0) {
        for (int j = 0; j <= i; j++) {
          grad[i*(i+1)/2 + j] += occ[i] * occ[j];
        }
      }
    }
  }
}


long SJastrow::getNumVariables() const
{
  long norbs = SpinCorrelator.rows();
  return norbs*(norbs+1)/2;
}


void SJastrow::getVariables(Eigen::VectorBlock<VectorXd> &v) const
{
  int numVars = 0;
  for (int i = 0; i < SpinCorrelator.rows(); i++) {
    for (int j = 0; j <= i; j++) {
      v[numVars] = SpinCorrelator(i,j);
      numVars++;
    }
  }
}

void SJastrow::updateVariables(const Eigen::VectorBlock<VectorXd> &v)
{
  int numVars = 0;
  for (int i = 0; i < SpinCorrelator.rows(); i++) {
    for (int j = 0; j <= i; j++) {
      SpinCorrelator(i,j) = v[numVars];
      numVars++;
    }
  }
}

void SJastrow::addNoise() 
{
  SpinCorrelator += 0.01 * MatrixXd::Random(Determinant::norbs, Determinant::norbs);
}

void SJastrow::printVariables() const
{
  cout << "SJastrow" << endl;
  cout << SpinCorrelator << endl << endl;
}
