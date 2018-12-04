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
#include "Gutzwiller.h"
#include "Correlator.h"
#include "Determinants.h"
#include <boost/container/static_vector.hpp>
#include <fstream>
#include "input.h"

using namespace Eigen;

Gutzwiller::Gutzwiller () {    
  //g = (VectorXd::Constant(Determinant::norbs, 1.0) + VectorXd::Random(Determinant::norbs))/20;
  int norbs = Determinant::norbs;
  g = VectorXd::Constant(Determinant::norbs, 1.);
  bool readGutz = false;
  char file[5000];
  sprintf(file, "Gutzwiller.txt");
  ifstream ofile(file);
  if (ofile)
    readGutz = true;
  if (readGutz) {
    for (int i = 0; i < norbs; i++) {
      ofile >> g(i);
    }
  }
};


double Gutzwiller::Overlap(const Determinant &d) const
{
  int norbs = Determinant::norbs;
  double ovlp = 1.0;
  for (int i = 0; i < norbs; i++) {
    if (d.getoccA(i) && d.getoccB(i)) ovlp *= g(i);
  }
  return ovlp;
}


double Gutzwiller::OverlapRatio (const Determinant &d1, const Determinant &d2) const {
  return Overlap(d1)/Overlap(d2);
}


double Gutzwiller::OverlapRatio(int i, int a, const Determinant &dcopy, const Determinant &d) const
{
  return OverlapRatio(dcopy, d);
}

double Gutzwiller::OverlapRatio(int i, int j, int a, int b, const Determinant &dcopy, const Determinant &d) const
{
  return OverlapRatio(dcopy, d);
}

void Gutzwiller::OverlapWithGradient(const Determinant& d, 
                              VectorXd& grad,
                              const double& ovlp) const {
  if (schd.optimizeCps) {
    int norbs = Determinant::norbs;
    for (int i = 0; i < norbs; i++) {
      if (d.getoccA(i) && d.getoccB(i)) grad[i] = 1/g(i);
    }
  }
}

long Gutzwiller::getNumVariables() const
{
  return Determinant::norbs;
}


void Gutzwiller::getVariables(Eigen::VectorXd &v) const
{
  for (int i = 0; i < Determinant::norbs; i++)
    v[i] = g[i];
}

void Gutzwiller::updateVariables(const Eigen::VectorXd &v)
{
  for (int i = 0; i < Determinant::norbs; i++)
    g[i] = v[i];
}

void Gutzwiller::printVariables() const
{
  cout << "Gutzwiller"<< endl;
  //for (int i=0; i<SpinCorrelator.rows(); i++)
  //  for (int j=0; j<=i; j++)
  //    cout << SpinCorrelator(i,j);
  cout << g << endl << endl;
}
