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

#include <fstream>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include "igl/slice.h"
#include "igl/slice_into.h"

#include "Determinants.h"
#include "PfaffianWalker.h"
#include "Pfaffian.h"
#include "global.h"
#include "input.h"

using namespace Eigen;

Pfaffian::Pfaffian() 
{
  int norbs = Determinant::norbs;
  //pairMat = MatrixXd::Identity(norbs, norbs) + MatrixXd::Random(norbs, norbs);
  pairMat = MatrixXd::Zero(norbs, norbs);
  readPairMat(pairMat);
}

void Pfaffian::initWalker(PfaffianWalker &walk) const 
{
  walk = PfaffianWalker(*this);
}

void Pfaffian::initWalker(PfaffianWalker &walk, const Determinant &d) const
{
  walk = PfaffianWalker(*this, d);
}

double Pfaffian::Overlap(const PfaffianWalker &walk) const
{
  return walk.getDetOverlap(*this);
}

double Pfaffian::OverlapRatio(int i, int a, const PfaffianWalker& walk, bool doparity) const 
{
  return walk.getDetFactor(i, a, *this);
}

double Pfaffian::OverlapRatio(int I, int J, int A, int B, const PfaffianWalker& walk, bool doparity) const 
{
  //single excitation
  if (J == 0 && B == 0) return OverlapRatio(I, A, walk, doparity);  
  return walk.getDetFactor(I, J, A, B, *this);
}

void Pfaffian::OverlapWithGradient(const PfaffianWalker & walk,
                                 const double &factor,
                                 Eigen::VectorBlock<VectorXd> &grad) const
{  
  walk.OverlapWithGradient(*this, grad, 1.0);
}

void Pfaffian::getVariables(Eigen::VectorBlock<VectorXd> &v) const
{ 
  int norbs = Determinant::norbs;
  for (int i = 0; i < 2 * norbs; i++) {
    for (int j = 0; j < 2 * norbs; j++) 
      v[2 * i * norbs + j] = pairMat(i, j);
  }
}

long Pfaffian::getNumVariables() const
{
  int norbs = Determinant::norbs;
  return 4 * norbs * norbs;
}

void Pfaffian::updateVariables(const Eigen::VectorBlock<VectorXd> &v) 
{  
  int norbs = Determinant::norbs;  
  for (int i = 0; i < 2 * norbs; i++) {
    for (int j = 0; j < 2 * norbs; j++) { 
      pairMat(i, j) = v[2 * i * norbs + j];
      //pairMat(j, i) = pairMat(i,j);  
    }
  }
  MatrixXd transpose = pairMat.transpose();
  pairMat = (pairMat - transpose)/2;
}

void Pfaffian::printVariables() const
{
  cout << endl << "pairMat" << endl;
  for (int i = 0; i < pairMat.rows(); i++) {
    for (int j = 0; j < pairMat.rows(); j++)
      cout << "  " << pairMat(i, j);
    cout << endl;
  }
  cout << endl;
}

