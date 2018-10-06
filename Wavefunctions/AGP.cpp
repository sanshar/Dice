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
#include "AGPWalker.h"
#include "AGP.h"
#include "global.h"
#include "input.h"

using namespace Eigen;

AGP::AGP() 
{
  int norbs = Determinant::norbs;
  //pairMat = MatrixXd::Identity(norbs, norbs) + MatrixXd::Random(norbs, norbs);
  pairMat = MatrixXd::Zero(norbs, norbs);
  readPairMat(pairMat);
}

void AGP::initWalker(AGPWalker &walk) const 
{
  walk = AGPWalker(*this);
}

void AGP::initWalker(AGPWalker &walk, const Determinant &d) const
{
  walk = AGPWalker(*this, d);
}

double AGP::Overlap(const AGPWalker &walk) const
{
  return walk.getDetOverlap(*this);
}

double AGP::OverlapRatio(int i, int a, const AGPWalker& walk, bool doparity) const 
{
  return walk.getDetFactor(i, a, *this);
}

double AGP::OverlapRatio(int I, int J, int A, int B, const AGPWalker& walk, bool doparity) const 
{
  //single excitation
  if (J == 0 && B == 0) return OverlapRatio(I, A, walk, doparity);  
  return walk.getDetFactor(I, J, A, B, *this);
}

void AGP::OverlapWithGradient(const AGPWalker & walk,
                                 const double &factor,
                                 Eigen::VectorBlock<VectorXd> &grad) const
{  
  walk.OverlapWithGradient(*this, grad, 1.0);
}

void AGP::getVariables(Eigen::VectorBlock<VectorXd> &v) const
{ 
  int norbs = Determinant::norbs;
  for (int i = 0; i < norbs; i++) {
    for (int j = 0; j < norbs; j++) 
      v[i * norbs + j] = pairMat(i, j);
  }
}

long AGP::getNumVariables() const
{
  int norbs = Determinant::norbs;
  return norbs * norbs;
}

void AGP::updateVariables(const Eigen::VectorBlock<VectorXd> &v) 
{  
  int norbs = Determinant::norbs;  
  for (int i = 0; i < norbs; i++) {
    for (int j = 0; j < norbs; j++) { 
      pairMat(i, j) = v[i * norbs + j];
      //pairMat(j, i) = pairMat(i,j);  
    }
  }
  //cout << "before\n" << endl;
  //printVariables();
  //MatrixXd transpose = pairMat.transpose();
  //pairMat = (pairMat + transpose)/2;
  //cout << "after\n" << endl;
  //printVariables();
}

void AGP::printVariables() const
{
  cout << endl << "pairMat" << endl;
  for (int i = 0; i < pairMat.rows(); i++) {
    for (int j = 0; j < pairMat.rows(); j++)
      cout << "  " << pairMat(i, j);
    cout << endl;
  }
  cout << endl;
}

