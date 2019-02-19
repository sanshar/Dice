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
#include "AGP.h"
#include "global.h"
#include "input.h"

using namespace Eigen;

AGP::AGP() 
{
  int norbs = Determinant::norbs;
  pairMat = MatrixXcd::Zero(norbs, norbs);
  readMat(pairMat, "pairMat.txt");
  if (schd.ifComplex && pairMat.imag().isZero(0)) pairMat.imag() = 0.01 * MatrixXd::Random(norbs, norbs);
  pairMat = (pairMat + pairMat.transpose().eval()) / 2;
}

void AGP::getVariables(Eigen::VectorBlock<VectorXd> &v) const
{ 
  int norbs = Determinant::norbs;
  for (int i = 0; i < norbs; i++) {
    for (int j = 0; j < norbs; j++) { 
      v[2 * i * norbs + 2 * j] = pairMat(i, j).real();
      v[2 * i * norbs + 2 * j + 1] = pairMat(i, j).imag();
    }
  }
}

long AGP::getNumVariables() const
{
  int norbs = Determinant::norbs;
  return 2 * norbs * norbs;
}

void AGP::updateVariables(const Eigen::VectorBlock<VectorXd> &v) 
{  
  int norbs = Determinant::norbs;  
  for (int i = 0; i < norbs; i++) {
    for (int j = 0; j < norbs; j++) { 
      pairMat(i, j) = std::complex<double>(v[2 * i * norbs + 2 * j], v[2 * i * norbs + 2 * j + 1]);
      //pairMat(j, i) = pairMat(i,j);
    }
  }
  if (!schd.uagp) pairMat = (pairMat + pairMat.transpose().eval())/2;
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

