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
#include "Pfaffian.h"
#include "global.h"
#include "input.h"

using namespace Eigen;

Pfaffian::Pfaffian() 
{
  int norbs = Determinant::norbs;
  pairMat = MatrixXcd::Zero(2*norbs, 2*norbs);
  readMat(pairMat, "pairMat.txt");
  if (schd.ifComplex && pairMat.imag().isZero(0)) pairMat.imag() = 0.01 * MatrixXd::Random(2*norbs, 2*norbs);
  pairMat = (pairMat - pairMat.transpose().eval()) / 2;
}

void Pfaffian::getVariables(Eigen::VectorBlock<VectorXd> &v) const
{ 
  int norbs = Determinant::norbs;
  for (int i = 0; i < 2 * norbs; i++) {
    for (int j = 0; j < 2 * norbs; j++) { 
      v[4 * i * norbs + 2 * j] = pairMat(i, j).real();
      v[4 * i * norbs + 2 * j + 1] = pairMat(i, j).imag();
    }
  }
}

long Pfaffian::getNumVariables() const
{
  int norbs = Determinant::norbs;
  return 8 * norbs * norbs;
}

void Pfaffian::updateVariables(const Eigen::VectorBlock<VectorXd> &v) 
{  
  int norbs = Determinant::norbs;  
  for (int i = 0; i < 2 * norbs; i++) {
    for (int j = 0; j < 2 * norbs; j++) { 
      pairMat(i, j) = std::complex<double>(v[4 * i * norbs + 2 * j], v[4 * i * norbs + 2 * j + 1]);
      //pairMat(j, i) = pairMat(i,j);
    }
  }
  pairMat = (pairMat - pairMat.transpose().eval())/2;
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

