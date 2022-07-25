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

#include "Determinants.h"
#include "MultiSlater.h"
#include "global.h"
#include "input.h"


using namespace Eigen;

MultiSlater::MultiSlater()
{
  initHforbs();
  initCiExpansion();
}

void MultiSlater::initHforbs()
{
  ifstream dump("hf.txt");
  if (dump) {
    int norbs = Determinant::norbs;
    Hforbs = MatrixXcd::Zero(2*norbs, 2*norbs);
    readMat(Hforbs, "hf.txt");
    if (schd.ifComplex && Hforbs.imag().isZero(0)) Hforbs.imag() = 0.1 * MatrixXd::Random(2*norbs, 2*norbs) / norbs;
  }
}

void MultiSlater::initCiExpansion()
{
  string fname = "dets";
  if (schd.ghfDets) readDeterminantsGHF(fname, ref, open, ciExcitations, ciParity, ciCoeffs);
  else readDeterminants(fname, ref, open, ciExcitations, ciParity, ciCoeffs);
  numDets = ciCoeffs.size();
}

void MultiSlater::getVariables(Eigen::VectorBlock<VectorXd> &v) const
{
  int norbs = Determinant::norbs;
  for (size_t i = 0; i < numDets; i++) v[i] = ciCoeffs[i];
  for (int i = 0; i < 2*norbs; i++) {
    for (int j = 0; j < 2*norbs; j++) {
        v[numDets + 4 * i * norbs + 2 * j] = Hforbs(i, j).real();
        v[numDets + 4 * i * norbs + 2 * j + 1] = Hforbs(i, j).imag();
    }
  }
}

size_t MultiSlater::getNumVariables() const
{
  return numDets + 2 * Hforbs.rows() * Hforbs.rows();
}

void MultiSlater::updateVariables(const Eigen::VectorBlock<VectorXd> &v)
{
  int norbs = Determinant::norbs;
  for (size_t i = 0; i < numDets; i++) ciCoeffs[i] = v[i];
  for (int i = 0; i < 2*norbs; i++) {
    for (int j = 0; j < 2*norbs; j++) {
        Hforbs(i, j) = std::complex<double>(v[numDets + 4 * i * norbs + 2 * j], v[numDets + 4 * i * norbs + 2 * j + 1]);
    }
  }
}

void MultiSlater::printVariables() const
{
  cout << "\nciCoeffs\n";
  for (size_t i = 0; i < numDets; i++) cout << ciCoeffs[i] << endl;
  cout << "\nHforbs\n";
  for (int i = 0; i < Hforbs.rows(); i++) {
    for (int j = 0; j < Hforbs.rows(); j++) cout << "  " << Hforbs(i, j);
    cout << endl;
  }
  cout << endl;
}
