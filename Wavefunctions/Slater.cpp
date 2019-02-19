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
#include "Slater.h"
#include "global.h"
#include "input.h"


using namespace Eigen;

Slater::Slater() 
{
  initHforbs();
  initDets();
}

int Slater::getNumOfDets() const {return determinants.size();}

//void Slater::initWalker(HFWalker &walk) const 
//{
//walk = HFWalker(*this);
//}

//void Slater::initWalker(HFWalker &walk, const Determinant &d) const
//{
//walk = HFWalker(*this, d);
//}

void Slater::initHforbs() 
{
  int norbs = Determinant::norbs;
  int size; //dimension of the mo coeff matrix
  //initialize hftype and hforbs
  if (schd.hf == "uhf") {
    hftype = HartreeFock::UnRestricted;
    MatrixXcd Hforbs = MatrixXcd::Zero(norbs, 2*norbs);
    readMat(Hforbs, "hf.txt");
    if (schd.ifComplex && Hforbs.imag().isZero(0)) Hforbs.imag() = 0.01 * MatrixXd::Random(norbs, 2*norbs);
    HforbsA = Hforbs.block(0, 0, norbs, norbs);
    HforbsB = Hforbs.block(0, norbs, norbs, norbs);
  }
  else {
    if (schd.hf == "rhf") {
      hftype = HartreeFock::Restricted;
      size = norbs;
    }
    else if (schd.hf == "ghf") {
      hftype = HartreeFock::Generalized;
      size = 2*norbs;
    }
    MatrixXcd Hforbs = MatrixXcd::Zero(size, size);
    readMat(Hforbs, "hf.txt");
    if (schd.ifComplex && Hforbs.imag().isZero(0)) Hforbs.imag() = 0.01 * MatrixXd::Random(size, size);
    HforbsA = Hforbs;
    HforbsB = Hforbs;
  }
}

void Slater::initDets() 
{  
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;
  
  //initialize slater determinants
  if (boost::iequals(schd.determinantFile, "") || boost::iequals(schd.determinantFile, "bestDet")) {
    determinants.resize(1);
    ciExpansion.resize(1, 1.0);
    if (schd.hf == "rhf" || schd.hf == "uhf") {
      for (int i = 0; i < nalpha; i++)
        determinants[0].setoccA(i, true);
      for (int i = 0; i < nbeta; i++)
        determinants[0].setoccB(i, true);
    }
    else {
    //jailbreaking existing dets for ghf use, filling alpha orbitals first and then beta, a ghf det wrapper maybe cleaner
      int nelec = nalpha + nbeta;
      if (nelec <= norbs) {
        for (int i = 0; i < nelec; i++)
          determinants[0].setoccA(i, true);
      }
      else  {
        for (int i = 0; i < norbs; i++)
          determinants[0].setoccA(i, true);
        for (int i = 0; i < nelec-norbs; i++)
          determinants[0].setoccB(i, true);
      }
    }
  }
  else {
    readDeterminants(schd.determinantFile, determinants, ciExpansion);
  }
}

void Slater::getVariables(Eigen::VectorBlock<VectorXd> &v) const
{ 
  int norbs = Determinant::norbs;  
  for (int i = 0; i < determinants.size(); i++)
    v[i] = ciExpansion[i];
  int numDeterminants = determinants.size();
  if (hftype == Generalized) {
    for (int i = 0; i < 2*norbs; i++) {
      for (int j = 0; j < 2*norbs; j++) { 
          v[numDeterminants + 4 * i * norbs + 2 * j] = HforbsA(i, j).real();
          v[numDeterminants + 4 * i * norbs + 2 * j + 1] = HforbsA(i, j).imag();
      }
    }
  }
  else {
    for (int i = 0; i < norbs; i++) {
      for (int j = 0; j < norbs; j++) {
        if (hftype == Restricted) {
          v[numDeterminants + 2 * i * norbs + 2 * j] = HforbsA(i, j).real();
          v[numDeterminants + 2 * i * norbs + 2 * j + 1] = HforbsA(i, j).imag();
        }
        else {
          v[numDeterminants + 2 * i * norbs + 2 * j] = HforbsA(i, j).real();
          v[numDeterminants + 2 * i * norbs + 2 * j + 1] = HforbsA(i, j).imag();
          v[numDeterminants + 2 * norbs * norbs + 2 * i * norbs + 2 * j] = HforbsB(i, j).real();
          v[numDeterminants + 2 * norbs * norbs + 2 * i * norbs + 2 * j + 1] = HforbsB(i, j).imag();
        }
      }
    }
  }
}

long Slater::getNumVariables() const
{
  long numVars = 0;
  numVars += determinants.size();
  if (hftype == UnRestricted)
  //if (hftype == 1)
    numVars += 4 * HforbsA.rows() * HforbsA.rows();
  else
    numVars += 2 * HforbsA.rows() * HforbsA.rows();
  return numVars;
}

void Slater::updateVariables(const Eigen::VectorBlock<VectorXd> &v) 
{  
  int norbs = Determinant::norbs;  
  for (int i = 0; i < determinants.size(); i++)
    ciExpansion[i] = v[i];
  int numDeterminants = determinants.size();
  if (hftype == Generalized) {
    for (int i = 0; i < 2*norbs; i++) {
      for (int j = 0; j < 2*norbs; j++) {
          HforbsA(i, j) = std::complex<double>(v[numDeterminants + 4 * i * norbs + 2 * j], v[numDeterminants + 4 * i * norbs + 2 * j + 1]);
          HforbsB(i, j) = HforbsA(i, j);
      }
    } 
  }
  else {
    for (int i = 0; i < norbs; i++) {
      for (int j = 0; j < norbs; j++) {
        if (hftype == Restricted) {
          HforbsA(i, j) = std::complex<double>(v[numDeterminants + 2 * i * norbs + 2 * j], v[numDeterminants + 2 * i * norbs + 2 * j + 1]);
          HforbsB(i, j) = HforbsA(i, j);
        }
        else {
          HforbsA(i, j) = std::complex<double>(v[numDeterminants + 2 * i * norbs + 2 * j], v[numDeterminants + 2 * i * norbs + 2 * j + 1]);
          HforbsB(i, j) = std::complex<double>(v[numDeterminants + 2 * norbs * norbs + 2 * i * norbs + 2 * j], v[numDeterminants + 2 * norbs * norbs + 2 * i * norbs + 2 * j + 1]);
        }
      }
    }
  }
}

void Slater::printVariables() const
{
  cout << endl<<"CI-expansion"<<endl;
  for (int i = 0; i < determinants.size(); i++) {
    cout << "  " << ciExpansion[i] << endl;
  }
  cout << endl<<"DeterminantA"<<endl;
  //for r/ghf
  for (int i = 0; i < HforbsA.rows(); i++) {
    for (int j = 0; j < HforbsA.rows(); j++)
      cout << "  " << HforbsA(i, j);
    cout << endl;
  }
  if (hftype == UnRestricted) {
  //if (hftype == 1) {
    cout << endl
         << "DeterminantB" << endl;
    for (int i = 0; i < HforbsB.rows(); i++) {
      for (int j = 0; j < HforbsB.rows(); j++)
        cout << "  " << HforbsB(i, j);
      cout << endl;
    }
  }
  cout << endl;

}


