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
  if (schd.hf == "rhf") {
    hftype = HartreeFock::Restricted;
    //hftype = 0;
    size = norbs;
  }
  else if (schd.hf == "uhf") {
    hftype = HartreeFock::UnRestricted;
    //hftype = 1;
    size = norbs;
  }
  else if (schd.hf == "ghf") {
    hftype = HartreeFock::Generalized;
    //hftype = 2;
    size = 2*norbs;
  }
  HforbsA = MatrixXd::Zero(size, size);
  HforbsB = MatrixXd::Zero(size, size);
  readHF(HforbsA, HforbsB, schd.hf);
}

void Slater::initDets() 
{  
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;
  
  //initialize slater determinants
  if (boost::iequals(schd.determinantFile, "")) {
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

/*
double Slater::Overlap(const HFWalker &walk) const
{
  return walk.getDetOverlap(*this);
}

double Slater::OverlapRatio(int i, int a, const HFWalker& walk, bool doparity) const 
{
  return walk.getDetFactor(i, a, *this);
}

double Slater::OverlapRatio(int I, int J, int A, int B, const HFWalker& walk, bool doparity) const 
{
  //single excitation
  if (J == 0 && B == 0) return OverlapRatio(I, A, walk, doparity);  
  return walk.getDetFactor(I, J, A, B, *this);
}

void Slater::OverlapWithGradient(const HFWalker & walk,
                                 const double &factor,
                                 Eigen::VectorBlock<VectorXd> &grad) const
{  
  int norbs = Determinant::norbs;
  double detovlp = walk.getDetOverlap(*this);
  for (int k = 0; k < ciExpansion.size(); k++)
    grad[k] += walk.getIndividualDetOverlap(k) / detovlp;
  if (determinants.size() <= 1 && schd.optimizeOrbs) {
    //if (hftype == UnRestricted)
    VectorXd gradOrbitals;
    if (hftype == UnRestricted) {
      gradOrbitals = VectorXd::Zero(2*HforbsA.rows()*HforbsA.rows());
      walk.OverlapWithGradient(*this, gradOrbitals, detovlp);
    }
    else {
      gradOrbitals = VectorXd::Zero(HforbsA.rows()*HforbsA.rows());
      if (hftype == Restricted) walk.OverlapWithGradient(*this, gradOrbitals, detovlp);
      else walk.OverlapWithGradientGhf(*this, gradOrbitals, detovlp);
    }
    for (int i=0; i<gradOrbitals.size(); i++)
      grad[ciExpansion.size() + i] += gradOrbitals[i];
  }
}
*/
void Slater::getVariables(Eigen::VectorBlock<VectorXd> &v) const
{ 
  int norbs = Determinant::norbs;  
  for (int i = 0; i < determinants.size(); i++)
    v[i] = ciExpansion[i];
  int numDeterminants = determinants.size();
  if (hftype == Generalized) {
  //if (hftype == 2) {
    for (int i = 0; i < 2*norbs; i++) {
      for (int j = 0; j < 2*norbs; j++) 
          v[numDeterminants + 2 * i * norbs + j] = HforbsA(i, j);
    }
  }
  else {
    for (int i = 0; i < norbs; i++) {
      for (int j = 0; j < norbs; j++) {
        if (hftype == Restricted) {
        //if (hftype == 0) {
          v[numDeterminants + i * norbs + j] = HforbsA(i, j);
          //v[numDeterminants + i * norbs + j] = HforbsB(i, j);
        }
        else {
          v[numDeterminants + i * norbs + j] = HforbsA(i, j);
          v[numDeterminants + norbs * norbs + i * norbs + j] = HforbsB(i, j);
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
    numVars += 2 * HforbsA.rows() * HforbsA.rows();
  else
    numVars += HforbsA.rows() * HforbsA.rows();
  return numVars;
}

void Slater::updateVariables(const Eigen::VectorBlock<VectorXd> &v) 
{  
  int norbs = Determinant::norbs;  
  for (int i = 0; i < determinants.size(); i++)
    ciExpansion[i] = v[i];
  int numDeterminants = determinants.size();
  if (hftype == Generalized) {
  //if (hftype == 2) {
    for (int i = 0; i < 2*norbs; i++) {
      for (int j = 0; j < 2*norbs; j++) {
          HforbsA(i, j) = v[numDeterminants + 2 * i * norbs + j];
          HforbsB(i, j) = v[numDeterminants + 2 * i * norbs + j];
      }
    } 
  }
  else {
    for (int i = 0; i < norbs; i++) {
      for (int j = 0; j < norbs; j++) {
        if (hftype == Restricted) {
        //if (hftype == 0) {
          HforbsA(i, j) = v[numDeterminants + i * norbs + j];
          HforbsB(i, j) = v[numDeterminants + i * norbs + j];
        }
        else {
          HforbsA(i, j) = v[numDeterminants + i * norbs + j];
          HforbsB(i, j) = v[numDeterminants + norbs * norbs + i * norbs + j];
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


//void Slater::getDetMatrix(Determinant &d, Eigen::MatrixXd &DetAlpha, Eigen::MatrixXd &DetBeta)
//{
//  //alpha and beta orbitals of the walker determinant d
//  std::vector<int> alpha, beta;
//  d.getAlphaBeta(alpha, beta);
//  int nalpha = alpha.size(), nbeta = beta.size();
//
//  //alpha and beta orbitals of the reference determinant
//  std::vector<int> alphaRef, betaRef;
//  determinants[0].getAlphaBeta(alphaRef, betaRef);
//
//  Eigen::Map<VectorXi> RowAlpha(&alpha[0], alpha.size()),
//      RowBeta (&beta [0], beta .size()),
//      ColAlpha(&alphaRef[0], alphaRef.size()),
//      ColBeta (&betaRef[0], betaRef.size());
//
//  DetAlpha = MatrixXd::Zero(nalpha, nalpha);
//  DetBeta = MatrixXd::Zero(nbeta, nbeta);
//
//  igl::slice(HforbsA, RowAlpha, ColAlpha, DetAlpha);
//  igl::slice(HforbsB, RowBeta, ColBeta, DetBeta);
//  
//  return;
//}

//This is expensive and not recommended
//double Slater::Overlap(Determinant &d)
//{
//  Eigen::MatrixXd DetAlpha, DetBeta;
//  getDetMatrix(d, DetAlpha, DetBeta);
//  return DetAlpha.determinant() * DetBeta.determinant();
//}


