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

#include "HFWalker.h"
#include "Slater.h"
#include "global.h"
#include "input.h"
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

using namespace Eigen;

Slater::Slater() { readDefault();}

void Slater::readDefault() {
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

  HforbsA = MatrixXd::Zero(norbs, norbs);
  HforbsB = MatrixXd::Zero(norbs, norbs);

  readHF(HforbsA, HforbsB, schd.uhf);

  if (boost::iequals(schd.determinantFile, ""))
  {
    determinants.resize(1);
    ciExpansion.resize(1, 1.0);
    for (int i = 0; i < nalpha; i++)
      determinants[0].setoccA(i, true);
    for (int i = 0; i < nbeta; i++)
      determinants[0].setoccB(i, true);
  }
  else
  {
    readDeterminants(schd.determinantFile, determinants, ciExpansion);
  }

}


void Slater::initWalker(HFWalker& walk) {


  //initialize the walker
  Determinant& d = walk.d;
  bool readDeterminant = false;
  char file[5000];

  sprintf(file, "BestDeterminant.txt");

  {
    ifstream ofile(file);
    if (ofile)
      readDeterminant = true;
  }
  //readDeterminant = false;

  if (readDeterminant)
    readBestDeterminant(d);
  else
    guessBestDeterminant(d);
    
  walk.initUsingWave(*this); 
}


void Slater::initWalker(HFWalker& walk, Determinant& d) {
  walk.d = d;
  walk.initUsingWave(*this);
}

void Slater::getDetMatrix(Determinant &d, Eigen::MatrixXd &DetAlpha, Eigen::MatrixXd &DetBeta)
{
  //alpha and beta orbitals of the walker determinant d
  std::vector<int> alpha, beta;
  d.getAlphaBeta(alpha, beta);
  int nalpha = alpha.size(), nbeta = beta.size();

  //alpha and beta orbitals of the reference determinant
  std::vector<int> alphaRef, betaRef;
  determinants[0].getAlphaBeta(alphaRef, betaRef);

  Eigen::Map<VectorXi> RowAlpha(&alpha[0], alpha.size()),
      RowBeta (&beta [0], beta .size()),
      ColAlpha(&alphaRef[0], alphaRef.size()),
      ColBeta (&betaRef[0], betaRef.size());

  DetAlpha = MatrixXd::Zero(nalpha, nalpha);
  DetBeta = MatrixXd::Zero(nbeta, nbeta);

  igl::slice(HforbsA, RowAlpha, ColAlpha, DetAlpha);
  igl::slice(HforbsB, RowBeta, ColBeta, DetBeta);
  
  return;
}

double Slater::Overlap(HFWalker &walk)
{
  return walk.getDetOverlap(*this);
}

//This is expensive and not recommended
double Slater::Overlap(Determinant &d)
{
  Eigen::MatrixXd DetAlpha, DetBeta;
  getDetMatrix(d, DetAlpha, DetBeta);
  return DetAlpha.determinant() * DetBeta.determinant();
}



double Slater::OverlapRatio(int i, int a, HFWalker& walk, bool doparity) {
  return walk.getDetFactor(i, a, *this, doparity);
}

double Slater::OverlapRatio(int I, int J, int A, int B, HFWalker& walk, bool doparity) {
  //singleexcitation
  if (J == 0 && B == 0) return OverlapRatio(I, A, walk, doparity);  
  return walk.getDetFactor(I, J, A, B, *this, doparity);
}


void Slater::OverlapWithGradient(HFWalker & walk,
                                 double &factor,
                                 Eigen::VectorBlock<VectorXd> &grad) {
  int norbs = Determinant::norbs;
  
  double detovlp = walk.getDetOverlap(*this);
    
  for (int k = 0; k < ciExpansion.size(); k++)
    grad[k] += walk.alphaDet[k] * walk.betaDet[k] / detovlp;
    
    
  if (determinants.size() <= 1 && schd.optimizeOrbs)
  {
    VectorXd gradOrbitals = schd.uhf ? VectorXd::Zero(2*norbs*norbs) : VectorXd::Zero(norbs*norbs);
    walk.OverlapWithGradient(*this, gradOrbitals, detovlp);

    for (int i=0; i<gradOrbitals.size(); i++)
      grad[ciExpansion.size() + i] += gradOrbitals[i];

  }
};

void Slater::getVariables(Eigen::VectorBlock<VectorXd> &v) {
  int norbs = Determinant::norbs;
    
  for (int i = 0; i < determinants.size(); i++)
    v[i] = ciExpansion[i];
    
  int numDeterminants = determinants.size();
    
  for (int i = 0; i < norbs; i++)
    for (int j = 0; j < norbs; j++)
      if (!schd.uhf)
      {
        v[numDeterminants + i * norbs + j] = HforbsA(i, j);
        v[numDeterminants + i * norbs + j] = HforbsB(i, j);
      }
      else
      {
        v[numDeterminants + i * norbs + j] = HforbsA(i, j);
        v[numDeterminants + norbs * norbs + i * norbs + j] = HforbsB(i, j);
      }
};


void Slater::updateVariables(Eigen::VectorBlock<VectorXd> &v) {
  int norbs = Determinant::norbs;
    
  for (int i = 0; i < determinants.size(); i++)
    ciExpansion[i] = v[i];
    
  int numDeterminants = determinants.size();
    
  for (int i = 0; i < norbs; i++)
    for (int j = 0; j < norbs; j++)
      if (!schd.uhf)
      {
        HforbsA(i, j) = v[numDeterminants + i * norbs + j];
        HforbsB(i, j) = v[numDeterminants + i * norbs + j];
      }
      else
      {
        HforbsA(i, j) = v[numDeterminants + i * norbs + j];
        HforbsB(i, j) = v[numDeterminants + norbs * norbs + i * norbs + j];
      }
};

void Slater::printVariables()
{
  cout << endl<<"CI-expansion"<<endl;
  for (int i = 0; i < determinants.size(); i++)
  {
    cout << "  " << ciExpansion[i] << endl;
  }

  cout << endl<<"DeterminantA"<<endl;
  int norbs = Determinant::norbs;
  for (int i = 0; i < norbs; i++) {
    for (int j = 0; j < norbs; j++)
      cout << "  " << HforbsA(i, j);
    cout << endl;
  }

  if (schd.uhf)
  {
    cout << endl
         << "DeterminantB" << endl;
    for (int i = 0; i < norbs; i++) {
      for (int j = 0; j < norbs; j++)
        cout << "  " << HforbsB(i, j);
      cout << endl;
    }
  }

  cout << endl;
}


long Slater::getNumVariables()
{
  int norbs = Determinant::norbs;
  long numVars = 0;

  numVars += determinants.size();
  if (schd.uhf)
    numVars += 2 * norbs * norbs;
  else
    numVars += norbs * norbs;

  return numVars;
}




void Slater::readBestDeterminant(Determinant& d) {
  if (commrank == 0)
  {
    char file[5000];
    sprintf(file, "BestDeterminant.txt");
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    load >> d;
  }
#ifndef SERIAL
  MPI_Bcast(&d.reprA, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&d.reprB, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
}
  
void Slater::guessBestDeterminant(Determinant& d) {
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

  d = Determinant();
  for (int i = 0; i < nalpha; i++) {
    int bestorb = 0;
    double maxovlp = 0;
    for (int j = 0; j < norbs; j++)
    {
      if (abs(HforbsA(i, j)) > maxovlp && !d.getoccA(j))
      {
        maxovlp = abs(HforbsA(i, j));
        bestorb = j;
      }
    }
    d.setoccA(bestorb, true);
  }
  for (int i = 0; i < nbeta; i++) {
    int bestorb = 0;
    double maxovlp = 0;
    for (int j = 0; j < norbs; j++)
    {
      if (abs(HforbsB(i, j)) > maxovlp && !d.getoccB(j))
      {
        bestorb = j;
        maxovlp = abs(HforbsB(i, j));
      }
    }
    d.setoccB(bestorb, true);
  }
}

