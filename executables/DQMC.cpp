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
//#define EIGEN_USE_MKL_ALL
#include <algorithm>
#include <utility>
#include <random>
#include <chrono>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#ifndef SERIAL
#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include <boost/function.hpp>
#include <boost/functional.hpp>
#include <boost/bind.hpp>
#include "Determinants.h"
#include "input.h"
#include "integral.h"
#include "math.h"
#include "Profile.h"
#include "SHCIshm.h"

using namespace Eigen;
using namespace boost;
using namespace std;

using matPair = pair<MatrixXcd, MatrixXcd>;
using vecPair = pair<VectorXcd, VectorXcd>;

// aka transition rdm 
// left is adjointed
void calcGreensFunction(matPair& left, matPair& right, matPair& green) 
{
  green.first = right.first * (left.first * right.first).inverse() * left.first;
  green.second = right.second * (left.second * right.second).inverse() * left.second;
}


// transition Hamiltonian matrix element < phi_1 | H | phi_2 > / < phi_1 | phi_2 >
// green: transition rdm between phi_1 and phi_2
complex<double> calcHamiltonianElement(matPair& green, double enuc, MatrixXd& h1, vector<MatrixXd>& chol) 
{ 
  // core energy
  complex<double> ene = enuc;

  // one body part
  ene += green.first.cwiseProduct(h1).sum() + green.second.cwiseProduct(h1).sum();
  complex<double> e1 = ene - enuc;

  // two body part
  for (int i = 0; i < chol.size(); i++) {
    complex<double> cup = green.first.cwiseProduct(chol[i]).sum();
    complex<double> cdn = green.second.cwiseProduct(chol[i]).sum();
    MatrixXcd eup = chol[i] * green.first;
    MatrixXcd edn = chol[i] * green.second;
    ene += (cup * cup + cdn * cdn + 2 * cup * cdn - eup.cwiseProduct(eup.transpose()).sum() - edn.cwiseProduct(edn.transpose()).sum()) / 2;
  }

  return ene;
}


// reads jastrow in our VMC format (not exponential)
// makes hs operators including mean field substraction and the offshoot one body operator
// the operators are diagonal matrices in the Jastrow basis, represented using vectors 
void prepJastrowHS(matPair& ref, vector<vecPair>& hsOperators, vecPair& oneBodyOperator)
{
  size_t norbs = Determinant::norbs;
  
  // read jastrow
  MatrixXd jastrow = MatrixXd::Zero(2*norbs, 2*norbs);
  readMat(jastrow, "Jastrow.txt");

  // VMC format: only lower triangle + diagonal has information
  // even indices are up, odd are down
  // convert to exponential form
  MatrixXd expJastrow = MatrixXd::Zero(2*norbs, 2*norbs);
  for (int i = 0; i < 2*norbs; i++) 
    expJastrow(i, i) = log(jastrow(i, i));
  for (int i = 0; i < 2*norbs; i++) { 
    for (int j = 0; j < i; j++) {
      expJastrow(i, j) = log(jastrow(i, j))/2;
      expJastrow(j, i) = log(jastrow(i, j))/2;
    }
  }

  // calculate hs operators by diagonalizing expJastrow
  SelfAdjointEigenSolver<MatrixXd> eigensolver(expJastrow);
  VectorXd eigenvalues = eigensolver.eigenvalues();
  MatrixXd eigenvectors = eigensolver.eigenvectors();
 
  oneBodyOperator.first = VectorXcd::Zero(norbs);
  oneBodyOperator.second = VectorXcd::Zero(norbs);

  // calculate rdm for mean field shifts
  matPair refT;
  refT.first = ref.first.adjoint();
  refT.second = ref.second.adjoint();
  matPair green;
  calcGreensFunction(refT, ref, green);
  
  for (int i = 0; i < 2*norbs; i++) {
    VectorXcd up(norbs), dn(norbs);
    for (int j = 0; j < 2*norbs; j++) {
      if (j%2 == 0) up(j/2) = eigenvectors(j, i);
      else dn(j/2) = eigenvectors(j, i);
    }
    
    // calculate shifts
    complex<double> mfShiftUp = 1. * green.first.diagonal().cwiseProduct(up).sum();
    complex<double> mfShiftDn = 1. * green.second.diagonal().cwiseProduct(dn).sum();
    
    // update one body ops
    oneBodyOperator.first += 1. * 2 * eigenvalues(i) * (mfShiftUp + mfShiftDn) * up;
    oneBodyOperator.second += 1. * 2 * eigenvalues(i) * (mfShiftUp + mfShiftDn) * dn;
    
    // make shifted hs ops
    up -= VectorXcd::Constant(norbs, mfShiftUp/Determinant::nalpha);
    dn -= VectorXcd::Constant(norbs, mfShiftDn/Determinant::nbeta);
    vecPair op;
    op.first = sqrt(complex<double>(2*eigenvalues(i), 0.)) * up;
    op.second = sqrt(complex<double>(2*eigenvalues(i), 0.)) * dn;
    hsOperators.push_back(op);
  }
 
}


// makes hs operators from cholesky matrices including mean field subtraction
void prepPropagatorHS()
{
  return;
}


// calculates variational jastrow slater energy estimate using importance sampling
void calcJastrowEnergyMetropolis(double enuc, MatrixXd& h1, vector<MatrixXd>& chol)
{
  size_t norbs = Determinant::norbs;
  size_t nsweeps = schd.stochasticIter;
  double stepsize = schd.stepsize;
  
  // prep and init
  // this is for mean field subtraction
  MatrixXd hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair rhf;
  rhf.first = hf.block(0, 0, norbs, Determinant::nalpha);
  rhf.second = hf.block(0, 0, norbs, Determinant::nbeta);
  
  vector<vecPair> hsOperators;
  vecPair oneBodyOperator;
  prepJastrowHS(rhf, hsOperators, oneBodyOperator);
  
  // this is the actual reference in J | ref >
  hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "ref.txt");
  matPair ref;
  ref.first = hf.block(0, 0, norbs, Determinant::nalpha);
  ref.second = hf.block(0, 0, norbs, Determinant::nbeta);

  vecPair expOneBodyOperator;
  expOneBodyOperator.first = oneBodyOperator.first.array().exp();
  expOneBodyOperator.second = oneBodyOperator.second.array().exp();
  ref.first = expOneBodyOperator.first.asDiagonal() * ref.first;
  ref.second = expOneBodyOperator.second.asDiagonal() * ref.second;
  
  // fields : right first
  // initialized to random values
  size_t nfields = hsOperators.size();
  pair<VectorXd, VectorXd> fields;
  fields.first = schd.stepsize * VectorXd::Random(nfields);
  fields.second = schd.stepsize * VectorXd::Random(nfields);
  vecPair propRight, propLeft;
  propRight.first = VectorXcd::Zero(norbs);
  propRight.second = VectorXcd::Zero(norbs);
  propLeft.first = VectorXcd::Zero(norbs);
  propLeft.second = VectorXcd::Zero(norbs);
  for (int i = 0; i < nfields; i++) {
    propRight.first += fields.first(i) * hsOperators[i].first;
    propRight.second += fields.first(i) * hsOperators[i].second;
    propLeft.first += fields.second(i) * hsOperators[i].first;
    propLeft.second += fields.second(i) * hsOperators[i].second;
  }
  matPair right, left;
  right.first = propRight.first.array().exp().matrix().asDiagonal() * ref.first;
  right.second = propRight.second.array().exp().matrix().asDiagonal() * ref.second;
  left.first = ref.first.adjoint() * propLeft.first.array().exp().matrix().asDiagonal();
  left.second = ref.second.adjoint() * propLeft.second.array().exp().matrix().asDiagonal();
  complex<double> overlap = (left.first * right.first).determinant() * (left.second * right.second).determinant();
  VectorXcd overlaps = VectorXcd::Zero(2*nsweeps), num = VectorXcd::Zero(nsweeps), denom = VectorXcd::Zero(nsweeps);

  // metropolis sweep
  size_t accepted = 0;
  uniform_real_distribution<double> uniformStep(-stepsize, stepsize);
  uniform_real_distribution<double> uniform(0., 1.); 
  auto iterTime = getTime();
  for (int n = 0; n < 2*nsweeps; n++) {
    if (n % (2*nsweeps/5) == 0 && commrank == 0) cout << n / 2 << "  " << getTime() - iterTime << endl;
    VectorXd proposedFields = VectorXd::Zero(nfields);
    vecPair proposedProp;
    proposedProp.first = VectorXcd::Zero(norbs);
    proposedProp.second = VectorXcd::Zero(norbs);
    double expRatio = 1.;

    // right jastrow
    if (n%2 == 0) {
      // propose move
      for (int i = 0; i < nfields; i++) {
        proposedFields(i) = fields.first(i) + uniformStep(generator);
        expRatio *= exp((fields.first(i) * fields.first(i) - proposedFields(i) * proposedFields(i))/2);
        proposedProp.first += proposedFields(i) * hsOperators[i].first;
        proposedProp.second += proposedFields(i) * hsOperators[i].second;
      }
      matPair proposedRight;
      proposedRight.first = proposedProp.first.array().exp().matrix().asDiagonal() * ref.first;
      proposedRight.second = proposedProp.second.array().exp().matrix().asDiagonal() * ref.second;
      complex<double> proposedOverlap = (left.first * proposedRight.first).determinant() * (left.second * proposedRight.second).determinant();

      // accept / reject
      if (expRatio * abs(proposedOverlap) / abs(overlap) >= uniform(generator)) {
        accepted++;
        fields.first = proposedFields;
        overlap = proposedOverlap;
        right = proposedRight;
      }

      // measure only for even n
      matPair green;
      calcGreensFunction(left, right, green);
      denom(n/2) = overlap / abs(overlap);
      num(n/2) = denom(n/2) * calcHamiltonianElement(green, enuc, h1, chol);
    }
    
    // left jastrow
    else {
      // propose move
      for (int i = 0; i < nfields; i++) {
        proposedFields(i) = fields.second(i) + uniformStep(generator);
        expRatio *= exp((fields.second(i) * fields.second(i) - proposedFields(i) * proposedFields(i))/2);
        proposedProp.first += proposedFields(i) * hsOperators[i].first;
        proposedProp.second += proposedFields(i) * hsOperators[i].second;
      }
      matPair proposedLeft;
      proposedLeft.first = ref.first.adjoint() * proposedProp.first.array().exp().matrix().asDiagonal();
      proposedLeft.second = ref.second.adjoint() * proposedProp.second.array().exp().matrix().asDiagonal();
      complex<double> proposedOverlap = (proposedLeft.first * right.first).determinant() * (proposedLeft.second * right.second).determinant();

      // accept / reject
      if (expRatio * abs(proposedOverlap) / abs(overlap) >= uniform(generator)) {
        accepted++;
        fields.second = proposedFields;
        overlap = proposedOverlap;
        left = proposedLeft;
      }
    }
    overlaps(n) = overlap;
  }

  complex<double> numMean = num.mean();
  complex<double> denomMean = denom.mean();
  complex<double> energyTotAll[commsize];
  complex<double> numTotAll[commsize];
  complex<double> denomTotAll[commsize];
  for (int i = 0; i < commsize; i++) {
    energyTotAll[i] = complex<double>(0., 0.);
    numTotAll[i] = complex<double>(0., 0.);
    denomTotAll[i] = complex<double>(0., 0.);
  }
  complex<double> energyProc = numMean / denomMean;
  MPI_Gather(&(energyProc), 1, MPI_DOUBLE_COMPLEX, &(energyTotAll), 1, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
  MPI_Gather(&(numMean), 1, MPI_DOUBLE_COMPLEX, &(numTotAll), 1, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
  MPI_Gather(&(denomMean), 1, MPI_DOUBLE_COMPLEX, &(denomTotAll), 1, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &energyProc, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &numMean, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &denomMean, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
  
  energyProc /= commsize;
  numMean /= commsize;
  denomMean /= commsize;
  double stddev = 0.;
  for (int i = 0; i < commsize; i++) {
    //if (commrank == 0) cout << i << "  " << numTotAll[i] << "  " << denomTotAll[i] << "  " << energyTotAll[i] << endl;
    stddev += pow(abs(energyTotAll[i] - energyProc), 2);
  }
  stddev /= (commsize - 1);
  stddev = sqrt(stddev / commsize);
  double acceptanceRatio = accepted / (2. * nsweeps);

  if (commrank == 0) {
    cout << "Acceptance ratio:  " << acceptanceRatio << endl;
    cout << "Numerator:  " << numMean << ", Denominator:  " << denomMean << endl;
    cout << "Energy:  " << energyProc << " (" << stddev << ")\n";
  }
  
  //MPI_Allreduce(MPI_IN_PLACE, &numMean, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
  //MPI_Allreduce(MPI_IN_PLACE, &denomMean, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
 
  //numMean /= commsize;
  //denomMean /= commsize;
  //auto energy = numMean / denomMean;
  //double acceptanceRatio = accepted / (2. * nsweeps);

  //if (commrank == 0) {
  //  cout << "Acceptance ratio:  " << acceptanceRatio << endl;
  //  cout << "Numerator:  " << numMean << ", Denominator:  " << denomMean << endl;
  //  cout << "Energy:  " << energy << endl;
  //}
}


int main(int argc, char *argv[])
{

#ifndef SERIAL
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
#endif
  startofCalc = getTime();

  initSHM();
  //license();
  if (commrank == 0) {
    std::system("echo User:; echo $USER");
    std::system("echo Hostname:; echo $HOSTNAME");
    std::system("echo CPU info:; lscpu | head -15");
    std::system("echo Computation started at:; date");
    cout << "git commit: " << GIT_HASH << ", branch: " << GIT_BRANCH << ", compiled at: " << COMPILE_TIME << endl << endl;
    cout << "nproc used: " << commsize << " (NB: stochasticIter below is per proc)" << endl << endl; 
  }

  string inputFile = "input.dat";
  if (argc > 1)
    inputFile = string(argv[1]);
  readInput(inputFile, schd, false);

  generator = std::mt19937(schd.seed + commrank);

  MatrixXd h1, h1Mod;
  vector<MatrixXd> chol;
  readIntegralsCholeskyAndInitializeDeterminantStaticVariables("FCIDUMP", h1, h1Mod, chol);

  calcJastrowEnergyMetropolis(coreE, h1, chol);

  if (commrank == 0) cout << "Total calculation time:  " << getTime() - startofCalc << endl;
  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shmcas.c_str());
  return 0;
}
