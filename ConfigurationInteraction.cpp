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
#include <algorithm>
#include <random>
#include <chrono>
#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#ifndef SERIAL
//#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "evaluatePT.h"
#include "MoDeterminants.h"
#include "Determinants.h"
#include "CPS.h"
#include "Wfn.h"
#include "input.h"
#include "integral.h"
#include "SHCIshm.h"
#include "math.h"
#include "Walker.h"
#include "Davidson.h"

using namespace Eigen;
using namespace boost;
using namespace std;

void getNVariables(int excitationLevel, vector<int> &SingleIndices,
                   vector<int>& DoubleIndices, int norbs, twoIntHeatBathSHM &I2hb);
void getStochasticGradientContinuousTimeCI(CPSSlater &w, double &E0, vector<int> &SingleIndices,
                                           vector<int>& DoubleIndices, double &stddev, 
					   int &nalpha, int &nbeta, int &norbs,
                                           oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb,
                                           double &coreE, VectorXd &vars, double &rk,
                                           int niter, double targetError);
void getDeterministicCI(CPSSlater &w, double &E0, vector<int> &SingleIndices,
                        vector<int>& DoubleIndices, double &stddev, 
			int &nalpha, int &nbeta, int &norbs,
                        oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb,
                        double &coreE, VectorXd &vars, double &rk,
                        int niter, double targetError);
void generateAllDeterminants(vector<Determinant> &allDets, int norbs, int nalpha, int nbeta);
double calcTcorr(vector<double> &v); //in evaluateE.cpp

int main(int argc, char *argv[])
{

#ifndef SERIAL
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
#endif
  startofCalc = getTime();

  initSHM();
  license();

  string inputFile = "input.dat";
  if (argc > 1)
    inputFile = string(argv[1]);
  if (commrank == 0)
    readInput(inputFile, schd);
#ifndef SERIAL
  mpi::broadcast(world, schd, 0);
#endif

  generator = std::mt19937(schd.seed + commrank);

  twoInt I2;
  oneInt I1;
  int norbs, nalpha, nbeta;
  double coreE = 0.0;
  std::vector<int> irrep;
  readIntegrals("FCIDUMP", I2, I1, nalpha, nbeta, norbs, coreE, irrep);

  //initialize the heatbath integrals
  std::vector<int> allorbs;
  for (int i = 0; i < norbs; i++)
    allorbs.push_back(i);
  twoIntHeatBath I2HB(1.e-10);
  twoIntHeatBathSHM I2HBSHM(1.e-10);
  if (commrank == 0)
    I2HB.constructClass(allorbs, I2, I1, norbs);
  I2HBSHM.constructClass(norbs, I2HB);

  //Setup static variables
  Determinant::EffDetLen = (norbs) / 64 + 1;
  Determinant::norbs = norbs;
  MoDeterminant::norbs = norbs;
  MoDeterminant::nalpha = nalpha;
  MoDeterminant::nbeta = nbeta;

  //Setup Slater Determinants
  HforbsA = MatrixXd::Zero(norbs, norbs);
  HforbsB = MatrixXd::Zero(norbs, norbs);
  readHF(HforbsA, HforbsB, schd.uhf);

  //Setup CPS wavefunctions
  std::vector<Correlator> nSiteCPS;
  for (auto it = schd.correlatorFiles.begin(); it != schd.correlatorFiles.end();
       it++)
  {
    readCorrelator(it->second, it->first, nSiteCPS);
  }

  vector<Determinant> detList;
  vector<double> ciExpansion;

  if (boost::iequals(schd.determinantFile, ""))
  {
    detList.resize(1);
    ciExpansion.resize(1, 1.0);
    for (int i = 0; i < nalpha; i++)
      detList[0].setoccA(i, true);
    for (int i = 0; i < nbeta; i++)
      detList[0].setoccB(i, true);
  }
  else
  {
    readDeterminants(schd.determinantFile, detList, ciExpansion);
  }
  //setup up wavefunction
  CPSSlater wave(nSiteCPS, detList, ciExpansion);

  size_t size;
  ifstream file("params.bin", ios::in | ios::binary | ios::ate);
  if (commrank == 0)
  {
    size = file.tellg();
  }
#ifndef SERIAL
  MPI_Bcast(&size, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  Eigen::VectorXd vars = Eigen::VectorXd::Zero(size / sizeof(double));
  if (commrank == 0)
  {
    file.seekg(0, ios::beg);
    file.read((char *)(&vars[0]), size);
    file.close();
  }

#ifndef SERIAL
  MPI_Bcast(&vars[0], vars.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

  if ((schd.uhf && vars.size() != wave.getNumVariables() + 2 * norbs * norbs) ||
      (!schd.uhf && vars.size() != wave.getNumVariables() + norbs * norbs))
  {
    cout << "number of variables on disk: " << vars.size() << " is not equal to wfn parameters: " << wave.getNumVariables() << endl;
    exit(0);
  }

  wave.updateVariables(vars);
  int numVars = wave.getNumVariables();

  for (int i = 0; i < norbs; i++)
  {
    for (int j = 0; j < norbs; j++)
    {
      if (!schd.uhf)
      {
        HforbsA(i, j) = vars[numVars + i * norbs + j];
        HforbsB(i, j) = vars[numVars + i * norbs + j];
      }
      else
      {
        HforbsA(i, j) = vars[numVars + i * norbs + j];
        HforbsB(i, j) = vars[numVars + norbs * norbs + i * norbs + j];
      }
    }
  }

  MatrixXd alpha(norbs, nalpha), beta(norbs, nbeta);
  alpha = HforbsA.block(0, 0, norbs, nalpha);
  beta = HforbsB.block(0, 0, norbs, nbeta);
  MoDeterminant det(alpha, beta);

  vector<int> SingleSpinIndices, DoubleSpinIndices;
  getNVariables(schd.excitationLevel, SingleSpinIndices, DoubleSpinIndices, norbs, I2HBSHM);

  //we assume intermediate normalization
  Eigen::VectorXd civars = Eigen::VectorXd::Zero(SingleSpinIndices.size() / 2 + 
						 1 + DoubleSpinIndices.size() / 4);

  double rt = 0., stddev, E0 = 0.;

  if (!schd.deterministic)
    getStochasticGradientContinuousTimeCI(wave, E0, SingleSpinIndices, DoubleSpinIndices,
					  stddev, nalpha, nbeta, norbs,
                                          I1, I2, I2HBSHM, coreE, civars, rt,
                                          schd.stochasticIter, 0.5e-3);
  else
    getDeterministicCI(wave, E0, SingleSpinIndices, DoubleSpinIndices, 
		       stddev, nalpha, nbeta, norbs,
                       I1, I2, I2HBSHM, coreE, civars, rt,
                       schd.stochasticIter, 0.5e-3);

  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());
  return 0;
}

void getNVariables(int excitationLevel, vector<int> &SingleIndices, vector<int>& DoubleIndices,
                   int norbs, twoIntHeatBathSHM &I2hb)
{
  for (int i = 0; i < 2 * norbs; i++)
    for (int j = 0; j < 2 * norbs; j++)
    {
      //if (I2hb.Singles(i, j) > schd.epsilon ) 
      if (i%2 == j%2)
      {
        SingleIndices.push_back(i);
        SingleIndices.push_back(j);
      }
    }

  for (int i=0; i < 2*norbs; i++) {
    for (int j=i+1; j < 2*norbs; j++) {
      int pair = (j/2) * (j/2 + 1)/2 + i/2;

      size_t start = i%2 == j%2 ? I2hb.startingIndicesSameSpin[pair]   : I2hb.startingIndicesOppositeSpin[pair];
      size_t end   = i%2 == j%2 ? I2hb.startingIndicesSameSpin[pair+1] : I2hb.startingIndicesOppositeSpin[pair+1];
      float *integrals = i%2 == j%2 ? I2hb.sameSpinIntegrals : I2hb.oppositeSpinIntegrals;
      short *orbIndices = i%2 == j%2 ? I2hb.sameSpinPairs : I2hb.oppositeSpinPairs;

      for (size_t index = start; index < end; index++) {
	if (fabs(integrals[index]) < schd.epsilon)
	  break;
	int a = 2 * orbIndices[2* index] + i%2, b = 2 * orbIndices[2 * index] + j%2;

	DoubleIndices.push_back(i); DoubleIndices.push_back(j); 
	DoubleIndices.push_back(a); DoubleIndices.push_back(b);
      }
    }
  }

  //SingleIndices.resize(4); DoubleIndices.resize(8);
  /*
  for (int i=0; i<SingleIndices.size()/2; i++) {
    for (int j=i+1; j<SingleIndices.size()/2; j++) {
      int I = SingleIndices[2*i], A = SingleIndices[2*i+1], 
	J = SingleIndices[2*j], B = SingleIndices[2*j+1];

      if (I == A || J == A || I == B || J == B ||
	  I == J || A == B) continue;

      DoubleIndices.push_back(I);
      DoubleIndices.push_back(J);
      DoubleIndices.push_back(A);
      DoubleIndices.push_back(B);
      //cout << I<<"  "<<J<<"  "<<A<<"  "<<B<<endl;
    }
  }
  */
  //exit(0);
  return;
}



void getStochasticGradientContinuousTimeCI(CPSSlater &w, double &E0, vector<int> &SingleIndices,
                                           vector<int>& DoubleIndices, double &stddev, 
					   int &nalpha, int &nbeta, int &norbs,
                                           oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb,
                                           double &coreE, VectorXd &civars, double &rk,
                                           int niter, double targetError)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));

  //initialize the walker
  Determinant d;
  bool readDeterminant = false;
  char file[5000];

  sprintf(file, "BestDeterminant.txt");

  {
    ifstream ofile(file);
    if (ofile)
      readDeterminant = true;
  }

  if (readDeterminant)
  {
    if (commrank == 0)
    {
      std::ifstream ifs(file, std::ios::binary);
      boost::archive::binary_iarchive load(ifs);
      load >> d;
    }
#ifndef SERIAL
    MPI_Bcast(&d.reprA, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d.reprB, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  }

  Walker walk(d);
  walk.initUsingWave(w);

  int maxTerms = (nalpha) * (norbs - nalpha); //pick a small number that will be incremented later
  vector<double> ovlpRatio(maxTerms);
  vector<size_t> excitation1(maxTerms), excitation2(maxTerms);
  vector<double> HijElements(maxTerms);
  int nExcitations = 0;

  vector<double> ovlpRatioForM(maxTerms);
  vector<size_t> excitation1ForM(maxTerms), excitation2ForM(maxTerms);
  vector<double> HijElementsForM(maxTerms);
  int nExcitationsForM = 0;

  stddev = 1.e4;
  int iter = 0;
  double M1 = 0., S1 = 0., Eavg = 0.;
  double Eloc = 0.;
  double ham = 0., ovlp = 0.;
  double scale = 1.0;

  VectorXd hamRatio = VectorXd::Zero(civars.rows());
  VectorXd gradRatio = VectorXd::Zero(civars.rows());
  MatrixXd Hamiltonian(civars.rows(), civars.rows());
  MatrixXd Overlap(civars.rows(), civars.rows());
  MatrixXd iterHamiltonian(civars.rows(), civars.rows());
  MatrixXd iterOverlap(civars.rows(), civars.rows());
  VectorXd localGrad;

  double bestOvlp = 0.;
  Determinant bestDet = d;

  //schd.epsilon = -1;
  nExcitations = 0;
  E0 = walk.d.Energy(I1, I2, coreE);
  w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE, ovlpRatio,
                       excitation1, excitation2, HijElements, nExcitations, false);

  gradRatio.setZero();
  double factor = 1.0;
  w.OvlpRatioCI(walk, gradRatio, I1, I2, SingleIndices, DoubleIndices, I2hb, coreE, factor);

  hamRatio = E0 * gradRatio;
  for (int m = 0; m < nExcitations; m++)
  {

    Walker wtmp = walk;
    //this is the new walker m, later I should try to get rid of this step
    wtmp.updateWalker(w, excitation1[m], excitation2[m]);

    double factor = HijElements[m] * ovlpRatio[m];
    w.OvlpRatioCI(wtmp, hamRatio,  I1, I2,
		  SingleIndices, DoubleIndices, I2hb, coreE, factor);
  }

  //updateHamOverlap(iterHamiltonian, iterOverlap, hamRatio, gradRatio, SingleIndices);
  iterHamiltonian = gradRatio*hamRatio.transpose();
  iterOverlap     = gradRatio*gradRatio.transpose();
  Hamiltonian = iterHamiltonian;
  Overlap = iterOverlap;

  int nstore = 1000000 / commsize;
  int gradIter = min(nstore, niter);

  std::vector<double> gradError(gradIter * commsize, 0);
  bool reset = false;
  double cumdeltaT = 0., cumdeltaT2 = 0.;

  while (iter < niter && stddev > targetError)
  {

    double cumovlpRatio = 0;
    //when using uniform probability 1./numConnection * max(1, pi/pj)
    for (int i = 0; i < nExcitations; i++)
    {
      cumovlpRatio += abs(ovlpRatio[i]);
      //cumovlpRatio += min(1.0, pow(ovlpRatio[i], 2));
      ovlpRatio[i] = cumovlpRatio;
    }

    //double deltaT = -log(random())/(cumovlpRatio);
    double deltaT = 1.0 / (cumovlpRatio);
    double nextDetRandom = random() * cumovlpRatio;
    int nextDet = std::lower_bound(ovlpRatio.begin(), (ovlpRatio.begin() + nExcitations),
                                   nextDetRandom) -
                  ovlpRatio.begin();

    cumdeltaT += deltaT;
    cumdeltaT2 += deltaT * deltaT;

    double Elocold = Eloc;

    double ratio = deltaT / cumdeltaT;
    Hamiltonian += ratio * (iterHamiltonian - Hamiltonian);
    Overlap += ratio * (iterOverlap - Overlap);

    Eloc = Eloc + deltaT * (ham - Eloc) / (cumdeltaT); //running average of energy
    //cout << Hamiltonian(0,0)<<"  "<<Eloc<<"  "<<iterHamiltonian(1,1)<<"  "<<iterOverlap(1,1)<<endl;

    S1 = S1 + (ham - Elocold) * (ham - Eloc);

    if (iter < gradIter)
      gradError[iter + commrank * gradIter] = ham;

    iter++;

    //update the walker
    if (true)
    {
      walk.updateWalker(w, excitation1[nextDet], excitation2[nextDet]);
    }

    nExcitations = 0;
    double E0 = walk.d.Energy(I1, I2, coreE);
    w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE, ovlpRatio,
                         excitation1, excitation2, HijElements, nExcitations, false);

    gradRatio.setZero();
    double factor = 1.0;
    //gradRatio[0] = 1.0;
    w.OvlpRatioCI(walk, gradRatio,  I1, I2, SingleIndices, DoubleIndices, I2hb, coreE, factor);

    //<m|Psi_i>/<n|Psi0>
    hamRatio = E0 * gradRatio;
    for (int m = 0; m < nExcitations; m++)
    {

      Walker wtmp = walk;
      wtmp.updateWalker(w, excitation1[m], excitation2[m]);

      double factor = HijElements[m] * ovlpRatio[m];
      //hamRatio[0] += factor;
      w.OvlpRatioCI(wtmp, hamRatio,  I1, I2,
		    SingleIndices, DoubleIndices, I2hb, coreE, factor);
    }

    iterHamiltonian = gradRatio*hamRatio.transpose();
    iterOverlap     = gradRatio*gradRatio.transpose();
    //updateHamOverlap(iterHamiltonian, iterOverlap, hamRatio, gradRatio, SingleIndices);

    if (abs(ovlp) > bestOvlp)
    {
      bestOvlp = abs(ovlp);
      bestDet = walk.d;
    }
  }

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(gradError[0]), gradError.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Hamiltonian(0, 0)), Hamiltonian.rows() * Hamiltonian.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Overlap(0, 0)), Overlap.rows() * Overlap.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &Eloc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  //if (commrank == 0)
  rk = calcTcorr(gradError);

  Hamiltonian /= (commsize);
  Overlap /= (commsize);
  E0 = Eloc / commsize;

  stddev = sqrt(S1 * rk / (niter - 1) / niter / commsize);
#ifndef SERIAL
  MPI_Bcast(&stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

  if (commrank == 0)
  {
    MatrixXcd eigenvectors; VectorXcd eigenvalues;
    VectorXd betas;
    GeneralizedEigen(Hamiltonian, Overlap, eigenvalues, eigenvectors, betas);
    //GeneralizedEigenSolver<MatrixXd> ges(Hamiltonian, Overlap);
    double lowest = Hamiltonian(0,0)/Overlap(0,0);
    
    MatrixXcd eigenvecInv = eigenvectors.inverse();
    int vecIndex = 0;
    for (int i=0; i<Hamiltonian.rows(); i++)  {
      if (abs(betas.transpose()(i)) > 1.e-10) {
	if (eigenvalues.transpose()(i).real() < lowest) {
	  lowest = eigenvalues.transpose()(i).real();
	  vecIndex = i;
	}
      }
    }

    double norm = std::abs((eigenvectors.col(vecIndex).adjoint()*(Overlap*eigenvectors.col(vecIndex)))(0,0));
    double a0 = std::norm((Overlap*eigenvectors.col(vecIndex))(0,0)/pow(Overlap(0,0),0.5))/norm;


    double e0 = (Hamiltonian(0,0)/Overlap(0,0));
    std::cout << format("%18s    : %14.8f (%8.2e)") % ("VMC energy") %  e0 % stddev<<endl;
    std::cout << format("%18s    : %14.8f (%8.2e)") % ("CISD energy") %  (lowest) % stddev<<endl;
    
    double de = lowest - e0;
    double plusQ = (1 - a0)*de/a0;

    std::cout << format("%18s    : %14.8f (%8.2e)") % ("CISD+Q energy") %  (lowest+plusQ) % stddev<<endl;
  }

  if (commrank == 0) {
    Hamiltonian /= Overlap(0,0);
    Overlap /= Overlap(0,0);

    double E0 = Hamiltonian(0,0)/Overlap(0,0);
    MatrixXd Uo = MatrixXd::Zero(Hamiltonian.rows(), Hamiltonian.cols());
    Uo(0,0) = 1.0;
    for (int i=1; i<Uo.rows(); i++) {
      Uo(0, i) = - Overlap(0,i);
      Uo(i, i) = 1.0;
    }


    Overlap = Uo.transpose()* (Overlap * Uo);
    Hamiltonian = Uo.transpose()* (Hamiltonian * Uo);

    VectorXd Vpsi0 = Hamiltonian.block(1,0,Overlap.rows()-1, 1);

    MatrixXd temp = Overlap;
    Overlap     = temp.block(1,1, Overlap.rows()-1, Overlap.rows()-1);
    temp = Hamiltonian;
    Hamiltonian = temp.block(1,1, temp.rows()-1, temp.rows()-1);

    MatrixXd eigenvectors; VectorXd eigenvalues;
    SelfAdjointEigen(Overlap, eigenvalues, eigenvectors);


    int nCols = 0;

    for (int i=0; i<Overlap.rows(); i++) 
      if ( eigenvalues(i) > 1.e-8)
	nCols ++;

    MatrixXd U = MatrixXd::Zero(Overlap.rows(), nCols);
    int index = 0;
  
    for (int i=0; i<Overlap.rows(); i++) 
      if ( eigenvalues(i) > 1.e-8) {
	U.col(index) = eigenvectors.col(i)/pow(eigenvalues(i), 0.5);
	index++;
      }
   
    MatrixXd Hprime = U.transpose()*(Hamiltonian * U);
    MatrixXd Hprime_E0 = 1.*Hprime;
    for (int i=0; i<Hprime.rows(); i++) {
      Hprime_E0(i,i) -= E0;
    }

    VectorXd temp1 = Vpsi0;
    Vpsi0 = U.transpose()*temp1;
    
    VectorXd psi1;
    SolveEigen(Hprime_E0, Vpsi0, psi1);

    double E2 = -Vpsi0.transpose()*psi1;
    std::cout << format("%18s    : %14.8f (%8.2e)") % ("E0+E2 energy") %  (E0+E2) % stddev<<endl;
  }
}

void getDeterministicCI(CPSSlater &w, double &E0, vector<int> &SingleIndices,
                        vector<int>& DoubleIndices, double &stddev, 
			int &nalpha, int &nbeta, int &norbs,
                        oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb,
                        double &coreE, VectorXd &civars, double &rk,
                        int niter, double targetError)
{

  vector<Determinant> allDets;
  generateAllDeterminants(allDets, norbs, nalpha, nbeta);

  int maxTerms = 1000000; //pick a small number that will be incremented later
  vector<double> ovlpRatio(maxTerms);
  vector<size_t> excitation1(maxTerms), excitation2(maxTerms);
  vector<double> HijElements(maxTerms);
  int nExcitations = 0;

  double Energy = 0.0;
  VectorXd hamRatio = VectorXd::Zero(civars.rows());
  VectorXd gradRatio = VectorXd::Zero(civars.rows());
  MatrixXd Hamiltonian = MatrixXd::Zero(civars.rows(), civars.rows());
  MatrixXd Overlap = MatrixXd::Zero(civars.rows(), civars.rows());
  VectorXd localGrad;

  for (int i = commrank; i < allDets.size(); i += commsize)
  {
    Walker walk(allDets[i]);
    walk.initUsingWave(w);

    double ham, ovlp;
    {
      nExcitations = 0;
      E0 = walk.d.Energy(I1, I2, coreE);
      w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE, ovlpRatio,
                           excitation1, excitation2, HijElements, nExcitations, false);

      gradRatio.setZero();
      double factor = 1.0;
      w.OvlpRatioCI(walk, gradRatio,  I1, I2, SingleIndices, DoubleIndices, I2hb, coreE, factor);

      hamRatio = E0 * gradRatio;
      for (int m = 0; m < nExcitations; m++)
      {

        Walker wtmp = walk;
        //this is the new walker m, later I should try to get rid of this step
        wtmp.updateWalker(w, excitation1[m], excitation2[m]);

        double factor = HijElements[m] * ovlpRatio[m];
        w.OvlpRatioCI(wtmp, hamRatio,  I1, I2,
                      SingleIndices, DoubleIndices, I2hb, coreE, factor);
      }
    }

    Hamiltonian += gradRatio*hamRatio.transpose() * ovlp * ovlp;
    Overlap += gradRatio*gradRatio.transpose() * ovlp * ovlp;
    Energy += ham * ovlp * ovlp;
  }

  //cout << Energy/Overlap(0,0) << endl;
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(Hamiltonian(0, 0)), Hamiltonian.rows() * Hamiltonian.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Overlap(0, 0)), Overlap.rows() * Overlap.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  //MPI_Allreduce(MPI_IN_PLACE, &Eloc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  //Hamiltonian /= (commsize);
  //Overlap /= (commsize);
  //E0 = Eloc / commsize;


  if (commrank == 0)
  {
    MatrixXcd eigenvectors; VectorXcd eigenvalues;
    VectorXd betas;
    GeneralizedEigen(Hamiltonian, Overlap, eigenvalues, eigenvectors, betas);
    //GeneralizedEigenSolver<MatrixXd> ges(Hamiltonian, Overlap);
    double lowest = Hamiltonian(0,0)/Overlap(0,0);
    
    MatrixXcd eigenvecInv = eigenvectors.inverse();
    int vecIndex = 0;
    for (int i=0; i<Hamiltonian.rows(); i++)  {
      if (abs(betas.transpose()(i)) > 1.e-10) {
	if (eigenvalues.transpose()(i).real() < lowest) {
	  lowest = eigenvalues.transpose()(i).real();
	  vecIndex = i;
	}
      }
    }

    double norm = std::abs((eigenvectors.col(vecIndex).adjoint()*(Overlap*eigenvectors.col(vecIndex)))(0,0));
    double a0 = std::norm((Overlap*eigenvectors.col(vecIndex))(0,0)/pow(Overlap(0,0),0.5))/norm;


    double e0 = (Hamiltonian(0,0)/Overlap(0,0));
    std::cout << format("%18s    : %14.8f (%8.2e)") % ("VMC energy") %  e0 % stddev<<endl;
    std::cout << format("%18s    : %14.8f (%8.2e)") % ("CISD energy") %  (lowest) % stddev<<endl;
    
    double de = lowest - e0;
    double plusQ = (1 - a0)*de/a0;

    std::cout << format("%18s    : %14.8f (%8.2e)") % ("CISD+Q energy") %  (lowest+plusQ) % stddev<<endl;
  }

  if (commrank == 0) {
    Hamiltonian /= Overlap(0,0);
    Overlap /= Overlap(0,0);

    double E0 = Hamiltonian(0,0)/Overlap(0,0);
    MatrixXd Uo = MatrixXd::Zero(Hamiltonian.rows(), Hamiltonian.cols());
    Uo(0,0) = 1.0;
    for (int i=1; i<Uo.rows(); i++) {
      Uo(0, i) = - Overlap(0,i);
      Uo(i, i) = 1.0;
    }


    Overlap = Uo.transpose()* (Overlap * Uo);
    Hamiltonian = Uo.transpose()* (Hamiltonian * Uo);

    VectorXd Vpsi0 = Hamiltonian.block(1,0,Overlap.rows()-1, 1);

    MatrixXd temp = Overlap;
    Overlap     = temp.block(1,1, Overlap.rows()-1, Overlap.rows()-1);
    temp = Hamiltonian;
    Hamiltonian = temp.block(1,1, temp.rows()-1, temp.rows()-1);

    MatrixXd eigenvectors; VectorXd eigenvalues;
    SelfAdjointEigen(Overlap, eigenvalues, eigenvectors);


    int nCols = 0;

    for (int i=0; i<Overlap.rows(); i++) 
      if ( eigenvalues(i) > 1.e-8)
	nCols ++;

    MatrixXd U = MatrixXd::Zero(Overlap.rows(), nCols);
    int index = 0;
  
    for (int i=0; i<Overlap.rows(); i++) 
      if ( eigenvalues(i) > 1.e-8) {
	U.col(index) = eigenvectors.col(i)/pow(eigenvalues(i), 0.5);
	index++;
      }
   
    MatrixXd Hprime = U.transpose()*(Hamiltonian * U);
    MatrixXd Hprime_E0 = 1.*Hprime;
    for (int i=0; i<Hprime.rows(); i++) {
      Hprime_E0(i,i) -= E0;
    }

    VectorXd temp1 = Vpsi0;
    Vpsi0 = U.transpose()*temp1;
    
    VectorXd psi1;
    SolveEigen(Hprime_E0, Vpsi0, psi1);

    double E2 = -Vpsi0.transpose()*psi1;
    std::cout << format("%18s    : %14.8f (%8.2e)") % ("E0+E2 energy") %  (E0+E2) % stddev<<endl;
  }

}
