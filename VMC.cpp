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
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <boost/format.hpp>
#ifndef SERIAL
//#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "evaluateE.h"
#include "MoDeterminants.h"
#include "Determinants.h"
#include "CPS.h"
#include "Wfn.h"
#include "input.h"
#include "integral.h"
#include "SHCIshm.h"
#include "math.h"
#include "diis.h"

using namespace Eigen;
using namespace boost;
using namespace std;


int main(int argc, char* argv[]) {

#ifndef SERIAL
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
#endif
  double startofCalc = getTime();

  initSHM();
  license();

  string inputFile = "input.dat";
  if (argc > 1)
    inputFile = string(argv[1]);
  if (commrank == 0) readInput(inputFile, schd);
#ifndef SERIAL
  mpi::broadcast(world, schd, 0);
#endif

  generator = std::mt19937(getTime()+commrank);

  twoInt I2; oneInt I1; 
  int norbs, nalpha, nbeta; 
  double coreE=0.0;
  std::vector<int> irrep;
  readIntegrals("FCIDUMP", I2, I1, nalpha, nbeta, norbs, coreE, irrep);
  
  //Setup static variables
  Determinant::EffDetLen = (norbs)/64+1;
  Determinant::norbs    = norbs;
  HalfDet::norbs        = norbs;
  MoDeterminant::norbs  = norbs;
  MoDeterminant::nalpha = nalpha;
  MoDeterminant::nbeta  = nbeta;

  //Setup Slater Determinants
  MatrixXd Hforbs = MatrixXd::Zero(norbs, norbs);
  readHF(Hforbs);
  MatrixXd alpha(norbs, nalpha), beta(norbs, nbeta);
  alpha = Hforbs.block(0, 0, norbs, nalpha);
  beta  = Hforbs.block(0, 0, norbs, nbeta );
  MoDeterminant det(alpha, beta);

  //Setup CPS wavefunctions
  std::vector<CPS> twoSiteCPS;
  for (auto it = schd.correlatorFiles.begin(); it != schd.correlatorFiles.end();
       it++) {
    readCorrelator(it->second, it->first, twoSiteCPS);
  }
  /*
  for (int i=0; i<norbs; i++)
  for (int j=i; j<norbs; j++) {
    if (j >= i)
    {
      vector<int> asites(1,i), bsites(1,j);
      twoSiteCPS.push_back(CPS(asites, bsites));
    }
    if (j >i ) {
      vector<int> asites(2,0), bsites;
      asites[0] = i; asites[1] = j;
      twoSiteCPS.push_back(CPS(asites, bsites));
      twoSiteCPS.push_back(CPS(bsites, asites));
    }

  }
  */

  //setup up wavefunction
  CPSSlater wave(twoSiteCPS, det);
  if (schd.restart) {
    wave.readWave();
  }
  cout.precision(10);
  
  DIIS diis;
  if (commrank == 0) diis.init(schd.diisSize, wave.getNumVariables());

  double gradnorm=10.0;
  double E0=0, ovlp=0;

  if (schd.deterministic)
    E0 = evaluateEDeterministic(wave, nalpha, nbeta, norbs, I1, I2, coreE);
  else
    E0 = evaluateEStochastic(wave, nalpha, nbeta, norbs, I1, I2, coreE, 100000, 1.e-6);

  if (!schd.davidsonPrecondition) {
    for (int iter =0; iter<schd.maxIter && gradnorm>schd.tol; iter++) {
      Eigen::VectorXd grad = Eigen::VectorXd::Zero(wave.getNumVariables());
      if (schd.deterministic)
	getGradient(wave, E0, nalpha, nbeta, norbs, I1, I2, coreE, grad);
      else
	getStochasticGradient(wave, E0, nalpha, nbeta, norbs, I1, I2, coreE, grad, 100000, 1e-6);

      gradnorm = grad.squaredNorm();
      grad *= -0.01;
      VectorXd vars = VectorXd::Zero(wave.getNumVariables());wave.getVariables(vars);
      vars = vars+grad;
      if (commrank == 0)
	diis.update(vars, grad);
#ifndef SERIAL
      MPI_Bcast(&(grad[0]),     grad.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&(vars[0]),     vars.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
      wave.updateVariables(vars);
      wave.writeWave();
      //wave.incrementVariables(grad);
      if (commrank == 0)
	std::cout << format("%6i   %14.8f  %14.8f %8.2f\n") %iter 
	  % E0 %gradnorm %( (getTime()-startofCalc));

      if (schd.deterministic)
	E0 = evaluateEDeterministic(wave, nalpha, nbeta, norbs, I1, I2, coreE);
      else
	E0 = evaluateEStochastic(wave, nalpha, nbeta, norbs, I1, I2, coreE, 100000, 1.e-6);
    }

  }
  else {
    for (int iter =0; iter<schd.maxIter && gradnorm>schd.tol; iter++) {

      if (commrank == 0)
	std::cout << format("%6i   %14.8f ") %iter % E0;
      
      Eigen::VectorXd grad = Eigen::VectorXd::Zero(wave.getNumVariables());
      
      if (schd.deterministic)
	getGradientUsingDavidson(wave, E0, nalpha, nbeta, norbs, I1, I2, coreE, grad);
      else
	getStochasticGradientUsingDavidson(wave, E0, nalpha, nbeta, norbs, I1, I2, coreE, grad, 100000, 1.e-6);      

      gradnorm = grad.squaredNorm();
      
      VectorXd vars = VectorXd::Zero(wave.getNumVariables());wave.getVariables(vars);
      
      //grad *= 0.1;
      vars = vars+grad;
      if (commrank == 0) diis.update(vars, grad);
#ifndef SERIAL
      MPI_Bcast(&(grad[0]),     grad.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&(vars[0]),     vars.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
      wave.updateVariables(vars);
      wave.writeWave();

      if (schd.deterministic)
	E0 = evaluateEDeterministic(wave, nalpha, nbeta, norbs, I1, I2, coreE);
      else
	E0 = evaluateEStochastic(wave, nalpha, nbeta, norbs, I1, I2, coreE, 100000, 1.e-6);
      
      if (commrank == 0)
	std::cout << format("  %14.8f  %8.2f\n") %gradnorm %( (getTime()-startofCalc));    
    }
  }

  exit(0);
  /*
  auto random = std::bind(std::uniform_real_distribution<double>(0,1),
			  mt19937(getTime()+commrank));

  int iter = 0, newiter=0; double cumulative=0.0;
  double ovlp, ham;
  det.HamAndOvlp(alphaOrbs, betaOrbs, ovlp, ham, I1, I2, coreE);
  bool update = true;

  while (iter < 100000) {

    double Eloc = ham/ovlp; 

    cumulative += Eloc;
    iter ++;
    if (iter %1000 == 0) {
      double cum = cumulative;
      int cumiter = iter*commsize;
      MPI_Allreduce(&cumulative, &cum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      if (commrank == 0)
	std::cout << format("%6i   %14.8f  %14.8f  %14.8f \n") %iter % ovlp % Eloc % (cum/cumiter);
    }

    //pick a random occupied orbital
    int i = floor( random()*(alphaOrbs.size()+betaOrbs.size()) );
    if (i < alphaOrbs.size()) {
      int a = floor(random()*alphaOpen.size());
      std::swap(alphaOrbs[i], alphaOpen[a]);

      double newovlp = det.Overlap(alphaOrbs, betaOrbs);
      if (pow(newovlp/ovlp,2) > random() ) {
	newiter++;
	det.HamAndOvlp(alphaOrbs, betaOrbs, ovlp, ham, I1, I2, coreE);
      }
      else 
	std::swap(alphaOrbs[i], alphaOpen[a]);
    }
    else {
      i = i - alphaOrbs.size();
      int a = floor( random()*betaOpen.size());
      std::swap(betaOrbs[i], betaOpen[a]);

      double newovlp = det.Overlap(alphaOrbs, betaOrbs);
      if (pow(newovlp/ovlp,2) > random() ) {
	det.HamAndOvlp(alphaOrbs, betaOrbs, ovlp, ham, I1, I2, coreE);
	newiter++;
      }
      else 
	std::swap(betaOrbs[i], betaOpen[a]);

    }

  }
  */
  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  return 0;
}
