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
  std::vector<CPS> nSiteCPS;
  for (auto it = schd.correlatorFiles.begin(); it != schd.correlatorFiles.end();
       it++) {
    readCorrelator(it->second, it->first, nSiteCPS);
  }

  //setup up wavefunction
  CPSSlater wave(nSiteCPS, det);
  if (schd.restart) {
    wave.readWave();
  }
  cout.precision(10);
  
  DIIS diis;
  if (commrank == 0) diis.init(schd.diisSize, wave.getNumVariables());

  double gradnorm=10.0;
  double E0=0, ovlp=0;
  double stddev=0;

  if (schd.deterministic)
    E0 = evaluateEDeterministic(wave, nalpha, nbeta, norbs, I1, I2, coreE);
  else
    E0 = evaluateEStochastic(wave, nalpha, nbeta, norbs, I1, I2, coreE, stddev, schd.stochasticIter, 1.e-6);

  Method m = schd.m;

  //{
  //Eigen::VectorXd grad = Eigen::VectorXd::Zero(wave.getNumVariables());
  //davidsonDirect(nalpha, nbeta, norbs, I1, I2, coreE, grad);
  //exit(0);
  //}
  if (schd.davidsonPrecondition) {
    Eigen::VectorXd prevGrad = Eigen::VectorXd::Zero(wave.getNumVariables());
    for (int iter =0; iter<schd.maxIter && gradnorm>schd.tol; iter++) {
      Eigen::VectorXd grad = Eigen::VectorXd::Zero(wave.getNumVariables());
      Eigen::VectorXd grad2 = Eigen::VectorXd::Zero(wave.getNumVariables());
      
      if (schd.deterministic) 
	getGradientUsingDavidson(wave, E0, nalpha, nbeta, norbs, I1, I2, coreE, grad);
	//getGradient(wave, E0, nalpha, nbeta, norbs, I1, I2, coreE, grad);
      else
	getStochasticGradient(wave, E0, nalpha, nbeta, norbs, I1, I2, coreE, grad, 500000, 1e-6);

      //cout << grad <<endl;
      //exit(0);

      gradnorm = grad.squaredNorm();
      if (commrank == 0)
	std::cout << format("%6i   %14.8f (%8.2e)  %14.8f %8.2f\n") %iter 
	  % E0 % stddev % gradnorm %( (getTime()-startofCalc));

      VectorXd vars = VectorXd::Zero(wave.getNumVariables());wave.getVariables(vars);
      if (commrank == 0) {
	vars += grad;
	vars *= wave.cpsArray.size()/wave.cpsArray[0].Variables[0];
	diis.update(vars, grad);
      }

#ifndef SERIAL
      MPI_Bcast(&(grad[0]),     grad.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&(vars[0]),     vars.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
      wave.updateVariables(vars);
      wave.writeWave();
      //wave.incrementVariables(grad);
      
      if (schd.deterministic)
	E0 = evaluateEDeterministic(wave, nalpha, nbeta, norbs, I1, I2, coreE);
      else
	E0 = evaluateEStochastic(wave, nalpha, nbeta, norbs, I1, I2, coreE, stddev, 1000000, 1.e-6);
    }


  }
  if (m == sgd) {
    Eigen::VectorXd prevGrad = Eigen::VectorXd::Zero(wave.getNumVariables());
    
    double momentum = 0.90;
    for (int iter =0; iter<schd.maxIter && gradnorm>schd.tol; iter++) {
      Eigen::VectorXd grad = Eigen::VectorXd::Zero(wave.getNumVariables());
      
      if (schd.deterministic)
	getGradient(wave, E0, nalpha, nbeta, norbs, I1, I2, coreE, grad);
      else
	getStochasticGradient(wave, E0, nalpha, nbeta, norbs, I1, I2, coreE, grad, 500000, 1e-6);


      //if (iter %100 == 0 && iter != 0) schd.gradientFactor *= 0.1;
      gradnorm = grad.squaredNorm();
      for (int i=0; i<grad.rows(); i++) {
	grad(i) = momentum*prevGrad(i)+schd.gradientFactor*grad(i);
      }
      prevGrad = grad;
      
      VectorXd vars = VectorXd::Zero(wave.getNumVariables());wave.getVariables(vars);
      if (commrank == 0) {
	vars = vars-grad;
      }
#ifndef SERIAL
      MPI_Bcast(&(grad[0]),     grad.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&(vars[0]),     vars.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
      wave.updateVariables(vars);
      wave.writeWave();
      //wave.incrementVariables(grad);

      if (commrank == 0)
	std::cout << format("%6i   %14.8f (%8.2e)  %14.8f %8.2f\n") %iter 
	  % E0 % stddev % gradnorm %( (getTime()-startofCalc));
      
      if (schd.deterministic)
	E0 = evaluateEDeterministic(wave, nalpha, nbeta, norbs, I1, I2, coreE);
      else
	E0 = evaluateEStochastic(wave, nalpha, nbeta, norbs, I1, I2, coreE, stddev, 1000000, 1.e-6);
    }

  }
  else if (m==nestorov ) { ///RMSProp
    Eigen::VectorXd prevGrad = Eigen::VectorXd::Zero(wave.getNumVariables());
    
    double momentum = 0.90;
    for (int iter =0; iter<schd.maxIter && gradnorm>schd.tol; iter++) {
      Eigen::VectorXd grad = Eigen::VectorXd::Zero(wave.getNumVariables());
      
      //Nestorov's trick
      {
	VectorXd tempvars = VectorXd::Zero(grad.rows());
	wave.getVariables(tempvars);
	tempvars -= momentum*prevGrad;
	wave.updateVariables(tempvars);
      }
      
      if (schd.deterministic)
	getGradient(wave, E0, nalpha, nbeta, norbs, I1, I2, coreE, grad);
      else
	getStochasticGradient(wave, E0, nalpha, nbeta, norbs, I1, I2, coreE, grad, 500000, 1e-6);
      
      gradnorm = grad.squaredNorm();
      for (int i=0; i<grad.rows(); i++) {
	grad(i) = momentum*prevGrad(i)+schd.gradientFactor*grad(i);
      }
      prevGrad = grad;
      
      VectorXd vars = VectorXd::Zero(wave.getNumVariables());wave.getVariables(vars);
      if (commrank == 0) {
	vars = vars-grad;
      }
#ifndef SERIAL
      MPI_Bcast(&(grad[0]),     grad.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&(vars[0]),     vars.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
      wave.updateVariables(vars);
      wave.writeWave();
      //wave.incrementVariables(grad);

      if (commrank == 0)
	std::cout << format("%6i   %14.8f (%8.2e)  %14.8f %8.2f\n") %iter 
	  % E0 % stddev % gradnorm %( (getTime()-startofCalc));
      //if (commrank == 0)
      //std::cout << format("%6i   %14.8f  %14.8f %8.2f\n") %iter 
      //% E0 %gradnorm %( (getTime()-startofCalc));
      
      if (schd.deterministic)
	E0 = evaluateEDeterministic(wave, nalpha, nbeta, norbs, I1, I2, coreE);
      else
	E0 = evaluateEStochastic(wave, nalpha, nbeta, norbs, I1, I2, coreE, stddev, 1000000, 1.e-6);
    }
  }
  else if (m==rmsprop ) { ///RMSProp
    Eigen::VectorXd prevGrad = Eigen::VectorXd::Zero(wave.getNumVariables());
    Eigen::VectorXd sumsqGrad = Eigen::VectorXd::Zero(wave.getNumVariables());

    double momentum, momentumdecay = schd.momentumDecay, decay = schd.decay;
    double lrt = schd.gradientFactor; int epoch = schd.learningEpoch;
    for (int iter =0; iter<schd.maxIter && gradnorm>schd.tol; iter++) {
      Eigen::VectorXd grad = Eigen::VectorXd::Zero(wave.getNumVariables());

      momentum = schd.momentum*exp(-momentumdecay*iter);
      //Nestorov's trick
      VectorXd vars = VectorXd::Zero(wave.getNumVariables());wave.getVariables(vars);
      if (abs(momentum ) > 1.e-5)
      {
	VectorXd tempvars = VectorXd::Zero(grad.rows());
	wave.getVariables(tempvars);
	tempvars -= momentum*prevGrad;
	wave.updateVariables(tempvars);
      }


      if (schd.deterministic)
	getGradient(wave, E0, nalpha, nbeta, norbs, I1, I2, coreE, grad);
      else {
	getStochasticGradient(wave, E0, nalpha, nbeta, norbs, I1, I2, coreE, grad, schd.stochasticIter, 1e-6);
      }

      lrt = max(0.001, schd.gradientFactor/pow(2.0, floor( (1+iter)/epoch)));
      gradnorm = grad.squaredNorm();
      for (int i=0; i<grad.rows(); i++) {
	sumsqGrad(i) = decay*sumsqGrad(i)+ (1-decay)*grad(i)*grad(i);
	grad(i) = momentum*prevGrad(i)+lrt*grad(i)/sqrt(sumsqGrad(i)+1e-8);
	vars(i) -= grad(i);
      }

      prevGrad = grad;

      //if (iter%50 == 0 && iter != 0)
      //{//reset optimization
      //prevGrad.setZero(); sumsqGrad.setZero();
      wave.updateVariables(vars);
      wave.normalizeAllCPS();
      //}

#ifndef SERIAL
      MPI_Bcast(&(grad[0]),     grad.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&(vars[0]),     vars.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
      wave.updateVariables(vars);
      wave.writeWave();
      //wave.incrementVariables(grad);
      if (commrank == 0)
	std::cout << format("%6i   %14.8f (%8.2e) %14.8f %8.3f %8.2f\n") %iter 
	  % E0 % stddev %(wave.approximateNorm()) %(lrt) %( (getTime()-startofCalc));
      
      if (schd.deterministic)
	E0 = evaluateEDeterministic(wave, nalpha, nbeta, norbs, I1, I2, coreE);
      else
	E0 = evaluateEStochastic(wave, nalpha, nbeta, norbs, I1, I2, coreE, stddev, schd.stochasticIter, 1.e-6);
    }
  }
  else if (m == adam) {//ADAM
    Eigen::VectorXd prevGrad = Eigen::VectorXd::Zero(wave.getNumVariables());
    Eigen::VectorXd m = Eigen::VectorXd::Zero(wave.getNumVariables());
    Eigen::VectorXd v = Eigen::VectorXd::Zero(wave.getNumVariables());
    
    //Algorithm ADAM https://arxiv.org/pdf/1412.6980.pdf
    double Alpha =0.001, beta1 = 0.9, beta2 = 0.999, epsilon=1.e-8;
    //double Alpha =0.01, beta1 = 0.95, beta2 = 0.99, epsilon=1.e-8;
    
    for (int iter =0; iter<schd.maxIter && gradnorm>schd.tol; iter++) {
      Eigen::VectorXd grad = Eigen::VectorXd::Zero(wave.getNumVariables());
      
      if (schd.deterministic)
	getGradient(wave, E0, nalpha, nbeta, norbs, I1, I2, coreE, grad);
      else
	getStochasticGradient(wave, E0, nalpha, nbeta, norbs, I1, I2, coreE, grad, schd.stochasticIter, 1e-6);
      
      VectorXd vars = VectorXd::Zero(wave.getNumVariables());wave.getVariables(vars);
      gradnorm = grad.squaredNorm();
      m = beta1*m + (1.-beta1)*grad;    
      double alphat = Alpha * sqrt(1. - pow(1.*beta2, iter+1))/(1 - pow(beta1, iter+1));
      for (int i=0; i<grad.rows(); i++) {
	v(i) = beta2*v(i) + (1.-beta2)*grad(i)*grad(i); 
	vars(i) = vars(i) - alphat*m(i)/(sqrt(v(i)) + epsilon);
      }
      
      
#ifndef SERIAL
      MPI_Bcast(&(grad[0]),     grad.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&(vars[0]),     vars.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
      wave.updateVariables(vars);
      wave.normalizeAllCPS();
      wave.writeWave();
      //wave.incrementVariables(grad);
      if (commrank == 0)
	std::cout << format("%6i   %14.8f (%8.2e)  %14.8f %8.2f\n") %iter 
	  % E0 % stddev % gradnorm %( (getTime()-startofCalc));
      //if (commrank == 0)
      //std::cout << format("%6i   %14.8f  %14.8f %8.2f\n") %iter 
      //% E0 %gradnorm %( (getTime()-startofCalc));
      
      if (schd.deterministic)
	E0 = evaluateEDeterministic(wave, nalpha, nbeta, norbs, I1, I2, coreE);
      else
	E0 = evaluateEStochastic(wave, nalpha, nbeta, norbs, I1, I2, coreE, stddev, schd.stochasticIter, 1.e-6);
      
      
    }
  }
  else if (m == amsgrad) {//AMSGRAD
    Eigen::VectorXd prevGrad = Eigen::VectorXd::Zero(wave.getNumVariables());
    Eigen::VectorXd m = Eigen::VectorXd::Zero(wave.getNumVariables());
    Eigen::VectorXd v = Eigen::VectorXd::Zero(wave.getNumVariables());
    
    //Algorithm ADAM https://arxiv.org/pdf/1412.6980.pdf
    double Alpha =0.01, beta1 = 0.9, beta2 = 0.999, epsilon=1.e-8, momentum=0.9;
    //double Alpha =0.01, beta1 = 0.95, beta2 = 0.99, epsilon=1.e-8;
    
    for (int iter =0; iter<schd.maxIter && gradnorm>schd.tol; iter++) {
      Eigen::VectorXd grad = Eigen::VectorXd::Zero(wave.getNumVariables());
      
      if (schd.deterministic)
	getGradient(wave, E0, nalpha, nbeta, norbs, I1, I2, coreE, grad);
      else
	getStochasticGradient(wave, E0, nalpha, nbeta, norbs, I1, I2, coreE, grad, 500000, 1e-6);
      
      VectorXd vars = VectorXd::Zero(wave.getNumVariables());wave.getVariables(vars);
      gradnorm = grad.squaredNorm();
      m = beta1*m + (1.-beta1)*grad;    
      for (int i=0; i<grad.rows(); i++) {
	v(i) = max(v(i), beta2*v(i) + (1.-beta2)*grad(i)*grad(i)); 
	vars(i) = vars(i)  - Alpha*m(i)/(sqrt(v(i)) + epsilon);
      }
      
#ifndef SERIAL
      MPI_Bcast(&(grad[0]),     grad.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&(vars[0]),     vars.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
      wave.updateVariables(vars);
      wave.writeWave();
      //wave.incrementVariables(grad);
      if (commrank == 0)
	std::cout << format("%6i   %14.8f (%8.2e)  %14.8f %8.2f\n") %iter 
	  % E0 % stddev % gradnorm %( (getTime()-startofCalc));
      //if (commrank == 0)
      //std::cout << format("%6i   %14.8f  %14.8f %8.2f\n") %iter 
      //% E0 %gradnorm %( (getTime()-startofCalc));
      
      if (schd.deterministic)
	E0 = evaluateEDeterministic(wave, nalpha, nbeta, norbs, I1, I2, coreE);
      else
	E0 = evaluateEStochastic(wave, nalpha, nbeta, norbs, I1, I2, coreE, stddev, 1000000, 1.e-6);
      
      
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
