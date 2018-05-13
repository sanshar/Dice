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
#include "evaluatePT.h"
#include "MoDeterminants.h"
#include "Determinants.h"
#include "CPS.h"
#include "Wfn.h"
#include "input.h"
#include "integral.h"
#include "SHCIshm.h"
#include "math.h"
#include "diis.h"
#include "optimizer.h"

using namespace Eigen;
using namespace boost;
using namespace std;


int main(int argc, char* argv[]) {

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
  if (commrank == 0) readInput(inputFile, schd);
#ifndef SERIAL
  mpi::broadcast(world, schd, 0);
#endif

  generator = std::mt19937(schd.seed+commrank);

  twoInt I2; oneInt I1; 
  int norbs, nalpha, nbeta; 
  double coreE=0.0;
  std::vector<int> irrep;
  readIntegrals("FCIDUMP", I2, I1, nalpha, nbeta, norbs, coreE, irrep);
  
  //initialize the heatbath integrals
  std::vector<int> allorbs;
  for (int i=0; i<norbs; i++)
    allorbs.push_back(i);
  twoIntHeatBath I2HB(1.e-10);
  twoIntHeatBathSHM I2HBSHM(1.e-10);
  if (commrank == 0) I2HB.constructClass(allorbs, I2, I1, norbs);
  I2HBSHM.constructClass(norbs, I2HB);

  
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

  //setup up wavefunction and read from disk
  CPSSlater wave(nSiteCPS, det);
  if (schd.restart) {
    ifstream file ("params_min.bin", ios::in|ios::binary|ios::ate);
    size_t size = file.tellg();
    Eigen::VectorXd vars = Eigen::VectorXd::Zero(size/sizeof(double));
    file.seekg (0, ios::beg);
    file.read ( (char*)(&vars[0]), size);
    file.close();
    
    if (vars.size() != wave.getNumVariables()) {
      cout << "number of variables on disk: "<<vars.size()<<" is not equal to wfn parameters: "<<wave.getNumVariables()<<endl;
      exit(0);
    }
    
    wave.updateVariables(vars);


    double stddev=0., lambda = schd.PTlambda;
    double unscaledE = 0;
    if (schd.deterministic) {
      double scaledE0 = evaluateScaledEDeterministic(wave, lambda, unscaledE, 
						     nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE);
      if (commrank == 0) cout << format("%14.8f (%8.2e) , E(lambda) = %14.8f\n") 
			   %(unscaledE) % (stddev) %(scaledE0);
      //double PT = evaluatePTDeterministic(wave, scaledE0, nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE);
      //double PT = evaluatePTDeterministicC(wave, scaledE0, nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE);
      double PT = evaluatePTDeterministicD(wave, scaledE0, nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE);
      if (commrank == 0) cout << format("%14.8f (%8.2e) \n") %(unscaledE+PT) % (stddev);
    }
    else {
      //double scaledE0 = evaluateScaledEStochastic(wave, lambda, unscaledE, 
      //nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE,
      //stddev, schd.stochasticIter);

      double scaledE0 = evaluateScaledEDeterministic(wave, lambda, unscaledE, 
      nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE);
      //double scaledE0 = -74.65542308;

      if (commrank == 0) cout << format("%14.8f (%8.2e) , E(lambda) = %14.8f\n") 
			   %(unscaledE) % (stddev) %(scaledE0);
      double A2, B, C, A3;
      double stddevA, stddevB, stddevC;
      double PT;

      /*
      {
	PT = evaluatePTStochasticMethodD(wave, scaledE0, nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE, stddevA,  stddevB, stddevC, schd.stochasticIter, A2, B, C);
	if (commrank == 0) cout <<format("%14.8f (%6.2e)  %14.8f (%6.2e)  %14.8f (%6.2e)   %14.8f\n") 
			     %(A2) %(stddevA) %B  %stddevB  %C %stddevC  %(A2/B);
      }
      */

      {
	double PT = evaluatePTStochasticMethodC(wave, scaledE0, nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE, stddevA,  stddevB, stddevC, schd.stochasticIter, A2, B, C);
      //double PT = evaluatePTStochastic3rdOrder(wave, scaledE0, nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE, stddev, schd.stochasticIter, A2, B, C, A3);
      //double PT = evaluatePTStochasticMethodA(wave, scaledE0, nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE, stddevA, schd.stochasticIter, A2, B, C);
	if (commrank == 0) cout <<format("%14.8f (%6.2e)  %14.8f (%6.2e)  %14.8f (%6.2e)   %14.8f\n") 
			     %(A2) %(stddevA) %B  %stddevB  %C %stddevC  %(A2+B*B/C);
      }

      if (commrank == 0) cout << format("%14.8f (%8.2e) \n") %(unscaledE+PT) % (stddev);
    }
  }
  else {
    cout << "Use the python script to optimize the wavefunction."<<endl;
    exit(0);
  }


  cout.precision(10);



  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  return 0;
}
