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
  MoDeterminant::norbs  = norbs;
  MoDeterminant::nalpha = nalpha;
  MoDeterminant::nbeta  = nbeta;

  //Setup Slater Determinants
  HforbsA = MatrixXd::Zero(norbs, norbs);
  HforbsB = MatrixXd::Zero(norbs, norbs);
  readHF(HforbsA, HforbsB, schd.uhf);


  //Setup CPS wavefunctions
  std::vector<Correlator> nSiteCPS;
  for (auto it = schd.correlatorFiles.begin(); it != schd.correlatorFiles.end();
       it++) {
    readCorrelator(it->second, it->first, nSiteCPS);
  }

  vector<Determinant> detList; vector<double> ciExpansion;

  if (boost::iequals(schd.determinantFile, "") )
  {
    detList.resize(1); ciExpansion.resize(1, 1.0);
    for (int i=0; i<nalpha; i++)
      detList[0].setoccA(i, true);
    for (int i=0; i<nbeta; i++)
      detList[0].setoccB(i, true);
  }
  else 
  {
    readDeterminants(schd.determinantFile, detList, ciExpansion);
  }
  //setup up wavefunction
  CPSSlater wave(nSiteCPS, detList, ciExpansion);

  size_t size;
  ifstream file ("params.bin", ios::in|ios::binary|ios::ate);
  if (commrank == 0) {
    size = file.tellg();
  }
#ifndef SERIAL
  MPI_Bcast(&size, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  Eigen::VectorXd vars = Eigen::VectorXd::Zero(size/sizeof(double));
  if (commrank == 0) {
    file.seekg (0, ios::beg);
    file.read ( (char*)(&vars[0]), size);
    file.close();
  }

#ifndef SERIAL
  MPI_Bcast(&vars[0], vars.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

  if ( (schd.uhf && vars.size() != wave.getNumVariables()+2*norbs*norbs) ||
       (!schd.uhf && vars.size() != wave.getNumVariables()+norbs*norbs) ){
    cout << "number of variables on disk: "<<vars.size()<<" is not equal to wfn parameters: "<<wave.getNumVariables()<<endl;
    exit(0);
  }

  wave.updateVariables(vars);
  int numVars = wave.getNumVariables();

  for (int i=0; i<norbs; i++) {
    for (int j=0; j<norbs; j++) {
      if (!schd.uhf) {
	HforbsA(i,j) = vars[numVars + i *norbs + j];
	HforbsB(i,j) = vars[numVars + i *norbs + j];
      }
      else {
	HforbsA(i,j) = vars[numVars + i *norbs + j];
	HforbsB(i,j) = vars[numVars + norbs*norbs + i *norbs + j];
      }
    }
  }

  MatrixXd alpha(norbs, nalpha), beta(norbs, nbeta);
  alpha = HforbsA.block(0, 0, norbs, nalpha);
  beta  = HforbsB.block(0, 0, norbs, nbeta );
  MoDeterminant det(alpha, beta);

  /*
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
    /
    
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
*/
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(vars.size());
  double unscaledE, stddev, rk;
  /*
  double scaledE0 = evaluateScaledEDeterministic(wave, schd.PTlambda, unscaledE, 
						   nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE);
  double PTdet = evaluatePTDeterministic(wave, scaledE0, 
					 nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE);
  */
  //if (commrank == 0) cout << scaledE0<<"  "<<unscaledE<<"  "<<PTdet<<endl;

  double scaledE = evaluateScaledEStochastic(wave, schd.PTlambda, unscaledE, nalpha, nbeta, norbs, 
					     I1, I2, I2HBSHM, coreE, stddev, rk,
					     schd.stochasticIter, 0.5e-3);

  double A, B, C, stddev2;
  double PTstoc = evaluatePTStochasticMethodB(wave, unscaledE, nalpha, nbeta, norbs, 
					      I1, I2, I2HBSHM, coreE, stddev2, rk,
					      schd.stochasticIter, A, B, C);
  if (commrank == 0) cout << PTstoc<<"  "<<stddev2<<endl;
  if (commrank == 0) {
    std::cout << format("%14.8f (%8.2e) %14.8f (%8.2e) %10.2f %10i %8.2f\n") 
      %unscaledE % stddev %(unscaledE+A+B*B/C) %stddev2 %rk %(schd.stochasticIter) %( (getTime()-startofCalc));

  }

  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());
  return 0;
}
