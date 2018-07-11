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
  //license();

  string inputFile = "input.dat";
  if (argc > 1)
    inputFile = string(argv[1]);
  if (commrank == 0) readInput(inputFile, schd, false);
#ifndef SERIAL
  mpi::broadcast(world, schd, 0);
#endif

  generator = std::mt19937(schd.seed+commrank);
  //generator = std::mt19937(commrank);

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

  ifstream file ("params.bin", ios::in|ios::binary|ios::ate);
  size_t size = file.tellg();
  Eigen::VectorXd vars = Eigen::VectorXd::Zero(size/sizeof(double));
  file.seekg (0, ios::beg);
  file.read ( (char*)(&vars[0]), size);
  file.close();

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

  Eigen::VectorXd grad = Eigen::VectorXd::Zero(vars.size());
  Eigen::MatrixXd Hessian, Smatrix;
  if (schd.doHessian) {
    Hessian.resize(vars.size()+1, vars.size()+1);
    Smatrix.resize(vars.size()+1, vars.size()+1);
    Hessian.setZero(); Smatrix.setZero();
  }

  double E0=0.0, stddev, rt=0;
  if (schd.deterministic) {
    if (!schd.doHessian) {
      getGradientDeterministic(wave, E0, nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE, grad);
      stddev = 0.0;
    }
    else {
      getGradientHessianDeterministic(wave, E0, nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE, grad, Hessian, Smatrix);
      stddev = 0.0;
    }
  }
  else {
    //getStochasticGradient(wave, E0, stddev, nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE, grad, rt, schd.stochasticIter, 0.5e-3);
    if (!schd.doHessian) 
      getStochasticGradientContinuousTime(wave, E0, stddev, nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE, grad, rt, schd.stochasticIter, 0.5e-3);
    else
      getStochasticGradientHessianContinuousTime(wave, E0, stddev, nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE, grad, Hessian, Smatrix, rt, schd.stochasticIter, 0.5e-3);

  }

  //for (int i=wave.getNumJastrowVariables(); i<wave.getNumVariables(); i++) {
  //grad[i] *= getParityForDiceToAlphaBeta(wave.determinants[i-wave.getNumJastrowVariables()]);
  //}

  if (commrank == 0)
    std::cout << format("%14.8f (%8.2e) %14.8f %8.1f %10i %8.2f\n") 
      %E0 % stddev %(grad.norm()) %(rt)  %(schd.stochasticIter) %( (getTime()-startofCalc));

  {
    ofstream file ("grad.bin", ios::out|ios::binary);
    file.write ( (char*)(&grad[0]), size);
    file.close();

    if (schd.doHessian)
    {
      ofstream hfile("hessian.bin", ios::out | ios::binary);
      hfile.write((char *)(&Hessian(0,0)), Hessian.rows()*Hessian.cols()*sizeof(double));
      hfile.close();

      ofstream sfile("smatrix.bin", ios::out | ios::binary);
      sfile.write((char *)(&Smatrix(0,0)), Hessian.rows()*Hessian.cols()*sizeof(double));
      sfile.close();
    }
  }

  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());
  return 0;
}
