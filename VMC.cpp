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
  //generator = std::mt19937(commrank);

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


  //optimize the wavefunction
  if (schd.m == rmsprop) 
    optimizer::rmsprop(wave, I1, I2, coreE);

  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  return 0;
}
