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
#include <boost/function.hpp>
#include <boost/functional.hpp>
#include <boost/bind.hpp>
#include "evaluateE.h"
#include "Determinants.h"
#include "CPSSlater.h"
#include "HFWalker.h"
#include "input.h"
#include "integral.h"
#include "SHCIshm.h"
#include "math.h"
#include "Profile.h"
#include "CIWavefunction.h"
#include "propagate.h"

using namespace Eigen;
using namespace boost;
using namespace std;

int main(int argc, char *argv[])
{

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
  readInput(inputFile, schd, false);

  generator = std::mt19937(schd.seed + commrank);

  readIntegralsAndInitializeDeterminantStaticVariables("FCIDUMP");


  workingArray work;
  if (schd.wavefunctionType == "CPSSlater") {
    //initialize wavefunction
    CPSSlater wave; HFWalker walk;
    wave.readWave(); wave.initWalker(walk);

    //calculate the energy as a initial guess for shift
    double ham, stddev, rk;
    getStochasticEnergyContinuousTime(wave, walk, ham, stddev, rk, schd.stochasticIter, 1.e-5);
    work.clear();
    if (commrank == 0) cout << "Energy of VMC wavefunction: "<<ham <<"("<<stddev<<")"<<endl;

    //do the GFMC continous time
    doGFMCCT(wave, walk, ham);
  }
  else if (schd.wavefunctionType == "CICPSSlater") {
    CIWavefunction<CPSSlater, HFWalker, Operator> wave; HFWalker walk;
    wave.readWave(); wave.initWalker(walk);

    //calculate the energy as a initial guess for shift
    double ham, stddev, rk;
    getStochasticEnergyContinuousTime(wave, walk, ham, stddev, rk, schd.stochasticIter, 1.e-5);
    work.clear();
    if (commrank == 0) cout << "Energy of VMC wavefunction: "<<ham <<"("<<stddev<<")"<<endl;

    //do the GFMC continous time
    doGFMCCT(wave, walk, ham);
  }


  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());
  return 0;
}

