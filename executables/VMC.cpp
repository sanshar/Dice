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
#include "CPSGHFSlater.h"
#include "GHFWalker.h"
#include "input.h"
#include "integral.h"
#include "SHCIshm.h"
#include "math.h"
#include "Profile.h"
#include "amsgrad.h"
#include "CIWavefunction.h"


using namespace Eigen;
using namespace boost;
using namespace std;

typedef boost::function<void (VectorXd&, VectorXd&, double&, double&, double&)> functor1;

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


  //calculate the hessian/gradient
  if (schd.wavefunctionType == "CPSSlater") {
    CPSSlater wave; HFWalker walk;
    wave.readDefault();
    if (schd.restart) wave.readWave();
    VectorXd vars; wave.getVariables(vars);

    getGradientWrapper<CPSSlater, HFWalker> wrapper(wave, walk, schd.stochasticIter);

    if (schd.method == amsgrad) {
      AMSGrad optimizer(schd.stepsize, schd.decay1, schd.decay2, schd.maxIter);
      functor1 getStochasticGradient = boost::bind(&getGradientWrapper<CPSSlater, HFWalker>::getGradient, &wrapper, _1, _2, _3, _4, _5, schd.deterministic);
      optimizer.optimize(vars, getStochasticGradient, schd.restart);
    }
    else if (schd.method == linearmethod) {

    }
  }
  else if (schd.wavefunctionType == "CPSGHFSlater") {
    CPSGHFSlater wave; GHFWalker walk;
    wave.readDefault();
    if (schd.restart) wave.readWave();
    VectorXd vars; wave.getVariables(vars);

    getGradientWrapper<CPSGHFSlater, GHFWalker> wrapper(wave, walk, schd.stochasticIter);

    if (schd.method == amsgrad) {
      AMSGrad optimizer(schd.stepsize, schd.decay1, schd.decay2, schd.maxIter);
      functor1 getStochasticGradient = boost::bind(&getGradientWrapper<CPSGHFSlater, GHFWalker>::getGradient, &wrapper, _1, _2, _3, _4, _5, schd.deterministic);
      optimizer.optimize(vars, getStochasticGradient, schd.restart);
    }
    else if (schd.method == linearmethod) {

    }
  }
  else if (schd.wavefunctionType == "CICPSSlater") {
    //CIWavefunction<CPSSlater, HFWalker, Operator> wave;
    CIWavefunction<CPSSlater, HFWalker, SpinFreeOperator> wave;
    wave.appendSinglesToOpList(); wave.appendScreenedDoublesToOpList(0.0);
    HFWalker walk;
    VectorXd vars; wave.getVariables(vars);

    //getGradientWrapper<CIWavefunction<CPSSlater, HFWalker, Operator>, HFWalker> wrapper(wave, walk, schd.stochasticIter);
    getGradientWrapper<CIWavefunction<CPSSlater, HFWalker, SpinFreeOperator>, HFWalker> wrapper(wave, walk, schd.stochasticIter);

    if (schd.method == amsgrad) {
      AMSGrad optimizer(schd.stepsize, schd.decay1, schd.decay2, schd.maxIter);
      //functor1 getStochasticGradient = boost::bind(&getGradientWrapper<CIWavefunction<CPSSlater, HFWalker, Operator>, HFWalker>::getGradient, &wrapper, _1, _2, _3, _4, _5, schd.deterministic);
      functor1 getStochasticGradient = boost::bind(&getGradientWrapper<CIWavefunction<CPSSlater, HFWalker, SpinFreeOperator>, HFWalker>::getGradient, &wrapper, _1, _2, _3, _4, _5, schd.deterministic);
      optimizer.optimize(vars, getStochasticGradient, schd.restart);
      if (commrank == 0) wave.printVariables();
    }
  }

  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());
  return 0;
}
