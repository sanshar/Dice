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
#include "CPSAGP.h"
#include "AGPWalker.h"
#include "input.h"
#include "integral.h"
#include "SHCIshm.h"
#include "math.h"
#include "Profile.h"
#include "CIWavefunction.h"
#include "runVMC.h"

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


  //calculate the hessian/gradient
  if (schd.wavefunctionType == "CPSSlater") {
    CPSSlater wave; HFWalker walk;
    runVMC(wave, walk);
  }
  
  else if (schd.wavefunctionType == "CICPSSlater") {
    CIWavefunction<CPSSlater, HFWalker, SpinFreeOperator> wave; HFWalker walk;
    wave.appendSinglesToOpList(); wave.appendScreenedDoublesToOpList(0.0);
    runVMC(wave, walk);
  }
  
  
  else if (schd.wavefunctionType == "CPSAGP") {
    CPSAGP wave; AGPWalker walk;
    if (schd.restart) wave.readWave();
    VectorXd vars; wave.getVariables(vars);

    getGradientWrapper<CPSAGP, AGPWalker> wrapper(wave, walk, schd.stochasticIter);
    functor1 getStochasticGradient = boost::bind(&getGradientWrapper<CPSAGP, AGPWalker>::getGradient, &wrapper, _1, _2, _3, _4, _5, schd.deterministic);

    if (schd.method == amsgrad) {
      AMSGrad optimizer(schd.stepsize, schd.decay1, schd.decay2, schd.maxIter);
      optimizer.optimize(vars, getStochasticGradient, schd.restart);
      //if (commrank == 0) wave.printVariables();
    }
    else if (schd.method == sgd) {
      SGD optimizer(schd.stepsize, schd.maxIter);
      optimizer.optimize(vars, getStochasticGradient, schd.restart);
    }
    else if (schd.method == linearmethod) {

    }
  }
  
  else if (schd.wavefunctionType == "LanczosCPSSlater") {
    //CIWavefunction<CPSSlater, HFWalker, Operator> wave;
    CPSSlater wave; HFWalker walk;
    wave.readWave();
    wave.initWalker(walk); 
    Eigen::VectorXd stddev = Eigen::VectorXd::Zero(4);
    Eigen::VectorXd rk = Eigen::VectorXd::Zero(4);
    //double rk = 0;
    Eigen::VectorXd lanczosCoeffs = Eigen::VectorXd::Zero(4);
    double alpha = 0.1;
    if (schd.deterministic) getLanczosCoeffsDeterministic(wave, walk, alpha, lanczosCoeffs);
    else getLanczosCoeffsContinuousTime(wave, walk, alpha, lanczosCoeffs, stddev, rk, schd.stochasticIter, 1.e-5);
    //getLanczosMatrixContinuousTime(wave, walk, lanczosMat, stddev, rk, schd.stochasticIter, 1.e-5);
    double a = lanczosCoeffs[2]/lanczosCoeffs[3];
    double b = lanczosCoeffs[1]/lanczosCoeffs[3];
    double c = lanczosCoeffs[0]/lanczosCoeffs[3];
    double delta = pow(a, 2) + 4 * pow(b, 3) - 6 * a * b * c - 3 * pow(b * c, 2) + 4 * a * pow(c, 3);
    double numP = a - b * c + pow(delta, 0.5);
    double numM = a - b * c - pow(delta, 0.5);
    double denom = 2 * pow(b, 2) - 2 * a * c;
    double alphaP = numP/denom;
    double alphaM = numM/denom;
    double eP = (a * pow(alphaP, 2) + 2 * b * alphaP + c) / (b * pow(alphaP, 2) + 2 * c * alphaP + 1);
    double eM = (a * pow(alphaM, 2) + 2 * b * alphaM + c) / (b * pow(alphaM, 2) + 2 * c * alphaM + 1);
    if (commrank == 0) {
      cout << "lanczosCoeffs\n";
      cout << lanczosCoeffs << endl;
      cout << "stddev\n";
      cout << stddev << endl;
      cout << "rk\n";
      cout << rk << endl;
      cout << "alpha(+/-)   " << alphaP << "   " << alphaM << endl;
      cout << "energy(+/-)   " << eP << "   " << eM << endl;
      //cout << "rk\n" << rk << endl << endl;
      //cout << "stddev\n" << stddev << endl << endl;
    }

    //vector<double> alpha{0., 0.1, 0.2, -0.1, -0.2}; 
    //vector<double> Ealpha{0., 0., 0., 0., 0.}; 
    //double stddev, rk;
    //for (int i = 0; i < alpha.size(); i++) {
    //  vars[0] = alpha[i];
    //  wave.updateVariables(vars);
    //  wave.initWalker(walk);
    //  getStochasticEnergyContinuousTime(wave, walk, Ealpha[i], stddev, rk, schd.stochasticIter, 1.e-5);
    //  if (commrank == 0) cout << alpha[i] << "   " << Ealpha[i] << "   " << stddev << endl;
    //}

    //getGradientWrapper<CIWavefunction<CPSSlater, HFWalker, Operator>, HFWalker> wrapper(wave, walk, schd.stochasticIter);
    //getGradientWrapper<Lanczos<CPSSlater, HFWalker>, HFWalker> wrapper(wave, walk, schd.stochasticIter);
    //  functor1 getStochasticGradient = boost::bind(&getGradientWrapper<Lanczos<CPSSlater, HFWalker>, HFWalker>::getGradient, &wrapper, _1, _2, _3, _4, _5, schd.deterministic);

    //if (schd.method == amsgrad) {
    //  AMSGrad optimizer(schd.stepsize, schd.decay1, schd.decay2, schd.maxIter);
    //  //functor1 getStochasticGradient = boost::bind(&getGradientWrapper<CIWavefunction<CPSSlater, HFWalker, Operator>, HFWalker>::getGradient, &wrapper, _1, _2, _3, _4, _5, schd.deterministic);
    //  optimizer.optimize(vars, getStochasticGradient, schd.restart);
    //  //if (commrank == 0) wave.printVariables();
    //}
    //else if (schd.method == sgd) {
    //  SGD optimizer(schd.stepsize, schd.maxIter);
    //  optimizer.optimize(vars, getStochasticGradient, schd.restart);
    //}
  }

  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());
  return 0;
}
