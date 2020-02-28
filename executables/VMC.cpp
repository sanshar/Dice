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
//#define EIGEN_USE_MKL_ALL
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
#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include <boost/function.hpp>
#include <boost/functional.hpp>
#include <boost/bind.hpp>
#include "evaluateE.h"
#include "Determinants.h"
#include "input.h"
#include "integral.h"
#include "SHCIshm.h"
#include "math.h"
#include "Profile.h"
#include "CIWavefunction.h"
#include "CorrelatedWavefunction.h"
#include "ResonatingWavefunction.h"
#include "TRWavefunction.h"
#include "PermutedWavefunction.h"
#include "PermutedTRWavefunction.h"
#include "SelectedCI.h"
#include "SimpleWalker.h"
#include "Lanczos.h"
#include "SCCI.h"
#include "SCPT.h"
#include "MRCI.h"
#include "EOM.h"
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
  if (commrank == 0) {
    system("echo User:; echo $USER");
    system("echo Hostname:; echo $HOSTNAME");
    system("echo CPU info:; lscpu | head -15");
    system("echo Computation started at:; date");
    cout << "git commit: " << GIT_HASH << ", branch: " << GIT_BRANCH << ", compiled at: " << COMPILE_TIME << endl << endl;
    cout << "nproc used: " << commsize << " (NB: stochasticIter below is per proc)" << endl << endl; 
  }

  string inputFile = "input.dat";
  if (argc > 1)
    inputFile = string(argv[1]);
  readInput(inputFile, schd, false);

  generator = std::mt19937(schd.seed + commrank);

  readIntegralsAndInitializeDeterminantStaticVariables("FCIDUMP");
  if (schd.numActive == -1) schd.numActive = Determinant::norbs;

  //calculate the hessian/gradient
  if (schd.wavefunctionType == "cpsslater") {
    CorrelatedWavefunction<CPS, Slater> wave; Walker<CPS, Slater> walk;
    runVMC(wave, walk);
  }
  
  else if (schd.wavefunctionType == "cpsagp") {
    CorrelatedWavefunction<CPS, AGP> wave; Walker<CPS, AGP> walk;
    runVMC(wave, walk);
  }
  
  else if (schd.wavefunctionType == "cpspfaffian") {
    CorrelatedWavefunction<CPS, Pfaffian> wave; Walker<CPS, Pfaffian> walk;
    runVMC(wave, walk);
  }
  
  if (schd.wavefunctionType == "jastrowslater") {
    CorrelatedWavefunction<Jastrow, Slater> wave; Walker<Jastrow, Slater> walk;
    runVMC(wave, walk);
  }
  
  if (schd.wavefunctionType == "resonatingwavefunction") {
    ResonatingWavefunction wave; ResonatingWalker walk;
    runVMC(wave, walk);
  }
  
  if (schd.wavefunctionType == "trwavefunction") {
    TRWavefunction wave; TRWalker walk;
    runVMC(wave, walk);
  }
  
  if (schd.wavefunctionType == "permutedwavefunction") {
    PermutedWavefunction wave; PermutedWalker walk;
    runVMC(wave, walk);
  }
  
  if (schd.wavefunctionType == "permutedtrwavefunction") {
    PermutedTRWavefunction wave; PermutedTRWalker walk;
    runVMC(wave, walk);
  }
  
  
  else if (schd.wavefunctionType == "jastrowagp") {
    CorrelatedWavefunction<Jastrow, AGP> wave; Walker<Jastrow, AGP> walk;
    runVMC(wave, walk);
  }
  
  else if (schd.wavefunctionType == "jastrowpfaffian") {
    CorrelatedWavefunction<Jastrow, Pfaffian> wave; Walker<Jastrow, Pfaffian> walk;
    runVMC(wave, walk);
  }
  
  else if (schd.wavefunctionType == "rbm") {
    CorrelatedWavefunction<RBM, Slater> wave; Walker<RBM, Slater> walk;
    runVMC(wave, walk);
  }
  
  else if (schd.wavefunctionType == "jrbms") {
    CorrelatedWavefunction<JRBM, Slater> wave; Walker<JRBM, Slater> walk;
    runVMC(wave, walk);
  }
  
  else if (schd.wavefunctionType == "jrbmp") {
    CorrelatedWavefunction<JRBM, Pfaffian> wave; Walker<JRBM, Pfaffian> walk;
    runVMC(wave, walk);
  }
  
  else if (schd.wavefunctionType == "cicpsslater") {
    CIWavefunction<CorrelatedWavefunction<CPS, Slater>, Walker<CPS, Slater>, SpinFreeOperator> wave; Walker<CPS, Slater> walk;
    wave.appendSinglesToOpList(0.0); wave.appendScreenedDoublesToOpList(0.0);
    runVMC(wave, walk);
  }
  
  else if (schd.wavefunctionType == "cicpsagp") {
    CIWavefunction<CorrelatedWavefunction<CPS, AGP>, Walker<CPS, AGP>, SpinFreeOperator> wave; Walker<CPS, AGP> walk;
    wave.appendSinglesToOpList(0.0); wave.appendScreenedDoublesToOpList(0.0);
    runVMC(wave, walk);
  }
  
  else if (schd.wavefunctionType == "cicpspfaffian") {
    CIWavefunction<CorrelatedWavefunction<CPS, Pfaffian>, Walker<CPS, Pfaffian>, SpinFreeOperator> wave; Walker<CPS, Pfaffian> walk;
    wave.appendSinglesToOpList(0.0); wave.appendScreenedDoublesToOpList(0.0);
    runVMC(wave, walk);
  }
  
  else if (schd.wavefunctionType == "cijastrowslater") {
    //CIWavefunction<CorrelatedWavefunction<Jastrow, Slater>, Walker<Jastrow, Slater>, SpinFreeOperator> wave; Walker<Jastrow, Slater> walk;
    CIWavefunction<CorrelatedWavefunction<Jastrow, Slater>, Walker<Jastrow, Slater>, Operator> wave; Walker<Jastrow, Slater> walk;
    wave.appendSinglesToOpList(0.0); wave.appendScreenedDoublesToOpList(0.0);
    runVMC(wave, walk);
  }
  
  else if (schd.wavefunctionType == "cijastrowagp") {
    CIWavefunction<CorrelatedWavefunction<Jastrow, AGP>, Walker<Jastrow, AGP>, SpinFreeOperator> wave; Walker<Jastrow, AGP> walk;
    wave.appendSinglesToOpList(0.0); wave.appendScreenedDoublesToOpList(0.0);
    runVMC(wave, walk);
  }
  
  else if (schd.wavefunctionType == "cijastrowpfaffian") {
    CIWavefunction<CorrelatedWavefunction<Jastrow, Pfaffian>, Walker<Jastrow, Pfaffian>, SpinFreeOperator> wave; Walker<Jastrow, Pfaffian> walk;
    wave.appendSinglesToOpList(0.0); wave.appendScreenedDoublesToOpList(0.0);
    runVMC(wave, walk);
  }
  
  else if (schd.wavefunctionType == "sci") {
    CIWavefunction<SelectedCI, SimpleWalker, Operator> wave; SimpleWalker walk;
    wave.appendSinglesToOpList(0.0); wave.appendScreenedDoublesToOpList(0.0);
    runVMC(wave, walk);
  }
  
  else if (schd.wavefunctionType == "lanczoscpslater") {
    Lanczos<CorrelatedWavefunction<CPS, Slater>> wave; Walker<CPS, Slater> walk;
    wave.initWalker(walk);
    wave.optimizeWave(walk);
    wave.writeWave();
  }
  
  else if (schd.wavefunctionType == "lanczoscpsagp") {
    Lanczos<CorrelatedWavefunction<CPS, AGP>> wave; Walker<CPS, AGP> walk;
    wave.initWalker(walk);
    wave.optimizeWave(walk);
    wave.writeWave();
  }
  
  else if (schd.wavefunctionType == "lanczoscpspfaffian") {
    Lanczos<CorrelatedWavefunction<CPS, Pfaffian>> wave; Walker<CPS, Pfaffian> walk;
    wave.initWalker(walk);
    wave.optimizeWave(walk);
    wave.writeWave();
  }
  
  else if (schd.wavefunctionType == "lanczosjastrowslater") {
    Lanczos<CorrelatedWavefunction<Jastrow, Slater>> wave; Walker<Jastrow, Slater> walk;
    wave.initWalker(walk);
    wave.optimizeWave(walk);
    wave.writeWave();
  }
  
  else if (schd.wavefunctionType == "lanczosjastrowagp") {
    Lanczos<CorrelatedWavefunction<Jastrow, AGP>> wave; Walker<Jastrow, AGP> walk;
    wave.initWalker(walk);
    wave.optimizeWave(walk);
    wave.writeWave();
  }
  
  else if (schd.wavefunctionType == "lanczosjastrowpfaffian") {
    Lanczos<CorrelatedWavefunction<Jastrow, Pfaffian>> wave; Walker<Jastrow, Pfaffian> walk;
    wave.initWalker(walk);
    wave.optimizeWave(walk);
    wave.writeWave();
  }
  
  else if (schd.wavefunctionType == "lanczossci") {
    Lanczos<SelectedCI> wave; SimpleWalker walk;
    wave.initWalker(walk);
    double alpha = wave.optimizeWave(walk, schd.alpha);
    wave.writeWave();
  }
  
  else if (schd.wavefunctionType == "scci") {
    SCCI<SelectedCI> wave; SimpleWalker walk;
    wave.initWalker(walk);
    if (schd.method == linearmethod) {
      wave.optimizeWaveCTDirect(walk); 
      wave.optimizeWaveCTDirect(walk);
    }
    else {
      runVMC(wave, walk);
      wave.calcEnergy(walk);
    }
    wave.writeWave();
  }
  
  else if (schd.wavefunctionType == "MRCI") {
    MRCI<Jastrow, Slater> wave; MRCIWalker<Jastrow, Slater> walk;
    //wave.initWalker(walk);
    runVMC(wave, walk);
  }
  
  else if (schd.wavefunctionType == "eom") {
    EOM<Jastrow, Slater> wave; Walker<Jastrow, Slater> walk;
    //cout << "detetrministic\n\n";
    if (schd.deterministic) wave.optimizeWaveDeterministic(walk); 
    else {
      wave.initWalker(walk);
      wave.optimizeWaveCT(walk); 
    }
    //wave.calcPolDeterministic(walk); 
    //wave.calcPolCT(walk); 
    //wave.initWalker(walk);
    //runVMC(wave, walk);
  }
  
  else if (schd.wavefunctionType == "pol") {
    EOM<Jastrow, Slater> wave; Walker<Jastrow, Slater> walk;
    //cout << "detetrministic\n\n";
    //wave.optimizeWaveDeterministic(walk); 
    //cout << "stochastic\n\n";
    //wave.optimizeWaveCT(walk); 
    if (schd.deterministic) wave.calcPolDeterministic(walk); 
    else {
      wave.initWalker(walk);
      wave.calcPolCT(walk); 
    }
    //wave.initWalker(walk);
    //runVMC(wave, walk);
  }
  
  else if (schd.wavefunctionType == "scpt") {
    SCPT<SelectedCI> wave; SimpleWalker walk;
    wave.initWalker(walk);
    wave.optimizeWaveCT(walk);
    wave.optimizeWaveCT(walk);
  }

  else if (schd.wavefunctionType == "slaterrdm") {
    CorrelatedWavefunction<Jastrow, Slater> wave; Walker<Jastrow, Slater> walk;
    wave.readWave();
    MatrixXd oneRdm0, oneRdm1, corr;
    //getOneRdmDeterministic(wave, walk, oneRdm0, 0);
    //getOneRdmDeterministic(wave, walk, oneRdm1, 1);
    //getDensityCorrelationsDeterministic(wave, walk, corr);
    getStochasticOneRdmContinuousTime(wave, walk, oneRdm0, 0, schd.stochasticIter);
    getStochasticOneRdmContinuousTime(wave, walk, oneRdm1, 1, schd.stochasticIter);
    getStochasticDensityCorrelationsContinuousTime(wave, walk, corr, schd.stochasticIter);
    if (commrank == 0) {
      cout << "oneRdm0\n" << oneRdm0 << endl << endl;
      cout << "oneRdm1\n" << oneRdm1 << endl << endl;
      cout << "Density correlations\n" << corr << endl << endl;
    }
  }
  
  else if (schd.wavefunctionType == "agprdm") {
    CorrelatedWavefunction<Jastrow, AGP> wave; Walker<Jastrow, AGP> walk;
    wave.readWave();
    MatrixXd oneRdm0, oneRdm1, corr;
    //getOneRdmDeterministic(wave, walk, oneRdm0, 0);
    //getOneRdmDeterministic(wave, walk, oneRdm1, 1);
    //getDensityCorrelationsDeterministic(wave, walk, corr);
    getStochasticOneRdmContinuousTime(wave, walk, oneRdm0, 0, schd.stochasticIter);
    getStochasticOneRdmContinuousTime(wave, walk, oneRdm1, 1, schd.stochasticIter);
    getStochasticDensityCorrelationsContinuousTime(wave, walk, corr, schd.stochasticIter);
    if (commrank == 0) {
      cout << "oneRdm0\n" << oneRdm0 << endl << endl;
      cout << "oneRdm1\n" << oneRdm1 << endl << endl;
      cout << "Density correlations\n" << corr << endl << endl;
    }
  }
  
  else if (schd.wavefunctionType == "pfaffianrdm") {
    CorrelatedWavefunction<Jastrow, Pfaffian> wave; Walker<Jastrow, Pfaffian> walk;
    wave.readWave();
    MatrixXd oneRdm0, oneRdm1, corr;
    //getOneRdmDeterministic(wave, walk, oneRdm0, 0);
    //getOneRdmDeterministic(wave, walk, oneRdm1, 1);
    //getDensityCorrelationsDeterministic(wave, walk, corr);
    getStochasticOneRdmContinuousTime(wave, walk, oneRdm0, 0, schd.stochasticIter);
    getStochasticOneRdmContinuousTime(wave, walk, oneRdm1, 1, schd.stochasticIter);
    getStochasticDensityCorrelationsContinuousTime(wave, walk, corr, schd.stochasticIter);
    if (commrank == 0) {
      cout << "oneRdm0\n" << oneRdm0 << endl << endl;
      cout << "oneRdm1\n" << oneRdm1 << endl << endl;
      cout << "Density correlations\n" << corr << endl << endl;
    }
  }
  
  //else if (schd.wavefunctionType == "LanczosCPSSlater") {
  //  //CIWavefunction<CPSSlater, HFWalker, Operator> wave;
  //  CPSSlater wave; HFWalker walk;
  //  wave.readWave();
  //  wave.initWalker(walk); 
  //  Eigen::VectorXd stddev = Eigen::VectorXd::Zero(4);
  //  Eigen::VectorXd rk = Eigen::VectorXd::Zero(4);
  //  //double rk = 0;
  //  Eigen::VectorXd lanczosCoeffs = Eigen::VectorXd::Zero(4);
  //  double alpha = 0.1;
  //  if (schd.deterministic) getLanczosCoeffsDeterministic(wave, walk, alpha, lanczosCoeffs);
  //  else getLanczosCoeffsContinuousTime(wave, walk, alpha, lanczosCoeffs, stddev, rk, schd.stochasticIter, 1.e-5);
  //  //getLanczosMatrixContinuousTime(wave, walk, lanczosMat, stddev, rk, schd.stochasticIter, 1.e-5);
  //  double a = lanczosCoeffs[2]/lanczosCoeffs[3];
  //  double b = lanczosCoeffs[1]/lanczosCoeffs[3];
  //  double c = lanczosCoeffs[0]/lanczosCoeffs[3];
  //  double delta = pow(a, 2) + 4 * pow(b, 3) - 6 * a * b * c - 3 * pow(b * c, 2) + 4 * a * pow(c, 3);
  //  double numP = a - b * c + pow(delta, 0.5);
  //  double numM = a - b * c - pow(delta, 0.5);
  //  double denom = 2 * pow(b, 2) - 2 * a * c;
  //  double alphaP = numP/denom;
  //  double alphaM = numM/denom;
  //  double eP = (a * pow(alphaP, 2) + 2 * b * alphaP + c) / (b * pow(alphaP, 2) + 2 * c * alphaP + 1);
  //  double eM = (a * pow(alphaM, 2) + 2 * b * alphaM + c) / (b * pow(alphaM, 2) + 2 * c * alphaM + 1);
  //  if (commrank == 0) {
  //    cout << "lanczosCoeffs\n";
  //    cout << lanczosCoeffs << endl;
  //    cout << "stddev\n";
  //    cout << stddev << endl;
  //    cout << "rk\n";
  //    cout << rk << endl;
  //    cout << "alpha(+/-)   " << alphaP << "   " << alphaM << endl;
  //    cout << "energy(+/-)   " << eP << "   " << eM << endl;
  //    //cout << "rk\n" << rk << endl << endl;
  //    //cout << "stddev\n" << stddev << endl << endl;
  //  }

  //  //vector<double> alpha{0., 0.1, 0.2, -0.1, -0.2}; 
  //  //vector<double> Ealpha{0., 0., 0., 0., 0.}; 
  //  //double stddev, rk;
  //  //for (int i = 0; i < alpha.size(); i++) {
  //  //  vars[0] = alpha[i];
  //  //  wave.updateVariables(vars);
  //  //  wave.initWalker(walk);
  //  //  getStochasticEnergyContinuousTime(wave, walk, Ealpha[i], stddev, rk, schd.stochasticIter, 1.e-5);
  //  //  if (commrank == 0) cout << alpha[i] << "   " << Ealpha[i] << "   " << stddev << endl;
  //  //}

  //  //getGradientWrapper<CIWavefunction<CPSSlater, HFWalker, Operator>, HFWalker> wrapper(wave, walk, schd.stochasticIter);
  //  //getGradientWrapper<Lanczos<CPSSlater, HFWalker>, HFWalker> wrapper(wave, walk, schd.stochasticIter);
  //  //  functor1 getStochasticGradient = boost::bind(&getGradientWrapper<Lanczos<CPSSlater, HFWalker>, HFWalker>::getGradient, &wrapper, _1, _2, _3, _4, _5, schd.deterministic);

  //  //if (schd.method == amsgrad) {
  //  //  AMSGrad optimizer(schd.stepsize, schd.decay1, schd.decay2, schd.maxIter);
  //  //  //functor1 getStochasticGradient = boost::bind(&getGradientWrapper<CIWavefunction<CPSSlater, HFWalker, Operator>, HFWalker>::getGradient, &wrapper, _1, _2, _3, _4, _5, schd.deterministic);
  //  //  optimizer.optimize(vars, getStochasticGradient, schd.restart);
  //  //  //if (commrank == 0) wave.printVariables();
  //  //}
  //  //else if (schd.method == sgd) {
  //  //  SGD optimizer(schd.stepsize, schd.maxIter);
  //  //  optimizer.optimize(vars, getStochasticGradient, schd.restart);
  //  //}
  //}

  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shmcas.c_str());
  return 0;
}
