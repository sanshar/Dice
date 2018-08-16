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
#include "Determinants.h"
#include "CPSSlater.h"
#include "CPSSlaterWalker.h"
#include "input.h"
#include "integral.h"
#include "SHCIshm.h"
#include "math.h"
#include "Profile.h"


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

  //Set up the gradient hessian overlap matrices
  double E0 = 0.0, stddev, rt = 0;
  int norbs = Determinant::norbs;
  Eigen::VectorXd grad;
  Eigen::MatrixXd Hessian, Smatrix;

  //calculate the hessian/gradient
  if (schd.wavefunctionType == "CPSSlater") {
    CPSSlater wave; CPSSlaterWalker walk;
    wave.read();
    int nvars = schd.uhf ? wave.getNumVariables() + 2 * norbs * norbs : wave.getNumVariables() + norbs * norbs;
    grad = Eigen::VectorXd::Zero(nvars);
    if (schd.doHessian)
    {
      Hessian = Eigen::MatrixXd::Zero(nvars + 1, nvars + 1);
      Smatrix = Eigen::MatrixXd::Zero(nvars + 1, nvars + 1);
    }

    wave.initWalker(walk);
    getStochasticGradientContinuousTime(wave, walk, E0, stddev, grad, rt, schd.stochasticIter, 0.5e-3);
  }

  //write the results
  if (commrank == 0)
  {
    std::cout << format("%14.8f (%8.2e) %14.8f %8.1f %10i %8.2f\n") % E0 % stddev % (grad.norm()) % (rt) % (schd.stochasticIter) % ((getTime() - startofCalc));

    size_t size;
    {
      ifstream file("params.bin", ios::in | ios::binary | ios::ate);
      size = file.tellg();
    }

    ofstream file("grad.bin", ios::out | ios::binary);
    file.write((char *)(&grad[0]), size);
    file.close();

    ofstream filee("E0.bin", ios::out | ios::binary);
    filee.write((char *)(&E0), sizeof(double));
    filee.close();

    if (schd.doHessian)
    {
      ofstream hfile("hessian.bin", ios::out | ios::binary);
      hfile.write((char *)(&Hessian(0, 0)), Hessian.rows() * Hessian.cols() * sizeof(double));
      hfile.close();

      ofstream sfile("smatrix.bin", ios::out | ios::binary);
      sfile.write((char *)(&Smatrix(0, 0)), Hessian.rows() * Hessian.cols() * sizeof(double));
      sfile.close();
    }
  }
/*

  //Setup Slater Determinants


  MatrixXd alpha(norbs, nalpha), beta(norbs, nbeta);
  alpha = HforbsA.block(0, 0, norbs, nalpha);
  beta = HforbsB.block(0, 0, norbs, nbeta);
  //MoDeterminant det(alpha, beta);

  Eigen::VectorXd grad = Eigen::VectorXd::Zero(vars.size());
  Eigen::MatrixXd Hessian, Smatrix;
  if (schd.doHessian)
  {
    Hessian.resize(vars.size() + 1, vars.size() + 1);
    Smatrix.resize(vars.size() + 1, vars.size() + 1);
    Hessian.setZero();
    Smatrix.setZero();
  }

  //if (commrank == 0)
  //std::cout << format("Finished reading from disk: %8.2f\n")
  //%( (getTime()-startofCalc));

  double E0 = 0.0, stddev, rt = 0;
  if (schd.deterministic)
  {
    if (!schd.doHessian)
    {
      getGradientDeterministic(wave, E0, nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE, grad);
      stddev = 0.0;
    }
    else
    {
      getGradientHessianDeterministic(wave, E0, nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE, grad, Hessian, Smatrix);
      stddev = 0.0;
    }
  }
  else
  {
    //getStochasticGradient(wave, E0, stddev, nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE, grad, rt, schd.stochasticIter, 0.5e-3);
    if (!schd.doHessian) {
      Walker walk;
      initWalker<CPSSlater, Walker>(wave, walk, nalpha, nbeta, norbs);
      getStochasticGradientContinuousTime<CPSSlater, Walker>(wave, walk, E0, stddev, nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE, grad, rt, schd.stochasticIter, 0.5e-3);
    }
    //getStochasticGradient(wave, E0, stddev, nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE, grad, rt, schd.stochasticIter, 0.5e-3);
    else
      getStochasticGradientHessianContinuousTime(wave, E0, stddev, nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE, grad, Hessian, Smatrix, rt, schd.stochasticIter, 0.5e-3);
  }

  //for (int i=wave.getNumJastrowVariables(); i<wave.getNumVariables(); i++) {
  //grad[i] *= getParityForDiceToAlphaBeta(wave.determinants[i-wave.getNumJastrowVariables()]);
  //}

  if (commrank == 0)
  {
    std::cout << format("%14.8f (%8.2e) %14.8f %8.1f %10i %8.2f\n") % E0 % stddev % (grad.norm()) % (rt) % (schd.stochasticIter) % ((getTime() - startofCalc));

    //cout << prof.SinglesTime<<"  "<<prof.SinglesCount<<endl;
    //cout << prof.DoubleTime<<"  "<<prof.DoubleCount<<endl;

    ofstream file("grad.bin", ios::out | ios::binary);
    file.write((char *)(&grad[0]), size);
    file.close();

    ofstream filee("E0.bin", ios::out | ios::binary);
    filee.write((char *)(&E0), sizeof(double));
    filee.close();

    if (schd.doHessian)
    {
      ofstream hfile("hessian.bin", ios::out | ios::binary);
      hfile.write((char *)(&Hessian(0, 0)), Hessian.rows() * Hessian.cols() * sizeof(double));
      hfile.close();

      ofstream sfile("smatrix.bin", ios::out | ios::binary);
      sfile.write((char *)(&Smatrix(0, 0)), Hessian.rows() * Hessian.cols() * sizeof(double));
      sfile.close();
    }
  }
*/
  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());
  return 0;
}
