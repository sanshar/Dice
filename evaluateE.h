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
#ifndef EvalE_HEADER_H
#define EvalE_HEADER_H
#include <Eigen/Dense>
#include <vector>
#ifndef SERIAL
#include "mpi.h"
#endif
class Wfn;
class CPSSlater;
class oneInt;
class twoInt;
class twoIntHeatBathSHM;
class MoDeterminant;

//generate all the alpha or beta strings
void comb(int N, int K, std::vector<std::vector<int> >& combinations);

//calculate reblocking analysis to find correlation length
double calcTcorr(std::vector<double>& v);


//evaluate energy and gradient using stochastic or deterministic algorithm
double evaluateEDeterministic(Wfn& w, int& nalpha, int& nbeta, int& norbs,
			      oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb,
			      double& coreE);

double evaluateEStochastic(CPSSlater& w, int& nalpha, int& nbeta, int& norbs,
			   oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb,
			   double& coreE, double& stddev,
			   int niter=10000, double targetError = 1.e-3);

void getGradient(Wfn& w, double& E0, int& alpha, int& nbeta, int& norbs,
		 oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, double& coreE,
		 Eigen::VectorXd& grad);

void getStochasticGradient(CPSSlater& w, double& E0, double& stddev, 
			   int& nalpha, int& nbeta, int& norbs,
			   oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, double& coreE,
			   Eigen::VectorXd& grad, double& rk,
			   int niter, double targetError);

void getStochasticGradientContinuousTime(CPSSlater& w, double& E0, double& stddev,
					 int& nalpha, int& nbeta, int& norbs,
					 oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, double& coreE, 
					 Eigen::VectorXd& grad, double& rk, 
					 int niter, double targetError);


#endif
