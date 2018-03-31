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
class Wfn;
class CPSSlater;
class oneInt;
class twoInt;

double evaluateEDeterministic(Wfn& w, int& nalpha, int& nbeta, int& norbs,
			      oneInt& I1, twoInt& I2, double& coreE);

double evaluateEStochastic(CPSSlater& w, int& nalpha, int& nbeta, int& norbs,
			   oneInt& I1, twoInt& I2, double& coreE, double& stddev,
			   int niter=10000, double targetError = 1.e-3);

void getGradient(Wfn& w, double& E0, int& alpha, int& nbeta, int& norbs,
		 oneInt& I1, twoInt& I2, double& coreE,
		 Eigen::VectorXd& grad);

void getStochasticGradient(CPSSlater& w, double& E0, double& stddev, 
			   int& nalpha, int& nbeta, int& norbs,
			   oneInt& I1, twoInt& I2, double& coreE,
			   Eigen::VectorXd& grad, int niter, double targetError);

void comb(int N, int K, std::vector<std::vector<int> >& combinations);

void getGradientUsingDavidson(Wfn& w, double& E0, int& nalpha, int& nbeta, int& norbs,
			      oneInt& I1, twoInt& I2, double& coreE,
			      Eigen::VectorXd& grad);

void davidsonDirect(int nalpha, int nbeta, int norbs, oneInt& I1,
		    twoInt& I2, double& coreE, Eigen::VectorXd& vars);

#endif
