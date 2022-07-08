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
#ifndef EvalPT_HEADER_H
#define EvalPT_HEADER_H
#include <Eigen/Dense>
#include <vector>
#ifndef SERIAL
#include "mpi.h"
#endif

class CPSSlater;
class oneInt;
class twoInt;
class twoIntHeatBathSHM;
class MoDeterminant;


//evaluate PT correction 
double evaluateScaledEDeterministic(CPSSlater& w, double& lambda, double& unscaledE,
				    int& nalpha, int& nbeta, int& norbs,
				    oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, 
				    double& coreE);


double evaluatePTDeterministic(CPSSlater& w, double&  E0, int& nalpha, int& nbeta, int& norbs,
			       oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, 
			       double& coreE);
double evaluatePTDeterministicB(CPSSlater& w, double& E0, int& nalpha, int& nbeta, int& norbs,
				oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, double& coreE);
double evaluatePTDeterministicC(CPSSlater& w, double&  E0, int& nalpha, int& nbeta, int& norbs,
			       oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, double& coreE);



double evaluateScaledEStochastic(CPSSlater& w, double& lambda, double& unscaledE,
				 int& nalpha, int& nbeta, int& norbs,
				 oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, 
				 double& coreE, double& stddev, double& rk,
				 int niter=10000, double targetError = 1e-3);



double evaluatePTStochasticMethodA(CPSSlater& w, double&  E0, int& nalpha, int& nbeta, int& norbs,
			    oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, 
			    double& coreE, double& stddev, int niter, double& A, double& B, double& C);

double evaluatePTStochasticMethodB(CPSSlater& w, double&  E0, int& nalpha, int& nbeta, int& norbs,
				   oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, 
				   double& coreE, double& stddev, double& rk, int niter, double& A, double& B, 
				   double& C);

double evaluatePTStochasticMethodC(CPSSlater& w, double&  E0, int& nalpha, int& nbeta, int& norbs,
				   oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, 
				   double& coreE, double& stddevA, double& stddevB, double& stddevC,
				   int niter, double& A, double& B, 
				   double& C);

double evaluatePTStochastic3rdOrder(CPSSlater& w, double&  E0, int& nalpha, int& nbeta, int& norbs,
				    oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, 
				    double& coreE, double& stddev, int niter, double& A2, double& B, 
				    double& C, double& A3);



#endif
