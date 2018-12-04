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

#pragma once

#include "input.h"
#include "evaluateE.h"
#include "amsgrad.h"
#include "sgd.h"
#include "sr.h"

using functor1 = boost::function<void (VectorXd&, VectorXd&, double&, double&, double&)>;
using functor2 = boost::function<void (VectorXd&, VectorXd&, MatrixXd&, double&, double&, double&)>;

template<typename Wave, typename Walker>
void runVMC(Wave& wave, Walker& walk) {

  if (schd.restart) wave.readWave();
  VectorXd vars; wave.getVariables(vars);
  getGradientWrapper<Wave, Walker> wrapper(wave, walk, schd.stochasticIter, schd.ctmc);
  functor1 getStochasticGradient = boost::bind(&getGradientWrapper<Wave, Walker>::getGradient, &wrapper, _1, _2, _3, _4, _5, schd.deterministic);
  functor2 getStochasticGradientMetric = boost::bind(&getGradientWrapper<Wave, Walker>::getMetric, &wrapper, _1, _2, _3, _4, _5, _6, schd.deterministic);

  if (schd.method == amsgrad || schd.method == amsgrad_sgd) {
    AMSGrad optimizer(schd.stepsize, schd.decay1, schd.decay2, schd.maxIter, schd.avgIter);
    optimizer.optimize(vars, getStochasticGradient, schd.restart);
  }
  else if (schd.method == sgd) {
    SGD optimizer(schd.stepsize, schd.maxIter);
    optimizer.optimize(vars, getStochasticGradient, schd.restart);
  }
  else if (schd.method == sr) {
    SR optimizer(schd.stepsize, schd.maxIter);
    optimizer.optimize(vars, getStochasticGradientMetric, schd.restart);
  }
  else if (schd.method == linearmethod) {
    
  }
  if (schd.printVars && commrank==0) wave.printVariables();
  
}




