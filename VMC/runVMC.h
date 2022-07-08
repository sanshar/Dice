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

#include <sys/stat.h>
#include "input.h"
#include "evaluateE.h"
#include "amsgrad.h"
#include "sgd.h"
#include "ftrl.h"
#include "sr.h"
#include <functional>

namespace ph = std::placeholders;
using functor1 = std::function<void (VectorXd&, VectorXd&, double&, double&, double&)>;
using functor2 = std::function<void (VectorXd&, VectorXd&, VectorXd&, DirectMetric&, double&, double&, double&)>;

template<typename Wave, typename Walker>
void runVMC(Wave& wave, Walker& walk) {

  if (schd.restart || schd.fullRestart) wave.readWave();
  VectorXd vars; wave.getVariables(vars);
  getGradientWrapper<Wave, Walker> wrapper(wave, walk, schd.stochasticIter, schd.ctmc);
  functor1 getStochasticGradient = std::bind(&getGradientWrapper<Wave, Walker>::getGradient, &wrapper, ph::_1, ph::_2, ph::_3, ph::_4, ph::_5, schd.deterministic);
  functor2 getStochasticGradientMetric = std::bind(&getGradientWrapper<Wave, Walker>::getMetric, &wrapper, ph::_1, ph::_2, ph::_3, ph::_4, ph::_5, ph::_6, ph::_7, schd.deterministic);

  if (schd.method == amsgrad || schd.method == amsgrad_sgd) {
    AMSGrad optimizer(schd.stepsize, schd.decay1, schd.decay2, schd.maxIter, schd.avgIter);
    optimizer.optimize(vars, getStochasticGradient, schd.restart);
  }
  else if (schd.method == sgd) {
    SGD optimizer(schd.stepsize, schd.momentum, schd.maxIter);
    optimizer.optimize(vars, getStochasticGradient, schd.restart);
  }
  else if (schd.method == ftrl) {
    SGD optimizer(schd.alpha, schd.beta, schd.maxIter);
    optimizer.optimize(vars, getStochasticGradient, schd.restart);
  }
  else if (schd.method == sr) {
/*
    mkdir("./Metric", 0777); 
    mkdir("./T", 0777); 
*/
    SR optimizer(schd.stepsize, schd.maxIter);
    optimizer.optimize(vars, getStochasticGradientMetric, schd.restart);
  }
  else if (schd.method == linearmethod) {
    
  }
  if (schd.printVars && commrank==0) wave.printVariables();
  
}




