/*
  Developed by Sandeep Sharma with contributions from James E. T. Smith and Adam A. Holmes, 2017
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
#ifndef SHCI_SAMPLEDETERMINANTS_H
#define SHCI_SAMPLEDETERMINANTS_H
#include <vector>
#include <Eigen/Dense>
#include <set>
#include <list>
#include <tuple>
#include <map>

using namespace std;
using namespace Eigen;
#include "Dice/Utils/global.h"

namespace SHCIsampledeterminants {


  void sample_round(MatrixXx& ci, double eps, std::vector<int>& Sample1, std::vector<CItype>& newWts);
  void setUpAliasMethod(MatrixXx& ci, double& cumulative, std::vector<int>& alias, std::vector<double>& prob) ;
  int sample_N2_alias(MatrixXx& ci, double& cumulative, std::vector<int>& Sample1, std::vector<CItype>& newWts, std::vector<int>& alias, std::vector<double>& prob) ;
  int sample_N2_withoutalias(MatrixXx& ci, double& cumulative, std::vector<int>& Sample1, std::vector<CItype>& newWts);
  int sample_N(MatrixXx& ci, double& cumulative, std::vector<int>& Sample1, std::vector<CItype>& newWts)
;
  void setUpAliasMethod(CItype *ci, int DetsSize, double& cumulative, std::vector<int>& alias, std::vector<double>& prob) ;
  int sample_N2_alias(CItype *ci, int DetsSize, double& cumulative, std::vector<int>& Sample1, std::vector<CItype>& newWts, std::vector<int>& alias, std::vector<double>& prob) ;

};

#endif
