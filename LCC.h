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
#ifndef LCC_HEADER_H
#define LCC_HEADER_H
#include <vector>
#include <Eigen/Dense>
#include <set>
#include <list>
#include <tuple>
#include <map>
#include <boost/serialization/serialization.hpp>

using namespace std;
using namespace Eigen;

namespace LCC {
  double doLCC(
          Determinant *Dets, CItype *ci,
          int DetsSize, double& E0, 
          oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2HB, 
          vector<int>& irrep, schedule& schd, double coreE, 
          int nelec, int root);

  void getDeterminantsLCC(
          Determinant& d, double epsilon, CItype ci1, CItype ci2,
          oneInt& int1, twoInt& int2, twoIntHeatBathSHM& I2hb,
          vector<int>& irreps, double coreE, double E0, 
          std::vector<Determinant>& dets, std::vector<CItype>& numerator, std::vector<double>& energy, 
          schedule& schd, int Nmc, int nelec) ;
};

#endif
