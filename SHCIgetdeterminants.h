/*
  Developed by Sandeep Sharma with contributions from James E. T. Smith and Adam
  A. Holmes, 2017 Copyright (c) 2017, Sandeep Sharma

  This file is part of DICE.

  This program is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or (at your option) any later
  version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with
  this program. If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef SHCI_GETDETERMINANTS_H
#define SHCI_GETDETERMINANTS_H
#include <Eigen/Dense>
#include <list>
#include <map>
#include <set>
#include <tuple>
#include <vector>

using namespace std;
using namespace Eigen;
class Determinant;
class HalfDet;
class oneInt;
class twoInt;
class twoIntHeatBath;
class twoIntHeatBathSHM;
class schedule;
class StichDEH;

namespace SHCIgetdeterminants {
void getDeterminants(Determinant &d, int det_ind, double epsilon, CItype ci1,
                     CItype ci2, oneInt &int1, twoInt &int2,
                     twoIntHeatBathSHM &I2hb, vector<int> &irreps, double coreE,
                     double E0, StitchDEH &uniqueDEH, schedule &schd, int Nmc,
                     int nelec, bool keepRefDets = false);

void getDeterminantsVariational(Determinant &d, double epsilon, CItype ci1,
                                CItype ci2, oneInt &int1, twoInt &int2,
                                twoIntHeatBathSHM &I2hb, vector<int> &irreps,
                                double coreE, double E0,
                                std::vector<Determinant> &dets, schedule &schd,
                                int Nmc, int nelec);

void getDeterminantsVariationalApprox(
    Determinant &d, double epsilon, CItype ci1, CItype ci2, oneInt &int1,
    twoInt &int2, twoIntHeatBathSHM &I2hb, vector<int> &irreps, double coreE,
    double E0, std::vector<Determinant> &dets, schedule &schd, int Nmc,
    int nelec, Determinant *SortedDets, int SortedDetsSize);

void getDeterminantsStochastic2Epsilon(
    Determinant &d, double epsilon, double epsilonLarge, CItype ci1, CItype ci2,
    oneInt &int1, twoInt &int2, twoIntHeatBathSHM &I2hb, vector<int> &irreps,
    double coreE, double E0, std::vector<Determinant> &dets,
    std::vector<CItype> &numerator1A, vector<CItype> &numerator2A,
    vector<char> &present, std::vector<double> &energy, schedule &schd, int Nmc,
    int nelec);

void getDeterminantsDeterministicPTWithSOC(
    Determinant det, int det_ind, double epsilon1, CItype ci1, double epsilon2,
    CItype ci2, oneInt &int1, twoInt &int2, twoIntHeatBathSHM &I2hb,
    vector<int> &irreps, double coreE, std::vector<Determinant> &dets,
    std::vector<CItype> &numerator1, std::vector<CItype> &numerator2,
    std::vector<double> &energy, schedule &schd, int nelec);

}; // namespace SHCIgetdeterminants
#endif
