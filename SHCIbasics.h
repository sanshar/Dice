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
#ifndef SHCI_HEADER_H
#define SHCI_HEADER_H
#include <Eigen/Dense>
#include <boost/serialization/serialization.hpp>
#include <list>
#include <map>
#include <set>
#include <tuple>
#include <vector>
#include "global.h"

using namespace std;
using namespace Eigen;
namespace SHCImakeHamiltonian {
struct HamHelpers2;
struct SparseHam;
};  // namespace SHCImakeHamiltonian

class Determinant;
class HalfDet;
class oneInt;
class twoInt;
class twoIntHeatBath;
class twoIntHeatBathSHM;
class schedule;

namespace SHCIbasics {

void writeHelperIntermediate(std::map<HalfDet, int>& BetaN,
                             std::map<HalfDet, int>& AlphaN,
                             std::map<HalfDet, vector<int> >& BetaNm1,
                             std::map<HalfDet, vector<int> >& AlphaNm1,
                             schedule& schd, int iter);

void readHelperIntermediate(std::map<HalfDet, int>& BetaN,
                            std::map<HalfDet, int>& AlphaN,
                            std::map<HalfDet, vector<int> >& BetaNm1,
                            std::map<HalfDet, vector<int> >& AlphaNm1,
                            schedule& schd, int iter);

void readVariationalResult(int& iter, vector<MatrixXx>& ci,
                           vector<Determinant>& Dets,
                           SHCImakeHamiltonian::SparseHam& sparseHam,
                           vector<double>& E0, bool& converged, schedule& schd,
                           SHCImakeHamiltonian::HamHelpers2& helper2);

void writeVariationalResult(int iter, vector<MatrixXx>& ci,
                            vector<Determinant>& Dets,
                            SHCImakeHamiltonian::SparseHam& sparseHam,
                            vector<double>& E0, bool converged, schedule& schd,
                            SHCImakeHamiltonian::HamHelpers2& helper2);

void writeVariationalResult(int iter, vector<MatrixXx>& ci,
                            vector<Determinant>& Dets,
                            vector<vector<int> >& connections,
                            vector<vector<size_t> >& orbDifference,
                            vector<vector<CItype> >& Helements,
                            vector<double>& E0, bool converged, schedule& schd,
                            std::map<HalfDet, std::vector<int> >& BetaN,
                            std::map<HalfDet, std::vector<int> >& AlphaNm1);
void readVariationalResult(int& iter, vector<MatrixXx>& ci,
                           vector<Determinant>& Dets,
                           vector<vector<int> >& connections,
                           vector<vector<size_t> >& orbDifference,
                           vector<vector<CItype> >& Helements,
                           vector<double>& E0, bool& converged, schedule& schd,
                           std::map<HalfDet, std::vector<int> >& BetaN,
                           std::map<HalfDet, std::vector<int> >& AlphaNm1);
vector<double> DoVariational(vector<MatrixXx>& ci, vector<Determinant>& Dets,
                             schedule& schd, twoInt& I2,
                             twoIntHeatBathSHM& I2HB, vector<int>& irrep,
                             oneInt& I1, double& coreE, int nelec,
                             bool DoRDM = false);

vector<double> DoVariationalDirect(vector<MatrixXx>& ci,
                                   vector<Determinant>& Dets, schedule& schd,
                                   twoInt& I2, twoIntHeatBathSHM& I2HB,
                                   vector<int>& irrep, oneInt& I1,
                                   double& coreE, int nelec,
                                   bool DoRDM = false);

double DoPerturbativeStochastic2SingleListDoubleEpsilon2AllTogether(
    Determinant* Dets, CItype* ci, int DetsSize, double& E0, oneInt& I1,
    twoInt& I2, twoIntHeatBathSHM& I2HB, vector<int>& irrep, schedule& schd,
    double coreE, int nelec, int root);
void DoPerturbativeStochastic2SingleListDoubleEpsilon2OMPTogether(
    vector<Determinant>& Dets, MatrixXx& ci, double& E0, oneInt& I1, twoInt& I2,
    twoIntHeatBathSHM& I2HB, vector<int>& irrep, schedule& schd, double coreE,
    int nelec, int root);
void DoPerturbativeStochastic2SingleListDoubleEpsilon2(
    vector<Determinant>& Dets, MatrixXx& ci, double& E0, oneInt& I1, twoInt& I2,
    twoIntHeatBathSHM& I2HB, vector<int>& irrep, schedule& schd, double coreE,
    int nelec, int root);

void DoPerturbativeStochastic2SingleList(vector<Determinant>& Dets,
                                         MatrixXx& ci, double& E0, oneInt& I1,
                                         twoInt& I2, twoIntHeatBathSHM& I2HB,
                                         vector<int>& irrep, schedule& schd,
                                         double coreE, int nelec, int root);

double DoPerturbativeDeterministic(Determinant* Dets, CItype* ci, int DetsSize,
                                   double& E0, oneInt& I1, twoInt& I2,
                                   twoIntHeatBathSHM& I2HB, vector<int>& irrep,
                                   schedule& schd, double coreE, int nelec,
                                   int root, vector<MatrixXx>& vdVector,
                                   double& Psi1Norm,
                                   bool appendPsi1ToPsi0 = false);

void DoPerturbativeDeterministicOffdiagonal(
    vector<Determinant>& Dets, MatrixXx& ci1, double& E01, MatrixXx& ci2,
    double& E02, int DetsSize, oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2HB,
    vector<int>& irrep, schedule& schd, double coreE, int nelec, int root,
    CItype& EPT1, CItype& EPT2, CItype& EPT12, std::vector<MatrixXx>& spinRDM);

}  // namespace SHCIbasics

#endif
