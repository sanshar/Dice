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
#include "SHCIsampledeterminants.h"
using namespace Eigen;

void SHCIsampledeterminants::sample_round(MatrixXx& ci, double eps,
                                          std::vector<int>& Sample1,
                                          std::vector<CItype>& newWts) {
  for (int i = 0; i < ci.rows(); i++) {
    if (abs(ci(i, 0)) > eps) {
      Sample1.push_back(i);
      newWts.push_back(ci(i, 0));
    } else if (((double)rand() / (RAND_MAX)) * eps < abs(ci(i, 0))) {
      Sample1.push_back(i);
      newWts.push_back(eps * ci(i, 0) / abs(ci(i, 0)));
    }
  }
}

void SHCIsampledeterminants::setUpAliasMethod(MatrixXx& ci, double& cumulative,
                                              std::vector<int>& alias,
                                              std::vector<double>& prob) {
  alias.resize(ci.rows());
  prob.resize(ci.rows());

  std::vector<double> larger, smaller;
  for (int i = 0; i < ci.rows(); i++) {
    prob[i] = abs(ci(i, 0)) * ci.rows() / cumulative;
    if (prob[i] < 1.0)
      smaller.push_back(i);
    else
      larger.push_back(i);
  }

  while (larger.size() > 0 && smaller.size() > 0) {
    int l = larger[larger.size() - 1];
    larger.pop_back();
    int s = smaller[smaller.size() - 1];
    smaller.pop_back();

    alias[s] = l;
    prob[l] = prob[l] - (1.0 - prob[s]);
    if (prob[l] < 1.0)
      smaller.push_back(l);
    else
      larger.push_back(l);
  }
}

void SHCIsampledeterminants::setUpAliasMethod(CItype* ci, int DetsSize,
                                              double& cumulative,
                                              std::vector<int>& alias,
                                              std::vector<double>& prob) {
  alias.resize(DetsSize);
  prob.resize(DetsSize);

  std::vector<double> larger, smaller;
  for (int i = 0; i < DetsSize; i++) {
    prob[i] = abs(ci[i]) * DetsSize / cumulative;
    if (prob[i] < 1.0)
      smaller.push_back(i);
    else
      larger.push_back(i);
  }

  while (larger.size() > 0 && smaller.size() > 0) {
    int l = larger[larger.size() - 1];
    larger.pop_back();
    int s = smaller[smaller.size() - 1];
    smaller.pop_back();

    alias[s] = l;
    prob[l] = prob[l] - (1.0 - prob[s]);
    if (prob[l] < 1.0)
      smaller.push_back(l);
    else
      larger.push_back(l);
  }
}

int SHCIsampledeterminants::sample_N2_alias(MatrixXx& ci, double& cumulative,
                                            std::vector<int>& Sample1,
                                            std::vector<CItype>& newWts,
                                            std::vector<int>& alias,
                                            std::vector<double>& prob) {
  int niter = Sample1.size();  // Sample1.resize(0); newWts.resize(0);

  int sampleIndex = 0;
  for (int index = 0; index < niter; index++) {
    int detIndex = floor(1. * ((double)rand() / (RAND_MAX)) * ci.rows());

    double rand_no = ((double)rand() / (RAND_MAX));
    if (rand_no >= prob[detIndex]) detIndex = alias[detIndex];

    std::vector<int>::iterator it =
        find(Sample1.begin(), Sample1.end(), detIndex);
    if (it == Sample1.end()) {
      Sample1[sampleIndex] = detIndex;
      newWts[sampleIndex] = cumulative * ci(detIndex, 0) / abs(ci(detIndex, 0));
      // newWts[sampleIndex] = ci(detIndex,0) < 0. ? -cumulative : cumulative;
      sampleIndex++;
    } else {
      newWts[distance(Sample1.begin(), it)] +=
          cumulative * ci(detIndex, 0) / abs(ci(detIndex, 0));
      // newWts[distance(Sample1.begin(), it) ] += ci(detIndex,0) < 0. ?
      // -cumulative : cumulative;
    }
  }

  for (int i = 0; i < niter; i++) newWts[i] /= niter;
  return sampleIndex;
}

int SHCIsampledeterminants::sample_N2_withoutalias(
    MatrixXx& ci, double& cumulative, std::vector<int>& Sample1,
    std::vector<CItype>& newWts) {
  double prob = 1.0;
  int niter = Sample1.size();
  int totalSample = 0;
  for (int index = 0; index < niter;) {
    double rand_no = ((double)rand() / (RAND_MAX)) * cumulative;
    for (int i = 0; i < ci.rows(); i++) {
      if (rand_no < abs(ci(i, 0))) {
        std::vector<int>::iterator it = find(Sample1.begin(), Sample1.end(), i);
        if (it == Sample1.end()) {
          Sample1[index] = i;
          newWts[index] = cumulative * ci(i, 0) / abs(ci(i, 0));
          // newWts[index] = ci(i,0) < 0. ? -cumulative : cumulative;
          index++;
          totalSample++;
        } else {
          newWts[distance(Sample1.begin(), it)] +=
              cumulative * ci(i, 0) / abs(ci(i, 0));
          // newWts[ distance(Sample1.begin(), it) ] += ci(i,0) < 0. ?
          // -cumulative : cumulative;
          totalSample++;
        }
        break;
      }
      rand_no -= abs(ci(i, 0));
    }
  }

  for (int i = 0; i < niter; i++) newWts[i] /= totalSample;
  return totalSample;
}

int SHCIsampledeterminants::sample_N2_alias(CItype* ci, int DetsSize,
                                            double& cumulative,
                                            std::vector<int>& Sample1,
                                            std::vector<CItype>& newWts,
                                            std::vector<int>& alias,
                                            std::vector<double>& prob) {
  int niter = Sample1.size();  // Sample1.resize(0); newWts.resize(0);

  int sampleIndex = 0;
  for (int index = 0; index < niter; index++) {
    int detIndex = floor(1. * ((double)rand() / (RAND_MAX)) * DetsSize);

    double rand_no = ((double)rand() / (RAND_MAX));
    if (rand_no >= prob[detIndex]) detIndex = alias[detIndex];

    std::vector<int>::iterator it =
        find(Sample1.begin(), Sample1.end(), detIndex);
    if (it == Sample1.end()) {
      Sample1[sampleIndex] = detIndex;
      newWts[sampleIndex] = cumulative * ci[detIndex] / abs(ci[detIndex]);
      // newWts[sampleIndex] = ci(detIndex,0) < 0. ? -cumulative : cumulative;
      sampleIndex++;
    } else {
      newWts[distance(Sample1.begin(), it)] +=
          cumulative * ci[detIndex] / abs(ci[detIndex]);
      // newWts[distance(Sample1.begin(), it) ] += ci(detIndex,0) < 0. ?
      // -cumulative : cumulative;
    }
  }

  for (int i = 0; i < niter; i++) newWts[i] /= niter;
  return sampleIndex;
}
