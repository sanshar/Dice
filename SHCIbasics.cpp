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
#include "SHCIbasics.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include <fstream>
#include <map>
#include <tuple>
#include <vector>
#include "Davidson.h"
#include "Determinants.h"
#include "Hmult.h"
#include "SHCISortMpiUtils.h"
#include "SHCIgetdeterminants.h"
#include "SHCImakeHamiltonian.h"
#include "SHCIrdm.h"
#include "SHCIsampledeterminants.h"
#include "SHCIshm.h"
#include "SHCItime.h"
#include "boost/format.hpp"
#include "input.h"
#include "integral.h"
#include "math.h"

#include "communicate.h"

using namespace std;
using namespace Eigen;
using namespace boost;
using namespace SHCISortMpiUtils;

double SHCIbasics::DoPerturbativeStochastic2SingleListDoubleEpsilon2AllTogether(
    Determinant *Dets, CItype *ci, int DetsSize, double &E0, oneInt &I1,
    twoInt &I2, twoIntHeatBathSHM &I2HB, vector<int> &irrep, schedule &schd,
    double coreE, int nelec, int root) {
  if (schd.nPTiter == 0)
    return 0;
  pout << format("Performing (semi)stochastic PT for state: %3i") % (root)
       << endl;

  double epsilon2 = schd.epsilon2;
  schd.epsilon2 = schd.epsilon2Large;
  vector<MatrixXx> vdVector;
  double Psi1Norm;
  double EptLarge = 0.0;
  if (schd.epsilon2 < 999)
    pout << endl
         << "1/ Deterministic calculation with epsilon2=" << schd.epsilon2
         << endl;
    EptLarge = DoPerturbativeDeterministic(Dets, ci, DetsSize, E0, I1, I2, I2HB,
                                           irrep, schd, coreE, nelec, root,
                                           vdVector, Psi1Norm);

  schd.epsilon2 = epsilon2;
  pout << endl
       << "2/ Stochastic calculation with epsilon2=" << schd.epsilon2 << endl;

  int norbs = Determinant::norbs;
  Determinant *SortedDets;
  std::vector<Determinant> SortedDetsvec;
  if (commrank == 0) {
    for (int i = 0; i < DetsSize; i++) SortedDetsvec.push_back(Dets[i]);
    std::sort(SortedDetsvec.begin(), SortedDetsvec.end());
  }
#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  SHMVecFromVecs(SortedDetsvec, SortedDets, shciSortedDets, SortedDetsSegment,
                 regionSortedDets);
  SortedDetsvec.clear();

  int niter = schd.nPTiter;
  // double eps = 0.001;
  int Nsample = schd.SampleN;
  double AvgenergyEN = 0.0, AvgenergyEN2 = 0.0, stddev = 0.0;
  double AverageDen = 0.0;
  int currentIter = 0;
  int sampleSize = 0;
  int num_thrds = 1;

  double cumulative = 0.0;
  for (int i = 0; i < DetsSize; i++) cumulative += abs(ci[i]);

  std::vector<int> alias;
  std::vector<double> prob;
  if (commrank == 0)
    SHCIsampledeterminants::setUpAliasMethod(ci, DetsSize, cumulative, alias,
                                             prob);

  StitchDEH uniqueDEH;
  double totalPT = 0.0;
  double totalPTLargeEps = 0;
  size_t ntries = 0;
  int AllDistinctSample = 0;
  size_t Nmc = commsize * num_thrds * Nsample;
  std::vector<int> allSample(Nmc, -1);
  std::vector<CItype> allwts(Nmc, 0.);
  pout << format("%6s  %18s  %5s %18s %10s  %10s") % ("Iter") % ("EPTcurrent") %
              ("State") % ("EPTavg") % ("Error") % ("Time(s)")
       << endl;

  int size = commsize, rank = commrank;

  for (int iter = 0; iter < niter; iter++) {
    std::vector<CItype> wts1(Nsample, 0.0);
    std::vector<int> Sample1(Nsample, -1);
    int ithrd = 0;
    vector<size_t> all_to_all(size * size, 0);

    if (commrank == 0) {
      std::fill(allSample.begin(), allSample.end(), -1);
      AllDistinctSample = SHCIsampledeterminants::sample_N2_alias(
          ci, DetsSize, cumulative, allSample, allwts, alias, prob);
    }

#ifndef SERIAL
    MPI_Bcast(&allSample[0], allSample.size(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&AllDistinctSample, 1, MPI_INT, 0, MPI_COMM_WORLD);
#ifndef Complex
    MPI_Bcast(&allwts[0], allwts.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
    MPI_Bcast(&allwts[0], 2 * allwts.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
#endif

    int distinctSample = 0;
    for (int i = 0; i < AllDistinctSample; i++) {
      if (i % size != rank) continue;
      wts1[distinctSample] = allwts[i];
      Sample1[distinctSample] = allSample[i];
      distinctSample++;
    }
    double norm = 0.0;

    for (int i = 0; i < distinctSample; i++) {
      int I = Sample1[i];
      SHCIgetdeterminants::getDeterminantsStochastic2Epsilon(
          Dets[I], schd.epsilon2 / abs(ci[I]), schd.epsilon2Large / abs(ci[I]),
          wts1[i], ci[I], I1, I2, I2HB, irrep, coreE, E0, *uniqueDEH.Det,
          *uniqueDEH.Num, *uniqueDEH.Num2, *uniqueDEH.present,
          *uniqueDEH.Energy, schd, Nmc, nelec);
    }

    if (commsize > 1) {
      std::shared_ptr<vector<Determinant>> &Det = uniqueDEH.Det;
      std::shared_ptr<vector<CItype>> &Num = uniqueDEH.Num;
      std::shared_ptr<vector<CItype>> &Num2 = uniqueDEH.Num2;
      std::shared_ptr<vector<double>> &Energy = uniqueDEH.Energy;
      std::shared_ptr<vector<char>> &present = uniqueDEH.present;

      std::vector<size_t> hashValues(Det->size());

      std::vector<size_t> all_to_all_cumulative(size, 0);
      for (int i = 0; i < Det->size(); i++) {
        hashValues[i] = Det->at(i).getHash();
        all_to_all[rank * size + hashValues[i] % size]++;
      }
      for (int i = 0; i < size; i++)
        all_to_all_cumulative[i] =
            i == 0 ? all_to_all[rank * size + i]
                   : all_to_all_cumulative[i - 1] + all_to_all[rank * size + i];

      size_t dsize = Det->size() == 0 ? 1 : Det->size();
      vector<Determinant> atoaDets(dsize);
      vector<CItype> atoaNum(dsize);
      vector<CItype> atoaNum2(dsize);
      vector<double> atoaE(dsize);
      vector<char> atoaPresent(dsize);

#ifndef SERIAL
      vector<size_t> all_to_allCopy = all_to_all;
      MPI_Allreduce(&all_to_allCopy[0], &all_to_all[0], 2 * size * size,
                    MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
      vector<size_t> counter(size, 0);
      for (int i = 0; i < Det->size(); i++) {
        int toProc = hashValues[i] % size;
        size_t index =
            toProc == 0 ? counter[0]
                        : counter[toProc] + all_to_all_cumulative[toProc - 1];

        atoaDets[index] = Det->at(i);
        atoaNum[index] = Num->at(i);
        atoaNum2[index] = Num2->at(i);
        atoaE[index] = Energy->at(i);
        atoaPresent[index] = present->at(i);

        counter[toProc]++;
      }

      vector<int> sendcts(size, 0), senddisp(size, 0), recvcts(size, 0),
          recvdisp(size, 0);
      vector<int> sendctsDets(size, 0), senddispDets(size, 0),
          recvctsDets(size, 0), recvdispDets(size, 0);
      vector<int> sendctsPresent(size, 0), senddispPresent(size, 0),
          recvctsPresent(size, 0), recvdispPresent(size, 0);

      size_t recvSize = 0;
      for (int i = 0; i < size; i++) {
        sendcts[i] =
            all_to_all[rank * size + i] * sizeof(CItype) / sizeof(double);
        senddisp[i] = i == 0 ? 0 : senddisp[i - 1] + sendcts[i - 1];
        recvcts[i] =
            all_to_all[i * size + rank] * sizeof(CItype) / sizeof(double);
        recvdisp[i] = i == 0 ? 0 : recvdisp[i - 1] + recvcts[i - 1];

        sendctsDets[i] =
            all_to_all[rank * size + i] * sizeof(Determinant) / sizeof(double);
        senddispDets[i] = i == 0 ? 0 : senddispDets[i - 1] + sendctsDets[i - 1];
        recvctsDets[i] =
            all_to_all[i * size + rank] * sizeof(Determinant) / sizeof(double);
        recvdispDets[i] = i == 0 ? 0 : recvdispDets[i - 1] + recvctsDets[i - 1];

        sendctsPresent[i] = all_to_all[rank * size + i];
        senddispPresent[i] =
            i == 0 ? 0 : senddispPresent[i - 1] + sendctsPresent[i - 1];
        recvctsPresent[i] = all_to_all[i * size + rank];
        recvdispPresent[i] =
            i == 0 ? 0 : recvdispPresent[i - 1] + recvctsPresent[i - 1];

        recvSize += all_to_all[i * size + rank];
      }

      recvSize = recvSize == 0 ? 1 : recvSize;
      Det->resize(recvSize), Num->resize(recvSize), Energy->resize(recvSize);
      Num2->resize(recvSize), present->resize(recvSize);

#ifndef SERIAL
      MPI_Alltoallv(&atoaNum.at(0), &sendcts[0], &senddisp[0], MPI_DOUBLE,
                    &Num->at(0), &recvcts[0], &recvdisp[0], MPI_DOUBLE,
                    MPI_COMM_WORLD);
      MPI_Alltoallv(&atoaNum2.at(0), &sendcts[0], &senddisp[0], MPI_DOUBLE,
                    &Num2->at(0), &recvcts[0], &recvdisp[0], MPI_DOUBLE,
                    MPI_COMM_WORLD);
      MPI_Alltoallv(&atoaE.at(0), &sendctsPresent[0], &senddispPresent[0],
                    MPI_DOUBLE, &Energy->at(0), &recvctsPresent[0],
                    &recvdispPresent[0], MPI_DOUBLE, MPI_COMM_WORLD);
      MPI_Alltoallv(&atoaPresent.at(0), &sendctsPresent[0], &senddispPresent[0],
                    MPI_CHAR, &present->at(0), &recvctsPresent[0],
                    &recvdispPresent[0], MPI_CHAR, MPI_COMM_WORLD);
      MPI_Alltoallv(&atoaDets.at(0).repr[0], &sendctsDets[0], &senddispDets[0],
                    MPI_DOUBLE, &(Det->at(0).repr[0]), &recvctsDets[0],
                    &recvdispDets[0], MPI_DOUBLE, MPI_COMM_WORLD);
#endif
    }
    uniqueDEH.MergeSort();

    double energyEN = 0.0, energyENLargeEps = 0.0;

    vector<Determinant> &Psi1 = *uniqueDEH.Det;
    vector<CItype> &numerator1A = *uniqueDEH.Num;
    vector<CItype> &numerator2A = *uniqueDEH.Num2;
    vector<char> &present = *uniqueDEH.present;
    vector<double> &det_energy = *uniqueDEH.Energy;

    CItype currentNum1A = 0.;
    CItype currentNum2A = 0.;
    CItype currentNum1B = 0.;
    CItype currentNum2B = 0.;
    size_t vec_it = 0, i = 0;

    while (i < Psi1.size() && vec_it < DetsSize) {
      if (Psi1[i] < SortedDets[vec_it]) {
        currentNum1A += numerator1A[i];
        currentNum2A += numerator2A[i];
        if (present[i]) {
          currentNum1B += numerator1A[i];
          currentNum2B += numerator2A[i];
        }

        if (i == Psi1.size() - 1) {
#ifndef Complex
          energyEN +=
              (pow(abs(currentNum1A), 2) * Nmc / (Nmc - 1) - currentNum2A) /
              (det_energy[i] - E0);
          energyENLargeEps +=
              (pow(abs(currentNum1B), 2) * Nmc / (Nmc - 1) - currentNum2B) /
              (det_energy[i] - E0);
#else
          energyEN += (pow(abs(currentNum1A), 2) * Nmc / (Nmc - 1) -
                       currentNum2A.real()) /
                      (det_energy[i] - E0);
          energyENLargeEps += (pow(abs(currentNum1B), 2) * Nmc / (Nmc - 1) -
                               currentNum2B.real()) /
                              (det_energy[i] - E0);
#endif
        } else if (!(Psi1[i] == Psi1[i + 1])) {
#ifndef Complex
          energyEN +=
              (pow(abs(currentNum1A), 2) * Nmc / (Nmc - 1) - currentNum2A) /
              (det_energy[i] - E0);
          energyENLargeEps +=
              (pow(abs(currentNum1B), 2) * Nmc / (Nmc - 1) - currentNum2B) /
              (det_energy[i] - E0);
#else
          energyEN += (pow(abs(currentNum1A), 2) * Nmc / (Nmc - 1) -
                       currentNum2A.real()) /
                      (det_energy[i] - E0);
          energyENLargeEps += (pow(abs(currentNum1B), 2) * Nmc / (Nmc - 1) -
                               currentNum2B.real()) /
                              (det_energy[i] - E0);
#endif
          currentNum1A = 0.;
          currentNum2A = 0.;
          currentNum1B = 0.;
          currentNum2B = 0.;
        }
        i++;
      } else if (SortedDets[vec_it] < Psi1[i] && vec_it != DetsSize) {
        vec_it++;
      } else {
        if (i == Psi1.size() - 1 || Psi1[i] == Psi1[i + 1]) {
          i++;
        } else {
          vec_it++;
          i++;
        }
      }
    }

    while (i < Psi1.size()) {
      currentNum1A += numerator1A[i];
      currentNum2A += numerator2A[i];
      if (present[i]) {
        currentNum1B += numerator1A[i];
        currentNum2B += numerator2A[i];
      }

      if (i == Psi1.size() - 1) {
#ifndef Complex
        energyEN +=
            (pow(abs(currentNum1A), 2) * Nmc / (Nmc - 1) - currentNum2A) /
            (det_energy[i] - E0);
        energyENLargeEps +=
            (pow(abs(currentNum1B), 2) * Nmc / (Nmc - 1) - currentNum2B) /
            (det_energy[i] - E0);
#else
        energyEN += (pow(abs(currentNum1A), 2) * Nmc / (Nmc - 1) -
                     currentNum2A.real()) /
                    (det_energy[i] - E0);
        energyENLargeEps += (pow(abs(currentNum1B), 2) * Nmc / (Nmc - 1) -
                             currentNum2B.real()) /
                            (det_energy[i] - E0);
#endif
      } else if (!(Psi1[i] == Psi1[i + 1])) {
#ifndef Complex
        energyEN +=
            (pow(abs(currentNum1A), 2) * Nmc / (Nmc - 1) - currentNum2A) /
            (det_energy[i] - E0);
        energyENLargeEps +=
            (pow(abs(currentNum1B), 2) * Nmc / (Nmc - 1) - currentNum2B) /
            (det_energy[i] - E0);
#else
        energyEN += (pow(abs(currentNum1A), 2) * Nmc / (Nmc - 1) -
                     currentNum2A.real()) /
                    (det_energy[i] - E0);
        energyENLargeEps += (pow(abs(currentNum1B), 2) * Nmc / (Nmc - 1) -
                             currentNum2B.real()) /
                            (det_energy[i] - E0);
#endif
        currentNum1A = 0.;
        currentNum2A = 0.;
        currentNum1B = 0.;
        currentNum2B = 0.;
      }
      i++;
    }

    totalPT = 0;
    totalPTLargeEps = 0;

    totalPT += energyEN;
    totalPTLargeEps += energyENLargeEps;

    double finalE = 0., finalELargeEps = 0;
#ifndef SERIAL
    MPI_Allreduce(&totalPT, &finalE, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&totalPTLargeEps, &finalELargeEps, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
#else
    finalE = totalPT;
    finalELargeEps = totalPTLargeEps;
#endif

    if (commrank == 0) {
      currentIter++;
      AvgenergyEN += -finalE + finalELargeEps + EptLarge;
      AvgenergyEN2 += pow(-finalE + finalELargeEps + EptLarge, 2);
      stddev = currentIter < 5
                   ? 1e4
                   : pow((currentIter * AvgenergyEN2 - pow(AvgenergyEN, 2)) /
                             currentIter / (currentIter - 1) / currentIter,
                         0.5);
      if (currentIter < 5)
        std::cout << format("%6i  %18.10f  %5i %18.10f %10s  %10.2f") %
                         (currentIter) %
                         (E0 - finalE + finalELargeEps + EptLarge) % (root) %
                         (E0 + AvgenergyEN / currentIter) % "--" %
                         (getTime() - startofCalc);
      else
        std::cout << format("%6i  %18.10f  %5i %18.10f %10.2e  %10.2f") %
                         (currentIter) %
                         (E0 - finalE + finalELargeEps + EptLarge) % (root) %
                         (E0 + AvgenergyEN / currentIter) % stddev %
                         (getTime() - startofCalc);
      pout << endl;
    }

#ifndef SERIAL
    MPI_Bcast(&currentIter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&AvgenergyEN, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

    uniqueDEH.clear();
    if (stddev < schd.targetError) {
      AvgenergyEN /= currentIter;
      // pout << "Standard Error : " << stddev << " less than " <<
      // schd.targetError << endl;
      pout << "Semistochastic PT calculation converged" << endl;
      pout << format("PTEnergy: %18.10f") % (E0 + AvgenergyEN) << " +/- ";
      pout << format("%8.2e") % (stddev) << endl;
      pout << format("Time(s):  %10.2f") % (getTime() - startofCalc) << endl;
      break;
    }
  }
  return AvgenergyEN;
}

double SHCIbasics::DoPerturbativeDeterministic(
    Determinant *Dets, CItype *ci, int DetsSize, double &E0, oneInt &I1,
    twoInt &I2, twoIntHeatBathSHM &I2HB, vector<int> &irrep, schedule &schd,
    double coreE, int nelec, int root, vector<MatrixXx> &vdVector,
    double &Psi1Norm, bool appendPsi1ToPsi0) {
#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  int norbs = Determinant::norbs;

  Determinant *SortedDets;
  std::vector<Determinant> SortedDetsvec;
  if (commrank == 0) {
    for (int i = 0; i < DetsSize; i++) SortedDetsvec.push_back(Dets[i]);
    std::sort(SortedDetsvec.begin(), SortedDetsvec.end());
  }
#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  SHMVecFromVecs(SortedDetsvec, SortedDets, shciSortedDets, SortedDetsSegment,
                 regionSortedDets);
  SortedDetsvec.clear();

  double energyEN = 0.0;
  double Psi1NormProc = 0.0;

  StitchDEH uniqueDEH;
  double totalPT = 0.0;
  int ntries = 0;

  int size = commsize, rank = commrank;
  vector<size_t> all_to_all(size * size, 0);

  if (schd.DoRDM || schd.doResponse) {
    uniqueDEH.extra_info = true;
    for (int i = 0; i < DetsSize; i++) {
      if (i % size != rank) continue;
      SHCIgetdeterminants::getDeterminantsDeterministicPTKeepRefDets(
          Dets[i], i, abs(schd.epsilon2 / ci[i]), ci[i], I1, I2, I2HB, irrep,
          coreE, E0, *uniqueDEH.Det, *uniqueDEH.Num, *uniqueDEH.Energy,
          *uniqueDEH.var_indices_beforeMerge,
          *uniqueDEH.orbDifference_beforeMerge, schd, nelec);
    }
  } else {
    for (int i = 0; i < DetsSize; i++) {
      if ((i % size != rank)) continue;

      SHCIgetdeterminants::getDeterminantsDeterministicPT(
          Dets[i], abs(schd.epsilon2 / ci[i]), ci[i], 0.0, I1, I2, I2HB, irrep,
          coreE, E0, *uniqueDEH.Det, *uniqueDEH.Num, *uniqueDEH.Energy, schd, 0,
          nelec);
      // if (i%100000 == 0 && omp_get_thread_num()==0 && commrank == 0) pout <<
      // "# " << i << endl;
    }
  }

  if (commsize > 1) {
    std::shared_ptr<vector<Determinant>> &Det = uniqueDEH.Det;
    std::shared_ptr<vector<CItype>> &Num = uniqueDEH.Num;
    std::shared_ptr<vector<double>> &Energy = uniqueDEH.Energy;
    std::shared_ptr<vector<int>> &var_indices =
        uniqueDEH.var_indices_beforeMerge;
    std::shared_ptr<vector<size_t>> &orbDifference =
        uniqueDEH.orbDifference_beforeMerge;

    std::vector<size_t> hashValues(Det->size());

    std::vector<size_t> all_to_all_cumulative(size, 0);
    for (int i = 0; i < Det->size(); i++) {
      hashValues[i] = Det->at(i).getHash();
      all_to_all[rank * size + hashValues[i] % size]++;
    }
    for (int i = 0; i < size; i++)
      all_to_all_cumulative[i] =
          i == 0 ? all_to_all[rank * size + i]
                 : all_to_all_cumulative[i - 1] + all_to_all[rank * size + i];

    size_t dsize = Det->size() == 0 ? 1 : Det->size();
    vector<Determinant> atoaDets(dsize);
    vector<CItype> atoaNum(dsize);
    vector<double> atoaE(dsize);
    vector<int> atoaVarIndices;
    vector<size_t> atoaOrbDiff;
    if (schd.DoRDM || schd.doResponse) {
      atoaVarIndices.resize(dsize);
      atoaOrbDiff.resize(dsize);
    }

#ifndef SERIAL
    vector<size_t> all_to_allCopy = all_to_all;
    MPI_Allreduce(&all_to_allCopy[0], &all_to_all[0], 2 * size * size, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);
#endif
    vector<size_t> counter(size, 0);
    for (int i = 0; i < Det->size(); i++) {
      int toProc = hashValues[i] % size;
      size_t index = toProc == 0
                         ? counter[0]
                         : counter[toProc] + all_to_all_cumulative[toProc - 1];

      atoaDets[index] = Det->at(i);
      atoaNum[index] = Num->at(i);
      atoaE[index] = Energy->at(i);
      if (schd.DoRDM || schd.doResponse) {
        atoaVarIndices[index] = var_indices->at(i);
        atoaOrbDiff[index] = orbDifference->at(i);
      }
      counter[toProc]++;
    }

    vector<int> sendcts(size, 0), senddisp(size, 0), recvcts(size, 0),
        recvdisp(size, 0);
    vector<int> sendctsDets(size, 0), senddispDets(size, 0),
        recvctsDets(size, 0), recvdispDets(size, 0);
    vector<int> sendctsVarDiff(size, 0), senddispVarDiff(size, 0),
        recvctsVarDiff(size, 0), recvdispVarDiff(size, 0);

    size_t recvSize = 0;
    for (int i = 0; i < size; i++) {
      sendcts[i] =
          all_to_all[rank * size + i] * sizeof(CItype) / sizeof(double);
      senddisp[i] = i == 0 ? 0 : senddisp[i - 1] + sendcts[i - 1];
      recvcts[i] =
          all_to_all[i * size + rank] * sizeof(CItype) / sizeof(double);
      recvdisp[i] = i == 0 ? 0 : recvdisp[i - 1] + recvcts[i - 1];

      sendctsDets[i] =
          all_to_all[rank * size + i] * sizeof(Determinant) / sizeof(double);
      senddispDets[i] = i == 0 ? 0 : senddispDets[i - 1] + sendctsDets[i - 1];
      recvctsDets[i] =
          all_to_all[i * size + rank] * sizeof(Determinant) / sizeof(double);
      recvdispDets[i] = i == 0 ? 0 : recvdispDets[i - 1] + recvctsDets[i - 1];

      sendctsVarDiff[i] = all_to_all[rank * size + i];
      senddispVarDiff[i] =
          i == 0 ? 0 : senddispVarDiff[i - 1] + sendctsVarDiff[i - 1];
      recvctsVarDiff[i] = all_to_all[i * size + rank];
      recvdispVarDiff[i] =
          i == 0 ? 0 : recvdispVarDiff[i - 1] + recvctsVarDiff[i - 1];

      recvSize += all_to_all[i * size + rank];
    }

    recvSize = recvSize == 0 ? 1 : recvSize;
    Det->resize(recvSize), Num->resize(recvSize), Energy->resize(recvSize);
    if (schd.DoRDM || schd.doResponse) {
      var_indices->resize(recvSize);
      orbDifference->resize(recvSize);
    }

#ifndef SERIAL
    MPI_Alltoallv(&atoaNum.at(0), &sendcts[0], &senddisp[0], MPI_DOUBLE,
                  &Num->at(0), &recvcts[0], &recvdisp[0], MPI_DOUBLE,
                  MPI_COMM_WORLD);
    MPI_Alltoallv(&atoaE.at(0), &sendctsVarDiff[0], &senddispVarDiff[0],
                  MPI_DOUBLE, &Energy->at(0), &recvctsVarDiff[0],
                  &recvdispVarDiff[0], MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Alltoallv(&atoaDets.at(0).repr[0], &sendctsDets[0], &senddispDets[0],
                  MPI_DOUBLE, &(Det->at(0).repr[0]), &recvctsDets[0],
                  &recvdispDets[0], MPI_DOUBLE, MPI_COMM_WORLD);

    if (schd.DoRDM || schd.doResponse) {
      MPI_Alltoallv(&atoaVarIndices.at(0), &sendctsVarDiff[0],
                    &senddispVarDiff[0], MPI_INT, &(var_indices->at(0)),
                    &recvctsVarDiff[0], &recvdispVarDiff[0], MPI_INT,
                    MPI_COMM_WORLD);
      MPI_Alltoallv(&atoaOrbDiff.at(0), &sendctsVarDiff[0], &senddispVarDiff[0],
                    MPI_DOUBLE, &(orbDifference->at(0)), &recvctsVarDiff[0],
                    &recvdispVarDiff[0], MPI_DOUBLE, MPI_COMM_WORLD);
    }
#endif
    uniqueDEH.Num2->clear();
  }
  uniqueDEH.MergeSortAndRemoveDuplicates();
  uniqueDEH.RemoveDetsPresentIn(SortedDets, DetsSize);

  vector<Determinant> &hasHEDDets = *uniqueDEH.Det;
  vector<CItype> &hasHEDNumerator = *uniqueDEH.Num;
  vector<double> &hasHEDEnergy = *uniqueDEH.Energy;

  double PTEnergy = 0.0;
  double psi1normthrd = 0.0;
  for (size_t i = 0; i < hasHEDDets.size(); i++) {
    psi1normthrd += pow(abs(hasHEDNumerator[i] / (E0 - hasHEDEnergy[i])), 2);
    PTEnergy += pow(abs(hasHEDNumerator[i]), 2) / (E0 - hasHEDEnergy[i]);
  }

  Psi1NormProc += psi1normthrd;
  totalPT += PTEnergy;

  /*
  //PROBABLY SHOULD DELETE IT
  {
    char file [5000];
    sprintf (file, "%s/%d-PTdata.bkp" , schd.prefix[0].c_str(), commrank );
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    printf("writing file %s\n", file);
    save << uniqueDEH;
    ofs.close();
  }
  */

  double finalE = 0.;
#ifndef SERIAL
  mpi::all_reduce(world, totalPT, finalE, std::plus<double>());
  mpi::all_reduce(world, Psi1NormProc, Psi1Norm, std::plus<double>());
#else
  finalE = totalPT;
#endif

  if (commrank == 0) {
    pout << "Deterministic PT calculation converged" << endl;
    pout << format("PTEnergy: %18.10f") % (E0 + finalE) << endl;
    pout << format("Time(s):  %10.2f") % (getTime() - startofCalc) << endl;
  }

  if (schd.doResponse || schd.DoRDM) {  // build RHS for the lambda equation
    pout << endl << "Now calculating PT RDM" << endl;
    MatrixXx s2RDM, twoRDM;
    SHCIrdm::loadRDM(schd, s2RDM, twoRDM, root);
#ifndef SERIAL
    mpi::broadcast(world, s2RDM, 0);
    if (schd.DoSpinRDM) mpi::broadcast(world, twoRDM, 0);
#endif
    if (commrank != 0) {
      s2RDM = 0. * s2RDM;
      twoRDM = 0. * twoRDM;
    }
    // SHCIrdm::ComputeEnergyFromSpatialRDM(norbs/2, nelec, I1, I2, coreE,
    // s2RDM);
    SHCIrdm::UpdateRDMResponsePerturbativeDeterministic(
        Dets, DetsSize, ci, E0, I1, I2, schd, coreE, nelec, norbs, uniqueDEH,
        root, Psi1Norm, s2RDM, twoRDM);
    // SHCIrdm::ComputeEnergyFromSpatialRDM(norbs/2, nelec, I1, I2, coreE,
    // s2RDM);
    SHCIrdm::saveRDM(schd, s2RDM, twoRDM, root);

    if (schd.RdmType == RELAXED) {
      // construct the vector Via x da
      // where Via is the perturbation matrix element
      // da are the elements of the PT wavefunctions
      vdVector[root] = MatrixXx::Zero(DetsSize, 1);

      vector<Determinant> &uniqueDets = *uniqueDEH.Det;

      vector<double> &uniqueEnergy = *uniqueDEH.Energy;
      vector<CItype> &uniqueNumerator = *uniqueDEH.Num;
      vector<vector<int>> &uniqueVarIndices = *uniqueDEH.var_indices;
      vector<vector<size_t>> &uniqueOrbDiff = *uniqueDEH.orbDifference;

      for (int a = 0; a < uniqueDets.size(); a++) {
        CItype da = uniqueNumerator[a] /
                    (E0 - uniqueEnergy[a]);  // coefficient for det a
        for (int i = 0; i < uniqueVarIndices[a].size(); i++) {
          int I = uniqueVarIndices[a][i];  // index of the Var determinant
          size_t orbDiff;
#ifndef Complex
          vdVector[root](I, 0) -=
              da * Hij(Dets[I], uniqueDets[a], I1, I2, coreE, orbDiff);
#else
          vdVector[root](I, 0) -=
              conj(da) * Hij(Dets[I], uniqueDets[a], I1, I2, coreE, orbDiff);
#endif
        }
      }

#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &vdVector[root](0, 0), vdVector[root].rows(),
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    }
  }
  return finalE;
}

void unpackTrevState(vector<Determinant> &Dets, int &DetsSize,
                     vector<MatrixXx> &ci, SparseHam &sparseHam, bool DoRDM,
                     oneInt &I1, twoInt &I2, double &coreE) {
#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  int oldLen = Dets.size();
  vector<int> partnerLocation(oldLen, -1);
  if (commrank == 0) {
    if (Determinant::Trev != 0) {
      int numDets = 0;
      for (int i = 0; i < DetsSize; i++) {
        if (Dets[i].hasUnpairedElectrons())
          numDets += 2;
        else
          numDets += 1;
      }
      Dets.resize(numDets);
      vector<MatrixXx> cibkp = ci;
      for (int i = 0; i < ci.size(); i++) {
        ci[i].resize(Dets.size(), 1);
        ci[i].block(0, 0, cibkp[i].rows(), 1) = 1. * cibkp[i];
      }

      int newIndex = 0, oldLen = cibkp[0].rows();
      for (int i = 0; i < oldLen; i++) {
        if (Dets[i].hasUnpairedElectrons()) {
          partnerLocation[i] = newIndex;
          Dets[newIndex + oldLen] = Dets[i];
          Dets[newIndex + oldLen].flipAlphaBeta();
          for (int j = 0; j < ci.size(); j++) {
            ci[j](i, 0) = cibkp[j](i, 0) / sqrt(2.0);
            double parity = Dets[i].parityOfFlipAlphaBeta();
            ci[j](newIndex + oldLen, 0) =
                Determinant::Trev * parity * cibkp[j](i, 0) / sqrt(2.0);
          }
          newIndex++;
        }
      }
    }
  }
  DetsSize = Dets.size();
#ifndef SERIAL
  MPI_Bcast(&DetsSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&oldLen, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
}

// this takes in a ci vector for determinants placed in Dets
// it then does a SHCI varitional calculation and the resulting
// ci and dets are returned here
// At input usually the Dets will just have a HF or some such determinant
// and ci will be just 1.0
vector<double> SHCIbasics::DoVariational(vector<MatrixXx> &ci,
                                         vector<Determinant> &Dets,
                                         schedule &schd, twoInt &I2,
                                         twoIntHeatBathSHM &I2HB,
                                         vector<int> &irrep, oneInt &I1,
                                         double &coreE, int nelec, bool DoRDM) {
  int proc = 0, nprocs = 1;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  boost::mpi::communicator world;
#endif

  // Put determinants on the shared memory
  Determinant *SHMDets, *SortedDets;
  SHMVecFromVecs(Dets, SHMDets, shciDetsCI, DetsCISegment, regionDetsCI);
  if (proc != 0) Dets.resize(0);
  std::vector<Determinant> SortedDetsvec;  // only proc 1 has it
  if (commrank == 0) {
    SortedDetsvec = Dets;
    std::sort(SortedDetsvec.begin(), SortedDetsvec.end());
  }
  int SortedDetsSize = SortedDetsvec.size(), DetsSize = Dets.size();
  SHMVecFromVecs(SortedDetsvec, SortedDets, shciSortedDets, SortedDetsSegment,
                 regionSortedDets);
  Dets.clear();
  SortedDetsvec.clear();
#ifndef SERIAL
  mpi::broadcast(world, SortedDetsSize, 0);
  mpi::broadcast(world, DetsSize, 0);
#endif

  // sometimes coreenergy is huge and it kills the stability of davidson solver
  // so make it zero and just add it to the converged energy
  double coreEbkp = coreE;
  coreE = 0.0;

  if (schd.outputlevel > 0 && commrank == 0)
    Time::print_time("start variation");

  // initialize the sparse Hamiltonian and helpers
  int nroots = ci.size();
  SHCImakeHamiltonian::HamHelpers2 helper2;
  SHCImakeHamiltonian::SparseHam sparseHam;
  if (schd.DavidsonType == DISK) {
    sparseHam.diskio = true;
    sparseHam.BatchSize = 100000;
    // sparseHam.BatchSize = 10;
    sparseHam.setNbatches(DetsSize);
    sparseHam.prefix = schd.prefix[0];
  }

  MatrixXx diag;

  size_t norbs = 2. * I2.Direct.rows();
  int Norbs = norbs;

  CItype e0 = SHMDets[0].Energy(I1, I2, coreE);
  size_t orbDiff;
  if (Determinant::Trev != 0)
    updateHijForTReversal(e0, SHMDets[0], SHMDets[0], I1, I2, coreE, orbDiff);
  vector<double> E0(nroots, abs(e0));

  // make helpers and then put them on the shared memory
  if (proc == 0) {
    helper2.PopulateHelpers(SHMDets, DetsSize, 0);
  }
  helper2.MakeSHMHelpers();

  // if it is not direct Hamiltonian then generate it
  if (schd.DavidsonType != DIRECT) {
    sparseHam.makeFromHelper(helper2, SHMDets, 0, DetsSize, Norbs, I1, I2,
                             coreE, schd.DoRDM || schd.DoOneRDM);
  }

  // update the Hamiltonian with SOC terms
#ifdef Complex
  SHCImakeHamiltonian::updateSOCconnections(
      SHMDets, 0, DetsSize, SortedDets, sparseHam.connections,
      sparseHam.orbDifference, sparseHam.Helements, norbs, I1, nelec, false);
#endif

  pout << format("%4s %4s  %10s  %10.2e   %18s   %9s  %10s\n") % ("Iter") %
              ("Root") % ("Eps1 ") % ("#Var. Det.") % ("Energy") %
              ("#Davidson") % ("Time(s)");

  int prevSize = 0;

  // If this is a restart calculation then read from disk
  int iterstart = 0;
  if (schd.restart || schd.fullrestart) {
    bool converged;
    readVariationalResult(iterstart, ci, Dets, sparseHam, E0, converged, schd,
                          helper2);

    // after reading restart put dets on shared memory
    SHMVecFromVecs(Dets, SHMDets, shciDetsCI, DetsCISegment, regionDetsCI);
    DetsSize = Dets.size();
    SortedDetsSize = DetsSize;

    // put sorted dets on shared memory as well
    SHMVecFromVecs(Dets, SortedDets, shciSortedDets, SortedDetsSegment,
                   regionSortedDets);
#ifndef SERIAL
    mpi::broadcast(world, SortedDetsSize, 0);
    mpi::broadcast(world, DetsSize, 0);
#endif
    if (localrank == 0) std::sort(SortedDets, SortedDets + SortedDetsSize);
#ifndef SERIAL
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    Dets.clear();

    // Make helpers and make sparse hamiltonian if full restart and not direct
    helper2.MakeSHMHelpers();
    if (!(schd.DavidsonType == DIRECT ||
          (!schd.fullrestart && converged &&
           iterstart >= schd.epsilon1.size() - 1))) {
      sparseHam.clear();
      sparseHam.makeFromHelper(helper2, SHMDets, 0, DetsSize, Norbs, I1, I2,
                               coreE, schd.DoRDM || schd.DoOneRDM);
    }

    for (int i = 0; i < E0.size(); i++)
      pout << format("%4i %4i  %10.2e  %10.2e -   %18.10f  %10.2f\n") %
                  (iterstart) % (i) % schd.epsilon1[iterstart] % DetsSize %
                  (E0[i] + coreEbkp) % (getTime() - startofCalc);

    if (!schd.fullrestart)
      iterstart++;
    else
      iterstart = 0;

    // if the calculation is converged then exit
    if (schd.outputlevel > 0)
      pout << "Converged: " << converged
           << " loaded iter: " << iterstart
           << " max iter: " << schd.epsilon1.size()
           << " loaded eps1: " << schd.epsilon1[iterstart-1]
           << " min eps1: " << schd.epsilon1[schd.epsilon1.size()-1]
           << endl;
    if (converged && iterstart-1 <= schd.epsilon1.size() &&
        schd.epsilon1[iterstart-1] <= schd.epsilon1[schd.epsilon1.size()-1]) {
      for (int i = 0; i < E0.size(); i++)
        E0[i] += coreEbkp;
      coreE = coreEbkp;
      pout << "# restarting from a converged calculation, moving to "
              "perturbative part.!!"
           << endl;
      Dets.resize(DetsSize);
      for (int i = 0; i < DetsSize; i++) Dets[i] = SHMDets[i];

#ifndef SERIAL
      MPI_Barrier(MPI_COMM_WORLD);
      mpi::broadcast(world, E0, 0);
#endif
      unpackTrevState(Dets, DetsSize, ci, sparseHam, false, I1, I2, coreE);

      return E0;
    }
  }

  for (int iter = iterstart; iter < schd.epsilon1.size(); iter++) {
    double epsilon1 = schd.epsilon1[iter];
    StitchDEH uniqueDEH;

    // for multiple states, use the sum of squares of states
    // to do the seclection process
    if (schd.outputlevel > 0)
      pout << format("#-------------Iter=%4i---------------") % iter << endl;

    CItype *cMaxSHM;
    vector<CItype> cMax;
    if (proc == 0) {
      cMax.resize(ci[0].rows(), 0);
      for (int j = 0; j < ci[0].rows(); j++) {
        for (int i = 0; i < ci.size(); i++) cMax[j] += pow(abs(ci[i](j, 0)), 2);
        cMax[j] = pow(cMax[j], 0.5);
      }
    }

    SHMVecFromVecs(cMax, cMaxSHM, shcicMax, cMaxSegment, regioncMax);
    cMax.clear();

    CItype zero = 0.0;

    for (int i = 0; i < SortedDetsSize; i++) {
      if (i % (commsize) != commrank) continue;
#ifndef Complex
      SHCIgetdeterminants::getDeterminantsVariationalApprox(
          SHMDets[i], epsilon1 / abs(cMaxSHM[i]), cMaxSHM[i], zero, I1, I2,
          I2HB, irrep, coreE, E0[0], *uniqueDEH.Det, schd, 0, nelec, SortedDets,
          SortedDetsSize);

#else
      SHCIgetdeterminants::getDeterminantsVariational(
          SHMDets[i], epsilon1 / abs(cMaxSHM[i]), cMaxSHM[i], zero, I1, I2,
          I2HB, irrep, coreE, E0[0], *uniqueDEH.Det, schd, 0, nelec);
#endif
    }

    if (Determinant::Trev != 0) {
      for (int i = 0; i < uniqueDEH.Det->size(); i++)
        uniqueDEH.Det->at(i).makeStandard();
    }

    //*********
    // Remove duplicates and put all the dets on all the nodes

    sort(uniqueDEH.Det->begin(), uniqueDEH.Det->end());
    uniqueDEH.Det->erase(unique(uniqueDEH.Det->begin(), uniqueDEH.Det->end()),
                         uniqueDEH.Det->end());

    if (Determinant::Trev != 0)
      uniqueDEH.RemoveOnlyDetsPresentIn(SortedDets, SortedDetsSize);
#ifdef Complex
    uniqueDEH.RemoveOnlyDetsPresentIn(SortedDets, SortedDetsSize);
#endif

#ifndef SERIAL
    for (int level = 0; level < ceil(log2(nprocs)); level++) {
      if (proc % ipow(2, level + 1) == 0 && proc + ipow(2, level) < nprocs) {
        int getproc = proc + ipow(2, level);
        long numDets = 0;
        long oldSize = uniqueDEH.Det->size();
        long maxint = 26843540;
        MPI_Recv(&numDets, 1, MPI_DOUBLE, getproc, getproc, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        long totalMemory = numDets * DetLen;

        if (totalMemory != 0) {
          uniqueDEH.Det->resize(oldSize + numDets);
          for (int i = 0; i < (totalMemory / maxint); i++)
            MPI_Recv(&(uniqueDEH.Det->at(oldSize).repr[0]) + i * maxint, maxint,
                     MPI_DOUBLE, getproc, getproc, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
          MPI_Recv(&(uniqueDEH.Det->at(oldSize).repr[0]) +
                       (totalMemory / maxint) * maxint,
                   totalMemory - (totalMemory / maxint) * maxint, MPI_DOUBLE,
                   getproc, getproc, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          sort(uniqueDEH.Det->begin(), uniqueDEH.Det->end());
          uniqueDEH.Det->erase(
              unique(uniqueDEH.Det->begin(), uniqueDEH.Det->end()),
              uniqueDEH.Det->end());
        }

      } else if (proc % ipow(2, level + 1) == 0 &&
                 proc + ipow(2, level) >= nprocs) {
        continue;
      } else if (proc % ipow(2, level) == 0) {
        int toproc = proc - ipow(2, level);
        int proc = commrank;
        long numDets = uniqueDEH.Det->size();
        long maxint = 26843540;
        long totalMemory = numDets * DetLen;
        MPI_Send(&numDets, 1, MPI_DOUBLE, toproc, proc, MPI_COMM_WORLD);

        if (totalMemory != 0) {
          for (int i = 0; i < (totalMemory / maxint); i++)
            MPI_Send(&(uniqueDEH.Det->at(0).repr[0]) + i * maxint, maxint,
                     MPI_DOUBLE, toproc, proc, MPI_COMM_WORLD);
          MPI_Send(
              &(uniqueDEH.Det->at(0).repr[0]) + (totalMemory / maxint) * maxint,
              totalMemory - (totalMemory / maxint) * maxint, MPI_DOUBLE, toproc,
              proc, MPI_COMM_WORLD);
          uniqueDEH.clear();
        }
      }
    }
    //*************

#endif

    //**********
    // Resize X0 and dets and sorteddets
    vector<MatrixXx> X0;
    X0.resize(ci.size());
    vector<Determinant> &newDets = *uniqueDEH.Det;
    if (proc == 0) {
      X0 = vector<MatrixXx>(ci.size(), MatrixXx(DetsSize + newDets.size(), 1));
      for (int i = 0; i < ci.size(); i++) {
        X0[i].setZero(DetsSize + newDets.size(), 1);
        X0[i].block(0, 0, ci[i].rows(), 1) = 1. * ci[i];
      }

      Dets.resize(DetsSize + newDets.size());
      for (int i = 0; i < DetsSize; i++) Dets[i] = SHMDets[i];
      for (int i = 0; i < newDets.size(); i++) Dets[i + DetsSize] = newDets[i];

      DetsSize = Dets.size();
    }
    uniqueDEH.resize(0);
    SHMVecFromVecs(Dets, SHMDets, shciDetsCI, DetsCISegment, regionDetsCI);
    Dets.clear();

#ifndef SERIAL
    mpi::broadcast(world, DetsSize, 0);
#endif
    //************
    if (commrank == 0 && schd.DavidsonType == DIRECT)
      printf("New size of determinant space %8i\n", DetsSize);

    //*************
    // make Helpers and Hamiltonian
    if (proc == 0) {
      helper2.PopulateHelpers(SHMDets, DetsSize, SortedDetsSize);
    }
    helper2.MakeSHMHelpers();
    if (schd.DavidsonType != DIRECT) {
      sparseHam.makeFromHelper(helper2, SHMDets, SortedDetsSize, DetsSize,
                               Norbs, I1, I2, coreE,
                               schd.DoRDM || schd.DoOneRDM);
    }
    //************

    // we update the sharedvectors after Hamiltonian is formed because needed
    // the dets size from previous iterations
    SHMVecFromVecs(SHMDets, DetsSize, SortedDets, shciSortedDets,
                   SortedDetsSegment, regionSortedDets);
    if (localrank == 0) std::sort(SortedDets, SortedDets + DetsSize);
#ifndef SERIAL
    MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef Complex
    SHCImakeHamiltonian::updateSOCconnections(
        SHMDets, SortedDetsSize, DetsSize, SortedDets, sparseHam.connections,
        sparseHam.orbDifference, sparseHam.Helements, norbs, I1, nelec, false);
#endif

    SortedDetsSize = DetsSize;
#ifndef SERIAL
    mpi::broadcast(world, SortedDetsSize, 0);
#endif

    // MERGE 2018.07.13, don't know where that comes from, commenting it
    // if (proc == 0) {
    //  MatrixXx diagbkp = diag;
    //  diag =MatrixXx::Zero(DetsSize,1);
    //  for (int k=0; k<diagbkp.rows(); k++)
    //    diag(k,0) = diagbkp(k,0);

    //  for (size_t k=diagbkp.rows(); k<DetsSize; k++) {
    //    CItype hij = SHMDets[k].Energy(I1, I2, coreE);
    //    diag(k,0) = hij;
    //  }
    //}
    //
    //
    // double prevE0 = E0[0];
    // if (iter == 0) prevE0 = -10.0;
    // Hmult2 H(sparseHam);
    // HmultDirect Hdirect(helper2, SHMDets, DetsSize, 0, Norbs,
    //		I1, I2, coreE, diag);
    // if (schd.DavidsonType == DISK) sparseHam.setNbatches(DetsSize);
    ////pout << "nbatches : " << sparseHam.Nbatches << endl;
    ////cout << commrank << "  " << sparseHam.Helements[0][0] << endl;
    // int numIter = 0;

    // Make the diagonal elements so that preconditioner can be applied in
    // davidson
    if (proc == 0) {
      MatrixXx diagbkp = diag;
      diag = MatrixXx::Zero(DetsSize, 1);
      for (int k = 0; k < diagbkp.rows(); k++) diag(k, 0) = diagbkp(k, 0);

      for (size_t k = diagbkp.rows(); k < DetsSize; k++) {
        CItype hij = SHMDets[k].Energy(I1, I2, coreE);
        diag(k, 0) = hij;
      }
    }

    // Hmult has a operator() that lets you multiply a vector with H to generate
    // and output
    double prevE0 = E0[0];
    if (iter == 0) prevE0 = -10.0;
    Hmult2 H(sparseHam);
    HmultDirect Hdirect(helper2, SHMDets, DetsSize, 0, Norbs, I1, I2, coreE,
                        diag);
    if (schd.DavidsonType == DISK) sparseHam.setNbatches(DetsSize);
    int numIter = 0;

    // do the davidson calculation
    if (schd.DavidsonType == DIRECT)
      E0 = davidsonDirect(Hdirect, X0, diag, schd.nroots + 2,
                          schd.davidsonTolLoose, numIter, schd.outputlevel > 0);
    else
      E0 = davidson(H, X0, diag, schd.nroots + 4, schd.davidsonTolLoose,
                    numIter, schd.outputlevel > 0);

    if (schd.outputlevel > 0 && commrank == 0)
      Time::print_time("davidson finished");

#ifndef SERIAL
    mpi::broadcast(world, E0, 0);
#endif

    pout << format("%4i %4i  %10.2e  %10.2e") % (iter) % (0) %
                schd.epsilon1[iter] % (newDets.size() + DetsSize);
    pout << format("   %18.10f  %9i  %10.2f\n") % (E0[0] + coreEbkp) %
                (numIter) % (getTime() - startofCalc);

    for (int i = 1; i < E0.size(); i++)
      pout << format("%4i %4i  %10.2e  %10.2e   %18.10f  %9i  %10.2f\n") %
                  (iter) % (i) % schd.epsilon1[iter] % DetsSize %
                  (E0[i] + coreEbkp) % (numIter) % (getTime() - startofCalc);
    if (E0.size() > 1) pout << endl;

    // update the civector
    if (proc == 0) {
      for (int i = 0; i < E0.size(); i++) {
        ci[i].resize(DetsSize, 1);
        ci[i] = 1.0 * X0[i];
        X0[i].resize(0, 0);
      }
    }

    // the variational step has converged
    if (abs(E0[0] - prevE0) < schd.dE || iter == schd.epsilon1.size() - 1) {
      pout << "Performing final tight davidson with tol: " << schd.davidsonTol
           << endl;

      if (schd.DavidsonType == DIRECT)
        E0 = davidsonDirect(Hdirect, ci, diag, schd.nroots + 4,
                            schd.davidsonTol, numIter, true);
      else
        E0 = davidson(H, ci, diag, schd.nroots + 4, schd.davidsonTol, numIter,
                      false);

#ifndef SERIAL
      mpi::broadcast(world, E0, 0);
#endif
      pout << "Exiting variational iterations" << endl;
      if (commrank == 0) {
        Dets.resize(DetsSize);
        for (int i = 0; i < DetsSize; i++) Dets[i] = SHMDets[i];
      }
#ifndef SERIAL
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      if (schd.io) {
        writeVariationalResult(iter, ci, Dets, sparseHam, E0, true, schd,
                               helper2);
      }

      unpackTrevState(Dets, DetsSize, ci, sparseHam,
                      schd.DoRDM || schd.doResponse, I1, I2, coreE);
      if (Determinant::Trev != 0) {
        // we add additional determinats
        SHMVecFromVecs(Dets, SHMDets, shciDetsCI, DetsCISegment, regionDetsCI);
      }

      if (schd.DoOneRDM) {
        pout << "\nCalculating 1-RDM" << endl;
        for (int i = 0; i < schd.nroots; i++) {
          MatrixXx s1RDM, oneRDM;
          oneRDM = MatrixXx::Zero(norbs, norbs);
          s1RDM = MatrixXx::Zero(norbs / 2, norbs / 2);
          CItype *SHMci;
          SHMVecFromMatrix(ci[i], SHMci, shciDetsCI, DavidsonSegment,
                           regionDavidson);
          SHCIrdm::EvaluateOneRDM(sparseHam.connections, SHMDets, DetsSize,
                                  SHMci, SHMci, sparseHam.orbDifference, nelec,
                                  schd, i, oneRDM, s1RDM);
          SHCIrdm::save1RDM(schd, s1RDM, oneRDM, i);
        }
      }

      if (DoRDM || schd.doResponse) {
        // if (schd.DavidsonType == DIRECT) {
        // pout << "RDM not implemented with direct davidson." << endl;
        // exit(0);
        //}
        pout << "\nCalculating 2-RDM" << endl;
        int trev = Determinant::Trev;
        Determinant::Trev = 0;
        for (int i = 0; i < schd.nroots; i++) {
          CItype *SHMci;
          SHMVecFromMatrix(ci[i], SHMci, shciDetsCI, DavidsonSegment,
                           regionDavidson);

          MatrixXx twoRDM;
          if (schd.DoSpinRDM)
            twoRDM = MatrixXx::Zero(norbs * (norbs + 1) / 2,
                                    norbs * (norbs + 1) / 2);
          MatrixXx s2RDM =
              MatrixXx::Zero((norbs / 2) * norbs / 2, (norbs / 2) * norbs / 2);

          if ((trev != 0 || schd.DavidsonType == DIRECT)) {
            // now that we have unpacked Trev list, we can forget about
            // t-reversal symmetry
            if (proc == 0) {
              helper2.clear();
              helper2.PopulateHelpers(SHMDets, DetsSize, 0);
            }
            if (i == 0) helper2.MakeSHMHelpers();
            SHCIrdm::makeRDM(
                helper2.AlphaMajorToBetaLen, helper2.AlphaMajorToBetaSM,
                helper2.AlphaMajorToDetSM, helper2.BetaMajorToAlphaLen,
                helper2.BetaMajorToAlphaSM, helper2.BetaMajorToDetSM,
                helper2.SinglesFromAlphaLen, helper2.SinglesFromAlphaSM,
                helper2.SinglesFromBetaLen, helper2.SinglesFromBetaSM, SHMDets,
                DetsSize, Norbs, nelec, SHMci, SHMci, s2RDM);

            // if (schd.outputlevel>0)
            SHCIrdm::ComputeEnergyFromSpatialRDM(norbs / 2, nelec, I1, I2,
                                                 coreEbkp, s2RDM);
            SHCIrdm::saveRDM(schd, s2RDM, twoRDM, i);

          } else {
            SHCIrdm::EvaluateRDM(sparseHam.connections, SHMDets, DetsSize,
                                 SHMci, SHMci, sparseHam.orbDifference, nelec,
                                 schd, i, twoRDM, s2RDM);
            // if (schd.outputlevel>0)
            SHCIrdm::ComputeEnergyFromSpatialRDM(norbs / 2, nelec, I1, I2,
                                                 coreEbkp, s2RDM);
            SHCIrdm::saveRDM(schd, s2RDM, twoRDM, i);

            boost::interprocess::shared_memory_object::remove(
                shciDetsCI.c_str());
          }
        }  // for i
      }
      sparseHam.resize(0);

      break;
    } else {
      if (schd.io) {
        if (commrank == 0) {
          Dets.resize(DetsSize);
          for (int i = 0; i < DetsSize; i++) Dets[i] = SHMDets[i];
        }
        writeVariationalResult(iter, ci, Dets, sparseHam, E0, true, schd,
                               helper2);
      }
      Dets.clear();
    }

    if (schd.outputlevel > 0)
      pout << format(
                  "###########################################      %10.2f ") %
                  (getTime() - startofCalc)
           << endl;
  }

  boost::interprocess::shared_memory_object::remove(shciDetsCI.c_str());
  boost::interprocess::shared_memory_object::remove(shciHelper.c_str());

  pout << endl;
  pout << "Variational calculation result" << endl;
  pout << format("%4s %18s  %10s\n") % ("Root") % ("Energy") % ("Time(s)");
  for (int i = 0; i < E0.size(); i++) {
    E0[i] += coreEbkp;
    pout << format("%4i  %18.10f  %10.2f\n") % (i) % (E0[i]) %
                (getTime() - startofCalc);
  }
  pout << endl;
  coreE = coreEbkp;
  return E0;
}

void SHCIbasics::writeVariationalResult(
    int iter, vector<MatrixXx> &ci, vector<Determinant> &Dets,
    vector<vector<int>> &connections, vector<vector<size_t>> &orbdifference,
    vector<vector<CItype>> &Helements, vector<double> &E0, bool converged,
    schedule &schd, std::map<HalfDet, std::vector<int>> &BetaN,
    std::map<HalfDet, std::vector<int>> &AlphaNm1) {
#ifndef SERIAL
  boost::mpi::communicator world;
#endif

  if (schd.outputlevel > 0)
    pout << format("#Begin writing variational wf %29.2f\n") %
                (getTime() - startofCalc);

  {
    char file[5000];
    sprintf(file, "%s/%d-variational.bkp", schd.prefix[0].c_str(), commrank);
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << iter << Dets;
    save << ci;
    save << E0;
    save << converged;
    ofs.close();
  }

  if (converged) {
    char file[5000];
    sprintf(file, "%s/%d-hamiltonian.bkp", schd.prefix[0].c_str(), commrank);
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << connections << Helements << orbdifference;
  }

  {
    char file[5000];
    sprintf(file, "%s/%d-helpers.bkp", schd.prefix[0].c_str(), commrank);
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << BetaN << AlphaNm1;
  }

  if (schd.outputlevel > 0)
    pout << format("#End   writing variational wf %29.2f\n") %
                (getTime() - startofCalc);
}

void SHCIbasics::readVariationalResult(
    int &iter, vector<MatrixXx> &ci, vector<Determinant> &Dets,
    vector<vector<int>> &connections, vector<vector<size_t>> &orbdifference,
    vector<vector<CItype>> &Helements, vector<double> &E0, bool &converged,
    schedule &schd, std::map<HalfDet, std::vector<int>> &BetaN,
    std::map<HalfDet, std::vector<int>> &AlphaNm1) {
#ifndef SERIAL
  boost::mpi::communicator world;
#endif

  if (schd.outputlevel > 0)
    pout << format("#Begin reading variational wf %29.2f\n") %
                (getTime() - startofCalc);

  {
    char file[5000];
    sprintf(file, "%s/%d-variational.bkp", schd.prefix[0].c_str(), commrank);
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);

    load >> iter >> Dets;
    ci.resize(1, MatrixXx(Dets.size(), 1));

    load >> ci;
    load >> E0;
    load >> converged;
    if (schd.outputlevel > 0)
      pout << "Load converged: " << converged << endl;
    if (schd.onlyperturbative) {
      ifs.close();
      return;
    }
  }

  char file[5000];
  sprintf(file, "%s/%d-hamiltonian.bkp", schd.prefix[0].c_str(), commrank);
  std::ifstream ifs(file, std::ios::binary);
  boost::archive::binary_iarchive load(ifs);
  load >> connections >> Helements >> orbdifference;

  {
    char file[5000];
    sprintf(file, "%s/%d-helpers.bkp", schd.prefix[0].c_str(), commrank);
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    load >> BetaN >> AlphaNm1;
    ifs.close();
  }

  if (schd.outputlevel > 0)
    pout << format("#End   reading variational wf %29.2f\n") %
                (getTime() - startofCalc);
}

void SHCIbasics::writeVariationalResult(
    int iter, vector<MatrixXx> &ci, vector<Determinant> &Dets,
    SHCImakeHamiltonian::SparseHam &sparseHam, vector<double> &E0,
    bool converged, schedule &schd, SHCImakeHamiltonian::HamHelpers2 &helper2) {
#ifndef SERIAL
  boost::mpi::communicator world;
#endif

  if (schd.outputlevel > 0)
    pout << format("#Begin writing variational wf %29.2f\n") %
                (getTime() - startofCalc);

  {
    char file[5000];
    sprintf(file, "%s/%d-variational.bkp", schd.prefix[0].c_str(), commrank);
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << iter << Dets;
    save << ci;
    save << E0;
    save << converged;
    ofs.close();
  }

  if (converged) {
    char file[5000];
    sprintf(file, "%s/%d-hamiltonian.bkp", schd.prefix[0].c_str(), commrank);
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << sparseHam.connections << sparseHam.Helements
         << sparseHam.orbDifference;
  }

  if (commrank == 0) {
    char file[5000];
    sprintf(file, "%s/%d-helpers.bkp", schd.prefix[0].c_str(), commrank);
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << helper2.AlphaMajorToBeta << helper2.AlphaMajorToDet
         << helper2.BetaMajorToAlpha << helper2.BetaMajorToDet
         << helper2.SinglesFromAlpha << helper2.SinglesFromBeta << helper2.BetaN
         << helper2.AlphaN;
  }

  if (schd.outputlevel > 0)
    pout << format("#End   writing variational wf %29.2f\n") %
                (getTime() - startofCalc);
}

void SHCIbasics::readVariationalResult(
    int &iter, vector<MatrixXx> &ci, vector<Determinant> &Dets,
    SHCImakeHamiltonian::SparseHam &sparseHam, vector<double> &E0,
    bool &converged, schedule &schd,
    SHCImakeHamiltonian::HamHelpers2 &helper2) {
#ifndef SERIAL
  boost::mpi::communicator world;
#endif

  if (schd.outputlevel > 0)
    pout << format("#Begin reading variational wf %29.2f\n") %
                (getTime() - startofCalc);

  {
    char file[5000];
    sprintf(file, "%s/%d-variational.bkp", schd.prefix[0].c_str(), commrank);
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);

    std::vector<Determinant> sorted;
    load >> iter >> Dets;  // >>sorted ;
    load >> ci;
    load >> E0;
    load >> converged;
    if (schd.outputlevel > 0)
      pout << "Load converged: " << converged << endl;
    if (schd.onlyperturbative) {
      ifs.close();
      return;
    }
  }

  /*
  if (schd.DavidsonType != DIRECT)
  {
    char file [5000];
    sprintf (file, "%s/%d-hamiltonian.bkp", schd.prefix[0].c_str(), commrank );
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    load >> sparseHam.connections >> sparseHam.Helements
  >>sparseHam.orbDifference;
  }
  */

  if (commrank == 0) {
    char file[5000];
    sprintf(file, "%s/%d-helpers.bkp", schd.prefix[0].c_str(), commrank);
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    load >> helper2.AlphaMajorToBeta >> helper2.AlphaMajorToDet >>
        helper2.BetaMajorToAlpha >> helper2.BetaMajorToDet >>
        helper2.SinglesFromAlpha >> helper2.SinglesFromBeta >> helper2.BetaN >>
        helper2.AlphaN;

    ifs.close();
  }

  if (schd.outputlevel > 0)
    pout << format("#End   reading variational wf %29.2f\n") %
                (getTime() - startofCalc);
}

void SHCIbasics::writeHelperIntermediate(
    std::map<HalfDet, int> &BetaN, std::map<HalfDet, int> &AlphaN,
    std::map<HalfDet, vector<int>> &BetaNm1,
    std::map<HalfDet, vector<int>> &AlphaNm1, schedule &schd, int iter) {
#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  if (commrank == 0) {
    char file[5000];
    sprintf(file, "%s/%d-helpers-Intermediate-%d.bkp", schd.prefix[0].c_str(),
            commrank, iter);
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << BetaN << AlphaN << BetaNm1 << AlphaNm1;
  }
}

void SHCIbasics::readHelperIntermediate(
    std::map<HalfDet, int> &BetaN, std::map<HalfDet, int> &AlphaN,
    std::map<HalfDet, vector<int>> &BetaNm1,
    std::map<HalfDet, vector<int>> &AlphaNm1, schedule &schd, int iter) {
#ifndef SERIAL
  boost::mpi::communicator world;
#endif

  if (commrank == 0) {
    char file[5000];
    sprintf(file, "%s/%d-helpers-Intermediate-%d.bkp", schd.prefix[0].c_str(),
            commrank, iter);
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    load >> BetaN >> AlphaN >> BetaNm1 >> AlphaNm1;
    ifs.close();
  }
}
