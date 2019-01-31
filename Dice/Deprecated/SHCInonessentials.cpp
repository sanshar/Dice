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

#include "SHCInonessentials."
#include "Dice/Davidson.h"
#include "Dice/Hmult.h"
#include "Dice/SHCIbasics.h"
#include "boost/format.hpp"
#include "Dice/Utils/input.h"
#include "Dice/Utils/integral.h"
#include "math.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <fstream>
#include <map>
#include <tuple>
#include <vector>

#ifndef SERIAL
#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#endif
#include "Dice/Utils/communicate.h"

using namespace std;
using namespace Eigen;
using namespace boost;

void SHCIbasics::DoPerturbativeDeterministicOffdiagonal(
    vector<Determinant> &Dets, MatrixXx &ci1, double &E01, MatrixXx &ci2,
    double &E02, oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2HB,
    vector<int> &irrep, schedule &schd, double coreE, int nelec, int root,
    CItype &EPT1, CItype &EPT2, CItype &EPT12, std::vector<MatrixXx> &spinRDM) {

#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  int norbs = Determinant::norbs;
  std::vector<Determinant> SortedDets = Dets;
  std::sort(SortedDets.begin(), SortedDets.end());

  double energyEN = 0.0;

  std::vector<StitchDEH> uniqueDEH(num_thrds);
  std::vector<std::vector<std::vector<vector<Determinant>>>> hashedDetBeforeMPI(
      commsize, std::vector<std::vector<vector<Determinant>>>(num_thrds));
  std::vector<std::vector<std::vector<vector<Determinant>>>> hashedDetAfterMPI(
      commsize, std::vector<std::vector<vector<Determinant>>>(num_thrds));
  std::vector<std::vector<std::vector<vector<CItype>>>> hashedNumBeforeMPI(
      commsize, std::vector<std::vector<vector<CItype>>>(num_thrds));
  std::vector<std::vector<std::vector<vector<CItype>>>> hashedNumAfterMPI(
      commsize, std::vector<std::vector<vector<CItype>>>(num_thrds));
  std::vector<std::vector<std::vector<vector<CItype>>>> hashedNum2BeforeMPI(
      commsize, std::vector<std::vector<vector<CItype>>>(num_thrds));
  std::vector<std::vector<std::vector<vector<CItype>>>> hashedNum2AfterMPI(
      commsize, std::vector<std::vector<vector<CItype>>>(num_thrds));
  std::vector<std::vector<std::vector<vector<double>>>> hashedEnergyBeforeMPI(
      commsize, std::vector<std::vector<vector<double>>>(num_thrds));
  std::vector<std::vector<std::vector<vector<double>>>> hashedEnergyAfterMPI(
      commsize, std::vector<std::vector<vector<double>>>(num_thrds));
  CItype totalPT1 = 0.0, totalPT2 = 0., totalPT12 = 0.;
  int ntries = 1;

  for (int i = 0; i < Dets.size(); i++) {
    if (i % (omp_get_num_threads() * commsize) !=
        commrank * omp_get_num_threads() + omp_get_thread_num()) {
      continue;
    }
    SHCIgetdeterminants::getDeterminantsDeterministicPTWithSOC(
        Dets[i], i, abs(schd.epsilon2 / ci1(i, 0)), ci1(i, 0),
        abs(schd.epsilon2 / ci2(i, 0)), ci2(i, 0), I1, I2, I2HB, irrep, coreE,
        *uniqueDEH[omp_get_thread_num()].Det,
        *uniqueDEH[omp_get_thread_num()].Num,
        *uniqueDEH[omp_get_thread_num()].Num2,
        *uniqueDEH[omp_get_thread_num()].Energy, schd, nelec);
  }

  uniqueDEH[omp_get_thread_num()].MergeSortAndRemoveDuplicates();
  uniqueDEH[omp_get_thread_num()].RemoveDetsPresentIn(SortedDets);

  if (commsize > 1 || num_thrds > 1) {
    StitchDEH uniqueDEH_afterMPI;
    if (schd.DoRDM || schd.doResponse)
      uniqueDEH_afterMPI.extra_info = true;

    for (int proc = 0; proc < commsize; proc++) {
      hashedDetBeforeMPI[proc][omp_get_thread_num()].resize(num_thrds);
      hashedNumBeforeMPI[proc][omp_get_thread_num()].resize(num_thrds);
      hashedNum2BeforeMPI[proc][omp_get_thread_num()].resize(num_thrds);
      hashedEnergyBeforeMPI[proc][omp_get_thread_num()].resize(num_thrds);
    }

    if (omp_get_thread_num() == 0) {
      ntries = uniqueDEH[omp_get_thread_num()].Det->size() * DetLen * 2 *
                   omp_get_num_threads() / 268435400 +
               1;
      if (commsize == 1)
        ntries = 1;
#ifndef SERIAL
      mpi::broadcast(world, ntries, 0);
#endif
    }

    size_t batchsize = uniqueDEH[omp_get_thread_num()].Det->size() / ntries;
    // ntries = 1;
    for (int tries = 0; tries < ntries; tries++) {

      size_t start = (ntries - 1 - tries) * batchsize;
      size_t end = tries == 0 ? uniqueDEH[omp_get_thread_num()].Det->size()
                              : (ntries - tries) * batchsize;
      for (size_t j = start; j < end; j++) {
        size_t lOrder = uniqueDEH[omp_get_thread_num()].Det->at(j).getHash();
        size_t procThrd = lOrder % (commsize * num_thrds);
        int proc = abs(procThrd / num_thrds), thrd = abs(procThrd % num_thrds);
        hashedDetBeforeMPI[proc][omp_get_thread_num()][thrd].push_back(
            uniqueDEH[omp_get_thread_num()].Det->at(j));
        hashedNumBeforeMPI[proc][omp_get_thread_num()][thrd].push_back(
            uniqueDEH[omp_get_thread_num()].Num->at(j));
        hashedNum2BeforeMPI[proc][omp_get_thread_num()][thrd].push_back(
            uniqueDEH[omp_get_thread_num()].Num2->at(j));
        hashedEnergyBeforeMPI[proc][omp_get_thread_num()][thrd].push_back(
            uniqueDEH[omp_get_thread_num()].Energy->at(j));
      }

      uniqueDEH[omp_get_thread_num()].resize(start);

#ifndef SERIAL
      mpi::all_to_all(world, hashedDetBeforeMPI, hashedDetAfterMPI);
      mpi::all_to_all(world, hashedNumBeforeMPI, hashedNumAfterMPI);
      mpi::all_to_all(world, hashedNum2BeforeMPI, hashedNum2AfterMPI);
      mpi::all_to_all(world, hashedEnergyBeforeMPI, hashedEnergyAfterMPI);
#else
      hashedDetAfterMPI = hashedDetBeforeMPI;
      hashedNumAfterMPI = hashedNumBeforeMPI;
      hashedNum2AfterMPI = hashedNum2BeforeMPI;
      // hashedpresentAfterMPI = hashedpresentBeforeMPI;
      hashedEnergyAfterMPI = hashedEnergyBeforeMPI;
#endif

      for (int proc = 0; proc < commsize; proc++) {
        for (int thrd = 0; thrd < num_thrds; thrd++) {
          hashedDetBeforeMPI[proc][thrd][omp_get_thread_num()].clear();
          hashedNumBeforeMPI[proc][thrd][omp_get_thread_num()].clear();
          hashedNum2BeforeMPI[proc][thrd][omp_get_thread_num()].clear();
          hashedEnergyBeforeMPI[proc][thrd][omp_get_thread_num()].clear();
        }
      }

      for (int proc = 0; proc < commsize; proc++) {
        for (int thrd = 0; thrd < num_thrds; thrd++) {

          for (int j = 0;
               j < hashedDetAfterMPI[proc][thrd][omp_get_thread_num()].size();
               j++) {
            uniqueDEH_afterMPI.Det->push_back(
                hashedDetAfterMPI[proc][thrd][omp_get_thread_num()].at(j));
            uniqueDEH_afterMPI.Num->push_back(
                hashedNumAfterMPI[proc][thrd][omp_get_thread_num()].at(j));
            uniqueDEH_afterMPI.Num2->push_back(
                hashedNum2AfterMPI[proc][thrd][omp_get_thread_num()].at(j));
            uniqueDEH_afterMPI.Energy->push_back(
                hashedEnergyAfterMPI[proc][thrd][omp_get_thread_num()].at(j));
          }
          hashedDetAfterMPI[proc][thrd][omp_get_thread_num()].clear();
          hashedNumAfterMPI[proc][thrd][omp_get_thread_num()].clear();
          hashedNum2AfterMPI[proc][thrd][omp_get_thread_num()].clear();
          hashedEnergyAfterMPI[proc][thrd][omp_get_thread_num()].clear();
        }
      }
    }

    *uniqueDEH[omp_get_thread_num()].Det = *uniqueDEH_afterMPI.Det;
    *uniqueDEH[omp_get_thread_num()].Num = *uniqueDEH_afterMPI.Num;
    *uniqueDEH[omp_get_thread_num()].Num2 = *uniqueDEH_afterMPI.Num2;
    *uniqueDEH[omp_get_thread_num()].Energy = *uniqueDEH_afterMPI.Energy;
    uniqueDEH_afterMPI.clear();
    uniqueDEH[omp_get_thread_num()].MergeSortAndRemoveDuplicates();
  }

  vector<Determinant> &hasHEDDets = *uniqueDEH[omp_get_thread_num()].Det;
  vector<CItype> &hasHEDNumerator = *uniqueDEH[omp_get_thread_num()].Num;
  vector<CItype> &hasHEDNumerator2 = *uniqueDEH[omp_get_thread_num()].Num2;
  vector<double> &hasHEDEnergy = *uniqueDEH[omp_get_thread_num()].Energy;

  CItype PTEnergy1 = 0.0, PTEnergy2 = 0.0, PTEnergy12 = 0.0;

  for (size_t i = 0; i < hasHEDDets.size(); i++) {
    PTEnergy1 += pow(abs(hasHEDNumerator[i]), 2) / (E01 - hasHEDEnergy[i]);
    PTEnergy12 += 0.5 * (conj(hasHEDNumerator[i]) * hasHEDNumerator2[i] /
                             (E01 - hasHEDEnergy[i]) +
                         conj(hasHEDNumerator[i]) * hasHEDNumerator2[i] /
                             (E02 - hasHEDEnergy[i]));
    PTEnergy2 += pow(abs(hasHEDNumerator2[i]), 2) / (E02 - hasHEDEnergy[i]);
  }

  totalPT1 += PTEnergy1;
  totalPT12 += PTEnergy12;
  totalPT2 += PTEnergy2;

  EPT1 = 0.0;
  EPT2 = 0.0;
  EPT12 = 0.0;
#ifndef SERIAL
  mpi::all_reduce(world, totalPT1, EPT1, std::plus<CItype>());
  mpi::all_reduce(world, totalPT2, EPT2, std::plus<CItype>());
  mpi::all_reduce(world, totalPT12, EPT12, std::plus<CItype>());
#else
  EPT1 = totalPT1;
  EPT2 = totalPT2;
  EPT12 = totalPT12;
#endif

  if (schd.doGtensor) { // DON'T PERFORM doGtensor

    if (commrank != 0) {
      spinRDM[0].setZero(spinRDM[0].rows(), spinRDM[0].cols());
      spinRDM[1].setZero(spinRDM[1].rows(), spinRDM[1].cols());
      spinRDM[2].setZero(spinRDM[2].rows(), spinRDM[2].cols());
    }

    vector<vector<MatrixXx>> spinRDM_thrd(num_thrds, vector<MatrixXx>(3));
#pragma omp parallel
    {
      for (int thrd = 0; thrd < num_thrds; thrd++) {
        if (thrd != omp_get_thread_num())
          continue;

        spinRDM_thrd[thrd][0].setZero(spinRDM[0].rows(), spinRDM[0].cols());
        spinRDM_thrd[thrd][1].setZero(spinRDM[1].rows(), spinRDM[1].cols());
        spinRDM_thrd[thrd][2].setZero(spinRDM[2].rows(), spinRDM[2].cols());

        vector<Determinant> &hasHEDDets = *uniqueDEH[thrd].Det;
        vector<CItype> &hasHEDNumerator = *uniqueDEH[thrd].Num;
        vector<CItype> &hasHEDNumerator2 = *uniqueDEH[thrd].Num2;
        vector<double> &hasHEDEnergy = *uniqueDEH[thrd].Energy;

        for (int x = 0; x < Dets.size(); x++) {
          Determinant &d = Dets[x];

          vector<int> closed(nelec, 0);
          vector<int> open(norbs - nelec, 0);
          d.getOpenClosed(open, closed);
          int nclosed = nelec;
          int nopen = norbs - nclosed;

          for (int ia = 0; ia < nopen * nclosed; ia++) {
            int i = ia / nopen, a = ia % nopen;

            Determinant di = d;
            di.setocc(open[a], true);
            di.setocc(closed[i], false);

            auto lower =
                std::lower_bound(hasHEDDets.begin(), hasHEDDets.end(), di);
            // map<Determinant, int>::iterator it = SortedDets.find(di);
            if (di == *lower) {
              double sgn = 1.0;
              d.parity(min(open[a], closed[i]), max(open[a], closed[i]), sgn);
              int y = distance(hasHEDDets.begin(), lower);
              // states "a" and "b"
              //"0" order and "1" order corrections
              // in all 4 states "0a" "1a"  "0b"  "1b"
              CItype complex1 =
                  1.0 * (conj(hasHEDNumerator[y]) * ci1(x, 0) /
                         (E01 - hasHEDEnergy[y]) * sgn); //<1a|v|0a>
              CItype complex2 =
                  1.0 * (conj(hasHEDNumerator2[y]) * ci2(x, 0) /
                         (E02 - hasHEDEnergy[y]) * sgn); //<1b|v|0b>
              CItype complex12 =
                  1.0 * (conj(hasHEDNumerator[y]) * ci2(x, 0) /
                         (E01 - hasHEDEnergy[y]) * sgn); //<1a|v|0b>
              CItype complex12b =
                  1.0 * (conj(ci1(x, 0)) * hasHEDNumerator2[y] /
                         (E02 - hasHEDEnergy[y]) * sgn); //<0a|v|1b>

              spinRDM_thrd[thrd][0](open[a], closed[i]) += complex1;
              spinRDM_thrd[thrd][1](open[a], closed[i]) += complex2;
              spinRDM_thrd[thrd][2](open[a], closed[i]) += complex12;

              spinRDM_thrd[thrd][0](closed[i], open[a]) += conj(complex1);
              spinRDM_thrd[thrd][1](closed[i], open[a]) += conj(complex2);
              spinRDM_thrd[thrd][2](closed[i], open[a]) += complex12b;

              /*
              spinRDM[0](open[a], closed[i]) += complex1;
              spinRDM[1](open[a], closed[i]) += complex2;
              spinRDM[2](open[a], closed[i]) += complex12;

              spinRDM[0](closed[i], open[a]) += conj(complex1);
              spinRDM[1](closed[i], open[a]) += conj(complex2);
              spinRDM[2](closed[i], open[a]) += complex12b;
              */
            }
          }
        }
      }
    }

    for (int thrd = 0; thrd < num_thrds; thrd++) {
      spinRDM[0] += spinRDM_thrd[thrd][0];
      spinRDM[1] += spinRDM_thrd[thrd][1];
      spinRDM[2] += spinRDM_thrd[thrd][2];
    }

#ifndef SERIAL
#ifndef Complex
    MPI_Allreduce(MPI_IN_PLACE, &spinRDM[0](0, 0),
                  spinRDM[0].rows() * spinRDM[0].cols(), MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &spinRDM[1](0, 0),
                  spinRDM[1].rows() * spinRDM[1].cols(), MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &spinRDM[2](0, 0),
                  spinRDM[2].rows() * spinRDM[2].cols(), MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    // boost::mpi::all_reduce(world, boost::mpi::inplace_t<double*
    // >(&spinRDM[0](0,0)), spinRDM[0].rows()*spinRDM[0].cols(),
    // std::plus<double>()); boost::mpi::all_reduce(world,
    // boost::mpi::inplace_t<double* >(&spinRDM[1](0,0)),
    // spinRDM[1].rows()*spinRDM[1].cols(), std::plus<double>());
    // boost::mpi::all_reduce(world, boost::mpi::inplace_t<double*
    // >(&spinRDM[2](0,0)), spinRDM[2].rows()*spinRDM[2].cols(),
    // std::plus<double>());
#else
    MPI_Allreduce(MPI_IN_PLACE, &spinRDM[0](0, 0),
                  2 * spinRDM[0].rows() * spinRDM[0].cols(), MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &spinRDM[1](0, 0),
                  2 * spinRDM[1].rows() * spinRDM[1].cols(), MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &spinRDM[2](0, 0),
                  2 * spinRDM[2].rows() * spinRDM[2].cols(), MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    // boost::mpi::all_reduce(world, boost::mpi::inplace_t<std::complex<double>*
    // >(&spinRDM[0](0,0)), spinRDM[0].rows()*spinRDM[0].cols(), sumComplex);
    // boost::mpi::all_reduce(world, boost::mpi::inplace_t<std::complex<double>*
    // >(&spinRDM[1](0,0)), spinRDM[1].rows()*spinRDM[1].cols(), sumComplex);
    // boost::mpi::all_reduce(world, boost::mpi::inplace_t<std::complex<double>*
    // >(&spinRDM[2](0,0)), spinRDM[2].rows()*spinRDM[2].cols(), sumComplex);
#endif
#endif
  }
}

void SHCImakeHamiltonian::MakeHfromHelpers(
    std::map<HalfDet, std::vector<int>> &BetaN,
    std::map<HalfDet, std::vector<int>> &AlphaNm1,
    std::vector<Determinant> &Dets, int StartIndex,
    std::vector<std::vector<int>> &connections,
    std::vector<std::vector<CItype>> &Helements, int Norbs, oneInt &I1,
    twoInt &I2, double &coreE, std::vector<std::vector<size_t>> &orbDifference,
    bool DoRDM) {

#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  int nprocs = commsize, proc = commrank;

  size_t norbs = Norbs;

  for (size_t k = StartIndex; k < connections.size(); k++) {
    if (k % (nprocs) != proc)
      continue;
    connections[k].push_back(k);
    CItype hij = Dets[k].Energy(I1, I2, coreE);
    if (Determinant::Trev != 0)
      updateHijForTReversal(hij, Dets[k], Dets[k], I1, I2, coreE);
    Helements[k].push_back(hij);
    if (DoRDM)
      orbDifference[k].push_back(0);
  }

  std::map<HalfDet, std::vector<int>>::iterator ita = BetaN.begin();
  int index = 0;
  for (; ita != BetaN.end(); ita++) {
    std::vector<int> &detIndex = ita->second;
    int localStart = detIndex.size();
    for (int j = 0; j < detIndex.size(); j++)
      if (detIndex[j] >= StartIndex) {
        localStart = j;
        break;
      }

    for (int k = localStart; k < detIndex.size(); k++) {
      if (detIndex[k] % (nprocs) != proc)
        continue;

      for (int j = 0; j < k; j++) {
        size_t J = detIndex[j];
        size_t K = detIndex[k];
        if (Dets[J].connected(Dets[K]) ||
            (Determinant::Trev != 0 &&
             Dets[J].connectedToFlipAlphaBeta(Dets[K]))) {

          size_t orbDiff;
          CItype hij = Hij(Dets[J], Dets[K], I1, I2, coreE, orbDiff);
          if (Determinant::Trev != 0)
            updateHijForTReversal(hij, Dets[J], Dets[K], I1, I2, coreE);

          if (abs(hij) < 1.e-10)
            continue;
          Helements[K].push_back(hij);
          connections[K].push_back(J);

          if (DoRDM)
            orbDifference[K].push_back(orbDiff);
        }
      }
    }
  }

  ita = AlphaNm1.begin();
  index = 0;
  for (; ita != AlphaNm1.end(); ita++) {
    std::vector<int> &detIndex = ita->second;
    int localStart = detIndex.size();
    for (int j = 0; j < detIndex.size(); j++)
      if (detIndex[j] >= StartIndex) {
        localStart = j;
        break;
      }

    for (int k = localStart; k < detIndex.size(); k++) {
      if (detIndex[k] % (nprocs) != proc)
        continue;

      for (int j = 0; j < k; j++) {
        size_t J = detIndex[j];
        size_t K = detIndex[k];
        if (Dets[J].connected(Dets[K]) ||
            (Determinant::Trev != 0 &&
             Dets[J].connectedToFlipAlphaBeta(Dets[K]))) {
          if (find(connections[K].begin(), connections[K].end(), J) ==
              connections[K].end()) {
            size_t orbDiff;
            CItype hij = Hij(Dets[J], Dets[K], I1, I2, coreE, orbDiff);
            if (Determinant::Trev != 0)
              updateHijForTReversal(hij, Dets[J], Dets[K], I1, I2, coreE);

            if (abs(hij) < 1.e-10)
              continue;
            connections[K].push_back(J);
            Helements[K].push_back(hij);

            if (DoRDM)
              orbDifference[K].push_back(orbDiff);
          }
        }
      }
    }
  }
}

void SHCImakeHamiltonian::PopulateHelperLists(
    std::map<HalfDet, std::vector<int>> &BetaN,
    std::map<HalfDet, std::vector<int>> &AlphaNm1,
    std::vector<Determinant> &Dets, int StartIndex) {
  for (int i = StartIndex; i < Dets.size(); i++) {
    HalfDet da = Dets[i].getAlpha(), db = Dets[i].getBeta();

    BetaN[db].push_back(i);

    int norbs = 64 * DetLen;
    std::vector<int> closeda(norbs / 2); //, closedb(norbs);
    int ncloseda = da.getClosed(closeda);
    // int nclosedb = db.getClosed(closedb);
    for (int j = 0; j < ncloseda; j++) {
      da.setocc(closeda[j], false);
      AlphaNm1[da].push_back(i);
      da.setocc(closeda[j], true);
    }

    // When Treversal symmetry is used
    if (Determinant::Trev != 0 && Dets[i].hasUnpairedElectrons()) {
      BetaN[da].push_back(i);

      std::vector<int> closedb(norbs / 2); //, closedb(norbs);
      int nclosedb = db.getClosed(closedb);
      for (int j = 0; j < nclosedb; j++) {
        db.setocc(closedb[j], false);
        AlphaNm1[db].push_back(i);
        db.setocc(closedb[j], true);
      }
    }
  }
}

void SHCImakeHamiltonian::MakeHfromHelpers(
    int *&BetaVecLen, vector<int *> &BetaVec, int *&AlphaVecLen,
    vector<int *> &AlphaVec, Determinant *Dets, int StartIndex,
    std::vector<std::vector<int>> &connections,
    std::vector<std::vector<CItype>> &Helements, int Norbs, oneInt &I1,
    twoInt &I2, double &coreE, std::vector<std::vector<size_t>> &orbDifference,
    bool DoRDM) {

  int proc = 0, nprocs = 1;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#endif

  size_t norbs = Norbs;

  for (size_t k = StartIndex; k < connections.size(); k++) {
    if (k % (nprocs) != proc)
      continue;
    connections[k].push_back(k);
    CItype hij = Dets[k].Energy(I1, I2, coreE);
    if (Determinant::Trev != 0)
      updateHijForTReversal(hij, Dets[k], Dets[k], I1, I2, coreE);
    Helements[k].push_back(hij);
    if (DoRDM)
      orbDifference[k].push_back(0);
  }

  int index = 0;
  for (int i = 0; i < BetaVec.size(); i++) {
    int *detIndex = BetaVec[i];
    int localStart = BetaVecLen[i];
    for (int j = 0; j < BetaVecLen[i]; j++)
      if (detIndex[j] >= StartIndex) {
        localStart = j;
        break;
      }

    for (int k = localStart; k < BetaVecLen[i]; k++) {

      if (detIndex[k] % (nprocs) != proc)
        continue;

      for (int j = 0; j < k; j++) {
        size_t J = detIndex[j];
        size_t K = detIndex[k];
        size_t orbDiff;
        CItype hij = Hij(Dets[J], Dets[K], I1, I2, coreE, orbDiff);
        if (Determinant::Trev != 0)
          updateHijForTReversal(hij, Dets[J], Dets[K], I1, I2, coreE);

        if (abs(hij) < 1.e-10)
          continue;
        Helements[K].push_back(hij);
        connections[K].push_back(J);

        if (DoRDM)
          orbDifference[K].push_back(orbDiff);
      }
    }

    index++;
  }

  index = 0;
  for (int i = 0; i < AlphaVec.size(); i++) {
    int *detIndex = AlphaVec[i];
    int localStart = AlphaVecLen[i];
    for (int j = 0; j < AlphaVecLen[i]; j++)
      if (detIndex[j] >= StartIndex) {
        localStart = j;
        break;
      }

    for (int k = localStart; k < AlphaVecLen[i]; k++) {
      if (detIndex[k] % (nprocs) != proc)
        continue;

      for (int j = 0; j < k; j++) {
        size_t J = detIndex[j];
        size_t K = detIndex[k];
        if (Dets[J].connected(Dets[K]) ||
            (Determinant::Trev != 0 &&
             Dets[J].connectedToFlipAlphaBeta(Dets[K]))) {
          if (find(connections[K].begin(), connections[K].end(), J) ==
              connections[K].end()) {
            size_t orbDiff;
            CItype hij = Hij(Dets[J], Dets[K], I1, I2, coreE, orbDiff);
            if (Determinant::Trev != 0)
              updateHijForTReversal(hij, Dets[J], Dets[K], I1, I2, coreE);

            if (abs(hij) < 1.e-10)
              continue;
            connections[K].push_back(J);
            Helements[K].push_back(hij);

            if (DoRDM)
              orbDifference[K].push_back(orbDiff);
          }
        }
      }
    }
    index++;
  }
}

void SHCImakeHamiltonian::MakeHfromHelpers2(
    vector<vector<int>> &AlphaMajorToBeta, vector<vector<int>> &AlphaMajorToDet,
    vector<vector<int>> &BetaMajorToAlpha, vector<vector<int>> &BetaMajorToDet,
    vector<vector<int>> &SinglesFromAlpha, vector<vector<int>> &SinglesFromBeta,
    std::vector<Determinant> &Dets, int StartIndex,
    std::vector<std::vector<int>> &connections,
    std::vector<std::vector<CItype>> &Helements, int Norbs, oneInt &I1,
    twoInt &I2, double &coreE, std::vector<std::vector<size_t>> &orbDifference,
    bool DoRDM) {

  int proc = 0, nprocs = 1;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#endif

  size_t norbs = Norbs;

  // diagonal element
  for (size_t k = StartIndex; k < Dets.size(); k++) {
    if (k % (nprocs) != proc)
      continue;
    connections.push_back(vector<int>(1, k));
    // connections[k].push_back(k);
    CItype hij = Dets[k].Energy(I1, I2, coreE);
    if (Determinant::Trev != 0)
      updateHijForTReversal(hij, Dets[k], Dets[k], I1, I2, coreE);
    Helements.push_back(vector<CItype>(1, hij));
    // Helements[k].push_back(hij);
    if (DoRDM)
      orbDifference.push_back(vector<size_t>(1, 0));
  }

  // alpha-beta excitation
  for (int i = 0; i < AlphaMajorToBeta.size(); i++) {
    for (int ii = 0; ii < AlphaMajorToBeta[i].size(); ii++) {
      if (AlphaMajorToDet[i][ii] < StartIndex ||
          AlphaMajorToDet[i][ii] % nprocs != proc)
        continue;
      int Astring = i, Bstring = AlphaMajorToBeta[i][ii],
          DetI = AlphaMajorToDet[i][ii];

      // singles from Astring
      for (int j = 0; j < SinglesFromAlpha[Astring].size(); j++) {
        int Asingle = SinglesFromAlpha[Astring][j];

        int index = binarySearch(&BetaMajorToAlpha[Bstring][0], 0,
                                 BetaMajorToAlpha[Bstring].size() - 1, Asingle);
        if (index != -1) {
          int DetJ = BetaMajorToDet[Bstring][index];

          if (DetJ < DetI) {
            size_t orbDiff;
            CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
            if (abs(hij) > 1.e-10) {
              connections[DetI / nprocs].push_back(DetJ);
              Helements[DetI / nprocs].push_back(hij);
              if (DoRDM)
                orbDifference[DetI / nprocs].push_back(orbDiff);
            }
          }
        }

        int SearchStartIndex = 0;
        for (int k = 0; k < SinglesFromBeta[Bstring].size(); k++) {
          int &Bsingle = SinglesFromBeta[Bstring][k];

          if (SearchStartIndex >= AlphaMajorToBeta[Asingle].size())
            break;

          int index = SearchStartIndex;
          for (; index < AlphaMajorToBeta[Asingle].size(); index++)
            if (AlphaMajorToBeta[Asingle][index] >= Bsingle)
              break;
          SearchStartIndex = index;
          if (index < AlphaMajorToBeta[Asingle].size() &&
              AlphaMajorToBeta[Asingle][index] == Bsingle) {
            int DetJ = AlphaMajorToDet[Asingle][index];

            // int index = binarySearch(&AlphaMajorToBeta[Asingle][0],
            // SearchStartIndex, AlphaMajorToBeta[Asingle].size()-1, Bsingle);
            // if (index != -1 ) { SearchStartIndex = index; int DetJ =
            // AlphaMajorToDet[Asingle][index];

            // auto itb =
            // lower_bound(AlphaMajorToBeta[Asingle].begin()+SearchStartIndex,
            // AlphaMajorToBeta[Asingle].end(), Bsingle); if (itb !=
            // AlphaMajorToBeta[Asingle].end() && *itb == Bsingle) { int DetJ =
            // AlphaMajorToDet[Asingle][SearchStartIndex];

            if (DetJ < DetI) { // single beta, single alpha
              size_t orbDiff;
              CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
              if (abs(hij) > 1.e-10) {
                connections[DetI / nprocs].push_back(DetJ);
                Helements[DetI / nprocs].push_back(hij);
                if (DoRDM)
                  orbDifference[DetI / nprocs].push_back(orbDiff);
              }
            } // DetJ <Det I
          }   //*itb == Bsingle
        }     // k 0->SinglesFromBeta
      }       // j singles fromAlpha

      // singles from Bstring
      for (int j = 0; j < SinglesFromBeta[Bstring].size(); j++) {
        int Bsingle = SinglesFromBeta[Bstring][j];

        int index = binarySearch(&AlphaMajorToBeta[Astring][0], 0,
                                 AlphaMajorToBeta[Astring].size() - 1, Bsingle);
        if (index != -1) {
          int DetJ = AlphaMajorToDet[Astring][index];
          // auto itb = lower_bound(AlphaMajorToBeta[Astring].begin(),
          // AlphaMajorToBeta[Astring].end(), Bsingle); if (itb !=
          // AlphaMajorToBeta[Astring].end() && *itb == Bsingle) { int DetJ =
          // AlphaMajorToDet[Astring][itb-AlphaMajorToBeta[Astring].begin()];

          if (DetJ < DetI) {
            size_t orbDiff;
            CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
            if (abs(hij) < 1.e-10)
              continue;
            connections[DetI / nprocs].push_back(DetJ);
            Helements[DetI / nprocs].push_back(hij);
            if (DoRDM)
              orbDifference[DetI / nprocs].push_back(orbDiff);
          }
        }
      }

      // double beta excitation
      for (int j = 0; j < AlphaMajorToBeta[i].size(); j++) {
        int DetJ = AlphaMajorToDet[i][j];

        if (DetJ < DetI && Dets[DetJ].ExcitationDistance(Dets[DetI]) == 2) {
          size_t orbDiff;
          CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
          if (abs(hij) > 1.e-10) {
            connections[DetI / nprocs].push_back(DetJ);
            Helements[DetI / nprocs].push_back(hij);
            if (DoRDM)
              orbDifference[DetI / nprocs].push_back(orbDiff);
          }
        }
      }

      // double Alpha excitation
      for (int j = 0; j < BetaMajorToAlpha[Bstring].size(); j++) {
        int DetJ = BetaMajorToDet[Bstring][j];

        if (DetJ < DetI && Dets[DetJ].ExcitationDistance(Dets[DetI]) == 2) {
          size_t orbDiff;
          CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
          if (abs(hij) > 1.e-10) {
            connections[DetI / nprocs].push_back(DetJ);
            Helements[DetI / nprocs].push_back(hij);
            if (DoRDM)
              orbDifference[DetI / nprocs].push_back(orbDiff);
          }
        }
      }
    }
  }
}

void SHCImakeHamiltonian::MakeSHMHelpers(
    std::map<HalfDet, std::vector<int>> &BetaN,
    std::map<HalfDet, std::vector<int>> &AlphaN, int *&betaVecLenSHM,
    vector<int *> &betaVecSHM, int *&alphaVecLenSHM,
    vector<int *> &alphaVecSHM) {
  int comm_rank = 0, comm_size = 1;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
#endif
  boost::interprocess::shared_memory_object::remove(shciHelper.c_str());

  size_t totalMemory = 0, nBeta = 0, nAlpha = 0;
  vector<int> betaveclen(BetaN.size(), 0), alphaveclen(AlphaN.size(), 0);
  if (comm_rank == 0) {
    // Now put it on shared memory
    auto ita = BetaN.begin();
    for (; ita != BetaN.end(); ita++) {
      betaveclen[nBeta] = ita->second.size();
      nBeta++;
      totalMemory += sizeof(int); // write how many elements in the vector
      totalMemory +=
          sizeof(int) * ita->second.size(); // memory to store the vector
    }

    ita = AlphaN.begin();
    for (; ita != AlphaN.end(); ita++) {
      alphaveclen[nAlpha] = ita->second.size();
      nAlpha++;
      totalMemory += sizeof(int); // write how many elements in the vector
      totalMemory +=
          sizeof(int) * ita->second.size(); // memory to store the vector
    }
  }
#ifndef SERIAL
  MPI_Bcast(&totalMemory, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nBeta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nAlpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (comm_rank != 0) {
    betaveclen.resize(nBeta);
    alphaveclen.resize(nAlpha);
  }
  MPI_Bcast(&betaveclen[0], nBeta, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&alphaveclen[0], nAlpha, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  hHelpersSegment.truncate(totalMemory);
  regionHelpers = boost::interprocess::mapped_region{
      hHelpersSegment, boost::interprocess::read_write};
  memset(regionHelpers.get_address(), 0., totalMemory);

#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  betaVecLenSHM = static_cast<int *>(regionHelpers.get_address());
  betaVecSHM.resize(nBeta);
  size_t counter = 0;
  for (int i = 0; i < nBeta; i++) {
    betaVecSHM[i] = static_cast<int *>(betaVecLenSHM + nBeta + counter);
    betaVecLenSHM[i] = betaveclen[i];
    counter += betaveclen[i];
  }

  int *beginAlpha = betaVecSHM[0] + counter;
  alphaVecLenSHM = static_cast<int *>(beginAlpha);
  alphaVecSHM.resize(nAlpha);
  counter = 0;
  for (int i = 0; i < nAlpha; i++) {
    alphaVecSHM[i] = static_cast<int *>(beginAlpha + nAlpha + counter);
    alphaVecLenSHM[i] = alphaveclen[i];
    counter += alphaveclen[i];
  }

  // now fill the memory
  if (comm_rank == 0) {
    size_t nBeta = 0, nAlpha = 0;
    auto ita = BetaN.begin();
    for (; ita != BetaN.end(); ita++) {
      for (int j = 0; j < ita->second.size(); j++) {
        betaVecSHM[nBeta][j] = ita->second[j];
      }
      nBeta++;
    }

    ita = AlphaN.begin();
    for (; ita != AlphaN.end(); ita++) {
      for (int j = 0; j < ita->second.size(); j++)
        alphaVecSHM[nAlpha][j] = ita->second[j];
      nAlpha++;
    }
  }

  long intdim = totalMemory;
  long maxint =
      26843540; // mpi cannot transfer more than these number of doubles
  long maxIter = intdim / maxint;
#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);
  char *shrdMem = static_cast<char *>(regionHelpers.get_address());
  for (int i = 0; i < maxIter; i++) {
    MPI::COMM_WORLD.Bcast(shrdMem + i * maxint, maxint, MPI_CHAR, 0);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI::COMM_WORLD.Bcast(shrdMem + (maxIter)*maxint,
                        totalMemory - maxIter * maxint, MPI_CHAR, 0);
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

void SHCIbasics::DoPerturbativeStochastic2SingleListDoubleEpsilon2OMPTogether(
    vector<Determinant> &Dets, MatrixXx &ci, double &E0, oneInt &I1, twoInt &I2,
    twoIntHeatBathSHM &I2HB, vector<int> &irrep, schedule &schd, double coreE,
    int nelec, int root) {

  boost::mpi::communicator world;
  char file[5000];
  sprintf(file, "output-%d.bkp", world.rank());
  std::ofstream ofs;
  if (root == 0)
    ofs.open(file, std::ofstream::out);
  else
    ofs.open(file, std::ofstream::app);

  double epsilon2 = schd.epsilon2;
  schd.epsilon2 = schd.epsilon2Large;
  vector<MatrixXx> vdVector;
  double Psi1Norm;
  double EptLarge =
      DoPerturbativeDeterministic(Dets, ci, E0, I1, I2, I2HB, irrep, schd,
                                  coreE, nelec, root, vdVector, Psi1Norm);

  schd.epsilon2 = epsilon2;

  int norbs = Determinant::norbs;
  std::vector<Determinant> SortedDets = Dets;
  std::sort(SortedDets.begin(), SortedDets.end());
  int niter = schd.nPTiter;
  // double eps = 0.001;
  int Nsample = schd.SampleN;
  double AvgenergyEN = 0.0;
  double AverageDen = 0.0;
  int currentIter = 0;
  int sampleSize = 0;
  int num_thrds = omp_get_max_threads();

  double cumulative = 0.0;
  for (int i = 0; i < ci.rows(); i++)
    cumulative += abs(ci(i, 0));

  std::vector<int> alias;
  std::vector<double> prob;
  SHCIsampledeterminants::setUpAliasMethod(ci, cumulative, alias, prob);

  double totalPT = 0, totalPTLargeEps = 0;

  std::vector<std::vector<vector<Determinant>>> hashedDetBeforeMPI(
      num_thrds); // vector<Determinant> > >(num_thrds));
  std::vector<std::vector<vector<CItype>>> hashedNum1BeforeMPI(
      num_thrds); // std::vector<vector<double> > >(num_thrds));
  std::vector<std::vector<vector<CItype>>> hashedNum2BeforeMPI(
      num_thrds); // std::vector<vector<double> > >(num_thrds));
  std::vector<std::vector<vector<double>>> hashedEnergyBeforeMPI(
      num_thrds); // std::vector<vector<double> > >(num_thrds));
  std::vector<std::vector<vector<char>>> hashedpresentBeforeMPI(
      num_thrds); // std::vector<vector<char> > >(num_thrds));

  int AllDistinctSample = 0;
  int Nmc = Nsample * num_thrds;
  std::vector<CItype> allwts(Nmc, 0.0);
  std::vector<int> allSample(Nmc, -1);

#pragma omp parallel
  {
    for (int iter = 0; iter < niter; iter++) {
      std::vector<CItype> wts1(Nsample, 0.0);
      std::vector<int> Sample1(Nsample, -1);
      int distinctSample = 0;

      if (omp_get_thread_num() == 0) {
        std::fill(allSample.begin(), allSample.end(), -1);
        AllDistinctSample = SHCIsampledeterminants::sample_N2_alias(
            ci, cumulative, allSample, allwts, alias, prob);
      }
#pragma omp barrier
      if (omp_get_thread_num() < AllDistinctSample % num_thrds)
        distinctSample = AllDistinctSample / num_thrds + 1;
      else
        distinctSample = AllDistinctSample / num_thrds;

      size_t stride = omp_get_thread_num() < AllDistinctSample % num_thrds
                          ? omp_get_thread_num() * distinctSample
                          : omp_get_thread_num() * distinctSample +
                                AllDistinctSample % num_thrds;

      for (int i = 0; i < distinctSample; i++) {
        wts1[i] = allwts[i + stride];
        Sample1[i] = allSample[i + stride];
      }

      double norm = 0.0;

      std::vector<Determinant> Psi1;
      std::vector<CItype> numerator1A;
      vector<CItype> numerator2A;
      vector<char> present;
      std::vector<double> det_energy;

      for (int i = 0; i < distinctSample; i++) {
        int I = Sample1[i];
        SHCIgetdeterminants::getDeterminantsStochastic2Epsilon(
            Dets[I], schd.epsilon2 / abs(ci(I, 0)),
            schd.epsilon2Large / abs(ci(I, 0)), wts1[i], ci(I, 0), I1, I2, I2HB,
            irrep, coreE, E0, Psi1, numerator1A, numerator2A, present,
            det_energy, schd, Nmc, nelec);
      }

      if (num_thrds > 1) {
        hashedDetBeforeMPI[omp_get_thread_num()].resize(num_thrds);
        hashedNum1BeforeMPI[omp_get_thread_num()].resize(num_thrds);
        hashedNum2BeforeMPI[omp_get_thread_num()].resize(num_thrds);
        hashedEnergyBeforeMPI[omp_get_thread_num()].resize(num_thrds);
        hashedpresentBeforeMPI[omp_get_thread_num()].resize(num_thrds);

        for (int thrd = 0; thrd < num_thrds; thrd++) {
          hashedDetBeforeMPI[omp_get_thread_num()][thrd].reserve(Psi1.size() /
                                                                 num_thrds);
          hashedNum1BeforeMPI[omp_get_thread_num()][thrd].reserve(Psi1.size() /
                                                                  num_thrds);
          hashedNum2BeforeMPI[omp_get_thread_num()][thrd].reserve(Psi1.size() /
                                                                  num_thrds);
          hashedEnergyBeforeMPI[omp_get_thread_num()][thrd].reserve(
              Psi1.size() / num_thrds);
          hashedpresentBeforeMPI[omp_get_thread_num()][thrd].reserve(
              Psi1.size() / num_thrds);
        }

        for (int j = 0; j < Psi1.size(); j++) {
          size_t lOrder = Psi1.at(j).getHash();
          size_t thrd = lOrder % (num_thrds);
          hashedDetBeforeMPI[omp_get_thread_num()][thrd].push_back(Psi1.at(j));
          hashedNum1BeforeMPI[omp_get_thread_num()][thrd].push_back(
              numerator1A.at(j));
          hashedNum2BeforeMPI[omp_get_thread_num()][thrd].push_back(
              numerator2A.at(j));
          hashedEnergyBeforeMPI[omp_get_thread_num()][thrd].push_back(
              det_energy.at(j));
          hashedpresentBeforeMPI[omp_get_thread_num()][thrd].push_back(
              present.at(j));
        }
        Psi1.clear();
        numerator1A.clear();
        numerator2A.clear();
        det_energy.clear();
        present.clear();

#pragma omp barrier
        size_t totalSize = 0;
        for (int thrd = 0; thrd < num_thrds; thrd++)
          totalSize += hashedDetBeforeMPI[thrd][omp_get_thread_num()].size();

        Psi1.reserve(totalSize);
        numerator1A.reserve(totalSize);
        numerator2A.reserve(totalSize);
        det_energy.reserve(totalSize);
        present.reserve(totalSize);
        for (int thrd = 0; thrd < num_thrds; thrd++) {
          for (int j = 0;
               j < hashedDetBeforeMPI[thrd][omp_get_thread_num()].size(); j++) {
            Psi1.push_back(
                hashedDetBeforeMPI[thrd][omp_get_thread_num()].at(j));
            numerator1A.push_back(
                hashedNum1BeforeMPI[thrd][omp_get_thread_num()].at(j));
            numerator2A.push_back(
                hashedNum2BeforeMPI[thrd][omp_get_thread_num()].at(j));
            det_energy.push_back(
                hashedEnergyBeforeMPI[thrd][omp_get_thread_num()].at(j));
            present.push_back(
                hashedpresentBeforeMPI[thrd][omp_get_thread_num()].at(j));
          }
          hashedDetBeforeMPI[thrd][omp_get_thread_num()].clear();
          hashedNum1BeforeMPI[thrd][omp_get_thread_num()].clear();
          hashedNum2BeforeMPI[thrd][omp_get_thread_num()].clear();
          hashedEnergyBeforeMPI[thrd][omp_get_thread_num()].clear();
          hashedpresentBeforeMPI[thrd][omp_get_thread_num()].clear();
        }
      }

      std::vector<Determinant> Psi1copy = Psi1;
      vector<long> detIndex(Psi1.size(), 0);
      vector<long> detIndexcopy(Psi1.size(), 0);
      for (size_t i = 0; i < Psi1.size(); i++)
        detIndex[i] = i;
      mergesort(&Psi1copy[0], 0, Psi1.size() - 1, &detIndex[0],
                &(Psi1.operator[](0)), &detIndexcopy[0]);
      detIndexcopy.clear();
      Psi1copy.clear();
      reorder(numerator1A, detIndex);
      reorder(numerator2A, detIndex);
      reorder(det_energy, detIndex);
      reorder(present, detIndex);
      detIndex.clear();

      CItype currentNum1A = 0.;
      CItype currentNum2A = 0.;
      CItype currentNum1B = 0.;
      CItype currentNum2B = 0.;
      vector<Determinant>::iterator vec_it = SortedDets.begin();
      double energyEN = 0.0, energyENLargeEps = 0.0;
      // size_t effNmc = num_thrds*Nmc;

      for (int i = 0; i < Psi1.size();) {
        if (Psi1[i] < *vec_it) {
          currentNum1A += numerator1A[i];
          currentNum2A += numerator2A[i];
          if (present[i]) {
            currentNum1B += numerator1A[i];
            currentNum2B += numerator2A[i];
          }

          if (i == Psi1.size() - 1) {
#ifdef Complex
            energyEN += (pow(abs(currentNum1A), 2) * Nmc / (Nmc - 1) -
                         currentNum2A.real()) /
                        (det_energy[i] - E0);
            energyENLargeEps += (pow(abs(currentNum1B), 2) * Nmc / (Nmc - 1) -
                                 currentNum2B.real()) /
                                (det_energy[i] - E0);
#else
            energyEN +=
                (pow(abs(currentNum1A), 2) * Nmc / (Nmc - 1) - currentNum2A) /
                (det_energy[i] - E0);
            energyENLargeEps +=
                (pow(abs(currentNum1B), 2) * Nmc / (Nmc - 1) - currentNum2B) /
                (det_energy[i] - E0);
#endif
          } else if (!(Psi1[i] == Psi1[i + 1])) {
#ifdef Complex
            energyEN += (pow(abs(currentNum1A), 2) * Nmc / (Nmc - 1) -
                         currentNum2A.real()) /
                        (det_energy[i] - E0);
            energyENLargeEps += (pow(abs(currentNum1B), 2) * Nmc / (Nmc - 1) -
                                 currentNum2B.real()) /
                                (det_energy[i] - E0);
#else
            energyEN +=
                (pow(abs(currentNum1A), 2) * Nmc / (Nmc - 1) - currentNum2A) /
                (det_energy[i] - E0);
            energyENLargeEps +=
                (pow(abs(currentNum1B), 2) * Nmc / (Nmc - 1) - currentNum2B) /
                (det_energy[i] - E0);
#endif
            // energyEN += ( pow(abs(currentNum1A),2)*Nmc/(Nmc-1) -
            // currentNum2A)/(det_energy[i] - E0); energyENLargeEps += (
            // pow(abs(currentNum1B),2)*Nmc/(Nmc-1) -
            // currentNum2B)/(det_energy[i] - E0);
            currentNum1A = 0.;
            currentNum2A = 0.;
            currentNum1B = 0.;
            currentNum2B = 0.;
          }
          i++;
        } else if (*vec_it < Psi1[i] && vec_it != SortedDets.end())
          vec_it++;
        else if (*vec_it < Psi1[i] && vec_it == SortedDets.end()) {
          currentNum1A += numerator1A[i];
          currentNum2A += numerator2A[i];
          if (present[i]) {
            currentNum1B += numerator1A[i];
            currentNum2B += numerator2A[i];
          }

          if (i == Psi1.size() - 1) {
#ifdef Complex
            energyEN += (pow(abs(currentNum1A), 2) * Nmc / (Nmc - 1) -
                         currentNum2A.real()) /
                        (det_energy[i] - E0);
            energyENLargeEps += (pow(abs(currentNum1B), 2) * Nmc / (Nmc - 1) -
                                 currentNum2B.real()) /
                                (det_energy[i] - E0);
#else
            energyEN +=
                (pow(abs(currentNum1A), 2) * Nmc / (Nmc - 1) - currentNum2A) /
                (det_energy[i] - E0);
            energyENLargeEps +=
                (pow(abs(currentNum1B), 2) * Nmc / (Nmc - 1) - currentNum2B) /
                (det_energy[i] - E0);
#endif
            // energyEN += ( pow(abs(currentNum1A),2)*Nmc/(Nmc-1) -
            // currentNum2A)/(det_energy[i] - E0); energyENLargeEps += (
            // pow(abs(currentNum1B),2)*Nmc/(Nmc-1) -
            // currentNum2B)/(det_energy[i] - E0);
          } else if (!(Psi1[i] == Psi1[i + 1])) {
#ifdef Complex
            energyEN += (pow(abs(currentNum1A), 2) * Nmc / (Nmc - 1) -
                         currentNum2A.real()) /
                        (det_energy[i] - E0);
            energyENLargeEps += (pow(abs(currentNum1B), 2) * Nmc / (Nmc - 1) -
                                 currentNum2B.real()) /
                                (det_energy[i] - E0);
#else
            energyEN +=
                (pow(abs(currentNum1A), 2) * Nmc / (Nmc - 1) - currentNum2A) /
                (det_energy[i] - E0);
            energyENLargeEps +=
                (pow(abs(currentNum1B), 2) * Nmc / (Nmc - 1) - currentNum2B) /
                (det_energy[i] - E0);
#endif
            // energyEN += ( pow(abs(currentNum1A),2)*Nmc/(Nmc-1) -
            // currentNum2A)/(det_energy[i] - E0); energyENLargeEps += (
            // pow(abs(currentNum1B),2)*Nmc/(Nmc-1) -
            // currentNum2B)/(det_energy[i] - E0); energyEN +=
            // (currentNum1A*currentNum1A*Nmc/(Nmc-1) -
            // currentNum2A)/(det_energy[i] - E0); energyENLargeEps +=
            // (currentNum1B*currentNum1B*Nmc/(Nmc-1) -
            // currentNum2B)/(det_energy[i] - E0);
            currentNum1A = 0.;
            currentNum2A = 0.;
            currentNum1B = 0.;
            currentNum2B = 0.;
          }
          i++;
        } else {
          if (Psi1[i] == Psi1[i + 1])
            i++;
          else {
            vec_it++;
            i++;
          }
        }
      }

      totalPT = 0;
      totalPTLargeEps = 0;
#pragma omp barrier
#pragma omp critical
      {
        totalPT += energyEN;
        totalPTLargeEps += energyENLargeEps;
      }
#pragma omp barrier

      double finalE = totalPT, finalELargeEps = totalPTLargeEps;
      if (mpigetrank() == 0 && omp_get_thread_num() == 0) {
        AvgenergyEN += -finalE + finalELargeEps + EptLarge;
        currentIter++;
        cout << finalE << "  " << finalELargeEps << "  " << EptLarge << endl;
        std::cout << format("%6i  %14.8f  %s%i %14.8f   %10.2f  %10i") %
                         (currentIter) %
                         (E0 - finalE + finalELargeEps + EptLarge) % ("Root") %
                         root % (E0 + AvgenergyEN / currentIter) %
                         (getTime() - startofCalc) % AllDistinctSample;
        cout << endl;
      } else if (mpigetrank() != 0 && omp_get_thread_num() == 0) {
        AvgenergyEN += -finalE + finalELargeEps + EptLarge;
        currentIter++;
        ofs << format("%6i  %14.8f  %s%i %14.8f   %10.2f  %10i") %
                   (currentIter) % (E0 - finalE + finalELargeEps + EptLarge) %
                   ("Root") % root % (E0 + AvgenergyEN / currentIter) %
                   (getTime() - startofCalc) % AllDistinctSample;
        ofs << endl;
      }
    }
  }
}
void SHCIbasics::DoPerturbativeStochastic2SingleList(
    vector<Determinant> &Dets, MatrixXx &ci, double &E0, oneInt &I1, twoInt &I2,
    twoIntHeatBathSHM &I2HB, vector<int> &irrep, schedule &schd, double coreE,
    int nelec, int root) {

  boost::mpi::communicator world;
  char file[5000];
  sprintf(file, "output-%d.bkp", world.rank());
  // std::ofstream ofs(file);
  std::ofstream ofs;
  if (root == 0)
    ofs.open(file, std::ofstream::out);
  else
    ofs.open(file, std::ofstream::app);

  int norbs = Determinant::norbs;
  std::vector<Determinant> SortedDets = Dets;
  std::sort(SortedDets.begin(), SortedDets.end());
  int niter = schd.nPTiter;
  // int niter = 1000000;
  // double eps = 0.001;
  int Nsample = schd.SampleN;
  double AvgenergyEN = 0.0;
  double AverageDen = 0.0;
  int currentIter = 0;
  int sampleSize = 0;
  int num_thrds = omp_get_max_threads();

  double cumulative = 0.0;
  for (int i = 0; i < ci.rows(); i++)
    cumulative += abs(ci(i, 0));

  std::vector<int> alias;
  std::vector<double> prob;
  SHCIsampledeterminants::setUpAliasMethod(ci, cumulative, alias, prob);
#pragma omp parallel for schedule(dynamic)
  for (int iter = 0; iter < niter; iter++) {
    // cout << norbs<<"  "<<nelec<<endl;
    char psiArray[norbs];
    vector<int> psiClosed(nelec, 0);
    vector<int> psiOpen(norbs - nelec, 0);
    // char psiArray[norbs];
    std::vector<CItype> wts1(Nsample, 0.0);
    std::vector<int> Sample1(Nsample, -1);

    // int Nmc = sample_N2(ci, cumulative, Sample1, wts1);
    int distinctSample = SHCIsampledeterminants::sample_N2_alias(
        ci, cumulative, Sample1, wts1, alias, prob);
    int Nmc = Nsample;
    double norm = 0.0;

    size_t initSize = 100000;
    std::vector<Determinant> Psi1;
    std::vector<CItype> numerator1;
    std::vector<double> numerator2;
    std::vector<double> det_energy;
    Psi1.reserve(initSize);
    numerator1.reserve(initSize);
    numerator2.reserve(initSize);
    det_energy.reserve(initSize);
    for (int i = 0; i < distinctSample; i++) {
      int I = Sample1[i];
      SHCIgetdeterminants::getDeterminantsStochastic(
          Dets[I], schd.epsilon2 / abs(ci(I, 0)), wts1[i], ci(I, 0), I1, I2,
          I2HB, irrep, coreE, E0, Psi1, numerator1, numerator2, det_energy,
          schd, Nmc, nelec);
    }

    quickSort(&(Psi1[0]), 0, Psi1.size(), &numerator1[0], &numerator2[0],
              &det_energy);

    CItype currentNum1 = 0.;
    double currentNum2 = 0.;
    vector<Determinant>::iterator vec_it = SortedDets.begin();
    double energyEN = 0.0;

    for (int i = 0; i < Psi1.size();) {
      if (Psi1[i] < *vec_it) {
        currentNum1 += numerator1[i];
        currentNum2 += numerator2[i];
        if (i == Psi1.size() - 1)
          energyEN +=
              (pow(abs(currentNum1), 2) * Nmc / (Nmc - 1) - currentNum2) /
              (det_energy[i] - E0);
        else if (!(Psi1[i] == Psi1[i + 1])) {
          energyEN +=
              (pow(abs(currentNum1), 2) * Nmc / (Nmc - 1) - currentNum2) /
              (det_energy[i] - E0);
          currentNum1 = 0.;
          currentNum2 = 0.;
        }
        i++;
      } else if (*vec_it < Psi1[i] && vec_it != SortedDets.end())
        vec_it++;
      else if (*vec_it < Psi1[i] && vec_it == SortedDets.end()) {
        currentNum1 += numerator1[i];
        currentNum2 += numerator2[i];
        if (i == Psi1.size() - 1)
          energyEN +=
              (pow(abs(currentNum1), 2) * Nmc / (Nmc - 1) - currentNum2) /
              (det_energy[i] - E0);
        // energyEN += (currentNum1*currentNum1*Nmc/(Nmc-1) -
        // currentNum2)/(det_energy[i] - E0);
        else if (!(Psi1[i] == Psi1[i + 1])) {
          energyEN +=
              (pow(abs(currentNum1), 2) * Nmc / (Nmc - 1) - currentNum2) /
              (det_energy[i] - E0);
          // energyEN += (currentNum1*currentNum1*Nmc/(Nmc-1) -
          // currentNum2)/(det_energy[i] - E0);
          currentNum1 = 0.;
          currentNum2 = 0.;
        }
        i++;
      } else {
        if (Psi1[i] == Psi1[i + 1])
          i++;
        else {
          vec_it++;
          i++;
        }
      }
    }

    sampleSize = distinctSample;

#pragma omp critical
    {
      if (mpigetrank() == 0) {
        AvgenergyEN += energyEN;
        currentIter++;
        std::cout << format("%6i  %14.8f  %s%i %14.8f   %10.2f  %10i %4i") %
                         (currentIter) % (E0 - energyEN) % ("Root") % root %
                         (E0 - AvgenergyEN / currentIter) %
                         (getTime() - startofCalc) % sampleSize %
                         (omp_get_thread_num());
        cout << endl;
      } else {
        AvgenergyEN += energyEN;
        currentIter++;
        ofs << format("%6i  %14.8f  %s%i %14.8f   %10.2f  %10i %4i") %
                   (currentIter) % (E0 - energyEN) % ("Root") % root %
                   (E0 - AvgenergyEN / currentIter) %
                   (getTime() - startofCalc) % sampleSize %
                   (omp_get_thread_num());
        ofs << endl;
      }
    }
  }
  ofs.close();
}

void SHCIbasics::DoBatchDeterministic(vector<Determinant> &Dets, MatrixXx &ci,
                                      double &E0, oneInt &I1, twoInt &I2,
                                      twoIntHeatBath &I2HB, vector<int> &irrep,
                                      schedule &schd, double coreE, int nelec) {
  int nblocks = schd.nblocks;
  std::vector<int> blockSizes(nblocks, 0);
  for (int i = 0; i < nblocks; i++) {
    if (i != nblocks - 1)
      blockSizes[i] = Dets.size() / nblocks;
    else
      blockSizes[i] = Dets.size() - (nblocks - 1) * Dets.size() / (nblocks);
  }
  int norbs = Determinant::norbs;
  std::vector<Determinant> SortedDets = Dets;
  std::sort(SortedDets.begin(), SortedDets.end());
  double AvgenergyEN = 0.0;

#pragma omp parallel for schedule(dynamic)
  for (int inter1 = 0; inter1 < nblocks; inter1++) {

    vector<int> psiClosed(nelec, 0);
    vector<int> psiOpen(norbs - nelec, 0);

    std::vector<double> wts1(blockSizes[inter1]);
    std::vector<int> Sample1(blockSizes[inter1]);
    for (int i = 0; i < wts1.size(); i++) {
      wts1[i] = ci(i + (inter1)*Dets.size() / nblocks, 0);
      Sample1[i] = i + (inter1)*Dets.size() / nblocks;
    }

    map<Determinant, pair<double, double>> Psi1ab;
    for (int i = 0; i < Sample1.size(); i++) {
      int I = Sample1[i];
      SHCIbasics::getDeterminants(Dets[I], abs(schd.epsilon2 / ci(I, 0)),
                                  wts1[i], 0.0, I1, I2, I2HB, irrep, coreE, E0,
                                  Psi1ab, SortedDets, schd);
    }

    double energyEN = 0.0;
    for (map<Determinant, pair<double, double>>::iterator it = Psi1ab.begin();
         it != Psi1ab.end(); it++) {
      it->first.getOpenClosed(psiOpen, psiClosed);
      energyEN += it->second.first * it->second.first /
                  (Energy(psiClosed, nelec, I1, I2, coreE) - E0);
    }

    for (int i = Sample1[Sample1.size() - 1] + 1; i < Dets.size(); i++) {
      SHCIbasics::getDeterminants(Dets[i], abs(schd.epsilon2 / ci(i, 0)), 0.0,
                                  ci(i, 0), I1, I2, I2HB, irrep, coreE, E0,
                                  Psi1ab, SortedDets, schd);
      if (i % 1000 == 0 && omp_get_thread_num() == 0)
        cout << i << " out of " << Dets.size() - Sample1.size() << endl;
    }

    for (map<Determinant, pair<double, double>>::iterator it = Psi1ab.begin();
         it != Psi1ab.end(); it++) {
      it->first.getOpenClosed(psiOpen, psiClosed);
      energyEN += 2. * it->second.first * it->second.second /
                  (Energy(psiClosed, nelec, I1, I2, coreE) - E0);
    }

#pragma omp critical
    {
      AvgenergyEN += energyEN;

      std::cout << format("%6i  %14.8f   %10.2f  %10i %4i") % (inter1) %
                       (E0 - AvgenergyEN) % (getTime() - startofCalc) % inter1 %
                       (omp_get_thread_num());
      cout << endl;
    }
  }

  {
    std::cout << "FINAL ANSWER " << endl;
    std::cout << format("%4i  %14.8f   %10.2f  %10i %4i") % (nblocks) %
                     (E0 - AvgenergyEN) % (getTime() - startofCalc) % nblocks %
                     (0);
    cout << endl;
  }
}

void SHCIbasics::DoPerturbativeStochastic2(vector<Determinant> &Dets,
                                           MatrixXx &ci, double &E0, oneInt &I1,
                                           twoInt &I2, twoIntHeatBath &I2HB,
                                           vector<int> &irrep, schedule &schd,
                                           double coreE, int nelec) {

  cout << "This function is most likely broken, dont use it. Use the single "
          "list method instead!!!"
       << endl;
  exit(0);
  boost::mpi::communicator world;
  char file[5000];
  sprintf(file, "output-%d.bkp", world.rank());
  std::ofstream ofs(file);

  int norbs = Determinant::norbs;
  std::vector<Determinant> SortedDets = Dets;
  std::sort(SortedDets.begin(), SortedDets.end());
  int niter = 10000;
  // double eps = 0.001;
  int Nsample = schd.SampleN;
  double AvgenergyEN = 0.0;
  double AverageDen = 0.0;
  int currentIter = 0;
  int sampleSize = 0;
  int num_thrds = omp_get_max_threads();

  double cumulative = 0.0;
  for (int i = 0; i < ci.rows(); i++)
    cumulative += abs(ci(i, 0));

#pragma omp parallel for schedule(dynamic)
  for (int iter = 0; iter < niter; iter++) {
    // cout << norbs<<"  "<<nelec<<endl;
    char psiArray[norbs];
    vector<int> psiClosed(nelec, 0);
    vector<int> psiOpen(norbs - nelec, 0);
    // char psiArray[norbs];
    std::vector<double> wts1(Nsample, 0.0), wts2(Nsample, 0.0);
    std::vector<int> Sample1(Nsample, -1), Sample2(Nsample, -1);
    // wts1.reserve(Nsample); wts2.reserve(Nsample); Sample1.reserve(Nsample);
    // Sample2.reserve(Nsample);

    sample_N2(ci, cumulative, Sample1, wts1);
    sample_N2(ci, cumulative, Sample2, wts2);

    double norm = 0.0;
    for (int i = 0; i < Sample1.size(); i++) {
      double normi = 0.0;
      for (int j = 0; j < Sample2.size(); j++)
        if (Sample2[j] == Sample1[i])
          normi += wts1[i] * wts2[j];
      norm += normi;
    }

    map<Determinant, pair<double, double>> Psi1ab;
    for (int i = 0; i < Sample1.size(); i++) {
      int I = Sample1[i];
      std::vector<int>::iterator it = find(Sample2.begin(), Sample2.end(), I);
      if (it != Sample2.end())
        SHCIbasics::getDeterminants(
            Dets[I], abs(schd.epsilon2 / ci(I, 0)), wts1[i],
            wts2[distance(Sample2.begin(), it)], I1, I2, I2HB, irrep, coreE, E0,
            Psi1ab, SortedDets, schd);
      else
        SHCIbasics::getDeterminants(Dets[I], abs(schd.epsilon2 / ci(I, 0)),
                                    wts1[i], 0.0, I1, I2, I2HB, irrep, coreE,
                                    E0, Psi1ab, SortedDets, schd);
    }

    for (int i = 0; i < Sample2.size(); i++) {
      int I = Sample2[i];
      std::vector<int>::iterator it = find(Sample1.begin(), Sample1.end(), I);
      if (it == Sample1.end())
        SHCIbasics::getDeterminants(Dets[I], abs(schd.epsilon2 / ci(I, 0)), 0.0,
                                    wts2[i], I1, I2, I2HB, irrep, coreE, E0,
                                    Psi1ab, SortedDets, schd);
    }

    double energyEN = 0.0;
    for (map<Determinant, pair<double, double>>::iterator it = Psi1ab.begin();
         it != Psi1ab.end(); it++) {
      it->first.getOpenClosed(psiOpen, psiClosed);
      energyEN += it->second.first * it->second.second /
                  (Energy(psiClosed, nelec, I1, I2, coreE) - E0);
    }
    sampleSize = Sample1.size();
    AverageDen += norm;
#pragma omp critical
    {
      if (mpigetrank() == 0) {
        AvgenergyEN += energyEN;
        currentIter++;
        std::cout << format("%6i  %14.8f  %14.8f %14.8f   %10.2f  %10i %4i") %
                         (currentIter) % (E0 - energyEN) % (norm) %
                         (E0 - AvgenergyEN / currentIter) %
                         (getTime() - startofCalc) % sampleSize %
                         (omp_get_thread_num());
        cout << endl;
      } else {
        AvgenergyEN += energyEN;
        currentIter++;
        ofs << format("%6i  %14.8f  %14.8f %14.8f   %10.2f  %10i %4i") %
                   (currentIter) % (E0 - energyEN) % (norm) %
                   (E0 - AvgenergyEN / AverageDen) % (getTime() - startofCalc) %
                   sampleSize % (omp_get_thread_num());
        ofs << endl;
      }
    }
  }
  ofs.close();
}

void SHCIbasics::DoPerturbativeStochastic(vector<Determinant> &Dets,
                                          MatrixXx &ci, double &E0, oneInt &I1,
                                          twoInt &I2, twoIntHeatBath &I2HB,
                                          vector<int> &irrep, schedule &schd,
                                          double coreE, int nelec) {

  boost::mpi::communicator world;
  char file[5000];
  sprintf(file, "output-%d.bkp", world.rank());
  std::ofstream ofs(file);

  int norbs = Determinant::norbs;
  std::vector<Determinant> SortedDets = Dets;
  std::sort(SortedDets.begin(), SortedDets.end());
  int niter = 10000;
  // double eps = 0.001;
  double AvgenergyEN = 0.0;
  double AverageDen = 0.0;
  int currentIter = 0;
  int sampleSize = 0;
  int num_thrds = omp_get_max_threads();

#pragma omp parallel for schedule(dynamic)
  for (int iter = 0; iter < niter; iter++) {
    // cout << norbs<<"  "<<nelec<<endl;
    char psiArray[norbs];
    vector<int> psiClosed(nelec, 0);
    vector<int> psiOpen(norbs - nelec, 0);
    // char psiArray[norbs];
    std::vector<double> wts1, wts2;
    std::vector<int> Sample1, Sample2;
    wts1.reserve(1000);
    wts2.reserve(1000);
    Sample1.reserve(1000);
    Sample2.reserve(1000);

    Sample1.resize(0);
    wts1.resize(0);
    Sample2.resize(0);
    wts2.resize(0);
    sample_round(ci, schd.eps, Sample1, wts1);
    sample_round(ci, schd.eps, Sample2, wts2);

    double norm = 0.0;
    for (int i = 0; i < Sample1.size(); i++) {
      for (int j = 0; j < Sample2.size(); j++)
        if (Sample2[j] == Sample1[i]) {
          norm += wts1[i] * wts2[j];
          break;
        }
    }

    map<Determinant, pair<double, double>> Psi1ab;
    for (int i = 0; i < Sample1.size(); i++) {
      int I = Sample1[i];
      // SHCIbasics::getDeterminants(Dets[I], abs(schd.epsilon2/ci(I,0)),
      // wts1[i], 0.0, I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets,
      // schd);
      std::vector<int>::iterator it = find(Sample2.begin(), Sample2.end(), I);
      if (it != Sample2.end())
        SHCIbasics::getDeterminants(
            Dets[I], abs(schd.epsilon2 / ci(I, 0)), wts1[i],
            wts2[distance(Sample2.begin(), it)], I1, I2, I2HB, irrep, coreE, E0,
            Psi1ab, SortedDets, schd);
      else
        SHCIbasics::getDeterminants(Dets[I], abs(schd.epsilon2 / ci(I, 0)),
                                    wts1[i], 0.0, I1, I2, I2HB, irrep, coreE,
                                    E0, Psi1ab, SortedDets, schd);
    }

    for (int i = 0; i < Sample2.size(); i++) {
      int I = Sample2[i];
      std::vector<int>::iterator it = find(Sample1.begin(), Sample1.end(), I);
      if (it == Sample1.end())
        SHCIbasics::getDeterminants(Dets[I], abs(schd.epsilon2 / ci(I, 0)), 0.0,
                                    wts2[i], I1, I2, I2HB, irrep, coreE, E0,
                                    Psi1ab, SortedDets, schd);
    }

    double energyEN = 0.0;
    for (map<Determinant, pair<double, double>>::iterator it = Psi1ab.begin();
         it != Psi1ab.end(); it++) {
      it->first.getOpenClosed(psiOpen, psiClosed);
      energyEN += it->second.first * it->second.second /
                  (Energy(psiClosed, nelec, I1, I2, coreE) - E0);
    }
    sampleSize = Sample1.size();

#pragma omp critical
    {
      if (mpigetrank() == 0) {
        AvgenergyEN += energyEN;
        currentIter++;
        std::cout << format("%6i  %14.8f  %14.8f %14.8f   %10.2f  %10i %4i") %
                         (currentIter) % (E0 - energyEN) % (norm) %
                         (E0 - AvgenergyEN / currentIter) %
                         (getTime() - startofCalc) % sampleSize %
                         (omp_get_thread_num());
        cout << endl;
      } else {
        AvgenergyEN += energyEN;
        currentIter++;
        ofs << format("%6i  %14.8f  %14.8f %14.8f   %10.2f  %10i %4i") %
                   (currentIter) % (E0 - energyEN) % (norm) %
                   (E0 - AvgenergyEN / AverageDen) % (getTime() - startofCalc) %
                   sampleSize % (omp_get_thread_num());
        ofs << endl;
      }

      // AverageDen += norm;
      // AvgenergyEN += energyEN; currentIter++;
      // std::cout << format("%6i  %14.8f  %14.8f  %14.8f   %10.2f  %10i %4i")
      //%(currentIter) % (E0-energyEN) % (norm) % (E0-AvgenergyEN/currentIter) %
      //(getTime()-startofCalc) % sampleSize % (omp_get_thread_num()); cout <<
      // endl;
      //%(currentIter) % (E0-AvgenergyEN/currentIter) % (norm) %
      //(E0-AvgenergyEN/AverageDen) % (getTime()-startofCalc) % sampleSize %
      //(omp_get_thread_num()); cout << endl;
    }
  }
  ofs.close();
}

void SHCIbasics::DoPerturbativeStochasticSingleList(
    vector<Determinant> &Dets, MatrixXx &ci, double &E0, oneInt &I1, twoInt &I2,
    twoIntHeatBath &I2HB, vector<int> &irrep, schedule &schd, double coreE,
    int nelec) {

  boost::mpi::communicator world;
  char file[5000];
  sprintf(file, "output-%d.bkp", world.rank());
  std::ofstream ofs(file);

  int norbs = Determinant::norbs;
  std::vector<Determinant> SortedDets = Dets;
  std::sort(SortedDets.begin(), SortedDets.end());
  int niter = 10000;
  // double eps = 0.001;
  double AvgenergyEN = 0.0;
  double AverageDen = 0.0;
  int currentIter = 0;
  int sampleSize = 0;
  int num_thrds = omp_get_max_threads();

#pragma omp parallel for schedule(dynamic)
  for (int iter = 0; iter < niter; iter++) {
    // cout << norbs<<"  "<<nelec<<endl;
    char psiArray[norbs];
    vector<int> psiClosed(nelec, 0);
    vector<int> psiOpen(norbs - nelec, 0);
    // char psiArray[norbs];
    std::vector<double> wts1;
    std::vector<int> Sample1;
    wts1.reserve(1000);
    Sample1.reserve(1000);

    Sample1.resize(0);
    wts1.resize(0);
    sample_round(ci, schd.eps, Sample1, wts1);

    map<Determinant, pair<double, double>> Psi1ab;
    for (int i = 0; i < Sample1.size(); i++) {
      int I = Sample1[i];
      SHCIbasics::getDeterminants(Dets[I], abs(schd.epsilon2 / ci(I, 0)),
                                  wts1[i], ci(I, 0), I1, I2, I2HB, irrep, coreE,
                                  E0, Psi1ab, SortedDets, schd);
    }

    double energyEN = 0.0;
    for (map<Determinant, pair<double, double>>::iterator it = Psi1ab.begin();
         it != Psi1ab.end(); it++) {
      it->first.getOpenClosed(psiOpen, psiClosed);
      energyEN += (it->second.first * it->second.first - it->second.second) /
                  (Energy(psiClosed, nelec, I1, I2, coreE) - E0);
    }
    sampleSize = Sample1.size();

#pragma omp critical
    {
      if (mpigetrank() == 0) {
        AvgenergyEN += energyEN;
        currentIter++;
        std::cout << format("%6i  %14.8f  %14.8f %14.8f   %10.2f  %10i %4i") %
                         (currentIter) % (E0 - energyEN) % (1.0) %
                         (E0 - AvgenergyEN / currentIter) %
                         (getTime() - startofCalc) % sampleSize %
                         (omp_get_thread_num());
        cout << endl;
      } else {
        AvgenergyEN += energyEN;
        currentIter++;
        ofs << format("%6i  %14.8f  %14.8f %14.8f   %10.2f  %10i %4i") %
                   (currentIter) % (E0 - energyEN) % (1.0) %
                   (E0 - AvgenergyEN / AverageDen) % (getTime() - startofCalc) %
                   sampleSize % (omp_get_thread_num());
        ofs << endl;
      }

      // AverageDen += norm;
      // AvgenergyEN += energyEN; currentIter++;
      // std::cout << format("%6i  %14.8f  %14.8f  %14.8f   %10.2f  %10i %4i")
      //%(currentIter) % (E0-energyEN) % (norm) % (E0-AvgenergyEN/currentIter) %
      //(getTime()-startofCalc) % sampleSize % (omp_get_thread_num()); cout <<
      // endl;
      //%(currentIter) % (E0-AvgenergyEN/currentIter) % (norm) %
      //(E0-AvgenergyEN/AverageDen) % (getTime()-startofCalc) % sampleSize %
      //(omp_get_thread_num()); cout << endl;
    }
  }
  ofs.close();
}

class sort_indices {
private:
  Determinant *mparr;

public:
  sort_indices(Determinant *parr) : mparr(parr) {}
  bool operator()(int i, int j) const { return mparr[i] < mparr[j]; }
};

// this function is complicated because I wanted to make it general enough that
// deterministicperturbative and stochasticperturbative could use the same
// function in stochastic perturbative each determinant in Psi1 can come from
// the first replica of Psi0 or the second replica of Psi0. that is why you have
// a pair of doubles associated with each det in Psi1 and we pass ci1 and ci2
// which are the coefficients of d in replica 1 and replica2 of Psi0.
void SHCIbasics::getDeterminants(
    Determinant &d, double epsilon, double ci1, double ci2, oneInt &int1,
    twoInt &int2, twoIntHeatBath &I2hb, vector<int> &irreps, double coreE,
    double E0, std::map<Determinant, pair<double, double>> &Psi1,
    std::vector<Determinant> &Psi0, schedule &schd, int Nmc) {

  int norbs = d.norbs;
  int open[norbs], closed[norbs];
  char detArray[norbs], diArray[norbs];
  int nclosed = d.getOpenClosed(open, closed);
  int nopen = norbs - nclosed;
  d.getRepArray(detArray);

  std::map<Determinant, pair<double, double>>::iterator det_it;
  for (int ia = 0; ia < nopen * nclosed; ia++) {
    int i = ia / nopen, a = ia % nopen;
    if (open[a] / 2 > schd.nvirt + nclosed / 2)
      continue; // dont occupy above a certain orbital
    if (irreps[closed[i] / 2] != irreps[open[a] / 2])
      continue;
    double integral =
        Hij_1Excite(closed[i], open[a], int1, int2, detArray, norbs);
    if (fabs(integral) > epsilon) {
      Determinant di = d;
      di.setocc(open[a], true);
      di.setocc(closed[i], false);
      if (!(binary_search(Psi0.begin(), Psi0.end(), di))) {
        det_it = Psi1.find(di);

        if (schd.singleList && schd.SampleN != -1) {
          if (det_it == Psi1.end())
            Psi1[di] =
                make_pair(integral * ci1, integral * integral * ci1 *
                                              (ci1 * Nmc / (Nmc - 1) - ci2));
          else {
            det_it->second.first += integral * ci1;
            det_it->second.second +=
                integral * integral * ci1 * (ci1 * Nmc / (Nmc - 1) - ci2);
          }
        } else if (schd.singleList && schd.SampleN == -1) {
          if (det_it == Psi1.end())
            Psi1[di] = make_pair(integral * ci1, integral * integral * ci2 *
                                                     ci1 * (ci1 / ci2 - 1.));
          else {
            det_it->second.first += integral * ci1;
            det_it->second.second +=
                integral * integral * ci2 * ci1 * (ci1 / ci2 - 1.);
          }
        } else {
          if (det_it == Psi1.end())
            Psi1[di] = make_pair(integral * ci1, integral * ci2);
          else {
            det_it->second.first += integral * ci1;
            det_it->second.second += integral * ci2;
          }
        }
      }
    }
  }

  if (fabs(int2.maxEntry) < epsilon)
    return;

  //#pragma omp parallel for schedule(dynamic)
  for (int ij = 0; ij < nclosed * nclosed; ij++) {

    int i = ij / nclosed, j = ij % nclosed;
    if (i <= j)
      continue;
    int I = closed[i] / 2, J = closed[j] / 2;
    std::pair<int, int> IJpair(max(I, J), min(I, J));
    std::map<std::pair<int, int>,
             std::multimap<double, std::pair<int, int>, compAbs>>::iterator
        ints = closed[i] % 2 == closed[j] % 2 ? I2hb.sameSpin.find(IJpair)
                                              : I2hb.oppositeSpin.find(IJpair);

    // THERE IS A BUG IN THE CODE WHEN USING HEATBATH INTEGRALS
    if (true && (ints != I2hb.sameSpin.end() &&
                 ints != I2hb.oppositeSpin.end())) { // we have this pair stored
                                                     // in heat bath integrals
      for (std::multimap<double, std::pair<int, int>, compAbs>::reverse_iterator
               it = ints->second.rbegin();
           it != ints->second.rend(); it++) {
        if (fabs(it->first) < epsilon)
          break; // if this is small then all subsequent ones will be small
        int a = 2 * it->second.first + closed[i] % 2,
            b = 2 * it->second.second + closed[j] % 2;
        if (a / 2 > schd.nvirt + nclosed / 2 ||
            b / 2 > schd.nvirt + nclosed / 2)
          continue; // dont occupy above a certain orbital
        // cout << a/2<<"  "<<schd.nvirt<<"  "<<nclosed/2<<endl;
        if (!(d.getocc(a) || d.getocc(b))) {
          Determinant di = d;
          di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i], false),
              di.setocc(closed[j], false);
          if (!(binary_search(Psi0.begin(), Psi0.end(), di))) {
            double sgn = 1.0;
            {
              int A = (closed[i]), B = closed[j], I = a, J = b;
              sgn = parity(detArray, norbs, A) * parity(detArray, norbs, I) *
                    parity(detArray, norbs, B) * parity(detArray, norbs, J);
              if (B > J)
                sgn *= -1;
              if (I > J)
                sgn *= -1;
              if (I > B)
                sgn *= -1;
              if (A > J)
                sgn *= -1;
              if (A > B)
                sgn *= -1;
              if (A > I)
                sgn *= -1;
            }

            det_it = Psi1.find(di);

            if (schd.singleList && schd.SampleN != -1) {
              if (det_it == Psi1.end())
                Psi1[di] = make_pair(it->first * sgn * ci1,
                                     it->first * it->first * ci1 *
                                         (ci1 * Nmc / (Nmc - 1) - ci2));
              else {
                det_it->second.first += it->first * sgn * ci1;
                det_it->second.second +=
                    it->first * it->first * ci1 * (ci1 * Nmc / (Nmc - 1) - ci2);
              }
            } else if (schd.singleList && schd.SampleN == -1) {
              if (det_it == Psi1.end())
                Psi1[di] = make_pair(it->first * sgn * ci1,
                                     it->first * it->first * ci1 * (ci1 - ci2));
              else {
                det_it->second.first += it->first * sgn * ci1;
                det_it->second.second +=
                    it->first * it->first * ci1 * (ci1 - ci2);
              }
            } else {
              if (det_it == Psi1.end())
                Psi1[di] =
                    make_pair(it->first * sgn * ci1, it->first * sgn * ci2);
              else {
                det_it->second.first += it->first * sgn * ci1;
                det_it->second.second += it->first * sgn * ci2;
              }
            }
          }
        }
      }
    }
  }
  return;
}

// this function is complicated because I wanted to make it general enough that
// deterministicperturbative and stochasticperturbative could use the same
// function in stochastic perturbative each determinant in Psi1 can come from
// the first replica of Psi0 or the second replica of Psi0. that is why you have
// a pair of doubles associated with each det in Psi1 and we pass ci1 and ci2
// which are the coefficients of d in replica 1 and replica2 of Psi0.
void SHCIbasics::getDeterminants(
    Determinant &d, double epsilon, double ci1, double ci2, oneInt &int1,
    twoInt &int2, twoIntHeatBath &I2hb, vector<int> &irreps, double coreE,
    double E0, std::map<Determinant, std::tuple<double, double, double>> &Psi1,
    std::vector<Determinant> &Psi0, schedule &schd, int Nmc, int nelec) {

  int norbs = d.norbs;
  vector<int> closed(nelec, 0);
  vector<int> open(norbs - nelec, 0);
  d.getOpenClosed(open, closed);
  int nclosed = nelec;
  char detArray[norbs], diArray[norbs];
  int nopen = norbs - nclosed;
  d.getRepArray(detArray);

  double Energyd = Energy(closed, nclosed, int1, int2, coreE);

  std::map<Determinant, std::tuple<double, double, double>>::iterator det_it;
  for (int ia = 0; ia < nopen * nclosed; ia++) {
    int i = ia / nopen, a = ia % nopen;
    if (open[a] / 2 > schd.nvirt + nclosed / 2)
      continue; // dont occupy above a certain orbital
    if (irreps[closed[i] / 2] != irreps[open[a] / 2])
      continue;

    // double integral = Hij_1Excite(closed[i],open[a],int1,int2, detArray,
    // norbs);
    double integral = d.Hij_1Excite(closed[i], open[a], int1, int2);

    if (fabs(integral) > epsilon) {
      Determinant di = d;
      di.setocc(open[a], true);
      di.setocc(closed[i], false);
      if (!(binary_search(Psi0.begin(), Psi0.end(), di))) {
        det_it = Psi1.find(di);

        if (schd.singleList && schd.SampleN != -1) {
          if (det_it == Psi1.end()) {
            double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE,
                                             i, open[a], Energyd);
            Psi1[di] = std::tuple<double, double, double>(
                integral * ci1,
                integral * integral * ci1 * (ci1 * Nmc / (Nmc - 1) - ci2), E);
          } else {
            std::get<0>(det_it->second) += integral * ci1;
            std::get<1>(det_it->second) +=
                integral * integral * ci1 * (ci1 * Nmc / (Nmc - 1) - ci2);
          }
        }
      }
    }
  }

  if (fabs(int2.maxEntry) < epsilon)
    return;

  //#pragma omp parallel for schedule(dynamic)
  for (int ij = 0; ij < nclosed * nclosed; ij++) {

    int i = ij / nclosed, j = ij % nclosed;
    if (i <= j)
      continue;
    int I = closed[i] / 2, J = closed[j] / 2;
    std::pair<int, int> IJpair(max(I, J), min(I, J));
    std::map<std::pair<int, int>,
             std::multimap<double, std::pair<int, int>, compAbs>>::iterator
        ints = closed[i] % 2 == closed[j] % 2 ? I2hb.sameSpin.find(IJpair)
                                              : I2hb.oppositeSpin.find(IJpair);

    if (true && (ints != I2hb.sameSpin.end() &&
                 ints != I2hb.oppositeSpin.end())) { // we have this pair stored
                                                     // in heat bath integrals
      for (std::multimap<double, std::pair<int, int>, compAbs>::reverse_iterator
               it = ints->second.rbegin();
           it != ints->second.rend(); it++) {
        if (fabs(it->first) < epsilon)
          break; // if this is small then all subsequent ones will be small
        int a = 2 * it->second.first + closed[i] % 2,
            b = 2 * it->second.second + closed[j] % 2;
        if (a / 2 > schd.nvirt + nclosed / 2 ||
            b / 2 > schd.nvirt + nclosed / 2)
          continue; // dont occupy above a certain orbital

        // cout << a/2<<"  "<<schd.nvirt<<"  "<<nclosed/2<<endl;
        if (!(d.getocc(a) || d.getocc(b))) {
          Determinant di = d;
          di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i], false),
              di.setocc(closed[j], false);
          if (!(binary_search(Psi0.begin(), Psi0.end(), di))) {

            double sgn = 1.0;
            di.parity(a, b, closed[i], closed[j], sgn);

            det_it = Psi1.find(di);

            if (schd.singleList && schd.SampleN != -1) {
              if (det_it == Psi1.end()) {
                double E = EnergyAfterExcitation(closed, nclosed, int1, int2,
                                                 coreE, i, a, j, b, Energyd);
                Psi1[di] = std::tuple<double, double, double>(
                    it->first * sgn * ci1,
                    it->first * it->first * ci1 * (ci1 * Nmc / (Nmc - 1) - ci2),
                    E);
              } else {
                std::get<0>(det_it->second) += it->first * sgn * ci1;
                std::get<1>(det_it->second) +=
                    it->first * it->first * ci1 * (ci1 * Nmc / (Nmc - 1) - ci2);
              }
            }
          }
        }
      }
    }
  }
  return;
}

// this function is complicated because I wanted to make it general enough that
// deterministicperturbative and stochasticperturbative could use the same
// function in stochastic perturbative each determinant in Psi1 can come from
// the first replica of Psi0 or the second replica of Psi0. that is why you have
// a pair of doubles associated with each det in Psi1 and we pass ci1 and ci2
// which are the coefficients of d in replica 1 and replica2 of Psi0.
void SHCIbasics::getDeterminants2Epsilon(
    Determinant &d, double epsilon, double epsilonLarge, double ci1, double ci2,
    oneInt &int1, twoInt &int2, twoIntHeatBath &I2hb, vector<int> &irreps,
    double coreE, double E0,
    std::map<Determinant, std::tuple<double, double, double, double, double>>
        &Psi1,
    std::vector<Determinant> &Psi0, schedule &schd, int Nmc, int nelec) {

  int norbs = d.norbs;
  vector<int> closed(nelec, 0);
  vector<int> open(norbs - nelec, 0);
  d.getOpenClosed(open, closed);
  int nclosed = nelec;
  char detArray[norbs], diArray[norbs];
  int nopen = norbs - nclosed;
  d.getRepArray(detArray);

  double Energyd = Energy(closed, nclosed, int1, int2, coreE);

  std::map<Determinant,
           std::tuple<double, double, double, double, double>>::iterator det_it;
  for (int ia = 0; ia < nopen * nclosed; ia++) {
    int i = ia / nopen, a = ia % nopen;
    if (open[a] / 2 > schd.nvirt + nclosed / 2)
      continue; // dont occupy above a certain orbital
    if (irreps[closed[i] / 2] != irreps[open[a] / 2])
      continue;
    double integral =
        Hij_1Excite(closed[i], open[a], int1, int2, detArray, norbs);
    if (fabs(integral) > epsilon) {
      Determinant di = d;
      di.setocc(open[a], true);
      di.setocc(closed[i], false);
      if (!(binary_search(Psi0.begin(), Psi0.end(), di))) {
        det_it = Psi1.find(di);

        if (schd.singleList && schd.SampleN != -1) {
          if (det_it == Psi1.end()) {
            double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE,
                                             i, open[a], Energyd);
            Psi1[di] = std::tuple<double, double, double, double, double>(
                integral * ci1,
                integral * integral * ci1 * (ci1 * Nmc / (Nmc - 1) - ci2), E,
                0.0, 0.0);
          } else {
            std::get<0>(det_it->second) += integral * ci1;
            std::get<1>(det_it->second) +=
                integral * integral * ci1 * (ci1 * Nmc / (Nmc - 1) - ci2);
          }
        }

        if (fabs(integral) > epsilonLarge) {
          det_it = Psi1.find(di);
          std::get<3>(det_it->second) += integral * ci1;
          std::get<4>(det_it->second) +=
              integral * integral * ci1 * (ci1 * Nmc / (Nmc - 1) - ci2);
        }
      }
    }
  }

  if (fabs(int2.maxEntry) < epsilon)
    return;

  //#pragma omp parallel for schedule(dynamic)
  for (int ij = 0; ij < nclosed * nclosed; ij++) {

    int i = ij / nclosed, j = ij % nclosed;
    if (i <= j)
      continue;
    int I = closed[i] / 2, J = closed[j] / 2;
    std::pair<int, int> IJpair(max(I, J), min(I, J));
    std::map<std::pair<int, int>,
             std::multimap<double, std::pair<int, int>, compAbs>>::iterator
        ints = closed[i] % 2 == closed[j] % 2 ? I2hb.sameSpin.find(IJpair)
                                              : I2hb.oppositeSpin.find(IJpair);

    // THERE IS A BUG IN THE CODE WHEN USING HEATBATH INTEGRALS
    if (true && (ints != I2hb.sameSpin.end() &&
                 ints != I2hb.oppositeSpin.end())) { // we have this pair stored
                                                     // in heat bath integrals
      for (std::multimap<double, std::pair<int, int>, compAbs>::reverse_iterator
               it = ints->second.rbegin();
           it != ints->second.rend(); it++) {
        if (fabs(it->first) < epsilon)
          break; // if this is small then all subsequent ones will be small
        int a = 2 * it->second.first + closed[i] % 2,
            b = 2 * it->second.second + closed[j] % 2;
        if (a / 2 > schd.nvirt + nclosed / 2 ||
            b / 2 > schd.nvirt + nclosed / 2)
          continue; // dont occupy above a certain orbital

        // cout << a/2<<"  "<<schd.nvirt<<"  "<<nclosed/2<<endl;
        if (!(d.getocc(a) || d.getocc(b))) {
          Determinant di = d;
          di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i], false),
              di.setocc(closed[j], false);
          if (!(binary_search(Psi0.begin(), Psi0.end(), di))) {

            double sgn = 1.0;
            {
              int A = (closed[i]), B = closed[j], I = a, J = b;
              sgn = parity(detArray, norbs, A) * parity(detArray, norbs, I) *
                    parity(detArray, norbs, B) * parity(detArray, norbs, J);
              if (B > J)
                sgn *= -1;
              if (I > J)
                sgn *= -1;
              if (I > B)
                sgn *= -1;
              if (A > J)
                sgn *= -1;
              if (A > B)
                sgn *= -1;
              if (A > I)
                sgn *= -1;
            }

            det_it = Psi1.find(di);

            if (schd.singleList && schd.SampleN != -1) {
              if (det_it == Psi1.end()) {
                double E = EnergyAfterExcitation(closed, nclosed, int1, int2,
                                                 coreE, i, a, j, b, Energyd);
                Psi1[di] = std::tuple<double, double, double, double, double>(
                    it->first * sgn * ci1,
                    it->first * it->first * ci1 * (ci1 * Nmc / (Nmc - 1) - ci2),
                    E, 0.0, 0.0);
              } else {
                std::get<0>(det_it->second) += it->first * sgn * ci1;
                std::get<1>(det_it->second) +=
                    it->first * it->first * ci1 * (ci1 * Nmc / (Nmc - 1) - ci2);
              }
            }

            if (fabs(it->first) > epsilonLarge) {
              det_it = Psi1.find(di);
              std::get<3>(det_it->second) += it->first * sgn * ci1;
              std::get<4>(det_it->second) +=
                  it->first * it->first * ci1 * (ci1 * Nmc / (Nmc - 1) - ci2);
            }
          }
        }
      }
    }
  }
  return;
}

void SHCIbasics::updateConnections(vector<Determinant> &Dets,
                                   map<Determinant, int> &SortedDets, int norbs,
                                   oneInt &int1, twoInt &int2, double coreE,
                                   char *detArray,
                                   vector<vector<int>> &connections,
                                   vector<vector<double>> &Helements) {
  size_t prevSize = SortedDets.size();
  size_t Norbs = norbs;
  for (size_t i = prevSize; i < Dets.size(); i++) {
    SortedDets[Dets[i]] = i;
    connections[i].push_back(i);
    Helements[i].push_back(
        Energy(&detArray[i * Norbs], norbs, int1, int2, coreE));
  }

#pragma omp parallel for schedule(dynamic)
  for (size_t x = prevSize; x < Dets.size(); x++) {
    Determinant d = Dets[x];
    int open[norbs], closed[norbs];
    int nclosed = d.getOpenClosed(open, closed);
    int nopen = norbs - nclosed;

    if (x % 10000 == 0)
      cout << "update connections " << x << " out of " << Dets.size() - prevSize
           << endl;
    // loop over all single excitation and find if they are present in the list
    // on or before the current determinant
    for (int ia = 0; ia < nopen * nclosed; ia++) {
      int i = ia / nopen, a = ia % nopen;
      Determinant di = d;
      di.setocc(open[a], true);
      di.setocc(closed[i], false);

      map<Determinant, int>::iterator it = SortedDets.find(di);
      if (it != SortedDets.end()) {
        int y = it->second;
        if (y <= x) { // avoid double counting
          double integral = Hij_1Excite(closed[i], open[a], int1, int2,
                                        &detArray[x * Norbs], norbs);
          if (abs(integral) > 1.e-8) {
            connections[x].push_back(y);
            Helements[x].push_back(integral);
          }
          // connections[y].push_back(x);
          // Helements[y].push_back(integral);
        }
      }
    }

    for (int i = 0; i < nclosed; i++)
      for (int j = 0; j < i; j++) {
        for (int a = 0; a < nopen; a++) {
          for (int b = 0; b < a; b++) {
            Determinant di = d;
            di.setocc(open[a], true), di.setocc(open[b], true),
                di.setocc(closed[i], false), di.setocc(closed[j], false);

            map<Determinant, int>::iterator it = SortedDets.find(di);
            if (it != SortedDets.end()) {
              int y = it->second;
              if (y <= x) { // avoid double counting
                double integral =
                    Hij_2Excite(closed[i], closed[j], open[a], open[b], int2,
                                &detArray[x * Norbs], norbs);
                if (abs(integral) > 1.e-8) {
                  connections[x].push_back(y);
                  Helements[x].push_back(integral);
                  // cout << x<<"  "<<y<<"  "<<integral<<endl;
                }
                // connections[y].push_back(x);
                // Helements[y].push_back(integral);
              }
            }
          }
        }
      }
  }
}
void SHCIbasics::DoPerturbativeStochastic2SingleListDoubleEpsilon2AllTogether(
    vector<Determinant> &Dets, MatrixXx &ci, double &E0, oneInt &I1, twoInt &I2,
    twoIntHeatBathSHM &I2HB, vector<int> &irrep, schedule &schd, double coreE,
    int nelec, int root) {

  boost::mpi::communicator world;

  double epsilon2 = schd.epsilon2;
  schd.epsilon2 = schd.epsilon2Large;
  double EptLarge = DoPerturbativeDeterministic(Dets, ci, E0, I1, I2, I2HB,
                                                irrep, schd, coreE, nelec);

  schd.epsilon2 = epsilon2;

  int norbs = Determinant::norbs;
  std::vector<Determinant> SortedDets = Dets;
  std::sort(SortedDets.begin(), SortedDets.end());
  int niter = schd.nPTiter;
  // double eps = 0.001;
  int Nsample = schd.SampleN;
  double AvgenergyEN = 0.0;
  double AverageDen = 0.0;
  int currentIter = 0;
  int sampleSize = 0;
  int num_thrds = omp_get_max_threads();

  double cumulative = 0.0;
  for (int i = 0; i < ci.rows(); i++)
    cumulative += abs(ci(i, 0));

  std::vector<int> alias;
  std::vector<double> prob;
  setUpAliasMethod(ci, cumulative, alias, prob);

  double totalPT = 0, totalPTLargeEps = 0;

  std::vector<std::vector<std::vector<vector<Determinant>>>> hashedDetBeforeMPI(
      mpigetsize(), std::vector<std::vector<vector<Determinant>>>(num_thrds));
  std::vector<std::vector<std::vector<vector<Determinant>>>> hashedDetAfterMPI(
      mpigetsize(), std::vector<std::vector<vector<Determinant>>>(num_thrds));
  std::vector<std::vector<std::vector<vector<double>>>> hashedNum1BeforeMPI(
      mpigetsize(), std::vector<std::vector<vector<double>>>(num_thrds));
  std::vector<std::vector<std::vector<vector<double>>>> hashedNum1AfterMPI(
      mpigetsize(), std::vector<std::vector<vector<double>>>(num_thrds));
  std::vector<std::vector<std::vector<vector<double>>>> hashedNum2BeforeMPI(
      mpigetsize(), std::vector<std::vector<vector<double>>>(num_thrds));
  std::vector<std::vector<std::vector<vector<double>>>> hashedNum2AfterMPI(
      mpigetsize(), std::vector<std::vector<vector<double>>>(num_thrds));
  std::vector<std::vector<std::vector<vector<double>>>> hashedEnergyBeforeMPI(
      mpigetsize(), std::vector<std::vector<vector<double>>>(num_thrds));
  std::vector<std::vector<std::vector<vector<double>>>> hashedEnergyAfterMPI(
      mpigetsize(), std::vector<std::vector<vector<double>>>(num_thrds));
  std::vector<std::vector<std::vector<vector<char>>>> hashedpresentBeforeMPI(
      mpigetsize(), std::vector<std::vector<vector<char>>>(num_thrds));
  std::vector<std::vector<std::vector<vector<char>>>> hashedpresentAfterMPI(
      mpigetsize(), std::vector<std::vector<vector<char>>>(num_thrds));

#pragma omp parallel
  {
    for (int iter = 0; iter < niter; iter++) {
      std::vector<CItype> wts1(Nsample, 0.0);
      std::vector<int> Sample1(Nsample, -1);
      int distinctSample =
          sample_N2_alias(ci, cumulative, Sample1, wts1, alias, prob);
      for (int i = 0; i < Nsample; i++)
        wts1[i] /= (num_thrds * mpigetsize());
      int Nmc = Nsample;
      double norm = 0.0;

      std::vector<Determinant> Psi1;
      std::vector<CItype> numerator1A;
      vector<double> numerator2A;
      vector<char> present;
      std::vector<double> det_energy;

      for (int i = 0; i < distinctSample; i++) {
        int I = Sample1[i];
        SHCIbasics::getDeterminants2Epsilon(
            Dets[I], schd.epsilon2 / abs(ci(I, 0)),
            schd.epsilon2Large / abs(ci(I, 0)), wts1[i], ci(I, 0), I1, I2, I2HB,
            irrep, coreE, E0, Psi1, numerator1A, numerator2A, present,
            det_energy, schd, Nmc, nelec);
      }

      std::vector<Determinant> Psi1copy = Psi1;
      vector<long> detIndex(Psi1.size(), 0);
      vector<long> detIndexcopy(Psi1.size(), 0);
      for (size_t i = 0; i < Psi1.size(); i++)
        detIndex[i] = i;
      mergesort(&Psi1copy[0], 0, Psi1.size() - 1, &detIndex[0],
                &(Psi1.operator[](0)), &detIndexcopy[0]);
      detIndexcopy.clear();
      Psi1copy.clear();
      reorder(numerator1A, detIndex);
      reorder(numerator2A, detIndex);
      reorder(det_energy, detIndex);
      reorder(present, detIndex);
      detIndex.clear();

      // quickSort( &(Psi1[0]), 0, Psi1.size(), &numerator1A[0],
      // &numerator2A[0], &det_energy, &present);
      RemoveDetsPresentIn(SortedDets, Psi1, numerator1A, numerator2A,
                          det_energy, present);

      if (mpigetsize() > 1 || num_thrds > 1) {
        for (int proc = 0; proc < mpigetsize(); proc++) {
          hashedDetBeforeMPI[proc][omp_get_thread_num()].resize(num_thrds);
          hashedNum1BeforeMPI[proc][omp_get_thread_num()].resize(num_thrds);
          hashedNum2BeforeMPI[proc][omp_get_thread_num()].resize(num_thrds);
          hashedEnergyBeforeMPI[proc][omp_get_thread_num()].resize(num_thrds);
          hashedpresentBeforeMPI[proc][omp_get_thread_num()].resize(num_thrds);
        }
        for (int j = 0; j < Psi1.size(); j++) {
          size_t lOrder = Psi1.at(j).getHash();
          size_t procThrd = lOrder % (mpigetsize() * num_thrds);
          int proc = abs(procThrd / num_thrds),
              thrd = abs(procThrd % num_thrds);
          hashedDetBeforeMPI[proc][omp_get_thread_num()][thrd].push_back(
              Psi1.at(j));
          hashedNum1BeforeMPI[proc][omp_get_thread_num()][thrd].push_back(
              numerator1A.at(j));
          hashedNum2BeforeMPI[proc][omp_get_thread_num()][thrd].push_back(
              numerator2A.at(j));
          hashedEnergyBeforeMPI[proc][omp_get_thread_num()][thrd].push_back(
              det_energy.at(j));
          hashedpresentBeforeMPI[proc][omp_get_thread_num()][thrd].push_back(
              present.at(j));
        }
        Psi1.clear();
        numerator1A.clear();
        numerator2A.clear();
        det_energy.clear();
        present.clear();

        // if (mpigetrank() == 0 && omp_get_thread_num() == 0) cout << "#After
        // hash "<<getTime()-startofCalc<<endl;

#pragma omp barrier
        if (omp_get_thread_num() == 0) {
          mpi::all_to_all(world, hashedDetBeforeMPI, hashedDetAfterMPI);
          for (int proc = 0; proc < mpigetsize(); proc++)
            hashedDetBeforeMPI[proc][omp_get_thread_num()].clear();
          mpi::all_to_all(world, hashedNum1BeforeMPI, hashedNum1AfterMPI);
          for (int proc = 0; proc < mpigetsize(); proc++)
            hashedNum1BeforeMPI[proc][omp_get_thread_num()].clear();
          mpi::all_to_all(world, hashedNum2BeforeMPI, hashedNum2AfterMPI);
          for (int proc = 0; proc < mpigetsize(); proc++)
            hashedNum2BeforeMPI[proc][omp_get_thread_num()].clear();
          mpi::all_to_all(world, hashedEnergyBeforeMPI, hashedEnergyAfterMPI);
          for (int proc = 0; proc < mpigetsize(); proc++)
            hashedEnergyBeforeMPI[proc][omp_get_thread_num()].clear();
          mpi::all_to_all(world, hashedpresentBeforeMPI, hashedpresentAfterMPI);
          for (int proc = 0; proc < mpigetsize(); proc++)
            hashedpresentBeforeMPI[proc][omp_get_thread_num()].clear();
        }
#pragma omp barrier

        for (int proc = 0; proc < mpigetsize(); proc++) {
          for (int thrd = 0; thrd < num_thrds; thrd++) {
            for (int j = 0;
                 j < hashedDetAfterMPI[proc][thrd][omp_get_thread_num()].size();
                 j++) {
              Psi1.push_back(
                  hashedDetAfterMPI[proc][thrd][omp_get_thread_num()].at(j));
              numerator1A.push_back(
                  hashedNum1AfterMPI[proc][thrd][omp_get_thread_num()].at(j));
              numerator2A.push_back(
                  hashedNum2AfterMPI[proc][thrd][omp_get_thread_num()].at(j));
              det_energy.push_back(
                  hashedEnergyAfterMPI[proc][thrd][omp_get_thread_num()].at(j));
              present.push_back(
                  hashedpresentAfterMPI[proc][thrd][omp_get_thread_num()].at(
                      j));
            }
            hashedDetAfterMPI[proc][thrd][omp_get_thread_num()].clear();
            hashedNum1AfterMPI[proc][thrd][omp_get_thread_num()].clear();
            hashedNum2AfterMPI[proc][thrd][omp_get_thread_num()].clear();
            hashedEnergyAfterMPI[proc][thrd][omp_get_thread_num()].clear();
            hashedpresentAfterMPI[proc][thrd][omp_get_thread_num()].clear();
          }
        }

        std::vector<Determinant> Psi1copy = Psi1;
        vector<long> detIndex(Psi1.size(), 0);
        vector<long> detIndexcopy(Psi1.size(), 0);
        for (size_t i = 0; i < Psi1.size(); i++)
          detIndex[i] = i;
        mergesort(&Psi1copy[0], 0, Psi1.size() - 1, &detIndex[0],
                  &(Psi1.operator[](0)), &detIndexcopy[0]);
        detIndexcopy.clear();
        Psi1copy.clear();
        reorder(numerator1A, detIndex);
        reorder(numerator2A, detIndex);
        reorder(det_energy, detIndex);
        reorder(present, detIndex);
        detIndex.clear();
        // quickSort( &(Psi1[0]), 0, Psi1.size(), &numerator1A[0],
        // &numerator2A[0], &det_energy, &present);
      }

      CItype currentNum1A = 0.;
      double currentNum2A = 0.;
      CItype currentNum1B = 0.;
      double currentNum2B = 0.;
      vector<Determinant>::iterator vec_it = SortedDets.begin();
      double energyEN = 0.0, energyENLargeEps = 0.0;
      size_t effNmc = mpigetsize() * num_thrds * Nmc;

      for (int i = 0; i < Psi1.size();) {
        if (Psi1[i] < *vec_it) {
          currentNum1A += numerator1A[i];
          currentNum2A += numerator2A[i];
          if (present[i]) {
            currentNum1B += numerator1A[i];
            currentNum2B += numerator2A[i];
          }

          if (i == Psi1.size() - 1) {
            energyEN +=
                (pow(abs(currentNum1A), 2) * Nmc / (Nmc - 1) - currentNum2A) /
                (det_energy[i] - E0);
            energyENLargeEps +=
                (pow(abs(currentNum1B), 2) * Nmc / (Nmc - 1) - currentNum2B) /
                (det_energy[i] - E0);
          } else if (!(Psi1[i] == Psi1[i + 1])) {
            energyEN +=
                (pow(abs(currentNum1A), 2) * Nmc / (Nmc - 1) - currentNum2A) /
                (det_energy[i] - E0);
            energyENLargeEps +=
                (pow(abs(currentNum1B), 2) * Nmc / (Nmc - 1) - currentNum2B) /
                (det_energy[i] - E0);
            currentNum1A = 0.;
            currentNum2A = 0.;
            currentNum1B = 0.;
            currentNum2B = 0.;
          }
          i++;
        } else if (*vec_it < Psi1[i] && vec_it != SortedDets.end())
          vec_it++;
        else if (*vec_it < Psi1[i] && vec_it == SortedDets.end()) {
          currentNum1A += numerator1A[i];
          currentNum2A += numerator2A[i];
          if (present[i]) {
            currentNum1B += numerator1A[i];
            currentNum2B += numerator2A[i];
          }

          if (i == Psi1.size() - 1) {
            energyEN +=
                (pow(abs(currentNum1A), 2) * Nmc / (Nmc - 1) - currentNum2A) /
                (det_energy[i] - E0);
            energyENLargeEps +=
                (pow(abs(currentNum1B), 2) * Nmc / (Nmc - 1) - currentNum2B) /
                (det_energy[i] - E0);
          } else if (!(Psi1[i] == Psi1[i + 1])) {
            energyEN +=
                (pow(abs(currentNum1A), 2) * Nmc / (Nmc - 1) - currentNum2A) /
                (det_energy[i] - E0);
            energyENLargeEps +=
                (pow(abs(currentNum1B), 2) * Nmc / (Nmc - 1) - currentNum2B) /
                (det_energy[i] - E0);
            // energyEN += (currentNum1A*currentNum1A*Nmc/(Nmc-1) -
            // currentNum2A)/(det_energy[i] - E0); energyENLargeEps +=
            // (currentNum1B*currentNum1B*Nmc/(Nmc-1) -
            // currentNum2B)/(det_energy[i] - E0);
            currentNum1A = 0.;
            currentNum2A = 0.;
            currentNum1B = 0.;
            currentNum2B = 0.;
          }
          i++;
        } else {
          if (Psi1[i] == Psi1[i + 1])
            i++;
          else {
            vec_it++;
            i++;
          }
        }
      }

      totalPT = 0;
      totalPTLargeEps = 0;
#pragma omp barrier
#pragma omp critical
      {
        totalPT += energyEN;
        totalPTLargeEps += energyENLargeEps;
      }
#pragma omp barrier

      double finalE = 0., finalELargeEps = 0;
      if (omp_get_thread_num() == 0)
        mpi::all_reduce(world, totalPT, finalE, std::plus<double>());
      if (omp_get_thread_num() == 0)
        mpi::all_reduce(world, totalPTLargeEps, finalELargeEps,
                        std::plus<double>());

      if (mpigetrank() == 0 && omp_get_thread_num() == 0) {
        AvgenergyEN += -finalE + finalELargeEps + EptLarge;
        currentIter++;
        std::cout << format("%6i  %14.8f  %s%i %14.8f   %10.2f  %10i") %
                         (currentIter) %
                         (E0 - finalE + finalELargeEps + EptLarge) % ("Root") %
                         root % (E0 + AvgenergyEN / currentIter) %
                         (getTime() - startofCalc) % sampleSize;
        cout << endl;
      }
    }
  }
}

void getDeterminantsDeterministicPTInt1(
    Determinant det, int det_ind, double epsilon1, CItype ci1, double epsilon2,
    CItype ci2, oneInt &int1a, oneInt &int1, twoInt &int2, vector<int> &irreps,
    double coreE, std::vector<Determinant> &dets,
    std::vector<CItype> &numerator1, std::vector<CItype> &numerator2,
    std::vector<double> &energy, schedule &schd, int nelec) {

  int norbs = det.norbs;
  vector<int> closed(nelec, 0);
  vector<int> open(norbs - nelec, 0);
  det.getOpenClosed(open, closed);
  int nclosed = nelec;
  int nopen = norbs - nclosed;
  size_t orbDiff;

  for (int ia = 0; ia < nopen * nclosed; ia++) {
    int i = ia / nopen, a = ia % nopen;

    double sgn = 1.0;
    det.parity(min(open[a], closed[i]), max(open[a], closed[i]), sgn);
    CItype integral = int1a(open[a], closed[i]) * sgn;

    if (fabs(integral) > epsilon1 || fabs(integral) > epsilon2) {
      dets.push_back(det);
      Determinant &di = *dets.rbegin();
      di.setocc(open[a], true);
      di.setocc(closed[i], false);

      double E = di.Energy(int1, int2, coreE);
      energy.push_back(E);

      // if(fabs(integral) >epsilon1)
      numerator1.push_back(integral * ci1);
      // else
      // numerator1.push_back(0.0);

      // if(fabs(integral) > epsilon2)
      numerator2.push_back(integral * ci2);
      // else
      // numerator2.push_back(0.0);
    }
  }
  return;
}
