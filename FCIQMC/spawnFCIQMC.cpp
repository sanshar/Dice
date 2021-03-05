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

#include <iostream>
#include <vector>
#include <boost/serialization/serialization.hpp>
#include <Eigen/Dense>
#include "global.h"
#include "input.h"
#include "spawnFCIQMC.h"
#include "walkersFCIQMC.h"

#include "CorrelatedWavefunction.h"
#include "Jastrow.h"
#include "Slater.h"
#include "SelectedCI.h"
#include "trivialWF.h"
#include "trivialWalk.h"

template <typename T, typename Compare>
vector<size_t> sort_permutation(int const nDets, const vector<T>& vec, Compare compare)
{
  vector<size_t> p(nDets);
  iota(p.begin(), p.end(), 0);
  sort(p.begin(), p.end(),
    [&](size_t i, size_t j){ return compare(vec[i], vec[j]); });
  return p;
}

template <typename T>
void apply_permutation(int const nDets, const vector<T>& vec, vector<T>& sorted_vec, const vector<size_t>& p)
{
  transform(p.begin(), p.end(), sorted_vec.begin(),
    [&](size_t i){ return vec[i]; });
}

void apply_permutation(int const nDets, double** amps, double** sorted_amps, const vector<size_t>& p)
{
  for (int iDet = 0; iDet<nDets; iDet++) {
    int sorted_ind = p.at(iDet);
    for (int iReplica=0; iReplica<schd.nreplicas; iReplica++) {
      sorted_amps[iDet][iReplica] = amps[sorted_ind][iReplica];
    }
  }
}

spawnFCIQMC::spawnFCIQMC(int spawnSize, int DetLenLocal, int nreplicasLocal) {
  init(spawnSize, DetLenLocal, nreplicasLocal);
}

void spawnFCIQMC::init(int spawnSize, int DetLenLocal, int nreplicasLocal) {
  nDets = 0;
  nreplicas = nreplicasLocal;
  dets.resize(spawnSize);
  detsTemp.resize(spawnSize);
  amps = allocateAmpsArray(spawnSize, nreplicas, 0.0);
  ampsTemp = allocateAmpsArray(spawnSize, nreplicas, 0.0);

  if (schd.initiator) {
    flags.resize(spawnSize, 0);
    flagsTemp.resize(spawnSize, 0);
  }

  firstProcSlots.resize(commsize);
  currProcSlots.resize(commsize);

  nSlotsPerProc = spawnSize / commsize;
  for (int i=0; i<commsize; i++) {
    firstProcSlots[i] = i*nSlotsPerProc;
  }

  currProcSlots = firstProcSlots;

  DetLenMin = DetLenLocal;
}

spawnFCIQMC::~spawnFCIQMC() {
  dets.clear();
  detsTemp.clear();
  deleteAmpsArray(amps);
  deleteAmpsArray(ampsTemp);
  flags.clear();
  flagsTemp.clear();
  firstProcSlots.clear();
  currProcSlots.clear();
}

// Send spawned walkers to their correct processor
void spawnFCIQMC::communicate() {
#ifdef SERIAL
  nDets = currProcSlots[0] - firstProcSlots[0];
  // Copy spawning data to temp arrays
  for (int iDet=0; iDet<nDets; iDet++) {
    detsTemp[iDet] = dets[iDet];
    for (int iReplica = 0; iReplica<nreplicas; iReplica++) {
      ampsTemp[iDet][iReplica] = amps[iDet][iReplica];
    }
    if (schd.initiator) flagsTemp[iDet] = flags[iDet];
  }
#else
  int sendCounts[commsize], recvCounts[commsize];
  int sendDispls[commsize], recvDispls[commsize];
  int sendCountsDets[commsize], recvCountsDets[commsize];
  int sendDisplsDets[commsize], recvDisplsDets[commsize];
  int sendCountsAmps[commsize], recvCountsAmps[commsize];
  int sendDisplsAmps[commsize], recvDisplsAmps[commsize];

  // The number of determinants to send to each processor, and their
  // displacements in the spawning list
  for (int proc=0; proc<commsize; proc++) {
    sendCounts[proc] = currProcSlots[proc] - firstProcSlots[proc];
    sendDispls[proc] = firstProcSlots[proc];
  }
  // Communicate the number of dets to be sent and received
  MPI_Alltoall(sendCounts, 1, MPI_INTEGER, recvCounts, 1, MPI_INTEGER, MPI_COMM_WORLD);

  // Displacements of dets about to be received
  recvDispls[0] = 0;
  for (int proc=1; proc<commsize; proc++) {
    recvDispls[proc] = recvDispls[proc-1] + recvCounts[proc-1];
  }

  // Dets have width of 2*DetLen
  // They are stored contiguously in the vector, as required for MPI
  for (int proc=0; proc<commsize; proc++) {
    sendCountsDets[proc] = sendCounts[proc] * 2*DetLen;
    recvCountsDets[proc] = recvCounts[proc] * 2*DetLen;
    sendDisplsDets[proc] = sendDispls[proc] * 2*DetLen;
    recvDisplsDets[proc] = recvDispls[proc] * 2*DetLen;

    sendCountsAmps[proc] = sendCounts[proc] * nreplicas;
    recvCountsAmps[proc] = recvCounts[proc] * nreplicas;
    sendDisplsAmps[proc] = sendDispls[proc] * nreplicas;
    recvDisplsAmps[proc] = recvDispls[proc] * nreplicas;
  }

  MPI_Alltoallv(&dets.front(), sendCountsDets, sendDisplsDets, MPI_LONG,
                &detsTemp.front(), recvCountsDets, recvDisplsDets, MPI_LONG, MPI_COMM_WORLD);

  MPI_Alltoallv(&(amps[0][0]), sendCountsAmps, sendDisplsAmps, MPI_DOUBLE,
                &(ampsTemp[0][0]), recvCountsAmps, recvDisplsAmps, MPI_DOUBLE, MPI_COMM_WORLD);

  if (schd.initiator) {
    MPI_Alltoallv(&flags.front(), sendCounts, sendDispls, MPI_INTEGER,
                  &flagsTemp.front(), recvCounts, recvDispls, MPI_INTEGER, MPI_COMM_WORLD);
  }

  // The total number of determinants received
  nDets = recvDispls[commsize-1] + recvCounts[commsize-1];

#endif
}

// Merge multiple spawned walkers to the same determinant, so that each
// determinant only appears once
void spawnFCIQMC::compress(vector<double>& nAnnihil) {

  if (nDets > 0) {
    // Perform sort
    auto p = sort_permutation(nDets, detsTemp, [](simpleDet const& a, simpleDet const& b){ return (a < b); });

    apply_permutation( nDets, detsTemp, dets, p );
    apply_permutation( nDets, ampsTemp, amps, p );

    if (schd.initiator) {
      apply_permutation( nDets, flagsTemp, flags, p );
    }

    bool exitOuter = false;
    int j = 0, k = 0;

    // Now the array is sorted, loop through and merge repeats
    while (true) {
      dets[j] = dets[k];
      for (int iReplica=0; iReplica<nreplicas; iReplica++) {
        amps[j][iReplica] = amps[k][iReplica];
      }
      if (schd.initiator) flags[j] = flags[k];

      while (true) {
        k += 1;
        if (k == nDets) {
          exitOuter = true;
          break;
        }
        if ( dets[j] == dets[k] ) {
          for (int iReplica=0; iReplica<nreplicas; iReplica++) {
            if (amps[j][iReplica]*amps[k][iReplica] < 0.0) {
              nAnnihil[iReplica] += 2.0*min(abs(amps[j][iReplica]), abs(amps[k][iReplica]));
            }
            amps[j][iReplica] += amps[k][iReplica];
          }

          // If the parent of any of the child walkers on this
          // determinant was an initiator, then we want to allow the
          // spawn. So set the child's flag to specify this, if so.
          // (Bitwise OR operation).
          if (schd.initiator) {
            flags[j] |= flags[k];
          }

        } else {
          break;
        }
      }

      if (exitOuter) break;
      
      if (j == nDets-1) {
        break;
      } else {
        j += 1;
      }
    }
    nDets = j+1;
  }

}

// Wrapper function for merging the spawned list into the main list
// Two versions are used for optimization
template<typename Wave, typename TrialWalk>
void spawnFCIQMC::mergeIntoMain(Wave& wave, TrialWalk& walk,
                                walkersFCIQMC<TrialWalk>& walkers,
                                semiStoch& core, vector<double>& nAnnihil,
                                const double minPop, bool initiator,
                                workingArray& work) {

  if (initiator) {
    mergeIntoMain_Initiator(wave, walk, walkers, core, nAnnihil, minPop, work);
  } else {
    mergeIntoMain_NoInitiator(wave, walk, walkers, core, nAnnihil, minPop, work);
  }

}

// Move spawned walkers to the provided main walker list
template<typename Wave, typename TrialWalk>
void spawnFCIQMC::mergeIntoMain_NoInitiator(Wave& wave, TrialWalk& walk,
                                            walkersFCIQMC<TrialWalk>& walkers,
                                            semiStoch& core, vector<double>& nAnnihil,
                                            const double minPop, workingArray& work) {
  for (int i = 0; i<nDets; i++) {

    // Convert from simpleDet to Determinant object
    Determinant det_i = Determinant(dets[i]);

    // Is this spawned determinant already in the main list?
    if (walkers.ht.find(det_i) != walkers.ht.end()) {
      int iDet = walkers.ht[det_i];

      // If using semi-stochastic, don't round spawning to core dets
      bool round = true;
      if (schd.semiStoch) {
        round = core.ht.find(dets[i]) == core.ht.end();
      }

      for (int iReplica=0; iReplica<nreplicas; iReplica++) {
        double oldAmp = walkers.amps[iDet][iReplica];

        // To ensure the various replicas are statistically independent,
        // we should stochastically round the spawning on a determinant
        // if it is currently unoccupied, and the new walker is below
        // the minimum threshold (since this is what happens if all
        // replicas are unoccupied):
        if (round) {
          if (abs(oldAmp) < 1.0e-12 && abs(amps[i][iReplica]) < minPop) {
            bool keepDet;
            stochastic_round(minPop, amps[i][iReplica], keepDet);
          }
        }

        if (amps[i][iReplica]*oldAmp < 0.0) {
          nAnnihil[iReplica] += 2.0*min(abs(amps[i][iReplica]), abs(oldAmp));
        }
        double newAmp = amps[i][iReplica] + oldAmp;
        walkers.amps[iDet][iReplica] = newAmp;
      }
    }
    else
    {
      // New determinant:
      bool keepDetAny = false;
      for (int iReplica=0; iReplica<nreplicas; iReplica++) {
        // If smaller than the smallest allowed population, then stochastically
        // round up to the threshold or down to 0:
        if (abs(amps[i][iReplica]) < minPop) {
          bool keepDet;
          stochastic_round(minPop, amps[i][iReplica], keepDet);
          keepDetAny = keepDetAny || keepDet;
        } else {
          keepDetAny = true;
        }
      }

      if (keepDetAny) {
        int pos;
        // Check if a determinant has become unoccupied in the existing list
        // If so, insert into that position
        // If not, then increase the walkers.ndets by 1 and add it on the end
        if (walkers.firstEmpty <= walkers.lastEmpty) {
          pos = walkers.emptyDets[walkers.firstEmpty];
          walkers.firstEmpty += 1;
        }
        else
        {
          pos = walkers.nDets;
          walkers.nDets += 1;
        }
        walkers.dets[pos] = det_i;
        walkers.diagH[pos] = det_i.Energy(I1, I2, coreE);
        TrialWalk newWalk(wave, det_i);
        double ovlp, localE, SVTotal;
        wave.HamAndOvlpAndSVTotal(newWalk, ovlp, localE, SVTotal, work,
                                  schd.epsilon);
        walkers.ovlp[pos] = ovlp;
        walkers.localE[pos] = localE;
        walkers.SVTotal[pos] = SVTotal;
        walkers.trialWalk[pos] = newWalk;

        // Add in the new walker population
        for (int iReplica=0; iReplica<nreplicas; iReplica++) {
          walkers.amps[pos][iReplica] = amps[i][iReplica];
        }
        walkers.ht[det_i] = pos;
      }

    }
  }
}

// Move spawned walkers to the provided main walker list, while
// applying the initiator criteria
template<typename Wave, typename TrialWalk>
void spawnFCIQMC::mergeIntoMain_Initiator(Wave& wave, TrialWalk& walk,
                                          walkersFCIQMC<TrialWalk>& walkers,
                                          semiStoch& core, vector<double>& nAnnihil,
                                          const double minPop, workingArray& work) {
  for (int i = 0; i<nDets; i++) {

    // Convert from simpleDet to Determinant object
    Determinant det_i = Determinant(dets[i]);

    // Used for testing if an initiator flag is set:
    bitset<max_nreplicas> initFlags(flags[i]);

    // Is this spawned determinant already in the main list?
    if (walkers.ht.find(det_i) != walkers.ht.end()) {
      int iDet = walkers.ht[det_i];

      // When doing semi-stochastic, all spawnings to core dets are
      // allowed, and are never stochastically rounded
      bool coreDet = false;
      if (schd.semiStoch) {
        coreDet = core.ht.find(dets[i]) != core.ht.end();
      }

      for (int iReplica=0; iReplica<nreplicas; iReplica++) {
        double oldAmp = walkers.amps[iDet][iReplica];

        // For the initiator criteria: only add in the spawned walker
        // for this determinant if it is already on occupied on
        // *this* particular replica, or the initiator flag is set
        if ( initFlags.test(iReplica) || abs(oldAmp) > 1.0e-12 || coreDet ) {
          // To ensure the various replicas are statistically independent,
          // we should stochastically round the spawning on a determinant
          // if it is currently unoccupied, and the new walker is below
          // the minimum threshold (since this is what happens if all
          // replicas are unoccupied):
          if (!coreDet) {
            if (abs(oldAmp) < 1.0e-12 && abs(amps[i][iReplica]) < minPop) {
              bool keepDet;
              stochastic_round(minPop, amps[i][iReplica], keepDet);
            }
          }

          if (amps[i][iReplica]*oldAmp < 0.0) {
            nAnnihil[iReplica] += 2.0*min(abs(amps[i][iReplica]), abs(oldAmp));
          }
          double newAmp = amps[i][iReplica] + oldAmp;
          walkers.amps[iDet][iReplica] = newAmp;
        }
      }
    }
    else
    {
      // Note that a new determinant can never be a core determinant when
      // semi-stochastic is in use; core determinants are never removed
      // from the main walker list. So we don't need to check for this.

      // New determinant:
      bool keepDetAny = false;
      for (int iReplica=0; iReplica<nreplicas; iReplica++) {
        // Check if initiator flag is set for this replica:
        if ( initFlags.test(iReplica) ) {
          // If smaller than the smallest allowed population, then stochastically
          // round up to the threshold or down to 0:
          if (abs(amps[i][iReplica]) < minPop) {
            bool keepDet;
            stochastic_round(minPop, amps[i][iReplica], keepDet);
            keepDetAny = keepDetAny || keepDet;
          } else {
            keepDetAny = true;
          }
        }
      }

      if (keepDetAny) {
        int pos;
        // Check if a determinant has become unoccupied in the existing list
        // If so, insert into that position
        // If not, then increase the walkers.ndets by 1 and add it on the end
        if (walkers.firstEmpty <= walkers.lastEmpty) {
          pos = walkers.emptyDets[walkers.firstEmpty];
          walkers.firstEmpty += 1;
        }
        else
        {
          pos = walkers.nDets;
          walkers.nDets += 1;
        }
        walkers.dets[pos] = det_i;
        walkers.diagH[pos] = det_i.Energy(I1, I2, coreE);
        TrialWalk newWalk(wave, det_i);
        double ovlp, localE, SVTotal;
        wave.HamAndOvlpAndSVTotal(newWalk, ovlp, localE, SVTotal, work,
                                  schd.epsilon);
        walkers.ovlp[pos] = ovlp;
        walkers.localE[pos] = localE;
        walkers.SVTotal[pos] = SVTotal;
        walkers.trialWalk[pos] = newWalk;

        // Add in the new walker population, but only if allowed by the
        // initiator criteria (i.e. if the flag is set):
        for (int iReplica=0; iReplica<nreplicas; iReplica++) {
          if ( initFlags.test(iReplica) ) {
            walkers.amps[pos][iReplica] = amps[i][iReplica];
          }
        }
        walkers.ht[det_i] = pos;
      }

    }
  }
}

// Instantiate needed templates

// Trivial (no trial wave function)
template void spawnFCIQMC::mergeIntoMain(
    TrivialWF& wave, TrivialWalk& walk,
    walkersFCIQMC<TrivialWalk>& walkers,
    semiStoch& core,
    vector<double>& nAnnihil,
    const double minPop,
    bool initiator,
    workingArray& work);

template void spawnFCIQMC::mergeIntoMain_NoInitiator(
    TrivialWF& wave, TrivialWalk& walk,
    walkersFCIQMC<TrivialWalk>& walkers,
    semiStoch& core,
    vector<double>& nAnnihil,
    const double minPop,
    workingArray& work);

template void spawnFCIQMC::mergeIntoMain_Initiator(
    TrivialWF& wave, TrivialWalk& walk,
    walkersFCIQMC<TrivialWalk>& walkers,
    semiStoch& core,
    vector<double>& nAnnihil,
    const double minPop,
    workingArray& work);

// Jastrow-Slater
using JSWalker = Walker<Jastrow, Slater>;

template void spawnFCIQMC::mergeIntoMain(
    CorrelatedWavefunction<Jastrow, Slater>& wave,
    Walker<Jastrow, Slater>& walk,
    walkersFCIQMC<JSWalker>& walkers,
    semiStoch& core,
    vector<double>& nAnnihil,
    const double minPop,
    bool initiator,
    workingArray& work);

template void spawnFCIQMC::mergeIntoMain_NoInitiator(
    CorrelatedWavefunction<Jastrow, Slater>& wave,
    Walker<Jastrow, Slater>& walk,
    walkersFCIQMC<JSWalker>& walkers,
    semiStoch& core,
    vector<double>& nAnnihil,
    const double minPop,
    workingArray& work);

template void spawnFCIQMC::mergeIntoMain_Initiator(
    CorrelatedWavefunction<Jastrow, Slater>& wave,
    Walker<Jastrow, Slater>& walk,
    walkersFCIQMC<JSWalker>& walkers,
    semiStoch& core,
    vector<double>& nAnnihil,
    const double minPop,
    workingArray& work);

// SelectedCI
template void spawnFCIQMC::mergeIntoMain(
    SelectedCI& wave,
    SimpleWalker& walk,
    walkersFCIQMC<SimpleWalker>& walkers,
    semiStoch& core,
    vector<double>& nAnnihil,
    const double minPop,
    bool initiator,
    workingArray& work);

template void spawnFCIQMC::mergeIntoMain_NoInitiator(
    SelectedCI& wave,
    SimpleWalker& walk,
    walkersFCIQMC<SimpleWalker>& walkers,
    semiStoch& core,
    vector<double>& nAnnihil,
    const double minPop,
    workingArray& work);

template void spawnFCIQMC::mergeIntoMain_Initiator(
    SelectedCI& wave,
    SimpleWalker& walk,
    walkersFCIQMC<SimpleWalker>& walkers,
    semiStoch& core,
    vector<double>& nAnnihil,
    const double minPop,
    workingArray& work);
