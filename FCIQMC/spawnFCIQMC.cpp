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
    for (int iReplica=0; iReplica<nreplicas; iReplica++) {
      sorted_amps[iDet][iReplica] = amps[sorted_ind][iReplica];
    }
  }
}

spawnFCIQMC::spawnFCIQMC(int spawnSize, int DetLenLocal) {
  nDets = 0;
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
void spawnFCIQMC::compress() {

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

// Move spawned walkers to the provided main walker list
void spawnFCIQMC::mergeIntoMain(walkersFCIQMC& walkers, const double& minPop) {

  int pos;

  for (int i = 0; i<nDets; i++) {
    // Is this spawned determinant already in the main list?
    if (walkers.ht.find(dets[i]) != walkers.ht.end()) {
      int iDet = walkers.ht[dets[i]];
      for (int iReplica=0; iReplica<nreplicas; iReplica++) {
        double oldAmp = walkers.amps[iDet][iReplica];
        double newAmp = amps[i][iReplica] + oldAmp;
        walkers.amps[iDet][iReplica] = newAmp;
      }
    }
    else
    {
      // New determinant:

      // If the initiator flag is unset, then none of the parents which
      // spawned here were initiators. So cancel the spawn.
      if (schd.initiator) {
        if (flags[i] == 0) continue;
      }

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
        walkers.dets[pos] = Determinant(dets[i]);
        for (int iReplica=0; iReplica<nreplicas; iReplica++) {
          walkers.amps[pos][iReplica] = amps[i][iReplica];
        }
        walkers.ht[dets[i]] = pos;
      }

    }
  }
}
