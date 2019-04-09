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
#ifndef spawnFCIQMC_HEADER_H
#define spawnFCIQMC_HEADER_H

#include "global.h"
#include "walkersFCIQMC.h"
#include <iostream>
#include <vector>
#include <boost/serialization/serialization.hpp>
#include <Eigen/Dense>

class Determinant;

template <typename T, typename Compare>
std::vector<std::size_t> sort_permutation(int const nDets, const std::vector<T>& vec, Compare compare)
{
  std::vector<std::size_t> p(nDets);
  std::iota(p.begin(), p.end(), 0);
  std::sort(p.begin(), p.end(),
    [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
  return p;
}

template <typename T>
void apply_permutation(int const nDets, const std::vector<T>& vec, std::vector<T>& sorted_vec, const std::vector<std::size_t>& p)
{
  std::transform(p.begin(), p.end(), sorted_vec.begin(),
    [&](std::size_t i){ return vec[i]; });
}

using namespace std;

// Class for spawning arrays needed in FCIQMC
class spawnFCIQMC {

 public:
  // The number of determinants spawned to
  int nDets;
  // The list of determinants spawned to
  vector<simpleDet> dets;
  // Temporary space for communication and sorting
  vector<simpleDet> detsTemp;
  // The amplitudes of spawned walkers
  vector<double> amps;
  vector<double> ampsTemp;

  // The number of elements allocated for spawns to each processor
  int nSlotsPerProc;
  // The positions of the first elements for each processor in the
  // spawning array
  vector<int> firstProcSlots;
  // The current positions in the spawning array, in which to add
  // the next spawned walker to a given processor
  vector<int> currProcSlots;


  spawnFCIQMC(int spawnSize) {
    nDets = 0;
    dets.resize(spawnSize);
    amps.resize(spawnSize, 0.0);
    detsTemp.resize(spawnSize);
    ampsTemp.resize(spawnSize, 0.0);

    firstProcSlots.resize(commsize);
    currProcSlots.resize(commsize);

    nSlotsPerProc = spawnSize / commsize;
    for (int i=0; i<commsize; i++) {
      firstProcSlots[i] = i*nSlotsPerProc;
    }

    currProcSlots = firstProcSlots;
  }

  // Send spawned walkers to their correct processor
  void communicate() {
#ifdef SERIAL
    nDets = currProcSlots[0] - firstProcSlots[0];
    for (int i=0; i<nDets; i++) {
      detsTemp[i] = dets[i];
      ampsTemp[i] = amps[i];
    }
#else
    int sendCounts[commsize], recvCounts[commsize];
    int sendDispls[commsize], recvDispls[commsize];
    int sendCountsDets[commsize], recvCountsDets[commsize];
    int sendDisplsDets[commsize], recvDisplsDets[commsize];

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
    }

    MPI_Alltoallv(&dets.front(), sendCountsDets, sendDisplsDets, MPI_LONG,
                  &detsTemp.front(), recvCountsDets, recvDisplsDets, MPI_LONG, MPI_COMM_WORLD);

    MPI_Alltoallv(&amps.front(), sendCounts, sendDispls, MPI_DOUBLE,
                  &ampsTemp.front(), recvCounts, recvDispls, MPI_DOUBLE, MPI_COMM_WORLD);

    // The total number of determinants received
    nDets = recvDispls[commsize-1] + recvCounts[commsize-1];
#endif
  }
  
  // Merge multiple spawned walkers to the same determinant, so that each
  // determinant only appears once
  void compress() {

    if (nDets > 0) {
      // Perform sort
      auto p = sort_permutation(nDets, detsTemp, [](simpleDet const& a, simpleDet const& b){ return (a < b); });
      apply_permutation( nDets, detsTemp, dets, p );
      apply_permutation( nDets, ampsTemp, amps, p );
  
      bool exitOuter = false;
      int j = 0, k = 0;

      // Now the array is sorted, loop through and merge repeats
      while (true) {
        dets[j] = dets[k];
        amps[j] = amps[k];
        while (true) {
          k += 1;
          if (k == nDets) {
            exitOuter = true;
            break;
          }
          if ( dets[j] == dets[k] ) {
            amps[j] += amps[k];
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
  void mergeIntoMain(walkersFCIQMC& walkers, const double& minPop) {

    int pos;
    bool keepDet;
  
    for (int i = 0; i<nDets; i++) {
      keepDet = true;
      // Is this spawned determinant already in the main list?
      if (walkers.ht.find(dets[i]) != walkers.ht.end()) {
        int iDet = walkers.ht[dets[i]];
        double oldAmp = walkers.amps[iDet];
        double newAmp = amps[i] + oldAmp;
        walkers.amps[iDet] = newAmp;
      }
      else
      {
        // New determinant:
        if (abs(amps[i]) < minPop) {
          stochastic_round(minPop, amps[i], keepDet);
        }

        if (keepDet) {
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
          walkers.amps[pos] = amps[i];
          walkers.ht[dets[i]] = pos;
        }

      }
    }
  }

};

#endif
