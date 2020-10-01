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

#include <vector>
#include "Determinants.h"
#include "walkersFCIQMC.h"

class Determinant;

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
  double** amps;
  double** ampsTemp;

  // Flags for the spawned walkers
  vector<int> flags;
  vector<int> flagsTemp;

  // The number of elements allocated for spawns to each processor
  int nSlotsPerProc;
  // The positions of the first elements for each processor in the
  // spawning array
  vector<int> firstProcSlots;
  // The current positions in the spawning array, in which to add
  // the next spawned walker to a given processor
  vector<int> currProcSlots;

  // The number of 64-bit integers required to represent (the alpha or beta
  // part of) a determinant
  // Note, this is different to DetLen in global.h
  int DetLenMin;

  spawnFCIQMC(int spawnSize, int DetLenLocal);
  ~spawnFCIQMC();

  // Send spawned walkers to their correct processor
  void communicate();
  
  // Merge multiple spawned walkers to the same determinant, so that each
  // determinant only appears once
  void compress();
  
  // Move spawned walkers to the provided main walker list
  void mergeIntoMain(walkersFCIQMC& walkers, const double& minPop, bool initiator);
  void mergeIntoMain_NoInitiator(walkersFCIQMC& walkers, const double& minPop);
  void mergeIntoMain_Initiator(walkersFCIQMC& walkers, const double& minPop);

};

#endif
