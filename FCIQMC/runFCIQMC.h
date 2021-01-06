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

#ifndef runFCIQMC_HEADER_H
#define runFCIQMC_HEADER_H

#include "dataFCIQMC.h"
#include "excitGen.h"
#include "spawnFCIQMC.h"
#include "walkersFCIQMC.h"
#include "utilsFCIQMC.h"

template<typename Wave, typename TrialWalk>
void initFCIQMC(Wave& wave, TrialWalk& walk,
                const int norbs, const int nel, const int nalpha, const int nbeta,
                Determinant& HFDet, double& HFEnergy, heatBathFCIQMC& hb,
                walkersFCIQMC<TrialWalk>& walkers, spawnFCIQMC& spawn, workingArray& work);

template<typename Wave, typename TrialWalk>
void initWalkerListHF(Wave& wave, TrialWalk& walk, Determinant& HFDet, const int DetLenMin,
                      double& HFEnergy, walkersFCIQMC<TrialWalk>& walkers, workingArray& work);

template<typename Wave, typename TrialWalk>
void initWalkerListTrialWF(Wave& wave, TrialWalk& walk, walkersFCIQMC<TrialWalk>& walkers,
                           spawnFCIQMC& spawn, workingArray& work);

template<typename Wave, typename TrialWalk>
void runFCIQMC(Wave& wave, TrialWalk& walk, const int norbs, const int nel,
               const int nalpha, const int nbeta);

template<typename Wave, typename TrialWalk>
void attemptSpawning(Wave& wave, TrialWalk& walk, Determinant& parentDet, Determinant& childDet,
                     spawnFCIQMC& spawn, oneInt &I1, twoInt &I2, double& coreE, const int nAttemptsEach,
                     const double parentAmp, const int parentFlags, const int iReplica,
                     const double tau, const double minSpawn, const double pgen,
                     const int ex1, const int ex2);

template<typename Wave, typename TrialWalk>
void performDeath(const int iDet, walkersFCIQMC<TrialWalk>& walkers, oneInt &I1, twoInt &I2,
                  double& coreE, const vector<double>& Eshift, const double tau);

template<typename Wave, typename TrialWalk>
void performDeathAllWalkers(walkersFCIQMC<TrialWalk>& walkers, oneInt &I1, twoInt &I2,
                  double& coreE, const vector<double>& Eshift, const double tau);

template<typename Wave, typename TrialWalk>
void calcVarEnergy(walkersFCIQMC<TrialWalk>& walkers, const spawnFCIQMC& spawn, const oneInt& I1,
                   const twoInt& I2, double& coreE, const double tau,
                   double& varEnergyNum, double& varEnergyDenom);

template<typename Wave, typename TrialWalk>
void calcEN2Correction(walkersFCIQMC<TrialWalk>& walkers, const spawnFCIQMC& spawn, const oneInt& I1,
                       const twoInt& I2, double& coreE, const double tau, const double varEnergyNum,
                       const double varEnergyDenom, double& EN2All);

void communicateEstimates(dataFCIQMC& dat, const int nDets, const int nSpawnedDets, int& nDetsTot,
                          int& nSpawnedDetsTot);

void updateShift(vector<double>& Eshift, vector<bool>& varyShift, const vector<double>& walkerPop,
                 const vector<double>& walkerPopOld, const double targetPop,
                 const double shiftDamping, const double tau);

void printDataTableHeader();

void printDataTable(const dataFCIQMC& dat, const int iter, const int nDets, const int nSpawned,
                    const double iter_time);

void printFinalStats(const vector<double>& walkerPop, const int nDets,
                     const int nSpawnDets, const double total_time);

#endif
