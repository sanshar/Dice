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

#pragma once

#include <iomanip>
#include "Determinants.h"
#include "evaluateE.h"
#include "excitGen.h"
#include "global.h"
#include "input.h"
#include "spawnFCIQMC.h"
#include "Walker.h"
#include "walkersFCIQMC.h"
#include "utilsFCIQMC.h"

void attemptSpawning(Determinant& parentDet, Determinant& childDet, spawnFCIQMC& spawn,
                     oneInt &I1, twoInt &I2, double& coreE, const int& nAttemptsEach,
                     const double& parentAmp, const int& parentFlags, const int& iReplica,
                     const double& tau, const double& minSpawn, const double& pgen);

void performDeath(const int &iDet, walkersFCIQMC& walkers, oneInt &I1, twoInt &I2,
                  double& coreE, const vector<double>& Eshift, const double& tau);

void performDeathAllWalkers(walkersFCIQMC& walkers, oneInt &I1, twoInt &I2,
                  double& coreE, const vector<double>& Eshift, const double& tau);

void calcVarEnergy(walkersFCIQMC& walkers, const spawnFCIQMC& spawn, const oneInt& I1,
                   const twoInt& I2, double& coreE, const double& tau,
                   double& varEnergyNum, double& varEnergyDenom);

void communicateEstimates(const vector<double>& walkerPop, const vector<double>& EProj, const vector<double>& HFAmp,
                          const int& nDets, const int& nSpawnedDets, vector<double>& walkerPopTot,
                          vector<double>& EProjTot, vector<double>& HFAmpTot, int& nDetsTot, int& nSpawnedDetsTot);

void updateShift(vector<double>& Eshift, vector<bool>& varyShift, const vector<double>& walkerPop,
                 const vector<double>& walkerPopOld, const double& targetPop,
                 const double& shiftDamping, const double& tau);

void printDataTableHeader();

void printDataTable(const int iter, const int& nDets, const int& nSpawned, const vector<double>& shift,
                    const vector<double>& walkerPop, const vector<double>& EProj, const vector<double>& HFAmp,
                    const double& EVarNumAll, const double& EVarDenomAll, const double& iter_time);

void printFinalStats(const vector<double>& walkerPop, const int& nDets,
                     const int& nSpawnDets, const double& total_time);


void runFCIQMC() {

  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;
  int nel = nalpha + nbeta;

  // The number of 64-bit integers required to represent (the alpha or beta
  // part of) a determinant
  int DetLenAlpha = (nalpha-1)/64 + 1;
  int DetLenBeta = (nalpha-1)/64 + 1;
  int DetLenMin = max(DetLenAlpha, DetLenBeta);

  Determinant HFDet;
  for (int i = 0; i < nalpha; i++)
    HFDet.setoccA(i, true);
  for (int i = 0; i < nbeta; i++)
    HFDet.setoccB(i, true);

  // Processor that the HF determinant lives on
  int HFDetProc = getProc(HFDet, DetLenMin);

  int walkersSize = schd.targetPop * schd.mainMemoryFac / commsize;
  int spawnSize = schd.targetPop * schd.spawnMemoryFac / commsize;

  walkersFCIQMC walkers(walkersSize, DetLenMin);
  spawnFCIQMC spawn(spawnSize, DetLenMin);

  if (boost::iequals(schd.determinantFile, ""))
  {
    if (HFDetProc == commrank) {
      walkers.dets[0] = HFDet;
      walkers.ht[HFDet] = 0;
      // Set the population on the reference
      for (int iReplica=0; iReplica<nreplicas; iReplica++) {
        walkers.amps[0][iReplica] = schd.initialPop;
      }
      // The number of determinants in the walker list
      walkers.nDets = 1;
    }
  }
  else
  {
    //readDeterminants(schd.determinantFile, walkers.dets, walkers.amps);
  }

  // ----- FCIQMC data -----
  double pgen = 0.0, pgen2 = 0.0, parentAmp = 0.0;
  double time_start = 0.0, time_end = 0.0, iter_time = 0.0, total_time = 0.0;
  double EVarNumAll = 0.0, EVarDenomAll = 0.0;
  vector<double> walkerPop(nreplicas, 0.0);
  vector<double> EProj(nreplicas, 0.0);
  vector<double> HFAmp(nreplicas, 0.0);

  // Total quantities, after summing over processors
  vector<double> walkerPopTot(nreplicas, 0.0);
  vector<double> walkerPopOldTot(nreplicas, 0.0);
  vector<double> EProjTot(nreplicas, 0.0);
  vector<double> HFAmpTot(nreplicas, 0.0);
  int nDetsTot, nSpawnedDetsTot;

  int nAttempts = 0;
  Determinant childDet, childDet2;

  vector<bool> varyShift(nreplicas);
  std::fill(varyShift.begin(), varyShift.end(), false);

  vector<double> Eshift(nreplicas);
  double initEshift = HFDet.Energy(I1, I2, coreE) + schd.initialShift;
  std::fill(Eshift.begin(), Eshift.end(), initEshift);
  // -----------------------

  if (commrank == 0) {
    cout << "Hartree--Fock energy: " << HFDet.Energy(I1, I2, coreE) << endl << endl;
  }

  heatBathFCIQMC hb;
  if (schd.heatBathExGen || schd.heatBathUniformSingExGen) {
    if (commrank == 0) cout << "Starting heat bath excitation generator construction..." << endl;
    hb.createArrays(norbs, I2);
    if (commrank == 0) cout << "Heat bath excitation generator construction finished." << endl;
  }

  // The default excitation generator is the uniform one. If
  // the user has specified another, then turn this off.
  if (schd.heatBathExGen || schd.heatBathUniformSingExGen) {
    schd.uniformExGen = false;
  }

  // Get and print the initial stats
  walkers.calcStats(HFDet, walkerPop, EProj, HFAmp, I1, I2, coreE);
  communicateEstimates(walkerPop, EProj, HFAmp, walkers.nDets, spawn.nDets,
                       walkerPopTot, EProjTot, HFAmpTot, nDetsTot, nSpawnedDetsTot);
  calcVarEnergy(walkers, spawn, I1, I2, coreE,schd.tau, EVarNumAll, EVarDenomAll);
  walkerPopOldTot = walkerPopTot; 
  printDataTableHeader();
  printDataTable(0, nDetsTot, nSpawnedDetsTot, Eshift, walkerPopTot, EProjTot,
                 HFAmpTot, EVarNumAll, EVarDenomAll, iter_time);

  // Main FCIQMC loop
  for (int iter = 1; iter <= schd.maxIter; iter++) {
    time_start = getTime();

    walkers.firstEmpty = 0;
    walkers.lastEmpty = -1;
    spawn.nDets = 0;
    spawn.currProcSlots = spawn.firstProcSlots;

    fill(walkerPop.begin(), walkerPop.end(), 0.0);
    fill(EProj.begin(), EProj.end(), 0.0);
    fill(HFAmp.begin(), HFAmp.end(), 0.0);

    //cout << walkers << endl;

    // Loop over all walkers/determinants
    for (int iDet=0; iDet<walkers.nDets; iDet++) {
      // Is this unoccupied for all replicas? If so, add to the list of empty slots
      if (walkers.allUnoccupied(iDet)) {
        walkers.lastEmpty += 1;
        walkers.emptyDets[walkers.lastEmpty] = iDet;
        continue;
      }

      for (int iReplica=0; iReplica<nreplicas; iReplica++) {
        // Update the initiator flag, if necessary
        int parentFlags = 0;
        if (schd.initiator) {
          if (abs(walkers.amps[iDet][iReplica]) > schd.initiatorThresh) {
            // This walker is an initiator, so set the flag for the
            // appropriate replica
            parentFlags |= 1 << iReplica;
          }
        }

        // Number of spawnings to attempt
        nAttempts = max(1.0, round(abs(walkers.amps[iDet][iReplica]) * schd.nAttemptsEach));
        parentAmp = walkers.amps[iDet][iReplica] * schd.nAttemptsEach / nAttempts;

        // Perform one spawning attempt for each 'walker' of weight parentAmp
        for (int iAttempt=0; iAttempt<nAttempts; iAttempt++) {
          generateExcitation(hb, I1, I2, walkers.dets[iDet], nel, childDet, childDet2, pgen, pgen2);

          // pgen=0.0 is set when a null excitation is returned.
          if (pgen > 1.e-15) {
            attemptSpawning(walkers.dets[iDet], childDet, spawn, I1, I2, coreE, schd.nAttemptsEach,
                            parentAmp, parentFlags, iReplica, schd.tau, schd.minSpawn, pgen);
          }
          if (pgen2 > 1.e-15) {
            attemptSpawning(walkers.dets[iDet], childDet2, spawn, I1, I2, coreE, schd.nAttemptsEach,
                            parentAmp, parentFlags, iReplica, schd.tau, schd.minSpawn, pgen2);
          }

        } // Loop over spawning attempts
      } // Loop over replicas

    }

    // Perform annihilation
    spawn.communicate();
    spawn.compress();
    calcVarEnergy(walkers, spawn, I1, I2, coreE,schd.tau, EVarNumAll, EVarDenomAll);
    performDeathAllWalkers(walkers, I1, I2, coreE, Eshift, schd.tau);
    spawn.mergeIntoMain(walkers, schd.minPop, schd.initiator);

    // Stochastic rounding of small walkers
    walkers.stochasticRoundAll(schd.minPop);

    walkers.calcStats(HFDet, walkerPop, EProj, HFAmp, I1, I2, coreE);
    communicateEstimates(walkerPop, EProj, HFAmp, walkers.nDets, spawn.nDets,
                         walkerPopTot, EProjTot, HFAmpTot, nDetsTot, nSpawnedDetsTot);
    updateShift(Eshift, varyShift, walkerPopTot, walkerPopOldTot, schd.targetPop,
                schd.shiftDamping, schd.tau);
    printDataTable(iter, nDetsTot, nSpawnedDetsTot, Eshift, walkerPopTot, EProjTot,
                   HFAmpTot, EVarNumAll, EVarDenomAll, iter_time);

    walkerPopOldTot = walkerPopTot;

    time_end = getTime();
    iter_time = time_end - time_start;
  }

  total_time = getTime() - startofCalc;
  printFinalStats(walkerPop, walkers.nDets, spawn.nDets, total_time);

}


// Find the weight of the spawned walker
// If it is above a minimum threshold, then always spawn
// Otherwsie, stochastically round it up to the threshold or down to 0
void attemptSpawning(Determinant& parentDet, Determinant& childDet, spawnFCIQMC& spawn,
                     oneInt &I1, twoInt &I2, double& coreE, const int& nAttemptsEach,
                     const double& parentAmp, const int& parentFlags, const int& iReplica,
                     const double& tau, const double& minSpawn, const double& pgen)
{
  bool childSpawned = true;

  double pgen_tot = pgen * nAttemptsEach;
  double HElem = Hij(parentDet, childDet, I1, I2, coreE);
  double childAmp = - tau * parentAmp * HElem / pgen_tot;

  if (abs(childAmp) < minSpawn) {
    stochastic_round(minSpawn, childAmp, childSpawned);
  }

  if (childSpawned) {
    int proc = getProc(childDet, spawn.DetLenMin);
    // Find the appropriate place in the spawned list for the processor
    // of the newly-spawned walker
    int ind = spawn.currProcSlots[proc];
    spawn.dets[ind] = childDet.getSimpleDet();

    // Set the child amplitude for the correct replica - all others are 0
    for (int i=0; i<nreplicas; i++) {
      if (i == iReplica) {
        spawn.amps[ind][i] = childAmp;
      } else {
        spawn.amps[ind][i] = 0.0;
      }
    }

    if (schd.initiator) spawn.flags[ind] = parentFlags;
    spawn.currProcSlots[proc] += 1;
  }
}

void calcVarEnergy(walkersFCIQMC& walkers, const spawnFCIQMC& spawn, const oneInt& I1,
                   const twoInt& I2, double& coreE, const double& tau,
                   double& varEnergyNumAll, double& varEnergyDenomAll)
{
  double varEnergyNum = 0.0;
  double varEnergyDenom = 0.0;

  // Contributions from the diagonal of the Hamiltonian
  for (int iDet=0; iDet<walkers.nDets; iDet++) {
    double HDiag = walkers.dets[iDet].Energy(I1, I2, coreE);
    varEnergyNum += HDiag * walkers.amps[iDet][0] * walkers.amps[iDet][1];
    varEnergyDenom += walkers.amps[iDet][0] * walkers.amps[iDet][1];
  }

  // Contributions from the off-diagonal of the Hamiltonian
  for (int j=0; j<spawn.nDets; j++) {
    if (walkers.ht.find(spawn.dets[j]) != walkers.ht.end()) {
      int iDet = walkers.ht[spawn.dets[j]];
      double spawnAmp1 = spawn.amps[j][0];
      double spawnAmp2 = spawn.amps[j][1];
      double currAmp1 = walkers.amps[iDet][0];
      double currAmp2 = walkers.amps[iDet][1];
      varEnergyNum -= ( currAmp1 * spawnAmp2 + spawnAmp1 * currAmp2 ) / ( 2.0 * tau);
    }
  }

  MPI_Allreduce(&varEnergyNum,   &varEnergyNumAll,   1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&varEnergyDenom, &varEnergyDenomAll, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

void performDeath(const int &iDet, walkersFCIQMC& walkers, oneInt &I1, twoInt &I2,
                  double& coreE, const vector<double>& Eshift, const double& tau)
{
  double parentE = walkers.dets[iDet].Energy(I1, I2, coreE);
  for (int iReplica=0; iReplica<nreplicas; iReplica++) {
    double fac = tau * ( parentE - Eshift[iReplica] );
    walkers.amps[iDet][iReplica] -= fac * walkers.amps[iDet][iReplica];
  }
}

void performDeathAllWalkers(walkersFCIQMC& walkers, oneInt &I1, twoInt &I2,
                  double& coreE, const vector<double>& Eshift, const double& tau)
{
  for (int iDet=0; iDet<walkers.nDets; iDet++) {
    double parentE = walkers.dets[iDet].Energy(I1, I2, coreE);
    for (int iReplica=0; iReplica<nreplicas; iReplica++) {
      double fac = tau * ( parentE - Eshift[iReplica] );
      walkers.amps[iDet][iReplica] -= fac * walkers.amps[iDet][iReplica];
    }
  }
}

void communicateEstimates(const vector<double>& walkerPop, const vector<double>& EProj, const vector<double>& HFAmp,
                          const int& nDets, const int& nSpawnedDets, vector<double>& walkerPopTot,
                          vector<double>& EProjTot, vector<double>& HFAmpTot, int& nDetsTot, int& nSpawnedDetsTot)
{
#ifdef SERIAL
  walkerPopTot    = walkerPop;
  EProjTot        = EProj;
  HFAmpTot        = HFAmp;
  nDetsTot        = nDets;
  nSpawnedDetsTot = nSpawnedDets;
#else
  MPI_Allreduce(&walkerPop.front(),  &walkerPopTot.front(),  nreplicas,  MPI_DOUBLE,  MPI_SUM,  MPI_COMM_WORLD);
  MPI_Allreduce(&EProj.front(),      &EProjTot.front(),      nreplicas,  MPI_DOUBLE,  MPI_SUM,  MPI_COMM_WORLD);
  MPI_Allreduce(&HFAmp.front(),      &HFAmpTot.front(),      nreplicas,  MPI_DOUBLE,  MPI_SUM,  MPI_COMM_WORLD);
  MPI_Allreduce(&nDets,              &nDetsTot,              1,          MPI_INT,     MPI_SUM,  MPI_COMM_WORLD);
  MPI_Allreduce(&nSpawnedDets,       &nSpawnedDetsTot,       1,          MPI_INT,     MPI_SUM,  MPI_COMM_WORLD);
#endif
}

void updateShift(vector<double>& Eshift, vector<bool>& varyShift, const vector<double>& walkerPop,
                 const vector<double>& walkerPopOld, const double& targetPop,
                 const double& shiftDamping, const double& tau)
{
  for (int iReplica = 0; iReplica<nreplicas; iReplica++) {
    if ((!varyShift.at(iReplica)) && walkerPop[iReplica] > targetPop) {
      varyShift.at(iReplica) = true;
    }
    if (varyShift.at(iReplica)) {
      Eshift.at(iReplica) = Eshift.at(iReplica) - (shiftDamping/tau) * log(walkerPop.at(iReplica)/walkerPopOld.at(iReplica));
    }
  }
}

void printDataTableHeader()
{
  if (commrank == 0) {
    cout << "#  1. Iter";
    cout << "     2. nDets";
    cout << "  3. nSpawned";

    // This is the column label
    int label;

    // This loop is for properties printed on each replica
    for (int iReplica=0; iReplica<nreplicas; iReplica++) {

      for (int j=0; j<4; j++) {
        string header;
        label = 4*iReplica + j + 4;
        header.append(to_string(label));

        if (j==0) {
          header.append(". Shift_");
        } else if (j==1) {
          header.append(". nWalkers_");
        } else if (j==2) {
          header.append(". Energy_Num_");
        } else if (j==3) {
          header.append(". Energy_Denom_");
        }

        header.append(to_string(iReplica+1));
        cout << right << setw(21) << header;
      }

    }

    if (nreplicas == 2) {
      label += 1;
      cout << right << setw(6) << label << ". Var_Energy_Num";
      label += 1;
      cout << right << setw(4) << label << ". Var_Energy_Denom";
    }

    label += 1;
    cout << right << setw(5) << label << ". Time\n";
  }
}

void printDataTable(const int iter, const int& nDets, const int& nSpawned, const vector<double>& shift,
                    const vector<double>& walkerPop, const vector<double>& EProj, const vector<double>& HFAmp,
                    const double& EVarNumAll, const double& EVarDenomAll, const double& iter_time)
{
  if (commrank == 0) {
    printf ("%10d   ", iter);
    printf ("%10d   ", nDets);
    printf ("%10d   ", nSpawned);
    for (int iReplica=0; iReplica<nreplicas; iReplica++) {
      printf ("%18.10f   ", shift[iReplica]);
      printf ("%18.10f   ", walkerPop[iReplica]);
      printf ("%18.10f   ", EProj[iReplica]);
      printf ("%18.10f   ", HFAmp[iReplica]);
    }
    if (nreplicas == 2) {
      printf ("%.12e    ", EVarNumAll);
      printf ("%.12e   ", EVarDenomAll);
    }
    printf ("%8.4f\n", iter_time);
  }
}

void printFinalStats(const vector<double>& walkerPop, const int& nDets,
                     const int& nSpawnDets, const double& total_time)
{
  int parallelReport[commsize];
  double parallelReportD[commsize];

  if (commrank == 0) {
    cout << "# Total time:  " << getTime() - startofCalc << endl;
  }

#ifndef SERIAL
  MPI_Gather(&walkerPop[0], 1, MPI_DOUBLE, &parallelReportD, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (commrank == 0) {
    cout << "# Min # walkers on proc:   " << *min_element(parallelReportD, parallelReportD + commsize) << endl;
    cout << "# Max # walkers on proc:   " << *max_element(parallelReportD, parallelReportD + commsize) << endl;
  }

  MPI_Gather(&nDets, 1, MPI_INT, &parallelReport, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (commrank == 0) {
    cout << "# Min # determinants on proc:   " << *min_element(parallelReport, parallelReport + commsize) << endl;
    cout << "# Max # determinants on proc:   " << *max_element(parallelReport, parallelReport + commsize) << endl;
  }

  MPI_Gather(&nSpawnDets, 1, MPI_INT, &parallelReport, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (commrank == 0) {
    cout << "# Min # determinants spawned on proc:   " << *min_element(parallelReport, parallelReport + commsize) << endl;
    cout << "# Max # determinants spawned on proc:   " << *max_element(parallelReport, parallelReport + commsize) << endl;
  }
#endif
}
