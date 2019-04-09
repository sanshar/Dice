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
#include <algorithm>
#include <random>
#include <chrono>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#ifndef SERIAL
#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "Determinants.h"
#include "global.h"
#include "input.h"
#include "integral.h"
#include "spawnFCIQMC.h"
#include "SHCIshm.h"
#include "math.h"
#include "Profile.h"
#include "Walker.h"
#include "walkersFCIQMC.h"

using namespace Eigen;
using namespace boost;
using namespace std;


void generateExcitation(const Determinant& parentDet, Determinant& childDet, double& pgen);
void generateSingleExcit(const Determinant& parentDet, Determinant& childDet, double& pgen_ia);
void generateDoubleExcit(const Determinant& parentDet, Determinant& childDet, double& pgen_ijab);

void attemptSpawning(Determinant& parentDet, Determinant& childDet, spawnFCIQMC& spawn,
                     oneInt &I1, twoInt &I2, double& coreE, const int& nAttemptsEach, const double& parentAmp,
                     const double& tau, const double& minSpawn, const double& pgen);

void performDeath(Determinant& parentDet, double& detPopulation, oneInt &I1, twoInt &I2,
                  double& coreE, const double& Eshift, const double& tau);

void communicateEstimates(const double& walkerPop, const double& EProj, const double& HFAmp,
                          const int& nDets, const int& nSpawnedDets,
                          double& walkerPopTot, double& EProjTot, double& HFAmpTot,
                          int& nDetsTot, int& nSpawnedDetsTot);

void updateShift(double& Eshift, bool& varyShift, const double& walkerPop,
                 const double& walkerPopOld, const double& targetPop,
                 const double& shiftDamping, const double& tau);

void printDataTableHeader();

void printDataTable(const int iter, const int& nDets, const int& nSpawned,
                    const double& shift, const double& walkerPop, const double& EProj,
                    const double& HFAmp, const double& iter_time);

void printFinalStats(const double& walkerPop, const int& nDets,
                     const int& nSpawnDets, const double& total_time);


int main(int argc, char *argv[])
{
#ifndef SERIAL
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
#endif
  startofCalc = getTime();

  initSHM();
  //license();

  string inputFile = "input.dat";
  if (argc > 1)
    inputFile = string(argv[1]);
  readInput(inputFile, schd, false);

  generator = std::mt19937(schd.seed + commrank);

  readIntegralsAndInitializeDeterminantStaticVariables("FCIDUMP");
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

  Determinant HFDet;
  for (int i = 0; i < nalpha; i++)
    HFDet.setoccA(i, true);
  for (int i = 0; i < nbeta; i++)
    HFDet.setoccB(i, true);

  // Processor that the HF determinant lives on
  int HFDetProc = HFDet.getProc();

  // TODO: Update when code is parallelized
  int walkersSize = schd.targetPop * schd.mainMemoryFac;
  int spawnSize = schd.targetPop * schd.spawnMemoryFac;

  walkersFCIQMC walkers(walkersSize);
  spawnFCIQMC spawn(spawnSize);

  if (boost::iequals(schd.determinantFile, ""))
  {
    if (HFDetProc == commrank) {
      walkers.dets[0] = HFDet;
      walkers.ht[HFDet] = 0;
      // Set the population on the reference
      walkers.amps[0] = schd.initialPop;
      // The number of determinants in the walker list
      walkers.nDets = 1;
    }
  }
  else
  {
    readDeterminants(schd.determinantFile, walkers.dets, walkers.amps);
  }

  // ----- FCIQMC data -----
  double EProj = 0.0, HFAmp = 0.0, pgen = 0.0, parentAmp = 0.0, walkerPop = 0.0;
  double time_start = 0.0, time_end = 0.0, iter_time = 0.0, total_time = 0.0;

  int nAttempts = 0;
  Determinant childDet;

  // Total quantities, after summing over processors
  double walkerPopTot, walkerPopOldTot, EProjTot, HFAmpTot;
  int nDetsTot, nSpawnedDetsTot;

  bool varyShift = false;
  double Eshift = HFDet.Energy(I1, I2, coreE) + schd.initialShift;
  // -----------------------

  if (commrank == 0) {
    cout << "Number of processors: " << commsize << endl;
    cout << "Hartree--Fock energy: " << HFDet.Energy(I1, I2, coreE) << endl << endl;
  }

  // Get and print the initial stats
  walkers.calcStats(HFDet, walkerPop, EProj, HFAmp, I1, I2, coreE);
  communicateEstimates(walkerPop, EProj, HFAmp, walkers.nDets, spawn.nDets,
                       walkerPopTot, EProjTot, HFAmpTot, nDetsTot, nSpawnedDetsTot);
  walkerPopOldTot = walkerPopTot; 
  printDataTableHeader();
  printDataTable(0, nDetsTot, nSpawnedDetsTot, Eshift, walkerPopTot, EProjTot, HFAmpTot, iter_time);

  // Main FCIQMC loop
  for (int iter = 1; iter <= schd.maxIter; iter++) {
    time_start = getTime();

    walkers.firstEmpty = 0;
    walkers.lastEmpty = -1;
    spawn.nDets = 0;
    spawn.currProcSlots = spawn.firstProcSlots;
    walkerPop = 0.0;
    EProj = 0.0;
    HFAmp = 0.0;

    //cout << walkers << endl;

    // Loop over all walkers/determinants
    for (int iDet=0; iDet<walkers.nDets; iDet++) {
      // Is this unoccupied? If so, add to the list of empty slots
      if (abs(walkers.amps[iDet]) < 1.0e-12) {
        walkers.lastEmpty += 1;
        walkers.emptyDets[walkers.lastEmpty] = iDet;
        continue;
      }

      // Number of spawnings to attempt
      nAttempts = max(1.0, round(walkers.amps[iDet] * schd.nAttemptsEach));
      parentAmp = walkers.amps[iDet] * schd.nAttemptsEach / nAttempts;

      // Perform one spawning attempt for each 'walker' of weight parentAmp
      for (int iAttempt=0; iAttempt<nAttempts; iAttempt++) {
        generateExcitation(walkers.dets[iDet], childDet, pgen);

        attemptSpawning(walkers.dets[iDet], childDet, spawn, I1, I2, coreE,
                        schd.nAttemptsEach, parentAmp, schd.tau, schd.minSpawn, pgen);
      }
      performDeath(walkers.dets[iDet], walkers.amps[iDet], I1, I2, coreE, Eshift, schd.tau);
    }

    // Perform annihilation
    spawn.communicate();
    spawn.compress();
    spawn.mergeIntoMain(walkers, schd.minPop);
    // Stochastic rounding of small walkers
    walkers.stochasticRoundAll(schd.minPop);

    walkers.calcStats(HFDet, walkerPop, EProj, HFAmp, I1, I2, coreE);
    communicateEstimates(walkerPop, EProj, HFAmp, walkers.nDets, spawn.nDets,
                         walkerPopTot, EProjTot, HFAmpTot, nDetsTot, nSpawnedDetsTot);
    updateShift(Eshift, varyShift, walkerPopTot, walkerPopOldTot, schd.targetPop, schd.shiftDamping, schd.tau);
    printDataTable(iter, nDetsTot, nSpawnedDetsTot, Eshift, walkerPopTot, EProjTot, HFAmpTot, iter_time);

    walkerPopOldTot = walkerPopTot;

    time_end = getTime();
    iter_time = time_end - time_start;
  }

  total_time = getTime() - startofCalc;
  printFinalStats(walkerPop, walkers.nDets, spawn.nDets, total_time);

  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());
  return 0;
}


// Generate a random single or double excitation, and also return the
// probability that it was generated
void generateExcitation(const Determinant& parentDet, Determinant& childDet, double& pgen)
{
  double pSingle = 0.05;
  double pgen_ia, pgen_ijab;

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));

  if (random() < pSingle) {
    generateSingleExcit(parentDet, childDet, pgen_ia);
    pgen = pSingle * pgen_ia;
  } else {
    generateDoubleExcit(parentDet, childDet, pgen_ijab);
    pgen = (1 - pSingle) * pgen_ijab;
  }
}

// Generate a random single excitation, and also return the probability that
// it was generated
void generateSingleExcit(const Determinant& parentDet, Determinant& childDet, double& pgen_ia)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));

  vector<int> AlphaOpen;
  vector<int> AlphaClosed;
  vector<int> BetaOpen;
  vector<int> BetaClosed;

  parentDet.getOpenClosedAlphaBeta(AlphaOpen, AlphaClosed, BetaOpen, BetaClosed);

  childDet   = parentDet;
  int nalpha = AlphaClosed.size();
  int nbeta  = BetaClosed.size();
  int norbs  = Determinant::norbs;

  // Pick a random occupied orbital
  int i = floor(random() * (nalpha + nbeta));
  double pgen_i = 1.0/(nalpha + nbeta);

  // Pick an unoccupied orbital
  if (i < nalpha) // i is alpha
  {
    int a = floor(random() * (norbs - nalpha));
    int I = AlphaClosed[i];
    int A = AlphaOpen[a];

    childDet.setoccA(I, false);
    childDet.setoccA(A, true);
    pgen_ia = pgen_i / (norbs - nalpha);
  }
  else // i is beta
  {
    i = i - nalpha;
    int a = floor( random() * (norbs - nbeta));
    int I = BetaClosed[i];
    int A = BetaOpen[a];

    childDet.setoccB(I, false);
    childDet.setoccB(A, true);
    pgen_ia = pgen_i / (norbs - nbeta);
  }

  //cout << "parent:  " << parentDet << endl;
  //cout << "child:   " << childDet << endl;
}

// Generate a random double excitation, and also return the probability that
// it was generated
void generateDoubleExcit(const Determinant& parentDet, Determinant& childDet, double& pgen_ijab)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));

  vector<int> AlphaOpen;
  vector<int> AlphaClosed;
  vector<int> BetaOpen;
  vector<int> BetaClosed;

  int i, j, a, b, I, J, A, B;

  parentDet.getOpenClosedAlphaBeta(AlphaOpen, AlphaClosed, BetaOpen, BetaClosed);

  childDet   = parentDet;
  int nalpha = AlphaClosed.size();
  int nbeta  = BetaClosed.size();
  int norbs  = Determinant::norbs;
  int nel    = nalpha + nbeta;

  // Pick a combined ij index
  int ij = floor( random() * (nel*(nel-1))/2 ) + 1;
  // The probability of having picked this pair
  double pgen_ij = 2.0 / (nel * (nel-1));

  // Use triangular indexing scheme to obtain (i,j), with j>i
  j = floor(1.5 + sqrt(2*ij - 1.75)) - 1;
  i = ij - (j * (j - 1))/2 - 1;

  bool iAlpha = i < nalpha;
  bool jAlpha = j < nalpha;
  bool sameSpin = iAlpha == jAlpha;

  // Pick a and b
  if (sameSpin) {
    int nvirt;
    if (iAlpha)
    {
      nvirt = norbs - nalpha;
      // Pick a combined ab index
      int ab = floor( random() * (nvirt*(nvirt-1))/2 ) + 1;

      // Use triangular indexing scheme to obtain (a,b), with b>a
      b = floor(1.5 + sqrt(2*ab - 1.75)) - 1;
      a = ab - (b * (b - 1))/2 - 1;

      I = AlphaClosed[i];
      J = AlphaClosed[j];
      A = AlphaOpen[a];
      B = AlphaOpen[b];
    }
    else
    {
      i = i - nalpha;
      j = j - nalpha;

      nvirt = norbs - nbeta;
      // Pick a combined ab index
      int ab = floor( random() * (nvirt * (nvirt-1))/2 ) + 1;

      // Use triangular indexing scheme to obtain (a,b), with b>a
      b = floor(1.5 + sqrt(2*ab - 1.75)) - 1;
      a = ab - (b * (b - 1))/2 - 1;

      I = BetaClosed[i];
      J = BetaClosed[j];
      A = BetaOpen[a];
      B = BetaOpen[b];
    }
    pgen_ijab = pgen_ij * 2.0 / (nvirt * (nvirt-1));
  }
  else
  { // Opposite spin
    if (iAlpha) {
      a = floor(random() * (norbs - nalpha));
      I = AlphaClosed[i];
      A = AlphaOpen[a];

      j = j - nalpha;
      b = floor( random() * (norbs - nbeta));
      J = BetaClosed[j];
      B = BetaOpen[b];
    }
    else
    {
      i = i - nalpha;
      a = floor( random() * (norbs - nbeta));
      I = BetaClosed[i];
      A = BetaOpen[a];

      b = floor(random() * (norbs - nalpha));
      J = AlphaClosed[j];
      B = AlphaOpen[b];
    }
    pgen_ijab = pgen_ij / ( (norbs - nalpha) * (norbs - nbeta) );
  }

  if (iAlpha) {
    childDet.setoccA(I, false);
    childDet.setoccA(A, true);
  } else {
    childDet.setoccB(I, false);
    childDet.setoccB(A, true);
  }

  if (jAlpha) {
    childDet.setoccA(J, false);
    childDet.setoccA(B, true);
  } else {
    childDet.setoccB(J, false);
    childDet.setoccB(B, true);
  }

  //cout << "parent:  " << parentDet << endl;
  //cout << "child:   " << childDet << endl;
}

// Find the weight of the spawned walker
// If it is above amin threshold, then always spawn
// Otherwsie, stochastically round it up to the threshold or down to 0
void attemptSpawning(Determinant& parentDet, Determinant& childDet, spawnFCIQMC& spawn,
                     oneInt &I1, twoInt &I2, double& coreE, const int& nAttemptsEach, const double& parentAmp,
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
    int proc = childDet.getProc();
    // Find the appropriate place in the spawned list for the processor
    // of the newly-spawned walker
    int ind = spawn.currProcSlots[proc];
    spawn.dets[ind] = childDet.getSimpleDet();
    spawn.amps[ind] = childAmp;
    spawn.currProcSlots[proc] += 1;
  }
}

void performDeath(Determinant& parentDet, double& detPopulation, oneInt &I1, twoInt &I2,
                  double& coreE, const double& Eshift, const double& tau)
{
  double parentE = parentDet.Energy(I1, I2, coreE);
  double fac = tau * ( parentE - Eshift );
  detPopulation -= fac * detPopulation;
}

void communicateEstimates(const double& walkerPop, const double& EProj, const double& HFAmp,
                          const int& nDets, const int& nSpawnedDets,
                          double& walkerPopTot, double& EProjTot, double& HFAmpTot,
                          int& nDetsTot, int& nSpawnedDetsTot)
{
#ifdef SERIAL
  walkerPopTot    = walkerPop;
  EProjTot        = EProj;
  HFAmpTot        = HFAmp;
  nDetsTot        = nDets;
  nSpawnedDetsTot = nSpawnedDets;
#else
  MPI_Allreduce(&walkerPop,    &walkerPopTot,     1,  MPI_DOUBLE,  MPI_SUM,  MPI_COMM_WORLD);
  MPI_Allreduce(&EProj,        &EProjTot,         1,  MPI_DOUBLE,  MPI_SUM,  MPI_COMM_WORLD);
  MPI_Allreduce(&HFAmp,        &HFAmpTot,         1,  MPI_DOUBLE,  MPI_SUM,  MPI_COMM_WORLD);
  MPI_Allreduce(&nDets,        &nDetsTot,         1,  MPI_INT,     MPI_SUM,  MPI_COMM_WORLD);
  MPI_Allreduce(&nSpawnedDets, &nSpawnedDetsTot,  1,  MPI_INT,     MPI_SUM,  MPI_COMM_WORLD);
#endif
}

void updateShift(double& Eshift, bool& varyShift, const double& walkerPop,
                 const double& walkerPopOld, const double& targetPop,
                 const double& shiftDamping, const double& tau)
{
  if ((!varyShift) && walkerPop > targetPop) {
    varyShift = true;
  }
  if (varyShift) {
    Eshift = Eshift - (shiftDamping/tau) * log(walkerPop/walkerPopOld);
  }
}

void printDataTableHeader()
{
  if (commrank == 0) {
    printf ("#  1. Iter");
    printf ("     2. nDets");
    printf ("  3. nSpawned");
    printf ("             4. Shift");
    printf ("          5. nWalkers");
    printf ("       6. Energy num.");
    printf ("     7. Energy denom.");
    printf ("    8. Time\n");
  }
}

void printDataTable(const int iter, const int& nDets, const int& nSpawned,
                    const double& shift, const double& walkerPop, const double& EProj,
                    const double& HFAmp, const double& iter_time)
{
  if (commrank == 0) {
    printf ("%10d   ", iter);
    printf ("%10d   ", nDets);
    printf ("%10d   ", nSpawned);
    printf ("%18.10f   ", shift);
    printf ("%18.10f   ", walkerPop);
    printf ("%18.10f   ", EProj);
    printf ("%18.10f   ", HFAmp);
    printf ("%8.4f\n", iter_time);
  }
}

void printFinalStats(const double& walkerPop, const int& nDets,
                     const int& nSpawnDets, const double& total_time)
{
  int parallelReport[commsize];
  double parallelReportD[commsize];

  if (commrank == 0) {
    cout << "# Total time:  " << getTime() - startofCalc << endl;
  }

#ifndef SERIAL
  MPI_Gather(&walkerPop, 1, MPI_DOUBLE, &parallelReportD, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
