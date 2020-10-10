#include <iomanip>
#include "Determinants.h"
#include "evaluateE.h"
#include "global.h"
#include "input.h"
#include "Walker.h"

#include "runFCIQMC.h"

// Perform initialization of variables needed for the runFCIQMC routine.
// In particular the Hartree--Fock determinant and energy, the heat bath
// integrals (hb), and the walker and spawned walker objects.
void initFCIQMC(const int norbs, const int nel, const int nalpha, const int nbeta,
                Determinant& HFDet, double& HFEnergy, heatBathFCIQMC& hb,
                walkersFCIQMC& walkers, spawnFCIQMC& spawn) {

  // The number of 64-bit integers required to represent (the alpha
  // or beta part of) a determinant
  int DetLenAlpha = (nalpha-1)/64 + 1;
  int DetLenBeta = (nalpha-1)/64 + 1;
  int DetLenMin = max(DetLenAlpha, DetLenBeta);

  for (int i = 0; i < nalpha; i++)
    HFDet.setoccA(i, true);
  for (int i = 0; i < nbeta; i++)
    HFDet.setoccB(i, true);

  // Processor that the HF determinant lives on
  int HFDetProc = getProc(HFDet, DetLenMin);

  int walkersSize = schd.targetPop * schd.mainMemoryFac / commsize;
  int spawnSize = schd.targetPop * schd.spawnMemoryFac / commsize;

  // Resize and initialize the walker and spawned walker arrays
  walkers.init(walkersSize, DetLenMin, schd.nreplicas);
  spawn.init(spawnSize, DetLenMin, schd.nreplicas);

  // Set up the walker list to contain a single walker on the HF
  // determinant
  if (HFDetProc == commrank) {
    walkers.dets[0] = HFDet;
    walkers.ht[HFDet] = 0;
    // Set the population on the reference
    for (int iReplica=0; iReplica<schd.nreplicas; iReplica++) {
      walkers.amps[0][iReplica] = schd.initialPop;
    }
    // The number of determinants in the walker list
    walkers.nDets = 1;
  }

  HFEnergy = HFDet.Energy(I1, I2, coreE);
  if (commrank == 0) {
    cout << "Hartree--Fock energy: " << HFEnergy << endl << endl;
  }

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

}

// Perform the main FCIQMC loop
void runFCIQMC(const int norbs, const int nel, const int nalpha, const int nbeta) {

  // Objects needed for the FCIQMC simulation, which will be
  // initialized in initFCIQMC
  double HFEnergy;
  Determinant HFDet;
  heatBathFCIQMC hb;
  walkersFCIQMC walkers;
  spawnFCIQMC spawn;

  initFCIQMC(norbs, nel, nalpha, nbeta, HFDet, HFEnergy, hb, walkers, spawn);

  // ----- FCIQMC data -----
  double initEshift = HFEnergy + schd.initialShift;
  dataFCIQMC dat(schd.nreplicas, initEshift);

  double time_start = 0.0, time_end = 0.0, iter_time = 0.0, total_time = 0.0;
  int nDetsTot, nSpawnedDetsTot;
  // -----------------------

  // Get the initial stats
  walkers.calcStats(dat, HFDet, I1, I2, coreE);
  communicateEstimates(dat, walkers.nDets, spawn.nDets, nDetsTot, nSpawnedDetsTot);
  calcVarEnergy(walkers, spawn, I1, I2, coreE, schd.tau, dat.EVarNumAll, dat.EVarDenomAll);
  dat.walkerPopOldTot = dat.walkerPopTot;

  // Print the initial stats
  printDataTableHeader();
  printDataTable(dat, 0, nDetsTot, nSpawnedDetsTot, iter_time);

  // Main FCIQMC loop
  for (int iter = 1; iter <= schd.maxIter; iter++) {
    time_start = getTime();

    walkers.firstEmpty = 0;
    walkers.lastEmpty = -1;
    spawn.nDets = 0;
    spawn.currProcSlots = spawn.firstProcSlots;

    fill(dat.walkerPop.begin(), dat.walkerPop.end(), 0.0);
    fill(dat.EProj.begin(), dat.EProj.end(), 0.0);
    fill(dat.HFAmp.begin(), dat.HFAmp.end(), 0.0);

    //cout << walkers << endl;

    // Loop over all walkers/determinants
    for (int iDet=0; iDet<walkers.nDets; iDet++) {
      // Is this unoccupied for all replicas? If so, add to the list of empty slots
      if (walkers.allUnoccupied(iDet)) {
        walkers.lastEmpty += 1;
        walkers.emptyDets[walkers.lastEmpty] = iDet;
        continue;
      }

      for (int iReplica=0; iReplica<schd.nreplicas; iReplica++) {
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
        int nAttempts = max(1.0, round(abs(walkers.amps[iDet][iReplica]) * schd.nAttemptsEach));
        double parentAmp = walkers.amps[iDet][iReplica] * schd.nAttemptsEach / nAttempts;

        // Perform one spawning attempt for each 'walker' of weight parentAmp
        for (int iAttempt=0; iAttempt<nAttempts; iAttempt++) {
          double pgen = 0.0, pgen2 = 0.0;
          Determinant childDet, childDet2;

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

    // Perform annihilation of spawned walkers
    spawn.communicate();
    spawn.compress();

    // Calculate energies involving multiple replicas
    if (schd.nreplicas == 2) {
      calcVarEnergy(walkers, spawn, I1, I2, coreE, schd.tau, dat.EVarNumAll, dat.EVarDenomAll);
      if (schd.calcEN2) {
        calcEN2Correction(walkers, spawn, I1, I2, coreE, schd.tau,
                          dat.EVarNumAll, dat.EVarDenomAll, dat.EN2All);
      }
    }

    performDeathAllWalkers(walkers, I1, I2, coreE, dat.Eshift, schd.tau);
    spawn.mergeIntoMain(walkers, schd.minPop, schd.initiator);

    // Stochastic rounding of small walkers
    walkers.stochasticRoundAll(schd.minPop);

    walkers.calcStats(dat, HFDet, I1, I2, coreE);
    communicateEstimates(dat, walkers.nDets, spawn.nDets, nDetsTot, nSpawnedDetsTot);
    updateShift(dat.Eshift, dat.varyShift, dat.walkerPopTot, dat.walkerPopOldTot,
                schd.targetPop, schd.shiftDamping, schd.tau);
    printDataTable(dat, iter, nDetsTot, nSpawnedDetsTot, iter_time);

    dat.walkerPopOldTot = dat.walkerPopTot;

    time_end = getTime();
    iter_time = time_end - time_start;
  }

  total_time = getTime() - startofCalc;
  printFinalStats(dat.walkerPop, walkers.nDets, spawn.nDets, total_time);

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
    for (int i=0; i<schd.nreplicas; i++) {
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

// Calculate and return the numerator and denominator of the variational
// energy estimator. This can be used in replicas FCIQMC simulations.
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

// Calculate the numerator of the second-order Epstein-Nesbet correction
// to the energy. This is for use in replica initiator simulations.
void calcEN2Correction(walkersFCIQMC& walkers, const spawnFCIQMC& spawn, const oneInt& I1,
                       const twoInt& I2, double& coreE, const double& tau, const double& varEnergyNum,
                       const double& varEnergyDenom, double& EN2All)
{
  double EN2 = 0.0;
  double EVar = varEnergyNum / varEnergyDenom;

  // Loop over all spawned walkers
  for (int j=0; j<spawn.nDets; j++) {
    // If this determinant is not already occupied, then we may need
    // to cancel the spawning due to initiator rules:
    if (walkers.ht.find(spawn.dets[j]) == walkers.ht.end()) {
      // If the initiator flag is not set on both replicas, then both
      // spawnings are about to be cancelled
      bitset<max_nreplicas> initFlags(spawn.flags[j]);
      if ( (!initFlags.test(0)) && (!initFlags.test(1)) ) {
        double spawnAmp1 = spawn.amps[j][0];
        double spawnAmp2 = spawn.amps[j][1];
        Determinant fullDet(spawn.dets[j]);
        double HDiag = fullDet.Energy(I1, I2, coreE);
        EN2 += ( spawnAmp1 * spawnAmp2 ) / ( EVar - HDiag );
      }
    }
  }

  EN2 /= tau*tau;

  MPI_Allreduce(&EN2, &EN2All, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

// Perform the death step for the determinant at position iDet in
// the walkers.det array
void performDeath(const int &iDet, walkersFCIQMC& walkers, oneInt &I1, twoInt &I2,
                  double& coreE, const vector<double>& Eshift, const double& tau)
{
  double parentE = walkers.dets[iDet].Energy(I1, I2, coreE);
  for (int iReplica=0; iReplica<schd.nreplicas; iReplica++) {
    double fac = tau * ( parentE - Eshift[iReplica] );
    walkers.amps[iDet][iReplica] -= fac * walkers.amps[iDet][iReplica];
  }
}

// Perform death for *all* walkers in the walker array, held in
// walkers.dets
void performDeathAllWalkers(walkersFCIQMC& walkers, oneInt &I1, twoInt &I2,
                  double& coreE, const vector<double>& Eshift, const double& tau)
{
  for (int iDet=0; iDet<walkers.nDets; iDet++) {
    double parentE = walkers.dets[iDet].Energy(I1, I2, coreE);
    for (int iReplica=0; iReplica<schd.nreplicas; iReplica++) {
      double fac = tau * ( parentE - Eshift[iReplica] );
      walkers.amps[iDet][iReplica] -= fac * walkers.amps[iDet][iReplica];
    }
  }
}

// For values which are calculated on each MPI process, sum these
// quantities over all processes to obtain the final total values.
void communicateEstimates(dataFCIQMC& dat, const int& nDets, const int& nSpawnedDets,
                          int& nDetsTot, int& nSpawnedDetsTot)
{
#ifdef SERIAL
  dat.walkerPopTot = dat.walkerPop;
  dat.EProjTot     = dat.EProj;
  dat.HFAmpTot     = dat.HFAmp;
  nDetsTot         = nDets;
  nSpawnedDetsTot  = nSpawnedDets;
#else
  MPI_Allreduce(&dat.walkerPop.front(), &dat.walkerPopTot.front(), schd.nreplicas,  MPI_DOUBLE,  MPI_SUM,  MPI_COMM_WORLD);
  MPI_Allreduce(&dat.EProj.front(),     &dat.EProjTot.front(),     schd.nreplicas,  MPI_DOUBLE,  MPI_SUM,  MPI_COMM_WORLD);
  MPI_Allreduce(&dat.HFAmp.front(),     &dat.HFAmpTot.front(),     schd.nreplicas,  MPI_DOUBLE,  MPI_SUM,  MPI_COMM_WORLD);
  MPI_Allreduce(&nDets,                 &nDetsTot,                 1,               MPI_INT,     MPI_SUM,  MPI_COMM_WORLD);
  MPI_Allreduce(&nSpawnedDets,          &nSpawnedDetsTot,          1,               MPI_INT,     MPI_SUM,  MPI_COMM_WORLD);
#endif
}

// If the shift has started to vary (if varyShift is true) then update
// the shift estimator here. If not, then check if we should now start to
// vary the shift (if the walker population is above the target).
void updateShift(vector<double>& Eshift, vector<bool>& varyShift, const vector<double>& walkerPop,
                 const vector<double>& walkerPopOld, const double& targetPop,
                 const double& shiftDamping, const double& tau)
{
  for (int iReplica = 0; iReplica<schd.nreplicas; iReplica++) {
    if ((!varyShift.at(iReplica)) && walkerPop[iReplica] > targetPop) {
      varyShift.at(iReplica) = true;
    }
    if (varyShift.at(iReplica)) {
      Eshift.at(iReplica) = Eshift.at(iReplica) -
          (shiftDamping/tau) * log(walkerPop.at(iReplica)/walkerPopOld.at(iReplica));
    }
  }
}

// Print the column labels for the main data table
void printDataTableHeader()
{
  if (commrank == 0) {
    cout << "#  1. Iter";
    cout << "     2. nDets";
    cout << "  3. nSpawned";

    // This is the column label
    int label;

    // This loop is for properties printed on each replica
    for (int iReplica=0; iReplica<schd.nreplicas; iReplica++) {

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

    if (schd.nreplicas == 2) {
      label += 1;
      cout << right << setw(6) << label << ". Var_Energy_Num";
      label += 1;
      cout << right << setw(4) << label << ". Var_Energy_Denom";
      if (schd.calcEN2) {
        label += 1;
        cout << right << setw(6) << label << ". EN2_Numerator";
      }
    }

    label += 1;
    cout << right << setw(5) << label << ". Time\n";
  }
}

void printDataTable(const dataFCIQMC& dat, const int iter, const int& nDets, const int& nSpawned,
                    const double& iter_time)
{
  if (commrank == 0) {
    printf ("%10d   ", iter);
    printf ("%10d   ", nDets);
    printf ("%10d   ", nSpawned);

    for (int iReplica=0; iReplica<schd.nreplicas; iReplica++) {
      printf ("%18.10f   ", dat.Eshift[iReplica]);
      printf ("%18.10f   ", dat.walkerPopTot[iReplica]);
      printf ("%18.10f   ", dat.EProjTot[iReplica]);
      printf ("%18.10f   ", dat.HFAmpTot[iReplica]);
    }

    if (schd.nreplicas == 2) {
      printf ("%.12e    ", dat.EVarNumAll);
      printf ("%.12e   ", dat.EVarDenomAll);
      if (schd.calcEN2) {
        printf ("%.12e   ", dat.EN2All);
      }
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
