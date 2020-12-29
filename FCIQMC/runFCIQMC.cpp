#include <iomanip>
#include "Determinants.h"
#include "evaluateE.h"
#include "global.h"
#include "input.h"

#include "runFCIQMC.h"
#include "walkersFCIQMC.h"

#include "CorrelatedWavefunction.h"
#include "Jastrow.h"
#include "Slater.h"
#include "SelectedCI.h"
#include "trivialWF.h"
#include "trivialWalk.h"

// Perform initialization of variables needed for the runFCIQMC routine.
// In particular the Hartree--Fock determinant and energy, the heat bath
// integrals (hb), and the walker and spawned walker objects.
template<typename Wave, typename TrialWalk>
void initFCIQMC(Wave& wave, TrialWalk& walk,
                const int norbs, const int nel, const int nalpha, const int nbeta,
                Determinant& HFDet, double& HFEnergy, heatBathFCIQMC& hb,
                walkersFCIQMC<TrialWalk>& walkers, spawnFCIQMC& spawn,
                workingArray& work) {

  // The number of 64-bit integers required to represent (the alpha
  // or beta part of) a determinant
  int DetLenAlpha = (nalpha-1)/64 + 1;
  int DetLenBeta = (nalpha-1)/64 + 1;
  int DetLenMin = max(DetLenAlpha, DetLenBeta);

  for (int i = 0; i < nalpha; i++)
    HFDet.setoccA(i, true);
  for (int i = 0; i < nbeta; i++)
    HFDet.setoccB(i, true);

  int walkersSize = schd.targetPop * schd.mainMemoryFac / commsize;
  int spawnSize = schd.targetPop * schd.spawnMemoryFac / commsize;

  // Resize and initialize the walker and spawned walker arrays
  walkers.init(walkersSize, DetLenMin, schd.nreplicas);
  spawn.init(spawnSize, DetLenMin, schd.nreplicas);

  HFEnergy = HFDet.Energy(I1, I2, coreE);

  // Set up the initial walker list
  if (schd.trialInitFCIQMC) {
    initWalkerListTrialWF(wave, walk, walkers, spawn, work);
  } else {
    initWalkerListHF(wave, walk, HFDet, DetLenMin, HFEnergy, walkers, work);
  }

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

// This routine places all intial walkers on the Hartree--Fock determinant,
// and sets all attributes in the walkersFCIQMC object as appropriate for
// this state.
template<typename Wave, typename TrialWalk>
void initWalkerListHF(Wave& wave, TrialWalk& walk, Determinant& HFDet,
                      const int DetLenMin, double& HFEnergy,
                      walkersFCIQMC<TrialWalk>& walkers, workingArray& work) {
  // Processor that the HF determinant lives on
  int HFDetProc = getProc(HFDet, DetLenMin);

  if (HFDetProc == commrank) {
    walkers.dets[0] = HFDet;
    walkers.diagH[0] = HFEnergy;
    walkers.ht[HFDet] = 0;

    double HFOvlp, HFLocalE, HFSVTotal;
    TrialWalk HFWalk(wave, HFDet);
    wave.HamAndOvlpAndSVTotal(HFWalk, HFOvlp, HFLocalE, HFSVTotal, work,
                              schd.importanceSampling, schd.epsilon);
    walkers.ovlp[0] = HFOvlp;
    walkers.localE[0] = HFLocalE;
    walkers.SVTotal[0] = HFSVTotal;

    // Set the population on the reference
    for (int iReplica=0; iReplica<schd.nreplicas; iReplica++) {
      walkers.amps[0][iReplica] = schd.initialPop;
    }
    // The number of determinants in the walker list
    walkers.nDets = 1;
  }
}

// This routine creates the initial walker distribution by sampling the
// VMC wave function, using the CTMC algorithm. These are placed in the
// spawned array, and then communicated to the correct process,
// compressed (annihilating repeated determinants), and then moved to
// the walker array (merged). The walker population is then corrected
// with a constant multiplicative factor.
template<typename Wave, typename TrialWalk>
void initWalkerListTrialWF(Wave& wave, TrialWalk& walk, walkersFCIQMC<TrialWalk>& walkers,
                           spawnFCIQMC& spawn, workingArray& work) {
  double ham, ovlp;
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));

  // Split the generation of determinants across processes
  int nDetsThisProc = schd.initialNDets / commsize;
  // Select a new walker every 30 iterations
  const int nitersPerWalker = 30;
  int niter = nitersPerWalker * nDetsThisProc * schd.nreplicas;

  for (int iter=0; iter<niter; iter++) {
    wave.HamAndOvlp(walk, ovlp, ham, work);

    // If importance sampling is in use then the initial wave function
    // to sample has coefficients psi_i^2. If not, then the wave
    // function to sample has coefficients psi_i. In this latter case,
    // we sample psi_i^2 then include a factor of 1/ovlp (=1/psi_i)
    // to get the desired initial WF.
    double ISFactor = 1.0;
    if (! schd.importanceSampling) {
      ISFactor = 1/ovlp;
    }

    double cumOvlpRatio = 0;
    for (int i = 0; i < work.nExcitations; i++)
    {
      cumOvlpRatio += abs(work.ovlpRatio[i]);
      work.ovlpRatio[i] = cumOvlpRatio;
    }

    if ((iter+1) % 30 == 0) {
      // Put the current walker in the FCIQMC walker list. This
      // needs top be put in the spawning list to correctly perform
      // annihilation and send walkers to the correct processor.
      int proc = getProc(walk.d, spawn.DetLenMin);
      // The position in the spawned list for the walker
      int ind = spawn.currProcSlots[proc];
      int iReplica = iter / (nitersPerWalker * nDetsThisProc);
      spawn.dets[ind] = walk.d.getSimpleDet();
      // Set the amplitude on the correct replica
      for (int iSgn=0; iSgn<schd.nreplicas; iSgn++) {
        if (iSgn == iReplica) {
          spawn.amps[ind][iSgn] = ISFactor/cumOvlpRatio;
        } else {
          spawn.amps[ind][iSgn] = 0.0;
        }
      }
      spawn.currProcSlots[proc] += 1;
    }

    double nextDetRandom = random()*cumOvlpRatio;
    int nextDet = std::lower_bound(
        work.ovlpRatio.begin(),
		    work.ovlpRatio.begin()+work.nExcitations,
        nextDetRandom
    ) - work.ovlpRatio.begin();


    walk.updateWalker(
        wave.getRef(),
        wave.getCorr(),
        work.excitation1[nextDet],
        work.excitation2[nextDet]
    );
  }

  // Communicate and compress spawned walkers, then move them
  // to the main walker list
  spawn.communicate();
  spawn.compress();
  spawn.mergeIntoMain_NoInitiator(wave, walk, walkers, 0.0, work);

  vector<double> walkerPop, walkerPopTot;
  walkerPop.resize(schd.nreplicas, 0.0);
  walkerPopTot.resize(schd.nreplicas, 0.0);
  walkers.calcPop(walkerPop, walkerPopTot);

  // The population we want divided by the population we have
  vector<double> popFactor;
  popFactor.resize(schd.nreplicas, 0.0);
  for (int iReplica=0; iReplica<schd.nreplicas; iReplica++) {
    popFactor[iReplica] = schd.initialPop / walkerPopTot[iReplica];
  }

  // Renormalize the walkers to get the correct initial population
  for (int iDet=0; iDet<walkers.nDets; iDet++) {
    for (int iReplica=0; iReplica<schd.nreplicas; iReplica++) {
      walkers.amps[iDet][iReplica] *= popFactor[iReplica];
    }
  }

}

// Perform the main FCIQMC loop
template<typename Wave, typename TrialWalk>
void runFCIQMC(Wave& wave, TrialWalk& walk, const int norbs, const int nel,
               const int nalpha, const int nbeta) {

  // Objects needed for the FCIQMC simulation, which will be
  // initialized in initFCIQMC
  double HFEnergy;
  Determinant HFDet;
  heatBathFCIQMC hb;
  walkersFCIQMC<TrialWalk> walkers;
  spawnFCIQMC spawn;

  // Needed for calculating local energies
  workingArray work;

  initFCIQMC(wave, walk, norbs, nel, nalpha, nbeta,
      HFDet, HFEnergy, hb, walkers, spawn, work);

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
  for (int iter = 1; iter <= schd.maxIterFCIQMC; iter++) {
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

      TrialWalk& parentWalk = walkers.trialWalk[iDet];

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
          int ex1, ex2;
          Determinant childDet, childDet2;

          generateExcitation(hb, I1, I2, walkers.dets[iDet], nel, childDet, childDet2,
                             pgen, pgen2, ex1, ex2);

          // pgen=0.0 is set when a null excitation is returned.
          if (pgen > 1.e-15) {
            attemptSpawning(wave, parentWalk, walkers.dets[iDet], childDet, spawn, I1, I2,
                            coreE, schd.nAttemptsEach, parentAmp, parentFlags, iReplica,
                            schd.tau, schd.minSpawn, pgen, ex1, ex2);
          }
          if (pgen2 > 1.e-15) {
            attemptSpawning(wave, parentWalk, walkers.dets[iDet], childDet2, spawn, I1, I2,
                            coreE, schd.nAttemptsEach, parentAmp, parentFlags, iReplica,
                            schd.tau, schd.minSpawn, pgen2, ex1, ex2);
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
    spawn.mergeIntoMain(wave, walk, walkers, schd.minPop, schd.initiator, work);

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
template<typename Wave, typename TrialWalk>
void attemptSpawning(Wave& wave, TrialWalk& walk, Determinant& parentDet, Determinant& childDet,
                     spawnFCIQMC& spawn, oneInt &I1, twoInt &I2, double& coreE,
                     const int nAttemptsEach, const double parentAmp, const int parentFlags,
                     const int iReplica, const double tau, const double minSpawn,
                     const double pgen, const int ex1, const int ex2)
{
  bool childSpawned = true;

  double HElem = Hij(parentDet, childDet, I1, I2, coreE);

  double overlapRatio;
  if (schd.applyNodeFCIQMC || schd.importanceSampling) {
    // Calculate the ratio of overlaps, and the parity factor
    int norbs = Determinant::norbs;
    int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
    int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
    overlapRatio = wave.getOverlapFactor(I, J, A, B, walk, false);
    double parityFac = wave.parityFactor(parentDet, ex2, I, J, A, B);
    overlapRatio *= parityFac;
  }

  // Apply the fixed/partial node approximation
  if (schd.applyNodeFCIQMC) {
    if (HElem * overlapRatio > 0.0) {
      HElem *= (1.0 - schd.partialNodeFactor);
    }
  }

  if (schd.importanceSampling) {
    HElem *= overlapRatio;
  }

  double pgen_tot = pgen * nAttemptsEach;
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
template<typename TrialWalk>
void calcVarEnergy(walkersFCIQMC<TrialWalk>& walkers, const spawnFCIQMC& spawn,
                   const oneInt& I1, const twoInt& I2, double& coreE, const double tau,
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
template<typename TrialWalk>
void calcEN2Correction(walkersFCIQMC<TrialWalk>& walkers, const spawnFCIQMC& spawn, const oneInt& I1,
                       const twoInt& I2, double& coreE, const double tau, const double varEnergyNum,
                       const double varEnergyDenom, double& EN2All)
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
template<typename TrialWalk>
void performDeath(const int iDet, walkersFCIQMC<TrialWalk>& walkers, oneInt &I1, twoInt &I2,
                  double& coreE, const vector<double>& Eshift, const double tau)
{
  double parentE;
  if (schd.diagonalDumping) {
    parentE = walkers.diagH[iDet] + walkers.SVTotal[iDet] * schd.partialNodeFactor;
  } else {
    parentE = walkers.diagH[iDet];
  }
  for (int iReplica=0; iReplica<schd.nreplicas; iReplica++) {
    double fac = tau * ( parentE - Eshift[iReplica] );
    walkers.amps[iDet][iReplica] -= fac * walkers.amps[iDet][iReplica];
  }
}

// Perform death for *all* walkers in the walker array, held in
// walkers.dets
template<typename TrialWalk>
void performDeathAllWalkers(walkersFCIQMC<TrialWalk>& walkers, oneInt &I1, twoInt &I2,
                  double& coreE, const vector<double>& Eshift, const double tau)
{
  for (int iDet=0; iDet<walkers.nDets; iDet++) {
    double parentE;
    if (schd.diagonalDumping) {
      parentE = walkers.diagH[iDet] + walkers.SVTotal[iDet] * schd.partialNodeFactor;
    } else {
      parentE = walkers.diagH[iDet];
    }
    for (int iReplica=0; iReplica<schd.nreplicas; iReplica++) {
      double fac = tau * ( parentE - Eshift[iReplica] );
      walkers.amps[iDet][iReplica] -= fac * walkers.amps[iDet][iReplica];
    }
  }
}

// For values which are calculated on each MPI process, sum these
// quantities over all processes to obtain the final total values.
void communicateEstimates(dataFCIQMC& dat, const int nDets, const int nSpawnedDets,
                          int& nDetsTot, int& nSpawnedDetsTot)
{
#ifdef SERIAL
  dat.walkerPopTot  = dat.walkerPop;
  dat.EProjTot      = dat.EProj;
  dat.HFAmpTot      = dat.HFAmp;
  dat.trialEProjTot = dat.trialEProj;
  dat.ampSumTot     = dat.ampSum;
  nDetsTot          = nDets;
  nSpawnedDetsTot   = nSpawnedDets;
#else
  MPI_Allreduce(&dat.walkerPop.front(),  &dat.walkerPopTot.front(),  schd.nreplicas, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&dat.EProj.front(),      &dat.EProjTot.front(),      schd.nreplicas, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&dat.HFAmp.front(),      &dat.HFAmpTot.front(),      schd.nreplicas, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&dat.trialEProj.front(), &dat.trialEProjTot.front(), schd.nreplicas, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&dat.ampSum.front(),     &dat.ampSumTot.front(),     schd.nreplicas, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&nDets,                  &nDetsTot,                  1,              MPI_INT,    MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&nSpawnedDets,           &nSpawnedDetsTot,           1,              MPI_INT,    MPI_SUM, MPI_COMM_WORLD);
#endif
}

// If the shift has started to vary (if varyShift is true) then update
// the shift estimator here. If not, then check if we should now start to
// vary the shift (if the walker population is above the target).
void updateShift(vector<double>& Eshift, vector<bool>& varyShift,
                 const vector<double>& walkerPop,
                 const vector<double>& walkerPopOld, const double targetPop,
                 const double shiftDamping, const double tau)
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

    int nColPerReplica;
    if (schd.trialWFEstimator) {
      nColPerReplica = 6;
    } else {
      nColPerReplica = 4;
    }

    // This loop is for properties printed on each replica
    for (int iReplica=0; iReplica<schd.nreplicas; iReplica++) {

      for (int j=0; j<nColPerReplica; j++) {
        string header;
        label = nColPerReplica*iReplica + j + 4;
        header.append(to_string(label));

        if (j==0) {
          header.append(". Shift_");
        } else if (j==1) {
          header.append(". nWalkers_");
        } else if (j==2) {
          header.append(". Energy_Num_");
        } else if (j==3) {
          header.append(". Energy_Denom_");
        } else if (j==4) {
          header.append(". Trial_E_Num_");
        } else if (j==5) {
          header.append(". Trial_E_Denom_");
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

void printDataTable(const dataFCIQMC& dat, const int iter, const int nDets,
                    const int nSpawned, const double iter_time)
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
      if (schd.trialWFEstimator) {
        printf ("%18.10f   ", dat.trialEProjTot[iReplica]);
        printf ("%18.10f   ", dat.ampSumTot[iReplica]);
      }
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

void printFinalStats(const vector<double>& walkerPop, const int nDets,
                     const int nSpawnDets, const double total_time)
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

// Instantiate needed templates
template void runFCIQMC(TrivialWF& wave, TrivialWalk& walk,
                        const int norbs, const int nel,
                        const int nalpha, const int nbeta);

template void runFCIQMC(CorrelatedWavefunction<Jastrow, Slater>& wave,
                        Walker<Jastrow, Slater>& walk,
                        const int norbs, const int nel,
                        const int nalpha, const int nbeta);

template void runFCIQMC(SelectedCI& wave, SimpleWalker& walk,
                        const int norbs, const int nel,
                        const int nalpha, const int nbeta);
