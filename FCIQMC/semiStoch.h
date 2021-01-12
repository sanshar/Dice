#ifndef semiStoch_HEADER_H
#define semiStoch_HEADER_H

#include <fstream>
#include <string>
#include <vector>
#include "Determinants.h"
#include "walkersFCIQMC.h"
#include "utilsFCIQMC.h"

class semiStoch {

 public:
  // True if this instance is being used in a semi-stochastic
  // FCIQMC simulation (if init has been called)
  bool doingSemiStoch;
  // The number of determinants in the core space
  int nDets;
  // The number of determinants in the core space on this process
  int nDetsThisProc;
  // The number of determinants and displacements for each process -
  // these is used in MPI communications.
  int* determSizes;
  int* determDispls;
  // The list of determinants in the core space
  vector<simpleDet> dets;
  // The list of determinants in the core space on this process
  vector<simpleDet> detsThisProc;
  // Holds the CI coefficients read in from the selected CI calculation
  vector<double> sciAmps;
  // The number of replica simulations, which determines the width of
  // amps and ampsFull
  int nreplicas;
  // Used to hold the walker amplitudes of the core determinants, and
  // also to store the ouput of the projection each iteration
  double** amps;
  // Used to hold all walker amplitudes from all core determinants,
  // which is obtained by gathering the values in amps before projection
  double** ampsFull;
  // The positions of the core determinants in the main walker list
  vector<int> indices;
  // Deterministic flags of the walkers in the main walker list
  vector<int> flags;
  // Hash table to find the position of a determinant in dets
  unordered_map<simpleDet, int, boost::hash<simpleDet>> ht;
  // The positions of non-zero elements in the core Hamiltonian.
  // pos[i,j] is the j'th non-zero column index in row i.
  vector<vector<int>> pos;
  // The values of the non-zero elements in the core Hamiltonian.
  // ham[i,j] is the j'th non-zero element in row i.
  vector<vector<double>> ham;

  semiStoch() {
    doingSemiStoch = false;
  }

  template<typename Wave, typename TrialWalk>
  void init(std::string SHCIFile, Wave& wave, TrialWalk& walk,
            walkersFCIQMC<TrialWalk>& walkers, int DetLenMin,
            int nreplicasLocal, bool importanceSampling,
            bool semiStochInit, double targetPop, workingArray& work) {

    doingSemiStoch = true;

    nDets = 0;
    nDetsThisProc = 0;
    nreplicas = nreplicasLocal;

    ifstream dump(SHCIFile.c_str());

    int index = 0;
    double bestCoeff = 0.0;

    int orbsToLoopOver;
    int offset;

    orbsToLoopOver = Determinant::norbs;

    // Loop through all lines in the SHCI file
    while (dump.good()) {
      std::string Line;
      std::getline(dump, Line);

      boost::trim_if(Line, boost::is_any_of(", \t\n"));
      
      vector<string> tok;
      boost::split(tok, Line, boost::is_any_of(", \t\n"), boost::token_compress_on);

      if (tok.size() > 2 ) {
        double ci = atof(tok[0].c_str());
        Determinant det ;

        for (int i=0; i<orbsToLoopOver; i++)
        {
          if (boost::iequals(tok[1+i], "2")) {
            det.setoccA(i, true);
            det.setoccB(i, true);
          }
          else if (boost::iequals(tok[1+i], "a")) {
            det.setoccA(i, true);
            det.setoccB(i, false);
          }
          if (boost::iequals(tok[1+i], "b")) {
            det.setoccA(i, false);
            det.setoccB(i, true);
          }
          if (boost::iequals(tok[1+i], "0")) {
            det.setoccA(i, false);
            det.setoccB(i, false);
          }
        }

        nDets += 1;

        int proc = getProc(det, DetLenMin);
        // If the determinant belongs to this process, store it
        if (proc == commrank) {
          nDetsThisProc += 1;
          detsThisProc.push_back(det.getSimpleDet());
          sciAmps.push_back(ci);
        }
        
      }
    }
    // Finished looping over the SHCI file

    // Next we need to accumualte the core determinants from each
    // process, in the correct order (proc 0 dets, proc 1 dets, etc.)

    determSizes = new int[commsize];
    determDispls = new int[commsize];

#ifdef SERIAL
    determSizes[0] = nDetsThisProc;
#else
    MPI_Allgather(&nDetsThisProc, 1, MPI_INTEGER, determSizes, 1,
                  MPI_INTEGER, MPI_COMM_WORLD);
#endif

    determDispls[0] = 0;
    for (int i = 1; i<commsize; i++) {
      determDispls[i] = determDispls[i-1] + determSizes[i-1];
    }

    int determSizesDets[commsize];
    int determDisplsDets[commsize];

    for (int i=0; i<commsize; i++) {
      determSizesDets[i] = determSizes[i] * 2*DetLen;
      determDisplsDets[i] = determDispls[i] * 2*DetLen;
    }

    // Gather the determinants into the dets array
    dets.resize(nDets);
#ifdef SERIAL
    for (int i=0; i<nDetsThisProc; i++) {
      dets[i] = detsThisProc[i];
    }
#else
    MPI_Allgatherv(&detsThisProc.front(), nDetsThisProc*2*DetLen, MPI_LONG,
                   &dets.front(), determSizesDets, determDisplsDets,
                   MPI_LONG, MPI_COMM_WORLD);
#endif

    // Create the hash table, mapping determinants to their position
    // in the full list of core determinants
    for (int i=0; i<nDets; i++) {
      ht[ dets[i] ] = i;
    }

    if (importanceSampling) {
      createCoreHamil_IS(wave, walk);
    } else {
      createCoreHamil();
    }

    // Create arrays that will store the walker amplitudes in the
    // core space
    amps = allocateAmpsArray(nDetsThisProc, nreplicas, 0.0);
    ampsFull = allocateAmpsArray(nDets, nreplicas, 0.0);

    // If true, then initialise from the SCI wave function
    if (semiStochInit) {
      if (importanceSampling) {
        setAmpsToSCIWF_IS(wave, walk, targetPop);
      } else {
        setAmpsToSCIWF(targetPop);
      }
    }

    indices.resize(nDetsThisProc, 0);

    // Add the core determinants to the main list with zero amplitude
    addCoreDetsToMainList(wave, walk, walkers, work);
  }

  void setAmpsToSCIWF(double targetPop) {
    // Set the amplitudes to those from the read-in SCI wave function

    // Find the total population
    double totPop = 0.0, totPopAll = 0.0;
    for (int i=0; i<nDetsThisProc; i++) {
      totPop += abs(sciAmps[i]);
    }

#ifdef SERIAL
    totPopAll = totPop;
#else
    MPI_Allreduce(&totPop, &totPopAll, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    // Factor to scale up the amplitudes to the correct initial
    // walker population
    double fac = targetPop/totPopAll;

    for (int i=0; i<nDetsThisProc; i++) {
      for (int j=0; j<nreplicas; j++) {
        amps[i][j] = sciAmps[i] * fac;
      }
    }
  }

  template<typename Wave, typename TrialWalk>
  void setAmpsToSCIWF_IS(Wave& wave, TrialWalk& walk, double targetPop) {
    // Set the amplitudes to those from the read-in SCI wave function.
    // This uses the importance sampled Hamiltonian, where the amplitudes
    // should be C_i * psi_i^T, i.e. they include a factor from the trial
    // wave function, psi_i^T, also.

    // Find the total population
    double totPop = 0.0, totPopAll = 0.0;
    for (int i=0; i<nDetsThisProc; i++) {
      Determinant det_i(detsThisProc[i]);
      TrialWalk walk_i(wave, det_i);
      double ovlp = wave.Overlap(walk_i);

      totPop += abs(sciAmps[i]*ovlp);
    }

#ifdef SERIAL
    totPopAll = totPop;
#else
    MPI_Allreduce(&totPop, &totPopAll, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    // Factor to scale up the amplitudes to the correct initial
    // walker population
    double fac = targetPop/totPopAll;

    for (int i=0; i<nDetsThisProc; i++) {
      Determinant det_i(detsThisProc[i]);
      TrialWalk walk_i(wave, det_i);
      double ovlp = wave.Overlap(walk_i);

      for (int j=0; j<nreplicas; j++) {
        amps[i][j] = sciAmps[i] * ovlp * fac;
      }
    }
  }

  template<typename Wave, typename TrialWalk>
  void addCoreDetsToMainList(Wave& wave, TrialWalk& walk,
                             walkersFCIQMC<TrialWalk>& walkers,
                             workingArray& work) {

    for (int i = 0; i<nDetsThisProc; i++) {

      Determinant det_i = Determinant(detsThisProc[i]);

      // Is this spawned determinant already in the main list?
      if (walkers.ht.find(det_i) != walkers.ht.end()) {
        int iDet = walkers.ht[det_i];
        for (int iReplica=0; iReplica<nreplicas; iReplica++) {
          double oldAmp = walkers.amps[iDet][iReplica];
          double newAmp = amps[i][iReplica] + oldAmp;
          walkers.amps[iDet][iReplica] = newAmp;
        }
      }
      else
      {
        // New determinant
        int pos = walkers.nDets;
        walkers.dets[pos] = det_i;
        walkers.diagH[pos] = det_i.Energy(I1, I2, coreE);
        TrialWalk newWalk(wave, det_i);
        double ovlp, localE, SVTotal;
        wave.HamAndOvlpAndSVTotal(newWalk, ovlp, localE, SVTotal, work,
                                  schd.importanceSampling, schd.epsilon);
        walkers.ovlp[pos] = ovlp;
        walkers.localE[pos] = localE;
        walkers.SVTotal[pos] = SVTotal;
        walkers.trialWalk[pos] = newWalk;

        // Add in the new walker population
        for (int iReplica=0; iReplica<nreplicas; iReplica++) {
          walkers.amps[pos][iReplica] = amps[i][iReplica];
        }
        walkers.ht[det_i] = pos;

        walkers.nDets += 1;
      }
    }

  }

  template<typename TrialWalk>
  void copyWalkerAmps(walkersFCIQMC<TrialWalk>& walkers) {

    // Run through the main list and copy the core amplitudes to the
    // core.amps array. This is needed for printing the correct stats
    // in the first iteration, for some cases.
    int nCoreFound = 0;
    for (int iDet=0; iDet<walkers.nDets; iDet++) {
      simpleDet det_i = walkers.dets[iDet].getSimpleDet();
      if (ht.find(det_i) != ht.end()) {
        for (int iReplica=0; iReplica<schd.nreplicas; iReplica++) {
          amps[nCoreFound][iReplica] = walkers.amps[iDet][iReplica];
        }
        // Store the position of this core determinant in the main list
        indices[nCoreFound] = iDet;
        nCoreFound += 1;
      }
    }

  }

  ~semiStoch() {
    if (doingSemiStoch) {
      delete[] determSizes;
      delete[] determDispls;
      dets.clear();
      detsThisProc.clear();
      sciAmps.clear();
      deleteAmpsArray(amps);
      deleteAmpsArray(ampsFull);
      indices.clear();
      flags.clear();
      ht.clear();

      for (auto pos_i: pos) {
        pos_i.clear();
      }
      pos.clear();

      for (auto ham_i: ham) {
        ham_i.clear();
      }
      ham.clear();
    }
  }

  void createCoreHamil() {

    // These will be used to hold the positions and elements
    // for each row of the core Hamiltonian
    vector<int> tempPos;
    vector<double> tempHam;

    for (int i=0; i<nDetsThisProc; i++) {
      Determinant det_i(detsThisProc[i]);

      for (int j=0; j<nDets; j++) {
        Determinant det_j(dets[j]);

        double HElem = 0.0;
        if (det_i == det_j) {
          // Don't include the diagonal contributions - these are
          // taken care of in the death step
          HElem = 0.0;
        } else {
          HElem = Hij(det_i, det_j, I1, I2, coreE);
        }

        if (abs(HElem) > 1.e-12) {
          tempPos.push_back(j);
          tempHam.push_back(HElem);
        }
      }

      // Add the elements for this row to the arrays
      pos.push_back(tempPos);
      ham.push_back(tempHam);
      tempPos.clear();
      tempHam.clear();
    }

  }

  template<typename Wave, typename TrialWalk>
  void createCoreHamil_IS(Wave& wave, TrialWalk& walk) {

    // These will be used to hold the positions and elements
    // for each row of the core Hamiltonian
    vector<int> tempPos;
    vector<double> tempHam;

    for (int i=0; i<nDetsThisProc; i++) {
      Determinant det_i(detsThisProc[i]);
      TrialWalk walk_i(wave, det_i);

      for (int j=0; j<nDets; j++) {
        Determinant det_j(dets[j]);

        double HElem = 0.0;
        if (det_i == det_j) {
          // Don't include the diagonal contributions - these are
          // taken care of in the death step
          HElem = 0.0;
        } else {
          HElem = Hij(det_i, det_j, I1, I2, coreE);
        }

        if (abs(HElem) > 1.e-12) {
          // Apply importane sampling factor
          HElem /= wave.getOverlapFactor(walk_i, det_j, true);

          tempPos.push_back(j);
          tempHam.push_back(HElem);
        }
      }

      // Add the elements for this row to the arrays
      pos.push_back(tempPos);
      ham.push_back(tempHam);
      tempPos.clear();
      tempHam.clear();
    }

  }

  void determProjection(double tau) {

    int determSizesAmps[commsize];
    int determDisplsAmps[commsize];

    for (int i=0; i<commsize; i++) {
      determSizesAmps[i] = determSizes[i] * nreplicas;
      determDisplsAmps[i] = determDispls[i] * nreplicas;
    }

#ifdef SERIAL
    for (int i=0; i<nDetsThisProc; i++) {
      for (int j=0; j<nreplicas; j++) {
        ampsFull[i][j] = amps[i][j];
      }
    }
#else
    MPI_Allgatherv(&amps[0][0], nDetsThisProc*nreplicas, MPI_DOUBLE,
                   &ampsFull[0][0], determSizesAmps, determDisplsAmps,
                   MPI_DOUBLE, MPI_COMM_WORLD);
#endif

    // Zero the amps array, which will be used for accumulating the
    // results of the projection
    for (int iDet=0; iDet<nDetsThisProc; iDet++) {
      for (int iReplica=0; iReplica<nreplicas; iReplica++) {
        amps[iDet][iReplica] = 0.0;
      }
    }

    // Perform the multiplication by the core Hamiltonian
    for (int iDet=0; iDet<nDetsThisProc; iDet++) {
      for (int jDet=0; jDet < pos[iDet].size(); jDet++) {
        int colInd = pos[iDet][jDet];
        double HElem = ham[iDet][jDet];

        for (int iReplica=0; iReplica<nreplicas; iReplica++) {
          amps[iDet][iReplica] -= HElem * ampsFull[colInd][iReplica];
        }
      }
    }

    // Now multiply by the time step to get the final projected vector
    for (int iDet=0; iDet<nDetsThisProc; iDet++) {
      for (int iReplica=0; iReplica<nreplicas; iReplica++) {
        amps[iDet][iReplica] *= tau;
      }
    }

  }

  void determAnnihilation(double** walkerAmps) {
    for (int iDet=0; iDet<nDetsThisProc; iDet++) {
      // The position of this core determinant in the main list
      int ind = indices[iDet];
      for (int iReplica=0; iReplica<nreplicas; iReplica++) {
        // Add the deterministic projection amplitudes into the main
        // walker list amplitudes
        walkerAmps[ind][iReplica] += amps[iDet][iReplica];
      }
    }
  }

};

#endif
