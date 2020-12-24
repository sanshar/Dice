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

#include "global.h"
#include "walkersFCIQMC.h"

#include "CorrelatedWavefunction.h"
#include "Jastrow.h"
#include "Slater.h"

void stochastic_round(const double minPop, double& amp, bool& roundedUp) {
  auto random = bind(uniform_real_distribution<double>(0, 1), ref(generator));
  double pAccept = abs(amp)/minPop;
  if (random() < pAccept) {
    amp = copysign(minPop, amp);
    roundedUp = true;
  } else {
    amp = 0.0;
    roundedUp = false;
  }
}

walkersFCIQMC::walkersFCIQMC(int arrayLength, int DetLenLocal, int nreplicasLocal) {
  init(arrayLength, DetLenLocal, nreplicasLocal);
}

// Define a init function, so that a walkersFCIQMC object can be
// initialized after it is constructed, useful in some cases
void walkersFCIQMC::init(int arrayLength, int DetLenLocal, int nreplicasLocal) {
  nDets = 0;
  nreplicas = nreplicasLocal;
  dets.resize(arrayLength);
  diagH.resize(arrayLength);
  localE.resize(arrayLength);
  ovlp.resize(arrayLength);
  amps = allocateAmpsArray(arrayLength, nreplicas, 0.0);
  emptyDets.resize(arrayLength);
  firstEmpty = 0;
  lastEmpty = -1;
  DetLenMin = DetLenLocal;
}

walkersFCIQMC::~walkersFCIQMC() {
  dets.clear();
  diagH.clear();
  localE.clear();
  ovlp.clear();
  deleteAmpsArray(amps);
  ht.clear();
  emptyDets.clear();
}

// return true if all replicas for determinant i are unoccupied
bool walkersFCIQMC::allUnoccupied(const int i) const {
  return all_of(&amps[i][0], &amps[i][nreplicas], [](double x) { return abs(x)<1.0e-12; });
}

void walkersFCIQMC::stochasticRoundAll(const double minPop) {

  for (int iDet=0; iDet<nDets; iDet++) {
    // To be a valid walker in the main list, there must be a corresponding
    // hash table entry *and* the amplitude must be non-zero for a replica
    if ( ht.find(dets[iDet]) != ht.end() && !allUnoccupied(iDet) ) {

      bool keepDetAny = false;

      for (int iReplica=0; iReplica<nreplicas; iReplica++) {
        if (abs(amps[iDet][iReplica]) < minPop) {
          bool keepDet;
          stochastic_round(minPop, amps[iDet][iReplica], keepDet);
          keepDetAny = keepDetAny || keepDet;
        } else {
          keepDetAny = true;
        }
      }

      // If the population is now 0 on all replicas then remove the ht entry
      if (!keepDetAny) {
        ht.erase(dets[iDet]);
        lastEmpty += 1;
        emptyDets[lastEmpty] = iDet;
      }

    } else {

      if ( !allUnoccupied(iDet) ) {
        // This should never happen - the hash table entry should not be
        // removed unless the walker population becomes zero for all replicas
        cout << "#Error: Non-empty det no hash table entry found." << endl;
        // Print determinant and all amplitudes
        cout << dets[iDet];
        for (int iReplica=0; iReplica<nreplicas; iReplica++) {
          cout << "    " << amps[iDet][iReplica];
        }
        cout << endl;

      }
    }

  }
}

void walkersFCIQMC::calcStats(dataFCIQMC& dat, Determinant& HFDet,
                              oneInt& I1, twoInt& I2, double& coreE) {

  int excitLevel = 0;
  double overlapRatio = 0.0;
  std::fill(dat.walkerPop.begin(),   dat.walkerPop.end(), 0.0);
  std::fill(dat.EProj.begin(),       dat.EProj.end(), 0.0);
  std::fill(dat.HFAmp.begin(),       dat.HFAmp.end(), 0.0);
  std::fill(dat.trialEProj.begin(),  dat.trialEProj.end(), 0.0);
  std::fill(dat.ampSum.begin(),      dat.ampSum.end(), 0.0);

  for (int iDet=0; iDet<nDets; iDet++) {

    if ( ht.find(dets[iDet]) != ht.end() ) {
      excitLevel = HFDet.ExcitationDistance(dets[iDet]);

      for (int iReplica=0; iReplica<nreplicas; iReplica++) {

        // To be a valid walker in the main list, there must be a corresponding
        // hash table entry *and* the amplitude must be non-zero
        if ( abs(amps[iDet][iReplica]) > 1.0e-12 ) {

          dat.walkerPop.at(iReplica) += abs(amps[iDet][iReplica]);

          // Trial-WF-based estimator data
          dat.trialEProj.at(iReplica) += localE[iDet] * amps[iDet][iReplica];
          dat.ampSum.at(iReplica) += amps[iDet][iReplica];
          
          // HF-based estimator data
          if (excitLevel == 0) {
            dat.HFAmp.at(iReplica) = amps[iDet][iReplica] / ovlp[iDet];
            dat.EProj.at(iReplica) += amps[iDet][iReplica] * HFDet.Energy(I1, I2, coreE) / ovlp[iDet];
          } else if (excitLevel <= 2) {
            dat.EProj.at(iReplica) += amps[iDet][iReplica] * Hij(HFDet, dets[iDet], I1, I2, coreE) / ovlp[iDet];
          }
        }

      } // Loop over replicas

    } // If hash table entry exists

  } // Loop over all entries in list
}

void walkersFCIQMC::calcPop(vector<double>& walkerPop, vector<double>& walkerPopTot) {

  std::fill(walkerPop.begin(), walkerPop.end(), 0.0);

  for (int iDet=0; iDet<nDets; iDet++) {
    if ( ht.find(dets[iDet]) != ht.end() ) {
      for (int iReplica=0; iReplica<nreplicas; iReplica++) {
        // To be a valid walker in the main list, there must be a corresponding
        // hash table entry *and* the amplitude must be non-zero
        if ( abs(amps[iDet][iReplica]) > 1.0e-12 ) {
          walkerPop.at(iReplica) += abs(amps[iDet][iReplica]);
        }
      } // Loop over replicas
    } // If hash table entry exists
  } // Loop over all entries in list

  // Sum walker population from each process
#ifdef SERIAL
  walkerPopTot = walkerPop;
#else
  MPI_Allreduce(
      &walkerPop.front(),
      &walkerPopTot.front(),
      nreplicas,
      MPI_DOUBLE,
      MPI_SUM,
      MPI_COMM_WORLD
  );
#endif
}
