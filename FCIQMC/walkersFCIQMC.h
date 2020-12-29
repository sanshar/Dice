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
#ifndef walkersFCIQMC_HEADER_H
#define walkersFCIQMC_HEADER_H

#include <iostream>
#include <vector>
#include <unordered_map>
#include "Determinants.h"
#include "dataFCIQMC.h"

class Determinant;

void stochastic_round(const double minPop, double& amp, bool& roundedUp);

template <typename T>
T** allocateAmpsArray(const int nrows, const int ncols, const T& init = T())
{
  // array to hold pointers to the first elements for each row
  T** amps = nullptr;
  // a single block of contiguous memory
  T* pool = nullptr;

  // allocate pointers
  amps = new T*[nrows];
  // allocate the block of memory
  pool = new T[nrows*ncols]{init};

  // point to the first position in each row
  for (int i = 0; i < nrows; ++i, pool += ncols) {
    amps[i] = pool;
  }

  return amps;
}

template <typename T>
void deleteAmpsArray(T** amps)
{
  delete [] amps[0];
  delete [] amps;
}

// Class for main walker list in FCIQMC
template<typename TrialWalk>
class walkersFCIQMC {

 public:
  // The index of the final occupied determinant
  // Note that some determinants with lower indicies may be unoccupied
  int nDets;
  // The number of replicas simulations being performed
  // (i.e. the number of amplitudes to store per determinant)
  int nreplicas;
  // The list of determinants. The total size is constant, and elements
  // beyond nDets are not filled, and so should not be used
  vector<Determinant> dets;
  // List of diagonal Hamiltonian elements for the occupied determinants
  vector<double> diagH;
  // When using a trial wave function, this holds a list of the overlaps
  // between the wave function and the occupied determinants
  vector<double> ovlp;
  // When using a trial wave function, this holds a list of the local
  // energies for the occupied determinants
  vector<double> localE;
  // When using a trial wave function, this holds a list of the sum of
  // the sign violating terms (in the impoprtance-sampled Hamiltonian)
  // for each occupied determinant
  vector<double> SVTotal;
  // VMC walker used to calculate properties involving the trial wave
  // function, such as the overlap and local energy
  vector<TrialWalk> trialWalk;
  // List of walkers amplitudes
  double** amps;
  // Hash table to access the walker array
  unordered_map<Determinant, int> ht;

  // Positions in main walker list which have become unoccupied,
  // and so can be freely overwritten
  vector<int> emptyDets;
  int firstEmpty, lastEmpty;

  // The number of 64-bit integers required to represent (the alpha or beta
  // part of) a determinant
  // Note, this is different to DetLen in global.h
  int DetLenMin;

  walkersFCIQMC() {};

  walkersFCIQMC(int arrayLength, int DetLenLocal, int nreplicasLocal) {
    init(arrayLength, DetLenLocal, nreplicasLocal);
  }

  // Define a init function, so that a walkersFCIQMC object can be
  // initialized after it is constructed, useful in some cases
  void init(int arrayLength, int DetLenLocal, int nreplicasLocal) {
    nDets = 0;
    nreplicas = nreplicasLocal;
    dets.resize(arrayLength);
    diagH.resize(arrayLength);
    ovlp.resize(arrayLength);
    localE.resize(arrayLength);
    SVTotal.resize(arrayLength);
    trialWalk.resize(arrayLength);
    amps = allocateAmpsArray(arrayLength, nreplicas, 0.0);
    emptyDets.resize(arrayLength);
    firstEmpty = 0;
    lastEmpty = -1;
    DetLenMin = DetLenLocal;
  }

  ~walkersFCIQMC() {
    dets.clear();
    diagH.clear();
    ovlp.clear();
    localE.clear();
    SVTotal.clear();
    trialWalk.clear();
    deleteAmpsArray(amps);
    ht.clear();
    emptyDets.clear();
  }

  // return true if all replicas for determinant i are unoccupied
  bool allUnoccupied(const int i) const {
    return all_of(&amps[i][0], &amps[i][nreplicas], [](double x) { return abs(x)<1.0e-12; });
  }

  void stochasticRoundAll(const double minPop) {

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

  void calcStats(dataFCIQMC& dat, Determinant& HFDet, oneInt& I1, twoInt& I2, double& coreE) {

    int excitLevel = 0;
    double overlapRatio = 0.0;
    std::fill(dat.walkerPop.begin(),   dat.walkerPop.end(), 0.0);
    std::fill(dat.EProj.begin(),       dat.EProj.end(), 0.0);
    std::fill(dat.HFAmp.begin(),       dat.HFAmp.end(), 0.0);
    std::fill(dat.trialEProj.begin(),  dat.trialEProj.end(), 0.0);
    std::fill(dat.ampSum.begin(),      dat.ampSum.end(), 0.0);

    for (int iDet=0; iDet<nDets; iDet++) {

      // If using importance sampling then the wave function sampled
      // is psi_i^T*C_i. If not, then it is just C_i. So we have the
      // extra factors of psi_i^T to include in estimators when using
      // importance sampling.
      double ISFactor = 1.0;
      if (schd.importanceSampling) {
        ISFactor = ovlp[iDet];
      }

      if ( ht.find(dets[iDet]) != ht.end() ) {
        excitLevel = HFDet.ExcitationDistance(dets[iDet]);

        for (int iReplica=0; iReplica<nreplicas; iReplica++) {

          // To be a valid walker in the main list, there must be a corresponding
          // hash table entry *and* the amplitude must be non-zero
          if ( abs(amps[iDet][iReplica]) > 1.0e-12 ) {

            dat.walkerPop.at(iReplica) += abs(amps[iDet][iReplica]);

            // Trial-WF-based estimator data
            dat.trialEProj.at(iReplica) += localE[iDet] * amps[iDet][iReplica] * ovlp[iDet] / ISFactor;
            dat.ampSum.at(iReplica) += amps[iDet][iReplica] * ovlp[iDet] / ISFactor;

            // HF-based estimator data
            if (excitLevel == 0) {
              dat.HFAmp.at(iReplica) = amps[iDet][iReplica];
              dat.EProj.at(iReplica) += amps[iDet][iReplica] * HFDet.Energy(I1, I2, coreE) / ISFactor;
            } else if (excitLevel <= 2) {
              dat.EProj.at(iReplica) += amps[iDet][iReplica] * Hij(HFDet, dets[iDet], I1, I2, coreE) / ISFactor;
            }
          }

        } // Loop over replicas

      } // If hash table entry exists

    } // Loop over all entries in list
  }

  void calcPop(vector<double>& walkerPop, vector<double>& walkerPopTot) {

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

  // Print the determinants and hash table
  friend ostream& operator<<(ostream& os, const walkersFCIQMC& walkers) {
    os << "Walker list:" << endl;
    for (int iDet=0; iDet<walkers.nDets; iDet++) {
      os << iDet << "   " << walkers.dets[iDet];
      for (int iReplica=0; iReplica<walkers.nreplicas; iReplica++) {
        os << "   " << walkers.amps[iDet][iReplica];
      }
      os << endl;
    }

    os << "Hash table:" << endl;
    for (auto kv : walkers.ht) {
      os << kv.first << "   " << kv.second << "   " << endl;
    } 
    return os;
  }

};
#endif
