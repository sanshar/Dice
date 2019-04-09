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

#include "global.h"
#include <iostream>
#include <vector>
#include <boost/serialization/serialization.hpp>
#include <Eigen/Dense>

class Determinant;

void stochastic_round(const double& minPop, double& amp, bool& roundedUp) {
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));
  double pAccept = abs(amp)/minPop;
  if (random() < pAccept) {
    amp = copysign(minPop, amp);
    roundedUp = true;
  } else {
    amp = 0.0;
    roundedUp = false;
  }
}

using namespace std;

// Class for main walker list in FCIQMC
class walkersFCIQMC {

 public:
  // The index of the final occupied determinant
  // Note that some determinants with lower indicies may be unoccupied
  int nDets;
  // The list of determinants. The total size is constant, and elements
  // beyond nDets are not filled, and so should not be used
  vector<Determinant> dets;
  // List of walkers amplitudes
  vector<double> amps;
  // Hash table to access the walker array
  unordered_map<Determinant, int> ht;

  // Positions in main walker list which have become unoccupied,
  // and so can be freely overwritten
  vector<int> emptyDets;
  int firstEmpty, lastEmpty;

  walkersFCIQMC(int arrayLength) {
    nDets = 0;
    dets.resize(arrayLength);
    amps.resize(arrayLength, 0.0);
    emptyDets.resize(arrayLength);
    firstEmpty = 0;
    lastEmpty = -1;
  }

  void stochasticRoundAll(const double& minPop) {
    bool keepDet;
    for (int iDet=0; iDet<nDets; iDet++) {

      // To be a valid walker in the main list, there must be a corresponding
      // hash table entry *and* the amplitude must be non-zero
      if ( ht.find(dets[iDet]) != ht.end() && abs(amps[iDet]) > 1.0e-12 ) {
        if (abs(amps[iDet]) < minPop) {
          stochastic_round(minPop, amps[iDet], keepDet);

          if (!keepDet) {
            ht.erase(dets[iDet]);
            lastEmpty += 1;
            emptyDets[lastEmpty] = iDet;
          }
        }

      } else {
        if (abs(amps[iDet]) > 1.0e-12) {
          // This should never happen - the hash table entry should not be
          // removed unless the walker population becomes zero
          cout << "#Error: Non-empty det no hash table entry found." << endl;
          cout << dets[iDet] << "    " << amps[iDet] << endl;
        }
      }

    }
  }

  void calcStats(Determinant& HFDet, double& walkerPop, double& EProj, double& HFAmp,
                 oneInt& I1, twoInt& I2, double& coreE) {

    int excitLevel = 0;
    walkerPop = 0.0;
    EProj = 0.0;
    HFAmp = 0.0;

    for (int iDet=0; iDet<nDets; iDet++) {

      // To be a valid walker in the main list, there must be a corresponding
      // hash table entry *and* the amplitude must be non-zero
      if ( ht.find(dets[iDet]) != ht.end() && abs(amps[iDet]) > 1.0e-12 ) {

        walkerPop += abs(amps[iDet]);
        excitLevel = HFDet.ExcitationDistance(dets[iDet]);

        if (excitLevel == 0) {
          HFAmp = amps[iDet];
          EProj += amps[iDet] * HFDet.Energy(I1, I2, coreE);
        } else if (excitLevel <= 2) {
          EProj += amps[iDet] * Hij(HFDet, dets[iDet], I1, I2, coreE);
        }

      } // If a valid walker

    } // Loop over all entries in list
  }

  // Print the determinants and hash table
  friend ostream& operator<<(ostream& os, const walkersFCIQMC& walkers) {
    os << "Walker list:" << endl;
    for (int i=0; i<walkers.nDets; i++) {
      os << i << "   " << walkers.dets[i] << "   " << walkers.amps[i] << "   " << endl;
    }
    os << "Hash table:" << endl;
    for (auto kv : walkers.ht) {
      os << kv.first << "   " << kv.second << "   " << endl;
    } 
    return os;
  }
  
};
#endif
