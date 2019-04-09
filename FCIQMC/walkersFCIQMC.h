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

using namespace std;

class Determinant;

void stochastic_round(const double& minPop, double& amp, bool& roundedUp);

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

  walkersFCIQMC(int arrayLength);

  void stochasticRoundAll(const double& minPop);

  void calcStats(Determinant& HFDet, double& walkerPop, double& EProj, double& HFAmp,
                 oneInt& I1, twoInt& I2, double& coreE);

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
