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

void stochastic_round(const double& minPop, double& amp, bool& roundedUp);

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
  walkersFCIQMC(int arrayLength, int DetLenLocal, int nreplicasLocal);
  ~walkersFCIQMC();

  // Function to initialize walkersFCIQMC. Useful if the object is created
  // with the default constructor and needs to be initialized later
  void init(int arrayLength, int DetLenLocal, int nreplicasLocal);

  void stochasticRoundAll(const double& minPop);

  bool allUnoccupied(const int i) const;

  void calcStats(dataFCIQMC& dat, Determinant& HFDet, oneInt& I1, twoInt& I2, double& coreE);

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
