/*
  Developed by Sandeep Sharma with contributions from James E. T. Smith and Adam
  A. Holmes, 2017 Copyright (c) 2017, Sandeep Sharma

  This file is part of DICE.

  This program is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or (at your option) any later
  version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with
  this program. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include <vector>
#include <boost/serialization/serialization.hpp>

using namespace std;
using namespace boost;

//this class imposes restrictions on the number of electrons in a given
//subset of orbitals. It works by restricting the determinants that are
//generated during excitation. It stores the inital number of electrons
//in some orbitals in a determinant. For every excitation out of the
//determinant it return whether that determinant is allowed or not

struct OccRestrictions {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    for (int i=0; i<DetLen/2; i++)
      ar & minElec
          & maxElec
          & orbs
          & initElec;
  }

  int minElec;
  int maxElec;
  vector<int> orbs;
  int currentElec;
  
  OccRestrictions(int pmin, int pmax, vector<int>& porbs) :
      minElec(pmin), maxElec(pmax), orbs(porbs) {};


  void setElec(vector<int>& closed) {
    for (int i=0; i<closed.size(); i++)
      for (int x=0; x<orbs.size(); x++)
        if (closed[i] == orbs[x]) currentElec++;
  }
  
  void setElec(int elec){
    currentElec = elec;
  }

  bool oneElecAllowed(int i, int a) {
    int elec = currentElec;
    for (int x=0; x<orbs.size(); x++) {
      if (orbs[x] == i) elec--;
      if (orbs[x] == a) elec++;
    }

    if (elec < minElec || elec > maxElec) return false;
    return true;
  }

  bool twoElecAllowed(int i, int j, int a, int b) {
    int elec = currentElec;
    for (int x=0; x<orbs.size(); x++) {
      if (orbs[x] == i || orbs[x] == j) elec--;
      if (orbs[x] == a || orbs[x] == b) elec++;
    }

    if (elec < minElec || elec > maxElec) return false;
    return true;
  }
  
};

