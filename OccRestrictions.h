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
#include <iostream>

using namespace std;
using namespace boost;
class schedule;

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
      ar & minElec
          & maxElec
          & orbs
          & currentElec;
  }
 public:
  int minElec;
  int maxElec;
  vector<int> orbs;
  int currentElec;
  OccRestrictions() : minElec(0), maxElec(10000) {};
  OccRestrictions(int pmin, int pmax, vector<int>& porbs) :
      minElec(pmin), maxElec(pmax), orbs(porbs) {};


  void setElec(vector<int>& closed);
  
  void setElec(int elec);

  bool oneElecAllowed(int i, int a);

  bool twoElecAllowed(int i, int j, int a, int b);

  friend ostream& operator<<(ostream& os, OccRestrictions& occ) {
    os << "restrict, "<<occ.minElec<<","<<occ.maxElec<<"  ";
    for (int i=0; i<occ.orbs.size(); i++)
      os <<","<< occ.orbs[i];
    os << endl;
    return os;
  };
};

void initiateRestrictions(schedule& schd, vector<int>& closed);
bool satisfiesRestrictions(schedule& schd, int i, int a) ;
bool satisfiesRestrictions(schedule& schd, int i, int j, int a, int b) ;


