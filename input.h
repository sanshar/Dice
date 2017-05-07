/*
Developed by Sandeep Sharma with contributions from James E. Smith and Adam A. Homes, 2017
Copyright (c) 2017, Sandeep Sharma

This file is part of DICE.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef INPUT_HEADER_H
#define INPUT_HEADER_H
#include <vector>
#include <list>
#include <boost/serialization/serialization.hpp>

using namespace std;

struct schedule{
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & davidsonTol				\
      & epsilon2				\
      & epsilon2Large                           \
      & SampleN					\
      & epsilon1				\
      & onlyperturbative			\
      & restart					\
      & fullrestart				\
      & dE					\
      & eps					\
      & prefix					\
      & stochastic				\
      & nblocks					\
      & excitation				\
      & nvirt					\
      & singleList				\
      & io                                      \
      & nroots                                  \
      & nPTiter                                 \
      & DoRDM                                   \
      & DoSpinRDM                               \
      & quasiQ                                  \
      & quasiQEpsilon                           \
      & doSOC                                   \
      & doSOCQDPT                               \
      & randomSeed                              \
      & doGtensor                               \
      & integralFile                            \
      & doResponse                              \
      & responseFile                            \
      & socmultiplier                           \
      & targetError;
  }

public:
  double davidsonTol;
  double epsilon2;
  double epsilon2Large;
  int SampleN;
  std::vector<double> epsilon1;
  bool onlyperturbative;
  bool restart;
  bool fullrestart;
  double dE;
  double eps;
  vector<string> prefix;
  bool stochastic;
  int nblocks;
  int excitation;
  int nvirt;
  bool singleList;
  bool io;
  int nroots;
  int nPTiter;
  bool DoRDM;
  bool DoSpinRDM;
  bool quasiQ;
  double quasiQEpsilon;
  bool doSOC;
  bool doSOCQDPT;
  unsigned int randomSeed;
  bool doGtensor;
  string integralFile;
  bool doResponse;
  string responseFile;
  double socmultiplier;
  double targetError;
};

#endif
