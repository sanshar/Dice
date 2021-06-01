/*
  Developed by Sandeep Sharma with contributions from James E. T. Smith and Adam A. Holmes, 2017
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
#ifndef INPUT_HEADER_H
#define INPUT_HEADER_H
#include <vector>
#include <list>
#include <boost/serialization/serialization.hpp>
#include "Determinants.h"
#include "OccRestrictions.h"

using namespace std;

enum davidsonType {DIRECT, DISK, MEMORY};
enum rdmType {RELAXED, UNRELAXED};

struct schedule{
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & davidsonTol                          \
    & davidsonTolLoose                        \
    & RdmType                                 \
    & DavidsonType                            \
    & epsilon2                                \
    & epsilon2Large                           \
    & SampleN                                 \
    & epsilon1                                \
    & onlyperturbative                        \
    & restart                                 \
    & fullrestart                             \
    & dE                                      \
    & eps                                     \
    & prefix                                  \
    & stochastic                              \
    & nblocks                                 \
    & excitation                              \
    & nvirt                                   \
    & singleList                              \
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
    & targetError                             \
    & num_thrds                               \
    & Trev                                    \
    & algorithm                               \
    & outputlevel                             \
    & printBestDeterminants                   \
    & writeBestDeterminants                   \
    & extrapolate                             \
    & extrapolationFactor                     \
    & enforceSeniority                        \
    & maxSeniority                            \
    & enforceExcitation                       \
    & maxExcitation                           \
    & HF                                      \
    & enforceSenioExc                         \
    & ncore                                   \
    & nact                                    \
    & doLCC                                   \
    & pointGroup                              \
    & spin                                    \
    & irrep                                   \
    & DoSpinOneRDM                            \
    & DoOneRDM                                \
    & DoThreeRDM                              \
    & DoFourRDM                               \
    & restrictions;
  }

public:
  double davidsonTol;
  double davidsonTolLoose;
  rdmType RdmType;
  davidsonType DavidsonType;
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
  int num_thrds;
  char Trev;
  int algorithm;
  int outputlevel;
  int printBestDeterminants;
  int writeBestDeterminants;
  bool extrapolate;
  double extrapolationFactor;
  bool enforceSeniority;
  int maxSeniority;
  bool enforceExcitation;
  int maxExcitation;
  Determinant HF;
  bool enforceSenioExc;
  int ncore;
  int nact;
  bool doLCC;
  string pointGroup;
  int spin;
  int irrep;
  bool DoSpinOneRDM;
  bool DoOneRDM;
  bool DoThreeRDM;
  bool DoFourRDM;

  vector<OccRestrictions> restrictions;
};

#endif
