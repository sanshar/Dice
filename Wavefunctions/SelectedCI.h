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
#ifndef SelectedCI_HEADER_H
#define SelectedCI_HEADER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include "SimpleWalker.h"
#include <google/dense_hash_map>
//#include <sparsehash/dense_hash_map>


class SelectedCI
{
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    //ar & DetsMap & bestDeterminant;
    ar & bestDeterminant;
  }

 public:
  using CorrType = Determinant;
  using ReferenceType = Determinant;
  
  google::dense_hash_map<shortSimpleDet, double, boost::hash<shortSimpleDet> > DetsMap;
  //unordered_map<Determinant, double, boost::hash<Determinant> > DetsMap;
  Determinant bestDeterminant;

  SelectedCI();
  void readWave();

  void initWalker(SimpleWalker &walk); 

  void initWalker(SimpleWalker &walk, Determinant& d);

  double getOverlapFactor(SimpleWalker& walk, Determinant& dcopy, bool doparity=true) ;
  
  double getOverlapFactor(int I, int A, SimpleWalker& walk, bool doparity);

  double Overlap(SimpleWalker& walk);
  double Overlap(Determinant& d);
  inline double Overlap(shortSimpleDet& d);

  
  double getOverlapFactor(int I, int J, int A, int B,
                          SimpleWalker& walk, bool doparity);
  
  void OverlapWithGradient(SimpleWalker &walk,
			   double &factor,
			   Eigen::VectorXd &grad);

  // This version of HamAndOvlp is the standard version, appropriate when
  // performing VMC in the usual way, using a SelectedCI wave function.
  // This function calculations both ovlp and ham. ovlp is the overlap of
  // walk.d with the selected CI wave function. ham is the local energy
  // on determinant walk.d, including the 1/ovlp factor.
  void HamAndOvlp(SimpleWalker &walk, double &ovlp, double &ham,
                  workingArray& work, double epsilon);

  void HamAndOvlpAndSVTotal(SimpleWalker &walk, double &ovlp,
                            double &ham, double& SVTotal,
                            workingArray& work, double epsilon);

  // This version of HamAndOvlp is used for MRCI and NEVPT calculations,
  // where excitations occur into the first-order interacting space, but
  // the selected CI wave function only has non-zero coefficients
  // within the complete active space.
  // *IMPORTANT* - ham here is <n|H|phi0>, *not* the ratio, to avoid out
  // of active space singularitites. Also, ovlp = ham when ham is calculated.
  void HamAndOvlp(SimpleWalker &walk, double &ovlp, double &ham,
                  workingArray& work, bool dontCalcEnergy=true);
  
  void HamAndOvlpLanczos(SimpleWalker &walk,
                         Eigen::VectorXd &lanczosCoeffsSample,
                         double &ovlpSample,
                         workingArray& work,
                         workingArray& moreWork, double &alpha) ;

  // For some situation, such as FCIQMC, we want to know the ratio of
  // overlaps with the correct parity. This function will calculate
  // this parity, relative to what is returned by getOverlapFactor.
  // For a SelectedCI wave function, this is always equal to 1.
  double parityFactor(Determinant& d, const int ex2, const int i,
                      const int j, const int a, const int b) const {
    return 1.0;
  }

  //void getVariables(Eigen::VectorXd &v);

  //long getNumVariables();

  //void updateVariables(Eigen::VectorXd &v);

  //void printVariables();
  Determinant& getRef() { return bestDeterminant; } //no ref and corr for selectedCI, defined to work with other wavefunctions
  Determinant& getCorr() { return bestDeterminant; }
  std::string getfileName() const { return "SelectedCI"; }
};

#endif
