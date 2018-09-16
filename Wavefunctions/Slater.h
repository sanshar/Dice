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
#ifndef Slater_HEADER_H
#define Slater_HEADER_H
#include <vector>
#include <set>
#include "Determinants.h"
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

class oneInt;
class twoInt;
class twoIntHeatBathSHM;
class HFWalker;
class workingArray;

enum HartreeFock {Restricted, UnRestricted, Generalized};

/**
* This is the wavefunction, it is a product of the CPS and a linear combination of
* slater determinants
*/
class Slater {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & determinants
       & ciExpansion 
        & HforbsA
        & HforbsB;
  }

  HartreeFock hftype;
  vector<Determinant> determinants; //The set of determinants
  vector<double> ciExpansion;       //The ci expansion

  MatrixXx HforbsA, HforbsB;

  
  void readDefault();

  void getDetMatrix(Determinant &, Eigen::MatrixXd &alpha, Eigen::MatrixXd &beta);
  void readBestDeterminant(Determinant&);
  void guessBestDeterminant(Determinant&);
 public:

   Slater();
   void initWalker(HFWalker &walk);
   void initWalker(HFWalker &walk, Determinant &d);



   /**
   *This calculates the overlap of the walker with the
   *jastrow and the ciexpansion 
   */
   double Overlap(HFWalker &walk);

   /**
   * This is expensive and is not recommended because 
   * one has to generate the overlap determinants w.r.t to the
   * ciExpansion from scratch
   */
  double Overlap(Determinant &);

  double OverlapRatio(int i, int a, HFWalker& w, bool doparity);
  double OverlapRatio(int i, int j, int a, int b, HFWalker& w, bool doparity);


   /**
   * This basically calls the overlapwithgradient(determinant, factor, grad)
   */
  void OverlapWithGradient(HFWalker & walk,
                           double &factor,
                           Eigen::VectorBlock<VectorXd> &grad);

  //d (<n|H|Psi>/<n|Psi>)/dc_i
   void derivativeOfLocalEnergy(HFWalker &,
                              double &factor,
                              Eigen::VectorXd &hamRatio);

  void getVariables(Eigen::VectorBlock<VectorXd> &v);

  long getNumVariables();

  void updateVariables(Eigen::VectorBlock<VectorXd> &v);

  void printVariables();
  void writeWave();
  void readWave();
  vector<Determinant> &getDeterminants() { return determinants; }
  vector<double> &getciExpansion() { return ciExpansion; }
  MatrixXd& getHforbsA() {return HforbsA;}
  MatrixXd& getHforbsB() {return HforbsB;}
};


#endif
