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
#ifndef CPSGHFSlater_HEADER_H
#define CPSGHFSlater_HEADER_H
#include <vector>
#include <set>
#include "Determinants.h"
#include "CPS.h"
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

class oneInt;
class twoInt;
class twoIntHeatBathSHM;
class GHFWalker;
class workingArray;

/**
* This is the wavefunction, it is a product of the CPS and a linear combination of
* slater determinants
*/
class CPSGHFSlater {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & cps
       & determinants
       & ciExpansion 
       & GHFOrbs;
  }

 public:
   CPS cps; //The jastrow factors
   vector<Determinant> determinants; //The set of determinants
   vector<double> ciExpansion;       //The ci expansion
   MatrixXx GHFOrbs;


   CPSGHFSlater();
   void readDefault();
   void initWalker(GHFWalker &walk);
   void initWalker(GHFWalker &walk, Determinant &d);

   /**
 * Generates the determinant of overlap of the input determinant d and the
 * zeroth determinant determinants[0] in the wavefunction
 */
  // void getDetMatrix(Determinant &, Eigen::MatrixXd &alpha, Eigen::MatrixXd &beta);

   /**
 * This just generates the overlap of a walker with the
 * determinants in the ciExpansion
 */
   double getOverlapWithDeterminants(GHFWalker &walk);

   /**
   *This calculates the overlap of the walker with the
   *jastrow and the ciexpansion 
   */
   double Overlap(GHFWalker &walk);

   /**
   * This is expensive and is not recommended because 
   * one has to generate the overlap determinants w.r.t to the
   * ciExpansion from scratch
   */
   double Overlap(Determinant &);

   double getOverlapFactor(int i, int a, GHFWalker& w, bool doparity);
   double getOverlapFactor(int i, int j, int a, int b, GHFWalker& w, bool doparity);

   double getJastrowFactor(int i, int a, Determinant &dcopy, Determinant &d);
   double getJastrowFactor(int i, int j, int a, int b, Determinant &dcopy, Determinant &d);


   /**
   * This basically calls the overlapwithgradient(determinant, factor, grad)
   */
   void OverlapWithGradient(GHFWalker &,
                            double &factor,
                            Eigen::VectorXd &grad);

   /**
 * Calculates the overlap, hamiltonian,
 * actually it only calculates 
 * ham      = hamiltonian/overlap
 * 
 * it also is able to calculate the overlap_d'/overlap_d, the ratio of
 * overlaps of all d' connected to the determinant d (walk.d)
 */
   void HamAndOvlp(GHFWalker &walk,
                   double &ovlp, double &ham, 
		   workingArray& work, bool fillExcitations = true);

  //d (<n|H|Psi>/<n|Psi>)/dc_i
   void derivativeOfLocalEnergy(GHFWalker &,
                              double &factor,
                              Eigen::VectorXd &hamRatio);

   void getVariables(Eigen::VectorXd &v);
   long getNumVariables();
   long getNumJastrowVariables();
   void updateVariables(Eigen::VectorXd &dv);
   void printVariables();
   void writeWave();
   void readWave();
   vector<Determinant> &getDeterminants() { return determinants; }
   vector<double> &getciExpansion() { return ciExpansion; }
   MatrixXd& getGHFOrbs() {return GHFOrbs;}
};


#endif
