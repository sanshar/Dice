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
#ifndef CPSSlater_HEADER_H
#define CPSSlater_HEADER_H
#include <vector>
#include <set>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include "Slater.h"
#include "CPS.h"

class oneInt;
class twoInt;
class twoIntHeatBathSHM;
class workingArray;
class Determinant;
class HFWalker;


/**
* This is the wavefunction, it is a product of the CPS and a linear combination of
* slater determinants
*/
class CPSSlater {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & cps
       & slater;
  }

 public:
   CPS cps; //The jastrow factors
   Slater slater; //reference

   double getJastrowFactor(int i, int a, Determinant &dcopy, Determinant &d);
   double getJastrowFactor(int i, int j, int a, int b, Determinant &dcopy, Determinant &d);
   Slater& getRef() { return slater; }

   CPSSlater();
   void initWalker(HFWalker &walk);
   void initWalker(HFWalker &walk, Determinant &d);

   /**
   *This calculates the overlap of the walker with the
   *jastrow and the ciexpansion 
   */
   double Overlap(HFWalker &walk);


  double getOverlapFactor(HFWalker& w, Determinant& dcopy, bool doparity=false);
  double getOverlapFactor(int i, int a, HFWalker& w, bool doparity);
  double getOverlapFactor(int i, int j, int a, int b, HFWalker& w, bool doparity);


   /**
   * This basically calls the overlapwithgradient(determinant, factor, grad)
   */
   void OverlapWithGradient(HFWalker &,
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
   void HamAndOvlp(HFWalker &walk,
                   double &ovlp, double &ham, 
		   workingArray& work, bool fillExcitations = true);

  //d (<n|H|Psi>/<n|Psi>)/dc_i
   void derivativeOfLocalEnergy(HFWalker &,
                              double &factor,
                              Eigen::VectorXd &hamRatio);

   void getVariables(Eigen::VectorXd &v);
   long getNumVariables();
   long getNumJastrowVariables();
   void updateVariables(Eigen::VectorXd &dv);
   void printVariables();
   void writeWave();
   void readWave();
   
   /**
   * This is expensive and is not recommended because 
   * one has to generate the overlap determinants w.r.t to the
   * ciExpansion from scratch
   */
  //double Overlap(Determinant &);
};


#endif
