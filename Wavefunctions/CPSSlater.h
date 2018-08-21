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
#include "Determinants.h"
#include "CPS.h"
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

class oneInt;
class twoInt;
class twoIntHeatBathSHM;
class HFWalker;
class workingArray;

/**
* This is the wavefunction, it is a product of the CPS and a linear combination of
* slater determinants
*/
class CPSSlater {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & cpsArray
       & determinants
       & ciExpansion 
       & orbitalToCPS
       & workingVectorOfCPS
       & HforbsA
       & HforbsB;
  }

 public:
   std::vector<Correlator> cpsArray; //The jastrow factors
   vector<Determinant> determinants; //The set of determinants
   vector<double> ciExpansion;       //The ci expansion
   vector<vector<int>> orbitalToCPS; //for each orbital all CPS that it belongs to
   vector<int> workingVectorOfCPS;
   MatrixXx HforbsA, HforbsB;


   CPSSlater();
   void readDefault();
   void initWalker(HFWalker &walk);
   void initWalker(HFWalker &walk, Determinant &d);

   /**
 * Generates the determinant of overlap of the input determinant d and the
 * zeroth determinant determinants[0] in the wavefunction
 */
   void getDetMatrix(Determinant &, Eigen::MatrixXd &alpha, Eigen::MatrixXd &beta);

   /**
 * This just generates the overlap of a walker with the
 * determinants in the ciExpansion
 */
   double getOverlapWithDeterminants(HFWalker &walk);

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

   double getJastrowFactor(int i, int a, Determinant &dcopy, Determinant &d);
   double getJastrowFactor(int i, int j, int a, int b, Determinant &dcopy, Determinant &d);

   /**
   * This calculates the overlap of the determinant with the
   * gradient of the wavefunction w.r.t to the jastrow parameters,
   * divided by the overlap of the determinant to the wavefunction
   * <d|Psi_x>/<d|Psi>, where x is a jastrow parameter
   * Right now, the walker does not have any useful information to 
   * evaluate this efficiently
   */
   void OverlapWithGradient(Determinant &,
                            double &factor,
                            Eigen::VectorXd &grad);

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
   vector<Determinant> &getDeterminants() { return determinants; }
   vector<double> &getciExpansion() { return ciExpansion; }
   MatrixXd& getHforbsA() {return HforbsA;}
   MatrixXd& getHforbsB() {return HforbsB;}
};


#endif
