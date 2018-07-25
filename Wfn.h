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
#ifndef Wfn_HEADER_H
#define Wfn_HEADER_H
#include <vector>
#include <set>
#include "MoDeterminants.h"
#include "Determinants.h"
#include "CPS.h"
#include <boost/serialization/vector.hpp>

class oneInt;
class twoInt;
class twoIntHeatBathSHM;
class Walker;


/**
* This is the wavefunction, it is a product of the CPS and a linear combination of
* slater determinants
*/
class CPSSlater  {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & cpsArray
       & determinants
       & ciExpansion ;
  }

 public:
  std::vector<Correlator> cpsArray    ;     //The jastrow factors
  vector<Determinant>     determinants;     //The set of determinants
  vector<double>          ciExpansion ;     //The ci expansion
  vector<vector<int> >             orbitalToCPS;     //for each orbital all CPS that it belongs to
  vector<int>             workingVectorOfCPS;

  CPSSlater( std::vector<Correlator> & pcpsArray,
             std::vector<Determinant>& pdeterminants,
             std::vector<double>     & pciExpansion
	     );

/**
 * Generates the determinant of overlap of the input determinant d and the
 * zeroth determinant determinants[0] in the wavefunction
 */
  void getDetMatrix(Determinant&, Eigen::MatrixXd& alpha, Eigen::MatrixXd& beta);

/**
 * This just generates the overlap of a walker with the
 * determinants in the ciExpansion
 */
  double getOverlapWithDeterminants(Walker& walk);

  /**
   *This calculates the overlap of the walker with the
   *jastrow and the ciexpansion 
   */
  double Overlap(Walker& walk);

  /**
   * This is expensive and is not recommended because 
   * one has to generate the overlap determinants w.r.t to the
   * ciExpansion from scratch
   */
  double Overlap(Determinant&);
 
  double getJastrowFactor(int i, int a, Determinant& dcopy, Determinant& d);
  double getJastrowFactor(int i, int j, int a, int b, Determinant& dcopy, Determinant& d);

  /**
   * This calculates the overlap of the determinant with the
   * gradient of the wavefunction w.r.t to the jastrow parameters,
   * divided by the overlap of the determinant to the wavefunction
   * <d|Psi_x>/<d|Psi>, where x is a jastrow parameter
   * Right now, the walker does not have any useful information to 
   * evaluate this efficiently
   */ 
  void OverlapWithGradient(Determinant&,
			   double& factor,
			   Eigen::VectorXd& grad);

  /**
   * This basically calls the overlapwithgradient(determinant, factor, grad)
   */ 
  void OverlapWithGradient(Walker&,
			   double& factor,
			   Eigen::VectorXd& grad);



/**
 * Calculates the overlap, hamiltonian and gradient,
 * actually it only calculates 
 * ham      = hamiltonian/overlap
 * gradient = gradient/overlap
 * 
 * it also is able to calculate the overlap_d'/overlap_d, the ratio of
 * overlaps of all d' connected to the determinant d (walk.d)
 */ 
  void HamAndOvlpGradient(Walker& walk,
			  double& ovlp, double& ham, VectorXd& grad,
			  oneInt& I1, twoInt& I2,
			  twoIntHeatBathSHM& I2hb, double& coreE,
			  vector<double>& ovlpRatio, vector<size_t>& excitation1,
			  vector<size_t>& excitation2, vector<double>& HijElement,
			  int& nExcitations,
        bool doGradient=true, bool fillExcitations=true);

/**ham is incremented and returnWalkder, coeffWalker are appended to. 
* so make sure they are empty at the
* begining if you don't want to just append things
* it gives the following output, where i can be nsingles+ndoubles
* <psi|H|D_j> = \sum_i (H_ij/pi) *(<psi|D_i>/<psi|D_j>) <psi|D_j>
*/
  void HamAndOvlpStochastic(Walker& walk,
				    double& ovlp, double& ham,
				    oneInt& I1, twoInt& I2,
				    twoIntHeatBathSHM& I2hb, double& coreE,
				    int nterms,
				    vector<Walker>& returnWalker,
				    vector<double>& coeffWalker,
				    bool fillWalker);

  
  void PTcontribution2ndOrder(Walker& , double& E0,
			      oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb,
			      double& coreE, double& A, double& B, double& C,
			      vector<double>& ovlpRatio, vector<size_t>& excitation1,
			      vector<size_t>& excitation2, bool doGradient=true);

  void PTcontribution3rdOrder(Walker& , double& E0,
			      oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb,
			      double& coreE, double& A2, double& B, double& C, double& A3,
			      vector<double>& ovlpRatio, vector<size_t>& excitation1,
			      vector<size_t>& excitation2, bool doGradient=true);


  void exciteWalker(Walker& w, int excite1, int excite2, int norbs) ;
  void getVariables(Eigen::VectorXd& v);
  long getNumVariables();
  long getNumJastrowVariables();
  void updateVariables(Eigen::VectorXd& dv);
  void printVariables();
  void writeWave();
  void readWave();

 
};

#endif
