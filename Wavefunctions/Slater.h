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
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <Eigen/Dense>
#include "Determinants.h" 

class oneInt;
class twoInt;
class twoIntHeatBathSHM;
class HFWalker;
class workingArray;
enum HartreeFock {Restricted, UnRestricted, Generalized};


/**
 * This is the wavefunction, it is a linear combination of
 * slater determinants made of Hforbs
 */
class Slater {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & hftype
        & determinants
        & ciExpansion 
        & HforbsA
        & HforbsB;
  }
  
  
 public:
  
  HartreeFock hftype;                       //r/u/ghf
  std::vector<Determinant> determinants; //The set of determinants 
  std::vector<double> ciExpansion;       //The ci expansion
  Eigen::MatrixXd HforbsA, HforbsB;        //mo coeffs, HforbsA=HforbsB for r/ghf

  //read mo coeffs fomr hf.txt
  void initHforbs();
   
  //init ref dets either by reading from a file or filling the first nelec orbs
  void initDets();
  
  /**
   * constructor
   */
  Slater();
  void initWalker(HFWalker &walk) const;
  void initWalker(HFWalker &walk, const Determinant &d) const;

  /**
   * This calculates the overlap of the walker with the
   * the reference
   */
  double Overlap(const HFWalker &walk) const;

  /**
   *  returns < m | Psi0 >/< w.d | Psi0 >, where m is obtained by exciting the walker with 
   * spin orbital excitations i->a, j->b
   */
  double OverlapRatio(int i, int a, const HFWalker& w, bool doparity) const;
  double OverlapRatio(int i, int j, int a, int b, const HFWalker& w, bool doparity) const;

  /**
   * This basically calls the overlapwithgradient(determinant, factor, grad)
   * fills grad with wfn derivatives w.r.t. ci coeffs and mo coeffs 
   */
  void OverlapWithGradient(const HFWalker & walk,
                           const double &factor,
                           Eigen::VectorBlock<Eigen::VectorXd> &grad) const;

  //d (<n|H|Psi>/<n|Psi>)/dc_i
  void derivativeOfLocalEnergy(const HFWalker &,
                               const double &factor,
                               Eigen::VectorXd &hamRatio) const;

  //variables are ordered as:
  //cicoeffs of the reference multidet expansion, followed by hforbs (row major)
  //in case of uhf all alpha first followed by beta
  void getVariables(Eigen::VectorBlock<Eigen::VectorXd> &v) const;
  long getNumVariables() const;
  void updateVariables(const Eigen::VectorBlock<Eigen::VectorXd> &v);
  void printVariables() const;
  const std::vector<Determinant> &getDeterminants() const { return determinants; }
int getNumOfDets() const;
  const std::vector<double> &getciExpansion() const { return ciExpansion; }
  const Eigen::MatrixXd& getHforbsA() const { return HforbsA;}
  const Eigen::MatrixXd& getHforbsB() const  {return HforbsB;}
  const Eigen::MatrixXd& getHforbs(bool sz = 0) const { if(sz == 0) return HforbsA; else return HforbsB;}
    
    //void writeWave();
    //void readWave();
    //void getDetMatrix(Determinant &, Eigen::MatrixXd &alpha, Eigen::MatrixXd &beta);//don't know how to extend to ghf, also non-essential
  
    /**
     * This is expensive and is not recommended because 
     * one has to generate the overlap determinants w.r.t to the
     * ciExpansion from scratch
     * uses getdetmatrix, not used anywhere
     */
    //double Overlap(Determinant &);

};


#endif
