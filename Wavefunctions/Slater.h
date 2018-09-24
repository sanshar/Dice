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


class oneInt;
class twoInt;
class twoIntHeatBathSHM;
class Determinant;
class HFWalker;
class workingArray;
enum HartreeFock;

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
   vector<Determinant> determinants; //The set of determinants 
   vector<double> ciExpansion;       //The ci expansion
   MatrixXx HforbsA, HforbsB;        //mo coeffs, HforbsA=HforbsB for r/ghf

   //read mo coeffs fomr hf.txt
   void initHforbs();
   
   //init ref dets either by reading from a file or filling the first nelec orbs
   void initDets();

   /**
    * constructor
    */
   Slater();
   void initWalker(HFWalker &walk);
   void initWalker(HFWalker &walk, Determinant &d);

   /**
    * This calculates the overlap of the walker with the
    * the reference
    */
   double Overlap(HFWalker &walk);

   /**
    *  returns < m | Psi0 >/< w.d | Psi0 >, where m is obtained by exciting the walker with 
    * spin orbital excitations i->a, j->b
    */
   double OverlapRatio(int i, int a, HFWalker& w, bool doparity);
   double OverlapRatio(int i, int j, int a, int b, HFWalker& w, bool doparity);

   /**
    * This basically calls the overlapwithgradient(determinant, factor, grad)
    * fills grad with wfn derivatives w.r.t. ci coeffs and mo coeffs 
    */
   void OverlapWithGradient(HFWalker & walk,
                           double &factor,
                           Eigen::VectorBlock<VectorXd> &grad);

  //d (<n|H|Psi>/<n|Psi>)/dc_i
   void derivativeOfLocalEnergy(HFWalker &,
                              double &factor,
                              Eigen::VectorXd &hamRatio);

  //variables are ordered as:
  //cicoeffs of the reference multidet expansion, followed by hforbs (row major)
  //in case of uhf all alpha first followed by beta
  void getVariables(Eigen::VectorBlock<VectorXd> &v);
  long getNumVariables();
  void updateVariables(Eigen::VectorBlock<VectorXd> &v);
  void printVariables();
  vector<Determinant> &getDeterminants() { return determinants; }
  int getNumOfDets() {return determinants.size();}
  vector<double> &getciExpansion() { return ciExpansion; }
  MatrixXd& getHforbsA() { return HforbsA;}
  MatrixXd& getHforbsB() {return HforbsB;}
  MatrixXd& getHforbs(bool sz = 0) { if(sz == 0) return HforbsA; else return HforbsB;}
  
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
