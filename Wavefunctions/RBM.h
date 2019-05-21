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

#ifndef RBM_HEADER_H
#define RBM_HEADER_H
#include "Correlator.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <string>

class Determinant;

class RBM {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize (Archive & ar, const unsigned int version) {
      ar & numHidden;
  }
 public:
  int numHidden; //number of hidden neurons
  Eigen::ArrayXXd wMat; //edge coeffs, column-major in vars
  Eigen::ArrayXd bVec; //hidden neuron local fields
  Eigen::ArrayXd aVec; //visible neuron local fields
  //all visible neuron indices referring to spin-orbitals are ordered in an alpha-beta alternating pattern
  Eigen::ArrayXd bwn; //b + w.n, intermediate, updated by the walker (should be in the walker)
  double coshbwn;//prod cosh(b + w.n)
  //std::array<Eigen::ArrayXXd, 2> intermediates; //doesn't depend on the walker, norb x norb (one for each excitation) matrix for up and down spins

  //reads correlator file and makes cpsArray, orbitalToRBM
  RBM ();
  
  double Overlap(const Determinant& d) const ;

  /*
   * Takes an occupation number representation of two determinants
   * in the local orbital basis and calculates the ratio of overlaps 
   * <d1|RBM>/<d2|RBM>
   * PARAMS:
   * 
   * Determinant: the occupation number representation of dets
   * 
   * RETURN:
   * <d1|RBM>/<d2|RBM>
   *
   */
  double OverlapRatio(const Determinant& d1, const Determinant& d2) const ;
  
  /*
   * return ratio of overlaps of RBM with d and (i->a,j->b)excited-d (=dcopy)
   */
  double OverlapRatio(int i, int a, const Determinant &dcopy, const Determinant &d) const ;
  double OverlapRatio(int i, int j, int a, int b, const Determinant &dcopy, const Determinant &d) const;
  
  /*
   * Takes an occupation number representation of a determinant
   * in the local orbital basis and calculates the overlap 
   * the RBM and also the overlap of the determinant
   * with respect to the tangent vector w.r.t to all the 
   * parameters in the wavefuncion
   * 
   * PARAMS:
   * 
   * Determinant: the occupation number representation of the determinant
   * grad       : the vector of the gradient. This vector is long and contains
   *              the space for storing gradients with respect to all the parameters
   *              in the wavefunction and not just this RBM and so the index startIndex
   *              is needed to indentify at what index should we start populating the
   *              gradient w.r.t to this RBM in the vector
   * ovlp       : <d|RBM>
   * startIndex : The location in the vector gradient from where to start populating
   *              the gradient w.r.t to the current Correlator
   */
  void   OverlapWithGradient  (const Determinant& d, 
                               Eigen::VectorXd& grad,
                               const double& ovlp) const;

  void getVariables(Eigen::VectorXd &v) const ;
  long getNumVariables() const;
  void updateVariables(const Eigen::VectorXd &v);
  void printVariables() const;
  std::string getfileName() const {return "RBM";};
};


#endif
