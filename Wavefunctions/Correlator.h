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
#ifndef Correlator_HEADER_H
#define Correlator_HEADER_H
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

class Determinant;
class BigDeterminant;

/**
 * A correlator contains a tuple of local sites and contains
 * a set of 4^n parameters, where n is the number of sites in the 
 * correlator
 */
class Correlator {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & asites & bsites & Variables;
  }

 public:

  std::vector<int>    asites, 
    bsites;  // Separately store the alpha and beta sites, although in current version they are always the same

  std::vector<double> Variables;    //the total number of variables in a single Correlator 2^{na+nb} number of variables

  //Dummy constructor
  Correlator () {};

  /**
   * Takes the set of alpha sites and beta sites and initial set of parameters
   * Internally these sites are always arranges in asending order
   */
  Correlator (std::vector<int>& pasites, std::vector<int>& pbsites, double iv=1.0);

  /**
   * Takes an occupation number representation of a determinant
   * in the local orbital basis and calculates the overlap with
   * the correlator
   * PARAMS:
   * 
   * Determinant: the occupation number representation as an input
   * 
   * RETURN:
   * <d|correlator>
   */
  double Overlap              (const Determinant& d) const;
  double Overlap              (const BigDeterminant& d) const;

  /**
   * Takes an occupation number representation of two determinants
   * in the local orbital basis and calculates the ratio of overlaps 
   * <d1|correlator>/<d2|correlator>
   * PARAMS:
   * 
   * Determinant: the occupation number representation as an input
   * 
   * RETURN:
   * <d1|correlator>/<d2|correlator>
   */
  double OverlapRatio(const Determinant& d1, const Determinant& d2) const;
  double OverlapRatio(const BigDeterminant& d1, const BigDeterminant& d2) const;

  /**
   * Takes an occupation number representation of a determinant
   * in the local orbital basis and calculates the overlap 
   * the correlator and also the overlap of the determinant
   * with the tangent vector w.r.t to all the 
   * parameters in this correlator
   * 
   * PARAMS:
   * 
   * Determinant: the occupation number representation of the determinant
   * grad       : the vector of the gradient. This vector is long and contains
   *              the space for storing gradients with respect to all the parameters
   *              in the wavefunction and not just this correlator and so the index startIndex
   *              is needed to indentify at what index should we start populating the
   *              gradient w.r.t to this correlator in the vector
   * ovlp       : <d|correlator> 
   * startIndex : The location in the vector gradient from where to start populating
   *              the gradient w.r.t to the current correlator
   */
  void   OverlapWithGradient  (const Determinant& d, 
                               Eigen::VectorBlock<Eigen::VectorXd>& grad,
                               const double& ovlp,
                               const long& startIndex) const;

  friend std::ostream& operator<<(std::ostream& os, const Correlator& c); 
};


#endif
