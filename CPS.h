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
#ifndef CPS_HEADER_H
#define CPS_HEADER_H
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

class Determinant;

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
  Correlator (std::vector<int>& pasites, std::vector<int>& pbsites, double iv=1.0) : asites(pasites), bsites(pbsites) {
    if (asites.size()+bsites.size() > 20) {
      std::cout << "Cannot handle correlators of size greater than 20."<<std::endl;
      exit(0);
    }
    std::sort(asites.begin(), asites.end());
    std::sort(bsites.begin(), bsites.end());
    Variables.resize( pow(2,asites.size()+bsites.size()), iv);
  }

/**
 * Takes an occupation number representation of a determinant
 * in the local orbital basis and calculates the overlap with
 * the correlator
 * PARAMS:
 * 
 * Determinant: the occupation number representation as an input
 * 
 * RETURN:
 * the value of the overlap
 */
  double Overlap              (const Determinant& d);

/**
 * Takes an occupation number representation of a determinant
 * in the local orbital basis and calculates the overlap 
 * the correlator and also the overlap of the determinant
 * with respect to the tangent vector w.r.t to all the 
 * parameters in the wavefuncion
 * 
 * PARAMS:
 * 
 * Determinant: the occupation number representation of the determinant
 * grad       : the vector of the gradient. This vector is long and contains
 *              the space for storing gradients with respect to all the parameters
 *              in the wavefunction and not just this CPS and so the index startIndex
 *              is needed to indentify at what index should we start populating the
 *              gradient w.r.t to this Correlator in the vector
 * ovlp       : The overlap of the CPS w.r.t to the Correlator
 * startIndex : The location in the vector gradient from where to start populating
 *              the gradient w.r.t to the current Correlator
 */
  void   OverlapWithGradient  (const Determinant& d, 
			                         Eigen::VectorXd& grad,
                      			   const double& ovlp,
                      			   const long& startIndex);

  friend std::ostream& operator<<(std::ostream& os, const Correlator& c); 
};


#endif
