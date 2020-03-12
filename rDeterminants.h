/*
  Developed by Sandeep Sharma with contributions from James E. T. Smith and Adam A. Holmes, 2017
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
#ifndef RDeterminants_HEADER_H
#define RDeterminants_HEADER_H

#include "global.h"
#include <iostream>
#include <vector>
#include <boost/serialization/serialization.hpp>
#include <Eigen/Dense>


struct rDeterminant {
  std::vector<Eigen::Vector3d> coord;
  static int nalpha;
  static int nbeta;
  static int nelec;

  rDeterminant() {
    coord.resize(nelec);
  }

  friend std::ostream& operator<<(std::ostream& os, const rDeterminant& d) {
    for (int i=0; i<d.coord.size(); i++)
      os << "{" << d.coord[i](0) << ", " << d.coord[i](1) << ", " << d.coord[i](2) << "}," << std::endl;
    return os;
  }
  
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & coord;
  }
  
};

#endif
