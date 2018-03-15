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
#ifndef IOWRAPPER_HEADER_H
#define IOWRAPPER_HEADER_H
#include <Eigen/Dense>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/complex.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

using namespace Eigen;

namespace boost {
  namespace serialization {

    template<class Archive>
      void serialize(Archive & ar, MatrixXcd& a, const unsigned int version)
      {
	int dim1 = a.rows(), dim2 = a.cols();
	ar & dim1 & dim2;
	if(dim1 != a.rows() || dim2 != a.cols())
	  a.resize(dim1, dim2);
	for(int i=0;i<a.rows();++i)
	for(int j=0;j<a.cols();++j)
	  ar & a(i,j);
      }

    template<class Archive>
      void serialize(Archive & ar, MatrixXd& a, const unsigned int version)
      {
	int dim1 = a.rows(), dim2 = a.cols();
	ar & dim1 & dim2;
	if(dim1 != a.rows() || dim2 != a.cols())
	  a.resize(dim1, dim2);
	for(int i=0;i<a.rows();++i)
	  for(int j=0;j<a.cols();++j)
	    ar & a(i,j);
      }
  }
}


#endif
