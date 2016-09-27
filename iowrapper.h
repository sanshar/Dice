#ifndef IOWRAPPER_HEADER_H
#define IOWRAPPER_HEADER_H
#include <Eigen/Dense>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>


using namespace Eigen;

namespace boost {
  namespace serialization {

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
