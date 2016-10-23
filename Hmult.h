#ifndef HMULT_HEADER_H
#define HMULT_HEADER_H
#include <Eigen/Dense>
#include <Eigen/Core>
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "communicate.h"

using namespace Eigen;


struct Hmult2 {
  std::vector<std::vector<int> >& connections;
  std::vector<std::vector<double> >& Helements;
  
  Hmult2(std::vector<std::vector<int> >& connections_, std::vector<std::vector<double> >& Helements_)
  : connections(connections_), Helements(Helements_) {}

  template <typename Derived>
  void operator()(MatrixBase<Derived>& x, MatrixBase<Derived>& y) {
    y*=0.0;
    
#ifndef SERIAL
    boost::mpi::communicator world;
#endif
    
    int num_thrds = omp_get_max_threads();
    if (num_thrds >1) {
      std::vector<MatrixXd> yarray(num_thrds);
      
      for (int i=0; i<num_thrds; i++) {
	yarray[i] = Eigen::MatrixXd::Zero(y.rows(),1);
	//yarray[i] = 0.*y;
      }
      
#pragma omp parallel for schedule(dynamic)
      for (int i=0; i<x.rows(); i++) {
	if ((i/omp_get_num_threads())%world.size() != world.rank()) continue;
	for (int j=0; j<connections[i].size(); j++) {
	  double hij = Helements[i][j];
	  int J = connections[i][j];
	  yarray[omp_get_thread_num()](i,0) += hij*x(J,0);
	  if (i!= J) yarray[omp_get_thread_num()](J,0) += hij*x(i,0);
	}
      }
      
      for (int i=1; i<num_thrds; i++)
	yarray[0] += yarray[i];
      
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &yarray[0](0,0), yarray[0].rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      y = 1.*yarray[0];
    }
    else {
      for (int i=0; i<x.rows(); i++) {
	if (i%world.size() != world.rank()) continue;
	for (int j=0; j<connections[i].size(); j++) {
	  double hij = Helements[i][j];
	  int J = connections[i][j];
	  y(i,0) += hij*x(J,0);
	  if (i!= J) y(J,0) += hij*x(i,0);
	}
      }
      
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &y(0,0), y.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      
    }
  }
  
};
  

#endif
