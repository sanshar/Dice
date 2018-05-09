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
#ifndef Hmult_HEADER_H
#define Hmult_HEADER_H
#include <Eigen/Dense>
#include <vector>
#ifndef SERIAL
#include "mpi.h"
#endif

struct SparseHam {
  std::vector<std::vector<int> > connections;  
  std::vector<std::vector<double> > Helements;
};

struct Hmult2 {
  SparseHam& sparseHam;

  Hmult2(SparseHam& p_sparseHam) : sparseHam(p_sparseHam) {}
  
  void operator()(double *x, double *y) {
    
  int comm_rank=0, comm_size=1;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
#endif
  int size = comm_size, rank = comm_rank;
    
    int numDets = sparseHam.connections.size(), localDets = sparseHam.connections.size();
#ifndef SERIAL
    MPI_Allreduce(&localDets, &numDets, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
    
    std::vector<double> ytemp(numDets, 0);
    
    for (int i=0; i<sparseHam.connections.size(); i++) {
      for (int j=0; j<sparseHam.connections[i].size(); j++) {
	double hij = sparseHam.Helements[i][j];
	int J = sparseHam.connections[i][j];
	ytemp[i*size+rank] += hij*x[J];
	if (J != i*size+rank)
	  ytemp[J] += hij*x[i*size+rank];
      }
    }
    
#ifndef SERIAL
    if (rank == 0) {
      MPI_Reduce(MPI_IN_PLACE, &ytemp[0],  numDets, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      for (int j=0; j<numDets; j++)
	y[j] = ytemp[j];
    } else {
      MPI_Reduce(&ytemp[0], &ytemp[0],  numDets, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
#else
    for (int j=0; j<numDets; j++)
      y[j] = ytemp[j];
#endif
    
  }; 
}; 

#endif
