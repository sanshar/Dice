/*
Developed by Sandeep Sharma with contributions from James E. Smith and Adam A. Homes, 2017
Copyright (c) 2017, Sandeep Sharma

This file is part of DICE.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef SHCI_SHM_H
#define SHCI_SHM_H
#include <vector>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "global.h"
#include <boost/interprocess/managed_shared_memory.hpp>
#include "hdf5.h"

using namespace Eigen;

void initSHM();
void removeSHM();

template <typename T>
void SHMVecFromVecs(std::vector<T>& vec, T* &SHMvec, std::string& SHMname, 
		    boost::interprocess::shared_memory_object& SHMsegment,
		    boost::interprocess::mapped_region& SHMregion) {

  size_t totalMemory = 0;
  int comm_rank=0, comm_size=1;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
#endif
  
  if (comm_rank == 0) 
    totalMemory = vec.size()*sizeof(T);
#ifndef SERIAL
  MPI_Bcast(&totalMemory, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  
  SHMsegment.truncate(totalMemory);
  SHMregion = boost::interprocess::mapped_region{SHMsegment, boost::interprocess::read_write};
  if (localrank == 0)
    memset(SHMregion.get_address(), 0., totalMemory);
  SHMvec = (T*)(SHMregion.get_address());
  
#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  
  if (comm_rank == 0) {
    for (size_t i=0; i<vec.size(); i++) 
      SHMvec[i] = vec[i];
  }
#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  if (localrank == 0) {
    long intdim  = totalMemory;
    long maxint  = 26843540; //mpi cannot transfer more than these number of doubles
    long maxIter = intdim/maxint;
#ifndef SERIAL
    MPI_Barrier(shmcomm);
    
    char* shrdMem = static_cast<char*>(SHMregion.get_address());
    for (int i=0; i<maxIter; i++) {
      MPI_Bcast  ( shrdMem+i*maxint,       maxint,                       MPI_CHAR, 0, shmcomm);
      MPI_Barrier(shmcomm);
    }
    
    MPI_Bcast  ( shrdMem+(maxIter)*maxint, totalMemory - maxIter*maxint, MPI_CHAR, 0, shmcomm);
#endif
  }
#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  boost::interprocess::shared_memory_object::remove(SHMname.c_str());

}


template <typename T>
void SHMVecFromVecs(T *vec, int vecsize, T* &SHMvec, std::string& SHMname, 
		    boost::interprocess::shared_memory_object& SHMsegment,
		    boost::interprocess::mapped_region& SHMregion) {

  boost::interprocess::shared_memory_object::remove(SHMname.c_str());
  size_t totalMemory = 0;
  int comm_rank=0, comm_size=1;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
#endif
  
  if (comm_rank == 0) 
    totalMemory = vecsize*sizeof(T);
#ifndef SERIAL
  MPI_Bcast(&totalMemory, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  
  SHMsegment.truncate(totalMemory);
  SHMregion = boost::interprocess::mapped_region{SHMsegment, boost::interprocess::read_write};
  if (localrank == 0)
    memset(SHMregion.get_address(), 0., totalMemory);
  SHMvec = (T*)(SHMregion.get_address());
  
#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  
  if (comm_rank == 0) {
    for (size_t i=0; i<vecsize; i++) 
      SHMvec[i] = vec[i];
  }
#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  if (localrank == 0) {
    long intdim  = totalMemory;
    long maxint  = 26843540; //mpi cannot transfer more than these number of doubles
    long maxIter = intdim/maxint;
#ifndef SERIAL
    MPI_Barrier(shmcomm);
    
    char* shrdMem = static_cast<char*>(SHMregion.get_address());
    for (int i=0; i<maxIter; i++) {
      MPI_Bcast  ( shrdMem+i*maxint,       maxint,                       MPI_CHAR, 0, shmcomm);
      MPI_Barrier(shmcomm);
    }
    
    MPI_Bcast  ( shrdMem+(maxIter)*maxint, totalMemory - maxIter*maxint, MPI_CHAR, 0, shmcomm);
#endif
  }
#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}


void readHDF5ToSHM(hid_t file, std::string datasetName, size_t size, double* &SHMAddress, std::string SHMName, 
		    boost::interprocess::shared_memory_object& SHMSegment,
		    boost::interprocess::mapped_region& SHMRegion); 


#endif
