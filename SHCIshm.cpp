#include "SHCIshm.h"
#include <boost/interprocess/managed_shared_memory.hpp>
#include "Determinants.h"
#include <vector>
#ifndef SERIAL
#include "mpi.h"
#endif

using namespace std;

void SHMVecFromMatrix(MatrixXx& vec, CItype* &SHMvec, std::string& SHMname,
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
    totalMemory = vec.rows()*sizeof(CItype);
#ifndef SERIAL
  MPI_Bcast(&totalMemory, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  
  
  SHMsegment.truncate(totalMemory);
  SHMregion = boost::interprocess::mapped_region{SHMsegment, boost::interprocess::read_write};
  memset(SHMregion.get_address(), 0., totalMemory);
  SHMvec = (CItype*)(SHMregion.get_address());
  
#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  
  if (comm_rank == 0) {
    for (size_t i=0; i<vec.rows(); i++) 
      SHMvec[i] = vec(i,0);
  }
#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  long intdim  = totalMemory;
  long maxint  = 26843540; //mpi cannot transfer more than these number of doubles
  long maxIter = intdim/maxint;
#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);
  
  char* shrdMem = static_cast<char*>(SHMregion.get_address());
  for (int i=0; i<maxIter; i++) {
    MPI_Bcast  ( shrdMem+i*maxint,       maxint,                       MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  
  MPI_Bcast  ( shrdMem+(maxIter)*maxint, totalMemory - maxIter*maxint, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}
