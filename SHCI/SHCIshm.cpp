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
#include "SHCIshm.h"
#include <boost/interprocess/managed_shared_memory.hpp>
#include "Determinants.h"
#include <vector>
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#include "mpi.h"
#endif

using namespace std;

boost::interprocess::shared_memory_object int2Segment;
boost::interprocess::mapped_region regionInt2;
string shciint2;
boost::interprocess::shared_memory_object int2SHMSegment;
boost::interprocess::mapped_region regionInt2SHM;
string shciint2shm;
boost::interprocess::shared_memory_object hHelpersSegment;
boost::interprocess::mapped_region regionHelpers;
string shciHelper;
boost::interprocess::shared_memory_object DetsCISegment;
boost::interprocess::mapped_region regionDetsCI;
std::string shciDetsCI;
boost::interprocess::shared_memory_object DetsCISegment2;
boost::interprocess::mapped_region regionDetsCI2;
std::string shciDetsCI2;
boost::interprocess::shared_memory_object SortedDetsSegment;
boost::interprocess::mapped_region regionSortedDets;
std::string shciSortedDets;
boost::interprocess::shared_memory_object DavidsonSegment;
boost::interprocess::mapped_region regionDavidson;
std::string shciDavidson;
boost::interprocess::shared_memory_object cMaxSegment;
boost::interprocess::mapped_region regioncMax;
std::string shcicMax;
#ifndef SERIAL
MPI_Comm shmcomm, localcomm;
#endif
int commrank, shmrank, localrank;
int commsize, shmsize, localsize;

void initSHM() {
#ifndef SERIAL
  boost::mpi::communicator world;

  MPI_Comm_rank(MPI_COMM_WORLD, &commrank);
  MPI_Comm_size(MPI_COMM_WORLD, &commsize);

  if (true){
    char hostname[MPI_MAX_PROCESSOR_NAME]; int resultsize;
    MPI_Get_processor_name(hostname, &resultsize);
    string HName(hostname);
    vector<string> AllHostNames(commsize);
    AllHostNames[commrank] = HName;
    // all nodes will not have node names for all procs
    for (int i=0; i<commsize; i++)
      boost::mpi::broadcast(world, AllHostNames[i], i);
    
    std::sort( AllHostNames.begin(), AllHostNames.end());
    AllHostNames.erase(std::unique(AllHostNames.begin(), AllHostNames.end()), AllHostNames.end());
    
    int color = -1;
    for (int i=0; i<AllHostNames.size(); i++)
      if (HName == AllHostNames[i])
	color = i;
    
    MPI_Comm_split(MPI_COMM_WORLD, color, commrank, &localcomm);

    MPI_Comm_rank(localcomm, &localrank);
    MPI_Comm_size(localcomm, &localsize);
  }
  else {
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
			MPI_INFO_NULL, &localcomm);

    MPI_Comm_rank(localcomm, &localrank);
    MPI_Comm_size(localcomm, &localsize);
  }



  int localrankcopy = localrank;
  MPI_Comm_split(MPI_COMM_WORLD, localrankcopy, commrank, &shmcomm);

  MPI_Comm_rank(shmcomm, &shmrank);
  MPI_Comm_size(shmcomm, &shmsize);


  //MPI_Comm_free(&localcomm);
#else
  commrank=0;shmrank=0;localrank=0;
  commsize=1;shmsize=1;localsize=1;
#endif

  //set up shared memory files to store the integrals
  shciint2 = "SHCIint2" + to_string(static_cast<long long>(time(NULL) % 1000000));
  shciint2shm = "SHCIint2shm" + to_string(static_cast<long long>(time(NULL) % 1000000));
  shciHelper = "SHCIhelpershm" + to_string(static_cast<long long>(time(NULL) % 1000000));
  shciDetsCI = "SHCIDetsCIshm" + to_string(static_cast<long long>(time(NULL) % 1000000));
  shciDetsCI2 = "SHCIDetsCIshm2" + to_string(static_cast<long long>(time(NULL) % 1000000));
  shciSortedDets = "SHCISortedDetsshm" + to_string(static_cast<long long>(time(NULL) % 1000000));
  shciDavidson = "SHCIDavidsonshm" + to_string(static_cast<long long>(time(NULL) % 1000000));
  shcicMax = "SHCIcMaxshm" + to_string(static_cast<long long>(time(NULL) % 1000000));
  int2Segment = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, shciint2.c_str(), boost::interprocess::read_write);
  int2SHMSegment = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, shciint2shm.c_str(), boost::interprocess::read_write);
  hHelpersSegment = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, shciHelper.c_str(), boost::interprocess::read_write);
  DetsCISegment = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, shciDetsCI.c_str(), boost::interprocess::read_write);
  DetsCISegment2 = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, shciDetsCI2.c_str(), boost::interprocess::read_write);
  SortedDetsSegment = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, shciSortedDets.c_str(), boost::interprocess::read_write);
  DavidsonSegment = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, shciDavidson.c_str(), boost::interprocess::read_write);
  cMaxSegment = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, shcicMax.c_str(), boost::interprocess::read_write);

}

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
  if (localrank == 0)
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
