#include "SHCIshm.h"
#include <boost/interprocess/managed_shared_memory.hpp>
#include <vector>
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#include "mpi.h"
#endif

using namespace std;

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
  int2Segment = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, shciint2.c_str(), boost::interprocess::read_write);
  shciint2shm = "SHCIint2shm" + to_string(static_cast<long long>(time(NULL) % 1000000));
  int2SHMSegment = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, shciint2shm.c_str(), boost::interprocess::read_write);
  shciint2shmcas = "SHCIint2shmcas" + to_string(static_cast<long long>(time(NULL) % 1000000));
  int2SHMCASSegment = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, shciint2shmcas.c_str(), boost::interprocess::read_write);

}

void removeSHM(){
  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
}
