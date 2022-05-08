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
  
  // used for afqmc
  cholSHMName = "chol" + to_string(static_cast<long long>(time(NULL) % 1000000));
  cholSegment = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, cholSHMName.c_str(), boost::interprocess::read_write);
  cholSHMNameUp = "cholUp" + to_string(static_cast<long long>(time(NULL) % 1000000));
  cholSegmentUp = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, cholSHMNameUp.c_str(), boost::interprocess::read_write);
  cholSHMNameDn = "cholDn" + to_string(static_cast<long long>(time(NULL) % 1000000));
  cholSegmentDn = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, cholSHMNameDn.c_str(), boost::interprocess::read_write);
  floatCholSHMName = "floatChol" + to_string(static_cast<long long>(time(NULL) % 1000000));
  floatCholSegment = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, floatCholSHMName.c_str(), boost::interprocess::read_write);
  rotCholSHMName = "rotChol" + to_string(static_cast<long long>(time(NULL) % 1000000));
  rotCholSegment = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, rotCholSHMName.c_str(), boost::interprocess::read_write);

}

void removeSHM(){
  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shmcas.c_str());
  boost::interprocess::shared_memory_object::remove(cholSHMName.c_str());
  boost::interprocess::shared_memory_object::remove(cholSHMNameUp.c_str());
  boost::interprocess::shared_memory_object::remove(cholSHMNameDn.c_str());
  boost::interprocess::shared_memory_object::remove(floatCholSHMName.c_str());
  boost::interprocess::shared_memory_object::remove(rotCholSHMName.c_str());
}


void readHDF5ToSHM(hid_t file, std::string datasetName, size_t size, double* &SHMAddress, std::string SHMName, 
		    boost::interprocess::shared_memory_object& SHMSegment,
		    boost::interprocess::mapped_region& SHMRegion) 
{
  boost::interprocess::shared_memory_object::remove(SHMName.c_str());
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
 
  size_t totalMemory = size * sizeof(double);
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  SHMSegment.truncate(totalMemory);
  SHMRegion = boost::interprocess::mapped_region{SHMSegment, boost::interprocess::read_write};
  if (comm_rank == 0) memset(SHMRegion.get_address(), 0., totalMemory);
  SHMAddress = (double*)(SHMRegion.get_address());
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  if (comm_rank == 0) { // read dataset into shared memory
    double* data = new double[size];
    for (size_t i = 0; i < size; i++) data[i] = 0.;
    
    hid_t dataset = (-1);
    herr_t status;
    H5E_BEGIN_TRY {
      dataset = H5Dopen(file, datasetName.c_str(), H5P_DEFAULT);
      status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    } H5E_END_TRY
    if (dataset < 0) {
      if (commrank == 0) std::cout << datasetName << " dataset could not be read." << std::endl;
      exit(1);
    }
    
    for (size_t i = 0; i < size; i++) SHMAddress[i] = data[i];
    delete [] data;
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  if (localrank == 0) { // send to all nodes
    long intdim = totalMemory;
    long maxint = 26843540; // mpi cannot transfer more than these number of doubles; probably should not be hand-coded
    long maxIter = intdim / maxint;
    MPI_Barrier(shmcomm);
    
    char* shrdMem = static_cast<char*>(SHMRegion.get_address());
    for (int i = 0; i < maxIter; i++) {
      MPI_Bcast(shrdMem + i*maxint, maxint, MPI_CHAR, 0, shmcomm);
      MPI_Barrier(shmcomm);
    }
    MPI_Bcast(shrdMem + maxIter * maxint, totalMemory - maxIter*maxint, MPI_CHAR, 0, shmcomm);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
}
