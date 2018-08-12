#include "time.h"
#include "Determinants.h"
#include "SHCIbasics.h"
#include "SHCIgetdeterminants.h"
#include "SHCIsampledeterminants.h"
#include "SHCIrdm.h"
#include "SHCISortMpiUtils.h"
#include "SHCImake4cHamiltonian.h"
#include "input.h"
#include "integral.h"
#include <vector>
#include "math.h"
#include "Hmult.h"
#include <tuple>
#include <map>
#include "Davidson.h"
#include "boost/format.hpp"
#include <fstream>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "communicate.h"
#include <boost/interprocess/managed_shared_memory.hpp>

using namespace std;
using namespace Eigen;
using namespace boost;
using namespace SHCISortMpiUtils;

void SHCImake4cHamiltonian::HamHelper4c::MakeSHMHelpers() {
  SHCImake4cHamiltonian::MakeSMHelpers();
}
void SHCImake4cHamiltonian::MakeSMHelpers(
  vector<vector<int>>& Nminus2ToDet,
  int* &Nminus2ToDetLen,
  vector<int*>& Nminus2ToDetSM
)
{
  int comm_rank = 0, comm_size = 1;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
#endif
  boost::interprocess::shared_memory_object::remove(shciHelper.c_str());

  size_t totalMemory = 0;
  size_t nNminus2 = Nminus2.size();
  vector<int> Nminus2ToDetTemp(nNminus2, 0);
  for (int i=0; i<nNminus2; i++) {
    totalMemory += 2*sizeof(int);
    totalMemory += 2*sizeof(int) * Nminus2ToDet[i].size();
    Nminus2ToDetTemp[i] = Nminus2ToDet[i].size();
  }
#ifndef SERIAL
#endif

  hHelpersSegment.truncate(totalMemory);
  regionHelpers = boost::interprocess::mapped_region{hHelpersSegment, boost::interprocess::read_write};
  memset(regionHelpers.get_address(), 0., totalMemory);

#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  Nminus2ToDetLen = static_cast<int*>(regionHelpers.get_address());

  for (int i = 0; i < nNminus2; i++) Nminus2ToDetLen[i] = Nminus2ToDetTemp[i];
  Nminus2ToDetSM.resize(nNminus2);

  int* begin = Nminus2ToDetLen + nNminus2;
  size_t counter = 0;
  for (int i = 0; i < nNminus2; i++) {
    Nminus2ToDetSM[i] = begin + counter; counter += Nminus2ToDetLen[i];
  }

  if (comm_rank == 0) {
    for (int i=0; i<nNminu2; i++) {
      for (int j=0; j<Nminus2ToDet[i].size(); j++) {
        Nminus2ToDetSM[i][j] = Nminus2ToDet[i][j];
      }
    }
  }

  long intdim = totalMemory;
  long maxint = 26843540;
  long maxIter = intdim/maxint;

#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);

  char* shrdMem = static_cast<char*>(regionHelpers.get_address());
  for (int i=0; i<maxIter; i++) {
    MPI_Bcast (shrdMem+i*maxint, maxint, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Bcast(shrdMem+(maxIter)*maxint, totalMemory - maxIter*maxint, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
#endif

}
void SHCImakeHamiltonian::SparseHam::makeFromHelper(
        HamHelper4c& helper, Determinant * SHMDets,
        int startIndex, int endIndex,
        int Norbs, oneInt& I1, towInt& I2, double& coreE, bool DoRDM) 
{
  SHCImake4cHamiltonian::MakeHfromSMHelpers(Nminus2ToDet,
  Dets, startIndex, endIndex, diskio, *this, I1, I2, coreE, DoRDM);
}

void SHCImake4cHamiltonian::HamHelpers::PopulateHelpers(
        Determinant* SHMDets,
        int DetsSize, int startIndex)
{
  SHCImake4cHamiltonian::PopulateHelperLists(
    Nminus2, Nminus2ToDet,
    SHMDets, DetsSize, startIndex);
}

void PopulateHelperLists(map<Determinants, int>& Nminus2s,
 map<Determinants, int> Nminus1s, 
 vector<vector<int>> Nminus2ToDets, vector<vector<int>> Nminus1to2, Determinant *Dets, int DetsSize, int StartIndex) {
#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  if (commrank == 0) {
    for (int i = StartIndex; i < DetsSize; i++) {
      for (int j = 0; j<nocc; j++) {
        for (int k = 0; k < j; k++) {
          Determinant Nminus2 = removeTewElectron(Dets[i], j, k);
          updateNminus(Nminus2, Nminus2s, Nminus2ToDets, Dets, DetsSize, StartIndex);
        }
      }
    }
  }
}

void updateNminus(Determinant & Nminus2, std::map<Determinant, int>&Nminus2s, vector<vector<int>> Nminus2ToDets, Determinant* Dets, int DetsSize, int StartIndex) {
  auto iter = Nminus2s.find(Nminus2);
  if (iter == Nminus2s.end()) {
    auto newElement = Nminus2s.insert(std::pair<Determinant, int>(Nminus2, Nminus2ToDets.size()));
    iter = newElement.first;
    Nminus2ToDets.resize(iter->second+1);

    int norbs = 64*DetLen;
  }
}

void SHCImake4cHamiltonian::MakeSMHelpers(
  vector<vector<int>>& Nminus2ToDet,
)
void SHCImake4cHamiltonian::MakeHfromSMHelpers(
  vector<int* > &Nminus2ToDet,
  Determinant* Dets, int StartIndex, int EndIndex,
  bool diskio, SparseHam& sparseHam,
  int Norbs, oneInt& I1, twoInt& I2, double& coreE,
  bool DoRDM) {
  int proc = 0, nprocs = 1;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#endif
  size_t norbs = Norbs;
  iont offSet = 0;
  std::vector<std::vecto<int>> & connections = sparseHam.connections;
  std::vector<std::vector<CItype>> & Helements = sparseHam.Helements;
  std::vector<std::vector<size_t>> & orbDifference = sparseHam.orbDifference;

  for (size_t k = StartIndex; k < EndIndex; k++) {
    if (k%(nprocs) != proc || k < max(StartIndex, offSet)) continue;
    connections.push_back(vector<int>(1,k));
    CItype hij = Dets[k].Energy(I1, I2, coreE);
    size_t orbDiff;
    if (DoRDM) orbDifference.push_back(vector<size_t>(1,0));
  }

  for (int i = 0; i < Nminus2ToDet.size(); i++) {
    for (auto iter=Nminus2ToDet[i].begin(), iter !=Nminus2ToDet[i].end(), iter++) {
      for (auto iter2=++iter, iter!=Nminus2ToDet[i].end(), iter2++) {
        CItype hij = Hij(Dets[*iter2], Dets[*iter], I1, I2, coreE, orbDiff);
        if std::abs(hij) > 1.e-10 {
          connections[*iter].push_back(*iter2);
          Helements[*iter].push_back(hij);
          if (DoRDM) orbDifference[*iter].push_back(orbDiff);
        }
      }
    }
  }
}

void SHCImake4cHamiltonian::SparseHam::setNbatches(int DetSize) {
  int proc = 0, nprocs = 1;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#endif
  Nbatches = diskio ? (DetSize)/BatchSize/nprocs : 1;
  if ((DetSize) > Nbatches*BatchSize*nprocs && diskio) Nbatches += 1;
}

void SHCImake4cHamiltonian::SparseHam::writeBatch(int batch) {
  char file [5000];
  sprintf (file, "%s/%d-4chamiltonian-batch%d.bkp", prefix.c_str(), commrank, batch);
  std::ofstream ofs(file, std::ios::binary);
  boost::archive::binary_oarchive save(ofs);
  save << connections << Helements << orbDifference;
}

void SHCImakeHamiltonian::SparseHam::readBatch (int batch) {
  char file [5000];
  sprintf (file, "%s/%d-4chamiltonian-batch%d.bkp" , prefix.c_str(), commrank, batch);
  std::ifstream ifs(file, std::ios::binary);
  boost::archive::binary_iarchive load(ifs);
  load >> connections >> Helements >> orbDifference;
}
