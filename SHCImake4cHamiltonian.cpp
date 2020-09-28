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
  SHCImake4cHamiltonian::MakeSMHelpers(Nminus1, Nminus1ToDet, Nminus1ToDetLen, Nminus1ToDetSM,
    Nminus2, Nminus2ToDet, Nminus2ToDetLen, Nminus2ToDetSM);
}
void SHCImake4cHamiltonian::MakeSMHelpers(
  map& Nminus1,
  vector<vector<int>>& Nminus1ToDet,
  int* &Nminus1ToDetLen,
  vector<int*>& Nminus1ToDetSM,
  map& Nminus2,
  vector<vector<int>>& Nminus2ToDet,
  int* &Nminus2ToDetLen,
  vector<int*>& Nminus2ToDetSM)
{
  int comm_rank = 0, comm_size = 1;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
#endif
  boost::interprocess::shared_memory_object::remove(shciHelper.c_str());

  size_t totalMemory = 0;
  size_t nNminus2 = Nminus2.size();
  size_t nNminus1 = Nminus1.size();
  vector<int> Nminus2ToDetTemp(nNminus2, 0);
  vector<int> Nminus1ToDetTemp(nNminus1, 0);
  for (int i=0; i<nNminus2; i++) {
    totalMemory += 2*sizeof(int);
    totalMemory += 2*sizeof(int) * Nminus2ToDet[i].size();
    Nminus2ToDetTemp[i] = Nminus2ToDet[i].size();
  }
  for (int i=0; i<nNminus1; i++) {
    totalMemory += 2*sizeof(int);
    totalMemory += 2*sizeof(int) * Nminus1ToDet[i].size();
    Nminus1ToDetTemp[i] = Nminus1ToDet[i].size();
  }
#ifndef SERIAL
  MPI_Bcast(&totalMemory, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nNminus1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nNminus2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (comm_rank != 0) {
    Nminus1ToDetTemp.resize(nNminus1);
    Nminus2ToDetTemp.resize(nNminus2);
  }
  MPI_Bcast(&Nminus1ToDetTemp[0], nNminus1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Nminus2ToDetTemp[0], nNminus2, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  hHelpersSegment.truncate(totalMemory);
  regionHelpers = boost::interprocess::mapped_region{hHelpersSegment, boost::interprocess::read_write};
  memset(regionHelpers.get_address(), 0., totalMemory);

#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  Nminus2ToDetLen = static_cast<int*>(regionHelpers.get_address());
  Nminus1ToDetLen = Nminus2ToDetLen + nNminus2;
  for (int i = 0; i < nNminus2; i++) Nminus2ToDetLen[i] = Nminus2ToDetTemp[i];
  for (int i = 0; i < nNminus1; i++) Nminus1ToDetLen[i] = Nminus1ToDetTemp[i];
  Nminus2ToDetSM.resize(nNminus2);
  Nminus1ToDetSM.resize(nNminus1);

  int* begin = Nminus2ToDetLen + nNminus2 + nNminus1;
  size_t counter = 0;
  for (int i = 0; i < nNminus2; i++) {
    Nminus2ToDetSM[i] = begin + counter; counter += Nminus2ToDetLen[i];
  }
  for (int i = 0; i < nNminus1; i++) {
    Nminus1ToDetSM[i] = begin + counter; counter += Nminus1ToDetLen[i];
  }

  if (comm_rank == 0) {
    for (int i=0; i<nNminus2; i++) {
      for (int j=0; j<Nminus2ToDet[i].size(); j++) {
        Nminus2ToDetSM[i][j] = Nminus2ToDet[i][j];
      }
    }
    for (int i=0; i<nNminus1; i++) {
      for (int j=0; j<Nminus1ToDet[i].size(); j++) {
        Nminus1ToDetSM[i][j] = Nminus1ToDet[i][j];
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
void SHCImake4cHamiltonian::SparseHam::makeFromHelper(
        HamHelper4c& helper, Determinant * SHMDets,
        int startIndex, int endIndex,
        int Norbs, oneInt& I1, twoInt& I2, schedule& schd, double& coreE, bool DoRDM) 
{
  SHCImake4cHamiltonian::MakeHfromSMHelpers(helper.Nminus1ToDetSM, helper.Nminus1ToDetLen,
  helper.Nminus2ToDetSM, helper.Nminus2ToDetLen,
  SHMDets, startIndex, endIndex, diskio, *this, Norbs, I1, I2, schd, coreE, DoRDM);
}

void SHCImake4cHamiltonian::HamHelper4c::PopulateHelpers(
        Determinant* SHMDets,
        int DetsSize, int startIndex)
{
  SHCImake4cHamiltonian::PopulateHelperLists(
    Nminus1, Nminus1ToDet,
    Nminus2, Nminus2ToDet,
    SHMDets, DetsSize, startIndex);
}
void updateNminus(Determinant & Nminus2, SHCImake4cHamiltonian::map& Nminus2s, 
vector<vector<int>>& Nminus2ToDets, Determinant* Dets, int DetsSize, int StartIndex, int DetIndex) {
  auto iter = Nminus2s.find(Nminus2);
  if (iter == Nminus2s.end()) {
    //auto newElement = Nminus2s.insert(std::pair<Determinant, int>(Nminus2, Nminus2ToDets.size()));
    Nminus2s[Nminus2] = Nminus2ToDets.size();
    Nminus2ToDets.push_back(vector<int>(1, DetIndex));
    int norbs = 64*DetLen;
  }
  else {
    Nminus2ToDets[iter->second].push_back(DetIndex);
  }
}

void SHCImake4cHamiltonian::PopulateHelperLists(
  map& Nminus1s, vector<vector<int>> & Nminus1ToDet, map& Nminus2s, vector<vector<int>> & Nminus2ToDet, 
  Determinant *Dets, int DetsSize, int StartIndex) {
#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  int norbs = Dets[0].norbs;
  if (commrank == 0) {
    for (int i = StartIndex; i < DetsSize; i++) {
      for (int j = 0; j<norbs; j++) {
        if (!Dets[i].getocc(j)) continue;
        Determinant Nminus1 = Dets[i];
        Nminus1.setocc(j, false);
        updateNminus(Nminus1, Nminus1s, Nminus1ToDet, Dets, DetsSize, StartIndex, i);
        for (int k = 0; k < j; k++) {
          if (!Dets[i].getocc(k)) continue;
          Determinant Nminus2 = Dets[i];
          Nminus2.setocc(j, false), Nminus2.setocc(k, false);
          updateNminus(Nminus2, Nminus2s, Nminus2ToDet, Dets, DetsSize, StartIndex, i);
        }
      }
    }
  }
}



void SHCImake4cHamiltonian::MakeHfromSMHelpers(
  vector<int* > &Nminus1ToDetSM, int* Nminus1ToDetLen,
  vector<int* > &Nminus2ToDetSM, int* Nminus2ToDetLen,
  Determinant* Dets, int StartIndex, int EndIndex,
  bool diskio, SparseHam& sparseHam,
  int Norbs, oneInt& I1, twoInt& I2, schedule& schd, double& coreE,
  bool DoRDM) {
  int proc = 0, nprocs = 1;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#endif
  size_t norbs = Norbs;
  size_t orbDiff;
  int offSet = 0;
  std::vector<std::vector<int>> & connections = sparseHam.connections;
  std::vector<std::vector<CItype>> & Helements = sparseHam.Helements;
  std::vector<std::vector<size_t>> & orbDifference = sparseHam.orbDifference;

  for (size_t k = StartIndex; k < EndIndex; k++) {
    if (k%(nprocs) != proc || k < max(StartIndex, offSet)) continue;
    connections.push_back(vector<int>(1,k));
    CItype hij = Dets[k].Energy(I1, I2, coreE);
    Helements.push_back(vector<CItype>(1,hij));
    if (DoRDM) orbDifference.push_back(vector<size_t>(1,0));
  }
  for (int i = 0; i < Nminus1ToDetSM.size(); i++) {
    for (int j = 0; j<Nminus1ToDetLen[i]; j++) {
      for (int k = j+1; k<Nminus1ToDetLen[i]; k++) {
        int DetI = Nminus1ToDetSM[i][j];
        int DetJ = Nminus1ToDetSM[i][k];
        if (DetI < StartIndex && DetJ < StartIndex)     
          continue;
        if (DetI % nprocs != proc || DetI < 0) continue;
        CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
        if (std::abs(hij) > schd.thresh_hij) {
          //pout << Dets[Nminus1ToDetSM[i][j]].ExcitationDistance(Dets[Nminus1ToDetSM[i][k]]);
        //pout << "(" << Nminus1ToDetSM[i][j] << " " << Nminus1ToDetSM[i][k] << ")" << " " << hij.real() << " " << hij.imag() << endl;
          //cout << "Hij : " << abs(hij) << " greater than threshold " << schd.thresh_hij << endl;
          connections[DetI/nprocs].push_back(DetJ);
          Helements[DetI/nprocs].push_back(hij);
          if (DoRDM) orbDifference[DetI/nprocs].push_back(orbDiff);
        }
      }
    }
  }
  for (int i = 0; i < Nminus2ToDetSM.size(); i++) {
    //if (i % nprocs != proc) continue;
    for (int j = 0; j<Nminus2ToDetLen[i]; j++) {
      for (int k=j+1; k<Nminus2ToDetLen[i]; k++) {
        int DetI = Nminus2ToDetSM[i][j];
        int DetJ = Nminus2ToDetSM[i][k];
        if (DetI < StartIndex && DetJ < StartIndex)     
          continue;
        if (DetI % nprocs != proc || DetI < 0) continue;
        CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
        //pout  << hij << endl;
        if (std::abs(hij) > schd.thresh_hij) {
          if (Dets[Nminus2ToDetSM[i][j]].ExcitationDistance(Dets[Nminus2ToDetSM[i][k]]) != 2) continue;
          connections[DetI/nprocs].push_back(Nminus2ToDetSM[i][k]);
          Helements[DetI/nprocs].push_back(hij);
          if (DoRDM) orbDifference[DetI/nprocs].push_back(orbDiff);
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

void SHCImake4cHamiltonian::SparseHam::readBatch (int batch) {
  char file [5000];
  sprintf (file, "%s/%d-4chamiltonian-batch%d.bkp" , prefix.c_str(), commrank, batch);
  std::ifstream ifs(file, std::ios::binary);
  boost::archive::binary_iarchive load(ifs);
  load >> connections >> Helements >> orbDifference;
}
