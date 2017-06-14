/*
Developed by Sandeep Sharma with contributions from James E. Smith and Adam A. Homes, 2017
Copyright (c) 2017, Sandeep Sharma

This file is part of DICE.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "omp.h"
#include "Determinants.h"
#include "SHCIbasics.h"
#include "SHCIgetdeterminants.h"
#include "SHCIsampledeterminants.h"
#include "SHCIrdm.h"
#include "SHCISortMpiUtils.h"
#include "SHCImakeHamiltonian.h"
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


void SHCImakeHamiltonian::regenerateH(std::vector<Determinant>& Dets,
				 std::vector<std::vector<int> >&connections,
				 std::vector<std::vector<CItype> >& Helements,
				 oneInt& I1,
				 twoInt& I2,
				 double& coreE
			    ) {

#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  int size = mpigetsize(), rank = mpigetrank();

#pragma omp parallel
  {
    for (int i=0; i<connections.size(); i++) {
      if ((i/omp_get_num_threads())%size != rank) continue;
      Helements[i][0] = Dets[i].Energy(I1, I2, coreE);
      for (int j=1; j<connections[i].size(); j++) {
	size_t orbDiff;
	CItype hij = Hij(Dets[i], Dets[connections[i][j]], I1, I2, coreE, orbDiff);
	if (Determinant::Trev != 0) updateHijForTReversal(hij, Dets[i], Dets[connections[i][j]], I1, I2, coreE);
	Helements[i][j] = hij;
      }
    }
  }
}

void SHCImakeHamiltonian::MakeHfromHelpers(std::map<HalfDet, std::vector<int> >& BetaN,
				 std::map<HalfDet, std::vector<int> >& AlphaNm1,
				 std::vector<Determinant>& Dets,
				 int StartIndex,
				 std::vector<std::vector<int> >&connections,
				 std::vector<std::vector<CItype> >& Helements,
				 int Norbs,
				 oneInt& I1,
				 twoInt& I2,
				 double& coreE,
				 std::vector<std::vector<size_t> >& orbDifference,
				 bool DoRDM) {

#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  int nprocs= mpigetsize(), proc = mpigetrank();

  size_t norbs = Norbs;

  for (size_t k=StartIndex; k<connections.size(); k++) {
    if (k%(nprocs) != proc) continue;
    connections[k].push_back(k);
    CItype hij = Dets[k].Energy(I1, I2, coreE);
    if (Determinant::Trev != 0) updateHijForTReversal(hij, Dets[k], Dets[k], I1, I2, coreE);
    Helements[k].push_back(hij);
    if (DoRDM) orbDifference[k].push_back(0);
  }
  

  std::map<HalfDet, std::vector<int> >::iterator ita = BetaN.begin();
  int index = 0;
  for (; ita!=BetaN.end(); ita++) {
    std::vector<int>& detIndex = ita->second;
    int localStart = detIndex.size();
    for (int j=0; j<detIndex.size(); j++)
      if (detIndex[j] >= StartIndex) {
	localStart = j; break;
      }

    for (int k=localStart; k<detIndex.size(); k++) {
      if (detIndex[k]%(nprocs) != proc) continue;

      for(int j=0; j<k; j++) {
	size_t J = detIndex[j];size_t K = detIndex[k];
	if (Dets[J].connected(Dets[K]) ||  (Determinant::Trev!=0 && Dets[J].connectedToFlipAlphaBeta(Dets[K]))) {
	  
	  size_t orbDiff;
	  CItype hij = Hij(Dets[J], Dets[K], I1, I2, coreE, orbDiff);
	  if (Determinant::Trev != 0) updateHijForTReversal(hij, Dets[J], Dets[K], I1, I2, coreE);
	  
	  if (abs(hij) <1.e-10) continue;
	  Helements[K].push_back(hij);
	  connections[K].push_back(J);
	  
	  if (DoRDM)
	    orbDifference[K].push_back(orbDiff);
	}
      }
    }
  }

  ita = AlphaNm1.begin();
  index = 0;
  for (; ita!=AlphaNm1.end(); ita++) {
    std::vector<int>& detIndex = ita->second;
    int localStart = detIndex.size();
    for (int j=0; j<detIndex.size(); j++)
      if (detIndex[j] >= StartIndex) {
	localStart = j; break;
      }

    for (int k=localStart; k<detIndex.size(); k++) {
      if (detIndex[k]%(nprocs) != proc) continue;

      for(int j=0; j<k; j++) {
	size_t J = detIndex[j];size_t K = detIndex[k];
	if (Dets[J].connected(Dets[K]) ||  (Determinant::Trev!=0 && Dets[J].connectedToFlipAlphaBeta(Dets[K]))) {
	  if (find(connections[K].begin(), connections[K].end(), J) == connections[K].end()){
	    size_t orbDiff;
	    CItype hij = Hij(Dets[J], Dets[K], I1, I2, coreE, orbDiff);
	    if (Determinant::Trev != 0) updateHijForTReversal(hij, Dets[J], Dets[K], I1, I2, coreE);
	    
	    if (abs(hij) <1.e-10) continue;
	    connections[K].push_back(J);
	    Helements[K].push_back(hij);
	    
	    if (DoRDM)
	      orbDifference[K].push_back(orbDiff);
	  }
	}
      }
    }
    
  }
  
}


void SHCImakeHamiltonian::PopulateHelperLists(std::map<HalfDet, std::vector<int> >& BetaN,
				    std::map<HalfDet, std::vector<int> >& AlphaNm1,
				    std::vector<Determinant>& Dets,
				    int StartIndex)
{
  for (int i=StartIndex; i<Dets.size(); i++) {
    HalfDet da = Dets[i].getAlpha(), db = Dets[i].getBeta();
    
    
    BetaN[db].push_back(i);

    int norbs = 64*DetLen;
    std::vector<int> closeda(norbs/2);//, closedb(norbs);
    int ncloseda = da.getClosed(closeda);
    //int nclosedb = db.getClosed(closedb);
    for (int j=0; j<ncloseda; j++) {
      da.setocc(closeda[j], false);
      AlphaNm1[da].push_back(i);
      da.setocc(closeda[j], true);
    }

    //When Treversal symmetry is used
    if (Determinant::Trev!=0 && Dets[i].hasUnpairedElectrons()) {
      BetaN[da].push_back(i);

      std::vector<int> closedb(norbs/2);//, closedb(norbs);
      int nclosedb = db.getClosed(closedb);
      for (int j=0; j<nclosedb; j++) {
	db.setocc(closedb[j], false);
	AlphaNm1[db].push_back(i);
	db.setocc(closedb[j], true);
      }
    }
  }
}

void SHCImakeHamiltonian::MakeHfromHelpers(int* &BetaVecLen, vector<int*> &BetaVec,
					   int* &AlphaVecLen, vector<int*> &AlphaVec,
					   Determinant *Dets,
					   int StartIndex,
					   std::vector<std::vector<int> >&connections,
					   std::vector<std::vector<CItype> >& Helements,
					   int Norbs,
					   oneInt& I1,
					   twoInt& I2,
					   double& coreE,
					   std::vector<std::vector<size_t> >& orbDifference,
					   bool DoRDM) {
  
  int proc=0, nprocs=1;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#endif

  size_t norbs = Norbs;

  for (size_t k=StartIndex; k<connections.size(); k++) {
    if (k%(nprocs) != proc) continue;
    connections[k].push_back(k);
    CItype hij = Dets[k].Energy(I1, I2, coreE);
    if (Determinant::Trev != 0) updateHijForTReversal(hij, Dets[k], Dets[k], I1, I2, coreE);
    Helements[k].push_back(hij);
    if (DoRDM) orbDifference[k].push_back(0);
  }


  int index = 0;
  for (int i=0;i<BetaVec.size(); i++) {
    int* detIndex = BetaVec[i];
    int localStart = BetaVecLen[i];
    for (int j=0; j<BetaVecLen[i]; j++)
      if (detIndex[j] >= StartIndex) {
	localStart = j; break;
      }

    for (int k=localStart; k<BetaVecLen[i]; k++) {
      
      if (detIndex[k]%(nprocs) != proc) continue;
      
      for(int j=0; j<k; j++) {
	size_t J = detIndex[j];size_t K = detIndex[k];
	size_t orbDiff;
	CItype hij = Hij(Dets[J], Dets[K], I1, I2, coreE, orbDiff);
	if (Determinant::Trev != 0) updateHijForTReversal(hij, Dets[J], Dets[K], I1, I2, coreE);
	
	if (abs(hij) <1.e-10) continue;
	Helements[K].push_back(hij);
	connections[K].push_back(J);
	
	if (DoRDM)
	  orbDifference[K].push_back(orbDiff);
      }
    }
    
    index++;
  }


  index = 0;
  for (int i=0;i<AlphaVec.size(); i++) {
    int* detIndex = AlphaVec[i];
    int localStart = AlphaVecLen[i];
    for (int j=0; j<AlphaVecLen[i]; j++)
      if (detIndex[j] >= StartIndex) {
	localStart = j; break;
      }
    
    for (int k=localStart; k<AlphaVecLen[i]; k++) {
      if (detIndex[k]%(nprocs) != proc) continue;

      for(int j=0; j<k; j++) {
	size_t J = detIndex[j];size_t K = detIndex[k];
	if (Dets[J].connected(Dets[K]) ||  (Determinant::Trev!=0 && Dets[J].connectedToFlipAlphaBeta(Dets[K]))) {
	  if (find(connections[K].begin(), connections[K].end(), J) == connections[K].end()){
	    size_t orbDiff;
	    CItype hij = Hij(Dets[J], Dets[K], I1, I2, coreE, orbDiff);
	    if (Determinant::Trev != 0) updateHijForTReversal(hij, Dets[J], Dets[K], I1, I2, coreE);
	    
	    if (abs(hij) <1.e-10) continue;
	    connections[K].push_back(J);
	    Helements[K].push_back(hij);
	    
	    if (DoRDM)
	      orbDifference[K].push_back(orbDiff);
	  }
	}
      }
    }
    index++;
  }
  
}


//check if the cost can be reduced by half by only storiong connection > than the index
void SHCImakeHamiltonian::PopulateHelperLists2(std::map<HalfDet, int >& BetaN,
					       std::map<HalfDet, int >& AlphaN,
					       std::map<HalfDet, vector<int> >& BetaNm1,
					       std::map<HalfDet, vector<int> >& AlphaNm1,
					       vector<vector<int> >& AlphaMajorToBeta,
					       vector<vector<int> >& AlphaMajorToDet,
					       vector<vector<int> >& BetaMajorToAlpha,
					       vector<vector<int> >& BetaMajorToDet,
					       vector< vector<int> >& SinglesFromAlpha,
					       vector< vector<int> >& SinglesFromBeta,
					       Determinant *Dets, int DetsSize,
					       int StartIndex)
{
  //ith vector of AlphaMajor contains all Determinants that have
  //ith Alpha string, and the 2j and 2j+1 elements of this ith vector
  //are the indices of the beta string and the determinant respectively
#ifndef SERIAL
  boost::mpi::communicator world;
#endif

  if (mpigetrank() == 0) {
    for (int i=StartIndex; i<DetsSize; i++) {
      HalfDet da = Dets[i].getAlpha(), db = Dets[i].getBeta();

      auto itb = BetaN.find(db);
      if (itb == BetaN.end()) {
	auto ret = BetaN.insert( std::pair<HalfDet, int>(db, BetaMajorToDet.size()));
	itb = ret.first;
	BetaMajorToAlpha.resize(itb->second+1);
	BetaMajorToDet.resize(itb->second+1);
	SinglesFromBeta.resize(itb->second+1);
	
	int norbs = 64*DetLen;
	std::vector<int> closedb(norbs/2);//, closedb(norbs);
	int nclosedb = db.getClosed(closedb);
	//int nclosedb = db.getClosed(closedb);
	for (int j=0; j<nclosedb; j++) {
	  db.setocc(closedb[j], false);
	  std::vector<int>& dbvec = BetaNm1[db];
	  dbvec.push_back(itb->second);
	  db.setocc(closedb[j], true);
	  
	  for (int k=0; k<dbvec.size()-1; k++) { //the last one in the list is db
	    SinglesFromBeta[itb->second].push_back(dbvec[k]);
	    SinglesFromBeta[dbvec[k]].push_back(itb->second);
	  }
	}
	
      }
      
      auto ita = AlphaN.find(da);
      if (ita == AlphaN.end()) {
	auto ret = AlphaN.insert( std::pair<HalfDet, int>(da, AlphaMajorToDet.size()));
	ita = ret.first;
	AlphaMajorToBeta.resize(ita->second+1);
	AlphaMajorToDet.resize(ita->second+1);
	SinglesFromAlpha.resize(ita->second+1);
	
	int norbs = 64*DetLen;
	std::vector<int> closeda(norbs/2);//, closedb(norbs);
	int ncloseda = da.getClosed(closeda);
	
	for (int j=0; j<ncloseda; j++) {
	  da.setocc(closeda[j], false);
	  std::vector<int>& davec = AlphaNm1[da];
	  davec.push_back(ita->second);
	  da.setocc(closeda[j], true);
	  
	  for (int k=0; k<davec.size()-1; k++) { //the last one in the list is da
	    SinglesFromAlpha[ita->second].push_back(davec[k]);
	    SinglesFromAlpha[davec[k]].push_back(ita->second);
	  }
	}
      }
      
      AlphaMajorToBeta[ita->second].push_back(itb->second);
      AlphaMajorToDet[ita->second].push_back(i);
      
      
      BetaMajorToAlpha[itb->second].push_back(ita->second);
      BetaMajorToDet[itb->second].push_back(i);
      
      
    }
    size_t max=0, min=0, avg=0;
    for (int i=0; i<AlphaMajorToBeta.size(); i++)
      {
	vector<int> betacopy = AlphaMajorToBeta[i];
	vector<int> detIndex(betacopy.size(), 0), detIndexCopy(betacopy.size(), 0);
	for (int j=0; j<detIndex.size(); j++)
	  detIndex[j] = j;
	mergesort(&betacopy[0], 0, betacopy.size()-1, &detIndex[0], &AlphaMajorToBeta[i][0], &detIndexCopy[0]);
	detIndexCopy.clear();
	reorder(AlphaMajorToDet[i], detIndex);
	
	std::sort(SinglesFromAlpha[i].begin(), SinglesFromAlpha[i].end());
	
      }
    
    for (int i=0; i<BetaMajorToAlpha.size(); i++)
      {
	vector<int> betacopy = BetaMajorToAlpha[i];
	vector<int> detIndex(betacopy.size(), 0), detIndexCopy(betacopy.size(), 0);
	for (int j=0; j<detIndex.size(); j++)
	  detIndex[j] = j;
	mergesort(&betacopy[0], 0, betacopy.size()-1, &detIndex[0], &BetaMajorToAlpha[i][0], &detIndexCopy[0]);
	detIndexCopy.clear();
	reorder(BetaMajorToDet[i], detIndex);
	
	std::sort(SinglesFromBeta[i].begin(), SinglesFromBeta[i].end());
	
      }

    /*
    double totalMemoryBm = 0, totalMemoryB=0, totalMemoryAm=0, totalMemoryA=0;
    auto it = BetaNm1.begin();
    for (;it != BetaNm1.end(); it++) {
      totalMemoryBm += it->second.size();
      totalMemoryBm += sizeof(HalfDet)/sizeof(int);
    }
    for (int i=0; i<BetaMajorToAlpha.size(); i++) {
      totalMemoryB += 2.*BetaMajorToAlpha[i].size();
      totalMemoryB += SinglesFromBeta[i].size();
    }
    
    it = AlphaNm1.begin();
    for (;it != AlphaNm1.end(); it++) {
      totalMemoryAm += it->second.size();
      totalMemoryAm += sizeof(HalfDet)/sizeof(int);
    }
    for (int i=0; i<AlphaMajorToBeta.size(); i++) {
      totalMemoryA += 2.*AlphaMajorToBeta[i].size();
      totalMemoryA += SinglesFromAlpha[i].size();
    }
    pout << BetaN.size()<<"  "<<AlphaN.size()<<endl;
    pout << totalMemoryA<<"  "<<totalMemoryB<<"  "<<totalMemoryAm<<"  "<<totalMemoryBm<<endl;
    */
  }
}


void SHCImakeHamiltonian::MakeHfromHelpers2(vector<vector<int> >& AlphaMajorToBeta,
					    vector<vector<int> >& AlphaMajorToDet,
					    vector<vector<int> >& BetaMajorToAlpha,
					    vector<vector<int> >& BetaMajorToDet,
					    vector<vector<int> >& SinglesFromAlpha,
					    vector<vector<int> >& SinglesFromBeta,
					    std::vector<Determinant>& Dets,
					    int StartIndex,
					    std::vector<std::vector<int> >&connections,
					    std::vector<std::vector<CItype> >& Helements,
					    int Norbs,
					    oneInt& I1,
					    twoInt& I2,
					    double& coreE,
					    std::vector<std::vector<size_t> >& orbDifference,
					    bool DoRDM) {


  int proc=0, nprocs=1;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#endif

  size_t norbs = Norbs;

  //diagonal element
  for (size_t k=StartIndex; k<Dets.size(); k++) {
    if (k%(nprocs) != proc) continue;
    connections[k].push_back(k);
    CItype hij = Dets[k].Energy(I1, I2, coreE);
    if (Determinant::Trev != 0) updateHijForTReversal(hij, Dets[k], Dets[k], I1, I2, coreE);
    Helements[k].push_back(hij);
    if (DoRDM) orbDifference[k].push_back(0);
  }

  //alpha-beta excitation
  for (int i=0; i<AlphaMajorToBeta.size(); i++) {
    for (int ii=0; ii<AlphaMajorToBeta[i].size(); ii++) {
      if (AlphaMajorToDet[i][ii] < StartIndex || AlphaMajorToDet[i][ii]%nprocs != proc) continue;
      int Astring = i, Bstring = AlphaMajorToBeta[i][ii], DetI = AlphaMajorToDet[i][ii];

      //singles from Astring
      for (int j=0; j<SinglesFromAlpha[Astring].size(); j++) {
	int Asingle = SinglesFromAlpha[Astring][j];


	int index = binarySearch(&BetaMajorToAlpha[Bstring][0], 0, BetaMajorToAlpha[Bstring].size()-1, Asingle);
	if (index != -1 ) {
	  int DetJ = BetaMajorToDet[Bstring][index];

	  if (DetJ < DetI) {
	    size_t orbDiff;
	    CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
	    if (abs(hij) >1.e-10) {
	      connections[DetI].push_back(DetJ);
	      Helements[DetI].push_back(hij);
	      if (DoRDM) orbDifference[DetI].push_back(orbDiff);
	    }
	  }
	}

	int SearchStartIndex = 0;
	for (int k=0; k<SinglesFromBeta[Bstring].size(); k++) {
	  int& Bsingle = SinglesFromBeta[Bstring][k];

	  if (SearchStartIndex >= AlphaMajorToBeta[Asingle].size()) break;

	  int index=SearchStartIndex;
	  for (; index <AlphaMajorToBeta[Asingle].size(); index++)
	    if (AlphaMajorToBeta[Asingle][index] >= Bsingle) break;
	  SearchStartIndex = index;
	  if (index <AlphaMajorToBeta[Asingle].size() && AlphaMajorToBeta[Asingle][index] == Bsingle) {
	    int DetJ = AlphaMajorToDet[Asingle][index];

	    //int index = binarySearch(&AlphaMajorToBeta[Asingle][0], SearchStartIndex, AlphaMajorToBeta[Asingle].size()-1, Bsingle);
	    //if (index != -1 ) {
	    //SearchStartIndex = index;
	    //int DetJ = AlphaMajorToDet[Asingle][index];


	    //auto itb = lower_bound(AlphaMajorToBeta[Asingle].begin()+SearchStartIndex, AlphaMajorToBeta[Asingle].end(), Bsingle);
	    //if (itb != AlphaMajorToBeta[Asingle].end() && *itb == Bsingle) {
	    //int DetJ = AlphaMajorToDet[Asingle][SearchStartIndex];

	    if (DetJ < DetI) { //single beta, single alpha
	      size_t orbDiff;
	      CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
	      if (abs(hij) >1.e-10) {
		connections[DetI].push_back(DetJ);
		Helements[DetI].push_back(hij);
		if (DoRDM) orbDifference[DetI].push_back(orbDiff);
	      }
	    } //DetJ <Det I
	  } //*itb == Bsingle
	} //k 0->SinglesFromBeta
      } //j singles fromAlpha


      //singles from Bstring
      for (int j=0; j<SinglesFromBeta[Bstring].size(); j++) {
	int Bsingle = SinglesFromBeta[Bstring][j];

	int index = binarySearch(&AlphaMajorToBeta[Astring][0], 0, AlphaMajorToBeta[Astring].size()-1, Bsingle);
	if (index != -1 ) {
	  int DetJ = AlphaMajorToDet[Astring][index];
	  //auto itb = lower_bound(AlphaMajorToBeta[Astring].begin(), AlphaMajorToBeta[Astring].end(), Bsingle);
	  //if (itb != AlphaMajorToBeta[Astring].end() && *itb == Bsingle) {
	  //int DetJ = AlphaMajorToDet[Astring][itb-AlphaMajorToBeta[Astring].begin()];

	  if (DetJ < DetI) {
	    size_t orbDiff;
	    CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
	    if (abs(hij) <1.e-10) continue;
	    connections[DetI].push_back(DetJ);
	    Helements[DetI].push_back(hij);
	    if (DoRDM) orbDifference[DetI].push_back(orbDiff);
	  }
	}
      }


      //double beta excitation
      for (int j=0; j<AlphaMajorToBeta[i].size(); j++) {
	int DetJ = AlphaMajorToDet[i][j];

	if (DetJ < DetI && Dets[DetJ].ExcitationDistance(Dets[DetI]) == 2) {
	  size_t orbDiff;
	  CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
	  if (abs(hij) >1.e-10) {
	    connections[DetI].push_back(DetJ);
	    Helements[DetI].push_back(hij);
	    if (DoRDM) orbDifference[DetI].push_back(orbDiff);
	  }
	}
      }

      //double Alpha excitation
      for (int j=0; j<BetaMajorToAlpha[Bstring].size(); j++) {
	int DetJ = BetaMajorToDet[Bstring][j];

	if (DetJ < DetI && Dets[DetJ].ExcitationDistance(Dets[DetI]) == 2) {
	  size_t orbDiff;
	  CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
	  if (abs(hij) >1.e-10) {
	    connections[DetI].push_back(DetJ);
	    Helements[DetI].push_back(hij);
	    if (DoRDM) orbDifference[DetI].push_back(orbDiff);
	  }
	}
      }

    }
  }
}


void SHCImakeHamiltonian::MakeHfromSMHelpers2(int*          &AlphaMajorToBetaLen, 
					      vector<int* > &AlphaMajorToBeta   ,
					      vector<int* > &AlphaMajorToDet    ,
					      int*          &BetaMajorToAlphaLen, 
					      vector<int* > &BetaMajorToAlpha   ,
					      vector<int* > &BetaMajorToDet     ,
					      int*          &SinglesFromAlphaLen, 
					      vector<int* > &SinglesFromAlpha   ,
					      int*          &SinglesFromBetaLen , 
					      vector<int* > &SinglesFromBeta    ,
					      Determinant* Dets,
					      int StartIndex,
					      std::vector<std::vector<int> >&connections,
					      std::vector<std::vector<CItype> >& Helements,
					      int Norbs,
					      oneInt& I1,
					      twoInt& I2,
					      double& coreE,
					      std::vector<std::vector<size_t> >& orbDifference,
					      bool DoRDM) {


  int proc=0, nprocs=1;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#endif
  size_t norbs = Norbs;

  //diagonal element
  for (size_t k=StartIndex; k<connections.size(); k++) {
    if (k%(nprocs) != proc) continue;
    connections[k].push_back(k);
    CItype hij = Dets[k].Energy(I1, I2, coreE);
    if (Determinant::Trev != 0) updateHijForTReversal(hij, Dets[k], Dets[k], I1, I2, coreE);
    Helements[k].push_back(hij);
    if (DoRDM) orbDifference[k].push_back(0);
  }

  //alpha-beta excitation
  for (int i=0; i<AlphaMajorToBeta.size(); i++) {

    for (int ii=0; ii<AlphaMajorToBetaLen[i]; ii++) {

      if (AlphaMajorToDet[i][ii]%nprocs != proc) 
	continue;

      int Astring = i, 
	  Bstring = AlphaMajorToBeta[i][ii], 
	  DetI    = AlphaMajorToDet [i][ii];

      int maxBToA = BetaMajorToAlpha[Bstring][BetaMajorToAlphaLen[Bstring]-1];
      //singles from Astring
      for (int j=0; j<SinglesFromAlphaLen[Astring]; j++) {
	int Asingle = SinglesFromAlpha[Astring][j];

	//if (Asingle > maxBToA) break;
	int index = binarySearch ( &BetaMajorToAlpha[Bstring][0] , 
				   0                             , 
				   BetaMajorToAlphaLen[Bstring]-1, 
				   Asingle                       );
	if (index != -1 ) {
	  int DetJ = BetaMajorToDet[Bstring][index];
	  if (DetJ < StartIndex && DetI < StartIndex) continue;
	  size_t orbDiff;
	  CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
	  if (abs(hij) >1.e-10) {
	    connections[DetI].push_back(DetJ);
	    Helements  [DetI].push_back(hij);
	    if (DoRDM) orbDifference[DetI].push_back(orbDiff);
	  }
	}
      }

      //single Alpha and single Beta
      for (int j=0; j<SinglesFromAlphaLen[Astring]; j++) {
	int Asingle = SinglesFromAlpha[Astring][j];

	int SearchStartIndex = 0, AlphaToBetaLen = AlphaMajorToBetaLen[Asingle],
	  SinglesFromBLen  = SinglesFromBetaLen[Bstring];
	int maxAToB = AlphaMajorToBeta[Asingle][AlphaMajorToBetaLen[Asingle]-1];
	for (int k=0; k<SinglesFromBLen; k++) {
	  int& Bsingle = SinglesFromBeta[Bstring][k];

	  if (SearchStartIndex >= AlphaToBetaLen) break;

	  int index=SearchStartIndex;
	  for (; index <AlphaToBetaLen && AlphaMajorToBeta[Asingle][index] < Bsingle; index++) {}

	  SearchStartIndex = index;
	  if (index <AlphaToBetaLen && AlphaMajorToBeta[Asingle][index] == Bsingle) {
	    
	    int DetJ = AlphaMajorToDet[Asingle][SearchStartIndex];
	    if (DetJ < StartIndex && DetI < StartIndex) continue;
	    size_t orbDiff;
	    CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
	    if (abs(hij) >1.e-10) {
	      connections[DetI].push_back(DetJ);
	      Helements[DetI].push_back(hij);
	      if (DoRDM) orbDifference[DetI].push_back(orbDiff);
	    }
	  } //*itb == Bsingle
	} //k 0->SinglesFromBeta
      } //j singles fromAlpha


      
      //singles from Bstring
      int maxAtoB = AlphaMajorToBeta[Astring][AlphaMajorToBetaLen[Astring]-1];
      for (int j=0; j< SinglesFromBetaLen[Bstring]; j++) {
	int Bsingle =  SinglesFromBeta   [Bstring][j];

	//if (Bsingle > maxAtoB) break;
	int index = binarySearch( &AlphaMajorToBeta[Astring][0] , 
				  0                             , 
				  AlphaMajorToBetaLen[Astring]-1, 
				  Bsingle                        );

	if (index != -1 ) {
	  int DetJ = AlphaMajorToDet[Astring][index];
	  //if (DetJ < StartIndex) continue;
	  if (DetJ < StartIndex && DetI < StartIndex) continue;

	  size_t orbDiff;
	  CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
	  if (abs(hij) <1.e-10) continue;
	  connections[DetI].push_back(DetJ);
	  Helements[DetI].push_back(hij);
	  if (DoRDM) orbDifference[DetI].push_back(orbDiff);
	}
      }


      //double beta excitation
      for (int j=0; j< AlphaMajorToBetaLen[i]; j++) {
	int DetJ     = AlphaMajorToDet    [i][j];
	//if (DetJ < StartIndex) continue;
	if (DetJ < StartIndex && DetI < StartIndex) continue;

	if (Dets[DetJ].ExcitationDistance(Dets[DetI]) == 2) {
	  size_t orbDiff;
	  CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
	  if (abs(hij) >1.e-10) {
	    connections[DetI].push_back(DetJ);
	    Helements[DetI].push_back(hij);
	    if (DoRDM) orbDifference[DetI].push_back(orbDiff);
	  }
	}
      }

      //double Alpha excitation
      for (int j=0; j < BetaMajorToAlphaLen[Bstring]; j++) {
	int DetJ      = BetaMajorToDet     [Bstring][j];
	//if (DetJ < StartIndex) continue;
	if (DetJ < StartIndex && DetI < StartIndex) continue;

	if (Dets[DetJ].ExcitationDistance(Dets[DetI]) == 2) {
	  size_t orbDiff;
	  CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
	  if (abs(hij) >1.e-10) {
	    connections[DetI].push_back(DetJ);
	    Helements[DetI].push_back(hij);
	    if (DoRDM) orbDifference[DetI].push_back(orbDiff);
	  }
	}
      }

    }
  }
}

void SHCImakeHamiltonian::MakeSMHelpers(vector<vector<int> >& AlphaMajorToBeta,
					vector<vector<int> >& AlphaMajorToDet,
					vector<vector<int> >& BetaMajorToAlpha,
					vector<vector<int> >& BetaMajorToDet,
					vector<vector<int> >& SinglesFromAlpha,
					vector<vector<int> >& SinglesFromBeta,
					int* &AlphaMajorToBetaLen, vector<int* >& AlphaMajorToBetaSM,
					vector<int* >& AlphaMajorToDetSM,
					int* &BetaMajorToAlphaLen, vector<int* >& BetaMajorToAlphaSM,
					vector<int* >& BetaMajorToDetSM,
					int* &SinglesFromAlphaLen, vector<int* >& SinglesFromAlphaSM,
					int* &SinglesFromBetaLen, vector<int* >& SinglesFromBetaSM) {

  int comm_rank=0, comm_size=1;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
#endif
  boost::interprocess::shared_memory_object::remove(shciHelper.c_str());

  size_t totalMemory = 0, nBeta=BetaMajorToAlpha.size(), nAlpha =AlphaMajorToBeta.size();
  vector<int> AlphaToBetaTemp(nAlpha,0), BetaToAlphaTemp(nBeta,0), 
    AlphaSinglesTemp(nAlpha,0), BetaSinglesTemp(nBeta,0);

  if (comm_rank == 0) {
    for (int i=0; i<nAlpha; i++) {
      totalMemory        += 2*sizeof(int);
      totalMemory        += 2*sizeof(int) * AlphaMajorToBeta[i].size();
      totalMemory        +=   sizeof(int) * SinglesFromAlpha[i].size();
      AlphaToBetaTemp[i]  = AlphaMajorToBeta[i].size();
      AlphaSinglesTemp[i] = SinglesFromAlpha[i].size();
    }
    for (int i=0; i<nBeta; i++) {
      totalMemory       += 2*sizeof(int);
      totalMemory       += 2*sizeof(int) * BetaMajorToAlpha[i].size();
      totalMemory       +=   sizeof(int) * SinglesFromBeta[i].size();
      BetaToAlphaTemp[i] = BetaMajorToAlpha[i].size();
      BetaSinglesTemp[i] = SinglesFromBeta[i].size();
    }
  }

#ifndef SERIAL
  MPI_Bcast(&totalMemory, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nBeta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nAlpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (comm_rank != 0) {
    BetaToAlphaTemp.resize(nBeta); 
    AlphaToBetaTemp.resize(nAlpha);
    BetaSinglesTemp.resize(nBeta);
    AlphaSinglesTemp.resize(nAlpha);
  }
  MPI_Bcast(&BetaToAlphaTemp[0] , nBeta , MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&BetaSinglesTemp[0] , nBeta , MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&AlphaToBetaTemp[0] , nAlpha, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&AlphaSinglesTemp[0], nAlpha, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  hHelpersSegment.truncate(totalMemory);
  regionHelpers = boost::interprocess::mapped_region{hHelpersSegment, boost::interprocess::read_write};
  memset(regionHelpers.get_address(), 0., totalMemory);

#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  AlphaMajorToBetaLen = static_cast<int*>(regionHelpers.get_address());
  SinglesFromAlphaLen = AlphaMajorToBetaLen + nAlpha;
  BetaMajorToAlphaLen = SinglesFromAlphaLen + nAlpha;
  SinglesFromBetaLen  = BetaMajorToAlphaLen + nBeta;

  for (int i=0; i<nAlpha; i++) {
    AlphaMajorToBetaLen[i] = AlphaToBetaTemp [i];
    SinglesFromAlphaLen[i] = AlphaSinglesTemp[i];
  }
  for (int i=0; i<nBeta; i++) {
    BetaMajorToAlphaLen[i] = BetaToAlphaTemp[i];
    SinglesFromBetaLen [i] = BetaSinglesTemp[i];
  }

  AlphaMajorToBetaSM.resize(nAlpha); 
  AlphaMajorToDetSM .resize(nAlpha); 
  SinglesFromAlphaSM.resize(nAlpha);

  BetaMajorToAlphaSM.resize(nBeta); 
  BetaMajorToDetSM  .resize(nBeta); 
  SinglesFromBetaSM .resize(nBeta);

  int* begin = SinglesFromBetaLen + nBeta;
  size_t counter = 0;
  for (int i=0; i<nAlpha; i++) {
    AlphaMajorToBetaSM[i] = begin + counter;  counter += AlphaMajorToBetaLen[i];
    AlphaMajorToDetSM [i] = begin + counter;  counter += AlphaMajorToBetaLen[i];
    SinglesFromAlphaSM[i] = begin + counter;  counter += SinglesFromAlphaLen[i];
  }

  for (int i=0; i<nBeta; i++) {
    BetaMajorToAlphaSM[i] = begin + counter;  counter += BetaMajorToAlphaLen[i];
    BetaMajorToDetSM  [i] = begin + counter;  counter += BetaMajorToAlphaLen[i];
    SinglesFromBetaSM [i] = begin + counter;  counter += SinglesFromBetaLen [i];
  }


  //now fill the memory
  if (comm_rank == 0) {

    for (int i=0; i<nAlpha; i++) {

      for (int j=0; j<AlphaMajorToBeta[i].size(); j++) {
	AlphaMajorToBetaSM[i][j]  =  AlphaMajorToBeta[i][j];
	AlphaMajorToDetSM [i][j]  =  AlphaMajorToDet [i][j];
      }
      for (int j=0; j<SinglesFromAlpha[i].size(); j++) 
	SinglesFromAlphaSM[i][j]  =  SinglesFromAlpha[i][j];
    }


    for (int i=0; i<nBeta; i++) {

      for (int j=0; j<BetaMajorToAlpha[i].size(); j++) {
	BetaMajorToAlphaSM[i][j]  =  BetaMajorToAlpha[i][j];
	BetaMajorToDetSM  [i][j]  =  BetaMajorToDet  [i][j];
      }
      for (int j=0; j<SinglesFromBeta[i].size(); j++) 
	SinglesFromBetaSM[i][j]  =  SinglesFromBeta[i][j];
    }

  }


  long intdim  = totalMemory;
  long maxint  = 26843540; //mpi cannot transfer more than these number of doubles
  long maxIter = intdim/maxint;
#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);

  char* shrdMem = static_cast<char*>(regionHelpers.get_address());
  for (int i=0; i<maxIter; i++) {
    MPI_Bcast  ( shrdMem+i*maxint,       maxint,                       MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Bcast  ( shrdMem+(maxIter)*maxint, totalMemory - maxIter*maxint, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
#endif

}

void SHCImakeHamiltonian::MakeSHMHelpers(std::map<HalfDet, std::vector<int> >& BetaN,
		    std::map<HalfDet, std::vector<int> >& AlphaN,
		    int* &betaVecLenSHM, vector<int*>& betaVecSHM,
		    int* &alphaVecLenSHM, vector<int*>& alphaVecSHM) {
  int comm_rank=0, comm_size=1;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
#endif
  boost::interprocess::shared_memory_object::remove(shciHelper.c_str());

  size_t totalMemory = 0, nBeta=0, nAlpha=0;
  vector<int> betaveclen(BetaN.size(),0), alphaveclen(AlphaN.size(),0);
  if (comm_rank == 0) {
    //Now put it on shared memory
    auto ita = BetaN.begin();
    for (; ita != BetaN.end(); ita++) {
      betaveclen[nBeta] = ita->second.size();
      nBeta++;
      totalMemory += sizeof(int); //write how many elements in the vector
      totalMemory += sizeof(int)*ita->second.size(); //memory to store the vector      
    }
    
    
    ita = AlphaN.begin();
    for (; ita != AlphaN.end(); ita++) {
      alphaveclen[nAlpha] = ita->second.size();
      nAlpha++;
      totalMemory += sizeof(int); //write how many elements in the vector
      totalMemory += sizeof(int)*ita->second.size(); //memory to store the vector
    }
  }
#ifndef SERIAL
  MPI_Bcast(&totalMemory, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nBeta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nAlpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (comm_rank != 0) {betaveclen.resize(nBeta); alphaveclen.resize(nAlpha);}
  MPI_Bcast(&betaveclen[0], nBeta, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&alphaveclen[0], nAlpha, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  hHelpersSegment.truncate(totalMemory);
  regionHelpers = boost::interprocess::mapped_region{hHelpersSegment, boost::interprocess::read_write};
  memset(regionHelpers.get_address(), 0., totalMemory);

#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  betaVecLenSHM = static_cast<int*>(regionHelpers.get_address());
  betaVecSHM.resize(nBeta);
  size_t counter = 0;
  for (int i=0; i<nBeta; i++) {
    betaVecSHM[i] = static_cast<int*>(betaVecLenSHM+nBeta+counter);
    betaVecLenSHM[i] = betaveclen[i];
    counter += betaveclen[i];
  }

  int* beginAlpha = betaVecSHM[0]+counter;
  alphaVecLenSHM = static_cast<int*>(beginAlpha);
  alphaVecSHM.resize(nAlpha);
  counter = 0;
  for (int i=0; i<nAlpha; i++) {
    alphaVecSHM[i] = static_cast<int*>(beginAlpha+nAlpha+counter);
    alphaVecLenSHM[i] = alphaveclen[i];
    counter += alphaveclen[i];
  }

 
  //now fill the memory
  if (comm_rank == 0) {
    size_t nBeta=0, nAlpha=0;
    auto ita = BetaN.begin();
    for (; ita != BetaN.end(); ita++) {
      for (int j=0; j<ita->second.size(); j++) {
	betaVecSHM[nBeta][j] = ita->second[j];
      }
      nBeta++;
    }
    
    
    ita = AlphaN.begin();
    for (; ita != AlphaN.end(); ita++) {
      for (int j=0; j<ita->second.size(); j++) 
	alphaVecSHM[nAlpha][j] = ita->second[j];
      nAlpha++;
    }
    
  }

  long intdim = totalMemory;
  long  maxint = 26843540; //mpi cannot transfer more than these number of doubles
  long maxIter = intdim/maxint;
#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);
  char* shrdMem = static_cast<char*>(regionHelpers.get_address());
  for (int i=0; i<maxIter; i++) {
    MPI::COMM_WORLD.Bcast(shrdMem+i*maxint, maxint, MPI_CHAR, 0);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI::COMM_WORLD.Bcast(shrdMem+(maxIter)*maxint, totalMemory - maxIter*maxint, MPI_CHAR, 0);
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}



void SHCImakeHamiltonian::updateSOCconnections(Determinant *Dets, int prevSize, vector<vector<int> >& connections, vector<vector<size_t> >& orbDifference, vector<vector<CItype> >& Helements, int norbs, oneInt& int1, int nelec, bool includeSz) {

  size_t Norbs = norbs;

  map<Determinant, int> SortedDets;
  for (int i=0; i<connections.size(); i++)
    SortedDets[Dets[i]] = i;

  int nprocs= mpigetsize(), proc = mpigetrank();

  //#pragma omp parallel for schedule(dynamic)
#pragma omp parallel
  {

    for (int x=prevSize; x<connections.size(); x++) {
      if (x%(nprocs*omp_get_num_threads()) != proc*omp_get_num_threads()+omp_get_thread_num()) continue;
      Determinant& d = Dets[x];

      vector<int> closed(nelec,0);
      vector<int> open(norbs-nelec,0);
      d.getOpenClosed(open, closed);
      int nclosed = nelec;
      int nopen = norbs-nclosed;

      //loop over all single excitation and find if they are present in the list
      //on or before the current determinant
      for (int ia=0; ia<nopen*nclosed; ia++){
	int i=ia/nopen, a=ia%nopen;

	CItype integral = int1(open[a],closed[i]);
	if (abs(integral) < 1.e-8) continue;

	Determinant di = d;
	if (open[a]%2 == closed[i]%2 && !includeSz) continue;

	di.setocc(open[a], true); di.setocc(closed[i],false);
	double sgn = 1.0;
	d.parity(min(open[a],closed[i]), max(open[a],closed[i]),sgn);


	map<Determinant, int>::iterator it = SortedDets.find(di);
	if (it != SortedDets.end() ) {
	  int y = it->second;
	  if (y < x) {
	    connections[x].push_back(y);
	    Helements[x].push_back(integral*sgn);
	    orbDifference[x].push_back(open[a]*norbs+closed[i]);
	  }
	}
      }
    }
  }
}
