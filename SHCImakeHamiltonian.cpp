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


void SHCImakeHamiltonian::SparseHam::setNbatches(int DetSize) {

  int proc=0, nprocs=1;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#endif

  Nbatches = diskio ? (DetSize)/BatchSize/nprocs : 1;
  if ((DetSize) > Nbatches*BatchSize*nprocs && diskio)
    Nbatches +=1;
}

void SHCImakeHamiltonian::SparseHam::writeBatch(int batch) {


  char file [5000];
  sprintf (file, "%s/%d-hamiltonian-batch%d.bkp" , prefix.c_str(), commrank, batch );
  std::ofstream ofs(file, std::ios::binary);
  boost::archive::binary_oarchive save(ofs);
  save << connections << Helements << orbDifference;

}


void SHCImakeHamiltonian::SparseHam::readBatch (int batch) {

  char file [5000];
  sprintf (file, "%s/%d-hamiltonian-batch%d.bkp" , prefix.c_str(), commrank, batch );
  std::ifstream ifs(file, std::ios::binary);
  boost::archive::binary_iarchive load(ifs);
  load >> connections >> Helements >> orbDifference;

}

void SHCImakeHamiltonian::HamHelpers2::MakeSHMHelpers() {
  SHCImakeHamiltonian::MakeSMHelpers( AlphaMajorToBeta, AlphaMajorToDet,
				      BetaMajorToAlpha, BetaMajorToDet ,
				      SinglesFromAlpha, SinglesFromBeta,
				      AlphaMajorToBetaLen, AlphaMajorToBetaSM,
				      AlphaMajorToDetSM,
				      BetaMajorToAlphaLen, BetaMajorToAlphaSM,
				      BetaMajorToDetSM ,
				      SinglesFromAlphaLen, SinglesFromAlphaSM,
				      SinglesFromBetaLen , SinglesFromBetaSM);

}

void SHCImakeHamiltonian::SparseHam::makeFromHelper(HamHelpers2& helpers2, Determinant *SHMDets, int startIndex, int endIndex, int Norbs, oneInt& I1, twoInt& I2, double& coreE, bool DoRDM) {

  std::vector<std::vector<size_t> > orbDifference;

  SHCImakeHamiltonian::MakeHfromSMHelpers2(helpers2.AlphaMajorToBetaLen, 
					   helpers2.AlphaMajorToBetaSM ,
					   helpers2.AlphaMajorToDetSM  ,
					   helpers2.BetaMajorToAlphaLen, 
					   helpers2.BetaMajorToAlphaSM ,
					   helpers2.BetaMajorToDetSM   ,
					   helpers2.SinglesFromAlphaLen, 
					   helpers2.SinglesFromAlphaSM ,
					   helpers2.SinglesFromBetaLen , 
					   helpers2.SinglesFromBetaSM  ,
					   SHMDets, startIndex, endIndex,
					   diskio, *this, Norbs,
					   I1, I2, coreE, 
					   DoRDM);

}

void SHCImakeHamiltonian::HamHelpers2::PopulateHelpers ( Determinant* SHMDets,
							 int DetsSize, 
							 int startIndex) {

  SHCImakeHamiltonian::PopulateHelperLists2(BetaN, AlphaN,  
					    AlphaMajorToBeta, AlphaMajorToDet,
					    BetaMajorToAlpha, BetaMajorToDet, 
					    SinglesFromAlpha, SinglesFromBeta,
					    SHMDets, DetsSize, startIndex);
}



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
  int size = commsize, rank = commrank;

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

//check if the cost can be reduced by half by only storiong connection > than the index
void SHCImakeHamiltonian::PopulateHelperLists2(std::map<HalfDet, int >& BetaN,
					       std::map<HalfDet, int >& AlphaN,
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

  if (commrank == 0) {
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
	std::vector<int> openb(norbs/2,0);
	int nclosedb = db.getOpenClosed(openb, closedb);

	for (int j=0; j<nclosedb; j++)
	  for (int k=0; k<norbs/2-nclosedb; k++) {
	    HalfDet dbcopy = db;
	    dbcopy.setocc(closedb[j], false);
	    dbcopy.setocc(openb[k], true);
	    auto itbcopy = BetaN.find(dbcopy);
	    if (itbcopy != BetaN.end()) {
	      SinglesFromBeta[itb->second].push_back(itbcopy->second);
	      SinglesFromBeta[itbcopy->second].push_back(itb->second);
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
	std::vector<int> opena(norbs/2,0);
	int ncloseda = da.getOpenClosed(opena, closeda);

	for (int j=0; j<ncloseda; j++)
	  for (int k=0; k<norbs/2-ncloseda; k++) {
	    HalfDet dacopy = da;
	    dacopy.setocc(closeda[j], false);
	    dacopy.setocc(opena[k], true);
	    auto itacopy = AlphaN.find(dacopy);
	    if (itacopy != AlphaN.end()) {
	      SinglesFromAlpha[ita->second].push_back(itacopy->second);
	      SinglesFromAlpha[itacopy->second].push_back(ita->second);
	    }
	  }      
	
      }
      
      AlphaMajorToBeta[ita->second].push_back(itb->second);
      AlphaMajorToDet[ita->second].push_back(i);
      
      
      BetaMajorToAlpha[itb->second].push_back(ita->second);
      BetaMajorToDet[itb->second].push_back(i);
      
      
    }
    //printf("Nalpha: %12d,  Nbeta: %12d\n", AlphaN.size(), BetaN.size());
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
    //pout << " ."<<endl;

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
					      int EndIndex,
					      bool diskio,
					      SparseHam& sparseHam,
					      int Norbs,
					      oneInt& I1,
					      twoInt& I2,
					      double& coreE,
					      bool DoRDM) {


  int proc=0, nprocs=1;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#endif
  size_t norbs = Norbs;
  int offSet = 0;

  std::vector<std::vector<int> >& connections = sparseHam.connections;
  std::vector<std::vector<CItype> >& Helements = sparseHam.Helements;
  std::vector<std::vector<size_t> >& orbDifference = sparseHam.orbDifference;
  
  //diagonal element
  for (size_t k=StartIndex; k<EndIndex; k++) {
    if (k%(nprocs) != proc || k < max(StartIndex, offSet) ) continue;
    connections.push_back(vector<int>(1,k));
    CItype hij = Dets[k].Energy(I1, I2, coreE);
    if (Determinant::Trev != 0) updateHijForTReversal(hij, Dets[k], Dets[k], I1, I2, coreE);
    Helements.push_back(vector<CItype>(1,hij));
    if (DoRDM) orbDifference.push_back(vector<size_t>(1,0));
  }
    
  //alpha-beta excitation
  for (int i=0; i<AlphaMajorToBeta.size(); i++) {
    
    for (int ii=0; ii<AlphaMajorToBetaLen[i]; ii++) {
      
      int Astring = i, 
	Bstring = AlphaMajorToBeta[i][ii], 
	DetI    = AlphaMajorToDet [i][ii];
      
      if (DetI%nprocs != proc || DetI < StartIndex) 
	continue;
      
      
      int maxBToA = BetaMajorToAlpha[Bstring][BetaMajorToAlphaLen[Bstring]-1];
      //singles from Astring
      for (int j=0; j<SinglesFromAlphaLen[Astring]; j++) {
	int Asingle = SinglesFromAlpha[Astring][j];
	
	int index = binarySearch ( &BetaMajorToAlpha[Bstring][0] , 
				   0                             , 
				   BetaMajorToAlphaLen[Bstring]-1, 
				   Asingle                       );
	if (index != -1 ) {
	  int DetJ = BetaMajorToDet[Bstring][index];
	  //if (DetJ < max(offSet, StartIndex) && DetI < max(offSet, StartIndex)) continue;
	  if (DetJ >  DetI) continue;
	  size_t orbDiff;
	  CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
	  if (abs(hij) >1.e-10) {
	    connections[(DetI-offSet)/nprocs].push_back(DetJ);
	    Helements  [(DetI-offSet)/nprocs].push_back(hij);
	    if (DoRDM) orbDifference[(DetI-offSet)/nprocs].push_back(orbDiff);
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
	    //if (DetJ < max(offSet, StartIndex) && DetI < max(offSet, StartIndex)) continue;
	    if (DetJ >  DetI) continue;
	    size_t orbDiff;
	    CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
	    if (abs(hij) >1.e-10) {
	      connections[(DetI-offSet)/nprocs].push_back(DetJ);
	      Helements[(DetI-offSet)/nprocs].push_back(hij);
	      if (DoRDM) orbDifference[(DetI-offSet)/nprocs].push_back(orbDiff);
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
	  //if (DetJ < max(offSet, StartIndex) && DetI < max(offSet, StartIndex)) continue;
	  if (DetJ > DetI) continue;

	  size_t orbDiff;
	  CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
	  if (abs(hij) <1.e-10) continue;
	  connections[(DetI-offSet)/nprocs].push_back(DetJ);
	  Helements[(DetI-offSet)/nprocs].push_back(hij);
	  if (DoRDM) orbDifference[(DetI-offSet)/nprocs].push_back(orbDiff);
	}
      }
      
      
      //double beta excitation
      for (int j=0; j< AlphaMajorToBetaLen[i]; j++) {
	int DetJ     = AlphaMajorToDet    [i][j];
	//if (DetJ < StartIndex) continue;
	//if (DetJ < max(offSet, StartIndex) && DetI < max(offSet, StartIndex)) continue;
	if (DetJ > DetI) continue;

	if (Dets[DetJ].ExcitationDistance(Dets[DetI]) == 2) {
	  size_t orbDiff;
	  CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
	  if (abs(hij) >1.e-10) {
	    connections[(DetI-offSet)/nprocs].push_back(DetJ);
	    Helements[(DetI-offSet)/nprocs].push_back(hij);
	    if (DoRDM) orbDifference[(DetI-offSet)/nprocs].push_back(orbDiff);
	  }
	}
      }
      
      //double Alpha excitation
      for (int j=0; j < BetaMajorToAlphaLen[Bstring]; j++) {
	int DetJ      = BetaMajorToDet     [Bstring][j];
	//if (DetJ < StartIndex) continue;
	//if (DetJ < max(offSet, StartIndex) && DetI < max(offSet, StartIndex)) continue;
	if (DetJ > DetI) continue;

	if (Dets[DetJ].ExcitationDistance(Dets[DetI]) == 2) {
	  size_t orbDiff;
	  CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
	  if (abs(hij) >1.e-10) {
	    connections[(DetI-offSet)/nprocs].push_back(DetJ);
	    Helements[(DetI-offSet)/nprocs].push_back(hij);
	    if (DoRDM) orbDifference[(DetI-offSet)/nprocs].push_back(orbDiff);
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


void SHCImakeHamiltonian::updateSOCconnections(Determinant *Dets, int prevSize, int DetsSize, 
					       Determinant *SortedDets, vector<vector<int>>& connections,
					       vector<vector<size_t> >& orbDifference, 
					       vector<vector<CItype> >& Helements, 
					       int norbs, oneInt& int1, int nelec, bool includeSz) {

  size_t Norbs = norbs;

  int nprocs= commsize, proc = commrank;

  for (int x=prevSize; x<DetsSize; x++) {
    if (x%(nprocs) != proc) continue;
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
      
      
      //map<Determinant, int>::iterator it = SortedDets.find(di);
      Determinant* it = std::lower_bound(SortedDets, SortedDets+DetsSize, di);
      if (it != SortedDets+DetsSize ) {
	int y = it - SortedDets;
	if (y < x) {
	  connections[x/nprocs].push_back(y);
	  Helements[x/nprocs].push_back(integral*sgn);
	  if (orbDifference.size() != 0) orbDifference[x/nprocs].push_back(open[a]*norbs+closed[i]);
	}
      }
    }
  }
}

