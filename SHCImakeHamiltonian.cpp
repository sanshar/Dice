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

#pragma omp parallel
  {
    int ithrd = omp_get_thread_num();
    int nthrd = omp_get_num_threads();
    for (size_t k=StartIndex; k<Dets.size(); k++) {
      if (k%(nprocs*nthrd) != proc*nthrd+ithrd) continue;
      connections[k].push_back(k);
      CItype hij = Dets[k].Energy(I1, I2, coreE);
      if (Determinant::Trev != 0) updateHijForTReversal(hij, Dets[k], Dets[k], I1, I2, coreE);
      Helements[k].push_back(hij);
      if (DoRDM) orbDifference[k].push_back(0);
    }
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

#pragma omp parallel
    {
      int ithrd = omp_get_thread_num();
      int nthrd = omp_get_num_threads();
      for (int k=localStart; k<detIndex.size(); k++) {

	if (detIndex[k]%(nprocs*nthrd) != proc*nthrd+ithrd) continue;

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
    index++;
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

#pragma omp parallel
    {
      int ithrd = omp_get_thread_num();
      int nthrd = omp_get_num_threads();

      for (int k=localStart; k<detIndex.size(); k++) {
	if (detIndex[k]%(nprocs*nthrd) != proc*nthrd+ithrd) continue;

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
    index++;
  }

}

void SHCImakeHamiltonian::MakeHfromHelpers(int* &BetaVecLen, vector<int*> &BetaVec,
					   int* &AlphaVecLen, vector<int*> &AlphaVec,
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

#pragma omp parallel
  {
    int ithrd = omp_get_thread_num();
    int nthrd = omp_get_num_threads();
    for (size_t k=StartIndex; k<Dets.size(); k++) {
      if (k%(nprocs*nthrd) != proc*nthrd+ithrd) continue;
      connections[k].push_back(k);
      CItype hij = Dets[k].Energy(I1, I2, coreE);
      if (Determinant::Trev != 0) updateHijForTReversal(hij, Dets[k], Dets[k], I1, I2, coreE);
      Helements[k].push_back(hij);
      if (DoRDM) orbDifference[k].push_back(0);
    }
  }

  int index = 0;
  for (int i=0;i<BetaVec.size(); i++) {
    int* detIndex = BetaVec[i];
    int localStart = BetaVecLen[i];
    for (int j=0; j<BetaVecLen[i]; j++)
      if (detIndex[j] >= StartIndex) {
	localStart = j; break;
      }

    int ithrd = omp_get_thread_num();
    int nthrd = omp_get_num_threads();
    for (int k=localStart; k<BetaVecLen[i]; k++) {
      
      if (detIndex[k]%(nprocs*nthrd) != proc*nthrd+ithrd) continue;
      
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
    
    int ithrd = omp_get_thread_num();
    int nthrd = omp_get_num_threads();
    
    for (int k=localStart; k<AlphaVecLen[i]; k++) {
      if (detIndex[k]%(nprocs*nthrd) != proc*nthrd+ithrd) continue;

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
					       vector<vector<int> >& AlphaMajorToBeta,
					       vector<vector<int> >& AlphaMajorToDet,
					       vector<vector<int> >& BetaMajorToAlpha,
					       vector<vector<int> >& BetaMajorToDet,
					       vector< vector<int> >& SinglesFromAlpha,
					       vector< vector<int> >& SinglesFromBeta,
					       vector< vector<int> >& DoublesFromAlpha,
					       vector< vector<int> >& DoublesFromBeta,
					       std::vector<Determinant>& Dets,
					       int StartIndex)
{
  //ith vector of AlphaMajor contains all Determinants that have
  //ith Alpha string, and the 2j and 2j+1 elements of this ith vector
  //are the indices of the beta string and the determinant respectively

  for (int i=StartIndex; i<Dets.size(); i++) {
    HalfDet da = Dets[i].getAlpha(), db = Dets[i].getBeta();

    auto itb = BetaN.find(db);
    if (itb == BetaN.end()) {
      auto ret = BetaN.insert( std::pair<HalfDet, int>(db, BetaMajorToDet.size()));
      itb = ret.first;
      BetaMajorToAlpha.resize(itb->second+1);
      BetaMajorToDet.resize(itb->second+1);

      SinglesFromBeta.resize(itb->second+1);
      DoublesFromBeta.resize(itb->second+1);
      auto it = BetaN.begin();
      for (; it != BetaN.end(); it++) {
	if (db.ExcitationDistance(it->first) == 1) {
	  SinglesFromBeta[itb->second].push_back(it->second);
	  SinglesFromBeta[it->second].push_back(itb->second);
	}
	if (db.ExcitationDistance(it->first) == 2) {
	  //if (itb->second <it->second)
	    DoublesFromBeta[itb->second].push_back(it->second);
	    //else
	    DoublesFromBeta[it->second].push_back(itb->second);
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
      DoublesFromAlpha.resize(ita->second+1);
      auto it = AlphaN.begin();
      for (; it != AlphaN.end(); it++) {
	if (da.ExcitationDistance(it->first) == 1) {
	  SinglesFromAlpha[ita->second].push_back(it->second);
	  SinglesFromAlpha[it->second].push_back(ita->second);
	}

	if (da.ExcitationDistance(it->first) == 2) {
	  //if (ita->second > it->second)
	    DoublesFromAlpha[ita->second].push_back(it->second);
	    //else
	    DoublesFromAlpha[it->second].push_back(ita->second);
	}
      }

    }

    AlphaMajorToBeta[ita->second].push_back(itb->second);
    AlphaMajorToDet[ita->second].push_back(i);


    BetaMajorToAlpha[itb->second].push_back(ita->second);
    BetaMajorToDet[itb->second].push_back(i);


  }

  for (int i=0; i<AlphaMajorToBeta.size(); i++)
  {
    vector<int> betacopy = AlphaMajorToBeta[i];
    vector<int> detIndex(betacopy.size(), 0), detIndexCopy(betacopy.size(), 0);
    for (int i=0; i<detIndex.size(); i++)
      detIndex[i] = i;
    mergesort(&betacopy[0], 0, betacopy.size()-1, &detIndex[0], &AlphaMajorToBeta[i][0], &detIndexCopy[0]);
    detIndexCopy.clear();
    reorder(AlphaMajorToDet[i], detIndex);

    std::sort(SinglesFromAlpha[i].begin(), SinglesFromAlpha[i].end());
    std::sort(DoublesFromAlpha[i].begin(), DoublesFromAlpha[i].end());

  }

  for (int i=0; i<BetaMajorToAlpha.size(); i++)
  {
    vector<int> betacopy = BetaMajorToAlpha[i];
    vector<int> detIndex(betacopy.size(), 0), detIndexCopy(betacopy.size(), 0);
    for (int i=0; i<detIndex.size(); i++)
      detIndex[i] = i;
    mergesort(&betacopy[0], 0, betacopy.size()-1, &detIndex[0], &BetaMajorToAlpha[i][0], &detIndexCopy[0]);
    detIndexCopy.clear();
    reorder(BetaMajorToDet[i], detIndex);

    std::sort(SinglesFromBeta[i].begin(), SinglesFromBeta[i].end());
    std::sort(DoublesFromBeta[i].begin(), DoublesFromBeta[i].end());

  }

}


void SHCImakeHamiltonian::MakeHfromHelpers2(vector<vector<int> >& AlphaMajorToBeta,
					    vector<vector<int> >& AlphaMajorToDet,
					    vector<vector<int> >& BetaMajorToAlpha,
					    vector<vector<int> >& BetaMajorToDet,
					    vector<vector<int> >& SinglesFromAlpha,
					    vector<vector<int> >& SinglesFromBeta,
					    vector<vector<int> >& DoublesFromAlpha,
					    vector<vector<int> >& DoublesFromBeta,
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


	//auto itb = lower_bound(BetaMajorToAlpha[Bstring].begin(), BetaMajorToAlpha[Bstring].end(), Asingle);
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
	  int index = binarySearch(&AlphaMajorToBeta[Asingle][0], SearchStartIndex, AlphaMajorToBeta[Asingle].size()-1, Bsingle);
	  if (index != -1 ) {
	    SearchStartIndex = index;
	    int DetJ = AlphaMajorToDet[Asingle][index];
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


      //doubles from Astring
      for (int j=0; j<DoublesFromAlpha[Astring].size(); j++) {
	int Adouble = DoublesFromAlpha[Astring][j];

	int index = binarySearch(&BetaMajorToAlpha[Bstring][0], 0, BetaMajorToAlpha[Bstring].size()-1, Adouble);
	if (index != -1 ) {
	  int DetJ = BetaMajorToDet[Bstring][index];
	  //auto itb = lower_bound(BetaMajorToAlpha[Bstring].begin(), BetaMajorToAlpha[Bstring].end(), Adouble);
	  //if (itb != BetaMajorToAlpha[Bstring].end() && *itb == Adouble) {
	  //int DetJ = BetaMajorToDet[Bstring][itb-BetaMajorToAlpha[Bstring].begin()];

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
      }

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

      //Doubles from Bstring
      for (int j=0; j<DoublesFromBeta[Bstring].size(); j++) {
	int Bdouble = DoublesFromBeta[Bstring][j];

	int index = binarySearch(&AlphaMajorToBeta[Astring][0], 0, AlphaMajorToBeta[Astring].size()-1, Bdouble);
	if (index != -1 ) {
	  int DetJ = AlphaMajorToDet[Astring][index];
	  //auto itb = lower_bound(AlphaMajorToBeta[Astring].begin(), AlphaMajorToBeta[Astring].end(), Bdouble);
	  //if (itb != AlphaMajorToBeta[Astring].end() && *itb == Bdouble) {
	  //int DetJ = AlphaMajorToDet[Astring][itb-AlphaMajorToBeta[Astring].begin()];

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



void SHCImakeHamiltonian::updateSOCconnections(vector<Determinant>& Dets, int prevSize, vector<vector<int> >& connections, vector<vector<size_t> >& orbDifference, vector<vector<CItype> >& Helements, int norbs, oneInt& int1, int nelec, bool includeSz) {

  size_t Norbs = norbs;

  map<Determinant, int> SortedDets;
  for (int i=0; i<Dets.size(); i++)
    SortedDets[Dets[i]] = i;

  int nprocs= mpigetsize(), proc = mpigetrank();

  //#pragma omp parallel for schedule(dynamic)
#pragma omp parallel
  {

    for (int x=prevSize; x<Dets.size(); x++) {
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
