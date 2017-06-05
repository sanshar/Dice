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
	  //if (Dets[J].connected(Dets[K]) ||  (Determinant::Trev!=0 && Dets[J].connectedToFlipAlphaBeta(Dets[K]))) {

	    size_t orbDiff;
	    CItype hij = Hij(Dets[J], Dets[K], I1, I2, coreE, orbDiff);
	    if (Determinant::Trev != 0) updateHijForTReversal(hij, Dets[J], Dets[K], I1, I2, coreE);
	    
	    if (abs(hij) <1.e-10) continue;
	    Helements[K].push_back(hij);
	    connections[K].push_back(J);
	    
	    if (DoRDM)
	      orbDifference[K].push_back(orbDiff);
	    //}
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
