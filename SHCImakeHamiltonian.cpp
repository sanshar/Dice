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

#pragma omp parallel
  {
    for (int i=0; i<connections.size(); i++) {
      if ((i/omp_get_num_threads())%world.size() != world.rank()) continue;
      Helements[i][0] = Dets[i].Energy(I1, I2, coreE);
      for (int j=1; j<connections[i].size(); j++) {
	size_t orbDiff;
	Helements[i][j] = Hij(Dets[i], Dets[connections[i][j]], I1, I2, coreE, orbDiff);
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
    for (size_t k=StartIndex; k<Dets.size(); k++) {
      if (k%(nprocs*omp_get_num_threads()) != proc*omp_get_num_threads()+omp_get_thread_num()) continue;
      connections[k].push_back(k);
      CItype hij = Dets[k].Energy(I1, I2, coreE);
      Helements[k].push_back(hij);
      if (DoRDM) orbDifference[k].push_back(0);
    }
  }

  std::map<HalfDet, std::vector<int> >::iterator ita = BetaN.begin();
  int index = 0;
  pout <<"# "<< Dets.size()<<"  "<<BetaN.size()<<"  "<<AlphaNm1.size()<<endl;
  pout << "#";
  for (; ita!=BetaN.end(); ita++) {
    std::vector<int>& detIndex = ita->second;
    int localStart = detIndex.size();
    for (int j=0; j<detIndex.size(); j++)
      if (detIndex[j] >= StartIndex) {
	localStart = j; break;
      }

#pragma omp parallel
    {
      for (int k=localStart; k<detIndex.size(); k++) {

	if (detIndex[k]%(nprocs*omp_get_num_threads()) != proc*omp_get_num_threads()+omp_get_thread_num()) continue;

	for(int j=0; j<k; j++) {
	  size_t J = detIndex[j];size_t K = detIndex[k];
	  if (Dets[J].connected(Dets[K])) 	  {
	    connections[K].push_back(J);
	    size_t orbDiff;
	    CItype hij = Hij(Dets[J], Dets[K], I1, I2, coreE, orbDiff);
	    Helements[K].push_back(hij);
	    if (DoRDM)
	      orbDifference[K].push_back(orbDiff);
	  }
	}
      }
    }
    index++;
    if (index%1000000 == 0 && index!= 0) {pout <<". ";}
  }
  pout << format("BetaN    %49.2f\n#")
    % (getTime()-startofCalc);

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
      for (int k=localStart; k<detIndex.size(); k++) {
	if (detIndex[k]%(nprocs*omp_get_num_threads()) != proc*omp_get_num_threads()+omp_get_thread_num()) continue;

	for(int j=0; j<k; j++) {
	  size_t J = detIndex[j];size_t K = detIndex[k];
	  if (Dets[J].connected(Dets[K]) ) {
	    if (find(connections[K].begin(), connections[K].end(), J) == connections[K].end()){
	      connections[K].push_back(J);
	      size_t orbDiff;
	      CItype hij = Hij(Dets[J], Dets[K], I1, I2, coreE, orbDiff);
	      Helements[K].push_back(hij);

	      if (DoRDM)
		orbDifference[K].push_back(orbDiff);
	    }
	  }
	}
      }
    }
    index++;
    if (index%1000000 == 0 && index!= 0) {pout <<". ";}
  }

  pout << format("AlphaN-1 %49.2f\n")
    % (getTime()-startofCalc);


}

void SHCImakeHamiltonian::PopulateHelperLists(std::map<HalfDet, std::vector<int> >& BetaN,
				    std::map<HalfDet, std::vector<int> >& AlphaNm1,
				    std::vector<Determinant>& Dets,
				    int StartIndex)
{
  pout << format("#Making Helpers %43.2f\n")
    % (getTime()-startofCalc);
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
  }

}



void SHCImakeHamiltonian::updateSOCconnections(vector<Determinant>& Dets, int prevSize, vector<vector<int> >& connections, vector<vector<CItype> >& Helements, int norbs, oneInt& int1, int nelec, bool includeSz) {

  size_t Norbs = norbs;

  map<Determinant, int> SortedDets;
  for (int i=0; i<Dets.size(); i++)
    SortedDets[Dets[i]] = i;


  //#pragma omp parallel for schedule(dynamic)
  for (int x=prevSize; x<Dets.size(); x++) {
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

	}
      }
    }
  }
}
