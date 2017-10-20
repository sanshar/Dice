/* Developed by Sandeep Sharma with contributions from James E. Smith and Adam A. Homes, 2017
Copyright (c) 2017, Sandeep Sharma

This file is part of DICE.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

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
#include "SHCIshm.h"
#include "LCC.h"

#include "communicate.h"


using namespace std;
using namespace Eigen;
using namespace boost;
using namespace SHCISortMpiUtils;


double LCC::doLCC(Determinant *Dets, CItype *ci, int DetsSize, double& E0, oneInt& I1, twoInt& I2,
			 twoIntHeatBathSHM& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec, int root) {



#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  int norbs = Determinant::norbs;

  Determinant* SortedDets;
  std::vector<Determinant> SortedDetsvec;
  if (commrank == 0 ) {  
    for (int i=0; i<DetsSize; i++)
      SortedDetsvec.push_back(Dets[i]);
    std::sort(SortedDetsvec.begin(), SortedDetsvec.end());
  }
#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  SHMVecFromVecs(SortedDetsvec, SortedDets, shciSortedDets, SortedDetsSegment, regionSortedDets);
  SortedDetsvec.clear();

  double energyEN = 0.0;
  double Psi1NormProc = 0.0;

  StitchDEH uniqueDEH;
  double totalPT = 0.0;
  int ntries = 0;

  int size = commsize, rank = commrank;
  vector<size_t> all_to_all(size*size,0);

  for (int i=0; i<DetsSize; i++) {
    if ((i%size != rank)) continue;
    
    LCC::getDeterminantsLCC(Dets[i], abs(schd.epsilon2/ci[i]), ci[i], 0.0,
			    I1, I2, I2HB, irrep, coreE, E0,
			    *uniqueDEH.Det,
			    *uniqueDEH.Num,
			    *uniqueDEH.Energy,
			    schd,0, nelec);
  }
  
  uniqueDEH.MergeSortAndRemoveDuplicates();
  uniqueDEH.RemoveDetsPresentIn(SortedDets, DetsSize);

  for (int level = 0; level <ceil(log2(size)); level++) {
    
    //cout <<level<<" "<<rank<<" recv/send "<<rank%ipow(2, level+1)<<"  "<<rank%ipow(2, level)<<"  "<<ceil(log2(size))<<endl;
    if (rank%ipow(2, level+1) == 0 && rank + ipow(2, level) < size) {
      //cout <<rank<<" recv "<<level<<"  "<<ceil(log2(size))<<endl;
      int getproc = rank+ipow(2,level);
      long numDets = 0;
      long oldSize = uniqueDEH.Det->size();
      long maxint = 26843540;
      MPI_Recv(&numDets, 1, MPI_DOUBLE, getproc, getproc, 
	       MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
      long totalMemory = numDets*DetLen;
      
      if (totalMemory != 0) {
	uniqueDEH.Det->resize(oldSize+numDets);
	uniqueDEH.Num->resize(oldSize+numDets);
	uniqueDEH.Energy->resize(oldSize+numDets);
	for (int i=0; i<(totalMemory/maxint); i++) 
	  MPI_Recv(&(uniqueDEH.Det->at(oldSize).repr[0])+i*maxint, 
		   maxint, MPI_DOUBLE, getproc, getproc, 
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&(uniqueDEH.Det->at(oldSize).repr[0])+(totalMemory/maxint)*maxint, 
		 totalMemory-(totalMemory/maxint)*maxint, MPI_DOUBLE, 
		 getproc, getproc, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
	for (int i=0; i<(numDets/maxint); i++) 
	  MPI_Recv(&(uniqueDEH.Num->at(oldSize))+i*maxint, 
		   maxint, MPI_DOUBLE, getproc, getproc, 
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&(uniqueDEH.Num->at(oldSize))+(numDets/maxint)*maxint, 
		 numDets-(numDets/maxint)*maxint, MPI_DOUBLE, 
		 getproc, getproc, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	for (int i=0; i<(numDets/maxint); i++) 
	  MPI_Recv(&(uniqueDEH.Energy->at(oldSize))+i*maxint, 
		   maxint, MPI_DOUBLE, getproc, getproc, 
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&(uniqueDEH.Energy->at(oldSize))+(numDets/maxint)*maxint, 
		 numDets-(numDets/maxint)*maxint, MPI_DOUBLE, 
		 getproc, getproc, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
	
	uniqueDEH.MergeSortAndRemoveDuplicates();
	//cout << uniqueDEH.Det->size()<<" recv "<<commrank<<endl;
	
      }
    }
    else if ( rank%ipow(2, level+1) == 0 && rank + ipow(2, level) >= size) {
      //cout <<rank<<" should not be here "<<level<<"  "<<ceil(log2(size))<<endl;
      continue ;
    }
    else if ( rank%ipow(2, level) == 0) {
      //cout <<rank<<" send "<<level<<"  "<<ceil(log2(size))<<endl;
      int toproc = rank-ipow(2,level);
      int proc = commrank;
      long numDets = uniqueDEH.Det->size();
      long maxint = 26843540;
      long totalMemory = numDets*DetLen;	
      MPI_Send(&numDets, 1, MPI_DOUBLE, toproc, proc, MPI_COMM_WORLD); 
      
      if (totalMemory != 0) {
	for (int i=0; i<(totalMemory/maxint); i++)
	  MPI_Send(&(uniqueDEH.Det->at(0).repr[0])+i*maxint, 
		   maxint, MPI_DOUBLE, toproc, proc, MPI_COMM_WORLD);
	MPI_Send(&(uniqueDEH.Det->at(0).repr[0])+(totalMemory/maxint)*maxint, 
		 totalMemory-(totalMemory/maxint)*maxint, MPI_DOUBLE, 
		 toproc, proc, MPI_COMM_WORLD);	
	
	for (int i=0; i<(numDets/maxint); i++)
	  MPI_Send(&(uniqueDEH.Num->at(0))+i*maxint, 
		   maxint, MPI_DOUBLE, toproc, proc, MPI_COMM_WORLD);
	MPI_Send(&(uniqueDEH.Num->at(0))+(numDets/maxint)*maxint, 
		 numDets-(numDets/maxint)*maxint, MPI_DOUBLE, 
		 toproc, proc, MPI_COMM_WORLD);	

	for (int i=0; i<(numDets/maxint); i++)
	  MPI_Send(&(uniqueDEH.Energy->at(0))+i*maxint, 
		   maxint, MPI_DOUBLE, toproc, proc, MPI_COMM_WORLD);
	MPI_Send(&(uniqueDEH.Energy->at(0))+(numDets/maxint)*maxint, 
		 numDets-(numDets/maxint)*maxint, MPI_DOUBLE, 
		 toproc, proc, MPI_COMM_WORLD);	
	
	uniqueDEH.clear();
	//cout << uniqueDEH.Det->size()<<" sent "<<commrank<<endl;
      }
    }
  }


  
  vector<Determinant>& Vpsi0Dets = *uniqueDEH.Det;
  vector<CItype>& Vpsi0 = *uniqueDEH.Num;
  //cout << Vpsi0Dets.size()<<"  "<<Vpsi0.size()<<"  "<<commrank<<endl;
  boost::mpi::broadcast(world, Vpsi0Dets, 0);
  boost::mpi::broadcast(world, Vpsi0, 0);


  SHCImakeHamiltonian::HamHelpers2 helper2;
  SHCImakeHamiltonian::SparseHam sparseHam;

  if (commrank == 0) {
    helper2.PopulateHelpers(&Vpsi0Dets[0], Vpsi0Dets.size(), 0);
  }	
  helper2.MakeSHMHelpers();

  if (schd.DavidsonType != DIRECT)
    sparseHam.makeFromHelper(helper2, &Vpsi0Dets[0], 0, Vpsi0Dets.size(), norbs, I1, I2, coreE, false);
  
  std::vector<MatrixXx> Psi1(1, MatrixXx::Zero(Vpsi0Dets.size(), 1));
  MatrixXx Vpsi = MatrixXx::Zero(Vpsi0Dets.size(), 1);
  for (int i=0; i<Vpsi0Dets.size(); i++)
    Vpsi(i, 0) = Vpsi0[i]; 
  std::vector<CItype*> proj;
  Hmult2 H(sparseHam);
  LinearSolver(H, E0, Psi1[0], Vpsi, proj, 1.e-5, false);

  return 0.0;
}


void LCC::getDeterminantsLCC(Determinant& d, double epsilon, CItype ci1, CItype ci2, oneInt& int1, twoInt& int2, twoIntHeatBathSHM& I2hb, vector<int>& irreps, double coreE, double E0, std::vector<Determinant>& dets, std::vector<CItype>& numerator, std::vector<double>& energy, schedule& schd, int Nmc, int nelec) {

  int norbs = d.norbs;
  vector<int> closed(nelec,0);
  vector<int> open(norbs-nelec,0);
  d.getOpenClosed(open, closed);
  int nclosed = nelec;
  int nopen = norbs-nclosed;
  //d.getRepArray(detArray);

  double Energyd = d.Energy(int1, int2, coreE);

  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    //CItype integral = d.Hij_1Excite(closed[i],open[a],int1,int2);
    if (closed[i]%2 != open[a]%2 || irreps[closed[i]/2] != irreps[open[a]/2]) continue;
    CItype integral = Hij_1Excite(open[a],closed[i],int1,int2, &closed[0], nclosed);

    if (fabs(integral) > epsilon ) {
      dets.push_back(d); Determinant& di = *dets.rbegin();
      di.setocc(open[a], true); di.setocc(closed[i],false);

      double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, open[a], Energyd);

      numerator.push_back(integral*ci1);
      energy.push_back(E);
    }
  }

  if (fabs(int2.maxEntry) <epsilon) return;

  //#pragma omp parallel for schedule(dynamic)
  for (int ij=0; ij<nclosed*nclosed; ij++) {

    int i=ij/nclosed, j = ij%nclosed;
    if (i<=j) continue;
    int I = closed[i]/2, J = closed[j]/2;
    int X = max(I, J), Y = min(I, J);

    int pairIndex = X*(X+1)/2+Y;
    size_t start = closed[i]%2==closed[j]%2 ? I2hb.startingIndicesSameSpin[pairIndex] : I2hb.startingIndicesOppositeSpin[pairIndex];
    size_t end = closed[i]%2==closed[j]%2 ? I2hb.startingIndicesSameSpin[pairIndex+1] : I2hb.startingIndicesOppositeSpin[pairIndex+1];
    float* integrals = closed[i]%2==closed[j]%2 ?  I2hb.sameSpinIntegrals : I2hb.oppositeSpinIntegrals;
    short* orbIndices = closed[i]%2==closed[j]%2 ?  I2hb.sameSpinPairs : I2hb.oppositeSpinPairs;


    for (size_t index=start; index<end; index++) {
      if (fabs(integrals[index]) <epsilon) break;
      int a = 2* orbIndices[2*index] + closed[i]%2, b= 2*orbIndices[2*index+1]+closed[j]%2;

      if (!(d.getocc(a) || d.getocc(b))) {
	dets.push_back(d);
	Determinant& di = *dets.rbegin();
	di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);

	double sgn = 1.0;
	di.parity(a, b, closed[i], closed[j], sgn);
	numerator.push_back(integrals[index]*sgn*ci1);

	double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, a, j, b, Energyd);
	energy.push_back(E);
      }
    }
  }
  return;
}
