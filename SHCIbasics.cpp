/* Developed by Sandeep Sharma with contributions from James E. Smith and Adam A. Homes, 2017
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
#include "SHCIshm.h"

#include "communicate.h"
#include "omp.h"

using namespace std;
using namespace Eigen;
using namespace boost;
using namespace SHCISortMpiUtils;


double SHCIbasics::DoPerturbativeStochastic2SingleListDoubleEpsilon2AllTogether(Determinant *Dets, CItype *ci, int DetsSize, double& E0, oneInt& I1, twoInt& I2,
									  twoIntHeatBathSHM& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec, int root) {

  if (schd.nPTiter == 0) return 0;
  pout << "Peforming semistochastiPT for state: "<<root<<endl;

  double epsilon2 = schd.epsilon2;
  schd.epsilon2 = schd.epsilon2Large;
  vector<MatrixXx> vdVector;
  double Psi1Norm;
  double EptLarge = DoPerturbativeDeterministic(Dets, ci, DetsSize, E0, I1, I2, I2HB, irrep, schd, coreE, nelec, root,  vdVector, Psi1Norm);

  schd.epsilon2 = epsilon2;

  int norbs = Determinant::norbs;
  Determinant* SortedDets;
  std::vector<Determinant> SortedDetsvec;
  if (commrank == 0 ) {  
    for (int i=0; i<DetsSize; i++)
      SortedDetsvec.push_back(Dets[i]);
    std::sort(SortedDetsvec.begin(), SortedDetsvec.end());
  }
  SHMVecFromVecs(SortedDetsvec, SortedDets, shciSortedDets, SortedDetsSegment, regionSortedDets);
  SortedDetsvec.clear();

  int niter = schd.nPTiter;
  //double eps = 0.001;
  int Nsample = schd.SampleN;
  double AvgenergyEN = 0.0, AvgenergyEN2=0.0, stddev=0.0;
  double AverageDen = 0.0;
  int currentIter = 0;
  int sampleSize = 0;
  int num_thrds = 1;

  double cumulative = 0.0;
  for (int i=0; i<DetsSize; i++)
    cumulative += abs(ci[i]);

  std::vector<int> alias; std::vector<double> prob;
  if (commrank == 0)
    SHCIsampledeterminants::setUpAliasMethod(ci, DetsSize, cumulative, alias, prob);


  StitchDEH uniqueDEH;
  double totalPT = 0.0;
  double totalPTLargeEps=0;
  size_t ntries = 0;
  int AllDistinctSample = 0;
  size_t Nmc = commsize*num_thrds*Nsample;
  std::vector<int> allSample(Nmc, -1);
  std::vector<CItype> allwts(Nmc, 0.);
  pout <<endl<<format("%6s  %14s  %5s %14s %10s  %10s")
    %("Iter") % ("EPTcurrent") %("State") %("EPTavg") %("Error")%("Time(s)")<<endl;

  int size = commsize, rank = commrank;

  for (int iter=0; iter<niter; iter++) {
    std::vector<CItype> wts1(Nsample,0.0); std::vector<int> Sample1(Nsample,-1);
    int ithrd = 0;
    vector<size_t> all_to_all(size*size,0);
    
    if (commrank == 0) {
      std::fill(allSample.begin(), allSample.end(), -1);
      AllDistinctSample = SHCIsampledeterminants::sample_N2_alias(ci, DetsSize, cumulative, allSample, allwts, alias, prob);
    }

#ifndef SERIAL
    MPI_Bcast(&allSample[0], allSample.size(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&AllDistinctSample, 1, MPI_INT, 0, MPI_COMM_WORLD);
#ifndef Complex
    MPI_Bcast(&allwts[0], allwts.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
    MPI_Bcast(&allwts[0], 2*allwts.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
#endif


    int distinctSample = 0;
    for (int i = 0; i < AllDistinctSample; i++) {
      if ( i% size != rank) continue;
      wts1   [distinctSample] = allwts   [i];
      Sample1[distinctSample] = allSample[i];
      distinctSample++;
    }
    double norm = 0.0;
    
    for (int i=0; i<distinctSample; i++) {
      int I = Sample1[i];
      SHCIgetdeterminants::getDeterminantsStochastic2Epsilon(Dets[I], schd.epsilon2/abs(ci[I]),
							     schd.epsilon2Large/abs(ci[I]), wts1[i],
							     ci[I], I1, I2, I2HB, irrep, coreE, E0,
							     *uniqueDEH.Det,
							     *uniqueDEH.Num,
							     *uniqueDEH.Num2,
							     *uniqueDEH.present,
							     *uniqueDEH.Energy,
							     schd, Nmc, nelec);
      }
    
      if(commsize >1 ) {

	boost::shared_ptr<vector<Determinant> >& Det = uniqueDEH.Det;
	boost::shared_ptr<vector<CItype> >& Num = uniqueDEH.Num;
	boost::shared_ptr<vector<CItype> >& Num2 = uniqueDEH.Num2;
	boost::shared_ptr<vector<double> >& Energy = uniqueDEH.Energy;
	boost::shared_ptr<vector<char> >& present = uniqueDEH.present;
	
	std::vector<size_t> hashValues(Det->size());
	
	std::vector<size_t> all_to_all_cumulative(size,0);
	for (int i=0; i<Det->size(); i++) {
	  hashValues[i] = Det->at(i).getHash();
	  all_to_all[rank*size+hashValues[i]%size]++; 
	}
	for (int i=0; i<size; i++)
	  all_to_all_cumulative[i] = i == 0 ? all_to_all[rank*size+i] :  all_to_all_cumulative[i-1]+all_to_all[rank*size+i];
	
	vector<Determinant> atoaDets(Det->size());
	vector<CItype> atoaNum(Det->size());
	vector<CItype> atoaNum2(Det->size());
	vector<double> atoaE(Det->size());
	vector<char> atoaPresent(Det->size());
	
	
	vector<size_t> all_to_allCopy = all_to_all;
	MPI_Allreduce( &all_to_allCopy[0], &all_to_all[0], 2*size*size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	
	vector<size_t> counter(size, 0);
	for (int i=0; i<Det->size(); i++) {
	  int toProc = hashValues[i]%size;
	  size_t index = toProc==0 ? counter[0] : counter[toProc] + all_to_all_cumulative[toProc-1];
	  
	  atoaDets[index ] = Det->at(i);
	  atoaNum[index] = Num->at(i);
	  atoaNum2[index] = Num2->at(i);
	  atoaE[index] = Energy->at(i);
	  atoaPresent[index] = present->at(i);
	  
	  counter[toProc]++;
	}
	
	
	
	
	vector<int> sendcts(size,0), senddisp(size,0), recvcts(size,0), recvdisp(size,0);
	vector<int> sendctsDets(size,0), senddispDets(size,0), recvctsDets(size,0), recvdispDets(size,0);
	vector<int> sendctsPresent(size,0), senddispPresent(size,0), recvctsPresent(size,0), recvdispPresent(size,0);
	
	size_t recvSize = 0;
	for (int i=0; i<size; i++) {
	  sendcts[i] = all_to_all[rank*size+i]* sizeof(CItype)/sizeof(double);
	  senddisp[i] = i==0? 0 : senddisp[i-1]+sendcts[i-1];
	  recvcts[i] = all_to_all[i*size+rank]*sizeof(CItype)/sizeof(double);
	  recvdisp[i] = i==0? 0 : recvdisp[i-1]+recvcts[i-1];
	  
	  sendctsDets[i] = all_to_all[rank*size+i]* sizeof(Determinant)/sizeof(double);
	  senddispDets[i] = i==0? 0 : senddispDets[i-1]+sendctsDets[i-1];
	  recvctsDets[i] = all_to_all[i*size+rank]*sizeof(Determinant)/sizeof(double);
	  recvdispDets[i] = i==0? 0 : recvdispDets[i-1]+recvctsDets[i-1];
	  
	  sendctsPresent[i] = all_to_all[rank*size+i];
	  senddispPresent[i] = i==0? 0 : senddispPresent[i-1]+sendctsPresent[i-1];
	  recvctsPresent[i] = all_to_all[i*size+rank];
	  recvdispPresent[i] = i==0? 0 : recvdispPresent[i-1]+recvctsPresent[i-1];
	  
	  recvSize += all_to_all[i*size+rank];
	}
	
	Det->resize(recvSize), Num->resize(recvSize), Energy->resize(recvSize);
	Num2->resize(recvSize), present->resize(recvSize);
	
	MPI_Alltoallv(&atoaNum.at(0), &sendcts[0], &senddisp[0], MPI_DOUBLE, &Num->at(0), &recvcts[0], &recvdisp[0], MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Alltoallv(&atoaNum2.at(0), &sendcts[0], &senddisp[0], MPI_DOUBLE, &Num2->at(0), &recvcts[0], &recvdisp[0], MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Alltoallv(&atoaE.at(0), &sendctsPresent[0], &senddispPresent[0], MPI_DOUBLE, &Energy->at(0), &recvctsPresent[0], &recvdispPresent[0], MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Alltoallv(&atoaPresent.at(0), &sendctsPresent[0], &senddispPresent[0], MPI_CHAR, &present->at(0), &recvctsPresent[0], &recvdispPresent[0], MPI_CHAR, MPI_COMM_WORLD);
	MPI_Alltoallv(&atoaDets.at(0).repr[0], &sendctsDets[0], &senddispDets[0], MPI_DOUBLE, &(Det->at(0).repr[0]), &recvctsDets[0], &recvdispDets[0], MPI_DOUBLE, MPI_COMM_WORLD);
	
      }
      uniqueDEH.MergeSort();
      
      
      double energyEN = 0.0, energyENLargeEps = 0.0;
      
      vector<Determinant>& Psi1 = *uniqueDEH.Det;
      vector<CItype>& numerator1A = *uniqueDEH.Num;
      vector<CItype>& numerator2A = *uniqueDEH.Num2;
      vector<char>& present = *uniqueDEH.present;
      vector<double>& det_energy = *uniqueDEH.Energy;
      
      CItype currentNum1A=0.; CItype currentNum2A=0.;
      CItype currentNum1B=0.; CItype currentNum2B=0.;
      int vec_it = 0;
      
      for (int i=0;i<Psi1.size();) {
	if (Psi1[i] < SortedDets[vec_it]) {
	  currentNum1A += numerator1A[i];
	  currentNum2A += numerator2A[i];
	  if (present[i]) {
	    currentNum1B += numerator1A[i];
	    currentNum2B += numerator2A[i];
	  }
	  
	  if ( i == Psi1.size()-1) {
#ifndef Complex
	    energyEN += (pow(abs(currentNum1A),2)*Nmc/(Nmc-1) - currentNum2A)/(det_energy[i] - E0);
	    energyENLargeEps += (pow(abs(currentNum1B),2)*Nmc/(Nmc-1) - currentNum2B)/(det_energy[i] - E0);
#else
	    energyEN += (pow(abs(currentNum1A),2)*Nmc/(Nmc-1) - currentNum2A.real())/(det_energy[i] - E0);
	    energyENLargeEps += (pow(abs(currentNum1B),2)*Nmc/(Nmc-1) - currentNum2B.real())/(det_energy[i] - E0);
#endif
	  }
	  else if (!(Psi1[i] == Psi1[i+1])) {
#ifndef Complex
	    energyEN += (pow(abs(currentNum1A),2)*Nmc/(Nmc-1) - currentNum2A)/(det_energy[i] - E0);
	    energyENLargeEps += (pow(abs(currentNum1B),2)*Nmc/(Nmc-1) - currentNum2B)/(det_energy[i] - E0);
#else
	    energyEN += (pow(abs(currentNum1A),2)*Nmc/(Nmc-1) - currentNum2A.real())/(det_energy[i] - E0);
	    energyENLargeEps += (pow(abs(currentNum1B),2)*Nmc/(Nmc-1) - currentNum2B.real())/(det_energy[i] - E0);
#endif
	    currentNum1A = 0.;
	    currentNum2A = 0.;
	    currentNum1B = 0.;
	    currentNum2B = 0.;
	  }
	  i++;
	}
	else if (SortedDets[vec_it] <Psi1[i] && vec_it != DetsSize)
	  vec_it++;
	else if (SortedDets[vec_it] <Psi1[i] && vec_it == DetsSize) {
	  currentNum1A += numerator1A[i];
	  currentNum2A += numerator2A[i];
	  if (present[i]) {
	    currentNum1B += numerator1A[i];
	    currentNum2B += numerator2A[i];
	  }

	  if ( i == Psi1.size()-1) {
#ifndef Complex
	    energyEN += (pow(abs(currentNum1A),2)*Nmc/(Nmc-1) - currentNum2A)/(det_energy[i] - E0);
	    energyENLargeEps += (pow(abs(currentNum1B),2)*Nmc/(Nmc-1) - currentNum2B)/(det_energy[i] - E0);
#else
	    energyEN += (pow(abs(currentNum1A),2)*Nmc/(Nmc-1) - currentNum2A.real())/(det_energy[i] - E0);
	    energyENLargeEps += (pow(abs(currentNum1B),2)*Nmc/(Nmc-1) - currentNum2B.real())/(det_energy[i] - E0);
#endif
	  }
	  else if (!(Psi1[i] == Psi1[i+1])) {
#ifndef Complex
	    energyEN += (pow(abs(currentNum1A),2)*Nmc/(Nmc-1) - currentNum2A)/(det_energy[i] - E0);
	    energyENLargeEps += (pow(abs(currentNum1B),2)*Nmc/(Nmc-1) - currentNum2B)/(det_energy[i] - E0);
#else
	    energyEN += (pow(abs(currentNum1A),2)*Nmc/(Nmc-1) - currentNum2A.real())/(det_energy[i] - E0);
	    energyENLargeEps += (pow(abs(currentNum1B),2)*Nmc/(Nmc-1) - currentNum2B.real())/(det_energy[i] - E0);
#endif
	    currentNum1A = 0.;
	    currentNum2A = 0.;
	    currentNum1B = 0.;
	    currentNum2B = 0.;
	  }
	  i++;
	}
	else {
	  if (Psi1[i] == Psi1[i+1])
	    i++;
	  else {
	    vec_it++; i++;
	  }
	}
      }
      
      totalPT=0; totalPTLargeEps=0;
      
      totalPT += energyEN;
      totalPTLargeEps += energyENLargeEps;
      
      
      double finalE = 0., finalELargeEps=0;
#ifndef SERIAL
      MPI_Allreduce(&totalPT, &finalE, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&totalPTLargeEps, &finalELargeEps, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
      finalE = totalPT;
      finalELargeEps = totalPTLargeEps;
#endif

      if (commrank == 0 ) {
	currentIter++;
	AvgenergyEN += -finalE+finalELargeEps+EptLarge;
	AvgenergyEN2 += pow(-finalE+finalELargeEps+EptLarge,2);
	stddev = currentIter < 5 ? 1e4 : pow( (currentIter*AvgenergyEN2 - pow(AvgenergyEN,2))/currentIter/(currentIter-1)/currentIter, 0.5);
	if (currentIter < 5)
	  std::cout << format("%6i  %14.8f  %5i %14.8f %10s  %10.2f")
	    %(currentIter) % (E0-finalE+finalELargeEps+EptLarge) %(root) %(E0+AvgenergyEN/currentIter) %"--" %(getTime()-startofCalc) ;
	else
	  std::cout << format("%6i  %14.8f  %5i %14.8f %10.2e  %10.2f")
	    %(currentIter) % (E0-finalE+finalELargeEps+EptLarge) %(root) %(E0+AvgenergyEN/currentIter) %stddev %(getTime()-startofCalc) ;
	pout << endl;
      }
      
#ifndef SERIAL
      MPI_Bcast(&currentIter, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&AvgenergyEN, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
      
      uniqueDEH.clear();
      if (stddev < schd.targetError) {
	AvgenergyEN /= currentIter;
	//pout << "Standard Error : "<<stddev<<" less than "<<schd.targetError<<endl;
	pout << "Semistochastic PT calculation converged"<<endl; 
	pout << "epsilon2: "<<schd.epsilon2<<endl<<"PTEnergy: "<<E0+AvgenergyEN<<" +/- "<<format("%8.2e") %(stddev)<<endl<<"Time(s):  "<<getTime()-startofCalc<<endl;

	break;
      }
  }
  return AvgenergyEN;
}


double SHCIbasics::DoPerturbativeDeterministic(Determinant* Dets, CItype* ci, int DetsSize, double& E0, oneInt& I1, twoInt& I2,
					       twoIntHeatBathSHM& I2HB, vector<int>& irrep, schedule& schd, double coreE,
					       int nelec, int root,  vector<MatrixXx>& vdVector, double& Psi1Norm,
					       bool appendPsi1ToPsi0) {

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
  SHMVecFromVecs(SortedDetsvec, SortedDets, shciSortedDets, SortedDetsSegment, regionSortedDets);
  SortedDetsvec.clear();

  double energyEN = 0.0;
  double Psi1NormProc = 0.0;

  StitchDEH uniqueDEH;
  double totalPT = 0.0;
  int ntries = 0;

  int size = commsize, rank = commrank;
  vector<size_t> all_to_all(size*size,0);

  if (schd.DoRDM || schd.doResponse) {
    uniqueDEH.extra_info = true;
    for (int i=0; i<DetsSize; i++) {
      if (i%size != rank) continue;
      SHCIgetdeterminants::getDeterminantsDeterministicPTKeepRefDets(Dets[i], i, abs(schd.epsilon2/ci[i]), ci[i],
								     I1, I2, I2HB, irrep, coreE, E0,
								     *uniqueDEH.Det,
								     *uniqueDEH.Num,
								     *uniqueDEH.Energy,
								     *uniqueDEH.var_indices_beforeMerge,
								     *uniqueDEH.orbDifference_beforeMerge,
								     schd, nelec);
    }
  }
  else {
    for (int i=0; i<DetsSize; i++) {
      if ((i%size != rank)) continue;
      
      SHCIgetdeterminants::getDeterminantsDeterministicPT(Dets[i], abs(schd.epsilon2/ci[i]), ci[i], 0.0,
							  I1, I2, I2HB, irrep, coreE, E0,
							  *uniqueDEH.Det,
							  *uniqueDEH.Num,
							  *uniqueDEH.Energy,
							  schd,0, nelec);
      //if (i%100000 == 0 && omp_get_thread_num()==0 && commrank == 0) pout <<"# "<<i<<endl;
    }
  }
  
  
  if(commsize >1 ) {
    boost::shared_ptr<vector<Determinant> >& Det = uniqueDEH.Det;
    boost::shared_ptr<vector<CItype> >& Num = uniqueDEH.Num;
    boost::shared_ptr<vector<double> >& Energy = uniqueDEH.Energy;
    boost::shared_ptr<vector<int > >& var_indices = uniqueDEH.var_indices_beforeMerge;
    boost::shared_ptr<vector<size_t > >& orbDifference = uniqueDEH.orbDifference_beforeMerge;
    
    std::vector<size_t> hashValues(Det->size());
    
    std::vector<size_t> all_to_all_cumulative(size,0);
    for (int i=0; i<Det->size(); i++) {
      hashValues[i] = Det->at(i).getHash();
      all_to_all[rank*size+hashValues[i]%size]++; 
    }
    for (int i=0; i<size; i++)
      all_to_all_cumulative[i] = i == 0 ? all_to_all[rank*size+i] :  all_to_all_cumulative[i-1]+all_to_all[rank*size+i];
    
    vector<Determinant> atoaDets(Det->size());
    vector<CItype> atoaNum(Det->size());
    vector<double> atoaE(Det->size());
    vector<int > atoaVarIndices;
    vector<size_t > atoaOrbDiff;
    if (schd.DoRDM || schd.doResponse) {
      atoaVarIndices.resize(Det->size()); atoaOrbDiff.resize(Det->size());
    }
    
    
    vector<size_t> all_to_allCopy = all_to_all;
    MPI_Allreduce( &all_to_allCopy[0], &all_to_all[0], 2*size*size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    vector<size_t> counter(size, 0);
    for (int i=0; i<Det->size(); i++) {
      int toProc = hashValues[i]%size;
      size_t index = toProc==0 ? counter[0] : counter[toProc] + all_to_all_cumulative[toProc-1];
      
      atoaDets[index ] = Det->at(i);
      atoaNum[index] = Num->at(i);
      atoaE[index] = Energy->at(i);
      if (schd.DoRDM||schd.doResponse) {
	atoaVarIndices[index] = var_indices->at(i);
	atoaOrbDiff[index] = orbDifference->at(i);
      }
      counter[toProc]++;
    }
    
    
    
    
    vector<int> sendcts(size,0), senddisp(size,0), recvcts(size,0), recvdisp(size,0);
    vector<int> sendctsDets(size,0), senddispDets(size,0), recvctsDets(size,0), recvdispDets(size,0);
    vector<int> sendctsVarDiff(size,0), senddispVarDiff(size,0), recvctsVarDiff(size,0), recvdispVarDiff(size,0);
    
    size_t recvSize = 0;
    for (int i=0; i<size; i++) {
      sendcts[i] = all_to_all[rank*size+i]* sizeof(CItype)/sizeof(double);
      senddisp[i] = i==0? 0 : senddisp[i-1]+sendcts[i-1];
      recvcts[i] = all_to_all[i*size+rank]*sizeof(CItype)/sizeof(double);
      recvdisp[i] = i==0? 0 : recvdisp[i-1]+recvcts[i-1];
      
      sendctsDets[i] = all_to_all[rank*size+i]* sizeof(Determinant)/sizeof(double);
      senddispDets[i] = i==0? 0 : senddispDets[i-1]+sendctsDets[i-1];
      recvctsDets[i] = all_to_all[i*size+rank]*sizeof(Determinant)/sizeof(double);
      recvdispDets[i] = i==0? 0 : recvdispDets[i-1]+recvctsDets[i-1];
      
      sendctsVarDiff[i] = all_to_all[rank*size+i];
      senddispVarDiff[i] = i==0? 0 : senddispVarDiff[i-1]+sendctsVarDiff[i-1];
      recvctsVarDiff[i] = all_to_all[i*size+rank];
      recvdispVarDiff[i] = i==0? 0 : recvdispVarDiff[i-1]+recvctsVarDiff[i-1];
      
      recvSize += all_to_all[i*size+rank];
    }
    
    Det->resize(recvSize), Num->resize(recvSize), Energy->resize(recvSize);
    if (schd.DoRDM||schd.doResponse) {
      var_indices->resize(recvSize);
      orbDifference->resize(recvSize);
    }
    
    MPI_Alltoallv(&atoaNum.at(0), &sendcts[0], &senddisp[0], MPI_DOUBLE, &Num->at(0), &recvcts[0], &recvdisp[0], MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Alltoallv(&atoaE.at(0), &sendctsVarDiff[0], &senddispVarDiff[0], MPI_DOUBLE, &Energy->at(0), &recvctsVarDiff[0], &recvdispVarDiff[0], MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Alltoallv(&atoaDets.at(0).repr[0], &sendctsDets[0], &senddispDets[0], MPI_DOUBLE, &(Det->at(0).repr[0]), &recvctsDets[0], &recvdispDets[0], MPI_DOUBLE, MPI_COMM_WORLD);
    
    if (schd.DoRDM || schd.doResponse) {
      MPI_Alltoallv(&atoaVarIndices.at(0), &sendctsVarDiff[0], &senddispVarDiff[0], MPI_INT, &(var_indices->at(0)), &recvctsVarDiff[0], &recvdispVarDiff[0], MPI_INT, MPI_COMM_WORLD);
      MPI_Alltoallv(&atoaOrbDiff.at(0), &sendctsVarDiff[0], &senddispVarDiff[0], MPI_DOUBLE, &(orbDifference->at(0)), &recvctsVarDiff[0], &recvdispVarDiff[0], MPI_DOUBLE, MPI_COMM_WORLD);
    }
    uniqueDEH.Num2->clear();
    
  }
  uniqueDEH.MergeSortAndRemoveDuplicates();
  uniqueDEH.RemoveDetsPresentIn(SortedDets, DetsSize);
  
  
  vector<Determinant>& hasHEDDets = *uniqueDEH.Det;
  vector<CItype>& hasHEDNumerator = *uniqueDEH.Num;
  vector<double>& hasHEDEnergy    = *uniqueDEH.Energy;
  
  double PTEnergy = 0.0;
  double psi1normthrd=0.0;
  for (size_t i=0; i<hasHEDDets.size();i++) {
    psi1normthrd += pow(abs(hasHEDNumerator[i]/(E0-hasHEDEnergy[i])),2);
    PTEnergy += pow(abs(hasHEDNumerator[i]),2)/(E0-hasHEDEnergy[i]);
  }

  Psi1NormProc += psi1normthrd;
  totalPT += PTEnergy;

  


  double finalE = 0.;
#ifndef SERIAL
  mpi::all_reduce(world, totalPT, finalE, std::plus<double>());
  mpi::all_reduce(world, Psi1NormProc, Psi1Norm, std::plus<double>());
#else
  finalE = totalPT;
#endif
  
  if (commrank == 0) {
    pout << "Deterministic PT calculation converged"<<endl; 
    pout << "epsilon2: "<<schd.epsilon2<<endl<<"PTEnergy: "<<E0+finalE<<endl<<"Time(s):  "<<getTime()-startofCalc<<endl;
  }
  
  if (schd.doResponse || schd.DoRDM) { //build RHS for the lambda equation
    pout << "Now calculating PT RDM"<<endl;
    MatrixXx s2RDM, twoRDM;
    SHCIrdm::loadRDM(schd, s2RDM, twoRDM, root);
#ifndef SERIAL
    mpi::broadcast(world, s2RDM, 0);
    if (schd.DoSpinRDM)
      mpi::broadcast(world, twoRDM, 0);
#endif
    if (commrank != 0) {
      s2RDM = 0.*s2RDM;
      twoRDM = 0.*twoRDM;
    }
    SHCIrdm::UpdateRDMResponsePerturbativeDeterministic(Dets, DetsSize, ci, E0, I1, I2, schd, coreE,
							nelec, norbs, uniqueDEH, root, Psi1Norm,
							s2RDM, twoRDM);
    SHCIrdm::saveRDM(schd, s2RDM, twoRDM, root);

    //construct the vector Via x da
    //where Via is the perturbation matrix element
    //da are the elements of the PT wavefunctions
    vdVector[root]= MatrixXx::Zero(DetsSize,1);

    vector<Determinant>& uniqueDets = *uniqueDEH.Det;
    vector<double>& uniqueEnergy = *uniqueDEH.Energy;
    vector<CItype>& uniqueNumerator = *uniqueDEH.Num;
    vector<vector<int> >& uniqueVarIndices = *uniqueDEH.var_indices;
    vector<vector<size_t> >& uniqueOrbDiff = *uniqueDEH.orbDifference;
    for (int a=0; a<uniqueDets.size(); a++) {
      CItype da = uniqueNumerator[a]/(E0-uniqueEnergy[a]); //coefficient for det a
      for (int i=0; i<uniqueVarIndices[a].size(); i++) {
	int I = uniqueVarIndices[a][i]; //index of the Var determinant
	size_t orbDiff;
	vdVector[root](I,0) -= conj(da)*Hij(uniqueDets[a], Dets[I], I1, I2, coreE, orbDiff);
      }
    }
    
#ifndef SERIAL
    MPI_Allreduce(MPI_IN_PLACE, &vdVector[root](0,0), vdVector[root].rows(),
		  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  }

  return finalE;
}


void SHCIbasics::DoPerturbativeDeterministicOffdiagonal(vector<Determinant>& Dets, MatrixXx& ci1, double& E01,
							MatrixXx&ci2, double& E02, oneInt& I1, twoInt& I2,
							twoIntHeatBathSHM& I2HB, vector<int>& irrep,
							schedule& schd, double coreE, int nelec, int root,
							CItype& EPT1, CItype& EPT2, CItype& EPT12,
							std::vector<MatrixXx>& spinRDM) {

#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  int norbs = Determinant::norbs;
  std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());

  double energyEN = 0.0;
  int num_thrds = omp_get_max_threads();

  std::vector<StitchDEH> uniqueDEH(num_thrds);
  std::vector<std::vector< std::vector<vector<Determinant> > > > hashedDetBeforeMPI(commsize, std::vector<std::vector<vector<Determinant> > >(num_thrds));
  std::vector<std::vector< std::vector<vector<Determinant> > > > hashedDetAfterMPI(commsize, std::vector<std::vector<vector<Determinant> > >(num_thrds));
  std::vector<std::vector< std::vector<vector<CItype> > > > hashedNumBeforeMPI(commsize, std::vector<std::vector<vector<CItype> > >(num_thrds));
  std::vector<std::vector< std::vector<vector<CItype> > > > hashedNumAfterMPI(commsize, std::vector<std::vector<vector<CItype> > >(num_thrds));
  std::vector<std::vector< std::vector<vector<CItype> > > > hashedNum2BeforeMPI(commsize, std::vector<std::vector<vector<CItype> > >(num_thrds));
  std::vector<std::vector< std::vector<vector<CItype> > > > hashedNum2AfterMPI(commsize, std::vector<std::vector<vector<CItype> > >(num_thrds));
  std::vector<std::vector< std::vector<vector<double> > > > hashedEnergyBeforeMPI(commsize, std::vector<std::vector<vector<double> > >(num_thrds));
  std::vector<std::vector< std::vector<vector<double> > > > hashedEnergyAfterMPI(commsize, std::vector<std::vector<vector<double> > >(num_thrds));
  CItype totalPT1 = 0.0, totalPT2=0., totalPT12=0.;
  int ntries = 1;

#pragma omp parallel
  {

    for (int i=0; i<Dets.size(); i++) {
      if (i%(omp_get_num_threads()*commsize) != commrank*omp_get_num_threads()+omp_get_thread_num()) {continue;}
      SHCIgetdeterminants::getDeterminantsDeterministicPTWithSOC(Dets[i], i, abs(schd.epsilon2/ci1(i,0)), ci1(i,0),
						      abs(schd.epsilon2/ci2(i,0)), ci2(i,0),
						      I1, I2, I2HB, irrep, coreE,
						      *uniqueDEH[omp_get_thread_num()].Det,
						      *uniqueDEH[omp_get_thread_num()].Num,
						      *uniqueDEH[omp_get_thread_num()].Num2,
						      *uniqueDEH[omp_get_thread_num()].Energy,
						      schd, nelec);
    }

    uniqueDEH[omp_get_thread_num()].MergeSortAndRemoveDuplicates();
    uniqueDEH[omp_get_thread_num()].RemoveDetsPresentIn(SortedDets);

    if(commsize >1 || num_thrds >1) {
      StitchDEH uniqueDEH_afterMPI;
      if (schd.DoRDM || schd.doResponse) uniqueDEH_afterMPI.extra_info = true;


      for (int proc=0; proc<commsize; proc++) {
	hashedDetBeforeMPI[proc][omp_get_thread_num()].resize(num_thrds);
	hashedNumBeforeMPI[proc][omp_get_thread_num()].resize(num_thrds);
	hashedNum2BeforeMPI[proc][omp_get_thread_num()].resize(num_thrds);
	hashedEnergyBeforeMPI[proc][omp_get_thread_num()].resize(num_thrds);
      }

      if (omp_get_thread_num()==0) {
	ntries = uniqueDEH[omp_get_thread_num()].Det->size()*DetLen*2*omp_get_num_threads()/268435400+1;
	if (commsize == 1)
	  ntries = 1;
#ifndef SERIAL
	mpi::broadcast(world, ntries, 0);
#endif
      }
#pragma omp barrier

      size_t batchsize = uniqueDEH[omp_get_thread_num()].Det->size()/ntries;
      //ntries = 1;
      for (int tries = 0; tries<ntries; tries++) {

        size_t start = (ntries-1-tries)*batchsize;
        size_t end   = tries==0 ? uniqueDEH[omp_get_thread_num()].Det->size() : (ntries-tries)*batchsize;
	for (size_t j=start; j<end; j++) {
	  size_t lOrder = uniqueDEH[omp_get_thread_num()].Det->at(j).getHash();
	  size_t procThrd = lOrder%(commsize*num_thrds);
	  int proc = abs(procThrd/num_thrds), thrd = abs(procThrd%num_thrds);
	  hashedDetBeforeMPI[proc][omp_get_thread_num()][thrd].push_back(uniqueDEH[omp_get_thread_num()].Det->at(j));
	  hashedNumBeforeMPI[proc][omp_get_thread_num()][thrd].push_back(uniqueDEH[omp_get_thread_num()].Num->at(j));
	  hashedNum2BeforeMPI[proc][omp_get_thread_num()][thrd].push_back(uniqueDEH[omp_get_thread_num()].Num2->at(j));
	  hashedEnergyBeforeMPI[proc][omp_get_thread_num()][thrd].push_back(uniqueDEH[omp_get_thread_num()].Energy->at(j));
	}

	uniqueDEH[omp_get_thread_num()].resize(start);


#pragma omp barrier
	if (omp_get_thread_num()==num_thrds-1) {
#ifndef SERIAL
	  mpi::all_to_all(world, hashedDetBeforeMPI, hashedDetAfterMPI);
	  mpi::all_to_all(world, hashedNumBeforeMPI, hashedNumAfterMPI);
	  mpi::all_to_all(world, hashedNum2BeforeMPI, hashedNum2AfterMPI);
	  mpi::all_to_all(world, hashedEnergyBeforeMPI, hashedEnergyAfterMPI);
#else
	  hashedDetAfterMPI = hashedDetBeforeMPI;
	  hashedNumAfterMPI = hashedNumBeforeMPI;
	  hashedNum2AfterMPI = hashedNum2BeforeMPI;
	  //hashedpresentAfterMPI = hashedpresentBeforeMPI;
	  hashedEnergyAfterMPI = hashedEnergyBeforeMPI;
#endif
	}
#pragma omp barrier

	for (int proc=0; proc<commsize; proc++) {
	  for (int thrd=0; thrd<num_thrds; thrd++) {
	    hashedDetBeforeMPI[proc][thrd][omp_get_thread_num()].clear();
	    hashedNumBeforeMPI[proc][thrd][omp_get_thread_num()].clear();
	    hashedNum2BeforeMPI[proc][thrd][omp_get_thread_num()].clear();
	    hashedEnergyBeforeMPI[proc][thrd][omp_get_thread_num()].clear();
	  }
	}



	for (int proc=0; proc<commsize; proc++) {
	  for (int thrd=0; thrd<num_thrds; thrd++) {

	    for (int j=0; j<hashedDetAfterMPI[proc][thrd][omp_get_thread_num()].size(); j++) {
	      uniqueDEH_afterMPI.Det->push_back(hashedDetAfterMPI[proc][thrd][omp_get_thread_num()].at(j));
	      uniqueDEH_afterMPI.Num->push_back(hashedNumAfterMPI[proc][thrd][omp_get_thread_num()].at(j));
	      uniqueDEH_afterMPI.Num2->push_back(hashedNum2AfterMPI[proc][thrd][omp_get_thread_num()].at(j));
	      uniqueDEH_afterMPI.Energy->push_back(hashedEnergyAfterMPI[proc][thrd][omp_get_thread_num()].at(j));
	    }
	    hashedDetAfterMPI[proc][thrd][omp_get_thread_num()].clear();
	    hashedNumAfterMPI[proc][thrd][omp_get_thread_num()].clear();
	    hashedNum2AfterMPI[proc][thrd][omp_get_thread_num()].clear();
	    hashedEnergyAfterMPI[proc][thrd][omp_get_thread_num()].clear();
	  }
	}
      }


      *uniqueDEH[omp_get_thread_num()].Det = *uniqueDEH_afterMPI.Det;
      *uniqueDEH[omp_get_thread_num()].Num = *uniqueDEH_afterMPI.Num;
      *uniqueDEH[omp_get_thread_num()].Num2 = *uniqueDEH_afterMPI.Num2;
      *uniqueDEH[omp_get_thread_num()].Energy = *uniqueDEH_afterMPI.Energy;
      uniqueDEH_afterMPI.clear();
      uniqueDEH[omp_get_thread_num()].MergeSortAndRemoveDuplicates();


    }

    vector<Determinant>& hasHEDDets = *uniqueDEH[omp_get_thread_num()].Det;
    vector<CItype>& hasHEDNumerator = *uniqueDEH[omp_get_thread_num()].Num;
    vector<CItype>& hasHEDNumerator2 = *uniqueDEH[omp_get_thread_num()].Num2;
    vector<double>& hasHEDEnergy = *uniqueDEH[omp_get_thread_num()].Energy;

    CItype PTEnergy1 = 0.0, PTEnergy2 = 0.0, PTEnergy12 = 0.0;

    for (size_t i=0; i<hasHEDDets.size();i++) {
      PTEnergy1 += pow(abs(hasHEDNumerator[i]),2)/(E01-hasHEDEnergy[i]);
      PTEnergy12 += 0.5*(conj(hasHEDNumerator[i])*hasHEDNumerator2[i]/(E01-hasHEDEnergy[i])+conj(hasHEDNumerator[i])*hasHEDNumerator2[i]/(E02-hasHEDEnergy[i]));
      PTEnergy2 += pow(abs(hasHEDNumerator2[i]),2)/(E02-hasHEDEnergy[i]);
    }
#pragma omp critical
    {
      totalPT1 += PTEnergy1;
      totalPT12 += PTEnergy12;
      totalPT2 += PTEnergy2;
    }

  }


  EPT1=0.0;EPT2=0.0;EPT12=0.0;
#ifndef SERIAL
  mpi::all_reduce(world, totalPT1, EPT1, std::plus<CItype>());
  mpi::all_reduce(world, totalPT2, EPT2, std::plus<CItype>());
  mpi::all_reduce(world, totalPT12, EPT12, std::plus<CItype>());
#else
  EPT1 = totalPT1;
  EPT2 = totalPT2;
  EPT12 = totalPT12;
#endif

  if (schd.doGtensor) {//DON'T PERFORM doGtensor

    if (commrank != 0) {
      spinRDM[0].setZero(spinRDM[0].rows(), spinRDM[0].cols());
      spinRDM[1].setZero(spinRDM[1].rows(), spinRDM[1].cols());
      spinRDM[2].setZero(spinRDM[2].rows(), spinRDM[2].cols());
    }

    vector< vector<MatrixXx> > spinRDM_thrd(num_thrds, vector<MatrixXx>(3));
#pragma omp parallel
    {
      for (int thrd=0; thrd<num_thrds; thrd++) {
	if (thrd != omp_get_thread_num()) continue;

	spinRDM_thrd[thrd][0].setZero(spinRDM[0].rows(), spinRDM[0].cols());
	spinRDM_thrd[thrd][1].setZero(spinRDM[1].rows(), spinRDM[1].cols());
	spinRDM_thrd[thrd][2].setZero(spinRDM[2].rows(), spinRDM[2].cols());

	vector<Determinant>& hasHEDDets = *uniqueDEH[thrd].Det;
	vector<CItype>& hasHEDNumerator = *uniqueDEH[thrd].Num;
	vector<CItype>& hasHEDNumerator2 = *uniqueDEH[thrd].Num2;
	vector<double>& hasHEDEnergy = *uniqueDEH[thrd].Energy;


	for (int x=0; x<Dets.size(); x++) {
	  Determinant& d = Dets[x];

	  vector<int> closed(nelec,0);
	  vector<int> open(norbs-nelec,0);
	  d.getOpenClosed(open, closed);
	  int nclosed = nelec;
	  int nopen = norbs-nclosed;


	  for (int ia=0; ia<nopen*nclosed; ia++){
	    int i=ia/nopen, a=ia%nopen;

	    Determinant di = d;
	    di.setocc(open[a], true); di.setocc(closed[i],false);


	    auto lower = std::lower_bound(hasHEDDets.begin(), hasHEDDets.end(), di);
	    //map<Determinant, int>::iterator it = SortedDets.find(di);
	    if (di == *lower ) {
	      double sgn = 1.0;
	      d.parity(min(open[a],closed[i]), max(open[a],closed[i]),sgn);
	      int y = distance(hasHEDDets.begin(), lower);
	      //states "a" and "b"
	      //"0" order and "1" order corrections
	      //in all 4 states "0a" "1a"  "0b"  "1b"
	      CItype complex1 = 1.0*( conj(hasHEDNumerator[y])*ci1(x,0)/(E01-hasHEDEnergy[y])*sgn); //<1a|v|0a>
	      CItype complex2 = 1.0*( conj(hasHEDNumerator2[y])*ci2(x,0)/(E02-hasHEDEnergy[y])*sgn); //<1b|v|0b>
	      CItype complex12= 1.0*( conj(hasHEDNumerator[y])*ci2(x,0)/(E01-hasHEDEnergy[y])*sgn); //<1a|v|0b>
	      CItype complex12b= 1.0*( conj(ci1(x,0))*hasHEDNumerator2[y]/(E02-hasHEDEnergy[y])*sgn);//<0a|v|1b>

	      spinRDM_thrd[thrd][0](open[a], closed[i]) += complex1;
	      spinRDM_thrd[thrd][1](open[a], closed[i]) += complex2;
	      spinRDM_thrd[thrd][2](open[a], closed[i]) += complex12;

	      spinRDM_thrd[thrd][0](closed[i], open[a]) += conj(complex1);
	      spinRDM_thrd[thrd][1](closed[i], open[a]) += conj(complex2);
	      spinRDM_thrd[thrd][2](closed[i], open[a]) += complex12b;

	      /*
	      spinRDM[0](open[a], closed[i]) += complex1;
	      spinRDM[1](open[a], closed[i]) += complex2;
	      spinRDM[2](open[a], closed[i]) += complex12;

	      spinRDM[0](closed[i], open[a]) += conj(complex1);
	      spinRDM[1](closed[i], open[a]) += conj(complex2);
	      spinRDM[2](closed[i], open[a]) += complex12b;
	      */
	    }
	  }
	}


      }
    }

    for (int thrd=0; thrd<num_thrds; thrd++) {
    spinRDM[0] += spinRDM_thrd[thrd][0];
    spinRDM[1] += spinRDM_thrd[thrd][1];
    spinRDM[2] += spinRDM_thrd[thrd][2];
    }

#ifndef SERIAL
#ifndef Complex
    MPI_Allreduce(MPI_IN_PLACE, &spinRDM[0](0,0), spinRDM[0].rows()*spinRDM[0].cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &spinRDM[1](0,0), spinRDM[1].rows()*spinRDM[1].cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &spinRDM[2](0,0), spinRDM[2].rows()*spinRDM[2].cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //boost::mpi::all_reduce(world, boost::mpi::inplace_t<double* >(&spinRDM[0](0,0)), spinRDM[0].rows()*spinRDM[0].cols(), std::plus<double>());
    //boost::mpi::all_reduce(world, boost::mpi::inplace_t<double* >(&spinRDM[1](0,0)), spinRDM[1].rows()*spinRDM[1].cols(), std::plus<double>());
    //boost::mpi::all_reduce(world, boost::mpi::inplace_t<double* >(&spinRDM[2](0,0)), spinRDM[2].rows()*spinRDM[2].cols(), std::plus<double>());
#else
    MPI_Allreduce(MPI_IN_PLACE, &spinRDM[0](0,0), 2*spinRDM[0].rows()*spinRDM[0].cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &spinRDM[1](0,0), 2*spinRDM[1].rows()*spinRDM[1].cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &spinRDM[2](0,0), 2*spinRDM[2].rows()*spinRDM[2].cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //boost::mpi::all_reduce(world, boost::mpi::inplace_t<std::complex<double>* >(&spinRDM[0](0,0)), spinRDM[0].rows()*spinRDM[0].cols(), sumComplex);
    //boost::mpi::all_reduce(world, boost::mpi::inplace_t<std::complex<double>* >(&spinRDM[1](0,0)), spinRDM[1].rows()*spinRDM[1].cols(), sumComplex);
    //boost::mpi::all_reduce(world, boost::mpi::inplace_t<std::complex<double>* >(&spinRDM[2](0,0)), spinRDM[2].rows()*spinRDM[2].cols(), sumComplex);
#endif
#endif
  }
}


//this takes in a ci vector for determinants placed in Dets
//it then does a SHCI varitional calculation and the resulting
//ci and dets are returned here
//At input usually the Dets will just have a HF or some such determinant
//and ci will be just 1.0
vector<double> SHCIbasics::DoVariational(vector<MatrixXx>& ci, vector<Determinant>& Dets, schedule& schd,
					 twoInt& I2, twoIntHeatBathSHM& I2HB, vector<int>& irrep, oneInt& I1, double& coreE
					 , int nelec, bool DoRDM) {
  
  int proc=0, nprocs=1, localrank=0;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  boost::mpi::communicator world;
  MPI_Comm localComm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
		      MPI_INFO_NULL, &localComm);
  MPI_Comm_rank(localComm, &localrank);
  MPI_Comm_free(&localComm);
#endif

  Determinant* SHMDets, *SortedDets;
  SHMVecFromVecs(Dets, SHMDets, shciDetsCI, DetsCISegment, regionDetsCI);
  if (proc != 0) Dets.resize(0);
  std::vector<Determinant> SortedDetsvec; //only proc 1 has it
  if (commrank == 0) {
    SortedDetsvec = Dets; 
    std::sort(SortedDetsvec.begin(), SortedDetsvec.end());
  }
  int SortedDetsSize = SortedDetsvec.size(), DetsSize = Dets.size();
  SHMVecFromVecs(SortedDetsvec, SortedDets, shciSortedDets, SortedDetsSegment, regionSortedDets);


  Dets.clear();
  SortedDetsvec.clear();
#ifndef SERIAL
  mpi::broadcast(world, SortedDetsSize, 0);
  mpi::broadcast(world, DetsSize, 0);
#endif


  double coreEbkp = coreE;
  coreE = 0.0;

  pout << "**************************************************************"<<endl;
  pout << "VARIATIONAL STEP  "<<endl;
  pout << "**************************************************************"<<endl;


  int nroots = ci.size();
  SHCImakeHamiltonian::HamHelpers2 helper2;
  SHCImakeHamiltonian::SparseHam sparseHam;
  if (schd.DavidsonType == DISK) {
    sparseHam.diskio = true;
    sparseHam.BatchSize = 100000;
    //sparseHam.BatchSize = 10;
    sparseHam.setNbatches(DetsSize);
    sparseHam.prefix = schd.prefix[0];
  }




  MatrixXx diag;
  
  size_t norbs = 2.*I2.Direct.rows();
  int Norbs = norbs;
  vector<double> E0(nroots,SHMDets[0].Energy(I1, I2, coreE));


  //if I1[1].store.size() is not zero then soc integrals is active so populate AlphaN
  if (proc == 0) {
    helper2.PopulateHelpers(SHMDets, DetsSize, 0);
  }	
  helper2.MakeSHMHelpers();

  if (schd.DavidsonType != DIRECT)
    sparseHam.makeFromHelper(helper2, SHMDets, 0, DetsSize, Norbs, I1, I2, coreE, schd.DoRDM);
  


#ifdef Complex
  SHCImakeHamiltonian::updateSOCconnections(SHMDets, 0, DetsSize, SortedDets, sparseHam.connections, 
					    sparseHam.orbDifference, 
					    sparseHam.Helements,
					    norbs, I1, nelec, false);
#endif


  pout << format("%4s %4s  %10s  %10.2e   %18s   %9s  %10s\n") %("Iter") %("Root") %("Eps1 ") %("#Var. Det.") %("Energy") %("#Davidson") %("Time(s)");


  int prevSize = 0;

  //If this is a restart calculation then read from disk
  int iterstart = 0;
  if (schd.restart || schd.fullrestart) {
    bool converged;
    readVariationalResult(iterstart, ci, Dets, sparseHam, E0, converged, schd, helper2);

    SHMVecFromVecs(Dets, SHMDets, shciDetsCI, 
		   DetsCISegment, regionDetsCI);

    DetsSize = Dets.size(); SortedDetsSize = DetsSize;
    SHMVecFromVecs(Dets, SortedDets, shciSortedDets, SortedDetsSegment, regionSortedDets);
#ifndef SERIAL
    mpi::broadcast(world, SortedDetsSize, 0);
    mpi::broadcast(world, DetsSize, 0);
#endif
    if (localrank == 0) 
      std::sort(SortedDets, SortedDets+SortedDetsSize);
    MPI_Barrier(MPI_COMM_WORLD);
    Dets.clear();

    helper2.MakeSHMHelpers();
    if (schd.DavidsonType != DIRECT)
      sparseHam.makeFromHelper(helper2, SHMDets, 0, DetsSize, Norbs, I1, I2, coreE, schd.DoRDM);


    for (int i=0; i<E0.size(); i++)
      pout << format("%4i %4i  %10.2e  %10.2e -   %18.10f  %10.2f\n")
	%(iterstart) %(i) % schd.epsilon1[iterstart] % DetsSize % (E0[i]+coreEbkp) % (getTime()-startofCalc);

    if (!schd.fullrestart) iterstart++;
    else iterstart = 0;
    

    if (converged && iterstart >= schd.epsilon1.size()) {
      for (int i=0; i<E0.size(); i++) 
	E0[i] += coreEbkp;
      coreE = coreEbkp;
      pout << "# restarting from a converged calculation, moving to perturbative part.!!"<<endl;
      Dets.resize(DetsSize);
      for (int i=0; i<DetsSize; i++)
	Dets[i] = SHMDets[i];

      return E0;
    }
  }


  for (int iter=iterstart; iter<schd.epsilon1.size(); iter++) {

    double epsilon1 = schd.epsilon1[iter];    
    StitchDEH uniqueDEH;
    
    //for multiple states, use the sum of squares of states 
    //to do the seclection process
    if(schd.outputlevel>0) pout << format("#-------------Iter=%4i---------------") % iter<<endl;

    CItype *cMaxSHM; vector<CItype> cMax;
    if (proc == 0) {
      cMax.resize(ci[0].rows(),0);
      for (int j=0; j<ci[0].rows(); j++) {
	for (int i=0; i<ci.size(); i++)
	  cMax[j] += pow( abs(ci[i](j,0)), 2);	
	cMax[j] = pow( cMax[j], 0.5);
      }      
    }

    SHMVecFromVecs(cMax, cMaxSHM, shcicMax, cMaxSegment, regioncMax);
    cMax.clear();

    
    CItype zero = 0.0;
    
    for (int i=0; i<SortedDetsSize; i++) {
      if (i%(commsize) != commrank) continue;
      SHCIgetdeterminants::getDeterminantsVariationalApprox(SHMDets[i], 
							    epsilon1/abs(cMaxSHM[i]), cMaxSHM[i], zero,
							    I1, I2, I2HB, irrep, coreE, E0[0],
							    *uniqueDEH.Det,
							    schd,0, nelec, SortedDets, SortedDetsSize);
      /*
      SHCIgetdeterminants::getDeterminantsVariational(SHMDets[i], 
							    epsilon1/abs(cMaxSHM[i]), cMaxSHM[i], zero,
							    I1, I2, I2HB, irrep, coreE, E0[0],
							    *uniqueDEH.Det,
							    schd,0, nelec);
      */
    }

    if (Determinant::Trev != 0) {
      for (int i=0; i<uniqueDEH.Det->size(); i++) 
	uniqueDEH.Det->at(i).makeStandard();
    }

    sort( uniqueDEH.Det->begin(), uniqueDEH.Det->end() );
    uniqueDEH.Det->erase( unique( uniqueDEH.Det->begin(), uniqueDEH.Det->end() ), uniqueDEH.Det->end() );

    if (Determinant::Trev != 0) 
      uniqueDEH.RemoveOnlyDetsPresentIn(SortedDets, SortedDetsSize);

#ifndef SERIAL
    for (int level = 0; level <ceil(log2(nprocs)); level++) {
      
      if (proc%ipow(2, level+1) == 0 && proc + ipow(2, level) < nprocs) {
	int getproc = proc+ipow(2,level);
	long numDets = 0;
	long oldSize = uniqueDEH.Det->size();
	long maxint = 26843540;
	MPI_Recv(&numDets, 1, MPI_DOUBLE, getproc, getproc, MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
	long totalMemory = numDets*DetLen;

	if (totalMemory != 0) {
	  uniqueDEH.Det->resize(oldSize+numDets);
	  for (int i=0; i<(totalMemory/maxint); i++)
	    MPI_Recv(&(uniqueDEH.Det->at(oldSize).repr[0])+i*maxint, 
		     maxint, MPI_DOUBLE, getproc, getproc, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(&(uniqueDEH.Det->at(oldSize).repr[0])+(totalMemory/maxint)*maxint, 
		   totalMemory-(totalMemory/maxint)*maxint, MPI_DOUBLE, 
		   getproc, getproc, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  
	  sort( uniqueDEH.Det->begin(), uniqueDEH.Det->end() );
	  uniqueDEH.Det->erase( unique( uniqueDEH.Det->begin(), uniqueDEH.Det->end() ), uniqueDEH.Det->end() );
	}

      }
      else if ( proc%ipow(2, level+1) == 0 && proc + ipow(2, level) >= nprocs) {
	continue ;
      }
      else if ( proc%ipow(2, level) == 0) {
	int toproc = proc-ipow(2,level);
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
	  uniqueDEH.clear();
	}
      }
    }
    
    
#endif
    
    vector<MatrixXx> X0; X0.resize(ci.size());
    vector<Determinant>& newDets = *uniqueDEH.Det;

    if (proc == 0) {
      X0 = vector<MatrixXx>(ci.size(), MatrixXx(DetsSize+newDets.size(), 1));
      for (int i=0; i<ci.size(); i++) {
	X0[i].setZero(DetsSize+newDets.size(),1);
	X0[i].block(0,0,ci[i].rows(),1) = 1.*ci[i];
      }

      Dets.resize(DetsSize+newDets.size());
      for (int i=0; i<DetsSize; i++)
	Dets[i] = SHMDets[i];
      for (int i=0; i<newDets.size(); i++)
	Dets[i+DetsSize] = newDets[i];

      DetsSize = Dets.size();
    }
    uniqueDEH.resize(0);
    SHMVecFromVecs(Dets, SHMDets, shciDetsCI, DetsCISegment, regionDetsCI);
    Dets.clear();

#ifndef SERIAL
    mpi::broadcast(world, DetsSize, 0);
#endif
    if (commrank == 0) printf("New size of determinant space %8i\n", DetsSize);
    
    if (proc == 0) {
      helper2.PopulateHelpers(SHMDets, DetsSize,SortedDetsSize);
    }	
    helper2.MakeSHMHelpers();
    if (schd.DavidsonType != DIRECT )
      sparseHam.makeFromHelper(helper2, SHMDets, SortedDetsSize, DetsSize, Norbs, I1, I2, coreE, schd.DoRDM);
    
    SHMVecFromVecs(SHMDets, DetsSize, SortedDets, shciSortedDets, SortedDetsSegment, regionSortedDets);
    
    if (localrank == 0) 
      std::sort(SortedDets, SortedDets+DetsSize);
  
#ifdef Complex
      SHCImakeHamiltonian::updateSOCconnections(SHMDets, SortedDetsSize, DetsSize, SortedDets,
						sparseHam.connections, sparseHam.orbDifference,
						sparseHam.Helements, norbs, I1, nelec, false);
#endif

    SortedDetsSize = DetsSize;
#ifndef SERIAL
  mpi::broadcast(world, SortedDetsSize, 0);
#endif
  
    if (proc == 0) {
      MatrixXx diagbkp = diag;
      diag =MatrixXx::Zero(DetsSize,1);
      for (int k=0; k<diagbkp.rows(); k++)
	diag(k,0) = diagbkp(k,0);

      for (size_t k=diagbkp.rows(); k<DetsSize ; k++) {
	CItype hij = SHMDets[k].Energy(I1, I2, coreE);
	diag(k,0) = hij;
      }
    }  
    
    
    double prevE0 = E0[0];
    if (iter == 0) prevE0 = -10.0;
    Hmult2 H(sparseHam);
    HmultDirect Hdirect(helper2, SHMDets, DetsSize, 0, Norbs,
			I1, I2, coreE, diag); 
    if (schd.DavidsonType == DISK) sparseHam.setNbatches(DetsSize);
    //pout << "nbatches : "<<sparseHam.Nbatches<<endl;
    //cout <<commrank<<"  "<< sparseHam.Helements[0][0]<<endl;
    int numIter = 0;
    if (schd.DavidsonType == DIRECT)
      E0 = davidsonDirect(Hdirect, X0, diag, schd.nroots+4, schd.davidsonTolLoose, numIter, true);
    else
      E0 = davidson(H, X0, diag, schd.nroots+4, schd.davidsonTolLoose, numIter, false);

#ifndef SERIAL
    mpi::broadcast(world, E0, 0);
#endif

    pout << format("%4i %4i  %10.2e  %10.2e") 
      %(iter) %(0) % schd.epsilon1[iter] % (newDets.size()+DetsSize) ;
    pout << format("   %18.10f  %9i  %10.2f\n") 
      % (E0[0]+coreEbkp) % (numIter) %(getTime()-startofCalc);

    for (int i=1; i<E0.size(); i++)
      pout << format("%4i %4i  %10.2e  %10.2e   %18.10f  %9i  %10.2f\n") 
	%(iter) %(i) % schd.epsilon1[iter] % DetsSize % (E0[i]+coreEbkp) % (numIter) %(getTime()-startofCalc);
    if (E0.size() >1) pout <<endl;

    if (proc == 0) {
      for (int i=0; i<E0.size(); i++) {
	ci[i].resize(DetsSize,1); ci[i] = 1.0*X0[i];
	X0[i].resize(0,0);
      }
    }


    if (abs(E0[0]-prevE0) < schd.dE || iter == schd.epsilon1.size()-1)  {
      pout << endl<<"Performing final tight davidson with tol: "<<schd.davidsonTol<<endl;

      if (schd.DavidsonType == DIRECT)
	E0 = davidsonDirect(Hdirect, ci, diag, schd.nroots+4, schd.davidsonTol, numIter, true);
      else
	E0 = davidson(H, ci, diag, schd.nroots+4, schd.davidsonTol, numIter, false);

#ifndef SERIAL
      mpi::broadcast(world, E0, 0);
      mpi::broadcast(world, ci, 0);
#endif
      pout <<endl<<"Exiting variational iterations"<<endl;
      if (commrank == 0) {
	Dets.resize(DetsSize);
	for (int i=0; i<DetsSize; i++)
	  Dets[i] = SHMDets[i];
      }
      writeVariationalResult(iter, ci, Dets, sparseHam, E0, true, schd, helper2);



      /*
      if (Determinant::Trev != 0) {
	int numDets = 0;
	for (int i=0; i<DetsSize; i++) {
	  if (Dets[i].hasUnpairedElectrons()) 
	    numDets += 2;
	  else
	    numDets += 1;
	}
	Dets.resize(numDets);
	vector<MatrixXx> cibkp = ci;
	for (int i=0; i<E0.size(); i++) {
	  ci[i].resize(Dets.size(), 1); 
	  ci[i].block(0, 0, cibkp[i].rows(), 1) = 1.*cibkp[i];
	}

	int newIndex=0, oldLen = cibkp[0].rows();
	vector<int> partnerLocation(oldLen,-1);
	for (int i=0; i<oldLen; i++) {
	  if (Dets[i].hasUnpairedElectrons()) {
	    partnerLocation[i] = newIndex;
	    Dets[newIndex+oldLen] = Dets[i];
	    Dets[newIndex+oldLen].flipAlphaBeta();
	    for (int j=0; j<E0.size(); j++) {
	      ci[j](i,0) = cibkp[j](i,0)/sqrt(2.0);
	      double parity = Dets[i].parityOfFlipAlphaBeta();
	      ci[j](newIndex+oldLen,0) = Determinant::Trev*parity*cibkp[j](i,0)/sqrt(2.0);
	    }
	    newIndex++;
	  }
	    
	}

	if (DoRDM || schd.doResponse) {
	  int proc=0, nprocs=1;
#ifndef SERIAL
	  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
	  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#endif
	  connections.resize(Dets.size());
	  orbDifference.resize(Dets.size());

	  for (int i=oldLen; i<Dets.size(); i++) {
	    if (i%nprocs != proc) continue;
	    connections[i].push_back(i); 
	    orbDifference[i].push_back(0);
	  }

	  for (int i=0; i<oldLen; i++) {
	    if (i%nprocs != proc) continue;
	    if (Dets[i].hasUnpairedElectrons()) {
	      if (Dets[i].connected(Dets[partnerLocation[i]+oldLen]) ) {
		size_t orbDiff; 
		CItype hij = Hij(Dets[i], 
				 Dets[partnerLocation[i]+oldLen], I1, I2, coreE, orbDiff);
		connections[partnerLocation[i]+oldLen].push_back(i);
		orbDifference[partnerLocation[i]+oldLen].push_back(orbDiff);
	      }
	    }
	    for (int j=1; j<connections[i].size(); j++) {
	      int J = connections[i][j];

	      if (Dets[i].hasUnpairedElectrons() &&
		  Dets[J].hasUnpairedElectrons()) {

		if (Dets[i].connected(Dets[J])  && !Dets[i].connected(Dets[partnerLocation[J]+oldLen])) {
		  size_t orbDiff; 
		  CItype hij = Hij(Dets[partnerLocation[J]+oldLen], 
				   Dets[partnerLocation[i]+oldLen], I1, I2, coreE, orbDiff);
		  connections[partnerLocation[i]+oldLen].push_back(partnerLocation[J]+oldLen);
		  orbDifference[partnerLocation[i]+oldLen].push_back(orbDiff);
					
		}
		else if (Dets[i].connected(Dets[J]) && Dets[i].connected(Dets[partnerLocation[J]+oldLen]) ) {
		  size_t orbDiff; 
		  CItype hij = Hij(Dets[i], 
				   Dets[partnerLocation[J]+oldLen], I1, I2, coreE, orbDiff);
		  connections[partnerLocation[J]+oldLen].push_back(i);
		  orbDifference[partnerLocation[J]+oldLen].push_back(orbDiff);

		  hij = Hij(Dets[J], 
			    Dets[partnerLocation[i]+oldLen], I1, I2, coreE, orbDiff);
		  connections[partnerLocation[i]+oldLen].push_back(J);
		  orbDifference[partnerLocation[i]+oldLen].push_back(orbDiff);

		  hij = Hij(Dets[partnerLocation[J]+oldLen], 
			    Dets[partnerLocation[i]+oldLen], I1, I2, coreE, orbDiff);
		  connections[partnerLocation[i]+oldLen].push_back(partnerLocation[J]+oldLen);
		  orbDifference[partnerLocation[i]+oldLen].push_back(orbDiff);
		  }
		else if (!Dets[i].connected(Dets[J]) && Dets[i].connected(Dets[partnerLocation[J]+oldLen]) ) {
		  size_t orbDiff; 
		  CItype hij = Hij(Dets[partnerLocation[J]+oldLen], 
				   Dets[i], I1, I2, coreE, orbDiff);
		  connections[i][j] = partnerLocation[J]+oldLen;
		  orbDifference[i][j] = orbDiff;
		  

		  hij = Hij(Dets[J], 
			    Dets[partnerLocation[i]+oldLen], I1, I2, coreE, orbDiff);
		  connections[partnerLocation[i]+oldLen].push_back(J);
		  orbDifference[partnerLocation[i]+oldLen].push_back(orbDiff);
		}
	      }
	      else if (!Dets[i].hasUnpairedElectrons() &&
		       Dets[J].hasUnpairedElectrons()) {
		size_t orbDiff; 
		CItype hij = Hij(Dets[i], 
				 Dets[partnerLocation[J]+oldLen], I1, I2, coreE, orbDiff);
		connections[partnerLocation[J]+oldLen].push_back(i);
		orbDifference[partnerLocation[J]+oldLen].push_back(orbDiff);
		
	      }
	      else if (Dets[i].hasUnpairedElectrons() &&
		       !Dets[J].hasUnpairedElectrons()) {
		size_t orbDiff; 
		CItype hij = Hij(Dets[J], 
				 Dets[partnerLocation[i]+oldLen], I1, I2, coreE, orbDiff);
		connections[partnerLocation[i]+oldLen].push_back(J);
		orbDifference[partnerLocation[i]+oldLen].push_back(orbDiff);
		
	      }

	    }
	  }

	}
      }
      */

      if (DoRDM || schd.doResponse) {
	if (schd.DavidsonType == DIRECT) {
	  pout << "RDM not implemented with direct davidson."<<endl;
	  exit(0);
	}
	pout <<"Calculating RDM"<<endl;
	for (int i=0; i<schd.nroots; i++) {
	  MatrixXx twoRDM;
	  if (schd.DoSpinRDM )
	    twoRDM = MatrixXx::Zero(norbs*(norbs+1)/2, norbs*(norbs+1)/2);
	  MatrixXx s2RDM = MatrixXx::Zero((norbs/2)*norbs/2, (norbs/2)*norbs/2);
	  SHCIrdm::EvaluateRDM(sparseHam.connections, SHMDets, DetsSize, ci[i], 
			       ci[i], sparseHam.orbDifference, nelec, schd, i, twoRDM, s2RDM);
	  //if (schd.outputlevel>0) 
	    SHCIrdm::ComputeEnergyFromSpatialRDM(norbs/2, nelec, I1, I2, coreEbkp, s2RDM);
	  SHCIrdm::saveRDM(schd, s2RDM, twoRDM, i);
        } // for i
      }
      sparseHam.resize(0);

      break;
    }
    else {
      if (schd.io) {
	if (commrank == 0) {
	  Dets.resize(DetsSize);
	  for (int i=0; i<DetsSize; i++)
	    Dets[i] = SHMDets[i];	  
	}
      writeVariationalResult(iter, ci, Dets, sparseHam, E0, true, schd, helper2);
      }
      Dets.clear();
    }
    
    if (schd.outputlevel>0) pout << format("###########################################      %10.2f ") %(getTime()-startofCalc)<<endl;
  }

  boost::interprocess::shared_memory_object::remove(shciDetsCI.c_str());
  boost::interprocess::shared_memory_object::remove(shciHelper.c_str());


  pout << "VARIATIONAL CALCULATION RESULT"<<endl;
  pout << "------------------------------"<<endl;
  pout << format("%4s %18s  %10s\n") %("Root") %("Energy") %("Time(s)");
  for (int i=0; i<E0.size(); i++) {
    E0[i] += coreEbkp;
    pout << format("%4i  %18.10f  %10.2f\n") 
      %(i) % (E0[i]) % (getTime()-startofCalc);
  }
  pout <<endl<<endl;
  coreE = coreEbkp;
  return E0;

}



void SHCIbasics::writeVariationalResult(int iter, vector<MatrixXx>& ci, vector<Determinant>& Dets,
				        vector<vector<int> >& connections, vector<vector<size_t> >&orbdifference,
				       vector<vector<CItype> >& Helements,
				       vector<double>& E0, bool converged, schedule& schd,
				       std::map<HalfDet, std::vector<int> >& BetaN,
				       std::map<HalfDet, std::vector<int> >& AlphaNm1) {

#ifndef SERIAL
  boost::mpi::communicator world;
#endif

  if (schd.outputlevel>0) pout << format("#Begin writing variational wf %29.2f\n")
    % (getTime()-startofCalc);

  {
    char file [5000];
    sprintf (file, "%s/%d-variational.bkp" , schd.prefix[0].c_str(), commrank );
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << iter <<Dets;
    save << ci;
    save << E0;
    save << converged;
    ofs.close();
  }

  {
    char file [5000];
    sprintf (file, "%s/%d-hamiltonian.bkp" , schd.prefix[0].c_str(), commrank );
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << connections<<Helements<<orbdifference;
  }

  {
    char file [5000];
    sprintf (file, "%s/%d-helpers.bkp" , schd.prefix[0].c_str(), commrank );
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << BetaN<< AlphaNm1;
  }

  if (schd.outputlevel>0) pout << format("#End   writing variational wf %29.2f\n")
    % (getTime()-startofCalc);
}


void SHCIbasics::readVariationalResult(int& iter, vector<MatrixXx>& ci, vector<Determinant>& Dets,
				       vector<vector<int> >& connections, vector<vector<size_t> >& orbdifference,
				      vector<vector<CItype> >& Helements,
				      vector<double>& E0, bool& converged, schedule& schd,
				      std::map<HalfDet, std::vector<int> >& BetaN,
				      std::map<HalfDet, std::vector<int> >& AlphaNm1) {


#ifndef SERIAL
  boost::mpi::communicator world;
#endif

  if (schd.outputlevel>0) pout << format("#Begin reading variational wf %29.2f\n")
    % (getTime()-startofCalc);

  {
    char file [5000];
    sprintf (file, "%s/%d-variational.bkp" , schd.prefix[0].c_str(), commrank );
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);

    load >> iter >> Dets ;
    ci.resize(1, MatrixXx(Dets.size(),1));

    load >> ci;
    load >> E0;
    if (schd.onlyperturbative) {ifs.close();return;}
    load >> converged;
  }

  {
    char file [5000];
    sprintf (file, "%s/%d-hamiltonian.bkp" , schd.prefix[0].c_str(), commrank );
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    load >> connections >> Helements >>orbdifference;
  }

  {
    char file [5000];
    sprintf (file, "%s/%d-helpers.bkp" , schd.prefix[0].c_str(), commrank );
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    load >> BetaN>> AlphaNm1;
    ifs.close();
  }

  if (schd.outputlevel >0) pout << format("#End   reading variational wf %29.2f\n")
    % (getTime()-startofCalc);
}


void SHCIbasics::writeVariationalResult(int iter, vector<MatrixXx>& ci, vector<Determinant>& Dets,
					SHCImakeHamiltonian::SparseHam& sparseHam,
					vector<double>& E0, bool converged, schedule& schd,
					SHCImakeHamiltonian::HamHelpers2& helper2) {
  
#ifndef SERIAL
  boost::mpi::communicator world;
#endif

  if (schd.outputlevel>0) pout << format("#Begin writing variational wf %29.2f\n")
    % (getTime()-startofCalc);

  {
    char file [5000];
    sprintf (file, "%s/%d-variational.bkp" , schd.prefix[0].c_str(), commrank );
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << iter <<Dets;
    save << ci;
    save << E0;
    save << converged;
    ofs.close();
  }

  /*
  {
    char file [5000];
    sprintf (file, "%s/%d-hamiltonian.bkp" , schd.prefix[0].c_str(), commrank );
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << sparseHam.connections<<sparseHam.Helements<<sparseHam.orbDifference;
  }
  */

  if (world.rank() == 0)
  {
    char file [5000];
    sprintf (file, "%s/%d-helpers.bkp" , schd.prefix[0].c_str(), commrank );
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << helper2.AlphaMajorToBeta<<helper2.AlphaMajorToDet
	 << helper2.BetaMajorToAlpha<<helper2.BetaMajorToDet
	 << helper2.SinglesFromAlpha<<helper2.SinglesFromBeta
	 << helper2.BetaN            << helper2.AlphaN;
  }

  if (schd.outputlevel>0) pout << format("#End   writing variational wf %29.2f\n")
    % (getTime()-startofCalc);
}


void SHCIbasics::readVariationalResult(int& iter, vector<MatrixXx>& ci, vector<Determinant>& Dets,
				       SHCImakeHamiltonian::SparseHam& sparseHam,
				       vector<double>& E0, bool& converged, schedule& schd,
				       SHCImakeHamiltonian::HamHelpers2& helper2) {
			    


#ifndef SERIAL
  boost::mpi::communicator world;
#endif

  if (schd.outputlevel>0) pout << format("#Begin reading variational wf %29.2f\n")
    % (getTime()-startofCalc);

  {
    char file [5000];
    sprintf (file, "%s/%d-variational.bkp" , schd.prefix[0].c_str(), commrank );
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);

    std::vector<Determinant> sorted;
    load >> iter >> Dets;// >>sorted ;
    load >> ci;
    load >> E0;
    if (schd.onlyperturbative) {ifs.close();return;}
    load >> converged;
  }

  /*
  if (schd.DavidsonType != DIRECT)
  {
    char file [5000];
    sprintf (file, "%s/%d-hamiltonian.bkp" , schd.prefix[0].c_str(), commrank );
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    load >> sparseHam.connections >> sparseHam.Helements >>sparseHam.orbDifference;
  }
  */

  if (world.rank() == 0)
  {
    char file [5000];
    sprintf (file, "%s/%d-helpers.bkp" , schd.prefix[0].c_str(), commrank );
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    load >> helper2.AlphaMajorToBeta >> helper2.AlphaMajorToDet
	 >> helper2.BetaMajorToAlpha >> helper2.BetaMajorToDet 
	 >> helper2.SinglesFromAlpha >> helper2.SinglesFromBeta
         >> helper2.BetaN            >> helper2.AlphaN;
    
    ifs.close();
  }

  if (schd.outputlevel >0) pout << format("#End   reading variational wf %29.2f\n")
    % (getTime()-startofCalc);
}


void SHCIbasics::writeHelperIntermediate( std::map<HalfDet, int >& BetaN,
					  std::map<HalfDet, int >& AlphaN,
					  std::map<HalfDet, vector<int> >& BetaNm1,
					  std::map<HalfDet, vector<int> >& AlphaNm1, schedule& schd, int iter){
  
#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  if (world.rank() == 0)
  {
    char file [5000];
    sprintf (file, "%s/%d-helpers-Intermediate-%d.bkp" , schd.prefix[0].c_str(), commrank, iter );
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << BetaN<< AlphaN << BetaNm1 << AlphaNm1;
  }

}


void SHCIbasics::readHelperIntermediate(std::map<HalfDet, int >& BetaN,
				       std::map<HalfDet, int >& AlphaN,
				       std::map<HalfDet, vector<int> >& BetaNm1,
					std::map<HalfDet, vector<int> >& AlphaNm1, schedule& schd, int iter){
			   

#ifndef SERIAL
  boost::mpi::communicator world;
#endif

  if (world.rank() == 0)
  {
    char file [5000];
    sprintf (file, "%s/%d-helpers-Intermediate-%d.bkp" , schd.prefix[0].c_str(), commrank , iter);
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    load >> BetaN>> AlphaN >> BetaNm1 >> AlphaNm1;
    ifs.close();
  }

}

