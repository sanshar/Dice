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




void SHCIbasics::DoPerturbativeStochastic2SingleListDoubleEpsilon2(vector<Determinant>& Dets, MatrixXx& ci, double& E0, oneInt& I1, twoInt& I2,
								  twoIntHeatBathSHM& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec, int root) {

  boost::mpi::communicator world;
  char file [5000];
  sprintf (file, "output-%d.bkp" , world.rank() );
  std::ofstream ofs;
  if (root == 0)
    ofs.open(file, std::ofstream::out);
  else
    ofs.open(file, std::ofstream::app);

  double epsilon2 = schd.epsilon2;
  schd.epsilon2 = schd.epsilon2Large;
  vector<MatrixXx> vdVector;
  double Psi1Norm;
  double EptLarge = DoPerturbativeDeterministic(Dets, ci, E0, I1, I2, I2HB, irrep, schd, coreE, nelec, root,  vdVector, Psi1Norm);

  schd.epsilon2 = epsilon2;

  int norbs = Determinant::norbs;
  std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
  int niter = schd.nPTiter;
  //double eps = 0.001;
  int Nsample = schd.SampleN;
  double AvgenergyEN = 0.0;
  double AverageDen = 0.0;
  int currentIter = 0;
  int sampleSize = 0;
  int num_thrds = omp_get_max_threads();

  double cumulative = 0.0;
  for (int i=0; i<ci.rows(); i++)
    cumulative += abs(ci(i,0));

  std::vector<int> alias; std::vector<double> prob;
  SHCIsampledeterminants::setUpAliasMethod(ci, cumulative, alias, prob);
#pragma omp parallel for schedule(dynamic)
  for (int iter=0; iter<niter; iter++) {
    //cout << norbs<<"  "<<nelec<<endl;
    char psiArray[norbs];
    vector<int> psiClosed(nelec,0);
    vector<int> psiOpen(norbs-nelec,0);
    //char psiArray[norbs];
    std::vector<CItype> wts1(Nsample,0.0); std::vector<int> Sample1(Nsample,-1);

    //int Nmc = sample_N2(ci, cumulative, Sample1, wts1);
    int distinctSample = SHCIsampledeterminants::sample_N2_alias(ci, cumulative, Sample1, wts1, alias, prob);
    int Nmc = Nsample;
    double norm = 0.0;

    //map<Determinant, std::tuple<double,double,double, double, double> > Psi1ab;
    std::vector<Determinant> Psi1; std::vector<CItype>  numerator1A; vector<double> numerator2A;
    vector<char> present;
    std::vector<double>  det_energy;
    for (int i=0; i<distinctSample; i++) {
      int I = Sample1[i];
      SHCIgetdeterminants::getDeterminantsStochastic2Epsilon(Dets[I], schd.epsilon2/abs(ci(I,0)), schd.epsilon2Large/abs(ci(I,0)), wts1[i], ci(I,0), I1, I2, I2HB, irrep, coreE, E0, Psi1, numerator1A, numerator2A, present, det_energy, schd, Nmc, nelec);
    }


    quickSort( &(Psi1[0]), 0, Psi1.size(), &numerator1A[0], &numerator2A[0], &det_energy, &present);

    CItype currentNum1A=0.; double currentNum2A=0.;
    CItype currentNum1B=0.; double currentNum2B=0.;
    vector<Determinant>::iterator vec_it = SortedDets.begin();
    double energyEN = 0.0, energyENLargeEps = 0.0;

    for (int i=0;i<Psi1.size();) {
      if (Psi1[i] < *vec_it) {
	currentNum1A += numerator1A[i];
	currentNum2A += numerator2A[i];
	if (present[i]) {
	  currentNum1B += numerator1A[i];
	  currentNum2B += numerator2A[i];
	}

	if ( i == Psi1.size()-1) {
	  energyEN += (pow(abs(currentNum1A),2)*Nmc/(Nmc-1) - currentNum2A)/(det_energy[i] - E0);
	  energyENLargeEps += (pow(abs(currentNum1B),2)*Nmc/(Nmc-1) - currentNum2B)/(det_energy[i] - E0);
	}
	else if (!(Psi1[i] == Psi1[i+1])) {
	  energyEN += ( pow(abs(currentNum1A),2)*Nmc/(Nmc-1) - currentNum2A)/(det_energy[i] - E0);
	  energyENLargeEps += ( pow(abs(currentNum1B),2)*Nmc/(Nmc-1) - currentNum2B)/(det_energy[i] - E0);
	  currentNum1A = 0.;
	  currentNum2A = 0.;
	  currentNum1B = 0.;
	  currentNum2B = 0.;
	}
	i++;
      }
      else if (*vec_it <Psi1[i] && vec_it != SortedDets.end())
	vec_it++;
      else if (*vec_it <Psi1[i] && vec_it == SortedDets.end()) {
	currentNum1A += numerator1A[i];
	currentNum2A += numerator2A[i];
	if (present[i]) {
	  currentNum1B += numerator1A[i];
	  currentNum2B += numerator2A[i];
	}

	if ( i == Psi1.size()-1) {
	  energyEN += ( pow(abs(currentNum1A),2)*Nmc/(Nmc-1) - currentNum2A)/(det_energy[i] - E0);
	  energyENLargeEps += ( pow(abs(currentNum1B),2)*Nmc/(Nmc-1) - currentNum2B)/(det_energy[i] - E0);
	}
	else if (!(Psi1[i] == Psi1[i+1])) {
	  energyEN += ( pow(abs(currentNum1A),2)*Nmc/(Nmc-1) - currentNum2A)/(det_energy[i] - E0);
	  energyENLargeEps += ( pow(abs(currentNum1B),2)*Nmc/(Nmc-1) - currentNum2B)/(det_energy[i] - E0);
	  //energyEN += (currentNum1A*currentNum1A*Nmc/(Nmc-1) - currentNum2A)/(det_energy[i] - E0);
	  //energyENLargeEps += (currentNum1B*currentNum1B*Nmc/(Nmc-1) - currentNum2B)/(det_energy[i] - E0);
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

    sampleSize = distinctSample;


#pragma omp critical
    {
      if (mpigetrank() == 0) {
	AvgenergyEN += -energyEN+energyENLargeEps+EptLarge; currentIter++;
	std::cout << format("%6i  %14.8f  %s%i %14.8f   %10.2f  %10i %4i")
	  %(currentIter) % (E0-energyEN+energyENLargeEps+EptLarge) % ("Root") % root % (E0+AvgenergyEN/currentIter) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	cout << endl;
      }
      else {
	AvgenergyEN += -energyEN+energyENLargeEps+EptLarge; currentIter++;
	ofs << format("%6i  %14.8f  %s%i %14.8f   %10.2f  %10i %4i")
	  %(currentIter) % (E0-energyEN+energyENLargeEps+EptLarge) % ("Root") % root % (E0+AvgenergyEN/currentIter) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	ofs << endl;

      }
    }
  }
  ofs.close();

}




void SHCIbasics::DoPerturbativeStochastic2SingleListDoubleEpsilon2OMPTogether(vector<Determinant>& Dets, MatrixXx& ci, double& E0, oneInt& I1, twoInt& I2,
									     twoIntHeatBathSHM& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec, int root) {

  boost::mpi::communicator world;
  char file [5000];
  sprintf (file, "output-%d.bkp" , world.rank() );
  std::ofstream ofs;
  if (root == 0)
    ofs.open(file, std::ofstream::out);
  else
    ofs.open(file, std::ofstream::app);

  double epsilon2 = schd.epsilon2;
  schd.epsilon2 = schd.epsilon2Large;
  vector<MatrixXx> vdVector;
  double Psi1Norm;
  double EptLarge = DoPerturbativeDeterministic(Dets, ci, E0, I1, I2, I2HB, irrep, schd, coreE, nelec, root,  vdVector, Psi1Norm);

  schd.epsilon2 = epsilon2;

  int norbs = Determinant::norbs;
  std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
  int niter = schd.nPTiter;
  //double eps = 0.001;
  int Nsample = schd.SampleN;
  double AvgenergyEN = 0.0;
  double AverageDen = 0.0;
  int currentIter = 0;
  int sampleSize = 0;
  int num_thrds = omp_get_max_threads();

  double cumulative = 0.0;
  for (int i=0; i<ci.rows(); i++)
    cumulative += abs(ci(i,0));

  std::vector<int> alias; std::vector<double> prob;
  SHCIsampledeterminants::setUpAliasMethod(ci, cumulative, alias, prob);

  double totalPT=0, totalPTLargeEps=0;

  std::vector< std::vector<vector<Determinant> > > hashedDetBeforeMPI   (num_thrds);// vector<Determinant> > >(num_thrds));
  std::vector< std::vector<vector<CItype> > >      hashedNum1BeforeMPI  (num_thrds);// std::vector<vector<double> > >(num_thrds));
  std::vector< std::vector<vector<double> > >      hashedNum2BeforeMPI  (num_thrds);// std::vector<vector<double> > >(num_thrds));
  std::vector< std::vector<vector<double> > >      hashedEnergyBeforeMPI(num_thrds);// std::vector<vector<double> > >(num_thrds));
  std::vector< std::vector<vector<char> > >        hashedpresentBeforeMPI(num_thrds);// std::vector<vector<char> > >(num_thrds));

  int AllDistinctSample = 0;
  int Nmc = Nsample*num_thrds;
  std::vector<CItype> allwts(Nmc, 0.0); std::vector<int> allSample(Nmc, -1);

#pragma omp parallel
  {
    for (int iter=0; iter<niter; iter++) {
      std::vector<CItype> wts1(Nsample,0.0); std::vector<int> Sample1(Nsample,-1);
      int distinctSample = 0;

      if (omp_get_thread_num() == 0) {
	std::fill(allSample.begin(), allSample.end(), -1);
	AllDistinctSample = SHCIsampledeterminants::sample_N2_alias(ci, cumulative, allSample, allwts, alias, prob);
      }
#pragma omp barrier
      if (omp_get_thread_num() <  AllDistinctSample%num_thrds)
	distinctSample = AllDistinctSample/num_thrds + 1;
      else
	distinctSample = AllDistinctSample/num_thrds;


      size_t stride = omp_get_thread_num() < AllDistinctSample%num_thrds \
					     ? omp_get_thread_num()*distinctSample \
					     : omp_get_thread_num()*distinctSample + AllDistinctSample%num_thrds;

      for (int i = 0; i < distinctSample; i++) {
	wts1   [i] = allwts   [i + stride];
	Sample1[i] = allSample[i + stride];
      }

      double norm = 0.0;

      std::vector<Determinant> Psi1; std::vector<CItype>  numerator1A; vector<double> numerator2A;
      vector<char> present;
      std::vector<double>  det_energy;

      for (int i=0; i<distinctSample; i++) {
	int I = Sample1[i];
	SHCIgetdeterminants::getDeterminantsStochastic2Epsilon(Dets[I], schd.epsilon2/abs(ci(I,0)),
						      schd.epsilon2Large/abs(ci(I,0)), wts1[i],
						      ci(I,0), I1, I2, I2HB, irrep, coreE, E0,
						      Psi1,
						      numerator1A,
						      numerator2A,
						      present,
						      det_energy,
						      schd, Nmc, nelec);
      }

      if(num_thrds >1) {
	hashedDetBeforeMPI[omp_get_thread_num()].resize(num_thrds);
	hashedNum1BeforeMPI[omp_get_thread_num()].resize(num_thrds);
	hashedNum2BeforeMPI[omp_get_thread_num()].resize(num_thrds);
	hashedEnergyBeforeMPI[omp_get_thread_num()].resize(num_thrds);
	hashedpresentBeforeMPI[omp_get_thread_num()].resize(num_thrds);

	for (int thrd=0; thrd<num_thrds; thrd++) {
	  hashedDetBeforeMPI[omp_get_thread_num()][thrd].reserve(Psi1.size()/num_thrds);
	  hashedNum1BeforeMPI[omp_get_thread_num()][thrd].reserve(Psi1.size()/num_thrds);
	  hashedNum2BeforeMPI[omp_get_thread_num()][thrd].reserve(Psi1.size()/num_thrds);
	  hashedEnergyBeforeMPI[omp_get_thread_num()][thrd].reserve(Psi1.size()/num_thrds);
	  hashedpresentBeforeMPI[omp_get_thread_num()][thrd].reserve(Psi1.size()/num_thrds);
	}

	for (int j=0; j<Psi1.size(); j++) {
	  size_t lOrder = Psi1.at(j).getHash();
	  size_t thrd = lOrder%(num_thrds);
	  hashedDetBeforeMPI[omp_get_thread_num()][thrd].push_back(Psi1.at(j));
	  hashedNum1BeforeMPI[omp_get_thread_num()][thrd].push_back(numerator1A.at(j));
	  hashedNum2BeforeMPI[omp_get_thread_num()][thrd].push_back(numerator2A.at(j));
	  hashedEnergyBeforeMPI[omp_get_thread_num()][thrd].push_back(det_energy.at(j));
	  hashedpresentBeforeMPI[omp_get_thread_num()][thrd].push_back(present.at(j));
	}
	Psi1.clear(); numerator1A.clear(); numerator2A.clear(); det_energy.clear(); present.clear();

#pragma omp barrier
	size_t totalSize = 0;
	for (int thrd=0; thrd<num_thrds; thrd++)
	  totalSize += hashedDetBeforeMPI[thrd][omp_get_thread_num()].size();

	Psi1.reserve(totalSize);
	numerator1A.reserve(totalSize);
	numerator2A.reserve(totalSize);
	det_energy.reserve(totalSize);
	present.reserve(totalSize);
	for (int thrd=0; thrd<num_thrds; thrd++) {
	  for (int j=0; j<hashedDetBeforeMPI[thrd][omp_get_thread_num()].size(); j++) {
	    Psi1.push_back(hashedDetBeforeMPI[thrd][omp_get_thread_num()].at(j));
	    numerator1A.push_back(hashedNum1BeforeMPI[thrd][omp_get_thread_num()].at(j));
	    numerator2A.push_back(hashedNum2BeforeMPI[thrd][omp_get_thread_num()].at(j));
	    det_energy.push_back(hashedEnergyBeforeMPI[thrd][omp_get_thread_num()].at(j));
	    present.push_back(hashedpresentBeforeMPI[thrd][omp_get_thread_num()].at(j));
	  }
	  hashedDetBeforeMPI[thrd][omp_get_thread_num()].clear();
	  hashedNum1BeforeMPI[thrd][omp_get_thread_num()].clear();
	  hashedNum2BeforeMPI[thrd][omp_get_thread_num()].clear();
	  hashedEnergyBeforeMPI[thrd][omp_get_thread_num()].clear();
	  hashedpresentBeforeMPI[thrd][omp_get_thread_num()].clear();
	}
      }

      std::vector<Determinant> Psi1copy = Psi1;
      vector<long> detIndex(Psi1.size(), 0);
      vector<long> detIndexcopy(Psi1.size(), 0);
      for (size_t i=0; i<Psi1.size(); i++)
	detIndex[i] = i;
      mergesort(&Psi1copy[0], 0, Psi1.size()-1, &detIndex[0], &( Psi1.operator[](0)), &detIndexcopy[0]);
      detIndexcopy.clear(); Psi1copy.clear();
      reorder(numerator1A, detIndex);
      reorder(numerator2A, detIndex);
      reorder(det_energy, detIndex);
      reorder(present, detIndex);
      detIndex.clear();


      CItype currentNum1A=0.; double currentNum2A=0.;
      CItype currentNum1B=0.; double currentNum2B=0.;
      vector<Determinant>::iterator vec_it = SortedDets.begin();
      double energyEN = 0.0, energyENLargeEps = 0.0;
      //size_t effNmc = num_thrds*Nmc;

      for (int i=0;i<Psi1.size();) {
	if (Psi1[i] < *vec_it) {
	  currentNum1A += numerator1A[i];
	  currentNum2A += numerator2A[i];
	  if (present[i]) {
	    currentNum1B += numerator1A[i];
	    currentNum2B += numerator2A[i];
	  }

	  if ( i == Psi1.size()-1) {
	    energyEN += (pow(abs(currentNum1A),2)*Nmc/(Nmc-1) - currentNum2A)/(det_energy[i] - E0);
	    energyENLargeEps += (pow(abs(currentNum1B),2)*Nmc/(Nmc-1) - currentNum2B)/(det_energy[i] - E0);
	  }
	  else if (!(Psi1[i] == Psi1[i+1])) {
	    energyEN += ( pow(abs(currentNum1A),2)*Nmc/(Nmc-1) - currentNum2A)/(det_energy[i] - E0);
	    energyENLargeEps += ( pow(abs(currentNum1B),2)*Nmc/(Nmc-1) - currentNum2B)/(det_energy[i] - E0);
	    currentNum1A = 0.;
	    currentNum2A = 0.;
	    currentNum1B = 0.;
	    currentNum2B = 0.;
	  }
	  i++;
	}
	else if (*vec_it <Psi1[i] && vec_it != SortedDets.end())
	  vec_it++;
	else if (*vec_it <Psi1[i] && vec_it == SortedDets.end()) {
	  currentNum1A += numerator1A[i];
	  currentNum2A += numerator2A[i];
	  if (present[i]) {
	    currentNum1B += numerator1A[i];
	    currentNum2B += numerator2A[i];
	  }

	  if ( i == Psi1.size()-1) {
	    energyEN += ( pow(abs(currentNum1A),2)*Nmc/(Nmc-1) - currentNum2A)/(det_energy[i] - E0);
	    energyENLargeEps += ( pow(abs(currentNum1B),2)*Nmc/(Nmc-1) - currentNum2B)/(det_energy[i] - E0);
	  }
	  else if (!(Psi1[i] == Psi1[i+1])) {
	    energyEN += ( pow(abs(currentNum1A),2)*Nmc/(Nmc-1) - currentNum2A)/(det_energy[i] - E0);
	    energyENLargeEps += ( pow(abs(currentNum1B),2)*Nmc/(Nmc-1) - currentNum2B)/(det_energy[i] - E0);
	    //energyEN += (currentNum1A*currentNum1A*Nmc/(Nmc-1) - currentNum2A)/(det_energy[i] - E0);
	    //energyENLargeEps += (currentNum1B*currentNum1B*Nmc/(Nmc-1) - currentNum2B)/(det_energy[i] - E0);
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
#pragma omp barrier
#pragma omp critical
      {
	totalPT += energyEN;
	totalPTLargeEps += energyENLargeEps;
      }
#pragma omp barrier

      double finalE = totalPT, finalELargeEps=totalPTLargeEps;
      if (mpigetrank() == 0 && omp_get_thread_num() == 0) {
	AvgenergyEN += -finalE+finalELargeEps+EptLarge; currentIter++;
	std::cout << format("%6i  %14.8f  %s%i %14.8f   %10.2f  %10i")
	  %(currentIter) % (E0-finalE+finalELargeEps+EptLarge) % ("Root") % root % (E0+AvgenergyEN/currentIter) % (getTime()-startofCalc) % AllDistinctSample ;
	cout << endl;
      }
      else if (mpigetrank() != 0 && omp_get_thread_num() == 0) {
	AvgenergyEN += -finalE+finalELargeEps+EptLarge; currentIter++;
	ofs << format("%6i  %14.8f  %s%i %14.8f   %10.2f  %10i")
	  %(currentIter) % (E0-finalE+finalELargeEps+EptLarge) % ("Root") % root % (E0+AvgenergyEN/currentIter) % (getTime()-startofCalc) % AllDistinctSample ;
	ofs << endl;
      }
    }
  }

}

void SHCIbasics::DoPerturbativeStochastic2SingleList(vector<Determinant>& Dets, MatrixXx& ci, double& E0, oneInt& I1, twoInt& I2,
						    twoIntHeatBathSHM& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec, int root) {

  boost::mpi::communicator world;
  char file [5000];
  sprintf (file, "output-%d.bkp" , world.rank() );
  //std::ofstream ofs(file);
  std::ofstream ofs;
  if (root == 0)
    ofs.open(file, std::ofstream::out);
  else
    ofs.open(file, std::ofstream::app);

  int norbs = Determinant::norbs;
  std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
  int niter = schd.nPTiter;
  //int niter = 1000000;
  //double eps = 0.001;
  int Nsample = schd.SampleN;
  double AvgenergyEN = 0.0;
  double AverageDen = 0.0;
  int currentIter = 0;
  int sampleSize = 0;
  int num_thrds = omp_get_max_threads();

  double cumulative = 0.0;
  for (int i=0; i<ci.rows(); i++)
    cumulative += abs(ci(i,0));

  std::vector<int> alias; std::vector<double> prob;
  SHCIsampledeterminants::setUpAliasMethod(ci, cumulative, alias, prob);
#pragma omp parallel for schedule(dynamic)
  for (int iter=0; iter<niter; iter++) {
    //cout << norbs<<"  "<<nelec<<endl;
    char psiArray[norbs];
    vector<int> psiClosed(nelec,0);
    vector<int> psiOpen(norbs-nelec,0);
    //char psiArray[norbs];
    std::vector<CItype> wts1(Nsample,0.0); std::vector<int> Sample1(Nsample,-1);

    //int Nmc = sample_N2(ci, cumulative, Sample1, wts1);
    int distinctSample = SHCIsampledeterminants::sample_N2_alias(ci, cumulative, Sample1, wts1, alias, prob);
    int Nmc = Nsample;
    double norm = 0.0;

    size_t initSize = 100000;
    std::vector<Determinant> Psi1; std::vector<CItype>  numerator1; std::vector<double> numerator2;
    std::vector<double>  det_energy;
    Psi1.reserve(initSize); numerator1.reserve(initSize); numerator2.reserve(initSize); det_energy.reserve(initSize);
    for (int i=0; i<distinctSample; i++) {
      int I = Sample1[i];
      SHCIgetdeterminants::getDeterminantsStochastic(Dets[I], schd.epsilon2/abs(ci(I,0)), wts1[i], ci(I,0), I1, I2, I2HB, irrep, coreE, E0, Psi1, numerator1, numerator2, det_energy, schd, Nmc, nelec);
    }

    quickSort( &(Psi1[0]), 0, Psi1.size(), &numerator1[0], &numerator2[0], &det_energy);

    CItype currentNum1=0.; double currentNum2=0.;
    vector<Determinant>::iterator vec_it = SortedDets.begin();
    double energyEN = 0.0;

    for (int i=0;i<Psi1.size();) {
      if (Psi1[i] < *vec_it) {
	currentNum1 += numerator1[i];
	currentNum2 += numerator2[i];
	if ( i == Psi1.size()-1)
	  energyEN += (pow(abs(currentNum1),2)*Nmc/(Nmc-1) - currentNum2)/(det_energy[i] - E0);
	else if (!(Psi1[i] == Psi1[i+1])) {
	  energyEN += ( pow(abs(currentNum1),2)*Nmc/(Nmc-1) - currentNum2)/(det_energy[i] - E0);
	  currentNum1 = 0.;
	  currentNum2 = 0.;
	}
	i++;
      }
      else if (*vec_it <Psi1[i] && vec_it != SortedDets.end())
	vec_it++;
      else if (*vec_it <Psi1[i] && vec_it == SortedDets.end()) {
	currentNum1 += numerator1[i];
	currentNum2 += numerator2[i];
	if ( i == Psi1.size()-1)
	  energyEN += (pow(abs(currentNum1),2)*Nmc/(Nmc-1) - currentNum2)/(det_energy[i] - E0);
	//energyEN += (currentNum1*currentNum1*Nmc/(Nmc-1) - currentNum2)/(det_energy[i] - E0);
	else if (!(Psi1[i] == Psi1[i+1])) {
	  energyEN += ( pow(abs(currentNum1),2)*Nmc/(Nmc-1) - currentNum2)/(det_energy[i] - E0);
	  //energyEN += (currentNum1*currentNum1*Nmc/(Nmc-1) - currentNum2)/(det_energy[i] - E0);
	  currentNum1 = 0.;
	  currentNum2 = 0.;
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

    sampleSize = distinctSample;

#pragma omp critical
    {
      if (mpigetrank() == 0) {
	AvgenergyEN += energyEN; currentIter++;
	std::cout << format("%6i  %14.8f  %s%i %14.8f   %10.2f  %10i %4i")
	  %(currentIter) % (E0-energyEN) % ("Root") % root % (E0-AvgenergyEN/currentIter) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	cout << endl;
      }
      else {
	AvgenergyEN += energyEN; currentIter++;
	ofs << format("%6i  %14.8f  %s%i %14.8f   %10.2f  %10i %4i")
	  %(currentIter) % (E0-energyEN) % ("Root") % root % (E0-AvgenergyEN/currentIter) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	ofs << endl;

      }
    }
  }
  ofs.close();

}

double SHCIbasics::DoPerturbativeDeterministic(vector<Determinant>& Dets, MatrixXx& ci, double& E0, oneInt& I1, twoInt& I2,
					       twoIntHeatBathSHM& I2HB, vector<int>& irrep, schedule& schd, double coreE, 
					       int nelec, int root,  vector<MatrixXx>& vdVector, double& Psi1Norm,
					       bool appendPsi1ToPsi0) {

  boost::mpi::communicator world;
  int norbs = Determinant::norbs;
  std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());

  double energyEN = 0.0;
  double Psi1NormProc = 0.0;
  int num_thrds = omp_get_max_threads();

  std::vector<StitchDEH> uniqueDEH(num_thrds);
  std::vector<std::vector< std::vector<vector<Determinant> > > > hashedDetBeforeMPI(mpigetsize(), std::vector<std::vector<vector<Determinant> > >(num_thrds));
  std::vector<std::vector< std::vector<vector<Determinant> > > > hashedDetAfterMPI(mpigetsize(), std::vector<std::vector<vector<Determinant> > >(num_thrds));
  std::vector<std::vector< std::vector<vector<CItype> > > > hashedNumBeforeMPI(mpigetsize(), std::vector<std::vector<vector<CItype> > >(num_thrds));
  std::vector<std::vector< std::vector<vector<CItype> > > > hashedNumAfterMPI(mpigetsize(), std::vector<std::vector<vector<CItype> > >(num_thrds));
  std::vector<std::vector< std::vector<vector<double> > > > hashedEnergyBeforeMPI(mpigetsize(), std::vector<std::vector<vector<double> > >(num_thrds));
  std::vector<std::vector< std::vector<vector<double> > > > hashedEnergyAfterMPI(mpigetsize(), std::vector<std::vector<vector<double> > >(num_thrds));
  std::vector<std::vector< std::vector<vector<vector<int> > > > > hashedVarIndicesBeforeMPI(mpigetsize(), std::vector<std::vector<vector<vector<int> > > >(num_thrds));
  std::vector<std::vector< std::vector<vector<vector<int> > > > > hashedVarIndicesAfterMPI(mpigetsize(), std::vector<std::vector<vector<vector<int> > > >(num_thrds));
  std::vector<std::vector< std::vector<vector<vector<size_t> > > > > hashedOrbdiffBeforeMPI(mpigetsize(), std::vector<std::vector<vector<vector<size_t> > > >(num_thrds));
  std::vector<std::vector< std::vector<vector<vector<size_t> > > > > hashedOrbdiffAfterMPI(mpigetsize(), std::vector<std::vector<vector<vector<size_t> > > >(num_thrds));
  double totalPT = 0.0;
  int ntries = 0;

#pragma omp parallel
  {
    if (schd.DoRDM || schd.doResponse) {
      uniqueDEH[omp_get_thread_num()].extra_info = true;
      for (int i=0; i<Dets.size(); i++) {
	if (i%(omp_get_num_threads()*mpigetsize()) != mpigetrank()*omp_get_num_threads()+omp_get_thread_num()) {continue;}
	SHCIgetdeterminants::getDeterminantsDeterministicPTKeepRefDets(Dets[i], i, abs(schd.epsilon2/ci(i,0)), ci(i,0),
						I1, I2, I2HB, irrep, coreE, E0,
						*uniqueDEH[omp_get_thread_num()].Det,
						*uniqueDEH[omp_get_thread_num()].Num,
						*uniqueDEH[omp_get_thread_num()].Energy,
						*uniqueDEH[omp_get_thread_num()].var_indices_beforeMerge,
						*uniqueDEH[omp_get_thread_num()].orbDifference_beforeMerge,
						schd, nelec);
	if (i%100000 == 0 && omp_get_thread_num()==0 && mpigetrank() == 0) cout <<"# "<<i<<endl;
      }
    }
    else {
      for (int i=0; i<Dets.size(); i++) {
	if (i%(omp_get_num_threads()*mpigetsize()) != mpigetrank()*omp_get_num_threads()+omp_get_thread_num()) {continue;}
	SHCIgetdeterminants::getDeterminantsDeterministicPT(Dets[i], abs(schd.epsilon2/ci(i,0)), ci(i,0), 0.0,
				   I1, I2, I2HB, irrep, coreE, E0,
				   *uniqueDEH[omp_get_thread_num()].Det,
				   *uniqueDEH[omp_get_thread_num()].Num,
				   *uniqueDEH[omp_get_thread_num()].Energy,
				   schd,0, nelec);
	if (i%100000 == 0 && omp_get_thread_num()==0 && mpigetrank() == 0) cout <<"# "<<i<<endl;
      }
    }


    uniqueDEH[omp_get_thread_num()].MergeSortAndRemoveDuplicates();
    uniqueDEH[omp_get_thread_num()].RemoveDetsPresentIn(SortedDets);

    if(mpigetsize() >1 || num_thrds >1) {
      StitchDEH uniqueDEH_afterMPI;
      if (schd.DoRDM || schd.doResponse) uniqueDEH_afterMPI.extra_info = true;
      if (mpigetrank() == 0 && omp_get_thread_num() == 0) cout << "#Before hash "<<getTime()-startofCalc<<endl;


      for (int proc=0; proc<mpigetsize(); proc++) {
	hashedDetBeforeMPI[proc][omp_get_thread_num()].resize(num_thrds);
	hashedNumBeforeMPI[proc][omp_get_thread_num()].resize(num_thrds);
	hashedEnergyBeforeMPI[proc][omp_get_thread_num()].resize(num_thrds);
	if (schd.DoRDM || schd.doResponse) {
	  hashedVarIndicesBeforeMPI[proc][omp_get_thread_num()].resize(num_thrds);
	  hashedOrbdiffBeforeMPI[proc][omp_get_thread_num()].resize(num_thrds);
	}
      }

      if (omp_get_thread_num()==0 ) {
	if (mpigetsize() == 1)
	  ntries = 1;
	else {
	  ntries = uniqueDEH[omp_get_thread_num()].Det->size()*DetLen*2*omp_get_num_threads()/268435400+1;
	  mpi::broadcast(world, ntries, 0);
	}
      }
#pragma omp barrier

      size_t batchsize = uniqueDEH[omp_get_thread_num()].Det->size()/ntries;
      //ntries = 1;
      for (int tries = 0; tries<ntries; tries++) {

        size_t start = (ntries-1-tries)*batchsize;
        size_t end   = tries==0 ? uniqueDEH[omp_get_thread_num()].Det->size() : (ntries-tries)*batchsize;

	for (size_t j=start; j<end; j++) {
	  size_t lOrder = uniqueDEH[omp_get_thread_num()].Det->at(j).getHash();
	  size_t procThrd = lOrder%(mpigetsize()*num_thrds);
	  int proc = abs(procThrd/num_thrds), thrd = abs(procThrd%num_thrds);
	  hashedDetBeforeMPI[proc][omp_get_thread_num()][thrd].push_back(uniqueDEH[omp_get_thread_num()].Det->at(j));
	  hashedNumBeforeMPI[proc][omp_get_thread_num()][thrd].push_back(uniqueDEH[omp_get_thread_num()].Num->at(j));
	  hashedEnergyBeforeMPI[proc][omp_get_thread_num()][thrd].push_back(uniqueDEH[omp_get_thread_num()].Energy->at(j));
	  if (schd.DoRDM || schd.doResponse) {
	    hashedVarIndicesBeforeMPI[proc][omp_get_thread_num()][thrd].push_back(uniqueDEH[omp_get_thread_num()].var_indices->at(j));
	    hashedOrbdiffBeforeMPI[proc][omp_get_thread_num()][thrd].push_back(uniqueDEH[omp_get_thread_num()].orbDifference->at(j));
	  }
	}

	uniqueDEH[omp_get_thread_num()].resize(start);

	if (mpigetrank() == 0 && omp_get_thread_num() == 0) cout << "#After hash "<<getTime()-startofCalc<<endl;

#pragma omp barrier
	if (omp_get_thread_num()==num_thrds-1) {
	  mpi::all_to_all(world, hashedDetBeforeMPI, hashedDetAfterMPI);
	  mpi::all_to_all(world, hashedNumBeforeMPI, hashedNumAfterMPI);
	  mpi::all_to_all(world, hashedEnergyBeforeMPI, hashedEnergyAfterMPI);
	  if (schd.DoRDM || schd.doResponse) {
	    mpi::all_to_all(world, hashedVarIndicesBeforeMPI, hashedVarIndicesAfterMPI);
	    mpi::all_to_all(world, hashedOrbdiffBeforeMPI, hashedOrbdiffAfterMPI);
	  }
	}
#pragma omp barrier

	for (int proc=0; proc<mpigetsize(); proc++) {
	  for (int thrd=0; thrd<num_thrds; thrd++) {
	    hashedDetBeforeMPI[proc][thrd][omp_get_thread_num()].clear();
	    hashedNumBeforeMPI[proc][thrd][omp_get_thread_num()].clear();
	    hashedEnergyBeforeMPI[proc][thrd][omp_get_thread_num()].clear();
	    if (schd.DoRDM || schd.doResponse) {
	      hashedVarIndicesBeforeMPI[proc][thrd][omp_get_thread_num()].clear();
	      hashedOrbdiffBeforeMPI[proc][thrd][omp_get_thread_num()].clear();
	    }
	  }
	}


	if (mpigetrank() == 0 && omp_get_thread_num() == 0) cout << "#After all_to_all "<<getTime()-startofCalc<<endl;

	for (int proc=0; proc<mpigetsize(); proc++) {
	  for (int thrd=0; thrd<num_thrds; thrd++) {
	    for (int j=0; j<hashedDetAfterMPI[proc][thrd][omp_get_thread_num()].size(); j++) {
	      uniqueDEH_afterMPI.Det->push_back(hashedDetAfterMPI[proc][thrd][omp_get_thread_num()].at(j));
	      uniqueDEH_afterMPI.Num->push_back(hashedNumAfterMPI[proc][thrd][omp_get_thread_num()].at(j));
	      uniqueDEH_afterMPI.Energy->push_back(hashedEnergyAfterMPI[proc][thrd][omp_get_thread_num()].at(j));
	      if (schd.DoRDM || schd.doResponse) {
		uniqueDEH_afterMPI.var_indices->push_back(hashedVarIndicesAfterMPI[proc][thrd][omp_get_thread_num()].at(j));
		uniqueDEH_afterMPI.orbDifference->push_back(hashedOrbdiffAfterMPI[proc][thrd][omp_get_thread_num()].at(j));
	      }
	    }
	    hashedDetAfterMPI[proc][thrd][omp_get_thread_num()].clear();
	    hashedNumAfterMPI[proc][thrd][omp_get_thread_num()].clear();
	    hashedEnergyAfterMPI[proc][thrd][omp_get_thread_num()].clear();
	    if (schd.DoRDM || schd.doResponse) {
	      hashedVarIndicesAfterMPI[proc][thrd][omp_get_thread_num()].clear();
	      hashedOrbdiffAfterMPI[proc][thrd][omp_get_thread_num()].clear();
	    }
	  }
	}
      }


      *uniqueDEH[omp_get_thread_num()].Det = *uniqueDEH_afterMPI.Det;
      *uniqueDEH[omp_get_thread_num()].Num = *uniqueDEH_afterMPI.Num;
      *uniqueDEH[omp_get_thread_num()].Energy = *uniqueDEH_afterMPI.Energy;
      if (schd.DoRDM || schd.doResponse) {
	*uniqueDEH[omp_get_thread_num()].var_indices = *uniqueDEH_afterMPI.var_indices;
	*uniqueDEH[omp_get_thread_num()].orbDifference = *uniqueDEH_afterMPI.orbDifference;
      }
      uniqueDEH_afterMPI.clear();

      uniqueDEH[omp_get_thread_num()].Num2->clear();
      uniqueDEH[omp_get_thread_num()].MergeSortAndRemoveDuplicates();


    }
    if (mpigetrank() == 0 && omp_get_thread_num() == 0) cout << "#After collecting "<<getTime()-startofCalc<<endl;

    //uniqueDEH[omp_get_thread_num()].RemoveDetsPresentIn(SortedDets);
    if (mpigetrank() == 0 && omp_get_thread_num() == 0) cout << "#Unique determinants "<<getTime()-startofCalc<<"  "<<endl;


    vector<Determinant>& hasHEDDets = *uniqueDEH[omp_get_thread_num()].Det;
    vector<CItype>& hasHEDNumerator = *uniqueDEH[omp_get_thread_num()].Num;
    vector<double>& hasHEDEnergy = *uniqueDEH[omp_get_thread_num()].Energy;

    double PTEnergy = 0.0;
    double psi1normthrd=0.0;
    for (size_t i=0; i<hasHEDDets.size();i++) {
      psi1normthrd += pow(abs(hasHEDNumerator[i]/(E0-hasHEDEnergy[i])),2);
      PTEnergy += pow(abs(hasHEDNumerator[i]),2)/(E0-hasHEDEnergy[i]);
    }
#pragma omp critical
    {
      Psi1NormProc += psi1normthrd;
      totalPT += PTEnergy;
    }

  }


  double finalE = 0.;
  mpi::all_reduce(world, totalPT, finalE, std::plus<double>());
  mpi::all_reduce(world, Psi1NormProc, Psi1Norm, std::plus<double>());

  if (mpigetrank() == 0) cout << "#Done energy "<<E0+finalE<<"  "<<getTime()-startofCalc<<endl;


  if (schd.doResponse || schd.DoRDM) { //build RHS for the lambda equation
    MatrixXx s2RDM, twoRDM;
    SHCIrdm::loadRDM(schd, s2RDM, twoRDM, root);
    mpi::broadcast(world, s2RDM, 0);
    if (schd.DoSpinRDM) 
      mpi::broadcast(world, twoRDM, 0);
    if (mpigetrank() != 0) {
      s2RDM = 0.*s2RDM;
      twoRDM = 0.*twoRDM;
    }
    SHCIrdm::UpdateRDMResponsePerturbativeDeterministic(Dets, ci, E0, I1, I2, schd, coreE, 
							nelec, norbs, uniqueDEH, root, Psi1Norm,
							s2RDM, twoRDM); 
    SHCIrdm::saveRDM(schd, s2RDM, twoRDM, root);
    //construct the vector Via x da
    //where Via is the perturbation matrix element
    //da are the elements of the PT wavefunctions
    vdVector[root]= MatrixXx::Zero(Dets.size(),1);
    for (int thrd=0; thrd <num_thrds; thrd++) {
      vector<Determinant>& uniqueDets = *uniqueDEH[thrd].Det;
      vector<double>& uniqueEnergy = *uniqueDEH[thrd].Energy;
      vector<CItype>& uniqueNumerator = *uniqueDEH[thrd].Num;
      vector<vector<int> >& uniqueVarIndices = *uniqueDEH[thrd].var_indices;
      vector<vector<size_t> >& uniqueOrbDiff = *uniqueDEH[thrd].orbDifference;
      for (int a=0; a<uniqueDets.size(); a++) {
	CItype da = uniqueNumerator[a]/(E0-uniqueEnergy[a]); //coefficient for det a
	for (int i=0; i<uniqueVarIndices[a].size(); i++) {
	  int I = uniqueVarIndices[a][i]; //index of the Var determinant
	  size_t orbDiff;
	  vdVector[root](I,0) -= conj(da)*Hij(uniqueDets[a], Dets[I], I1, I2, coreE, orbDiff);
	}
      }
    } 
   
    MPI_Allreduce(MPI_IN_PLACE, &vdVector[root](0,0), vdVector[root].rows(), 
		  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
 
  }

  return finalE;
}


void SHCIbasics::DoPerturbativeDeterministicOffdiagonal(vector<Determinant>& Dets, MatrixXx& ci1, double& E01, 
							MatrixXx&ci2, double& E02, oneInt& I1, twoInt& I2,
							twoIntHeatBathSHM& I2HB, vector<int>& irrep, 
							schedule& schd, double coreE, int nelec, int root, 
							CItype& EPT1, CItype& EPT2, CItype& EPT12,
							std::vector<MatrixXx>& spinRDM) {

  boost::mpi::communicator world;
  int norbs = Determinant::norbs;
  std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());

  double energyEN = 0.0;
  int num_thrds = omp_get_max_threads();

  std::vector<StitchDEH> uniqueDEH(num_thrds);
  std::vector<std::vector< std::vector<vector<Determinant> > > > hashedDetBeforeMPI(mpigetsize(), std::vector<std::vector<vector<Determinant> > >(num_thrds));
  std::vector<std::vector< std::vector<vector<Determinant> > > > hashedDetAfterMPI(mpigetsize(), std::vector<std::vector<vector<Determinant> > >(num_thrds));
  std::vector<std::vector< std::vector<vector<CItype> > > > hashedNumBeforeMPI(mpigetsize(), std::vector<std::vector<vector<CItype> > >(num_thrds));
  std::vector<std::vector< std::vector<vector<CItype> > > > hashedNumAfterMPI(mpigetsize(), std::vector<std::vector<vector<CItype> > >(num_thrds));
  std::vector<std::vector< std::vector<vector<CItype> > > > hashedNum2BeforeMPI(mpigetsize(), std::vector<std::vector<vector<CItype> > >(num_thrds));
  std::vector<std::vector< std::vector<vector<CItype> > > > hashedNum2AfterMPI(mpigetsize(), std::vector<std::vector<vector<CItype> > >(num_thrds));
  std::vector<std::vector< std::vector<vector<double> > > > hashedEnergyBeforeMPI(mpigetsize(), std::vector<std::vector<vector<double> > >(num_thrds));
  std::vector<std::vector< std::vector<vector<double> > > > hashedEnergyAfterMPI(mpigetsize(), std::vector<std::vector<vector<double> > >(num_thrds));
  CItype totalPT1 = 0.0, totalPT2=0., totalPT12=0.;
  int ntries = 1;

#pragma omp parallel
  {

    for (int i=0; i<Dets.size(); i++) {
      if (i%(omp_get_num_threads()*mpigetsize()) != mpigetrank()*omp_get_num_threads()+omp_get_thread_num()) {continue;}
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

    if(mpigetsize() >1 || num_thrds >1) {
      StitchDEH uniqueDEH_afterMPI;
      if (schd.DoRDM || schd.doResponse) uniqueDEH_afterMPI.extra_info = true;


      for (int proc=0; proc<mpigetsize(); proc++) {
	hashedDetBeforeMPI[proc][omp_get_thread_num()].resize(num_thrds);
	hashedNumBeforeMPI[proc][omp_get_thread_num()].resize(num_thrds);
	hashedNum2BeforeMPI[proc][omp_get_thread_num()].resize(num_thrds);
	hashedEnergyBeforeMPI[proc][omp_get_thread_num()].resize(num_thrds);
      }

      if (omp_get_thread_num()==0) {
	ntries = uniqueDEH[omp_get_thread_num()].Det->size()*DetLen*2*omp_get_num_threads()/268435400+1;
	if (mpigetsize() == 1)
	  ntries = 1;
	mpi::broadcast(world, ntries, 0);
      }
#pragma omp barrier

      size_t batchsize = uniqueDEH[omp_get_thread_num()].Det->size()/ntries;
      //ntries = 1;
      for (int tries = 0; tries<ntries; tries++) {

        size_t start = (ntries-1-tries)*batchsize;
        size_t end   = tries==0 ? uniqueDEH[omp_get_thread_num()].Det->size() : (ntries-tries)*batchsize;
	for (size_t j=start; j<end; j++) {
	  size_t lOrder = uniqueDEH[omp_get_thread_num()].Det->at(j).getHash();
	  size_t procThrd = lOrder%(mpigetsize()*num_thrds);
	  int proc = abs(procThrd/num_thrds), thrd = abs(procThrd%num_thrds);
	  hashedDetBeforeMPI[proc][omp_get_thread_num()][thrd].push_back(uniqueDEH[omp_get_thread_num()].Det->at(j));
	  hashedNumBeforeMPI[proc][omp_get_thread_num()][thrd].push_back(uniqueDEH[omp_get_thread_num()].Num->at(j));
	  hashedNum2BeforeMPI[proc][omp_get_thread_num()][thrd].push_back(uniqueDEH[omp_get_thread_num()].Num2->at(j));
	  hashedEnergyBeforeMPI[proc][omp_get_thread_num()][thrd].push_back(uniqueDEH[omp_get_thread_num()].Energy->at(j));
	}

	uniqueDEH[omp_get_thread_num()].resize(start);


#pragma omp barrier
	if (omp_get_thread_num()==num_thrds-1) {
	  mpi::all_to_all(world, hashedDetBeforeMPI, hashedDetAfterMPI);
	  mpi::all_to_all(world, hashedNumBeforeMPI, hashedNumAfterMPI);
	  mpi::all_to_all(world, hashedNum2BeforeMPI, hashedNum2AfterMPI);
	  mpi::all_to_all(world, hashedEnergyBeforeMPI, hashedEnergyAfterMPI);
	}
#pragma omp barrier

	for (int proc=0; proc<mpigetsize(); proc++) {
	  for (int thrd=0; thrd<num_thrds; thrd++) {
	    hashedDetBeforeMPI[proc][thrd][omp_get_thread_num()].clear();
	    hashedNumBeforeMPI[proc][thrd][omp_get_thread_num()].clear();
	    hashedNum2BeforeMPI[proc][thrd][omp_get_thread_num()].clear();
	    hashedEnergyBeforeMPI[proc][thrd][omp_get_thread_num()].clear();
	  }
	}



	for (int proc=0; proc<mpigetsize(); proc++) {
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
  mpi::all_reduce(world, totalPT1, EPT1, std::plus<CItype>());
  mpi::all_reduce(world, totalPT2, EPT2, std::plus<CItype>());
  mpi::all_reduce(world, totalPT12, EPT12, std::plus<CItype>());

  if (schd.doGtensor && true) {//DON'T PERFORM doGtensor

    if (mpigetrank() != 0) {
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
    boost::mpi::all_reduce(world, boost::mpi::inplace_t<double* >(&spinRDM[0](0,0)), spinRDM[0].rows()*spinRDM[0].cols(), std::plus<double>()); 
    boost::mpi::all_reduce(world, boost::mpi::inplace_t<double* >(&spinRDM[1](0,0)), spinRDM[1].rows()*spinRDM[1].cols(), std::plus<double>()); 
    boost::mpi::all_reduce(world, boost::mpi::inplace_t<double* >(&spinRDM[2](0,0)), spinRDM[2].rows()*spinRDM[2].cols(), std::plus<double>()); 
#else
    boost::mpi::all_reduce(world, boost::mpi::inplace_t<std::complex<double>* >(&spinRDM[0](0,0)), spinRDM[0].rows()*spinRDM[0].cols(), sumComplex); 
    boost::mpi::all_reduce(world, boost::mpi::inplace_t<std::complex<double>* >(&spinRDM[1](0,0)), spinRDM[1].rows()*spinRDM[1].cols(), sumComplex); 
    boost::mpi::all_reduce(world, boost::mpi::inplace_t<std::complex<double>* >(&spinRDM[2](0,0)), spinRDM[2].rows()*spinRDM[2].cols(), sumComplex); 
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

  int nroots = ci.size();
  std::map<HalfDet, std::vector<int> > BetaN, AlphaNm1, AlphaN;
  //if I1[1].store.size() is not zero then soc integrals is active so populate AlphaN
  SHCImakeHamiltonian::PopulateHelperLists(BetaN, AlphaNm1, Dets, 0);

#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  int num_thrds = omp_get_max_threads();

  std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());

  size_t norbs = 2.*I2.Direct.rows();
  int Norbs = norbs;
  vector<double> E0(nroots,Dets[0].Energy(I1, I2, coreE));

  pout << "#HF = "<<E0[0]<<std::endl;

  //this is essentially the hamiltonian, we have stored it in a sparse format
  std::vector<std::vector<int> > connections; connections.resize(Dets.size());
  std::vector<std::vector<CItype> > Helements;Helements.resize(Dets.size());
  std::vector<std::vector<size_t> > orbDifference;orbDifference.resize(Dets.size());
  SHCImakeHamiltonian::MakeHfromHelpers(BetaN, AlphaNm1, Dets, 0, connections, Helements,
					norbs, I1, I2, coreE, orbDifference, DoRDM||schd.doResponse);

#ifdef Complex
  SHCImakeHamiltonian::updateSOCconnections(Dets, 0, connections, orbDifference, Helements, 
					    norbs, I1, nelec, false);
#endif


  //keep the diagonal energies of determinants so we only have to generated
  //this for the new determinants in each iteration and not all determinants
  MatrixXx diagOld(Dets.size(),1);
  for (int i=0; i<Dets.size(); i++)
    diagOld(i,0) = Dets[i].Energy(I1, I2, coreE);
  int prevSize = 0;

  //If this is a restart calculation then read from disk
  int iterstart = 0;
  if (schd.restart || schd.fullrestart) {
    bool converged;
    readVariationalResult(iterstart, ci, Dets, SortedDets, diagOld, connections, orbDifference, Helements, E0, converged, schd, BetaN, AlphaNm1);
    if (schd.fullrestart)
      iterstart = 0;
    pout << format("# %4i  %10.2e  %10.2e   %14.8f  %10.2f\n")
      %(iterstart) % schd.epsilon1[iterstart] % Dets.size() % E0[0] % (getTime()-startofCalc);
    if (!schd.fullrestart)
      iterstart++;
    else
      SHCImakeHamiltonian::regenerateH(Dets, connections, Helements, I1, I2, coreE);
    if (schd.onlyperturbative)
      return E0;

    if (converged && iterstart >= schd.epsilon1.size()) {
      pout << "# restarting from a converged calculation, moving to perturbative part.!!"<<endl;
      return E0;
    }
  }


  //do the variational bit
  for (int iter=iterstart; iter<schd.epsilon1.size(); iter++) {
    double epsilon1 = schd.epsilon1[iter];

    std::vector<StitchDEH> uniqueDEH(num_thrds);

    //for multiple states, use the sum of squares of states to do the seclection process
    pout << format("#-------------Iter=%4i---------------") % iter<<endl;
    MatrixXx cMax(ci[0].rows(),1); cMax = 0.*ci[0];
    for (int j=0; j<ci[0].rows(); j++) {
      for (int i=0; i<ci.size(); i++) 
	cMax(j,0) += pow( abs(ci[i](j,0)), 2);
	
      cMax(j,0) = pow( cMax(j,0), 0.5);
    }


    CItype zero = 0.0;

#pragma omp parallel
    {
      for (int i=0; i<SortedDets.size(); i++) {
	if (i%(mpigetsize()*omp_get_num_threads()) != mpigetrank()*omp_get_num_threads()+omp_get_thread_num()) continue;
	SHCIgetdeterminants::getDeterminantsVariational(Dets[i], epsilon1/abs(cMax(i,0)), cMax(i,0), zero,
					       I1, I2, I2HB, irrep, coreE, E0[0],
					       *uniqueDEH[omp_get_thread_num()].Det,
					       schd,0, nelec);
      }
      uniqueDEH[omp_get_thread_num()].Energy->resize(uniqueDEH[omp_get_thread_num()].Det->size(),0.0);
      uniqueDEH[omp_get_thread_num()].Num->resize(uniqueDEH[omp_get_thread_num()].Det->size(),0.0);


      uniqueDEH[omp_get_thread_num()].QuickSortAndRemoveDuplicates();
      uniqueDEH[omp_get_thread_num()].RemoveDetsPresentIn(SortedDets);


#pragma omp barrier
      if (omp_get_thread_num() == 0) {
	for (int thrd=1; thrd<num_thrds; thrd++) {
	  uniqueDEH[0].merge(uniqueDEH[thrd]);
	  uniqueDEH[thrd].clear();
	  uniqueDEH[0].RemoveDuplicates();
	}
      }
    }

    uniqueDEH.resize(1);

#ifndef SERIAL
    for (int level = 0; level <ceil(log2(mpigetsize())); level++) {

      if (mpigetrank()%ipow(2, level+1) == 0 && mpigetrank() + ipow(2, level) < mpigetsize()) {
	StitchDEH recvDEH;
	int getproc = mpigetrank()+ipow(2,level);
	world.recv(getproc, mpigetsize()*level+getproc, recvDEH);
	uniqueDEH[0].merge(recvDEH);
	uniqueDEH[0].RemoveDuplicates();
      }
      else if ( mpigetrank()%ipow(2, level+1) == 0 && mpigetrank() + ipow(2, level) >= mpigetsize()) {
	continue ;
      }
      else if ( mpigetrank()%ipow(2, level) == 0) {
	int toproc = mpigetrank()-ipow(2,level);
	world.send(toproc, mpigetsize()*level+mpigetrank(), uniqueDEH[0]);
      }
    }


    mpi::broadcast(world, uniqueDEH[0], 0);
#endif

    vector<Determinant>& newDets = *uniqueDEH[0].Det;
    //cout << newDets.size()<<"  "<<Dets.size()<<endl;
    vector<MatrixXx> X0(ci.size(), MatrixXx(Dets.size()+newDets.size(), 1));
    for (int i=0; i<ci.size(); i++) {
      X0[i].setZero(Dets.size()+newDets.size(),1);
      X0[i].block(0,0,ci[i].rows(),1) = 1.*ci[i];

    }

    vector<Determinant>::iterator vec_it = SortedDets.begin();
    int ciindex = 0;
    double EPTguess = 0.0;
    for (vector<Determinant>::iterator it=newDets.begin(); it!=newDets.end(); ) {
      if (schd.excitation != 1000 ) {
	if (it->ExcitationDistance(Dets[0]) > schd.excitation) continue;
      }
      Dets.push_back(*it);
      it++;
    }
    uniqueDEH.resize(0);

    if (iter != 0) pout << str(boost::format("#Initial guess(PT) : %18.10g  \n") %(E0[0]+EPTguess) );

    MatrixXx diag(Dets.size(), 1); diag.setZero(diag.size(),1);
    if (mpigetrank() == 0) diag.block(0,0,ci[0].rows(),1)= 1.*diagOld;




    connections.resize(Dets.size());
    Helements.resize(Dets.size());
    orbDifference.resize(Dets.size());



    SHCImakeHamiltonian::PopulateHelperLists(BetaN, AlphaNm1, Dets, ci[0].size());
    SHCImakeHamiltonian::MakeHfromHelpers(BetaN, AlphaNm1, Dets, SortedDets.size(), 
					  connections, Helements,
					  norbs, I1, I2, coreE, orbDifference, DoRDM || schd.doResponse);
#ifdef Complex
    SHCImakeHamiltonian::updateSOCconnections(Dets, SortedDets.size(), connections, orbDifference,
					      Helements, norbs, I1, nelec, false);
#endif

#pragma omp parallel
    {
      for (size_t k=SortedDets.size(); k<Dets.size() ; k++) {
	if (k%(mpigetsize()*omp_get_num_threads()) != mpigetrank()*omp_get_num_threads()+omp_get_thread_num()) continue;
	diag(k,0) = Helements[k].at(0);
      }
    }

#ifndef SERIAL
#ifndef Complex
    MPI_Allreduce(MPI_IN_PLACE, &diag(0,0), diag.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
    boost::mpi::all_reduce(world, boost::mpi::inplace_t<std::complex<double>* >(&diag(0,0)), diag.rows(), sumComplex); 
#endif
#endif

    for (size_t i=SortedDets.size(); i<Dets.size(); i++)
      SortedDets.push_back(Dets[i]);
    std::sort(SortedDets.begin(), SortedDets.end());



    double prevE0 = E0[0];
    if (iter == 0) prevE0 = -10.0;
    Hmult2 H(connections, Helements);

    E0 = davidson(H, X0, diag, schd.nroots+10, schd.davidsonTol, false);
    mpi::broadcast(world, E0, 0);
    mpi::broadcast(world, X0, 0);

    for (int i=0; i<E0.size(); i++) {
      ci[i].resize(Dets.size(),1); ci[i] = 1.0*X0[i];
      X0[i].resize(0,0);
    }

    diagOld.resize(Dets.size(),1); diagOld = 1.0*diag;

    if (abs(E0[0]-prevE0) < schd.dE || iter == schd.epsilon1.size()-1)  {

      writeVariationalResult(iter, ci, Dets, SortedDets, diag, connections, orbDifference, Helements, E0, true, schd, BetaN, AlphaNm1);
      if (DoRDM || schd.doResponse) {
	Helements.resize(0); BetaN.clear(); AlphaNm1.clear();
	for (int i=0; i<schd.nroots; i++) {
	  MatrixXx twoRDM;
	  if (schd.DoSpinRDM )
	    twoRDM = MatrixXx::Zero(norbs*(norbs+1)/2, norbs*(norbs+1)/2);
	  MatrixXx s2RDM = MatrixXx::Zero((norbs/2)*norbs/2, (norbs/2)*norbs/2);
	  SHCIrdm::EvaluateRDM(connections, Dets, ci[i], ci[i], orbDifference, nelec, schd, i, twoRDM, s2RDM);
	  SHCIrdm::ComputeEnergyFromSpatialRDM(norbs/2, nelec, I1, I2, coreE, s2RDM);
	  SHCIrdm::saveRDM(schd, s2RDM, twoRDM, i);
        } // for i
      }

      break;
    }
    else {
      if (schd.io) writeVariationalResult(iter, ci, Dets, SortedDets, diag, connections,
					  orbDifference, Helements, E0, false, schd, BetaN, AlphaNm1);
    }

    pout << format("###########################################      %10.2f ") %(getTime()-startofCalc)<<endl;
  }
  return E0;

}


void SHCIbasics::writeVariationalResult(int iter, vector<MatrixXx>& ci, vector<Determinant>& Dets, vector<Determinant>& SortedDets,
				       MatrixXx& diag, vector<vector<int> >& connections, vector<vector<size_t> >&orbdifference,
				       vector<vector<CItype> >& Helements,
				       vector<double>& E0, bool converged, schedule& schd,
				       std::map<HalfDet, std::vector<int> >& BetaN,
				       std::map<HalfDet, std::vector<int> >& AlphaNm1) {

#ifndef SERIAL
  boost::mpi::communicator world;
#endif

  pout << format("#Begin writing variational wf %29.2f\n")
    % (getTime()-startofCalc);

  {
    char file [5000];
    sprintf (file, "%s/%d-variational.bkp" , schd.prefix[0].c_str(), world.rank() );
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << iter <<Dets<<SortedDets;
    int diagrows = diag.rows();
    save << diagrows;
    for (int i=0; i<diag.rows(); i++)
      save << diag(i,0);
    save << ci;
    save << E0;
    save << converged;
    ofs.close();
  }

  {
    char file [5000];
    sprintf (file, "%s/%d-hamiltonian.bkp" , schd.prefix[0].c_str(), world.rank() );
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << connections<<Helements<<orbdifference;
  }

  {    
    char file [5000];
    sprintf (file, "%s/%d-helpers.bkp" , schd.prefix[0].c_str(), world.rank() );
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << BetaN<< AlphaNm1;
  }

  pout << format("#End   writing variational wf %29.2f\n")
    % (getTime()-startofCalc);
}


void SHCIbasics::readVariationalResult(int& iter, vector<MatrixXx>& ci, vector<Determinant>& Dets, vector<Determinant>& SortedDets,
				      MatrixXx& diag, vector<vector<int> >& connections, vector<vector<size_t> >& orbdifference,
				      vector<vector<CItype> >& Helements,
				      vector<double>& E0, bool& converged, schedule& schd,
				      std::map<HalfDet, std::vector<int> >& BetaN,
				      std::map<HalfDet, std::vector<int> >& AlphaNm1) {


#ifndef SERIAL
  boost::mpi::communicator world;
#endif

  pout << format("#Begin reading variational wf %29.2f\n")
    % (getTime()-startofCalc);

  {
    char file [5000];
    sprintf (file, "%s/%d-variational.bkp" , schd.prefix[0].c_str(), world.rank() );
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    
    load >> iter >> Dets >> SortedDets ;
    int diaglen;
    load >> diaglen;
    ci.resize(1, MatrixXx(diaglen,1)); diag.resize(diaglen,1);
    for (int i=0; i<diag.rows(); i++)
      load >> diag(i,0);
    
    load >> ci;
    load >> E0;
    if (schd.onlyperturbative) {ifs.close();return;}
    load >> converged;
  }

  {
    char file [5000];
    sprintf (file, "%s/%d-hamiltonian.bkp" , schd.prefix[0].c_str(), world.rank() );
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    load >> connections >> Helements >>orbdifference;
  }

  {
    char file [5000];
    sprintf (file, "%s/%d-helpers.bkp" , schd.prefix[0].c_str(), world.rank() );
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    load >> BetaN>> AlphaNm1;
    ifs.close();
  }
    
  pout << format("#End   reading variational wf %29.2f\n")
    % (getTime()-startofCalc);
}



