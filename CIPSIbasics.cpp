#include "Determinants.h"
#include "CIPSIbasics.h"
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

int ipow(int base, int exp)
{
  int result = 1;
  while (exp)
    {
      if (exp & 1)
	result *= base;
      exp >>= 1;
      base *= base;
    }

  return result;
}

void merge(Determinant *a, size_t low, size_t high, size_t mid, double* x, double* y, Determinant* c, double* cx, double* cy)
{
  size_t i, j, k;
  i = low;
  k = low;
  j = mid + 1;
  while (i <= mid && j <= high)
    {
      if (a[i] < a[j])
        {
	  c[k] = a[i];
	  cx[k] = x[i];
	  cy[k] = y[i];
	  k++;
	  i++;
        }
      else
        {
	  c[k] = a[j];
	  cx[k] = x[j];
	  cy[k] = y[j];
	  k++;
	  j++;
        }
    }
  while (i <= mid)
    {
      c[k] = a[i];
      cx[k] = x[i];
      cy[k] = y[i];
      k++;
      i++;
    }
  while (j <= high)
    {
      c[k] = a[j];
      cx[k] = x[j];
      cy[k] = y[j];
      k++;
      j++;
    }
  for (i = low; i < k; i++)
    {
      a[i] =  c[i];
      x[i] = cx[i];
      y[i] = cy[i];
    }
}
void mergesort(Determinant *a, int low, int high, double* x, double* y, Determinant* c, double* cx, double *cy)
{
  int mid;
  if (low < high)
    {
      mid=(low+high)/2;
      mergesort(a,low,mid, x, y, c, cx, cy);
      mergesort(a,mid+1,high, x, y, c, cx, cy);
      merge(a,low,high,mid, x, y, c, cx, cy);
    }
  return;
}

//for each element in ci stochastic round to eps and put all the nonzero elements in newWts and their corresponding
//indices in Sample1
int sample_round(MatrixXd& ci, double eps, std::vector<int>& Sample1, std::vector<double>& newWts){
  for (int i=0; i<ci.rows(); i++) {
    if (abs(ci(i,0)) > eps) {
      Sample1.push_back(i);
      newWts.push_back(ci(i,0));
    }
    else if (((double) rand() / (RAND_MAX))*eps < abs(ci(i,0))) {
      Sample1.push_back(i);
      newWts.push_back( eps*ci(i,0)/abs(ci(i,0)));
    }
  }
}

void setUpAliasMethod(MatrixXd& ci, double& cumulative, std::vector<int>& alias, std::vector<double>& prob) {
  alias.resize(ci.rows());
  prob.resize(ci.rows());
  
  std::vector<double> larger, smaller;
  for (int i=0; i<ci.rows(); i++) {
    prob[i] = abs(ci(i,0))*ci.rows()/cumulative;
    if (prob[i] < 1.0)
      smaller.push_back(i);
    else
      larger.push_back(i);
  }

  while (larger.size() >0 && smaller.size() >0) {
    int l = larger[larger.size()-1]; larger.pop_back();
    int s = smaller[smaller.size()-1]; smaller.pop_back();
    
    alias[s] = l;
    prob[l] = prob[l] - (1.0 - prob[s]);
    if (prob[l] < 1.0)
      smaller.push_back(l);
    else
      larger.push_back(l);
  }
}

int sample_N2_alias(MatrixXd& ci, double& cumulative, std::vector<int>& Sample1, std::vector<double>& newWts, std::vector<int>& alias, std::vector<double>& prob) {

  int niter = Sample1.size(); //Sample1.resize(0); newWts.resize(0);

  int sampleIndex = 0;
  for (int index = 0; index<niter; index++) {
    int detIndex = floor(1.* ((double) rand() / (RAND_MAX))*ci.rows() );

    double rand_no = ((double) rand()/ (RAND_MAX));
    if (rand_no >= prob[detIndex]) 
      detIndex = alias[detIndex];

    std::vector<int>::iterator it = find(Sample1.begin(), Sample1.end(), detIndex);
    if (it == Sample1.end()) {
      Sample1[sampleIndex] = detIndex;
      newWts[sampleIndex] = ci(detIndex,0) < 0. ? -cumulative : cumulative;
      sampleIndex++;
    }
    else {
      newWts[distance(Sample1.begin(), it) ] += ci(detIndex,0) < 0. ? -cumulative : cumulative;
    }
  }

  for (int i=0; i<niter; i++)
    newWts[i] /= niter;
  return sampleIndex;
}

int sample_N2(MatrixXd& ci, double& cumulative, std::vector<int>& Sample1, std::vector<double>& newWts){
  double prob = 1.0;
  int niter = Sample1.size();
  int totalSample = 0;
  for (int index = 0; index<niter; ) {

    double rand_no = ((double) rand() / (RAND_MAX))*cumulative;
    for (int i=0; i<ci.rows(); i++) {
      if (rand_no < abs(ci(i,0))) {
	std::vector<int>::iterator it = find(Sample1.begin(), Sample1.end(), i);
	if (it == Sample1.end()) {
	  Sample1[index] = i;
	  newWts[index] = ci(i,0) < 0. ? -cumulative : cumulative;
	  index++; totalSample++;
	}
	else {
	  //if (Sample1[distance(Sample1.begin(), it) ] != i) {cout << i<<" "<<*it <<endl; exit(0);}
	  //int oldindex = distance(Sample1.begin(), it);
	  //cout << oldindex<<"  "<<newWts[oldindex]<<"  "<<cumulative/Sample1.size()<<"  "<<Sample1.size()<<endl;
	  newWts[ distance(Sample1.begin(), it) ] += ci(i,0) < 0. ? -cumulative : cumulative;
	  totalSample++;
	}
	break;
      }
      rand_no -= abs(ci(i,0));
    }
  }

  for (int i=0; i<niter; i++)
    newWts[i] /= totalSample;
  return totalSample;
}

int sample_N(MatrixXd& ci, double& cumulative, std::vector<int>& Sample1, std::vector<double>& newWts){
  double prob = 1.0;
  int niter = Sample1.size();

  for (int index = 0; index<niter; ) {

    double rand_no = ((double) rand() / (RAND_MAX))*cumulative;
    for (int i=0; i<ci.rows(); i++) {
      if (rand_no < abs(ci(i,0))) {

	Sample1[index] = i;
	newWts[index] = ci(i,0) < 0. ? -cumulative/Sample1.size() : cumulative/Sample1.size();
	index++;
	break;
      }
      rand_no -= abs(ci(i,0));
    }
  }

}


void CIPSIbasics::DoBatchDeterministic(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
				       twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) {
  int nblocks = schd.nblocks;
  std::vector<int> blockSizes(nblocks,0);
  for (int i=0; i<nblocks; i++) {
    if (i!=nblocks-1)
      blockSizes[i] = Dets.size()/nblocks;
    else
      blockSizes[i] = Dets.size() - (nblocks-1)*Dets.size()/(nblocks);
  }
  int norbs = Determinant::norbs;
  std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
  double AvgenergyEN = 0.0;

#pragma omp parallel for schedule(dynamic) 
  for (int inter1 = 0; inter1 < nblocks; inter1++) {

    vector<int> psiClosed(nelec,0); 
    vector<int> psiOpen(norbs-nelec,0);

    std::vector<double> wts1(blockSizes[inter1]);
    std::vector<int> Sample1(blockSizes[inter1]);
    for (int i=0; i<wts1.size(); i++) {
      wts1[i] = ci(i+(inter1)*Dets.size()/nblocks ,0);
      Sample1[i] = i+(inter1)*Dets.size()/nblocks ;
    }

    map<Determinant, pair<double,double> > Psi1ab; 
    for (int i=0; i<Sample1.size(); i++) {
      int I = Sample1[i];
      CIPSIbasics::getDeterminants(Dets[I], abs(schd.epsilon2/ci(I,0)), wts1[i], 0.0, I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, schd);
    }

    double energyEN = 0.0;
    for (map<Determinant, pair<double, double> >::iterator it = Psi1ab.begin(); it != Psi1ab.end(); it++) {
      it->first.getOpenClosed(psiOpen, psiClosed);
      energyEN += it->second.first*it->second.first/(Energy(psiClosed, nelec, I1, I2, coreE)-E0); 
    }
    

    for (int i=Sample1[Sample1.size()-1]+1; i<Dets.size(); i++) {
      CIPSIbasics::getDeterminants(Dets[i], abs(schd.epsilon2/ci(i,0)), 0.0, ci(i,0), I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, schd);
      if (i%1000 == 0 && omp_get_thread_num() == 0) cout <<i <<" out of "<<Dets.size()-Sample1.size()<<endl; 
    }

    for (map<Determinant, pair<double, double> >::iterator it = Psi1ab.begin(); it != Psi1ab.end(); it++) {
      it->first.getOpenClosed(psiOpen, psiClosed);
      energyEN += 2.*it->second.first*it->second.second/(Energy(psiClosed, nelec, I1, I2, coreE)-E0); 
    }

#pragma omp critical 
    {
      AvgenergyEN += energyEN;
      
      std::cout << format("%6i  %14.8f   %10.2f  %10i %4i") 
	%(inter1) % (E0-AvgenergyEN) % (getTime()-startofCalc) % inter1 % (omp_get_thread_num());
      cout << endl;
    }
  }
  
  {
    std::cout <<"FINAL ANSWER "<<endl;
    std::cout << format("%4i  %14.8f   %10.2f  %10i %4i") 
      %(nblocks) % (E0-AvgenergyEN) % (getTime()-startofCalc) % nblocks % (0);
    cout << endl;
  }

}



void CIPSIbasics::DoPerturbativeStochastic2(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
					   twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) {

  cout << "This function is most likely broken, dont use it. Use the single list method instead!!!"<<endl;
  exit(0);
  boost::mpi::communicator world;
  char file [5000];
  sprintf (file, "output-%d.bkp" , world.rank() );
  std::ofstream ofs(file);

    int norbs = Determinant::norbs;
    std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
    int niter = 10000;
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

#pragma omp parallel for schedule(dynamic) 
    for (int iter=0; iter<niter; iter++) {
      //cout << norbs<<"  "<<nelec<<endl;
      char psiArray[norbs]; 
      vector<int> psiClosed(nelec,0); 
      vector<int> psiOpen(norbs-nelec,0);
      //char psiArray[norbs];
      std::vector<double> wts1(Nsample,0.0), wts2(Nsample,0.0); std::vector<int> Sample1(Nsample,-1), Sample2(Nsample,-1);
      //wts1.reserve(Nsample); wts2.reserve(Nsample); Sample1.reserve(Nsample); Sample2.reserve(Nsample);
      

      sample_N2(ci, cumulative, Sample1, wts1);
      sample_N2(ci, cumulative, Sample2, wts2);

      double norm = 0.0;
      for (int i=0; i<Sample1.size(); i++) {
	double normi = 0.0;
	for (int j=0; j<Sample2.size(); j++)
	  if (Sample2[j] == Sample1[i]) normi += wts1[i]*wts2[j];
	norm += normi;
      }
      
      map<Determinant, pair<double,double> > Psi1ab; 
      for (int i=0; i<Sample1.size(); i++) {
	int I = Sample1[i];
	std::vector<int>::iterator it = find(Sample2.begin(), Sample2.end(), I);
	if (it != Sample2.end())
	  CIPSIbasics::getDeterminants(Dets[I], abs(schd.epsilon2/ci(I,0)), wts1[i], wts2[ distance(Sample2.begin(), it) ], I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, schd);
	else
	  CIPSIbasics::getDeterminants(Dets[I], abs(schd.epsilon2/ci(I,0)), wts1[i], 0.0, I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, schd);
      }

      for (int i=0; i<Sample2.size(); i++) {
	int I = Sample2[i];
	std::vector<int>::iterator it = find(Sample1.begin(), Sample1.end(), I);
	if (it == Sample1.end())
	  CIPSIbasics::getDeterminants(Dets[I], abs(schd.epsilon2/ci(I,0)), 0.0, wts2[i], I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, schd);
      }


      double energyEN = 0.0;
      for (map<Determinant, pair<double, double> >::iterator it = Psi1ab.begin(); it != Psi1ab.end(); it++) {
	it->first.getOpenClosed(psiOpen, psiClosed);
	energyEN += it->second.first*it->second.second/(Energy(psiClosed, nelec, I1, I2, coreE)-E0); 
      }
      sampleSize = Sample1.size();
      AverageDen += norm;
#pragma omp critical 
      {
	if (mpigetrank() == 0) {
	  AvgenergyEN += energyEN; currentIter++;
	  std::cout << format("%6i  %14.8f  %14.8f %14.8f   %10.2f  %10i %4i") 
	    %(currentIter) % (E0-energyEN) % (norm) % (E0-AvgenergyEN/currentIter) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	  cout << endl;
	}
	else {
	  AvgenergyEN += energyEN; currentIter++;
	  ofs << format("%6i  %14.8f  %14.8f %14.8f   %10.2f  %10i %4i") 
	    %(currentIter) % (E0-energyEN) % (norm) % (E0-AvgenergyEN/AverageDen) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	  ofs << endl;

	}
      }
    }
    ofs.close();

}

void CIPSIbasics::DoPerturbativeStochastic2SingleListDoubleEpsilon2(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
								    twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) {

  boost::mpi::communicator world;
  char file [5000];
  sprintf (file, "output-%d.bkp" , world.rank() );
  std::ofstream ofs(file);

  double epsilon2 = schd.epsilon2;
  schd.epsilon2 = schd.epsilon2Large;
  double EptLarge = DoPerturbativeDeterministic(Dets, ci, E0, I1, I2, I2HB, irrep, schd, coreE, nelec);
  schd.epsilon2 = epsilon2;

  int norbs = Determinant::norbs;
  std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
  int niter = 1000000;
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
  setUpAliasMethod(ci, cumulative, alias, prob);
#pragma omp parallel for schedule(dynamic) 
  for (int iter=0; iter<niter; iter++) {
      //cout << norbs<<"  "<<nelec<<endl;
    char psiArray[norbs]; 
    vector<int> psiClosed(nelec,0); 
    vector<int> psiOpen(norbs-nelec,0);
    //char psiArray[norbs];
    std::vector<double> wts1(Nsample,0.0); std::vector<int> Sample1(Nsample,-1);
    
    //int Nmc = sample_N2(ci, cumulative, Sample1, wts1);
    int distinctSample = sample_N2_alias(ci, cumulative, Sample1, wts1, alias, prob);
    int Nmc = Nsample;
    double norm = 0.0;
    
    map<Determinant, std::tuple<double,double,double, double, double> > Psi1ab; 
    for (int i=0; i<distinctSample; i++) {
      int I = Sample1[i];
      CIPSIbasics::getDeterminants2Epsilon(Dets[I], abs(schd.epsilon2/ci(I,0)), abs(schd.epsilon2Large/ci(I,0)), wts1[i], ci(I,0), I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, schd, Nmc, nelec);
    }
    
    
    double energyEN = 0.0, energyENLargeEps = 0.0;
    for (map<Determinant, std::tuple<double, double, double, double, double> >::iterator it = Psi1ab.begin(); it != Psi1ab.end(); it++) {
      it->first.getOpenClosed(psiOpen, psiClosed);
      double A = std::get<0>(it->second), B = std::get<1>(it->second), C = std::get<2>(it->second);
      double D = std::get<3>(it->second), E = std::get<4>(it->second);
      energyEN += (A*A *Nmc/(Nmc-1)  - B)/(C-E0); 
      energyENLargeEps += (D*D *Nmc/(Nmc-1)  - E)/(C-E0); 
    }
    sampleSize = distinctSample;
    
#pragma omp critical 
    {
      if (mpigetrank() == 0) {
	AvgenergyEN += -energyEN+energyENLargeEps-EptLarge; currentIter++;
	std::cout << format("%6i  %14.8f  %14.8f %14.8f   %10.2f  %10i %4i") 
	  %(currentIter) % (E0-energyEN+energyENLargeEps-EptLarge) % (norm) % (E0+AvgenergyEN/currentIter) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	cout << endl;
      }
      else {
	AvgenergyEN += -energyEN+energyENLargeEps-EptLarge; currentIter++;
	ofs << format("%6i  %14.8f  %14.8f %14.8f   %10.2f  %10i %4i") 
	  %(currentIter) % (E0-energyEN+energyENLargeEps-EptLarge) % (norm) % (E0+AvgenergyEN/AverageDen) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	ofs << endl;
	
      }
    }
  }
  ofs.close();
  
}


void CIPSIbasics::DoPerturbativeStochastic2SingleList(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
						      twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) {

  boost::mpi::communicator world;
  char file [5000];
  sprintf (file, "output-%d.bkp" , world.rank() );
  std::ofstream ofs(file);

    int norbs = Determinant::norbs;
    std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
    int niter = 1000000;
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
    setUpAliasMethod(ci, cumulative, alias, prob);
#pragma omp parallel for schedule(dynamic) 
    for (int iter=0; iter<niter; iter++) {
      //cout << norbs<<"  "<<nelec<<endl;
      char psiArray[norbs]; 
      vector<int> psiClosed(nelec,0); 
      vector<int> psiOpen(norbs-nelec,0);
      //char psiArray[norbs];
      std::vector<double> wts1(Nsample,0.0); std::vector<int> Sample1(Nsample,-1);

      //int Nmc = sample_N2(ci, cumulative, Sample1, wts1);
      int distinctSample = sample_N2_alias(ci, cumulative, Sample1, wts1, alias, prob);
      int Nmc = Nsample;
      double norm = 0.0;
      
      map<Determinant, std::tuple<double,double,double> > Psi1ab; 
      for (int i=0; i<distinctSample; i++) {
       int I = Sample1[i];
       CIPSIbasics::getDeterminants(Dets[I], abs(schd.epsilon2/ci(I,0)), wts1[i], ci(I,0), I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, schd, Nmc, nelec);
      }


      double energyEN = 0.0;
      for (map<Determinant, std::tuple<double, double, double> >::iterator it = Psi1ab.begin(); it != Psi1ab.end(); it++) {
	it->first.getOpenClosed(psiOpen, psiClosed);
	double A = std::get<0>(it->second), B = std::get<1>(it->second), C = std::get<2>(it->second);
	energyEN += (A*A *Nmc/(Nmc-1)  - B)/(C-E0); 
      }
      sampleSize = distinctSample;

#pragma omp critical 
      {
	if (mpigetrank() == 0) {
	  AvgenergyEN += energyEN; currentIter++;
	  std::cout << format("%6i  %14.8f  %14.8f %14.8f   %10.2f  %10i %4i") 
	    %(currentIter) % (E0-energyEN) % (norm) % (E0-AvgenergyEN/currentIter) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	  cout << endl;
	}
	else {
	  AvgenergyEN += energyEN; currentIter++;
	  ofs << format("%6i  %14.8f  %14.8f %14.8f   %10.2f  %10i %4i") 
	    %(currentIter) % (E0-energyEN) % (norm) % (E0-AvgenergyEN/AverageDen) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	  ofs << endl;

	}
      }
    }
    ofs.close();

}


void CIPSIbasics::DoPerturbativeStochastic(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
					   twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) {

  boost::mpi::communicator world;
  char file [5000];
  sprintf (file, "output-%d.bkp" , world.rank() );
  std::ofstream ofs(file);

    int norbs = Determinant::norbs;
    std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
    int niter = 10000;
    //double eps = 0.001;
    double AvgenergyEN = 0.0;
    double AverageDen = 0.0;
    int currentIter = 0;
    int sampleSize = 0;
    int num_thrds = omp_get_max_threads();

#pragma omp parallel for schedule(dynamic) 
    for (int iter=0; iter<niter; iter++) {
      //cout << norbs<<"  "<<nelec<<endl;
      char psiArray[norbs]; 
      vector<int> psiClosed(nelec,0); 
      vector<int> psiOpen(norbs-nelec,0);
      //char psiArray[norbs];
      std::vector<double> wts1, wts2; std::vector<int> Sample1, Sample2;
      wts1.reserve(1000); wts2.reserve(1000); Sample1.reserve(1000); Sample2.reserve(1000);
      
      Sample1.resize(0); wts1.resize(0); Sample2.resize(0); wts2.resize(0);
      sample_round(ci, schd.eps, Sample1, wts1);
      sample_round(ci, schd.eps, Sample2, wts2);

      double norm = 0.0;
      for (int i=0; i<Sample1.size(); i++) {
	for (int j=0; j<Sample2.size(); j++)
	  if (Sample2[j] == Sample1[i]) {
	    norm += wts1[i]*wts2[j];
	    break;
	  }
      }
      
      map<Determinant, pair<double,double> > Psi1ab; 
      for (int i=0; i<Sample1.size(); i++) {
	int I = Sample1[i];
	//CIPSIbasics::getDeterminants(Dets[I], abs(schd.epsilon2/ci(I,0)), wts1[i], 0.0, I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, schd);
	std::vector<int>::iterator it = find(Sample2.begin(), Sample2.end(), I);
	if (it != Sample2.end())
	  CIPSIbasics::getDeterminants(Dets[I], abs(schd.epsilon2/ci(I,0)), wts1[i], wts2[ distance(Sample2.begin(), it) ], I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, schd);
	else
	  CIPSIbasics::getDeterminants(Dets[I], abs(schd.epsilon2/ci(I,0)), wts1[i], 0.0, I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, schd);
      }


      for (int i=0; i<Sample2.size(); i++) {
	int I = Sample2[i];
	std::vector<int>::iterator it = find(Sample1.begin(), Sample1.end(), I);
	if (it == Sample1.end())
	  CIPSIbasics::getDeterminants(Dets[I], abs(schd.epsilon2/ci(I,0)), 0.0, wts2[i], I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, schd);
      }


      double energyEN = 0.0;
      for (map<Determinant, pair<double, double> >::iterator it = Psi1ab.begin(); it != Psi1ab.end(); it++) {
	it->first.getOpenClosed(psiOpen, psiClosed);
	energyEN += it->second.first*it->second.second/(Energy(psiClosed, nelec, I1, I2, coreE)-E0); 
      }
      sampleSize = Sample1.size();

#pragma omp critical 
      {
	if (mpigetrank() == 0) {
	  AvgenergyEN += energyEN; currentIter++;
	  std::cout << format("%6i  %14.8f  %14.8f %14.8f   %10.2f  %10i %4i") 
	    %(currentIter) % (E0-energyEN) % (norm) % (E0-AvgenergyEN/currentIter) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	  cout << endl;
	}
	else {
	  AvgenergyEN += energyEN; currentIter++;
	  ofs << format("%6i  %14.8f  %14.8f %14.8f   %10.2f  %10i %4i") 
	    %(currentIter) % (E0-energyEN) % (norm) % (E0-AvgenergyEN/AverageDen) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	  ofs << endl;

	}

	//AverageDen += norm;
	//AvgenergyEN += energyEN; currentIter++;
	//std::cout << format("%6i  %14.8f  %14.8f  %14.8f   %10.2f  %10i %4i") 
	//%(currentIter) % (E0-energyEN) % (norm) % (E0-AvgenergyEN/currentIter) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	//cout << endl;
	//%(currentIter) % (E0-AvgenergyEN/currentIter) % (norm) % (E0-AvgenergyEN/AverageDen) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	//cout << endl;
      }
    }
    ofs.close();
}


void CIPSIbasics::DoPerturbativeStochasticSingleList(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
						     twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) {

  boost::mpi::communicator world;
  char file [5000];
  sprintf (file, "output-%d.bkp" , world.rank() );
  std::ofstream ofs(file);

    int norbs = Determinant::norbs;
    std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
    int niter = 10000;
    //double eps = 0.001;
    double AvgenergyEN = 0.0;
    double AverageDen = 0.0;
    int currentIter = 0;
    int sampleSize = 0;
    int num_thrds = omp_get_max_threads();

#pragma omp parallel for schedule(dynamic) 
    for (int iter=0; iter<niter; iter++) {
      //cout << norbs<<"  "<<nelec<<endl;
      char psiArray[norbs]; 
      vector<int> psiClosed(nelec,0); 
      vector<int> psiOpen(norbs-nelec,0);
      //char psiArray[norbs];
      std::vector<double> wts1; std::vector<int> Sample1;
      wts1.reserve(1000);  Sample1.reserve(1000);
      
      Sample1.resize(0); wts1.resize(0);
      sample_round(ci, schd.eps, Sample1, wts1);
      
      map<Determinant, pair<double,double> > Psi1ab; 
      for (int i=0; i<Sample1.size(); i++) {
	int I = Sample1[i];
	CIPSIbasics::getDeterminants(Dets[I], abs(schd.epsilon2/ci(I,0)), wts1[i], ci(I,0), I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, schd);
      }



      double energyEN = 0.0;
      for (map<Determinant, pair<double, double> >::iterator it = Psi1ab.begin(); it != Psi1ab.end(); it++) {
	it->first.getOpenClosed(psiOpen, psiClosed);
	energyEN += (it->second.first*it->second.first - it->second.second)/(Energy(psiClosed, nelec, I1, I2, coreE)-E0); 
      }
      sampleSize = Sample1.size();

#pragma omp critical 
      {
	if (mpigetrank() == 0) {
	  AvgenergyEN += energyEN; currentIter++;
	  std::cout << format("%6i  %14.8f  %14.8f %14.8f   %10.2f  %10i %4i") 
	    %(currentIter) % (E0-energyEN) % (1.0) % (E0-AvgenergyEN/currentIter) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	  cout << endl;
	}
	else {
	  AvgenergyEN += energyEN; currentIter++;
	  ofs << format("%6i  %14.8f  %14.8f %14.8f   %10.2f  %10i %4i") 
	    %(currentIter) % (E0-energyEN) % (1.0) % (E0-AvgenergyEN/AverageDen) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	  ofs << endl;

	}

	//AverageDen += norm;
	//AvgenergyEN += energyEN; currentIter++;
	//std::cout << format("%6i  %14.8f  %14.8f  %14.8f   %10.2f  %10i %4i") 
	//%(currentIter) % (E0-energyEN) % (norm) % (E0-AvgenergyEN/currentIter) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	//cout << endl;
	//%(currentIter) % (E0-AvgenergyEN/currentIter) % (norm) % (E0-AvgenergyEN/AverageDen) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	//cout << endl;
      }
    }
    ofs.close();
}


class sort_indices
{
private:
  Determinant* mparr;
public:
  sort_indices(Determinant* parr) : mparr(parr) {}
  bool operator()(int i, int j) const { return mparr[i]<mparr[j]; }
};

double CIPSIbasics::DoPerturbativeDeterministic(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
					      twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) {

    int norbs = Determinant::norbs;
    std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
    char psiArray[norbs]; vector<int> psiClosed(nelec,0), psiOpen(norbs-nelec,0);
    //char psiArray[norbs]; int psiOpen[nelec], psiClosed[norbs-nelec];
    double energyEN = 0.0;
    int num_thrds = omp_get_max_threads();


    std::vector<std::map<Determinant, pair<double,double> > > det_map(num_thrds);
    std::map<Determinant, pair<double,double> >::iterator det_it;

#pragma omp parallel 
    {
      std::vector<Determinant> Psi1(num_thrds); std::vector<double>  numerator(num_thrds);
      std::vector<double>  det_energy(num_thrds);

      for (int i=0; i<Dets.size(); i++) {
	if (i%(mpigetsize()*omp_get_num_threads()) != mpigetrank()*omp_get_num_threads()+omp_get_thread_num()) continue;
	CIPSIbasics::getDeterminants(Dets[i], abs(schd.epsilon2/ci(i,0)), ci(i,0), 0.0, I1, I2, I2HB, irrep, coreE, E0, Psi1, numerator, det_energy, schd,0, nelec);
	if (i%1000 == 0 && omp_get_thread_num()==0) cout <<"# "<<i<<endl;
      }


      if (omp_get_thread_num() == 0) cout << "Before sort "<<getTime()-startofCalc<<endl;
      int index = 0;

      for (int i=0; i<Psi1.size(); i++) {      
	det_it = det_map[omp_get_thread_num()].lower_bound(Psi1[i]);
	if (det_it != det_map[omp_get_thread_num()].end() && Psi1[i] == det_it->first) {
	  det_it->second.first += numerator[i];
	}
	else {
	  det_map[omp_get_thread_num()].insert(det_it, map<Determinant, pair<double, double> >::value_type(Psi1[i], pair<double, double>(numerator[i], det_energy[i]) ));

	}

      }

      if (omp_get_thread_num() == 0) cout << "Unique determinants "<<getTime()-startofCalc<<endl;

      for (int level=0; level<ceil(log2(omp_get_num_threads())); level++) {
#pragma omp barrier
	if (omp_get_thread_num()%ipow(2,level+1) == 0 && omp_get_thread_num() + ipow(2,level) < omp_get_num_threads() ) {

	  int other_thrd = omp_get_thread_num()+ipow(2,level);

	  for (map<Determinant, pair<double, double> >::iterator it=det_map[other_thrd].begin(); it!=det_map[other_thrd].end(); ++it)  {
	    map<Determinant, pair<double, double> >::iterator det_it = det_map[omp_get_thread_num()].lower_bound( it->first );
	    
	    if(det_it != det_map[omp_get_thread_num()].end() && det_it->first == it->first)  
	      det_it->second.first += it->second.first;
	    else
	      det_map[omp_get_thread_num()].insert(det_it, map<Determinant, pair<double, double> >::value_type(it->first, pair<double, double>(it->second.first, it->second.second) ) );
	  }
	  det_map[other_thrd].clear();
	}	  

      }

      if (omp_get_thread_num() == 0) cout << "Merging "<<getTime()-startofCalc<<endl;

    }
    det_map.resize(1);


    double totalPT=0.0;
#pragma omp parallel 
    {
      double PTEnergy = 0.0;
      map<Determinant, pair<double, double> >::iterator map_it = det_map[0].begin();

      vector<Determinant>::iterator vec_it = SortedDets.begin();
      size_t startIter = omp_get_thread_num() * (det_map[0].size()/omp_get_num_threads()) ;
      size_t endIter = startIter +  (det_map[0].size()/omp_get_num_threads()) ;
      if (omp_get_thread_num() == num_thrds-1) endIter = det_map[0].size();

      size_t cnt = 0;
      while(map_it != det_map[0].end() ) {
	if (cnt <startIter) {cnt++; map_it++;}
	else if (cnt >=endIter) break;
	else if ( map_it->first < *vec_it ) {
	  PTEnergy += map_it->second.first*map_it->second.first/(E0-map_it->second.second); 
	  map_it++;
	  cnt++;
	}
	else if (*vec_it < map_it->first && vec_it != SortedDets.end())
	  vec_it++;
	else if (*vec_it < map_it->first && vec_it == SortedDets.end()) {
	  PTEnergy += map_it->second.first*map_it->second.first/(E0-map_it->second.second); 
	  map_it++;
	  cnt++;
	}
	else {
	  vec_it++; map_it++;cnt++;
	}
      }

#pragma omp critical
      {
	totalPT += PTEnergy;
      }
    }

    cout << "Done energy "<<totalPT<<"  "<<getTime()-startofCalc<<endl;
    return energyEN;
}

void MakeHfromHelpers(std::map<HalfDet, std::vector<int> >& BetaN,
		      std::map<HalfDet, std::vector<int> >& AlphaNm1,
		      std::vector<Determinant>& Dets,
		      int StartIndex,
		      std::vector<std::vector<int> >&connections,
		      std::vector<std::vector<double> >& Helements,
		      char* detChar,
		      int Norbs,
		      oneInt& I1,
		      twoInt& I2,
		      double& coreE) {

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
      double hij = Energy(&detChar[norbs*k], Norbs, I1, I2, coreE);
      Helements[k].push_back(hij);
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
      for (int k=localStart; k<detIndex.size(); k++) {
      
	if (detIndex[k]%(nprocs*omp_get_num_threads()) != proc*omp_get_num_threads()+omp_get_thread_num()) continue;
	//if (detIndex[j]%omp_get_num_threads() != omp_get_thread_num()) continue;
	
	for(int j=0; j<k; j++) {
	  size_t J = detIndex[j];size_t K = detIndex[k];
	  if (Dets[J].connected(Dets[K])) 	  {
	      connections[K].push_back(J);
	      //double hij = Hij(&detChar[norbs*J], &detChar[norbs*K], Norbs, I1, I2, coreE);
	      double hij = Hij(Dets[J], Dets[K], I1, I2, coreE);
	      Helements[K].push_back(hij);
	  }
	}
      }
    }
    index++;
    if (index%100000 == 0 && index!= 0) {pout <<"#bn "<<index<<endl;}
  }

  pout <<"# "<< Dets.size()<<"  "<<BetaN.size()<<"  "<<AlphaNm1.size()<<endl;
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
	      double hij = Hij(Dets[J], Dets[K], I1, I2, coreE);
	      //double hij = Hij(&detChar[norbs*J], &detChar[norbs*K], Norbs, I1, I2, coreE);
	      Helements[K].push_back(hij);
	    }
	  }
	}
      }
    }
    index++;
    if (index%100000 == 0 && index!= 0) {pout <<"#an-1 "<<index<<endl;}
  }
  pout << format("#AlphaN-1 %49.2f\n")
      % (getTime()-startofCalc);


    
}
  
void PopulateHelperLists(std::map<HalfDet, std::vector<int> >& BetaN,
			 std::map<HalfDet, std::vector<int> >& AlphaNm1,
			 std::vector<Determinant>& Dets,
			 int StartIndex) {
  pout << format("#Making Helpers %43.2f\n")
      % (getTime()-startofCalc);
  for (int i=StartIndex; i<Dets.size(); i++) {
    HalfDet da = Dets[i].getAlpha(), db = Dets[i].getBeta();

    BetaN[db].push_back(i);

    int norbs = 64*DetLen;
    std::vector<int> closeda(norbs);//, closedb(norbs);
    int ncloseda = da.getClosed(closeda);
    //int nclosedb = db.getClosed(closedb);

    
    for (int j=0; j<ncloseda; j++) {
      da.setocc(closeda[j], false);
      AlphaNm1[da].push_back(i);
      da.setocc(closeda[j], true);
    }
  }

}

//this takes in a ci vector for determinants placed in Dets
//it then does a CIPSI varitional calculation and the resulting
//ci and dets are returned here
//at input usually the Dets will just have a HF or some such determinant
//and ci will be just 1.0
double CIPSIbasics::DoVariational(MatrixXd& ci, vector<Determinant>& Dets, schedule& schd,
				  twoInt& I2, twoIntHeatBath& I2HB, vector<int>& irrep, oneInt& I1, double& coreE) {

  std::map<HalfDet, std::vector<int> > BetaN, AlphaNm1;
  PopulateHelperLists(BetaN, AlphaNm1, Dets, 0);


#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  int num_thrds = omp_get_max_threads();

  std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());

  size_t norbs = 2.*I2.Direct.rows();
  int Norbs = norbs;
  std::vector<char> detChar(norbs); Dets[0].getRepArray(&detChar[0]);
  double E0 = Energy(&detChar[0], Norbs, I1, I2, coreE);

  pout << "#HF = "<<E0<<std::endl;

  //this is essentially the hamiltonian, we have stored it in a sparse format
  std::vector<std::vector<int> > connections(1, std::vector<int>(1,0));
  std::vector<std::vector<double> > Helements(1, std::vector<double>(1,E0));
  if (mpigetrank() != 0) {connections.resize(0); Helements.resize(0);}


  //keep the diagonal energies of determinants so we only have to generated
  //this for the new determinants in each iteration and not all determinants
  MatrixXd diagOld(1,1); diagOld(0,0) = E0;
  int prevSize = 0;

  int iterstart = 0;

  if (schd.restart || schd.fullrestart) {
    bool converged;
    readVariationalResult(iterstart, ci, Dets, SortedDets, diagOld, connections, Helements, E0, converged, schd, BetaN, AlphaNm1);
    pout << format("# %4i  %10.2e  %10.2e   %14.8f  %10.2f\n") 
      %(iterstart) % schd.epsilon1[iterstart] % Dets.size() % E0 % (getTime()-startofCalc);
    iterstart++;
    if (schd.onlyperturbative)
      return E0;

    detChar.resize(Dets.size()*norbs);
    for (int i=0; i<Dets.size(); i++) 
      Dets[i].getRepArray(&detChar[i*norbs]);
    if (converged && !schd.fullrestart) {
      pout << "# restarting from a converged calculation, moving to perturbative part.!!"<<endl;
      return E0;
    }
  }


  //do the variational bit
  for (int iter=iterstart; iter<schd.epsilon1.size(); iter++) {
    double epsilon1 = schd.epsilon1[iter];
    std::vector<map<Determinant, pair<double,double> > > newDets(num_thrds); //also include the connection magnitude so we can calculate the pt

#pragma omp parallel for schedule(static)
    for (int i=0; i<SortedDets.size(); i++) {
      if (i%mpigetsize() != mpigetrank()) continue;
      //if (world.rank() != 0) continue;
      getDeterminants(Dets[i], abs(epsilon1/ci(i,0)), ci(i,0), 0.0, I1, I2, I2HB, irrep, coreE, E0, newDets[omp_get_thread_num()], SortedDets, schd);
    }

    for (int thrd=1; thrd<num_thrds; thrd++) {
      for (map<Determinant, pair<double, double> >::iterator it=newDets[thrd].begin(); it!=newDets[thrd].end(); ++it)  {
	if(newDets[0].find(it->first) == newDets[0].end())
	  newDets[0][it->first].first = it->second.first;
	else
	  newDets[0][it->first].first += it->second.first;
      }
      newDets[thrd].clear();
    }


#ifndef SERIAL
    //mpi send
    if (mpigetrank() != 0) {
      for (int proc = 1; proc<mpigetsize(); proc++) {
	if (proc == mpigetrank()) {
	  size_t newdetssize = newDets[0].size();
	  world.send(0, proc, newdetssize);
	  if (newDets[0].size() > 0) {
	    world.send(0, proc, newDets[0]);
	  }
	  newDets[0].clear();
	}
      }
    }

    //mpi recv
    if (mpigetrank() == 0) {
      for (int proc = 1; proc < world.size(); proc++) {
	map<Determinant, pair<double, double> > newDetsRecv;
	size_t newdetsrecvsize=-1;
	world.recv(proc, proc, newdetsrecvsize);
	if (newdetsrecvsize > 0) {
	  world.recv(proc, proc, newDetsRecv);
	  for (map<Determinant, pair<double, double> >::iterator it=newDetsRecv.begin(); it!=newDetsRecv.end(); ++it) {
	    if(newDets[0].find(it->first) == newDets[0].end())
	      newDets[0][it->first].first = it->second.first;
	    else
	      newDets[0][it->first].first += it->second.first;
	  }
	}
      }
    }

    mpi::broadcast(world, newDets[0], 0);
#endif

    for (map<Determinant, pair<double, double> >::iterator it=newDets[0].begin(); it!=newDets[0].end(); ++it) {
      if (schd.excitation != 1000 ) {
	if (it->first.ExcitationDistance(Dets[0]) > schd.excitation) continue;
      }
      Dets.push_back(it->first);
    }

    //now diagonalize the hamiltonian
    detChar.resize(norbs* Dets.size()); 
    MatrixXd X0(Dets.size(), 1); X0.setZero(Dets.size(),1); X0.block(0,0,ci.rows(),1) = 1.*ci; 
    MatrixXd diag(Dets.size(), 1); diag.setZero(diag.size(),1);
    if (mpigetrank() == 0) diag.block(0,0,ci.rows(),1)= 1.*diagOld;


    double estimatedCorrection = 0.0;
#pragma omp parallel for schedule(dynamic) reduction(+:estimatedCorrection)
    for (size_t k=SortedDets.size(); k<Dets.size() ; k++) {
      if (k % mpigetsize() != mpigetrank() ) continue;
      Dets[k].getRepArray(&detChar[norbs*k]);
      diag(k,0) = Energy(&detChar[norbs*k], Norbs, I1, I2, coreE);
      //double numerator = newDets[0][Dets[k]].first;
      //estimatedCorrection += numerator*numerator/(diag(k,0) - E0);
      if (k%1000000 == 0 && k!=0) cout <<"#"<< k<<"Hdiag out of "<<Dets.size()<<endl;     
    }
    newDets[0].clear();

#ifndef SERIAL
    //MPI_Allreduce(MPI_IN_PLACE, &estimatedCorrection, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &diag(0,0), diag.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    size_t intdim = (Dets.size()-SortedDets.size())*norbs;
    long  maxint = 26843540; //mpi cannot transfer more than these number of doubles
    long maxIter = intdim/maxint; 
    for (size_t ii=0; ii<maxIter; ii++) {
      MPI_Allreduce(MPI_IN_PLACE, &detChar[SortedDets.size()*norbs+ii*maxint], maxint, MPI_CHAR, MPI_SUM, MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Allreduce(MPI_IN_PLACE, &detChar[SortedDets.size()*norbs+maxIter*maxint], intdim-maxIter*maxint, MPI_CHAR, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    //MPI_Allreduce(MPI_IN_PLACE, &detChar[SortedDets.size()*norbs], (Dets.size()-SortedDets.size())*norbs, MPI_CHAR, MPI_SUM, MPI_COMM_WORLD);
#endif

    connections.resize(Dets.size());
    Helements.resize(Dets.size());


    
    if (true) {
      PopulateHelperLists(BetaN, AlphaNm1, Dets, ci.size());
      MakeHfromHelpers(BetaN, AlphaNm1, Dets, SortedDets.size(), connections, Helements,
		       &detChar[0], norbs, I1, I2, coreE);
    }
    //if (Dets.size() > 10000000) {
    else if (false){
      size_t oldDetSize = SortedDets.size();
      map<Determinant, int> SortedDets;
      cout << "about to sort"<<endl;
      for (int i=0; i<oldDetSize; i++)
	SortedDets[Dets[i]] = i;
      cout << "update connections"<<endl;
      updateConnections(Dets, SortedDets, norbs, I1, I2, coreE, &detChar[0], connections, Helements);
    //update connetions
    }
    else
    {
#pragma omp parallel for schedule(dynamic)
      for (size_t i=0; i<Dets.size() ; i++) {
	if (i%mpigetsize() != mpigetrank()) continue;
	
	for (size_t j=max(SortedDets.size(),i); j<Dets.size(); j++) {
	  if (Dets[i].connected(Dets[j])) {
	    double hij = Hij(&detChar[norbs*i], &detChar[norbs*j], Norbs, I1, I2, coreE);
	    
	    if (abs(hij) > 1.e-10) {
	      connections[i].push_back(j);
	      Helements[i].push_back(hij);
	    }
          }
	}

        if (i%1000000 == 0 && i!=0) cout <<"#"<< i<<" Hij out of "<<Dets.size()<<endl;     

        //if (i%10000) cout << i<<"  "<<Dets.size()<<endl;
      }
    } 
     
    for (size_t i=SortedDets.size(); i<Dets.size(); i++)
      SortedDets.push_back(Dets[i]);
    std::sort(SortedDets.begin(), SortedDets.end());
 

    
    double prevE0 = E0;
    //Hmult H(&detChar[0], norbs, I1, I2, coreE);
    Hmult2 H(connections, Helements);
    
    E0 = davidson(H, X0, diag, 10, schd.davidsonTol, false);
    pout << format("# %4i  %10.2e  %10.2e   %14.8f  %10.2f") 
      %(iter) % epsilon1 % Dets.size() % E0 % (getTime()-startofCalc);
    pout << endl;

    ci.resize(Dets.size(),1); ci = 1.0*X0;
    diagOld.resize(Dets.size(),1); diagOld = 1.0*diag;

    
    if (abs(E0-prevE0) < schd.dE)  {
      writeVariationalResult(iter, ci, Dets, SortedDets, diag, connections, Helements, E0, true, schd, BetaN, AlphaNm1);
      break;
    }
    else {
      if (schd.io) writeVariationalResult(iter, ci, Dets, SortedDets, diag, connections, Helements, E0, false, schd, BetaN, AlphaNm1);
    }


  }
  return E0;

}



void CIPSIbasics::writeVariationalResult(int iter, MatrixXd& ci, vector<Determinant>& Dets, vector<Determinant>& SortedDets,
					 MatrixXd& diag, vector<vector<int> >& connections, vector<vector<double> >& Helements, 
					 double& E0, bool converged, schedule& schd,   
					 std::map<HalfDet, std::vector<int> >& BetaN, 
					 std::map<HalfDet, std::vector<int> >& AlphaNm1) {

#ifndef SERIAL
  boost::mpi::communicator world;
#endif

    pout << format("#Begin writing variational wf %29.2f\n")
      % (getTime()-startofCalc);

    char file [5000];
    sprintf (file, "%s/%d-variational.bkp" , schd.prefix.c_str(), world.rank() );
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << iter <<Dets<<SortedDets;
    int diagrows = diag.rows();
    save << diagrows;
    for (int i=0; i<diag.rows(); i++)
      save << diag(i,0);
    for (int i=0; i<ci.rows(); i++)
      save << ci(i,0);
    save << E0;
    save << converged;
    save << connections<<Helements;
    save << BetaN<< AlphaNm1;
    ofs.close();

    pout << format("#End   writing variational wf %29.2f\n")
      % (getTime()-startofCalc);
}


void CIPSIbasics::readVariationalResult(int& iter, MatrixXd& ci, vector<Determinant>& Dets, vector<Determinant>& SortedDets,
					MatrixXd& diag, vector<vector<int> >& connections, vector<vector<double> >& Helements, 
					double& E0, bool& converged, schedule& schd,
					std::map<HalfDet, std::vector<int> >& BetaN, 
					std::map<HalfDet, std::vector<int> >& AlphaNm1) {


#ifndef SERIAL
  boost::mpi::communicator world;
#endif

    pout << format("#Begin reading variational wf %29.2f\n")
      % (getTime()-startofCalc);

    char file [5000];
    sprintf (file, "%s/%d-variational.bkp" , schd.prefix.c_str(), world.rank() );
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    
    load >> iter >> Dets >> SortedDets ;
    int diaglen;
    load >> diaglen;
    ci.resize(diaglen,1); diag.resize(diaglen,1);
    for (int i=0; i<diag.rows(); i++)
      load >> diag(i,0);
    for (int i=0; i<ci.rows(); i++)
      load >>  ci(i,0);
    load >> E0;
    if (schd.onlyperturbative) {ifs.close();return;}
    load >> converged;

    load >> connections >> Helements;
    load >> BetaN>> AlphaNm1;
    ifs.close();

    pout << format("#End   reading variational wf %29.2f\n")
      % (getTime()-startofCalc);
}


//this function is complicated because I wanted to make it general enough that deterministicperturbative and stochasticperturbative could use the same function
//in stochastic perturbative each determinant in Psi1 can come from the first replica of Psi0 or the second replica of Psi0. that is why you have a pair of doubles associated with each det in Psi1 and we pass ci1 and ci2 which are the coefficients of d in replica 1 and replica2 of Psi0.
void CIPSIbasics::getDeterminants(Determinant& d, double epsilon, double ci1, double ci2, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::vector<Determinant>& dets, std::vector<double>& numerator, std::vector<double>& energy, schedule& schd, int Nmc, int nelec) {

  int norbs = d.norbs;
  char detArray[norbs], diArray[norbs];
  vector<int> closed(nelec,0); 
  vector<int> open(norbs-nelec,0);
  d.getOpenClosed(open, closed);
  int nclosed = nelec;
  int nopen = norbs-nclosed;
  d.getRepArray(detArray);

  double Energyd = Energy(closed, nclosed, int1, int2, coreE);
  

  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    if (open[a]/2 > schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital
    if (irreps[closed[i]/2] != irreps[open[a]/2]) continue;

    double integral = Hij_1Excite(closed[i],open[a],int1,int2, detArray, norbs);

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
    std::pair<int,int> IJpair(max(I,J), min(I,J));
    std::map<std::pair<int,int>, std::multimap<double, std::pair<int,int>, compAbs > >::iterator ints = closed[i]%2==closed[j]%2 ? I2hb.sameSpin.find(IJpair) : I2hb.oppositeSpin.find(IJpair);

    if (true && (ints != I2hb.sameSpin.end() && ints != I2hb.oppositeSpin.end())) { //we have this pair stored in heat bath integrals
      for (std::multimap<double, std::pair<int,int>,compAbs >::reverse_iterator it=ints->second.rbegin(); it!=ints->second.rend(); it++) {
	if (fabs(it->first) <epsilon) break; //if this is small then all subsequent ones will be small
	int a = 2* it->second.first + closed[i]%2, b= 2*it->second.second+closed[j]%2;
	if (a/2 > schd.nvirt+nclosed/2 || b/2 >schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital

	if (!(d.getocc(a) || d.getocc(b))) {
	  dets.push_back(d);
	  Determinant& di = *dets.rbegin();
	  di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);	    

	  double sgn = 1.0;
	  di.parity(a, b, closed[i], closed[j], sgn);
	  numerator.push_back(it->first*sgn*ci1);

	  double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, a, j, b, Energyd);	    
	  energy.push_back(E);

	}
      }
    }
  }
  return;
}


//this function is complicated because I wanted to make it general enough that deterministicperturbative and stochasticperturbative could use the same function
//in stochastic perturbative each determinant in Psi1 can come from the first replica of Psi0 or the second replica of Psi0. that is why you have a pair of doubles associated with each det in Psi1 and we pass ci1 and ci2 which are the coefficients of d in replica 1 and replica2 of Psi0.
void CIPSIbasics::getDeterminants(Determinant& d, double epsilon, double ci1, double ci2, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::vector<Determinant>& dets, schedule& schd, int Nmc, int nelec) {

  int norbs = d.norbs;
  char detArray[norbs], diArray[norbs];
  vector<int> closed(nelec,0); 
  vector<int> open(norbs-nelec,0);
  d.getOpenClosed(open, closed);
  int nclosed = nelec;
  int nopen = norbs-nclosed;
  d.getRepArray(detArray);


  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    if (open[a]/2 > schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital
    if (irreps[closed[i]/2] != irreps[open[a]/2]) continue;

    double integral = Hij_1Excite(closed[i],open[a],int1,int2, detArray, norbs);

    if (fabs(integral) > epsilon ) {
      dets.push_back(d); Determinant& di = *dets.rbegin();
      di.setocc(open[a], true); di.setocc(closed[i],false);
    }
  }
  
  if (fabs(int2.maxEntry) <epsilon) return;

  //#pragma omp parallel for schedule(dynamic)
  for (int ij=0; ij<nclosed*nclosed; ij++) {

    int i=ij/nclosed, j = ij%nclosed;
    if (i<=j) continue;
    int I = closed[i]/2, J = closed[j]/2;
    std::pair<int,int> IJpair(max(I,J), min(I,J));
    std::map<std::pair<int,int>, std::multimap<double, std::pair<int,int>, compAbs > >::iterator ints = closed[i]%2==closed[j]%2 ? I2hb.sameSpin.find(IJpair) : I2hb.oppositeSpin.find(IJpair);

    if (true && (ints != I2hb.sameSpin.end() && ints != I2hb.oppositeSpin.end())) { //we have this pair stored in heat bath integrals
      for (std::multimap<double, std::pair<int,int>,compAbs >::reverse_iterator it=ints->second.rbegin(); it!=ints->second.rend(); it++) {
	if (fabs(it->first) <epsilon) break; //if this is small then all subsequent ones will be small
	int a = 2* it->second.first + closed[i]%2, b= 2*it->second.second+closed[j]%2;
	if (a/2 > schd.nvirt+nclosed/2 || b/2 >schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital

	if (!(d.getocc(a) || d.getocc(b))) {
	  dets.push_back(d);
	  Determinant& di = *dets.rbegin();
	  di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);	    
	}
      }
    }
  }
  return;
}


  

//this function is complicated because I wanted to make it general enough that deterministicperturbative and stochasticperturbative could use the same function
//in stochastic perturbative each determinant in Psi1 can come from the first replica of Psi0 or the second replica of Psi0. that is why you have a pair of doubles associated with each det in Psi1 and we pass ci1 and ci2 which are the coefficients of d in replica 1 and replica2 of Psi0.
void CIPSIbasics::getDeterminants(Determinant& d, double epsilon, double ci1, double ci2, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::map<Determinant, pair<double,double> >& Psi1, std::vector<Determinant>& Psi0, schedule& schd, int Nmc) {

  int norbs = d.norbs;
  int open[norbs], closed[norbs]; char detArray[norbs], diArray[norbs];
  int nclosed = d.getOpenClosed(open, closed);
  int nopen = norbs-nclosed;
  d.getRepArray(detArray);
  
  std::map<Determinant, pair<double,double> >::iterator det_it;
  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    if (open[a]/2 > schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital
    if (irreps[closed[i]/2] != irreps[open[a]/2]) continue;
    double integral = Hij_1Excite(closed[i],open[a],int1,int2, detArray, norbs);
    if (fabs(integral) > epsilon ) {
      Determinant di = d;
      di.setocc(open[a], true); di.setocc(closed[i],false);
      if (!(binary_search(Psi0.begin(), Psi0.end(), di))) {
	det_it = Psi1.find(di);

	if (schd.singleList && schd.SampleN != -1) {
	  if (det_it == Psi1.end()) Psi1[di] = make_pair(integral*ci1, integral*integral*ci1*(ci1*Nmc/(Nmc-1)-ci2));
	  else {det_it->second.first +=integral*ci1;det_it->second.second += integral*integral*ci1*(ci1*Nmc/(Nmc-1)-ci2);}
	}
	else if (schd.singleList && schd.SampleN == -1) {
	  if (det_it == Psi1.end()) Psi1[di] = make_pair(integral*ci1, integral*integral*ci2*ci1*(ci1/ci2-1.));
	  else {det_it->second.first +=integral*ci1;det_it->second.second += integral*integral*ci2*ci1*(ci1/ci2-1.);}
	}
	else {
	  if (det_it == Psi1.end()) Psi1[di] = make_pair(integral*ci1, integral*ci2);
	  else {det_it->second.first +=integral*ci1;det_it->second.second +=integral*ci2;}
	}
      }
    }
  }
  
  if (fabs(int2.maxEntry) <epsilon) return;

  //#pragma omp parallel for schedule(dynamic)
  for (int ij=0; ij<nclosed*nclosed; ij++) {

    int i=ij/nclosed, j = ij%nclosed;
    if (i<=j) continue;
    int I = closed[i]/2, J = closed[j]/2;
    std::pair<int,int> IJpair(max(I,J), min(I,J));
    std::map<std::pair<int,int>, std::multimap<double, std::pair<int,int>, compAbs > >::iterator ints = closed[i]%2==closed[j]%2 ? I2hb.sameSpin.find(IJpair) : I2hb.oppositeSpin.find(IJpair);

    //THERE IS A BUG IN THE CODE WHEN USING HEATBATH INTEGRALS
    if (true && (ints != I2hb.sameSpin.end() && ints != I2hb.oppositeSpin.end())) { //we have this pair stored in heat bath integrals
      for (std::multimap<double, std::pair<int,int>,compAbs >::reverse_iterator it=ints->second.rbegin(); it!=ints->second.rend(); it++) {
	if (fabs(it->first) <epsilon) break; //if this is small then all subsequent ones will be small
	int a = 2* it->second.first + closed[i]%2, b= 2*it->second.second+closed[j]%2;
	if (a/2 > schd.nvirt+nclosed/2 || b/2 >schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital
	//cout << a/2<<"  "<<schd.nvirt<<"  "<<nclosed/2<<endl;
	if (!(d.getocc(a) || d.getocc(b))) {
	  Determinant di = d;
	  di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);	    
	  if (!(binary_search(Psi0.begin(), Psi0.end(), di))) {
	    double sgn = 1.0;
	    {
	      int A = (closed[i]), B = closed[j], I= a, J = b; 
	      sgn = parity(detArray,norbs,A)*parity(detArray,norbs,I)*parity(detArray,norbs,B)*parity(detArray,norbs,J);
	      if (B > J) sgn*=-1 ;
	      if (I > J) sgn*=-1 ;
	      if (I > B) sgn*=-1 ;
	      if (A > J) sgn*=-1 ;
	      if (A > B) sgn*=-1 ;
	      if (A > I) sgn*=-1 ;
	    }

	    det_it = Psi1.find(di);

	    if (schd.singleList && schd.SampleN != -1) {
	      if (det_it == Psi1.end()) Psi1[di] = make_pair(it->first*sgn*ci1, it->first*it->first*ci1*(ci1*Nmc/(Nmc-1)-ci2));
	      else {det_it->second.first +=it->first*sgn*ci1;det_it->second.second += it->first*it->first*ci1*(ci1*Nmc/(Nmc-1)-ci2);}
	    }
	    else if (schd.singleList && schd.SampleN == -1) {
	      if (det_it == Psi1.end()) Psi1[di] = make_pair(it->first*sgn*ci1, it->first*it->first*ci1*(ci1-ci2));
	      else {det_it->second.first +=it->first*sgn*ci1;det_it->second.second += it->first*it->first*ci1*(ci1-ci2);}
	    }
	    else {
	      if (det_it == Psi1.end()) Psi1[di] = make_pair(it->first*sgn*ci1, it->first*sgn*ci2);
	      else {det_it->second.first += it->first*sgn*ci1; det_it->second.second += it->first*sgn*ci2;}
	    }
	  }
	}
      }
    }
  }
  return;
}


//this function is complicated because I wanted to make it general enough that deterministicperturbative and stochasticperturbative could use the same function
//in stochastic perturbative each determinant in Psi1 can come from the first replica of Psi0 or the second replica of Psi0. that is why you have a pair of doubles associated with each det in Psi1 and we pass ci1 and ci2 which are the coefficients of d in replica 1 and replica2 of Psi0.
void CIPSIbasics::getDeterminants(Determinant& d, double epsilon, double ci1, double ci2, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::map<Determinant, std::tuple<double,double,double> >& Psi1, std::vector<Determinant>& Psi0, schedule& schd, int Nmc, int nelec) {

  int norbs = d.norbs;
  vector<int> closed(nelec,0); 
  vector<int> open(norbs-nelec,0);
  d.getOpenClosed(open, closed);
  int nclosed = nelec;
  char detArray[norbs], diArray[norbs];
  int nopen = norbs-nclosed;
  d.getRepArray(detArray);


  double Energyd = Energy(closed, nclosed, int1, int2, coreE);
  
  std::map<Determinant, std::tuple<double,double, double> >::iterator det_it;
  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    if (open[a]/2 > schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital
    if (irreps[closed[i]/2] != irreps[open[a]/2]) continue;

    //double integral = Hij_1Excite(closed[i],open[a],int1,int2, detArray, norbs);
    double integral = d.Hij_1Excite(closed[i], open[a], int1, int2);

    if (fabs(integral) > epsilon ) {
      Determinant di = d;
      di.setocc(open[a], true); di.setocc(closed[i],false);
      if (!(binary_search(Psi0.begin(), Psi0.end(), di))) {
	det_it = Psi1.find(di);

	if (schd.singleList && schd.SampleN != -1) {
	  if (det_it == Psi1.end()) {
	    double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, open[a], Energyd);
	    Psi1[di] = std::tuple<double, double, double>(integral*ci1, integral*integral*ci1*(ci1*Nmc/(Nmc-1)-ci2), E);
	  }
	  else {std::get<0>(det_it->second) +=integral*ci1; std::get<1>(det_it->second) += integral*integral*ci1*(ci1*Nmc/(Nmc-1)-ci2);}
	}
      }
    }
  }
  
  if (fabs(int2.maxEntry) <epsilon) return;

  //#pragma omp parallel for schedule(dynamic)
  for (int ij=0; ij<nclosed*nclosed; ij++) {

    int i=ij/nclosed, j = ij%nclosed;
    if (i<=j) continue;
    int I = closed[i]/2, J = closed[j]/2;
    std::pair<int,int> IJpair(max(I,J), min(I,J));
    std::map<std::pair<int,int>, std::multimap<double, std::pair<int,int>, compAbs > >::iterator ints = closed[i]%2==closed[j]%2 ? I2hb.sameSpin.find(IJpair) : I2hb.oppositeSpin.find(IJpair);

    if (true && (ints != I2hb.sameSpin.end() && ints != I2hb.oppositeSpin.end())) { //we have this pair stored in heat bath integrals
      for (std::multimap<double, std::pair<int,int>,compAbs >::reverse_iterator it=ints->second.rbegin(); it!=ints->second.rend(); it++) {
	if (fabs(it->first) <epsilon) break; //if this is small then all subsequent ones will be small
	int a = 2* it->second.first + closed[i]%2, b= 2*it->second.second+closed[j]%2;
	if (a/2 > schd.nvirt+nclosed/2 || b/2 >schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital


	//cout << a/2<<"  "<<schd.nvirt<<"  "<<nclosed/2<<endl;
	if (!(d.getocc(a) || d.getocc(b))) {
	  Determinant di = d;
	  di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);	    
	  if (!(binary_search(Psi0.begin(), Psi0.end(), di))) {

	    double sgn = 1.0;
	    di.parity(a, b, closed[i], closed[j], sgn);

	    det_it = Psi1.find(di);


	    if (schd.singleList && schd.SampleN != -1) {
	      if (det_it == Psi1.end()) {
		double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, a, j, b, Energyd);	    
		Psi1[di] = std::tuple<double,double,double>(it->first*sgn*ci1, it->first*it->first*ci1*(ci1*Nmc/(Nmc-1)-ci2),E);
	      }
	      else {std::get<0>(det_it->second) +=it->first*sgn*ci1;std::get<1>(det_it->second) += it->first*it->first*ci1*(ci1*Nmc/(Nmc-1)-ci2);}
	    }
	  }
	}
      }
    }
  }
  return;
}


//this function is complicated because I wanted to make it general enough that deterministicperturbative and stochasticperturbative could use the same function
//in stochastic perturbative each determinant in Psi1 can come from the first replica of Psi0 or the second replica of Psi0. that is why you have a pair of doubles associated with each det in Psi1 and we pass ci1 and ci2 which are the coefficients of d in replica 1 and replica2 of Psi0.
void CIPSIbasics::getDeterminants2Epsilon(Determinant& d, double epsilon, double epsilonLarge, double ci1, double ci2, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::map<Determinant, std::tuple<double, double,double,double,double> >& Psi1, std::vector<Determinant>& Psi0, schedule& schd, int Nmc, int nelec) {

  int norbs = d.norbs;
  vector<int> closed(nelec,0); 
  vector<int> open(norbs-nelec,0);
  d.getOpenClosed(open, closed);
  int nclosed = nelec;
  char detArray[norbs], diArray[norbs];
  int nopen = norbs-nclosed;
  d.getRepArray(detArray);


  double Energyd = Energy(closed, nclosed, int1, int2, coreE);
  
  std::map<Determinant, std::tuple<double, double, double,double, double> >::iterator det_it;
  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    if (open[a]/2 > schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital
    if (irreps[closed[i]/2] != irreps[open[a]/2]) continue;
    double integral = Hij_1Excite(closed[i],open[a],int1,int2, detArray, norbs);
    if (fabs(integral) > epsilon ) {
      Determinant di = d;
      di.setocc(open[a], true); di.setocc(closed[i],false);
      if (!(binary_search(Psi0.begin(), Psi0.end(), di))) {
	det_it = Psi1.find(di);

	if (schd.singleList && schd.SampleN != -1) {
	  if (det_it == Psi1.end()) {
	    double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, open[a], Energyd);
	    Psi1[di] = std::tuple<double, double, double, double, double>(integral*ci1, integral*integral*ci1*(ci1*Nmc/(Nmc-1)-ci2), E,0.0,0.0);
	  }
	  else {std::get<0>(det_it->second) +=integral*ci1; std::get<1>(det_it->second) += integral*integral*ci1*(ci1*Nmc/(Nmc-1)-ci2);}
	}

	if (fabs(integral) > epsilonLarge ) {
	  det_it = Psi1.find(di);
	  std::get<3>(det_it->second) +=integral*ci1; std::get<4>(det_it->second) += integral*integral*ci1*(ci1*Nmc/(Nmc-1)-ci2);
	}

      }
    }
  }
  
  if (fabs(int2.maxEntry) <epsilon) return;

  //#pragma omp parallel for schedule(dynamic)
  for (int ij=0; ij<nclosed*nclosed; ij++) {

    int i=ij/nclosed, j = ij%nclosed;
    if (i<=j) continue;
    int I = closed[i]/2, J = closed[j]/2;
    std::pair<int,int> IJpair(max(I,J), min(I,J));
    std::map<std::pair<int,int>, std::multimap<double, std::pair<int,int>, compAbs > >::iterator ints = closed[i]%2==closed[j]%2 ? I2hb.sameSpin.find(IJpair) : I2hb.oppositeSpin.find(IJpair);

    //THERE IS A BUG IN THE CODE WHEN USING HEATBATH INTEGRALS
    if (true && (ints != I2hb.sameSpin.end() && ints != I2hb.oppositeSpin.end())) { //we have this pair stored in heat bath integrals
      for (std::multimap<double, std::pair<int,int>,compAbs >::reverse_iterator it=ints->second.rbegin(); it!=ints->second.rend(); it++) {
	if (fabs(it->first) <epsilon) break; //if this is small then all subsequent ones will be small
	int a = 2* it->second.first + closed[i]%2, b= 2*it->second.second+closed[j]%2;
	if (a/2 > schd.nvirt+nclosed/2 || b/2 >schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital


	//cout << a/2<<"  "<<schd.nvirt<<"  "<<nclosed/2<<endl;
	if (!(d.getocc(a) || d.getocc(b))) {
	  Determinant di = d;
	  di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);	    
	  if (!(binary_search(Psi0.begin(), Psi0.end(), di))) {

	    double sgn = 1.0;
	    {
	      int A = (closed[i]), B = closed[j], I= a, J = b; 
	      sgn = parity(detArray,norbs,A)*parity(detArray,norbs,I)*parity(detArray,norbs,B)*parity(detArray,norbs,J);
	      if (B > J) sgn*=-1 ;
	      if (I > J) sgn*=-1 ;
	      if (I > B) sgn*=-1 ;
	      if (A > J) sgn*=-1 ;
	      if (A > B) sgn*=-1 ;
	      if (A > I) sgn*=-1 ;
	    }

	    det_it = Psi1.find(di);


	    if (schd.singleList && schd.SampleN != -1) {
	      if (det_it == Psi1.end()) {
		double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, a, j, b, Energyd);	    
		Psi1[di] = std::tuple<double,double,double,double,double>(it->first*sgn*ci1, it->first*it->first*ci1*(ci1*Nmc/(Nmc-1)-ci2),E,0.0,0.0);
	      }
	      else {std::get<0>(det_it->second) +=it->first*sgn*ci1;std::get<1>(det_it->second) += it->first*it->first*ci1*(ci1*Nmc/(Nmc-1)-ci2);}
	    }

	    if (fabs(it->first) > epsilonLarge ) {
	      det_it = Psi1.find(di);
	      std::get<3>(det_it->second) +=it->first*sgn*ci1;std::get<4>(det_it->second) += it->first*it->first*ci1*(ci1*Nmc/(Nmc-1)-ci2);
	    }

	  }
	}
      }
    }
  }
  return;
}



void CIPSIbasics::updateConnections(vector<Determinant>& Dets, map<Determinant,int>& SortedDets, int norbs, oneInt& int1, twoInt& int2, double coreE, char* detArray, vector<vector<int> >& connections, vector<vector<double> >& Helements) {
  size_t prevSize = SortedDets.size();
  size_t Norbs = norbs;
  for (size_t i=prevSize; i<Dets.size(); i++) {
    SortedDets[Dets[i]] = i;
    connections[i].push_back(i);
    Helements[i].push_back(Energy(&detArray[i*Norbs], norbs, int1, int2, coreE));
  }

#pragma omp parallel for schedule(dynamic)
  for (size_t x=prevSize; x<Dets.size(); x++) {
    Determinant d = Dets[x];
    int open[norbs], closed[norbs]; 
    int nclosed = d.getOpenClosed(open, closed);
    int nopen = norbs-nclosed;
    
    if (x%10000 == 0) cout <<"update connections "<<x<<" out of "<<Dets.size()-prevSize<<endl;
    //loop over all single excitation and find if they are present in the list
    //on or before the current determinant
    for (int ia=0; ia<nopen*nclosed; ia++){
      int i=ia/nopen, a=ia%nopen;
      Determinant di = d;
      di.setocc(open[a], true); di.setocc(closed[i],false);
      
      map<Determinant, int>::iterator it = SortedDets.find(di);
      if (it != SortedDets.end()) {
	int y = it->second;
	if (y <= x) { //avoid double counting
	  double integral = Hij_1Excite(closed[i],open[a],int1,int2, &detArray[x*Norbs], norbs);
	  if (abs(integral) > 1.e-8) {
	    connections[x].push_back(y);
	    Helements[x].push_back(integral);
	  }
	  //connections[y].push_back(x);
	  //Helements[y].push_back(integral);
	}
      }
    }


    for (int i=0; i<nclosed; i++)
      for (int j=0; j<i; j++) {
	for (int a=0; a<nopen; a++){
	  for (int b=0; b<a; b++){
	    Determinant di = d;
	    di.setocc(open[a], true), di.setocc(open[b], true), di.setocc(closed[i],false), di.setocc(closed[j], false);

	    map<Determinant, int>::iterator it = SortedDets.find(di);
	    if (it != SortedDets.end()) {
	      int y = it->second;
	      if (y <= x) { //avoid double counting
		double integral = Hij_2Excite(closed[i], closed[j], open[a], open[b], int2, &detArray[x*Norbs], norbs);
		if (abs(integral) > 1.e-8) {
		  connections[x].push_back(y);
		  Helements[x].push_back(integral);
		  //cout << x<<"  "<<y<<"  "<<integral<<endl;
		}
		//connections[y].push_back(x);
		//Helements[y].push_back(integral);
	      }
	    }
	  }
	}
      }
  }

  
}
