#include "Determinants.h"
#include "CIPSIbasics.h"
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

  int niter = Sample1.size();
  int totalSample = 0;
  for (int index = 0; index<niter; ) {
    int detIndex = floor(1.* ((double) rand() / (RAND_MAX))*ci.rows() );

    double rand_no = ((double) rand()/ (RAND_MAX));
    if (rand_no >= prob[detIndex]) 
      detIndex = alias[detIndex];

    std::vector<int>::iterator it = find(Sample1.begin(), Sample1.end(), detIndex);
    if (it == Sample1.end()) {
      Sample1[index] = detIndex;
      newWts[index] = ci(detIndex,0) < 0. ? -cumulative : cumulative;
      index++; totalSample++;
    }
    else {
      newWts[ distance(Sample1.begin(), it) ] += ci(detIndex,0) < 0. ? -cumulative : cumulative;
      totalSample++;
    }
  }

  for (int i=0; i<niter; i++)
    newWts[i] /= totalSample;
  return totalSample;
  

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
      int Nmc = sample_N2_alias(ci, cumulative, Sample1, wts1, alias, prob);

      double norm = 0.0;
      
      map<Determinant, pair<double,double> > Psi1ab; 
      for (int i=0; i<Sample1.size(); i++) {
       int I = Sample1[i];
       CIPSIbasics::getDeterminants(Dets[I], abs(schd.epsilon2/ci(I,0)), wts1[i], ci(I,0), I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, schd, Nmc);
      }


      double energyEN = 0.0;
      for (map<Determinant, pair<double, double> >::iterator it = Psi1ab.begin(); it != Psi1ab.end(); it++) {
	it->first.getOpenClosed(psiOpen, psiClosed);
	energyEN += (it->second.first*it->second.first *Nmc/(Nmc-1)  - it->second.second)/(Energy(psiClosed, nelec, I1, I2, coreE)-E0); 
	//energyEN += it->second.first*it->second.second/(Energy(psiClosed, nelec, I1, I2, coreE)-E0); 
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



void CIPSIbasics::DoPerturbativeDeterministic(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
					      twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) {

    int norbs = Determinant::norbs;
    std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
    char psiArray[norbs]; vector<int> psiClosed(nelec,0), psiOpen(norbs-nelec,0);
    //char psiArray[norbs]; int psiOpen[nelec], psiClosed[norbs-nelec];
    double energyEN = 0.0;
    int num_thrds = omp_get_max_threads();

    std::vector<std::map<Determinant, pair<double,double> > > Psi1(num_thrds);
#pragma omp parallel for schedule(dynamic)
    for (int i=0; i<Dets.size(); i++) {
      CIPSIbasics::getDeterminants(Dets[i], abs(schd.epsilon2/ci(i,0)), ci(i,0), 0.0, I1, I2, I2HB, irrep, coreE, E0, Psi1[omp_get_thread_num()], SortedDets, schd);
      //cout << Dets.size()<<"  "<<i <<"  "<<Psi1[0].size()<<endl;
      if (i%1000 == 0 && omp_get_thread_num()==0) cout <<"# "<<i<<endl;
    }

    for (int thrd=1; thrd<num_thrds; thrd++) 
      for (map<Determinant, pair<double, double> >::iterator it=Psi1[thrd].begin(); it!=Psi1[thrd].end(); ++it)  {
	if(Psi1[0].find(it->first) == Psi1[0].end())
	  Psi1[0][it->first].first = it->second.first;
	else
	  Psi1[0][it->first].first += it->second.first;
      }
    Psi1.resize(1);


    //cout << "adding contributions from "<<Psi1[0].size()<<" perturber states"<<endl;

    //MatrixXd c1(Psi1[0].size(),1);
    //double norm = 0.0;
    cout << Psi1[0].size()<<endl;
#pragma omp parallel
    {
      vector<int> psiOpen(norbs-nelec,0), psiClosed(nelec,0);
      double thrdEnergy = 0.0;
      size_t cnt = 0;
      for (map<Determinant, pair<double, double> >::iterator it = Psi1[0].begin(); it != Psi1[0].end(); it++, cnt++) {
	if (cnt%num_thrds == omp_get_thread_num()) {
	  if (it->first.ExcitationDistance(Dets[0]) > schd.excitation) continue;
	  it->first.getOpenClosed(psiOpen, psiClosed);
	  double e = Energy(psiClosed, nelec, I1, I2, coreE)-E0; 
	  thrdEnergy += it->second.first*it->second.first/e; 
	  //cout << it->first<<endl;
	  //c1(cnt,0) = -it->second/e;
	  //norm += pow(it->second/e,2);
	  //cout << -it->second/e<<"   "<<it->second<<"   "<<e<<endl;
	}
      }
#pragma omp critical
      {
	//	cout << omp_get_thread_num()<<"  "<<thrdEnergy<<endl;
	energyEN += thrdEnergy;
      }
    }

    cout <<energyEN<<"  "<< -energyEN+E0<<"  "<<getTime()-startofCalc<<endl;
    /*
    cout << "norm "<<norm<<endl;
    if (true) {
      std::vector<Determinant> Psi1vec;
      for (map<Determinant, double>::iterator it=Psi1[0].begin(); it!=Psi1[0].end(); ++it)  {
	Psi1vec.push_back( it->first);
      }
      Psi1.clear();

      std::vector<std::vector<int> > connections(Psi1vec.size());
      std::vector<std::vector<double> > Helements(Psi1vec.size());

      std::vector<char> detChar(norbs*Psi1vec.size(),0);
#pragma omp parallel for schedule(dynamic)
      for (size_t i=0; i<Psi1vec.size() ; i++) 
	Psi1vec[i].getRepArray(&detChar[norbs*i]);

      double pt3 = 0.0;
      MatrixXd c2 = 0.*c1;
      cout << "psi1 norm "<<c1.transpose()*c1<<endl;
#pragma omp parallel for schedule(dynamic) reduction(+:pt3)
      for (int i=0; i<c1.rows(); i++) {
	for (int j=i+1; j<c1.rows(); j++)
	  if (Psi1vec[i].connected(Psi1vec[j])) {
	    double hij = Hij(&detChar[norbs*i], &detChar[norbs*j], norbs, I1, I2, coreE);

	    if (abs(hij) > 1.e-10) {
	      connections[i].push_back(j);
	      Helements[i].push_back(hij);
	    }
	  
	    pt3 += 2.*hij*c1(i,0)*c1(j,0);
	  }
	if (i%10000 == 0) cout << i<<"  out of "<<c1.rows()<<endl; 
      }
      cout << "pt3 "<<c2.transpose()*c2<<"  "<<c2.transpose()*c1<<"  "<<pt3<<"  "<<pt3-energyEN+E0<<endl;

      Hmult2 H(connections, Helements);
      c2 = 0.*c2;
      H(c1,c2);
      cout <<c2.transpose()*c1<<"  "<<c2.transpose()*c2<<endl;
    }
    */
}

void MakeHfromHelpers(std::map<HalfDet, std::vector<int> >& AlphaN,
		      std::vector<int>& AlphaNr,
		      std::map<HalfDet, std::vector<int> >& BetaN,
		      std::vector<int>& BetaNr,		      
		      std::map<HalfDet, std::vector<int> >& AlphaNm1,
		      std::map<HalfDet, std::vector<int> >& BetaNm1,
		      std::map<HalfDet, std::vector<int> >& AlphaNm2,
		      std::map<HalfDet, std::vector<int> >& BetaNm2,
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
  int nprocs= world.size(), proc = world.rank();

  //alpha double excitation
  //for each alpha(N-2) string we know all the determinants [An,Bn] 
  //and then for each [An,Bn] we have to know the beta(N) determinant  
  size_t norbs = Norbs;
  //cout << "alphan-2  "<<AlphaNm2.size()<<endl;

  for (size_t k=StartIndex; k<Dets.size(); k++) {
    if (k%(nprocs*omp_get_num_threads()) != proc*omp_get_num_threads()+omp_get_thread_num()) continue;
    connections[k].push_back(k);
    double hij = Energy(&detChar[norbs*k], Norbs, I1, I2, coreE);
    Helements[k].push_back(hij);
  }


  pout <<"# "<< Dets.size()<<"  "<<AlphaN.size()<<"  "<<AlphaNm1.size()<<"  "<<AlphaNm2.size()<<endl;
  std::map<HalfDet, std::vector<int> >::iterator ita = AlphaNm1.begin();
  int index = 0;
  for (; ita!=AlphaNm1.end(); ita++) {
    std::vector<int>& detIndex = ita->second;
    int localStart = detIndex.size();
    for (int j=0; j<detIndex.size(); j++)
      if (detIndex[j] >= StartIndex) {
	localStart = j; break;
      }

#pragma omp parallel 
    {
      for(int j=0; j<detIndex.size(); j++) {
	if (detIndex[j]%(nprocs*omp_get_num_threads()) != proc*omp_get_num_threads()+omp_get_thread_num()) continue;
	for (int k=max(localStart,j+1); k<detIndex.size(); k++) {
	  size_t J = detIndex[j];size_t K = detIndex[k];
	  if (Dets[J].connected(Dets[K]) ) {
	    if (find(connections[J].begin(), connections[J].end(), K) == connections[J].end()){
	      connections[J].push_back(K);
	      double hij = Hij(&detChar[norbs*J], &detChar[norbs*K], Norbs, I1, I2, coreE);
	      Helements[J].push_back(hij);
	    }
	  }
	}
      }
    }
    index++;
    if (index%100000 == 0 && index!= 0) {pout <<"#an-1 "<<index<<endl;}
  }
  pout << "#AlphaN-1"<<endl;

  //pout << Dets.size()<<"  "<<BetaN.size()<<"  "<<BetaNm1.size()<<"  "<<BetaNm2.size()<<endl;
  ita = AlphaNm2.begin();
  index = 0;
  for (; ita!=AlphaNm2.end(); ita++) {
    std::vector<int>& detIndex = ita->second;
    int localStart = detIndex.size();
    for (int j=0; j<detIndex.size(); j++)
      if (detIndex[j] >= StartIndex) {
	localStart = j; break;
      }

#pragma omp parallel 
    {
      for(int j=0; j<detIndex.size(); j++) {
	if (detIndex[j]%(nprocs*omp_get_num_threads()) != proc*omp_get_num_threads()+omp_get_thread_num()) continue;
	//if (detIndex[j]%omp_get_num_threads() != omp_get_thread_num()) continue;
	for (int k=max(localStart,j); k<detIndex.size(); k++) {
	  size_t J = detIndex[j];size_t K = detIndex[k];
	  if (K >= StartIndex && BetaNr[J] == BetaNr[K]) 	  {
	    if (find(connections[J].begin(), connections[J].end(), K) == connections[J].end()){
	      connections[J].push_back(K);
	      double hij = Hij(&detChar[norbs*J], &detChar[norbs*K], Norbs, I1, I2, coreE);
	      Helements[J].push_back(hij);
	    }
	  }
	}
      }
    }
    index++;
    if (index%100000 == 0 && index!= 0) {pout <<"#an-2 "<<index<<endl;}
  }
  pout << "#alphaN-2"<<endl;
  /*
  //pout << connections[0].size()<<"  2 alpha "<<endl;
  //exit(0);
  //beta double excitation
  std::map<HalfDet, std::vector<int> >::iterator it = BetaNm2.begin();
  for (; it!=BetaNm2.end(); it++) {
    std::vector<int>& detIndex = it->second;
    int localStart = detIndex.size();
    for (int j=0; j<detIndex.size(); j++)
      if (detIndex[j] >= StartIndex) {
	localStart = j; break;
      }

#pragma omp parallel 
    {
      for(int j=0; j<detIndex.size(); j++) {
	if (detIndex[j]%omp_get_num_threads() != omp_get_thread_num()) continue;
	for (int k=max(localStart,j+1); k<detIndex.size(); k++) {
	  int J = detIndex[j];int K = detIndex[k];
	  if (K >= StartIndex && AlphaNr[J] == AlphaNr[K]) {
	    if (find(connections[J].begin(), connections[J].end(), K) == connections[J].end()) {
	      connections[J].push_back(K);
	      double hij = Hij(&detChar[Norbs*J], &detChar[norbs*K], Norbs, I1, I2, coreE);
	      Helements[J].push_back(hij);
	    }
	  }
	}
      }
    }
  }
  pout << "betaN-2"<<endl;
  //pout << connections[0].size()<<"  2 beta "<<endl;

  
  //beta-single excitation
  it = BetaNm1.begin();
  for (; it!=BetaNm1.end(); it++) {
    std::vector<int>& detIndex = it->second;
    int localStart = detIndex.size();
    for (int j=0; j<detIndex.size(); j++)
      if (detIndex[j] >= StartIndex) {
	localStart = j; break;
      }

#pragma omp parallel 
    {
      for(int j=0; j<detIndex.size(); j++) {
	if (detIndex[j]%omp_get_num_threads() != omp_get_thread_num()) continue;
	for (int k=max(localStart,j+1); k<detIndex.size(); k++) {
	  int J = detIndex[j];int K = detIndex[k];
	  if (Dets[K].connected(Dets[J]))
	    if (find(connections[J].begin(), connections[J].end(), K) == connections[J].end()) {
	      connections[J].push_back(K);
	      double hij = Hij(&detChar[Norbs*J], &detChar[norbs*K], Norbs, I1, I2, coreE);
	      Helements[J].push_back(hij);
	    }
	}
      }
    }
  }
  pout << "alphabetaN-2"<<endl;
  
  //pout << connections[0].size()<<"  alpha-beta "<<endl;
  
  //alpha-single excitation
  it = AlphaNm1.begin();
  for (; it!=AlphaNm1.end(); it++) {
    std::vector<int>& detIndex = it->second;
    int localStart = detIndex.size();
    for (int j=0; j<detIndex.size(); j++)
      if (detIndex[j] >= StartIndex) {
	localStart = j; break;
      }

#pragma omp parallel 
    {
      for(int j=0; j<detIndex.size(); j++) {
	if (detIndex[j]%omp_get_num_threads() != omp_get_thread_num()) continue;
	for (int k=max(localStart,j+1); k<detIndex.size(); k++) {
	  int J = detIndex[j];int K = detIndex[k];
	  if (K >= StartIndex && BetaNr[K] == BetaNr[J]) { //Dets j&k are related by single beta excitatin
	    if (find(connections[J].begin(), connections[J].end(), K) == connections[J].end()) {
	      connections[J].push_back(K);
	      double hij = Hij(&detChar[Norbs*J], &detChar[norbs*K], Norbs, I1, I2, coreE);
	      Helements[J].push_back(hij);
	    }
	  }
	}
      }
    }
    //pout << connections[0].size()<<"  1 alpha "<<endl;
  }  
    pout << "alphaN-1"<<endl;
  */    
    
}
  
void PopulateHelperLists(std::map<HalfDet, std::vector<int> >& AlphaN,
			 std::vector<int>& AlphaNr,
			 std::map<HalfDet, std::vector<int> >& BetaN,
			 std::vector<int>& BetaNr,
			 std::map<HalfDet, std::vector<int> >& AlphaNm1,
			 std::map<HalfDet, std::vector<int> >& BetaNm1,
			 std::map<HalfDet, std::vector<int> >& AlphaNm2,
			 std::map<HalfDet, std::vector<int> >& BetaNm2,
			 std::vector<Determinant>& Dets,
			 int StartIndex) {
  pout <<"#Making Helpers"<<endl;
  for (int i=StartIndex; i<Dets.size(); i++) {
    HalfDet da = Dets[i].getAlpha(), db = Dets[i].getBeta();

    AlphaN[da].push_back(i);
    BetaN[db].push_back(i);

    int norbs = 64*DetLen;
    std::vector<int> closeda(norbs), closedb(norbs);
    int ncloseda = da.getClosed(closeda);
    int nclosedb = db.getClosed(closedb);

    
    for (int j=0; j<ncloseda; j++) {
      HalfDet daj = da; daj.setocc(closeda[j], false);
      AlphaNm1[daj].push_back(i);
      for (int k=j+1; k<ncloseda; k++) {
	HalfDet dajk = daj; dajk.setocc(closeda[k], false);
	AlphaNm2[dajk].push_back(i);
      }
    }
  }

  std::map<HalfDet, std::vector<int> >::iterator ita = AlphaN.begin();
  int index = 0;
  AlphaNr.resize(Dets.size(),0);
  for (;ita!= AlphaN.end(); ita++) {
    std::vector<int>& detIndex = ita->second;
    for(int j=0; j<detIndex.size(); j++) 
      AlphaNr[detIndex[j]] = index;
    index++;
  }

  std::map<HalfDet, std::vector<int> >::iterator itb = BetaN.begin();
  index = 0;
  BetaNr.resize(Dets.size(),0);
  for (;itb!= BetaN.end(); itb++) {
    std::vector<int>& detIndex = itb->second;
    for(int j=0; j<detIndex.size(); j++) 
      BetaNr[detIndex[j]] = index;
    index++;
  }

    /*
    for (int j=0; j<nclosedb; j++) {
      HalfDet dbj = db; dbj.setocc(closedb[j], false);
      BetaNm1[dbj].push_back(i);
      for (int k=j+1; k<nclosedb; k++) {
	HalfDet dbjk = dbj; dbjk.setocc(closedb[k], false);
	BetaNm2[dbjk].push_back(i);
      }

    }

  }

  
    */

}

//this takes in a ci vector for determinants placed in Dets
//it then does a CIPSI varitional calculation and the resulting
//ci and dets are returned here
//at input usually the Dets will just have a HF or some such determinant
//and ci will be just 1.0
double CIPSIbasics::DoVariational(MatrixXd& ci, vector<Determinant>& Dets, schedule& schd,
				  twoInt& I2, twoIntHeatBath& I2HB, vector<int>& irrep, oneInt& I1, double& coreE) {

  std::map<HalfDet, std::vector<int> > AlphaN, BetaN, AlphaNm1, BetaNm1, AlphaNm2, BetaNm2;
  std::vector<int> AlphaNr, BetaNr;
  PopulateHelperLists(AlphaN, AlphaNr, BetaN, BetaNr, AlphaNm1, BetaNm1, AlphaNm2, BetaNm2, Dets, 0);


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
    readVariationalResult(iterstart, ci, Dets, SortedDets, diagOld, connections, Helements, E0, converged, schd, AlphaN, BetaN, AlphaNm1, AlphaNm2, AlphaNr, BetaNr);
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
      if (i%world.size() != world.rank()) continue;
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


    //mpi send
    if (mpigetrank() != 0) {
      for (int proc = 1; proc<world.size(); proc++) {
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
      if (k % world.size() != world.rank() ) continue;
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
      PopulateHelperLists(AlphaN, AlphaNr, BetaN, BetaNr, AlphaNm1, BetaNm1, AlphaNm2, BetaNm2, Dets, SortedDets.size());
      MakeHfromHelpers(AlphaN, AlphaNr, BetaN, BetaNr, AlphaNm1, BetaNm1, AlphaNm2, BetaNm2, Dets, SortedDets.size(), connections, Helements,
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
	if (i%world.size() != world.rank()) continue;
	
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
      if (schd.io) writeVariationalResult(iter, ci, Dets, SortedDets, diag, connections, Helements, E0, true, schd, AlphaN, BetaN, AlphaNm1, AlphaNm2, AlphaNr, BetaNr);
      break;
    }
    else {
      if (schd.io) writeVariationalResult(iter, ci, Dets, SortedDets, diag, connections, Helements, E0, false, schd, AlphaN, BetaN, AlphaNm1, AlphaNm2, AlphaNr, BetaNr);
    }


  }
  return E0;

}



void CIPSIbasics::writeVariationalResult(int iter, MatrixXd& ci, vector<Determinant>& Dets, vector<Determinant>& SortedDets,
					 MatrixXd& diag, vector<vector<int> >& connections, vector<vector<double> >& Helements, 
					 double& E0, bool converged, schedule& schd,   
					 std::map<HalfDet, std::vector<int> >& AlphaN, 
					 std::map<HalfDet, std::vector<int> >& BetaN, 
					 std::map<HalfDet, std::vector<int> >& AlphaNm1, 
					 std::map<HalfDet, std::vector<int> >& AlphaNm2, 
					 std::vector<int>& AlphaNr, std::vector<int>& BetaNr) {
#ifndef SERIAL
  boost::mpi::communicator world;
#endif


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
    save << AlphaN << AlphaNr<< BetaN<< BetaNr<< AlphaNm1<< AlphaNm2;
    ofs.close();
}


void CIPSIbasics::readVariationalResult(int& iter, MatrixXd& ci, vector<Determinant>& Dets, vector<Determinant>& SortedDets,
					MatrixXd& diag, vector<vector<int> >& connections, vector<vector<double> >& Helements, 
					double& E0, bool& converged, schedule& schd,
					std::map<HalfDet, std::vector<int> >& AlphaN, 
					std::map<HalfDet, std::vector<int> >& BetaN, 
					std::map<HalfDet, std::vector<int> >& AlphaNm1, 
					std::map<HalfDet, std::vector<int> >& AlphaNm2, 
					std::vector<int>& AlphaNr, std::vector<int>& BetaNr) {

#ifndef SERIAL
  boost::mpi::communicator world;
#endif


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
    load >> AlphaN >> AlphaNr>> BetaN>> BetaNr>> AlphaNm1>> AlphaNm2;
    ifs.close();
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
	  //if (det_it == Psi1.end()) Psi1[di] = make_pair(integral*ci1, integral*integral*ci2*ci1*(ci1/ci2-1.));
	  if (det_it == Psi1.end()) Psi1[di] = make_pair(integral*ci1, integral*integral*ci1*(ci1*Nmc/(Nmc-1)-ci2));
	  //else {det_it->second.first +=integral*ci1;det_it->second.second += integral*integral*ci2*ci1*(ci1/ci2-1.);}
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
    /*
    else {
      for (int a=0; a<nopen; a++){
	for (int b=0; b<a; b++){
	  double integral = int2(closed[i],open[a],closed[j],open[b]) - int2(closed[i],open[b],closed[j],open[a]);
	  
	  if (open[a]/2 > schd.nvirt+nclosed/2 || open[b]/2 >schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital
	  if (fabs(integral) > epsilon ) {
	    Determinant di = d;
	    di.setocc(open[a], true), di.setocc(open[b], true), di.setocc(closed[i],false), di.setocc(closed[j], false);
	    if (!(binary_search(Psi0.begin(), Psi0.end(), di))) {

	      {
		int A = (closed[i]), B = closed[j], I= open[a], J = open[b]; 
		double sgn = parity(detArray,norbs,A)*parity(detArray,norbs,I)*parity(detArray,norbs,B)*parity(detArray,norbs,J);
		if (B > J) sgn*=-1 ;
		if (I > J) sgn*=-1 ;
		if (I > B) sgn*=-1 ;
		if (A > J) sgn*=-1 ;
		if (A > B) sgn*=-1 ;
		if (A > I) sgn*=-1 ;
		integral *= sgn;
	      }
	      det_it = Psi1.find(di);

	      if (det_it == Psi1.end()) Psi1[di] = make_pair(integral*ci1, integral*ci2);
	      else {det_it->second.first += integral*ci1; det_it->second.second += integral*ci2;}
	    }
	  }
	}
      }
    }
    */
    
  //for (int thrd=0; thrd<omp_get_max_threads(); thrd++)
  //for (int i=0; i<thrdDeterminants[thrd].size(); i++)
  //dets.insert(thrdDeterminants[thrd][i]);
  
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
  

/*

void CIPSIbasics::DoPerturbativeDeterministicLCC(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
						 twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) {

    int norbs = Determinant::norbs;
    std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
    char psiArray[norbs]; vector<int> psiClosed(nelec,0), psiOpen(norbs-nelec,0);
    //char psiArray[norbs]; int psiOpen[nelec], psiClosed[norbs-nelec];
    double energyEN = 0.0;
    int num_thrds = omp_get_max_threads();

    std::vector<std::map<Determinant, double>> Psi1(num_thrds);
#pragma omp parallel for schedule(dynamic)
    for (int i=0; i<Dets.size(); i++) {
      CIPSIbasics::getDeterminants(Dets[i], abs(schd.epsilon2/ci(i,0)), ci(i,0), I1, I2, I2HB, irrep, coreE, E0, Psi1[omp_get_thread_num()], SortedDets);
      if (i%1000 == 0 && omp_get_thread_num()==0) cout <<"# "<<i<<endl;
    }


    for (int thrd=1; thrd<num_thrds; thrd++) 
      for (map<Determinant, double>::iterator it=Psi1[thrd].begin(); it!=Psi1[thrd].end(); ++it)  {
	if(Psi1[0].find(it->first) == Psi1[0].end())
	  Psi1[0][it->first] = it->second;
	else
	  Psi1[0][it->first] += it->second;
      }
    Psi1.resize(1);

    std::vector<Determinant> Psi1vec(Psi1[0].size()); MatrixXd b(Psi1[0].size(),1);

    int iter = 0;
    for (map<Determinant, double>::iterator it=Psi1[0].begin(); it!=Psi1[0].end(); ++it)  {
      Psi1vec[iter]= it->first; b(iter,0) = it->second;
      iter++;
    }
    Psi1.clear();

    cout << "# total number of dets in Psi1 "<<Psi1vec.size()<<endl;
    //this is essentially the hamiltonian, we have stored it in a sparse format
    std::vector<std::vector<int> > connections(Psi1vec.size());
    std::vector<std::vector<double> > Helements(Psi1vec.size());

    std::vector<char> detChar(norbs*Psi1vec.size(),0);
#pragma omp parallel for schedule(dynamic)
    for (size_t i=0; i<Psi1vec.size() ; i++) 
      Psi1vec[i].getRepArray(&detChar[norbs*i]);

#pragma omp parallel for schedule(dynamic)
    for (size_t i=0; i<Psi1vec.size() ; i++) {
      for (size_t j=i; j<Psi1vec.size(); j++) {
	if (Psi1vec[i].connected(Psi1vec[j])) {
	  double hij = Hij(&detChar[norbs*i], &detChar[norbs*j], norbs, I1, I2, coreE);
	  
	  if (abs(hij) > 1.e-10) {
	    connections[i].push_back(j);
	    if (i == j) {
	      Helements[i].push_back(hij - E0);
	    }
	    else {
	      Helements[i].push_back(hij);
	    }
	  }
	}
      }
      if (i%10000 == 0) cout <<i<<"  "<<Helements[i].size()<<"  out of "<<Psi1vec.size()<<endl;
      //if (i%1000000 == 0 && i!=0) cout <<"#"<< i<<" Hij out of "<<Dets.size()<<endl;     
      //if (i%10000) cout << i<<"  "<<Dets.size()<<endl;
    }

    MatrixXd X0 = 0.*b;
    Hmult2 H(connections, Helements);
    double Ep2 = LinearSolver(H, X0, b, schd.davidsonTol, true);

    cout <<Ep2<<"  "<< Ep2+E0<<"  "<<getTime()-startofCalc<<endl;

}



*/
