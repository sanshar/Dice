#include "global.h"
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include "Determinants.h"
#include "integral.h"
#include "Hmult.h"
#include "CIPSIbasics.h"
#include "Davidson.h"
#include <Eigen/Dense>
#include <set>
#include <list>
#include <tuple>
#include "boost/format.hpp"

using namespace Eigen;
using namespace boost;
int Determinant::norbs = 1; //spin orbitals
int Determinant::EffDetLen = 1;
//get the current time
double getTime() {
  struct timeval start;
  gettimeofday(&start, NULL);
  return start.tv_sec + 1.e-6*start.tv_usec;
}
double startofCalc = getTime();

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


int sample_N(MatrixXd& ci, double& cumulative, std::vector<int>& Sample1, std::vector<double>& newWts){
  double prob = 1.0;
  for (int s=0; s<Sample1.size(); s++) {

    double rand_no = ((double) rand() / (RAND_MAX))*cumulative;
    for (int i=0; i<ci.rows(); i++) {
      if (rand_no < abs(ci(i,0))) {
	Sample1[s] = i;
	//newWts[s] = ci(i,0);
	//prob = prob*ci(i,0)/cumulative;
	newWts[s] = cumulative/Sample1.size();
	//prob = prob*ci(i,0)/cumulative;
	break;
      }
      rand_no -= abs(ci(i,0));
    }
  }

  /*
  for (int s=0; s<Sample1.size(); s++) 
    newWts[s] = newWts[s]/prob;
  */
}


void readInput(string input, std::vector<int>& occupied, CIPSIbasics::schedule& schd);


int main(int argc, char* argv[]) {
  startofCalc=getTime();
  std::cout.precision(15);

  //read the hamiltonian (integrals, orbital irreps, num-electron etc.)
  twoInt I2; oneInt I1; int nelec; int norbs; double coreE, eps;
  std::vector<int> irrep;
  readIntegrals("FCIDUMP", I2, I1, nelec, norbs, coreE, irrep);
  norbs *=2;
  Determinant::norbs = norbs; //spin orbitals
  Determinant::EffDetLen = norbs/64+1;
  if (Determinant::EffDetLen >DetLen) {
    cout << "change DetLen in global.h to "<<Determinant::EffDetLen<<" and recompile "<<endl;
    exit(0);
  }

  //initialize the heatback integral
  std::vector<int> allorbs;
  for (int i=0; i<norbs/2; i++)
    allorbs.push_back(i);
  twoIntHeatBath I2HB(1.e-10);
  I2HB.constructClass(allorbs, I2, norbs/2);


  int num_thrds;
  std::vector<int> HFoccupied; double epsilon1, epsilon2, tol, dE;
  CIPSIbasics::schedule schd;
  readInput("input.dat", HFoccupied, schd); //epsilon1, epsilon2, tol, num_thrds, eps, dE);


  //make HF determinant
  Determinant d;
  for (int i=0; i<HFoccupied.size(); i++) {
    d.setocc(HFoccupied[i], true);
  }
  MatrixXd ci(1,1); ci(0,0) = 1.0;
  std::vector<Determinant> Dets(1,d);
  
  double E0 = CIPSIbasics::DoVariational(ci, Dets, schd, I2, I2HB, I1, coreE);

  //print the 5 most important determinants and their weights
  MatrixXd prevci = 1.*ci;
  for (int i=0; i<5; i++) {
    compAbs comp;
    int m = distance(&prevci(0,0), max_element(&prevci(0,0), &prevci(0,0)+prevci.rows(), comp));
    cout <<"#"<< i<<"  "<<prevci(m,0)<<"  "<<Dets[m]<<endl;
    prevci(m,0) = 0.0;
  }
  prevci.resize(0,0);



  //now do the perturbative bit
  if (false) {
    std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
    char psiArray[norbs]; vector<int> psiClosed(nelec,0), psiOpen(norbs-nelec,0);
    //char psiArray[norbs]; int psiOpen[nelec], psiClosed[norbs-nelec];
    double energyEN = 0.0;


    std::vector<std::map<Determinant, double>> Psi1(num_thrds);
#pragma omp parallel for schedule(dynamic)
    for (int i=0; i<Dets.size(); i++) {
      CIPSIbasics::getDeterminants(Dets[i], abs(epsilon2/ci(i,0)), ci(i,0), I1, I2, I2HB, irrep, coreE, E0, Psi1[omp_get_thread_num()], SortedDets);
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


    cout << "adding contributions from "<<Psi1[0].size()<<" perturber states"<<endl;

#pragma omp parallel
    {
      vector<int> psiOpen(norbs-nelec,0), psiClosed(nelec,0);
      double thrdEnergy = 0.0;
      size_t cnt = 0;
      for (map<Determinant, double>::iterator it = Psi1[0].begin(); it != Psi1[0].end(); it++, cnt++) {
	if (cnt%num_thrds == omp_get_thread_num()) {
	  it->first.getOpenClosed(psiOpen, psiClosed);
	  thrdEnergy += it->second*it->second/(Energy(psiClosed, nelec, I1, I2, coreE)-E0); 
	}
      }
#pragma omp critical
      {
	cout << omp_get_thread_num()<<"  "<<thrdEnergy<<endl;
	energyEN += thrdEnergy;
      }
    }

    cout <<energyEN<<"  "<< -energyEN+E0<<"  "<<getTime()-startofCalc<<endl;
  }
  else if(true){
    std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
    int niter = 10000;
    //double eps = 0.001;
    double AvgenergyEN = 0.0;
    int currentIter = 0;
    int sampleSize = 0;

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
      
      map<Determinant, pair<double,double> > Psi1ab; 
      for (int i=0; i<Sample1.size(); i++) {
	int I = Sample1[i];
	CIPSIbasics::getDeterminants(Dets[I], abs(epsilon2/ci(I,0)), wts1[i], I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, 0);
      }
      
      for (int i=0; i<Sample2.size(); i++) {
	int I = Sample2[i];
	CIPSIbasics::getDeterminants(Dets[I], abs(epsilon2/ci(I,0)), wts2[i], I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, 1);
      }
      
      double energyEN = 0.0;
      for (map<Determinant, pair<double, double> >::iterator it = Psi1ab.begin(); it != Psi1ab.end(); it++) {
	it->first.getOpenClosed(psiOpen, psiClosed);
	energyEN += it->second.first*it->second.second/(Energy(psiClosed, nelec, I1, I2, coreE)-E0); 
      }
      sampleSize = Sample1.size();

#pragma omp critical 
      {
	AvgenergyEN += energyEN; currentIter++;
	std::cout << format("%4i  %14.8f  %14.8f   %10.2f  %10i %4i") 
	  %(currentIter) % (E0-energyEN) % (E0-AvgenergyEN/currentIter) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	cout << endl;
      }
      
    }
  }
  else { //THIS CURRENTLY BROKEN
    std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
    int niter = 100000;
    //double eps = 0.001;
    double AvgenergyEN = 0.0;
    int currentIter = 0;
    int sampleSize = 0;
    int N = 20;
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
      std::vector<double> wts1(N,0.0), wts2(N,0.0); std::vector<int> Sample1(N,-1), Sample2(N,-1);

      sample_N(ci, cumulative, Sample1, wts1);
      sample_N(ci, cumulative, Sample2, wts2);
      
      map<Determinant, pair<double,double> > Psi1ab; 
      for (int i=0; i<Sample1.size(); i++) {
	int I = Sample1[i];
	CIPSIbasics::getDeterminants(Dets[I], abs(epsilon2/ci(I,0)), wts1[i], I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, 0);
      }
      
      for (int i=0; i<Sample2.size(); i++) {
	int I = Sample2[i];
	CIPSIbasics::getDeterminants(Dets[I], abs(epsilon2/ci(I,0)), wts2[i], I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, 1);
      }
      
      double energyEN = 0.0;
      for (map<Determinant, pair<double, double> >::iterator it = Psi1ab.begin(); it != Psi1ab.end(); it++) {
	it->first.getOpenClosed(psiOpen, psiClosed);
	energyEN += it->second.first*it->second.second/(Energy(psiClosed, nelec, I1, I2, coreE)-E0); 
      }
      sampleSize = Sample1.size();

#pragma omp critical 
      {
	AvgenergyEN += energyEN; currentIter++;
	std::cout << format("%4i  %14.8f  %14.8f   %10.2f  %10i %4i") 
	  %(currentIter) % (E0-energyEN) % (E0-AvgenergyEN/currentIter) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	cout << endl;
      }
      
    }
    
  }

  return 0;
}
