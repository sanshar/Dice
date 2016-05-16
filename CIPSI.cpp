#include <ctime>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <fstream>
#include "global.h"
#include "Determinants.h"
#include "integral.h"
#include "Hmult.h"
#include "CIPSIbasics.h"
#include "Davidson.h"
#include <Eigen/Dense>
#include <set>
#include <list>
#include <tuple>

using namespace Eigen;
int Determinant::norbs = 1; //spin orbitals

int sample_N(MatrixXd& ci, std::vector<int>& Sample1, std::vector<double>& newWts){
  double cumulative = 0.;
  for (int i=0; i<ci.rows(); i++)
    cumulative += abs(ci(i,0));
  /*
  double prob = 1.0;
  for (int s=0; s<Sample1.size(); s++) {
    double rand_no = ((double) rand() / (RAND_MAX))*cumulative;
    for (int i=0; i<wts.size(); i++) {
      if (rand_no <abs(wts[i])) {
	Sample1[s] = i;
	newWts[s] = wts[i];
	prob = prob*abs(wts[i])/cumulative;
	break;
      }
      rand_no -= abs(wts[i]);
    }
  }

  for (int s=0; s<Sample1.size(); s++)
    newWts[s] /= prob;
  */
}


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


void readInput(string input, std::vector<int>& occupied, double& eps1, double& eps2, double& tol,int& num_thrds, double& epsilon, double& dE) {
  ifstream dump(input.c_str());
  int nocc; dump >>nocc;
  occupied.resize(nocc);
  for (int i=0; i<nocc; i++)
    dump >> occupied[i];
  dump >> eps1 >>eps2>>tol>>num_thrds>>epsilon>>dE;
}

double getTime() {
  struct timeval start;
  gettimeofday(&start, NULL);
  return start.tv_sec + 1.e-6*start.tv_usec;
}


int main(int argc, char* argv[]) {
  double startofCalc=getTime();
  std::cout.precision(15);
  twoInt I2; oneInt I1; int nelec; int norbs; double coreE, eps;
  std::vector<int> irrep;
  readIntegrals("FCIDUMP", I2, I1, nelec, norbs, coreE, irrep);
  irrep.resize(norbs);
  norbs *=2;
  Determinant::norbs = norbs; //spin orbitals

  int num_thrds;
  std::vector<int> HFoccupied; double epsilon1, epsilon2, tol, dE;
  readInput("input.dat", HFoccupied, epsilon1, epsilon2, tol, num_thrds, eps, dE);

  std::vector<int> allorbs;
  for (int i=0; i<norbs/2; i++)
    allorbs.push_back(i);
  twoIntHeatBath I2HB(epsilon2);
  I2HB.constructClass(allorbs, I2, norbs/2);

  //make HF determinant
  Determinant d;
  //std::cout << norbs<<std::endl;
  for (int i=0; i<HFoccupied.size(); i++) {
    //std::cout <<i<<"  "<< HFoccupied[i]<<std::endl;
    d.setocc(HFoccupied[i], true);
  }

  char detchar[norbs]; d.getRepArray(detchar);
  double EHF= Energy(detchar,norbs,I1,I2,coreE);
  std::cout << "#HF = "<<EHF<<std::endl;

  unsigned short closed[nelec], open[norbs-nelec];
  int o = d.getOpenClosed(open, closed); int v=norbs-o;

  std::vector<Determinant> Dets(1,d), prevDets(1,d);
  std::vector<Determinant> SortedDets(1,d);
  MatrixXd ci(1,1), prevci(1,1); ci(0,0) = 1.0; prevci *= 0.0;

  double E0 = EHF;
  std::vector<char> detChar(norbs); d.getRepArray(&detChar[0]);
  MatrixXd diagOld(1,1); diagOld(0,0) = EHF;
  int prevSize = 0;
  std::vector<std::vector<int> > connections(1, std::vector<int>(1,0));
  std::vector<std::vector<double> > Helements(1, std::vector<double>(1,EHF));

  omp_set_num_threads(num_thrds);
  //std::cout << "max thrds "<<num_thrds<<std::endl;
  int iter = 0, maxiter = 40; 
  //do the variational bit
  while(iter<maxiter){

    std::vector<set<Determinant> > newDets(num_thrds);
#pragma omp parallel for schedule(dynamic)
    for (int i=0; i<prevDets.size(); i++) 
      getDeterminants(Dets[i], abs(epsilon1/ci(i,0)), I1, I2, I2HB, coreE, E0, newDets[omp_get_thread_num()]);

    for (int thrd=1; thrd<num_thrds; thrd++)
      for (set<Determinant>::iterator it=newDets[thrd].begin(); it!=newDets[thrd].end(); ++it) 
	if(newDets[0].find(*it) == newDets[0].end())
	  newDets[0].insert(*it);

    for (set<Determinant>::iterator it=newDets[0].begin(); it!=newDets[0].end(); ++it) 
      if (!(binary_search(SortedDets.begin(), SortedDets.end(), *it))) 
	Dets.push_back(*it);

    
    //now diagonalize the hamiltonian
    detChar.resize(norbs* Dets.size()); 
    MatrixXd X0(Dets.size(), 1); X0 *= 0.0; X0.block(0,0,ci.rows(),1) = 1.*ci; prevci = X0;
    MatrixXd diag(Dets.size(), 1); diag.block(0,0,ci.rows(),1)= 1.*diagOld;

#pragma omp parallel for schedule(dynamic)
    for (int k=prevDets.size(); k<Dets.size(); k++) {
      Dets[k].getRepArray(&detChar[norbs*k]);
      diag(k,0) = Energy(&detChar[norbs*k], norbs, I1, I2, coreE);
    }

    //update connetions
    connections.resize(Dets.size());
    Helements.resize(Dets.size());
#pragma omp parallel for schedule(dynamic)
    for (size_t i=0; i<Dets.size(); i++) 
      for (int j=max(prevDets.size(),i); j<Dets.size(); j++) {
	if (Dets[i].connected(Dets[j])) {
	  double hij = Hij(&detChar[norbs*i], &detChar[norbs*j], norbs, I1, I2, coreE);
	  //double hij = Hij(Dets[i], Dets[j], norbs, I1, I2, coreE);
	  if (abs(hij) > 1.e-10) {
	    connections[i].push_back(j);
	    Helements[i].push_back(hij);
	  }
	}
      }


    double prevE0 = E0;
    //Hmult H(&detChar[0], norbs, I1, I2, coreE);
    Hmult2 H(connections, Helements);
    E0 = davidson(H, X0, diag, 5, tol, true);
    std::cout <<"#"<<iter<<"  "<<Dets.size()<<"  "<< E0 <<"  "<<getTime()-startofCalc<<std::endl;iter++;
    ci.resize(Dets.size(),1); ci = 1.0*X0;
    diagOld.resize(Dets.size(),1); diagOld = 1.0*diag;

    if (abs(E0-prevE0) < dE)  {
      break;
    }
    prevSize = prevDets.size();

    prevDets.clear();prevDets=Dets;
    SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
  }
  //now do the perturbative bit


  prevci = 1.*ci;
  for (int i=0; i<5; i++) {
    compAbs comp;
    int m = distance(&prevci(0,0), max_element(&prevci(0,0), &prevci(0,0)+prevci.rows(), comp));
    cout << i<<"  "<<prevci(m,0)<<"  "<<Dets[m]<<endl;
    prevci(m,0) = 0.0;
  }
  prevci.resize(0,0);

  if (false) {
    SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
    char psiArray[norbs]; vector<int> psiClosed(nelec,0), psiOpen(norbs-nelec,0);
    //char psiArray[norbs]; int psiOpen[nelec], psiClosed[norbs-nelec];
    double energyEN = 0.0;


    std::map<Determinant, double> Psi1;
    for (int i=0; i<Dets.size(); i++) {
      getDeterminants(Dets[i], abs(epsilon2/ci(i,0)), ci(i,0), I1, I2, I2HB, irrep, coreE, E0, Psi1, SortedDets);
      if (i%1000 == 0) cout <<"# "<<i<<endl;
    }


    //#pragma omp parallel for reduction(+:energyEN)
    for (map<Determinant, double>::iterator it = Psi1.begin(); it != Psi1.end(); it++) {
      it->first.getOpenClosed(psiOpen, psiClosed);
      energyEN += it->second*it->second/(Energy(psiClosed, nelec, I1, I2, coreE)-E0); 
      //it->first.getRepArray(psiArray);
      //energyEN += it->second*it->second/(Energy(psiArray, norbs, I1, I2, coreE)-E0);
    }
    cout <<energyEN<<"  "<< -energyEN+E0<<"  "<<getTime()-startofCalc<<endl;
  }
  else if(true){
    SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
    int niter = 10000;
    //double eps = 0.001;
    double AvgenergyEN = 0.0, energyEN=0.0;
    double biasedEN = 0.0;
    int parallelLoops = niter/num_thrds; int parLoopIter = 0;
    int sampleSize = 0;
    while(parLoopIter < parallelLoops) {
      energyEN = 0.0;
#pragma omp parallel for schedule(static) reduction(+:energyEN)
      for (int iter=0; iter<num_thrds; iter++) {
	//cout << norbs<<"  "<<nelec<<endl;
	char psiArray[norbs]; 
	vector<int> psiClosed(nelec,0); 
	vector<int> psiOpen(norbs-nelec,0);
	//char psiArray[norbs];
	std::vector<double> wts1, wts2; std::vector<int> Sample1, Sample2;
	wts1.reserve(1000); wts2.reserve(1000); Sample1.reserve(1000); Sample2.reserve(1000);
	
	Sample1.resize(0); wts1.resize(0); Sample2.resize(0); wts2.resize(0);
	sample_round(ci, eps, Sample1, wts1);
	sample_round(ci, eps, Sample2, wts2);

	map<Determinant, double> Psi1a, Psi1b; 
	for (int i=0; i<Sample1.size(); i++) {
	  int I = Sample1[i];
	  getDeterminants(Dets[I], abs(epsilon2/ci(I,0)), wts1[i], I1, I2, I2HB, irrep, coreE, E0, Psi1a, SortedDets);
	}

	Psi1b.insert(Psi1a.begin(), Psi1a.end());
	for (map<Determinant, double>::iterator it = Psi1b.begin(); it != Psi1b.end(); it++) 
	  it->second = 0.0;
	for (int i=0; i<Sample2.size(); i++) {
	  int I = Sample2[i];
	  getDeterminants(Dets[I], abs(epsilon2/ci(I,0)), wts2[i], I1, I2, I2HB, irrep, coreE, E0, Psi1b, SortedDets, false);
	}


	map<Determinant, double>::iterator it2 = Psi1b.begin();
	for (map<Determinant, double>::iterator it = Psi1a.begin(); it != Psi1a.end(); it++) {
	  it->first.getOpenClosed(psiOpen, psiClosed);
	  energyEN += it->second*it2->second/(Energy(psiClosed, nelec, I1, I2, coreE)-E0); 
	  it2++;
	}
	sampleSize = Sample1.size();
      }
      parLoopIter++;
      AvgenergyEN += energyEN;
      cout <<(parLoopIter)*num_thrds <<"  "<<E0-energyEN/num_thrds<<"  "<<E0-AvgenergyEN/num_thrds/parLoopIter<<"  "<<getTime()-startofCalc<<"  "<<sampleSize<<endl;
    }

  }
  else {
    SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
    int niter = 10000;
    double eps = 0.2;
    double energyEN = 0.0;
    double biasedEN = 0.0;
    int parallelLoops = niter/num_thrds; int parLoopIter = 0;
    int sampleSize = 0;
    int N = 100;
    while(parLoopIter < parallelLoops) {

#pragma omp parallel for schedule(static) reduction(+:energyEN)
      for (int iter=0; iter<num_thrds; iter++) {
	char psiArray[norbs];
	std::vector<double> wts1(N,0), wts2(N,0); std::vector<int> Sample1(N,0), Sample2(N,0);
	
	sample_N(ci, Sample1, wts1);
	sample_N(ci, Sample2, wts2);

	map<Determinant, double> Psi1a, Psi1b; 
	for (int i=0; i<Sample1.size(); i++) {
	  int I = Sample1[i];
	  getDeterminants(Dets[I], abs(epsilon2/ci(I,0)), wts1[i], I1, I2, I2HB, irrep, coreE, E0, Psi1a, SortedDets);
	}

	Psi1b.insert(Psi1a.begin(), Psi1a.end());
	for (map<Determinant, double>::iterator it = Psi1b.begin(); it != Psi1b.end(); it++) 
	  it->second = 0.0;
	for (int i=0; i<Sample2.size(); i++) {
	  int I = Sample2[i];
	  getDeterminants(Dets[I], abs(epsilon2/ci(I,0)), wts2[i], I1, I2, I2HB, irrep, coreE, E0, Psi1b, SortedDets, false);
	}


	map<Determinant, double>::iterator it2 = Psi1b.begin();
	for (map<Determinant, double>::iterator it = Psi1a.begin(); it != Psi1a.end(); it++) {
	  it->first.getRepArray(psiArray);
	  energyEN += it->second*it2->second/(Energy(psiArray, norbs, I1, I2, coreE)-E0); it2++;
	}
	sampleSize = Sample1.size();
      }
      parLoopIter++;
      cout << "done "<<(parLoopIter)*num_thrds <<"  "<<E0-energyEN/parLoopIter/num_thrds<<" "<<"  "<<getTime()-startofCalc<<"  "<<sampleSize<<endl;
    }

  }

  return 0;
}
