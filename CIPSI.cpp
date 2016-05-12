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
using namespace Eigen;
int Determinant::norbs = 1; //spin orbitals

int sample_N(std::vector<double> wts, double cumulative, std::vector<int>& Sample1){
  for (int s=0; s<Sample1.size(); s++) {
    double rand_no = ((double) rand() / (RAND_MAX))*cumulative;
    for (int i=0; i<wts.size(); i++) {
      if (rand_no <wts[i]) {
	Sample1[s] = i;
	cumulative -= wts[i];
	wts[i] = 0.0;
      }
      rand_no -= wts[i];
    }
  }
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


void readInput(string input, std::vector<int>& occupied, double& eps1, double& eps2, double& tol,int& num_thrds) {
  ifstream dump(input.c_str());
  int nocc; dump >>nocc;
  occupied.resize(nocc);
  for (int i=0; i<nocc; i++)
    dump >> occupied[i];
  dump >> eps1 >>eps2>>tol>>num_thrds;
}

double getTime() {
  struct timeval start;
  gettimeofday(&start, NULL);
  return start.tv_sec + 1.e-6*start.tv_usec;
}


int main(int argc, char* argv[]) {
  double startofCalc=getTime();
  std::cout.precision(15);
  twoInt I2; oneInt I1; int nelec; int norbs; double coreE;
  readIntegrals("FCIDUMP", I2, I1, nelec, norbs, coreE);
  norbs *=2;
  Determinant::norbs = norbs; //spin orbitals

  int num_thrds;
  std::vector<int> HFoccupied; double epsilon1, epsilon2, tol;
  readInput("input.dat", HFoccupied, epsilon1, epsilon2, tol, num_thrds);
  //make HF determinant
  Determinant d;
  std::cout << norbs<<std::endl;
  for (int i=0; i<HFoccupied.size(); i++) {
    std::cout <<i<<"  "<< HFoccupied[i]<<std::endl;
    d.setocc(HFoccupied[i], true);
  }

  char detchar[norbs]; d.getRepArray(detchar);
  double EHF= Energy(detchar,norbs,I1,I2,coreE);
  std::cout << "HF = "<<EHF<<std::endl;

  unsigned short closed[nelec], open[norbs-nelec];
  int o = d.getOpenClosed(open, closed); int v=norbs-o;
  std::vector<Determinant> Dets(1,d), prevDets(1,d);
  std::vector<Determinant> SortedDets(1,d);
  MatrixXd ci(1,1), prevci(1,1); ci(0,0) = 1.0; prevci *= 0.0;

  int niter = 5;
  double E0 = EHF;
  std::vector<char> detChar(norbs); d.getRepArray(&detChar[0]);
  MatrixXd diagOld(1,1); diagOld(0,0) = EHF;
  int prevSize = 0;
  std::vector<std::vector<int> > connections(1, std::vector<int>(1,0));
  std::vector<std::vector<double> > Helements(1, std::vector<double>(1,EHF));

  omp_set_num_threads(num_thrds);
  std::cout << "max thrds "<<num_thrds<<std::endl;
  int iter = 0;
  //do the variational bit
  while(true){

    /*
    set<Determinant> newDets;
    getDeterminants(prevDets,  abs(epsilon1), I1, I2, coreE, E0, newDets, ci);
    //for (int k=0; k<newDets.size(); k++) {
    for (set<Determinant>::iterator it=newDets.begin(); it!=newDets.end(); ++it) {
      if (find(Dets.begin(), Dets.end(), *it) == Dets.end()) 
	Dets.push_back(*it);
    }
    */

    std::vector<set<Determinant> > newDets(num_thrds);
#pragma omp parallel for schedule(dynamic)
    for (int i=0; i<prevDets.size(); i++) 
      getDeterminants(Dets[i], abs(epsilon1/ci(i,0)), I1, I2, coreE, E0, newDets[omp_get_thread_num()]);

    for (int thrd=0; thrd<num_thrds; thrd++) 
      for (set<Determinant>::iterator it=newDets[thrd].begin(); it!=newDets[thrd].end(); ++it) 
	if  (find(Dets.begin(), Dets.end(), *it) == Dets.end())
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
	  if (abs(hij) > 1.e-10) {
	    connections[i].push_back(j);
	    Helements[i].push_back(hij);
	  }
	}
      }


    //Hmult H(&detChar[0], norbs, I1, I2, coreE);
    Hmult2 H(connections, Helements);
    E0 = davidson(H, X0, diag, 5, tol, true);
    std::cout <<iter<<"  "<<Dets.size()<<"  "<< E0 <<"  "<<getTime()-startofCalc<<std::endl;iter++;
    ci.resize(Dets.size(),1); ci = 1.0*X0;
    diagOld.resize(Dets.size(),1); diagOld = 1.0*diag;

    if (1.*(Dets.size()-prevDets.size())/prevDets.size() < 0.01)  {
      break;
    }
    prevSize = prevDets.size();

    prevDets.clear();prevDets=Dets;
  }
  //now do the perturbative bit

  if (false) {
    char psiArray[norbs];
    double energyEN = 0.0;
    set<Determinant> Psi1;
    for (int i=0; i<Dets.size(); i++) {
      std::vector<Determinant> newDets;
      getDeterminants(Dets[i], abs(epsilon2/ci(i,0)), I1, I2, coreE, E0, newDets);
      
      for (int j=0; j<newDets.size(); j++) {
	if ( (find(Dets.begin(), Dets.end(), newDets[j]) == Dets.end()) &&
	     (Psi1.find(newDets[j]) == Psi1.end()) ) {
	  Psi1.insert(newDets[j]);
	  
	  double integral = 0.0;
	  newDets[j].getRepArray(psiArray);
	  //#pragma omp parallel for reduction(+:integral)
	  for (int k=0; k<Dets.size(); k++) {
	    if (Dets[k].connected(newDets[j])) 
	      integral += Hij(&detChar[k*norbs], psiArray, norbs, I1, I2, coreE)*ci(k,0); 
	  }
	  energyEN += integral*integral/(Energy(psiArray, norbs, I1, I2, coreE)-E0);
	}
      }     
      
      if (i%100 == 0)
	cout << "done "<<i <<"  "<<energyEN<<" "<<getTime()-startofCalc<<endl;
    }
    cout <<energyEN<<"  "<< -energyEN+E0<<endl;
  }
  else {
    SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
    int niter = 1000;
    double eps = 0.1;
    double energyEN = 0.0;

    int parallelLoops = niter/num_thrds; int parLoopIter = 0;
    while(parLoopIter < parallelLoops) {

#pragma omp parallel for schedule(static) reduction(+:energyEN)
      for (int iter=0; iter<num_thrds; iter++) {
	char psiArray[norbs];
	std::vector<double> wts1, wts2; std::vector<int> Sample1, Sample2;
	wts1.reserve(1000); wts2.reserve(1000); Sample1.reserve(1000); Sample2.reserve(1000);
	
	Sample1.resize(0); wts1.resize(0); Sample2.resize(0); wts2.resize(0);
	sample_round(ci, eps, Sample1, wts1);
	sample_round(ci, eps, Sample2, wts2);
	
	set<Determinant> Psi1;
	for (int i=0; i<Sample1.size(); i++) {
	  int I = Sample1[i];
	  std::vector<Determinant> newDets;
	  getDeterminants(Dets[I], abs(epsilon2/wts1[i]), I1, I2, coreE, E0, newDets);
	  
	  for (int j=0; j<newDets.size(); j++) {
	    if ( (!binary_search(SortedDets.begin(), SortedDets.end(), newDets[j])) &&
		 (Psi1.find(newDets[j]) == Psi1.end()) ) {
	      Psi1.insert(newDets[j]);
	      
	      double integral1 = 0.0, integral2 = 0.0;
	      
	      newDets[j].getRepArray(psiArray);
	      for (int k=0; k<Sample1.size(); k++) {
		int K = Sample1[k];
		if (Dets[K].connected(newDets[j])) 
		  integral1 += Hij(&detChar[K*norbs], psiArray, norbs, I1, I2, coreE)*wts1[k]; 
	      }
	      for (int k=0; k<Sample2.size(); k++) {
		int K = Sample2[k];
		if (Dets[K].connected(newDets[j])) 
		  integral2 += Hij(&detChar[K*norbs], psiArray, norbs, I1, I2, coreE)*wts2[k]; 
	      }
	      
	      energyEN += integral1*integral2/(Energy(psiArray, norbs, I1, I2, coreE)-E0);
	    }
	  }
	}
      }
      parLoopIter++;
      cout << "done "<<(parLoopIter)*num_thrds <<"  "<<E0-energyEN/parLoopIter/num_thrds<<" "<<getTime()-startofCalc<<endl;
    }
    //cout <<energyEN/niter<<"  "<< -energyEN/niter+E0<<endl;

  }
  return 0;
}
