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
  if (!schd.stochastic) {
    CIPSIbasics::DoPerturbativeDeterministic(Dets, ci, E0, I1, I2, I2HB, irrep, schd, coreE, nelec);
  }
  else if (true){
    CIPSIbasics::DoPerturbativeStochastic(Dets, ci, E0, I1, I2, I2HB, irrep, schd, coreE, nelec);
  }
  else { 
    //Here I will implement the alias method
  }

  return 0;
}
