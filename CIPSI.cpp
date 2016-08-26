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
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "communicate.h"

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




void readInput(string input, std::vector<int>& occupied, CIPSIbasics::schedule& schd);


int main(int argc, char* argv[]) {

#ifndef SERIAL
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
#endif

  startofCalc=getTime();
  srand(startofCalc+world.rank());
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
  std::vector<int> HFoccupied; //double epsilon1, epsilon2, tol, dE;
  CIPSIbasics::schedule schd;
  if (mpigetrank() == 0) readInput("input.dat", HFoccupied, schd); //epsilon1, epsilon2, tol, num_thrds, eps, dE);

#ifndef SERIAL
  mpi::broadcast(world, HFoccupied, 0);
  mpi::broadcast(world, schd, 0);
#endif

  //make HF determinant
  Determinant d;
  for (int i=0; i<HFoccupied.size(); i++) {
    d.setocc(HFoccupied[i], true);
  }

  //have the dets, ci coefficient and diagnoal on all processors
  MatrixXd ci(1,1); ci(0,0) = 1.0;
  std::vector<Determinant> Dets(1,d);

  double E0 = CIPSIbasics::DoVariational(ci, Dets, schd, I2, I2HB, irrep, I1, coreE);

  //print the 5 most important determinants and their weights
  MatrixXd prevci = 1.*ci;
  for (int i=0; i<10; i++) {
    compAbs comp;
    int m = distance(&prevci(0,0), max_element(&prevci(0,0), &prevci(0,0)+prevci.rows(), comp));
    pout <<"#"<< i<<"  "<<prevci(m,0)<<"  "<<Dets[m]<<endl;
    prevci(m,0) = 0.0;
  }
  prevci.resize(0,0);



  //now do the perturbative bit
  if (!schd.stochastic && schd.nblocks == 1) {
    //CIPSIbasics::DoPerturbativeDeterministicLCC(Dets, ci, E0, I1, I2, I2HB, irrep, schd, coreE, nelec);
    CIPSIbasics::DoPerturbativeDeterministic(Dets, ci, E0, I1, I2, I2HB, irrep, schd, coreE, nelec);
  }
  else if (!schd.stochastic) {
    CIPSIbasics::DoBatchDeterministic(Dets, ci, E0, I1, I2, I2HB, irrep, schd, coreE, nelec);
  }
  else if (schd.SampleN == -1 && schd.singleList){
    CIPSIbasics::DoPerturbativeStochasticSingleList(Dets, ci, E0, I1, I2, I2HB, irrep, schd, coreE, nelec);
  }
  else if (schd.SampleN == -1 && !schd.singleList){
    CIPSIbasics::DoPerturbativeStochastic(Dets, ci, E0, I1, I2, I2HB, irrep, schd, coreE, nelec);
  }
  else if (schd.SampleN != -1 && schd.singleList){
    CIPSIbasics::DoPerturbativeStochastic2SingleList(Dets, ci, E0, I1, I2, I2HB, irrep, schd, coreE, nelec);
  }
  else { 
    //Here I will implement the alias method
    CIPSIbasics::DoPerturbativeStochastic2(Dets, ci, E0, I1, I2, I2HB, irrep, schd, coreE, nelec);
  }

  return 0;
}
