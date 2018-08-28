/*
  Developed by Sandeep Sharma with contributions from James E. T. Smith and Adam A. Holmes, 2017
  Copyright (c) 2017, Sandeep Sharma
  
  This file is part of DICE.
  
  This program is free software: you can redistribute it and/or modify it under the terms
  of the GNU General Public License as published by the Free Software Foundation, 
  either version 3 of the License, or (at your option) any later version.
  
  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
  See the GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License along with this program. 
  If not, see <http://www.gnu.org/licenses/>.
*/
#include "global.h"
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include "Determinants.h"
#include "SHCImakeHamiltonian.h"
#include "SHCIshm.h"
#include "input.h"
#include "integral.h"
#include "Hmult.h"
#include "SHCIbasics.h"
#include "Davidson.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <set>
#include <list>
#include <tuple>
#include "boost/format.hpp"
#include "new_anglib.h"
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/serialization/vector.hpp>
#include "communicate.h"
#include "SOChelper.h"
#include "symmetry.h"

using namespace Eigen;
using namespace boost;
using namespace SHCIbasics;
int HalfDet::norbs = 1; //spin orbitals
int Determinant::norbs = 1; //spin orbitals
int Determinant::EffDetLen = 1;
char Determinant::Trev = 0;  // Time reversal
Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> Determinant::LexicalOrder ;
//get the current time
double getTime() {
  struct timeval start;
  gettimeofday(&start, NULL);
  return start.tv_sec + 1.e-6*start.tv_usec;
}
double startofCalc = getTime();

// boost::interprocess::shared_memory_object int2Segment;
// boost::interprocess::mapped_region regionInt2;
// boost::interprocess::shared_memory_object int2SHMSegment;
// boost::interprocess::mapped_region regionInt2SHM;

void readInput(string input, vector<std::vector<int> >& occupied, schedule& schd);
double getdEusingDeterministicPT(vector<Determinant>& Dets, vector<MatrixXx>& ci, int DetsSize, 
				 vector<double>& E0, oneInt& I1, twoInt& I2,
				 twoIntHeatBathSHM& I2HB, vector<int>& irrep,
				 schedule& schd, double coreE, int nelec) ;

void initDets(vector<MatrixXx>& ci, vector<Determinant>& Dets,
	      schedule& schd, vector<vector<int> >& HFoccupied);

int main(int argc, char* argv[]) {

#ifndef SERIAL
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
#endif

  // Initialize
  initSHM();

  //Read the input file
  std::vector<std::vector<int> > HFoccupied;
  schedule schd;
  string inputFile = "input.dat";
  if (argc > 1) inputFile = string(argv[1]);
  if (commrank == 0) readInput(inputFile, HFoccupied, schd);
  if(DetLen % 2 ==1) {
    pout << "Change DetLen in global to an even number and recompile." << endl;
    exit(0);
  }

#ifndef SERIAL
  mpi::broadcast(world, HFoccupied, 0);
  mpi::broadcast(world, schd, 0);
#endif


  //set the random seed
  startofCalc=getTime();
  srand(schd.randomSeed + commrank);
  if (schd.outputlevel > 1) pout << "#using seed: " << schd.randomSeed << endl;

  std::cout.precision(15);

  //set up shared memory files to store the integrals
  // string shciint2 = "SHCIint2" + to_string(static_cast<long long>(time(NULL) % 1000000));
  // string shciint2shm = "SHCIint2shm" + to_string(static_cast<long long>(time(NULL) % 1000000));
  // int2Segment = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, shciint2.c_str(), boost::interprocess::read_write);
  // int2SHMSegment = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, shciint2shm.c_str(), boost::interprocess::read_write);


  //read the hamiltonian (integrals, orbital irreps, num-electron etc.)
  twoInt I2; 
  oneInt I1; 
  int nelec; 
  int norbs; 
  double coreE, eps;
  std::vector<int> irrep;
  readIntegrals(schd.integralFile, I2, I1, nelec, norbs, coreE, irrep);

  int num_thrds;

  // Check
  if (HFoccupied[0].size() != nelec) {
    pout << "The number of electrons given in the FCIDUMP should be";
    pout << " equal to the nocc given in the shci input file." << endl;
    exit(0);
  }

  // Setup the lexical table for the determinants
  norbs *=2;
  Determinant::norbs = norbs; //spin orbitals
  HalfDet::norbs = norbs; //spin orbitals
  Determinant::EffDetLen = norbs/64+1;
  Determinant::initLexicalOrder(nelec);
  if (Determinant::EffDetLen >DetLen) {
    cout << "change DetLen in global.h to " << Determinant::EffDetLen <<" and recompile " << endl;
    exit(0);
  }

  // Initialize the Heat-Bath integrals
  std::vector<int> allorbs;
  for (int i = 0; i < norbs / 2; i++) allorbs.push_back(i);
  twoIntHeatBath I2HB(1.e-10);
  twoIntHeatBathSHM I2HBSHM(1.e-10);
  if (commrank == 0) I2HB.constructClass(allorbs, I2, I1, norbs / 2);
  I2HBSHM.constructClass(norbs / 2, I2HB);

  readSOCIntegrals(I1, norbs, "SOC");
  // vector<MatrixXx> citmp(2 , MatrixXx::Zero(HFoccupied.size(), 1));
  // vector<Determinant> Detstmp;
  // vector<double> E0tmp = SHCIbasics::DoVariational(citmp, Detstmp, schd, I2, I2HBSHM,
	// 					  irrep, I1, coreE, nelec, schd.DoRDM);
  // for(int ii=0;ii<citmp.size();ii++)
  // for(int jj=0;jj<citmp.size();jj++){
  //   cout << citmp[ii][jj];
  //   if(jj==citmp.size()-1) cout << endl;
  // }
  // exit(0);

#ifndef SERIAL
  mpi::broadcast(world, I1, 0);
#endif

  //initialize L and S integrals
  vector<oneInt> L(3), S(3);
  for (int i=0; i<3; i++) {
    L[i].store.resize(norbs*norbs, 0.0);
    L[i].norbs = norbs;

    S[i].store.resize(norbs*norbs, 0.0);
    S[i].norbs = norbs;
  }
  //read L integrals
  readGTensorIntegrals(L, norbs, "GTensor");

  //generate S integrals
  double ge = 2.002319304;
  for (int a=1; a<norbs/2+1; a++) {
    S[0](2*(a-1), 2*(a-1)+1) += ge/2.;  //alpha beta
    S[0](2*(a-1)+1, 2*(a-1)) += ge/2.;  //beta alpha

    S[1](2*(a-1), 2*(a-1)+1) += std::complex<double>(0, -ge/2.);  //alpha beta
    S[1](2*(a-1)+1, 2*(a-1)) += std::complex<double>(0,  ge/2.);  //beta alpha

    S[2](2*(a-1), 2*(a-1)) +=  ge/2.;  //alpha alpha
    S[2](2*(a-1)+1, 2*(a-1)+1) += -ge/2.;  //beta beta
  }

  // std::cout.precision(15);
  vector<MatrixXx> ci(2 , MatrixXx::Zero(HFoccupied.size(), 1));
  vector<Determinant> Dets;

  //the perturbation is S[i]+L[i]
  double epsilon = 1.e-4;

  vector<double> fpm(6,0.0); 
  vector<double> dpm(6,0.0);//these are f+ and f- function evaluations
  MatrixXx Gtensor = MatrixXx::Zero(3,3);

  for (int a=0; a<3; a++) {
    initDets(ci, Dets, schd, HFoccupied);
    for (int i=0; i<I1.store.size(); i++) {
      I1.store.at(i) += (1.*L[a].store.at(i)+S[a].store.at(i))*epsilon;
    }
    vector<double> E0 = SHCIbasics::DoVariational(ci, Dets, schd, I2, I2HBSHM,
						  irrep, I1, coreE, nelec, schd.DoRDM);

    pout << ((E0[1]-E0[0])/epsilon -ge)*1e6<<"  "<<endl;
    int DetsSize = Dets.size();
    if (!schd.stochastic) {
      fpm[2*a] = pow(getdEusingDeterministicPT(Dets, ci, DetsSize,  E0, I1, I2, I2HBSHM, irrep, schd, coreE, nelec),2);
      // pout << "We can't support perturbation calculation now" << endl;
      // pout << "Please change stochastic input to 1 to use varitianal energy directly." << endl;
      // exit(0);
      pout << "PT" << fpm[2*a] << endl;
      pout << "D"  << pow(E0[1]-E0[0],2) << endl;
    }
    else
      fpm[2*a] = pow(E0[1]-E0[0],2);


    // pout << (sqrt(fpm[2*a])/epsilon - ge)*1e6<<endl;
    for (int i=0; i<I1.store.size(); i++) {
      I1.store.at(i) -= (1.*L[a].store.at(i)+S[a].store.at(i))*epsilon;
    }

    initDets(ci, Dets, schd, HFoccupied);
    for (int i=0; i<I1.store.size(); i++) {
      I1.store.at(i) -= (1.*L[a].store.at(i)+S[a].store.at(i))*epsilon;
    }
    E0 = SHCIbasics::DoVariational(ci, Dets, schd, I2, I2HBSHM,
				   irrep, I1, coreE, nelec, schd.DoRDM);
    DetsSize = Dets.size();

    if (!schd.stochastic) {
      fpm[2*a+1] = pow(getdEusingDeterministicPT(Dets, ci, DetsSize,  E0, I1, I2, I2HBSHM, irrep, schd, coreE, nelec),2);
      // pout << "We can't support perturbation calculation now" << endl;
      // pout << "Please change stochastic input to 1 to use varitianal energy directly." << endl;
      // exit(0);
      pout << "PT" << fpm[2*a+1] << endl;
      pout << "D"  << pow(E0[1]-E0[0],2) << endl;
    }
    else
      fpm[2*a+1] = pow(E0[1]-E0[0],2);
  

    for (int i=0; i<I1.store.size(); i++) {
      I1.store.at(i) += (1.*L[a].store.at(i)+S[a].store.at(i))*epsilon;
    }

    Gtensor(a,a) = (fpm[2*a] + fpm[2*a+1])/(epsilon*epsilon)/2;
  }

  int count=0;

  for (int a=0; a<3; a++)
  for (int b=0; b<a; b++)
  {
    double plusplus, minusminus;
    initDets(ci, Dets, schd, HFoccupied);
    for (int i=0; i<I1.store.size(); i++) {
      I1.store.at(i) += (1.*L[a].store.at(i)+S[a].store.at(i))*epsilon;
      I1.store.at(i) += (1.*L[b].store.at(i)+S[b].store.at(i))*epsilon;
    }
    vector<double> E0 = SHCIbasics::DoVariational(ci, Dets, schd, I2, I2HBSHM,
						  irrep, I1, coreE, nelec, schd.DoRDM);
    int DetsSize = Dets.size();

    if (!schd.stochastic) {
      plusplus = pow(getdEusingDeterministicPT(Dets, ci, DetsSize,  E0, I1, I2, I2HBSHM, irrep, schd, coreE, nelec),2);
      // pout << "We can't support perturbation calculation now" << endl;
      // pout << "Please change stochastic input to 1 to use varitianal energy directly." << endl;
      // exit(0);
      pout << "PT" << plusplus << endl;
      pout << "D"  << pow(E0[1]-E0[0],2) << endl;
    }
    else
      plusplus = pow(E0[1]-E0[0],2);


    for (int i=0; i<I1.store.size(); i++) {
      I1.store.at(i) -= (1.*L[a].store.at(i)+S[a].store.at(i))*epsilon;
      I1.store.at(i) -= (1.*L[b].store.at(i)+S[b].store.at(i))*epsilon;
    }


    initDets(ci, Dets, schd, HFoccupied);
    for (int i=0; i<I1.store.size(); i++) {
      I1.store.at(i) -= (L[a].store.at(i)+S[a].store.at(i))*epsilon;
      I1.store.at(i) -= (L[b].store.at(i)+S[b].store.at(i))*epsilon;
    }
    E0 = SHCIbasics::DoVariational(ci, Dets, schd, I2, I2HBSHM,
				   irrep, I1, coreE, nelec, schd.DoRDM);
    DetsSize = Dets.size();

    if (!schd.stochastic) {
      minusminus = pow(getdEusingDeterministicPT(Dets, ci, DetsSize,  E0, I1, I2, I2HBSHM, irrep, schd, coreE, nelec),2);
      // pout << "We can't support perturbation calculation now" << endl;
      // pout << "Please change stochastic input to 1 to use varitianal energy directly." << endl;
      // exit(0);    
      pout << "PT" << minusminus << endl;
      pout << "D"  << pow(E0[1]-E0[0],2) << endl;
    }
    else
      minusminus = pow(E0[1]-E0[0],2);


    for (int i=0; i<I1.store.size(); i++) {
      I1.store.at(i) += (L[a].store.at(i)+S[a].store.at(i))*epsilon;
      I1.store.at(i) += (L[b].store.at(i)+S[b].store.at(i))*epsilon;
    }

    dpm[2*count] = plusplus;
    dpm[2*count+1] = minusminus;

    count++;

    Gtensor(a,b) = (plusplus  - fpm[2*a] - fpm[2*b] - fpm[2*a+1] - fpm[2*b+1] + minusminus)/2/(epsilon*epsilon)/2;
    Gtensor(b,a) = Gtensor(a,b);
  }

  pout << endl;
  pout << Gtensor<<endl;

  SelfAdjointEigenSolver<MatrixXx> eigensolver(Gtensor);
  if (eigensolver.info() != Success) abort();
  cout <<endl<< "Gtensor eigenvalues for epsilon = " << epsilon <<endl;
  cout << str(boost::format("g1= %9.6f,  shift: %6.0f\n")%pow(eigensolver.eigenvalues()[0],0.5) % ((-ge+pow(eigensolver.eigenvalues()[0],0.5))*1.e6) );
  cout << str(boost::format("g2= %9.6f,  shift: %6.0f\n")%pow(eigensolver.eigenvalues()[1],0.5) % ((-ge+pow(eigensolver.eigenvalues()[1],0.5))*1.e6) );
  cout << str(boost::format("g3= %9.6f,  shift: %6.0f\n")%pow(eigensolver.eigenvalues()[2],0.5) % ((-ge+pow(eigensolver.eigenvalues()[2],0.5))*1.e6) );
  cout <<endl<< "Corresponding eigenvectors :" << endl;
  cout << eigensolver.eigenvectors() << endl;
  return 0;
}


void initDets(vector<MatrixXx>& ci, vector<Determinant>& Dets,
	      schedule& schd, vector<vector<int> >& HFoccupied) {

#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  ci.clear(); Dets.clear();
  ci.resize(schd.nroots, MatrixXx::Zero(HFoccupied.size(),1));
  Dets.resize(HFoccupied.size());
  for (int d=0;d<HFoccupied.size(); d++) {
    for (int i=0; i<HFoccupied[d].size(); i++) {
      Dets[d].setocc(HFoccupied[d][i], true);
    }
  }
  if (commrank == 0) {
    for (int j=0; j<ci[0].rows(); j++)
      ci[0](j,0) = 1.0;
    ci[0] = ci[0]/ci[0].norm();
  }
#ifndef SERIAL
  mpi::broadcast(world, ci, 0);
#endif
}

double getdEusingDeterministicPT(vector<Determinant>& Dets, vector<MatrixXx>& ci, int DetsSize, 
			       vector<double>& E0, oneInt& I1, twoInt& I2,
			       twoIntHeatBathSHM& I2HBSHM, vector<int>& irrep,
			       schedule& schd, double coreE, int nelec) {

  vector<MatrixXx> spinRDM(3);

  MatrixXx Heff = MatrixXx::Zero(E0.size(), E0.size());
  for (int root1 =0 ;root1<schd.nroots; root1++) {
    for (int root2=root1+1 ;root2<schd.nroots; root2++) {
      Heff(root1, root1) = 0.0; Heff(root2, root2) = 0.0; Heff(root1, root2) = 0.0;
      DoPerturbativeDeterministicOffdiagonal(Dets, ci[root1], E0[root1], ci[root2],
							 E0[root2], DetsSize,  I1,
							 I2, I2HBSHM, irrep, schd,
							 coreE, nelec, root1, Heff(root1,root1),
							 Heff(root2, root2), Heff(root1, root2),
							 spinRDM);
      Heff(root2, root1) = conj(Heff(root1, root2));
      pout << "Heff" << endl << Heff << endl;
    }
  }
  for (int root1 =0 ;root1<schd.nroots; root1++)
    Heff(root1, root1) += E0[root1];

  schd.doGtensor = true;

  SelfAdjointEigenSolver<MatrixXx> eigensolver(Heff);
  pout << "energyPT" << eigensolver.eigenvalues()(1,0)<<"  "<<eigensolver.eigenvalues()(0,0)<<endl;
  //exit(0);
  return eigensolver.eigenvalues()(1,0)-eigensolver.eigenvalues()(0,0);


}
