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
#include "Dice/Utils/global.h"
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include "Dice/Utils/Determinants.h"
#include "Dice/SHCImakeHamiltonian.h"
#include "Dice/Utils/input.h"
#include "Dice/Utils/integral.h"
#include "Dice/Hmult.h"
#include "Dice/SHCIbasics.h"
#include "Dice/Davidson.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <set>
#include <list>
#include <tuple>
#include "boost/format.hpp"
#include "Dice/new_anglib.h"
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include <boost/serialization/vector.hpp>
#include "Dice/Utils/communicate.h"
#include "Dice/SOChelper.h"

using namespace Eigen;
using namespace boost;
int HalfDet::norbs = 1; //spin orbitals
int Determinant::norbs = 1; //spin orbitals
int Determinant::EffDetLen = 1;
Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> Determinant::LexicalOrder ;
//get the current time
double getTime() {
  struct timeval start;
  gettimeofday(&start, NULL);
  return start.tv_sec + 1.e-6*start.tv_usec;
}
double startofCalc = getTime();

boost::interprocess::shared_memory_object int2Segment;
boost::interprocess::mapped_region regionInt2;
boost::interprocess::shared_memory_object int2SHMSegment;
boost::interprocess::mapped_region regionInt2SHM;


void readInput(string input, vector<std::vector<int> >& occupied, schedule& schd);
double getdEusingDeterministicPT(vector<Determinant>& Dets, vector<MatrixXx>& ci,
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

  //Read the input file
  std::vector<std::vector<int> > HFoccupied;
  schedule schd;
  if (mpigetrank() == 0) readInput("input.dat", HFoccupied, schd);

#ifndef SERIAL
  mpi::broadcast(world, HFoccupied, 0);
  mpi::broadcast(world, schd, 0);
#endif


  //set the random seed
  startofCalc=getTime();
  srand(schd.randomSeed+mpigetrank());


  //set up shared memory files to store the integrals
  string shciint2 = "SHCIint2" + to_string(static_cast<long long>(time(NULL) % 1000000));
  string shciint2shm = "SHCIint2shm" + to_string(static_cast<long long>(time(NULL) % 1000000));
  int2Segment = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, shciint2.c_str(), boost::interprocess::read_write);
  int2SHMSegment = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, shciint2shm.c_str(), boost::interprocess::read_write);


  //read the hamiltonian (integrals, orbital irreps, num-electron etc.)
  twoInt I2; oneInt I1; int nelec; int norbs; double coreE, eps;
  std::vector<int> irrep;
  readIntegrals("FCIDUMP", I2, I1, nelec, norbs, coreE, irrep);

  int num_thrds;

  norbs *=2;
  Determinant::norbs = norbs; //spin orbitals
  HalfDet::norbs = norbs; //spin orbitals
  Determinant::EffDetLen = norbs/64+1;
  Determinant::initLexicalOrder(nelec);
  if (Determinant::EffDetLen >DetLen) {
    cout << "change DetLen in global.h to "<<Determinant::EffDetLen<<" and recompile "<<endl;
    exit(0);
  }

  std::vector<int> allorbs;
  for (int i=0; i<norbs/2; i++)
    allorbs.push_back(i);
  twoIntHeatBath I2HB(1.e-10);
  twoIntHeatBathSHM I2HBSHM(1.e-10);
  if (mpigetrank() == 0) I2HB.constructClass(allorbs, I2, norbs/2);
  I2HBSHM.constructClass(norbs/2, I2HB);

  readSOCIntegrals(I1, norbs, "SOC");


  //initialize L and S integrals
  vector<oneInt> LplusS(3), L(3), S(3);
  for (int i=0; i<3; i++) {
    LplusS[i].store.resize(norbs*norbs, 0.0);
    LplusS[i].norbs = norbs;
  }
  //read L integrals
  readGTensorIntegrals(LplusS, norbs, "GTensor");

  //generate S integrals
  double ge = 2.002319304;
  for (int a=1; a<norbs/2+1; a++) {
    LplusS[0](2*(a-1), 2*(a-1)+1) += ge/2.;  //alpha beta
    LplusS[0](2*(a-1)+1, 2*(a-1)) += ge/2.;  //beta alpha

    LplusS[1](2*(a-1), 2*(a-1)+1) += std::complex<double>(0, -ge/2.);  //alpha beta
    LplusS[1](2*(a-1)+1, 2*(a-1)) += std::complex<double>(0,  ge/2.);  //beta alpha

    LplusS[2](2*(a-1), 2*(a-1)) +=  ge/2.;  //alpha alpha
    LplusS[2](2*(a-1)+1, 2*(a-1)+1) += -ge/2.;  //beta beta
  }

  std::cout.precision(15);
  vector<MatrixXx> ci;
  vector<Determinant> Dets;

  double Ezero = 0.0;
  initDets(ci, Dets, schd, HFoccupied);
  vector<double> E0 = SHCIbasics::DoVariational(ci, Dets, schd, I2, I2HBSHM,
						irrep, I1, coreE, nelec, schd.DoRDM);



  if (!schd.stochastic) {
    Ezero = getdEusingDeterministicPT(Dets, ci, E0, I1, I2, I2HBSHM, irrep, schd, coreE, nelec);
  }
  else
    Ezero = E0[0];


  //the perturbation is S[i]+L[i]
  double epsilon = 5.e-4;

  vector<double> Bfpm(6,0.0); //these are f+ and f- function evaluations
  MatrixXx Gtensor = MatrixXx::Zero(3,3);
  for (int a=0; a<3; a++) {
    initDets(ci, Dets, schd, HFoccupied);
    for (int i=0; i<I1.store.size(); i++) {
      I1.store.at(i) += (1.*L[a].store.at(i)+S[a].store.at(i))*epsilon;
    }
    vector<double> E0 = SHCIbasics::DoVariational(ci, Dets, schd, I2, I2HBSHM,
						  irrep, I1, coreE, nelec, schd.DoRDM);

    if (!schd.stochastic) {
      Bfpm[2*a] = getdEusingDeterministicPT(Dets, ci, E0, I1, I2, I2HBSHM, irrep, schd, coreE, nelec);
    }
    else
      Bfpm[2*a] = E0[0];

    for (int i=0; i<I1.store.size(); i++) {
      I1.store.at(i) -= (1.*L[a].store.at(i)+S[a].store.at(i))*epsilon;
    }

    initDets(ci, Dets, schd, HFoccupied);
    for (int i=0; i<I1.store.size(); i++) {
      I1.store.at(i) -= (1.*L[a].store.at(i)+S[a].store.at(i))*epsilon;
    }
    E0 = SHCIbasics::DoVariational(ci, Dets, schd, I2, I2HBSHM,
				   irrep, I1, coreE, nelec, schd.DoRDM);

    if (!schd.stochastic) {
      Bfpm[2*a+1] = getdEusingDeterministicPT(Dets, ci, E0, I1, I2, I2HBSHM, irrep, schd, coreE, nelec);
    }
    else
      Bfpm[2*a+1] = E0[0];

    for (int i=0; i<I1.store.size(); i++) {
      I1.store.at(i) += (1.*L[a].store.at(i)+S[a].store.at(i))*epsilon;
    }
  }


  vector<double> Sfpm(6,0.0); //these are f+ and f- function evaluations
  for (int a=0; a<3; a++) {
    initDets(ci, Dets, schd, HFoccupied);
    for (int i=0; i<I1.store.size(); i++) {
      I1.store.at(i) += (0.*L[a].store.at(i)+S[a].store.at(i))*epsilon;
    }
    vector<double> E0 = SHCIbasics::DoVariational(ci, Dets, schd, I2, I2HBSHM,
						  irrep, I1, coreE, nelec, schd.DoRDM);

    if (!schd.stochastic) {
      Sfpm[2*a] = getdEusingDeterministicPT(Dets, ci, E0, I1, I2, I2HBSHM, irrep, schd, coreE, nelec);
    }
    else
      Sfpm[2*a] = E0[0];

    for (int i=0; i<I1.store.size(); i++) {
      I1.store.at(i) -= (0.*L[a].store.at(i)+S[a].store.at(i))*epsilon;
    }

    initDets(ci, Dets, schd, HFoccupied);
    for (int i=0; i<I1.store.size(); i++) {
      I1.store.at(i) -= (0.*L[a].store.at(i)+S[a].store.at(i))*epsilon;
    }
    E0 = SHCIbasics::DoVariational(ci, Dets, schd, I2, I2HBSHM,
				   irrep, I1, coreE, nelec, schd.DoRDM);

    if (!schd.stochastic) {
      Sfpm[2*a+1] = getdEusingDeterministicPT(Dets, ci, E0, I1, I2, I2HBSHM, irrep, schd, coreE, nelec);
    }
    else
      Sfpm[2*a+1] = E0[0];

    for (int i=0; i<I1.store.size(); i++) {
      I1.store.at(i) += (0.*L[a].store.at(i)+S[a].store.at(i))*epsilon;
    }

  }

  for (int a=0; a<3; a++)
  for (int b=0; b<a+1; b++)
  {
    double plusplus, minusminus;
    initDets(ci, Dets, schd, HFoccupied);
    for (int i=0; i<I1.store.size(); i++) {
      I1.store.at(i) += (1.*L[a].store.at(i)+S[a].store.at(i))*epsilon;
      I1.store.at(i) += (0.*L[b].store.at(i)+S[b].store.at(i))*epsilon;
    }
    vector<double> E0 = SHCIbasics::DoVariational(ci, Dets, schd, I2, I2HBSHM,
						  irrep, I1, coreE, nelec, schd.DoRDM);

    if (!schd.stochastic) {
      plusplus = getdEusingDeterministicPT(Dets, ci, E0, I1, I2, I2HBSHM, irrep, schd, coreE, nelec);
    }
    else
      plusplus = E0[0];



    for (int i=0; i<I1.store.size(); i++) {
      I1.store.at(i) -= (1.*L[a].store.at(i)+S[a].store.at(i))*epsilon;
      I1.store.at(i) -= (0.*L[b].store.at(i)+S[b].store.at(i))*epsilon;
    }


    initDets(ci, Dets, schd, HFoccupied);
    for (int i=0; i<I1.store.size(); i++) {
      I1.store.at(i) -= (L[a].store.at(i)+S[a].store.at(i))*epsilon;
      I1.store.at(i) -= (0.*L[b].store.at(i)+S[b].store.at(i))*epsilon;
    }
    E0 = SHCIbasics::DoVariational(ci, Dets, schd, I2, I2HBSHM,
				   irrep, I1, coreE, nelec, schd.DoRDM);

    if (!schd.stochastic) {
      minusminus = getdEusingDeterministicPT(Dets, ci, E0, I1, I2, I2HBSHM, irrep, schd, coreE, nelec);
    }
    else
      minusminus = E0[0];

    for (int i=0; i<I1.store.size(); i++) {
      I1.store.at(i) += (L[a].store.at(i)+S[a].store.at(i))*epsilon;
      I1.store.at(i) += (0.*L[b].store.at(i)+S[b].store.at(i))*epsilon;
    }

    cout << plusplus<<"  "<<Bfpm[0]<<"  "<<Sfpm[0]<<"  "<<Ezero<<"  "<<Bfpm[1]<<"  "<<Sfpm[1]<<"  "<<minusminus<<endl;
    Gtensor(a,b) = (plusplus  - Bfpm[2*a] - Sfpm[2*b] +2*Ezero - Bfpm[2*a+1] - Sfpm[2*b+1]+minusminus)/2/(epsilon*epsilon);
    Gtensor(b,a) = Gtensor(a,b);
    cout << Gtensor(0,0)<<endl;
    exit(0);
  }


  cout << Gtensor<<endl;

  SelfAdjointEigenSolver<MatrixXx> eigensolver(Gtensor);
  if (eigensolver.info() != Success) abort();
  cout <<endl<< "Gtensor eigenvalues"<<endl;
  cout << str(boost::format("g1= %9.6f,  shift: %6.0f\n")%pow(eigensolver.eigenvalues()[0],1.0) % ((-ge+pow(eigensolver.eigenvalues()[0],1.0))*1.e6) );
  cout << str(boost::format("g2= %9.6f,  shift: %6.0f\n")%pow(eigensolver.eigenvalues()[1],1.0) % ((-ge+pow(eigensolver.eigenvalues()[1],1.0))*1.e6) );
  cout << str(boost::format("g3= %9.6f,  shift: %6.0f\n")%pow(eigensolver.eigenvalues()[2],1.0) % ((-ge+pow(eigensolver.eigenvalues()[2],1.0))*1.e6) );

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
  if (mpigetrank() == 0) {
    for (int j=0; j<ci[0].rows(); j++)
      ci[0](j,0) = 1.0;
    ci[0] = ci[0]/ci[0].norm();
  }
#ifndef SERIAL
  mpi::broadcast(world, ci, 0);
#endif
}

double getdEusingDeterministicPT(vector<Determinant>& Dets, vector<MatrixXx>& ci,
			       vector<double>& E0, oneInt& I1, twoInt& I2,
			       twoIntHeatBathSHM& I2HBSHM, vector<int>& irrep,
			       schedule& schd, double coreE, int nelec) {



  schd.doGtensor = false; ///THIS IS DONE BECAUSE WE DONT WANT TO doperturbativedeterministicoffdiagonal to calculate rdm
  vector<MatrixXx> spinRDM(3);
  MatrixXx Heff = MatrixXx::Zero(E0.size(), E0.size());
  for (int root1 =0 ;root1<schd.nroots; root1++) {
    for (int root2=root1+1 ;root2<schd.nroots; root2++) {
      Heff(root1, root1) = 0.0; Heff(root2, root2) = 0.0; Heff(root1, root2) = 0.0;
      SHCIbasics::DoPerturbativeDeterministicOffdiagonal(Dets, ci[root1], E0[root1], ci[root2],
							 E0[root2], I1,
							 I2, I2HBSHM, irrep, schd,
							 coreE, nelec, root1, Heff(root1,root1),
							 Heff(root2, root2), Heff(root1, root2),
							 spinRDM);
      Heff(root2, root1) = conj(Heff(root1, root2));
    }
  }
  for (int root1 =0 ;root1<schd.nroots; root1++)
    Heff(root1, root1) += E0[root1];

  schd.doGtensor = true;

  SelfAdjointEigenSolver<MatrixXx> eigensolver(Heff);
  //cout << eigensolver.eigenvalues()(1,0)<<"  "<<eigensolver.eigenvalues()(0,0)<<endl;
  //exit(0);
  return eigensolver.eigenvalues()(0,0);


}
