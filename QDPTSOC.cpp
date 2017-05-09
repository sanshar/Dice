/*
Developed by Sandeep Sharma with contributions from James E. Smith and Adam A. Homes, 2017
Copyright (c) 2017, Sandeep Sharma

This file is part of DICE.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "global.h"
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include "Determinants.h"
#include "SHCImakeHamiltonian.h"
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
#include <boost/serialization/vector.hpp>
#include "communicate.h"
#include "SOChelper.h"

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
  int nelec = HFoccupied[0].size();


  //set the random seed
  startofCalc=getTime();
  srand(schd.randomSeed+mpigetrank());


  int num_thrds;

  int norbs = readNorbs("FCIDUMP");
  norbs *=2; //spin orbitals
  Determinant::norbs = norbs; //spin orbitals
  HalfDet::norbs = norbs; //spin orbitals
  Determinant::EffDetLen = norbs/64+1;
  Determinant::initLexicalOrder(nelec);
  oneInt I1; I1.store.resize(norbs*norbs, 0.0); I1.norbs = norbs;
  readSOCIntegrals(I1, norbs, "SOC");


  //have the dets, ci coefficient and diagnoal on all processors
  vector<MatrixXd> ci;
  std::vector<Determinant> Dets, SortedDets;
  std::vector<double> Energy;
  MatrixXd diag;
  for (int i=0; i<schd.prefix.size(); i++)
  {
    char file [5000];
    sprintf (file, "%s/%d-variational.bkp" , schd.prefix[i].c_str(), mpigetrank() );
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);

    vector<MatrixXd> Localci;
    std::vector<Determinant> LocalDets, LocalSortedDets;
    std::vector<double> LocalEnergy;
    MatrixXd Localdiag;

    int iter;
    load >> iter >> LocalDets >> LocalSortedDets ;
    LocalSortedDets.size();
    int diaglen;
    load >> diaglen;
    Localdiag.resize(diaglen,1);
    for (int j=0; j<Localdiag.rows(); j++)
      load >> Localdiag(j,0);
    load >> Localci;
    load >> LocalEnergy;

    //We assume that the states from different prefixes are of different
    //symmetry
    if (i == 0) {
      ci = Localci;
      diag = Localdiag;
    }
    else {
      std::vector<MatrixXd> bkpci = ci;
      ci.clear();
      ci.resize(bkpci.size()+Localci.size(), MatrixXd::Zero(Localci[0].rows()+bkpci[0].rows(),1));
      for (int j=0; j<bkpci.size(); j++)
	ci[j].block(0,0,bkpci[0].rows(),1) = 1.*bkpci[j];
      for (int j=0; j<Localci.size(); j++) {
	ci[j+bkpci.size()].block(bkpci[0].rows(),0,Localci[0].rows(),1) = 1.*Localci[j];
      }
    }
    Dets.insert(Dets.end(), LocalDets.begin(), LocalDets.end());
    Energy.insert(Energy.end(), LocalEnergy.begin(), LocalEnergy.end());
  }
#ifndef SERIAL
  mpi::broadcast(world, ci, 0);
  mpi::broadcast(world, Dets, 0);
  mpi::broadcast(world, diag, 0);
  mpi::broadcast(world, Energy, 0);
#endif

  //Locate all determinants that are S+ |dets>  and S- |dets>
  //and add them to determinants. Rougly increases the space 3 times.

  //vector<Determinant> VecSplusDets;
  vector<vector<size_t> > SplusConnections;
  {
    //Now find out the Ms and S^2
    std::map<Determinant, vector<size_t> > SplusDets;
    std::map<Determinant, vector<size_t> > SminusDets;

    for (int i=0; i<Dets.size(); i++) { //loop over all determinants
      for (int j=0; j<norbs/2 ; j++) {  //loop over spatial orbitals
	if (Dets[i].getocc(2*j+1) &&
	    !Dets[i].getocc(2*j  )   ) {//if beta-filled and alpha-empty

	  Determinant d = Dets[i];
	  d.setocc(2*j+1, false); //empty the beta
	  d.setocc(2*j  , true);  //fill up the alpha

	  SplusDets[d].push_back(i);
	}//if
      }// j (orbitals)
    }//i dets


    SplusConnections.reserve(SplusDets.size());
    //add the Splus determinants to Dets
    for (auto it = SplusDets.begin(); it != SplusDets.end(); it++) {
      Dets.push_back(it->first);
      SplusConnections.push_back(it->second);
      it->second.clear();
    }


    for (int i=0; i<ci[0].rows(); i++) { //loop over all determinants
      for (int j=0; j<norbs/2 ; j++) {  //loop over spatial orbitals
	if (Dets[i].getocc(2*j) &&
	    !Dets[i].getocc(2*j+1  )   ) {//if beta-filled and alpha-empty

	  Determinant d = Dets[i];
	  d.setocc(2*j, false); //empty the beta
	  d.setocc(2*j+1, true);  //fill up the alpha

	  SminusDets[d].push_back(i);
	}//if
      }// j (orbitals)
    }//i dets

    //add S- dets to the Determinants
    for (auto it = SminusDets.begin(); it != SminusDets.end(); it++) {
      Dets.push_back(it->first);
    }

  }


  //for (int i=0; i<Dets.size(); i++)
  //cout << i<<"  "<<Dets[i]<<endl;

  //beings0<---->beginSp are original dets
  //beginSp<---->beginSm are the S+ dets
  //beginSm<---->Dets.end() are the S- dets
  auto beginS0 = Dets.begin();
  auto beginSp = Dets.begin()+ci[0].rows();
  auto beginSm = Dets.begin()+ci[0].rows()+SplusConnections.size();


  int Sz = Dets[0].Nalpha()-Dets[0].Nbeta();
  //Now calculate S^2 for each state
  std::vector<double> SpinSquare(ci.size(), 0.0);

  //Sz(Sz+1)
  for (int j=0; j<ci.size(); j++) {
    for (int i=0; i<ci[0].rows(); i++) {
      int M = Dets[i].Nalpha() - Dets[i].Nbeta();
      if (M != Sz) {
	cout << "Sz is not the same for all determinants."<<endl;
	exit(0);
      }
      SpinSquare[j] += ci[j](i,0)*ci[j](i,0)*M*0.5*(M*0.5+1);
    }
  }

  int hsubspaceSize = 0;
  //S+S-
  for (int detIndex = ci[0].rows(); detIndex<ci[0].rows()+SplusConnections.size(); detIndex++) {
    std::vector<size_t>& connections = SplusConnections[detIndex-ci[0].rows()];
    for (int k=0; k<ci.size(); k++) {

      double afterSplus = 0.0;
      //sum up all the contributions toward determinant it->first
      //\sum_j S+c_j|Dj>  -> sum_j c_j |it->first>
      for (int i=0; i<connections.size(); i++) {
	int I = connections[i];
	afterSplus += ci[k](I,0);
      }

      for (int j=0; j<connections.size(); j++) {
	int J = connections[j];
	SpinSquare[k] += ci[k](J,0)*afterSplus;
      }

    }
  }

  //Now we have S^2 we want 2S from it
  vector<int> Spin(ci.size(), 0);
  vector<int> rowIndex(ci.size(), 0);
  //calculate S from <S^2> = S(S+1)
  for (int j=0; j<ci.size(); j++) {
    int s = static_cast<int>( (-1.0 + pow(1+4*SpinSquare[j], 0.5)) + 0.5); //2S
    Spin[j] = s;
    hsubspaceSize += s+1; //2s+1
    if (j != 0)
      rowIndex[j] = rowIndex[j-1]+Spin[j-1]+1;
  }

  cout << str(boost::format("PERFORMING QDPTSOC with %s roots\n")%ci.size());
  for (int j=0; j<ci.size(); j++)
    cout << str(boost::format("State: %3d,  2S: %3d,  S^2: %8.2g,  dE: %10.2f\n")%j %Spin[j] %SpinSquare[j] %( (Energy[j]-Energy[0])*219470));

  MatrixXx Hsubspace = MatrixXx::Zero(hsubspaceSize, hsubspaceSize);

  for (int i=0; i<Spin.size(); i++)
    for (int row=rowIndex[i]; row < rowIndex[i]+Spin[i]+1; row++)
      Hsubspace(row, row) = (Energy[i]-Energy[0]);


  //update the connections
  //this will just update the connections and Helements with the SOC integrals
  std::vector<std::vector<int> > connections; connections.resize(Dets.size());
  std::vector<std::vector<CItype> > Helements;Helements.resize(Dets.size());
  std::vector<std::vector<size_t> > orbDifference; orbDifference.resize(Dets.size());
  SHCImakeHamiltonian::updateSOCconnections(Dets, 0, connections, orbDifference, Helements, norbs, I1, nelec);
  pout <<"Updated connections."<<endl;

  // calculate the matrix elements <S1|SOC|S2>
  for (int i=0; i<ci.size(); i++) {
    for (int j=i+1; j<ci.size(); j++) {
      int s1 = Spin[i], s2=Spin[j];
      SOChelper::calculateMatrixElements(s1, s2, Sz, rowIndex[i], rowIndex[j], ci[i], ci[j],
					 connections, Helements, Hsubspace, Dets, norbs,
					 beginS0, beginSp, beginSm);
    }
  }
  Hsubspace *= 219470.;

  //print out the hsoc matrix
  cout <<endl<< "SOC Matrix "<<endl;
  cout << Hsubspace<<endl<<endl<<endl;

  //print out the eigenvalues
  SelfAdjointEigenSolver<MatrixXx> eigensolver(Hsubspace);
  if (eigensolver.info() != Success) abort();
  for (int i=0; i<Hsubspace.rows(); i++)
    cout << "Root: "<<i<<" -> Energy: "<<eigensolver.eigenvalues()[i]<<endl;



  //NOW DO THE GTENSOR CALCULATION
  if (schd.doGtensor) {

    if (abs(eigensolver.eigenvalues()[0] - eigensolver.eigenvalues()[1]) > 10.0) {
      cout <<"Energy difference between kramer's pair greater than 10."<<endl;
      cout << "Quitting the g-tensor calcualtion"<<endl;
      exit(0);
    }
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

      S[1](2*(a-1), 2*(a-1)+1) += std::complex<double>(0,  -ge/2.);  //alpha beta
      S[1](2*(a-1)+1, 2*(a-1)) += std::complex<double>(0,   ge/2.);  //beta alpha

      S[2](2*(a-1), 2*(a-1)) +=  ge/2.;  //alpha alpha
      S[2](2*(a-1)+1, 2*(a-1)+1) += -ge/2.;  //beta beta
    }

    //The  La+ge Sa matrices where a is x,y,z
    vector<MatrixXx> LplusS(3, MatrixXx::Zero(hsubspaceSize, hsubspaceSize));

    //First calcualte S
    for (int a=0; a<3; a++) {
      std::vector<std::vector<int> > connections; connections.resize(Dets.size());
      std::vector<std::vector<CItype> > Helements;Helements.resize(Dets.size());
      std::vector<std::vector<size_t> > orbDifference; orbDifference.resize(Dets.size());

      oneInt LplusSInt;
      LplusSInt.store.resize(norbs*norbs, 0.0);
      LplusSInt.norbs = norbs;
      for (int i=0; i<L[a].store.size(); i++)
	LplusSInt.store[i] = 1.*L[a].store[i]+1.*S[a].store[i];

      //updateSOCconnections does not update the diagonal elements
      //these have to be done separately
      for (int i=0; i<Dets.size(); i++) {
	connections[i].push_back(i);
	CItype energy = 0.0;
	for (int j=0; j<norbs; j++)
	  if (Dets[i].getocc(j)) {
	    energy += LplusSInt(j,j);
	  }
	Helements[i].push_back(energy);
      }
      SHCImakeHamiltonian::updateSOCconnections(Dets, 0, connections, orbDifference, Helements, norbs, LplusSInt, nelec);
      Hmult2 H(connections, Helements);
      for (int j=0; j<ci.size(); j++) {//bra
	for (int i=j; i<ci.size(); i++) {//ket
	  for (int sz2=-Spin[j]; sz2<=Spin[j]; sz2+=2) {//bra Sz
	    for (int sz1=-Spin[i]; sz1<=Spin[i]; sz1+=2) {//ket	Sz
	      MatrixXx c2extended = MatrixXx::Zero(Dets.size(), 1);  //bra
	      MatrixXx c1extended = MatrixXx::Zero(Dets.size(), 1);  //ket
	      MatrixXx Hc1 = MatrixXx::Zero(Dets.size(), 1);

	      //bra
	      if (sz2==-Spin[j])
		SOChelper::getSminus(ci[j], c2extended, Dets, beginS0, beginSp, beginSm, norbs);
	      else
		c2extended.block(0,0,ci[j].rows(),1) = 1.*ci[j];

	      //ket
	      if (sz1==-Spin[i])
		SOChelper::getSminus(ci[i], c1extended, Dets, beginS0, beginSp, beginSm, norbs);
	      else
		c1extended.block(0,0,ci[i].rows(),1) = 1.*ci[i];

	      H(c1extended, Hc1);
	      LplusS[a](rowIndex[j]+ (-sz2+Spin[j])/2, rowIndex[i] + (-sz1+Spin[i])/2) = (c2extended.adjoint()*Hc1)(0,0);
	      LplusS[a](rowIndex[i]+ (-sz1+Spin[i])/2, rowIndex[j] + (-sz2+Spin[j])/2) = (Hc1.adjoint()*c2extended)(0,0);

	    }
	  }
	}
      }
    }
    //cout << LplusS[0]<<endl;
    vector<MatrixXx> Intermediate = vector<MatrixXx>(3, MatrixXx::Zero(2,2));
    for (int a=0; a<3; a++)
      Intermediate[a] = eigensolver.eigenvectors().block(0,0,hsubspaceSize, 2).adjoint() *(LplusS[a]*eigensolver.eigenvectors().block(0,0,hsubspaceSize, 2));

    MatrixXx Gtensor = MatrixXx::Zero(3,3);

    for (int i=0; i<3; i++)
      for (int j=0; j<3; j++)
	Gtensor(i,j) += 2.*(Intermediate[i].adjoint()*Intermediate[j]).trace();

    //cout << Gtensor<<endl;
    SelfAdjointEigenSolver<MatrixXx> eigensolver(Gtensor);
    if (eigensolver.info() != Success) abort();
    cout <<endl<< "Gtensor eigenvalues"<<endl;
    cout << str(boost::format("g1= %9.6f,  shift: %6.0f\n")%pow(eigensolver.eigenvalues()[0],0.5) % ((-ge+pow(eigensolver.eigenvalues()[0],0.5))*1.e6) );
    cout << str(boost::format("g2= %9.6f,  shift: %6.0f\n")%pow(eigensolver.eigenvalues()[1],0.5) % ((-ge+pow(eigensolver.eigenvalues()[1],0.5))*1.e6) );
    cout << str(boost::format("g3= %9.6f,  shift: %6.0f\n")%pow(eigensolver.eigenvalues()[2],0.5) % ((-ge+pow(eigensolver.eigenvalues()[2],0.5))*1.e6) );

  }

  return 0;
}
