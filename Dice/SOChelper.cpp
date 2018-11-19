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
#include "communicate.h"
#include "SOChelper.h"
#include <boost/serialization/vector.hpp>


void SOChelper::calculateSpinRDM(vector<MatrixXx>& spinRDM, MatrixXx& ci1, MatrixXx& ci2,
				 Determinant* Dets1, int Detssize, int norbs, int nelec)
{
  size_t Norbs = norbs;

  map<Determinant, int> SortedDets;
  for (int i=0; i<Detssize; i++)
    SortedDets[Dets1[i]] = i;


  vector<int> closed(nelec,0);
  vector<int> open(norbs-nelec,0);

  for (int x=0; x<Detssize; x++) {
    if (x%commsize != commrank) continue;

    Determinant& d = Dets1[x];

    d.getOpenClosed(open, closed);
    int nclosed = nelec;
    int nopen = norbs-nclosed;

    for (int i=0; i<nclosed; i++){

      spinRDM[0](closed[i], closed[i]) += conj(ci1(x,0))*ci1(x,0);
      spinRDM[1](closed[i], closed[i]) += conj(ci2(x,0))*ci2(x,0);
      spinRDM[2](closed[i], closed[i]) += conj(ci1(x,0))*ci2(x,0);
    }

    //loop over all single excitation and find if they are present in the list
    //On or before the current determinant
    for (int ia=0; ia<nopen*nclosed; ia++){
      int i=ia/nopen, a=ia%nopen;

      Determinant di = d;
      di.setocc(open[a], true); di.setocc(closed[i],false);
      double sgn = 1.0;
      d.parity(min(open[a],closed[i]), max(open[a],closed[i]),sgn);


      map<Determinant, int>::iterator it = SortedDets.find(di);
      if (it != SortedDets.end() ) {
	int y = it->second;
	if (y != x) {
	  spinRDM[0](open[a], closed[i]) += conj(ci1(y,0))*ci1(x,0)*sgn;
	  spinRDM[1](open[a], closed[i]) += conj(ci2(y,0))*ci2(x,0)*sgn;
	  spinRDM[2](open[a], closed[i]) += conj(ci1(y,0))*ci2(x,0)*sgn;
	}
      }
    }
  }


#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &spinRDM[0](0, 0), 2*spinRDM[0].rows() * spinRDM[0].cols(),
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &spinRDM[1](0, 0), 2*spinRDM[1].rows() * spinRDM[1].cols(),
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &spinRDM[2](0, 0), 2*spinRDM[2].rows() * spinRDM[2].cols(),
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
}

void SOChelper::doGTensor(vector<MatrixXx>& ci, Determinant* Dets, vector<double>& E0, int Detssize, int norbs, int nelec, vector<MatrixXx>& spinRDM) {

  if (abs(E0[0] - E0[1])*219470. > 10.0) {
    cout <<"Energy difference between kramer's pair greater than 10."<<endl;
    cout << "Quitting the g-tensor calcualtion"<<endl;
    exit(0);
  }

  //initialize L and S integrals
  vector<oneInt> L(3), LplusS(3);
  for (int i=0; i<3; i++) {
    L[i].store.resize(norbs*norbs, 0.0);
    L[i].norbs = norbs;

    LplusS[i].store.resize(norbs*norbs, 0.0);
    LplusS[i].norbs = norbs;
  }

  //read L integrals
  readGTensorIntegrals(L, norbs, "GTensor");
#ifndef SERIAL
  boost::mpi::communicator world;
  boost::mpi::broadcast(world, L, 0);
#endif

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

  for (int a=0; a<3; a++)
    for (int i=0; i<norbs; i++)
      for (int j=0; j<norbs; j++)
	LplusS[a](i,j) += L[a](i,j);

  //The  La+ge Sa matrices where a is x,y,z
  vector<MatrixXx> Intermediate = vector<MatrixXx>(3, MatrixXx::Zero(2,2));

  //calcualte S+L
  for (int a=0; a<3; a++) {
    for (int i=0; i<norbs; i++)
      for (int j=0; j<norbs; j++) {
	Intermediate[a](0,0) += (LplusS[a](i,j))*spinRDM[0](i,j);
	Intermediate[a](1,1) += (LplusS[a](i,j))*spinRDM[1](i,j);
	Intermediate[a](0,1) += (LplusS[a](i,j))*spinRDM[2](i,j);
      }
    Intermediate[a](1,0) = conj(Intermediate[a](0,1));

    SelfAdjointEigenSolver<MatrixXx> eigensolver(Intermediate[a]);
    if (eigensolver.info() != Success) abort();
  }

  MatrixXx Gtensor = MatrixXx::Zero(3,3);

  for (int i=0; i<3; i++)
    for (int j=0; j<3; j++)
      Gtensor(i,j) += 2.*(Intermediate[i].adjoint()*Intermediate[j]).trace();


  SelfAdjointEigenSolver<MatrixXx> eigensolver(Gtensor);
  if (eigensolver.info() != Success) abort();
  pout <<endl<< "Gtensor eigenvalues"<<endl;
  pout << str(boost::format("g1= %9.6f,  shift: %6.0f\n")%pow(eigensolver.eigenvalues()[0],0.5) % ((-ge+pow(eigensolver.eigenvalues()[0],0.5))*1.e6) );
  pout << str(boost::format("g2= %9.6f,  shift: %6.0f\n")%pow(eigensolver.eigenvalues()[1],0.5) % ((-ge+pow(eigensolver.eigenvalues()[1],0.5))*1.e6) );
  pout << str(boost::format("g3= %9.6f,  shift: %6.0f\n")%pow(eigensolver.eigenvalues()[2],0.5) % ((-ge+pow(eigensolver.eigenvalues()[2],0.5))*1.e6) );

}

//GTENSOR IS NOT WORKING NOW
void SOChelper::doGTensor(vector<MatrixXx>& ci, vector<Determinant>& Dets, vector<double>& E0, int norbs, int nelec) {

  if (abs(E0[0] - E0[1])*219470. > 10.0) {
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

    S[1](2*(a-1), 2*(a-1)+1) += std::complex<double>(0, -ge/2.);  //alpha beta
    S[1](2*(a-1)+1, 2*(a-1)) += std::complex<double>(0,  ge/2.);  //beta alpha

    S[2](2*(a-1), 2*(a-1)) +=  ge/2.;  //alpha alpha
    S[2](2*(a-1)+1, 2*(a-1)+1) += -ge/2.;  //beta beta
  }

  //The  La+ge Sa matrices where a is x,y,z
  vector<MatrixXx> Intermediate = vector<MatrixXx>(3, MatrixXx::Zero(2,2));

  //First calcualte S
  for (int a=0; a<3; a++) {
    std::vector<std::vector<int> > connections; connections.resize(Dets.size());
    std::vector<std::vector<CItype> > Helements;Helements.resize(Dets.size());
    std::vector<std::vector<size_t> > orbDifference; orbDifference.resize(Dets.size());

    oneInt LplusS;
    LplusS.store.resize(norbs*norbs, 0.0);
    LplusS.norbs = norbs;
    for (int i=0; i<L[a].store.size(); i++)
      LplusS.store[i] = L[a].store[i]+S[a].store[i];
    //updateSOCconnections does not update the diagonal elements
    //these have to be done separately
    for (int i=0; i<Dets.size(); i++) {
      connections[i].push_back(i);
      CItype energy = 0.0;
      for (int j=0; j<norbs; j++)
	if (Dets[i].getocc(j)) {
	  energy += S[a](j,j)+L[a](j,j);
	}
      Helements[i].push_back(energy);
    }
    
    //****FIX THIS
    //SHCImakeHamiltonian::updateSOCconnections(Dets, 0, connections, orbDifference, Helements, norbs, LplusS, nelec);

    MatrixXx Hc = MatrixXx::Zero(Dets.size(), 1);
    //Hmult2 H(connections, Helements);
    //H(ci[0], Hc);

    Intermediate[a](0,0) = (ci[0].adjoint()*Hc)(0,0);
    Intermediate[a](1,0) = (ci[1].adjoint()*Hc)(0,0);
    Intermediate[a](0,1) = conj(Intermediate[a](1,0));

    Hc *= 0.0;
    //H(ci[1], Hc);
    Intermediate[a](1,1) = (ci[1].adjoint()*Hc)(0,0);
  }

  MatrixXx Gtensor = MatrixXx::Zero(3,3);


  for (int i=0; i<3; i++)
    for (int j=0; j<3; j++)
      Gtensor(i,j) += 2.*(Intermediate[i].adjoint()*Intermediate[j]).trace();

  SelfAdjointEigenSolver<MatrixXx> eigensolver(Gtensor);
  if (eigensolver.info() != Success) abort();
  cout <<endl<< "Gtensor eigenvalues"<<endl;
  cout << str(boost::format("g1= %9.6f,  shift: %6.0f\n")%pow(eigensolver.eigenvalues()[0],0.5) % ((-ge+pow(eigensolver.eigenvalues()[0],0.5))*1.e6) );
  cout << str(boost::format("g2= %9.6f,  shift: %6.0f\n")%pow(eigensolver.eigenvalues()[1],0.5) % ((-ge+pow(eigensolver.eigenvalues()[1],0.5))*1.e6) );
  cout << str(boost::format("g3= %9.6f,  shift: %6.0f\n")%pow(eigensolver.eigenvalues()[2],0.5) % ((-ge+pow(eigensolver.eigenvalues()[2],0.5))*1.e6) );

}


void SOChelper::getSplus(const MatrixXx& c2, MatrixXx& c2splus, vector<Determinant>& Dets, vector<Determinant>::iterator& beginS0, vector<Determinant>::iterator& beginSp, vector<Determinant>::iterator& beginSm, int norbs)
{
  for (int i=0; i<c2.rows(); i++) { //loop over all determinants
    for (int j=0; j<norbs/2 ; j++) {  //loop over spatial orbitals
      if (Dets[i].getocc(2*j+1) &&
	  !Dets[i].getocc(2*j  )   ) {//if beta-filled and alpha-empty

	Determinant d = Dets[i];
	d.setocc(2*j+1, false); //empty the beta
	d.setocc(2*j  , true);  //fill up the alpha

	auto location = lower_bound(beginSp, beginSm, d);
	c2splus(location-Dets.begin(), 0) += c2(i,0);
      }//if
    }// j (orbitals)
  }//i dets
}

void SOChelper::getSminus(const MatrixXx& c2, MatrixXx& c2sminus, vector<Determinant>& Dets, vector<Determinant>::iterator& beginS0, vector<Determinant>::iterator& beginSp, vector<Determinant>::iterator& beginSm, int norbs)
{
  for (int i=0; i<c2.rows(); i++) { //loop over all determinants
    for (int j=0; j<norbs/2 ; j++) {  //loop over spatial orbitals
      if (Dets[i].getocc(2*j) &&
	  !Dets[i].getocc(2*j +1 )   ) {//if beta-filled and alpha-empty

	Determinant d = Dets[i];
	d.setocc(2*j+1, true); //fill up the beta
	d.setocc(2*j  , false);  //empty the alpha

	auto location = lower_bound(beginSm, Dets.end(), d);
	c2sminus(location-Dets.begin(), 0) += c2(i,0);
      }//if
    }// j (orbitals)
  }//i dets
}


void SOChelper::calculateMatrixElements(int spin1, int spin2, int Sz, int rowIndex1, int rowIndex2,
			     const MatrixXx& c1, const MatrixXx& c2, vector<vector<int> >& connections,
			     vector<vector<CItype> >& Helements, MatrixXx& Hsubspace,
			     vector<Determinant>& Dets, int norbs,
			     vector<Determinant>::iterator& beginS0,
			     vector<Determinant>::iterator& beginSp,
			     vector<Determinant>::iterator& beginSm) {

  //Hmult2 H(connections, Helements);

  int orbM = 0;
  vector<CItype> RME (3,0.0);
  int Sz2list [3] = {Sz, Sz+2, Sz-2};
  int Sz1list [3] = {Sz, Sz+2, Sz-2};
  int Jop = 2;
  for (int Mop = -2; Mop<=2; Mop+=2) {

    for (int ij=0; ij<9; ij++) { //Sz2 Sz1
      int i=ij/3, j = ij%3;

      int Sz2 = Sz2list[i];
      int Sz1 = Sz1list[j];
      if (Sz2 > spin2 || Sz2 <-spin2) continue;
      if (Sz1 > spin1 || Sz1 <-spin1) continue;
      double cg = clebsch(spin2, Sz2, Jop, Mop, spin1, Sz1) ;

      if (abs(cg) < 1.e-8) continue;

      MatrixXx c2extended = MatrixXx::Zero(Dets.size(), 1);
      MatrixXx c1extended = MatrixXx::Zero(Dets.size(), 1);
      MatrixXx Hc2 = MatrixXx::Zero(Dets.size(), 1);
      if (i == 0) {
	c2extended.block(0,0,c2.rows(),1) = 1.*c2;
      }
      else if (i==1) {
	getSplus(c2, c2extended, Dets, beginS0, beginSp, beginSm, norbs);
      }
      else if (i == 2) {
	getSminus(c2, c2extended, Dets, beginS0, beginSp, beginSm, norbs);
      }

      if (j == 0) {
	c1extended.block(0,0,c1.rows(),1) = 1.*c1;
      }
      else if (j==1) {
	getSplus(c1, c1extended, Dets, beginS0, beginSp, beginSm, norbs);
      }
      else if (j == 2) {
	getSminus(c1, c1extended, Dets, beginS0, beginSp, beginSm, norbs);
      }

      //H(c2extended, Hc2);
      CItype element = (c1extended.adjoint()*Hc2)(0,0)/c2extended.norm()/c1extended.norm();
      RME[ (Mop+2)/2] = element/cg;
      break;
    }//for ij Sz1 Sz2
  }//Mop

  //now populate all elements of hsubspace
  for (int Sz1 = -spin1; Sz1 <=spin1; Sz1+=2)
    for (int Sz2 = -spin2; Sz2 <=spin2; Sz2+=2) {
      for (int Mop = -2; Mop<=2; Mop+=2) {
	double cg = clebsch(spin2, Sz2, Jop, Mop, spin1, Sz1);
	Hsubspace(rowIndex1 + (-Sz1+spin1)/2, rowIndex2 + (-Sz2+spin2)/2) += RME[ (Mop+2)/2]*cg;
	if (rowIndex1 != rowIndex2 )
	  Hsubspace(rowIndex2 + (-Sz2+spin2)/2, rowIndex1 + (-Sz1+spin1)/2) += conj(RME[(Mop+2)/2]*cg);
      }
    }
  return;
}


void SOChelper::calculateMatrixElementsForgTensor(int spin1, int spin2, int Sz, int rowIndex1, int rowIndex2,
						  const MatrixXx& c1, const MatrixXx& c2, vector<vector<int> >& connections,
						  vector<vector<CItype> >& Helements, vector<MatrixXx>& Hsubspace,
						  vector<Determinant>& Dets, int norbs,
						  vector<Determinant>::iterator& beginS0,
						  vector<Determinant>::iterator& beginSp,
						  vector<Determinant>::iterator& beginSm) {

  //  Hmult2 H(connections, Helements);

  int orbM = 0;
  vector<CItype> RME (3,0.0);
  int Sz2list [3] = {Sz, Sz+2, Sz-2};
  int Sz1list [3] = {Sz, Sz+2, Sz-2};
  int Jop = 2;
  for (int Mop = -2; Mop<=2; Mop+=2) {

    for (int ij=0; ij<9; ij++) { //Sz2 Sz1
      int i=ij/3, j = ij%3;

      int Sz2 = Sz2list[i];
      int Sz1 = Sz1list[j];
      if (Sz2 > spin2 || Sz2 <-spin2) continue;
      if (Sz1 > spin1 || Sz1 <-spin1) continue;
      double cg = clebsch(spin2, Sz2, Jop, Mop, spin1, Sz1) ;

      if (abs(cg) < 1.e-8) continue;

      MatrixXx c2extended = MatrixXx::Zero(Dets.size(), 1);
      MatrixXx c1extended = MatrixXx::Zero(Dets.size(), 1);
      MatrixXx Hc2 = MatrixXx::Zero(Dets.size(), 1);
      if (i == 0) {
	c2extended.block(0,0,c2.rows(),1) = 1.*c2;
      }
      else if (i==1) {
	getSplus(c2, c2extended, Dets, beginS0, beginSp, beginSm, norbs);
      }
      else if (i == 2) {
	getSminus(c2, c2extended, Dets, beginS0, beginSp, beginSm, norbs);
      }

      if (j == 0) {
	c1extended.block(0,0,c1.rows(),1) = 1.*c1;
      }
      else if (j==1) {
	getSplus(c1, c1extended, Dets, beginS0, beginSp, beginSm, norbs);
      }
      else if (j == 2) {
	getSminus(c1, c1extended, Dets, beginS0, beginSp, beginSm, norbs);
      }

      //H(c2extended, Hc2);
      CItype element = (c1extended.adjoint()*Hc2)(0,0)/c2extended.norm()/c1extended.norm();
      RME[ (Mop+2)/2] = element/cg;
      break;
    }//for ij Sz1 Sz2
  }//Mop

  //now populate all elements of hsubspace
  for (int Sz1 = -spin1; Sz1 <=spin1; Sz1+=2)
    for (int Sz2 = -spin2; Sz2 <=spin2; Sz2+=2) {
      for (int Mop = -2; Mop<=2; Mop+=2) {
	double cg = clebsch(spin2, Sz2, Jop, Mop, spin1, Sz1);
	Hsubspace[(Mop+2)/2](rowIndex1 + (-Sz1+spin1)/2,
			     rowIndex2 + (-Sz2+spin2)/2) += RME[ (Mop+2)/2]*cg;
	if (rowIndex1 != rowIndex2 )
	  Hsubspace[(Mop+2)/2](rowIndex2 + (-Sz2+spin2)/2,
			       rowIndex1 + (-Sz1+spin1)/2) += conj(RME[(Mop+2)/2]*cg);
      }
    }
  return;
}
