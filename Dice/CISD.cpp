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
#include <iostream>
#include "global.h"
#include "input.h"
#include "Determinants.h"
#include "integral.h"
#include "Hmult.h"
#include "CIPSIbasics.h"
#include "Davidson.h"
#include <Eigen/Dense>
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif

using namespace Eigen;
int HalfDet::norbs = 1; //spin orbitals
int Determinant::norbs = 1; //spin orbitals
int Determinant::EffDetLen = 1;


void readInput(string input, std::vector<int>& occupied, schedule& schd);

int main(int argc, char* argv[]) {
#ifndef SERIAL
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
#endif
  std::cout.precision(15);
  twoInt I2; oneInt I1; int nelec; int norbs; double coreE;
  std::vector<int> irrep;
  readIntegrals("FCIDUMP", I2, I1, nelec, norbs, coreE, irrep);
  norbs *=2;
  Determinant::norbs = norbs; //spin orbitals

  //make HF determinant
  schedule schd;
  std::vector<int> HFoccupied; //double epsilon1, epsilon2, tol, dE;
  if (mpigetrank() == 0) readInput("input.dat", HFoccupied, schd); //epsilon1, epsilon2, tol, num_thrds, eps, dE);

#ifndef SERIAL
  boost::mpi::broadcast(world, HFoccupied, 0);
  boost::mpi::broadcast(world, schd, 0);
#endif

  //make HF determinant
  Determinant d;
  for (int i=0; i<HFoccupied.size(); i++) {
    d.setocc(HFoccupied[i], true);
  }


  char detchar[norbs]; d.getRepArray(detchar);
  std::cout << Energy(detchar,norbs,I1,I2,coreE)<<" "<<coreE<<std::endl;

  vector<int> closed(nelec,0), open(norbs-nelec,0);
  int o = nelec;
  d.getOpenClosed(open, closed); int v=norbs-o;
  std::vector<Determinant> dets(o*(o-1)*v*(v-1)/4+1+o*v);
  dets[0] = d;

  cout << o<<"  "<<v<<"  "<<dets.size()<<endl;
  int index = 1;

  for (int a=0; a<v; a++){
    for (int i=0; i<o; i++) {
      Determinant di = d;
      di.setocc(open[a], true); di.setocc(closed[i],false);
      dets[index]=di;
      index++;
    }
  }
  for (int a=0; a<v; a++){
    for (int b=0; b<v; b++){
      if (b >= a) continue;
      for (int i=0; i<o; i++) {
	for (int j=0; j<o; j++) {
	  if (j >= i) continue;
	  Determinant di = d;
	  di.setocc(open[a], true), di.setocc(open[b], true), di.setocc(closed[i],false), di.setocc(closed[j], false);
	  dets[index] = di;
	  index++;
	}
      }
    }
    cout << index<<"  "<<dets[index-1]<<"  "<<dets.size()<<endl;
  }


  MatrixXd X0(dets.size(), 1); X0 *= 0.0; X0(0,0) = 1.0;
  MatrixXd diag(dets.size(), 1); diag *= 0.0;
  char detChar[norbs*dets.size()];
  cout << dets.size()<<endl;
  std::vector<std::vector<int> > connections(dets.size(), std::vector<int>(1,0));
  std::vector<std::vector<double> > Helements(dets.size(), std::vector<double>(1,0.0));

#pragma omp parallel for schedule(dynamic)
  for (int k=0; k<dets.size(); k++) {
    dets[k].getRepArray(detChar+norbs*k);
    diag(k,0) = Energy(detChar+norbs*k, norbs, I1, I2, coreE);
  }

  MatrixXd diagcopy = 1.*diag;
  for (int i=0; i<5; i++) {
    compAbs comp;
    int m = distance(&diagcopy(0,0), max_element(&diagcopy(0,0), &diagcopy(0,0)+diagcopy.rows(), comp));
    pout <<"#"<< i<<"  "<<diagcopy(m,0)<<"  "<<dets[m]<<endl;
    diagcopy(m,0) = 0.0;
  }


  int Norbs = norbs;
#pragma omp parallel for schedule(dynamic)
  for (size_t i=0; i<dets.size() ; i++) {
    if (i%world.size() != world.rank()) continue;

    for (size_t j=i; j<dets.size(); j++) {
      if (dets[i].connected(dets[j])) {
	double hij = Hij(&detChar[norbs*i], &detChar[norbs*j], Norbs, I1, I2, coreE);

	if (abs(hij) > 1.e-10) {
	  connections[i].push_back(j);
	  Helements[i].push_back(hij);
	}
      }
    }
  }
  Hmult2 H(connections, Helements);
  double E0 = davidson(H, X0, diag, 10, schd.davidsonTol, false);
  pout << "energy " << E0<<endl;

  for (int i=0; i<5; i++) {
    compAbs comp;
    int m = distance(&X0(0,0), max_element(&X0(0,0), &X0(0,0)+X0.rows(), comp));
    pout <<"#"<< i<<"  "<<X0(m,0)<<"  "<<dets[m]<<endl;
    X0(m,0) = 0.0;
  }

  return 0;
}
