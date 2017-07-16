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
#include "integral.h"
#include "Hmult.h"
#include "Davidson.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <set>
#include <list>
#include <tuple>
#include "boost/format.hpp"
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include <boost/serialization/vector.hpp>
#include "communicate.h"
#include <boost/interprocess/managed_shared_memory.hpp>
#include "SHCIshm.h"

using namespace Eigen;
using namespace boost;
int HalfDet::norbs = 1; //spin orbitals
int Determinant::norbs = 1; //spin orbitals
int Determinant::EffDetLen = 1;
char Determinant::Trev = 0; //Time reversal
Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> Determinant::LexicalOrder ;
//get the current time
double getTime() {
  struct timeval start;
  gettimeofday(&start, NULL);
  return start.tv_sec + 1.e-6*start.tv_usec;
}
double startofCalc = getTime();


void license() {
  pout << endl;
  pout << endl;
  pout << "**************************************************************"<<endl;
  pout << "Dice  Copyright (C) 2017  Sandeep Sharma"<<endl;
  pout <<"This program is distributed in the hope that it will be useful,"<<endl;
  pout <<"but WITHOUT ANY WARRANTY; without even the implied warranty of"<<endl;
  pout <<"MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE."<<endl;  
  pout <<"See the GNU General Public License for more details."<<endl;
  pout << endl<<endl;
  pout << "Author:       Sandeep Sharma"<<endl;
  pout << "Contributors: James E Smith, Adam A Holmes, Bastien Mussard"<<endl;
  pout << "For detailed documentation on Dice please visit"<<endl;
  pout << "https://sanshar.github.io/Dice/"<<endl;
  pout << "Please visit our group page for up to date information on other projects"<<endl;
  pout << "http://www.colorado.edu/lab/sharmagroup/"<<endl;
  pout << "**************************************************************"<<endl;
  pout << endl;
  pout << endl;
}

double calcGreensFunction(int i, int j, Determinant* Dets, CItype* ci, int DetsSize, 
        Determinant* DetsNm1, int DetsNm1Size, Hmult2& HNm1, double E0) ; 


int main(int argc, char* argv[]) {

#ifndef SERIAL
  //initialize mpi environment
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
#endif
  //initialize
  initSHM();

  license();

  std::cout.precision(15);

  //read the hamiltonian (integrals, orbital irreps, num-electron etc.)
  twoInt I2; oneInt I1; int nelec; int norbs; double coreE=0.0, eps;
  std::vector<int> irrep;
  readIntegrals("FCIDUMP", I2, I1, nelec, norbs, coreE, irrep);


  //setup the lexical table for the determinants
  norbs *=2;
  Determinant::norbs = norbs; //spin orbitals
  HalfDet::norbs = norbs; //spin orbitals
  Determinant::EffDetLen = norbs/64+1;
  Determinant::initLexicalOrder(nelec);
  if (Determinant::EffDetLen >DetLen) {
    pout << "change DetLen in global.h to "<<Determinant::EffDetLen<<" and recompile "<<endl;
    exit(0);
  }

  //read dets and ci coeffs of the variational result psi0
  char file [5000];
  sprintf (file, "%d-variational.bkp" , 0 );
  std::ifstream ifs(file, std::ios::binary);
  boost::archive::binary_iarchive load(ifs);
  //Dets has deteminants with nonzero psi0 overlap, ci has the corresponding coeffs
  int iter; std::vector<Determinant> Dets; std::vector<MatrixXx> ci; std::vector<double> E0;
  load >> iter >> Dets;
  load >> ci;
  load >> E0;
  ifs.close();


  //to store dets in psi0 less one electron, to construct H0 on N-1 electron space 
  //calc only in 0 proc
  std::vector<Determinant> DetsNm1;
  
  //needs to be parallelized
  if (commrank == 0) {
      //fill up DetsNm1
      for(int k=0; k<Dets.size(); k++){
          
          //get the occupied orbs in Dets[k]
          std::vector<int> closed(64*DetLen);
          int nclosed = Dets[k].getClosed(closed);
          
          //remove electrons from occupied orbs
          for(int l=0; l<nclosed; l++){
              Dets[k].setocc(closed[l], false);
              DetsNm1.push_back(Dets[k]);
              Dets[k].setocc(closed[l], true);
          }
      }

      //remove duplicates
      SHCISortMpiUtils::RemoveDuplicates(DetsNm1);
  }


  //put dets in shared memory: haven't figured out how this works yet
  Determinant* SHMDets, *SHMDetsNm1; CItype *SHMci;
  int DetsSize = Dets.size(), DetsNm1Size = DetsNm1.size();
  SHMVecFromVecs(Dets, SHMDets, shciDetsCI, DetsCISegment, regionDetsCI);
  Dets.clear();
  SHMVecFromVecs(DetsNm1, SHMDetsNm1, shciDetsNm1, DetsNm1Segment, regionDetsNm1);
  DetsNm1.clear();
  SHMVecFromMatrix(ci[0], SHMci, shcicMax, cMaxSegment, regioncMax);
  ci.clear();

#ifndef SERIAL
  mpi::broadcast(world, DetsNm1Size, 0);
#endif

  //make H0 in the DetsNm1 space
  SHCImakeHamiltonian::HamHelpers2 helper2;
  SHCImakeHamiltonian::SparseHam sparseHam;
  int Norbs = 2.*I2.Direct.rows();
  
  if (commrank == 0) {
    helper2.PopulateHelpers(SHMDetsNm1, DetsNm1Size, 0);
  }	
  helper2.MakeSHMHelpers();
  sparseHam.makeFromHelper(helper2, SHMDetsNm1, 0, DetsNm1Size, Norbs, I1, I2, coreE, false);
  Hmult2 H(sparseHam);

  double g_ij = 0;
  for(int i=0; i<norbs; i++){
      for(int j=0; j<=i; j++){
          g_ij = calcGreensFunction(i, j, SHMDets, SHMci, DetsSize, SHMDetsNm1, DetsNm1Size, H,  E0[0]);
          pout << endl << "G_" << i << "_" << j << "  " << g_ij << endl; 
      }
  }
   
  boost::interprocess::shared_memory_object::remove(shciDetsCI.c_str());
  boost::interprocess::shared_memory_object::remove(shciDetsNm1.c_str());
  boost::interprocess::shared_memory_object::remove(shcicMax.c_str());
  boost::interprocess::shared_memory_object::remove(shciHelper.c_str());
  
  return 0;

}

//to calculate G_ij(w=0) = <psi0| a_i^(dag) 1/H0-E0 a_j |psi0>
//psi0 is the variational ground state, a_i, a_j are annihilation operators 
//H0 is the Hamiltonian and E0 is the N electron variational ground state energy 
//Dets has dets in psi0, ci has the corresponding coeffs
//DetsNm1 dets in psi0 less one electron, HNm1 is H0 on the N-1 electron space
double calcGreensFunction(int i, int j, Determinant* Dets, CItype* ci, int DetsSize, Determinant* DetsNm1, int DetsNm1Size, Hmult2& HNm1, double E0) {

#ifndef SERIAL
    boost::mpi::communicator world;
#endif

    //to store ci coeffs in a_j|psi0> and a_i|psi0>, corresponding to dets in DetsNm1
    MatrixXx ciNm1 = MatrixXx::Zero(DetsNm1Size, 1);  
    
    //calc a_j|psi0>
    int n = 0;
    Determinant detTemp;
    for(int k=0; k<DetsSize; k++){
        if(Dets[k].getocc(j)) {
            detTemp = Dets[k];
            detTemp.setocc(j, false);
            n = std::distance(DetsNm1, std::find(DetsNm1, DetsNm1+DetsNm1Size, detTemp));
            ciNm1(n) = ci[k];
        }
    }
     
#ifndef SERIAL
    world.barrier();
#endif
    
    
    //solve for x0 = 1/H0-E0 a_j |psi0>
    MatrixXx x0 = MatrixXx::Zero(DetsNm1Size, 1);
    vector<CItype*> proj;
    LinearSolver(HNm1, E0, x0, ciNm1, proj, 1.e-5, false);
    
#ifndef SERIAL
    world.barrier();
#endif
    
    
    //calc a_i|psi0>
    ciNm1 = MatrixXx::Zero(DetsNm1Size, 1);
    for(int k=0; k<DetsSize; k++){
        if(Dets[k].getocc(i)) {
            detTemp = Dets[k];
            detTemp.setocc(i, false);
            n = std::distance(DetsNm1, std::find(DetsNm1, DetsNm1+DetsNm1Size, detTemp));
            if (n >= DetsNm1Size || n < 0) std::cout << commrank << " n = " << n << " k " << k << std::endl; 
            ciNm1(n) = ci[k];
        }
    }
    
#ifndef SERIAL
    world.barrier();
#endif
    

    double dotProduct = 0.0;
    for(int k=0; k<DetsNm1Size; k++){
        dotProduct += ciNm1(k)*x0(k);
    }
    //}
    
    return dotProduct;


}
