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
#include "SHCIgetdeterminants.h"
#include "SHCISortMpiUtils.h"
#include "SHCImakeHamiltonian.h"
#include "integral.h"
#include "Hmult.h"
#include "Davidson.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <set>
#include <list>
#include <tuple>
#include <numeric>
#include "boost/format.hpp"
#include <boost/serialization/shared_ptr.hpp>
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include <boost/serialization/vector.hpp>
#include "communicate.h"
#include <boost/interprocess/managed_shared_memory.hpp>
#include "SHCIshm.h"
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>

using namespace Eigen;
using namespace boost;
using namespace SHCISortMpiUtils;
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


class schedule {
public:
  bool exact;
  std::vector<int> ij; //i and j in G(i,j,w)
  std::vector<double> w; // w1Real, w2Real, wImag, dw (same for w1 and w2) -> (w1, w2) with dw spacing
  //default constructor
  schedule() {
      exact=false;
      ij.push_back(0);
      ij.push_back(0);
      w.push_back(-2.00);//w1R
      w.push_back(2.00);//w2R
      w.push_back(0.01);//wI
      w.push_back(0.02);//dw
  }
  //read input
  void readInput(std::string input) {
      std::ifstream dump(input.c_str());
      while (dump.good()){
          std::string Line;
          std::getline(dump, Line);
          trim(Line);
          std::vector<string> tok;
          boost::split(tok, Line, is_any_of(", \t\n"), token_compress_on);//break up an input line into pieces
          std::string ArgName = *tok.begin();
          if (!ArgName.empty() && (boost::iequals(tok[0].substr(0,1), "#"))) continue;
          else if (ArgName.empty()) continue;
          else if (boost::iequals(ArgName, "exact")) exact=true;
          else if (boost::iequals(ArgName, "ij")) {
              ij[0]=std::atoi(tok[1].c_str());
              ij[1]=std::atoi(tok[2].c_str());
          }
          else if (boost::iequals(ArgName, "w")) {
              w[0]=std::atof(tok[1].c_str());
              w[1]=std::atof(tok[2].c_str());
              w[2]=std::atof(tok[3].c_str());
              w[3]=std::atof(tok[4].c_str());
          }
      
      }
  }
} schd;

class stitched {
    public:
        Determinant det;
        CItype matElement;
        double energy;
        stitched(Determinant& d, CItype& me, double& ene) {
            det = d;
            matElement = me;
            energy = ene;
        }
        
        bool operator<(const stitched& s) const {
            if (det < s.det) return true;
            else return false;
        }
        
        bool operator==(const stitched& s) const {
            if (det == s.det) return true;
            else return false;
        }
};

class stitchedList {
    public:
        std::vector<stitched> stitchedVec;
        stitchedList();
        stitchedList(std::vector<Determinant>& dets, Determinant* varSortedDets, int DetsSize, std::vector<CItype>& matElements, std::vector<double>& ene) {
            for(int i=0; i<dets.size(); i++) {
                if (!std::binary_search(varSortedDets, varSortedDets+DetsSize, dets[i]))
                    stitchedVec.push_back(stitched(dets[i], matElements[i], ene[i]));
            }
        }
        void sort() {
            std::sort(stitchedVec.begin(), stitchedVec.end());
        }
        stitched operator()(int k) const {
            return stitchedVec[k];
        }
};



int main(int argc, char* argv[]) {

#ifndef SERIAL
  //initialize mpi environment
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
#endif
  //initialize
  initSHM();

  //license();

  std::cout.precision(15);
  //read input
  std::string inputFile = "ginput.dat";
  if (argc > 1) inputFile = std::string(argv[1]);
  if (commrank == 0) schd.readInput(inputFile);

  //read the hamiltonian (integrals, orbital irreps, num-electron etc.)
  twoInt I2; oneInt I1; int nelec; int norbs; double coreE=0.0, eps;
  std::vector<int> irrep;
  readIntegrals("FCIDUMP", I2, I1, nelec, norbs, coreE, irrep);

  //setup the lexical table for the determinants
  norbs *=2;
  Determinant::norbs = norbs; //spin orbitals
  HalfDet::norbs = norbs; //spin orbitals
  Determinant::EffDetLen = norbs/64+1;
  Determinant::initLexicalOrder(nelec); //or nelec-1?
  if (Determinant::EffDetLen >DetLen) {
    pout << "change DetLen in global.h to "<<Determinant::EffDetLen<<" and recompile "<<endl;
    exit(0);
  }

  //read dets and ci coeffs of the variational result psi0
  char file [5000];
  sprintf (file, "%d-variational.bkp" , 0 );
  std::ifstream ifs(file, std::ios::binary);
  boost::archive::binary_iarchive load(ifs);
  //Dets has deteminants in psi0, ci has the corresponding coeffs
  int iter; std::vector<Determinant> Dets; std::vector<MatrixXd> ciReal; std::vector<double> E0;
  load >> iter >> Dets;
  load >> ciReal;
  load >> E0;
  ifs.close();
  MatrixXx ci = ciReal[0];
  ciReal.clear(); 
  
  //sort variational dets by ci coeffs
  std::vector<Determinant> SortedCIDetsVec;
  std::vector<size_t> idx(Dets.size());
  if(commrank == 0){
      std::iota(idx.begin(), idx.end(), 0);//idx=0,1,2,...,DetsSize
      std::sort(idx.begin(), idx.end(), [&ci](size_t i1, size_t i2){return ci(i1,0).real()*ci(i1,0).real()>ci(i2,0).real()*ci(i2).real();});//defined comparison function inline (lambda)
      //now 'sorting' Dets
      for(int i=0; i<Dets.size(); i++) SortedCIDetsVec.push_back(Dets[idx[i]]); 
  }

  int DetsSize = Dets.size();
  //SortedCIDetsVec.resize(DetsSize);

  //sort variational dets
  std::vector<Determinant> SortedDetsVec = SortedCIDetsVec; //dets in psi0 sorted 
  std::sort(SortedDetsVec.begin(), SortedDetsVec.end());
  
  
#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  
  //put dets in shared memory
  Determinant* SHMSortedDets, *SHMDets;//don't need non-sorted shared dets, delete later
  SHMVecFromVecs(SortedDetsVec, SHMSortedDets, shciSortedDets, SortedDetsSegment, regionSortedDets);
  SHMVecFromVecs(SortedCIDetsVec, SHMDets, shciDetsCI, DetsCISegment, regionDetsCI);
  SortedDetsVec.clear();
  SortedCIDetsVec.clear();
  Dets.clear();

  //preparing first order perturbation calculation
  //initialize the heatbath integral
  std::vector<int> allorbs;
  for (int i=0; i<norbs/2; i++) allorbs.push_back(i);
  twoIntHeatBath I2HB(1e-10);
  twoIntHeatBathSHM I2HBSHM(1e-10);
  if (commrank == 0) I2HB.constructClass(allorbs, I2, I1, norbs/2);
  I2HBSHM.constructClass(norbs/2, I2HB);

  //build varHam
  MatrixXx varHam = MatrixXx::Zero(DetsSize, DetsSize);
  if (commrank==0) {
      for (int i=0; i<DetsSize; i++) {
          varHam(i,i) = SHMDets[i].Energy(I1, I2, coreE);
          for (int j=i+1; j<DetsSize; j++) {
              if (SHMDets[i].connected(SHMDets[j])) {
                  size_t orbDiff = 0;
                  varHam(i,j) = Hij(SHMDets[i], SHMDets[j], I1, I2, coreE, orbDiff).real();
                  varHam(j,i) = varHam(i,j);
              }
          }
      }
  }

  int size = commsize, rank = commrank;
  int prevSlope = 0;
  std::vector<double> specPeaks;
  double prevSpecVal = 0.0;
  double dw = 0.001, eneRef = E0[0];
  int kLeft = -1000, kRight=3000;
  for (int k=kLeft; k<kRight; k++) {
      std::complex<double> w (k*dw+eneRef, 0.01);
      //MatrixXx selfEnergy = MatrixXx::Zero(DetsSize, DetsSize);
      std::complex<double> selfEnergy[DetsSize*(DetsSize+1)/2];
      std::complex<double> selfEnergyGlob[DetsSize*(DetsSize+1)/2];
      for (int i=0; i<DetsSize*(DetsSize+1)/2; i++) {
          selfEnergy[i]=0.; selfEnergyGlob[i]=0.;
      }
      //std::complex<double> w (1., 0.01);
      for (int i=0; i<DetsSize; i++) {
          for (int j=0; j<=i; j++) {
              std::vector<Determinant> iDets, jDets;
              std::vector<CItype> hik, hjk;
              std::vector<double> iEnergy, jEnergy;
              if ((i*(i+1)/2+j)%size != rank) continue;
              SHCIgetdeterminants::getDeterminantsDeterministicPT(SHMDets[i], 0.0, 1.0, 0.0,
      						  I1, I2, I2HBSHM, irrep, coreE, E0[0],
      						  iDets,
      						  hik,
      						  iEnergy,
      						  schd,0, nelec);
              SHCIgetdeterminants::getDeterminantsDeterministicPT(SHMDets[j], 0.0, 1.0, 0.0,
      						  I1, I2, I2HBSHM, irrep, coreE, E0[0],
      						  jDets,
      						  hjk,
      						  jEnergy,
      						  schd,0, nelec);
              stitchedList iStitched(iDets, SHMSortedDets, DetsSize, hik, iEnergy);
              stitchedList jStitched(jDets, SHMSortedDets, DetsSize, hjk, jEnergy);
              iDets.clear(); jDets.clear(); 
              hik.clear(); hjk.clear(); 
              iEnergy.clear(); jEnergy.clear();
              iStitched.sort(); jStitched.sort();
              int k1=0, k2=0;
              while (k1<iStitched.stitchedVec.size() && k2<jStitched.stitchedVec.size()) {
                  if (iStitched(k1).det == jStitched(k2).det) {
                      //selfEnergy(i,j)+=iStitched(k1).matElement*jStitched(k2).matElement/(w-iStitched(k1).energy);
                      selfEnergy[i*(i+1)/2+j] += iStitched(k1).matElement*jStitched(k2).matElement/(w-iStitched(k1).energy);
                      k1++; k2++;
                  }
                  else if (iStitched(k1).det < jStitched(k2).det)
                      k1++;
                  else 
                      k2++;
              }
              iStitched.stitchedVec.clear();
              jStitched.stitchedVec.clear();
              //selfEnergy(j,i)=selfEnergy(i,j);
          }
      }
      
#ifndef SERIAL
      MPI_Reduce(selfEnergy, selfEnergyGlob, DetsSize*(DetsSize+1)/2, MPI_C_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
      if (commrank==0) {
          MatrixXx selfEnergyMat = MatrixXx::Zero(DetsSize, DetsSize);
          for (int i=0; i<DetsSize; i++) {
              for (int j=0; j<=i; j++) {
                  selfEnergyMat(i,j) = selfEnergyGlob[i*(i+1)/2+j];
                  selfEnergyMat(j,i) = selfEnergyMat(i,j);
              }
          }
          MatrixXx w_varHam = w*MatrixXx::Identity(DetsSize, DetsSize)-varHam; //MatrixXx::Zero(DetsSize, DetsSize);
          //    SelfAdjointEigenSolver<MatrixXx> eigensolver(varHam);
          //    pout << eigensolver.eigenvalues() << endl;
          MatrixXx resolvent = MatrixXx::Zero(DetsSize, DetsSize);
          resolvent = (w_varHam-selfEnergyMat).inverse();
          if (k!=kLeft && k!=kRight-1) {
              int newSlope = -std::imag(resolvent(0,0)) > prevSpecVal ? 1 : -1;
              if (newSlope==-1 && prevSlope==1) specPeaks.push_back(std::real(w)-dw);
              prevSlope = newSlope;
          }
          pout << w << "   " << -std::imag(resolvent(0,0)) << endl;
          prevSpecVal = -std::imag(resolvent(0,0)); 
      }
  }
  pout << "# of peaks: " << specPeaks.size() <<  endl;
  //pout << "eneRef " << eneRef <<  endl;
  for (int i=0; i<specPeaks.size(); i++) printf("%0.3f\n", specPeaks[i]);
  return 0;
}
