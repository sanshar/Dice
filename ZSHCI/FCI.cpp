/*
  Developed by Xubo Wang 
  2025 Copyright (c) 2025, Xubo Wang

  This program is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or (at your option) any later
  version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with
  this program. If not, see <http://www.gnu.org/licenses/>.
*/
#include <stdio.h>
#include <stdlib.h>
#define EIGEN_USE_LAPACKE
#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>
#include <list>
#include <set>
#include <tuple>
#include "Davidson.h"
#include "Determinants.h"
#include "Hmult.h"
#include "global.h"
#include "input.h"
#include "integral.h"
#ifndef SERIAL
#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#endif
#include <boost/format.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/serialization/vector.hpp>
#include <cstdlib>
#include <numeric>

#include "SHCIrdm.h"
#include "SHCIshm.h"
#include "SHCItime.h"
#include "communicate.h"
#include <unistd.h> 

#include <algorithm>
#include <boost/bind.hpp>

// Initialize
//using namespace Eigen;
//using namespace boost;

int HalfDet::norbs = 1;      // spin orbitals
int Determinant::norbs = 1;  // spin orbitals
int Determinant::EffDetLen = 1;
char Determinant::Trev = 0;  // Time reversal
Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> Determinant::LexicalOrder;

void gosper_combination(vector<Determinant>& dets, int n, int k) {
    size_t x = (1 << k) - 1;
    while (x < (1 << n)) {
        Determinant d;
        d.repr[0] = x;
        dets.push_back(d);

        // Gosper's Hack
        int lowest_bit = x & -x;
        int next_x = x + lowest_bit;  
        x = next_x | (((x ^ next_x) / lowest_bit) >> 2);  
    }
}

// Get the current time
double getTime() {
  struct timeval start;
  gettimeofday(&start, NULL);
  return start.tv_sec + 1.e-6 * start.tv_usec;
}

double startofCalc = getTime();

// License
void license(char* argv[]) {
  pout << endl;

  char *user;
  user=(char *)malloc(10*sizeof(char));
  user=getlogin();

  time_t t = time(NULL);
  struct tm *tm = localtime(&t);
  char date[64];
  strftime(date, sizeof(date), "%c", tm);

  printf("User:             %s\n",user);
  printf("Date:             %s\n",date);
  printf("PID:              %d\n",getpid());
  pout << endl;
  printf("Path:             %s\n",argv[0]);
  printf("Commit:           %s\n",GIT_HASH);
  printf("Branch:           %s\n",GIT_BRANCH);
  printf("Compilation Date: %s %s\n",__DATE__,__TIME__);
}

// Read Input
void readInput(string input, vector<std::vector<int> >& occupied,
               schedule& schd);

// Main
int main(int argc, char* argv[]) {
#ifndef SERIAL
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
#endif

  // #####################################################################
  // Misc Initialize
  // #####################################################################

  // Initialize
  initSHM();
  if (commrank == 0) license(argv);

  // Read the input file
  string inputFile = "input.dat";
  if (argc > 1) inputFile = string(argv[1]);
  schedule schd;
  vector<vector<int> > HFoccupied;
  if (commrank == 0) readInput(inputFile, HFoccupied, schd);
  if (schd.outputlevel > 0 && commrank == 0) Time::print_time("begin");
  if (DetLen % 2 == 1) {
    pout << "Change DetLen in global to an even number and recompile." << endl;
    exit(0);
  }

#ifndef SERIAL
  mpi::broadcast(world, schd, 0);
#endif

  std::cout.precision(15);

  // Read the Hamiltonian (integrals, orbital irreps, num-electron etc.)
  twoInt I2;
  oneInt I1;
  int nelec;
  int norbs;
  double coreE = 0.0, eps;
  std::vector<int> irrep;
  
  if (commrank == 0) Time::print_time("begin_integral");
  readIntegrals(schd.integralFile, I2, I1, nelec, norbs, coreE, irrep, schd.ReadTxt);
  if (commrank == 0) Time::print_time("end_integral");

  // Setup the lexical table for the determinants
  // For relativistic, norbs is already the spin orbitals
  Determinant::norbs = norbs;  // spin orbitals
  HalfDet::norbs = norbs;      // spin orbitals
  Determinant::EffDetLen = norbs / 64 + 1;
  cout << "lexical order" << endl;
  Determinant::initLexicalOrder(nelec);
  cout << "lexical order" << endl;

  if (Determinant::EffDetLen > DetLen) {
    pout << "change DetLen in global.h to " << Determinant::EffDetLen
         << " and recompile " << endl;
    exit(0);
  }

  // Initialize the Heat-Bath integrals
  std::vector<int> allorbs;
  for (int i = 0; i < norbs; i++)
    allorbs.push_back(i);
  twoIntHeatBath I2HB(1e-12);
  twoIntHeatBathSHM I2HBSHM(1e-12);

  cout << "I2HB" << endl;
  if (commrank == 0) I2HB.constructClass(allorbs, I2, I1, norbs);
  I2HBSHM.constructClass(norbs, I2HB);
  int num_thrds;

  // Unlink the integral shared memory
  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());

  vector<double> E0;
  vector<Determinant> Dets;
  
  gosper_combination(Dets, norbs, nelec);
  cout << "gosper" << endl;
  int dets_size = Dets.size();
  int nclosed = nelec;
  int nopen   = Dets[0].norbs-nclosed;
  int nsingles = nclosed * nopen;
  MatrixXx diag = MatrixXx::Zero(dets_size, 1);
  cout << dets_size << " " << nsingles << endl;

  vector<size_t> singles;
  
  if (commrank == 0) {
  singles.resize(dets_size * nsingles);
    for (auto i=0; i < dets_size; i++) {
      auto d = Dets[i];
      CItype hij =Dets[i].Energy(I1, I2, coreE);
      diag(i, 0) = hij;
      vector<int> closed(nclosed,0);
      vector<int> open(nopen,0);
      d.getOpenClosed(open, closed);
      
      for (auto j=0; j<nclosed; j++) {
        for (auto k=0; k<nopen; k++) {
          auto d_jk = d;
          d_jk.setocc(closed[j], false);
          d_jk.setocc(open[k], true);
          singles[i*nsingles+j*nopen+k] = d_jk.getHash();
        }
      }
    }
  }

  Determinant* SHMDets;
  size_t* SHMsingles;

  SHMVecFromVecs(Dets, SHMDets, shciDetsCI, DetsCISegment, regionDetsCI);
  SHMVecFromVecs(singles, SHMsingles, shciHelper, hHelpersSegment, regionHelpers);

  Time::print_time("single excitation list constructed");
  size_t orbDiff;
  /*for (int i = 0; i < dets_size; i++) {
    auto idet = i;
    auto det_i = SHMDets[i];
    hamiltonian(idet, idet) = det_i.Energy(I1, I2, coreE);
    for (int j = 0; j < nsingles; j++) {
      auto jdet = SHMsingles[i*nsingles+j];
      auto det_j = SHMDets[jdet];
      hamiltonian(idet, jdet) = Hij(det_i, det_j, I1, I2, coreE, orbDiff);
      for (int k = 0; k < nsingles; k++) {
        auto kdet = SHMsingles[jdet*nsingles+k];
        auto det_k = SHMDets[kdet];
        if (det_i.ExcitationDistance(det_k) <= 1) continue;
        hamiltonian(idet, kdet) = Hij(det_i, det_k, I1, I2, coreE, orbDiff);
      }
    }
  }*/
  
  vector<MatrixXx> X0;
  vector<MatrixXx> ci(schd.nroots, MatrixXx::Zero(dets_size, 1));

  if (commrank==0) {
    cout << "nsingles " << nsingles << " dets_size:" << dets_size << endl;
    MatrixXx pspace(nsingles+1, nsingles+1);
    vector<int> pspace_idx(nsingles+1);
    pspace_idx[0] = 0;
    for (int i = 1; i < nsingles+1; i++) {
      pspace_idx[i] = singles[i-1];
    }
    for (int i = 0; i < nsingles+1; i++) {
      for (int j = 0; j < nsingles+1; j++) {
        if (i == j) {
          pspace(i, j) = Dets[pspace_idx[i]].Energy(I1, I2, coreE);
        } else {
          auto distance = Dets[pspace_idx[i]].ExcitationDistance(Dets[pspace_idx[j]]);
          size_t orbDiff;
          if (distance < 3){
            pspace(i, j) = Hij(Dets[pspace_idx[i]], Dets[pspace_idx[j]], I1, I2, coreE, orbDiff);
          }
          else pspace(i, j) = 0.0;
        }
      }
    }

    SelfAdjointEigenSolver<MatrixXx> eigensolver(pspace);
    pout << eigensolver.eigenvalues() << endl;

    auto eigen_vectors = eigensolver.eigenvectors();
    for (int i = 0; i < schd.nroots; i++) {
      ci[i](i,0) = complex<double>{1.0,0.0};
      //cout << i << " " << dets_size << endl;
      //for (int i_idx = 0; i_idx < nsingles+1; i_idx++) {
      //  ci[i](i_idx,0) = eigen_vectors(i_idx, i);
      //}
    }
    Time::print_time("CIS intial guess constructed");

    if (dets_size < 10000) // full diag for 4f systems with 14 spinors 
    {
    MatrixXx hamiltonian(dets_size, dets_size);
    for (auto i=0; i<dets_size; i++) {
      for (auto j=0; j<dets_size; j++) {
        if (i==j) {
          hamiltonian(i, j) = Dets[i].Energy(I1, I2, coreE);
          cout << Dets[i] << " " << hamiltonian(i,j) << endl;
        } else {
          auto distance = Dets[i].ExcitationDistance(Dets[j]);
          size_t orbDiff;
          if (distance < 3){
            hamiltonian(i, j) = Hij(Dets[i], Dets[j], I1, I2, coreE, orbDiff);
            //cout << Dets[i] << " " << Dets[j] << " " << hamiltonian(i,j) << endl;
          }
          else hamiltonian(i, j) = 0.0;
        }
      }
    }

    Time::print_time("construct full Hamiltonian");
    SelfAdjointEigenSolver<MatrixXx> eigensolver(hamiltonian);
    Time::print_time("Eigen solve for all eigenvalues");
    pout << eigensolver.eigenvalues() << endl;

    auto eigen_vectors = eigensolver.eigenvectors();
    for (int i = 0; i < schd.nroots; i++) {
        auto vec_i = eigen_vectors.col(i);
      for (int j = 0; j < dets_size; j++) {
        ci[i](j, 0) = vec_i[j]; 
      }
    }
    }
    Time::print_time("Full diagonalization");
  }
  MPI_Barrier(MPI_COMM_WORLD);
  mpi::broadcast(world, ci, 0);

  X0 = ci;
  //HmultDirectFCI H(SHMDets, SHMsingles, dets_size, nelec, norbs, I1, I2, coreE, diag);
  SHCImake4cHamiltonian::SparseHam sparse_ham;
  sparse_ham.makeFCIHamitlonianFromSingles(SHMDets, SHMsingles, schd, dets_size, nelec, norbs, I1, I2, coreE, schd.DoRDM);
  Hmult2 H(sparse_ham);
  
  Time::print_time("Sparse Hamiltonian constructed.");

  auto numIter = schd.nroots*100;
  E0 = davidson(H, ci, diag, max(100, schd.nroots*4), schd.davidsonTol, numIter, schd.outputlevel>0, schd.nroots);
  for (auto &E : E0) {
    E += coreE;
  }
#ifndef SERIAL
  mpi::broadcast(world, ci, 0);
#endif

  SHMVecFromVecs(Dets, SHMDets, shciDetsCI, DetsCISegment, regionDetsCI);
  int DetsSize = Dets.size();
#ifndef SERIAL
  mpi::broadcast(world, DetsSize, 0);
#endif
  Dets.clear();

  if (commrank == 0) {
    std::string efile;
    efile = str(boost::format("%s%s") % schd.prefix[0].c_str() % "/shci.e");
    FILE* f = fopen(efile.c_str(), "wb");
    for (int j = 0; j < schd.nroots; ++j) {
      fwrite(&E0[j], 1, sizeof(double), f);
    }
    fclose(f);
  
  if (commrank == 0) {
    if (schd.writeBestDeterminants > 0) {
      int num = min(schd.writeBestDeterminants, static_cast<int>(DetsSize));
      int nspatorbs = Determinant::norbs/2;
      int nspinorbs = Determinant::norbs;
      for (int root = 0; root < schd.nroots; root++) {
        string fname;
        fname = "dets_";
        fname.append(to_string(root));
        fname.append(".bin");
        ofstream fout = ofstream(fname, ios::binary);
        fout.write((char*) &num, sizeof(int));
        fout.write((char*) &nspinorbs, sizeof(int));
        MatrixXx prevci = 1. * ci[root];
        std::vector<size_t> idx(static_cast<int>(DetsSize));
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&prevci](size_t i1, size_t i2){return abs(prevci(i1, 0)) > abs(prevci(i2, 0));});
        for (int i = 0; i < num; i++) {
          int m = idx[i];
          fout.write((char*) &prevci(m, 0), sizeof(complex<double>));
          Determinant wdet = SHMDets[m];
          char det[norbs];
          wdet.getRepArray(det);
          for (int i = 0; i < nspatorbs; i++) {
            char detocc;
            if (det[2 * i] == false && det[2 * i + 1] == false)
              detocc = '0';
            else if (det[2 * i] == true && det[2 * i + 1] == false)
              detocc = 'a';
            else if (det[2 * i] == false && det[2 * i + 1] == true)
              detocc = 'b';
            else if (det[2 * i] == true && det[2 * i + 1] == true)
              detocc = '2';
            fout.write((char*) &detocc, sizeof(char));
          }
        }
        fout.close();
      }
    }
  }
  
  for (int root = 0; root < schd.nroots; root++) {
    pout << format("State : %3i  Energy = %18.10f") % (root) % E0[root] << endl;
  }
  // #####################################################################
  // Print the 5 most important determinants and their weights
  // #####################################################################
  pout << "Printing most important determinants" << endl;
  pout << format("%4s %10s  Determinant string") % ("Det") % ("weight")
       << endl;
  for (int root = 0; root < schd.nroots; root++) {
    pout << format("State : %3i  Energy = %18.10f") % (root) % E0[root] << endl;
    MatrixXx prevci = 1. * ci[root];
    int num = max(6, schd.printBestDeterminants);
    for (int i = 0; i < min(num, static_cast<int>(DetsSize)); i++) {
      compAbs comp;
      int m = distance(&prevci(0, 0),
          max_element(&prevci(0, 0), &prevci(0, 0) + prevci.rows(), comp));
        pout << format("%4i %16.10f %16.10f ") % (i) % prevci(m, 0).real()
                % prevci(m, 0).imag();
        pout << SHMDets[m] << endl;
        prevci(m, 0) = 0.0;
      }
    }  // end root
    pout << std::flush;
    }    
#ifndef SERIAL
    world.barrier();
#endif
  if (commrank == 0) {
    std::string efile;
    efile = str(boost::format("%s%s") % schd.prefix[0].c_str() % "/shci.e");
    FILE* f = fopen(efile.c_str(), "wb");
    for (int j = 0; j < E0.size(); ++j) {
      fwrite(&E0[j], 1, sizeof(double), f);
    }
    fclose(f);
  }
 
  // #####################################################################
  // RDMs
  // #####################################################################
  if (schd.DoRDM) {
    pout << endl;
    pout << endl;
    pout << "**************************************************************"
         << endl;
    pout << "RDMs CALCULATIONS" << endl;
    pout << "**************************************************************"
         << endl;
    for (int i=0; i<schd.nroots; i++) {
      CItype * SHMci;
      SHMVecFromMatrix(ci[i], SHMci, shciDetsCI, DavidsonSegment, regionDavidson);
      MatrixXx twoRDM = MatrixXx::Zero(norbs*norbs, norbs*norbs);
      SHCIrdm::EvaluateRDM(sparse_ham.connections, SHMDets, DetsSize, SHMci, SHMci, sparse_ham.orbDifference, nelec, schd, i, twoRDM);
      pout << "compute energy from rdm" << endl;
      SHCIrdm::ComputeEnergyFromSpinRDM(norbs, nelec, I1, I2, coreE, twoRDM);
      if (commrank==0) SHCIrdm::saveRDM_hdf5(schd, twoRDM, i);

      MatrixXx oneRDM = MatrixXx::Zero(norbs, norbs);
      SHCIrdm::EvaluateOneRDM(sparse_ham.connections, SHMDets, DetsSize, SHMci, SHMci, sparse_ham.orbDifference, nelec, schd, i, oneRDM);
      SHCIrdm::save1RDM_hdf5(schd, oneRDM, i, i);
    }
  }
  if (schd.DoTDM) {
    pout << endl;
    pout << "**************************************************************"
         << endl;
    pout << "TDMs CALCULATIONS" << endl;
    pout << "**************************************************************"
         << endl;

    // Rebuild sparse_ham with thresh_hij overridden so that single-excitation
    // pairs with vanishing Hij (e.g. Kramers partners under symmetry) still
    // enter connections/orbDifference. Those pairs contribute to the TDM
    // but get filtered by the production thresh_hij used for Hmult.
    double saved_thresh = schd.thresh_hij;
    schd.thresh_hij = -1.0;
    sparse_ham.clear();
    sparse_ham.makeFCIHamitlonianFromSingles(
        SHMDets, SHMsingles, schd, DetsSize, nelec, norbs,
        I1, I2, coreE, true);
    schd.thresh_hij = saved_thresh;

    CItype * SHMci;
    CItype * ciroot;
    SHMVecFromMatrix(ci[0], SHMci, shciDetsCI, DavidsonSegment, regionDavidson);
    for (int i=1; i<schd.nroots; i++) {
      SHMVecFromMatrix(ci[i], ciroot, shcicMax, cMaxSegment, regioncMax);
      MatrixXx TRDM = MatrixXx::Zero(norbs, norbs);
      SHCIrdm::EvaluateOneRDM(sparse_ham.connections, SHMDets, DetsSize, SHMci, ciroot, sparse_ham.orbDifference, nelec, schd, i, TRDM);
      SHCIrdm::save1RDM_hdf5(schd, TRDM, 0, i);
    }
  }

#ifndef SERIAL
  world.barrier();
#endif

  pout << endl;
  pout << endl;
  pout << "**************************************************************"
       << endl;
  pout << "Returning without error" << endl;
  pout << "**************************************************************"
       << endl;
  pout << endl << endl;

  return 0;
}
