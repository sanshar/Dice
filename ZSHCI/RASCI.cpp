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
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/IterativeSolvers>
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
#include <highfive/H5File.hpp>

// Initialize
//using namespace Eigen;
//using namespace boost;
namespace h5 = HighFive;
typedef complex<double> dcomplex;

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
  Determinant::initLexicalOrder(nelec);

  if (Determinant::EffDetLen > DetLen) {
    pout << "change DetLen in global.h to " << Determinant::EffDetLen
         << " and recompile " << endl;
    exit(0);
  }

  // Initialize the Heat-Bath integrals
  std::vector<int> allorbs;
  for (int i = 0; i < norbs; i++)
    allorbs.push_back(i);

  int num_thrds;

  // Unlink the integral shared memory
  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());

  vector<double> E0(schd.nroots);
  vector<Determinant> Dets;
  
  // rasci 
  // generate det list for each subspace and concat them together.
  // then use the sparseHam scheme in HCI to construct Hamiltonian 
  // 
  gosper_combination(Dets, norbs, nelec);
  int dets_size = Dets.size();
  int nclosed = nelec;
  int nopen   = Dets[0].norbs-nclosed;
  int nsingles = nclosed * nopen;
  MatrixXx diag = MatrixXx::Zero(dets_size, 1);
  cout << dets_size << " " << nsingles << endl;
  MatrixXx hamiltonian(dets_size, dets_size);

  /*
    Single excitation list is used to construct sparse Hamiltonian.
    Sparse Hamiltonian with "connections" is used to evaluate RDMs.
  */ 
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
  for (int i = 0; i < dets_size; i++) {
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
  }
  Time::print_time("Full Hamiltonian constructed.");

  vector<MatrixXx> X0;
  vector<MatrixXx> ci(schd.nroots, MatrixXx::Zero(dets_size, 1));

  SHCImake4cHamiltonian::SparseHam sparse_ham;
  sparse_ham.makeFCIHamitlonianFromSingles(SHMDets, SHMsingles, schd, dets_size, nelec, norbs, I1, I2, coreE, schd.DoRDM);
  Time::print_time("Sparse Hamiltonian constructed.");

  SelfAdjointEigenSolver<MatrixXx> eigensolver(hamiltonian);
  Time::print_time("FCI full matrix solved.");

  auto eigen_vectors = eigensolver.eigenvectors();
  auto eigen_values = eigensolver.eigenvalues();
  cout << eigen_vectors.rows() << " " << eigen_vectors.cols() << endl;
  cout << ci.size() << " " << schd.nroots << endl;
  cout << ci[0].rows() << " " << ci[0].cols() << endl;
  cout << eigensolver.eigenvalues().cols() << eigensolver.eigenvalues().rows() << endl;
  for (int i = 0; i < schd.nroots; i++) {
    ci[i](i,0) = complex<double>{1.0,0.0};
    E0[i] = eigen_values(i);
    for (int i_idx = 0; i_idx < dets_size; i_idx++) {
      ci[i](i_idx,0) = eigen_vectors(i_idx, i);
    }
  }

  world.barrier();
  mpi::broadcast(world, ci, 0);

  if (commrank == 0) {
    std::string efile;
    efile = str(boost::format("%s%s") % schd.prefix[0].c_str() % "/shci.e");
    FILE* f = fopen(efile.c_str(), "wb");
    for (int j = 0; j < schd.nroots; ++j) {
      fwrite(&E0[j], 1, sizeof(double), f);
      cout << boost::format("Energy of state %d: %16.10f") % j % E0[j] << endl;
    }
    fclose(f);
  
  // #####################################################################
  // Print the 5 most important determinants and their weights
  // #####################################################################
  pout << "Printing most important determinants" << endl;
  pout << format("%4s %10s  Determinant string") % ("Det") % ("weight")
       << endl;
  for (int root = 0; root < schd.nroots; root++) {
    pout << format("State : %3i") % (root) << endl;
    MatrixXx prevci = 1. * ci[root];
    int num = max(6, schd.printBestDeterminants);
    for (int i = 0; i < min(num, static_cast<int>(dets_size)); i++) {
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
      SHCIrdm::EvaluateRDM(sparse_ham.connections, SHMDets, dets_size, SHMci, SHMci, sparse_ham.orbDifference, nelec, schd, i, twoRDM);
      pout << "compute energy from rdm" << endl;
      SHCIrdm::ComputeEnergyFromSpinRDM(norbs, nelec, I1, I2, coreE, twoRDM);
      if (commrank==0) SHCIrdm::saveRDM_hdf5(schd, twoRDM, i);
    }
  }

#ifndef SERIAL
  world.barrier();
#endif

  pout << endl << endl;
  pout << "**************************************************************"
       << endl;
  pout << "Returning without error" << endl;
  pout << "**************************************************************"
       << endl << endl << endl;

  // Start of SSMRPT calculation.

  // Read integrals from HDF5 file.
  // Read integral pphh, where np and nh are the number of particles and holes.
  // np and nh are stored as nparticle and nhole in the HDF5 file.


  // 8 class of excitations in total.
  /*
    For CAS reference.
    Single excitations
    class 1: i->t
    class 2: i->a it->ua
    class 3: t->a, tu->va
    
    Double excitations
    class 4: ij->tu
    class 5: ij->ta
    class 6: ij->ab
    class 7: it->ab
    class 8: ij->ab
    
    For selected CI reference
    t->u
    tu->vw
  */
  int na;
  int nc;
  int nv;
  int np;
  int nh;
  int norb;
  vector<dcomplex> pphh_vec;
  vector<dcomplex> fpq_vec;
  h5::File feri("eri.chk", h5::File::ReadOnly);
  feri.getDataSet("nc").read(nc);
  feri.getDataSet("na").read(na);
  feri.getDataSet("nv").read(nv);
  feri.getDataSet("pphh").read(pphh_vec);
  feri.getDataSet("fpq").read(fpq_vec);
  np = na + nc;
  nh = na + nv;
  norb = nc + na + nv;
  Eigen::Map<const Eigen::MatrixXcd> pphh(pphh_vec.data(), np*np, nh*nh);
  Eigen::Map<const Eigen::MatrixXcd> fpq(fpq_vec.data(), norb, norb);
  cout << "pphh: " << pphh.cols() << pphh.rows() << endl;
  cout << "nc: " << nc << endl;
  cout << "na: " << na << endl;
  cout << "nv: " << nv << endl;
  
  // Create a GMRES solver instance
  GMRES<MatrixXcd> gmres_solver;
  gmres_solver.setMaxIterations(100);  // Set maximum iterations
  gmres_solver.setTolerance(1e-6);       // Set convergence tolerance
  
  MatrixXcd h_munu(dets_size, dets_size);

  auto civector = ci[0];
  auto e_cas = E0[0];
  for (int idet=0; idet<dets_size; idet++) {
    for (int jdet=0; jdet<dets_size; jdet++) {
      if (idet == jdet)
        h_munu(idet, idet) = complex<double>{0.0, 0.0};
      h_munu(idet, jdet) = hamiltonian(idet, jdet) * civector(idet, 0) / civector(jdet, 0);
    }
  }
  auto iden = MatrixXcd::Identity(dets_size, dets_size);
  MatrixXcd x_mat(dets_size, dets_size);
  VectorXcd v_vec = VectorXcd::Zero(dets_size);
  VectorXcd t_vec = VectorXcd::Zero(dets_size);

  for (int i = 0; i < nc; i++) {
    for (int j = 0; j < i; j++) {
      for (int a = na; a < np; a++) {
        for (int b = na; b < a; b++) {
          auto ijab = fpq(i,i) + fpq(j,j) - fpq(a+nc,a+nc) - fpq(b+nc,b+nc);
          v_vec.setConstant(pphh(i*np+j, a*nh+b)-pphh(i*np+j, b*nh+a));
          for (int idet = 0; idet < dets_size; idet++) {
            auto denom = complex<double>(e_cas) - hamiltonian(idet, idet) + ijab;
            v_vec(idet) /= denom;
            x_mat.col(idet) = h_munu.col(idet) / (e_cas - hamiltonian(idet, idet) + denom); 
          }
          MatrixXcd A = iden + x_mat;
          gmres_solver.compute(A);
          t_vec = gmres_solver.solve(v_vec);
          cout << "Number of iterations: " << gmres_solver.iterations() << endl;
          cout << "Estimated error: " << gmres_solver.error() << endl;
        }
      }
    }
  }
  
  return 0;
}
