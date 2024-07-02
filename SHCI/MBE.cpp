/*
  Developed by Sandeep Sharma
  with contributions from James E. T. Smith and Adam A. Holmes
  2017 Copyright (c) 2017, Sandeep Sharma

  This file is part of DICE.

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
#include <fstream>
#include <list>
#include <set>
#include <tuple>

#include "Davidson.h"
#include "Determinants.h"
#include "Hmult.h"
#include "SHCIbasics.h"
#include "SHCIgetdeterminants.h"
#include "SHCImakeHamiltonian.h"
#include "SHCIrdm.h"
#include "SHCItime.h"
#include "boost/format.hpp"
#include "global.h"
#include "input.h"
#include "integral.h"
#include "SchmidtStates.h"

#ifndef SERIAL
#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#endif
#include <unistd.h>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/serialization/vector.hpp>
#include <cstdlib>
#include <numeric>
#ifndef Complex
#include "cdfci.h"
#endif
#include "LCC.h"
#include "SHCIshm.h"
#include "SOChelper.h"
#include "communicate.h"
#include "symmetry.h"
MatrixXd symmetry::product_table;
#include <algorithm>

// Initialize
using namespace Eigen;
using namespace boost;
int HalfDet::norbs = 1;      // spin orbitals
int Determinant::norbs = 1;  // spin orbitals
int Determinant::EffDetLen = 1;
char Determinant::Trev = 0;  // Time reversal
Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> Determinant::LexicalOrder;

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
  pout << "     ____  _\n";
  pout << "    |  _ \\(_) ___ ___\n";
  pout << "    | | | | |/ __/ _ \\\n";
  pout << "    | |_| | | (_|  __/\n";
  pout << "    |____/|_|\\___\\___|   v1.0\n";
  pout << endl;
  pout << endl;
  pout << "**************************************************************"
       << endl;
  pout << "Dice  Copyright (C) 2017  Sandeep Sharma" << endl;
  pout << endl;
  pout << "This program is distributed in the hope that it will be useful,"
       << endl;
  pout << "but WITHOUT ANY WARRANTY; without even the implied warranty of"
       << endl;
  pout << "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE." << endl;
  pout << "See the GNU General Public License for more details." << endl;
  pout << endl;
  pout << "Author:       Sandeep Sharma" << endl;
  pout << "Contributors: James E Smith, Adam A Holmes, Bastien Mussard" << endl;
  pout << "For detailed documentation on Dice please visit" << endl;
  pout << "https://sanshar.github.io/Dice/" << endl;
  pout << "and our group page for up to date information on other projects"
       << endl;
  pout << "http://www.colorado.edu/lab/sharmagroup/" << endl;
  pout << "**************************************************************"
       << endl;
  pout << endl;

  char* user;
  user = (char*)malloc(10 * sizeof(char));
  user = getlogin();

  time_t t = time(NULL);
  struct tm* tm = localtime(&t);
  char date[64];
  strftime(date, sizeof(date), "%c", tm);

  printf("User:             %s\n", user);
  printf("Date:             %s\n", date);
  printf("PID:              %d\n", getpid());
  pout << endl;
  printf("Path:             %s\n", argv[0]);
  printf("Commit:           %s\n", GIT_HASH);
  printf("Branch:           %s\n", GIT_BRANCH);
  printf("Compilation Date: %s %s\n", __DATE__, __TIME__);
  // printf("Cores:            %s\n","TODO");
}

// Read Input
void readInput(string input, vector<std::vector<int> >& occupied,
               schedule& schd);

// PT message
void log_pt(schedule& schd) {
  pout << endl;
  pout << endl;
  pout << "**************************************************************"
       << endl;
  pout << "PERTURBATION THEORY STEP  " << endl;
  pout << "**************************************************************"
       << endl;
  if (schd.stochastic == true && schd.DoRDM) {
    schd.DoRDM = false;
    pout << "(We cannot perform PT RDM with stochastic PT. Disabling RDM.)"
         << endl
         << endl;
  }
}

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
  std::vector<std::vector<int> > HFoccupied;
  schedule schd;
  if (commrank == 0) readInput(inputFile, HFoccupied, schd);
  if (schd.outputlevel > 0 && commrank == 0) Time::print_time("begin");
  if (DetLen % 2 == 1) {
    pout << "Change DetLen in global to an even number and recompile." << endl;
    exit(0);
  }

#ifndef SERIAL
  mpi::broadcast(world, HFoccupied, 0);
  mpi::broadcast(world, schd, 0);
#endif

  // Time reversal symmetry
  if (HFoccupied[0].size() % 2 != 0 && schd.Trev != 0) {
    pout << endl
         << "Cannot use time reversal symmetry for odd electron system."
         << endl;
    schd.Trev = 0;
  }
  Determinant::Trev = schd.Trev;

  // Set the random seed
  startofCalc = getTime();
  srand(schd.randomSeed + commrank);
  if (schd.outputlevel > 1) pout << "#using seed: " << schd.randomSeed << endl;

  std::cout.precision(15);

  // Read the Hamiltonian (integrals, orbital irreps, num-electron etc.)
  twoInt I2;
  oneInt I1;
  int nelec;
  int norbs;
  double coreE = 0.0, eps;
  std::vector<int> irrep;
  readIntegrals(schd.integralFile, I2, I1, nelec, norbs, coreE, irrep);

  // Check
  if (HFoccupied[0].size() != nelec) {
    pout << "The number of electrons given in the FCIDUMP should be";
    pout << " equal to the nocc given in the shci input file." << endl;
    exit(0);
  }

  // LCC
  if (schd.doLCC) {
    // no nact was given in the input file
    if (schd.nact == 1000000)
      schd.nact = norbs - schd.ncore;
    else if (schd.nact + schd.ncore > norbs) {
      pout << "core + active orbitals = " << schd.nact + schd.ncore;
      pout << " greater than orbitals " << norbs << endl;
      exit(0);
    }
  }

  // Setup the lexical table for the determinants
  norbs *= 2;
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
  for (int i = 0; i < norbs / 2; i++) allorbs.push_back(i);
  twoIntHeatBath I2HB(1.e-10);
  twoIntHeatBathSHM I2HBSHM(1.e-10);
  if (commrank == 0) I2HB.constructClass(allorbs, I2, I1, norbs / 2);
  I2HBSHM.constructClass(norbs / 2, I2HB);


  //if the calculation is not doing CISD then modify the restrict
  //input to do that
  if (commrank == 0) {
    if (schd.restrictionsV.size() == 0) {
      int minElec=0, maxElec=2;
      int nocc = (HFoccupied.size()+1)/2;

      vector<int> virtOrbs;
      for (int i=nocc; i<norbs/2; i++)
        virtOrbs.push_back(i);

      schd.restrictionsV.push_back(OccRestrictions(minElec, maxElec, virtOrbs));
    }
  }


  int num_thrds;

  // If SOC is true then read the SOC integrals
#ifndef Complex
  if (schd.doSOC) {
    pout << "doSOC option works with complex coefficients." << endl;
    pout << "Uncomment the -Dcomplex in the make file and recompile." << endl;
    exit(0);
  }
#else
  if (schd.doSOC) {
    readSOCIntegrals(I1, norbs, "SOC", schd);
#ifndef SERIAL
    mpi::broadcast(world, I1, 0);
#endif
  }
#endif

  // Unlink the integral shared memory
  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());

  // Have the dets, ci coefficient and diagnoal on all processors
  vector<MatrixXx> ci(schd.nroots, MatrixXx::Zero(HFoccupied.size(), 1));
  vector<MatrixXx> vdVector(schd.nroots);  // these vectors are used to
                                           // calculate the response equations
  double Psi1Norm = 0.0;

  // #####################################################################
  // Reference determinant
  // #####################################################################
  pout << endl;
  pout << endl;
  pout << "**************************************************************\n";
  pout << "SELECTING REFERENCE DETERMINANT(S)\n";
  pout << "**************************************************************\n";

  // Make HF determinant
  int lowestEnergyDet = 0;
  double lowestEnergy = 1.e12;
  vector<Determinant> Dets(HFoccupied.size());
  for (int d = 0; d < HFoccupied.size(); d++) {
    for (int i = 0; i < HFoccupied[d].size(); i++) {
      if (Dets[d].getocc(HFoccupied[d][i])) {
        pout << "orbital " << HFoccupied[d][i]
             << " appears twice in input determinant number " << d << endl;
        exit(0);
      }
      Dets[d].setocc(HFoccupied[d][i], true);
    }
    if (Determinant::Trev != 0) Dets[d].makeStandard();
    for (int i = 0; i < d; i++) {
      if (Dets[d] == Dets[i]) {
        pout << "Determinant " << Dets[d]
             << " appears twice in the input determinant list." << endl;
        exit(0);
      }
    }
    double E = Dets.at(d).Energy(I1, I2, coreE);
    pout << Dets[d] << " Given Ref. Energy:    "
         << format("%18.10f") % (E) << endl;
    if (E < lowestEnergy) {
      lowestEnergy = E;
      lowestEnergyDet = d;
    }
  }

  if (schd.searchForLowestEnergyDet) {
    // Set up the symmetry class
    symmetry molSym(schd.pointGroup, irrep, schd.irrep);

    // Check the users specified determinants. If they use multiple spins and/or
    // irreps, ignore the give targetIrrep and just use the give determinants.
    molSym.checkTargetStates(Dets, schd.spin);

    if (schd.pointGroup != "dooh" && schd.pointGroup != "coov" &&
        molSym.init_success) {
      vector<Determinant> tempDets(Dets);

      bool spin_specified = true;
      if (schd.spin == -1) {  // Set spin if none specified by user
        spin_specified = false;
        schd.spin = Dets[0].Nalpha() - Dets[0].Nbeta();
        pout << "No spin specified, using spin from first reference "
                "determinant. "
                "Setting target spin to "
             << schd.spin << endl;
      }
      for (int d = 0; d < HFoccupied.size(); d++) {
        // Guess the lowest energy det with given symmetry from one body
        // integrals.
        molSym.estimateLowestEnergyDet(schd.spin, I1, irrep, HFoccupied.at(d),
                                       tempDets.at(d));
        // Generate list of connected determinants to guess determinant.
        SHCIgetdeterminants::getDeterminantsVariational(
            tempDets.at(d), 0.00001, 1, 0.0, I1, I2, I2HBSHM, irrep, coreE, 0,
            tempDets, schd, 0, nelec);

        // If spin is specified we assume the user wants a particular
        // determinant even if it's higher in energy than the HF so we keep it.
        // If the user didn't specify then we keep the lowest energy determinant
        // Check all connected and find lowest energy.
        int spin_HF = Dets[d].Nalpha() - Dets[d].Nbeta();
        if (spin_specified && spin_HF != schd.spin) {
          Dets.at(d) = tempDets.at(d);
        }

        // Same for irrep
        if (molSym.targetIrrep != molSym.getDetSymmetry(Dets[d])) {
          Dets.at(d) = tempDets.at(d);
          pout << "WARNING: Given determinants have different irrep than the "
                  "target irrep\n\tspecified. Using the specified irrep."
               << endl;
        }

        for (int cd = 0; cd < tempDets.size(); cd++) {
          if (tempDets.at(d).connected(tempDets.at(cd))) {
            bool correct_spin = abs(tempDets.at(cd).Nalpha() -
                                    tempDets.at(cd).Nbeta()) == schd.spin;
            if (correct_spin) {
              bool lower_energy = Dets.at(d).Energy(I1, I2, coreE) >
                                  tempDets.at(cd).Energy(I1, I2, coreE);
              bool correct_irrep =
                  molSym.getDetSymmetry(tempDets.at(cd)) == molSym.targetIrrep;
              if (lower_energy && correct_irrep) {
                Dets.at(d) = tempDets.at(cd);
              }
            }
          }
        }  // end cd
        pout << Dets[d] << " Starting Det. Energy: "
             << format("%18.10f") % (Dets[d].Energy(I1, I2, coreE)) << endl;
      }  // end d

    } else {
      pout << "\nWARNING: Skipping Ref. Determinant Search for pointgroup "
           << schd.pointGroup << "\nUsing given determinants as reference"
           << endl;
    }  // End if (Search for Ref. Det)
  }    // end searchForLowestEnergyDet

  schd.HF = Dets[0];

  if (commrank == 0) {
    for (int j = 0; j < ci[0].rows(); j++) ci[0](j, 0) = 1.0;
    ci[0](lowestEnergyDet,0) += Dets.size();
    ci[0] = ci[0] / ci[0].norm();
  }

#ifndef SERIAL
  mpi::broadcast(world, ci, 0);
#endif

  // #####################################################################
  // Variational step
  // #####################################################################
  pout << endl;
  pout << endl;
  pout << "**************************************************************"
       << endl;
  pout << "VARIATIONAL STEP  " << endl;
  pout << "**************************************************************"
       << endl;

  vector<double> E0;
#ifndef Complex
  if (schd.cdfci_on == 0 && schd.restart) {
      cdfci::sequential_solve_omp(schd, I1, I2, I2HBSHM, irrep, coreE, E0, ci, Dets);
  }
  else {
    E0 = SHCIbasics::DoVariational(
        ci, Dets, schd, I2, I2HBSHM, irrep, I1, coreE, nelec, schd.DoRDM);
  }
#else
    E0 = SHCIbasics::DoVariational(
        ci, Dets, schd, I2, I2HBSHM, irrep, I1, coreE, nelec, schd.DoRDM);
#endif

  MatrixXd AA, BB, AB, A, B;
  MatrixXi AAidx, BBidx, ABidx;
  getMatrix(Dets, norbs, nelec/2, ci[0], A, B, AA, BB, AB, AAidx, BBidx, ABidx);



  Determinant* SHMDets;
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
    for (int j = 0; j < E0.size(); ++j) {
      // pout << "Writing energy " << E0[j] << "  to file: " << efile << endl;
      fwrite(&E0[j], 1, sizeof(double), f);
    }
    fclose(f);
  }
  
  if (commrank == 0) {
    if (schd.printAllDeterminants) {
      pout << "Printing all determinants"<<endl;
      pout << format("%4s %10s  ") %("Det") %("weight"); pout << "Determinant string"<<endl;
      for (int root=0; root<schd.nroots; root++) {
        pout << "State :"<<root<<endl;
        MatrixXx prevci = 1.*ci[root];
        for (int i=0; i<static_cast<int>(DetsSize); i++) {
          double parity = getParityForDiceToAlphaBeta(SHMDets[i]);
#ifdef Complex
          pout << format("%4i %18.8e  ") %(i) %(abs(prevci(i,0))); pout << SHMDets[i]<<endl;
#else
          pout << format("%18.8e  ") %(prevci(i,0)*parity); pout << SHMDets[i]<<endl;
#endif
        }
      }
    }
    if (schd.writeBestDeterminants > 0) {
      int num = min(schd.writeBestDeterminants, static_cast<int>(DetsSize));
      int nspatorbs = Determinant::norbs/2;
      for (int root = 0; root < schd.nroots; root++) {
        string fname;
        if (root == 0) fname = "dets.bin";
        else {
          fname = "dets_";
          fname.append(to_string(root));
          fname.append(".bin");
        }
        ofstream fout = ofstream(fname, ios::binary);
        fout.write((char*) &num, sizeof(int));
        fout.write((char*) &nspatorbs, sizeof(int));
        MatrixXx prevci = 1. * ci[root];
        std::vector<size_t> idx(static_cast<int>(DetsSize));
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&prevci](size_t i1, size_t i2){return abs(prevci(i1, 0)) > abs(prevci(i2, 0));});
        for (int i = 0; i < num; i++) {
          int m = idx[i];
          double parity = getParityForDiceToAlphaBeta(SHMDets[m]);
          double wciCoeff = parity * std::real(prevci(m, 0));
          fout.write((char*) &wciCoeff, sizeof(double));
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

//#ifdef Complex
//  //make the largest magnitude ci coefficient real
//  for (int root = 0; root < schd.nroots; root++) {
//    MatrixXx& prevci =  ci[root];
//    compAbs comp;
//    int m = distance(
//        &prevci(0, 0),
//        max_element(&prevci(0, 0), &prevci(0, 0) + prevci.rows(), comp));
//    complex<double> maxC = prevci(m,0);
//    for (int i=0; i< static_cast<int>(DetsSize); i++)
//      prevci(i,0) = prevci(i,0)*abs(maxC)/maxC;
//  }
//#endif
  
  // #####################################################################
  // Print the 5 most important determinants and their weights
  // #####################################################################
  pout << "Printing most important determinants" << endl;
  pout << format("%4s %10s  Determinant string") % ("Det") % ("weight") << endl;
  for (int root = 0; root < schd.nroots; root++) {
    pout << format("State : %3i") % (root) << endl;
    MatrixXx prevci = 1. * ci[root];
    int num = max(6, schd.printBestDeterminants);
    complex<double> maxC = 0;
    for (int i = 0; i < min(num, static_cast<int>(DetsSize)); i++) {
      compAbs comp;
      int m = distance(
          &prevci(0, 0),
          max_element(&prevci(0, 0), &prevci(0, 0) + prevci.rows(), comp));
      double parity = getParityForDiceToAlphaBeta(SHMDets[m]);
      if (i == 0) maxC = prevci(m,0);
#ifdef Complex
      pout << format("%4i %18.10f %18.10f ") % (i) % (prevci(m, 0)*abs(maxC)/maxC) % (abs(prevci(m,0)));
      pout << SHMDets[m] << endl;
#else
      pout << format("%4i %18.10f  ") % (i) % (prevci(m, 0)*parity);
      pout << SHMDets[m] << endl;
#endif
      // pout <<"#"<< i<<"  "<<prevci(m,0)<<"  "<<abs(prevci(m,0))<<"
      // "<<Dets[m]<<endl;
      prevci(m, 0) = 0.0;
    }
  }  // end root
  pout << std::flush;
 


  return 0;
}
