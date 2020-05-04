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
#include "LCC.h"
#include "SHCIshm.h"
#include "SOChelper.h"
#include "communicate.h"

#include "symmetry.h"
MatrixXd symmetry::product_table;
#include <algorithm>
#include <boost/bind.hpp>

// Initialize
using namespace Eigen;
using namespace boost;
// int HalfDet::norbs = 1;  // spin orbitals // JETS: rm

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
  // if (commrank == 0)
  license(argv);

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
  // twoInt I2;
  // oneInt I1;
  // int nelec;
  // int norbs;
  // double coreE = 0.0;  //, eps; JETS: eps is unused
  std::vector<int> irrep;
  // readIntegrals(schd.integralFile, I2, I1, nelec, norbs, coreE, irrep); //
  // From Old Dice
  readIntegralsAndInitializeDeterminantStaticVariables(schd.integralFile);

  int norbs, nelec, n_spinorbs;
  norbs = Determinant::norbs;
  n_spinorbs = Determinant::n_spinorbs;
  nelec = Determinant::nalpha + Determinant::nbeta;
  // pout << "norbs " << norbs << std::endl;  // JETS: rm
  // exit(0);                                 // JETS: rm

  // Check
  if (HFoccupied[0].size() != (size_t)nelec) {
    pout << "The number of electrons given in the FCIDUMP should be";
    pout << " equal to the nocc given in the shci input file." << endl;
    pout << "Nelec from FCIDUMP " << nelec << endl;
    pout << "Nelec from SHCI input " << HFoccupied[0].size() << endl;
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
  // norbs *= 2;
  // JETS: These are already set by
  // readIntegralsAndInitializeDeterminantStaticVariables

  // Determinant::norbs = norbs;  // spin orbitals
  // HalfDet::norbs = norbs;      // spin orbitals
  // Determinant::EffDetLen = norbs / 64 + 1;
  // Determinant::initLexicalOrder(nelec);
  // if (Determinant::EffDetLen > DetLen) {
  //   pout << "change DetLen in global.h to " << Determinant::EffDetLen
  //        << " and recompile " << endl;
  //   exit(0);
  // }

  // Initialize the Heat-Bath integrals
  std::vector<int> allorbs;
  for (int i = 0; i < n_spinorbs / 2; i++) allorbs.push_back(i);
  twoIntHeatBath I2HB(1.e-10);
  twoIntHeatBathSHM I2HBSHM(1.e-10);
  if (commrank == 0) I2HB.constructClass(allorbs, I2, I1, n_spinorbs / 2);
  I2HBSHM.constructClass(n_spinorbs / 2, I2HB);

  // int num_thrds; // JETS: num_threads is unused

  // If SOC is true then read the SOC integrals
#ifndef Complex
  if (schd.doSOC) {
    pout << "doSOC option works with complex coefficients." << endl;
    pout << "Uncomment the -Dcomplex in the make file and recompile." << endl;
    exit(0);
  }
#else
  if (schd.doSOC) {
    readSOCIntegrals(I1, n_spinorbs, "SOC");
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
  vector<Determinant> Dets(HFoccupied.size());
  for (size_t d = 0; d < HFoccupied.size(); d++) {
    for (size_t i = 0; i < HFoccupied[d].size(); i++) {
      if (Dets[d].getocc(HFoccupied[d][i])) {
        pout << "orbital " << HFoccupied[d][i]
             << " appears twice in input determinant number " << d << endl;
        exit(0);
      }
      Dets[d].setocc(HFoccupied[d][i], true);
    }
    if (Determinant::Trev != 0) Dets[d].makeStandard();
    for (size_t i = 0; i < d; i++) {
      if (Dets[d] == Dets[i]) {
        pout << "Determinant " << Dets[d]
             << " appears twice in the input determinant list." << endl;
        exit(0);
      }
    }
  }
  pout << Dets[0] << " Given HF Energy:      "
       << format("%18.10f") % (Dets.at(0).Energy(I1, I2, coreE)) << endl;

  // symmetry molSym(schd.pointGroup, irrep);

  // if (schd.pointGroup != "dooh" && schd.pointGroup != "coov" &&
  //     molSym.init_success) {
  //   vector<Determinant> tempDets(Dets);
  //   bool spin_specified = true;
  //   if (schd.spin == -1) {  // Set spin if none specified by user
  //     spin_specified = false;
  //     schd.spin = Dets[0].Nalpha() - Dets[0].Nbeta();
  //     // pout << "Setting spin to " << schd.spin << endl;
  //   }
  //   for (int d = 0; d < HFoccupied.size(); d++) {
  //     // Guess the lowest energy det with given symmetry from one body
  //     // integrals.
  //     molSym.estimateLowestEnergyDet(schd.spin, schd.irrep, I1, irrep,
  //                                    HFoccupied.at(d), tempDets.at(d));

  //     // Generate list of connected determinants to guess determinant.
  //     SHCIgetdeterminants::getDeterminantsVariational(
  //         tempDets.at(d), 0.00001, 1, 0.0, I1, I2, I2HBSHM, irrep, coreE, 0,
  //         tempDets, schd, 0, nelec);

  //     // If spin is specified we assume the user wants a particular
  //     determinant
  //     // even if it's higher in energy than the HF so we keep it. If the user
  //     // didn't specify then we keep the lowest energy determinant
  //     // Check all connected and find lowest energy.
  //     int spin_HF = Dets[d].Nalpha() - Dets[d].Nbeta();
  //     if (spin_specified && spin_HF != schd.spin) {
  //       Dets.at(d) = tempDets.at(0);
  //     }
  //     for (int cd = 0; cd < tempDets.size(); cd++) {
  //       if (tempDets.at(d).connected(tempDets.at(cd))) {
  //         if (abs(tempDets.at(cd).Nalpha() - tempDets.at(cd).Nbeta()) ==
  //             schd.spin) {
  //           char repArray[tempDets.at(cd).norbs];
  //           tempDets.at(cd).getRepArray(
  //               repArray);  // JETS: This breaks b/c we don't have
  //                           // `getRepArray()` anymore. I'll fix this later
  //           if (Dets.at(d).Energy(I1, I2, coreE) >
  //                   tempDets.at(cd).Energy(I1, I2, coreE) &&
  //               molSym.getSymmetry(repArray, irrep) == schd.irrep) {
  //             Dets.at(d) = tempDets.at(cd);
  //           }
  //         }
  //       }
  //     }  // end cd
  //     pout << Dets[d] << " Starting Det. Energy: "
  //          << format("%18.10f") % (Dets[d].Energy(I1, I2, coreE)) << endl;
  //   }  // end d

  // } else {
  //   pout << "Skipping Ref. Determinant Search for pointgroup "
  //        << schd.pointGroup << "\nUsing HF as ref determinant" << endl;
  // }  // End if (Search for Ref. Det)

  schd.HF = Dets[0];

  if (commrank == 0) {
    for (int j = 0; j < ci[0].rows(); j++) ci[0](j, 0) = 1.0;
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

  vector<double> E0 = SHCIbasics::DoVariational(
      ci, Dets, schd, I2, I2HBSHM, irrep, I1, coreE, nelec, schd.DoRDM);
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
    for (size_t j = 0; j < E0.size(); ++j) {
      // pout << "Writing energy " << E0[j] << "  to file: " << efile << endl;
      fwrite(&E0[j], 1, sizeof(double), f);
    }
    fclose(f);
  }

  // #####################################################################
  // Print the 5 most important determinants and their weights
  // #####################################################################
  pout << "Printing most important determinants" << endl;
  pout << format("%4s %10s  Determinant string") % ("Det") % ("weight") << endl;
  for (int root = 0; root < schd.nroots; root++) {
    pout << format("State : %3i") % (root) << endl;
    MatrixXx prevci = 1. * ci[root];
    int num = max(6, schd.printBestDeterminants);
    for (int i = 0; i < min(num, static_cast<int>(DetsSize)); i++) {
      compAbs comp;
      int m = distance(
          &prevci(0, 0),
          max_element(&prevci(0, 0), &prevci(0, 0) + prevci.rows(), comp));
      double parity = getParityForDiceToAlphaBeta(SHMDets[m]);
#ifdef Complex
      pout << format("%4i %18.10f  ") % (i) % (abs(prevci(m, 0)));
      pout << SHMDets[m] << endl;
#else
      pout << format("%4i %18.10f  ") % (i) % (prevci(m, 0) * parity);
      pout << SHMDets[m] << endl;
#endif
      // pout <<"#"<< i<<"  "<<prevci(m,0)<<"  "<<abs(prevci(m,0))<<"
      // "<<Dets[m]<<endl;
      prevci(m, 0) = 0.0;
    }
  }  // end root
  pout << std::flush;

  // #####################################################################
  // RDMs
  // #####################################################################
  if (schd.doSOC || schd.DoOneRDM || schd.DoThreeRDM || schd.DoFourRDM) {
    pout << endl;
    pout << endl;
    pout << "**************************************************************"
         << endl;
    pout << "RDMs CALCULATIONS" << endl;
    pout << "**************************************************************"
         << endl;
  }

  /*
  if (schd.quasiQ) {
    double bkpepsilon2 = schd.epsilon2;
    schd.epsilon2 = schd.quasiQEpsilon;
    for (int root=0; root<schd.nroots;root++) {
      E0[root] += SHCIbasics::DoPerturbativeDeterministic(SHMDets, ci[root],
                                                          E0[root], I1, I2,
  I2HBSHM, irrep, schd, coreE, nelec, root, vdVector, Psi1Norm, true);
      ci[root] = ci[root]/ci[root].norm();
    }
    schd.epsilon2 = bkpepsilon2;
  }
  */

#ifndef SERIAL
  world.barrier();
#endif

  // SpinRDM
  vector<MatrixXx> spinRDM(3, MatrixXx::Zero(norbs, norbs));
  // SOC
#ifdef Complex
  if (schd.doSOC) {
    pout << "PERFORMING G-tensor calculations" << endl;

    // dont do this here, if perturbation theory is switched on
    if (schd.doGtensor) {
#ifndef SERIAL
      mpi::broadcast(world, ci, 0);
#endif
      SOChelper::calculateSpinRDM(spinRDM, ci[0], ci[1], SHMDets, DetsSize,
                                  norbs, nelec);
      SOChelper::doGTensor(ci, SHMDets, E0, DetsSize, norbs, nelec, spinRDM);
      if (commrank != 0) {
        ci[0].resize(1, 1);
        ci[1].resize(1, 1);
      }
    }
  }
#endif

  // JETS: Unused at the moment
  // if (schd.DoOneRDM) {
  //   pout << "Calculating 1-RDM..." << endl;
  //   MatrixXx s1RDM;
  //   CItype* ciroot;
  // }

  // 3RDM
  if (schd.DoThreeRDM) {
    pout << "Calculating 3-RDM..." << endl;
    MatrixXx s3RDM, threeRDM;
    CItype* ciroot;
    SHMVecFromMatrix(ci[0], ciroot, shcicMax, cMaxSegment, regioncMax);

    if (schd.DoSpinRDM)
      threeRDM.setZero(norbs * norbs * norbs, norbs * norbs * norbs);
    s3RDM.setZero(norbs * norbs * norbs / 8, norbs * norbs * norbs / 8);
    SHCIrdm::Evaluate3RDM(SHMDets, DetsSize, ciroot, ciroot, nelec, schd, 0,
                          threeRDM, s3RDM);
    SHCIrdm::save3RDM(schd, threeRDM, s3RDM, 0, norbs);
  }

  // 4RDM
  if (schd.DoFourRDM) {
    pout << "Calculating 4-RDM..." << endl;
    MatrixXx s4RDM, fourRDM;
    CItype* ciroot;
    SHMVecFromMatrix(ci[0], ciroot, shcicMax, cMaxSegment, regioncMax);

    if (schd.DoSpinRDM)
      fourRDM.setZero(norbs * norbs * norbs * norbs,
                      norbs * norbs * norbs * norbs);
    s4RDM.setZero(norbs * norbs * norbs * norbs / 16,
                  norbs * norbs * norbs * norbs / 16);
    SHCIrdm::Evaluate4RDM(SHMDets, DetsSize, ciroot, ciroot, nelec, schd, 0,
                          fourRDM, s4RDM);
    SHCIrdm::save4RDM(schd, fourRDM, s4RDM, 0, norbs);
  }

  // #####################################################################
  // PT
  // #####################################################################
  if (schd.doSOC && !schd.stochastic) {  // deterministic SOC calculation
    log_pt(schd);
    if (schd.doGtensor) {
      pout << "Gtensor calculation not supported with deterministic PT for "
              "more than 2 roots."
           << endl;
      pout << "Just performing the ZFS calculations." << endl;
    }
    MatrixXx Heff = MatrixXx::Zero(E0.size(), E0.size());

    /*
    for (int root1 =0 ;root1<schd.nroots; root1++) {
      for (int root2=root1+1 ;root2<schd.nroots; root2++) {
        Heff(root1, root1) = 0.0; Heff(root2, root2) = 0.0; Heff(root1, root2)
= 0.0; SHCIbasics::DoPerturbativeDeterministicOffdiagonal(Dets, ci[root1],
E0[root1], ci[root2], E0[root2], I1, I2, I2HBSHM, irrep, schd, coreE, nelec,
root1, Heff(root1,root1), Heff(root2, root2), Heff(root1, root2), spinRDM);
#ifdef Complex
        Heff(root2, root1) = conj(Heff(root1, root2));
#else
        Heff(root2, root1) = Heff(root1, root2);
#endif
      }
    }
    */

    for (int root1 = 0; root1 < schd.nroots; root1++)
      Heff(root1, root1) += E0[root1];

    SelfAdjointEigenSolver<MatrixXx> eigensolver(Heff);
    for (int j = 0; j < eigensolver.eigenvalues().rows(); j++) {
      E0[j] = eigensolver.eigenvalues()(j, 0);
      pout << str(
          boost::format("State: %3d,  E: %18.10f, dE: %10.2f\n") % j %
          (eigensolver.eigenvalues()(j, 0)) %
          ((eigensolver.eigenvalues()(j, 0) - eigensolver.eigenvalues()(0, 0)) *
           219470));
    }

    std::string efile;
    efile = str(boost::format("%s%s") % schd.prefix[0].c_str() % "/shci.e");
    FILE* f = fopen(efile.c_str(), "wb");
    for (size_t j = 0; j < E0.size(); ++j) {
      fwrite(&E0[j], 1, sizeof(double), f);
    }
    fclose(f);

#ifdef Complex
    // SOChelper::doGTensor(ci, Dets, E0, norbs, nelec, spinRDM);
    return 0;
#endif
  } else if (schd.doLCC) {
    log_pt(schd);
#ifndef Complex
    for (int root = 0; root < schd.nroots; root++) {
      CItype* ciroot;
      SHMVecFromMatrix(ci[root], ciroot, shcicMax, cMaxSegment, regioncMax);
      LCC::doLCC(SHMDets, ciroot, DetsSize, E0[root], I1, I2, I2HBSHM, irrep,
                 schd, coreE, nelec, root);
    }
#else
    pout << " Not for Complex" << endl;
#endif
  } else if (!schd.stochastic && schd.nblocks == 1) {
    log_pt(schd);
    double ePT = 0.0;
    std::string efile;
    efile = str(boost::format("%s%s") % schd.prefix[0].c_str() % "/shci.e");
    FILE* f = fopen(efile.c_str(), "wb");
    for (int root = 0; root < schd.nroots; root++) {
      CItype* ciroot;
      SHMVecFromMatrix(ci[root], ciroot, shcicMax, cMaxSegment, regioncMax);
      ePT = SHCIbasics::DoPerturbativeDeterministic(
          SHMDets, ciroot, DetsSize, E0[root], I1, I2, I2HBSHM, irrep, schd,
          coreE, nelec, root, vdVector, Psi1Norm);
      ePT += E0[root];
      // pout << "Writing energy " << ePT << "  to file: " << efile << endl;
      if (commrank == 0) fwrite(&ePT, 1, sizeof(double), f);
    }
    fclose(f);
  } else if (schd.SampleN != -1 && schd.singleList) {
    if (schd.nPTiter != 0) {
      log_pt(schd);
      vector<double> ePT(schd.nroots, 0.0);
      std::string efile;
      efile = str(boost::format("%s%s") % schd.prefix[0].c_str() % "/shci.e");
      FILE* f = fopen(efile.c_str(), "wb");
      for (int root = 0; root < schd.nroots; root++) {
        CItype* ciroot;
        SHMVecFromMatrix(ci[root], ciroot, shcicMax, cMaxSegment, regioncMax);
        ePT[root] = SHCIbasics::
            DoPerturbativeStochastic2SingleListDoubleEpsilon2AllTogether(
                SHMDets, ciroot, DetsSize, E0[root], I1, I2, I2HBSHM, irrep,
                schd, coreE, nelec, root);
        ePT[root] += E0[root];
        // pout << "Writing energy " << E0[root] << "  to file: " << efile <<
        // endl;
        if (commrank == 0) fwrite(&ePT[root], 1, sizeof(double), f);
      }
      fclose(f);

      if (schd.doSOC) {
        for (size_t j = 0; j < E0.size(); j++)
          pout << str(boost::format("State: %3d,  E: %18.10f, dE: %10.2f\n") %
                      j % (ePT[j]) % ((ePT[j] - ePT[0]) * 219470));
      }
    }  // end if iter!=0
  } else {
#ifndef SERIAL
    world.barrier();
#endif
    pout << "Error here" << endl;
    exit(0);
  }

  // THIS IS USED FOR RDM CALCULATION FOR DETERMINISTIC PT
  if ((schd.doResponse || schd.DoRDM) && schd.RdmType == RELAXED &&
      (!schd.stochastic && schd.nblocks == 1)) {
    if (schd.DavidsonType == DIRECT) {
      pout << "PT RDM not implemented with direct davidson." << endl;
      exit(0);
    }
    std::vector<MatrixXx> lambda(schd.nroots, MatrixXx::Zero(Dets.size(), 1));
    SHCImakeHamiltonian::SparseHam sparseHam;
    {
      char file[5000];
      sprintf(file, "%s/%d-hamiltonian.bkp", schd.prefix[0].c_str(), commrank);
      std::ifstream ifs(file, std::ios::binary);
      boost::archive::binary_iarchive load(ifs);
      load >> sparseHam.connections >> sparseHam.Helements >>
          sparseHam.orbDifference;
    }

    vector<CItype*> ciroot(schd.nroots);
    SHMVecFromMatrix(ci[0], ciroot[0], shcicMax, cMaxSegment, regioncMax);
    Hmult2 H(sparseHam);
    LinearSolver(H, E0[0], lambda[0], vdVector[0], ciroot, 1.e-5, false);
#ifndef SERIAL
    mpi::broadcast(world, lambda[0], 0);
#endif

    MatrixXx s2RDM, twoRDM;
    s2RDM.setZero(norbs * norbs / 4, norbs * norbs / 4);
    if (schd.DoSpinRDM)
      twoRDM.setZero(norbs * (norbs + 1) / 2, norbs * (norbs + 1) / 2);

    SHCIrdm::EvaluateRDM(sparseHam.connections, SHMDets, DetsSize,
                         &lambda[0](0, 0), ciroot[0], sparseHam.orbDifference,
                         nelec, schd, 0, twoRDM, s2RDM);

    // Add DoOneRDM Block

    if (commrank == 0) {
      MatrixXx s2RDMdisk, twoRDMdisk;
      SHCIrdm::loadRDM(schd, s2RDMdisk, twoRDMdisk, 0);
      s2RDMdisk = s2RDMdisk + s2RDM.adjoint() + s2RDM;
      SHCIrdm::saveRDM(schd, s2RDMdisk, twoRDMdisk, 0);
    }

    // pout << " response ";
    // SHCIrdm::ComputeEnergyFromSpatialRDM(norbs, nelec, I1, I2, coreE,
    // s2RDM);
  }  // end if doResponse||DoRDM && RdmType && !stochastic...

  // #####################################################################
  // Extrapolate
  // #####################################################################
  if (schd.extrapolate) {  // performing extrapolation
    if (schd.nroots > 1) {
      pout << " extrapolation only supported for single root " << endl;
      exit(0);
    }
    for (int root = 0; root < schd.nroots; root++) {
      vector<double> var(4, 0.0), PT(4, 0.0);
      vector<int> nDets(4, 0);
      nDets[0] = DetsSize;
      var[0] = E0[0];
      std::string efile;
      efile = str(boost::format("%s%s") % schd.prefix[0].c_str() % "/shci.e");
      FILE* f = fopen(efile.c_str(), "rb");
      if (commrank == 0) fread(&PT[0], 1, sizeof(double), f);
#ifndef SERIAL
      mpi::broadcast(world, PT, 0);
#endif
      PT[0] -= var[0];

      // do 4 iterations for extrapolation
      for (int iter = 0; iter < 3; iter++) {
        if (commrank == 0) {
          char file[5000];
          sprintf(file, "%s/%d-variational.bkp", schd.prefix[0].c_str(),
                  commrank);
          std::ifstream ifs(file, std::ios::binary);
          boost::archive::binary_iarchive load(ifs);
          ci.clear();
          Dets.clear();
          int niter;
          load >> niter >> Dets;
          load >> ci;
          if (iter == 0) nDets[0] = Dets.size();
          DetsSize = Dets.size();
        }
        SHMVecFromVecs(Dets, SHMDets, shciDetsCI, DetsCISegment, regionDetsCI);
        if (commrank == 0) {
          std::vector<size_t> indices(DetsSize);
          for (int i = 0; i < DetsSize; i++) indices[i] = i;

          sort(indices.begin(), indices.end(), [&ci](size_t i1, size_t i2) {
            return abs(ci[0](i1, 0)) > abs(ci[0](i2, 0));
          });

          DetsSize = DetsSize * schd.extrapolationFactor;
          Dets.resize(DetsSize);
          MatrixXx cicopy = MatrixXx::Zero(DetsSize, 1);
          for (size_t i = 0; i < (size_t)DetsSize; i++) {
            Dets[i] = SHMDets[indices[i]];
            cicopy(i, 0) = ci[root](indices[i], 0);
          }
          ci[root].resize(DetsSize, 1);
          for (size_t i = 0; i < (size_t)DetsSize; i++) {
            ci[root](i, 0) = cicopy(i, 0);
          }
          ci[root] = ci[root] / ci[root].norm();
        }

#ifndef SERIAL
        mpi::broadcast(world, DetsSize, 0);
#endif
        nDets[iter + 1] = DetsSize;
        schd.epsilon1.resize(1);
        schd.epsilon1[0] = 1.e10;  // very large
        schd.restart = false;
        schd.fullrestart = false;
        schd.DoRDM = false;
        E0 = SHCIbasics::DoVariational(ci, Dets, schd, I2, I2HBSHM, irrep, I1,
                                       coreE, nelec, false);
        var[iter + 1] = E0[0];

        DetsSize = Dets.size();
#ifndef SERIAL
        mpi::broadcast(world, DetsSize, 0);
#endif
        SHMVecFromVecs(Dets, SHMDets, shciDetsCI, DetsCISegment, regionDetsCI);
        Dets.clear();

        CItype* ciroot;
        SHMVecFromMatrix(ci[root], ciroot, shcicMax, cMaxSegment, regioncMax);
        if (!schd.stochastic)
          PT[iter + 1] = SHCIbasics::DoPerturbativeDeterministic(
              SHMDets, ciroot, DetsSize, E0[root], I1, I2, I2HBSHM, irrep, schd,
              coreE, nelec, root, vdVector, Psi1Norm);
        else
          PT[iter + 1] = SHCIbasics::
              DoPerturbativeStochastic2SingleListDoubleEpsilon2AllTogether(
                  SHMDets, ciroot, DetsSize, E0[root], I1, I2, I2HBSHM, irrep,
                  schd, coreE, nelec, root);
      }  // end iter

      if (commrank == 0)
        printf("Ndet         Evar                  Ept               \n");
      for (int iter = 0; iter < 4; iter++)
        if (commrank == 0)
          printf("%10i   %18.10g    %18.10g \n", nDets[iter], var[iter],
                 PT[iter]);
    }  // end root
  }    // end extrapolate

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
  // std::system("rm -rf /dev/shm* 2>/dev/null");

  return 0;
}
