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
#include "Mytime.h"
#include "global.h"
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include "Determinants.h"
//#include "SHCImakeHamiltonian.h"
#include "SHCImake4cHamiltonian.h"
#include "SHCIrdm.h"
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
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include <boost/serialization/vector.hpp>
#include "communicate.h"
#include <boost/interprocess/managed_shared_memory.hpp>
#include "SOChelper.h"
#include "SHCIshm.h"
#include "LCC.h"
#include <numeric>
#include <cstdlib>

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


void readInput(string input, vector<std::vector<int> >& occupied, schedule& schd);


int main(int argc, char* argv[]) {

#ifndef SERIAL
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
#endif

  initSHM();

  license();

  string inputFile = "input.dat";
  if (argc > 1)
    inputFile = string(argv[1]);
  //Read the input file
  std::vector<std::vector<int> > HFoccupied;
  schedule schd;
  if (commrank == 0) readInput(inputFile, HFoccupied, schd);
  if (schd.outputlevel > 0 && commrank == 0) Time::print_time("begin");
  if (DetLen%2 == 1) {
    pout << "Change DetLen in global to an even number and recompile."<<endl;
    exit(0);
  }

#ifndef SERIAL
  mpi::broadcast(world, HFoccupied, 0);
  mpi::broadcast(world, schd, 0);
#endif
  

  if (HFoccupied[0].size()%2 != 0 && schd.Trev !=0) {
    pout << "Cannot use time reversal symmetry for odd electron system."<<endl;
    schd.Trev = 0;
  }
  Determinant::Trev = schd.Trev;


  //set the random seed
  startofCalc=getTime();
  srand(schd.randomSeed+commrank);
  if (schd.outputlevel>1) pout<<"#using seed: "<<schd.randomSeed<<endl;



  std::cout.precision(15);
  //read the hamiltonian (integrals, orbital irreps, num-electron etc.)
  twoInt I2; oneInt I1; int nelec; int norbs; double coreE=0.0, eps;
  std::vector<int> irrep;
  readIntegrals(schd.integralFile, I2, I1, nelec, norbs, coreE, irrep);
  if (HFoccupied[0].size() != nelec) {
    pout << "The number of electrons given in the FCIDUMP should be equal to the nocc given in the shci input file."<<endl;
    exit(0);
  }
  if (schd.doLCC) {
    //no nact was given in the input file
    if (schd.nact == 1000000) 
      schd.nact = norbs - schd.ncore;
    else if (schd.nact+schd.ncore > norbs) {
      pout << "core + active orbitals = " << schd.nact+schd.ncore << 
	            " greater than orbitals " << norbs << endl;
      exit(0);
    }
  }

  //setup the lexical table for the determinants
  
  //For relativistic, norbs is already the spin orbitals
  //norbs *=2;
  
  Determinant::norbs = norbs; //spin orbitals
  HalfDet::norbs = norbs; //spin orbitals
  Determinant::EffDetLen = norbs/64+1;
  Determinant::initLexicalOrder(nelec);
  if (Determinant::EffDetLen >DetLen) {
    pout << "change DetLen in global.h to "<<Determinant::EffDetLen<<" and recompile "<<endl;
    exit(0);
  }


  //initialize the heatbath integral
  std::vector<int> allorbs;
  //for (int i=0; i<norbs/2; i++)
  for (int i=0; i<norbs; i++)
    allorbs.push_back(i);
  twoIntHeatBath I2HB(1.e-10);
  twoIntHeatBathSHM I2HBSHM(1.e-10);
  //if (commrank == 0) I2HB.constructClass(allorbs, I2, I1, norbs/2);
  //I2HBSHM.constructClass(norbs/2, I2HB);
  if (commrank == 0) I2HB.constructClass(allorbs, I2, I1, norbs);
  I2HBSHM.constructClass(norbs, I2HB);
  int num_thrds;

  //IF SOC is true then read the SOC integrals
#ifndef Complex
  if (schd.doSOC) {
    pout << "doSOC option only works with the complex coefficients. Uncomment the -Dcomplex in the make file and recompile."<<endl;
    exit(0);
  }
#else
  if (schd.doSOC) {
    readSOCIntegrals(I1, norbs, "SOC");
#ifndef SERIAL
    mpi::broadcast(world, I1, 0);
#endif
  }
#endif
  
  //unlink the integral shared memory
  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());


  //have the dets, ci coefficient and diagnoal on all processors
  vector<MatrixXx> ci(schd.nroots, MatrixXx::Zero(HFoccupied.size(),1));
  vector<MatrixXx> vdVector(schd.nroots); //these vectors are used to calculate the response equations
  double Psi1Norm = 0.0;

  //make HF determinant
  vector<Determinant> Dets(HFoccupied.size());
  for (int d=0;d<HFoccupied.size(); d++) {
    for (int i=0; i<HFoccupied[d].size(); i++) {
      if (Dets[d].getocc(HFoccupied[d][i])) {
	      pout << "orbital "<<HFoccupied[d][i]<<" appears twice in input determinant number "<<d<<endl;
	      exit(0);
      }
      Dets[d].setocc(HFoccupied[d][i], true);
    }
    if (Determinant::Trev != 0)
      Dets[d].makeStandard();
    for (int i=0; i<d; i++) {
      if (Dets[d] == Dets[i]) {
	      pout << "Determinant "<<Dets[d]<<" appears twice in the input determinant list."<<endl;
	      exit(0);
      }
    }
  }
  schd.HF=Dets[0];

  if (commrank == 0) {
    for (int j=0; j<ci[0].rows(); j++)
      ci[0](j,0) = 1.0;
    ci[0] = ci[0]/ci[0].norm();
  }

#ifndef SERIAL
  mpi::broadcast(world, ci, 0);
#endif
  pout << "HF Energy: " << endl;
  pout << Dets[0].Energy(I1, I2, coreE) << endl;
  vector<double> E0 = SHCIbasics::DoVariational(ci, Dets, schd, I2, I2HBSHM, irrep, I1, coreE, nelec, schd.DoRDM);

  Determinant* SHMDets;
  SHMVecFromVecs(Dets, SHMDets, shciDetsCI, DetsCISegment, regionDetsCI);
  int DetsSize = Dets.size();
#ifndef SERIAL
  mpi::broadcast(world, DetsSize, 0);
#endif
  Dets.clear();

  if (commrank == 0) {
    std::string efile;
    efile = str(boost::format("%s%s") % schd.prefix[0].c_str() % "/shci.e" );
    FILE* f = fopen(efile.c_str(), "wb");
    for(int j=0;j<E0.size();++j) {
      pout << "Writing energy " << E0[j] << "  to file: " << efile << endl;
      fwrite( &E0[j], 1, sizeof(CItype), f);
    }
    fclose(f);
  }

    //print the 5 most important determinants and their weights
  pout << "Printing most important determinants"<<endl;
  pout << format("%4s %10s  ") %("Det") %("weight"); pout << "Determinant string"<<endl;
  for (int root=0; root<schd.nroots; root++) {
    pout << "State :"<<root<<endl;
    MatrixXx prevci = 1.*ci[root];
    int num = max(6, schd.printBestDeterminants);
    for (int i=0; i<min(num, static_cast<int>(DetsSize)); i++) {
      compAbs comp;
      int m = distance(&prevci(0,0), max_element(&prevci(0,0), &prevci(0,0) + prevci.rows(), comp));
#ifdef Complex
      pout << format("%4i %12.4e %12.4e  ") %(i) %(prevci(m,0).real()) %(prevci(m,0).imag()); pout << SHMDets[m]<<endl;
#else
      pout << format("%4i %18.10e ") %(i) %(prevci(m,0)); pout << SHMDets[m]<<endl;
#endif
      prevci(m,0) = 0.0;
    }
  }
  exit(0);
  pout << "### PERFORMING PERTURBATIVE CALCULATION"<<endl;

  if (schd.stochastic == true && schd.DoRDM) {
    schd.DoRDM = false;
    pout << "We cannot perform PT RDM with stochastic PT. Disabling RDM."<<endl;
  }


  /*
  if (schd.quasiQ) {
    double bkpepsilon2 = schd.epsilon2;
    schd.epsilon2 = schd.quasiQEpsilon;
    for (int root=0; root<schd.nroots;root++) {
      E0[root] += SHCIbasics::DoPerturbativeDeterministic(SHMDets, ci[root], E0[root], I1, I2, I2HBSHM, irrep, schd, coreE, nelec, root, vdVector, Psi1Norm, true);
      ci[root] = ci[root]/ci[root].norm();
    }
    schd.epsilon2 = bkpepsilon2;
  }
  */

#ifndef SERIAL
  world.barrier();
#endif

  vector<MatrixXx> spinRDM(3, MatrixXx::Zero(norbs, norbs));
#ifdef Complex
  if (schd.doSOC) {
    for (int j=0; j<E0.size(); j++)
      pout << str(boost::format("State: %3d,  E: %17.9f, dE: %10.2f\n")%j %(E0[j]) %( (E0[j]-E0[0])*219470));

    //dont do this here, if perturbation theory is switched on
    if (schd.doGtensor)  {
      SOChelper::calculateSpinRDM(spinRDM, ci[0], ci[1], Dets, norbs, nelec);
      pout << "VARIATIONAL G-TENSOR"<<endl;
      SOChelper::doGTensor(ci, Dets, E0, norbs, nelec, spinRDM);
    }
  }
#endif

  pout << endl;
  pout << "**************************************************************"<<endl;
  pout << "PERTURBATION THEORY STEP  "<<endl;
  pout << "**************************************************************"<<endl;


  if (schd.doSOC && !schd.stochastic) { //deterministic SOC calculation
    pout << "About to perform Perturbation theory"<<endl;
    if (schd.doGtensor) {
      pout << "Gtensor calculation not supported with deterministic PT for more than 2 roots."<<endl;
      pout << "Just performing the ZFS calculations."<<endl;
    }
    MatrixXx Heff = MatrixXx::Zero(E0.size(), E0.size());

    /*
    for (int root1 =0 ;root1<schd.nroots; root1++) {
      for (int root2=root1+1 ;root2<schd.nroots; root2++) {
	Heff(root1, root1) = 0.0; Heff(root2, root2) = 0.0; Heff(root1, root2) = 0.0;
	SHCIbasics::DoPerturbativeDeterministicOffdiagonal(Dets, ci[root1], E0[root1], ci[root2],
							   E0[root2], I1,
							   I2, I2HBSHM, irrep, schd,
							   coreE, nelec, root1, Heff(root1,root1),
							   Heff(root2, root2), Heff(root1, root2),
							   spinRDM);
	#ifdef Complex
  Heff(root2, root1) = conj(Heff(root1, root2));
#else
  Heff(root2, root1) = Heff(root1, root2);
#endif
      }
    }
    */
    for (int root1 =0 ;root1<schd.nroots; root1++)
      Heff(root1, root1) += E0[root1];

    SelfAdjointEigenSolver<MatrixXx> eigensolver(Heff);
    for (int j=0; j<eigensolver.eigenvalues().rows(); j++) {
      E0[j] = eigensolver.eigenvalues()(j,0);
      pout << str(boost::format("State: %3d,  E: %17.9f, dE: %10.2f\n")%j %(eigensolver.eigenvalues()(j,0)) %( (eigensolver.eigenvalues()(j,0)-eigensolver.eigenvalues()(0,0))*219470));
    }

    std::string efile;
    efile = str(boost::format("%s%s") % schd.prefix[0].c_str() % "/shci.e" );
    FILE* f = fopen(efile.c_str(), "wb");
    for(int j=0;j<E0.size();++j) {
      fwrite( &E0[j], 1, sizeof(double), f);
    }
    fclose(f);

//#ifdef Complex
//    SOChelper::doGTensor(ci, Dets, E0, norbs, nelec, spinRDM);
//    return 0;
//#endif
  }
  else if (schd.doLCC) {
    for (int root = 0; root<schd.nroots; root++) {
      CItype *ciroot;
      SHMVecFromMatrix(ci[root], ciroot, shcicMax, cMaxSegment, regioncMax);
      //LCC::doLCC(SHMDets, ciroot, DetsSize, E0[root], I1, I2, 
      //           I2HBSHM, irrep, schd, coreE, nelec, root);
    }    
  }
  else if (!schd.stochastic && schd.nblocks == 1) {
    double ePT = 0.0;
    std::string efile;
    efile = str(boost::format("%s%s") % schd.prefix[0].c_str() % "/shci.e" );
    FILE* f = fopen(efile.c_str(), "wb");
    for (int root=0; root<schd.nroots;root++) {
      CItype *ciroot;
      SHMVecFromMatrix(ci[root], ciroot, shcicMax, cMaxSegment, regioncMax);
      ePT = SHCIbasics::DoPerturbativeDeterministic(SHMDets, ciroot, DetsSize, E0[root], I1, I2,
						    I2HBSHM, irrep, schd, coreE, nelec,
						    root, vdVector, Psi1Norm);
      ePT += E0[root];
      pout << "Writing energy " << ePT << "  to file: " << efile << endl;
      if (commrank == 0) fwrite( &ePT, 1, sizeof(double), f);
    }
    fclose(f);

  }
  else if (schd.SampleN != -1 && schd.singleList){
    vector<double> ePT (schd.nroots,0.0);
    std::string efile;
    efile = str(boost::format("%s%s") % schd.prefix[0].c_str() % "/shci.e" );
    FILE* f = fopen(efile.c_str(), "wb");
    for (int root=0; root<schd.nroots;root++) {
      CItype *ciroot;
      SHMVecFromMatrix(ci[root], ciroot, shcicMax, cMaxSegment, regioncMax);
      ePT[root] = SHCIbasics::DoPerturbativeStochastic2SingleListDoubleEpsilon2AllTogether(SHMDets, ciroot, DetsSize, E0[root], I1, I2, I2HBSHM, irrep, schd, coreE, nelec, root);
      ePT[root] += E0[root];
      //pout << "Writing energy "<<E0[root]<<"  to file: "<<efile<<endl;
      if (commrank == 0) fwrite( &ePT[root], 1, sizeof(double), f);
    }
    fclose(f);

    if (schd.doSOC) {
      for (int j=0; j<E0.size(); j++)
	      pout << str(boost::format("State: %3d,  E: %17.9f, dE: %10.2f\n")%j %(ePT[j]) %( (ePT[j]-ePT[0])*219470));
    }
  }
  else {
#ifndef SERIAL
    world.barrier();
#endif
    pout << "Error here"<<endl;
    exit(0);
  }


  //THIS IS USED FOR RDM CALCULATION FOR DETERMINISTIC PT
  if ((schd.doResponse || schd.DoRDM) && 
      schd.RdmType == RELAXED &&
      (!schd.stochastic && schd.nblocks==1)) {

    if (schd.DavidsonType == DIRECT) {
      pout << "PT RDM not implemented with direct davidson."<<endl;
      exit(0);
    }

    std::vector<MatrixXx> lambda(schd.nroots, MatrixXx::Zero(Dets.size(),1));
    SHCImake4cHamiltonian::SparseHam sparseHam;
    {
      char file [5000];
      sprintf (file, "%s/%d-hamiltonian.bkp" , schd.prefix[0].c_str(), commrank );
      std::ifstream ifs(file, std::ios::binary);
      boost::archive::binary_iarchive load(ifs);
      load >> sparseHam.connections >> sparseHam.Helements >> sparseHam.orbDifference;
    }

    vector<CItype*> ciroot(schd.nroots);

    for (int iroot = 0; iroot < schd.nroots; iroot++) {
      SHMVecFromMatrix(ci[iroot], ciroot[iroot], shcicMax, cMaxSegment, regioncMax);
      Hmult2 H(sparseHam);
      //LinearSolver(H, E0[iroot], lambda[iroot], vdVector[iroot], ciroot, 1.e-5, false);
#ifndef SERIAL
      //mpi::broadcast(world, lambda[i], 0);
#endif
      MatrixXx s2RDM, twoRDM;
      s2RDM.setZero(norbs*norbs/4, norbs*norbs/4);
      if (schd.DoSpinRDM) twoRDM.setZero(norbs*norbs, norbs*norbs);
    
      if (schd.DoOneRDM) {
        MatrixXx s1RDM, oneRDM;
	      oneRDM = MatrixXx::Zero(norbs,norbs);
	      s1RDM = MatrixXx::Zero(norbs/2, norbs/2);
        SHCIrdm::EvaluateOneRDM(sparseHam.connections, SHMDets, DetsSize, ciroot[iroot], ciroot[iroot], 
			  sparseHam.orbDifference, nelec, schd, iroot, oneRDM, s1RDM);
        SHCIrdm::save1RDM(schd, s1RDM, oneRDM, iroot);
      }
    // Add DoOneRDM Block
    
    
      if (commrank == 0) {
        MatrixXx s2RDMdisk, twoRDMdisk;
        SHCIrdm::loadRDM(schd, s2RDMdisk, twoRDMdisk, iroot);
        s2RDMdisk = s2RDMdisk + s2RDM.adjoint() + s2RDM;
        SHCIrdm::saveRDM(schd, s2RDMdisk, twoRDMdisk, iroot);
        SHCIrdm::ComputeEnergyFromSpinRDM(norbs, nelec, I1, I2, coreE, twoRDMdisk);
      }
    }
    //pout <<" response ";
  }


  if (schd.extrapolate) {//performing extrapolation
    if (schd.nroots > 1 ) {
      cout <<" extrapolation only supported for single root "<<endl;
      exit(0);
    }
    for (int root = 0; root <schd.nroots; root++) {
      
      vector<double> var(4, 0.0), PT(4, 0.0);
      vector<int> nDets(4,0);
      nDets[0] = DetsSize;
      var[0] = E0[0]; 
      std::string efile;
      efile = str(boost::format("%s%s") % schd.prefix[0].c_str() % "/shci.e" );
      FILE* f = fopen(efile.c_str(), "rb");
      if (commrank == 0) fread( &PT[0], 1, sizeof(double), f);
#ifndef SERIAL
      mpi::broadcast(world, PT, 0);
#endif
      PT[0] -= var[0];
      
      //do 4 iterations for extrapolation
      for (int iter = 0; iter<3; iter++) {
	
	if (commrank == 0) {
	  char file [5000];
	  sprintf (file, "%s/%d-variational.bkp" , schd.prefix[0].c_str(),
		   commrank );
	  std::ifstream ifs(file, std::ios::binary);
	  boost::archive::binary_iarchive load(ifs);
	  ci.clear(); Dets.clear();
	  int niter;
	  load >> niter >> Dets;
	  load >> ci;
	  if (iter == 0) 
	    nDets[0] = Dets.size();
	  DetsSize = Dets.size();
	}
	SHMVecFromVecs(Dets, SHMDets, shciDetsCI, DetsCISegment, regionDetsCI);
	if (commrank == 0) {
	  std::vector<size_t> indices(DetsSize);
	  for (int i=0; i<DetsSize;i++) indices[i] = i;

	  sort(indices.begin(), indices.end(),
	       [&ci](size_t i1, size_t i2) {return abs(ci[0](i1,0)) > abs(ci[0](i2,0));});

	  DetsSize = DetsSize*schd.extrapolationFactor;
	  Dets.resize(DetsSize);
	  MatrixXx cicopy = MatrixXx::Zero(DetsSize,1);
	  for (size_t i=0; i<DetsSize; i++)  {
	    Dets[i] = SHMDets[indices[i]];
	    cicopy(i, 0)  = ci[root](indices[i],0);
	  }
	  ci[root].resize(DetsSize,1);
	  for (size_t i=0; i<DetsSize; i++) {
	    ci[root](i, 0)  = cicopy(i,0);
	  }
	  ci[root] = ci[root]/ci[root].norm();
	}

#ifndef SERIAL
	mpi::broadcast(world, DetsSize, 0);
#endif
	nDets[iter+1] = DetsSize;
	schd.epsilon1.resize(1); schd.epsilon1[0] = 1.e10; //very large
	schd.restart = false; schd.fullrestart = false;
	schd.DoRDM = false;
	E0 = SHCIbasics::DoVariational(ci, Dets, schd, I2, I2HBSHM, irrep, I1, coreE, nelec, false);
	var[iter+1] = E0[0];

	DetsSize = Dets.size();
#ifndef SERIAL
	mpi::broadcast(world, DetsSize, 0);
#endif
	SHMVecFromVecs(Dets, SHMDets, shciDetsCI, DetsCISegment, regionDetsCI);
	Dets.clear();
	
	CItype *ciroot;
	SHMVecFromMatrix(ci[root], ciroot, shcicMax, cMaxSegment, regioncMax);
	if (!schd.stochastic)
	  PT[iter+1] = SHCIbasics::DoPerturbativeDeterministic(SHMDets, ciroot, DetsSize, E0[root], I1,I2,I2HBSHM, irrep, schd, coreE, nelec, root, vdVector, Psi1Norm);
	else
	  PT[iter+1] = SHCIbasics::DoPerturbativeStochastic2SingleListDoubleEpsilon2AllTogether(SHMDets, ciroot, DetsSize, E0[root], I1, I2,I2HBSHM, irrep, schd, coreE, nelec, root);
	
      }
	
      if (commrank == 0) printf("Ndet         Evar                  Ept               \n");
      for (int iter=0; iter<4; iter++)
	if (commrank == 0) printf("%10i   %18.10g    %18.10g \n", nDets[iter], var[iter], PT[iter]);

    }
    
	
  }
#ifndef SERIAL
  world.barrier();
#endif

  std::system("rm -rf /dev/shm* 2>/dev/null");

  return 0;
}
