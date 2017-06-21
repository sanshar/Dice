/*
   Developed by Sandeep Sharma with contributions from James E. Smith and Adam A. Homes, 2017
   Copyright (c) 2017, Sandeep Sharma

   This file is part of DICE.
   This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "omp.h"
#include "global.h"
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include "Determinants.h"
#include "SHCImakeHamiltonian.h"
#include "SHCIgetdeterminants.h"
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

#include "symmetry.h"
MatrixXd symmetry::product_table;
#include <algorithm>
#include <boost/bind.hpp>

using namespace Eigen;
using namespace boost;
int HalfDet::norbs = 1; //spin orbitals
int Determinant::norbs = 1; //spin orbitals
int Determinant::EffDetLen = 1;
Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> Determinant::LexicalOrder;
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
	license();
	string inputFile = "input.dat";
	if (argc > 1)
		inputFile = string(argv[1]);
	//Read the input file
	std::vector<std::vector<int> > HFoccupied;
	schedule schd;
	if (mpigetrank() == 0) readInput(inputFile, HFoccupied, schd);
#ifndef SERIAL
	mpi::broadcast(world, HFoccupied, 0);
	mpi::broadcast(world, schd, 0);
#endif
	omp_set_num_threads(schd.num_thrds);


	//set the random seed
	startofCalc=getTime();
	srand(schd.randomSeed+mpigetrank());
	if (schd.outputlevel>1) pout<<"#using seed: "<<schd.randomSeed<<endl;



	//set up shared memory files to store the integrals
	string shciint2 = "SHCIint2" + to_string(static_cast<long long>(time(NULL) % 1000000));
	string shciint2shm = "SHCIint2shm" + to_string(static_cast<long long>(time(NULL) % 1000000));
	int2Segment = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, shciint2.c_str(), boost::interprocess::read_write);
	int2SHMSegment = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, shciint2shm.c_str(), boost::interprocess::read_write);




	std::cout.precision(15);

	//read the hamiltonian (integrals, orbital irreps, num-electron etc.)
	twoInt I2; oneInt I1; int nelec; int norbs; double coreE=0.0, eps;
	std::vector<int> irrep;
	readIntegrals(schd.integralFile, I2, I1, nelec, norbs, coreE, irrep);

	if (HFoccupied[0].size() != nelec) {
		pout << "The number of electrons given in the FCIDUMP should be equal to the nocc given in the shci input file."<<endl;
		exit(0);
	}


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


	//initialize the heatbath integral
	std::vector<int> allorbs;
	for (int i=0; i<norbs/2; i++)
		allorbs.push_back(i);
	twoIntHeatBath I2HB(1.e-10);
	twoIntHeatBathSHM I2HBSHM(1.e-10);
	if (mpigetrank() == 0) I2HB.constructClass(allorbs, I2, norbs/2);
	I2HBSHM.constructClass(norbs/2, I2HB);

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


	//have the dets, ci coefficient and diagnoal on all processors
	vector<MatrixXx> ci(schd.nroots, MatrixXx::Zero(HFoccupied.size(),1));
	vector<MatrixXx> vdVector(schd.nroots); //these vectors are used to calculate the response equations
	double Psi1Norm = 0.0;

	//make HF determinant
	vector<Determinant> Dets(HFoccupied.size());
	for (int d=0; d<HFoccupied.size(); d++) {

		for (int i=0; i<HFoccupied[d].size(); i++) {
			Dets[d].setocc(HFoccupied[d][i], true);
		}
	};

	pout << "**************************************************************\n";
	pout << "SELECTING REFERENCE DETERMINANT(S)\n";
	pout << "**************************************************************\n";
	pout << Dets[0] << " Give HF Energy: " << Dets.at(0).Energy(I1,I2,coreE) << "\n\n";

// #ifndef Complex
	symmetry molSym ( schd.pointGroup );
	vector<Determinant> tempDets ( Dets );
	for ( int d=0; d < HFoccupied.size(); d++ ) {
		// Guess the lowest energy det with given symmetry from one body integrals.
		molSym.estimateLowestEnergyDet(schd.spin, schd.irrep, I1, irrep,
		  HFoccupied.at(d), tempDets.at(d));
		cout << tempDets[d] << " Est. Det. Energy: " << tempDets.at(d).Energy(I1,I2,coreE) << "\n\n"; // TODO

		// Generate list of connected determinants to guess determinant.
		SHCIgetdeterminants::getDeterminantsVariational(tempDets.at(d), 0.00001, 1, 0.0,
		  I1, I2, I2HBSHM, irrep, coreE, 0, tempDets, schd, 0, nelec );

		// Check all connected and find lowest energy.
		int counter = 0;
		for ( int cd = 0; cd < tempDets.size(); cd++ ) {
			// cout << Dets.at(d).Energy(I1,I2,coreE) << endl;
			cout << tempDets.at(cd) << " ";
			cout << tempDets.at(cd).Energy(I1,I2,coreE) << endl;
			if ( abs(tempDets.at(cd).Nalpha() - tempDets.at(cd).Nbeta()) == schd.spin ) {
				// cout << tempDets.at(cd) << " Energy: " << tempDets.at(cd).Energy(I1,I2,coreE) << endl;
				counter++;
				// if ( Dets.at(d).Energy(I1,I2,coreE) >
				//   tempDets.at(cd).Energy(I1,I2,coreE) ) {
				//  Dets.at(d) = tempDets.at(cd);
				// }
			}
		}
		cout << counter << " " << tempDets.size() << endl;
		cout << "Predicted lowest energy det: " << Dets[d] << endl << endl;
	}

// #endif

	if (mpigetrank() == 0) {
		for (int j=0; j<ci[0].rows(); j++)
			ci[0](j,0) = 1.0;
		ci[0] = ci[0]/ci[0].norm();
	}

#ifndef SERIAL
	mpi::broadcast(world, ci, 0);
#endif


	vector<double> E0 = SHCIbasics::DoVariational(ci, Dets, schd, I2, I2HBSHM, irrep, I1, coreE, nelec, schd.DoRDM);


	if (mpigetrank() == 0) {
		std::string efile;
		efile = str(boost::format("%s%s") % schd.prefix[0].c_str() % "/shci.e" );
		FILE* f = fopen(efile.c_str(), "wb");
		for(int j=0; j<E0.size(); ++j) {
			//pout << "Writing energy "<<E0[j]<<"  to file: "<<efile<<endl;
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
		for (int i=0; i<6; i++) {
			compAbs comp;
			int m = distance(&prevci(0,0), max_element(&prevci(0,0), &prevci(0,0)+prevci.rows(), comp));
			pout << format("%4i %10.2e  ") %(i) %(abs(prevci(m,0))); pout << Dets[m]<<endl;
			//pout <<"#"<< i<<"  "<<prevci(m,0)<<"  "<<abs(prevci(m,0))<<"  "<<Dets[m]<<endl;
			prevci(m,0) = 0.0;
		}
	}
	//pout << "### PERFORMING PERTURBATIVE CALCULATION"<<endl;
	if (schd.stochastic == true && schd.DoRDM) {
		schd.DoRDM = false;
		pout << "We cannot perform PT RDM with stochastic PT. Disabling RDM."<<endl;
	}


	if (schd.quasiQ) {
		double bkpepsilon2 = schd.epsilon2;
		schd.epsilon2 = schd.quasiQEpsilon;
		for (int root=0; root<schd.nroots; root++) {
			E0[root] += SHCIbasics::DoPerturbativeDeterministic(Dets, ci[root], E0[root], I1, I2, I2HBSHM, irrep, schd, coreE, nelec, root, vdVector, Psi1Norm, true);
			ci[root] = ci[root]/ci[root].norm();
		}
		schd.epsilon2 = bkpepsilon2;
	}

#ifndef SERIAL
	world.barrier();
#endif
	boost::interprocess::shared_memory_object::remove(shciint2.c_str());

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

		for (int root1 =0; root1<schd.nroots; root1++) {
			for (int root2=root1+1; root2<schd.nroots; root2++) {
				Heff(root1, root1) = 0.0; Heff(root2, root2) = 0.0; Heff(root1, root2) = 0.0;
				SHCIbasics::DoPerturbativeDeterministicOffdiagonal(Dets, ci[root1], E0[root1], ci[root2],
				  E0[root2], I1,
				  I2, I2HBSHM, irrep, schd,
				  coreE, nelec, root1, Heff(root1,root1),
				  Heff(root2, root2), Heff(root1, root2),
				  spinRDM);
				Heff(root2, root1) = conj(Heff(root1, root2));
			}
		}
		for (int root1 =0; root1<schd.nroots; root1++)
			Heff(root1, root1) += E0[root1];

		SelfAdjointEigenSolver<MatrixXx> eigensolver(Heff);
		for (int j=0; j<eigensolver.eigenvalues().rows(); j++) {
			E0[j] = eigensolver.eigenvalues() (j,0);
			pout << str(boost::format("State: %3d,  E: %17.9f, dE: %10.2f\n")%j %(eigensolver.eigenvalues() (j,0)) %( (eigensolver.eigenvalues() (j,0)-eigensolver.eigenvalues() (0,0))*219470));
		}

		std::string efile;
		efile = str(boost::format("%s%s") % schd.prefix[0].c_str() % "/shci.e" );
		FILE* f = fopen(efile.c_str(), "wb");
		for(int j=0; j<E0.size(); ++j) {
			fwrite( &E0[j], 1, sizeof(double), f);
		}
		fclose(f);

#ifdef Complex
		SOChelper::doGTensor(ci, Dets, E0, norbs, nelec, spinRDM);
		return 0;
#endif
	}

	else if (!schd.stochastic && schd.nblocks == 1) {
		double ePT = 0.0;
		std::string efile;
		efile = str(boost::format("%s%s") % schd.prefix[0].c_str() % "/shci.e" );
		FILE* f = fopen(efile.c_str(), "wb");
		for (int root=0; root<schd.nroots; root++) {
			ePT = SHCIbasics::DoPerturbativeDeterministic(Dets, ci[root], E0[root], I1, I2,
			  I2HBSHM, irrep, schd, coreE, nelec,
			  root, vdVector, Psi1Norm);
			ePT += E0[root];
			//pout << "Writing energy "<<ePT<<"  to file: "<<efile<<endl;
			if (mpigetrank() == 0) fwrite( &ePT, 1, sizeof(double), f);
		}
		fclose(f);

	}
	else if (schd.SampleN != -1 && schd.singleList) {
		double ePT = 0.0;
		std::string efile;
		efile = str(boost::format("%s%s") % schd.prefix[0].c_str() % "/shci.e" );
		FILE* f = fopen(efile.c_str(), "wb");
		for (int root=0; root<schd.nroots; root++) {
			ePT = SHCIbasics::DoPerturbativeStochastic2SingleListDoubleEpsilon2AllTogether(Dets, ci[root], E0[root], I1, I2, I2HBSHM, irrep, schd, coreE, nelec, root);
			E0[root] += ePT;
			//pout << "Writing energy "<<E0[root]<<"  to file: "<<efile<<endl;
			if (mpigetrank() == 0) fwrite( &E0[root], 1, sizeof(double), f);
		}
		fclose(f);

		if (schd.doSOC) {
			for (int j=0; j<E0.size(); j++)
				pout << str(boost::format("State: %3d,  E: %17.9f, dE: %10.2f\n")%j %(E0[j]) %( (E0[j]-E0[0])*219470));
		}
	}
	else {
#ifndef SERIAL
		world.barrier();
#endif
		boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());
		pout << "Error here"<<endl;
		exit(0);
	}


	//THIS IS USED FOR RDM CALCULATION FOR DETERMINISTIC PT
	if ((schd.doResponse || schd.DoRDM) && (!schd.stochastic && schd.nblocks==1)) {
		std::vector<MatrixXx> lambda(schd.nroots, MatrixXx::Zero(Dets.size(),1));
		std::vector<std::vector<int> > connections;
		std::vector<std::vector<CItype> > Helements;
		std::vector<std::vector<size_t> > orbDifference;
		{
			char file [5000];
			sprintf (file, "%s/%d-hamiltonian.bkp", schd.prefix[0].c_str(), mpigetrank() );
			std::ifstream ifs(file, std::ios::binary);
			boost::archive::binary_iarchive load(ifs);
			load >> connections >> Helements >> orbDifference;
		}


		Hmult2 H(connections, Helements);
		LinearSolver(H, E0[0], lambda[0], vdVector[0], ci, 1.e-5, false);
#ifndef SERIAL
		mpi::broadcast(world, lambda[0], 0);
#endif

		MatrixXx s2RDM, twoRDM;
		s2RDM.setZero(norbs*norbs/4, norbs*norbs/4);
		if (schd.DoSpinRDM) twoRDM.setZero(norbs*(norbs+1)/2, norbs*(norbs+1)/2);
		SHCIrdm::EvaluateRDM(connections, Dets, lambda[0], ci[0], orbDifference, nelec, schd, 0, twoRDM, s2RDM);

		if (mpigetrank() == 0) {
			MatrixXx s2RDMdisk, twoRDMdisk;
			SHCIrdm::loadRDM(schd, s2RDMdisk, twoRDMdisk, 0);
			s2RDMdisk = s2RDMdisk + s2RDM.adjoint() + s2RDM;
			SHCIrdm::saveRDM(schd, s2RDMdisk, twoRDMdisk, 0);
		}
		//pout <<" response ";
		//SHCIrdm::ComputeEnergyFromSpatialRDM(norbs, nelec, I1, I2, coreE, s2RDM);
	}

#ifndef SERIAL
	world.barrier();
#endif
	boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());

	return 0;
}
