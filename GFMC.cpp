/*
  Developed by Sandeep Sharma 
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
#include <algorithm>
#include <random>
#include <chrono>
#include <stdlib.h>
#include <stdio.h>
#include <list>
#include <utility> 
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#ifndef SERIAL
//#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "evaluateE.h"
#include "MoDeterminants.h"
#include "Determinants.h"
#include "Walker.h"
#include "CPS.h"
#include "Wfn.h"
#include "input.h"
#include "integral.h"
#include "SHCIshm.h"
#include "math.h"
#include "diis.h"
#include "optimizer.h"

using namespace Eigen;
using namespace boost;
using namespace std;

void doGFMC(CPSSlater &w, double Eshift, oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, double coreE);
void doGFMCCT(CPSSlater &w, double Eshift, oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, double coreE);

int main(int argc, char* argv[]) {

#ifndef SERIAL
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
#endif
  startofCalc = getTime();

  initSHM();
  //license();

  string inputFile = "input.dat";
  if (argc > 1)
    inputFile = string(argv[1]);
  if (commrank == 0) readInput(inputFile, schd, false);
#ifndef SERIAL
  mpi::broadcast(world, schd, 0);
#endif

  generator = std::mt19937(schd.seed+commrank);
  //generator = std::mt19937(commrank);

  twoInt I2; oneInt I1; 
  int norbs, nalpha, nbeta; 
  double coreE=0.0;
  std::vector<int> irrep;
  readIntegrals("FCIDUMP", I2, I1, nalpha, nbeta, norbs, coreE, irrep);

  //initialize the heatbath integrals
  std::vector<int> allorbs;
  for (int i=0; i<norbs; i++)
    allorbs.push_back(i);
  twoIntHeatBath I2HB(1.e-10);
  twoIntHeatBathSHM I2HBSHM(1.e-10);
  if (commrank == 0) I2HB.constructClass(allorbs, I2, I1, norbs);
  I2HBSHM.constructClass(norbs, I2HB);
  
  //Setup static variables
  Determinant::EffDetLen = (norbs)/64+1;
  Determinant::norbs    = norbs;
  MoDeterminant::norbs  = norbs;
  MoDeterminant::nalpha = nalpha;
  MoDeterminant::nbeta  = nbeta;

  //Setup Slater Determinants
  HforbsA = MatrixXd::Zero(norbs, norbs);
  HforbsB = MatrixXd::Zero(norbs, norbs);
  readHF(HforbsA, HforbsB, schd.uhf);


  //Setup CPS wavefunctions
  std::vector<Correlator> nSiteCPS;
  for (auto it = schd.correlatorFiles.begin(); it != schd.correlatorFiles.end();
       it++) {
    readCorrelator(it->second, it->first, nSiteCPS);
  }
  
  vector<Determinant> detList; vector<double> ciExpansion;

  if (boost::iequals(schd.determinantFile, "") )
  {
    detList.resize(1); ciExpansion.resize(1, 1.0);
    for (int i=0; i<nalpha; i++)
      detList[0].setoccA(i, true);
    for (int i=0; i<nbeta; i++)
      detList[0].setoccB(i, true);
  }
  else 
  {
    readDeterminants(schd.determinantFile, detList, ciExpansion);
  }
  //setup up wavefunction
  CPSSlater wave(nSiteCPS, detList, ciExpansion);

  ifstream file ("params.bin", ios::in|ios::binary|ios::ate);
  size_t size = file.tellg();
  Eigen::VectorXd vars = Eigen::VectorXd::Zero(size/sizeof(double));
  file.seekg (0, ios::beg);
  file.read ( (char*)(&vars[0]), size);
  file.close();

  if ( (schd.uhf && vars.size() != wave.getNumVariables()+2*norbs*norbs) ||
       (!schd.uhf && vars.size() != wave.getNumVariables()+norbs*norbs) ){
    cout << "number of variables on disk: "<<vars.size()<<" is not equal to wfn parameters: "<<wave.getNumVariables()<<endl;
    exit(0);
  }

  wave.updateVariables(vars);
  int numVars = wave.getNumVariables();

  for (int i=0; i<norbs; i++) {
    for (int j=0; j<norbs; j++) {
      if (!schd.uhf) {
    HforbsA(i,j) = vars[numVars + i *norbs + j];
    HforbsB(i,j) = vars[numVars + i *norbs + j];
      }
      else {
    HforbsA(i,j) = vars[numVars + i *norbs + j];
    HforbsB(i,j) = vars[numVars + norbs*norbs + i *norbs + j];
      }
    }
  }

  MatrixXd alpha(norbs, nalpha), beta(norbs, nbeta);
  alpha = HforbsA.block(0, 0, norbs, nalpha);
  beta  = HforbsB.block(0, 0, norbs, nbeta );
  MoDeterminant det(alpha, beta);

  //Read the E0 energy
  
  double shift0 = 0;
  if (commrank == 0) {
    ifstream fileE ("E0.bin", ios::in|ios::binary);
    fileE.read ( (char*)(&shift0), sizeof(double));
    fileE.close();
  }
#ifndef SERIAL
  MPI_Bcast(&shift0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

  doGFMCCT(wave, shift0, I1, I2, I2HBSHM, coreE);
  //doGFMC(wave, shift0, I1, I2, I2HBSHM, coreE);

  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());
  return 0;
}


void generateWalkers(list<pair<Walker, double> >& Walkers, CPSSlater& w,
		     oneInt& I1, twoInt& I2, 
		     twoIntHeatBathSHM& I2hb, double coreE) {

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));
  
  VectorXd localGrad;

  //initialize the walker
  Determinant d;
  bool readDeterminant = false;
  char file [5000];

  sprintf (file, "BestDeterminant.txt");

  {
    ifstream ofile(file);
    if (ofile) readDeterminant = true;
  }
  //readDeterminant = false;

  if (readDeterminant) {
    if (commrank == 0) {
      std::ifstream ifs(file, std::ios::binary);
      boost::archive::binary_iarchive load(ifs);
      load >> d;
    }
#ifndef SERIAL
    MPI_Bcast(&d.reprA, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d.reprB, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  }
  
  Walker walk(d);
  walk.initUsingWave(w);
  
  //int maxTerms =  3*(nalpha) * (nbeta) * (norbs-nalpha) * (norbs-nbeta);
  int maxTerms =  1000000; //pick a small number that will be incremented later
  vector<double> ovlpRatio(maxTerms);
  vector<size_t> excitation1( maxTerms), excitation2( maxTerms);
  vector<double> HijElements(maxTerms);
  int nExcitations = 0;
  double ham, ovlp;

  int iter = 0;
  int numSample = Walkers.size();
  int niter = 30*numSample+1;
  auto it = Walkers.begin();
  while(iter < niter) {
    w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE, ovlpRatio,
			 excitation1, excitation2, HijElements, nExcitations, false);
    double cumovlpRatio = 0;
    //when using uniform probability 1./numConnection * max(1, pi/pj)
    for (int i = 0; i < nExcitations; i++)
    {
      cumovlpRatio += abs(ovlpRatio[i]);
      //cumovlpRatio += min(1.0, pow(ovlpRatio[i], 2));
      ovlpRatio[i] = cumovlpRatio;
    }

    if (iter % 30 == 0) {
      it->first = walk; it->second = 1.0/cumovlpRatio;
      it++;
      if (it == Walkers.end()) break;
    }

    
    double nextDetRandom = random()*cumovlpRatio;
    int nextDet = std::lower_bound(ovlpRatio.begin(), (ovlpRatio.begin()+nExcitations),
                                   nextDetRandom) -
      ovlpRatio.begin();
    walk.updateWalker(w, excitation1[nextDet], excitation2[nextDet]);
  }
}


void applyPropogator(Walker& walk, double& wt, 
		     CPSSlater& w, double tau, vector<double>& ovlpRatio, vector<size_t>& excitation1,
		     vector<size_t>& excitation2,
		     vector<double>& HijElements, vector<double>& cumHijElements, double& Eshift,
		     double& ham, double& ovlp, oneInt& I1, twoInt& I2, 
		     twoIntHeatBathSHM& I2hb, double coreE,
		     double fn_factor) {

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));
  double Ewalk = walk.d.Energy(I1, I2, coreE);
  
  int nExcitations = 0;
  VectorXd localGrad;
  w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE, ovlpRatio,
		       excitation1, excitation2, HijElements, nExcitations, false);

  if (cumHijElements.size()< nExcitations) cumHijElements.resize(nExcitations);

  double cumHij = 0;

  for (int i = 0; i < nExcitations; i++)
  {
    if (HijElements[i]*ovlpRatio[i] > 0.0) {
      Ewalk += HijElements[i]/ovlpRatio[i] * fn_factor;
      cumHij += (fn_factor-1) * abs(-tau*HijElements[i]*ovlpRatio[i]);
      cumHijElements[i] = cumHij; //the cumulative does not change and so wont be excited to
    }
    else {
      cumHij += abs(-tau*HijElements[i]*ovlpRatio[i]);
      cumHijElements[i] = cumHij;
    }
  }

  if (tau*(Ewalk - Eshift) > 1.0) {
    cout << "tau is too large"<<endl;
    cout << "IT should be atleast smaller than "<<1.0/(Ewalk-Eshift)<<endl;
    exit(0);
  }

  cumHij += (1.0 - tau*(Ewalk - Eshift));


  double nextDetRandom = random() * cumHij;
  wt *= cumHij;
  if (nextDetRandom > cumHijElements[nExcitations-1]) {
    return;
    //newWeights.push_back( cumHij * (wt/nwtsInt));
    //newWalkers.push_back(walk);
  }
  else {
    int nextDet = std::lower_bound(cumHijElements.begin(), (cumHijElements.begin() + nExcitations),
				   nextDetRandom) - cumHijElements.begin();

    walk.updateWalker(w, excitation1[nextDet], excitation2[nextDet]);
  }

}


void applyPropogatorContinousTime(Walker& walk, double& wt, 
				  CPSSlater& w, double& tau, vector<double>& ovlpRatio, vector<size_t>& excitation1,
				  vector<size_t>& excitation2,
				  vector<double>& HijElements, vector<double>& cumHijElements, double& Eshift,
				  double& ham, double& ovlp, oneInt& I1, twoInt& I2, 
				  twoIntHeatBathSHM& I2hb, double coreE,
				  double fn_factor) {

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));
  
  VectorXd localGrad;

  double t = tau;
  int index = 0;

  while (t > 0) {
    double Ewalk = walk.d.Energy(I1, I2, coreE);
    int nExcitations = 0;

    w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE, ovlpRatio,
			 excitation1, excitation2, HijElements, nExcitations, false);

    if (cumHijElements.size()< nExcitations) cumHijElements.resize(nExcitations);

    double cumHij = 0;

    for (int i = 0; i < nExcitations; i++)
    {
      //if sy,x is > 0 then add include contribution to the diagonal
      if (HijElements[i]*ovlpRatio[i] > 0.0) {
	Ewalk += HijElements[i]*ovlpRatio[i] * fn_factor;
	cumHij += (fn_factor-1.0)*abs(HijElements[i]*ovlpRatio[i]); //negatie of the contribution to local energy
	cumHijElements[i] = cumHij; 
      }
      else {
	cumHij += abs(HijElements[i]*ovlpRatio[i]); //negatie of the contribution to local energy
	cumHijElements[i] = cumHij;
      }
    }

    ham = Ewalk-cumHij;

    double tsample = -log(random())/cumHij;
    double deltaT = min(t, tsample);
    t -= tsample;

    wt = wt * exp(deltaT*(Eshift - (Ewalk-cumHij) ));

    if (t > 0.0)   {
      double nextDetRandom = random()*cumHij;
      int nextDet = std::lower_bound(cumHijElements.begin(), (cumHijElements.begin() + nExcitations),
				     nextDetRandom) - cumHijElements.begin();    
      walk.updateWalker(w, excitation1[nextDet], excitation2[nextDet]);

    }
    
  }
}

double reconfigure_splitjoin(list<pair<Walker,double> >& walkers, double targetwt, double tau, double Eest, double Eshift) {

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
			  std::ref(generator));

  /*
  auto it = walkers.begin();
  double newwt = 0.0;
  while (it != walkers.end()) {
    double wt = it->second;
    newwt += wt;
    it++;
  }

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &newwt, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  newwt /= commsize;

  double factor = pow((newwt/targetwt), schd.tau);

  for (it = walkers.begin(); it != walkers.end(); it++) { 
    it->second = it->second/factor;
  }
  /*/

  double factor= 1.0;

  auto prevSmall = walkers.begin();
  bool prevSmallAssigned = false;
  auto it = walkers.begin();

  while (it != walkers.end()) {
    double wt = it->second;

    if (wt < 0.5 && prevSmallAssigned == false) {
      prevSmall = it;
      prevSmallAssigned = true;
      it++;
    }
    else if (wt < 0.5 && prevSmallAssigned == true) {
      double oldwt = prevSmall->second;
      double rand = random()*( wt + oldwt);
      if (rand < oldwt) {
	it = walkers.erase(it);
	prevSmall->second = wt+oldwt;
      }
      else {
	walkers.erase(prevSmall);
	it->second = wt+oldwt;
	it++;
      }
      prevSmallAssigned = false;
    }
    else if (wt > 2.0) {
      int numReplicate = int(wt);
      it->second = it->second/(1.0*numReplicate);
      for (int i=0; i<numReplicate-1; i++) 
	walkers.push_front(*it); //add at front
      it++;
    }
    else
      it++;
  }
  int maxwalkers=walkers.size(), minwalkers=walkers.size();
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &maxwalkers, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &minwalkers, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
  //if (commrank == 0)
  //cout << minwalkers<<"  "<<maxwalkers<<endl;
  return factor;
}

double reconfigure_simple(list<pair<Walker,double> >& walkers, double targetwt, double tau, double Eest, double Eshift) {

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
			  std::ref(generator));


  auto it = walkers.begin();
  double sumwt = 0.0;
  int nwts = 0;
  while (it != walkers.end()) {
    double wt = it->second;
    sumwt += wt;
    it++;
    nwts++;
  }




  it = walkers.begin();

  while (it != walkers.end()) {
    double wt = it->second*nwts/sumwt;

    if (wt < 1.0 ) {
      if (wt+random() > 1.0) {
	wt = 1.0*sumwt/nwts;
	it++;
      }
      else 
	it = walkers.erase(it);
    }
    else if (wt > 2.0) {
      int numReplicate = int(wt);
      it->second = it->second/(1.0*numReplicate);
      for (int i=0; i<numReplicate-1; i++) 
	walkers.push_front(*it); //add at front
      it++;
    }
    else
      it++;
  }

  double factor = 1.0;
  return factor;
}


void doGFMCCT(CPSSlater &w, double Eshift, oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, double coreE)
{
  startofCalc = getTime();
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
			  std::ref(generator));
  
  //initialize the walker
  Determinant d;
  bool readDeterminant = false;
  char file[5000];
  
  sprintf(file, "BestDeterminant.txt");
  {
    ifstream ofile(file);
    if (ofile) readDeterminant = true;
  }
  
  if ( readDeterminant)
    {
      ifstream ofile(file);
      if (commrank == 0)
        {
	  std::ifstream ifs(file, std::ios::binary);
	  boost::archive::binary_iarchive load(ifs);
	  load >> d;
        }
#ifndef SERIAL
      MPI_Bcast(&d.reprA, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&d.reprB, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    }
  
  
  Walker walk(d);
  walk.initUsingWave(w);
  list<pair<Walker, double> > Walkers;
  for (int i=0; i<schd.nwalk; i++)
    Walkers.push_back(std::pair<Walker, double>(walk, 1.0));
  generateWalkers(Walkers, w, I1, I2, I2hb, coreE);

  int maxTerms = 1000000;
  vector<double> ovlpRatio(maxTerms);
  vector<size_t> excitation1(maxTerms), excitation2(maxTerms);
  vector<double> HijElements(maxTerms), cumHijElements(maxTerms);
  int nExcitations = 0;
  
  double stddev = 1.e4;
  int iter = 0;
  double M1 = 0., S1 = 0.;
  double Eloc = 0.;
  double ham = 0., ovlp = 0.;
  double scale = 1.0;
  int niter = schd.maxIter;
  double tau = schd.tau;
  
  double oldwt = .0;
  for (auto it=Walkers.begin(); it!=Walkers.end(); it++) oldwt += it->second;
  for (auto it=Walkers.begin(); it!=Walkers.end(); it++) {
    it->second = it->second*schd.nwalk/oldwt;
  }
  oldwt = schd.nwalk;
  double targetwt = schd.nwalk;
  
  int nGeneration = schd.nGeneration;
  int gradIter = min(niter, 100000);
  double EavgExpDecay = 0.0, Eavg;
  
  double Enum = 0.0, Eden = 0.0, totalwt = 0.0;
  bool importanceSample = true;
  double popControl = 1.0;
  double iterPop = 0.0, genPop = 0.0, olditerPop = oldwt, oldgenPop = oldwt*schd.nGeneration;

  while (iter < niter)
  {
    

    //One step iter
    int i = 0;
    for (auto it = Walkers.begin(); it != Walkers.end(); it++) {
      Walker& walk = it->first;
      double& wt = it->second;
      
      double ham=0., ovlp=0.;
      double wtsold = wt;
      applyPropogatorContinousTime(walk, wt, w, schd.tau,
				   ovlpRatio, excitation1, excitation2, 
				   HijElements, cumHijElements, Eshift,
				   ham, ovlp, I1, I2, I2hb, coreE,
				   schd.fn_factor);
      
      iterPop += abs(wt);
      Enum += popControl*wt*ham;
      Eden += popControl*wt;	

      i++;
    }
    genPop += iterPop;

    double factor = reconfigure_splitjoin(Walkers, targetwt, tau, Eavg, Eshift);

    Eshift = Eshift - 0.1/tau * log(iterPop/olditerPop);
    factor *= pow(olditerPop/iterPop, 0.1);
    popControl = pow(popControl, 0.99)*factor;

    olditerPop = iterPop;
    iterPop = 0.0;
    
    if (iter % nGeneration == 0 && iter != 0) {
      double Ecurrentavg = Enum/Eden;
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &Ecurrentavg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      //MPI_Allreduce(MPI_IN_PLACE, &Enum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      //MPI_Allreduce(MPI_IN_PLACE, &Eden, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &genPop, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      Ecurrentavg /= commsize;

      if (iter == nGeneration)
	EavgExpDecay = Ecurrentavg;
      EavgExpDecay = (0.9)*EavgExpDecay +  0.1*Ecurrentavg;

      if (iter/nGeneration == 4) 
	Eavg = Ecurrentavg;
      else if (iter/nGeneration > 4) {
	int oldIter = iter/nGeneration;
	Eavg = ((oldIter - 4)*Eavg +  Ecurrentavg)/(oldIter-3);
      }
      else 
	Eavg = EavgExpDecay;


      double Egr = Eshift ;
      
      if (commrank == 0)	  
	cout << format("%8i %14.8f  %14.8f  %14.8f  %10i  %8.2f   %14.8f   %8.2f\n") % iter % Ecurrentavg % (EavgExpDecay) %Eavg % (Walkers.size()) % (genPop/commsize/nGeneration) % Egr % (getTime()-startofCalc);
      
      oldgenPop = genPop;
      genPop = 0.0;
      
      Enum=0.0; Eden =0.0;
      popControl = 1.0;
    }
    iter++;
  }
}



void doGFMC(CPSSlater &w, double Eshift, oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, double coreE)
{
  startofCalc = getTime();
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
			  std::ref(generator));
  
  //initialize the walker
  Determinant d;
  bool readDeterminant = false;
  char file[5000];
  
  sprintf(file, "BestDeterminant.txt");
  {
    ifstream ofile(file);
    if (ofile) readDeterminant = true;
  }
  
  if ( readDeterminant)
    {
      ifstream ofile(file);
      if (commrank == 0)
        {
	  std::ifstream ifs(file, std::ios::binary);
	  boost::archive::binary_iarchive load(ifs);
	  load >> d;
        }
#ifndef SERIAL
      MPI_Bcast(&d.reprA, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&d.reprB, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    }
  
  
  Walker walk(d);
  walk.initUsingWave(w);
  list<pair<Walker, double> > Walkers;
  for (int i=0; i<schd.nwalk; i++)
    Walkers.push_back(std::pair<Walker, double>(walk, 1.0));
  generateWalkers(Walkers, w, I1, I2, I2hb, coreE);

  int maxTerms = 1000000;
  vector<double> ovlpRatio(maxTerms);
  vector<size_t> excitation1(maxTerms), excitation2(maxTerms);
  vector<double> HijElements(maxTerms), cumHijElements(maxTerms);
  int nExcitations = 0;
  
  double stddev = 1.e4;
  int iter = 0;
  double M1 = 0., S1 = 0.;
  double Eloc = 0.;
  double ham = 0., ovlp = 0.;
  double scale = 1.0;
  int niter = schd.maxIter;
  double tau = schd.tau;
  
  double oldwt = .0;
  for (auto it=Walkers.begin(); it!=Walkers.end(); it++) oldwt += it->second;
  for (auto it=Walkers.begin(); it!=Walkers.end(); it++) {
    it->second = it->second*schd.nwalk/oldwt;
  }
  oldwt = schd.nwalk;
  double targetwt = schd.nwalk;
  
  int nGeneration = schd.nGeneration;
  int gradIter = min(niter, 100000);
  double EavgExpDecay = 0.0, Eavg;
  
  double Enum = 0.0, Eden = 0.0, totalwt = 0.0;
  double popControl = 1.0;
  double iterPop = 0.0, genPop = 0.0, olditerPop = oldwt, oldgenPop = oldwt*schd.nGeneration;

  while (iter < niter)
  {
    

    //One step iter
    int i = 0;
    for (auto it = Walkers.begin(); it != Walkers.end(); it++) {
      Walker& walk = it->first;
      double& wt = it->second;
      
      double ham=0., ovlp=0.;
      double wtsold = wt;
      applyPropogator(walk, wt, w, schd.tau,
		      ovlpRatio, excitation1, excitation2, 
		      HijElements, cumHijElements, Eshift,
		      ham, ovlp, I1, I2, I2hb, coreE,
		      schd.fn_factor);
      
      iterPop += abs(wt);
      Enum += popControl*wtsold*ham;
      Eden += popControl*wtsold;	

      i++;
    }
    genPop += iterPop;

    double factor = reconfigure_splitjoin(Walkers, targetwt, tau, Eavg, Eshift);

    Eshift = Eshift - 0.1/tau * log(iterPop/olditerPop);
    factor *= pow(olditerPop/iterPop, 0.1);
    popControl = pow(popControl, 0.99)*factor;

    olditerPop = iterPop;
    iterPop = 0.0;
    
    if (iter % nGeneration == 0 && iter != 0) {
      double Ecurrentavg = Enum/Eden;
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &Ecurrentavg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      //MPI_Allreduce(MPI_IN_PLACE, &Enum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      //MPI_Allreduce(MPI_IN_PLACE, &Eden, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &genPop, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      Ecurrentavg /= commsize;

      if (iter == nGeneration)
	EavgExpDecay = Ecurrentavg;
      EavgExpDecay = (0.9)*EavgExpDecay +  0.1*Ecurrentavg;

      if (iter/nGeneration == 4) 
	Eavg = Ecurrentavg;
      else if (iter/nGeneration > 4) {
	int oldIter = iter/nGeneration;
	Eavg = ((oldIter - 4)*Eavg +  Ecurrentavg)/(oldIter-3);
      }
      else 
	Eavg = EavgExpDecay;


      double Egr = Eshift ;
      
      if (commrank == 0)	  
	cout << format("%8i %14.8f  %14.8f  %14.8f  %10i  %8.2f   %14.8f   %8.2f\n") % iter % Ecurrentavg % (EavgExpDecay) %Eavg % (Walkers.size()) % (genPop/commsize/nGeneration) % Egr % (getTime()-startofCalc);
      
      oldgenPop = genPop;
      genPop = 0.0;
      
      Enum=0.0; Eden =0.0;
      popControl = 1.0;
    }
    iter++;
  }
}
