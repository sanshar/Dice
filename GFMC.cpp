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
    cout << shift0<<endl;
  }
  MPI_Bcast(&shift0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  doGFMC(wave, shift0, I1, I2, I2HBSHM, coreE);

  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());
  return 0;
}

void updateWalker(Walker& newWalk, int Ex1, int Ex2, CPSSlater& w){
  int norbs = Determinant::norbs ;

  int I = Ex1 / 2 / norbs, A = Ex1 - 2 * norbs * I;
  int J = Ex2 / 2 / norbs, B = Ex2 - 2 * norbs * J;
  
  if (I % 2 == J % 2 && Ex2 != 0)
    {
      if (I % 2 == 1)
        {
	  newWalk.updateB(I / 2, J / 2, A / 2, B / 2, w);
        }
      else
        {
	  newWalk.updateA(I / 2, J / 2, A / 2, B / 2, w);
        }
    }
  else
    {
      if (I % 2 == 0)
	newWalk.updateA(I / 2, A / 2, w);
      else
	newWalk.updateB(I / 2, A / 2, w);
      
      if (Ex2 != 0)
        {
	  if (J % 2 == 1)
            {
	      newWalk.updateB(J / 2, B / 2, w);
            }
	  else
            {
	      newWalk.updateA(J / 2, B / 2, w);
            }
        }
    }
}

void applyPropogator(Walker& walk, double& wt, vector<Walker>& newWalkers, vector<double>& newWeights,
		     CPSSlater& w, double tau, vector<double>& ovlpRatio, vector<size_t>& excitation1,
		     vector<size_t>& excitation2,
		     vector<double>& HijElements, vector<double>& cumHijElements, double& Eshift,
		     double& ham, double& ovlp, oneInt& I1, twoInt& I2, 
		     twoIntHeatBathSHM& I2hb, double coreE,
		     double fn_factor) {

  bool importanceSample = true;
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
      cumHij += abs(-tau*HijElements[i]*ovlpRatio[i]* (1-fn_factor));
      cumHijElements[i] = cumHij; //the cumulative does not change and so wont be excited to
    }
    else {
      if (importanceSample)      cumHij += abs(-tau*HijElements[i]*ovlpRatio[i]);
      else                       cumHij += abs(-tau*HijElements[i]);
      cumHijElements[i] = cumHij;
    }
  }
  
  //cout << cumHij<<endl;
  cumHij += (1.0 - tau*(Ewalk - Eshift));


  int nwtsInt = int(abs(wt) + random());
  if (abs(wt) < 2.0)
    nwtsInt = 1;
  
  for (int t = 0; t < nwtsInt; t++){
    if (cumHij < 0.0) {
      cout << "probably need to reduce time step"<<endl;
      cout << t<<"  "<<nwtsInt<<"  "<<tau<<"  "<<Ewalk<<"  "<<Eshift<<"  "<<cumHij<<"  "<<(1.0 - tau*(Ewalk - Eshift))<<endl;
      cout << walk.d.Energy(I1, I2, coreE)<<endl;
      exit(0);
    }
    double nextDetRandom = random() * cumHij;
    if (nextDetRandom > cumHijElements[nExcitations-1]) {
      newWeights.push_back( cumHij * (wt/nwtsInt));
      newWalkers.push_back(walk);
    }
    else {
      int nextDet = std::lower_bound(cumHijElements.begin(), (cumHijElements.begin() + nExcitations),
				     nextDetRandom) - cumHijElements.begin();


      newWalkers.push_back(walk);
      newWeights.push_back( (wt / nwtsInt) * cumHij * -1.*abs(HijElements[nextDet]*ovlpRatio[nextDet])/(HijElements[nextDet]*ovlpRatio[nextDet]) );
      Walker& newWalk = *newWalkers.rbegin();
      updateWalker(newWalk, excitation1[nextDet], excitation2[nextDet], w);
    }
  }
  //exit(0);
}


void applyPropogatorContinousTime(Walker& walk, double& wt, 
				  CPSSlater& w, double tau, vector<double>& ovlpRatio, vector<size_t>& excitation1,
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
    //ham = 0.; ovlp = 0.;
    w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE, ovlpRatio,
			 excitation1, excitation2, HijElements, nExcitations, false);

    if (cumHijElements.size()< nExcitations) cumHijElements.resize(nExcitations);

    double cumHij = 0;


    //(Eshift - H), there is no tau
    for (int i = 0; i < nExcitations; i++)
    {
      //if sy,x is > 0 then add include contribution to the diagonal
      if (HijElements[i]*ovlpRatio[i] > 0.0) {
	Ewalk += HijElements[i]*ovlpRatio[i] * fn_factor;
	//cumHij += abs(-HijElements[i]*ovlpRatio[i]* (1-fn_factor));
	//cumHij += HijElements[i]*ovlpRatio[i]; //positive contribution
	cumHijElements[i] = cumHij; 
      }
      else {
	cumHij += abs(HijElements[i]*ovlpRatio[i]); //negatie of the contribution to local energy
	cumHijElements[i] = cumHij;
      }

    }

    if (1.0/cumHij > tau) {
      cout <<" Increase the value of "<<tau<<endl;
      exit(0);
    }
    //if (abs(ham - (Ewalk-cumHij)) > 1.e-10) {
    //cout << ham<<"  "<<Ewalk-cumHij<<endl;
    //exit(0);
    //}
    ham = Ewalk - cumHij;

    double deltaT = min(t, 1.0/cumHij);

    wt = wt * exp(deltaT*(Eshift - ham ));
    if (1.0/cumHij < t)   {
      double nextDetRandom = random()*cumHij;
      int nextDet = std::lower_bound(cumHijElements.begin(), (cumHijElements.begin() + nExcitations),
				     nextDetRandom) - cumHijElements.begin();    
      walk.updateWalker(w, excitation1[nextDet], excitation2[nextDet]);
      //updateWalker(walk, excitation1[nextDet], excitation2[nextDet], w);

    }
    
    t -= 1.0/cumHij;
  }
}

void removeWalkersAndAccumulate(vector<Walker>& newWalkers, vector<double>& newWeights,
				vector<Walker>& walkers   , vector<double>& weights) {

  copy(newWalkers.begin(), newWalkers.end(), std::back_inserter(walkers));
  copy(newWeights.begin(), newWeights.end(), std::back_inserter(weights));
  
  //Sort walkers
  std::vector<size_t> idx(weights.size());
  std::iota(idx.begin(), idx.end(), 0); //idx=0,1,2,...,
  std::sort(idx.begin(), idx.end(), [&walkers](size_t i1, size_t i2) { return walkers[i1] < walkers[i2]; });
  newWalkers.clear(); newWeights.clear();

  vector<Walker> sortedWalkers = walkers;
  vector<double> sortedWeights = weights;
  for (int i=0; i<idx.size(); i++) {
    sortedWalkers[i] = walkers[idx[i]];
    sortedWeights[i] = weights[idx[i]];
  }

  //Remove duplicates
  int uniqueIndex = 0, i =1;
  walkers[0] = sortedWalkers[0];
  weights[0] = sortedWeights[0];
  while ( i < sortedWalkers.size()) {
    if ( !(sortedWalkers[i] == walkers[uniqueIndex]) ) {
      uniqueIndex++;
      walkers[uniqueIndex] = sortedWalkers[i]; 
      weights[uniqueIndex] = sortedWeights[i];
    }
    else if (sortedWalkers[i] == walkers[uniqueIndex]) {
      weights[uniqueIndex] += sortedWeights[i];
    }
    else {
      cout <<"something is not right"<<endl;
      exit(0);
    }
    i++;
  }
  walkers.resize(uniqueIndex+1);
  weights.resize(uniqueIndex+1);
}

double reconfigure_splitjoin(list<pair<Walker,double> >& walkers, double targetwt, double tau, double Eest, double Eshift) {

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
			  std::ref(generator));

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

  double factor = (newwt/targetwt);

  for (it = walkers.begin(); it != walkers.end(); it++) { 
    it->second = it->second/factor;
  }



  auto prevSmall = walkers.begin();
  bool prevSmallAssigned = false;
  it = walkers.begin();

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


  return factor;
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
  //Eshift = -2.05;
  
  double oldwt = .0;
  for (auto it=Walkers.begin(); it!=Walkers.end(); it++) oldwt += it->second;
  
  
  double targetwt = oldwt;
  
  int reconfigureInterval = 1;
  int gradIter = min(niter, 100000);
  double Eavg = 0.0;
  
  double Enum = 0.0, Eden = 0.0, totalwt = 0.0;
  bool importanceSample = true;
  
  while (iter < niter)
  {
    
    Enum = 0.0; Eden = 0.0;
    for (auto it = Walkers.begin(); it != Walkers.end(); it++) {
      Walker& walk = it->first;
      double& wt = it->second;
      
      double ham=0., ovlp=0.;
      double wtsold = wt;
      applyPropogatorContinousTime(walk, wt, w, tau,
				   ovlpRatio, excitation1, excitation2, 
				   HijElements, cumHijElements, Eshift,
				   ham, ovlp, I1, I2, I2hb, coreE,
				   schd.fn_factor);
      
      totalwt += abs(wt);
      
      Enum += wt*ham;
      Eden += wt;	
    }
    
    //if (iter == 3) exit(0);
    if (iter % reconfigureInterval == 0 ) {
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &Enum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &Eden, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &totalwt, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      Eavg = Enum/Eden;
      Enum = 0.0; Eden = 0.0;
      
      double factor = reconfigure_splitjoin(Walkers, targetwt, tau, Eavg, Eshift);
      double Egr = Eshift - 1.0/(tau)*log(totalwt/targetwt/commsize);
      //double Egr = Eshift + 1.0/(tau) *(1. - totalwt/oldwt);
      if (iter==reconfigureInterval ) Egr = Eshift;
      
      if (commrank == 0)	  
	cout << format("%8i %14.8f  %10i  %8.2f  %8.2f  %14.8f   %8.2f\n") % iter % (Eavg) % (Walkers.size()) % (totalwt/commsize/reconfigureInterval) % (oldwt/commsize/reconfigureInterval) % Egr % (getTime()-startofCalc);
      
      oldwt = totalwt;
      totalwt = 0.0;
      
      Eavg = 0.0;
    }
    iter++;
  }
}
