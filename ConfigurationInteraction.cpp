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
#include "evaluatePT.h"
#include "MoDeterminants.h"
#include "Determinants.h"
#include "CPS.h"
#include "Wfn.h"
#include "input.h"
#include "integral.h"
#include "SHCIshm.h"
#include "math.h"
#include "Walker.h"

using namespace Eigen;
using namespace boost;
using namespace std;

void getNVariables(int excitationLevel, vector<int>& SingleIndices,
		     int norbs, twoIntHeatBathSHM& I2hb);
void getStochasticGradientContinuousTimeCI(CPSSlater &w, double &E0, vector<int>& SingleIndices,
					   double &stddev, int &nalpha, int &nbeta, int &norbs,
					   oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb, 
					   double &coreE,  VectorXd &vars, double &rk,
					   int niter, double targetError);
void getDeterministicCI(CPSSlater &w, double &E0, vector<int>& SingleIndices,
					   double &stddev, int &nalpha, int &nbeta, int &norbs,
					   oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb, 
					   double &coreE,  VectorXd &vars, double &rk,
					   int niter, double targetError);
void generateAllDeterminants(vector<Determinant>& allDets, int norbs, int nalpha, int nbeta);
double calcTcorr(vector<double> &v); //in evaluateE.cpp

int main(int argc, char* argv[]) {

#ifndef SERIAL
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
#endif
  startofCalc = getTime();

  initSHM();
  license();

  string inputFile = "input.dat";
  if (argc > 1)
    inputFile = string(argv[1]);
  if (commrank == 0) readInput(inputFile, schd);
#ifndef SERIAL
  mpi::broadcast(world, schd, 0);
#endif

  generator = std::mt19937(schd.seed+commrank);

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

  size_t size;
  ifstream file ("params.bin", ios::in|ios::binary|ios::ate);
  if (commrank == 0) {
    size = file.tellg();
  }
#ifndef SERIAL
  MPI_Bcast(&size, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  Eigen::VectorXd vars = Eigen::VectorXd::Zero(size/sizeof(double));
  if (commrank == 0) {
    file.seekg (0, ios::beg);
    file.read ( (char*)(&vars[0]), size);
    file.close();
  }

#ifndef SERIAL
  MPI_Bcast(&vars[0], vars.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

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

  vector<int> SingleSpinIndices;
  getNVariables(schd.excitationLevel, SingleSpinIndices, norbs, I2HBSHM);

  //we assume intermediate normalization
  Eigen::VectorXd civars = Eigen::VectorXd::Zero(SingleSpinIndices.size()/2+1);
  double rt=0., stddev, E0=0.;

  if (!schd.deterministic)
    getStochasticGradientContinuousTimeCI(wave, E0, SingleSpinIndices, stddev, nalpha, nbeta, norbs, 
					  I1, I2, I2HBSHM, coreE, civars, rt, 
					  schd.stochasticIter, 0.5e-3);
  else
    getDeterministicCI(wave, E0, SingleSpinIndices, stddev, nalpha, nbeta, norbs, 
		       I1, I2, I2HBSHM, coreE, civars, rt, 
		       schd.stochasticIter, 0.5e-3);

  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());
  return 0;
}


void getNVariables(int excitationLevel, vector<int>& SingleIndices,
		     int norbs, twoIntHeatBathSHM& I2hb) {
  if (excitationLevel > 1) {
    cout << "excitaitonLevel greater than 1 has not been implemented yet!!"<<endl;
    exit(0);
  }

  for (int i=0; i<2*norbs; i++) 
    for (int j=0; j<2*norbs; j++) {
      //if (i == j) continue;
      if (i%2 == j%2) {
	SingleIndices.push_back(i);
	SingleIndices.push_back(j);
      }
    }
  SingleIndices.resize(norbs*norbs);
  
}

int getIndex(int I, int A, int norbs) {
  int I1 = I/2, I2 = A/2;

  if (I%2 != A%2) {
    cout <<"something wrong "<<endl;
    exit(0);
  }
  if (I%2 == 0)
    return I1*norbs + I2 + 1;
  else
    return norbs*norbs + I1*norbs + I2 +1;

}

void updateHamOverlap(MatrixXd& Ham, MatrixXd& S, VectorXd& hamRatio, VectorXd& gradRatio,
		      vector<int>& SingleIndices) {
  VectorXd hamRatioSmall = VectorXd::Zero(Ham.rows());
  VectorXd gradRatioSmall = VectorXd::Zero(Ham.rows());

  hamRatioSmall[0]  = hamRatio[0];
  gradRatioSmall[0] = gradRatio[0];

  int norbs = Determinant::norbs;
  int index = 1;
  for (int i=0; i<SingleIndices.size()/2; i++) {
    int a = SingleIndices[2*i], b = SingleIndices[2*i+1];
    int longIndex = getIndex(a,b,norbs);
    hamRatioSmall[index] = hamRatio[longIndex];
    gradRatioSmall[index] = gradRatio[longIndex];
    index++;
  }

  Ham = gradRatioSmall * hamRatioSmall.transpose();
  S   = gradRatioSmall * gradRatioSmall.transpose();
  
  return;
}

void getStochasticGradientContinuousTimeCI(CPSSlater &w, double &E0, vector<int>& SingleIndices,
					   double &stddev, int &nalpha, int &nbeta, int &norbs,
					   oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb, 
					   double &coreE,  VectorXd &civars, double &rk,
					   int niter, double targetError)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));
  
  //initialize the walker
  Determinant d;
  bool readDeterminant = false;
  char file [5000];

  sprintf (file, "BestDeterminant.txt");

  {
    ifstream ofile(file);
    if (ofile) readDeterminant = true;
  }

  if ( readDeterminant )
  {
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
  
  int maxTerms =  (nalpha) * (norbs-nalpha); //pick a small number that will be incremented later
  vector<double> ovlpRatio(maxTerms);
  vector<size_t> excitation1( maxTerms), excitation2( maxTerms);
  vector<double> HijElements(maxTerms);
  int nExcitations = 0;

  vector<double> ovlpRatioForM(maxTerms);
  vector<size_t> excitation1ForM( maxTerms), excitation2ForM( maxTerms);
  vector<double> HijElementsForM(maxTerms);
  int nExcitationsForM = 0;
  
  
  stddev = 1.e4;
  int iter = 0;
  double M1 = 0., S1 = 0., Eavg = 0.;
  double Eloc = 0.;
  double ham = 0., ovlp = 0.;
  double scale = 1.0;

  VectorXd hamRatio  = VectorXd::Zero(2*norbs*norbs+1);
  VectorXd gradRatio = VectorXd::Zero(2*norbs*norbs+1);
  MatrixXd Hamiltonian(civars.rows(), civars.rows());
  MatrixXd Overlap    (civars.rows(), civars.rows());
  MatrixXd iterHamiltonian(civars.rows(), civars.rows());
  MatrixXd iterOverlap    (civars.rows(), civars.rows());
  VectorXd localGrad;

  double bestOvlp =0.;
  Determinant bestDet=d;
  

  //schd.epsilon = -1;
  nExcitations = 0;
  E0 = walk.d.Energy(I1, I2, coreE);
  w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE, ovlpRatio,
                       excitation1, excitation2, HijElements, nExcitations, false);

  gradRatio.setZero();
  double factor = 1.0;
  w.OvlpRatioCI(walk, gradRatio, getIndex, I1, I2, SingleIndices, I2hb, coreE, factor);


  hamRatio = E0*gradRatio;
  for (int m=0; m<nExcitations; m++) {
    
    Walker wtmp = walk;
    //this is the new walker m, later I should try to get rid of this step
    wtmp.updateWalker(w, excitation1[m], excitation2[m]); 
    
    double factor = HijElements[m]*ovlpRatio[m];
    w.OvlpRatioCI(wtmp, hamRatio, getIndex, I1, I2, 
		  SingleIndices, I2hb, coreE, factor);
  }  


  updateHamOverlap(iterHamiltonian, iterOverlap, hamRatio, gradRatio, SingleIndices);
  Hamiltonian = iterHamiltonian;
  Overlap     = iterOverlap;

  int nstore = 1000000/commsize;
  int gradIter = min(nstore, niter);

  std::vector<double> gradError(gradIter*commsize, 0);
  bool reset = false;
  double cumdeltaT = 0., cumdeltaT2 = 0.;

  while (iter < niter && stddev > targetError)
  {
    
    double cumovlpRatio = 0;
    //when using uniform probability 1./numConnection * max(1, pi/pj)
    for (int i = 0; i < nExcitations; i++)
    {
      cumovlpRatio += abs(ovlpRatio[i]);
      //cumovlpRatio += min(1.0, pow(ovlpRatio[i], 2));
      ovlpRatio[i] = cumovlpRatio;
    }

    //double deltaT = -log(random())/(cumovlpRatio);
    double deltaT = 1.0 / (cumovlpRatio);
    double nextDetRandom = random()*cumovlpRatio;
    int nextDet = std::lower_bound(ovlpRatio.begin(), (ovlpRatio.begin()+nExcitations),
                                   nextDetRandom) -
      ovlpRatio.begin();
    

    cumdeltaT += deltaT;
    cumdeltaT2 += deltaT * deltaT;
    
    double Elocold = Eloc;
    
    double ratio = deltaT/cumdeltaT;
    Hamiltonian += ratio * (iterHamiltonian - Hamiltonian);
    Overlap += ratio * (iterOverlap - Overlap);

    Eloc = Eloc + deltaT * (ham - Eloc) / (cumdeltaT);       //running average of energy
    //cout << Hamiltonian(0,0)<<"  "<<Eloc<<"  "<<iterHamiltonian(1,1)<<"  "<<iterOverlap(1,1)<<endl;
    
    S1 = S1 + (ham - Elocold) * (ham - Eloc);
    
    if (iter < gradIter)
      gradError[iter + commrank*gradIter] = ham;
    
    iter++;
    
    
    //update the walker
    if (true)
    {
      walk.updateWalker(w, excitation1[nextDet], excitation2[nextDet]);
    }

    nExcitations = 0;
    double E0 = walk.d.Energy(I1, I2, coreE);
    w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE, ovlpRatio,
			 excitation1, excitation2, HijElements, nExcitations, false);
  
    gradRatio.setZero();
    double factor = 1.0;
    //gradRatio[0] = 1.0;
    w.OvlpRatioCI(walk, gradRatio, getIndex, I1, I2, SingleIndices, I2hb, coreE, factor);
    /*
    //<n|Psi_i>/<n|Psi0>
    gradRatio[0] = 1;
    for (int m=0; m<nExcitations; m++) {
      if (excitation2[m] != 0) //this is the start of double excitation
	break;
      int I = excitation1[m] / 2 / norbs, A = excitation1[m] - 2 * norbs * I;    
      int index = getIndex(I, A, norbs);
      gradRatio[index] = ovlpRatio[m]; //-<n|a_i^dag a_a|Psi0>/<n|Psi0>
    }
    */

    //<m|Psi_i>/<n|Psi0>
    hamRatio = E0*gradRatio;
    for (int m=0; m<nExcitations; m++) {
      
      Walker wtmp = walk;
      wtmp.updateWalker(w, excitation1[m], excitation2[m]); 

      double factor = HijElements[m]*ovlpRatio[m];
      //hamRatio[0] += factor;
      w.OvlpRatioCI(wtmp, hamRatio, getIndex, I1, I2, 
		    SingleIndices, I2hb, coreE, factor);

      /*
      //this should only generate singles when excitationlevel = 1
      nExcitationsForM = 0;
      double ovlpForM, hamForM;
      w.HamAndOvlpGradient(wtmp, ovlpForM, hamForM, localGrad, I1, I2, I2hb, coreE, ovlpRatioForM,
			   excitation1ForM, excitation2ForM, HijElementsForM, nExcitationsForM, false);
      
      hamRatio[0] += HijElements[m] * ovlpRatio[m];
      
      //<m|Psi>/<m|Psi0> 
      for (int n=0; n<nExcitationsForM; n++) {
	if (excitation2ForM[n] != 0) //this is the start of double excitation
	  break;
	int I = excitation1ForM[n] / 2 / norbs, A = excitation1ForM[n] - 2 * norbs * I;    
	int index = getIndex(A, I, norbs);
	hamRatio[index] += HijElements[m] * ovlpRatio[m] * ovlpRatioForM[n];
      }
      */
    }

    updateHamOverlap(iterHamiltonian, iterOverlap, hamRatio, gradRatio, SingleIndices);

    if (abs(ovlp) > bestOvlp) {
      bestOvlp = abs(ovlp);
      bestDet = walk.d;
    }
  }
  
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(gradError[0]), gradError.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Hamiltonian(0,0)), Hamiltonian.rows()*Hamiltonian.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Overlap(0,0)),     Overlap.rows()*Overlap.cols(),         MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &Eloc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  //if (commrank == 0)
  rk = calcTcorr(gradError);

  Hamiltonian /= (commsize);
  Overlap /= (commsize);
  E0 = Eloc / commsize;
  
  stddev = sqrt(S1 * rk / (niter - 1) / niter / commsize);
#ifndef SERIAL
  MPI_Bcast(&stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

  if (commrank == 0) {
    //EigenSolver<MatrixXd> es(Overlap);
    //cout << es.eigenvalues().transpose()<<endl;
    GeneralizedEigenSolver<MatrixXd> ges(Hamiltonian, Overlap);
  //if (commrank == 0) {
    //cout << Hamiltonian <<endl;
    //cout << Overlap <<endl;
  //}
    //ges.compute(Hamiltonian, Overlap);
    cout << ges.eigenvalues().transpose()(0)<<"  ("<<stddev<<")"<<"  "<<Hamiltonian(0,0)/Overlap(0,0)<<endl;
  }
  
}



void getDeterministicCI(CPSSlater &w, double &E0, vector<int>& SingleIndices,
			double &stddev, int &nalpha, int &nbeta, int &norbs,
			oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb, 
			double &coreE,  VectorXd &civars, double &rk,
			int niter, double targetError)
{

  vector<Determinant> allDets;
  generateAllDeterminants(allDets, norbs, nalpha, nbeta);
 

  int maxTerms =  (nalpha) * (norbs-nalpha); //pick a small number that will be incremented later
  vector<double> ovlpRatio(maxTerms);
  vector<size_t> excitation1( maxTerms), excitation2( maxTerms);
  vector<double> HijElements(maxTerms);
  int nExcitations = 0;

  vector<double> ovlpRatioForM(maxTerms);
  vector<size_t> excitation1ForM( maxTerms), excitation2ForM( maxTerms);
  vector<double> HijElementsForM(maxTerms);
  int nExcitationsForM = 0;
  
  double Energy = 0.0;
  VectorXd hamRatio  = VectorXd::Zero(2*norbs*norbs+1);
  VectorXd gradRatio = VectorXd::Zero(2*norbs*norbs+1);
  MatrixXd Hamiltonian(civars.rows(), civars.rows());
  MatrixXd Overlap    (civars.rows(), civars.rows());
  MatrixXd iterHamiltonian(civars.rows(), civars.rows());
  MatrixXd iterOverlap    (civars.rows(), civars.rows());
  VectorXd localGrad;

  for (int i=commrank; i<allDets.size(); i+=commsize) {
    Walker walk(allDets[i]);
    walk.initUsingWave(w);

    schd.epsilon = -1;

    double ham, ovlp;
    {
      nExcitations = 0;
      E0 = walk.d.Energy(I1, I2, coreE);
      w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE, ovlpRatio,
			   excitation1, excitation2, HijElements, nExcitations, false);
  
      gradRatio.setZero();
      double factor = 1.0;
      w.OvlpRatioCI(walk, gradRatio, getIndex, I1, I2, SingleIndices, I2hb, coreE, factor);
      
      
      hamRatio = E0*gradRatio;
      for (int m=0; m<nExcitations; m++) {
	
	Walker wtmp = walk;
	//this is the new walker m, later I should try to get rid of this step
	wtmp.updateWalker(w, excitation1[m], excitation2[m]); 
	
	double factor = HijElements[m]*ovlpRatio[m];
	w.OvlpRatioCI(wtmp, hamRatio, getIndex, I1, I2, 
		      SingleIndices, I2hb, coreE, factor);
      }  
    }
    
    updateHamOverlap(iterHamiltonian, iterOverlap, hamRatio, gradRatio, SingleIndices);
    Hamiltonian += iterHamiltonian*ovlp*ovlp;
    Overlap     += iterOverlap*ovlp*ovlp;
    Energy += ham * ovlp * ovlp;
    //cout << iterHamiltonian(0,0)<<"  "<<ovlp<<"  "<<Energy<<endl;
  }

  cout << Energy<<endl;
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(Hamiltonian(0,0)), Hamiltonian.rows()*Hamiltonian.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Overlap(0,0)),     Overlap.rows()*Overlap.cols(),         MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  //MPI_Allreduce(MPI_IN_PLACE, &Eloc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif


  Hamiltonian /= (commsize);
  Overlap /= (commsize);
  //E0 = Eloc / commsize;
  
  if (commrank == 0) {
    EigenSolver<MatrixXd> es(Overlap);
    cout << es.eigenvalues().transpose()<<endl;
    GeneralizedEigenSolver<MatrixXd> ges(Hamiltonian, Overlap);
    //if (commrank == 0) {
    cout << Hamiltonian <<endl;
    cout << Overlap <<endl;
    //}
    //ges.compute(Hamiltonian, Overlap);
    cout << ges.eigenvalues().transpose()<<"  "<<Hamiltonian(0,0)/Overlap(0,0)<<endl;
  }
  
}
