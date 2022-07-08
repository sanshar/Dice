//#define EIGEN_USE_MKL_ALL
#include <iostream>
#include <fstream>
#ifndef SERIAL
#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "input.h"
#include "integral.h"
#include "SHCIshm.h"
//#include "DQMCSampling.h"
#include "ProjectedMF.h"
#include "RHF.h"
#include "UHF.h"
#include "GHF.h"
#include "KSGHF.h"
#include "Multislater.h"
#include "CCSD.h"
#include "UCCSD.h"
#include "sJastrow.h"
#include "MixedEstimator.h"

int main(int argc, char *argv[])
{

#ifndef SERIAL
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
#endif
  startofCalc = getTime();

  bool print = true;
  initSHM();
  //license();
 
  if (commrank == 0 && print) {
    std::system("echo '# User:' > afqmc.dat; echo $USER | sed 's/^/# /' >> afqmc.dat");
    std::system("echo '# Hostname:' >> afqmc.dat; echo $HOSTNAME | sed 's/^/# /' >> afqmc.dat");
    std::system("echo '# CPU info:' >> afqmc.dat; lscpu | head -15 | sed 's/^/# /' >> afqmc.dat");
    std::system("echo '# Computation started at:' >> afqmc.dat; date | sed 's/^/# /' >> afqmc.dat");
  }
  
  ofstream afqmcFile("afqmc.dat", ios::app);
  if (commrank == 0 && print) {
    afqmcFile << "# git commit: " << GIT_HASH << ", branch: " << GIT_BRANCH << ", compiled at: " << COMPILE_TIME << "\n#\n";
    afqmcFile << "# nproc used: " << commsize << "\n#\n"; 
    afqmcFile.flush();
  }

  string inputFile = "input.dat";
  if (argc > 1)
    inputFile = string(argv[1]);
  readInput(inputFile, schd, print);

  generator = std::mt19937(schd.seed + commrank);

  //MatrixXd h1, h1Mod;
  //vector<MatrixXd> chol;
  //int norbs, nalpha, nbeta;
  //double ecore;
  //readIntegralsCholeskyAndInitializeDeterminantStaticVariables(schd.integralsFile, norbs, nalpha, nbeta, ecore, h1, h1Mod, chol);
 
  Hamiltonian ham = Hamiltonian(schd.integralsFile, schd.soc, schd.intType);
  if (commrank == 0) {
    if (schd.soc || schd.intType == "g") afqmcFile << "# Number of orbitals:  " << ham.norbs << ", nelec:  " << ham.nelec << endl;
    else afqmcFile << "# Number of orbitals:  " << ham.norbs << ", nalpha:  " << ham.nalpha << ", nbeta:  " << ham.nbeta << endl;
    afqmcFile.flush();
  }
  afqmcFile.close();

  DQMCWalker walker;
  if (schd.soc || (schd.intType == "g")) walker = DQMCWalker(false, true, true);
  else {
    if (ham.nalpha != ham.nbeta) walker = DQMCWalker(false);
    if (schd.phaseless) {
      if (ham.nalpha == ham.nbeta) walker = DQMCWalker(true, true);
      else walker = DQMCWalker(false, true);
    }  
  }
  
  //walker = DQMCWalker(false, true);
  //std::array<Eigen::MatrixXcd, 2> det;
  //det[0] = Eigen::MatrixXcd::Zero(4, 2);
  //det[1] = Eigen::MatrixXcd::Zero(4, 2);
  //readMat(det[0], "up.dat");
  //readMat(det[1], "down.dat");
  //cout << "up\n" << det[0] << endl << endl;
  //cout << "dn\n" << det[1] << endl << endl;
  //walker.setDet(det);

  // left state
  Wavefunction *waveLeft;
  if (schd.leftWave == "rhf") {
    waveLeft = new RHF(ham, true); 
  }
  else if (schd.leftWave == "uhf") {
    waveLeft = new UHF(ham, true); 
  }
  else if (schd.leftWave == "ghf") {
    waveLeft = new GHF(ham, true); 
  }
  else if (schd.leftWave == "ksghf") {
    if (schd.optimizeOrbs)
      optimizeProjectedSlater(ham.ecore, ham.h1, ham.chol);
    MPI_Barrier(MPI_COMM_WORLD);
    waveLeft = new KSGHF(ham, true); 
  }
  else if (schd.leftWave == "multislater") {
    int nact = (schd.nciAct < 0) ? ham.norbs : schd.nciAct;
    waveLeft = new Multislater(ham, schd.determinantFile, nact, schd.nciCore); 
  }
  else if (schd.leftWave == "ccsd") {
    if (commrank == 0) cout << "Not supported yet\n";
    exit(0);
  }
  else if (schd.leftWave == "uccsd") {
    if (commrank == 0) cout << "Not supported yet\n";
    exit(0);
  }
  else if (schd.leftWave == "jastrow") {
    waveLeft = new sJastrow(ham.norbs, ham.nalpha, ham.nbeta);
  }
  else {
    if (commrank == 0) cout << "Left wave function not specified\n";
    exit(0);
  }

  //auto overlap = walker.overlap(*waveLeft);
  //Eigen::VectorXcd fb;
  //walker.forceBias(*waveLeft, ham, fb);
  //Eigen::MatrixXcd rdmSample;
  //walker.oneRDM(*waveLeft, rdmSample);
  //auto hamOverlap = walker.hamAndOverlap(*waveLeft, ham);
  //cout << "overlap:  " << overlap << endl << endl;
  //cout << "fb\n" << fb << endl << endl;
  //cout << "rdmSample\n" << rdmSample << endl << endl;
  //cout << "ham:  " << hamOverlap[0] << ",  overlap:  " <<  hamOverlap[1] << endl << endl;
  //cout << "eloc:  " << hamOverlap[0] / hamOverlap[1] << endl;
  //exit(0);

  // right state
  Wavefunction *waveRight;
  if (schd.rightWave == "rhf") {
    waveRight = new RHF(ham, false); 
  }
  else if (schd.rightWave == "uhf") {
    waveRight = new UHF(ham, false); 
    if(schd.phaseless) walker = DQMCWalker(false, true);
    else walker = DQMCWalker(false);
  }
  else if (schd.rightWave == "ghf") {
    waveRight = new GHF(ham, false); 
  }
  else if (schd.rightWave == "ksghf") {
    if (commrank == 0) cout << "Not supported yet\n";
    exit(0);
  }
  else if (schd.rightWave == "multislater") { 
    int nact = (schd.nciAct < 0) ? ham.norbs : schd.nciAct;
    waveLeft = new Multislater(ham, schd.determinantFile, nact, schd.nciCore, true); 
    walker = DQMCWalker(false);
  }
  else if (schd.rightWave == "ccsd") {
    waveRight = new CCSD(ham.norbs, ham.nalpha);
  }
  else if (schd.rightWave == "uccsd") {
    waveRight = new UCCSD(ham.norbs, ham.nalpha, ham.nbeta);
    walker = DQMCWalker(false);
  }
  else if (schd.rightWave == "jastrow") {
    waveRight = new sJastrow(ham.norbs, ham.nalpha, ham.nbeta);
  }
  else {
    if (commrank == 0) cout << "Right wave function not specified\n";
    exit(0);
  }

  if (schd.phaseless) {
    Wavefunction *waveGuide;
    waveGuide = new RHF(ham, false); 
    calcMixedEstimatorLongProp(*waveLeft, *waveRight, *waveGuide, walker, ham);
  }
  else {
    if (schd.dt == 0.) calcMixedEstimatorNoProp(*waveLeft, *waveRight, walker, ham);
    else calcMixedEstimator(*waveLeft, *waveRight, walker, ham);
  }
  //else calcMixedEstimatorLongProp(*waveLeft, *waveRight, walker, ham);
  
  if (commrank == 0) cout << "\nTotal calculation time:  " << getTime() - startofCalc << " s\n";
  
  removeSHM();
  
  return 0;
}
