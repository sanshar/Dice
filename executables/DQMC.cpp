//#define EIGEN_USE_MKL_ALL
#include <iostream>
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
#include "KSGHF.h"
#include "Multislater.h"
#include "CCSD.h"
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
    std::system("echo User:; echo $USER");
    std::system("echo Hostname:; echo $HOSTNAME");
    std::system("echo CPU info:; lscpu | head -15");
    std::system("echo Computation started at:; date");
    cout << "git commit: " << GIT_HASH << ", branch: " << GIT_BRANCH << ", compiled at: " << COMPILE_TIME << endl << endl;
    cout << "nproc used: " << commsize << " (NB: stochasticIter below is per proc)" << endl << endl; 
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
  
  Hamiltonian ham(schd.integralsFile);
  DQMCWalker walker;
  if (ham.nalpha != ham.nbeta) walker = DQMCWalker(false);

  // left state
  Wavefunction *waveLeft;
  if (schd.leftWave == "rhf") {
    waveLeft = new RHF(ham, true); 
  }
  else if (schd.leftWave == "uhf") {
    waveLeft = new UHF(ham, true); 
  }
  else if (schd.leftWave == "ksghf") {
    if (schd.optimizeOrbs)
      optimizeProjectedSlater(ham.ecore, ham.h1, ham.chol);
    MPI_Barrier(MPI_COMM_WORLD);
    waveLeft = new KSGHF(ham, true); 
  }
  else if (schd.leftWave == "multislater") {
    int nact = (schd.nciAct < 0) ? ham.norbs : schd.nciAct;
    waveLeft = new Multislater(schd.determinantFile, nact, schd.nciCore); 
  }
  else if (schd.leftWave == "ccsd") {
    if (commrank == 0) cout << "Not supported yet\n";
    exit(0);
  }
  else if (schd.leftWave == "jastrow") {
    if (commrank == 0) cout << "Not supported yet\n";
    exit(0);
  }
  
  // right state
  Wavefunction *waveRight;
  if (schd.rightWave == "rhf") {
    waveRight = new RHF(ham, false); 
  }
  else if (schd.rightWave == "uhf") {
    waveRight = new UHF(ham, false); 
    walker = DQMCWalker(false);
  }
  else if (schd.rightWave == "ksghf") {
    if (commrank == 0) cout << "Not supported yet\n";
    exit(0);
  }
  else if (schd.rightWave == "multislater") { 
    int nact = (schd.nciAct < 0) ? ham.norbs : schd.nciAct;
    waveLeft = new Multislater(schd.determinantFile, nact, schd.nciCore, true); 
    walker = DQMCWalker(false);
  }
  else if (schd.rightWave == "ccsd") {
    waveRight = new CCSD(ham.norbs, ham.nalpha);
  }
  else if (schd.rightWave == "jastrow") {
    waveRight = new sJastrow(ham.norbs, ham.nalpha, ham.nbeta);
  }
 
  calcMixedEstimator(*waveLeft, *waveRight, walker, ham);
  //calcMixedEstimatorNoProp(waveLeft, waveRight, walker, ham);
  //exit(0);

  //if (schd.wavefunctionType == "jastrow") {
  //  if (commrank == 0) cout << "\nUsing Jastrow RHF trial\n";
  //  calcEnergyJastrowDirect(coreE, h1, h1Mod, chol);
  //}
  //else if (schd.wavefunctionType == "multislater") {
  //  if (commrank == 0) cout << "\nUsing multiSlater trial\n";
  //  calcEnergyDirectMultiSlater(coreE, h1, h1Mod, chol);
  //}
  //else if (schd.wavefunctionType == "ccsd") {
  //  if (commrank == 0) cout << "\nUsing CCSD trial\n";
  //  calcEnergyCCSDDirect(coreE, h1, h1Mod, chol);
  //  //calcEnergyCCSDDirectVariational(coreE, h1, h1Mod, chol);
  //  //calcEnergyCCSDMultiSlaterDirect(coreE, h1, h1Mod, chol);
  //}
  //else if (schd.hf == "ghf") {
  //  if (schd.optimizeOrbs)
  //    optimizeProjectedSlater(coreE, h1, chol);
  //  MPI_Barrier(MPI_COMM_WORLD);
  //  if (commrank == 0) cout << "\nUsing GHF trial\n";
  //  calcEnergyDirectGHF(coreE, h1, h1Mod, chol);
  //}
  //else {
  //  if (commrank == 0) cout << "\nUsing RHF trial\n";
  //  calcEnergyDirect(coreE, h1, h1Mod, chol);
  //}
  if (commrank == 0) cout << "\nTotal calculation time:  " << getTime() - startofCalc << " s\n";
  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shmcas.c_str());
  return 0;
}
