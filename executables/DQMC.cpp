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
#include "DQMCSampling.h"
#include "ProjectedMF.h"

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

  MatrixXd h1, h1Mod;
  vector<MatrixXd> chol;
  readIntegralsCholeskyAndInitializeDeterminantStaticVariables(schd.integralsFile, h1, h1Mod, chol);
  

  if (schd.wavefunctionType == "jastrow") {
    if (commrank == 0) cout << "\nUsing Jastrow RHF trial\n";
    calcEnergyJastrowDirect(coreE, h1, h1Mod, chol);
  }
  else if (schd.wavefunctionType == "multislater") {
    if (commrank == 0) cout << "\nUsing multiSlater trial\n";
    calcEnergyDirectMultiSlater(coreE, h1, h1Mod, chol);
  }
  else if (schd.wavefunctionType == "ccsd") {
    if (commrank == 0) cout << "\nUsing CCSD trial\n";
    //calcEnergyCCSDDirect(coreE, h1, h1Mod, chol);
    //calcEnergyCCSDDirectVariational(coreE, h1, h1Mod, chol);
    calcEnergyCCSDMultiSlaterDirect(coreE, h1, h1Mod, chol);
  }
  else if (schd.hf == "ghf") {
    if (schd.optimizeOrbs)
      optimizeProjectedSlater(coreE, h1, chol);
    MPI_Barrier(MPI_COMM_WORLD);
    if (commrank == 0) cout << "\nUsing GHF trial\n";
    calcEnergyDirectGHF(coreE, h1, h1Mod, chol);
  }
  else {
    if (commrank == 0) cout << "\nUsing RHF trial\n";
    calcEnergyDirect(coreE, h1, h1Mod, chol);
  }
  if (commrank == 0) cout << "\nTotal calculation time:  " << getTime() - startofCalc << " s\n";
  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shmcas.c_str());
  return 0;
}
