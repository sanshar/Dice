#include <iostream>
#include <iomanip>
#include "global.h"
#include "input.h"
#include "DQMCStatistics.h"
#include "MixedEstimator.h"

using namespace std;
using namespace Eigen;

using matPair = std::array<MatrixXcd, 2>;

void calcMixedEstimator(Wavefunction& waveLeft, Wavefunction& waveRight, DQMCWalker& walker, Hamiltonian& ham)
{
  int norbs = ham.norbs;
  int nalpha = ham.nalpha;
  int nbeta = ham.nbeta;
  size_t nsweeps = schd.stochasticIter;
  size_t nsteps = schd.nsteps;
  size_t orthoSteps = schd.orthoSteps;
  double dt = schd.dt;
  vector<int> eneSteps = schd.eneSteps;

  MatrixXd hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair ref;
  ref[0] = hf.block(0, 0, norbs, nalpha);
  ref[1] = hf.block(0, 0, norbs, nbeta);
  
  auto hamOverlap = waveLeft.hamAndOverlap(ref, ham);
  complex<double> refEnergy = hamOverlap[0] / hamOverlap[1];
  complex<double> ene0;
  if (schd.ene0Guess == 1.e10) ene0 = refEnergy;
  else ene0 = schd.ene0Guess;
  if (commrank == 0) {
    cout << "Initial state energy:  " << setprecision(8) << refEnergy << endl;
    cout << "Ground state energy guess:  " << ene0 << endl << endl; 
  }
  
  int nchol = ham.chol.size();
  complex<double> delta(0., 0.);
  if (commrank == 0) cout << "Number of Cholesky vectors: " << nchol << endl;
  vector<int> ncholVec = { int(0.3 * nchol), int(0.4 * nchol), int(0.5 * nchol), int(0.6 * nchol), int(0.7 * nchol) };
  for (int i = 0; i < ncholVec.size(); i++) {
    ham.setNchol(ncholVec[i]);
    auto thamOverlap = waveLeft.hamAndOverlap(ref, ham);
    complex<double> trefEnergy = thamOverlap[0] / thamOverlap[1];
    if (abs(refEnergy - trefEnergy) < schd.choleskyThreshold) {
      nchol = ncholVec[i];
      delta = refEnergy - trefEnergy;
      if (commrank == 0) {
        cout << "Using truncated Cholesky with " << nchol << " vectors\n";
        cout << "Initial state energy with truncated Cholesky:  " << trefEnergy << endl << endl;
      }
      break;
    }
  }
  ham.setNchol(nchol);
  
  walker.prepProp(ref, ham, dt, ene0.real());
  int nEneSteps = eneSteps.size();
  DQMCStatistics stats(nEneSteps);
  auto iterTime = getTime();
  double propTime = 0., eneTime = 0.;
  ArrayXd iTime(nEneSteps);
  for (int i = 0; i < nEneSteps; i++) iTime(i) = dt * (eneSteps[i] + 1);
  if (commrank == 0) cout << "Starting sampling sweeps\n";
  
  for (int sweep = 0; sweep < nsweeps; sweep++) {
    if (stats.isConverged()) break;
    if (sweep != 0 && sweep % (schd.printFrequency) == 0) {
      if (commrank == 0) {
        cout << "\nSweep steps: " << sweep << endl << "Total walltime: " << setprecision(6) << getTime() - iterTime << " s\n";
        cout << "Propagation time:  " << propTime << " s\n";
        cout << "Energy evaluation time:  " << eneTime << " s\n\n";
      }
      stats.gatherAndPrintStatistics(iTime, delta);
    }
    
    matPair rn;
    waveRight.getSample(rn);

    walker.setDet(rn);

    // prop sampling
    int eneStepCounter = 0;
    ArrayXcd numSampleA(nEneSteps), denomSampleA(nEneSteps);
    numSampleA.setZero(); denomSampleA.setZero();
    for (int n = 0; n < nsteps; n++) {
      // prop
      double init = getTime();
      walker.propagate();
      propTime += getTime() - init;
      
      // orthogonalize for stability
      if (n % orthoSteps == 0 && n != 0) {
        walker.orthogonalize();
      }

      // measure
      init = getTime();
      if (n == eneSteps[eneStepCounter]) {
        if (stats.converged[eneStepCounter] == -1) {//-1 means this time slice has not yet converged
          auto hamOverlap = walker.hamAndOverlap(waveLeft, ham);
          numSampleA[eneStepCounter] = hamOverlap[0];
          denomSampleA[eneStepCounter] = hamOverlap[1];
        }
        eneStepCounter++;
      }
      eneTime += getTime() - init;
    }
    stats.addSamples(numSampleA, denomSampleA);
  }

  if (commrank == 0) {
    cout << "\nPropagation time:  " << propTime << " s\n";
    cout << "Energy evaluation time:  " << eneTime << " s\n\n";
  }

  stats.gatherAndPrintStatistics(iTime, delta);
  if (schd.printLevel > 10) stats.writeSamples();
};


void calcMixedEstimatorNoProp(Wavefunction& waveLeft, Wavefunction& waveRight, DQMCWalker& walker, Hamiltonian& ham)
{
  int norbs = ham.norbs;
  int nalpha = ham.nalpha;
  int nbeta = ham.nbeta;
  size_t nsweeps = schd.stochasticIter;

  MatrixXd hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair ref;
  ref[0] = hf.block(0, 0, norbs, nalpha);
  ref[1] = hf.block(0, 0, norbs, nbeta);
  
  auto hamOverlap = waveLeft.hamAndOverlap(ref, ham);
  complex<double> refEnergy = hamOverlap[0] / hamOverlap[1];
  if (commrank == 0) {
    cout << "Initial state energy:  " << setprecision(8) << refEnergy << endl;
  }
  
  int nchol = ham.chol.size();
  complex<double> delta(0., 0.);
  if (commrank == 0) cout << "Number of Cholesky vectors: " << nchol << endl;
  vector<int> ncholVec = { int(0.3 * nchol), int(0.4 * nchol), int(0.5 * nchol), int(0.6 * nchol), int(0.7 * nchol) };
  for (int i = 0; i < ncholVec.size(); i++) {
    ham.setNchol(ncholVec[i]);
    auto thamOverlap = waveLeft.hamAndOverlap(ref, ham);
    complex<double> trefEnergy = thamOverlap[0] / thamOverlap[1];
    if (abs(refEnergy - trefEnergy) < schd.choleskyThreshold) {
      nchol = ncholVec[i];
      delta = refEnergy - trefEnergy;
      if (commrank == 0) {
        cout << "Using truncated Cholesky with " << nchol << " vectors\n";
        cout << "Initial state energy with truncated Cholesky:  " << trefEnergy << endl << endl;
      }
      break;
    }
  }
  ham.setNchol(nchol);
  
  //walker.prepProp(ref, ham, dt, ene0.real());
  int nEneSteps = 1;
  DQMCStatistics stats(nEneSteps);
  auto iterTime = getTime();
  double propTime = 0., eneTime = 0.;
  ArrayXd iTime(nEneSteps);
  for (int i = 0; i < nEneSteps; i++) iTime(i) = 0.;
  if (commrank == 0) cout << "Starting sampling sweeps\n";
  
  for (int sweep = 0; sweep < nsweeps; sweep++) {
    if (stats.isConverged()) break;
    if (sweep != 0 && sweep % (schd.printFrequency) == 0) {
      if (commrank == 0) {
        cout << "\nSweep steps: " << sweep << endl << "Total walltime: " << setprecision(6) << getTime() - iterTime << " s\n";
        cout << "Energy evaluation time:  " << eneTime << " s\n\n";
      }
      stats.gatherAndPrintStatistics(iTime, delta);
    }
    
    matPair rn;
    waveRight.getSample(rn);
    walker.setDet(rn);

    double init = getTime();
    int eneStepCounter = 0;
    ArrayXcd numSampleA(nEneSteps), denomSampleA(nEneSteps);
    numSampleA.setZero(); denomSampleA.setZero();
    auto hamOverlap = walker.hamAndOverlap(waveLeft, ham);
    numSampleA[eneStepCounter] = hamOverlap[0];
    denomSampleA[eneStepCounter] = hamOverlap[1];
    eneTime += getTime() - init;
    stats.addSamples(numSampleA, denomSampleA);
  }

  if (commrank == 0) {
    cout << "\nEnergy evaluation time:  " << eneTime << " s\n\n";
  }

  stats.gatherAndPrintStatistics(iTime, delta);
  if (schd.printLevel > 10) stats.writeSamples();
};
