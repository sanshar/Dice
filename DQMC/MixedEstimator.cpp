#include <iostream>
#include <iomanip>
#include "global.h"
#include "input.h"
#include "DQMCStatistics.h"
#include "MixedEstimator.h"
#include <boost/format.hpp>

using namespace std;
using namespace Eigen;

using matPair = std::array<MatrixXcd, 2>;


std::array<double, 2> blocking(const ArrayXd& weights, const ArrayXd& energies, std::string fname = "blocking.tmp") 
{
  int nSamples = weights.size();
  VectorXi blockSizes {{ 1, 2, 5, 10, 20, 50, 70, 100, 200, 500 }};
  ofstream blockingFile(fname);
  blockingFile << "Number of samples: " << nSamples << endl;
  double prevError = 0., plateauError = -1.;
  ArrayXd weightedEnergies = weights * energies;
  std::array<double, 2> ene_err;
  double meanEnergy = weightedEnergies.sum() / weights.sum();
  ene_err[0] = meanEnergy;
  blockingFile << "Mean energy: " << boost::format("%.8e\n") % meanEnergy; 
  blockingFile << "Block size    # of blocks        Mean                Error" << endl;
  for (int i = 0; i < blockSizes.size(); i++) {
    if (blockSizes(i) > nSamples/2.) break;
    int nBlocks = nSamples / blockSizes(i);
    ArrayXd blockedWeights = ArrayXd::Zero(nBlocks);
    ArrayXd blockedEnergies = ArrayXd::Zero(nBlocks);
    for (int j = 0; j < nBlocks; j++) { 
      blockedWeights(j) = weights(Eigen::seq(j*blockSizes(i), (j+1)*blockSizes(i) - 1)).sum();
      blockedEnergies(j) = weightedEnergies(Eigen::seq(j*blockSizes(i), (j+1)*blockSizes(i) - 1)).sum() / blockedWeights(j);
    }
    double v1 = blockedWeights.sum();
    double v2 = (blockedWeights.pow(2)).sum();
    double mean = (blockedWeights * blockedEnergies).sum() / v1;
    double error = sqrt(((blockedWeights * (blockedEnergies - mean).pow(2)).sum() / (v1 - v2 / v1) / (nBlocks - 1)));
    blockingFile << boost::format("  %4d           %4d       %.8e       %.6e\n") % blockSizes(i) % nBlocks % mean % error;
    if (error < 1.05 * prevError && plateauError < 0.) plateauError = max(error, prevError);
    prevError = error;
  }
  if (plateauError > 0.) blockingFile << "Stochastic error estimate: " << plateauError << endl;
  blockingFile.close();
  ene_err[1] = plateauError;
  return ene_err;
}


void writeOneRDM(Eigen::MatrixXd oneRDM, double cumulativeWeight) 
{
  std::string scratch_dir = schd.scratchDir;
  string fname = scratch_dir + "/rdm_";
  fname.append(to_string(commrank));
  fname.append(".dat");
  ofstream rdmdump(fname);
  rdmdump << cumulativeWeight << endl;
  for (int i = 0; i < oneRDM.rows(); i++) {
    for (int j = 0; j < oneRDM.cols(); j++){
      rdmdump << oneRDM(i, j) << "  ";
    }
    rdmdump << endl;
  }
  rdmdump.close();
}


void writeOneRDM(std::array<Eigen::MatrixXd, 2> oneRDM, double cumulativeWeight) 
{
  std::string scratch_dir = schd.scratchDir;
  {
    string fname = scratch_dir + "/rdm_up_";
    fname.append(to_string(commrank));
    fname.append(".dat");
    ofstream rdmdump(fname);
    if (rdmdump.fail()) {
      if (commrank == 0) cout << scratch_dir + " does not exist!\n";
      rdmdump.close();
      exit(0);
    }
    rdmdump << cumulativeWeight << endl;
    for (int i = 0; i < oneRDM[0].rows(); i++) {
      for (int j = 0; j < oneRDM[0].cols(); j++){
        rdmdump << oneRDM[0](i, j) << "  ";
      }
      rdmdump << endl;
    }
    rdmdump.close();
  }
  
  {
    string fname = scratch_dir + "/rdm_dn_";
    fname.append(to_string(commrank));
    fname.append(".dat");
    ofstream rdmdump(fname);
    if (rdmdump.fail()) {
      if (commrank == 0) cout << scratch_dir + " does not exist!\n";
      rdmdump.close();
      exit(0);
    }
    rdmdump << cumulativeWeight << endl;
    for (int i = 0; i < oneRDM[1].rows(); i++) {
      for (int j = 0; j < oneRDM[1].cols(); j++){
        rdmdump << oneRDM[1](i, j) << "  ";
      }
      rdmdump << endl;
    }
    rdmdump.close();
  } 
}


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
  
  int nchol = ham.nchol;
  complex<double> delta(0., 0.);
  if (commrank == 0) cout << "Number of Cholesky vectors: " << nchol << endl;
  vector<int> ncholVec = { int(0.3 * nchol), int(0.4 * nchol), int(0.5 * nchol), int(0.6 * nchol), int(0.7 * nchol) };
  for (int i = 0; i < ncholVec.size(); i++) {
    ham.setNcholEne(ncholVec[i]);
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
  ham.setNcholEne(nchol);
  
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
      walker.propagate(ham);
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
  
  int nchol = ham.nchol;
  complex<double> delta(0., 0.);
  if (commrank == 0) cout << "Number of Cholesky vectors: " << nchol << endl;
  vector<int> ncholVec = { int(0.3 * nchol), int(0.4 * nchol), int(0.5 * nchol), int(0.6 * nchol), int(0.7 * nchol) };
  for (int i = 0; i < ncholVec.size(); i++) {
    ham.setNcholEne(ncholVec[i]);
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
  ham.setNcholEne(nchol);
  
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


void calcMixedEstimatorLongProp(Wavefunction& waveLeft, Wavefunction& waveRight, Wavefunction& waveGuide, DQMCWalker& walker, Hamiltonian& ham)
{
  int norbs = ham.norbs;
  int nalpha = ham.nalpha;
  int nbeta = ham.nbeta;
  int nelec = ham.nelec;
  size_t nsweeps = schd.stochasticIter;  // number of energy evaluations
  size_t nsteps = schd.nsteps;           // number of steps per energy evaluation
  size_t nburn = schd.burnIter;          // number of equilibration steps
  size_t orthoSteps = schd.orthoSteps;
  size_t nwalk = schd.nwalk;             // number of walkers per process
  double dt = schd.dt;
  ofstream afqmcFile("afqmc.dat", ios::app);

  matPair ref;
  MatrixXcd refSOC;
  std::array<std::complex<double>, 2> hamOverlap;
  if (schd.soc) {
    MatrixXcd hf = MatrixXcd::Zero(2*norbs, 2*norbs);
    readMat(hf, "ghf.txt");
    refSOC = hf.block(0, 0, 2*norbs, nelec);
    hamOverlap = waveLeft.hamAndOverlap(refSOC, ham);
  }
  else if (walker.szQ) {
    MatrixXd hf = MatrixXd::Zero(norbs, norbs);
    readMat(hf, "ghf.txt");
    refSOC = hf.block(0, 0, norbs, nelec);
    hamOverlap = waveLeft.hamAndOverlap(refSOC, ham);
  }
  else { 
    if (walker.rhfQ) { 
      MatrixXd hf = MatrixXd::Zero(norbs, norbs);
      readMat(hf, "rhf.txt");
      ref[0] = hf.block(0, 0, norbs, nalpha);
      ref[1] = hf.block(0, 0, norbs, nbeta);
    }
    else {
      MatrixXd hf = MatrixXd::Zero(norbs, 2*norbs);
      readMat(hf, "uhf.txt");
      ref[0] = hf.block(0, 0, norbs, nalpha);
      ref[1] = hf.block(0, norbs, norbs, nbeta);
    }
    
    hamOverlap = waveLeft.hamAndOverlap(ref, ham);
  }
  complex<double> refEnergy = hamOverlap[0] / hamOverlap[1];
  complex<double> ene0;
  if (schd.ene0Guess == 1.e10) ene0 = refEnergy;
  else ene0 = schd.ene0Guess;
  if (commrank == 0) {
    afqmcFile << "# Initial state energy:  " << refEnergy.real() << endl;
  }
  
  int nchol = ham.nchol;
  complex<double> delta(0., 0.);
  if (commrank == 0) afqmcFile << "# Number of Cholesky vectors: " << nchol << endl;
  vector<int> ncholVec = { int(0.3 * nchol), int(0.4 * nchol), int(0.5 * nchol), int(0.6 * nchol), int(0.7 * nchol) };
  for (int i = 0; i < ncholVec.size(); i++) {
    ham.setNcholEne(ncholVec[i]);
    std::array<complex<double>, 2> thamOverlap;
    if (walker.szQ) thamOverlap = waveLeft.hamAndOverlap(refSOC, ham);
    else thamOverlap = waveLeft.hamAndOverlap(ref, ham);
    complex<double> trefEnergy = thamOverlap[0] / thamOverlap[1];
    if (abs(refEnergy - trefEnergy) < schd.choleskyThreshold) {
      nchol = ncholVec[i];
      delta = refEnergy - trefEnergy;
      if (commrank == 0) {
        afqmcFile << "# Using truncated Cholesky with " << nchol << " vectors for energy calculations\n";
        afqmcFile << "# Initial state energy with truncated Cholesky:  " << trefEnergy << endl;
        afqmcFile.flush();
      }
      break;
    }
  }
  ham.setNcholEne(nchol);
  if (commrank == 0) {
    afqmcFile << "#\n";
    afqmcFile.flush();
  }

  vector<DQMCWalker> walkers;
  ArrayXd weights = ArrayXd::Zero(nwalk);
  ArrayXd localEnergy = ArrayXd::Zero(nwalk);
  ArrayXd totalWeights = ArrayXd::Zero(nsweeps);
  ArrayXd totalEnergies = ArrayXd::Zero(nsweeps);
  if (walker.szQ) walker.prepProp(refSOC, ham, dt, ene0.real());
  else walker.prepProp(ref, ham, dt, ene0.real());
  auto calcInitTime = getTime();
  for (int w = 0; w < nwalk; w++) {
    DQMCWalker walkerCopy = walker;
    if (walker.szQ) {
      MatrixXcd rn;
      waveRight.getSample(rn);
      walkerCopy.setDet(rn);
    }
    else {
      matPair rn;
      waveRight.getSample(rn);
      walkerCopy.setDet(rn);
    }
    walkers.push_back(walkerCopy);
    weights(w) = 1.;
    walkers[w].overlap(waveLeft);  // this initializes the trialOverlap in the walker, used in propagation
    //walkers[w].overlap(waveGuide);  // this initializes the trialOverlap in the walker, used in propagation
    hamOverlap = walkers[w].hamAndOverlap(waveLeft, ham);
    localEnergy(w) = (hamOverlap[0]/hamOverlap[1]).real();
  }
  
  totalWeights(0) = nwalk * commsize;
  double weightedEnergy = localEnergy.sum();
  MPI_Allreduce(MPI_IN_PLACE, &weightedEnergy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  totalEnergies(0) = weightedEnergy / totalWeights(0) + delta.real(); 
  if (commrank == 0) {
    double initializationTime = getTime() - calcInitTime;
    afqmcFile << "# block     propTime           eshift          weight             energy          cumulative_energy          walltime\n";
    afqmcFile << boost::format(" %5d     %.3e       %.5e     %.5e      %.9e       %9s                %.2e \n") % 0 % 0. % totalEnergies(0) % totalWeights(0) % totalEnergies(0) % '-' % initializationTime; 
    afqmcFile.flush();
    cout << "   Iter        Mean energy          Stochastic error       Walltime\n";
    cout << boost::format(" %5d      %.9e      %9s                %.2e \n") % 0 % totalEnergies(0) % '-' % initializationTime; 
  }

  double propTime = 0., eneTime = 0.;
  double eshift = totalEnergies(0);
  double totalWeight = nwalk * commsize;
  double cumulativeWeight = 0.;
  double averageEnergy = totalEnergies(0), averageNum = 0., averageDenom = 0.;
  double averageEnergyEql = totalEnergies(0), averageNumEql = 0., averageDenomEql = 0.;
  double eEstimate = totalEnergies(0);
  long nLargeDeviations = 0;
  int measureCounter = 1;
  MatrixXd oneRDM;
  if (schd.soc) oneRDM = MatrixXd::Zero(2*norbs, 2*norbs);
  else oneRDM = MatrixXd::Zero(norbs, norbs);
  std::array<MatrixXd, 2> oneRDMU;
  oneRDMU[0]= MatrixXd::Zero(norbs, norbs);
  oneRDMU[1]= MatrixXd::Zero(norbs, norbs);
  MatrixXcd rdmSample;
  if (schd.soc) rdmSample = MatrixXcd::Zero(2*norbs, 2*norbs);
  else rdmSample = MatrixXcd::Zero(norbs, norbs);
  std::array<MatrixXcd, 2> rdmSampleU;
  rdmSampleU[0] = MatrixXcd::Zero(norbs, norbs);
  rdmSampleU[1] = MatrixXcd::Zero(norbs, norbs);
  double weightCap = 0.;
  if (schd.weightCap > 0) weightCap = schd.weightCap;
  else weightCap = std::max(100., walkers.size() / 10.);
  for (int step = 1; step < nsweeps * nsteps; step++) {
    // average before eql
    if (step * dt < 10.) averageEnergy = averageEnergyEql;

    // propagate
    double init = getTime();
    for (int w = 0; w < walkers.size(); w++) {
      if (weights[w] > 1.e-8) weights[w] *= walkers[w].propagatePhaseless(waveLeft, ham, eshift);
      if (weights[w] > weightCap) {
        weights[w] = 0.;
        nLargeDeviations++;
      }
      //if (weights[w] > 1.e-8) weights[w] *= walkers[w].propagatePhaseless(waveGuide, ham, eshift);
    }
    propTime += getTime() - init;
    
    // population control
    totalWeight = weights.sum();
    MPI_Allreduce(MPI_IN_PLACE, &totalWeight, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //eshift = averageEnergy - 0.1 * log(totalWeight/(nwalk * commsize)) / dt;
    eshift = eEstimate - 0.1 * log(totalWeight/(nwalk * commsize)) / dt;

    // orthogonalize for stability
    if (step % orthoSteps == 0) {
      for (int w = 0; w < walkers.size(); w++) walkers[w].orthogonalize();
    }

    // measure
    if (step % nsteps == 0) { 
      measureCounter++;
      localEnergy.setZero();
      int block = step / nsteps;
      init = getTime();
      //ArrayXd overlapRatios = ArrayXd::Zero(walkers.size());
      for (int w = 0; w < walkers.size(); w++) {
        if (weights(w) != 0.) {
          // one rdm
          if (step * dt > 10. && step > schd.burnIter * nsteps) {
            if (ham.intType == "r" || ham.intType == "g") {
              walkers[w].oneRDM(waveLeft, rdmSample);
              oneRDM *= cumulativeWeight;
              oneRDM += weights(w) * rdmSample.real();
              cumulativeWeight += weights(w);
              oneRDM /= cumulativeWeight;
            }
            else if (ham.intType == "u") {
              walkers[w].oneRDM(waveLeft, rdmSampleU);
              oneRDMU[0] *= cumulativeWeight;
              oneRDMU[1] *= cumulativeWeight;
              oneRDMU[0] += weights(w) * rdmSampleU[0].real();
              oneRDMU[1] += weights(w) * rdmSampleU[1].real();
              cumulativeWeight += weights(w);
              oneRDMU[0] /= cumulativeWeight;
              oneRDMU[1] /= cumulativeWeight;
            }
          }

          // energy
          auto hamOverlap = walkers[w].hamAndOverlap(waveLeft, ham);
          localEnergy(w) = (hamOverlap[0]/hamOverlap[1]).real() + delta.real();
          if (std::isnan(localEnergy(w)) || std::isinf(localEnergy(w))) {
            cout << "local energy:  " << localEnergy(w) << endl;
            cout << "weight:  " << weights(w) << endl;
            cout << "overlap:  " << walkers[w].trialOverlap << "   " << hamOverlap[1] << endl;
            cout << "ham:  " << hamOverlap[0] << endl;
            cout << "orthofac:  " << walkers[w].orthoFac << endl;
            exit(0);
            localEnergy(w) = 0.;
            weights(w) = 0.;
          }
          else if (abs(localEnergy(w) - averageEnergy) > sqrt(2./dt)) {
            nLargeDeviations++;
            //weights(w) = 0.;
            if (localEnergy(w) > averageEnergy) localEnergy(w) = averageEnergy + sqrt(2./dt);
            else localEnergy(w) = averageEnergy - sqrt(2./dt);
          }
        }
        else localEnergy(w) = 0.;
      }
      eneTime += getTime() - init;
      weightedEnergy = (localEnergy * weights).sum();
      MPI_Allreduce(MPI_IN_PLACE, &weightedEnergy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      totalWeights(block) = totalWeight;
      totalEnergies(block) = weightedEnergy / totalWeight; 
      if (commrank == 0) {
        if (step * dt < 10. || step < schd.burnIter * nsteps) {
          averageNumEql += totalEnergies(block) * totalWeights(block);
          averageDenomEql += totalWeights(block);
          averageEnergyEql = averageNumEql / averageDenomEql;
          afqmcFile << boost::format(" %5d     %.3e       %.5e     %.5e      %.9e         %7s                %.2e \n") % block % (dt * step) % eshift % totalWeights(block) % totalEnergies(block) % '-' % (getTime() - calcInitTime); 
          afqmcFile.flush();
        }
        else {
          averageNum += totalEnergies(block) * totalWeights(block);
          averageDenom += totalWeights(block);
          averageEnergy = averageNum / averageDenom;
          afqmcFile << boost::format(" %5d     %.3e       %.5e     %.5e      %.9e        %.9e        %.2e \n") % block % (dt * step) % eshift % totalWeights(block) % totalEnergies(block) % averageEnergy % (getTime() - calcInitTime); 
          afqmcFile.flush();
        }
      }
      eEstimate = 0.9 * eEstimate + 0.1 * totalEnergies(block);
    }

    // reconfigure for efficiency
    if (step % nsteps == 0) {
      // serialize walkers
      int matSize;
      if (schd.soc) matSize = 2 * ham.norbs * ham.nelec;
      else if (walker.szQ) matSize = ham.norbs * ham.nelec;
      else matSize = ham.norbs * (ham.nalpha  + ham.nbeta);
      vector<complex<double>> serial(nwalk * matSize, complex<double>(0., 0.)), serialw(matSize, complex<double>(0., 0.)), overlaps(nwalk, complex<double>(0., 0.));
      for (int w = 0; w < walkers.size(); w++) {
        overlaps[w] = walkers[w].getDet(serialw);
        for (int i = 0; i < matSize; i++) serial[w * matSize + i] = serialw[i];
      }

      // gather on the root 
      vector<complex<double>> serialGather, overlapsGather;
      vector<double> weightsGather;
      if (commrank == 0) {
        serialGather = vector<complex<double>>(commsize * nwalk * matSize, complex<double>(0., 0.));
        overlapsGather = vector<complex<double>>(commsize * nwalk, complex<double>(0., 0.));
        weightsGather = vector<double>(commsize * nwalk, 0.);
      }
      MPI_Gather(serial.data(), nwalk * matSize, MPI_DOUBLE_COMPLEX, serialGather.data(), nwalk * matSize, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
      MPI_Gather(overlaps.data(), nwalk, MPI_DOUBLE_COMPLEX, overlapsGather.data(), nwalk, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
      MPI_Gather(weights.data(), nwalk, MPI_DOUBLE, weightsGather.data(), nwalk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      
      // reconfigure
      vector<complex<double>> newSerialGather, newOverlapsGather;
      double totalCumulativeWeight;
      if (commrank == 0) {
        if (std::isnan(averageEnergy)) {
          cout << "overlaps:  ";
          for (int i = 0; i < overlapsGather.size(); i++) cout << overlapsGather[i] << "  ";
          cout << "\n\nweights:  ";
          for (int i = 0; i < weightsGather.size(); i++) cout << weightsGather[i] << "  ";
          cout << endl << endl;
          exit(0);
        }
        vector<double> cumulativeWeights(weightsGather.size(), 0.);
        cumulativeWeights[0] = weightsGather[0];
        for (int i = 1; i < weightsGather.size(); i++) cumulativeWeights[i] = cumulativeWeights[i - 1] + weightsGather[i];
        totalCumulativeWeight = cumulativeWeights[cumulativeWeights.size() - 1];
        std::uniform_real_distribution<double> uniform = std::uniform_real_distribution<double>(0., 1.);
        double zeta = uniform(generator);
        newSerialGather = vector<complex<double>>(commsize * nwalk * matSize, complex<double>(0., 0.));
        newOverlapsGather = vector<complex<double>>(commsize * nwalk, complex<double>(0., 0.));
        for (int w = 0; w < commsize * nwalk; w++) {
          double z = (w + zeta) / nwalk / commsize;
          int index = std::lower_bound(cumulativeWeights.begin(), cumulativeWeights.end(), z * totalCumulativeWeight) - cumulativeWeights.begin();
          for (int i = 0; i < matSize; i++) newSerialGather[w * matSize + i] = serialGather[index * matSize + i];
          newOverlapsGather[w] = overlapsGather[index];
        }
      }

      // scatter
      MPI_Scatter(newSerialGather.data(), nwalk * matSize, MPI_DOUBLE_COMPLEX, serial.data(), nwalk * matSize, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
      MPI_Scatter(newOverlapsGather.data(), nwalk, MPI_DOUBLE_COMPLEX, overlaps.data(), nwalk, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
      MPI_Bcast(&totalCumulativeWeight, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      // deserialize
      for (int w = 0; w < walkers.size(); w++) {
        for (int i = 0; i < matSize; i++) serialw[i] = serial[w * matSize + i];
        walkers[w].setDet(serialw, overlaps[w]);
        weights[w] = totalCumulativeWeight / commsize / nwalk;
      }
      totalWeight = totalCumulativeWeight / commsize;
      //MPI_Barrier(MPI_COMM_WORLD);
      //if (commrank == 0) {
      //  for (int w = 0; w < walkers.size(); w++) {
      //    cout << w << "  " << overlaps[w] << "  " << walkers[w].trialOverlap << endl;
      //  }
      //  cout << "weights:\n" << weights << endl << endl;
      //  for (int w = 0; w < walkers.size(); w++) {
      //    cout << w << endl << walkers[w].det[0] << endl << endl;
      //  }
      //}
      //MPI_Barrier(MPI_COMM_WORLD);
      //exit(0);
    }

    // periodically carry out blocking analysis and print to disk
    if (commrank == 0) {
      if (step > max(40, schd.burnIter)*nsteps && step % (20*nsteps) == 0) {
        auto ene_err = blocking(totalWeights.head(measureCounter).tail(measureCounter - max(40, schd.burnIter)), totalEnergies.head(measureCounter).tail(measureCounter - max(40, schd.burnIter)));
        if (step % (100*nsteps) == 0) {
          int block = step / nsteps;
          if (ene_err[1] > 0.) cout << boost::format(" %5d      %.9e        %.9e        %.2e \n") % block % ene_err[0] % ene_err[1] % (getTime() - calcInitTime); 
          else cout << boost::format(" %5d      %.9e      %9s                %.2e \n") % block % ene_err[0] % '-' % (getTime() - calcInitTime); 
        }
      }
    }
    
    if (step > max(40, schd.burnIter)*nsteps && step % (20*nsteps) == 0) {
      if (schd.writeOneRDM) {
        std::string scratch_dir = schd.scratchDir;
        if (ham.intType == "r" || ham.intType == "g") {
          writeOneRDM(oneRDM, cumulativeWeight);
        }
        else if (ham.intType == "u") {
          writeOneRDM(oneRDMU, cumulativeWeight);
        }
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &nLargeDeviations, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

  if (commrank == 0) {
    double totalVhsTime = 0., totalExpTime = 0., totalFbTime = 0.;
    for (int w = 0; w < walkers.size(); w++) {
      totalVhsTime += walkers[w].vhsTime;
      totalExpTime += walkers[w].expTime;
      totalFbTime += walkers[w].fbTime;
    }
    afqmcFile << "#\n# Total propagation time:  " << propTime << " s\n"; 
    afqmcFile << "#    VHS Time: " << totalVhsTime << " s\n";
    afqmcFile << "#    Matmul Time: " << totalExpTime << " s\n";
    afqmcFile << "#    Force bias Time: " << totalFbTime << " s\n";
    afqmcFile << "# Energy evaluation time:  " << eneTime << " s\n#\n";
    afqmcFile << "# Number of large deviations:  " << nLargeDeviations << "\n";
    afqmcFile.flush();

    string fname = "samples.dat";
    ofstream samplesFile(fname);
    for (int i = 0; i < nsweeps; i++) {
        samplesFile << boost::format("%.7e      %.10e \n") % totalWeights(i) % totalEnergies(i);
    }
    samplesFile.close();
    if (measureCounter > 40) blocking(totalWeights.head(measureCounter).tail(measureCounter - max(40, schd.burnIter)), totalEnergies.head(measureCounter).tail(measureCounter - max(40, schd.burnIter)), "blocking.tmp");
  }
  afqmcFile.close(); 
  
  // write one rdm 
  if (schd.writeOneRDM) {
    std::string scratch_dir = schd.scratchDir;
    if (ham.intType == "r" || ham.intType == "g") {
      writeOneRDM(oneRDM, cumulativeWeight);
    }
    else if (ham.intType == "u") {
      writeOneRDM(oneRDMU, cumulativeWeight);
    }
  }

};

