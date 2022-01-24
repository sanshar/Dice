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


void calcMixedEstimatorLongProp(Wavefunction& waveLeft, Wavefunction& waveRight, Wavefunction& waveGuide, DQMCWalker& walker, Hamiltonian& ham)
{
  int norbs = ham.norbs;
  int nalpha = ham.nalpha;
  int nbeta = ham.nbeta;
  size_t nsweeps = schd.stochasticIter;  // number of energy evaluations
  size_t nsteps = schd.nsteps;           // number of steps per energy evaluation
  size_t nburn = schd.burnIter;          // number of equilibration steps
  size_t orthoSteps = schd.orthoSteps;
  size_t nwalk = schd.nwalk;             // number of walkers per process
  double dt = schd.dt;

  matPair ref;
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

  auto hamOverlap = waveLeft.hamAndOverlap(ref, ham);
  complex<double> refEnergy = hamOverlap[0] / hamOverlap[1];
  complex<double> ene0;
  if (schd.ene0Guess == 1.e10) ene0 = refEnergy;
  else ene0 = schd.ene0Guess;
  if (commrank == 0) {
    cout << "Initial state energy:  " << setprecision(8) << refEnergy << endl;
    //cout << "Ground state energy guess:  " << ene0 << endl << endl; 
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
        cout << "Using truncated Cholesky with " << nchol << " vectors for energy calculations\n";
        cout << "Initial state energy with truncated Cholesky:  " << trefEnergy << endl << endl;
      }
      break;
    }
  }
  ham.setNchol(nchol);
 
  vector<DQMCWalker> walkers;
  ArrayXd weights = ArrayXd::Zero(nwalk);
  ArrayXd localEnergy = ArrayXd::Zero(nwalk);
  ArrayXd totalWeights = ArrayXd::Zero(nsweeps);
  ArrayXd totalEnergies = ArrayXd::Zero(nsweeps);
  walker.prepProp(ref, ham, dt, ene0.real());
  auto calcInitTime = getTime();
  for (int w = 0; w < nwalk; w++) {
    DQMCWalker walkerCopy = walker;
    matPair rn;
    waveRight.getSample(rn);
    walkerCopy.setDet(rn);
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
    cout << "  block     propTime         eshift            weight             energy          cumulative_energy          walltime\n";
    cout << boost::format(" %5d     %.3e       %.5e     %.5e      %.9e       %9s                %.2e \n") % 0 % 0. % totalEnergies(0) % totalWeights(0) % totalEnergies(0) % '-' % initializationTime; 
  }

  double propTime = 0., eneTime = 0.;
  double eshift = totalEnergies(0);
  double totalWeight = nwalk * commsize;
  double cumulativeWeight = 0.;
  double averageEnergy = totalEnergies(0), averageNum = 0., averageDenom = 0.;
  double averageEnergyEql = totalEnergies(0), averageNumEql = 0., averageDenomEql = 0.;
  double eEstimate = totalEnergies(0);
  long nLargeDeviations = 0;
  MatrixXd oneRDM = MatrixXd::Zero(norbs, norbs);
  MatrixXcd rdmSample = MatrixXcd::Zero(norbs, norbs);
  for (int step = 1; step < nsweeps * nsteps; step++) {
    // average before eql
    if (step * dt < 10.) averageEnergy = averageEnergyEql;

    // propagate
    double init = getTime();
    for (int w = 0; w < walkers.size(); w++) {
      if (weights[w] > 1.e-8) weights[w] *= walkers[w].propagatePhaseless(waveLeft, ham, eshift);
      if (weights[w] > std::max(100., walkers.size() / 10.)) {
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
      localEnergy.setZero();
      int block = step / nsteps;
      init = getTime();
      //ArrayXd overlapRatios = ArrayXd::Zero(walkers.size());
      for (int w = 0; w < walkers.size(); w++) {
        if (weights(w) != 0.) {
          // one rdm
          if (step * dt > 10.) {
            walkers[w].oneRDM(waveLeft, rdmSample);
            oneRDM *= cumulativeWeight;
            oneRDM += weights(w) * rdmSample.real();
            cumulativeWeight += weights(w);
            oneRDM /= cumulativeWeight;
          }

          // energy
          auto hamOverlap = walkers[w].hamAndOverlap(waveLeft, ham);
          localEnergy(w) = (hamOverlap[0]/hamOverlap[1]).real() + delta.real();
          //localEnergy(w) = (hamOverlap[0]/hamOverlap[1]).real() * std::abs(hamOverlap[1]/walkers[w].trialOverlap) + delta.real();
          //overlapRatios(w) = std::abs(hamOverlap[1]/walkers[w].trialOverlap);
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
          //else if (step * dt < 10. && abs(localEnergy(w) - averageEnergy) > sqrt(2./dt)) {
          //  nLargeDeviations++;
          //  weights(w) = 0.;
          //  localEnergy(w) = 0.;
          //  //if (localEnergy(w) > averageEnergy) localEnergy(w) = averageEnergy + sqrt(2./dt);
          //  //else localEnergy(w) = averageEnergy - sqrt(2./dt);
          //}
          //else if (step * dt >= 10. && abs(localEnergy(w) - averageEnergy) > sqrt(0.1/dt)) {
          //  nLargeDeviations++;
          //  weights(w) = 0.;
          //  localEnergy(w) = 0.;
          //  //if (localEnergy(w) > averageEnergy) localEnergy(w) = averageEnergy + sqrt(0.1/dt);
          //  //else localEnergy(w) = averageEnergy - sqrt(0.1/dt);
          //}
        }
        else localEnergy(w) = 0.;
      }
      eneTime += getTime() - init;
      weightedEnergy = (localEnergy * weights).sum();
      //double ratioWeightedTotalWeight = (weights * overlapRatios).sum();
      MPI_Allreduce(MPI_IN_PLACE, &weightedEnergy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      //MPI_Allreduce(MPI_IN_PLACE, &ratioWeightedTotalWeight, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      totalWeights(block) = totalWeight;
      totalEnergies(block) = weightedEnergy / totalWeight; 
      //totalEnergies(block) = weightedEnergy / ratioWeightedTotalWeight; 
      if (commrank == 0) {
        if (step * dt < 10.) {
          averageNumEql += totalEnergies(block) * totalWeights(block);
          averageDenomEql += totalWeights(block);
          averageEnergyEql = averageNumEql / averageDenomEql;
          cout << boost::format(" %5d     %.3e       %.5e     %.5e      %.9e         %7s                %.2e \n") % block % (dt * step) % eshift % totalWeights(block) % totalEnergies(block) % '-' % (getTime() - calcInitTime); 
          //averageNumEql += totalEnergies(block) * ratioWeightedTotalWeight;
          //averageDenomEql +=  ratioWeightedTotalWeight;
          //averageEnergyEql = averageNumEql / averageDenomEql;
          //cout << boost::format(" %5d     %.3e       %.5e     %.5e      %.6e       %7s               %.2e \n") % block % (dt * step) % eshift % ratioWeightedTotalWeight % totalEnergies(block) % '-' % (getTime() - calcInitTime); 
        }
        else {
          averageNum += totalEnergies(block) * totalWeights(block);
          averageDenom += totalWeights(block);
          averageEnergy = averageNum / averageDenom;
          cout << boost::format(" %5d     %.3e       %.5e     %.5e      %.9e        %.9e        %.2e \n") % block % (dt * step) % eshift % totalWeights(block) % totalEnergies(block) % averageEnergy % (getTime() - calcInitTime); 
          //averageNum += totalEnergies(block) * ratioWeightedTotalWeight;
          //averageDenom += ratioWeightedTotalWeight;
          //averageEnergy = averageNum / averageDenom;
          //cout << boost::format(" %5d     %.3e       %.5e     %.5e      %.6e        %.6e        %.2e \n") % block % (dt * step) % eshift % ratioWeightedTotalWeight % totalEnergies(block) % averageEnergy % (getTime() - calcInitTime); 
        }
      }
      eEstimate = 0.9 * eEstimate + 0.1 * totalEnergies(block);
    }

    // reconfigure for efficiency
    if (step % nsteps == 0) {
      // serialize walkers
      int matSize = ham.norbs * (ham.nalpha  + ham.nbeta);
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
  }
  MPI_Allreduce(MPI_IN_PLACE, &nLargeDeviations, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

  if (commrank == 0) {
    cout << "\nPropagation time:  " << propTime << " s\n";
    cout << "Energy evaluation time:  " << eneTime << " s\n\n";
    cout << "Number of large deviations:  " << nLargeDeviations << "\n";
    string fname = "samples.dat";
    ofstream samplesFile(fname, ios::app);
    for (int i = 0; i < nsweeps; i++) {
        samplesFile << boost::format("%.7e      %.10e \n") % totalWeights(i) % totalEnergies(i);
    }
    samplesFile.close();
  }
  
  // write one rdm 
  string fname = "rdm_";
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
  
};
