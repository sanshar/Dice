#ifndef SERIAL
#include <iostream>
#include <iomanip>
#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include <boost/format.hpp>
#include "input.h"
#include "global.h"
#include "DQMCStatistics.h"

using namespace std;
using namespace Eigen;
using namespace boost;

// constructor
DQMCStatistics::DQMCStatistics(int pSampleSize) 
{
  nSamples = 0;
  sampleSize = pSampleSize;
  numMean = ArrayXcd::Zero(sampleSize);
  denomMean = ArrayXcd::Zero(sampleSize);
  denomAbsMean = ArrayXd::Zero(sampleSize);
  num2Mean = ArrayXcd::Zero(sampleSize);
  denom2Mean = ArrayXcd::Zero(sampleSize);
  num_denomMean = ArrayXcd::Zero(sampleSize);
  converged.resize(sampleSize,-1);
  convergedE = ArrayXcd::Zero(sampleSize);
  convergedDev = ArrayXd::Zero(sampleSize);
  convergedDev2 = ArrayXd::Zero(sampleSize);
  convergedPhase = ArrayXcd::Zero(sampleSize);
  errorTargets = schd.errorTargets;    // TODO: change this so that these are passed to the constructor
}


// store samples and update running averages
void DQMCStatistics::addSamples(ArrayXcd& numSample, ArrayXcd& denomSample)
{
  numMean += (numSample - numMean) / (nSamples + 1.);
  denomMean += (denomSample - denomMean) / (nSamples + 1.);
  denomAbsMean += (denomSample.abs() - denomAbsMean) / (nSamples + 1.);
  numSamples.push_back(numSample);
  denomSamples.push_back(denomSample);
  nSamples++;
}

// get the current number of samples
size_t DQMCStatistics::getNumSamples()
{
  return nSamples;
}

// calculates error by blocking data
// use after gathering data across processes for better estimates
void DQMCStatistics::calcError(ArrayXd& error, ArrayXd& error2, ArrayXcd& bias)
{
  ArrayXcd eneEstimates = numMean / denomMean;
  int nBlocks;
  size_t blockSize;
  if (nSamples <= 500) {
    nBlocks = 1;
    blockSize = nSamples;
  }
  else {
    nBlocks = 10;
    blockSize = size_t(nSamples / 10);
  }
  ArrayXd var(sampleSize), var2(sampleSize), denomVar(sampleSize), numVar(sampleSize);
  var.setZero(); var2.setZero(); denomVar.setZero(); numVar.setZero();
  ArrayXcd cov(sampleSize);
  cov.setZero();

  // calculate variance of blocked energies on each process
  for (int i = 0; i < nBlocks; i++) {
    ArrayXcd blockNum(sampleSize), blockDenom(sampleSize);
    blockNum.setZero(); blockDenom.setZero();
    for (int n = i * blockSize; n < (i + 1) * blockSize; n++) {
      blockNum += numSamples[n];
      blockDenom += denomSamples[n];
    }
    blockNum /= blockSize;
    blockDenom /= blockSize;
    ArrayXcd blockEne(sampleSize);
    blockEne = blockNum / blockDenom;
    var += (blockEne - eneEstimates).abs().pow(2);
    var2 += (blockEne - eneEstimates).abs().pow(4);
    numVar += (blockNum - numMean).abs().pow(2);
    denomVar += (blockDenom - denomMean).abs().pow(2);
    cov += (blockNum - numMean) * (blockDenom - denomMean);
  }
  
  // gather variance across processes
  MPI_Allreduce(MPI_IN_PLACE, var.data(), var.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, var2.data(), var2.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, numVar.data(), numVar.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, denomVar.data(), denomVar.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, cov.data(), cov.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

  int nBlockedSamples = nBlocks * commsize;
  var /= (nBlockedSamples - 1);
  var2 /= (nBlockedSamples);
  numVar /= (nBlockedSamples - 1);
  denomVar /= (nBlockedSamples - 1);
  cov /= (nBlockedSamples - 1);

  // bias
  bias = - (numMean * denomVar / denomMean.pow(3) - cov / denomMean.pow(2)) / nBlockedSamples;

  //if (commrank == 0) {
  //  cout << "\nnumMean:  " << numMean.transpose() << endl;
  //  cout << "numVar:  " << numVar.transpose() << endl;
  //  cout << "denomMean:  " << denomMean.transpose() << endl;
  //  cout << "denomVar:  " << denomVar.transpose() << endl;
  //  cout << "cov:  " << cov.transpose() << endl;
  //  cout << "bias (not added to the energies):  " << bias.transpose() << endl << endl;
  //}

  // calculate error estimates a la clt
  error = sqrt(var / nBlockedSamples);
  error2 = sqrt((var2 - (nBlockedSamples - 3) * var.pow(2) / (nBlockedSamples - 1)) / nBlockedSamples) / 2. / sqrt(var) / sqrt(sqrt(nBlockedSamples));
}
 

// gather data from all the processes and print quantities
// to be used at the end of a calculation
// iTime used only for printing
void DQMCStatistics::gatherAndPrintStatistics(ArrayXd iTime, complex<double> delta)
{
  ArrayXcd numMeanbkp = numMean;
  ArrayXcd denomMeanbkp = denomMean;
  ArrayXd denomAbsMeanbkp = denomAbsMean;

  // gather data across processes
  MPI_Allreduce(MPI_IN_PLACE, numMean.data(), numMean.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, denomMean.data(), denomMean.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, denomAbsMean.data(), denomAbsMean.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  
  numMean /= commsize;
  denomMean /= commsize;
  denomAbsMean /= commsize;
  ArrayXcd eneEstimates = numMean / denomMean;
  ArrayXcd avgPhase = denomMean / denomAbsMean;

  // calc error estimates
  ArrayXd error, error2;
  ArrayXcd bias;
  calcError(error, error2, bias);

  eneEstimates += delta;
  //eneEstimates += bias;

  // print
  if (commrank == 0) {
    cout << "          iTime                 Energy                     Energy error         Average phase\n";
    for (int n = 0; n < sampleSize; n++) {
      if (converged[n] == -1) {
        cout << format(" %14.2f   (%14.8f, %14.8f)   (%8.2e   (%8.2e))   (%3.3f, %3.3f) \n") % iTime(n) % eneEstimates(n).real() % eneEstimates(n).imag() % error(n) % error2(n) % avgPhase(n).real() % avgPhase(n).imag(); 

      }
      else { //after it has converged just use the old ones
        cout << format(" %14.2f   (%14.8f, %14.8f)   (%8.2e   (%8.2e))   (%3.3f, %3.3f) \n") % iTime(n) % convergedE(n).real() % convergedE(n).imag() % convergedDev(n) % convergedDev2(n) % convergedPhase(n).real() % convergedPhase(n).imag(); 
      }

    }
  }

  // if error falls below the specified threshold then stop calculating it
  for (int n = 0; n < sampleSize; n++) {
    if (error(n) < errorTargets[n]) {
      converged[n] = 1;
      convergedE(n) = eneEstimates(n);
      convergedDev(n) = error(n);
      convergedDev2(n) = error2(n);
      convergedPhase(n) = avgPhase(n);
    }
  }
  

  //restore the original running averages
  numMean = numMeanbkp;
  denomMean = denomMeanbkp;
  denomAbsMean = denomAbsMeanbkp;  
}

// if all energies are converged
bool DQMCStatistics::isConverged() 
{
  if (converged[sampleSize - 1]  == 1) return true;
  else return false;
};


// prints running averages from proc 0
void DQMCStatistics::printStatistics() 
{
  return;
};


// write samples to disk
void DQMCStatistics::writeSamples()
{
  string fname = "samples_";
  fname.append(to_string(commrank));
  fname.append(".dat");
  ofstream samplesFile(fname, ios::app);
  samplesFile << "num_i  denom_i\n";
  for (int i = 0; i < nSamples; i++) {
    for (int n = 0; n < sampleSize; n++)
      samplesFile << setprecision(8) << numSamples[i](n) << "  "  << denomSamples[i](n) << "  |  ";
    samplesFile << endl;
  }
  samplesFile.close();
  return;
};
