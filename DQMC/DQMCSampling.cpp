#ifndef SERIAL
#include <iomanip>
#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include <boost/format.hpp>
#include <unsupported/Eigen/MatrixFunctions>
#include "input.h"
#include "global.h"
#include "Determinants.h"
#include "DQMCUtils.h"
#include "DQMCMatrixElements.h"
#include "DQMCStatistics.h"
#include "DQMCSampling.h"
#include <iomanip> 

using namespace Eigen;
using namespace std;
using namespace boost;

using matPair = pair<MatrixXcd, MatrixXcd>;
using vecPair = pair<VectorXcd, VectorXcd>;


void calcEnergyDirectVartiational(double enuc, MatrixXd& h1, MatrixXd& h1Mod, vector<MatrixXd>& chol)
{
  size_t norbs = Determinant::norbs;
  size_t nfields = chol.size();
  size_t nsweeps = schd.stochasticIter;
  size_t nsteps = schd.nsteps;
  size_t orthoSteps = schd.orthoSteps;
  double dt = schd.dt;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;
  vector<int> eneSteps = schd.eneSteps;

  // prep and init
  // this is for mean field subtraction
  MatrixXd hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair rhf;
  rhf.first = hf.block(0, 0, norbs, Determinant::nalpha);
  rhf.second = hf.block(0, 0, norbs, Determinant::nbeta);
  
  vector<matPair> hsOperators;
  matPair oneBodyOperator;
  complex<double> mfConst = prepPropagatorHS(rhf, chol, hsOperators, oneBodyOperator);

  matPair expOneBodyOperator;
  expOneBodyOperator.first =  (-dt * (h1Mod - oneBodyOperator.first) / 2.).exp();
  expOneBodyOperator.second = (-dt * (h1Mod - oneBodyOperator.second) / 2.).exp();

  matPair refAd = rhf; refAd.first = rhf.first.adjoint(); refAd.second = rhf.second.adjoint(); 
  // rotate cholesky vectors
  pair<vector<MatrixXcd>, vector<MatrixXcd>> rotChol;
  for (int i = 0; i < chol.size(); i++) {
    MatrixXcd rotUp = refAd.first * chol[i];
    MatrixXcd rotDn = refAd.second * chol[i];
    rotChol.first.push_back(rotUp);
    rotChol.second.push_back(rotDn);
  }

  vector<VectorXd> fields;
  normal_distribution<double> normal(0., 1.);
  
  //matPair green;
  //calcGreensFunction(refT, ref, green);
  //complex<double> refEnergy = calcHamiltonianElement(green, enuc, h1, chol);
  complex<double> refEnergy = calcHamiltonianElement(refAd, rhf, enuc, h1, chol);
  complex<double> ene0;
  if (schd.ene0Guess == 1.e10) ene0 = refEnergy;
  else ene0 = schd.ene0Guess;
  if (commrank == 0) {
    cout << "Initial state energy:  " << refEnergy << endl;
    cout << "Ground state energy guess:  " << ene0 << endl << endl; 
  }
  
  int nEneSteps = eneSteps.size();
  DQMCStatistics stats(nEneSteps);
  //DQMCStatistics stats(2);
  auto iterTime = getTime();
  double propTime = 0., eneTime = 0.;
  ArrayXd iTime(nEneSteps);
  for (int i = 0; i < nEneSteps; i++) iTime(i) = dt * (eneSteps[i] + 1);
  if (commrank == 0) cout << "Starting sampling sweeps\n";

  int nstepsHalf = schd.nsteps/2 + schd.nsteps%2;

  MatrixXcd Ham  = MatrixXcd::Zero(nstepsHalf+1, nstepsHalf+1),
            Ovlp = MatrixXcd::Zero(nstepsHalf+1, nstepsHalf+1);

  for (int sweep = 0; sweep < nsweeps; sweep++) {

    if (sweep != 0 && sweep % (schd.printFrequency) == 0) {
      if (commrank == 0) {
        cout << "Sweep steps: " << sweep << endl << "Total walltime: " << getTime() - iterTime << " s\n";
        cout << "\nPropagation time:  " << propTime << " s\n";
        cout << "Energy evaluation time:  " << eneTime << " s\n\n";
      }
      stats.gatherAndPrintStatistics(iTime);
    }
    double init = getTime();
      
    complex<double> orthoFac = complex<double>(1., 0.);
    int eneStepCounter = 0;
    ArrayXcd numSampleA(nEneSteps), denomSampleA(nEneSteps);
    numSampleA.setZero(); denomSampleA.setZero();

    //matPair rn = rhf, ln = rhf;
    vector<MatrixXcd> rightPotMat(nstepsHalf, MatrixXcd::Zero(norbs, norbs)), 
                      leftPotMat(nstepsHalf, MatrixXcd::Zero(norbs, norbs));
    vector<MatrixXcd> rightPotProp(nstepsHalf, MatrixXcd::Zero(norbs, norbs)), 
                      leftPotProp(nstepsHalf, MatrixXcd::Zero(norbs, norbs));
    vector<matPair> leftKderiv(nstepsHalf+1, rhf),
                    rightKderiv(nstepsHalf+1, rhf);
    vector<complex<double>> orthoFacLeftKderiv(nstepsHalf+1, 1.),
                            orthoFacRightKderiv(nstepsHalf+1, 1.);

    propTime += getTime() - init;
    for (int n=0; n<nstepsHalf; n++) {
     for (int i=0; i<nfields; i++) {
        rightPotMat[n] += normal(generator) * hsOperators[i].first;
      }
      rightPotProp[n] = (sqrt(dt) * rightPotMat[n]).exp(); 
    }
    for (int n=0; n<nstepsHalf; n++) {
      for (int i=0; i<nfields; i++) {
        leftPotMat[n] += normal(generator) * hsOperators[i].first;
      }
      leftPotProp[n] = (sqrt(dt) * leftPotMat[n]).exp();
    }

    for (int n=0; n<nstepsHalf; n++) {
      rightKderiv[0].first = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * (expOneBodyOperator.first * (rightPotProp[n] * (expOneBodyOperator.first * rightKderiv[0].first)));
      leftKderiv[0].first = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * (expOneBodyOperator.first * (leftPotProp[n] * (expOneBodyOperator.first * leftKderiv[0].first)));
      rightKderiv[0].second = rightKderiv[0].first;
      leftKderiv[0].second = leftKderiv[0].first;

      for (int k=0; k<nstepsHalf; k++) {
        if (k < n) {
          rightKderiv[k+1].first = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * (expOneBodyOperator.first * (rightPotProp[n] * (expOneBodyOperator.first * rightKderiv[k+1].first)));        
          leftKderiv[k+1].first = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * (expOneBodyOperator.first * (leftPotProp[n] * (expOneBodyOperator.first * leftKderiv[k+1].first)));        
        }
        else if (k == n){
          rightKderiv[k+1].first = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * (h1Mod - oneBodyOperator.first) * (expOneBodyOperator.first * (rightPotProp[n] * (expOneBodyOperator.first * rightKderiv[k+1].first)));                
          leftKderiv[k+1].first = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * (h1Mod - oneBodyOperator.first) * (expOneBodyOperator.first * (leftPotProp[n] * (expOneBodyOperator.first * leftKderiv[k+1].first)));                
        }
        else {
          leftKderiv[k+1] = leftKderiv[0];
          rightKderiv[k+1] = rightKderiv[0];
        }
        leftKderiv[k+1].second = leftKderiv[k+1].first;
        rightKderiv[k+1].second = rightKderiv[k+1].first;
      }

      if (n != 0 && n % orthoSteps == 0) {
        for (int k=0; k<n+1; k++) {
          orthogonalize(leftKderiv[k], orthoFacLeftKderiv[k]);
          orthogonalize(rightKderiv[k], orthoFacRightKderiv[k]);
        }
      }
    }
    propTime += getTime() - init;

    init = getTime();
    for (int x = 0; x<nstepsHalf+1; x++) {
      matPair lnAd; lnAd.first = leftKderiv[x].first.adjoint(); lnAd.second = leftKderiv[x].second.adjoint();
      //for (int y = 0; y<nstepsHalf+1; y++) {
      for (int y = 0; y<x+1; y++) {
        complex<double> orthoFac = orthoFacLeftKderiv[x] * orthoFacRightKderiv[y];
        complex<double> overlap = orthoFac * (lnAd.first * rightKderiv[y].first).determinant() * (lnAd.second * rightKderiv[y].second).determinant(); 
        complex<double> numSample = overlap * calcHamiltonianElement(lnAd, rightKderiv[y], enuc, h1, chol); 
        Ovlp(x,y) += overlap;
        Ham (x,y) += numSample;
        Ovlp(y,x) = Ovlp(x,y);
        Ham(y,x) = Ham(x,y);
        if (x == 0 && y == 0) {
          numSampleA[eneStepCounter] = numSample;
          denomSampleA[eneStepCounter] = overlap;
        }
      }
    }
    eneTime += getTime() - init;
    stats.addSamples(numSampleA, denomSampleA);
  }

  MPI_Allreduce(MPI_IN_PLACE, &Ovlp(0,0), Ovlp.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &Ham(0,0), Ham.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

  if (commrank == 0 ){
    MatrixXcd OvlpInv = Ovlp.inverse();
    MatrixXcd OinvHam = OvlpInv * Ham;
    ComplexEigenSolver<MatrixXcd> ges;
    ges.compute(OinvHam);
    cout<<setprecision(12)<<endl;
    cout << Ovlp/Ovlp(0,0)<<endl<<endl;
    cout << Ham/Ovlp(0,0)<<endl<<endl;
    cout << ges.eigenvalues().transpose()<<endl;
    cout << ges.eigenvectors().col(nstepsHalf)<<endl;
    cout << Ovlp.size()<<endl;
  }
  stats.gatherAndPrintStatistics(iTime);

}


// calculates energy of the imaginary time propagated wave function using direct sampling of exponentials
// w/o jastrow
void calcEnergyDirect(double enuc, MatrixXd& h1, MatrixXd& h1Mod, vector<MatrixXd>& chol)
{
  size_t norbs = Determinant::norbs;
  size_t nfields = chol.size();
  size_t nsweeps = schd.stochasticIter;
  size_t nsteps = schd.nsteps;
  size_t orthoSteps = schd.orthoSteps;
  double dt = schd.dt;
  vector<int> eneSteps = schd.eneSteps;

  // prep and init
  // this is for mean field subtraction
  MatrixXd hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair rhf;
  rhf.first = hf.block(0, 0, norbs, Determinant::nalpha);
  rhf.second = hf.block(0, 0, norbs, Determinant::nbeta);
  
  //vector<matPair> hsOperators;
  vector<complex<double>> mfShifts;
  matPair oneBodyOperator;
  //complex<double> mfConst = prepPropagatorHS(rhf, chol, hsOperators, oneBodyOperator);
  complex<double> mfConst = prepPropagatorHS(rhf, chol, mfShifts, oneBodyOperator);
  
  // this is the initial state
  matPair ref;
  hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  ref.first = hf.block(0, 0, norbs, Determinant::nalpha);
  ref.second = hf.block(0, 0, norbs, Determinant::nbeta);
 
  // this is the left state
  matPair refT;
  if (schd.hf == "rhf") {
    refT.first = ref.first.adjoint();
    refT.second = ref.second.adjoint();
  }
  if (schd.hf == "rhfc") {
    MatrixXcd hfc = MatrixXcd::Zero(norbs, norbs);
    readMat(hfc, "rhfC.txt");
    refT.first = hfc.block(0, 0, norbs, Determinant::nalpha).adjoint();
    refT.second = hfc.block(0, 0, norbs, Determinant::nbeta).adjoint();
  }
  else if (schd.hf == "uhf") {
    hf = MatrixXd::Zero(norbs, 2*norbs);
    readMat(hf, "uhf.txt");
    refT.first = hf.block(0, 0, norbs, Determinant::nalpha).adjoint();
    refT.second = hf.block(0, norbs, norbs, Determinant::nbeta).adjoint();
  }
  else if (schd.hf == "uhfc") {
    MatrixXcd hfc = MatrixXcd::Zero(norbs, 2*norbs);
    readMat(hfc, "uhfC.txt");
    refT.first = hfc.block(0, 0, norbs, Determinant::nalpha).adjoint();
    refT.second = hfc.block(0, norbs, norbs, Determinant::nbeta).adjoint();
  }
 
  matPair expOneBodyOperator;
  expOneBodyOperator.first =  (-dt * (h1Mod - oneBodyOperator.first) / 2.).exp();
  expOneBodyOperator.second = (-dt * (h1Mod - oneBodyOperator.second) / 2.).exp();

  // rotate cholesky vectors
  pair<vector<MatrixXcd>, vector<MatrixXcd>> rotChol;
  for (int i = 0; i < chol.size(); i++) {
    MatrixXcd rotUp = refT.first * chol[i];
    MatrixXcd rotDn = refT.second * chol[i];
    rotChol.first.push_back(rotUp);
    rotChol.second.push_back(rotDn);
  }
  vector<MatrixXf> cholf; 
  for (int i = 0; i < chol.size(); i++) cholf.push_back(chol[i].cast<float>()); 
  
  // Gaussian sampling
  // field values arranged right to left
  vector<VectorXd> fields;
  normal_distribution<double> normal(0., 1.);
  
  //matPair green;
  //calcGreensFunction(refT, ref, green);
  //complex<double> refEnergy = calcHamiltonianElement(green, enuc, h1, chol);
  complex<double> refEnergy = calcHamiltonianElement(refT.first, ref.first, enuc, h1, rotChol.first);
  //complex<double> refEnergy = calcHamiltonianElement(refT, ref, enuc, h1, chol);
  complex<double> ene0;
  if (schd.ene0Guess == 1.e10) ene0 = refEnergy;
  else ene0 = schd.ene0Guess;
  if (commrank == 0) {
    cout << "Initial state energy:  " << setprecision(8) << refEnergy << endl;
    cout << "Ground state energy guess:  " << ene0 << endl << endl; 
  }
  
  if (commrank == 0) cout << "Number of Cholesky vectors: " << chol.size() << endl;
  
  //vector<int> eneSteps = { int(0.2*nsteps) - 1, int(0.4*nsteps) - 1, int(0.6*nsteps) - 1, int(0.8*nsteps) - 1, int(nsteps - 1) };
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
      stats.gatherAndPrintStatistics(iTime);
    }
    matPair rn;
    rn = ref;
    VectorXd fields = VectorXd::Zero(nfields);
    complex<double> orthoFac = complex<double>(1., 0.);
    int eneStepCounter = 0;
    ArrayXcd numSampleA(nEneSteps), denomSampleA(nEneSteps);
    numSampleA.setZero(); denomSampleA.setZero();

    for (int n = 0; n < nsteps; n++) {
      // sampling
      double init = getTime();
      //MatrixXcd prop = MatrixXcd::Zero(norbs, norbs);
      //MatrixXd prop = MatrixXd::Zero(norbs, norbs);
      MatrixXf prop = MatrixXf::Zero(norbs, norbs);
      complex<double> shift(0., 0.);
      for (int i = 0; i < nfields; i++) {
        double field_n_i = normal(generator);
        prop.noalias() += float(field_n_i) * cholf[i];
        shift += field_n_i * mfShifts[i];
      }
      //prop = (sqrt(dt) * prop).exp();
      //rn.first = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * (expOneBodyOperator.first * (prop * (expOneBodyOperator.first * rn.first)));
      
      rn.first = expOneBodyOperator.first * rn.first;
      MatrixXcd propc = sqrt(dt) * complex<double>(0, 1.) * prop.cast<double>();
      MatrixXcd temp = rn.first;
      for (int i = 1; i < 10; i++) {
        temp = propc * temp / i;
        rn.first += temp;
      }
      rn.first = exp(-sqrt(dt) * shift / (Determinant::nalpha + Determinant::nbeta)) * exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * expOneBodyOperator.first * rn.first;

      if (Determinant::nalpha == Determinant::nbeta) rn.second = rn.first;
      else {
        //prop.second = (sqrt(dt) * prop.second).exp();
        rn.second = exp(-sqrt(dt) * shift / (Determinant::nalpha + Determinant::nbeta)) * exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nbeta)) * expOneBodyOperator.second * propc * expOneBodyOperator.second * rn.second;
      }

      propTime += getTime() - init;
      
      // orthogonalize for stability
      if (n!= 0 && n % orthoSteps == 0) {
        //cout << "ortho"<<endl;
        orthogonalize(rn, orthoFac);
      }

      // measure
      init = getTime();
      if (n == eneSteps[eneStepCounter]) {
        if (stats.converged[eneStepCounter] == -1) {//-1 means this time slice has not yet converged
          complex<double> overlap = orthoFac * (refT.first * rn.first).determinant() * (refT.second * rn.second).determinant();
          complex<double> numSample;
          if (Determinant::nalpha == Determinant::nbeta) numSample = overlap * calcHamiltonianElement(refT.first, rn.first, enuc, h1, rotChol.first);
          else numSample = overlap * calcHamiltonianElement(refT, rn, enuc, h1, rotChol);
          //if (Determinant::nalpha == Determinant::nbeta) numSample = overlap * calcHamiltonianElement(refT, rn, enuc, h1, chol);
          //numSample = overlap * (refEnergy + calcHamiltonianElement_sRI(refT, rn, refT, ref, enuc, h1, chol, richol)); 
          //cout << orthoFac<<" "<<overlap<<"  "<<numSample<<endl;
          numSampleA[eneStepCounter] = numSample;
          denomSampleA[eneStepCounter] = overlap;
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

  stats.gatherAndPrintStatistics(iTime);
  if (schd.printLevel > 10) stats.writeSamples();
}

// calculates energy of the imaginary time propagated wave function using direct sampling of exponentials
// w/o jastrow
void calcEnergyDirectGHF(double enuc, MatrixXd& h1, MatrixXd& h1Mod, vector<MatrixXd>& chol)
{
  size_t norbs = Determinant::norbs;
  size_t nfields = chol.size();
  size_t nsweeps = schd.stochasticIter;
  size_t nsteps = schd.nsteps;
  size_t orthoSteps = schd.orthoSteps;
  size_t nalpha = Determinant::nalpha;
  size_t nbeta = Determinant::nbeta;
  size_t nelec = nalpha+nbeta;
  double dt = schd.dt;
  vector<int> eneSteps = schd.eneSteps;

  // prep and init
  // this is for mean field subtraction
  MatrixXd hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair rhf;
  rhf.first = hf.block(0, 0, norbs, Determinant::nalpha);
  rhf.second = hf.block(0, 0, norbs, Determinant::nbeta);
  
  vector<matPair> hsOperators;
  matPair oneBodyOperator;
  complex<double> mfConst = prepPropagatorHS(rhf, chol, hsOperators, oneBodyOperator);
  
  // this is the initial state
  matPair ref;
  hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  ref.first = hf.block(0, 0, norbs, Determinant::nalpha);
  ref.second = hf.block(0, 0, norbs, Determinant::nbeta);
 
  // this is the left state
  MatrixXcd refAd, refT;
  {
    MatrixXcd hf = MatrixXcd::Zero(2*norbs, 2*norbs);
    readMat(hf, "ghf.txt");
    refAd = 1.*hf.block(0,0,2*norbs, nelec).adjoint();// + 0.01*MatrixXcd::Random(2*norbs, nelec);
    refT = refAd.conjugate();
  }

// rotate cholesky vectors
  pair<vector<MatrixXcd>, vector<MatrixXcd>> rotCholAd;
  pair<vector<MatrixXcd>, vector<MatrixXcd>> rotCholT;
  for (int i = 0; i < chol.size(); i++) {
    MatrixXcd rotUp = refT.block(0,0,nelec,norbs) * chol[i];
    MatrixXcd rotDn = refT.block(0,norbs,nelec,norbs)  * chol[i];
    rotCholT.first.push_back(rotUp);
    rotCholT.second.push_back(rotDn);

    rotUp = refAd.block(0,0,nelec,norbs) * chol[i];
    rotDn = refAd.block(0,norbs,nelec,norbs)  * chol[i];
    rotCholAd.first.push_back(rotUp);
    rotCholAd.second.push_back(rotDn);

  }
  
  matPair expOneBodyOperator;
  expOneBodyOperator.first =  (-dt * (h1Mod - oneBodyOperator.first) / 2.).exp();
  expOneBodyOperator.second = (-dt * (h1Mod - oneBodyOperator.second) / 2.).exp();

   vector<VectorXd> fields;
  normal_distribution<double> normal(0., 1.);
  
  //matPair green;
  //calcGreensFunction(refAd, ref, green);
  //complex<double> refEnergy = calcHamiltonianElement(green, enuc, h1, chol);
  complex<double> ovlp1, ovlp2;
  complex<double> refEnergy1 = calcHamiltonianElement(refAd, ref, enuc, h1, chol, ovlp1);
  complex<double> refEnergy2 = calcHamiltonianElement(refT, ref, enuc, h1, chol, ovlp2);
  complex<double> refEnergy = (refEnergy1*ovlp1 + refEnergy2*ovlp2)/(ovlp1+ovlp2);

  complex<double> ene0;
  if (schd.ene0Guess == 1.e10) ene0 = refEnergy;
  else ene0 = schd.ene0Guess;
  if (commrank == 0) {
    cout << "Initial state energy:  " << setprecision(8) << refEnergy << endl;
    cout << "Ground state energy guess:  " << ene0 << endl << endl; 
  }
  
  //vector<int> eneSteps = { int(0.2*nsteps) - 1, int(0.4*nsteps) - 1, int(0.6*nsteps) - 1, int(0.8*nsteps) - 1, int(nsteps - 1) };
  int nEneSteps = eneSteps.size();
  DQMCStatistics stats(nEneSteps);
  auto iterTime = getTime();
  double propTime = 0., eneTime = 0., qrTime = 0.;
  if (commrank == 0) cout << "Starting sampling sweeps\n";
  ArrayXd iTime(nEneSteps);
  for (int i = 0; i < nEneSteps; i++) iTime(i) = dt * (eneSteps[i] + 1);

  for (int sweep = 0; sweep < nsweeps; sweep++) {
    if (stats.isConverged()) break;
    if (sweep != 0 && sweep % (schd.printFrequency) == 0) {
      if (commrank == 0) {
        cout << "Sweep steps: " << sweep << endl << "Total walltime: " << setprecision(6) << getTime() - iterTime << " s\n";
        cout << "\nPropagation time:  " << propTime << " s\n";
        cout << "Energy evaluation time:  " << eneTime << " s\n\n";
      }
      stats.gatherAndPrintStatistics(iTime);
    }

    matPair rn;
    rn = ref;
    VectorXd fields = VectorXd::Zero(nfields);
    complex<double> orthoFac = complex<double>(1., 0.);
    int eneStepCounter = 0;
    ArrayXcd numSampleA(nEneSteps), denomSampleA(nEneSteps);
    numSampleA.setZero(); denomSampleA.setZero();
    for (int n = 0; n < nsteps; n++) {
      // sampling
      double init = getTime();
      matPair prop;
      prop.first = MatrixXcd::Zero(norbs, norbs);
      prop.second = MatrixXcd::Zero(norbs, norbs);
      for (int i = 0; i < nfields; i++) {
        double field_n_i = normal(generator);
        prop.first += field_n_i * hsOperators[i].first;
        //prop.second += field_n_i * hsOperators[i].second;
      }
      prop.first = (sqrt(dt) * prop.first).exp();
      rn.first = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * (expOneBodyOperator.first * (prop.first * (expOneBodyOperator.first * rn.first)));
      if (Determinant::nalpha == Determinant::nbeta) rn.second = rn.first;
      else {
        //prop.second = (sqrt(dt) * prop.second).exp();
        rn.second = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nbeta)) * expOneBodyOperator.second * prop.first * expOneBodyOperator.second * rn.second;
      }

      propTime += getTime() - init;
      
      // orthogonalize for stability
      if (n % orthoSteps == 0) {
        orthogonalize(rn, orthoFac);
      }

      // measure
      init = getTime();
      if (n == eneSteps[eneStepCounter]) {
        if (stats.converged[eneStepCounter] == -1) {//-1 means this time slice has not yet converged
          complex<double> num, den;
          complex<double> overlapAd, overlapT;
          //complex<double> numSample = calcHamiltonianElement(refAd, rn, enuc, h1, chol, overlapAd);
          complex<double> numSample = calcHamiltonianElement(refAd, rn, enuc, h1, rotCholAd, overlapAd);
          overlapAd *= orthoFac;
          numSample *= overlapAd;
          num = numSample; den = overlapAd;

          //numSample = calcHamiltonianElement(refT, rn, enuc, h1, chol, overlapT);
          numSample = calcHamiltonianElement(refT, rn, enuc, h1, rotCholT, overlapT);
          overlapT *= orthoFac;
          numSample *= overlapT;
          num += numSample; den += overlapT;

          numSampleA[eneStepCounter] = num;
          denomSampleA[eneStepCounter] = den;
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

  stats.gatherAndPrintStatistics(iTime);
  if (schd.printLevel > 10) stats.writeSamples();
}



// calculates energy of the imaginary time propagated wave function using direct sampling of exponentials
// w/o jastrow
void findDtCorrelatedSamplingGHF(double enuc, MatrixXd& h1, MatrixXd& h1Mod, vector<MatrixXd>& chol)
{
  size_t norbs = Determinant::norbs;
  size_t nfields = chol.size();
  size_t nsweeps = schd.stochasticIter;
  size_t nsteps = schd.nsteps;
  size_t orthoSteps = schd.orthoSteps;
  size_t nalpha = Determinant::nalpha;
  size_t nbeta = Determinant::nbeta;
  size_t nelec = nalpha+nbeta;
  //double dt = schd.dt;
  vector<int> eneSteps = schd.eneSteps;

  // prep and init
  // this is for mean field subtraction
  MatrixXd hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair rhf;
  rhf.first = hf.block(0, 0, norbs, Determinant::nalpha);
  rhf.second = hf.block(0, 0, norbs, Determinant::nbeta);
  
  vector<matPair> hsOperators;
  matPair oneBodyOperator;
  complex<double> mfConst = prepPropagatorHS(rhf, chol, hsOperators, oneBodyOperator);
  
  // this is the initial state
  matPair ref;
  hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  ref.first = hf.block(0, 0, norbs, Determinant::nalpha);
  ref.second = hf.block(0, 0, norbs, Determinant::nbeta);
 
  // this is the left state
  MatrixXcd refAd, refT;
  {
    MatrixXcd hf = MatrixXcd::Zero(2*norbs, 2*norbs);
    readMat(hf, "ghf.txt");
    refAd = 1.*hf.block(0,0,2*norbs, nelec).adjoint();// + 0.01*MatrixXcd::Random(2*norbs, nelec);
    refT = refAd.conjugate();
  }
  int nCorrelatedT = 3;
  vector<double> dt(3, schd.dt);

  //largest to smallest with factors of 2
  for (int j=0; j<nCorrelatedT; j++) {
    dt[j] = schd.dt/pow(2.,j);
    cout <<j<<"  "<<dt[j]<<"  "<<schd.dt<<endl;
  }

  vector<matPair> expOneBodyOperator(nCorrelatedT);
  for (int j=0; j<nCorrelatedT; j++) {
    expOneBodyOperator[j].first =  (-dt[j] * (h1Mod - oneBodyOperator.first) / 2.).exp();
    expOneBodyOperator[j].second = (-dt[j] * (h1Mod - oneBodyOperator.second) / 2.).exp();
  }

  vector<VectorXd> fields;
  normal_distribution<double> normal(0., 1.);
  
  //matPair green;
  //calcGreensFunction(refAd, ref, green);
  //complex<double> refEnergy = calcHamiltonianElement(green, enuc, h1, chol);
  complex<double> ovlp1, ovlp2;
  complex<double> refEnergy1 = calcHamiltonianElement(refAd, ref, enuc, h1, chol, ovlp1);
  complex<double> refEnergy2 = calcHamiltonianElement(refT, ref, enuc, h1, chol, ovlp2);
  complex<double> refEnergy = (refEnergy1*ovlp1 + refEnergy2*ovlp2)/(ovlp1+ovlp2);

  complex<double> ene0;
  if (schd.ene0Guess == 1.e10) ene0 = refEnergy;
  else ene0 = schd.ene0Guess;
  if (commrank == 0) {
    cout << "Initial state energy:  " << refEnergy << endl;
    cout << "Ground state energy guess:  " << ene0 << endl << endl; 
  }
  
  //vector<int> eneSteps = { int(0.2*nsteps) - 1, int(0.4*nsteps) - 1, int(0.6*nsteps) - 1, int(0.8*nsteps) - 1, int(nsteps - 1) };

  int nEneSteps = eneSteps.size();
  vector<DQMCStatistics> statsVec(nCorrelatedT, DQMCStatistics(nEneSteps)); //a separate statistics class for each dT
  auto iterTime = getTime();
  double propTime = 0., eneTime = 0., qrTime = 0.;
  if (commrank == 0) cout << "Starting sampling sweeps\n";
  ArrayXd iTime(nEneSteps);
  for (int i = 0; i < nEneSteps; i++) iTime(i) = dt[0] * (eneSteps[i] + 1);

  for (int sweep = 0; sweep < nsweeps; sweep++) {
    if (sweep != 0 && sweep % (schd.printFrequency) == 0) {
      if (commrank == 0) {
        cout <<"sweep steps: "<< sweep <<endl<<"Total walltime: " << getTime() - iterTime << " s\n";
        cout << "\nPropagation time:  " << propTime << " s\n";
        cout << "Energy evaluation time:  " << eneTime << " s\n\n";
      }
      for (int j=0; j<nCorrelatedT; j++) 
        statsVec[j].gatherAndPrintStatistics(iTime);
    }

    vector<matPair> rn(nCorrelatedT, ref);
    VectorXd fields = VectorXd::Zero(nfields);
    vector<complex<double>> orthoFac(nCorrelatedT, complex<double>(1., 0.));
    int eneStepCounter = 0;

    vector<ArrayXcd> numSampleA(nCorrelatedT, ArrayXcd(nEneSteps)), denomSampleA(nCorrelatedT, ArrayXcd(nEneSteps));
    for (int j=0; j<nCorrelatedT; j++) {
      numSampleA[j].setZero(); denomSampleA[j].setZero();
    }

    for (int n = 0; n < nsteps; n++) {
      // sampling
      double init = getTime();
      vector<matPair> prop(nCorrelatedT);

      MatrixXcd hsProp = MatrixXcd::Zero(norbs, norbs);
      for (int i = 0; i < nfields; i++) {
        double field_n_i = normal(generator);
        hsProp += field_n_i * hsOperators[i].first;
        //prop.second += field_n_i * hsOperators[i].second;
      }

      //for each dt apply the Hamiltonian as many times as needed such that total time dt[0] is propogated
      for (int j=0; j<nCorrelatedT; j++) {
        int nterms = pow(2,j);
        prop[j].first = exp((ene0 - enuc - mfConst) * dt[j] / (2. * Determinant::nalpha)) * (expOneBodyOperator[j].first * ( (sqrt(dt[j]/nterms) * hsProp).exp() * expOneBodyOperator[j].first));

        for (int napply = 0; napply < nterms; napply++) {
          rn[j].first =  prop[j].first * rn[j].first;
        }

        if (Determinant::nalpha == Determinant::nbeta) rn[j].second = rn[j].first;
        else {
          for (int napply = 0; napply < pow(2, j); napply++) {
            rn[j].second = prop[j].first * rn[j].second;
          }
        }
        //cout << rn[j].first<<endl<<endl<<endl;
      }
      //cout << n<<"  "<<(prop[0].first - prop[1].first*prop[1].first).norm()<<endl;
      //exit(0);

      propTime += getTime() - init;
      
      // orthogonalize for stability
      if (n % orthoSteps == 0) {
        for (int j=0; j<nCorrelatedT; j++)
          orthogonalize(rn[j], orthoFac[j]);
      }

      // measure
      init = getTime();
      if (n == eneSteps[eneStepCounter]) {
        for (int j=0; j<nCorrelatedT; j++) {
          complex<double> num, den;
          complex<double> overlapAd, overlapT;
          complex<double> numSample = calcHamiltonianElement(refAd, rn[j], enuc, h1, chol, overlapAd);
          overlapAd *= orthoFac[j];
          numSample *= overlapAd;
          num = numSample; den = overlapAd;

          numSample = calcHamiltonianElement(refT, rn[j], enuc, h1, chol, overlapT);
          overlapT *= orthoFac[j];
          numSample *= overlapT;
          num += numSample; den += overlapT;

          //cout << j<<" e "<<num<<"  "<<den<<"  "<<num/den<<"  "<<orthoFac[j]<<"  "<<nCorrelatedT<<endl;
          numSampleA[j][eneStepCounter] = num;
          denomSampleA[j][eneStepCounter] = den;
        }
        eneStepCounter++;
        //exit(0);
      }
      
      eneTime += getTime() - init;
    }
    for (int j=0; j<nCorrelatedT; j++) {
      statsVec[j].addSamples(numSampleA[j], denomSampleA[j]);
      //cout << j<<"  "<<numSampleA[j][0]<<"  "<<denomSampleA[j][0]<<endl;
    }
  }

  if (commrank == 0) {
    cout << "\nPropagation time:  " << propTime << " s\n";
    cout << "Energy evaluation time:  " << eneTime << " s\n\n";
  }

  for (int j=0; j<nCorrelatedT; j++)
    statsVec[j].gatherAndPrintStatistics(iTime);
  //if (schd.printLevel > 10) stats.writeSamples();
}


// calculates mixed energy estimator of the imaginary time propagated wave function
// w jastrow
void calcEnergyJastrowDirect(double enuc, MatrixXd& h1, MatrixXd& h1Mod, vector<MatrixXd>& chol)
{
  size_t norbs = Determinant::norbs;
  size_t nfields = chol.size();
  size_t nsweeps = schd.stochasticIter;
  size_t nsteps = schd.nsteps;
  size_t orthoSteps = schd.orthoSteps;
  double dt = schd.dt;
  vector<int> eneSteps = schd.eneSteps;

  // prep and init
  // this is for mean field subtraction
  MatrixXd hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair rhf;
  rhf.first = hf.block(0, 0, norbs, Determinant::nalpha);
  rhf.second = hf.block(0, 0, norbs, Determinant::nbeta);
  
  // prop
  //vector<matPair> hsOperators;
  vector<complex<double>> mfShifts;
  matPair oneBodyOperator;
  //complex<double> mfConst = prepPropagatorHS(rhf, chol, hsOperators, oneBodyOperator);
  complex<double> mfConst = prepPropagatorHS(rhf, chol, mfShifts, oneBodyOperator);
  
  // this is the left state
  matPair refT;
  if (schd.hf == "rhf") {
     MatrixXcd hfc = MatrixXcd::Zero(norbs, norbs);
    readMat(hfc, "rhf.txt");
    refT.first = hfc.block(0, 0, norbs, Determinant::nalpha).adjoint();
    refT.second = hfc.block(0, 0, norbs, Determinant::nalpha).adjoint();
  }
  if (schd.hf == "rhfc") {
    MatrixXcd hfc = MatrixXcd::Zero(norbs, norbs);
    readMat(hfc, "rhfC.txt");
    refT.first = hfc.block(0, 0, norbs, Determinant::nalpha).adjoint();
    refT.second = hfc.block(0, 0, norbs, Determinant::nbeta).adjoint();
  }
  else if (schd.hf == "uhf") {
    hf = MatrixXd::Zero(norbs, 2*norbs);
    readMat(hf, "uhf.txt");
    refT.first = hf.block(0, 0, norbs, Determinant::nalpha).adjoint();
    refT.second = hf.block(0, norbs, norbs, Determinant::nbeta).adjoint();
  }
  else if (schd.hf == "uhfc") {
    MatrixXcd hfc = MatrixXcd::Zero(norbs, 2*norbs);
    readMat(hfc, "uhfC.txt");
    refT.first = hfc.block(0, 0, norbs, Determinant::nalpha).adjoint();
    refT.second = hfc.block(0, norbs, norbs, Determinant::nbeta).adjoint();
  }
  
  matPair expOneBodyOperator;
  expOneBodyOperator.first =  (-dt * (h1Mod - oneBodyOperator.first) / 2.).exp();
  expOneBodyOperator.second = (-dt * (h1Mod - oneBodyOperator.second) / 2.).exp();

  // this is the initial state
  // NB: refT need not be ref.adjoint()
  matPair ref;
  hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "ref.txt");
  ref.first = hf.block(0, 0, norbs, Determinant::nalpha);
  ref.second = hf.block(0, 0, norbs, Determinant::nbeta);

  // Jastrow HS
  matPair refAct;
  size_t numActOrbs;
  if (schd.nciAct < 0) numActOrbs = norbs;
  else numActOrbs = schd.nciAct;
  refAct.first = ref.first.block(0, 0, numActOrbs, Determinant::nalpha);
  refAct.second = ref.second.block(0, 0, numActOrbs, Determinant::nbeta);
  vector<vecPair> jhsOperators;
  vecPair joneBodyOperator;
  complex<double> jmfConst = prepJastrowHS(refAct, jhsOperators, joneBodyOperator);
  size_t jnfields = jhsOperators.size();


  // jref obtained by acting on refAct
  vecPair jexpOneBodyOperator;
  jexpOneBodyOperator.first = joneBodyOperator.first.array().exp();
  jexpOneBodyOperator.second = joneBodyOperator.second.array().exp();
  matPair jref;
  jref.first = jexpOneBodyOperator.first.asDiagonal() * refAct.first;
  jref.second = jexpOneBodyOperator.second.asDiagonal() * refAct.second;

  // rotate cholesky vectors
  pair<vector<MatrixXcd>, vector<MatrixXcd>> rotChol;
  for (int i = 0; i < chol.size(); i++) {
    MatrixXcd rotUp = refT.first * chol[i];
    MatrixXcd rotDn = refT.second * chol[i];
    rotChol.first.push_back(rotUp);
    rotChol.second.push_back(rotDn);
  }
  
  // Gaussian sampling
  // field values arranged right to left
  vector<VectorXd> fields;
  normal_distribution<double> normal(0., 1.);
  

  complex<double> refEnergy = calcHamiltonianElement(refT, ref, enuc, h1, rotChol);
  complex<double> ene0;
  if (schd.ene0Guess == 1.e10) ene0 = refEnergy;
  else ene0 = schd.ene0Guess;
  if (commrank == 0) {
    cout << "Initial state energy:  " << refEnergy << endl;
    cout << "Ground state energy guess:  " << ene0 << endl << endl; 
  }
 
  
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
      stats.gatherAndPrintStatistics(iTime);
    }
    
    matPair rn;
    rn = jref;

    // apply the jastrow on jref
    vecPair jprop;
    jprop.first = VectorXcd::Zero(numActOrbs);
    jprop.second = VectorXcd::Zero(numActOrbs);
    for (int i = 0; i < jnfields; i++) {
      double jfields = normal(generator);
      jprop.first += jfields * jhsOperators[i].first;
      jprop.second += jfields * jhsOperators[i].second;
    }
    rn.first = exp(jmfConst / (2. * Determinant::nalpha)) * jprop.first.array().exp().matrix().asDiagonal() * rn.first;
    rn.second = exp(jmfConst / (2. * Determinant::nbeta)) * jprop.second.array().exp().matrix().asDiagonal() * rn.second;

    VectorXd fields = VectorXd::Zero(nfields);
    complex<double> orthoFac = complex<double>(1., 0.);
    int eneStepCounter = 0;
    ArrayXcd numSampleA(nEneSteps), denomSampleA(nEneSteps);
    numSampleA.setZero(); denomSampleA.setZero();

    for (int n = 0; n < nsteps; n++) {
      // sampling
      double init = getTime();
      MatrixXd prop = MatrixXd::Zero(norbs, norbs);
      complex<double> shift(0., 0.);
      for (int i = 0; i < nfields; i++) {
        double field_n_i = normal(generator);
        prop.noalias() += field_n_i * chol[i];
        shift += field_n_i * mfShifts[i];
      }
      
      rn.first = expOneBodyOperator.first * rn.first;
      MatrixXcd propc = sqrt(dt) * complex<double>(0, 1.) * prop;
      MatrixXcd temp = rn.first;
      for (int i = 1; i < 10; i++) {
        temp = propc * temp / i;
        rn.first += temp;
      }
      rn.first = exp(-sqrt(dt) * shift / (Determinant::nalpha + Determinant::nbeta)) * exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * expOneBodyOperator.first * rn.first;
      
      rn.second = expOneBodyOperator.second * rn.second;
      temp = rn.second;
      for (int i = 1; i < 10; i++) {
        temp = propc * temp / i;
        rn.second += temp;
      }
      rn.second = exp(-sqrt(dt) * shift / (Determinant::nalpha + Determinant::nbeta)) * exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nbeta)) * expOneBodyOperator.second * rn.second;

      propTime += getTime() - init;
      
      // orthogonalize for stability
      if (n!= 0 && n % orthoSteps == 0) {
        orthogonalize(rn, orthoFac);
      }

      // measure
      init = getTime();
      if (n == eneSteps[eneStepCounter]) {
        if (stats.converged[eneStepCounter] == -1) {//-1 means this time slice has not yet converged
          complex<double> overlap = orthoFac * (refT.first * rn.first).determinant() * (refT.second * rn.second).determinant();
          complex<double> numSample;
          numSample = overlap * calcHamiltonianElement(refT, rn, enuc, h1, rotChol);
          numSampleA[eneStepCounter] = numSample;
          denomSampleA[eneStepCounter] = overlap;
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

  stats.gatherAndPrintStatistics(iTime);
  if (schd.printLevel > 10) stats.writeSamples();
}


void calcEnergyCCSDMultiSlaterDirect(double enuc, MatrixXd& h1, MatrixXd& h1Mod, vector<MatrixXd>& chol)
{
  size_t norbs = Determinant::norbs;
  size_t nfields = chol.size();
  size_t nsweeps = schd.stochasticIter;
  size_t nsteps = schd.nsteps;
  size_t orthoSteps = schd.orthoSteps;
  double dt = schd.dt;
  vector<int> eneSteps = schd.eneSteps;

  // prep and init
  // this is for mean field subtraction
  MatrixXd hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair rhf;
  rhf.first = hf.block(0, 0, norbs, Determinant::nalpha);
  rhf.second = hf.block(0, 0, norbs, Determinant::nbeta);
  
  // cc operators
  vector<MatrixXcd> cchsOperators;
  MatrixXcd cconeBodyOperator;
  prepCCHS(rhf.first, cchsOperators, cconeBodyOperator);
  size_t ccnfields = cchsOperators.size();

  // prop
  vector<complex<double>> mfShifts;
  matPair oneBodyOperator;
  complex<double> mfConst = prepPropagatorHS(rhf, chol, mfShifts, oneBodyOperator);
  
  matPair expOneBodyOperator;
  expOneBodyOperator.first =  (-dt * (h1Mod - oneBodyOperator.first) / 2.).exp();
  expOneBodyOperator.second = (-dt * (h1Mod - oneBodyOperator.second) / 2.).exp();
  
  // this is the trial state used to measure energy
  matPair ref;
  hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  ref.first = hf.block(0, 0, norbs, Determinant::nalpha);
  ref.second = hf.block(0, 0, norbs, Determinant::nbeta);
  matPair refT;
  refT.first = ref.first.adjoint();
  refT.second = ref.second.adjoint();
  
  string fname = schd.determinantFile;
  std::array<std::vector<int>, 2> refDet; 
  std::array<std::vector<std::array<Eigen::VectorXi, 2>>, 2> ciExcitations;
  std::vector<double> ciParity; 
  std::vector<double> ciCoeffs;
  // if using text file name it "dets", if using binary use anything other than dets
  if (fname == "dets") readDeterminants(fname, refDet, ciExcitations, ciParity, ciCoeffs);
  else readDeterminantsBinary(fname, refDet, ciExcitations, ciParity, ciCoeffs);
  
  // this is the initial state
  MatrixXcd ccexpOneBodyOperator = cconeBodyOperator.exp();
  MatrixXcd ccref = ccexpOneBodyOperator * ref.first;
  
  //complex<double> refEnergy = calcHamiltonianElement(refT, ref, enuc, h1, rotChol);
  auto refOverlapHam = calcHamiltonianElement(refT, ciExcitations, ciParity, ciCoeffs, ref, enuc, h1, chol);
  complex<double> refEnergy = refOverlapHam.second / refOverlapHam.first;
  complex<double> ene0;
  if (schd.ene0Guess == 1.e10) ene0 = refEnergy;
  else ene0 = schd.ene0Guess;
  if (commrank == 0) {
    cout << "Initial state energy:  " << setprecision(8) << refEnergy << endl;
    cout << "Ground state energy guess:  " << ene0 << endl << endl; 
  }
  
  int nchol = chol.size();
  complex<double> delta(0., 0.);
  if (commrank == 0) cout << "Number of Cholesky vectors: " << nchol << endl;
  vector<int> ncholVec = { int(0.3 * chol.size()), int(0.4 * chol.size()), int(0.5 * chol.size()), int(0.6 * chol.size()), int(0.7 * chol.size()) };
  for (int i = 0; i < ncholVec.size(); i++) {
    //complex<double> trefEnergy = calcHamiltonianElement(refT, ref, enuc, h1, rotChol, ncholVec[i]);
    auto trefOverlapHam = calcHamiltonianElement(refT, ciExcitations, ciParity, ciCoeffs, ref, enuc, h1, chol, ncholVec[i]);
    complex<double> trefEnergy = trefOverlapHam.second / trefOverlapHam.first;
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
  
  vector<MatrixXf> cholf; 
  for (int i = 0; i < chol.size(); i++) cholf.push_back(chol[i].cast<float>()); 
 
  // Gaussian sampling
  // field values arranged right to left
  normal_distribution<double> normal(0., 1.);
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
    
    MatrixXcd rn;
    complex<double> orthoFac = complex<double> (1., 0.);
    int eneStepCounter = 0;

    // cc sampling
    MatrixXcd ccprop = MatrixXcd::Zero(norbs, norbs);
    for (int i = 0; i < ccnfields; i++) {
      double ccfield_i = normal(generator);
      ccprop.noalias() += ccfield_i * cchsOperators[i];
    }
    rn = ccprop.exp() * ccref;

    // prop sampling
    ArrayXcd numSampleA(nEneSteps), denomSampleA(nEneSteps);
    numSampleA.setZero(); denomSampleA.setZero();
    for (int n = 0; n < nsteps; n++) {
      // prop
      double init = getTime();
      //MatrixXd prop = MatrixXd::Zero(norbs, norbs);
      MatrixXf prop = MatrixXf::Zero(norbs, norbs);
      complex<double> shift(0., 0.);
      for (int i = 0; i < nfields; i++) {
        double field_n_i = normal(generator);
        prop.noalias() += float(field_n_i) * cholf[i];
        shift += field_n_i * mfShifts[i];
      }
      rn = expOneBodyOperator.first * rn;
      MatrixXcd propc = sqrt(dt) * complex<double>(0, 1.) * prop.cast<double>();
      MatrixXcd temp = rn;
      for (int i = 1; i < 10; i++) {
        temp = propc * temp / i;
        rn += temp;
      }
      rn = exp(-sqrt(dt) * shift / (Determinant::nalpha + Determinant::nbeta)) * exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * expOneBodyOperator.first * rn;
      
      propTime += getTime() - init;
      
      // orthogonalize for stability
      if (n % orthoSteps == 0 && n != 0) {
        orthogonalize(rn, orthoFac);
      }

      // measure
      init = getTime();
      if (n == eneSteps[eneStepCounter] ) {
        if (stats.converged[eneStepCounter] == -1) {//-1 means this time slice has not yet converged
          matPair rnPair;
          rnPair.first = rn;
          rnPair.second = rn;
          auto overlapHam = calcHamiltonianElement(refT, ciExcitations, ciParity, ciCoeffs, rnPair, enuc, h1, chol, nchol, schd.nciAct, schd.nciCore);
          complex<double> overlap = orthoFac * orthoFac * overlapHam.first;
          complex<double> numSample = orthoFac * orthoFac * overlapHam.second;
          numSampleA[eneStepCounter] = numSample;
          denomSampleA[eneStepCounter] = overlap;
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
}


void calcEnergyCCSDDirectVariational(double enuc, MatrixXd& h1, MatrixXd& h1Mod, vector<MatrixXd>& chol)
{
  size_t norbs = Determinant::norbs;
  size_t nfields = chol.size();
  size_t nsweeps = schd.stochasticIter;
  size_t nsteps = schd.nsteps;
  size_t orthoSteps = schd.orthoSteps;
  double dt = schd.dt;
  vector<int> eneSteps = schd.eneSteps;

  // prep and init
  // this is for mean field subtraction
  MatrixXd hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair rhf;
  rhf.first = hf.block(0, 0, norbs, Determinant::nalpha);
  rhf.second = hf.block(0, 0, norbs, Determinant::nbeta);
  
  // cc operators
  vector<MatrixXcd> cchsOperators;
  MatrixXcd cconeBodyOperator;
  prepCCHS(rhf.first, cchsOperators, cconeBodyOperator);
  size_t ccnfields = cchsOperators.size();

  // prop
  vector<complex<double>> mfShifts;
  matPair oneBodyOperator;
  complex<double> mfConst = prepPropagatorHS(rhf, chol, mfShifts, oneBodyOperator);
  
  matPair expOneBodyOperator;
  expOneBodyOperator.first =  (-dt * (h1Mod - oneBodyOperator.first) / 2.).exp();
  expOneBodyOperator.second = (-dt * (h1Mod - oneBodyOperator.second) / 2.).exp();
  
  // this is the trial state used to measure energy
  matPair ref;
  hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  ref.first = hf.block(0, 0, norbs, Determinant::nalpha);
  ref.second = hf.block(0, 0, norbs, Determinant::nbeta);
  matPair refT;
  refT.first = ref.first.adjoint();
  refT.second = ref.second.adjoint();
  
  // this is the initial state
  MatrixXcd ccexpOneBodyOperator = cconeBodyOperator.exp();
  MatrixXcd ccref = ccexpOneBodyOperator * ref.first;
  MatrixXcd ccrefT = ccref.adjoint();

  complex<double> refEnergy = calcHamiltonianElement(refT, ref, enuc, h1, chol);
  complex<double> ene0;
  if (schd.ene0Guess == 1.e10) ene0 = refEnergy;
  else ene0 = schd.ene0Guess;
  if (commrank == 0) {
    cout << "Initial state energy:  " << setprecision(8) << refEnergy << endl;
    cout << "Ground state energy guess:  " << ene0 << endl << endl; 
  }
  
  //int nchol = chol.size();
  //complex<double> delta(0., 0.);
  //if (commrank == 0) cout << "Number of Cholesky vectors: " << nchol << endl;
  //vector<int> ncholVec = { int(0.3 * chol.size()), int(0.4 * chol.size()), int(0.5 * chol.size()), int(0.6 * chol.size()), int(0.7 * chol.size()) };
  //for (int i = 0; i < ncholVec.size(); i++) {
  //  complex<double> trefEnergy = calcHamiltonianElement(refT, ref, enuc, h1, rotChol, ncholVec[i]);
  //  if (abs(refEnergy - trefEnergy) < schd.choleskyThreshold) {
  //    nchol = ncholVec[i];
  //    delta = refEnergy - trefEnergy;
  //    if (commrank == 0) {
  //      cout << "Using truncated Cholesky with " << nchol << " vectors\n";
  //      cout << "Initial state energy with truncated Cholesky:  " << trefEnergy << endl << endl;
  //    }
  //    break;
  //  }
  //}
 
  // Gaussian sampling
  // field values arranged right to left
  normal_distribution<double> normal(0., 1.);
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
      stats.gatherAndPrintStatistics(iTime);
    }
    
    MatrixXcd rn;
    complex<double> orthoFac = complex<double> (1., 0.);

    // right cc sampling
    MatrixXcd ccprop = MatrixXcd::Zero(norbs, norbs);
    for (int i = 0; i < ccnfields; i++) {
      double ccfield_i = normal(generator);
      ccprop.noalias() += ccfield_i * cchsOperators[i];
    }
    rn = ccprop.exp() * ccref;

    // prop sampling
    int eneStepCounter = 0;
    ArrayXcd numSampleA(nEneSteps), denomSampleA(nEneSteps);
    numSampleA.setZero(); denomSampleA.setZero();
    for (int n = 0; n < nsteps; n++) {
      double init = getTime();
      MatrixXd prop = MatrixXd::Zero(norbs, norbs);
      complex<double> shift(0., 0.);
      for (int i = 0; i < nfields; i++) {
        double field_n_i = normal(generator);
        prop.noalias() += field_n_i * chol[i];
        shift += field_n_i * mfShifts[i];
      }
      rn = expOneBodyOperator.first * rn;
      MatrixXcd propc = sqrt(dt) * complex<double>(0, 1.) * prop;
      MatrixXcd temp = rn;
      for (int i = 1; i < 10; i++) {
        temp = propc * temp / i;
        rn += temp;
      }
      rn = exp(-sqrt(dt) * shift / (Determinant::nalpha + Determinant::nbeta)) * exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * expOneBodyOperator.first * rn;
      
      // orthogonalize for stability
      if (n % orthoSteps == 0 && n != 0) {
        orthogonalize(rn, orthoFac);
      }
      
      propTime += getTime() - init;

      // measure
      init = getTime();
      if (n == eneSteps[eneStepCounter] ) {
        if (stats.converged[eneStepCounter] == -1) {//-1 means this time slice has not yet converged
          // left cc sampling
          complex<double> ccOverlap(0., 0.), ccLocalEnergy(0., 0.);
          int nccSamples = schd.numJastrowSamples;
          for (int n = 0; n < nccSamples; n++) {
            MatrixXcd ccpropT = MatrixXcd::Zero(norbs, norbs);
            for (int i = 0; i < ccnfields; i++) {
              double ccfield_i = normal(generator);
              ccpropT.noalias() += ccfield_i * cchsOperators[i].adjoint();
            }
            //MatrixXcd lnT = ccrefT * ccpropT.exp();
            MatrixXcd lnT = ccrefT;
            lnT += lnT * ccpropT;
            
            complex<double> overlapSample = orthoFac * orthoFac * (lnT * rn).determinant() * (lnT * rn).determinant();
            ccOverlap += overlapSample;
            matPair lnP, rnP;
            lnP.first = lnT; lnP.second = lnT;
            rnP.first = rn; rnP.second = rn;
            ccLocalEnergy += overlapSample * calcHamiltonianElement(lnP, rnP, enuc, h1, chol);
          }
          numSampleA[eneStepCounter] = ccLocalEnergy / nccSamples;
          denomSampleA[eneStepCounter] = ccOverlap / nccSamples;
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

  stats.gatherAndPrintStatistics(iTime);
  if (schd.printLevel > 10) stats.writeSamples();
}


void calcEnergyCCSDDirect(double enuc, MatrixXd& h1, MatrixXd& h1Mod, vector<MatrixXd>& chol)
{
  size_t norbs = Determinant::norbs;
  size_t nfields = chol.size();
  size_t nsweeps = schd.stochasticIter;
  size_t nsteps = schd.nsteps;
  size_t orthoSteps = schd.orthoSteps;
  double dt = schd.dt;
  vector<int> eneSteps = schd.eneSteps;

  // prep and init
  // this is for mean field subtraction
  MatrixXd hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair rhf;
  rhf.first = hf.block(0, 0, norbs, Determinant::nalpha);
  rhf.second = hf.block(0, 0, norbs, Determinant::nbeta);
  
  // cc operators
  vector<MatrixXcd> cchsOperators;
  MatrixXcd cconeBodyOperator;
  prepCCHS(rhf.first, cchsOperators, cconeBodyOperator);
  size_t ccnfields = cchsOperators.size();

  // prop
  vector<complex<double>> mfShifts;
  matPair oneBodyOperator;
  complex<double> mfConst = prepPropagatorHS(rhf, chol, mfShifts, oneBodyOperator);
  
  matPair expOneBodyOperator;
  expOneBodyOperator.first =  (-dt * (h1Mod - oneBodyOperator.first) / 2.).exp();
  expOneBodyOperator.second = (-dt * (h1Mod - oneBodyOperator.second) / 2.).exp();
  
  // this is the trial state used to measure energy
  MatrixXcd ref;
  hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  ref = hf.block(0, 0, norbs, Determinant::nalpha);
  MatrixXcd refT = ref.adjoint();
  
  // this is the initial state
  MatrixXcd ccexpOneBodyOperator = cconeBodyOperator.exp();
  MatrixXcd ccref = ccexpOneBodyOperator * ref;

  // rotate cholesky vectors
  vector<MatrixXcd> rotChol;
  for (int i = 0; i < chol.size(); i++) {
    MatrixXcd rot = refT * chol[i];
    rotChol.push_back(rot);
  }
  
  complex<double> refEnergy = calcHamiltonianElement(refT, ref, enuc, h1, rotChol);
  complex<double> ene0;
  if (schd.ene0Guess == 1.e10) ene0 = refEnergy;
  else ene0 = schd.ene0Guess;
  if (commrank == 0) {
    cout << "Initial state energy:  " << refEnergy << endl;
    cout << "Ground state energy guess:  " << ene0 << endl << endl; 
  }
  
  int nchol = chol.size();
  complex<double> delta(0., 0.);
  if (commrank == 0) cout << "Number of Cholesky vectors: " << nchol << endl;
  vector<int> ncholVec = { int(0.3 * chol.size()), int(0.4 * chol.size()), int(0.5 * chol.size()), int(0.6 * chol.size()), int(0.7 * chol.size()) };
  for (int i = 0; i < ncholVec.size(); i++) {
    complex<double> trefEnergy = calcHamiltonianElement(refT, ref, enuc, h1, rotChol, ncholVec[i]);
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
 
  // Gaussian sampling
  // field values arranged right to left
  normal_distribution<double> normal(0., 1.);
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
    
    MatrixXcd rn;
    complex<double> orthoFac = complex<double> (1., 0.);
    int eneStepCounter = 0;

    // cc sampling
    MatrixXcd ccprop = MatrixXcd::Zero(norbs, norbs);
    for (int i = 0; i < ccnfields; i++) {
      double ccfield_i = normal(generator);
      ccprop.noalias() += ccfield_i * cchsOperators[i];
    }
    rn = ccprop.exp() * ccref;
    //if (commrank == 0) cout << "ccprop\n" << ccprop << endl << endl;
    //if (commrank == 0) cout << "rn\n" << rn << endl;
    //exit(0);

    // prop sampling
    ArrayXcd numSampleA(nEneSteps), denomSampleA(nEneSteps);
    numSampleA.setZero(); denomSampleA.setZero();
    for (int n = 0; n < nsteps; n++) {
      // prop
      double init = getTime();
      MatrixXd prop = MatrixXd::Zero(norbs, norbs);
      complex<double> shift(0., 0.);
      for (int i = 0; i < nfields; i++) {
        double field_n_i = normal(generator);
        prop.noalias() += field_n_i * chol[i];
        shift += field_n_i * mfShifts[i];
      }
      rn = expOneBodyOperator.first * rn;
      MatrixXcd propc = sqrt(dt) * complex<double>(0, 1.) * prop;
      MatrixXcd temp = rn;
      for (int i = 1; i < 10; i++) {
        temp = propc * temp / i;
        rn += temp;
      }
      rn = exp(-sqrt(dt) * shift / (Determinant::nalpha + Determinant::nbeta)) * exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * expOneBodyOperator.first * rn;
      
      propTime += getTime() - init;
      
      // orthogonalize for stability
      if (n % orthoSteps == 0 && n != 0) {
        orthogonalize(rn, orthoFac);
      }

      // measure
      init = getTime();
      if (n == eneSteps[eneStepCounter] ) {
        if (stats.converged[eneStepCounter] == -1) {//-1 means this time slice has not yet converged
          complex<double> overlap = orthoFac * orthoFac * (refT * rn).determinant() * (refT * rn).determinant();
          complex<double> numSample = overlap * calcHamiltonianElement(refT, rn, enuc, h1, rotChol, nchol);
          numSampleA[eneStepCounter] = numSample;
          denomSampleA[eneStepCounter] = overlap;
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
}


// calculates mixed energy estimator of the imaginary time propagated wave function
// w jastrow
void calcEnergyJastrowDirectVariational(double enuc, MatrixXd& h1, MatrixXd& h1Mod, vector<MatrixXd>& chol)
{
  size_t norbs = Determinant::norbs;
  size_t nfields = chol.size();
  size_t nsweeps = schd.stochasticIter;
  size_t nsteps = schd.nsteps;
  size_t orthoSteps = schd.orthoSteps;
  double dt = schd.dt;

  // prep and init
  // this is for mean field subtraction
  MatrixXd hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair rhf;
  rhf.first = hf.block(0, 0, norbs, Determinant::nalpha);
  rhf.second = hf.block(0, 0, norbs, Determinant::nbeta);
  
  // jastrow operators
  matPair refAct;
  size_t numActOrbs;
  if (schd.nciAct < 0) numActOrbs = norbs;
  else numActOrbs = schd.nciAct;
  refAct.first = rhf.first.block(0, 0, numActOrbs, Determinant::nalpha);
  refAct.second = rhf.second.block(0, 0, numActOrbs, Determinant::nbeta);
  vector<vecPair> jhsOperators;
  vecPair joneBodyOperator;
  complex<double> jmfConst = prepJastrowHS(refAct, jhsOperators, joneBodyOperator);
  size_t jnfields = jhsOperators.size();
  
  // prop
  vector<matPair> hsOperators;
  matPair oneBodyOperator;
  complex<double> mfConst = prepPropagatorHS(rhf, chol, hsOperators, oneBodyOperator);
  
  // this is the initial state
  matPair ref;
  if (schd.hf == "rhf") {
    hf = MatrixXd::Zero(norbs, norbs);
    readMat(hf, "rhf.txt");
    ref.first = hf.block(0, 0, norbs, Determinant::nalpha);
    ref.second = hf.block(0, 0, norbs, Determinant::nbeta);
  }
  else if (schd.hf == "uhf") {
    hf = MatrixXd::Zero(norbs, 2*norbs);
    readMat(hf, "uhf.txt");
    ref.first = hf.block(0, 0, norbs, Determinant::nalpha);
    ref.second = hf.block(0, norbs, norbs, Determinant::nbeta);
  }
  matPair refT;
  refT.first = ref.first.adjoint();
  refT.second = ref.second.adjoint();

  matPair expOneBodyOperator;
  expOneBodyOperator.first =  (-dt * (h1Mod - oneBodyOperator.first) / 2.).exp();
  expOneBodyOperator.second = (-dt * (h1Mod - oneBodyOperator.second) / 2.).exp();
  

  // rotate cholesky vectors
  pair<vector<MatrixXcd>, vector<MatrixXcd>> rotChol;
  for (int i = 0; i < chol.size(); i++) {
    MatrixXcd rotUp = refT.first * chol[i];
    MatrixXcd rotDn = refT.second * chol[i];
    rotChol.first.push_back(rotUp);
    rotChol.second.push_back(rotDn);
  }


  vecPair jexpOneBodyOperator;
  jexpOneBodyOperator.first = joneBodyOperator.first.array().exp();
  jexpOneBodyOperator.second = joneBodyOperator.second.array().exp();
  matPair jref;
  jref.first = jexpOneBodyOperator.first.asDiagonal() * refAct.first;
  jref.second = jexpOneBodyOperator.second.asDiagonal() * refAct.second;
  

  // Gaussian sampling
  // field values arranged right to left
  vector<VectorXd> fields;
  normal_distribution<double> normal(0., 1.);
  matPair jrefT;
  jrefT.first = jref.first.adjoint();
  jrefT.second = jref.second.adjoint();
  matPair green;
  calcGreensFunction(refT, ref, green);
  complex<double> refEnergy = calcHamiltonianElement(green, enuc, h1, chol);
  complex<double> ene0;
  if (schd.ene0Guess == 1.e10) ene0 = refEnergy;
  else ene0 = schd.ene0Guess;
  if (commrank == 0) {
    cout << "Initial state energy:  " << refEnergy << endl;
    cout << "Ground state energy guess:  " << ene0 << endl << endl; 
  }
  
  vector<int> eneSteps = { int(nsteps - 1) };
  int numEneSteps = eneSteps.size();
  vector<complex<double>> numMeanVar(numEneSteps, complex<double>(0., 0.)), denomMeanVar(numEneSteps, complex<double>(0., 0.)), denomAbsMeanVar(numEneSteps, complex<double>(0., 0.));
  vector<complex<double>> numMeanProj(numEneSteps, complex<double>(0., 0.)), denomMeanProj(numEneSteps, complex<double>(0., 0.)), denomAbsMeanProj(numEneSteps, complex<double>(0., 0.));
  auto iterTime = getTime();
  double qrTime = 0., propTime = 0., eneTime = 0.;
  if (commrank == 0) cout << "Starting sampling sweeps\n";

  MatrixXcd identity = MatrixXcd::Identity(norbs, norbs);
  for (int sweep = 0; sweep < nsweeps; sweep++) {
    if (sweep != 0 && sweep % (nsweeps/5) == 0 && commrank == 0) cout << sweep << "  " << getTime() - iterTime << " s\n";
    matPair rn;
    int eneStepCounter = 0;

    rn = ref;

    // sampling
    VectorXd fields = VectorXd::Zero(nfields);
    vector<MatrixXcd> expHam(nsteps/orthoSteps +1, identity);

    matPair prop;
    prop.first = MatrixXcd::Zero(norbs, norbs);
    prop.second = MatrixXcd::Zero(norbs, norbs);

    int hindex = 0;
    double init = getTime();
    // sample ham for time slice n
    for (int n = 0; n < nsteps; n++) {

      
      prop.first.setZero();  
      for (int i = 0; i < nfields; i++) {
        double field_n_i = normal(generator);
        prop.first += field_n_i * hsOperators[i].first;
      }

      prop.first = (sqrt(dt) * prop.first).exp();
      expHam[hindex] = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * expOneBodyOperator.first * prop.first * expOneBodyOperator.first * expHam[hindex];

      if (n!= 0 && n % orthoSteps == 0) hindex++;
      
    }
    propTime += getTime() - init;
    
  
    size_t numJastrowSamples = schd.numJastrowSamples;
    complex<double> jOverlapVar(0., 0.), jLocalEnergyVar(0., 0.);
    complex<double> jOverlapProj(0., 0.), jLocalEnergyProj(0., 0.);

    //measure energy numJastrowSamples times
    for (int i = 0; i < numJastrowSamples; i++) {

      // measure energy
      init = getTime();
      matPair ln, rn;

      //sample left jastrow
      {
        VectorXd jfields = VectorXd::Zero(jnfields);
        vecPair jpropLeft;
        jpropLeft.first = VectorXcd::Zero(numActOrbs);
        jpropLeft.second = VectorXcd::Zero(numActOrbs);
        for (int i = 0; i < jnfields; i++) {
          jfields(i) = normal(generator);
          jpropLeft.first += jfields(i) * jhsOperators[i].first;
          jpropLeft.second += jfields(i) * jhsOperators[i].second;
        }
        ln.first = exp(jmfConst / (2. * Determinant::nalpha)) * jrefT.first * jpropLeft.first.array().exp().matrix().asDiagonal();
        ln.second = exp(jmfConst / (2. * Determinant::nbeta)) * jrefT.second * jpropLeft.second.array().exp().matrix().asDiagonal();
      }
      //cout << "left prep"<<endl;
      //sample right jastrow
      {
        VectorXd jfields = VectorXd::Zero(jnfields);
        vecPair jpropRight;
        jpropRight.first = VectorXcd::Zero(numActOrbs);
        jpropRight.second = VectorXcd::Zero(numActOrbs);
        for (int i = 0; i < jnfields; i++) {
          jfields(i) = normal(generator);
          jpropRight.first  += jfields(i) * jhsOperators[i].first;
          jpropRight.second += jfields(i) * jhsOperators[i].second;
        }
        rn.first = exp(jmfConst / (2. * Determinant::nalpha)) *  jpropRight.first.array().exp().matrix().asDiagonal() * jref.first;
        rn.second = exp(jmfConst / (2. * Determinant::nbeta)) *  jpropRight.second.array().exp().matrix().asDiagonal() * jref.second;
       }    
      //cout << "right prep"<<endl;

      //apply ham to the ln
      matPair rnHam, lnHamAd, rnRefHam, lnRefHamAd;
      rnHam = rn; lnHamAd.first = 1.*ln.first.adjoint(); lnHamAd.second = 1.*ln.second.adjoint();
      rnRefHam = ref; lnRefHamAd = ref;
      
      complex<double> orthoFacR = complex<double> (1., 0.);
      complex<double> orthoFacRRef = complex<double> (1., 0.);
      for (int i=0; i<hindex; i++) {
        rnHam.first = expHam[i]*rnHam.first; 
        rnHam.second = expHam[i]*rnHam.second;
        orthogonalize(rnHam, orthoFacR);

        rnRefHam.first = expHam[i]*rnRefHam.first; 
        rnRefHam.second = expHam[i]*rnRefHam.second;
        orthogonalize(rnRefHam, orthoFacRRef);

      } 

      complex<double> orthoFacL = complex<double> (1., 0.);
      complex<double> orthoFacLRef = complex<double> (1., 0.);
      //cout << "before for"<<endl;
      for (int i=0; i<hindex; i++) {
        lnHamAd.first = expHam[i]*lnHamAd.first; 
        lnHamAd.second = expHam[i]*lnHamAd.second;
        orthogonalize(lnHamAd, orthoFacL);

        lnRefHamAd.first = expHam[i]*lnRefHamAd.first; 
        lnRefHamAd.second = expHam[i]*lnRefHamAd.second;
        orthogonalize(lnRefHamAd, orthoFacLRef);
      } 
      //cout<< "after for"<<endl;
      matPair lnHam, lnRefHam;
      lnHam.first = 1.*lnHamAd.first.adjoint(); lnHam.second = 1.*lnHamAd.second.adjoint();
      lnRefHam.first = 1.*lnRefHamAd.first.adjoint(); lnRefHam.second = 1.*lnRefHamAd.second.adjoint();

      propTime += getTime() - init;

      init = getTime();

      //Projected energy
      //cout << refT.first.rows()<<"  "<<refT.first.cols()<<"  "<<rnRefHam.first.rows()<<"  "<<rnRefHam.second.cols()<<endl;
      complex<double> overlapSample =  orthoFacRRef * (refT.first * rnRefHam.first).determinant() 
                                    * (refT.second * rnRefHam.second).determinant();
      jOverlapProj += overlapSample;
      //calcGreensFunction(ln, rnRefHam, green);
      jLocalEnergyProj += overlapSample * calcHamiltonianElement(refT, rnRefHam, enuc, h1, chol);

      overlapSample =  orthoFacLRef * (lnRefHam.first * rn.first).determinant() 
                                    * (lnRefHam.second * rn.second).determinant();
      jOverlapProj += overlapSample;
      //calcGreensFunction(lnRefHam, rn, green);
      jLocalEnergyProj += overlapSample * calcHamiltonianElement(lnRefHam, rn, enuc, h1, chol);

      //variational energy
      complex<double> overlapSampleR = orthoFacR * (ln.first * rnHam.first).determinant() 
                               * (ln.second * rnHam.second).determinant();
      complex<double>overlapSampleL =  orthoFacL * (lnHam.first * rn.first).determinant() 
                               * (lnHam.second * rn.second).determinant();
      jOverlapVar += (overlapSampleR + overlapSampleL);
      
      //calcGreensFunction(ln, rnHam, green);
      jLocalEnergyVar += overlapSampleR * calcHamiltonianElement(ln, rnHam, enuc, h1, chol);
      //calcGreensFunction(lnHam, rn, green);
      jLocalEnergyVar += overlapSampleL * calcHamiltonianElement(lnHam, rn, enuc, h1, chol);
      eneTime += getTime() - init;
    }

    jOverlapVar /= numJastrowSamples;
    jLocalEnergyVar /= numJastrowSamples;
    jOverlapProj /= numJastrowSamples;
    jLocalEnergyProj /= numJastrowSamples;

    denomMeanVar[eneStepCounter] += (jOverlapVar - denomMeanVar[eneStepCounter]) / (1.*(sweep + 1));
    denomAbsMeanVar[eneStepCounter] += (abs(jOverlapVar) - denomAbsMeanVar[eneStepCounter]) / (1.*(sweep + 1));
    numMeanVar[eneStepCounter] += (jLocalEnergyVar - numMeanVar[eneStepCounter]) / (1.*(sweep + 1));

    denomMeanProj[eneStepCounter] += (jOverlapProj - denomMeanProj[eneStepCounter]) / (1.*(sweep + 1));
    denomAbsMeanProj[eneStepCounter] += (abs(jOverlapProj) - denomAbsMeanProj[eneStepCounter]) / (1.*(sweep + 1));
    numMeanProj[eneStepCounter] += (jLocalEnergyProj - numMeanProj[eneStepCounter]) / (1.*(sweep + 1));
  }

  if (commrank == 0) {
    cout << "\nPropagation time:  " << propTime << " s\n";
    cout << "Energy evaluation time:  " << eneTime << " s\n\n";
    cout << "          iTime                 Energy                     Energy error         Average phase\n";
  }

  for (int n = 0; n < numEneSteps; n++) {
    complex<double> energyAll[commsize];
    for (int i = 0; i < commsize; i++) energyAll[i] = complex<double>(0., 0.);
    
    complex<double> energyProc = numMeanVar[n] / denomMeanVar[n];
    complex<double> numProc = numMeanVar[n];
    complex<double> denomProc = denomMeanVar[n];
    complex<double> denomAbsProc = denomAbsMeanVar[n];
    MPI_Gather(&(energyProc), 1, MPI_DOUBLE_COMPLEX, &(energyAll), 1, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &energyProc, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &numProc, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &denomProc, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &denomAbsProc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    energyProc /= commsize;
    numProc /= commsize;
    denomProc /= commsize;
    denomAbsProc /= commsize;
    double stddev = 0., stddev2 = 0.;
    for (int i = 0; i < commsize; i++) {
      stddev += pow(abs(energyAll[i] - energyProc), 2);
      stddev2 += pow(abs(energyAll[i] - energyProc), 4);
    }
    stddev /= (commsize - 1);
    stddev2 /= commsize;
    stddev2 = sqrt((stddev2 - (commsize - 3) * pow(stddev, 2) / (commsize - 1)) / commsize) / 2. / sqrt(stddev) / sqrt(sqrt(commsize));
    stddev = sqrt(stddev / commsize);

    if (commrank == 0) {
      cout << "Variational Energy"<<endl;
      cout << format(" %14.2f   (%14.8f, %14.8f)   (%8.2e   (%8.2e))   (%3.3f, %3.3f) \n") % ((eneSteps[n] + 1) * dt) % energyProc.real() % energyProc.imag() % stddev % stddev2 % (denomProc / denomAbsProc).real() % (denomProc / denomAbsProc).imag(); 
    }
  }

  for (int n = 0; n < numEneSteps; n++) {
    complex<double> energyAll[commsize];
    for (int i = 0; i < commsize; i++) energyAll[i] = complex<double>(0., 0.);
    
    complex<double> energyProc = numMeanProj[n] / denomMeanProj[n];
    complex<double> numProc = numMeanProj[n];
    complex<double> denomProc = denomMeanProj[n];
    complex<double> denomAbsProc = denomAbsMeanProj[n];
    MPI_Gather(&(energyProc), 1, MPI_DOUBLE_COMPLEX, &(energyAll), 1, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &energyProc, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &numProc, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &denomProc, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &denomAbsProc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    energyProc /= commsize;
    numProc /= commsize;
    denomProc /= commsize;
    denomAbsProc /= commsize;
    double stddev = 0., stddev2 = 0.;
    for (int i = 0; i < commsize; i++) {
      stddev += pow(abs(energyAll[i] - energyProc), 2);
      stddev2 += pow(abs(energyAll[i] - energyProc), 4);
    }
    stddev /= (commsize - 1);
    stddev2 /= commsize;
    stddev2 = sqrt((stddev2 - (commsize - 3) * pow(stddev, 2) / (commsize - 1)) / commsize) / 2. / sqrt(stddev) / sqrt(sqrt(commsize));
    stddev = sqrt(stddev / commsize);

    if (commrank == 0) {
      cout << "Projected Energy"<<endl;
      cout << format(" %14.2f   (%14.8f, %14.8f)   (%8.2e   (%8.2e))   (%3.3f, %3.3f) \n") % ((eneSteps[n] + 1) * dt) % energyProc.real() % energyProc.imag() % stddev % stddev2 % (denomProc / denomAbsProc).real() % (denomProc / denomAbsProc).imag(); 
    }
  }

}

// calculates energy of the imaginary time propagated wave function using direct sampling of exponentials
void calcEnergyDirectMultiSlater(double enuc, MatrixXd& h1, MatrixXd& h1Mod, vector<MatrixXd>& chol)
{
  size_t norbs = Determinant::norbs;
  size_t nfields = chol.size();
  size_t nsweeps = schd.stochasticIter;
  size_t nsteps = schd.nsteps;
  size_t orthoSteps = schd.orthoSteps;
  double dt = schd.dt;
  vector<int> eneSteps = schd.eneSteps;

  // prep and init
  // this is for mean field subtraction
  MatrixXd hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair rhf;
  rhf.first = hf.block(0, 0, norbs, Determinant::nalpha);
  rhf.second = hf.block(0, 0, norbs, Determinant::nbeta);
  
  //vector<matPair> hsOperators;
  vector<complex<double>> mfShifts;
  matPair oneBodyOperator;
  //complex<double> mfConst = prepPropagatorHS(rhf, chol, hsOperators, oneBodyOperator);
  complex<double> mfConst = prepPropagatorHS(rhf, chol, mfShifts, oneBodyOperator);
  
  // this is the initial state
  matPair ref;
  hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  ref.first = hf.block(0, 0, norbs, Determinant::nalpha);
  ref.second = hf.block(0, 0, norbs, Determinant::nbeta);
 
  // this is the left state
  matPair refT;
  refT.first = ref.first.adjoint();
  refT.second = ref.second.adjoint();
  string fname = schd.determinantFile;
  std::array<std::vector<int>, 2> refDet; 
  std::array<std::vector<std::array<Eigen::VectorXi, 2>>, 2> ciExcitations;
  std::vector<double> ciParity; 
  std::vector<double> ciCoeffs;
  // if using text file name it "dets", if using binary use anything other than dets
  if (fname == "dets") readDeterminants(fname, refDet, ciExcitations, ciParity, ciCoeffs);
  else readDeterminantsBinary(fname, refDet, ciExcitations, ciParity, ciCoeffs);

  double cumulative;
  vector<int> alias; vector<double> prob;
  setUpAliasMethod(&ciCoeffs[1], ciCoeffs.size()-1, cumulative, alias, prob);

  std::array<std::vector<std::array<Eigen::VectorXi, 2>>, 2> ciExcitationsSample;
  std::vector<double> ciParitySample; 
  std::vector<double> ciCoeffsSample;

  matPair expOneBodyOperator;
  expOneBodyOperator.first =  (-dt * (h1Mod - oneBodyOperator.first) / 2.).exp();
  expOneBodyOperator.second = (-dt * (h1Mod - oneBodyOperator.second) / 2.).exp();

  // Gaussian sampling
  // field values arranged right to left
  vector<VectorXd> fields;
  normal_distribution<double> normal(0., 1.);
  
  //matPair green;
  //calcGreensFunction(refT, ref, green);
  //complex<double> refEnergy = calcHamiltonianElement(green, enuc, h1, chol);
  vector<double> times = { 0., 0.};
  auto refOverlapHam = calcHamiltonianElement(refT, ciExcitations, ciParity, ciCoeffs, ref, enuc, h1, chol);
  complex<double> refEnergy = refOverlapHam.second / refOverlapHam.first;
  //complex<double> refEnergy = calcHamiltonianElement(refT, ref, enuc, h1, rotChol);
  complex<double> ene0;
  if (schd.ene0Guess == 1.e10) ene0 = refEnergy;
  else ene0 = schd.ene0Guess;
  if (commrank == 0) {
    cout << "Initial state energy:  " << setprecision(8) << refEnergy << endl;
    cout << "Ground state energy guess:  " << ene0 << endl << endl; 
  }
  
  int nchol = chol.size();
  complex<double> delta(0., 0.);
  if (commrank == 0) cout << "Number of Cholesky vectors: " << nchol << endl;
  vector<int> ncholVec = { int(0.3 * chol.size()), int(0.4 * chol.size()), int(0.5 * chol.size()), int(0.6 * chol.size()), int(0.7 * chol.size()) };
  for (int i = 0; i < ncholVec.size(); i++) {
    auto trefOverlapHam = calcHamiltonianElement(refT, ciExcitations, ciParity, ciCoeffs, ref, enuc, h1, chol, ncholVec[i]);
    complex<double> trefEnergy = trefOverlapHam.second / trefOverlapHam.first;
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

    //sample the determinants
    if (schd.sampleDeterminants != -1)
    {
      int nSample = schd.sampleDeterminants;
      ciExcitationsSample[0].clear(); ciExcitationsSample[1].clear();
      ciParitySample.clear(); ciCoeffsSample.clear();

      vector<int> sample(nSample,-1); vector<double> wts(nSample,0.); 
      sample_N2_alias(&ciCoeffs[1], cumulative, sample, wts, alias, prob);
      ciExcitationsSample[0].push_back(ciExcitations[0][0]);
      ciExcitationsSample[1].push_back(ciExcitations[1][0]);
      ciCoeffsSample.push_back(ciCoeffs[0]);
      ciParitySample.push_back(ciParity[0]);

      for (int j=0; j<sample.size(); j++) {
        if (sample[j] == -1) break;
        ciExcitationsSample[0].push_back(ciExcitations[0][sample[j]+1]);
        ciExcitationsSample[1].push_back(ciExcitations[1][sample[j]+1]);
        ciCoeffsSample.push_back(wts[j]);
        ciParitySample.push_back(ciParity[sample[j]+1]);
      }

    }

    matPair rn;
    rn = ref;
    VectorXd fields = VectorXd::Zero(nfields);
    complex<double> orthoFac = complex<double>(1., 0.);
    int eneStepCounter = 0;
    ArrayXcd numSampleA(nEneSteps), denomSampleA(nEneSteps);
    numSampleA.setZero(); denomSampleA.setZero();
    for (int n = 0; n < nsteps; n++) {
      // sampling
      double init = getTime();
      //MatrixXcd prop = MatrixXcd::Zero(norbs, norbs);
      MatrixXd prop = MatrixXd::Zero(norbs, norbs);
      complex<double> shift(0., 0.);
      for (int i = 0; i < nfields; i++) {
        double field_n_i = normal(generator);
        prop.noalias() += field_n_i * chol[i];
        shift += field_n_i * mfShifts[i];
        //prop += field_n_i * complex<double>(0., 1.) * chol[i];
        //prop.diagonal() -= field_n_i * VectorXcd::Constant(norbs, mfShifts[i]/(1. * (Determinant::nalpha + Determinant::nbeta)));
        //prop.second += field_n_i * hsOperators[i].second;
      }
      //prop = (sqrt(dt) * prop).exp();
      //rn.first = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * (expOneBodyOperator.first * (prop * (expOneBodyOperator.first * rn.first)));
      
      rn.first = expOneBodyOperator.first * rn.first;
      MatrixXcd propc = sqrt(dt) * complex<double>(0, 1.) * prop;
      MatrixXcd temp = rn.first;
      for (int i = 1; i < 10; i++) {
        temp = propc * temp / i;
        rn.first += temp;
      }
      rn.first = exp(-sqrt(dt) * shift / (Determinant::nalpha + Determinant::nbeta)) * exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * expOneBodyOperator.first * rn.first;
      
      if (Determinant::nalpha == Determinant::nbeta) rn.second = rn.first;
      else {
        //prop.second = (sqrt(dt) * prop.second).exp();
        rn.second = exp(-sqrt(dt) * shift / (Determinant::nalpha + Determinant::nbeta)) * exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nbeta)) * expOneBodyOperator.second * propc * expOneBodyOperator.second * rn.second;
      }

      propTime += getTime() - init;
      
      // orthogonalize for stability
      if (n % orthoSteps == 0) {
        orthogonalize(rn, orthoFac);
      }

      // measure
      init = getTime();
      if (n == eneSteps[eneStepCounter] ) {
        if (stats.converged[eneStepCounter] == -1) {//-1 means this time slice has not yet converged
          pair<complex<double>, complex<double>> overlapHam;
          if (schd.sampleDeterminants != -1)
            overlapHam = calcHamiltonianElement(refT, ciExcitationsSample, ciParitySample, ciCoeffsSample, rn, enuc, h1, chol);
          else 
            overlapHam = calcHamiltonianElement(refT, ciExcitations, ciParity, ciCoeffs, rn, enuc, h1, chol, nchol, schd.nciAct, schd.nciCore);
          //auto overlapHam = calcHamiltonianElement_sRI(refT, ciExcitations, ciParity, ciCoeffs, rn, enuc, h1, chol, std::real(refEnergy));

          complex<double> overlap = orthoFac * overlapHam.first;
          complex<double> numSample = orthoFac * overlapHam.second;
          numSampleA[eneStepCounter] = numSample;
          denomSampleA[eneStepCounter] = overlap;
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
    //cout << "Intermediates:  " << times[0] << endl;
    //cout << "citer:  " << times[1] << endl << endl;
  }

  stats.gatherAndPrintStatistics(iTime, delta);
  if (schd.printLevel > 10) stats.writeSamples();
}
