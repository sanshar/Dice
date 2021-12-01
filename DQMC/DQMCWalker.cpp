#include <iostream>
#include <fstream>
#include <unsupported/Eigen/MatrixFunctions>
#include "DQMCWalker.h"
#include "Hamiltonian.h"
#include "global.h"
#include "input.h"

using namespace std;
using namespace Eigen;
using matPair = std::array<MatrixXcd, 2>;

// constructor
DQMCWalker::DQMCWalker(bool prhfQ, bool pphaselessQ) 
{
  rhfQ = prhfQ;
  phaselessQ = pphaselessQ;
  orthoFac = complex<double> (1., 0.);
  normal = normal_distribution<double>(0., 1.);
};


void DQMCWalker::prepProp(std::array<Eigen::MatrixXcd, 2>& ref, Hamiltonian& ham, double pdt, double ene0)
{
  dt = pdt;

  int norbs = ham.norbs;
  int nfields = ham.chol.size();

  // read or calculate rdm for mean field shifts
  matPair green;
  bool readRDM = false;
  std::string fname = "spinRDM.0.0.txt";
  ifstream rdmfile(fname);
  if (rdmfile) readRDM = true;
  rdmfile.close();
  if (readRDM){
    if (commrank == 0) cout << "Reading RDM from disk for background subtraction\n\n";
    MatrixXd oneRDM, twoRDM;
    readSpinRDM("spinRDM.0.0.txt", oneRDM, twoRDM);
    green[0] = MatrixXcd::Zero(norbs, norbs);
    green[1] = MatrixXcd::Zero(norbs, norbs);
    for (int i = 0; i < 2*norbs; i++) {
      for (int j = 0; j < 2*norbs; j++) {
        if (i%2 == 0 && j%2 == 0) green[0](i/2, j/2) = oneRDM(i, j); 
        else if (i%2 == 1 && j%2 == 1) green[1](i/2, j/2) = oneRDM(i, j); 
      }
    }
  }
  else {
    if (commrank == 0) cout << "Using ROHF RDM for background subtraction\n\n";
    matPair refT;
    refT[0] = ref[0].adjoint();
    refT[1] = ref[1].adjoint();
    green[0] = ref[0] * (refT[0] * ref[0]).inverse() * refT[0];
    green[1] = ref[1] * (refT[1] * ref[1]).inverse() * refT[1];
  }

  // calculate mean field shifts
  MatrixXcd oneBodyOperator = ham.h1Mod;
  complex<double> constant(0., 0.);
  constant += ene0 - ham.ecore;
  for (int i = 0; i < nfields; i++) {
    MatrixXcd op = complex<double>(0., 1.) * ham.chol[i];
    complex<double> mfShift = 1. * green[0].cwiseProduct(op).sum() + 1. * green[1].cwiseProduct(op).sum();
    constant -= pow(mfShift, 2) / 2.;
    oneBodyOperator -= mfShift * op;
    if (phaselessQ) mfShifts.push_back(mfShift);
    else mfShifts.push_back(mfShift / (ham.nalpha + ham.nbeta));
  }

  if (phaselessQ) {
    propConstant[0] = constant - ene0;
    propConstant[1] = constant - ene0;
  }
  else {
    propConstant[0] = constant / ham.nalpha;
    propConstant[1] = constant / ham.nbeta;
  }
  expOneBodyOperator =  (-dt * oneBodyOperator / 2.).exp();

  //ham.floattenCholesky(floatChol);
};


void DQMCWalker::setDet(std::array<Eigen::MatrixXcd, 2> pdet) 
{
  det = pdet;
  orthoFac = complex<double> (1., 0.);
};
 

void DQMCWalker::setDet(std::vector<std::complex<double>>& serial, std::complex<double> ptrialOverlap)
{
  trialOverlap = ptrialOverlap;
  int norbs = det[0].rows(), nalpha = det[0].cols(), nbeta = det[1].cols();
  for (int i = 0; i < norbs; i++)
    for (int j = 0; j < nalpha; j++)
      det[0](i, j) = serial[i * nalpha + j];
  for (int i = 0; i < norbs; i++)
    for (int j = 0; j < nbeta; j++)
      det[1](i, j) = serial[nalpha * norbs + i * nbeta + j];
};


std::complex<double> DQMCWalker::getDet(std::vector<std::complex<double>>& serial)
{
  int norbs = det[0].rows(), nalpha = det[0].cols(), nbeta = det[1].cols();
  for (int i = 0; i < norbs; i++)
    for (int j = 0; j < nalpha; j++)
      serial[i * nalpha + j] = det[0](i, j);
  for (int i = 0; i < norbs; i++)
    for (int j = 0; j < nbeta; j++)
      serial[nalpha * norbs + i * nbeta + j] = det[1](i, j);
  return trialOverlap;
};


// for phaseless propagation the normalization does not matter
// it is calculated so as to not repeat it for the overlap ratio
void DQMCWalker::orthogonalize()
{
  complex<double> tempOrthoFac(1., 0.);
  HouseholderQR<MatrixXcd> qr1(det[0]);
  det[0] = qr1.householderQ() * MatrixXd::Identity(det[0].rows(), det[0].cols());
  for (int i = 0; i < qr1.matrixQR().diagonal().size(); i++) tempOrthoFac *= qr1.matrixQR().diagonal()(i);
  if (rhfQ) {
    orthoFac *= (tempOrthoFac * tempOrthoFac);
    if (phaselessQ) {
      trialOverlap /= (tempOrthoFac * tempOrthoFac);
      orthoFac = 1.;
    }
  }
  else {
    HouseholderQR<MatrixXcd> qr2(det[1]);
    det[1] = qr2.householderQ() * MatrixXd::Identity(det[1].rows(), det[1].cols());
    for (int i = 0; i < qr2.matrixQR().diagonal().size(); i++) tempOrthoFac *= qr2.matrixQR().diagonal()(i);
    orthoFac *= tempOrthoFac;
    if (phaselessQ) {
      trialOverlap /= tempOrthoFac;
      orthoFac = 1.;
    }
  }
};


void DQMCWalker::propagate(Hamiltonian& ham)
{
  int norbs = det[0].rows();
  int nfields = ham.floatChol.size(); 
  //MatrixXf prop = MatrixXf::Zero(norbs, norbs);
  vector<float> prop(norbs * (norbs + 1) / 2, 0.);
  complex<double> shift(0., 0.);
  VectorXd fields(nfields);
  fields.setZero();
  for (int n = 0; n < nfields; n++) {
    double field_n = normal(generator);
    fields(n) = field_n;
    for (int i = 0; i < norbs; i++)
      for (int j = 0; j <= i; j++)
        prop[i * (i + 1) / 2 + j] += float(field_n) * ham.floatChol[n][i * (i + 1) / 2 + j];
    //prop.noalias() += float(field_n) * floatChol[i];
    shift += field_n * mfShifts[n];
  }
  //MatrixXcd propc = sqrt(dt) * complex<double>(0, 1.) * prop.cast<double>();
  MatrixXcd propc = MatrixXcd::Zero(norbs, norbs);
  for (int i = 0; i < norbs; i++) {
    propc(i, i) = sqrt(dt) * complex<double>(0, 1.) * prop[i * (i + 1) / 2 + i];
    for (int j = 0; j < i; j++) {
      propc(i, j) = sqrt(dt) * complex<double>(0, 1.) * prop[i * (i + 1) / 2 + j];
      propc(j, i) = sqrt(dt) * complex<double>(0, 1.) * prop[i * (i + 1) / 2 + j];
    }
  }
  
  det[0] = expOneBodyOperator * det[0];
  MatrixXcd temp = det[0];
  for (int i = 1; i < 10; i++) {
    temp = propc * temp / i;
    det[0] += temp;
  }
  det[0] = exp(-sqrt(dt) * shift) * exp(propConstant[0] * dt / 2.) * expOneBodyOperator * det[0];


  if (rhfQ) det[1] = det[0];
  else {
    det[1] = expOneBodyOperator * det[1];
    temp = det[1];
    for (int i = 1; i < 10; i++) {
      temp = propc * temp / i;
      det[1] += temp;
    }
    det[1] = exp(-sqrt(dt) * shift) * exp(propConstant[1] * dt / 2.) * expOneBodyOperator * det[1];
  }
};


double DQMCWalker::propagatePhaseless(Wavefunction& wave, Hamiltonian& ham, double eshift)
{
  int norbs = det[0].rows();
  int nfields = ham.floatChol.size(); 
  VectorXcd fb(nfields);
  fb.setZero();
  this->forceBias(wave, ham, fb);
  // capping fb
  //for (int n = 0; n < nfields; n++) 
  //  if (std::abs(fb(n)) > 1./sqrt(dt)) fb(n) = fb(n) / std::abs(fb(n));
  MatrixXf prop = MatrixXf::Zero(norbs, norbs);
  vector<float> propr(norbs * (norbs + 1) / 2, 0.);
  vector<float> propi(norbs * (norbs + 1) / 2, 0.);
  //MatrixXcd propc = MatrixXcd::Zero(norbs, norbs);
  complex<double> shift(0., 0.), fbTerm(0., 0.);
  for (int n = 0; n < nfields; n++) {
    double field_n = normal(generator);
    complex<double> fieldShift = -sqrt(dt) * (complex<double>(0., 1.) * fb(n) - mfShifts[n]);
    //complex<double> fieldShift = complex<double>(0., 0.);
    //propc += sqrt(dt) * complex<double>(0., 1.) * (field_n - fieldShift) * ham.chol[n];
    for (int i = 0; i < norbs; i++) {
      for (int j = 0; j <= i; j++) {
        propr[i * (i + 1) / 2 + j] += float(field_n - fieldShift.real()) * ham.floatChol[n][i * (i + 1) / 2 + j];
        propi[i * (i + 1) / 2 + j] += float(- fieldShift.imag()) * ham.floatChol[n][i * (i + 1) / 2 + j];
      }
    }
    //prop.noalias() += float(field_n) * floatChol[i];
    shift += (field_n - fieldShift) * mfShifts[n];
    fbTerm += (field_n * fieldShift - fieldShift * fieldShift / 2);
  }

  //MatrixXcd propc = sqrt(dt) * complex<double>(0, 1.) * prop.cast<double>();
  MatrixXcd propc = MatrixXcd::Zero(norbs, norbs);
  for (int i = 0; i < norbs; i++) {
    propc(i, i) = sqrt(dt) * (complex<double>(0., 1.) * propr[i * (i + 1) / 2 + i] - propi[i * (i + 1) / 2 + i]);
    for (int j = 0; j < i; j++) {
      propc(i, j) = sqrt(dt) * (complex<double>(0., 1.) * propr[i * (i + 1) / 2 + j] - propi[i * (i + 1) / 2 + j]);
      propc(j, i) = sqrt(dt) * (complex<double>(0., 1.) * propr[i * (i + 1) / 2 + j] - propi[i * (i + 1) / 2 + j]);
    }
  }
  
  det[0] = expOneBodyOperator * det[0];
  MatrixXcd temp = det[0];
  for (int i = 1; i < 6; i++) {
    temp = propc * temp / i;
    det[0] += temp;
  }
  det[0] = expOneBodyOperator * det[0];


  if (rhfQ) det[1] = det[0];
  else {
    det[1] = expOneBodyOperator * det[1];
    temp = det[1];
    for (int i = 1; i < 6; i++) {
      temp = propc * temp / i;
      det[1] += temp;
    }
    det[1] = expOneBodyOperator * det[1];
  }
  
  // phaseless
  complex<double> oldOverlap = trialOverlap;
  complex<double> newOverlap = this->overlap(wave);
  complex<double> importanceFunction = exp(-sqrt(dt) * shift + fbTerm + dt * (eshift + propConstant[0])) * newOverlap / oldOverlap;
  double theta = std::arg( exp(-sqrt(dt) * shift) * newOverlap / oldOverlap );
  //if (commrank == 0) {
  ////cout << "det[0]:\n" << det[0] << endl;
  //  cout << "forceBias:\n" << fb << endl;
  //  cout << "\noldoverlap: " << oldOverlap << endl;
  //  cout << "newoverlap: " << newOverlap << endl;
  //  cout << "shift: " << shift << endl;
  //  cout << "fbTerm: " << fbTerm << endl;
  //  cout << "propConstant: " << propConstant[0] << endl;
  //  cout << "eshift: " << eshift << endl;
  //  cout << "importanceFunction: " << importanceFunction << endl;
  //  cout << "theta: " << theta << endl << endl;
  //}
  //exit(0);
  double importanceFunctionPhaseless = std::abs(importanceFunction) * cos(theta);
  if (importanceFunctionPhaseless < 1.e-3 || importanceFunctionPhaseless > 100. || std::isnan(importanceFunctionPhaseless)) importanceFunctionPhaseless = 0.; 
  return importanceFunctionPhaseless;
};


// orthoFac not considered
// only used in phaseless
std::complex<double> DQMCWalker::overlap(Wavefunction& wave)
{
  std::complex<double> overlap = rhfQ ? wave.overlap(det[0]) : wave.overlap(det);
  trialOverlap = overlap;
  return overlap;
};


void DQMCWalker::forceBias(Wavefunction& wave, Hamiltonian& ham, Eigen::VectorXcd& fb)
{
  if (rhfQ) wave.forceBias(det[0], ham, fb);
  else wave.forceBias(det, ham, fb);
};


std::array<std::complex<double>, 2> DQMCWalker::hamAndOverlap(Wavefunction& wave, Hamiltonian& ham)
{
  std::array<complex<double>, 2> hamOverlap = rhfQ ? wave.hamAndOverlap(det[0], ham) : wave.hamAndOverlap(det, ham);
  hamOverlap[0] *= orthoFac;
  hamOverlap[1] *= orthoFac;
  return hamOverlap;
};
