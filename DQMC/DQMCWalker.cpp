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
DQMCWalker::DQMCWalker(bool prhfQ, bool pphaselessQ, bool pszQ) 
{
  rhfQ = prhfQ;
  phaselessQ = pphaselessQ;
  szQ = pszQ;
  orthoFac = complex<double> (1., 0.);
  normal = normal_distribution<double>(0., 1.);
  vhsTime = 0.; expTime = 0.; fbTime = 0.;
};


void DQMCWalker::prepProp(std::array<Eigen::MatrixXcd, 2>& ref, Hamiltonian& ham, double pdt, double ene0)
{
  if (ham.intType == "r") prepPropR(ref, ham, pdt, ene0);
  else if (ham.intType == "u") prepPropU(ref, ham, pdt, ene0);
}


void DQMCWalker::prepPropR(std::array<Eigen::MatrixXcd, 2>& ref, Hamiltonian& ham, double pdt, double ene0)
{
  dt = pdt;

  int norbs = ham.norbs;
  int nfields = ham.nchol;

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
    MatrixXd chol = ham.chol[i];
    if (ham.rotFlag == true) {
      chol = MatrixXd::Zero(norbs, norbs);
      size_t triSize = (norbs * (norbs + 1) / 2);
      long counter = 0;
      for (int j = 0; j < norbs; j++) {
        for (int k = 0; k <= j; k++) {
          chol(j, k) = ham.floatChol[i * triSize + counter];
          chol(k, j) = ham.floatChol[i * triSize + counter];
          counter++;
        }
      }
    }
    //MatrixXcd op = complex<double>(0., 1.) * ham.chol[i];
    MatrixXcd op = complex<double>(0., 1.) * chol;
    complex<double> mfShift = 1. * green[0].cwiseProduct(op).sum() + 1. * green[1].cwiseProduct(op).sum();
    constant -= pow(mfShift, 2) / 2.;
    oneBodyOperator -= mfShift * op;
    if (phaselessQ) mfShifts.push_back(mfShift);
    else mfShifts.push_back(mfShift /(1. * (ham.nalpha + ham.nbeta)));
  }

  if (phaselessQ) {
    propConstant[0] = constant - ene0;
    propConstant[1] = constant - ene0;
  }
  else {
    propConstant[0] = constant / (1. * ham.nalpha);
    propConstant[1] = constant / (1. * ham.nbeta);
  }
  expOneBodyOperator =  (-dt * oneBodyOperator / 2.).exp();
};


void DQMCWalker::prepPropU(std::array<Eigen::MatrixXcd, 2>& ref, Hamiltonian& ham, double pdt, double ene0)
{
  dt = pdt;

  int norbs = ham.norbs;
  int nfields = ham.nchol;

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
    matPair refT;
    refT[0] = ref[0].adjoint();
    refT[1] = ref[1].adjoint();
    green[0] = ref[0] * (refT[0] * ref[0]).inverse() * refT[0];
    green[1] = ref[1] * (refT[1] * ref[1]).inverse() * refT[1];
  }

  // calculate mean field shifts
  std::array<MatrixXcd, 2> oneBodyOperator;
  oneBodyOperator[0] = ham.h1uMod[0];
  oneBodyOperator[1] = ham.h1uMod[1];
  complex<double> constant(0., 0.);
  constant += ene0 - ham.ecore;
  for (int i = 0; i < nfields; i++) {
    std::array<MatrixXd, 2> chol;
    chol[0] = ham.cholu[i][0];
    chol[1] = ham.cholu[i][1];
    if (ham.rotFlag == true) {
      size_t triSize = (norbs * (norbs + 1) / 2);
      chol[0] = MatrixXd::Zero(norbs, norbs);
      chol[1] = MatrixXd::Zero(norbs, norbs);
      long counter = 0;
      for (int j = 0; j < norbs; j++) {
        for (int k = 0; k <= j; k++) {
          chol[0](j, k) = ham.floatChol[i * triSize + counter];
          chol[0](k, j) = ham.floatChol[i * triSize + counter];
          chol[1](j, k) = ham.floatChol[nfields * triSize + i * triSize + counter];
          chol[1](k, j) = ham.floatChol[nfields * triSize + i * triSize + counter];
          counter++;
        }
      }
    }
    std::array<MatrixXcd, 2> op;
    op[0] = complex<double>(0., 1.) * chol[0];
    op[1] = complex<double>(0., 1.) * chol[1];
    complex<double> mfShift = 1. * green[0].cwiseProduct(op[0]).sum() + 1. * green[1].cwiseProduct(op[1]).sum();
    constant -= pow(mfShift, 2) / 2.;
    oneBodyOperator[0] -= mfShift * op[0];
    oneBodyOperator[1] -= mfShift * op[1];
    if (phaselessQ) mfShifts.push_back(mfShift);
    else mfShifts.push_back(mfShift /(1. * (ham.nalpha + ham.nbeta)));
  }

  if (phaselessQ) {
    propConstant[0] = constant - ene0;
    propConstant[1] = constant - ene0;
  }
  else {
    propConstant[0] = constant / (1. * ham.nalpha);
    propConstant[1] = constant / (1. * ham.nbeta);
  }
  expOneBodyOperatorU[0] =  (-dt * oneBodyOperator[0] / 2.).exp();
  expOneBodyOperatorU[1] =  (-dt * oneBodyOperator[1] / 2.).exp();
};


// for soc or gi
// only works for phaseless
// ene0 not used
void DQMCWalker::prepProp(Eigen::MatrixXcd& ref, Hamiltonian& ham, double pdt, double ene0)
{
  dt = pdt;
  int norbs = ham.norbs;
  int nfields = ham.nchol;

  // calculate rdm 
  MatrixXcd green = ref * ref.adjoint();
  MatrixXcd greenTrace;
  if (ham.socQ) greenTrace = green.block(0, 0, norbs, norbs) + green.block(norbs, norbs, norbs, norbs);

  // calculate mean field shifts
  complex<double> constant(0., 0.);
  MatrixXcd oneBodyOperator;
  if (ham.socQ) {
    oneBodyOperator = ham.h1socMod;
    constant += ene0 - ham.ecore;
    for (int i = 0; i < nfields; i++) {
      MatrixXcd op = complex<double>(0., 1.) * ham.chol[i];
      complex<double> mfShift = 1. * greenTrace.cwiseProduct(op).sum();
      constant -= pow(mfShift, 2) / 2.;
      oneBodyOperator.block(0, 0, norbs, norbs) -= mfShift * op;
      oneBodyOperator.block(norbs, norbs, norbs, norbs) -= mfShift * op;
      mfShifts.push_back(mfShift);
    }
  }
  else if (ham.intType == "g") {
    oneBodyOperator = ham.h1Mod;
    constant += ene0 - ham.ecore;
    for (int i = 0; i < nfields; i++) {
      MatrixXcd op = complex<double>(0., 1.) * ham.chol[i];
      complex<double> mfShift = 1. * green.cwiseProduct(op).sum();
      constant -= pow(mfShift, 2) / 2.;
      oneBodyOperator -= mfShift * op;
      mfShifts.push_back(mfShift);
    }
  }
  
  propConstant[0] = constant - ene0;
  propConstant[1] = constant - ene0;
  expOneBodyOperator =  (-dt * oneBodyOperator / 2.).exp();
};


void DQMCWalker::setDet(std::array<Eigen::MatrixXcd, 2> pdet) 
{
  det = pdet;
  orthoFac = complex<double> (1., 0.);
};


void DQMCWalker::setDet(Eigen::MatrixXcd pdet) 
{
  detG = pdet;
  orthoFac = complex<double> (1., 0.);
};
 

void DQMCWalker::setDet(std::vector<std::complex<double>>& serial, std::complex<double> ptrialOverlap)
{
  trialOverlap = ptrialOverlap;
  if (szQ) {
    int norbs = detG.rows(), nelec = detG.cols();
    for (int i = 0; i < norbs; i++)
      for (int j = 0; j < nelec; j++)
        detG(i, j) = serial[i * nelec + j];
  }
  else {
    int norbs = det[0].rows(), nalpha = det[0].cols(), nbeta = det[1].cols();
    for (int i = 0; i < norbs; i++)
      for (int j = 0; j < nalpha; j++)
        det[0](i, j) = serial[i * nalpha + j];
    for (int i = 0; i < norbs; i++)
      for (int j = 0; j < nbeta; j++)
        det[1](i, j) = serial[nalpha * norbs + i * nbeta + j];
  }
};


std::complex<double> DQMCWalker::getDet(std::vector<std::complex<double>>& serial)
{
  if (szQ) {
    int norbs = detG.rows(), nelec = detG.cols();
    for (int i = 0; i < norbs; i++)
      for (int j = 0; j < nelec; j++)
        serial[i * nelec + j] = detG(i, j);
  }
  else {
    int norbs = det[0].rows(), nalpha = det[0].cols(), nbeta = det[1].cols();
    for (int i = 0; i < norbs; i++)
      for (int j = 0; j < nalpha; j++)
        serial[i * nalpha + j] = det[0](i, j);
    for (int i = 0; i < norbs; i++)
      for (int j = 0; j < nbeta; j++)
        serial[nalpha * norbs + i * nbeta + j] = det[1](i, j);
  }
  return trialOverlap;
};


// for phaseless propagation the normalization does not matter
// it is calculated so as to not repeat it for the overlap ratio
void DQMCWalker::orthogonalize()
{
  complex<double> tempOrthoFac(1., 0.);
  if (szQ) {
    HouseholderQR<MatrixXcd> qr1(detG);
    detG = qr1.householderQ() * MatrixXd::Identity(detG.rows(), detG.cols());
    for (int i = 0; i < qr1.matrixQR().diagonal().size(); i++) tempOrthoFac *= qr1.matrixQR().diagonal()(i);
    trialOverlap /= tempOrthoFac;
  }
  else {
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
  }
};


// for free propagation only works for r integrals
void DQMCWalker::propagate(Hamiltonian& ham)
{
  int norbs = det[0].rows();
  int nfields = ham.nchol; 
  //MatrixXf prop = MatrixXf::Zero(norbs, norbs);
  vector<float> prop(norbs * (norbs + 1) / 2, 0.);
  complex<double> shift(0., 0.);
  VectorXd fields(nfields);
  fields.setZero();
  size_t triSize = (norbs * (norbs + 1)) / 2;
  for (int n = 0; n < nfields; n++) {
    double field_n = normal(generator);
    fields(n) = field_n;
    for (int i = 0; i < norbs; i++)
      for (int j = 0; j <= i; j++)
        prop[i * (i + 1) / 2 + j] += float(field_n) * ham.floatChol[n * triSize + i * (i + 1) / 2 + j];
    //prop.noalias() += float(field_n) * floatChol[i];
    shift += field_n * mfShifts[n];
  }
  //MatrixXcd propc = sqrt(dt) * complex<double>(0, 1.) * prop.cast<double>();
  MatrixXcd propc = MatrixXcd::Zero(norbs, norbs);
  for (int i = 0; i < norbs; i++) {
    propc(i, i) = sqrt(dt) * static_cast<complex<double>>(complex<float>(0, 1.) * prop[i * (i + 1) / 2 + i]);
    for (int j = 0; j < i; j++) {
      propc(i, j) = sqrt(dt) * static_cast<complex<double>>(complex<float>(0, 1.) * prop[i * (i + 1) / 2 + j]);
      propc(j, i) = sqrt(dt) * static_cast<complex<double>>(complex<float>(0, 1.) * prop[i * (i + 1) / 2 + j]);
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


void VHS(VectorXcf& fields, float* floatChol, MatrixXcd& propc) 
{
  int norbs = propc.rows();
  int nfields = fields.size();
  vector<float> propr(norbs * (norbs + 1) / 2, 0.);
  vector<float> propi(norbs * (norbs + 1) / 2, 0.);
  size_t triSize = (norbs * (norbs + 1)) / 2;
  for (int n = 0; n < nfields; n++) {
    for (int i = 0; i < norbs; i++) {
      for (int j = 0; j <= i; j++) {
        propr[i * (i + 1) / 2 + j] += fields(n).real() * floatChol[n * triSize + i * (i + 1) / 2 + j];
        propi[i * (i + 1) / 2 + j] += fields(n).imag() * floatChol[n * triSize + i * (i + 1) / 2 + j];
      }
    }
  }
  
  for (int i = 0; i < norbs; i++) {
    propc(i, i) = static_cast<complex<double>>(complex<float>(0., 1.) * propr[i * (i + 1) / 2 + i] - propi[i * (i + 1) / 2 + i]);
    for (int j = 0; j < i; j++) {
      propc(i, j) = static_cast<complex<double>>(complex<float>(0., 1.) * propr[i * (i + 1) / 2 + j] - propi[i * (i + 1) / 2 + j]);
      propc(j, i) = static_cast<complex<double>>(complex<float>(0., 1.) * propr[i * (i + 1) / 2 + j] - propi[i * (i + 1) / 2 + j]);
    }
  }
}

void VHS(VectorXcf& fields, Eigen::Map<Eigen::MatrixXf> floatCholMat, MatrixXcd& propc) 
{
  int norbs = propc.rows();
  int nfields = fields.size();
  size_t triSize = (norbs * (norbs + 1)) / 2;
  VectorXf propr = floatCholMat * fields.real();
  VectorXf propi = floatCholMat * fields.imag();
  
  for (int i = 0; i < norbs; i++) {
    propc(i, i) = static_cast<complex<double>>(complex<float>(0., 1.) * propr[i * (i + 1) / 2 + i] - propi[i * (i + 1) / 2 + i]);
    for (int j = 0; j < i; j++) {
      propc(i, j) = static_cast<complex<double>>(complex<float>(0., 1.) * propr[i * (i + 1) / 2 + j] - propi[i * (i + 1) / 2 + j]);
      propc(j, i) = static_cast<complex<double>>(complex<float>(0., 1.) * propr[i * (i + 1) / 2 + j] - propi[i * (i + 1) / 2 + j]);
    }
  }
}


void applyExp(MatrixXcd& propc, MatrixXcd& det) 
{
  MatrixXcd temp = det;
  for (int i = 1; i < 6; i++) {
    temp = propc * temp / i;
    det += temp;
  }
}


double DQMCWalker::propagatePhaseless(Wavefunction& wave, Hamiltonian& ham, double eshift)
{
  if (ham.intType == "r" || ham.intType == "g") return propagatePhaselessRG(wave, ham, eshift);
  else if (ham.intType == "u") return propagatePhaselessU(wave, ham, eshift);
}


double DQMCWalker::propagatePhaselessRG(Wavefunction& wave, Hamiltonian& ham, double eshift)
{
  int norbs = ham.norbs;
  int nelec = ham.nelec;
  int nfields = ham.nchol; 
  VectorXcd fb(nfields); fb.setZero();
  this->forceBias(wave, ham, fb);
  complex<double> shift(0., 0.), fbTerm(0., 0.);
  VectorXcf fields(nfields); fields.setZero();
  auto initTime = getTime();
  for (int n = 0; n < nfields; n++) {
    double field_n = normal(generator);
    complex<double> fieldShift = -sqrt(dt) * (complex<double>(0., 1.) * fb(n) - mfShifts[n]);
    fields(n) = complex<float>(field_n - fieldShift);
    shift += (field_n - fieldShift) * mfShifts[n];
    fbTerm += (field_n * fieldShift - fieldShift * fieldShift / 2.);
  }

  MatrixXcd propc = MatrixXcd::Zero(norbs, norbs);
  //VHS(fields, ham.floatChol, propc);
  VHS(fields, ham.floatCholMat[0], propc);
  propc *= sqrt(dt);
  vhsTime += getTime() - initTime;

  initTime = getTime();
  if (szQ && ham.socQ) {
    detG = expOneBodyOperator * detG;
    MatrixXcd temp = detG;
    for (int i = 1; i < 6; i++) {
      temp.block(0, 0, norbs, nelec)  = propc * temp.block(0, 0, norbs, nelec) / i;
      temp.block(norbs, 0, norbs, nelec)  = propc * temp.block(norbs, 0, norbs, nelec) / i;
      detG += temp;
    }
    detG = expOneBodyOperator * detG;
  }
  else if (szQ) {
    detG = expOneBodyOperator * detG;
    applyExp(propc, detG);
    detG = expOneBodyOperator * detG;
  }
  else {
    det[0] = expOneBodyOperator * det[0];
    applyExp(propc, det[0]);
    det[0] = expOneBodyOperator * det[0];

    if (rhfQ) det[1] = det[0];
    else {
      det[1] = expOneBodyOperator * det[1];
      applyExp(propc, det[1]);
      det[1] = expOneBodyOperator * det[1];
    }
  }
  expTime += getTime() - initTime;

  // phaseless
  complex<double> oldOverlap = trialOverlap;
  complex<double> newOverlap = this->overlap(wave);
  complex<double> importanceFunction = exp(-sqrt(dt) * shift + fbTerm + dt * (eshift + propConstant[0])) * newOverlap / oldOverlap;
  double theta = std::arg( exp(-sqrt(dt) * shift) * newOverlap / oldOverlap );
  double importanceFunctionPhaseless = std::abs(importanceFunction) * cos(theta);
  if (importanceFunctionPhaseless < 1.e-3 || importanceFunctionPhaseless > 100. || std::isnan(importanceFunctionPhaseless)) importanceFunctionPhaseless = 0.; 
  return importanceFunctionPhaseless;
};


double DQMCWalker::propagatePhaselessU(Wavefunction& wave, Hamiltonian& ham, double eshift)
{
  int norbs = ham.norbs;
  int nelec = ham.nelec;
  int nfields = ham.nchol; 
  VectorXcd fb(nfields); fb.setZero();
  this->forceBias(wave, ham, fb);
  MatrixXf propUp = MatrixXf::Zero(norbs, norbs);
  MatrixXf propDn = MatrixXf::Zero(norbs, norbs);
  complex<double> shift(0., 0.), fbTerm(0., 0.);
  VectorXcf fields(nfields); fields.setZero();
  auto initTime = getTime();
  for (int n = 0; n < nfields; n++) {
    double field_n = normal(generator);
    complex<double> fieldShift = -sqrt(dt) * (complex<double>(0., 1.) * fb(n) - mfShifts[n]);
    fields(n) = complex<float>(field_n - fieldShift);
    shift += (field_n - fieldShift) * mfShifts[n];
    fbTerm += (field_n * fieldShift - fieldShift * fieldShift / 2.);
  }

  MatrixXcd propUpc = MatrixXcd::Zero(norbs, norbs);
  MatrixXcd propDnc = MatrixXcd::Zero(norbs, norbs);
  //VHS(fields, ham.floatChol, propUpc);
  //VHS(fields, ham.floatChol + nfields * (norbs * (norbs+1)) / 2, propDnc);
  VHS(fields, ham.floatCholMat[0], propUpc);
  VHS(fields, ham.floatCholMat[1], propDnc);
  propUpc *= sqrt(dt);
  propDnc *= sqrt(dt);
  vhsTime += getTime() - initTime;

  initTime = getTime();
  det[0] = expOneBodyOperatorU[0] * det[0];
  applyExp(propUpc, det[0]);
  det[0] = expOneBodyOperatorU[0] * det[0];

  det[1] = expOneBodyOperatorU[1] * det[1];
  applyExp(propDnc, det[1]);
  det[1] = expOneBodyOperatorU[1] * det[1];
  expTime += getTime() - initTime;

  // phaseless
  complex<double> oldOverlap = trialOverlap;
  complex<double> newOverlap = this->overlap(wave);
  complex<double> importanceFunction = exp(-sqrt(dt) * shift + fbTerm + dt * (eshift + propConstant[0])) * newOverlap / oldOverlap;
  double theta = std::arg( exp(-sqrt(dt) * shift) * newOverlap / oldOverlap );
  double importanceFunctionPhaseless = std::abs(importanceFunction) * cos(theta);
  if (importanceFunctionPhaseless < 1.e-3 || importanceFunctionPhaseless > 100. || std::isnan(importanceFunctionPhaseless)) importanceFunctionPhaseless = 0.; 
  return importanceFunctionPhaseless;
};


// orthoFac not considered
// only used in phaseless
std::complex<double> DQMCWalker::overlap(Wavefunction& wave)
{
  std::complex<double> overlap;
  if (szQ) overlap = wave.overlap(detG); 
  else overlap = rhfQ ? wave.overlap(det[0]) : wave.overlap(det);
  trialOverlap = overlap;
  return overlap;
};


void DQMCWalker::forceBias(Wavefunction& wave, Hamiltonian& ham, Eigen::VectorXcd& fb)
{
  auto initTime = getTime();
  if (szQ) wave.forceBias(detG, ham, fb);
  else if (rhfQ) wave.forceBias(det[0], ham, fb);
  else wave.forceBias(det, ham, fb);
  fbTime += getTime() - initTime;
};


void DQMCWalker::oneRDM(Wavefunction& wave, Eigen::MatrixXcd& rdmSample)
{
  if (szQ) wave.oneRDM(detG, rdmSample);
  else wave.oneRDM(det, rdmSample);
};


void DQMCWalker::oneRDM(Wavefunction& wave, std::array<Eigen::MatrixXcd, 2>& rdmSample)
{
  wave.oneRDM(det, rdmSample);
};


std::array<std::complex<double>, 2> DQMCWalker::hamAndOverlap(Wavefunction& wave, Hamiltonian& ham)
{

  std::array<complex<double>, 2> hamOverlap;
  if (szQ) hamOverlap = wave.hamAndOverlap(detG, ham);
  else hamOverlap = rhfQ ? wave.hamAndOverlap(det[0], ham) : wave.hamAndOverlap(det, ham);
  hamOverlap[0] *= orthoFac;
  hamOverlap[1] *= orthoFac;
  return hamOverlap;
};


