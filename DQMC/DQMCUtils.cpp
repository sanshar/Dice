#include <iostream>
#include <fstream>
#include <string>
#include <unsupported/Eigen/MatrixFunctions>
#include "global.h"
#include "input.h"
#include "Determinants.h"
#include "DQMCUtils.h"
#include "DQMCMatrixElements.h"

using namespace Eigen;
using namespace std;

using matPair = pair<MatrixXcd, MatrixXcd>;
using vecPair = pair<VectorXcd, VectorXcd>;

// binning 
void binning(VectorXcd& samples, VectorXd& stdDev, VectorXi binSizes) 
{
  size_t numSamples = samples.size();
  stdDev = VectorXd::Zero(binSizes.size());
  
  // make bins
  for (int n = 0; n < binSizes.size(); n++) {
    ArrayXcd binned = VectorXcd::Zero(numSamples / binSizes(n));
    for (int i = 0; i < binned.size(); i++) {
      binned(i) = samples.segment(i * binSizes(n), binSizes(n)).mean();
    }
    stdDev(n) = sqrt((binned - binned.mean()).abs().square().sum() / (binned.size() - 1) / binned.size());
  }
}


// matrix exponential
MatrixXcd matExp(const MatrixXcd& mat, const int order) 
{
  if (order < 0)
    return mat.exp();
  else {
    size_t n = mat.rows();
    MatrixXcd temp = MatrixXcd::Identity(n, n);
    MatrixXcd mExp = MatrixXcd::Identity(n, n);
    for (int i = 1; i < order; i++) {
      temp = mat * temp / i;
      mExp += temp;
    }
    return mExp;
  }
}


void orthogonalize(matPair& rn, complex<double>& orthoFac){
  HouseholderQR<MatrixXcd> qr1(rn.first);
  HouseholderQR<MatrixXcd> qr2(rn.second);
  rn.first = qr1.householderQ() * MatrixXd::Identity(Determinant::norbs, Determinant::nalpha);
  rn.second = qr2.householderQ() * MatrixXd::Identity(Determinant::norbs, Determinant::nbeta);
  for (int i = 0; i < qr1.matrixQR().diagonal().size(); i++) orthoFac *= qr1.matrixQR().diagonal()(i);
  for (int i = 0; i < qr2.matrixQR().diagonal().size(); i++) orthoFac *= qr2.matrixQR().diagonal()(i);
}


// reads jastrow in VMC format (not exponential)
// makes hs operators including mean field subtraction and the offshoot one body operator, returns mean field constant 
complex<double> prepJastrowHS(matPair& ref, vector<vecPair>& hsOperators, vecPair& oneBodyOperator)
{

  size_t norbs = ref.first.rows();
  
  // read jastrow
  MatrixXd jastrow = MatrixXd::Zero(2*norbs, 2*norbs);
  readMat(jastrow, "Jastrow.txt");

  // VMC format: only lower triangle + diagonal has information
  // even indices are up, odd are down
  // convert to exponential form
  MatrixXd expJastrow = MatrixXd::Zero(2*norbs, 2*norbs);
  for (int i = 0; i < 2*norbs; i++) 
    expJastrow(i, i) = log(jastrow(i, i));
  for (int i = 0; i < 2*norbs; i++) { 
    for (int j = 0; j < i; j++) {
      expJastrow(i, j) = log(jastrow(i, j))/2;
      expJastrow(j, i) = log(jastrow(i, j))/2;
    }
  }

  // calculate hs operators by diagonalizing expJastrow
  SelfAdjointEigenSolver<MatrixXd> eigensolver(expJastrow);
  VectorXd eigenvalues = eigensolver.eigenvalues();
  MatrixXd eigenvectors = eigensolver.eigenvectors();
 
  oneBodyOperator.first = VectorXcd::Zero(norbs);
  oneBodyOperator.second = VectorXcd::Zero(norbs);

  // calculate rdm for mean field shifts
  matPair refT;
  refT.first = ref.first.adjoint();
  refT.second = ref.second.adjoint();
  matPair green;
  calcGreensFunction(refT, ref, green);
  
  complex<double> mfConstant(0., 0.);

  for (int i = 0; i < 2*norbs; i++) {
    VectorXcd up(norbs), dn(norbs);
    for (int j = 0; j < 2*norbs; j++) {
      if (j%2 == 0) up(j/2) = eigenvectors(j, i);
      else dn(j/2) = eigenvectors(j, i);
    }
    
    // calculate shifts
    complex<double> mfShiftUp = 1. * green.first.diagonal().cwiseProduct(up).sum();
    complex<double> mfShiftDn = 1. * green.second.diagonal().cwiseProduct(dn).sum();
    
    // update one body ops
    oneBodyOperator.first += 2 * eigenvalues(i) * (mfShiftUp + mfShiftDn) * up;
    oneBodyOperator.second +=  2 * eigenvalues(i) * (mfShiftUp + mfShiftDn) * dn;
    
    // constant
    mfConstant -= eigenvalues(i) * pow(mfShiftUp + mfShiftDn, 2);

    // make shifted hs ops
    up -= VectorXcd::Constant(norbs, mfShiftUp/(1.*Determinant::nalpha));
    dn -= VectorXcd::Constant(norbs, mfShiftDn/(1.*Determinant::nbeta));
    vecPair op;
    op.first = sqrt(complex<double>(2*eigenvalues(i), 0.)) * up;
    op.second = sqrt(complex<double>(2*eigenvalues(i), 0.)) * dn;
    hsOperators.push_back(op);
  }
 
  return mfConstant;
}


// reads jastrow in VMC format (not exponential)
// makes hs operators including mean field subtraction and the offshoot one body operator, returns mean field constant 
// alternative to the one above
complex<double> prepJastrowHS1(matPair& ref, vector<vecPair>& hsOperators, vecPair& oneBodyOperator)
{
  size_t norbs = Determinant::norbs;
  
  // read jastrow
  MatrixXd jastrow = MatrixXd::Zero(2*norbs, 2*norbs);
  readMat(jastrow, "Jastrow.txt");

  // VMC format: only lower triangle + diagonal has information
  // even indices are up, odd are down
  // convert to exponential form
  MatrixXd expJastrow = MatrixXd::Zero(2*norbs, 2*norbs);
  for (int i = 0; i < 2*norbs; i++) 
    expJastrow(i, i) = log(jastrow(i, i));
  for (int i = 0; i < 2*norbs; i++) { 
    for (int j = 0; j < i; j++) {
      expJastrow(i, j) = log(jastrow(i, j))/2;
      expJastrow(j, i) = log(jastrow(i, j))/2;
    }
  }

  // calculate hs operators for each number operator pair product
  oneBodyOperator.first = VectorXcd::Zero(norbs);
  oneBodyOperator.second = VectorXcd::Zero(norbs);

  // calculate rdm for mean field shifts
  matPair refT;
  refT.first = ref.first.adjoint();
  refT.second = ref.second.adjoint();
  matPair green;
  calcGreensFunction(refT, ref, green);
  
  complex<double> mfConstant(0., 0.);

  for (int i = 0; i < 2*norbs; i++) {
    // no shift for linear terms on the diagonal
    complex<double> mfShifti;
    if (i%2 == 0) {
      oneBodyOperator.first(i/2) += expJastrow(i, i);
      mfShifti = 1. * green.first.diagonal()(i/2);
    }
    else {
      oneBodyOperator.second(i/2) +=  expJastrow(i, i);
      mfShifti = 1. * green.second.diagonal()(i/2);
    }
    for (int j = 0; j < i; j++) {
      complex<double> sqrtFac = sqrt(complex<double>(2 * expJastrow(i, j), 0.));
      vecPair op;
      op.first = VectorXcd::Zero(norbs);
      op.second = VectorXcd::Zero(norbs);
      complex<double> mfShiftj;
      if (j%2 == 0) {
        mfShiftj = 1. * green.first.diagonal()(j/2);
        oneBodyOperator.first(j/2) -= 1. - 2. * (mfShifti + mfShiftj);
        op.first(j/2) += sqrtFac;
        op.first -= VectorXcd::Constant(norbs, sqrtFac * mfShiftj/(1.*Determinant::nalpha));
      }
      else {
        mfShiftj = 1. * green.second.diagonal()(j/2);
        oneBodyOperator.second(j/2) -= 1. - 2. * (mfShifti + mfShiftj);
        op.second(j/2) += sqrtFac;
        op.second -= VectorXcd::Constant(norbs, sqrtFac * mfShiftj/(1.*Determinant::nbeta));
      }

      if (i%2 == 0) {
        oneBodyOperator.first(i/2) -= 1. - 2. * (mfShifti + mfShiftj);
        op.first(i/2) += sqrtFac;
        op.first -= VectorXcd::Constant(norbs, sqrtFac * mfShifti/(1.*Determinant::nalpha));
      }
      else {
        oneBodyOperator.second(i/2) -= 1. - 2. * (mfShifti + mfShiftj);
        op.second(i/2) += sqrtFac;
        op.second -= VectorXcd::Constant(norbs, sqrtFac * mfShifti/(1.*Determinant::nbeta));
      }
      hsOperators.push_back(op);

      mfConstant -= expJastrow(i, j) * pow(mfShifti + mfShiftj, 2);
    }
  }
    
  return mfConstant;
}


// makes hs operators from cholesky matrices including mean field subtraction, returns mean field constant
complex<double> prepPropagatorHS(matPair& ref, vector<MatrixXd>& chol, vector<matPair>& hsOperators, matPair& oneBodyOperator)
{
  size_t norbs = Determinant::norbs;
  size_t nfields = chol.size();

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
    green.first = MatrixXcd::Zero(norbs, norbs);
    green.second = MatrixXcd::Zero(norbs, norbs);
    for (int i = 0; i < 2*norbs; i++) {
      for (int j = 0; j < 2*norbs; j++) {
        if (i%2 == 0 && j%2 == 0) green.first(i/2, j/2) = oneRDM(i, j); 
        else if (i%2 == 1 && j%2 == 1) green.second(i/2, j/2) = oneRDM(i, j); 
      }
    }
  }
  else {
    if (commrank == 0) cout << "Using RHF RDM for background subtraction\n\n";
    matPair refT;
    refT.first = ref.first.adjoint();
    refT.second = ref.second.adjoint();
    calcGreensFunction(refT, ref, green);
  }

  oneBodyOperator.first = MatrixXcd::Zero(norbs, norbs);
  oneBodyOperator.second = MatrixXcd::Zero(norbs, norbs);
  
  complex<double> mfConstant(0., 0.);

  for (int i = 0; i < nfields; i++) {
    matPair op;
    op.first = complex<double>(0., 1.) * chol[i];
    op.second = complex<double>(0., 1.) * chol[i];

    // calculate shifts
    complex<double> mfShiftUp = 1. * green.first.cwiseProduct(op.first).sum();
    complex<double> mfShiftDn = 1. * green.second.cwiseProduct(op.second).sum();
    
    // constant
    mfConstant += pow(mfShiftUp + mfShiftDn, 2);

    // update one body ops
    oneBodyOperator.first += (mfShiftUp + mfShiftDn) * op.first;
    oneBodyOperator.second += (mfShiftUp + mfShiftDn) * op.second;
    
    // make shifted hs ops
    op.first.diagonal() -= VectorXcd::Constant(norbs, mfShiftUp/(1.*Determinant::nalpha));
    op.second.diagonal() -= VectorXcd::Constant(norbs, mfShiftDn/(1.*Determinant::nbeta));
    hsOperators.push_back(op);
  }

  return mfConstant / 2.;
}
