#include <iostream>
#include <fstream>
#include <algorithm>
#include <unsupported/Eigen/MatrixFunctions>
#include "global.h"
#include "input.h"
#include "sJastrow.h"

using namespace std;
using namespace Eigen;

using matPair = std::array<MatrixXcd, 2>;
using vecPair = std::array<VectorXcd, 2>;

sJastrow::sJastrow(int pnorbs, int pnalpha, int pnbeta, std::string fname)
{
  norbs = pnorbs;

  // read jastrow
  MatrixXd jastrow = MatrixXd::Zero(norbs, norbs);
  readMat(jastrow, "Jastrow.txt");
  // vmc format jastrow has upper triangle equal to zero
  for (int i = 0; i < norbs; i++) {
    for (int j = 0; j < i; j++) {
      jastrow(i, j) /= 2.;
      jastrow(j, i) = jastrow(i, j);
    }
  }

  bool readRot = false;
  std::string fname1 = "basisRotation.txt";
  ifstream rotFile(fname1);
  if (rotFile) readRot = true;
  rotFile.close();
  if (readRot) {
    basisRotation = MatrixXd::Zero(norbs, norbs);
    readMat(basisRotation, fname1);
  }
  else basisRotation = MatrixXd::Identity(norbs, norbs);

  MatrixXd rhf, ref;
  rhf = MatrixXd::Zero(norbs, norbs);
  readMat(rhf, "rhf.txt");
  ref = MatrixXd::Zero(norbs, norbs);
  bool readRef = false;
  std::string fname2 = "ref.txt";
  ifstream refFile(fname2);
  if (refFile) readRef = true;
  refFile.close();
  if (readRef) {
    ref = MatrixXd::Zero(norbs, norbs);
    readMat(ref, fname2); 
  }
  else {
    ref = basisRotation.transpose() * rhf; 
  }
  
  refState[0] = ref.block(0, 0, norbs, pnalpha);
  refState[1] = ref.block(0, 0, norbs, pnbeta);

  if (commrank == 0) cout << "\nPreparing Jastrow HS operators\n";
  SelfAdjointEigenSolver<MatrixXd> eigensolver(jastrow);
  VectorXd eigenvalues = eigensolver.eigenvalues();
  MatrixXd eigenvectors = eigensolver.eigenvectors();

  oneBodyOperator = VectorXcd::Zero(norbs);
  matPair green, rhfState, rhfStateT;
  rhfState[0] = basisRotation.transpose() * rhf.block(0, 0, norbs, pnalpha);
  rhfState[1] = basisRotation.transpose() * rhf.block(0, 0, norbs, pnbeta);
  rhfStateT[0] = rhfState[0].adjoint();
  rhfStateT[1] = rhfState[1].adjoint();
  green[0] = rhfState[0] * (rhfStateT[0] * rhfState[0]).inverse() * rhfStateT[0];
  green[1] = rhfState[1] * (rhfStateT[1] * rhfState[1]).inverse() * rhfStateT[1];

  for (int i = 0; i < norbs; i++) {
    VectorXcd op(norbs);
    for (int j = 0; j < norbs; j++) {
      op(j) = eigenvectors(j, i);
    }
    op *= sqrt(complex<double>(2.* eigenvalues(i), 0.));
    complex<double> mfShift = green[0].diagonal().cwiseProduct(op).sum() + green[1].diagonal().cwiseProduct(op).sum();
    oneBodyOperator += mfShift * op;
    op -= VectorXcd::Constant(norbs, mfShift/(pnalpha + pnbeta));
    hsOperators.push_back(op);
  }
  
  if (commrank == 0) cout << "Finished preparing Jastrow HS operators\n\n";
  VectorXcd expOneBodyOperator = oneBodyOperator.array().exp();
  refState[0] = expOneBodyOperator.asDiagonal() * refState[0];
  refState[1] = expOneBodyOperator.asDiagonal() * refState[1];
  
  normal = normal_distribution<double>(0., 1.);
};


void sJastrow::getSample(std::array<Eigen::MatrixXcd, 2>& sampleDet)
{
  VectorXcd jprop = VectorXcd::Zero(norbs);
  for (int i = 0; i < hsOperators.size(); i++) {
    double jfield_i = normal(generator);
    jprop.noalias() += jfield_i * hsOperators[i];
  }
  jprop = jprop.array().exp();
  matPair det;
  det[0] = jprop.asDiagonal() * refState[0];
  det[1] = jprop.asDiagonal() * refState[1];

  sampleDet[0] = basisRotation * det[0];
  sampleDet[1] = basisRotation * det[1];
};


complex<double> calcHamiltonianElement(matPair& phi1T, matPair& phi2, Hamiltonian& ham) 
{ 
  // core energy
  complex<double> ene = ham.ecore;
  
  // calculate theta and green
  int numActOrbs = phi1T[0].cols();
  int numElecUp = phi2[0].cols();
  int numElecDn = phi2[1].cols();
  matPair theta, green;
  theta[0] = phi2[0] * (phi1T[0] * phi2[0].block(0, 0, numActOrbs, numElecUp)).inverse();
  theta[1] = phi2[1] * (phi1T[1] * phi2[1].block(0, 0, numActOrbs, numElecDn)).inverse();
  green[0] = (theta[0] * phi1T[0]).transpose();
  green[1] = (theta[1] * phi1T[1]).transpose();

  // one body part
  ene += green[0].cwiseProduct(ham.h1.block(0, 0, numActOrbs, ham.h1.rows())).sum() + green[1].cwiseProduct(ham.h1.block(0, 0, numActOrbs, ham.h1.rows())).sum();

  // two body part
  MatrixXcd fup = MatrixXcd::Zero(phi1T[0].rows(), phi1T[0].rows());
  MatrixXcd fdn = MatrixXcd::Zero(phi1T[1].rows(), phi1T[1].rows());
  MatrixXcd rotCholUp = MatrixXcd::Zero(phi1T[0].rows(), ham.h1.rows());
  MatrixXcd rotCholDn = MatrixXcd::Zero(phi1T[1].rows(), ham.h1.rows());
  for (int i = 0; i < ham.nchol; i++) {
    rotCholUp.noalias() = phi1T[0] * ham.chol[i].block(0, 0, numActOrbs, ham.h1.rows());
    rotCholDn.noalias() = phi1T[1] * ham.chol[i].block(0, 0, numActOrbs, ham.h1.rows());
    fup.noalias() = rotCholUp * theta[0];
    fdn.noalias() = rotCholDn * theta[1];
    complex<double> cup = fup.trace();
    complex<double> cdn = fdn.trace();
    ene += (cup * cup + cdn * cdn + 2. * cup * cdn - fup.cwiseProduct(fup.transpose()).sum() - fdn.cwiseProduct(fdn.transpose()).sum()) / 2.;
  }

  return ene;
}


std::complex<double> sJastrow::overlap(std::array<Eigen::MatrixXcd, 2>& psi)
{
  return std::complex<double>();
};


std::complex<double> sJastrow::overlap(Eigen::MatrixXcd& psi)
{
  return std::complex<double>();
};

void sJastrow::forceBias(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham, Eigen::VectorXcd& fb)
{
  return;
};


void sJastrow::forceBias(Eigen::MatrixXcd& psi, Hamiltonian& ham, Eigen::VectorXcd& fb)
{
  return;
};


// to be defined
std::array<std::complex<double>, 2> sJastrow::hamAndOverlap(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham)
{
  complex<double> overlap(0., 0.), localEnergy(0., 0.);
  int nSamples = schd.numJastrowSamples;
  for (int n = 0; n < nSamples; n++) {
    VectorXcd jprop = VectorXcd::Zero(norbs);
    for (int i = 0; i < hsOperators.size(); i++) {
      double jfield_i = normal(generator);
      jprop.noalias() += jfield_i * hsOperators[i];
    }
    jprop = jprop.array().exp();
    matPair detT;
    detT[0] = refState[0].adjoint() * jprop.asDiagonal();
    detT[1] = refState[1].adjoint() * jprop.asDiagonal();
    
    complex<double> overlapSample = (detT[0] * psi[0]).determinant() * (detT[1] * psi[1]).determinant();
    overlap += overlapSample;
    localEnergy += overlapSample * calcHamiltonianElement(detT, psi, ham);
  }
  
  std::array<complex<double>, 2> hamOverlap;
  hamOverlap[0] = localEnergy / nSamples;
  hamOverlap[1] = overlap / nSamples;
  return hamOverlap;
};


// to be defined
std::array<std::complex<double>, 2> sJastrow::hamAndOverlap(Eigen::MatrixXcd& psi, Hamiltonian& ham)
{
  matPair psiP;
  psiP[0] = psi;
  psiP[1] = psi;
  return hamAndOverlap(psiP, ham);
};
