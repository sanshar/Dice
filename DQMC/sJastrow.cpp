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
  else ref = rhf; 
  
  refState[0] = ref.block(0, 0, norbs, pnalpha);
  refState[1] = ref.block(0, 0, norbs, pnbeta);

  if (commrank == 0) cout << "\nPreparing Jastrow HS operators\n";
  SelfAdjointEigenSolver<MatrixXd> eigensolver(jastrow);
  VectorXd eigenvalues = eigensolver.eigenvalues();
  MatrixXd eigenvectors = eigensolver.eigenvectors();

  oneBodyOperator = VectorXcd::Zero(norbs);
  matPair green, rhfState, rhfStateT;
  rhfState[0] = rhf.block(0, 0, norbs, pnalpha);
  rhfState[1] = rhf.block(0, 0, norbs, pnbeta);
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


// to be defined
std::array<std::complex<double>, 2> sJastrow::hamAndOverlap(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham)
{
  return std::array<complex<double>, 2>();
};


// to be defined
std::array<std::complex<double>, 2> sJastrow::hamAndOverlap(Eigen::MatrixXcd& psi, Hamiltonian& ham)
{
  return std::array<complex<double>, 2>();
};
