#include <iostream>
#include <algorithm>
#include <unsupported/Eigen/MatrixFunctions>
#include "global.h"
#include "input.h"
#include "CCSD.h"

using namespace std;
using namespace Eigen;

using matPair = std::array<MatrixXcd, 2>;

CCSD::CCSD(int pnorbs, int pnocc, std::string fname)
{
  norbs = pnorbs;
  nocc = pnocc;
  int nopen = norbs - nocc;
  int nexc = nocc * nopen;

  // read amplitudes
  MatrixXd singles = MatrixXd::Zero(nocc, nopen);
  MatrixXd doubles = MatrixXd::Zero(nexc, nexc);
  basisRotation = MatrixXd::Zero(norbs, norbs);
  readCCSD(singles, doubles, basisRotation, fname);

  if (commrank == 0) cout << "\nPreparing CC HS operators\n";
  oneBodyOperator = singles.transpose();

  // calculate hs operators by diagonalizing doubles amplitudes
  SelfAdjointEigenSolver<MatrixXd> eigensolver(doubles);
  VectorXd eigenvalues = eigensolver.eigenvalues();
  MatrixXd eigenvectors = eigensolver.eigenvectors();

  for (int i = 0; i < nexc; i++) {
    MatrixXcd op = MatrixXcd::Zero(nopen, nocc);
    for (int j = 0; j < nexc; j++) {
      int p = j / nopen, k = j % nopen;
      op(k, p) += eigenvectors(j, i);
    }
    op *= sqrt(complex<double>(1.*eigenvalues(i), 0.));
    hsOperators.push_back(op);
  }
  
  if (commrank == 0) cout << "Finished preparing CC HS operators\n\n";
  
  normal = normal_distribution<double>(0., 1.);
};


void CCSD::getSample(std::array<Eigen::MatrixXcd, 2>& sampleDet)
{
  MatrixXcd ccref = MatrixXcd::Identity(norbs, nocc); 
  ccref.block(nocc, 0, norbs-nocc, nocc)  += oneBodyOperator;
  MatrixXcd ccprop = MatrixXcd::Zero(norbs-nocc, nocc);
  for (int i = 0; i < hsOperators.size(); i++) {
    double ccfield_i = normal(generator);
    ccprop.noalias() += ccfield_i * hsOperators[i];
  }
  MatrixXcd det = ccref;
  det.block(nocc, 0, norbs-nocc, nocc) += ccprop;

  // rotate to ham basis
  det = basisRotation * det;

  sampleDet[0] = det;
  sampleDet[1] = det;
};


// to be defined
std::array<std::complex<double>, 2> CCSD::hamAndOverlap(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham)
{
  return std::array<complex<double>, 2>();
};


// to be defined
std::array<std::complex<double>, 2> CCSD::hamAndOverlap(Eigen::MatrixXcd& psi, Hamiltonian& ham)
{
  return std::array<complex<double>, 2>();
};
