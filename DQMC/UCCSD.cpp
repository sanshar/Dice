#include <iostream>
#include <algorithm>
#include <unsupported/Eigen/MatrixFunctions>
#include "global.h"
#include "input.h"
#include "UCCSD.h"

using namespace std;
using namespace Eigen;

using matPair = std::array<MatrixXcd, 2>;

UCCSD::UCCSD(int pnorbs, int pnalpha, int pnbeta, std::string fname)
{
  norbs = pnorbs;
  nalpha = pnalpha;
  nbeta = pnbeta;
  std::array<int, 2> nocc = { nalpha, nbeta };
  std::array<int, 2> nopen = { norbs - nalpha, norbs - nbeta };
  std::array<int, 2> nexc = { nocc[0] * nopen[0], nocc[1] * nopen[1] };

  // read amplitudes
  // doubles: uu, dd, ud
  std::array<MatrixXd, 2> singles;
  std::array<MatrixXd, 3> doubles;
  for (int sz = 0; sz < 2; sz++) {
    singles[sz] = MatrixXd::Zero(nocc[sz], nopen[sz]);
    doubles[sz] = MatrixXd::Zero(nexc[sz], nexc[sz]);
  }
  doubles[2] = MatrixXd::Zero(nexc[0], nexc[1]);
  basisRotation = MatrixXd::Zero(norbs, norbs);
  readUCCSD(singles, doubles, basisRotation, fname);

  if (commrank == 0) cout << "\nPreparing CC HS operators\n";
  // calculate hs operators by diagonalizing doubles amplitudes
  MatrixXd doublesCombined = MatrixXd::Zero(nexc[0] + nexc[1], nexc[0] + nexc[1]);
  doublesCombined.block(0, 0, nexc[0], nexc[0]) = doubles[0] / 2;
  doublesCombined.block(nexc[0], nexc[0], nexc[1], nexc[1]) = doubles[1] / 2;
  doublesCombined.block(0, nexc[0], nexc[0], nexc[1]) = doubles[2];
  doublesCombined.block(nexc[0], 0, nexc[1], nexc[0]) = doubles[2].transpose();
  SelfAdjointEigenSolver<MatrixXd> eigensolver(doublesCombined);
  VectorXd eigenvalues = eigensolver.eigenvalues();
  MatrixXd eigenvectors = eigensolver.eigenvectors();

  oneBodyOperator[0] = singles[0].transpose();
  oneBodyOperator[1] = singles[1].transpose();
  for (int i = 0; i < nexc[0] + nexc[1]; i++) {
    for (int sz = 0; sz < 2; sz++) {
      int counter = 0;
      if (sz == 1) counter = nexc[0];
      MatrixXcd op = MatrixXcd::Zero(nopen[sz], nocc[sz]);
      for (int j = 0; j < nexc[sz]; j++) {
        int p = j / nopen[sz], k = j % nopen[sz];
        op(k, p) += eigenvectors(j + counter, i);
      }
      op *= sqrt(complex<double>(1.*eigenvalues(i), 0.));
      hsOperators[sz].push_back(op);
    }
  }
  
  if (commrank == 0) cout << "Finished preparing CC HS operators\n\n";
  
  normal = normal_distribution<double>(0., 1.);
};


void UCCSD::getSample(std::array<Eigen::MatrixXcd, 2>& sampleDet)
{
  matPair det;
  std::array<int, 2> nocc = { nalpha, nbeta };
  matPair ccprop, ccref;
  for (int sz = 0; sz < 2; sz++) {
    ccref[sz] = MatrixXcd::Identity(norbs, nocc[sz]); 
    ccref[sz].block(nocc[sz], 0, norbs-nocc[sz], nocc[sz]) += oneBodyOperator[sz];
    ccprop[sz] = MatrixXcd::Zero(norbs-nocc[sz], nocc[sz]);
  }
    
  for (int i = 0; i < hsOperators[0].size(); i++) {
    double ccfield_i = normal(generator);
    ccprop[0].noalias() += ccfield_i * hsOperators[0][i];
    ccprop[1].noalias() += ccfield_i * hsOperators[1][i];
  }

  for (int sz = 0; sz < 2; sz++) {
    det[sz] = ccref[sz];
    det[sz].block(nocc[sz], 0, norbs-nocc[sz], nocc[sz]) += ccprop[sz];

    // rotate to ham basis
    det[sz] = basisRotation * det[sz];
  }

  sampleDet = det;
};


// to be defined
std::array<std::complex<double>, 2> UCCSD::hamAndOverlap(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham)
{
  return std::array<complex<double>, 2>();
};


// to be defined
std::array<std::complex<double>, 2> UCCSD::hamAndOverlap(Eigen::MatrixXcd& psi, Hamiltonian& ham)
{
  return std::array<complex<double>, 2>();
};
