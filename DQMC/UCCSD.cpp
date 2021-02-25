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
  std::array<MatrixXd, 2> singles, doubles;
  for (int sz = 0; sz < 2; sz++) {
    singles[sz] = MatrixXd::Zero(nocc[sz], nopen[sz]);
    doubles[sz] = MatrixXd::Zero(nexc[sz], nexc[sz]);
    basisRotation[sz] = MatrixXd::Zero(norbs, norbs);
  }
  readUCCSD(singles, doubles, basisRotation, fname);

  cout << "singles[0]\n" << singles[0] << endl << endl;
  cout << "singles[1]\n" << singles[1] << endl << endl;
  cout << "doubles[0]\n" << doubles[0] << endl << endl;
  cout << "doubles[1]\n" << doubles[1] << endl << endl;
  cout << "basisRotation[0]\n" << basisRotation[0] << endl << endl;
  cout << "basisRotation[1]\n" << basisRotation[1] << endl << endl;

  if (commrank == 0) cout << "\nPreparing CC HS operators\n";
  for (int sz = 0; sz < 2; sz++) {
    oneBodyOperator[sz] = singles[sz].transpose();

    // calculate hs operators by diagonalizing doubles amplitudes
    SelfAdjointEigenSolver<MatrixXd> eigensolver(doubles[sz]);
    VectorXd eigenvalues = eigensolver.eigenvalues();
    MatrixXd eigenvectors = eigensolver.eigenvectors();

    for (int i = 0; i < nexc[sz]; i++) {
      MatrixXcd op = MatrixXcd::Zero(nopen[sz], nocc[sz]);
      for (int j = 0; j < nexc[sz]; j++) {
        int p = j / nopen[sz], k = j % nopen[sz];
        op(k, p) += eigenvectors(j, i);
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
  for (int sz = 0; sz < 2; sz++) {
    MatrixXcd ccref = MatrixXcd::Identity(norbs, nocc[sz]); 
    ccref.block(nocc[sz], 0, norbs-nocc[sz], nocc[sz])  += oneBodyOperator[sz];
    MatrixXcd ccprop = MatrixXcd::Zero(norbs-nocc[sz], nocc[sz]);
    for (int i = 0; i < hsOperators[sz].size(); i++) {
      double ccfield_i = normal(generator);
      ccprop.noalias() += ccfield_i * hsOperators[sz][i];
    }
    det[sz] = ccref;
    det[sz].block(nocc[sz], 0, norbs-nocc[sz], nocc[sz]) += ccprop;

    // rotate to ham basis
    det[sz] = basisRotation[sz] * det[sz];
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
