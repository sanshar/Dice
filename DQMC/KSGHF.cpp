#include "input.h"
#include "KSGHF.h"

using namespace std;
using namespace Eigen;

using matPair = std::array<MatrixXcd, 2>;

KSGHF::KSGHF(Hamiltonian& ham, bool pleftQ, std::string fname) 
{
  int norbs = ham.norbs;
  int nelec = ham.nalpha + ham.nbeta;
  MatrixXcd hf = MatrixXcd::Zero(2*norbs, 2*norbs);
  readMat(hf, fname);
  det = hf.block(0, 0, 2*norbs, nelec);
  detAd = det.adjoint();
  detT = det.transpose();
  leftQ = pleftQ;
  if (leftQ) {
    for (int i = 0; i < ham.nchol; i++) {
      MatrixXcd rotUp = detT.block(0, 0, nelec, norbs) * ham.chol[i];
      MatrixXcd rotDn = detT.block(0, norbs, nelec, norbs)  * ham.chol[i];
      rotCholT[0].push_back(rotUp);
      rotCholT[1].push_back(rotDn);

      rotUp = detAd.block(0, 0, nelec, norbs) * ham.chol[i];
      rotDn = detAd.block(0, norbs, nelec, norbs) * ham.chol[i];
      rotCholAd[0].push_back(rotUp);
      rotCholAd[1].push_back(rotDn);
    }
  }
};


// not supported yet
void KSGHF::getSample(std::array<Eigen::MatrixXcd, 2>& sampleDet) 
{
  return;
};

std::array<complex<double>, 2> calcHamiltonianElement(MatrixXcd& ghf, matPair& psi, Hamiltonian& ham, std::array<vector<MatrixXcd>, 2>& rotChol)
{
  // core energy
  complex<double> ene = ham.ecore;
  
  MatrixXcd B = ghf.adjoint();

  // calculate theta and green
  int numOrbs = psi[0].rows();
  int numElec = B.cols();
  int nAlpha = psi[0].cols();
  int nBeta = psi[1].cols();

  MatrixXcd Afull = 0*B, theta, green;
  Afull.block(0, 0, numOrbs, psi[0].cols()) = psi[0];
  Afull.block(numOrbs, psi[0].cols(), numOrbs, psi[1].cols()) = psi[1];

  MatrixXcd Ovlp = B.adjoint()*Afull;
  complex<double> ovlp = Ovlp.determinant();
  theta = Afull * Ovlp.inverse();
  green = (theta * B.adjoint()).transpose();

  // one body part
  ene += green.block(0, 0, numOrbs, numOrbs).cwiseProduct(ham.h1).sum() + green.block(numOrbs, numOrbs, numOrbs, numOrbs).cwiseProduct(ham.h1).sum();

  MatrixXcd thetaA = theta.block(0, 0, numOrbs, numElec), thetaB = theta.block(numOrbs, 0, numOrbs, numElec);

  // two body part
  vector<MatrixXcd> W(2, MatrixXcd(numElec, numOrbs));

  for (int i = 0; i < ham.nchol; i++) {
    W[0] = rotChol[0][i] * thetaA;
    W[1] = rotChol[1][i] * thetaB;

    complex<double> W0trace = W[0].trace(), W1trace = W[1].trace();
    complex<double> J = (W0trace + W1trace) * (W0trace + W1trace);
    ene += J/2.;

    complex<double> K = W[0].cwiseProduct(W[0].transpose()).sum() + 
                        W[1].cwiseProduct(W[1].transpose()).sum() +
                        W[0].cwiseProduct(W[1].transpose()).sum() +
                        W[1].cwiseProduct(W[0].transpose()).sum() ;

    ene -= K/2.; 
  }

  std::array<complex<double>, 2> hamOverlap;
  hamOverlap[0] = ene * ovlp;
  hamOverlap[1] = ovlp;
  return hamOverlap;
};

std::array<std::complex<double>, 2> KSGHF::hamAndOverlap(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham) 
{
  auto sample1 = calcHamiltonianElement(detAd, psi, ham, rotCholAd);
  auto sample2 = calcHamiltonianElement(detT, psi, ham, rotCholT);
  std::array<complex<double>, 2> hamOverlap;
  hamOverlap[0] = sample1[0] + sample2[0];
  hamOverlap[1] = sample1[1] + sample2[1];
  return hamOverlap;
};


std::array<std::complex<double>, 2> KSGHF::hamAndOverlap(Eigen::MatrixXcd& psi, Hamiltonian& ham) 
{
  matPair psiP;
  psiP[0] = psi;
  psiP[1] = psi;
  return hamAndOverlap(psiP, ham);
};
