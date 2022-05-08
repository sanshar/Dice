#include "input.h"
#include "GHF.h"

using namespace std;
using namespace Eigen;

using matPair = std::array<MatrixXcd, 2>;


GHF::GHF(Hamiltonian& ham, bool pleftQ, std::string fname) 
{
  int norbs = ham.norbs;
  int nelec = ham.nelec;
  MatrixXcd hf = MatrixXcd::Zero(2*norbs, 2*norbs);
  readMat(hf, fname);
  det = hf.block(0, 0, 2*norbs, nelec);
  detAd = det.adjoint();
  leftQ = pleftQ;
  if (leftQ) ham.rotateCholesky(detAd, rotChol);
};


void GHF::getSample(Eigen::MatrixXcd& sampleDet) 
{
  sampleDet = det;
};


std::complex<double> GHF::overlap(std::array<Eigen::MatrixXcd, 2>& psi)
{
  return std::complex<double>();
};


std::complex<double> GHF::overlap(Eigen::MatrixXcd& psi)
{
  complex<double> overlap = (detAd * psi).determinant();
  return overlap;
};


void GHF::forceBias(Eigen::MatrixXcd& psi, Hamiltonian& ham, Eigen::VectorXcd& fb)
{
  int norbs = ham.norbs, nelec = ham.nelec;
  MatrixXcd thetaT;
  thetaT = (psi * (detAd * psi).inverse()).transpose();
  fb = VectorXcd::Zero(rotChol.size());
  for (int i = 0; i < rotChol.size(); i++) {
    fb(i) = thetaT.block(0, 0, nelec, norbs).cwiseProduct(rotChol[i][0]).sum() + thetaT.block(0, norbs, nelec, norbs).cwiseProduct(rotChol[i][1]).sum();
  }
};


std::array<std::complex<double>, 2> GHF::hamAndOverlap(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham) 
{
  return std::array<std::complex<double>, 2>();
};


std::array<std::complex<double>, 2> GHF::hamAndOverlap(Eigen::MatrixXcd& psi, Hamiltonian& ham) 
{
  MatrixXcd overlapMat = detAd * psi;
  complex<double> overlap = overlapMat.determinant();
  complex<double> ene = ham.ecore;

  // calculate theta and green
  MatrixXcd theta = psi * overlapMat.inverse();
  MatrixXcd green = (theta * detAd).transpose();

  // one body part
  ene += green.cwiseProduct(ham.h1soc).sum();

  // two body part
  int norbs = ham.norbs, nelec = ham.nelec;
  MatrixXcd fup = MatrixXcd::Zero(nelec, nelec);
  MatrixXcd fdn = MatrixXcd::Zero(nelec, nelec);
  for (int i = 0; i < ham.ncholEne; i++) {
    fup.noalias() = rotChol[i][0] * theta.block(0, 0, norbs, nelec);
    fdn.noalias() = rotChol[i][1] * theta.block(norbs, 0, norbs, nelec);
    complex<double> cup = fup.trace();
    complex<double> cdn = fdn.trace();
    ene += ( cup * cup + cdn * cdn + 2. * cup * cdn 
           - fup.cwiseProduct(fup.transpose()).sum() - fdn.cwiseProduct(fdn.transpose()).sum() 
           - fup.cwiseProduct(fdn.transpose()).sum() - fdn.cwiseProduct(fup.transpose()).sum()) / 2.;
  }

  std::array<complex<double>, 2> hamOverlap;
  hamOverlap[0] = ene * overlap;
  hamOverlap[1] = overlap;
  return hamOverlap;
};
