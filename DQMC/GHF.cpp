#include "input.h"
#include "GHF.h"

using namespace std;
using namespace Eigen;

using matPair = std::array<MatrixXcd, 2>;


GHF::GHF(Hamiltonian& ham, bool pleftQ, std::string fname) 
{
  int norbs = ham.norbs;
  int nelec = ham.nelec;
  complexQ = ham.socQ;
  if (ham.socQ) {
    MatrixXcd hf = MatrixXcd::Zero(2*norbs, 2*norbs);
    readMat(hf, fname);
    detC = hf.block(0, 0, 2*norbs, nelec);
    detCAd = detC.adjoint();
  }
  else {
    MatrixXd hf = MatrixXd::Zero(norbs, norbs);
    readMat(hf, fname);
    det = hf.block(0, 0, norbs, nelec);
    detT = det.transpose();
  }
  leftQ = pleftQ;
  if (leftQ) {
    if (ham.socQ) ham.rotateCholesky(detCAd, rotCholC);
    else if (ham.intType == "g") ham.rotateCholesky(detT, rotChol, rotCholMat);
  }
};


void GHF::getSample(Eigen::MatrixXcd& sampleDet) 
{
  if (complexQ) sampleDet = detC;
  else sampleDet = det;
};


std::complex<double> GHF::overlap(std::array<Eigen::MatrixXcd, 2>& psi)
{
  return std::complex<double>();
};


std::complex<double> GHF::overlap(Eigen::MatrixXcd& psi)
{
  complex<double> overlap;
  if (complexQ) overlap = (detCAd * psi).determinant();
  else overlap = (detT * psi).determinant();
  return overlap;
};


void GHF::forceBias(Eigen::MatrixXcd& psi, Hamiltonian& ham, Eigen::VectorXcd& fb)
{
  int norbs = ham.norbs, nelec = ham.nelec;
  MatrixXcd thetaT;
  if (ham.socQ) {
    thetaT = (psi * (detCAd * psi).inverse()).transpose();
    fb = VectorXcd::Zero(rotCholC.size());
    for (int i = 0; i < rotCholC.size(); i++) {
      fb(i) = thetaT.block(0, 0, nelec, norbs).cwiseProduct(rotCholC[i][0]).sum() + thetaT.block(0, norbs, nelec, norbs).cwiseProduct(rotCholC[i][1]).sum();
    }
  }
  else {
    thetaT = (psi * (detT * psi).inverse()).transpose();
    Eigen::Map<VectorXcd> thetaTFlat(thetaT.data(), thetaT.rows() * thetaT.cols());
    fb = thetaTFlat.transpose() * rotCholMat[0];
  }
};
 

void GHF::oneRDM(Eigen::MatrixXcd& psi, Eigen::MatrixXcd& rdmSample)
{
  if (complexQ) rdmSample = (psi * (detCAd * psi).inverse() * detCAd).transpose();
  else rdmSample = (psi * (detT * psi).inverse() * detT).transpose();
}


std::array<std::complex<double>, 2> GHF::hamAndOverlap(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham) 
{
  return std::array<std::complex<double>, 2>();
};


std::array<std::complex<double>, 2> GHF::hamAndOverlap(Eigen::MatrixXcd& psi, Hamiltonian& ham) 
{
  complex<double> overlap, ene;
  if (ham.socQ) {
    MatrixXcd overlapMat = detCAd * psi;
    overlap = overlapMat.determinant();
    ene = ham.ecore;

    // calculate theta and green
    MatrixXcd theta = psi * overlapMat.inverse();
    MatrixXcd green = (theta * detCAd).transpose();

    // one body part
    ene += green.cwiseProduct(ham.h1soc).sum();

    // two body part
    int norbs = ham.norbs, nelec = ham.nelec;
    MatrixXcd fup = MatrixXcd::Zero(nelec, nelec);
    MatrixXcd fdn = MatrixXcd::Zero(nelec, nelec);
    for (int i = 0; i < ham.ncholEne; i++) {
      fup.noalias() = rotCholC[i][0] * theta.block(0, 0, norbs, nelec);
      fdn.noalias() = rotCholC[i][1] * theta.block(norbs, 0, norbs, nelec);
      complex<double> cup = fup.trace();
      complex<double> cdn = fdn.trace();
      ene += ( cup * cup + cdn * cdn + 2. * cup * cdn 
             - fup.cwiseProduct(fup.transpose()).sum() - fdn.cwiseProduct(fdn.transpose()).sum() 
             - fup.cwiseProduct(fdn.transpose()).sum() - fdn.cwiseProduct(fup.transpose()).sum()) / 2.;
    }
  }
  else {
    MatrixXcd overlapMat = detT * psi;
    overlap = overlapMat.determinant();
    ene = ham.ecore;

    // calculate theta and green
    MatrixXcd theta = psi * overlapMat.inverse();
    MatrixXcd green = (theta * detT).transpose();

    // one body part
    ene += green.cwiseProduct(ham.h1).sum();

    // two body part
    int norbs = ham.norbs, nelec = ham.nelec;
    MatrixXcd f = MatrixXcd::Zero(nelec, nelec);
    for (int i = 0; i < ham.ncholEne; i++) {
      f.noalias() = rotChol[i] * theta;
      complex<double> c = f.trace();
      ene += (c * c - f.cwiseProduct(f.transpose()).sum()) / 2.;
    }
  }

  std::array<complex<double>, 2> hamOverlap;
  hamOverlap[0] = ene * overlap;
  hamOverlap[1] = overlap;
  return hamOverlap;
};
