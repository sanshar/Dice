#ifndef GZHF_HEADER_H
#define GZHF_HEADER_H
#include "Hamiltonian.h"
#include "Wavefunction.h"
#include <utility>

// ghf wave function with all integrals to be complex
class GZHF : public Wavefunction {
public:
  Eigen::MatrixXcd detC, detCAd;
  std::vector<Eigen::Map<Eigen::MatrixXcd>> rotChol;
  std::vector<Eigen::Map<Eigen::MatrixXcd>> rotCholMat;
  bool leftQ;

  GZHF(Hamiltonian &ham, bool pleftQ, std::string fname = "ghf.txt");
  virtual void getSample(Eigen::MatrixXcd &sampleDet);
  virtual std::complex<double> overlap(std::array<Eigen::MatrixXcd, 2>& psi);
  virtual std::complex<double> overlap(Eigen::MatrixXcd &psi);
  // virtual void forceBias(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian&
  // ham, Eigen::VectorXcd& fb);
  virtual void forceBias(Eigen::MatrixXcd &psi, Hamiltonian &ham,
                         Eigen::VectorXcd &fb);
  virtual void oneRDM(Eigen::MatrixXcd &det, Eigen::MatrixXcd &rdmSample);
  virtual std::array<std::complex<double>, 2> hamAndOverlap(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham);
  virtual std::array<std::complex<double>, 2>
  hamAndOverlap(Eigen::MatrixXcd &psi, Hamiltonian &ham);
};
#endif
