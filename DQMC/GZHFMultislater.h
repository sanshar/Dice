#ifndef GZHFMultiSlater_HEADER_H
#define GZHFMultiSlater_HEADER_H
#include "DQMC/Hamiltonian.h"
#include "DQMC/Wavefunction.h"
#include <random>
#include <utility>

// multislater wave function
class GZHFMultislater : public Wavefunction {
public:
  Eigen::VectorXi refDet;
  std::vector<Eigen::VectorXi> dets;
  std::vector<std::array<Eigen::VectorXi, 2>> ciExcitations;
  std::vector<double> ciParity;
  std::vector<std::complex<double>> ciCoeffs;
  int nact, ncore;
  bool rightQ;
  bool slowQ;
  bool slowOverlapQ;
  bool slowForceBiasQ;
  bool slowLocalEnergyQ;
  std::vector<double> cumulativeCoeffs;
  std::vector<Eigen::Map<Eigen::MatrixXcd>> blockChol;
  std::uniform_real_distribution<double> uniform;

  GZHFMultislater(Hamiltonian &ham, std::string fname, int pnact, int pncore,
                  bool prightQ = false, bool pslowQ = false,
                  bool pslowOverlapQ = false,
                  bool pslowForceBiasQ = false,
                  bool pslowLocalEnergyQ = false);
  virtual void getSample(Eigen::MatrixXcd &sampleDet);
  std::complex<double> overlap(std::array<Eigen::MatrixXcd, 2> &psi){};
  virtual std::complex<double> overlap(Eigen::MatrixXcd &psi);
  virtual void forceBias(Eigen::MatrixXcd &psi, Hamiltonian &ham,
                         Eigen::VectorXcd &fb);
  virtual void oneRDM(Eigen::MatrixXcd &det, Eigen::MatrixXcd &rdmSample);
  std::array<std::complex<double>, 2>
  hamAndOverlap(std::array<Eigen::MatrixXcd, 2> &psi, Hamiltonian &ham){};
  virtual std::array<std::complex<double>, 2>
  hamAndOverlap(Eigen::MatrixXcd &psi, Hamiltonian &ham);
};

#endif
