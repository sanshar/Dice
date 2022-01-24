#ifndef MultiSlater_HEADER_H
#define MultiSlater_HEADER_H
#include <utility>
#include <random>
#include "Hamiltonian.h"
#include "Wavefunction.h"

// multislater wave function
class Multislater : public Wavefunction {
  public:
    std::array<std::vector<int>, 2> refDet; 
    std::array<std::vector<std::array<Eigen::VectorXi, 2>>, 2> ciExcitations;
    std::vector<double> ciParity; 
    std::vector<double> ciCoeffs;
    int nact, ncore;
    bool rightQ;
    std::vector<double> cumulativeCoeffs;
    std::uniform_real_distribution<double> uniform;

    Multislater(std::string fname, int pnact, int pncore, bool prightQ = false);
    virtual void getSample(std::array<Eigen::MatrixXcd, 2>& sampleDet);
    virtual std::complex<double> overlap(std::array<Eigen::MatrixXcd, 2>& psi);
    virtual std::complex<double> overlap(Eigen::MatrixXcd& psi);
    virtual void forceBias(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham, Eigen::VectorXcd& fb);
    virtual void forceBias(Eigen::MatrixXcd& psi, Hamiltonian& ham, Eigen::VectorXcd& fb);
    virtual void oneRDM(std::array<Eigen::MatrixXcd, 2>& psi, Eigen::MatrixXcd& rdmSample);
    virtual std::array<std::complex<double>, 2> hamAndOverlap(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham);
    virtual std::array<std::complex<double>, 2> hamAndOverlap(Eigen::MatrixXcd& psi, Hamiltonian& ham);
};
#endif
