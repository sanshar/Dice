#ifndef GHFMultiSlater_HEADER_H
#define GHFMultiSlater_HEADER_H
#include <utility>
#include <random>
#include "Hamiltonian.h"
#include "Wavefunction.h"

// multislater wave function
class GHFMultislater : public Wavefunction {
  public:
    Eigen::VectorXi refDet; 
    std::vector<std::array<Eigen::VectorXi, 2>> ciExcitations;
    std::vector<double> ciParity; 
    std::vector<double> ciCoeffs;
    int nact, ncore;
    bool rightQ;
    std::vector<double> cumulativeCoeffs;
    std::vector<Eigen::Map<Eigen::MatrixXd>> blockChol;
    std::uniform_real_distribution<double> uniform;

    GHFMultislater(Hamiltonian& ham, std::string fname, int pnact, int pncore, bool prightQ = false);
    virtual void getSample(Eigen::MatrixXcd& sampleDet);
    std::complex<double> overlap(std::array<Eigen::MatrixXcd, 2>& psi) {};
    virtual std::complex<double> overlap(Eigen::MatrixXcd& psi);
    virtual void forceBias(Eigen::MatrixXcd& psi, Hamiltonian& ham, Eigen::VectorXcd& fb);
    virtual void oneRDM(Eigen::MatrixXcd& det, Eigen::MatrixXcd& rdmSample);
    std::array<std::complex<double>, 2> hamAndOverlap(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham) {};
    virtual std::array<std::complex<double>, 2> hamAndOverlap(Eigen::MatrixXcd& psi, Hamiltonian& ham);
};

#endif
