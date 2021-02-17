#ifndef CCSD_HEADER_H
#define CCSD_HEADER_H
#include <utility>
#include <random>
#include "Hamiltonian.h"
#include "Wavefunction.h"

// ccsd wave function
class CCSD : public Wavefunction {
  public:
    std::vector<Eigen::MatrixXcd> hsOperators;
    Eigen::MatrixXcd oneBodyOperator;
    Eigen::MatrixXd basisRotation;
    int norbs, nocc;
    std::normal_distribution<double> normal;

    CCSD(int norbs, int nocc, std::string fname = "ccsd.h5");
    virtual void getSample(std::array<Eigen::MatrixXcd, 2>& sampleDet);
    virtual std::array<std::complex<double>, 2> hamAndOverlap(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham);
    virtual std::array<std::complex<double>, 2> hamAndOverlap(Eigen::MatrixXcd& psi, Hamiltonian& ham);
};
#endif
