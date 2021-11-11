#ifndef UCCSD_HEADER_H
#define UCCSD_HEADER_H
#include <utility>
#include <random>
#include "Hamiltonian.h"
#include "Wavefunction.h"

// uccsd wave function, using rohf ref
class UCCSD : public Wavefunction {
  public:
    std::array<std::vector<Eigen::MatrixXcd>, 2> hsOperators;
    std::array<Eigen::MatrixXcd, 2> oneBodyOperator;
    Eigen::MatrixXd basisRotation;
    int norbs, nalpha, nbeta;
    std::normal_distribution<double> normal;

    UCCSD(int norbs, int nalpha, int nbeta, std::string fname = "uccsd.h5");
    virtual void getSample(std::array<Eigen::MatrixXcd, 2>& sampleDet);
    virtual std::complex<double> overlap(std::array<Eigen::MatrixXcd, 2>& psi);
    virtual std::complex<double> overlap(Eigen::MatrixXcd& psi);
    virtual void forceBias(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham, Eigen::VectorXcd& fb);
    virtual void forceBias(Eigen::MatrixXcd& psi, Hamiltonian& ham, Eigen::VectorXcd& fb);
    virtual std::array<std::complex<double>, 2> hamAndOverlap(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham);
    virtual std::array<std::complex<double>, 2> hamAndOverlap(Eigen::MatrixXcd& psi, Hamiltonian& ham);
};
#endif
