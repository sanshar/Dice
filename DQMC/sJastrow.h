#ifndef sJastrow_HEADER_H
#define sJastrow_HEADER_H
#include <utility>
#include <random>
#include "Hamiltonian.h"
#include "Wavefunction.h"

// jastrow slater wave function
class sJastrow : public Wavefunction {
  public:
    std::vector<Eigen::VectorXcd> hsOperators;
    Eigen::VectorXcd oneBodyOperator;
    Eigen::MatrixXd basisRotation;
    std::array<Eigen::MatrixXcd, 2> refState;
    int norbs;
    std::normal_distribution<double> normal;

    sJastrow(int pnorbs, int pnalpha, int pnbeta, std::string fname = "ref.txt");
    virtual void getSample(std::array<Eigen::MatrixXcd, 2>& sampleDet);
    virtual std::complex<double> overlap(std::array<Eigen::MatrixXcd, 2>& psi);
    virtual std::complex<double> overlap(Eigen::MatrixXcd& psi);
    virtual void forceBias(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham, Eigen::VectorXcd& fb);
    virtual void forceBias(Eigen::MatrixXcd& psi, Hamiltonian& ham, Eigen::VectorXcd& fb);
    virtual std::array<std::complex<double>, 2> hamAndOverlap(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham);
    virtual std::array<std::complex<double>, 2> hamAndOverlap(Eigen::MatrixXcd& psi, Hamiltonian& ham);
};
#endif
