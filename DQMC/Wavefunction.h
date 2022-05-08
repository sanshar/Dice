#ifndef Wavefunction_HEADER_H
#define Wavefunction_HEADER_H
#include <utility>
#include "Hamiltonian.h"

// wave function interface
class Wavefunction {
  public:
    virtual void getSample(std::array<Eigen::MatrixXcd, 2>& det) { };
    virtual void getSample(Eigen::MatrixXcd& det) { };
    virtual std::complex<double> overlap(std::array<Eigen::MatrixXcd, 2>& det) = 0;
    virtual std::complex<double> overlap(Eigen::MatrixXcd& det) = 0;
    virtual void forceBias(std::array<Eigen::MatrixXcd, 2>& det, Hamiltonian& ham, Eigen::VectorXcd& fb) { };
    virtual void forceBias(Eigen::MatrixXcd& det, Hamiltonian& ham, Eigen::VectorXcd& fb) { };
    virtual void oneRDM(std::array<Eigen::MatrixXcd, 2>& det, Eigen::MatrixXcd& rdmSample) { };
    virtual void oneRDM(std::array<Eigen::MatrixXcd, 2>& det, std::array<Eigen::MatrixXcd, 2>& rdmSample) { };
    virtual void oneRDM(Eigen::MatrixXcd& det, Eigen::MatrixXcd& rdmSample) { };
    virtual std::array<std::complex<double>, 2> hamAndOverlap(std::array<Eigen::MatrixXcd, 2>& det, Hamiltonian& ham) = 0;
    virtual std::array<std::complex<double>, 2> hamAndOverlap(Eigen::MatrixXcd& det, Hamiltonian& ham) = 0;
};
#endif
