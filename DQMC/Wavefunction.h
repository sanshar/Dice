#ifndef Wavefunction_HEADER_H
#define Wavefunction_HEADER_H
#include <utility>
#include "Hamiltonian.h"

// wave function interface
class Wavefunction {
  public:
    virtual void getSample(std::array<Eigen::MatrixXcd, 2>& det) = 0;
    virtual std::array<std::complex<double>, 2> hamAndOverlap(std::array<Eigen::MatrixXcd, 2>& det, Hamiltonian& ham) = 0;
    virtual std::array<std::complex<double>, 2> hamAndOverlap(Eigen::MatrixXcd& det, Hamiltonian& ham) = 0;
};
#endif
