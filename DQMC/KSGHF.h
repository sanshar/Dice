#ifndef KSGHF_HEADER_H
#define KSGHF_HEADER_H
#include <utility>
#include "Hamiltonian.h"
#include "Wavefunction.h"

// rhf wave function
class KSGHF : public Wavefunction {
  public:
    Eigen::MatrixXcd det, detT, detAd;
    std::array<std::vector<Eigen::MatrixXcd>, 2> rotCholAd, rotCholT;
    bool leftQ;

    KSGHF(Hamiltonian& ham, bool pleftQ, std::string fname = "ghf.txt");
    virtual void getSample(std::array<Eigen::MatrixXcd, 2>& sampleDet);
    virtual std::array<std::complex<double>, 2> hamAndOverlap(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham);
    virtual std::array<std::complex<double>, 2> hamAndOverlap(Eigen::MatrixXcd& psi, Hamiltonian& ham);
};
#endif
