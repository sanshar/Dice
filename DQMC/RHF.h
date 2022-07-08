#ifndef RHF_HEADER_H
#define RHF_HEADER_H
#include <utility>
#include "Hamiltonian.h"
#include "Wavefunction.h"

// rhf wave function
class RHF : public Wavefunction {
  public:
    Eigen::MatrixXd det, detT;
    std::vector<Eigen::Map<Eigen::MatrixXd>> rotChol;
    std::vector<Eigen::Map<Eigen::MatrixXd>> rotCholMat;
    bool leftQ;

    RHF(Hamiltonian& ham, bool pleftQ, std::string fname = "rhf.txt");
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
