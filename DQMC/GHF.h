#ifndef GHF_HEADER_H
#define GHF_HEADER_H
#include <utility>
#include "Hamiltonian.h"
#include "Wavefunction.h"

// ghf wave function
// to be used with soc only
class GHF : public Wavefunction {
  public:
    Eigen::MatrixXd det, detT;
    Eigen::MatrixXcd detC, detCAd;
    std::vector<Eigen::Map<Eigen::MatrixXd>> rotChol;
    std::vector<Eigen::Map<Eigen::MatrixXd>> rotCholMat;
    std::vector<std::array<Eigen::MatrixXcd, 2>> rotCholC;
    bool leftQ, complexQ;

    GHF(Hamiltonian& ham, bool pleftQ, std::string fname = "ghf.txt");
    virtual void getSample(Eigen::MatrixXcd& sampleDet);
    virtual std::complex<double> overlap(std::array<Eigen::MatrixXcd, 2>& psi);
    virtual std::complex<double> overlap(Eigen::MatrixXcd& psi);
    //virtual void forceBias(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham, Eigen::VectorXcd& fb);
    virtual void forceBias(Eigen::MatrixXcd& psi, Hamiltonian& ham, Eigen::VectorXcd& fb);
    virtual void oneRDM(Eigen::MatrixXcd& det, Eigen::MatrixXcd& rdmSample);
    virtual std::array<std::complex<double>, 2> hamAndOverlap(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham);
    virtual std::array<std::complex<double>, 2> hamAndOverlap(Eigen::MatrixXcd& psi, Hamiltonian& ham);
};
#endif
