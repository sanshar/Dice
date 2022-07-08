#ifndef UHF_HEADER_H
#define UHF_HEADER_H
#include <utility>
#include "Hamiltonian.h"
#include "Wavefunction.h"

// rhf wave function
class UHF : public Wavefunction {
  public:
    std::array<Eigen::MatrixXd, 2> det;
    std::array<Eigen::MatrixXd, 2> detT;
    std::array<std::vector<Eigen::Map<Eigen::MatrixXd>>, 2> rotChol;
    std::array<std::vector<Eigen::Map<Eigen::MatrixXd>>, 2> rotCholMat;
    bool leftQ;

    UHF(Hamiltonian& ham, bool pleftQ, std::string fname = "uhf.txt");
    virtual void getSample(std::array<Eigen::MatrixXcd, 2>& sampleDet);
    virtual std::complex<double> overlap(std::array<Eigen::MatrixXcd, 2>& psi);
    virtual std::complex<double> overlap(Eigen::MatrixXcd& psi);
    virtual void forceBias(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham, Eigen::VectorXcd& fb);
    virtual void forceBias(Eigen::MatrixXcd& psi, Hamiltonian& ham, Eigen::VectorXcd& fb);
    virtual void oneRDM(std::array<Eigen::MatrixXcd, 2>& det, std::array<Eigen::MatrixXcd, 2>& rdmSample) ;
    virtual std::array<std::complex<double>, 2> hamAndOverlap(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham);
    virtual std::array<std::complex<double>, 2> hamAndOverlap(Eigen::MatrixXcd& psi, Hamiltonian& ham);
};
#endif
