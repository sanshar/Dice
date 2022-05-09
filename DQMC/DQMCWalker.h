#ifndef DQMCWalker_HEADER_H
#define DQMCWalker_HEADER_H
#include <random>
#include "Hamiltonian.h"
#include "Wavefunction.h"

class DQMCWalker {
  public:
    std::array<Eigen::MatrixXcd, 2> det;
    Eigen::MatrixXcd detG;
    std::complex<double> orthoFac, trialOverlap;
    std::vector<std::complex<double>> mfShifts;
    std::array<std::complex<double>, 2> propConstant;
    Eigen::MatrixXcd expOneBodyOperator;
    std::array<Eigen::MatrixXcd, 2> expOneBodyOperatorU;
    bool rhfQ, szQ, phaselessQ;  // come up with a better name for szQ (szQ == false for ghf dets)
    double dt, ene0;
    std::normal_distribution<double> normal;
    double vhsTime, expTime, fbTime;

    // constructor
    DQMCWalker(bool prhfQ = true, bool pphaselessQ = false, bool pszQ = false);

    void prepProp(std::array<Eigen::MatrixXcd, 2>& ref, Hamiltonian& ham, double pdt, double pene0);
    void prepPropR(std::array<Eigen::MatrixXcd, 2>& ref, Hamiltonian& ham, double pdt, double pene0);
    void prepPropU(std::array<Eigen::MatrixXcd, 2>& ref, Hamiltonian& ham, double pdt, double pene0);
    void prepProp(Eigen::MatrixXcd& ref, Hamiltonian& ham, double pdt, double pene0);

    void setDet(std::array<Eigen::MatrixXcd, 2> pdet);
    void setDet(Eigen::MatrixXcd pdet);
    void setDet(std::vector<std::complex<double>>& serial, std::complex<double> ptrialOverlap);
    std::complex<double> getDet(std::vector<std::complex<double>>& serial);

    void orthogonalize();

    void propagate(Hamiltonian& ham);
    double propagatePhaseless(Wavefunction& wave, Hamiltonian& ham, double eshift);
    double propagatePhaselessRG(Wavefunction& wave, Hamiltonian& ham, double eshift);
    double propagatePhaselessU(Wavefunction& wave, Hamiltonian& ham, double eshift);

    std::complex<double> overlap(Wavefunction& wave);
    void forceBias(Wavefunction& wave, Hamiltonian& ham, Eigen::VectorXcd& fb);
    void oneRDM(Wavefunction& wave, Eigen::MatrixXcd& rdmSample);
    void oneRDM(Wavefunction& wave, std::array<Eigen::MatrixXcd, 2>& rdmSample);
    std::array<std::complex<double>, 2> hamAndOverlap(Wavefunction& wave, Hamiltonian& ham);
};
#endif
