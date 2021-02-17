#ifndef DQMCWalker_HEADER_H
#define DQMCWalker_HEADER_H
#include <random>
#include "Hamiltonian.h"
#include "Wavefunction.h"

class DQMCWalker {
  public:
    std::array<Eigen::MatrixXcd, 2> det;
    std::complex<double> orthoFac;
    std::vector<std::complex<double>> mfShifts;
    std::array<std::complex<double>, 2> propConstant;
    Eigen::MatrixXcd expOneBodyOperator;
    //std::vector<Eigen::MatrixXf> floatChol;
    std::vector<std::vector<float>> floatChol;
    bool rhfQ;
    double dt, ene0;
    std::normal_distribution<double> normal;

    // constructor
    DQMCWalker(bool prhfQ = true);

    void prepProp(std::array<Eigen::MatrixXcd, 2>& ref, Hamiltonian& ham, double pdt, double pene0);

    void setDet(std::array<Eigen::MatrixXcd, 2> pdet);

    void orthogonalize();

    void propagate();

    std::array<std::complex<double>, 2> hamAndOverlap(Wavefunction& wave, Hamiltonian& ham);
};
#endif
