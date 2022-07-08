#include <Eigen/Dense>
#include <vector>
#include <complex>



void applyProjectorSz(Eigen::MatrixXcd& bra, std::vector<Eigen::MatrixXcd>& ketvec, 
                    std::vector<std::complex<double>>& coeffs, double Sz, int ngrid);

void optimizeProjectedSlater(double enuc, Eigen::MatrixXd& h1, std::vector<Eigen::Map<Eigen::MatrixXd>>& chol); 

std::complex<double> getGradientProjected(Eigen::MatrixXcd& ref, double enuc, Eigen::MatrixXd& h1, std::vector<Eigen::Map<Eigen::MatrixXcd>>& chol, Eigen::MatrixXcd& Grad); 
std::complex<double> getEnergyProjected(Eigen::MatrixXcd& ref, double enuc, Eigen::MatrixXd& h1, std::vector<Eigen::Map<Eigen::MatrixXd>>& chol);
