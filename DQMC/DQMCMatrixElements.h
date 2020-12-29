#ifndef DQMCMatrixElements_HEADER_H
#define DQMCMatrixElements_HEADER_H
#include <Eigen/Dense>
#include <utility>
#include "workingArray.h"

// left is adjointed
void calcGreensFunction(std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& leftT, 
    std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& right, 
    std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& green);


// Hamiltonian matrix element < phi_1 | H | phi_2 > / < phi_1 | phi_2 >
// green: rdm between phi_1 and phi_2
// leading cost: O(X M^3)
std::complex<double> calcHamiltonianElement(std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& green, double enuc, Eigen::MatrixXd& h1, std::vector<Eigen::MatrixXd>& chol);


// Hamiltonian matrix element < phi_1 | H | phi_2 > / < phi_1 | phi_2 >
// rotates cholesky vectors
// phi_1 can be an active space wave function
// TODO: allow core orbitals
// leading cost: O(X N A M)
std::complex<double> calcHamiltonianElement(std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& phi1T, std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& phi2, double enuc, Eigen::MatrixXd& h1, std::vector<Eigen::MatrixXd>& chol);


// Hamiltonian matrix element < phi_0 | H | psi > / < phi_0 | psi >
// using precalculated half rotated cholesky vectors
// leading cost: O(X N^2 M)
std::complex<double> calcHamiltonianElement(std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& phi0T, std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& psi, double enuc, Eigen::MatrixXd& h1, std::pair<std::vector<Eigen::MatrixXcd>, std::vector<Eigen::MatrixXcd>>& rotChol); 


// Hamiltonian matrix element < phi_1 | H | phi_2 > / < phi_1 | phi_2 >
// heat bath
std::complex<double> calcHamiltonianElement(std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& walker, workingArray& work, double dene); 


#endif
