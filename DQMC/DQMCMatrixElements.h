#ifndef DQMCMatrixElements_HEADER_H
#define DQMCMatrixElements_HEADER_H
#include <Eigen/Dense>
#include <utility>
#include "workingArray.h"

// left is adjointed
void calcGreensFunction(std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& leftT, 
    std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& right, 
    std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& green);


// left is adjointed
void calcGreensFunction(Eigen::MatrixXcd& leftT, Eigen::MatrixXcd& right, Eigen::MatrixXcd& green);


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

std::complex<double> calcHamiltonianElement_sRI(std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& phi1T, std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& phi2, 
                                                std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& phi0T, std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& phi0, double enuc, Eigen::MatrixXd& h1, std::vector<Eigen::MatrixXd>& chol, std::vector<Eigen::MatrixXd>& richol); 

// Hamiltonian matrix element < phi_0 | H | psi > / < phi_0 | psi >
// using precalculated half rotated cholesky vectors
// leading cost: O(X N^2 M)
std::complex<double> calcHamiltonianElement(std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& phi0T, std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& psi, double enuc, Eigen::MatrixXd& h1, std::pair<std::vector<Eigen::MatrixXcd>, std::vector<Eigen::MatrixXcd>>& rotChol, int nchol = -1); 

// Hamiltonian matrix element < phi_0 | H | psi > / < phi_0 | psi >
// using precalculated half rotated cholesky vectors
// leading cost: O(X N^2 M)
// assumes rhf
std::complex<double> calcHamiltonianElement(Eigen::MatrixXcd& phi0T, Eigen::MatrixXcd& psi, double enuc, Eigen::MatrixXd& h1, std::vector<Eigen::MatrixXcd>& rotChol, int nchol = -1); 


// Hamiltonian matrix element < \sum _i phi_i | H | psi > / < \sum_i phi_i | psi >
// multislater local energy
// leading cost: O(X N_c + X N^2 M)
std::pair<std::complex<double>, std::complex<double>> calcHamiltonianElement0(std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& phi0T, std::array<std::vector<std::array<Eigen::VectorXi, 2>>, 2>& ciExcitations, std::vector<double>& ciParity, std::vector<double>& ciCoeffs, std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& psi, double enuc, Eigen::MatrixXd& h1, std::vector<Eigen::MatrixXd>& chol); 


// Hamiltonian matrix element < \sum _i phi_i | H | psi > / < \sum_i phi_i | psi >
// multislater local energy
// leading cost: O(X N_c + X N M^2)
std::pair<std::complex<double>, std::complex<double>> calcHamiltonianElement(std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& phi0T, std::array<std::vector<std::array<Eigen::VectorXi, 2>>, 2>& ciExcitations, std::vector<double>& ciParity, std::vector<double>& ciCoeffs, std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& psi, double enuc, Eigen::MatrixXd& h1, std::vector<Eigen::MatrixXd>& chol, int nchol = -1, int nact = -1, int ncore = -1); 


// Hamiltonian matrix element < \sum _i phi_i | H | psi > / < \sum_i phi_i | psi >
// multislater local energy
// leading cost: O(N_c + X N^2 M^2)
std::pair<std::complex<double>, std::complex<double>> calcHamiltonianElement1(std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& phi0T, std::array<std::vector<std::array<Eigen::VectorXi, 2>>, 2>& ciExcitations, std::vector<double>& ciParity, std::vector<double>& ciCoeffs, std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& psi, double enuc, Eigen::MatrixXd& h1, std::vector<Eigen::MatrixXd>& chol); 


// Hamiltonian matrix element < \sum _i phi_i | H | psi > / < \sum_i phi_i | psi >
// multislater local energy
// leading cost: O(X N_c + X N^2 M)
std::pair<std::complex<double>, std::complex<double>> calcHamiltonianElement_sRI(std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& phi0T, std::array<std::vector<std::array<Eigen::VectorXi, 2>>, 2>& ciExcitations, std::vector<double>& ciParity, std::vector<double>& ciCoeffs, std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& psi, double enuc, Eigen::MatrixXd& h1, std::vector<Eigen::MatrixXd>& chol, double ene0); 

// Hamiltonian matrix element < phi_1 | H | phi_2 > / < phi_1 | phi_2 >
// heat bath
std::complex<double> calcHamiltonianElement(std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& walker, workingArray& work, double dene); 


std::complex<double> calcHamiltonianElement(Eigen::MatrixXcd& ghf, std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& A, double enuc, Eigen::MatrixXd& h1, std::vector<Eigen::MatrixXd>& chol, std::complex<double>& ovlp);
std::complex<double> calcHamiltonianElement(Eigen::MatrixXcd& ghf, std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& A, double enuc, Eigen::MatrixXd& h1, std::pair<std::vector<Eigen::MatrixXcd>, std::vector<Eigen::MatrixXcd>>& chol, std::complex<double>& ovlp) ;
std::complex<double> calcHamiltonianElement(Eigen::MatrixXcd& A, Eigen::MatrixXcd& B, double enuc, Eigen::MatrixXd& h1, std::vector<Eigen::MatrixXd>& chol);
std::complex<double> calcHamiltonianElement(Eigen::MatrixXcd& A, Eigen::MatrixXcd& B, double enuc, Eigen::MatrixXd& h1, std::pair<std::vector<Eigen::MatrixXcd>, std::vector<Eigen::MatrixXcd>>& rotChol);
std::complex<double> calcGradient(Eigen::MatrixXcd& At, Eigen::MatrixXcd& B, double enuc, Eigen::MatrixXd& h1, std::vector<Eigen::Map<Eigen::MatrixXd>>& chol, Eigen::MatrixXcd& Grad); 
std::complex<double> calcGradient(Eigen::MatrixXcd& At, Eigen::MatrixXcd& B, double enuc, Eigen::MatrixXd& h1, std::pair<std::vector<Eigen::MatrixXcd>, std::vector<Eigen::MatrixXcd>>& rotChol, Eigen::MatrixXcd& Grad); 
std::complex<double> calcHamiltonianElementNaive(Eigen::MatrixXcd& At, Eigen::MatrixXcd& B, double enuc, Eigen::MatrixXd& h1, std::vector<Eigen::MatrixXd>& chol);

#endif
