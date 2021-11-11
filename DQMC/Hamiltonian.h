#ifndef Hamiltonian_HEADER_H
#define Hamiltonian_HEADER_H
#include <string>
#include <vector>
#include <Eigen/Dense>

// cholesky vectors
class Hamiltonian {
  public:
    Eigen::MatrixXd h1, h1Mod;
    std::vector<Eigen::MatrixXd> chol;
    std::vector<std::vector<float>> floatChol;
    double ecore;
    int norbs, nalpha, nbeta, nchol;

    // constructor
    Hamiltonian(std::string fname);

    void setNchol(int pnchol);

    // rotate cholesky
    void rotateCholesky(Eigen::MatrixXd& phi, std::vector<Eigen::MatrixXd>& rotChol);

    // flatten and convert to float
    //void floattenCholesky(std::vector<Eigen::MatrixXf>& floatChol);
    //void floattenCholesky(std::vector<std::vector<float>>& floatChol);
    void floattenCholesky();
};
#endif
