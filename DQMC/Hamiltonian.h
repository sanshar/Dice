#ifndef Hamiltonian_HEADER_H
#define Hamiltonian_HEADER_H
#include <string>
#include <vector>
#include <Eigen/Dense>

// cholesky vectors
class Hamiltonian {
  public:
    Eigen::MatrixXd h1, h1Mod;
    Eigen::MatrixXcd h1soc, h1socMod;
    std::vector<Eigen::MatrixXd> chol;
    std::vector<std::vector<float>> floatChol;
    double ecore;
    int norbs, nalpha, nbeta, nelec, nchol;

    // constructor
    Hamiltonian(std::string fname, bool socQ = false);

    void setNchol(int pnchol);

    // rotate cholesky
    void rotateCholesky(Eigen::MatrixXd& phiT, std::vector<Eigen::MatrixXd>& rotChol, bool deleteOriginalChol=false);
    void rotateCholesky(Eigen::MatrixXcd& phiAd, std::vector<std::array<Eigen::MatrixXcd, 2>>& rotChol);

    // flatten and convert to float
    //void floattenCholesky(std::vector<Eigen::MatrixXf>& floatChol);
    //void floattenCholesky(std::vector<std::vector<float>>& floatChol);
    void floattenCholesky();
};
#endif
