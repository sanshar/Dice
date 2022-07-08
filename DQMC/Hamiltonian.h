#ifndef Hamiltonian_HEADER_H
#define Hamiltonian_HEADER_H
#include <string>
#include <vector>
#include <Eigen/Dense>

// cholesky vectors
class Hamiltonian {
  public:
    Eigen::MatrixXd h1, h1Mod;
    std::array<Eigen::MatrixXd, 2> h1u, h1uMod;
    Eigen::MatrixXcd h1soc, h1socMod;
    std::vector<Eigen::Map<Eigen::MatrixXd>> chol;
    std::vector<Eigen::Map<Eigen::MatrixXd>> cholMat;
    std::vector<std::array<Eigen::Map<Eigen::MatrixXd>, 2>> cholu;
    float* floatChol;
    std::vector<Eigen::Map<Eigen::MatrixXf>> floatCholMat;
    std::string intType;
    bool socQ;
    double ecore;
    bool rotFlag;
    int norbs, nalpha, nbeta, nelec, ncholEne, nchol;

    // constructor
    Hamiltonian(std::string fname, bool psocQ = false, std::string pintType = "r");

    void setNcholEne(int pnchol);

    // rotate cholesky
    void rotateCholesky(Eigen::MatrixXd& phiT, std::vector<Eigen::Map<Eigen::MatrixXd>>& rotChol, std::vector<Eigen::Map<Eigen::MatrixXd>>& rotCholMat, bool deleteOriginalChol=false);
    void rotateCholesky(std::array<Eigen::MatrixXd, 2>& phiT, std::array<std::vector<Eigen::Map<Eigen::MatrixXd>>, 2>& rotChol, std::array<std::vector<Eigen::Map<Eigen::MatrixXd>>, 2>& rotCholMat, bool deleteOriginalChol=false);
    void rotateCholesky(Eigen::MatrixXcd& phiAd, std::vector<std::array<Eigen::MatrixXcd, 2>>& rotChol);
    
    // block cholesky
    // for multislater where rotation <-> block
    void blockCholesky(std::vector<Eigen::Map<Eigen::MatrixXd>>& blockChol, int ncol);

    // flatten and convert to float
    void floattenCholesky();
};
#endif
