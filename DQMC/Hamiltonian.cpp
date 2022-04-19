#include "Hamiltonian.h"
#include "integral.h"

using namespace std;
using namespace Eigen;

// constructor
Hamiltonian::Hamiltonian(string fname, bool socQ) 
{
  if (socQ) {
    readIntegralsCholeskyAndInitializeDeterminantStaticVariablesSOC(fname, norbs, nelec, ecore, h1soc, h1socMod, chol);
    nalpha = 0;
    nbeta = 0;
  }
  else {
    readIntegralsCholeskyAndInitializeDeterminantStaticVariables(fname, norbs, nalpha, nbeta, ecore, h1, h1Mod, chol);
    nelec = nalpha + nbeta;
  }
  nchol = chol.size();
  floattenCholesky();
};


void Hamiltonian::setNchol(int pnchol) 
{
  nchol = pnchol;
};


// rotate cholesky
void Hamiltonian::rotateCholesky(Eigen::MatrixXd& phiT, std::vector<Eigen::MatrixXd>& rotChol, bool deleteOriginalChol) 
{
  for (int i = 0; i < chol.size(); i++) {
    MatrixXd rot = phiT * chol[i];
    rotChol.push_back(rot);
    if (deleteOriginalChol) chol[i].resize(0, 0);
  }
};


// rotate cholesky soc
void Hamiltonian::rotateCholesky(Eigen::MatrixXcd& phiAd, std::vector<std::array<Eigen::MatrixXcd, 2>>& rotChol) 
{
  for (int i = 0; i < chol.size(); i++) {
    std::array<Eigen::MatrixXcd, 2> rot;
    rot[0] = phiAd.block(0, 0, nelec, norbs) * chol[i];
    rot[1] = phiAd.block(0, norbs, nelec, norbs) * chol[i];
    rotChol.push_back(rot);
  }
};

// flatten and convert to float
//void Hamiltonian::floattenCholesky(std::vector<Eigen::MatrixXf>& floatChol)
//void Hamiltonian::floattenCholesky(std::vector<vector<float>>& floatChol)
void Hamiltonian::floattenCholesky()
{
  //for (int n = 0; n < chol.size(); n++) floatChol.push_back(chol[n].cast<float>());
  for (int n = 0; n < chol.size(); n++) {
    vector<float> cholVec;
    for (int i = 0; i < norbs; i++)
      for (int j = 0; j <= i; j++)
        cholVec.push_back(float(chol[n](i, j)));
    floatChol.push_back(cholVec);
  }
};
