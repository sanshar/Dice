#include "Hamiltonian.h"
#include "integral.h"

using namespace std;
using namespace Eigen;

// constructor
Hamiltonian::Hamiltonian(string fname) 
{
  readIntegralsCholeskyAndInitializeDeterminantStaticVariables(fname, norbs, nalpha, nbeta, ecore, h1, h1Mod, chol);
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
