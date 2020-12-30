#include "DQMCMatrixElements.h"
#include "Determinants.h"

using namespace Eigen;
using namespace std;

using matPair = pair<MatrixXcd, MatrixXcd>;
using vecPair = pair<VectorXcd, VectorXcd>;


// left is adjointed
void calcGreensFunction(matPair& leftT, matPair& right, matPair& green) 
{
  green.first = right.first * (leftT.first * right.first).inverse() * leftT.first;
  green.second = right.second * (leftT.second * right.second).inverse() * leftT.second;
}


// Hamiltonian matrix element < phi_1 | H | phi_2 > / < phi_1 | phi_2 >
// green: rdm between phi_1 and phi_2
// leading cost: O(X M^3)
complex<double> calcHamiltonianElement(matPair& green, double enuc, MatrixXd& h1, vector<MatrixXd>& chol) 
{ 
  // core energy
  complex<double> ene = enuc;

  // one body part
  ene += green.first.cwiseProduct(h1).sum() + green.second.cwiseProduct(h1).sum();

  // two body part
  MatrixXcd eup = MatrixXcd::Zero(h1.rows(), h1.rows());
  MatrixXcd edn = MatrixXcd::Zero(h1.rows(), h1.rows());
  for (int i = 0; i < chol.size(); i++) {
    complex<double> cup = green.first.cwiseProduct(chol[i]).sum();
    complex<double> cdn = green.second.cwiseProduct(chol[i]).sum();
    eup.noalias() = chol[i] * green.first;
    edn.noalias() = chol[i] * green.second;
    ene += (cup * cup + cdn * cdn + 2. * cup * cdn - eup.cwiseProduct(eup.transpose()).sum() - edn.cwiseProduct(edn.transpose()).sum()) / 2.;
  }

  return ene;
}


// Hamiltonian matrix element < phi_1 | H | phi_2 > / < phi_1 | phi_2 >
// rotates cholesky vectors
// phi_1 can be an active space wave function
// TODO: allow core orbitals
// leading cost: O(X N A M)
complex<double> calcHamiltonianElement(matPair& phi1T, matPair& phi2, double enuc, MatrixXd& h1, vector<MatrixXd>& chol) 
{ 
  // core energy
  complex<double> ene = enuc;
  
  // calculate theta and green
  int numActOrbs = phi1T.first.cols();
  int numElecUp = phi2.first.cols();
  int numElecDn = phi2.second.cols();
  matPair theta, green;
  theta.first = phi2.first * (phi1T.first * phi2.first.block(0, 0, numActOrbs, numElecUp)).inverse();
  theta.second = phi2.second * (phi1T.second * phi2.second.block(0, 0, numActOrbs, numElecDn)).inverse();
  green.first = (theta.first * phi1T.first).transpose();
  green.second = (theta.second * phi1T.second).transpose();

  // one body part
  ene += green.first.cwiseProduct(h1.block(0, 0, numActOrbs, h1.rows())).sum() + green.second.cwiseProduct(h1.block(0, 0, numActOrbs, h1.rows())).sum();

  // two body part
  MatrixXcd fup = MatrixXcd::Zero(phi1T.first.rows(), phi1T.first.rows());
  MatrixXcd fdn = MatrixXcd::Zero(phi1T.second.rows(), phi1T.second.rows());
  MatrixXcd rotCholUp = MatrixXcd::Zero(phi1T.first.rows(), h1.rows());
  MatrixXcd rotCholDn = MatrixXcd::Zero(phi1T.second.rows(), h1.rows());
  for (int i = 0; i < chol.size(); i++) {
    rotCholUp.noalias() = phi1T.first * chol[i].block(0, 0, numActOrbs, h1.rows());
    rotCholDn.noalias() = phi1T.second * chol[i].block(0, 0, numActOrbs, h1.rows());
    fup.noalias() = rotCholUp * theta.first;
    fdn.noalias() = rotCholDn * theta.second;
    complex<double> cup = fup.trace();
    complex<double> cdn = fdn.trace();
    ene += (cup * cup + cdn * cdn + 2. * cup * cdn - fup.cwiseProduct(fup.transpose()).sum() - fdn.cwiseProduct(fdn.transpose()).sum()) / 2.;
  }

  return ene;
}


// Hamiltonian matrix element < phi_0 | H | psi > / < phi_0 | psi >
// using precalculated half rotated cholesky vectors
complex<double> calcHamiltonianElement(matPair& phi0T, matPair& psi, double enuc, MatrixXd& h1, pair<vector<MatrixXcd>, vector<MatrixXcd>>& rotChol) 
{
  // core energy
  complex<double> ene = enuc;
  
  // calculate theta and green
  matPair theta, green;
  theta.first = psi.first * (phi0T.first * psi.first).inverse();
  theta.second = psi.second * (phi0T.second * psi.second).inverse();
  green.first = theta.first * phi0T.first;
  green.second = theta.second * phi0T.second;

  // one body part
  ene += green.first.cwiseProduct(h1).sum() + green.second.cwiseProduct(h1).sum();

  // two body part
  MatrixXcd fup = MatrixXcd::Zero(rotChol.first[0].rows(), rotChol.first[0].rows());
  MatrixXcd fdn = MatrixXcd::Zero(rotChol.second[0].rows(), rotChol.second[0].rows());
  for (int i = 0; i < rotChol.first.size(); i++) {
    fup.noalias() = rotChol.first[i] * theta.first;
    fdn.noalias() = rotChol.second[i] * theta.second;
    complex<double> cup = fup.trace();
    complex<double> cdn = fdn.trace();
    ene += (cup * cup + cdn * cdn + 2. * cup * cdn - fup.cwiseProduct(fup.transpose()).sum() - fdn.cwiseProduct(fdn.transpose()).sum()) / 2.;
  }

  return ene;
}


// Hamiltonian matrix element < phi_1 | H | phi_2 > / < phi_1 | phi_2 >
// heat bath
complex<double> calcHamiltonianElement(matPair& walker, workingArray& work, double dene) 
{ 
  size_t norbs = Determinant::norbs;
  size_t nalpha = Determinant::nalpha;
  size_t nbeta = Determinant::nbeta;

  // make tables
  matPair aInv, r_aInv;
  aInv.first = walker.first.block(0, 0, nalpha, nalpha).inverse();
  aInv.second = walker.second.block(0, 0, nbeta, nbeta).inverse();
  r_aInv.first = walker.first.block(nalpha, 0, norbs - nalpha, nalpha) * aInv.first;
  r_aInv.second = walker.second.block(nbeta, 0, norbs - nbeta, nbeta) * aInv.second;

  // core energy
  complex<double> ene = dene;

  // loop over screened excitations
  for (int i = 0; i < work.nExcitations; i++) {
    int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
    double tia = work.HijElement[i];
  
    int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
    int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
    
    // overlap ratio 
    complex<double> overlapRatio(1., 0.);
    if (ex2 == 0) {// single excitation
      if (I%2 == 0)
        overlapRatio = r_aInv.first(A/2 - nalpha, I/2);
      else
        overlapRatio = r_aInv.second(A/2 - nbeta, I/2);
    }
    else {// double excitation
      if (I%2 == J%2) {
        if (J%2 == 0) 
          overlapRatio = r_aInv.first(A/2 - nalpha, I/2) * r_aInv.first(B/2 - nalpha, J/2) - r_aInv.first(A/2 - nalpha, J/2) * r_aInv.first(B/2 - nalpha, I/2);
        else 
          overlapRatio = r_aInv.second(A/2 - nbeta, I/2) * r_aInv.second(B/2 - nbeta, J/2) - r_aInv.second(A/2 - nbeta, J/2) * r_aInv.second(B/2 - nbeta, I/2);
      }
      else {
        if (I%2 == 0)
          overlapRatio = r_aInv.first(A/2 - nalpha, I/2) * r_aInv.second(B/2 - nbeta, J/2);
        else
          overlapRatio = r_aInv.second(A/2 - nbeta, I/2) * r_aInv.first(B/2 - nalpha, J/2);
      }
    }

    ene += tia * overlapRatio;
  }

  return ene;
}
