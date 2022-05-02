#include "DQMCMatrixElements.h"
#include "Determinants.h"
#include <unsupported/Eigen/KroneckerProduct>

using namespace Eigen;
using namespace std;

using matPair = pair<MatrixXcd, MatrixXcd>;
using matArray = std::array<MatrixXcd, 2>;
using vecPair = pair<VectorXcd, VectorXcd>;


// left is adjointed
void calcGreensFunction(matPair& leftT, matPair& right, matPair& green) 
{
  green.first = right.first * (leftT.first * right.first).inverse() * leftT.first;
  green.second = right.second * (leftT.second * right.second).inverse() * leftT.second;
}


// left is adjointed
void calcGreensFunction(MatrixXcd& leftT, MatrixXcd& right, MatrixXcd& green) 
{
  green = right * (leftT * right).inverse() * leftT;
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

// Hamiltonian matrix element < GHF | H | UHF > / < GHF | UHF >
// rotates cholesky vectors
complex<double> calcHamiltonianElement(MatrixXcd& ghf, matPair& A, double enuc, MatrixXd& h1, vector<MatrixXd>& chol, complex<double>& ovlp) 
{ 
  // core energy
  complex<double> ene = enuc;
  
  MatrixXcd B = ghf.adjoint();

  // calculate theta and green
  int numOrbs = A.first.rows();
  int numElec = B.cols();
  int nAlpha = A.first.cols();
  int nBeta = A.second.cols();


  MatrixXcd Afull = 0*B, theta, green;
  Afull.block(0,0,numOrbs,A.first.cols()) = A.first;
  Afull.block(numOrbs,A.first.cols(),numOrbs,A.second.cols()) = A.second;

  MatrixXcd Ovlp = B.adjoint()*Afull;
  ovlp = Ovlp.determinant();
  theta = Ovlp.inverse()*B.adjoint();
  green = (Afull * theta).transpose();

  // one body part
  ene += green.block(0,0,numOrbs,numOrbs).cwiseProduct(h1).sum() + green.block(numOrbs,numOrbs,numOrbs,numOrbs).cwiseProduct(h1).sum();
  //cout << "ene "<<ene<<endl;
  //cout << ene<<endl;
  // two body part
  MatrixXcd W(nAlpha, nAlpha), U(nBeta, nBeta), S(nBeta, nAlpha), T(nAlpha, nBeta);

  MatrixXcd Ltilde = MatrixXcd::Zero(numOrbs, nAlpha);
  MatrixXcd Mtilde = MatrixXcd::Zero(numOrbs, nBeta);
  MatrixXcd ThetaAA = theta.block(0,0,nAlpha,numOrbs), ThetaAB = theta.block(0,numOrbs, nAlpha, numOrbs),
    ThetaBA = theta.block(nAlpha, 0, nBeta, numOrbs), ThetaBB = theta.block(nAlpha, numOrbs, nBeta, numOrbs);

  for (int i = 0; i < chol.size(); i++) {
    Ltilde.noalias() = chol[i]*A.first;
    Mtilde.noalias() = chol[i]*A.second;

    W.noalias() = ThetaAA * Ltilde ; U.noalias() = ThetaBB * Mtilde ; S.noalias() = ThetaBA * Ltilde ; T.noalias() = ThetaAB * Mtilde;
    complex<double> cup = W.trace();
    complex<double> cdn = U.trace();
    ene += (cup * cup + cdn * cdn + 2. * cup * cdn - W.cwiseProduct(W.transpose()).sum() - U.cwiseProduct(U.transpose()).sum() 
             - 2. * S.cwiseProduct(T.transpose()).sum() ) / 2.;
    //ene += (cup * cup + cdn * cdn + 2. * cup * cdn ) / 2.;
  }

  return ene;
}


// Hamiltonian matrix element < GHF | H | UHF > / < GHF | UHF >
// rotates cholesky vectors
complex<double> calcHamiltonianElement(MatrixXcd& ghf, matPair& A, double enuc, MatrixXd& h1, pair<vector<MatrixXcd>, vector<MatrixXcd>>& rotChol, complex<double>& ovlp) 
{ 
  // core energy
  complex<double> ene = enuc;
  
  MatrixXcd B = ghf.adjoint();

  // calculate theta and green
  int numOrbs = A.first.rows();
  int numElec = B.cols();
  int nAlpha = A.first.cols();
  int nBeta = A.second.cols();


  MatrixXcd Afull = 0*B, theta, green;
  Afull.block(0,0,numOrbs,A.first.cols()) = A.first;
  Afull.block(numOrbs,A.first.cols(),numOrbs,A.second.cols()) = A.second;

  MatrixXcd Ovlp = B.adjoint()*Afull;
  ovlp = Ovlp.determinant();
  theta = Afull * Ovlp.inverse();
  green = (theta * B.adjoint()).transpose();

  // one body part
  ene += green.block(0,0,numOrbs,numOrbs).cwiseProduct(h1).sum() + green.block(numOrbs,numOrbs,numOrbs,numOrbs).cwiseProduct(h1).sum();

  MatrixXcd thetaA = theta.block(0, 0, numOrbs, numElec),thetaB= theta.block(numOrbs, 0, numOrbs, numElec);

  // two body part
  //vector<MatrixXcd> rotChol(2, MatrixXcd(numElec, numOrbs)), W(2, MatrixXcd(numElec, numOrbs));
  vector<MatrixXcd> W(2, MatrixXcd(numElec, numOrbs));


  for (int i = 0; i < rotChol.first.size(); i++) {
    //rotChol[0] = ghf.block(0, 0, numElec, numOrbs) * chol[i];
    //rotChol[1] = ghf.block(0, numOrbs, numElec, numOrbs) * chol[i];

    W[0] = rotChol.first[i] * thetaA;
    W[1] = rotChol.second[i] * thetaB;

    complex<double> W0trace = W[0].trace(), W1trace = W[1].trace();
    complex<double> J = (W0trace + W1trace) * (W0trace + W1trace);
    ene += J/2.;

    complex<double> K = W[0].cwiseProduct(W[0].transpose()).sum() + 
                        W[1].cwiseProduct(W[1].transpose()).sum() +
                        W[0].cwiseProduct(W[1].transpose()).sum() +
                        W[1].cwiseProduct(W[0].transpose()).sum() ;

    ene -= K/2.; 

  }

  return ene;
}


complex<double> calcHamiltonianElementNaive(MatrixXcd& At, MatrixXcd& B, double enuc, MatrixXd& h1, vector<MatrixXd>& chol) {
  // core energy
  complex<double> ene = enuc;
 
  // calculate theta and green
  int numOrbs = B.rows()/2;
  int numElec = B.cols();
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

  vector<MatrixXd> cholBig(chol.size(), MatrixXd::Zero(2*numOrbs, 2*numOrbs));
  for (int i=0; i<chol.size(); i++) {
    cholBig[i].block(0,0,numOrbs,numOrbs) = 1.*chol[i];
    cholBig[i].block(numOrbs,numOrbs,numOrbs,numOrbs) = 1.*chol[i];
  }
  MatrixXd hbig(2*numOrbs, 2*numOrbs); hbig.block(0,0,numOrbs, numOrbs) = 1.*h1; hbig.block(numOrbs,numOrbs,numOrbs,numOrbs) = 1.*h1;
  MatrixXcd theta = B*(At * B).inverse();
  MatrixXcd green = (theta*At).transpose();
  ene += green.cwiseProduct(hbig).sum();

  MatrixXcd W;
  for (int i = 0; i < chol.size(); i++) {
    W = green.transpose() * cholBig[i];
    ene += 0.5*W.trace()*W.trace();
    ene -= 0.5*W.cwiseProduct(W.transpose()).sum();
  }
  
  return ene;  
}

// Hamiltonian matrix element < GHF | H | GHF > / < GHF | UHF >
// rotates cholesky vectors
complex<double> calcHamiltonianElement(MatrixXcd& At, MatrixXcd& B, double enuc, MatrixXd& h1, vector<MatrixXd>& chol) 
{ 
  // core energy
  complex<double> ene = enuc;
 
  // calculate theta and green
  int numOrbs = B.rows()/2;
  int numElec = B.cols();
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

  MatrixXcd theta = B*(At * B).inverse();
  MatrixXcd green = (theta*At);

  MatrixXcd thetaA = theta.block(0, 0, numOrbs, numElec),thetaB= theta.block(numOrbs, 0, numOrbs, numElec);

  // one body part
  ene += green.block(0,0,numOrbs,numOrbs).cwiseProduct(h1).sum() + green.block(numOrbs,numOrbs,numOrbs,numOrbs).cwiseProduct(h1).sum();

  
  vector<MatrixXcd> rotChol(2, MatrixXcd(numElec, numOrbs)), W(2, MatrixXcd(numElec, numOrbs));

  for (int i = 0; i < chol.size(); i++) {
    rotChol[0] = At.block(0,0,numElec,numOrbs) * chol[i];
    rotChol[1] = At.block(0,numOrbs,numElec,numOrbs) * chol[i]; //chol[i] * B.block(numOrbs,0,numOrbs, numElec);
    W[0] = rotChol[0] * thetaA;
    W[1] = rotChol[1] * thetaB;

    complex<double> W0trace = W[0].trace(), W1trace = W[1].trace();
    complex<double> J = (W0trace + W1trace) * (W0trace + W1trace);
    ene += J/2.;

    complex<double> K = W[0].cwiseProduct(W[0].transpose()).sum() + 
                        W[1].cwiseProduct(W[1].transpose()).sum() +
                        W[0].cwiseProduct(W[1].transpose()).sum() +
                        W[1].cwiseProduct(W[0].transpose()).sum() ;

    ene -= K/2.; 
  }
  
  return ene;
}

// Hamiltonian matrix element < GHF | H | GHF > / < GHF | UHF >
// rotates cholesky vectors
complex<double> calcHamiltonianElement(MatrixXcd& At, MatrixXcd& B, double enuc, MatrixXd& h1, std::pair<std::vector<Eigen::MatrixXcd>, std::vector<Eigen::MatrixXcd>>& rotChol) 
{ 
  // core energy
  complex<double> ene = enuc;
 
  // calculate theta and green
  int numOrbs = B.rows()/2;
  int numElec = B.cols();
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

  MatrixXcd theta = B*(At * B).inverse();
  MatrixXcd green = (theta*At);

  MatrixXcd thetaA = theta.block(0, 0, numOrbs, numElec),thetaB= theta.block(numOrbs, 0, numOrbs, numElec);

  // one body part
  ene += green.block(0,0,numOrbs,numOrbs).cwiseProduct(h1).sum() + green.block(numOrbs,numOrbs,numOrbs,numOrbs).cwiseProduct(h1).sum();

  
  vector<MatrixXcd> W(2, MatrixXcd(numElec, numOrbs));

  for (int i = 0; i < rotChol.first.size(); i++) {
    //rotChol[0] = At.block(0,0,numElec,numOrbs) * chol[i];
    //rotChol[1] = At.block(0,numOrbs,numElec,numOrbs) * chol[i]; //chol[i] * B.block(numOrbs,0,numOrbs, numElec);
    W[0] = rotChol.first[i] * thetaA;
    W[1] = rotChol.second[i] * thetaB;

    complex<double> W0trace = W[0].trace(), W1trace = W[1].trace();
    complex<double> J = (W0trace + W1trace) * (W0trace + W1trace);
    ene += J/2.;

    complex<double> K = W[0].cwiseProduct(W[0].transpose()).sum() + 
                        W[1].cwiseProduct(W[1].transpose()).sum() +
                        W[0].cwiseProduct(W[1].transpose()).sum() +
                        W[1].cwiseProduct(W[0].transpose()).sum() ;

    ene -= K/2.; 
  }
  
  return ene;
}

// Hamiltonian matrix element < d GHF/cxi | H | GHF > / < GHF | UHF >
// rotates cholesky vectors
complex<double> calcGradient(MatrixXcd& At, MatrixXcd& B, double enuc, MatrixXd& h1, vector<Eigen::Map<Eigen::MatrixXd>>& chol, MatrixXcd& Grad) 
{ 
  Grad.setZero();

  // core energy
  complex<double> ene = enuc;

  // calculate theta and green
  int numOrbs = B.rows()/2;
  int numElec = B.cols();
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

  MatrixXcd theta = B*(At * B).inverse();
  //MatrixXcd theta = B;//*(At * B).inverse();
  MatrixXcd green = (theta*At);

  //cout << (B.adjoint() * A) <<endl;exit(0);
  MatrixXcd thetaA = theta.block(0, 0, numOrbs, numElec),thetaB= theta.block(numOrbs, 0, numOrbs, numElec);
  MatrixXcd AtA = At.block(0, 0, numElec, numOrbs),AtB= At.block(0, numOrbs, numElec, numOrbs);

  // one body part
  ene += green.block(0,0,numOrbs,numOrbs).cwiseProduct(h1).sum() + green.block(numOrbs,numOrbs,numOrbs,numOrbs).cwiseProduct(h1).sum();
  Grad.block(0,0,numOrbs,numElec) += h1 * thetaA - thetaA * AtA*(h1 * thetaA) - thetaA * AtB * (h1 * thetaB);
  Grad.block(numOrbs,0,numOrbs,numElec) += h1 * thetaB - thetaB * AtA*(h1 * thetaA) - thetaB * AtB * (h1 * thetaB);;
  
  
  vector<MatrixXcd> rotChol(2, MatrixXcd(numElec, numOrbs)), 
                    W(2, MatrixXcd(numElec, numOrbs)),
                    X(2, MatrixXcd(numElec, numOrbs));
  for (int i = 0; i < chol.size(); i++) {
    rotChol[0] = chol[i] * thetaA;
    rotChol[1] = chol[i] * thetaB; 

    W[0] = AtA * rotChol[0];
    W[1] = AtB * rotChol[1];
    X[0] = rotChol[0] - thetaA * (W[0] + W[1]); 
    X[1] = rotChol[1] - thetaB * (W[0] + W[1]);

    complex<double> W0trace = W[0].trace(), W1trace = W[1].trace();
    Grad.block(0,0,numOrbs,numElec) += (rotChol[0] - thetaA * AtA * rotChol[0] - thetaA * AtB * rotChol[1])* (W0trace + W1trace);
    Grad.block(numOrbs,0,numOrbs,numElec) += (rotChol[1] - thetaB * AtA * rotChol[0] - thetaB * AtB * rotChol[1]) * (W0trace + W1trace);
    Grad.block(0,0,numOrbs,numElec) -= X[0] *(W[0] + W[1]);
    Grad.block(numOrbs,0,numOrbs,numElec) -= X[1] *(W[0] + W[1]);

    complex<double> J = (W0trace + W1trace) * (W0trace + W1trace);
    ene += J/2.;

    complex<double> K = W[0].cwiseProduct(W[0].transpose()+W[1].transpose()).sum() + 
                        W[1].cwiseProduct(W[0].transpose()+W[1].transpose()).sum() ;

    ene -= K/2.; 
  }

  return ene;
}


// Hamiltonian matrix element < phi_1 | H | phi_2 > / < phi_1 | phi_2 >
// rotates cholesky vectors
// phi_1 can be an active space wave function
// TODO: allow core orbitals
// leading cost: O(X N A M)
complex<double> calcHamiltonianElement_sRI(matPair& phi1T, matPair& phi2, matPair& phi0T, matPair& phi0, double enuc, MatrixXd& h1, vector<MatrixXd>& chol, vector<MatrixXd>& richol) 
{ 
  // core energy
  complex<double> ene = enuc;

  uniform_real_distribution<double> normal(-1., 1.);
  double Nri = 1.*richol.size();
  for (int j=0; j<richol.size(); j++) {
    richol[j].setZero();

    for (int i=0; i<chol.size(); i++) {
      double ran = normal(generator);
      richol[j] += ran/abs(ran) * chol[i];     
    }
    richol[j] /= Nri;
  }

  complex<double> Ephi = calcHamiltonianElement(phi1T, phi2, enuc, h1, richol);
  complex<double> Ephi0 = calcHamiltonianElement(phi0T, phi0, enuc, h1, richol);

  return Ephi-Ephi0;
}

// Hamiltonian matrix element < phi_0 | H | psi > / < phi_0 | psi >
// using precalculated half rotated cholesky vectors
complex<double> calcHamiltonianElement(matPair& phi0T, matPair& psi, double enuc, MatrixXd& h1, pair<vector<MatrixXcd>, vector<MatrixXcd>>& rotChol, int nchol) 
{

  if (nchol == -1) nchol = rotChol.first.size();
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
  for (int i = 0; i < nchol; i++) {
    fup.noalias() = rotChol.first[i] * theta.first;
    fdn.noalias() = rotChol.second[i] * theta.second;
    complex<double> cup = fup.trace();
    complex<double> cdn = fdn.trace();
    ene += (cup * cup + cdn * cdn + 2. * cup * cdn - fup.cwiseProduct(fup.transpose()).sum() - fdn.cwiseProduct(fdn.transpose()).sum()) / 2.;
  }

  return ene;
}


// Hamiltonian matrix element < phi_0 | H | psi > / < phi_0 | psi >
// using precalculated half rotated cholesky vectors
// rhf
complex<double> calcHamiltonianElement(MatrixXcd& phi0T, MatrixXcd& psi, double enuc, MatrixXd& h1, vector<MatrixXcd>& rotChol, int nchol) 
{
  if (nchol == -1) nchol = rotChol.size();
  // core energy
  complex<double> ene = enuc;
  
  // calculate theta and green
  MatrixXcd theta, green;
  theta = psi * (phi0T * psi).inverse();
  green = theta * phi0T;

  // one body part
  ene += 2. * green.cwiseProduct(h1).sum();

  // two body part
  MatrixXcd f = MatrixXcd::Zero(rotChol[0].rows(), rotChol[0].rows());
  for (int i = 0; i < nchol; i++) {
    f.noalias() = rotChol[i] * theta;
    complex<double> c = f.trace();
    ene += (2. * c * c - f.cwiseProduct(f.transpose()).sum());
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


// Hamiltonian matrix element < \sum _i phi_i | H | psi >
// multislater local energy
// leading cost: O(X N_c + X N M^2)
// returns both overlap and local energy
// TODO: use either array or pair, not both; make interface consistent with other local energy functions perhaps by changing all returns to <overlap, local energy> pairs
pair<complex<double>, complex<double>> calcHamiltonianElement0(matPair& phi0T, std::array<vector<std::array<VectorXi, 2>>, 2>& ciExcitations, vector<double>& ciParity, vector<double>& ciCoeffs, matPair& psi, double enuc, MatrixXd& h1, vector<MatrixXd>& chol)
{
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;
  std::array<int, 2> nelec{nalpha, nbeta};
  size_t ndets = ciCoeffs.size();

  complex<double> overlap(0., 0.);
  complex<double> ene(0., 0.);
  matArray theta, green, greenp;
  theta[0] = psi.first * (phi0T.first * psi.first).inverse();
  theta[1] = psi.second * (phi0T.second * psi.second).inverse();
  green[0] = (theta[0] * phi0T.first).transpose();
  green[1] = (theta[1] * phi0T.second).transpose();
  greenp[0] = green[0] - MatrixXcd::Identity(norbs, norbs);
  greenp[1] = green[1] - MatrixXcd::Identity(norbs, norbs);
  matArray greeno;
  greeno[0] = green[0].block(0, 0, nalpha, norbs);
  greeno[1] = green[1].block(0, 0, nbeta, norbs);
  //green[0] = green[0].block(0, 0, nalpha, norbs);
  //green[1] = green[1].block(0, 0, nbeta, norbs);

  // all quantities henceforth will be calculated in "units" of overlap0 = < phi_0 | psi >
  complex<double> overlap0 = (phi0T.first * psi.first).determinant() * (phi0T.second * psi.second).determinant();
 
  // ref contribution
  overlap += ciCoeffs[0];
  std::array<complex<double>, 2> hG;
  hG[0] = greeno[0].cwiseProduct(h1.block(0, 0, nalpha, norbs)).sum();
  hG[1] = greeno[1].cwiseProduct(h1.block(0, 0, nbeta, norbs)).sum();
  //hG[0] = green[0].cwiseProduct(h1).sum();
  //hG[1] = green[1].cwiseProduct(h1).sum();
  ene += ciCoeffs[0] * (hG[0] + hG[1]);
  
  // 1e intermediate
  matArray roth1;
  roth1[0] = (greeno[0] * h1) * greenp[0];
  roth1[1] = (greeno[1] * h1) * greenp[1];

  // G^{p}_{t} blocks
  vector<matArray> gBlocks;
  vector<std::array<complex<double>, 2>> gBlockDets;
  matArray empty;
  gBlocks.push_back(empty);
  std::array<complex<double>, 2> identity{1., 1.};
  gBlockDets.push_back(identity);
  
  // iterate over excitations
  for (int i = 1; i < ndets; i++) {
    matArray blocks;
    std::array<complex<double>, 2> oneEne, dets;
    for (int sz = 0; sz < 2; sz++) {
      int rank = ciExcitations[sz][i][0].size();
      if (rank == 0) {
        dets[sz] = 1.;
        oneEne[sz] = hG[sz];
      }
      else {
        blocks[sz] = MatrixXcd::Zero(rank, rank);
        for (int p = 0; p < rank; p++) 
          for (int t = 0; t < rank; t++) 
            blocks[sz](p, t) = green[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t));
        
        dets[sz] = blocks[sz].determinant();
        oneEne[sz] = hG[sz] * dets[sz];

        MatrixXcd temp;
        for (int p = 0; p < rank; p++) {
          temp = blocks[sz];
          for (int t = 0; t < rank; t++)
            temp(p, t) = roth1[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t));
          oneEne[sz] -= temp.determinant();
        }
      }
    }

    overlap += ciCoeffs[i] * ciParity[i] * dets[0] * dets[1];
    ene += ciCoeffs[i] * ciParity[i] * (oneEne[0] * dets[1] + dets[0] * oneEne[1]);
    gBlocks.push_back(blocks);
    gBlockDets.push_back(dets);
  }
  
  // 2e intermediates
  matArray int1, int2;
  int1[0] = 0. * greeno[0];
  int1[1] = 0. * greeno[0];
  int2[0] = 0. * greeno[0];
  int2[1] = 0. * greeno[0];
  // iterate over cholesky
  for (int n = 0; n < chol.size(); n++) {
    std::array<complex<double>, 2> lG, l2G2;
    matArray exc;
    for (int sz = 0; sz < 2; sz++) {
      exc[sz].noalias() = chol[n].block(0, 0, nelec[sz], norbs) * theta[sz];
      lG[sz] = exc[sz].trace();
      l2G2[sz] = lG[sz] * lG[sz] - exc[sz].cwiseProduct(exc[sz].transpose()).sum();
      //int2[sz].noalias() = (greeno[sz] * chol[n]) * greenp[sz];
      int2[sz].setZero();
      int2[sz].block(0, 0, nelec[sz], norbs) = (chol[n].block(0, 0, nelec[sz], norbs) * theta[sz]).transpose() * theta[sz].transpose();
      int2[sz].noalias() -= greeno[sz] * chol[n]; 
      //int2[sz].noalias() = (greeno[sz] * chol[n].block(0, 0, norbs, norbsAct)) * greenp[sz].block(0, 0, norbsAct, norbs);
      int1[sz] = lG[sz] * int2[sz];
      int1[sz].noalias() -= (greeno[sz] * chol[n].block(0, 0, norbs, nelec[sz])) * int2[sz];
    }

    // ref contribution
    ene += ciCoeffs[0] * (l2G2[0] + l2G2[1] + 2. * lG[0] * lG[1]) / 2.;
    
    // iterate over excitations
    for (int i = 1; i < ndets; i++) {
      std::array<complex<double>, 2> oneEne, twoEne;
      for (int sz = 0; sz < 2; sz++) {
        int rank = ciExcitations[sz][i][0].size();
        if (rank == 0) {
          oneEne[sz] = lG[sz];
          twoEne[sz] = l2G2[sz];
        }
        else if (rank == 1) {
          oneEne[sz] = lG[sz] * gBlockDets[i][sz] - int2[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0));
          twoEne[sz] = l2G2[sz] * gBlockDets[i][sz] - 2. * int1[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0));
        }
        else if (rank == 2) {
          oneEne[sz] = lG[sz] * gBlockDets[i][sz] 
                     - int2[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0)) * gBlocks[i][sz](1, 1)
                     + int2[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](1)) * gBlocks[i][sz](1, 0)
                     + int2[sz](ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](0)) * gBlocks[i][sz](0, 1)
                     - int2[sz](ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](1)) * gBlocks[i][sz](0, 0);

          twoEne[sz] = l2G2[sz] * gBlockDets[i][sz]
                     + 2. * (- int1[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0)) * gBlocks[i][sz](1, 1)
                             + int1[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](1)) * gBlocks[i][sz](1, 0)
                             + int1[sz](ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](0)) * gBlocks[i][sz](0, 1)
                             - int1[sz](ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](1)) * gBlocks[i][sz](0, 0)  )
                     + 2. * (  int2[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0))    
                             * int2[sz](ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](1)) 
                             - int2[sz](ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](0)) 
                             * int2[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](1))  );
        }
        else {
          // oneEne
          {
            oneEne[sz] = lG[sz] * gBlockDets[i][sz];

            MatrixXcd temp;
            for (int p = 0; p < rank; p++) {
              temp = gBlocks[i][sz];
              for (int t = 0; t < rank; t++)
                temp(p, t) = int2[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t));
              oneEne[sz] -= temp.determinant();
            }
          }
          
          // twoEne
          {
            // term 1
            twoEne[sz] = l2G2[sz] * gBlockDets[i][sz];

            // term 2
            MatrixXcd temp;
            for (int p = 0; p < rank; p++) {
              temp = gBlocks[i][sz];
              for (int t = 0; t < rank; t++)
                temp(p, t) = int1[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t));
              twoEne[sz] -= 2. * temp.determinant();
            }
            
            // term 3
            if (rank > 1) {
              vector<int> range(rank);
              for (int p = 0; p < rank; p++) range[p] = p;
              for (int p = 0; p < rank; p++) {
                for (int q = p + 1; q < rank; q++) {
                  double parity_pq = ((p + q + 1)%2 == 0) ? 1. : -1.;
                  vector<int> pq = {p, q};
                  vector<int> diff0;
                  std::set_difference(range.begin(), range.end(), pq.begin(), pq.end(), std::inserter(diff0, diff0.begin()));
                  for (int t = 0; t < rank; t++) {
                    for (int u = t + 1; u < rank; u++) {
                      double parity_tu = ((t + u + 1)%2 == 0) ? 1. : -1.;
                      vector<int> tu = {t, u};
                      vector<int> diff1;
                      std::set_difference(range.begin(), range.end(), tu.begin(), tu.end(), std::inserter(diff1, diff1.begin()));
                      
                      complex<double> blockBlockDet;
                      if (rank < 3) blockBlockDet = 1.;
                      else {
                        MatrixXcd blockBlock = MatrixXcd::Zero(rank - 2, rank - 2);
                        for (int mup = 0; mup < rank - 2; mup++) 
                          for (int nup = 0; nup < rank - 2; nup++) 
                            blockBlock(mup, nup) = gBlocks[i][sz](diff0[mup], diff1[nup]);
                        blockBlockDet = blockBlock.determinant();
                      }

                      twoEne[sz] += 2. * parity_pq * parity_tu * blockBlockDet * (  int2[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t)) 
                                                                                  * int2[sz](ciExcitations[sz][i][0](q), ciExcitations[sz][i][1](u)) 
                                                                                  - int2[sz](ciExcitations[sz][i][0](q), ciExcitations[sz][i][1](t)) 
                                                                                  * int2[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](u))  );
                    }
                  }
                }
              }
            }
          }
        } 
      } // sz

      ene += ciParity[i] * ciCoeffs[i] * (twoEne[0] * gBlockDets[i][1] + 2. * oneEne[0] * oneEne[1] + gBlockDets[i][0] * twoEne[1]) / 2.;
    
    } // dets
  } // chol

  overlap *= overlap0;
  ene *= overlap0;
  ene += enuc * overlap;
  return pair<complex<double>, complex<double>>(overlap, ene);
}


// Hamiltonian matrix element < \sum _i phi_i | H | psi >
// multislater local energy
// leading cost: O(X N_c + X N M^2)
// returns both overlap and local energy
// TODO: use either array or pair, not both; make interface consistent with other local energy functions perhaps by changing all returns to <overlap, local energy> pairs
pair<complex<double>, complex<double>> calcHamiltonianElement(matPair& phi0T, std::array<vector<std::array<VectorXi, 2>>, 2>& ciExcitations, vector<double>& ciParity, vector<double>& ciCoeffs, matPair& psi, double enuc, MatrixXd& h1, vector<MatrixXd>& chol, int nchol, int nact, int ncore)
{
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;
  std::array<int, 2> nelec{nalpha, nbeta};
  size_t ndets = ciCoeffs.size();
  if (nchol == -1) nchol = chol.size();
  if (nact == -1) nact = norbs;
  if (ncore == -1) ncore = 0;

  complex<double> overlap(0., 0.);
  complex<double> ene(0., 0.);
  matArray theta, green, greenp;
  theta[0] = psi.first * (phi0T.first * psi.first).inverse();
  theta[1] = psi.second * (phi0T.second * psi.second).inverse();
  green[0] = (theta[0] * phi0T.first).transpose();
  green[1] = (theta[1] * phi0T.second).transpose();
  greenp[0] = green[0] - MatrixXcd::Identity(norbs, norbs);
  greenp[1] = green[1] - MatrixXcd::Identity(norbs, norbs);
  matArray greeno;
  greeno[0] = green[0].block(0, 0, nalpha, norbs);
  greeno[1] = green[1].block(0, 0, nbeta, norbs);
  //green[0] = green[0].block(0, 0, nalpha, norbs);
  //green[1] = green[1].block(0, 0, nbeta, norbs);

  // all quantities henceforth will be calculated in "units" of overlap0 = < phi_0 | psi >
  complex<double> overlap0 = (phi0T.first * psi.first).determinant() * (phi0T.second * psi.second).determinant();
 
  // ref contribution
  overlap += ciCoeffs[0];
  std::array<complex<double>, 2> hG;
  hG[0] = greeno[0].cwiseProduct(h1.block(0, 0, nalpha, norbs)).sum();
  hG[1] = greeno[1].cwiseProduct(h1.block(0, 0, nbeta, norbs)).sum();
  //hG[0] = green[0].cwiseProduct(h1).sum();
  //hG[1] = green[1].cwiseProduct(h1).sum();
  ene += ciCoeffs[0] * (hG[0] + hG[1]);
  
  // 1e intermediate
  matArray roth1;
  roth1[0] = (greeno[0] * h1) * greenp[0];
  roth1[1] = (greeno[1] * h1) * greenp[1];

  // G^{p}_{t} blocks
  vector<matArray> gBlocks;
  vector<std::array<complex<double>, 2>> gBlockDets;
  matArray empty;
  gBlocks.push_back(empty);
  std::array<complex<double>, 2> identity{1., 1.};
  gBlockDets.push_back(identity);
  
  // iterate over excitations
  for (int i = 1; i < ndets; i++) {
    matArray blocks;
    std::array<complex<double>, 2> oneEne, dets;
    for (int sz = 0; sz < 2; sz++) {
      int rank = ciExcitations[sz][i][0].size();
      if (rank == 0) {
        dets[sz] = 1.;
        oneEne[sz] = hG[sz];
      }
      else {
        blocks[sz] = MatrixXcd::Zero(rank, rank);
        for (int p = 0; p < rank; p++) 
          for (int t = 0; t < rank; t++) 
            blocks[sz](p, t) = green[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t));
        
        dets[sz] = blocks[sz].determinant();
        oneEne[sz] = hG[sz] * dets[sz];

        MatrixXcd temp;
        for (int p = 0; p < rank; p++) {
          temp = blocks[sz];
          for (int t = 0; t < rank; t++)
            temp(p, t) = roth1[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t));
          oneEne[sz] -= temp.determinant();
        }
      }
    }

    overlap += ciCoeffs[i] * ciParity[i] * dets[0] * dets[1];
    ene += ciCoeffs[i] * ciParity[i] * (oneEne[0] * dets[1] + dets[0] * oneEne[1]);
    gBlocks.push_back(blocks);
    gBlockDets.push_back(dets);
  }
  
  // 2e intermediates
  matArray int1, int2;
  int1[0] = 0. * greeno[0].block(0, 0, nelec[0], nact + ncore);
  int1[1] = 0. * greeno[1].block(0, 0, nelec[1], nact + ncore);
  int2[0] = 0. * greeno[0].block(0, 0, nelec[0], nact + ncore);
  int2[1] = 0. * greeno[1].block(0, 0, nelec[1], nact + ncore);
  std::array<complex<double>, 2> l2G2Tot;
  l2G2Tot[0] = complex<double>(0., 0.);
  l2G2Tot[1] = complex<double>(0., 0.);
  
  // iterate over cholesky
  //for (int n = 0; n < chol.size(); n++) {
  for (int n = 0; n < nchol; n++) {
    std::array<complex<double>, 2> lG, l2G2;
    matArray exc;
    for (int sz = 0; sz < 2; sz++) {
      exc[sz].noalias() = chol[n].block(0, 0, nelec[sz], norbs) * theta[sz];
      lG[sz] = exc[sz].trace();
      l2G2[sz] = lG[sz] * lG[sz] - exc[sz].cwiseProduct(exc[sz].transpose()).sum();
      l2G2Tot[sz] += l2G2[sz];
      //int2[sz].noalias() = (greeno[sz] * chol[n]) * greenp[sz];
      //int2[sz].setZero();
      //int2[sz].block(0, 0, nelec[sz], norbs) = (chol[n].block(0, 0, nelec[sz], norbs) * theta[sz]).transpose() * theta[sz].transpose();
      //int2[sz].noalias() -= greeno[sz] * chol[n]; 
      int2[sz].noalias() = (greeno[sz] * chol[n].block(0, 0, norbs, nact + ncore)) * greenp[sz].block(0, ncore, nact + ncore, nact + ncore);
      int1[sz].noalias() += lG[sz] * int2[sz];
      int1[sz].noalias() -= (greeno[sz] * chol[n].block(0, 0, norbs, nelec[sz])) * int2[sz];
    }

    // ref contribution
    ene += ciCoeffs[0] * (l2G2[0] + l2G2[1] + 2. * lG[0] * lG[1]) / 2.;
    
    // iterate over excitations
    for (int i = 1; i < ndets; i++) {
      std::array<complex<double>, 2> oneEne, twoEne;
      for (int sz = 0; sz < 2; sz++) {
        int rank = ciExcitations[sz][i][0].size();
        if (rank == 0) {
          oneEne[sz] = lG[sz];
        }
        if (rank == 1) {
          oneEne[sz] = lG[sz] * gBlockDets[i][sz] - int2[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0) - ncore);
        }
        else if (rank == 2) {
          oneEne[sz] = lG[sz] * gBlockDets[i][sz] 
                     - int2[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0) - ncore) * gBlocks[i][sz](1, 1)
                     + int2[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](1) - ncore) * gBlocks[i][sz](1, 0)
                     + int2[sz](ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](0) - ncore) * gBlocks[i][sz](0, 1)
                     - int2[sz](ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](1) - ncore) * gBlocks[i][sz](0, 0);

          twoEne[sz] = 2. * (  int2[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0) - ncore)    
                             * int2[sz](ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](1) - ncore) 
                             - int2[sz](ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](0) - ncore) 
                             * int2[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](1) - ncore)  );
        }
        else if (rank == 3) {
          // oneEne
          {
            oneEne[sz] = lG[sz] * gBlockDets[i][sz];

            Matrix3cd temp;
            for (int p = 0; p < rank; p++) {
              temp = gBlocks[i][sz];
              for (int t = 0; t < rank; t++)
                temp(p, t) = int2[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t) - ncore);
              oneEne[sz] -= temp.determinant();
            }
          }
          
          // twoEne
          {
            twoEne[sz] = complex<double>(0., 0.);
            // term 3
            for (int p = 0; p < rank; p++) {
              for (int q = p + 1; q < rank; q++) {
                double parity_pq = ((p + q + 1)%2 == 0) ? 1. : -1.;
                for (int t = 0; t < rank; t++) {
                  for (int u = t + 1; u < rank; u++) {
                    double parity_tu = ((t + u + 1)%2 == 0) ? 1. : -1.;
                    complex<double> blockBlockDet = gBlocks[i][sz](3 - p - q, 3 - t - u);
                    twoEne[sz] += 2. * parity_pq * parity_tu * blockBlockDet * (  int2[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t) - ncore) 
                                                                                * int2[sz](ciExcitations[sz][i][0](q), ciExcitations[sz][i][1](u) - ncore) 
                                                                                - int2[sz](ciExcitations[sz][i][0](q), ciExcitations[sz][i][1](t) - ncore) 
                                                                                * int2[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](u) - ncore)  );
                  }
                }
              }
            }
          }
        }
        else if (rank == 4) {
          // oneEne
          {
            oneEne[sz] = lG[sz] * gBlockDets[i][sz];

            Matrix4cd temp;
            for (int p = 0; p < rank; p++) {
              temp = gBlocks[i][sz];
              for (int t = 0; t < rank; t++)
                temp(p, t) = int2[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t) - ncore);
              oneEne[sz] -= temp.determinant();
            }
          }
          
          // twoEne
          {
            twoEne[sz] = complex<double>(0., 0.);
            // term 3
            for (int p = 0; p < rank; p++) {
              for (int q = p + 1; q < rank; q++) {
                double parity_pq = ((p + q + 1)%2 == 0) ? 1. : -1.;
                int pp, qp;
                if (p == 0) {
                  pp = 3 - q + q/3; qp = 6 - q - pp;
                }
                else {
                  pp = 0; qp = 6 - p - q;
                }
                for (int t = 0; t < rank; t++) {
                  for (int u = t + 1; u < rank; u++) {
                    double parity_tu = ((t + u + 1)%2 == 0) ? 1. : -1.;
                    int tp, up;
                    if (t == 0) {
                      tp = 3 - u + u/3; up = 6 - u - tp;
                    }
                    else {
                      tp = 0; up = 6 - t - u;
                    }
                    
                    complex<double> blockBlockDet = gBlocks[i][sz](pp, tp) * gBlocks[i][sz](qp, up) - gBlocks[i][sz](pp, up) * gBlocks[i][sz](qp, tp);

                    twoEne[sz] += 2. * parity_pq * parity_tu * blockBlockDet * (  int2[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t) - ncore) 
                                                                                * int2[sz](ciExcitations[sz][i][0](q), ciExcitations[sz][i][1](u) - ncore) 
                                                                                - int2[sz](ciExcitations[sz][i][0](q), ciExcitations[sz][i][1](t) - ncore) 
                                                                                * int2[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](u) - ncore)  );
                  }
                }
              }
            }
          }
        } 
        else {
          // oneEne
          {
            oneEne[sz] = lG[sz] * gBlockDets[i][sz];

            MatrixXcd temp;
            for (int p = 0; p < rank; p++) {
              temp = gBlocks[i][sz];
              for (int t = 0; t < rank; t++)
                temp(p, t) = int2[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t) - ncore);
              oneEne[sz] -= temp.determinant();
            }
          }
          
          // twoEne
          {
            twoEne[sz] = complex<double>(0., 0.);
            // term 3
            vector<int> range(rank);
            for (int p = 0; p < rank; p++) range[p] = p;
            for (int p = 0; p < rank; p++) {
              for (int q = p + 1; q < rank; q++) {
                double parity_pq = ((p + q + 1)%2 == 0) ? 1. : -1.;
                vector<int> pq = {p, q};
                vector<int> diff0;
                std::set_difference(range.begin(), range.end(), pq.begin(), pq.end(), std::inserter(diff0, diff0.begin()));
                for (int t = 0; t < rank; t++) {
                  for (int u = t + 1; u < rank; u++) {
                    double parity_tu = ((t + u + 1)%2 == 0) ? 1. : -1.;
                    vector<int> tu = {t, u};
                    vector<int> diff1;
                    std::set_difference(range.begin(), range.end(), tu.begin(), tu.end(), std::inserter(diff1, diff1.begin()));
                    
                    MatrixXcd blockBlock = MatrixXcd::Zero(rank - 2, rank - 2);
                    for (int mup = 0; mup < rank - 2; mup++) 
                      for (int nup = 0; nup < rank - 2; nup++) 
                        blockBlock(mup, nup) = gBlocks[i][sz](diff0[mup], diff1[nup]);
                    complex<double> blockBlockDet = blockBlock.determinant();

                    twoEne[sz] += 2. * parity_pq * parity_tu * blockBlockDet * (  int2[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t) - ncore) 
                                                                                * int2[sz](ciExcitations[sz][i][0](q), ciExcitations[sz][i][1](u) - ncore) 
                                                                                - int2[sz](ciExcitations[sz][i][0](q), ciExcitations[sz][i][1](t) - ncore) 
                                                                                * int2[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](u) - ncore)  );
                  }
                }
              }
            }
          }
        } 
      } // sz

      ene += ciParity[i] * ciCoeffs[i] * (twoEne[0] * gBlockDets[i][1] + 2. * oneEne[0] * oneEne[1] + gBlockDets[i][0] * twoEne[1]) / 2.;
    
    } // dets
  } // chol
    
  // iterate over excitations
  for (int i = 1; i < ndets; i++) {
    std::array<complex<double>, 2> oneEne, twoEne;
    for (int sz = 0; sz < 2; sz++) {
      int rank = ciExcitations[sz][i][0].size();
      if (rank == 0) {
        twoEne[sz] = l2G2Tot[sz];
      }
      else if (rank == 1) {
        twoEne[sz] = l2G2Tot[sz] * gBlockDets[i][sz] - 2. * int1[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0) - ncore);
      }
      else if (rank == 2) {
        twoEne[sz] = l2G2Tot[sz] * gBlockDets[i][sz]
                   + 2. * (- int1[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0) - ncore) * gBlocks[i][sz](1, 1)
                           + int1[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](1) - ncore) * gBlocks[i][sz](1, 0)
                           + int1[sz](ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](0) - ncore) * gBlocks[i][sz](0, 1)
                           - int1[sz](ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](1) - ncore) * gBlocks[i][sz](0, 0)  );
      }
      else if (rank == 3) {
        // twoEne
        {
          // term 1
          twoEne[sz] = l2G2Tot[sz] * gBlockDets[i][sz];

          //term 2
          Matrix3cd temp;
          for (int p = 0; p < rank; p++) {
            temp = gBlocks[i][sz];
            for (int t = 0; t < rank; t++)
              temp(p, t) = int1[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t) - ncore);
            twoEne[sz] -= 2. * temp.determinant();
          }
        }
      } 
      else if (rank == 4) {
        // twoEne
        {
          // term 1
          twoEne[sz] = l2G2Tot[sz] * gBlockDets[i][sz];

          //term 2
          Matrix4cd temp;
          for (int p = 0; p < rank; p++) {
            temp = gBlocks[i][sz];
            for (int t = 0; t < rank; t++)
              temp(p, t) = int1[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t) - ncore);
            twoEne[sz] -= 2. * temp.determinant();
          }
        }
      } 
      else {
        // twoEne
        {
          // term 1
          twoEne[sz] = l2G2Tot[sz] * gBlockDets[i][sz];

          //term 2
          MatrixXcd temp;
          for (int p = 0; p < rank; p++) {
            temp = gBlocks[i][sz];
            for (int t = 0; t < rank; t++)
              temp(p, t) = int1[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t) - ncore);
            twoEne[sz] -= 2. * temp.determinant();
          }
        }
      } 
    } // sz

    ene += ciParity[i] * ciCoeffs[i] * (twoEne[0] * gBlockDets[i][1] + gBlockDets[i][0] * twoEne[1]) / 2.;
  
  } // dets

  overlap *= overlap0;
  ene *= overlap0;
  ene += enuc * overlap;
  return pair<complex<double>, complex<double>>(overlap, ene);
}


// Hamiltonian matrix element < \sum _i phi_i | H | psi >
// multislater local energy
// leading cost: O(N_c + X N^2 M^2)
// returns both overlap and local energy
// TODO: use either array or pair, not both; make interface consistent with other local energy functions perhaps by changing all returns to <overlap, local energy> pairs
pair<complex<double>, complex<double>> calcHamiltonianElement1(matPair& phi0T, std::array<vector<std::array<VectorXi, 2>>, 2>& ciExcitations, vector<double>& ciParity, vector<double>& ciCoeffs, matPair& psi, double enuc, MatrixXd& h1, vector<MatrixXd>& chol, int nchol, int nact, int ncore)
{
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;
  std::array<int, 2> nelec{nalpha, nbeta};
  size_t ndets = ciCoeffs.size();

  complex<double> overlap(0., 0.);
  complex<double> ene(0., 0.);
  matArray theta, green, greenp;
  theta[0] = psi.first * (phi0T.first * psi.first).inverse();
  theta[1] = psi.second * (phi0T.second * psi.second).inverse();
  green[0] = (theta[0] * phi0T.first).transpose();
  green[1] = (theta[1] * phi0T.second).transpose();
  greenp[0] = green[0] - MatrixXcd::Identity(norbs, norbs);
  greenp[1] = green[1] - MatrixXcd::Identity(norbs, norbs);
  matArray greeno;
  greeno[0] = green[0].block(0, 0, nalpha, norbs);
  greeno[1] = green[1].block(0, 0, nbeta, norbs);
  //green[0] = green[0].block(0, 0, nalpha, norbs);
  //green[1] = green[1].block(0, 0, nbeta, norbs);

  // all quantities henceforth will be calculated in "units" of overlap0 = < phi_0 | psi >
  complex<double> overlap0 = (phi0T.first * psi.first).determinant() * (phi0T.second * psi.second).determinant();
 
  // ref contribution
  overlap += ciCoeffs[0];
  std::array<complex<double>, 2> hG;
  hG[0] = greeno[0].cwiseProduct(h1.block(0, 0, nalpha, norbs)).sum();
  hG[1] = greeno[1].cwiseProduct(h1.block(0, 0, nbeta, norbs)).sum();
  //hG[0] = green[0].cwiseProduct(h1).sum();
  //hG[1] = green[1].cwiseProduct(h1).sum();
  ene += ciCoeffs[0] * (hG[0] + hG[1]);
  
  // 1e intermediate
  matArray roth1;
  roth1[0] = (greeno[0] * h1) * greenp[0];
  roth1[1] = (greeno[1] * h1) * greenp[1];

  // G^{p}_{t} blocks
  vector<matArray> gBlocks;
  vector<std::array<complex<double>, 2>> gBlockDets;
  matArray empty;
  gBlocks.push_back(empty);
  std::array<complex<double>, 2> identity{1., 1.};
  gBlockDets.push_back(identity);
  
  // iterate over excitations
  for (int i = 1; i < ndets; i++) {
    matArray blocks;
    std::array<complex<double>, 2> oneEne, dets;
    for (int sz = 0; sz < 2; sz++) {
      int rank = ciExcitations[sz][i][0].size();
      if (rank == 0) {
        dets[sz] = 1.;
        oneEne[sz] = hG[sz];
      }
      else {
        blocks[sz] = MatrixXcd::Zero(rank, rank);
        for (int p = 0; p < rank; p++) 
          for (int t = 0; t < rank; t++) 
            blocks[sz](p, t) = green[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t));
        
        dets[sz] = blocks[sz].determinant();
        oneEne[sz] = hG[sz] * dets[sz];

        MatrixXcd temp;
        for (int p = 0; p < rank; p++) {
          temp = blocks[sz];
          for (int t = 0; t < rank; t++)
            temp(p, t) = roth1[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t));
          oneEne[sz] -= temp.determinant();
        }
      }
    }

    overlap += ciCoeffs[i] * ciParity[i] * dets[0] * dets[1];
    ene += ciCoeffs[i] * ciParity[i] * (oneEne[0] * dets[1] + dets[0] * oneEne[1]);
    gBlocks.push_back(blocks);
    gBlockDets.push_back(dets);
  }
  
  // 2e intermediates
  matArray int1, int2;
  int1[0] = 0. * greeno[0];
  int1[1] = 0. * greeno[0];
  int2[0] = 0. * greeno[0];
  int2[1] = 0. * greeno[0];
  std::array<complex<double>, 2> l2G2Tot;
  l2G2Tot[0] = complex<double>(0., 0.);
  l2G2Tot[1] = complex<double>(0., 0.);
  std::array<MatrixXcd, 3> bigInt;
  bigInt[0] = MatrixXcd::Zero(nelec[0] * nelec[0], norbs * norbs);
  bigInt[1] = MatrixXcd::Zero(nelec[1] * nelec[1], norbs * norbs);
  //bigInt[2] = MatrixXcd::Zero(nelec[0] * nelec[1], norbs * norbs);

  // iterate over cholesky
  for (int n = 0; n < chol.size(); n++) {
    std::array<complex<double>, 2> lG, l2G2;
    matArray exc;
    for (int sz = 0; sz < 2; sz++) {
      exc[sz].noalias() = chol[n].block(0, 0, nelec[sz], norbs) * theta[sz];
      lG[sz] = exc[sz].trace();
      l2G2[sz] = lG[sz] * lG[sz] - exc[sz].cwiseProduct(exc[sz].transpose()).sum();
      l2G2Tot[sz] += l2G2[sz];
      //int2[sz].noalias() = (greeno[sz] * chol[n]) * greenp[sz];
      int2[sz].setZero();
      int2[sz].block(0, 0, nelec[sz], norbs) = (chol[n].block(0, 0, nelec[sz], norbs) * theta[sz]).transpose() * theta[sz].transpose();
      int2[sz].noalias() -= greeno[sz] * chol[n]; 
      //MatrixXcd temp = int2[sz];
      //bigInt[sz].noalias() += kroneckerProduct(temp, temp);
      //int2[sz].noalias() = (greeno[sz] * chol[n].block(0, 0, norbs, norbsAct)) * greenp[sz].block(0, 0, norbsAct, norbs);
      int1[sz].noalias() += lG[sz] * int2[sz];
      int1[sz].noalias() -= (greeno[sz] * chol[n].block(0, 0, norbs, nelec[sz])) * int2[sz];
      for (int p = 0; p < nelec[sz]; p++)
        for (int t = 0; t < norbs; t++)
          bigInt[sz].block(p * nelec[sz], t * norbs, nelec[sz], norbs) += int2[sz](p, t) * int2[sz];
    }

    // ref contribution
    ene += ciCoeffs[0] * (l2G2[0] + l2G2[1] + 2. * lG[0] * lG[1]) / 2.;
    
    // iterate over excitations
    for (int i = 1; i < ndets; i++) {
      std::array<complex<double>, 2> oneEne, twoEne;
      for (int sz = 0; sz < 2; sz++) {
        int rank = ciExcitations[sz][i][0].size();
        if (rank == 0) {
          oneEne[sz] = lG[sz];
        }
        if (rank == 1) {
          oneEne[sz] = lG[sz] * gBlockDets[i][sz] - int2[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0));
        }
        else if (rank == 2) {
          oneEne[sz] = lG[sz] * gBlockDets[i][sz] 
                     - int2[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0)) * gBlocks[i][sz](1, 1)
                     + int2[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](1)) * gBlocks[i][sz](1, 0)
                     + int2[sz](ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](0)) * gBlocks[i][sz](0, 1)
                     - int2[sz](ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](1)) * gBlocks[i][sz](0, 0);
        }
        else {
          // oneEne
          {
            oneEne[sz] = lG[sz] * gBlockDets[i][sz];

            MatrixXcd temp;
            for (int p = 0; p < rank; p++) {
              temp = gBlocks[i][sz];
              for (int t = 0; t < rank; t++)
                temp(p, t) = int2[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t));
              oneEne[sz] -= temp.determinant();
            }
          }
          
          // twoEne
          {
            twoEne[sz] = complex<double>(0., 0.);
            // term 3
            if (rank > 1) {
              vector<int> range(rank);
              for (int p = 0; p < rank; p++) range[p] = p;
              for (int p = 0; p < rank; p++) {
                for (int q = p + 1; q < rank; q++) {
                  double parity_pq = ((p + q + 1)%2 == 0) ? 1. : -1.;
                  vector<int> pq = {p, q};
                  vector<int> diff0;
                  std::set_difference(range.begin(), range.end(), pq.begin(), pq.end(), std::inserter(diff0, diff0.begin()));
                  for (int t = 0; t < rank; t++) {
                    for (int u = t + 1; u < rank; u++) {
                      double parity_tu = ((t + u + 1)%2 == 0) ? 1. : -1.;
                      vector<int> tu = {t, u};
                      vector<int> diff1;
                      std::set_difference(range.begin(), range.end(), tu.begin(), tu.end(), std::inserter(diff1, diff1.begin()));
                      
                      complex<double> blockBlockDet;
                      if (rank < 3) blockBlockDet = 1.;
                      else {
                        MatrixXcd blockBlock = MatrixXcd::Zero(rank - 2, rank - 2);
                        for (int mup = 0; mup < rank - 2; mup++) 
                          for (int nup = 0; nup < rank - 2; nup++) 
                            blockBlock(mup, nup) = gBlocks[i][sz](diff0[mup], diff1[nup]);
                        blockBlockDet = blockBlock.determinant();
                      }

                      twoEne[sz] += 2. * parity_pq * parity_tu * blockBlockDet * (  int2[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t)) 
                                                                                  * int2[sz](ciExcitations[sz][i][0](q), ciExcitations[sz][i][1](u)) 
                                                                                  - int2[sz](ciExcitations[sz][i][0](q), ciExcitations[sz][i][1](t)) 
                                                                                  * int2[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](u))  );
                    }
                  }
                }
              }
            }
          }
        } 
      } // sz

      ene += ciParity[i] * ciCoeffs[i] * (twoEne[0] * gBlockDets[i][1] + 2. * oneEne[0] * oneEne[1] + gBlockDets[i][0] * twoEne[1]) / 2.;
    
    } // dets
  } // chol
    
  // iterate over excitations
  for (int i = 1; i < ndets; i++) {
    std::array<complex<double>, 2> oneEne, twoEne;
    for (int sz = 0; sz < 2; sz++) {
      int rank = ciExcitations[sz][i][0].size();
      if (rank == 0) {
        twoEne[sz] = l2G2Tot[sz];
      }
      else if (rank == 1) {
        twoEne[sz] = l2G2Tot[sz] * gBlockDets[i][sz] - 2. * int1[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0));
      }
      else if (rank == 2) {
        twoEne[sz] = l2G2Tot[sz] * gBlockDets[i][sz]
                   + 2. * (- int1[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0)) * gBlocks[i][sz](1, 1)
                           + int1[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](1)) * gBlocks[i][sz](1, 0)
                           + int1[sz](ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](0)) * gBlocks[i][sz](0, 1)
                           - int1[sz](ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](1)) * gBlocks[i][sz](0, 0)  )
                   + 2. * (  bigInt[sz](ciExcitations[sz][i][0](0) * nelec[sz] + ciExcitations[sz][i][0](1),
                                        ciExcitations[sz][i][1](0) * norbs + ciExcitations[sz][i][1](1)) 
                           - bigInt[sz](ciExcitations[sz][i][0](0) * nelec[sz] + ciExcitations[sz][i][0](1),
                                        ciExcitations[sz][i][1](1) * norbs + ciExcitations[sz][i][1](0))  );
      }
      else {
        // twoEne
        {
          // term 1
          twoEne[sz] = l2G2Tot[sz] * gBlockDets[i][sz];

          //term 2
          MatrixXcd temp;
          for (int p = 0; p < rank; p++) {
            temp = gBlocks[i][sz];
            for (int t = 0; t < rank; t++)
              temp(p, t) = int1[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t));
            twoEne[sz] -= 2. * temp.determinant();
          }
        }
      } 
    } // sz

    ene += ciParity[i] * ciCoeffs[i] * (twoEne[0] * gBlockDets[i][1] + gBlockDets[i][0] * twoEne[1]) / 2.;
  
  } // dets

  overlap *= overlap0;
  ene *= overlap0;
  ene += enuc * overlap;
  return pair<complex<double>, complex<double>>(overlap, ene);
}


// Hamiltonian matrix element < \sum _i phi_i | H | psi >
// multislater local energy
// leading cost: O(X N_c + X N^2 M)
// returns both overlap and local energy
// TODO: use either array or pair, not both; make interface consistent with other local energy functions perhaps by changing all returns to <overlap, local energy> pairs
pair<complex<double>, complex<double>> calcHamiltonianElement_sRI(matPair& phi0T, std::array<vector<std::array<VectorXi, 2>>, 2>& ciExcitations, vector<double>& ciParity, vector<double>& ciCoeffs, matPair& psi, double enuc, MatrixXd& h1, vector<MatrixXd>& chol, double ene0)
{
  
  uniform_real_distribution<double> normal(-1., 1.);
  vector<MatrixXd> richolvec;
  MatrixXd richol = 0. * chol[0];

  for (int j = 0; j < 10; j++) {
    richol.setZero();
    for (int i = 0; i < chol.size(); i++) {
      double ran = normal(generator);
      richol += ran/abs(ran) * chol[i];     
    }
    richol /= 10.;
    richolvec.push_back(richol);
  }

  matPair phi0;
  phi0.first = phi0T.first.adjoint();
  phi0.second = phi0T.second.adjoint();

  auto overlapHam1 = calcHamiltonianElement(phi0T, ciExcitations, ciParity, ciCoeffs, psi, enuc, h1, richolvec); 
  auto overlapHam2 = calcHamiltonianElement(phi0T, ciExcitations, ciParity, ciCoeffs, phi0, enuc, h1, richolvec); 
 
  pair<complex<double>, complex<double>> overlapHam;
  overlapHam.first = overlapHam1.first;
  overlapHam.second = overlapHam1.first * ene0 + overlapHam1.second - overlapHam2.second * overlapHam1.first / overlapHam2.first;

  return overlapHam;
  //return calcHamiltonianElement(phi0T, ciExcitations, ciParity, ciCoeffs, psi, enuc, h1, chol);
}
