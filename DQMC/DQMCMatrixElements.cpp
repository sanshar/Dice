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

// Hamiltonian matrix element < d GHF/cxi | H | GHF > / < GHF | UHF >
// rotates cholesky vectors
complex<double> calcGradient(MatrixXcd& At, MatrixXcd& B, double enuc, MatrixXd& h1, vector<MatrixXd>& chol, MatrixXcd& Grad) 
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
