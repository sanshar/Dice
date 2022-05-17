#include "Hamiltonian.h"
#include "integral.h"
#include "SHCIshm.h"

using namespace std;
using namespace Eigen;

// constructor
Hamiltonian::Hamiltonian(string fname, bool psocQ, std::string pintType) 
{
  intType = pintType;
  socQ = psocQ;
  if (socQ) {
    readDQMCIntegralsSOC(fname, norbs, nelec, ecore, h1soc, h1socMod, chol);
    nalpha = 0;
    nbeta = 0;
    nchol = chol.size();
    ncholEne = chol.size();
  }
  else if (intType == "r") {
    readDQMCIntegralsRG(fname, norbs, nalpha, nbeta, ecore, h1, h1Mod, chol);
    nelec = nalpha + nbeta;
    nchol = chol.size();
    ncholEne = chol.size();
  }
  else if (intType == "u") {
    readDQMCIntegralsU(fname, norbs, nalpha, nbeta, ecore, h1u, h1uMod, cholu);
    nelec = nalpha + nbeta;
    nchol = cholu.size();
    ncholEne = cholu.size();
  }
  else if (intType == "g") {
    readDQMCIntegralsRG(fname, norbs, nalpha, nbeta, ecore, h1, h1Mod, chol, true);
    nelec = nalpha + nbeta;
    nalpha = 0;
    nbeta = 0;
    nchol = chol.size();
    ncholEne = chol.size();
  }
  floattenCholesky();
  rotFlag = false;
};


void Hamiltonian::setNcholEne(int pnchol) 
{
  ncholEne = pnchol;
};


// rotate cholesky ri or gi
void Hamiltonian::rotateCholesky(Eigen::MatrixXd& phiT, std::vector<Eigen::Map<Eigen::MatrixXd>>& rotChol, bool deleteOriginalChol) 
{
  double* rotCholSHM;
  double* rotChol0;
  size_t rotSize = phiT.rows() * norbs;
  size_t size = chol.size() * rotSize;

  if (commrank == 0) {
    rotChol0 = new double[size];
    for (int i = 0; i < chol.size(); i++) {
      MatrixXd rot = phiT * chol[i];
      for (int nu = 0; nu < rot.cols(); nu++)
        for (int mu = 0; mu < rot.rows(); mu++)
          rotChol0[i * rotSize + nu * rot.rows() + mu] = rot(mu, nu);
    }
  }
 
  MPI_Barrier(MPI_COMM_WORLD);
  SHMVecFromVecs(rotChol0, size, rotCholSHM, rotCholSHMName, rotCholSegment, rotCholRegion); 
  MPI_Barrier(MPI_COMM_WORLD);
  
  // create eigen matrix maps to shared memory
  for (size_t n = 0; n < chol.size(); n++) {
    Eigen::Map<MatrixXd> rotCholMat(static_cast<double*>(rotCholSHM) + n * rotSize, phiT.rows(), norbs);
    rotChol.push_back(rotCholMat);
  }
  
  if (commrank == 0) delete [] rotChol0; 
  if (deleteOriginalChol) rotFlag = true;
};


// rotate cholesky ri or ui
void Hamiltonian::rotateCholesky(std::array<Eigen::MatrixXd, 2>& phiT, std::array<std::vector<Eigen::Map<Eigen::MatrixXd>>, 2>& rotChol, bool deleteOriginalChol) 
{
  double* rotCholSHM;
  double* rotChol0;  // this zero referes to commarnk 0
  size_t rotSize0 = phiT[0].rows() * norbs; // this zero referes to spin
  size_t rotSize1 = phiT[1].rows() * norbs;
  size_t size = nchol * (rotSize0 + rotSize1);

  if (commrank == 0) {
    rotChol0 = new double[size];
    for (size_t i = 0; i < nchol; i++) {
      MatrixXd rot0;
      if (intType == "r") rot0 = phiT[0] * chol[i];
      else if (intType == "u") rot0 = phiT[0] * cholu[i][0];
      for (size_t nu = 0; nu < rot0.cols(); nu++)
        for (size_t mu = 0; mu < rot0.rows(); mu++)
          rotChol0[i * rotSize0 + nu * rot0.rows() + mu] = rot0(mu, nu);
      MatrixXd rot1;
      if (intType == "r") rot1 = phiT[1] * chol[i];
      else if (intType == "u") rot1 = phiT[1] * cholu[i][1];
      for (size_t nu = 0; nu < rot1.cols(); nu++)
        for (size_t mu = 0; mu < rot1.rows(); mu++)
          rotChol0[nchol * rotSize0 + i * rotSize1 + nu * rot1.rows() + mu] = rot1(mu, nu);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  SHMVecFromVecs(rotChol0, size, rotCholSHM, rotCholSHMName, rotCholSegment, rotCholRegion); 
  MPI_Barrier(MPI_COMM_WORLD);
  
  // create eigen matrix maps to shared memory
  for (size_t n = 0; n < nchol; n++) {
    Eigen::Map<MatrixXd> rotCholMat0(static_cast<double*>(rotCholSHM) + n * rotSize0, phiT[0].rows(), norbs);
    rotChol[0].push_back(rotCholMat0);
    Eigen::Map<MatrixXd> rotCholMat1(static_cast<double*>(rotCholSHM) + nchol * rotSize0 +  n * rotSize1, phiT[1].rows(), norbs);
    rotChol[1].push_back(rotCholMat1);
  }
  
  if (commrank == 0) delete [] rotChol0; 
  if (deleteOriginalChol) rotFlag = true;
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
void Hamiltonian::floattenCholesky()
{
  size_t triSize = (norbs * (norbs + 1)) / 2;
  size_t size;
  if (intType == "r" || intType == "g") size = nchol * triSize;
  else if (intType == "u") size = 2 * nchol * triSize;
  float* floatChol0;
  if (commrank == 0) {
    floatChol0 = new float[size];
    size_t counter = 0;
    for (int n = 0; n < nchol; n++) {
      for (int i = 0; i < norbs; i++) {
        for (int j = 0; j <= i; j++) {
          if (intType == "r" || intType == "g") floatChol0[counter] = float(chol[n](i, j));
          else if (intType == "u") {
            floatChol0[counter] = float(cholu[n][0](i, j));
            floatChol0[size/2 + counter] = float(cholu[n][1](i, j));
          }
          counter++;
        }
      }
    }
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  SHMVecFromVecs(floatChol0, size, floatChol, floatCholSHMName, floatCholSegment, floatCholRegion); 
  MPI_Barrier(MPI_COMM_WORLD);
    
  if (intType == "r" || intType == "g") {
    Eigen::Map<Eigen::MatrixXf> floatCholMatMap(static_cast<float*>(floatChol), triSize, nchol);
    floatCholMat.push_back(floatCholMatMap);
  }
  else if (intType == "u") {
    Eigen::Map<Eigen::MatrixXf> floatCholMatUp(static_cast<float*>(floatChol), triSize, nchol);
    floatCholMat.push_back(floatCholMatUp);
    Eigen::Map<Eigen::MatrixXf> floatCholMatDn(static_cast<float*>(floatChol) + size/2, triSize, nchol);
    floatCholMat.push_back(floatCholMatDn);
  }
  
  if (commrank == 0) delete [] floatChol0; 

};
