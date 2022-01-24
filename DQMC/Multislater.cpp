#include <algorithm>
#include <iostream>
#include "global.h"
#include "input.h"
#include "Multislater.h"

using namespace std;
using namespace Eigen;

using matPair = std::array<MatrixXcd, 2>;

Multislater::Multislater(std::string fname, int pnact, int pncore, bool prightQ) 
{
  if (fname == "dets") readDeterminants(fname, refDet, ciExcitations, ciParity, ciCoeffs);
  else readDeterminantsBinary(fname, refDet, ciExcitations, ciParity, ciCoeffs);
  nact = pnact;
  ncore = pncore;
  rightQ = prightQ;

  if (rightQ) {
    double sum = 0.;
    for (int i = 0; i < ciCoeffs.size(); i++) {
      sum += abs(ciCoeffs[i]);
      cumulativeCoeffs.push_back(sum);
    }
    uniform = uniform_real_distribution<double> (0., 1.);
  }
};


void Multislater::getSample(std::array<Eigen::MatrixXcd, 2>& sampleDet)
{
  double randNumber = uniform(generator) * cumulativeCoeffs[cumulativeCoeffs.size() - 1];
  auto sampleIt = lower_bound(cumulativeCoeffs.begin(), cumulativeCoeffs.end(), randNumber) - cumulativeCoeffs.begin();
  
  matPair phi0;
  int norbs = nact + ncore;
  int nalpha = refDet[0].size();
  int nbeta = refDet[1].size();
  phi0[0] = MatrixXcd::Zero(norbs, nalpha);
  phi0[1] = MatrixXcd::Zero(norbs, nbeta);
  for (int i = 0; i < nalpha; i++) phi0[0](refDet[0][i], i) = 1.;
  for (int i = 0; i < nbeta; i++) phi0[1](refDet[1][i], i) = 1.;

  for (int sz = 0; sz < 2; sz++) {
    for (int mu = 0; mu < ciExcitations[sz][sampleIt][0].size(); mu++) {
      int p = ciExcitations[sz][sampleIt][0](mu);
      p = lower_bound(refDet[sz].begin(), refDet[sz].end(), p) - refDet[sz].begin();
      int t = ciExcitations[sz][sampleIt][1](mu);
      phi0[sz](refDet[sz][p], p) = 0.;
      phi0[sz](t, p) = 1.;
    }
  }
  phi0[0].col(0) *= abs(ciCoeffs[sampleIt])/ciCoeffs[sampleIt];
  sampleDet = phi0;
};


std::complex<double> Multislater::overlap(std::array<Eigen::MatrixXcd, 2>& psi)
{
  int norbs = psi[0].rows();
  int nalpha = refDet[0].size();
  int nbeta = refDet[1].size();
  std::array<int, 2> nelec{nalpha, nbeta};
  size_t ndets = ciCoeffs.size();

  matPair phi0T;
  phi0T[0] = MatrixXcd::Zero(nalpha, norbs);
  phi0T[1] = MatrixXcd::Zero(nbeta, norbs);
  for (int i = 0; i < nalpha; i++) phi0T[0](i, refDet[0][i]) = 1.;
  for (int i = 0; i < nbeta; i++) phi0T[1](i, refDet[1][i]) = 1.;

  complex<double> overlap(0., 0.);
  matPair theta, green, greenp;
  theta[0] = psi[0] * (phi0T[0] * psi[0]).inverse();
  theta[1] = psi[1] * (phi0T[1] * psi[1]).inverse();
  green[0] = (theta[0] * phi0T[0]).transpose();
  green[1] = (theta[1] * phi0T[1]).transpose();
  greenp[0] = green[0] - MatrixXcd::Identity(norbs, norbs);
  greenp[1] = green[1] - MatrixXcd::Identity(norbs, norbs);
  matPair greeno;
  greeno[0] = green[0].block(0, 0, nalpha, norbs);
  greeno[1] = green[1].block(0, 0, nbeta, norbs);

  // all quantities henceforth will be calculated in "units" of overlap0 = < phi_0 | psi >
  complex<double> overlap0 = (phi0T[0] * psi[0]).determinant() * (phi0T[1] * psi[1]).determinant();
 
  // ref contribution
  overlap += ciCoeffs[0];

  // iterate over excitations
  for (int i = 1; i < ndets; i++) {
    std::array<complex<double>, 2> dets;
    for (int sz = 0; sz < 2; sz++) {
      int rank = ciExcitations[sz][i][0].size();
      if (rank == 0) {
        dets[sz] = 1.;
      }
      else if (rank == 1) {
        dets[sz] = green[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0));
      }
      else if (rank == 2) {
        dets[sz] = green[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0)) * green[sz](ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](1)) 
                 - green[sz](ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](0)) * green[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](1));
      }
      else if (rank == 3) {
        Matrix3cd temp = Matrix3cd::Zero(3, 3);
        for (int p = 0; p < rank; p++) 
          for (int t = 0; t < rank; t++) 
            temp(p, t) = green[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t));
        dets[sz] = temp.determinant();
      }
      else if (rank == 4) {
        Matrix4cd temp = Matrix4cd::Zero(4, 4);
        for (int p = 0; p < rank; p++) 
          for (int t = 0; t < rank; t++) 
            temp(p, t) = green[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t));
        dets[sz] = temp.determinant();
      }
      else {
        MatrixXcd temp = MatrixXcd::Zero(rank, rank);
        for (int p = 0; p < rank; p++) 
          for (int t = 0; t < rank; t++) 
            temp(p, t) = green[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t));
        dets[sz] = temp.determinant();
      }
    }

    overlap += ciCoeffs[i] * ciParity[i] * dets[0] * dets[1];
  }
  overlap *= overlap0;
  return overlap;
};


std::complex<double> Multislater::overlap(Eigen::MatrixXcd& psi)
{
  matPair psi2;
  psi2[0] = psi;
  psi2[1] = psi;
  return this->overlap(psi2);
};

void Multislater::forceBias(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham, Eigen::VectorXcd& fb)
{
  int norbs = ham.norbs;
  int nalpha = ham.nalpha;
  int nbeta = ham.nbeta;
  int nchol = ham.chol.size();
  std::array<int, 2> nelec{nalpha, nbeta};
  size_t ndets = ciCoeffs.size();
  fb = VectorXcd::Zero(ham.chol.size());

  //matPair phi0T;
  //phi0T[0] = MatrixXcd::Zero(nalpha, norbs);
  //phi0T[1] = MatrixXcd::Zero(nbeta, norbs);
  //for (int i = 0; i < nalpha; i++) phi0T[0](i, refDet[0][i]) = 1.;
  //for (int i = 0; i < nbeta; i++) phi0T[1](i, refDet[1][i]) = 1.;

  matPair theta, green, greenp, greeno;
  //theta[0] = psi[0] * (phi0T[0] * psi[0]).inverse();
  //theta[1] = psi[1] * (phi0T[1] * psi[1]).inverse();
  //green[0] = (theta[0] * phi0T[0]).transpose();
  //green[1] = (theta[1] * phi0T[1]).transpose();
  //greenp[0] = green[0] - MatrixXcd::Identity(norbs, norbs);
  //greenp[1] = green[1] - MatrixXcd::Identity(norbs, norbs);
  theta[0] = psi[0] * (psi[0].block(0, 0, nalpha, nalpha)).inverse();
  theta[1] = psi[1] * (psi[1].block(0, 0, nbeta, nbeta)).inverse();
  green[0] = MatrixXcd::Zero(norbs, norbs);
  green[1] = MatrixXcd::Zero(norbs, norbs);
  green[0].block(0, 0, nalpha, norbs) = theta[0].transpose();
  green[1].block(0, 0, nbeta, norbs) = theta[1].transpose();
  greenp[0] = green[0] - MatrixXcd::Identity(norbs, norbs);
  greenp[1] = green[1] - MatrixXcd::Identity(norbs, norbs);
  
  
  greeno[0] = green[0].block(0, 0, nalpha, norbs);
  greeno[1] = green[1].block(0, 0, nbeta, norbs);

  // most quantities henceforth will be calculated in "units" of overlap0 = < phi_0 | psi >
  complex<double> overlap0 = (psi[0].block(0, 0, nalpha, nalpha)).determinant() * (psi[1].block(0, 0, nbeta, nbeta)).determinant();
  complex<double> overlap(0., 0.);
  overlap += ciCoeffs[0];
  matPair intermediate, greenMulti;
  intermediate[0] = MatrixXcd::Zero(norbs, nalpha);
  intermediate[1] = MatrixXcd::Zero(norbs, nbeta);

  // iterate over excitations
  for (int i = 1; i < ndets; i++) {
    matPair temp;
    std::array<complex<double>, 2> dets;
    for (int sz = 0; sz < 2; sz++) {
      int rank = ciExcitations[sz][i][0].size();
      if (rank == 0) {
        dets[sz] = 1.;
      }
      else if (rank == 1) {
        dets[sz] = green[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0));
        temp[sz] = MatrixXcd::Zero(1, 1);
        temp[sz](0, 0) = -1.;
      }
      else if (rank == 2) {
        Matrix2cd cofactors = Matrix2cd::Zero(2, 2);
        cofactors(0, 0) = green[sz](ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](1));
        cofactors(1, 1) = green[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0));
        cofactors(0, 1) = -green[sz](ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](0));
        cofactors(1, 0) = -green[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](1));
        dets[sz] = cofactors.determinant();
        temp[sz] = -cofactors;
      }
      else if (rank == 3) {
        Matrix3cd cofactors = Matrix3cd::Zero(3, 3);
        Matrix2cd minorMat = Matrix2cd::Zero(2, 2);
        for (int p = 0; p < rank; p++) {
          for (int t = 0; t < rank; t++) {
            minorMat.setZero();
            for (int q = 0; q < rank - 1; q++) {
              int q1 = q < p ? q : q + 1;
              for (int u = 0; u < rank - 1; u++) { 
                int u1 = u < t ? u : u + 1;
                minorMat(q, u) = green[sz](ciExcitations[sz][i][0](q1), ciExcitations[sz][i][1](u1));
              }
            }
            double parity_pt = ((p + t)%2 == 0) ? 1. : -1.;
            cofactors(p, t) = parity_pt * minorMat.determinant();
            dets[sz] += green[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t)) * cofactors(p, t);
          }
        }
        temp[sz] = -cofactors;
      }
      else if (rank == 4) {
        Matrix4cd cofactors = Matrix4cd::Zero(4, 4);
        Matrix3cd minorMat = Matrix3cd::Zero(3, 3);
        for (int p = 0; p < rank; p++) {
          for (int t = 0; t < rank; t++) {
            minorMat.setZero();
            for (int q = 0; q < rank - 1; q++) {
              int q1 = q < p ? q : q + 1;
              for (int u = 0; u < rank - 1; u++) { 
                int u1 = u < t ? u : u + 1;
                minorMat(q, u) = green[sz](ciExcitations[sz][i][0](q1), ciExcitations[sz][i][1](u1));
              }
            }
            double parity_pt = ((p + t)%2 == 0) ? 1. : -1.;
            cofactors(p, t) = parity_pt * minorMat.determinant();
            dets[sz] += green[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t)) * cofactors(p, t);
          }
        }
        temp[sz] = -cofactors;
      }
      else {
        MatrixXcd cofactors = MatrixXcd::Zero(rank, rank);
        MatrixXcd minorMat = MatrixXcd::Zero(rank-1, rank-1);
        for (int p = 0; p < rank; p++) {
          for (int t = 0; t < rank; t++) {
            minorMat.setZero();
            for (int q = 0; q < rank - 1; q++) {
              int q1 = q < p ? q : q + 1;
              for (int u = 0; u < rank - 1; u++) { 
                int u1 = u < t ? u : u + 1;
                minorMat(q, u) = green[sz](ciExcitations[sz][i][0](q1), ciExcitations[sz][i][1](u1));
              }
            }
            double parity_pt = ((p + t)%2 == 0) ? 1. : -1.;
            cofactors(p, t) = parity_pt * minorMat.determinant();
            dets[sz] += green[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t)) * cofactors(p, t);
          }
        }
        temp[sz] = cofactors;
      }
    }

    overlap += ciCoeffs[i] * ciParity[i] * dets[0] * dets[1];
    for (int sz = 0; sz < 2; sz++) {
      int rank = ciExcitations[sz][i][0].size();
      if (rank > 0) {
        for (int p = 0; p < rank; p++) 
          for (int t = 0; t < rank; t++)
            intermediate[sz](ciExcitations[sz][i][1](t), ciExcitations[sz][i][0](p)) += ciCoeffs[i] * ciParity[i] * temp[sz](p, t) * dets[1 - sz];
      }
    }
  }
  
  greenMulti[0] = greenp[0] * intermediate[0] * greeno[0] + overlap * green[0];
  greenMulti[1] = greenp[1] * intermediate[1] * greeno[1] + overlap * green[1];
  overlap *= overlap0;
  greenMulti[0] *= (overlap0 / overlap);
  greenMulti[1] *= (overlap0 / overlap);
  //greenMulti[0] = greenMulti[0].block(0, 0, nalpha, norbs);
  //greenMulti[1] = greenMulti[1].block(0, 0, nbeta, norbs);
  //cout << "greenmulti[0]\n" << greenMulti[0] << endl << endl;
  //cout << "greenmulti[1]\n" << greenMulti[1] << endl << endl;
  greenMulti[0] = greenMulti[0].transpose().eval();
  greenMulti[1] = greenMulti[1].transpose().eval();
  for (int i = 0; i < ham.chol.size(); i++) 
    fb(i) = (greenMulti[0].block(0, 0, norbs, nact)).cwiseProduct(ham.chol[i].block(0, 0, norbs, nact)).sum() + (greenMulti[1].block(0, 0, norbs, nact)).cwiseProduct(ham.chol[i].block(0, 0, norbs, nact)).sum();
  //for (int i = 0; i < ham.chol.size(); i++) 
  //  fb(i) = (greenMulti[0].block(0, 0, nact, norbs)).cwiseProduct(ham.chol[i].block(0, 0, nact, norbs)).sum() + (greenMulti[1].block(0, 0, nact, norbs)).cwiseProduct(ham.chol[i].block(0, 0, nact, norbs)).sum();
    //fb(i) = (greenMulti[0]).cwiseProduct(ham.chol[i]).sum() + (greenMulti[1]).cwiseProduct(ham.chol[i]).sum();
};

void Multislater::oneRDM(std::array<Eigen::MatrixXcd, 2>& psi, Eigen::MatrixXcd& rdmSample) 
{ 
  int norbs = psi[0].rows();
  int nalpha = psi[0].cols();
  int nbeta = psi[1].cols();
  std::array<int, 2> nelec{nalpha, nbeta};
  size_t ndets = ciCoeffs.size();
  rdmSample = MatrixXcd::Zero(norbs, norbs);

  //matPair phi0T;
  //phi0T[0] = MatrixXcd::Zero(nalpha, norbs);
  //phi0T[1] = MatrixXcd::Zero(nbeta, norbs);
  //for (int i = 0; i < nalpha; i++) phi0T[0](i, refDet[0][i]) = 1.;
  //for (int i = 0; i < nbeta; i++) phi0T[1](i, refDet[1][i]) = 1.;

  matPair theta, green, greenp, greeno;
  //theta[0] = psi[0] * (phi0T[0] * psi[0]).inverse();
  //theta[1] = psi[1] * (phi0T[1] * psi[1]).inverse();
  //green[0] = (theta[0] * phi0T[0]).transpose();
  //green[1] = (theta[1] * phi0T[1]).transpose();
  //greenp[0] = green[0] - MatrixXcd::Identity(norbs, norbs);
  //greenp[1] = green[1] - MatrixXcd::Identity(norbs, norbs);
  theta[0] = psi[0] * (psi[0].block(0, 0, nalpha, nalpha)).inverse();
  theta[1] = psi[1] * (psi[1].block(0, 0, nbeta, nbeta)).inverse();
  green[0] = MatrixXcd::Zero(norbs, norbs);
  green[1] = MatrixXcd::Zero(norbs, norbs);
  green[0].block(0, 0, nalpha, norbs) = theta[0].transpose();
  green[1].block(0, 0, nbeta, norbs) = theta[1].transpose();
  greenp[0] = green[0] - MatrixXcd::Identity(norbs, norbs);
  greenp[1] = green[1] - MatrixXcd::Identity(norbs, norbs);
  
  
  greeno[0] = green[0].block(0, 0, nalpha, norbs);
  greeno[1] = green[1].block(0, 0, nbeta, norbs);

  // most quantities henceforth will be calculated in "units" of overlap0 = < phi_0 | psi >
  complex<double> overlap0 = (psi[0].block(0, 0, nalpha, nalpha)).determinant() * (psi[1].block(0, 0, nbeta, nbeta)).determinant();
  complex<double> overlap(0., 0.);
  overlap += ciCoeffs[0];
  matPair intermediate, greenMulti;
  intermediate[0] = MatrixXcd::Zero(norbs, nalpha);
  intermediate[1] = MatrixXcd::Zero(norbs, nbeta);

  // iterate over excitations
  for (int i = 1; i < ndets; i++) {
    matPair temp;
    std::array<complex<double>, 2> dets;
    for (int sz = 0; sz < 2; sz++) {
      int rank = ciExcitations[sz][i][0].size();
      if (rank == 0) {
        dets[sz] = 1.;
      }
      else if (rank == 1) {
        dets[sz] = green[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0));
        temp[sz] = MatrixXcd::Zero(1, 1);
        temp[sz](0, 0) = -1.;
      }
      else if (rank == 2) {
        Matrix2cd cofactors = Matrix2cd::Zero(2, 2);
        cofactors(0, 0) = green[sz](ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](1));
        cofactors(1, 1) = green[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0));
        cofactors(0, 1) = -green[sz](ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](0));
        cofactors(1, 0) = -green[sz](ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](1));
        dets[sz] = cofactors.determinant();
        temp[sz] = -cofactors;
      }
      else if (rank == 3) {
        Matrix3cd cofactors = Matrix3cd::Zero(3, 3);
        Matrix2cd minorMat = Matrix2cd::Zero(2, 2);
        for (int p = 0; p < rank; p++) {
          for (int t = 0; t < rank; t++) {
            minorMat.setZero();
            for (int q = 0; q < rank - 1; q++) {
              int q1 = q < p ? q : q + 1;
              for (int u = 0; u < rank - 1; u++) { 
                int u1 = u < t ? u : u + 1;
                minorMat(q, u) = green[sz](ciExcitations[sz][i][0](q1), ciExcitations[sz][i][1](u1));
              }
            }
            double parity_pt = ((p + t)%2 == 0) ? 1. : -1.;
            cofactors(p, t) = parity_pt * minorMat.determinant();
            dets[sz] += green[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t)) * cofactors(p, t);
          }
        }
        temp[sz] = -cofactors;
      }
      else if (rank == 4) {
        Matrix4cd cofactors = Matrix4cd::Zero(4, 4);
        Matrix3cd minorMat = Matrix3cd::Zero(3, 3);
        for (int p = 0; p < rank; p++) {
          for (int t = 0; t < rank; t++) {
            minorMat.setZero();
            for (int q = 0; q < rank - 1; q++) {
              int q1 = q < p ? q : q + 1;
              for (int u = 0; u < rank - 1; u++) { 
                int u1 = u < t ? u : u + 1;
                minorMat(q, u) = green[sz](ciExcitations[sz][i][0](q1), ciExcitations[sz][i][1](u1));
              }
            }
            double parity_pt = ((p + t)%2 == 0) ? 1. : -1.;
            cofactors(p, t) = parity_pt * minorMat.determinant();
            dets[sz] += green[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t)) * cofactors(p, t);
          }
        }
        temp[sz] = -cofactors;
      }
      else {
        MatrixXcd cofactors = MatrixXcd::Zero(rank, rank);
        MatrixXcd minorMat = MatrixXcd::Zero(rank-1, rank-1);
        for (int p = 0; p < rank; p++) {
          for (int t = 0; t < rank; t++) {
            minorMat.setZero();
            for (int q = 0; q < rank - 1; q++) {
              int q1 = q < p ? q : q + 1;
              for (int u = 0; u < rank - 1; u++) { 
                int u1 = u < t ? u : u + 1;
                minorMat(q, u) = green[sz](ciExcitations[sz][i][0](q1), ciExcitations[sz][i][1](u1));
              }
            }
            double parity_pt = ((p + t)%2 == 0) ? 1. : -1.;
            cofactors(p, t) = parity_pt * minorMat.determinant();
            dets[sz] += green[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t)) * cofactors(p, t);
          }
        }
        temp[sz] = cofactors;
      }
    }

    overlap += ciCoeffs[i] * ciParity[i] * dets[0] * dets[1];
    for (int sz = 0; sz < 2; sz++) {
      int rank = ciExcitations[sz][i][0].size();
      if (rank > 0) {
        for (int p = 0; p < rank; p++) 
          for (int t = 0; t < rank; t++)
            intermediate[sz](ciExcitations[sz][i][1](t), ciExcitations[sz][i][0](p)) += ciCoeffs[i] * ciParity[i] * temp[sz](p, t) * dets[1 - sz];
      }
    }
  }
  
  greenMulti[0] = greenp[0] * intermediate[0] * greeno[0] + overlap * green[0];
  greenMulti[1] = greenp[1] * intermediate[1] * greeno[1] + overlap * green[1];
  overlap *= overlap0;
  greenMulti[0] *= (overlap0 / overlap);
  greenMulti[1] *= (overlap0 / overlap);
  rdmSample = greenMulti[0] + greenMulti[1];
};

//void Multislater::forceBias(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham, Eigen::VectorXcd& fb)
//{
//  int norbs = ham.norbs;
//  int nalpha = ham.nalpha;
//  int nbeta = ham.nbeta;
//  int nchol = ham.chol.size();
//  std::array<int, 2> nelec{nalpha, nbeta};
//  size_t ndets = ciCoeffs.size();
//  fb = VectorXcd::Zero(ham.chol.size());
//
//  matPair phi0T;
//  phi0T[0] = MatrixXcd::Zero(nalpha, norbs);
//  phi0T[1] = MatrixXcd::Zero(nbeta, norbs);
//  for (int i = 0; i < nalpha; i++) phi0T[0](i, refDet[0][i]) = 1.;
//  for (int i = 0; i < nbeta; i++) phi0T[1](i, refDet[1][i]) = 1.;
//
//  complex<double> overlap(0., 0.);
//  complex<double> ene(0., 0.);
//  matPair theta, green, greenp;
//  theta[0] = psi[0] * (phi0T[0] * psi[0]).inverse();
//  theta[1] = psi[1] * (phi0T[1] * psi[1]).inverse();
//  green[0] = (theta[0] * phi0T[0]).transpose();
//  green[1] = (theta[1] * phi0T[1]).transpose();
//  greenp[0] = green[0] - MatrixXcd::Identity(norbs, norbs);
//  greenp[1] = green[1] - MatrixXcd::Identity(norbs, norbs);
//  matPair greeno;
//  greeno[0] = green[0].block(0, 0, nalpha, norbs);
//  greeno[1] = green[1].block(0, 0, nbeta, norbs);
//
//  // all quantities henceforth will be calculated in "units" of overlap0 = < phi_0 | psi >
//  complex<double> overlap0 = (phi0T[0] * psi[0]).determinant() * (phi0T[1] * psi[1]).determinant();
// 
//  // iterate over excitations
//  for (int gamma = 0; gamma < ham.chol.size(); gamma++) {
//    std::array<complex<double>, 2> hG;
//    hG[0] = greeno[0].cwiseProduct(ham.chol[gamma].block(0, 0, nalpha, norbs)).sum();
//    hG[1] = greeno[1].cwiseProduct(ham.chol[gamma].block(0, 0, nbeta, norbs)).sum();
//    complex<double> ene(0., 0.), overlap(0., 0.);
//    // ref contribution
//    overlap += ciCoeffs[0];
//    ene += ciCoeffs[0] * (hG[0] + hG[1]);
//    
//    // 1e intermediate
//    matPair roth1;
//    roth1[0] = (greeno[0] * ham.chol[gamma]) * greenp[0];
//    roth1[1] = (greeno[1] * ham.chol[gamma]) * greenp[1];
//    // G^{p}_{t} blocks
//    vector<matPair> gBlocks;
//    vector<std::array<complex<double>, 2>> gBlockDets;
//    matPair empty;
//    gBlocks.push_back(empty);
//    std::array<complex<double>, 2> identity{1., 1.};
//    gBlockDets.push_back(identity);
//      
//    // iterate over excitations
//    for (int i = 1; i < ndets; i++) {
//      matPair blocks;
//      std::array<complex<double>, 2> oneEne, dets;
//      for (int sz = 0; sz < 2; sz++) {
//        int rank = ciExcitations[sz][i][0].size();
//        if (rank == 0) {
//          dets[sz] = 1.;
//          oneEne[sz] = hG[sz];
//        }
//        else {
//          blocks[sz] = MatrixXcd::Zero(rank, rank);
//          for (int p = 0; p < rank; p++) 
//            for (int t = 0; t < rank; t++) 
//              blocks[sz](p, t) = green[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t));
//          
//          dets[sz] = blocks[sz].determinant();
//          oneEne[sz] = hG[sz] * dets[sz];
//
//          MatrixXcd temp;
//          for (int p = 0; p < rank; p++) {
//            temp = blocks[sz];
//            for (int t = 0; t < rank; t++)
//              temp(p, t) = roth1[sz](ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t));
//            oneEne[sz] -= temp.determinant();
//          }
//        }
//      }
//      overlap += ciCoeffs[i] * ciParity[i] * dets[0] * dets[1];
//      ene += ciCoeffs[i] * ciParity[i] * (oneEne[0] * dets[1] + dets[0] * oneEne[1]);
//    }
//    overlap *= overlap0;
//    ene *= (overlap0 / overlap);
//    fb(gamma) = ene;
//  }
//};


void Multislater::forceBias(Eigen::MatrixXcd& psi, Hamiltonian& ham, Eigen::VectorXcd& fb)
{
  matPair psi2;
  psi2[0] = psi;
  psi2[1] = psi;
  this->forceBias(psi2, ham, fb);
};

std::array<std::complex<double>, 2> Multislater::hamAndOverlap(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham)
{
  int norbs = ham.norbs;
  int nalpha = ham.nalpha;
  int nbeta = ham.nbeta;
  int nchol = ham.nchol;
  std::array<int, 2> nelec{nalpha, nbeta};
  size_t ndets = ciCoeffs.size();

  matPair phi0T;
  phi0T[0] = MatrixXcd::Zero(nalpha, norbs);
  phi0T[1] = MatrixXcd::Zero(nbeta, norbs);
  for (int i = 0; i < nalpha; i++) phi0T[0](i, refDet[0][i]) = 1.;
  for (int i = 0; i < nbeta; i++) phi0T[1](i, refDet[1][i]) = 1.;

  complex<double> overlap(0., 0.);
  complex<double> ene(0., 0.);
  matPair theta, green, greenp;
  theta[0] = psi[0] * (phi0T[0] * psi[0]).inverse();
  theta[1] = psi[1] * (phi0T[1] * psi[1]).inverse();
  green[0] = (theta[0] * phi0T[0]).transpose();
  green[1] = (theta[1] * phi0T[1]).transpose();
  greenp[0] = green[0] - MatrixXcd::Identity(norbs, norbs);
  greenp[1] = green[1] - MatrixXcd::Identity(norbs, norbs);
  matPair greeno;
  greeno[0] = green[0].block(0, 0, nalpha, norbs);
  greeno[1] = green[1].block(0, 0, nbeta, norbs);

  // all quantities henceforth will be calculated in "units" of overlap0 = < phi_0 | psi >
  complex<double> overlap0 = (phi0T[0] * psi[0]).determinant() * (phi0T[1] * psi[1]).determinant();
 
  // ref contribution
  overlap += ciCoeffs[0];
  std::array<complex<double>, 2> hG;
  hG[0] = greeno[0].cwiseProduct(ham.h1.block(0, 0, nalpha, norbs)).sum();
  hG[1] = greeno[1].cwiseProduct(ham.h1.block(0, 0, nbeta, norbs)).sum();
  ene += ciCoeffs[0] * (hG[0] + hG[1]);
  
  // 1e intermediate
  matPair roth1;
  roth1[0] = (greeno[0] * ham.h1) * greenp[0];
  roth1[1] = (greeno[1] * ham.h1) * greenp[1];

  // G^{p}_{t} blocks
  vector<matPair> gBlocks;
  vector<std::array<complex<double>, 2>> gBlockDets;
  matPair empty;
  gBlocks.push_back(empty);
  std::array<complex<double>, 2> identity{1., 1.};
  gBlockDets.push_back(identity);
  
  // iterate over excitations
  for (int i = 1; i < ndets; i++) {
    matPair blocks;
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
  matPair int1, int2;
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
    matPair exc;
    for (int sz = 0; sz < 2; sz++) {
      exc[sz].noalias() = ham.chol[n].block(0, 0, nelec[sz], norbs) * theta[sz];
      lG[sz] = exc[sz].trace();
      l2G2[sz] = lG[sz] * lG[sz] - exc[sz].cwiseProduct(exc[sz].transpose()).sum();
      l2G2Tot[sz] += l2G2[sz];
      int2[sz].noalias() = (greeno[sz] * ham.chol[n].block(0, 0, norbs, nact + ncore)) * greenp[sz].block(0, ncore, nact + ncore, nact + ncore);
      int1[sz].noalias() += lG[sz] * int2[sz];
      int1[sz].noalias() -= (greeno[sz] * ham.chol[n].block(0, 0, norbs, nelec[sz])) * int2[sz];
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
        else if (rank == 1) {
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
  ene += ham.ecore * overlap;
  std::array<complex<double>, 2> hamOverlap;
  hamOverlap[0] = ene;
  hamOverlap[1] = overlap;
  return hamOverlap;
};


std::array<std::complex<double>, 2> Multislater::hamAndOverlap(Eigen::MatrixXcd& psi, Hamiltonian& ham)
{
  int norbs = ham.norbs;
  int nalpha = ham.nalpha;
  int nbeta = ham.nbeta;
  int nchol = ham.nchol;
  std::array<int, 2> nelec{nalpha, nbeta};
  size_t ndets = ciCoeffs.size();

  MatrixXcd phi0T;
  phi0T = MatrixXcd::Zero(nalpha, norbs);
  for (int i = 0; i < nalpha; i++) phi0T(i, refDet[0][i]) = 1.;

  complex<double> overlap(0., 0.);
  complex<double> ene(0., 0.);
  MatrixXcd theta, green, greenp, greeno;
  theta = psi * (phi0T * psi).inverse();
  green = (theta * phi0T).transpose();
  greenp = green - MatrixXcd::Identity(norbs, norbs);
  greeno = green.block(0, 0, nalpha, norbs);

  // all quantities henceforth will be calculated in "units" of overlap0 = < phi_0 | psi >
  complex<double> overlap0 = (phi0T * psi).determinant() * (phi0T * psi).determinant();
 
  // ref contribution
  overlap += ciCoeffs[0];
  complex<double> hG;
  hG = greeno.cwiseProduct(ham.h1.block(0, 0, nalpha, norbs)).sum();
  ene += 2 * ciCoeffs[0] * hG;
  
  // 1e intermediate
  MatrixXcd roth1;
  roth1 = (greeno * ham.h1) * greenp;

  // G^{p}_{t} blocks
  vector<matPair> gBlocks;
  vector<std::array<complex<double>, 2>> gBlockDets;
  matPair empty;
  gBlocks.push_back(empty);
  std::array<complex<double>, 2> identity{1., 1.};
  gBlockDets.push_back(identity);
  
  // iterate over excitations
  for (int i = 1; i < ndets; i++) {
    matPair blocks;
    std::array<complex<double>, 2> oneEne, dets;
    for (int sz = 0; sz < 2; sz++) {
      int rank = ciExcitations[sz][i][0].size();
      if (rank == 0) {
        dets[sz] = 1.;
        oneEne[sz] = hG;
      }
      else {
        blocks[sz] = MatrixXcd::Zero(rank, rank);
        for (int p = 0; p < rank; p++) 
          for (int t = 0; t < rank; t++) 
            blocks[sz](p, t) = green(ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t));
        
        dets[sz] = blocks[sz].determinant();
        oneEne[sz] = hG * dets[sz];

        MatrixXcd temp;
        for (int p = 0; p < rank; p++) {
          temp = blocks[sz];
          for (int t = 0; t < rank; t++)
            temp(p, t) = roth1(ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t));
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
  MatrixXcd int1, int2;
  int1 = 0. * greeno.block(0, 0, nelec[0], nact + ncore);
  int2 = 0. * greeno.block(0, 0, nelec[0], nact + ncore);
  complex<double> l2G2Tot(0., 0.);
  
  for (int n = 0; n < nchol; n++) {
    complex<double> lG, l2G2;
    MatrixXcd exc;
    exc.noalias() = ham.chol[n].block(0, 0, nelec[0], norbs) * theta;
    lG = exc.trace();
    l2G2 = lG * lG - exc.cwiseProduct(exc.transpose()).sum();
    l2G2Tot += l2G2;
    int2.noalias() = (greeno * ham.chol[n].block(0, 0, norbs, nact + ncore)) * greenp.block(0, ncore, nact + ncore, nact + ncore);
    int1.noalias() += lG * int2;
    int1.noalias() -= (greeno * ham.chol[n].block(0, 0, norbs, nelec[0])) * int2;

    // ref contribution
    ene += ciCoeffs[0] * (l2G2 + lG * lG);
    
    // iterate over excitations
    for (int i = 1; i < ndets; i++) {
      std::array<complex<double>, 2> oneEne, twoEne;
      for (int sz = 0; sz < 2; sz++) {
        int rank = ciExcitations[sz][i][0].size();
        if (rank == 0) {
          oneEne[sz] = lG;
        }
        if (rank == 1) {
          oneEne[sz] = lG * gBlockDets[i][sz] - int2(ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0) - ncore);
        }
        else if (rank == 2) {
          oneEne[sz] = lG * gBlockDets[i][sz] 
                     - int2(ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0) - ncore) * gBlocks[i][sz](1, 1)
                     + int2(ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](1) - ncore) * gBlocks[i][sz](1, 0)
                     + int2(ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](0) - ncore) * gBlocks[i][sz](0, 1)
                     - int2(ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](1) - ncore) * gBlocks[i][sz](0, 0);

          twoEne[sz] = 2. * (  int2(ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0) - ncore)    
                             * int2(ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](1) - ncore) 
                             - int2(ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](0) - ncore) 
                             * int2(ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](1) - ncore)  );
        }
        else if (rank == 3) {
          // oneEne
          {
            oneEne[sz] = lG * gBlockDets[i][sz];

            Matrix3cd temp;
            for (int p = 0; p < rank; p++) {
              temp = gBlocks[i][sz];
              for (int t = 0; t < rank; t++)
                temp(p, t) = int2(ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t) - ncore);
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
                    twoEne[sz] += 2. * parity_pq * parity_tu * blockBlockDet * (  int2(ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t) - ncore) 
                                                                                * int2(ciExcitations[sz][i][0](q), ciExcitations[sz][i][1](u) - ncore) 
                                                                                - int2(ciExcitations[sz][i][0](q), ciExcitations[sz][i][1](t) - ncore) 
                                                                                * int2(ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](u) - ncore)  );
                  }
                }
              }
            }
          }
        }
        else if (rank == 4) {
          // oneEne
          {
            oneEne[sz] = lG * gBlockDets[i][sz];

            Matrix4cd temp;
            for (int p = 0; p < rank; p++) {
              temp = gBlocks[i][sz];
              for (int t = 0; t < rank; t++)
                temp(p, t) = int2(ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t) - ncore);
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

                    twoEne[sz] += 2. * parity_pq * parity_tu * blockBlockDet * (  int2(ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t) - ncore) 
                                                                                * int2(ciExcitations[sz][i][0](q), ciExcitations[sz][i][1](u) - ncore) 
                                                                                - int2(ciExcitations[sz][i][0](q), ciExcitations[sz][i][1](t) - ncore) 
                                                                                * int2(ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](u) - ncore)  );
                  }
                }
              }
            }
          }
        } 
        else {
          // oneEne
          {
            oneEne[sz] = lG * gBlockDets[i][sz];

            MatrixXcd temp;
            for (int p = 0; p < rank; p++) {
              temp = gBlocks[i][sz];
              for (int t = 0; t < rank; t++)
                temp(p, t) = int2(ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t) - ncore);
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

                    twoEne[sz] += 2. * parity_pq * parity_tu * blockBlockDet * (  int2(ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t) - ncore) 
                                                                                * int2(ciExcitations[sz][i][0](q), ciExcitations[sz][i][1](u) - ncore) 
                                                                                - int2(ciExcitations[sz][i][0](q), ciExcitations[sz][i][1](t) - ncore) 
                                                                                * int2(ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](u) - ncore)  );
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
        twoEne[sz] = l2G2Tot;
      }
      else if (rank == 1) {
        twoEne[sz] = l2G2Tot * gBlockDets[i][sz] - 2. * int1(ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0) - ncore);
      }
      else if (rank == 2) {
        twoEne[sz] = l2G2Tot * gBlockDets[i][sz]
                   + 2. * (- int1(ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](0) - ncore) * gBlocks[i][sz](1, 1)
                           + int1(ciExcitations[sz][i][0](0), ciExcitations[sz][i][1](1) - ncore) * gBlocks[i][sz](1, 0)
                           + int1(ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](0) - ncore) * gBlocks[i][sz](0, 1)
                           - int1(ciExcitations[sz][i][0](1), ciExcitations[sz][i][1](1) - ncore) * gBlocks[i][sz](0, 0)  );
      }
      else if (rank == 3) {
        // twoEne
        {
          // term 1
          twoEne[sz] = l2G2Tot * gBlockDets[i][sz];

          //term 2
          Matrix3cd temp;
          for (int p = 0; p < rank; p++) {
            temp = gBlocks[i][sz];
            for (int t = 0; t < rank; t++)
              temp(p, t) = int1(ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t) - ncore);
            twoEne[sz] -= 2. * temp.determinant();
          }
        }
      } 
      else if (rank == 4) {
        // twoEne
        {
          // term 1
          twoEne[sz] = l2G2Tot * gBlockDets[i][sz];

          //term 2
          Matrix4cd temp;
          for (int p = 0; p < rank; p++) {
            temp = gBlocks[i][sz];
            for (int t = 0; t < rank; t++)
              temp(p, t) = int1(ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t) - ncore);
            twoEne[sz] -= 2. * temp.determinant();
          }
        }
      } 
      else {
        // twoEne
        {
          // term 1
          twoEne[sz] = l2G2Tot * gBlockDets[i][sz];

          //term 2
          MatrixXcd temp;
          for (int p = 0; p < rank; p++) {
            temp = gBlocks[i][sz];
            for (int t = 0; t < rank; t++)
              temp(p, t) = int1(ciExcitations[sz][i][0](p), ciExcitations[sz][i][1](t) - ncore);
            twoEne[sz] -= 2. * temp.determinant();
          }
        }
      } 
    } // sz

    ene += ciParity[i] * ciCoeffs[i] * (twoEne[0] * gBlockDets[i][1] + gBlockDets[i][0] * twoEne[1]) / 2.;
  
  } // dets

  overlap *= overlap0;
  ene *= overlap0;
  ene += ham.ecore * overlap;
  std::array<complex<double>, 2> hamOverlap;
  hamOverlap[0] = ene;
  hamOverlap[1] = overlap;
  return hamOverlap;
};
