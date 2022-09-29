#include <algorithm>
#include <iostream>
#include "utils/global.h"
#include "utils/input.h"
#include "DQMC/GZHFMultislater.h"

using namespace std;
using namespace Eigen;

using matPair = std::array<MatrixXcd, 2>;

GZHFMultislater::GZHFMultislater(Hamiltonian& ham, std::string fname, int pnact, int pncore, bool prightQ) 
{
  std::vector<int> refDetVec;
  readDeterminantsGZHFBinary(fname, refDetVec, ciExcitations, ciParity, ciCoeffs);
  
  refDet = VectorXi::Zero(refDetVec.size());
  for (int i = 0; i < refDetVec.size(); i++) refDet[i] = refDetVec[i];

  nact = pnact;
  ncore = pncore;
  rightQ = prightQ;

  ham.blockCholesky(blockChol, ncore + nact);

  if (rightQ) {
    double sum = 0.;
    for (int i = 0; i < ciCoeffs.size(); i++) {
      sum += abs(ciCoeffs[i]);
      cumulativeCoeffs.push_back(sum);
    }
    uniform = uniform_real_distribution<double> (0., 1.);
  }
};


void GZHFMultislater::getSample(Eigen::MatrixXcd& sampleDet)
{
  return;
};


std::complex<double> GZHFMultislater::overlap(Eigen::MatrixXcd& psi)
{
  int norbs = psi.rows();
  int nelec = refDet.size();
  size_t ndets = ciCoeffs.size();

  MatrixXcd phi0T = MatrixXcd::Zero(nelec, norbs);
  for (int i = 0; i < nelec; i++) phi0T(i, refDet[i]) = 1.;

  complex<double> overlap(0., 0.);
  MatrixXcd theta, green, greenp;
  theta = psi * (phi0T * psi).inverse();
  green = (theta * phi0T).transpose();
  greenp = green - MatrixXcd::Identity(norbs, norbs);
  MatrixXcd greeno;
  greeno = green(refDet, Eigen::placeholders::all);

  // all quantities henceforth will be calculated in "units" of overlap0 = < phi_0 | psi >
  complex<double> overlap0 = (phi0T * psi).determinant();
 
  // ref contribution
  overlap += ciCoeffs[0];

  // iterate over excitations
  for (int i = 1; i < ndets; i++) {
    complex<double> dets;
    int rank = ciExcitations[i][0].size();
    if (rank == 0) {
      dets = 1.;
    }
    else if (rank == 1) {
      dets = greeno(ciExcitations[i][0](0), ciExcitations[i][1](0));
    }
    else if (rank == 2) {
      dets = greeno(ciExcitations[i][0](0), ciExcitations[i][1](0)) * greeno(ciExcitations[i][0](1), ciExcitations[i][1](1)) 
           - greeno(ciExcitations[i][0](1), ciExcitations[i][1](0)) * greeno(ciExcitations[i][0](0), ciExcitations[i][1](1));
    }
    else if (rank == 3) {
      Matrix3cd temp = Matrix3cd::Zero(3, 3);
      for (int p = 0; p < rank; p++) 
        for (int t = 0; t < rank; t++) 
          temp(p, t) = greeno(ciExcitations[i][0](p), ciExcitations[i][1](t));
      dets = temp.determinant();
    }
    else if (rank == 4) {
      Matrix4cd temp = Matrix4cd::Zero(4, 4);
      for (int p = 0; p < rank; p++) 
        for (int t = 0; t < rank; t++) 
          temp(p, t) = greeno(ciExcitations[i][0](p), ciExcitations[i][1](t));
      dets = temp.determinant();
    }
    else {
      MatrixXcd temp = MatrixXcd::Zero(rank, rank);
      for (int p = 0; p < rank; p++) 
        for (int t = 0; t < rank; t++) 
          temp(p, t) = greeno(ciExcitations[i][0](p), ciExcitations[i][1](t));
      dets = temp.determinant();
    }

    overlap += ciCoeffs[i] * ciParity[i] * dets;
  }
  overlap *= overlap0;
  return overlap;
};


void GZHFMultislater::forceBias(Eigen::MatrixXcd& psi, Hamiltonian& ham, Eigen::VectorXcd& fb)
{
  int norbs = ham.norbs;
  int nelec = refDet.size();
  int nchol = ham.nchol;
  size_t ndets = ciCoeffs.size();
  fb = VectorXcd::Zero(nchol);

  MatrixXcd phi0T;
  phi0T = MatrixXcd::Zero(nelec, norbs);
  for (int i = 0; i < nelec; i++) phi0T(i, refDet[i]) = 1.;

  MatrixXcd theta, green, greenp, greeno;
  theta = psi * (psi(refDet, Eigen::placeholders::all)).inverse();
  green = (theta * phi0T).transpose();
  greenp = green - MatrixXcd::Identity(norbs, norbs);
  greeno = green(refDet, Eigen::placeholders::all);

  // most quantities henceforth will be calculated in "units" of overlap0 = < phi_0 | psi >
  complex<double> overlap0 = (psi(refDet, Eigen::placeholders::all)).determinant();
  complex<double> overlap(0., 0.);
  overlap += ciCoeffs[0];
  MatrixXcd intermediate, greenMulti;
  intermediate = MatrixXcd::Zero(norbs, nelec);

  // iterate over excitations
  for (int i = 1; i < ndets; i++) {
    MatrixXcd temp;
    complex<double> dets;
    int rank = ciExcitations[i][0].size();
    if (rank == 0) {
      dets = 1.;
    }
    else if (rank == 1) {
      dets = greeno(ciExcitations[i][0](0), ciExcitations[i][1](0));
      temp = MatrixXcd::Zero(1, 1);
      temp(0, 0) = -1.;
    }
    else if (rank == 2) {
      Matrix2cd cofactors = Matrix2cd::Zero(2, 2);
      cofactors(0, 0) = greeno(ciExcitations[i][0](1), ciExcitations[i][1](1));
      cofactors(1, 1) = greeno(ciExcitations[i][0](0), ciExcitations[i][1](0));
      cofactors(0, 1) = -greeno(ciExcitations[i][0](1), ciExcitations[i][1](0));
      cofactors(1, 0) = -greeno(ciExcitations[i][0](0), ciExcitations[i][1](1));
      dets = cofactors.determinant();
      temp = -cofactors;
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
              minorMat(q, u) = greeno(ciExcitations[i][0](q1), ciExcitations[i][1](u1));
            }
          }
          double parity_pt = ((p + t)%2 == 0) ? 1. : -1.;
          cofactors(p, t) = parity_pt * minorMat.determinant();
          dets += greeno(ciExcitations[i][0](p), ciExcitations[i][1](t)) * cofactors(p, t);
        }
      }
      temp = -cofactors;
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
              minorMat(q, u) = greeno(ciExcitations[i][0](q1), ciExcitations[i][1](u1));
            }
          }
          double parity_pt = ((p + t)%2 == 0) ? 1. : -1.;
          cofactors(p, t) = parity_pt * minorMat.determinant();
          dets += greeno(ciExcitations[i][0](p), ciExcitations[i][1](t)) * cofactors(p, t);
        }
      }
      temp = -cofactors;
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
              minorMat(q, u) = greeno(ciExcitations[i][0](q1), ciExcitations[i][1](u1));
            }
          }
          double parity_pt = ((p + t)%2 == 0) ? 1. : -1.;
          cofactors(p, t) = parity_pt * minorMat.determinant();
          dets += greeno(ciExcitations[i][0](p), ciExcitations[i][1](t)) * cofactors(p, t);
        }
      }
      temp = cofactors;
    }

    overlap += ciCoeffs[i] * ciParity[i] * dets;
    if (rank > 0) {
      for (int p = 0; p < rank; p++) 
        for (int t = 0; t < rank; t++)
          intermediate(ciExcitations[i][1](t), ciExcitations[i][0](p)) += ciCoeffs[i] * ciParity[i] * temp(p, t);
    }
  }
  
  greenMulti = greenp * intermediate * greeno + overlap * green;
  overlap *= overlap0;
  greenMulti *= (overlap0 / overlap);
  greenMulti = greenMulti.transpose().eval();
  Eigen::Map<Eigen::VectorXcd> greenMultiVec(greenMulti.data(), greenMulti.rows() * greenMulti.cols());
  fb = greenMultiVec.transpose() * blockChol[0];
};


void GZHFMultislater::oneRDM(Eigen::MatrixXcd& psi, Eigen::MatrixXcd& rdmSample) 
{ 
  int norbs = psi.rows();
  int nelec = psi.cols();
  size_t ndets = ciCoeffs.size();
  rdmSample = MatrixXcd::Zero(norbs, norbs);

  MatrixXcd phi0T;
  phi0T = MatrixXcd::Zero(nelec, norbs);
  for (int i = 0; i < nelec; i++) phi0T(i, refDet[i]) = 1.;

  MatrixXcd theta, green, greenp, greeno;
  theta = psi * (psi(refDet, Eigen::placeholders::all)).inverse();
  green = (theta * phi0T).transpose();
  greenp = green - MatrixXcd::Identity(norbs, norbs);
  greeno = green(refDet, Eigen::placeholders::all);

  // most quantities henceforth will be calculated in "units" of overlap0 = < phi_0 | psi >
  complex<double> overlap0 = (psi(refDet, Eigen::placeholders::all)).determinant();
  complex<double> overlap(0., 0.);
  overlap += ciCoeffs[0];
  MatrixXcd intermediate, greenMulti;
  intermediate = MatrixXcd::Zero(norbs, nelec);

  // iterate over excitations
  for (int i = 1; i < ndets; i++) {
    MatrixXcd temp;
    complex<double> dets;
    int rank = ciExcitations[i][0].size();
    if (rank == 0) {
      dets = 1.;
    }
    else if (rank == 1) {
      dets = greeno(ciExcitations[i][0](0), ciExcitations[i][1](0));
      temp = MatrixXcd::Zero(1, 1);
      temp(0, 0) = -1.;
    }
    else if (rank == 2) {
      Matrix2cd cofactors = Matrix2cd::Zero(2, 2);
      cofactors(0, 0) = greeno(ciExcitations[i][0](1), ciExcitations[i][1](1));
      cofactors(1, 1) = greeno(ciExcitations[i][0](0), ciExcitations[i][1](0));
      cofactors(0, 1) = -greeno(ciExcitations[i][0](1), ciExcitations[i][1](0));
      cofactors(1, 0) = -greeno(ciExcitations[i][0](0), ciExcitations[i][1](1));
      dets = cofactors.determinant();
      temp = -cofactors;
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
              minorMat(q, u) = greeno(ciExcitations[i][0](q1), ciExcitations[i][1](u1));
            }
          }
          double parity_pt = ((p + t)%2 == 0) ? 1. : -1.;
          cofactors(p, t) = parity_pt * minorMat.determinant();
          dets += greeno(ciExcitations[i][0](p), ciExcitations[i][1](t)) * cofactors(p, t);
        }
      }
      temp = -cofactors;
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
              minorMat(q, u) = greeno(ciExcitations[i][0](q1), ciExcitations[i][1](u1));
            }
          }
          double parity_pt = ((p + t)%2 == 0) ? 1. : -1.;
          cofactors(p, t) = parity_pt * minorMat.determinant();
          dets += greeno(ciExcitations[i][0](p), ciExcitations[i][1](t)) * cofactors(p, t);
        }
      }
      temp = -cofactors;
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
              minorMat(q, u) = greeno(ciExcitations[i][0](q1), ciExcitations[i][1](u1));
            }
          }
          double parity_pt = ((p + t)%2 == 0) ? 1. : -1.;
          cofactors(p, t) = parity_pt * minorMat.determinant();
          dets += greeno(ciExcitations[i][0](p), ciExcitations[i][1](t)) * cofactors(p, t);
        }
      }
      temp = cofactors;
    }

    overlap += ciCoeffs[i] * ciParity[i] * dets;
    if (rank > 0) {
      for (int p = 0; p < rank; p++) 
        for (int t = 0; t < rank; t++)
          intermediate(ciExcitations[i][1](t), ciExcitations[i][0](p)) += ciCoeffs[i] * ciParity[i] * temp(p, t);
    }
  }
  
  greenMulti = greenp * intermediate * greeno + overlap * green;
  overlap *= overlap0;
  greenMulti *= (overlap0 / overlap);
  rdmSample = greenMulti;
}

//void GHFMultislater::forceBias(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham, Eigen::VectorXcd& fb)
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


std::array<std::complex<double>, 2> GZHFMultislater::hamAndOverlap(Eigen::MatrixXcd& psi, Hamiltonian& ham)
{
  int norbs = ham.norbs;
  int nelec = refDet.size();
  int nchol = ham.ncholEne;
  size_t ndets = ciCoeffs.size();

  MatrixXcd phi0T;
  phi0T = MatrixXcd::Zero(nelec, norbs);
  for (int i = 0; i < nelec; i++) phi0T(i, refDet[i]) = 1.;

  complex<double> overlap(0., 0.);
  complex<double> ene(0., 0.);
  MatrixXcd theta, green, greenp, greeno;
  theta = psi * (phi0T * psi).inverse();
  green = (theta * phi0T).transpose();
  greenp = green - MatrixXcd::Identity(norbs, norbs);
  greeno = green(refDet, Eigen::placeholders::all);

  // all quantities henceforth will be calculated in "units" of overlap0 = < phi_0 | psi >
  complex<double> overlap0 = (phi0T * psi).determinant();
 
  // ref contribution
  overlap += ciCoeffs[0];
  complex<double> hG;
  hG = greeno.cwiseProduct(ham.h1soc(refDet, Eigen::placeholders::all)).sum();
  ene += ciCoeffs[0] * hG;

  // 1e intermediate
  MatrixXcd roth1;
  roth1 = (greeno * ham.h1soc) * greenp;

  // G^{p}_{t} blocks
  vector<MatrixXcd> gBlocks;
  vector<complex<double>> gBlockDets;
  MatrixXcd empty;
  gBlocks.push_back(empty);
  gBlockDets.push_back(1.);
  
  // iterate over excitations
  for (int i = 1; i < ndets; i++) {
    MatrixXcd blocks;
    complex<double> oneEne, dets;
    int rank = ciExcitations[i][0].size();
    if (rank == 0) {
      dets = 1.;
      oneEne = hG;
    }
    else {
      blocks = MatrixXcd::Zero(rank, rank);
      for (int p = 0; p < rank; p++) 
        for (int t = 0; t < rank; t++) 
          blocks(p, t) = greeno(ciExcitations[i][0](p), ciExcitations[i][1](t));
      
      dets = blocks.determinant();
      oneEne = hG * dets;

      MatrixXcd temp;
      for (int p = 0; p < rank; p++) {
        temp = blocks;
        for (int t = 0; t < rank; t++)
          temp(p, t) = roth1(ciExcitations[i][0](p), ciExcitations[i][1](t));
        oneEne -= temp.determinant();
      }
    }

    overlap += ciCoeffs[i] * ciParity[i] * dets;
    ene += ciCoeffs[i] * ciParity[i] * oneEne;
    gBlocks.push_back(blocks);
    gBlockDets.push_back(dets);
  }

  // 2e intermediates
  MatrixXcd int1, int2;
  int1 = 0. * greeno.block(0, 0, nelec, nact);
  int2 = 0. * greeno.block(0, 0, nelec, nact);
  complex<double> l2G2Tot(0., 0.);
  
  for (int n = 0; n < nchol; n++) {
    complex<double> lG, l2G2;
    MatrixXcd exc;
    //exc.noalias() = ham.chol[n].block(0, 0, nelec[0], norbs) * theta;
    //lG = exc.trace();
    //l2G2 = lG * lG - exc.cwiseProduct(exc.transpose()).sum();
    //l2G2Tot += l2G2;
    //int2.noalias() = (greeno * ham.chol[n].block(0, 0, norbs, nact + ncore)) * greenp.block(0, ncore, nact + ncore, nact);
    //int1.noalias() += lG * int2;
    //int1.noalias() -= (greeno * ham.chol[n].block(0, 0, norbs, nelec[0])) * int2;
        
    exc.noalias() = ham.cholZ[n](refDet, Eigen::placeholders::all) * theta;
    lG = exc.trace();
    l2G2 = lG * lG - exc.cwiseProduct(exc.transpose()).sum();
    l2G2Tot += l2G2;
    int2.noalias() = (greeno * ham.cholZ[n].block(0, 0, norbs, nact + ncore)) * greenp.block(0, ncore, nact + ncore, nact);
    int1.noalias() += lG * int2;
    int1.noalias() -= (greeno * ham.cholZ[n](Eigen::placeholders::all, refDet)) * int2;

    // ref contribution
    ene += ciCoeffs[0] * l2G2 / 2.;
    
    // iterate over excitations
    for (int i = 1; i < ndets; i++) {
      complex<double> oneEne, twoEne;
      int rank = ciExcitations[i][0].size();
      if (rank == 0) {
        oneEne = lG;
      }
      if (rank == 1) {
        oneEne = lG * gBlockDets[i] - int2(ciExcitations[i][0](0), ciExcitations[i][1](0) - ncore);
      }
      else if (rank == 2) {
        oneEne = lG * gBlockDets[i] 
                   - int2(ciExcitations[i][0](0), ciExcitations[i][1](0) - ncore) * gBlocks[i](1, 1)
                   + int2(ciExcitations[i][0](0), ciExcitations[i][1](1) - ncore) * gBlocks[i](1, 0)
                   + int2(ciExcitations[i][0](1), ciExcitations[i][1](0) - ncore) * gBlocks[i](0, 1)
                   - int2(ciExcitations[i][0](1), ciExcitations[i][1](1) - ncore) * gBlocks[i](0, 0);

        twoEne = 2. * (  int2(ciExcitations[i][0](0), ciExcitations[i][1](0) - ncore)    
                           * int2(ciExcitations[i][0](1), ciExcitations[i][1](1) - ncore) 
                           - int2(ciExcitations[i][0](1), ciExcitations[i][1](0) - ncore) 
                           * int2(ciExcitations[i][0](0), ciExcitations[i][1](1) - ncore)  );
      }
      else if (rank == 3) {
        // oneEne
        {
          oneEne = lG * gBlockDets[i];

          Matrix3cd temp;
          for (int p = 0; p < rank; p++) {
            temp = gBlocks[i];
            for (int t = 0; t < rank; t++)
              temp(p, t) = int2(ciExcitations[i][0](p), ciExcitations[i][1](t) - ncore);
            oneEne -= temp.determinant();
          }
        }
        
        // twoEne
        {
          twoEne = complex<double>(0., 0.);
          // term 3
          for (int p = 0; p < rank; p++) {
            for (int q = p + 1; q < rank; q++) {
              double parity_pq = ((p + q + 1)%2 == 0) ? 1. : -1.;
              for (int t = 0; t < rank; t++) {
                for (int u = t + 1; u < rank; u++) {
                  double parity_tu = ((t + u + 1)%2 == 0) ? 1. : -1.;
                  complex<double> blockBlockDet = gBlocks[i](3 - p - q, 3 - t - u);
                  twoEne += 2. * parity_pq * parity_tu * blockBlockDet * (  int2(ciExcitations[i][0](p), ciExcitations[i][1](t) - ncore) 
                                                                              * int2(ciExcitations[i][0](q), ciExcitations[i][1](u) - ncore) 
                                                                              - int2(ciExcitations[i][0](q), ciExcitations[i][1](t) - ncore) 
                                                                              * int2(ciExcitations[i][0](p), ciExcitations[i][1](u) - ncore)  );
                }
              }
            }
          }
        }
      }
      else if (rank == 4) {
        // oneEne
        {
          oneEne = lG * gBlockDets[i];

          Matrix4cd temp;
          for (int p = 0; p < rank; p++) {
            temp = gBlocks[i];
            for (int t = 0; t < rank; t++)
              temp(p, t) = int2(ciExcitations[i][0](p), ciExcitations[i][1](t) - ncore);
            oneEne -= temp.determinant();
          }
        }
        
        // twoEne
        {
          twoEne = complex<double>(0., 0.);
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
                  
                  complex<double> blockBlockDet = gBlocks[i](pp, tp) * gBlocks[i](qp, up) - gBlocks[i](pp, up) * gBlocks[i](qp, tp);

                  twoEne += 2. * parity_pq * parity_tu * blockBlockDet * (  int2(ciExcitations[i][0](p), ciExcitations[i][1](t) - ncore) 
                                                                              * int2(ciExcitations[i][0](q), ciExcitations[i][1](u) - ncore) 
                                                                              - int2(ciExcitations[i][0](q), ciExcitations[i][1](t) - ncore) 
                                                                              * int2(ciExcitations[i][0](p), ciExcitations[i][1](u) - ncore)  );
                }
              }
            }
          }
        }
      } 
      else {
        // oneEne
        {
          oneEne = lG * gBlockDets[i];

          MatrixXcd temp;
          for (int p = 0; p < rank; p++) {
            temp = gBlocks[i];
            for (int t = 0; t < rank; t++)
              temp(p, t) = int2(ciExcitations[i][0](p), ciExcitations[i][1](t) - ncore);
            oneEne -= temp.determinant();
          }
        }
        
        // twoEne
        {
          twoEne = complex<double>(0., 0.);
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
                      blockBlock(mup, nup) = gBlocks[i](diff0[mup], diff1[nup]);
                  complex<double> blockBlockDet = blockBlock.determinant();

                  twoEne += 2. * parity_pq * parity_tu * blockBlockDet * (  int2(ciExcitations[i][0](p), ciExcitations[i][1](t) - ncore) 
                                                                              * int2(ciExcitations[i][0](q), ciExcitations[i][1](u) - ncore) 
                                                                              - int2(ciExcitations[i][0](q), ciExcitations[i][1](t) - ncore) 
                                                                              * int2(ciExcitations[i][0](p), ciExcitations[i][1](u) - ncore)  );
                }
              }
            }
          }
        }
      } 

      ene += ciParity[i] * ciCoeffs[i] * twoEne / 2.;
    
    } // dets
  } // chol
    
  // iterate over excitations
  for (int i = 1; i < ndets; i++) {
    complex<double> oneEne, twoEne;
    int rank = ciExcitations[i][0].size();
    if (rank == 0) {
      twoEne = l2G2Tot;
    }
    else if (rank == 1) {
      twoEne = l2G2Tot * gBlockDets[i] - 2. * int1(ciExcitations[i][0](0), ciExcitations[i][1](0) - ncore);
    }
    else if (rank == 2) {
      twoEne = l2G2Tot * gBlockDets[i]
                 + 2. * (- int1(ciExcitations[i][0](0), ciExcitations[i][1](0) - ncore) * gBlocks[i](1, 1)
                         + int1(ciExcitations[i][0](0), ciExcitations[i][1](1) - ncore) * gBlocks[i](1, 0)
                         + int1(ciExcitations[i][0](1), ciExcitations[i][1](0) - ncore) * gBlocks[i](0, 1)
                         - int1(ciExcitations[i][0](1), ciExcitations[i][1](1) - ncore) * gBlocks[i](0, 0)  );
    }
    else if (rank == 3) {
      // twoEne
      {
        // term 1
        twoEne = l2G2Tot * gBlockDets[i];

        //term 2
        Matrix3cd temp;
        for (int p = 0; p < rank; p++) {
          temp = gBlocks[i];
          for (int t = 0; t < rank; t++)
            temp(p, t) = int1(ciExcitations[i][0](p), ciExcitations[i][1](t) - ncore);
          twoEne -= 2. * temp.determinant();
        }
      }
    } 
    else if (rank == 4) {
      // twoEne
      {
        // term 1
        twoEne = l2G2Tot * gBlockDets[i];

        //term 2
        Matrix4cd temp;
        for (int p = 0; p < rank; p++) {
          temp = gBlocks[i];
          for (int t = 0; t < rank; t++)
            temp(p, t) = int1(ciExcitations[i][0](p), ciExcitations[i][1](t) - ncore);
          twoEne -= 2. * temp.determinant();
        }
      }
    } 
    else {
      // twoEne
      {
        // term 1
        twoEne = l2G2Tot * gBlockDets[i];

        //term 2
        MatrixXcd temp;
        for (int p = 0; p < rank; p++) {
          temp = gBlocks[i];
          for (int t = 0; t < rank; t++)
            temp(p, t) = int1(ciExcitations[i][0](p), ciExcitations[i][1](t) - ncore);
          twoEne -= 2. * temp.determinant();
        }
      }
    } 

    ene += ciParity[i] * ciCoeffs[i] * twoEne / 2.;
  
  } // dets

  overlap *= overlap0;
  ene *= overlap0;
  ene += ham.ecore * overlap;
  std::array<complex<double>, 2> hamOverlap;
  hamOverlap[0] = ene;
  hamOverlap[1] = overlap;
  return hamOverlap;
};
