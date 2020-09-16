#ifndef HMULTDIRECT_HEADER_H
#define HMULTDIRECT_HEADER_H
#include <Eigen/Dense>
#include <Eigen/Core>
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "communicate.h"
#include "global.h"
#include "Determinants.h"
#include <algorithm>
#include <chrono>
#include "SHCISortMpiUtils.h"
#include "SHCImake4cHamiltonian.h"
using namespace Eigen;
using namespace std;
using namespace SHCISortMpiUtils;
using namespace SHCImake4cHamiltonian;

std::complex<double> sumComplex(const std::complex<double>& a, const std::complex<double>& b) ;

namespace SHCISortMpiUtils{
  int ipow(int base, int exp);  
};

struct HmultDirect {
  int* & Nminus1ToDetLen;
  int* & Nminus2ToDetLen;
  vector<int*> & Nminus1ToDetSM;
  vector<int*> & Nminus2ToDetSM;
  Determinant* & Dets;
  int DetsSize;
  int StartIndex;
  int Norbs;
  oneInt& I1;
  twoInt& I2;
  double& coreE;
  MatrixXx & diag;

  HmultDirect(
    SHCImake4cHamiltonian::HamHelper4c& helper4c,
    Determinant* & pDets,
    int pDetsSize,
    int pStartIndex,
    int pNorbs,
    oneInt& pI1,
    twoInt& pI2,
    double& pcoreE,
    MatrixXx& pDiag
  ) : 
    Nminus1ToDetLen(helper4c.Nminus1ToDetLen),
    Nminus2ToDetLen(helper4c.Nminus2ToDetLen),
    Nminus1ToDetSM(helper4c.Nminus1ToDetSM),
    Nminus2ToDetSM(helper4c.Nminus2ToDetSM),
    Dets(pDets),
    DetsSize(pDetsSize),
    StartIndex(pStartIndex),
    Norbs(pNorbs),
    I1(pI1),
    I2(pI2),
    coreE(pcoreE),
    diag(pDiag) {};

  HmultDirect(
    int* & pNminus1ToDetLen,
    int* & pNminus2ToDetLen,
    vector<int*> & pNminus1ToDetSM,
    vector<int*> & pNminus2ToDetSM,
    Determinant* & pDets,
    int pDetsSize,
    int pStartIndex,
    int pNorbs,
    oneInt& pI1,
    twoInt& pI2,
    double& pcoreE,
    MatrixXx & pDiag
  ) : 
    Nminus1ToDetLen(pNminus1ToDetLen),
    Nminus2ToDetLen(pNminus2ToDetLen),
    Nminus1ToDetSM(pNminus1ToDetSM),
    Nminus2ToDetSM(pNminus2ToDetSM),
    Dets(pDets),
    DetsSize(pDetsSize),
    StartIndex(pStartIndex),
    Norbs(pNorbs),
    I1(pI1),
    I2(pI2),
    coreE(pcoreE),
    diag(pDiag) {};
  
  void operator() (CItype *x, CItype *y) {
    if (StartIndex >= DetsSize) return;

    #ifndef SERIAL
    boost::mpi::communicator world;
    #endif
    int nprocs = commsize, proc = commrank;
    size_t norbs = Norbs;

    //diagonal elements
    for (size_t k=StartIndex; k<DetsSize; k++) {
      if (k%(nprocs) != proc) continue;
      CItype hij = Dets[k].Energy(I1, I2, coreE);
      y[k] += hij*x[k];
    }

    size_t orbDiff;
    // single excitations
    for (int i = 0; i < Nminus1ToDetSM.size(); i++) {
      for (int j = 0; j<Nminus1ToDetLen[i]; j++) {
        int DetI = Nminus1ToDetSM[i][j];
        //if (DetI % nprocs != proc || DetI < 0) continue;
        for (int k = 0; k<j; k++) {
          int DetJ = Nminus1ToDetSM[i][k];
          if (DetI < StartIndex && DetJ < StartIndex) continue;
          CItype hij = Hij(Dets[DetI], Dets[DetJ], I1, I2, coreE, orbDiff);
          y[DetI] += hij*x[DetJ];
        }
        for (int k = j+1; k<Nminus1ToDetLen[i]; k++) {
          int DetJ = Nminus1ToDetSM[i][k];
          if (DetI < StartIndex && DetJ < StartIndex) continue;
          CItype hij = Hij(Dets[DetI], Dets[DetJ], I1, I2, coreE, orbDiff);
          y[DetI] += hij*x[DetJ];
        }
      }
    }

    for (int i = 0; i < Nminus2ToDetSM.size(); i++) {
    //if (i % nprocs != proc) continue;
      for (int j = 0; j<Nminus2ToDetLen[i]; j++) {
        int DetI = Nminus2ToDetSM[i][j];
        //if (DetI % nprocs != proc || DetI < 0) continue;
        for (int k=0; k<j; k++) {
          int DetJ = Nminus2ToDetSM[i][k];
          //if (DetI < StartIndex && DetJ < StartIndex) continue;
          if (Dets[Nminus2ToDetSM[i][j]].ExcitationDistance(Dets[Nminus2ToDetSM[i][k]]) != 2) continue;
          CItype hij = Hij(Dets[DetI], Dets[DetJ], I1, I2, coreE, orbDiff);
          y[DetI] += hij*x[DetJ];
        }
        for (int k=j+1; k<Nminus2ToDetLen[i]; k++) {
          int DetJ = Nminus2ToDetSM[i][k];
          //if (DetI < StartIndex && DetJ < StartIndex) continue;
          if (Dets[Nminus2ToDetSM[i][j]].ExcitationDistance(Dets[Nminus2ToDetSM[i][k]]) != 2) continue;
          CItype hij = Hij(Dets[DetI], Dets[DetJ], I1, I2, coreE, orbDiff);
          y[DetI] += hij*x[DetJ];
        }
      }
    }
  }
};
#endif