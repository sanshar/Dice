#pragma once

#include "CxMemoryStack.h"

class BasisShell;


struct LatticeSum {

  double RVolume, KVolume;
  vector<double> RLattice, KLattice;
  
  vector<double> Rcoord;
  vector<double> Rdist;

  vector<double> Kcoord;
  vector<double> Kdist;

  double Eta2Rho;
  LatticeSum(double* Lattice, double _Eta2Rho=15.0);
};

double calcCoulombIntegralPeriodic_Boys(int n1, double Ax, double Ay, double Az,
                                        double expA, double normA,
                                        int n2, double Bx, double By, double Bz,
                                        double expB, double normB,
                                        double Lx, double Ly, double Lz,
                                        double* IntOut);

void Int2e2c_EvalCoKernels(double *pCoFmT, uint TotalL,
                           BasisShell *pA, BasisShell *pC,
                           double Tx, double Ty, double Tz,
                           double PrefactorExt, double* pInv2Alpha, double* pInv2Gamma,
                           LatticeSum& latsum, int rindex, ct::FMemoryStack &Mem);

void Int2e2c_EvalCoShY(double *&pOutR, unsigned &TotalCo, BasisShell *pA,
                       BasisShell *pC, double Tx, double Ty, double Tz,
                       double Prefactor,   unsigned TotalLab,
                       double* pInv2Alpha, double* pInv2Gamma,
                       LatticeSum& latsum, ct::FMemoryStack& Mem);

void EvalInt2e2c( double *pOut, size_t StrideA, size_t StrideC,
                  BasisShell *pA, BasisShell *pC,
                  double Prefactor, bool Add,
                  LatticeSum& latsum, ct::FMemoryStack &Mem );

void Add2(double * pOut, double const * pIn, double f, size_t n);
void Scatter2e2c(double * pOut, size_t StrideA, size_t StrideC,
                 double const * pIn, size_t la, size_t lc, size_t nComp,
                 size_t nCoA, size_t nCoC, bool Add);
