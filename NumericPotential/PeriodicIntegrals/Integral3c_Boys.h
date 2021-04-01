#pragma once
#include <vector>

#include "CxMemoryStack.h"
#include "Kernel.h"

class Basis;
class BasisShell;
class LatticeSum;

void EvalInt2e3c(double *pOut, size_t *Strides,
                 BasisShell *pA, BasisShell *pB, BasisShell  *pCs,
                 size_t nC, double Prefactor, Kernel *pKernel,
                 LatticeSum& latsum, ct::FMemoryStack2 &Mem);

void EvalInt2e3cKKsum(double *pOut, BasisShell const *pA,
		      BasisShell const *pB, BasisShell const *pC,
		      double Prefactor, Kernel *pKernel,
		      LatticeSum& latsum, ct::FMemoryStack2 &Mem) ;

void ThreeCenterIntegrals(std::vector<int>& shls, BasisSet& basis,
			  std::vector<double>& Lattice, ct::FMemoryStack2 &Mem);

void PopulateAuxGMatrix(double* pOut, BasisSet& basis, std::vector<int>& shls,
			LatticeSum& latsum, ct::FMemoryStack2 &Mem);

void PopulateAuxGMatrix(double* pOut, BasisShell* pC, size_t offset,
			size_t nAuxBas, LatticeSum& latsum, ct::FMemoryStack2 &Mem);


void PopulatePairGMatrix(double* pOut, BasisShell* pA, BasisShell* pB, 
			 size_t nFnPair, LatticeSum& latsum, ct::FMemoryStack2 &Mem) ;

void Int2e3cRK(double *pOut, vector<int>& shls, BasisSet& basis, LatticeSum& latsum,
	       ct::FMemoryStack2 &Mem) ;
void Int2e3cRR(double *pIntFai, vector<int>& shls, BasisSet &basis, LatticeSum& latsum,
               ct::FMemoryStack2 &Mem);
