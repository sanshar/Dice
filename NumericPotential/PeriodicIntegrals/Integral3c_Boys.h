#pragma once

#include "CxMemoryStack.h"
#include "Kernel.h"

class BasisShell;
class LatticeSum;

void EvalInt2e3c(double *pOut, size_t *Strides,
                 BasisShell const *pA, BasisShell const *pB, BasisShell const *pCs,
                 size_t nC, double Prefactor, Kernel *pKernel,
                 LatticeSum& latsum, ct::FMemoryStack2 &Mem);
