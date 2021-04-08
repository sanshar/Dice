#pragma once
#include <vector>

#include "BasisShell.h"
#include "timer.h"

extern BasisSet basis;
extern cumulTimer realSumTime;
extern cumulTimer kSumTime;
extern cumulTimer latticeSumTime;
extern cumulTimer ksumTime1;
extern cumulTimer ksumTime2, ksumKsum;
extern cumulTimer pairRTime, pairKTime, coulombContractTime;
extern size_t add2;
extern "C" {
  void initPeriodic(int* pshls, int *patm, int pnatm,
                    int *pbas, int pnbas, double *penv,
                    double* lattice);
};
