#pragma once
#include <vector>

#include "BasisShell.h"
#include "timer.h"

extern BasisSet basis;
extern cumulTimer realSumTime;
extern cumulTimer kSumTime;
extern "C" {
  void initPeriodic(int* pshls, int *pao_loc, int *patm, int pnatm,
                    int *pbas, int pnbas, double *penv,
                    double* lattice);
};
