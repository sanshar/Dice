#pragma once
#include <vector>

#include "BasisShell.h"

extern BasisSet basis;

extern "C" {
  void initPeriodic(int* pshls, int *pao_loc, int *patm, int pnatm,
                    int *pbas, int pnbas, double *penv,
                    double* lattice);
}
