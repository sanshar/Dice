/* Copyright (c) 2012  Gerald Knizia
 * 
 * This file is part of the IR/WMME program
 * (See https://sites.psu.edu/knizia/)
 * 
 * IR/WMME is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 * 
 * IR/WMME is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with bfint (LICENSE). If not, see http://www.gnu.org/licenses/
 */

#include <stdexcept>
#include <cmath>
#include "IrBoysFn.h"
#include "IrBoysFn.inl"

namespace ir {

// store Boys function values Factor*Fm(T) at pOut[0...MaxM], MaxM inclusive
void IrBoysFn(double *pOut, double T, unsigned MaxM, double Factor)
{
   if (MaxM > TableMaxM)
      throw std::runtime_error("This version of IrBoysFn does not support the supplied M.");
   unsigned
      // index of table entry closest to T
      iTab = (unsigned)(T * StepsPerT + .5);

   if (iTab >= TableSize || T > 100.) {
      // large T, out of tabulated range.
      // calculate F(m=0,T) directly and get the other Fms by upward recursion.
      // Note: F0(T) = sqrt(pi/4) * Erf(sqrt(T))/sqrt(T)
      //       F_{m}(T) = ((2*m-1)*F_{m-1}(T) - exp(-T))/(2*T)
      double
         ExpT = 0.,
         Erf = 1.,
         SqrtT = std::sqrt(T),
         InvSqrtT = 1./SqrtT;
      if (T < 36 + 2*MaxM)
         // may need this in the upward recursion
         ExpT = std::exp(-T);
      if (T < 36) { // <- for T >= 36 1-erf(T) < 1e-16.
         // Target: Erf = std::erf(SqrtT);
         // p is a chebyshev approximation for erfc(x)*exp(x**2)*x
         // with x=T**.5 in the interval T=12..36
         double p =     -7.2491691372602426e-09;
         p = p * SqrtT + 3.8066154133517713e-07;
         p = p * SqrtT - 9.0705120587974647e-06;
         p = p * SqrtT + 1.2950910968223645e-04;
         p = p * SqrtT - 1.2316919353973963e-03;
         p = p * SqrtT + 8.1970126968511671e-03;
         p = p * SqrtT - 3.8969822025737025e-02;
         p = p * SqrtT + 1.3231983459351979e-01;
         p = p * SqrtT - 3.1347761821045855e-01;
         p = p * SqrtT + 4.8563266902187724e-01;
         p = p * SqrtT + 1.6056422712311058e-01;
         Erf = 1. - p*ExpT*InvSqrtT;
         // max error of fit for erf(sqrt(T)): 1.11e-16
      }

      double
         Inv2T = .5 * InvSqrtT * InvSqrtT; // 1/(2*T)
      ExpT *= Factor;

      pOut[0] = Factor * HalfSqrtPi * InvSqrtT * Erf;
      for (unsigned m = 1; m <= MaxM; ++ m)
         pOut[m] = ((2*m-1) * pOut[m-1] - ExpT)*Inv2T;
   } else {
      // T is in tabulated range. Calculate Fm for highest order M by
      // Taylor expansion and other Ts by downward recursion.
      double const
         // difference to next tabulated point
         Delta = T - iTab*TsPerStep,
         *pTab = &BoysData[iTab*TableOrder + MaxM];
      double
         Fm;
      Fm =              pTab[8]/(double)(1*2*3*4*5*6*7*8);
      Fm = Delta * Fm - pTab[7]/(double)(1*2*3*4*5*6*7);
      Fm = Delta * Fm + pTab[6]/(double)(1*2*3*4*5*6);
      Fm = Delta * Fm - pTab[5]/(double)(1*2*3*4*5);
      Fm = Delta * Fm + pTab[4]/(double)(1*2*3*4);
      Fm = Delta * Fm - pTab[3]/(double)(1*2*3);
      Fm = Delta * Fm + pTab[2]/(double)(1*2);
      Fm = Delta * Fm - pTab[1];
      Fm = Delta * Fm + pTab[0];

      pOut[MaxM] = Factor * Fm;
      if (MaxM != 0) {
         double
            ExpT = Factor * std::exp(-T);
         for (int m = MaxM-1; m != -1; --m)
            pOut[m] = (2*T * pOut[m+1] + ExpT)*Inv2mp1[m];
      }
   }
//    printf("IrBoysFn(T=%f, M=%i, F=%f) -> [0] = %f", T, MaxM, Factor, pOut[0]);
}


} // namespace ir
