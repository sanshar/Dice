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

/* IrAmrr.h v20150520 EST [storm, Gerald Knizia] */
#ifndef IR_RR_H
#define IR_RR_H

// IrAmrr -- Angular Momentum Recurrence Relations.
//
// This is generated code. Changes made here will be lost!

#include <stddef.h> // for size_t
//#include "CxDefs.h" // for assert and RESTRICT
#ifndef IR_RP
   #define IR_RP RESTRICT // restricted pointer
#endif

namespace ir {
   unsigned const
      MaxLa = 6,
      MaxLc = 6;

   // number of cartesian components with angular momentum <= l
   inline size_t nCartX(int l) { return static_cast<size_t>((l+1)*(l+2)*(l+3)/6); }
   // number of cartesians components with angular momentum == l
   inline size_t nCartY(int l) { return static_cast<size_t>((l+1)*(l+2)/2); }
   // number of solid harmonic components with angular momentum <= l
   inline size_t nSlmX(int l) { return static_cast<size_t>((l+1)*(l+1)); }
   // number of solid harmonic components with angular momentum == l
   inline size_t nSlmY(int l) { return static_cast<size_t>(2*l+1); }
   // index of solid harmonic component l,c (c = 0 .. 2*l+1) within 0..nSlmX(l).
   inline size_t iSlcX(int l, unsigned c) { return static_cast<size_t>(l*l + c); }

   typedef unsigned short
      cart_vec_t,
      cart_index_t;

   void OsrrA(double * pOut, double * pGm, unsigned lab, double PmAx, double PmAy, double PmAz, double PmQx, double PmQy, double PmQz, double rho, double InvEta);
   void ShTrN(double * pOut, double const * pIn, size_t N, unsigned l);
   void OsrrB_3c_shc(double * pOut, double const * pIn, double * pMem, int la, unsigned lab, unsigned lc, double fPmQx, double fPmQy, double fPmQz, double InvEtaABC, double riz);
   void OsrrB_3c_cac(double * pOut, double const * pIn, double * pMem, int la, unsigned lab, unsigned lc, double fPmQx, double fPmQy, double fPmQz, double InvEtaABC, double riz);
   extern cart_vec_t const ix2v[680];
   extern cart_index_t const iv2x[3151];
   void ShTrN_Indirect(double * pOut, size_t so, double const * pIn, size_t si, unsigned la, cart_index_t const *ii, size_t N, size_t M);
   void ShTrA_XY(double * pOut, double const * pIn, unsigned la, unsigned lab, size_t M);
   void ShTrA_XfY(double * pOut, double const * pIn, unsigned la, unsigned lab, size_t M);
   void ShTrA_YY(double * pOut, double const * pIn, unsigned la, unsigned lab, size_t M);
   void OsrrC(double * pOut, size_t sa, size_t sb, double const * p0Z, double AmBx, double AmBy, double AmBz, unsigned lb, size_t nCount);
   void ShTrN_TN(double * pOut, double const * pIn, size_t N, unsigned l);
   void AmrrDerivA1(double * pOut, double const * p0Z, double const * p2Z, unsigned lab, unsigned la, size_t nCount);
   void AmrrDerivA2(double * pOut, double const * p0Z, double const * p2Z, double const * p4Z, unsigned lab, unsigned la, size_t nCount);
   void AmrrDerivA0L(double * pOut, double const * p2Z, double const * p4Z, unsigned lab, unsigned la, size_t nCount);
   void OsrrC_dB1(double * pOut, size_t sa, size_t sb, size_t sd, double const * p0Z, double const * p2Z, double AmBx, double AmBy, double AmBz, unsigned lb, size_t nCount);
   void OsrrC_dB2(double * pOut, size_t sa, size_t sb, size_t sd, double const * p0Z, double const * p2Z, double const * p4Z, double AmBx, double AmBy, double AmBz, unsigned lb, size_t nCount);
   void OsrrC_dB0L(double * pOut, size_t sa, size_t sb, double const * p2Z, double const * p4Z, double AmBx, double AmBy, double AmBz, unsigned lb, size_t nCount);
   void CaTrC(double * pOut, double const * pIn, size_t N, unsigned l);
   void CaTrA(double * pOut, double const * pIn, size_t si, unsigned l);
   void OsrrRx(double * pOut, double const * pIn, size_t si, double AmBx, double AmBy, double AmBz, unsigned lb);
   void ShTrN_NN(double * pOut, double const * pIn, size_t N, unsigned l);
   void ShellMdrr(double * pOut, double const * pIn, double Rx, double Ry, double Rz, unsigned lab);
   void ShellLaplace(double * pOut, double const * pIn, unsigned LaplaceOrder, unsigned lab);

   extern unsigned char iCartPow[84][3];
}


#endif // IR_RR_H
