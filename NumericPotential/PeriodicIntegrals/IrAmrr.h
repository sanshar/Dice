/* IrAmrr.h v20210309 MST [hulk, Sandeep Sharma,,,] */
#ifndef IR_RR_H
#define IR_RR_H

// IrAmrr -- Angular Momentum Recurrence Relations.
//
// This is generated code. Changes made here will be lost!

#include <stddef.h> // for size_t
#include "CxDefs.h" // for assert and RESTRICT
#ifndef IR_RP
   #define IR_RP RESTRICT // restricted pointer
#endif

namespace ir {
   unsigned const
      MaxLa = 6,
      MaxLc = 6;

   // number of cartesian components with angular momentum <= l
   inline size_t nCartX(int l) { return static_cast<size_t>((l+1)*(l+2)*(l+3)/6); }
   // assuming la <= lab, return number of cartesian components with la <= l <= lab
   inline size_t nCartX(int lab, int la) { assert(la <= lab); return nCartX(lab) - nCartX(la-1); }
   // number of cartesians components with angular momentum == l
   inline size_t nCartY(int l) { return static_cast<size_t>((l+1)*(l+2)/2); }
   // number of solid harmonic components with angular momentum <= l
   inline size_t nSlmX(int l) { return static_cast<size_t>((l+1)*(l+1)); }
   // number of solid harmonic components with angular momentum == l
   inline size_t nSlmY(int l) { return static_cast<size_t>(2*l+1); }

   // index of solid harmonic component l,c (c \in {0,..., 2*l}) within 0...nSlcX(l).
   inline size_t iSlcX(int l, unsigned c) { assert(l >= 0 && int(c) <= int(2*l)); return static_cast<size_t>(l*l + c); }
   // index of solid harmonic component l,m (m \in {-l,..., l}) within 0...nSlmX(l).
   inline size_t iSlmX(int l, int m) { assert(l >= 0 && -l <= m && m <= l); return static_cast<size_t>(l*l + l + m); }

   typedef unsigned short
      cart_vec_t,
      cart_index_t;


   // Calculate (a0|00)^m for a = 0..lab, from [0]^m and (a0|00)^(m+1) for a = 0..(lab-1).
   // Inputs:
   // - pAm1: [nCartX(lab-1)]-array containing (a0|00)^(m+1) elements
   // - Im: the single base integral (00|00)^m at target m level
   // - PmA is (P-A) (make as (B-A) * (beta/zeta) for numerical stability! see doi.org/10.1021/ct200239p appendix C)
   // - PmQ is (P-Q), InvZeta is 1/zeta = 1/(alpha+beta). See PCCP 6 5119 (2004) eq. 11.
   // Output:
   // - pOut: a single [CartX(lab)]-array containing the (a0|00)^m integrals at the target m
   void OsrrA_StepM(double *IR_RP pOut, double const *IR_RP pAm1, double Im, unsigned lab, double PmAx, double PmAy, double PmAz, double PmQx, double PmQy, double PmQz, double rho, double InvZeta);
   // Calculate (a0|00)^0 from [0]^m, for a = 0..lab, m = 0..lab (both inclusive; see notes for (a0|00)^m with m!=0).
   // Inputs:
   // - pIm: input scalar integrals I[m] = (00|00)^m for m = 0..lab (inclusive!)
   // - PmA is (P-A) (make as (B-A) * (beta/zeta) for numerical stability! see doi.org/10.1021/ct200239p appendix C)
   // - PmQ is (P-Q), InvZeta is 1/zeta = 1/(alpha+beta). See PCCP 6 5119 (2004) eq. 11.
   // Output:
   // - pOut: a single [CartX(lab)]-array containing the (a0|00)^m integrals at the target m
   // Notes:
   // - This routine can also make 3-center integrals (a0|0)^m from [0]^m = (00|0)^m
   //   (use with delta=0, and therefore Q=C and Scd=1)
   // - To make a single set of (a0|00)^n with non-zero 'n' with this routine, provide
   //   (00|00)^m base integrals with m=n...(n+lab) instead of m=0..lab, ...by simply
   //   offsetting the input 'pIm' pointer by 'n' before calling;
   //   (e.g., in the 3c base case, one can make (a0|0)^m with m=lc by offsetting pIm by lc)
   // - The effect of this routine can also be achieved by stacking OsrrA_StepM calls. This one
   //   here makes less intermediates, tough, and is easier to use if only a single target m is needed.
   //   If (a0|00)^m with multiple m=0..lcd are needed, the easiest way to achieve it is to first call.
   //   this routine to assemble (a0|00)^m, and with m=lcd, then lcd times OsrrA_StepM to make the m=(lcd-1),...,0 sets.
   void OsrrA(double *IR_RP pOut, double const *IR_RP pIm, unsigned lab, double PmAx, double PmAy, double PmAz, double PmQx, double PmQy, double PmQz, double rho, double InvZeta);
   // Calculate (a0|c)^0 from (a0|0)^{m}, for a = la..lab, with solid harmonic 'c' functions
   // Notes:
   // - pMem must hold memory for nCartX(lab) x nCartX(lc) doubles.
   // Inputs:
   // - pIn: [nCartX(lab)]-array with elements (a0|0)^m for m=lc (only m=lc!)
   // - InvEtaABC = 1/(alpha+beta+gamma)
   // - PmQ = (P-Q) = (P-C) (since Q=C for 3c case here)
   // - rie = rho/eta = rho/(gamma+delta) = rho/gamma (since delta=0 for 3c case here)
   // Output:
   // - pOut: [nCartX(lab) - nCartX(la-1)] x [(2*lc+1)] array.
   void OsrrB_3c_shc(double *IR_RP pOut, double const *IR_RP pIn, double *IR_RP pMem, int la, unsigned lab, unsigned lc, double fPmQx, double fPmQy, double fPmQz, double InvEtaABC, double rie);
   // Calculate (a0|c)^0 from (a0|0)^{m}, for a = la..lab, with Hermite-cartesian 'c' functions
   // Notes:
   // - pMem must hold memory for nCartX(lab) x nCartX(lc) doubles.
   // Inputs:
   // - pIn: [nCartX(lab)]-array with elements (a0|0)^m for m=lc (only m=lc!)
   // - InvEtaABC = 1/(alpha+beta+gamma)
   // - PmQ = (P-Q) = (P-C) (since Q=C for 3c case here)
   // - rie = rho/eta = rho/(gamma+delta) = rho/gamma (since delta=0 for 3c case here)
   // Output:
   // - pOut: [nCartX(lab) - nCartX(la-1)] x [nCartY(lc)] array
   void OsrrB_3c_cac(double *IR_RP pOut, double const *IR_RP pIn, double *IR_RP pMem, int la, unsigned lab, unsigned lc, double fPmQx, double fPmQy, double fPmQz, double InvEtaABC, double rie);
   // Transform integrals over CartX[(x-A)^lb] cartesian functions, centered at A, to solid harmoincs Slc(x-B), centered at B.
   // Inputs:
   // - p0Z: [nCartX(lb+0)] x [nCount] array of unscaled (a0| sets.
   // Output:
   // - pOut: strided [(2l+1)] x [nCount] array; with strides sb for Slc(x-B) components, and sa for Count.
   void OsrrC(double *IR_RP pOut, size_t sa, size_t sb, double const *IR_RP p0Z, double AmBx, double AmBy, double AmBz, unsigned lb, size_t nCount);
   // Transform integrals over CartX[(x-A)^lb] cartesian functions, centered at A, to solid harmoincs Slc(x-B), centered at B.
   // Inputs:
   // - p0Z: [nCartX(la+lb+0,la)] array of unscaled (a0| sets.
   // Output:
   // - pOut: strided [(2la+1)] x [(2lb+1)] array; with strides sb for Slc(x-B) components, and sa for Count.
   void OsrrC_sha(double *IR_RP pOut, size_t sa, size_t sb, double const *IR_RP p0Z, unsigned la, double AmBx, double AmBy, double AmBz, unsigned lb);
   // Transform integrals over Cartesian functions, centered at A and multiplied by (2 ZetaB)^n,
   // to all 1st directional derivatives with respect to (Bx,By,Bz) of solid harmonics Slc(x-B) at center B.
   // Inputs:
   // - p0Z: [nCartX(lb-1)] x [nCount] array of unscaled (a0| sets.
   // - p2Z: [nCartX(lb+1)] x [nCount] array of (2 ZetaB)^1-scaled (a0| sets.
   // Output:
   // - pOut: strided [3] x [(2lb+1)] x [nCount] array, with
   //   strides sd (for derivatives x y z), sb (for Slc(x-B)), and sa (for nCount).
   void OsrrC_dB1(double *IR_RP pOut, size_t sa, size_t sb, size_t sd, double const *IR_RP p0Z, double const *IR_RP p2Z, double AmBx, double AmBy, double AmBz, unsigned lb, size_t nCount);
   // Transform integrals over Cartesian functions, centered at A and multiplied by (2 ZetaB)^n,
   // to all 2nd directional derivatives with respect to (Bx,By,Bz) of solid harmonics Slc(x-B) at center B.
   // Inputs:
   // - p0Z: [nCartX(lb-2)] x [nCount] array of unscaled (a0| sets.
   // - p2Z: [nCartX(lb+0)] x [nCount] array of (2 ZetaB)^1-scaled (a0| sets.
   // - p4Z: [nCartX(lb+2)] x [nCount] array of (2 ZetaB)^2-scaled (a0| sets.
   // Output:
   // - pOut: strided [6] x [(2lb+1)] x [nCount] array, with
   //   strides sd (for derivatives xx yy zz xy xz yz), sb (for Slc(x-B)), and sa (for nCount).
   void OsrrC_dB2(double *IR_RP pOut, size_t sa, size_t sb, size_t sd, double const *IR_RP p0Z, double const *IR_RP p2Z, double const *IR_RP p4Z, double AmBx, double AmBy, double AmBz, unsigned lb, size_t nCount);
   // Transform integrals over Cartesian functions, centered at A and multiplied by (2 ZetaB)^n,
   // to 1st Laplace derivatives with respect to vector B of solid harmonics Slc(x-B) at center B.
   // Inputs:
   // - p2Z: [nCartX(lb+0)] x [nCount] array of (2 ZetaB)^1-scaled (a0| sets.
   // - p4Z: [nCartX(lb+2)] x [nCount] array of (2 ZetaB)^2-scaled (a0| sets.
   // Output:
   // - pOut: strided [1] x [(2lb+1)] x [nCount] array, with
   //   strides sd (for derivatives s), sb (for Slc(x-B)), and sa (for nCount).
   void OsrrC_dB0L(double *IR_RP pOut, size_t sa, size_t sb, double const *IR_RP p2Z, double const *IR_RP p4Z, double AmBx, double AmBy, double AmBz, unsigned lb, size_t nCount);
   // Calculate (a0|c0)^0 from (a0|00)^{m}, for a = la..lab and c = lc..lcd, and input m = 0..lcd
   // Notes:
   // - pMem must hold memory for nCartX(lab) x nCartX(lcd) x lcd doubles.
   // Inputs:
   // - pIn: [nCartX(lab)] x [lcd+1] array with input (a0|0)^m components for m\in\{0,1,...,lcd\}
   // - i2e = 0.5/(gamma + delta)
   // - i2ze = 0.5/(alpha + beta + gamma + delta)
   // - fPmQ = (P-Q) on input (f: will be scaled inside function scope)
   // - QmC = (Q-C) = (D-C)*(delta/(gamma+delta)) (see comments on (P-A) in OsrrA)
   // Output:
   // - pOut: [nCartX(lab) - nCartX(la-1)] x [nCartX(lcd) - nCartX(lc-1)] array with (a0|c0)^0 components.
   void OsrrB_4c(double *IR_RP pOut, double const *IR_RP pIn, double *IR_RP pMem, double fPmQx, double fPmQy, double fPmQz, double QmCx, double QmCy, double QmCz, double rho, double i2e, double i2ze, int la, unsigned lab, unsigned lc, unsigned lcd);

   extern cart_vec_t const ix2v[680];
   extern cart_index_t const iv2x[3151];
   // Factorize a matrix (nCartX(lab)-nCartX(a-1)) x M into nCartX(lab-la) x (2*la+1) x M. Now la < lb is also allowed.
   void ShTrA_XY(double *IR_RP pOut, double const *IR_RP pIn, unsigned la, unsigned lab, size_t M);
   // Factorize a matrix nCartX(lab) x M into nCartX(lab-la) x (2*la+1) x M. Now la < lb is also allowed.
   void ShTrA_XfY(double *IR_RP pOut, double const *IR_RP pIn, unsigned la, unsigned lab, size_t M);
   // Factorize a matrix nCartY(lab) x M into nCartY(lab-la) x (2*la+1) x M. Now la < lb also allowed.
   void ShTrA_YY(double *IR_RP pOut, double const *IR_RP pIn, unsigned la, unsigned lab, size_t M);
   // Factorize one set of [CartX2(lab,la)] into [CartX(lab,la)] x [(2*la+1)]. Both la and (lab-la) must be <= 6
   void ShTrA1_XY(double *IR_RP pOut, double const *IR_RP pIn, unsigned la, unsigned lab);
   // Calculate [r]^0 for r\in CartY(lab) from input [0]^m vector with m=0..lab.
   // Effectively, this function calculates for all r\inCartY the derivatives:
   //    D^r f(T)
   // from f^[m](T) = (2 rho D/D{T})^m f(T) and R, where T = rho R^2, D^r means \prod_i (D/D{R_i})^{r_x},
   // and f is some arbitrary scalar function (of which you supply the m'th derivatives with
   // respect to T as [0]^m).
   void ShellMdrr(double *IR_RP pOut, double const *IR_RP pIn, double Rx, double Ry, double Rz, unsigned lab);
   // Form [R]_out := [R + 2ix]_in + [R + 2iy]_in + [R + 2iz]_in: Contract nCartY(lab+2) into nCartY(lab).
   // Note: can be chained multiple times to form higher order Laplace operators
   // (e.g., in an initially nCartX(lab) storage space, or swapping buffers after each invocation)
   void ShellLaplace(double *IR_RP pOut, double const *IR_RP pIn, unsigned LaplaceOrder, unsigned lab);

   // Transform cartesians centered at A and multiplied by (2 ZetaA)^n,
   // to all 1st directional derivatives with respect to (Ax,Ay,Az) at center A.
   // Inputs:
   // - p0Z: [nCartX(lb-1)] x [nCount] array of unscaled (a0| sets.
   // - p2Z: [nCartX(lb+1)] x [nCount] array of (2 ZetaA)^1-scaled (a0| sets.
   // Output:
   // - pOut: [nCartX(lb)] x [nCartY(la)] x [3] x [nCount] array, with middle dimension
   //   for derivatives x y z. Linear output, no strides.
   void AmrrDerivA1(double *IR_RP pOut, double const *IR_RP p0Z, double const *IR_RP p2Z, unsigned lab, unsigned la, size_t nCount);
   // Transform cartesians centered at A and multiplied by (2 ZetaA)^n,
   // to all 2nd directional derivatives with respect to (Ax,Ay,Az) at center A.
   // Inputs:
   // - p0Z: [nCartX(lb-2)] x [nCount] array of unscaled (a0| sets.
   // - p2Z: [nCartX(lb+0)] x [nCount] array of (2 ZetaA)^1-scaled (a0| sets.
   // - p4Z: [nCartX(lb+2)] x [nCount] array of (2 ZetaA)^2-scaled (a0| sets.
   // Output:
   // - pOut: [nCartX(lb)] x [nCartY(la)] x [6] x [nCount] array, with middle dimension
   //   for derivatives xx yy zz xy xz yz. Linear output, no strides.
   void AmrrDerivA2(double *IR_RP pOut, double const *IR_RP p0Z, double const *IR_RP p2Z, double const *IR_RP p4Z, unsigned lab, unsigned la, size_t nCount);
   // Transform cartesians centered at A and multiplied by (2 ZetaA)^n,
   // to 1st Laplace derivatives with respect to vector A at center A.
   // Inputs:
   // - p2Z: [nCartX(lb+2)] x [nCount] array of (2 ZetaA)^1-scaled (a0| sets.
   // - p4Z: [nCartX(lb+4)] x [nCount] array of (2 ZetaA)^2-scaled (a0| sets.
   // Output:
   // - pOut: [nCartX(lb)] x [nCartY(la)] x [1] x [nCount] array, with middle dimension
   //   for (\nabla_A)^2 times derivatives s. Linear output, no strides.
   void AmrrDerivA0L(double *IR_RP pOut, double const *IR_RP p2Z, double const *IR_RP p4Z, unsigned lab, unsigned la, size_t nCount);
   // Expand solid harmonic S[l=lb,m](r-B) centered at B into cartesians of degree 0..lb centered at A.
   void OsrrRx(double *IR_RP pOut, double const *IR_RP pIn, size_t si, double AmBx, double AmBy, double AmBz, unsigned lb);

   // Transform cartesians to Slc: map matrix [N] x [nCartY(l)] --> [N] x [(2*l+1)].
   void ShTrN(double *IR_RP pOut, double const *IR_RP pIn, size_t N, unsigned l);
   // Transforms [R] x [M] -> [r] x [S(la)] x [M] where R = lab and r == la.
   // In order to do that, the cartesian component
   // indices for the nCartX(lab-la) set of nCartY(la) monomials need to be input manually (via ii).
   // The actual parameters are thus: N == nSets (e.g., nCartX(lab-la)), ii: [nCartY(la)] x [N] integer array.
   // M: number of sets to transform (M could in principle be done as an outer loop over the function)
   void ShTrN_Indirect(double *IR_RP pOut, size_t so, double const *IR_RP pIn, size_t si, unsigned la, cart_index_t const *ii, size_t N, size_t M);
   // Transform cartesians to Slc: map matrix [nCartY(l)] x [N] --> [N] x [(2*l+1)].
   void ShTrN_TN(double *IR_RP pOut, double const *IR_RP pIn, size_t N, unsigned l);
   // Transform Slc to cartesians: map matrix [(2*l+1)] x [N] --> [nCartY(l)] x [N]
   void CaTrC(double *IR_RP pOut, double const *IR_RP pIn, size_t N, unsigned l);
   // Transform Slc to cartesians: map a single strided (si) [(2*l+1)] vector --> [nCartY(l)] vector
   void CaTrA(double *IR_RP pOut, double const *IR_RP pIn, size_t si, unsigned l);
   // Transform cartesians to Slc: map matrix nCartY(l) x N --> (2*l+1) x N.
   void ShTrN_NN(double *IR_RP pOut, double const *IR_RP pIn, size_t N, unsigned l);

   // Given l and c, return a descriptive label of the solid harmonic
   // Gaussian Slc(x,y,z) exp(-zeta r^2)
   char const *pSlcLabel(unsigned l, unsigned c);
   // Given l and c, returns the 'm' component of the real solid harmonic
   // S^l_m(R) (with m\in\{-l,...,+l\}) which is represented by the solid
   // harmonic component S^l_c(R), where 'c' (with c\in\{0,1,...,2l\}) denotes
   // a spherical component ('c') of the order decided at generation-time of IR
   int Sl_c2m(unsigned l, unsigned c);
   // Reverse of c2m: returns the component 'c' (with c\in\{0,1,..,2l\})
   // for a given m\in\{-l,..,l\}
   unsigned Sl_m2c(unsigned l, int m);
   // Calculate Slc(x,y,z) for all l,m with l <= L.
   // Output is addressed as pOut[iSlcX(l,c)]
   void EvalSlcX_Deriv0(double *IR_RP pOut, double x, double y, double z, unsigned L);
   // Calculate Slc(x,y,z) and 1st derivatives for all l,m with l <= L.
   // Output is addressed as pOut[iSlcX(l,c)*NCOMP_Deriv1 + iComp] with iComp \in {0,1,...,3} indexed by FSlcX::COMP__*
   // This function evaluates derivative components:
   //    [1, d/dx, d/dy, d/dz] Slc(x,y,z)
   void EvalSlcX_Deriv1(double *IR_RP pOut, double x, double y, double z, unsigned L);
   // Calculate Slc(x,y,z) and 1st+2nd derivatives for all l,m with l <= L.
   // Output is addressed as pOut[iSlcX(l,c)*NCOMP_Deriv2 + iComp] with iComp \in {0,1,...,9} indexed by FSlcX::COMP__*
   // This function evaluates derivative components:
   //    [1, d/dx, d/dy, d/dz, d^2/dxx, d^2/dyy, d^2/dzz, d^2/dxy, d^2/dxz, d^2/dyz] Slc(x,y,z)
   void EvalSlcX_Deriv2(double *IR_RP pOut, double x, double y, double z, unsigned L);
   // Calculate Slc(x,y,z) and 1st+2nd+3rd derivatives for all l,m with l <= L.
   // Output is addressed as pOut[iSlcX(l,c)*NCOMP_Deriv3 + iComp] with iComp \in {0,1,...,19} indexed by FSlcX::COMP__*
   // This function evaluates derivative components:
   //    [1, d/dx, d/dy, d/dz, d^2/dxx, d^2/dyy, d^2/dzz, d^2/dxy, d^2/dxz, d^2/dyz, d^3/dxxx, d^3/dyyy, d^3/dzzz, d^3/dxxy, d^3/dxxz, d^3/dxyy, d^3/dyyz, d^3/dxzz, d^3/dyzz, d^3/dxyz] Slc(x,y,z)
   void EvalSlcX_Deriv3(double *IR_RP pOut, double x, double y, double z, unsigned L);

   enum FSlcX {
      // maximal dimensions supported by this version of EvalSlcX_DerivN()
      MAX_AngMom = 6,
      MAX_DerivOrder = 3,
      // total number of output derivative components (=output stride) for EvalSlcX_DerivN()
      NCOMP_Deriv0 = 1,
      NCOMP_Deriv1 = 4,
      NCOMP_Deriv2 = 10,
      NCOMP_Deriv3 = 20,
      // order of output derivative components returned by EvalSlcX_DerivN()
      COMP_Value = 0,
      COMP_Dx = 1,
      COMP_Dy = 2,
      COMP_Dz = 3,
      COMP_Dxx = 4,
      COMP_Dyy = 5,
      COMP_Dzz = 6,
      COMP_Dxy = 7,
      COMP_Dxz = 8,
      COMP_Dyz = 9,
      COMP_Dxxx = 10,
      COMP_Dyyy = 11,
      COMP_Dzzz = 12,
      COMP_Dxxy = 13,
      COMP_Dxxz = 14,
      COMP_Dxyy = 15,
      COMP_Dyyz = 16,
      COMP_Dxzz = 17,
      COMP_Dyzz = 18,
      COMP_Dxyz = 19
   };
   extern unsigned char iCartPow[3654][3];
   extern unsigned char iSlcMirrorSymmetrySignatures[49];
}


#endif // IR_RR_H
