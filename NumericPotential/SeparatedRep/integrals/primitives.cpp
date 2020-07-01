#include <iostream>
#include <cmath>
#include "primitives.h"
#include "workArray.h"

using namespace std;
double nChoosek( size_t n, size_t k )
{
  if (k > n) return 0.;
  if (k * 2 > n) k = 1.*(n-k);
  if (k == 0) return 1.;

  double result = 1.*n;
  for( int i = 2; i <= k; ++i ) {
    result *= (n-i+1);
    result /= i;
  }
  return result;
}

double doubleFact(size_t n) {
  if (n == -1) return 1.;
  double val = 1.0;
  int limit = n/2;
  for (int i=0; i<limit; i++)
    val *= (n-2*i);
  return val;
}


//calculates the entire matrix \Omega_{ij} i,j<=max(n1,n2)
double calc1DOvlp(int n1, double A, double expA,
                  int n2, double B, double expB,
                  tensor& S) {
  double p = expA + expB,
      P = (expA * A + expB * B)/(expA + expB),
      mu = expA * expB / (expA + expB);
  if (mu*(A-B)*(A-B) > 40.) {
    return 0.0;
  }
  
  S(0,0) = sqrt(M_PI/p) * exp(-mu * (A-B) * (A-B));
  if (n1 == 0 && n2 == 0) return S(0,0);

  double f = 1./2./p;
  
  for (int L = 0; L< max(n1, n2); L++) {
    for (int i=0; i<= L; i++) {
      S(i, L+1) = (P-B) * S(i,L) ;
      S(i, L+1) += L != 0 ? L * S(i, L-1)*f : 0.;
      S(i, L+1) += i != 0 ? i * S(i-1, L)*f : 0.; 
    }

    for (int j=0; j<= L+1; j++) {
      S(L+1, j) = (P-A) * S(L, j) ;
      S(L+1, j) += L != 0 ? L * S(L-1, j)*f : 0.;
      S(L+1, j) += j != 0 ? j * S(L, j-1)*f : 0.; 
    }    
  }
  
  return S(n1, n2);
}


//actually it is \sqrt(pi/t) Theta3[0, exp(-pi*pi/t)]
double JacobiTheta(double z, double t, bool includeZero ) {

  double result = 0.0;
  if (t > 3.5) {//then do summation in direct space
    double prefactor = sqrt(t/M_PI);
    double Z = z/M_PI;
    Z = fmod(Z, 1.0) ; Z = Z < 0.0 ? 1.0 + Z : Z; //make sure that Z is between 0 and 1;
    
    for (int n = 0; n<=100; n++) {
      double term = exp(-t*(Z-n)*(Z-n));
      if (n != 0) term += exp(-t*(Z+n)*(Z+n));

      result += term ;
      if (abs(term) < 1.e-12) break;
    }
    if (!includeZero) result -= 1.0/prefactor;
    return result*prefactor;
  }
  else {
    double expt = exp(-M_PI*M_PI/t);
    int k0 = includeZero ? 0 : 1;
    for (int k = k0; k<=100; k++) {
      double term =  pow(expt, k*k) * cos(2*k*z);
      result += k==0 ? term : 2*term;

      if (abs(term) < 1.e-12) break;
    }
    return result;
  }
}

//actually it is \sqrt(pi/t) Theta3[0, exp(-pi*pi/t)]
void JacobiThetaDerivative(double z, double t, double* workArray, int nderiv, bool includeZero ) {

  //zero out all the derivatives
  std::fill(workArray, workArray+ 2*(nderiv+1), 0.0);

  if (t > 3.5) {//then do summation in direct space
    double prefactor = sqrt(t/M_PI);
    double Z = z/M_PI;
    Z = fmod(Z, 1.0) ; Z = Z < 0.0 ? 1.0 + Z : Z; //make sure that Z is between 0 and 1;

    double* intermediate = workArray + nderiv+1;
    for (int n = 0; n<=10; n++) {

      //do for positive n
      double nAB = Z - n;
      double expt1 = exp(-t * nAB * nAB), expt2 = 1.;
      
      intermediate[0] = expt1;
      if (nderiv >= 1)
        intermediate[1] = -expt1 * nAB * 2 * t;
      for (int k = 2; k <= nderiv; k++) 
        intermediate[k] = -2 * t * (intermediate[k-2] + nAB * intermediate[k-1])/k;

      double factorial = 1.0;
      workArray[0] += intermediate[0];
      for (int k = 1; k <= nderiv; k++) {
        factorial *= k/M_PI;
        workArray[k] += factorial * intermediate[k];
      }

      double maxIntermediate1 = max(1., abs(intermediate[0]));
      for (int i=1; i<=nderiv; i++)
        if (abs(intermediate[i]) > abs(maxIntermediate1))
          maxIntermediate1 = intermediate[i];

      //if n is not zero then do the summation for negative n
      if (n != 0) {
        nAB = Z + n;
        expt2 = exp(-t * nAB * nAB);
        
        intermediate[0] = expt2;
        if (nderiv >= 1)
          intermediate[1] = -expt2 * nAB * 2 * t;
        for (int k = 2; k <= nderiv; k++) 
          intermediate[k] = -2 * t * (intermediate[k-2] + nAB * intermediate[k-1])/k;

        double factorial = 1.0;
        workArray[0] += intermediate[0];
        for (int k = 1; k <= nderiv; k++) {
          factorial *= k/M_PI;
          workArray[k] += factorial * intermediate[k];
        }
      }

      double maxIntermediate = max(abs(maxIntermediate1), abs(intermediate[0]));
      for (int i=1; i<=nderiv; i++)
        if (abs(intermediate[i]) > abs(maxIntermediate))
          maxIntermediate = intermediate[i];
      
      //multiply this intermediate with another expt to estimate the next term
      if (abs(maxIntermediate) < 1.e-12)
        break;
    }

    workArray[0] = includeZero ? workArray[0] * prefactor : (workArray[0] - 1.0/prefactor) * prefactor;
    for (int i=1; i<=nderiv; i++)
      workArray[i] *= prefactor;
  }
  else {
    double expt = exp(-M_PI*M_PI/t);
    int n0 = includeZero ? 0 : 1;
    for (int n = n0; n<=100; n++) {
      double exptPown = pow(expt, n*n);

      workArray[0] += exptPown * cos(2*z*n) * (n==0 ? 1. : 2.);
      if (n > 0) {
        for (int k = 1; k<=nderiv; k++) 
          workArray[k] += 2 * pow(2*n, k) * exptPown * cos(k*M_PI/2 + 2*z*n);
      }

      //multiply current term with an additional expt to estimate the next term
      if (abs(exptPown*pow(2*M_PI,nderiv)*pow(expt,2*n+1)) < 1.e-12) break;
    }
  }
}


//calculates the entire matrix \Omega_{ij} i,j<=max(n1,n2)
double calc1DOvlpPeriodic(int n1, double A, double a,
                          int n2, double B, double b, double t,
                          int AdditionalDeriv, double L, 
                          tensor& S, double* workArray,
                          tensor& powPiOverL, tensor& pow1OverExpA,
                          tensor& pow1OverExpB) {

  double z = (A-B) * M_PI/L ;
  JacobiThetaDerivative(z, t*L*L, &workArray[0], (n1+n2+AdditionalDeriv), true);
  
  //convert derivatives of gaussian to polynomial times gaussians
  for (int i = 0; i <= n1; i++) {
    for (int j = 0; j <= n2+AdditionalDeriv; j++) {
      S(i, j) = 0.0;
      for (int ii = i%2; ii <= i; ii+=2) 
        for (int jj = j%2; jj <= j; jj+=2) {

          
          S(i, j) += (jj%2 == 0 ? 1 : -1) * DerivativeToPolynomial(i, ii) *
              DerivativeToPolynomial(j, jj) *
              workArray[ii+jj] * pow1OverExpA( (i+ii)/2) *
              pow1OverExpB( (j+jj)/2) * powPiOverL( jj+ii);
          
          /*S(i, j) += DerivativeToPolynomial(i, ii) *
              DerivativeToPolynomial(j, jj) *
              workArray[ii+jj] *
              pow(-1, jj) * pow(1./a, (i + ii)/2) * pow(1./b, (j+jj)/2)
              * pow(M_PI/L, ii+jj);
              */
        }
    }
  }

}


