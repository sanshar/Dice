#include <iostream>
#include "primitives.h"
#include "workArray.h"


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
                  MatrixXd& S) {
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



//calculates the entire matrix \Omega_{ij} i,j<=max(n1,n2)
double calc1DOvlpPeriodic(int n1, double A, double a,
                          int n2, double B, double b,
                          MatrixXd& S, double* workArray) {
  double t = a * b /(a + b);
  double p = a + b;
  double P = (a * A + b * B)/(A + B);

  //the number of elements stored in workArray are 4( n1/2+n2/2))
  //first let various derivatives
  std::fill(workArray, workArray+ 2*(n1+n2), 0.0);
  if (t > 3.5) {

    for (int n=0; n<=100; n++) {
      double nAB = (n - A + B);
      double expt = exp(-t * nAB * nAB);
      workArray[0] += expt ;    //00
      workArray[1] += 2*t * expt * nAB ;  //10 01
      workArray[2] += (2*t * expt - 4* t * t /a * expt * nAB * naB)/(4*a*b); //11

      if (n1+n2 < 2) continue;
      workArray[3] += t * expt * nAB * nAB /(a+b);  //20, 02
      workArray[4] += (-2 * t * expt * nAB + 2 * t * (t/a) * expT * nAB * nAB * nAB)/ (a + b);//30,21,12,03
      workArray[5] += (-2 * t * expt + 10 * t * (t/a) * expt * nAB * nAB
                       - 4 * t * t * t/(a * a) * expt * nAB * nAB * nAB * nAB)/(4*a*b*(a+b)); //31, 13

      if (n1+n2 < 4) continue;
      workArray[6] += -2 * t * expt * nAB * nAB / ((a+b) * (a+b)); //40, 22, 04
      workArray[7] +=  (4 * t * expt * nAB  - 8 * t * t * expt * nAB * nAB * nAB / a
                        + 2 * t * t * t/(a*a) * expt * pow(nAB, 5))/ ((a + b) * (a + b));
      //50, 41, 32, 23, 14, 05
      workArray[8] += (4 * t * expt - 32 * t * (t/a) * expt * nAB * nAB
                       + 26 * t * t * (t/a/a) * expt * pow(nAB, 4)
                       - 4 * pow(t,4)/(a*a*a) * expt * pow(nAB, 6))/ ((a + b) * (a + b) * 4 * a * b);
      //51, 33, 15
      if (n1+n2 < 6) continue;
      workArray[9]  +=  (6 * t * expt * nAB * nAB - 6 * t * t * expt * pow(nAB, 4)
                         + t * t * (t/a/a) * expt * pow(nAB, 6))/ pow(a+b,3); //40, 22, 04
      workArray[10] +=  (4 * t * expt * nAB  - 8 * t * t * expt * nAB * nAB * nAB / a
                        + 2 * t * t * t/(a*a) * expt * pow(nAB, 5))/ ((a + b) * (a + b));
      //50, 41, 32, 23, 14, 05
      workArray[11] += (4 * t * expt - 32 * t * (t/a) * expt * nAB * nAB
                       + 26 * t * t * (t/a/a) * expt * pow(nAB, 4)
                       - 4 * pow(t,4)/(a*a*a) * expt * pow(nAB, 6))/ ((a + b) * (a + b) * 4 * a * b);
    }

  }
}

