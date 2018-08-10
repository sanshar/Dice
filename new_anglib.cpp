/*                                                                           
  Developed by Roberto Olivares-Amaya and Garnet K.-L. Chan, 2012
  Copyright (c) 2012, Garnet K.-L. Chan
  
  This program is integrated in Molpro with the permission of
  Sandeep Sharma, Garnet K.-L. Chan and Roberto Olivares-Amaya
*/
#include <stdio.h>
#include <stdlib.h>
#include "new_anglib.h"
#include <cmath>
#include <algorithm>
#include "global.h"

#include <iostream>
using namespace std;

double nine_j(int na, int nb, int nc, int nd, int ne, int nf, int ng, int nh, int ni){
//In this case, we are not dividing by two because that will be done by the
//sixj routine.

  //Initializing
  double a=na;
  double b=nb;
  double c=nc;
  double d=nd;
  double e=ne;
  double f=nf;
  double g=ng;
  double h=nh;
  double i=ni;
	double ninej=0.0;

  //Checking triangle rules
  if(na+nb < nc || abs(na-nb) > nc)
    return ninej;
  if(nd + ne < nf || abs(nd-ne) > nf)
    return ninej;
  if(ng+nh < ni || abs(ng-nh) > ni)
    return ninej;
  if(na+nd < ng || abs(na-nd) > ng)
    return ninej;
  if(nb+ne < nh || abs(nb-ne) > nh)
    return ninej;
  if(nc+nf < ni || abs(nc-nf) > ni)
    return ninej;

  double  num = 0.0;
  double num1 = 0.0;
  double num2 = 0.0;
  double num3 = 0.0;
  double num4 = 0.0;
  int kmin =  max(max(abs(h-d), abs(b-f)), abs(a-i));
  int kmax =  min(min( h + d, b + f), a + i);
  int k;
  
  //pout << "kmin " << kmin << endl;
  //pout << "kmax " << kmax << endl;
  
  for (k = kmin; k <= kmax; k++) {
     num1 = k+1;
     num2 = six_j(a, b, c, f, i, k);
     num3 = six_j(d, e, f, b, k, h);
     num4 = six_j(g, h, i, k, a, d);
     num=mone(k)*num1*num2*num3*num4;
     ninej = ninej + num;
  }
  return ninej;
}

double six_j(int na, int nb, int nc, int nd, int ne, int nf){

   //Initializing
	double sixj=0.0;
  if((na+nb)%2 != nc%2)
    return sixj;
  if((nc+nd)%2 != ne%2)
    return sixj;
  if((na+ne)%2 != nf%2)
    return sixj;
  if((nb+nd)%2 != nf%2)
    return sixj;
  if(na + nb < nc || abs(na-nb)>nc)
    return sixj;
  if(nc+nd<ne || abs(nc-nd)>ne)
    return sixj;
  if(na+ne<nf || abs(na-ne)>nf)
    return sixj;
  if(nb+nd<nf || abs(nb-nd)>nf)
    return sixj;

  //Converting to half its value
  double a=na/2.;
  double b=nb/2.;
  double c=nc/2.;
  double d=nd/2.;
  double e=ne/2.;
  double f=nf/2.;

  double num1 = j6_delta(a, b, c);
  double num2 = j6_delta(c, d, e);
  double num3 = j6_delta(b, d, f);
  double den1 = j6_delta(a, e, f);

  double pref = num1*num2*num3/den1;

  double square = square_six(a, b, c, d, e, f);

  sixj=pref*square;

  return sixj;
}
// six_j

double three_j(int j1, int j2, int j3, int m1, int m2, int m3) {
   double cleb =0.0;
	double threej = 0.0;
   double fj1, fj2, fj3, fm3;
   fj1 = j1/2.;
   fj2 = j2/2.;
   fj3 = j3/2.;
   fm3 = m3/2.;
   cleb = clebsch(j1, m1, j2, m2, j3, m3);
   threej = mone(fj1-fj2+fm3)*cleb/sqrt(2*fj3+1);
   return threej;
}

double clebsch(int nj1, int nm1, int nj2, int nm2, int nj3, int nm3) {
   double j1, j2, j3;
   double m1, m2, m3;

   //Converting to half its value
   j1=nj1/2.;
   j2=nj2/2.;
   j3=nj3/2.;
   m1=nm1/2.;
   m2=nm2/2.;
   m3=nm3/2.;

   double cleb=0.0;
   if ( j1 < 0 || j2 < 0 || j3 < 0 || abs(m1) > j1 || abs(m2) > j2 ||
      abs(m3) > j3 || j1 + j2 < j3 || abs(j1-j2) > j3 || m1 + m2 != m3) {

      cleb=0.0;
   }
   else
   {
      double factor = 0.0;
      double sum = 0.0;
      int t;

      double num1 = pow(2*j3+1,2);
      double num2 = fbinom(j1+j2+j3+1, j1+j2-j3);
      double num3 = fbinom(2*j3, j3+m3);
      double den1 = (2*j1+1);
      double den2 = (2*j2+1);
      double den3 = fbinom(j1+j2+j3+1, j1-j2+j3);
      double den4 = fbinom(j1+j2+j3+1, j2-j1+j3);
      double den5 = fbinom(2*j1, j1+m1);
      double den6 = fbinom(2*j2, j2+m2);

      double num = num1*num2*num3;
      double den = den1*den2*den3*den4*den5*den6;
      factor = sqrt(num/den);

      double mint = max(max(0., j1-m1-(j3-m3)), j2 + m2 - (j3 + m3));
      double maxt = min(min(j1-m1, j2+m2),j1+j2-j3);

      //pout << "mint " << mint << endl;
      //pout << "maxt " << maxt << endl;
      double bin1;
      double bin2;
      double bin3;
      for (t=mint; t<=maxt; t++) {
         bin1=fbinom(j1+j2-j3, t);
         bin2=fbinom(j3-m3,     j1-m1-t);
         bin3=fbinom(j3+m3,     j2+m2-t);
         sum = sum + mone(t)*bin1*bin2*bin3;

         //pout << "t " << t << endl;
         //pout << "sum " << sum << endl;
         //pout << "bin1 " << bin1 << endl;
         //pout << "bin2 " << bin2 << endl;
         //pout << "bin3 " << bin3 << endl;
      }

      cleb = factor*sum;
      //pout << "factor: " << factor << endl;
      //pout << "sum: " << sum << endl;
      //pout << "Clebsch: " << cleb << endl;
   }
      return cleb;
}



double j6_delta(double a, double b, double c) {
	double prefac = 0.0;

   double den1 = fbinom(a+b+c+1,a+b-c);
   //den2 can be substituted to just den=2*c+1, since binom(2*c+1, 2*c) = 2*c+1
   //double den2 = fbinom(2*c+1, 2*c);
   double den2 = 2*c+1;
   double den3 = fbinom(2*c, b+c-a);

   double den = den1*den2*den3;
   prefac = 1/sqrt(den);
	return prefac;
}

double square_six(double a, double b, double c, double d, double e, double f){
/*
[a b c]
[d e f]
*/

   int nmin = max(max(max(a+e+f, b+d+f), c+d+e), a+b+c);
   int nmax = min(min(a+b+d+e, a+c+d+f), b+c+e+f);
   int n;
   double num1=0.0;
   double num2=0.0;
   double num3=0.0;
   double num4=0.0;
   double num=0.0;
   double sum=0.0;
   for (n=nmin; n<=nmax; n++){
      num1=fbinom(n+1, n-a-e-f);
      num2=fbinom(a+e-f, n-b-d-f);
      num3=fbinom(a-e+f, n-c-d-e);
      num4=fbinom(-a+e+f, n-a-b-c);

      num=mone(n)*num1*num2*num3*num4;

      sum=sum+num;
   }
   return sum;
}

















double facto(double n) {
	double fac;
   int nint;
   nint = get_cast(n);
	fac=1.0;
	int i;
	if (n==0 || n==1)
		return fac;
	for (i=2; i<=nint; i++)
		fac *= i;
	return fac;
}

int mone(double n) {
	int value;
   int nint;
   nint = get_cast(n);
   //pout << "nint %2 " << nint %2 << endl;
	if (nint % 2 == 0)
		value = 1;
	else
		value = -1;
	return (value);
}

int get_cast(double x) {
   int i;
   //i = (x / (int) x >= 1) ? (int) x : (int) x + 1 ;
   i = (int) x;
   //pout << "x " << x << " i " << i << endl;
   return i;
}

double fbinom(double dn, double dr)
{
  double res;
  int n = get_cast(dn);
  int r = get_cast(dr);


  if(n==r || r==0)
  {
    res = 1.0;
  }
  else if (r==1)
    res = n;
  else
    res = 1.0*n/(n-r)*fbinom((double)n-1,(double)r);
//    pout << n << " " << r<< " -> " << res << endl;
  return res;
}
