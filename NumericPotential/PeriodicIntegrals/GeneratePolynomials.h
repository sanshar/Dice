#pragma once
#include <complex>
#include "CxMemoryStack.h"


using namespace std;

double getHermiteReciprocal(int l, double* pout,
                          double Gx, double Gy, double Gz,
                          double Tx, double Ty, double Tz,
                          double exponentVal,
                          double Scale) ;

double getSphReciprocal(int la, int lb, double* pOut,
                        double* pSpha, double* pSphb,
                        double Gx, double Gy, double Gz,
                        double Tx, double Ty, double Tz,
                        double exponentVal,
                        double Scale);
double getSphReciprocal3(int la, int lb, int lc, double* pOut,
			 double* pSpha, double* pSphb, double* pSphc,
			 double Gx, double Gy, double Gz,
			 double Fx, double Fy, double Fz,
			 double TABx, double TABy, double TABz,
			 double TACx, double TACy, double TACz,
			 double exponentVal,
			 double Scale);
double getSphReciprocal2(int la, int lb, double* pOut,
			 double* pSpha, double* pSphb, 
			 double Gx, double Gy, double Gz,
			 double Fx, double Fy, double Fz,
			 double TAx, double TAy, double TAz,
			 double TBx, double TBy, double TBz,
			 double exponentVal,
			 double Scale);
  
double getSphReciprocal3sin(int la, int lb, double* pOut,
			    double* pSpha1, double* pSpha2,
			    double* pSphb1, double* pSphb2, 
			    double Gx, double Gy, double Gz,
			    double Fx, double Fy, double Fz,
			    double Ax, double Ay, double Az,
			    double Bx, double By, double Bz,
			    double exponentVal,
			    double Scale) ;
double getSphReciprocal3cos(int la, int lb, double* pOut,
			    double* pSpha1, double* pSpha2,
			    double* pSphb1, double* pSphb2, 
			    double Gx, double Gy, double Gz,
			    double Fx, double Fy, double Fz,
			    double Ax, double Ay, double Az,
			    double Bx, double By, double Bz,
			    double exponentVal,
			    double Scale) ;

double getSphRealRecursion(int la, int lb, double* pOutCos, double* pOutSin,
			   double Gx, double Gy, double Gz,
			   double Qx, double Qy, double Qz,
			   double Ax, double Ay, double Az,
			   double Bx, double By, double Bz,
			   double alpha, double a, double b,
			   double exponentVal,
			   double Scale, ct::FMemoryStack2& Mem) ;
