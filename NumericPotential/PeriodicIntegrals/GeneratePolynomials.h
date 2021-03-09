#pragma once

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
