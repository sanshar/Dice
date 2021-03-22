#include <cmath>
#include <iostream>
#include <algorithm>
#include "CxMemoryStack.h"
#include "GeneratePolynomials.h"
#include "CxAlgebra.h"
#include "IrAmrr.h"

double getHermiteReciprocal(int l, double* pOut,
                          double Gx, double Gy, double Gz,
                          double Tx, double Ty, double Tz,
                          double exponentVal,
                          double Scale) {
  
  double ExpVal = exponentVal;//exp(-exponentVal)/exponentVal; 
  //double ExpVal = exp(-exponentVal)/exponentVal; 
  if (l == 0) {
    pOut[0] += Scale * cos((Gx * Tx + Gy * Ty + Gz * Tz)) * ExpVal;
    return 1.0;
  }
  else if (l == 1) {
    double SinVal = -sin((Gx * Tx + Gy * Ty + Gz * Tz)) * ExpVal * Scale;
    pOut[0] += (Gx) * SinVal;
    pOut[1] += (Gy) * SinVal;
    pOut[2] += (Gz) * SinVal;
    return std::max(std::abs(Gx), std::max(std::abs(Gy), std::abs(Gz)));
  }
  else if (l == 2) {
    double CosVal = -cos((Gx * Tx + Gy * Ty + Gz * Tz)) * ExpVal * Scale;
    pOut[0] +=  CosVal * Gx * Gx;
    pOut[1] +=  CosVal * Gy * Gy;
    pOut[2] +=  CosVal * Gz * Gz;
    pOut[3] +=  CosVal * Gx * Gy;
    pOut[4] +=  CosVal * Gx * Gz;
    pOut[5] +=  CosVal * Gy * Gz;
    return std::max(Gx*Gx, std::max(Gy*Gy, Gz*Gz));
  }
  else if (l == 3) {
    double SinVal = sin((Gx * Tx + Gy * Ty + Gz * Tz)) * ExpVal * Scale;
    pOut[0] +=  SinVal * Gx * Gx * Gx; //3 0 0
    pOut[1] +=  SinVal * Gy * Gy * Gy; //0 3 0
    pOut[2] +=  SinVal * Gz * Gz * Gz; //0 0 3
    pOut[3] +=  SinVal * Gx * Gy * Gy; //1 2 0
    pOut[4] +=  SinVal * Gx * Gz * Gz; //1 0 2
    pOut[5] +=  SinVal * Gx * Gx * Gy; //2 1 0
    pOut[6] +=  SinVal * Gy * Gz * Gz; //0 1 2
    pOut[7] +=  SinVal * Gx * Gx * Gz; //2 0 1
    pOut[8] +=  SinVal * Gy * Gy * Gz; //0 2 1
    pOut[9] +=  SinVal * Gx * Gy * Gz; //1 1 1
    return std::max(std::abs(Gx*Gx*Gx), std::max(std::abs(Gy*Gy*Gy), std::abs(Gz*Gz*Gz)));
  }
  else if (l == 4) {
    double Gx0=1., Gx1, Gx2, Gx3, Gx4;
    double Gy0=1., Gy1, Gy2, Gy3, Gy4;
    double Gz0=1., Gz1, Gz2, Gz3, Gz4;
    double Cosval = cos((Gx * Tx + Gy * Ty + Gz * Tz)) * ExpVal * Scale;
    Gx1 = Gx * Gx0; Gx2 = Gx * Gx1; Gx3 = Gx * Gx2; Gx4 = Gx * Gx3;
    Gy1 = Gy * Gy0; Gy2 = Gy * Gy1; Gy3 = Gy * Gy2; Gy4 = Gy * Gy3;
    Gz1 = Gz * Gz0; Gz2 = Gz * Gz1; Gz3 = Gz * Gz2; Gz4 = Gz * Gz3;
    pOut[ 0 ]+= Cosval * Gx4 * Gy0 * Gz0;
    pOut[ 1 ]+= Cosval * Gx0 * Gy4 * Gz0;
    pOut[ 2 ]+= Cosval * Gx0 * Gy0 * Gz4;
    pOut[ 3 ]+= Cosval * Gx3 * Gy1 * Gz0;
    pOut[ 4 ]+= Cosval * Gx1 * Gy3 * Gz0;
    pOut[ 5 ]+= Cosval * Gx3 * Gy0 * Gz1;
    pOut[ 6 ]+= Cosval * Gx1 * Gy0 * Gz3;
    pOut[ 7 ]+= Cosval * Gx0 * Gy3 * Gz1;
    pOut[ 8 ]+= Cosval * Gx0 * Gy1 * Gz3;
    pOut[ 9 ]+= Cosval * Gx2 * Gy2 * Gz0;
    pOut[ 10 ]+= Cosval * Gx2 * Gy0 * Gz2;
    pOut[ 11 ]+= Cosval * Gx0 * Gy2 * Gz2;
    pOut[ 12 ]+= Cosval * Gx1 * Gy1 * Gz2;
    pOut[ 13 ]+= Cosval * Gx1 * Gy2 * Gz1;
    pOut[ 14 ]+= Cosval * Gx2 * Gy1 * Gz1;
    return std::max(std::abs(Gx4), std::max(std::abs(Gy4), std::abs(Gz4)));
  }
  else if (l == 5) {
    double Gx0=1., Gx1, Gx2, Gx4, Gx3, Gx5;
    double Gy0=1., Gy1, Gy2, Gy4, Gy3, Gy5;
    double Gz0=1., Gz1, Gz2, Gz4, Gz3, Gz5;
    double Sinval = -sin((Gx * Tx + Gy * Ty + Gz * Tz)) * ExpVal * Scale;
    Gx1 = Gx * Gx0; Gx2 = Gx * Gx1; Gx3 = Gx * Gx2; Gx4 = Gx * Gx3; Gx5 = Gx * Gx4;
    Gy1 = Gy * Gy0; Gy2 = Gy * Gy1; Gy3 = Gy * Gy2; Gy4 = Gy * Gy3; Gy5 = Gy * Gy4;
    Gz1 = Gz * Gz0; Gz2 = Gz * Gz1; Gz3 = Gz * Gz2; Gz4 = Gz * Gz3; Gz5 = Gz * Gz4;
    pOut[ 0 ]+= Sinval * Gx5 * Gy0 * Gz0;
    pOut[ 1 ]+= Sinval * Gx0 * Gy5 * Gz0;
    pOut[ 2 ]+= Sinval * Gx0 * Gy0 * Gz5;
    pOut[ 3 ]+= Sinval * Gx1 * Gy4 * Gz0;
    pOut[ 4 ]+= Sinval * Gx1 * Gy0 * Gz4;
    pOut[ 5 ]+= Sinval * Gx4 * Gy1 * Gz0;
    pOut[ 6 ]+= Sinval * Gx0 * Gy1 * Gz4;
    pOut[ 7 ]+= Sinval * Gx4 * Gy0 * Gz1;
    pOut[ 8 ]+= Sinval * Gx0 * Gy4 * Gz1;
    pOut[ 9 ]+= Sinval * Gx3 * Gy2 * Gz0;
    pOut[ 10 ]+= Sinval * Gx3 * Gy0 * Gz2;
    pOut[ 11 ]+= Sinval * Gx2 * Gy3 * Gz0;
    pOut[ 12 ]+= Sinval * Gx0 * Gy3 * Gz2;
    pOut[ 13 ]+= Sinval * Gx2 * Gy0 * Gz3;
    pOut[ 14 ]+= Sinval * Gx0 * Gy2 * Gz3;
    pOut[ 15 ]+= Sinval * Gx3 * Gy1 * Gz1;
    pOut[ 16 ]+= Sinval * Gx1 * Gy3 * Gz1;
    pOut[ 17 ]+= Sinval * Gx1 * Gy1 * Gz3;
    pOut[ 18 ]+= Sinval * Gx1 * Gy2 * Gz2;
    pOut[ 19 ]+= Sinval * Gx2 * Gy1 * Gz2;
    pOut[ 20 ]+= Sinval * Gx2 * Gy2 * Gz1;
    return std::max(std::abs(Gx5), std::max(std::abs(Gy5), std::abs(Gz5)));
    //return std::max(Gx5, std::max(Gy5, Gz5));
  }
  else if (l == 6) {
    double Gx0=1., Gx1, Gx2, Gx4, Gx3, Gx5, Gx6;
    double Gy0=1., Gy1, Gy2, Gy4, Gy3, Gy5, Gy6;
    double Gz0=1., Gz1, Gz2, Gz4, Gz3, Gz5, Gz6;

    double Cosval = -cos((Gx * Tx + Gy * Ty + Gz * Tz)) * ExpVal * Scale;
    
    Gx1 = Gx * Gx0; Gx2 = Gx * Gx1; Gx3 = Gx * Gx2; Gx4 = Gx * Gx3; Gx5 = Gx * Gx4; Gx6 = Gx * Gx5;
    Gy1 = Gy * Gy0; Gy2 = Gy * Gy1; Gy3 = Gy * Gy2; Gy4 = Gy * Gy3; Gy5 = Gy * Gy4; Gy6 = Gy * Gy5;
    Gz1 = Gz * Gz0; Gz2 = Gz * Gz1; Gz3 = Gz * Gz2; Gz4 = Gz * Gz3; Gz5 = Gz * Gz4; Gz6 = Gz * Gz5;
    
    pOut[ 0 ]+= Cosval * Gx6 * Gy0 * Gz0;
    pOut[ 1 ]+= Cosval * Gx0 * Gy6 * Gz0;
    pOut[ 2 ]+= Cosval * Gx0 * Gy0 * Gz6;
    pOut[ 3 ]+= Cosval * Gx5 * Gy1 * Gz0;
    pOut[ 4 ]+= Cosval * Gx1 * Gy5 * Gz0;
    pOut[ 5 ]+= Cosval * Gx5 * Gy0 * Gz1;
    pOut[ 6 ]+= Cosval * Gx1 * Gy0 * Gz5;
    pOut[ 7 ]+= Cosval * Gx0 * Gy5 * Gz1;
    pOut[ 8 ]+= Cosval * Gx0 * Gy1 * Gz5;
    pOut[ 9 ]+= Cosval * Gx4 * Gy2 * Gz0;
    pOut[ 10 ]+= Cosval * Gx4 * Gy0 * Gz2;
    pOut[ 11 ]+= Cosval * Gx2 * Gy4 * Gz0;
    pOut[ 12 ]+= Cosval * Gx2 * Gy0 * Gz4;
    pOut[ 13 ]+= Cosval * Gx0 * Gy4 * Gz2;
    pOut[ 14 ]+= Cosval * Gx0 * Gy2 * Gz4;
    pOut[ 15 ]+= Cosval * Gx3 * Gy3 * Gz0;
    pOut[ 16 ]+= Cosval * Gx3 * Gy0 * Gz3;
    pOut[ 17 ]+= Cosval * Gx0 * Gy3 * Gz3;
    pOut[ 18 ]+= Cosval * Gx1 * Gy1 * Gz4;
    pOut[ 19 ]+= Cosval * Gx1 * Gy4 * Gz1;
    pOut[ 20 ]+= Cosval * Gx4 * Gy1 * Gz1;
    pOut[ 21 ]+= Cosval * Gx3 * Gy1 * Gz2;
    pOut[ 22 ]+= Cosval * Gx1 * Gy3 * Gz2;
    pOut[ 23 ]+= Cosval * Gx3 * Gy2 * Gz1;
    pOut[ 24 ]+= Cosval * Gx1 * Gy2 * Gz3;
    pOut[ 25 ]+= Cosval * Gx2 * Gy3 * Gz1;
    pOut[ 26 ]+= Cosval * Gx2 * Gy1 * Gz3;
    pOut[ 27 ]+= Cosval * Gx2 * Gy2 * Gz2;
    return std::max(std::abs(Gx6), std::max(std::abs(Gy6), std::abs(Gz6)));
    //return std::max(Gx6, std::max(Gy6, Gz6));
  }
  else if (l == 7) {
    double Gx0=1., Gx1, Gx2, Gx4, Gx3, Gx5, Gx6, Gx7;
    double Gy0=1., Gy1, Gy2, Gy4, Gy3, Gy5, Gy6, Gy7;
    double Gz0=1., Gz1, Gz2, Gz4, Gz3, Gz5, Gz6, Gz7;
    double Sinval = sin((Gx * Tx + Gy * Ty + Gz * Tz)) * ExpVal * Scale;
    Gx1 = Gx * Gx0; Gx2 = Gx * Gx1; Gx3 = Gx * Gx2; Gx4 = Gx * Gx3; Gx5 = Gx * Gx4; Gx6 = Gx * Gx5; Gx7 = Gx * Gx6;
    Gy1 = Gy * Gy0; Gy2 = Gy * Gy1; Gy3 = Gy * Gy2; Gy4 = Gy * Gy3; Gy5 = Gy * Gy4; Gy6 = Gy * Gy5; Gy7 = Gy * Gy6;
    Gz1 = Gz * Gz0; Gz2 = Gz * Gz1; Gz3 = Gz * Gz2; Gz4 = Gz * Gz3; Gz5 = Gz * Gz4; Gz6 = Gz * Gz5; Gz7 = Gz * Gz6;
    pOut[ 0 ]+= Sinval * Gx7 * Gy0 * Gz0;
    pOut[ 1 ]+= Sinval * Gx0 * Gy7 * Gz0;
    pOut[ 2 ]+= Sinval * Gx0 * Gy0 * Gz7;
    pOut[ 3 ]+= Sinval * Gx1 * Gy6 * Gz0;
    pOut[ 4 ]+= Sinval * Gx1 * Gy0 * Gz6;
    pOut[ 5 ]+= Sinval * Gx6 * Gy1 * Gz0;
    pOut[ 6 ]+= Sinval * Gx0 * Gy1 * Gz6;
    pOut[ 7 ]+= Sinval * Gx6 * Gy0 * Gz1;
    pOut[ 8 ]+= Sinval * Gx0 * Gy6 * Gz1;
    pOut[ 9 ]+= Sinval * Gx5 * Gy2 * Gz0;
    pOut[ 10 ]+= Sinval * Gx5 * Gy0 * Gz2;
    pOut[ 11 ]+= Sinval * Gx2 * Gy5 * Gz0;
    pOut[ 12 ]+= Sinval * Gx0 * Gy5 * Gz2;
    pOut[ 13 ]+= Sinval * Gx2 * Gy0 * Gz5;
    pOut[ 14 ]+= Sinval * Gx0 * Gy2 * Gz5;
    pOut[ 15 ]+= Sinval * Gx3 * Gy4 * Gz0;
    pOut[ 16 ]+= Sinval * Gx3 * Gy0 * Gz4;
    pOut[ 17 ]+= Sinval * Gx4 * Gy3 * Gz0;
    pOut[ 18 ]+= Sinval * Gx0 * Gy3 * Gz4;
    pOut[ 19 ]+= Sinval * Gx4 * Gy0 * Gz3;
    pOut[ 20 ]+= Sinval * Gx0 * Gy4 * Gz3;
    pOut[ 21 ]+= Sinval * Gx5 * Gy1 * Gz1;
    pOut[ 22 ]+= Sinval * Gx1 * Gy5 * Gz1;
    pOut[ 23 ]+= Sinval * Gx1 * Gy1 * Gz5;
    pOut[ 24 ]+= Sinval * Gx1 * Gy4 * Gz2;
    pOut[ 25 ]+= Sinval * Gx1 * Gy2 * Gz4;
    pOut[ 26 ]+= Sinval * Gx4 * Gy1 * Gz2;
    pOut[ 27 ]+= Sinval * Gx2 * Gy1 * Gz4;
    pOut[ 28 ]+= Sinval * Gx4 * Gy2 * Gz1;
    pOut[ 29 ]+= Sinval * Gx2 * Gy4 * Gz1;
    pOut[ 30 ]+= Sinval * Gx3 * Gy3 * Gz1;
    pOut[ 31 ]+= Sinval * Gx3 * Gy1 * Gz3;
    pOut[ 32 ]+= Sinval * Gx1 * Gy3 * Gz3;
    pOut[ 33 ]+= Sinval * Gx3 * Gy2 * Gz2;
    pOut[ 34 ]+= Sinval * Gx2 * Gy3 * Gz2;
    pOut[ 35 ]+= Sinval * Gx2 * Gy2 * Gz3;
    return std::max(std::abs(Gx7), std::max(std::abs(Gy7), std::abs(Gz7)));
    //return std::max(Gx7, std::max(Gy7, Gz7));
  }
  else if (l == 8) {
    double Gx0=1., Gx1, Gx2, Gx4, Gx3, Gx5, Gx6, Gx7, Gx8;
    double Gy0=1., Gy1, Gy2, Gy4, Gy3, Gy5, Gy6, Gy7, Gy8;
    double Gz0=1., Gz1, Gz2, Gz4, Gz3, Gz5, Gz6, Gz7, Gz8;
    double Cosval = cos((Gx * Tx + Gy * Ty + Gz * Tz)) * ExpVal * Scale;
    Gx1 = Gx * Gx0; Gx2 = Gx * Gx1; Gx3 = Gx * Gx2; Gx4 = Gx * Gx3; Gx5 = Gx * Gx4; Gx6 = Gx * Gx5; Gx7 = Gx * Gx6; Gx8 = Gx * Gx7;
    Gy1 = Gy * Gy0; Gy2 = Gy * Gy1; Gy3 = Gy * Gy2; Gy4 = Gy * Gy3; Gy5 = Gy * Gy4; Gy6 = Gy * Gy5; Gy7 = Gy * Gy6; Gy8 = Gy * Gy7;
    Gz1 = Gz * Gz0; Gz2 = Gz * Gz1; Gz3 = Gz * Gz2; Gz4 = Gz * Gz3; Gz5 = Gz * Gz4; Gz6 = Gz * Gz5; Gz7 = Gz * Gz6; Gz8 = Gz * Gz7;
    pOut[ 0 ]+= Cosval * Gx8 * Gy0 * Gz0;
    pOut[ 1 ]+= Cosval * Gx0 * Gy8 * Gz0;
    pOut[ 2 ]+= Cosval * Gx0 * Gy0 * Gz8;
    pOut[ 3 ]+= Cosval * Gx7 * Gy1 * Gz0;
    pOut[ 4 ]+= Cosval * Gx1 * Gy7 * Gz0;
    pOut[ 5 ]+= Cosval * Gx7 * Gy0 * Gz1;
    pOut[ 6 ]+= Cosval * Gx1 * Gy0 * Gz7;
    pOut[ 7 ]+= Cosval * Gx0 * Gy7 * Gz1;
    pOut[ 8 ]+= Cosval * Gx0 * Gy1 * Gz7;
    pOut[ 9 ]+= Cosval * Gx6 * Gy2 * Gz0;
    pOut[ 10 ]+= Cosval * Gx6 * Gy0 * Gz2;
    pOut[ 11 ]+= Cosval * Gx2 * Gy6 * Gz0;
    pOut[ 12 ]+= Cosval * Gx2 * Gy0 * Gz6;
    pOut[ 13 ]+= Cosval * Gx0 * Gy6 * Gz2;
    pOut[ 14 ]+= Cosval * Gx0 * Gy2 * Gz6;
    pOut[ 15 ]+= Cosval * Gx5 * Gy3 * Gz0;
    pOut[ 16 ]+= Cosval * Gx3 * Gy5 * Gz0;
    pOut[ 17 ]+= Cosval * Gx5 * Gy0 * Gz3;
    pOut[ 18 ]+= Cosval * Gx3 * Gy0 * Gz5;
    pOut[ 19 ]+= Cosval * Gx0 * Gy5 * Gz3;
    pOut[ 20 ]+= Cosval * Gx0 * Gy3 * Gz5;
    pOut[ 21 ]+= Cosval * Gx4 * Gy4 * Gz0;
    pOut[ 22 ]+= Cosval * Gx4 * Gy0 * Gz4;
    pOut[ 23 ]+= Cosval * Gx0 * Gy4 * Gz4;
    pOut[ 24 ]+= Cosval * Gx1 * Gy1 * Gz6;
    pOut[ 25 ]+= Cosval * Gx1 * Gy6 * Gz1;
    pOut[ 26 ]+= Cosval * Gx6 * Gy1 * Gz1;
    pOut[ 27 ]+= Cosval * Gx5 * Gy1 * Gz2;
    pOut[ 28 ]+= Cosval * Gx1 * Gy5 * Gz2;
    pOut[ 29 ]+= Cosval * Gx5 * Gy2 * Gz1;
    pOut[ 30 ]+= Cosval * Gx1 * Gy2 * Gz5;
    pOut[ 31 ]+= Cosval * Gx2 * Gy5 * Gz1;
    pOut[ 32 ]+= Cosval * Gx2 * Gy1 * Gz5;
    pOut[ 33 ]+= Cosval * Gx3 * Gy1 * Gz4;
    pOut[ 34 ]+= Cosval * Gx1 * Gy3 * Gz4;
    pOut[ 35 ]+= Cosval * Gx3 * Gy4 * Gz1;
    pOut[ 36 ]+= Cosval * Gx1 * Gy4 * Gz3;
    pOut[ 37 ]+= Cosval * Gx4 * Gy3 * Gz1;
    pOut[ 38 ]+= Cosval * Gx4 * Gy1 * Gz3;
    pOut[ 39 ]+= Cosval * Gx4 * Gy2 * Gz2;
    pOut[ 40 ]+= Cosval * Gx2 * Gy4 * Gz2;
    pOut[ 41 ]+= Cosval * Gx2 * Gy2 * Gz4;
    pOut[ 42 ]+= Cosval * Gx3 * Gy3 * Gz2;
    pOut[ 43 ]+= Cosval * Gx3 * Gy2 * Gz3;
    pOut[ 44 ]+= Cosval * Gx2 * Gy3 * Gz3;
    return std::max(Gx8, std::max(Gy8, Gz8));
  }
  else if (l == 9) {
    double Gx0=1., Gx1, Gx2, Gx4, Gx3, Gx5, Gx6, Gx7, Gx8, Gx9;
    double Gy0=1., Gy1, Gy2, Gy4, Gy3, Gy5, Gy6, Gy7, Gy8, Gy9;
    double Gz0=1., Gz1, Gz2, Gz4, Gz3, Gz5, Gz6, Gz7, Gz8, Gz9;
    double Sinval = -sin((Gx * Tx + Gy * Ty + Gz * Tz)) * ExpVal * Scale;
    Gx1 = Gx * Gx0; Gx2 = Gx * Gx1; Gx3 = Gx * Gx2; Gx4 = Gx * Gx3; Gx5 = Gx * Gx4; Gx6 = Gx * Gx5; Gx7 = Gx * Gx6; Gx8 = Gx * Gx7; Gx9 = Gx * Gx8;
    Gy1 = Gy * Gy0; Gy2 = Gy * Gy1; Gy3 = Gy * Gy2; Gy4 = Gy * Gy3; Gy5 = Gy * Gy4; Gy6 = Gy * Gy5; Gy7 = Gy * Gy6; Gy8 = Gy * Gy7; Gy9 = Gy * Gy8;
    Gz1 = Gz * Gz0; Gz2 = Gz * Gz1; Gz3 = Gz * Gz2; Gz4 = Gz * Gz3; Gz5 = Gz * Gz4; Gz6 = Gz * Gz5; Gz7 = Gz * Gz6; Gz8 = Gz * Gz7; Gz9 = Gz * Gz8;
    pOut[ 0 ]+= Sinval * Gx9 * Gy0 * Gz0;
    pOut[ 1 ]+= Sinval * Gx0 * Gy9 * Gz0;
    pOut[ 2 ]+= Sinval * Gx0 * Gy0 * Gz9;
    pOut[ 3 ]+= Sinval * Gx1 * Gy8 * Gz0;
    pOut[ 4 ]+= Sinval * Gx1 * Gy0 * Gz8;
    pOut[ 5 ]+= Sinval * Gx8 * Gy1 * Gz0;
    pOut[ 6 ]+= Sinval * Gx0 * Gy1 * Gz8;
    pOut[ 7 ]+= Sinval * Gx8 * Gy0 * Gz1;
    pOut[ 8 ]+= Sinval * Gx0 * Gy8 * Gz1;
    pOut[ 9 ]+= Sinval * Gx7 * Gy2 * Gz0;
    pOut[ 10 ]+= Sinval * Gx7 * Gy0 * Gz2;
    pOut[ 11 ]+= Sinval * Gx2 * Gy7 * Gz0;
    pOut[ 12 ]+= Sinval * Gx0 * Gy7 * Gz2;
    pOut[ 13 ]+= Sinval * Gx2 * Gy0 * Gz7;
    pOut[ 14 ]+= Sinval * Gx0 * Gy2 * Gz7;
    pOut[ 15 ]+= Sinval * Gx3 * Gy6 * Gz0;
    pOut[ 16 ]+= Sinval * Gx3 * Gy0 * Gz6;
    pOut[ 17 ]+= Sinval * Gx6 * Gy3 * Gz0;
    pOut[ 18 ]+= Sinval * Gx0 * Gy3 * Gz6;
    pOut[ 19 ]+= Sinval * Gx6 * Gy0 * Gz3;
    pOut[ 20 ]+= Sinval * Gx0 * Gy6 * Gz3;
    pOut[ 21 ]+= Sinval * Gx5 * Gy4 * Gz0;
    pOut[ 22 ]+= Sinval * Gx5 * Gy0 * Gz4;
    pOut[ 23 ]+= Sinval * Gx4 * Gy5 * Gz0;
    pOut[ 24 ]+= Sinval * Gx0 * Gy5 * Gz4;
    pOut[ 25 ]+= Sinval * Gx4 * Gy0 * Gz5;
    pOut[ 26 ]+= Sinval * Gx0 * Gy4 * Gz5;
    pOut[ 27 ]+= Sinval * Gx7 * Gy1 * Gz1;
    pOut[ 28 ]+= Sinval * Gx1 * Gy7 * Gz1;
    pOut[ 29 ]+= Sinval * Gx1 * Gy1 * Gz7;
    pOut[ 30 ]+= Sinval * Gx1 * Gy6 * Gz2;
    pOut[ 31 ]+= Sinval * Gx1 * Gy2 * Gz6;
    pOut[ 32 ]+= Sinval * Gx6 * Gy1 * Gz2;
    pOut[ 33 ]+= Sinval * Gx2 * Gy1 * Gz6;
    pOut[ 34 ]+= Sinval * Gx6 * Gy2 * Gz1;
    pOut[ 35 ]+= Sinval * Gx2 * Gy6 * Gz1;
    pOut[ 36 ]+= Sinval * Gx5 * Gy3 * Gz1;
    pOut[ 37 ]+= Sinval * Gx5 * Gy1 * Gz3;
    pOut[ 38 ]+= Sinval * Gx3 * Gy5 * Gz1;
    pOut[ 39 ]+= Sinval * Gx3 * Gy1 * Gz5;
    pOut[ 40 ]+= Sinval * Gx1 * Gy5 * Gz3;
    pOut[ 41 ]+= Sinval * Gx1 * Gy3 * Gz5;
    pOut[ 42 ]+= Sinval * Gx1 * Gy4 * Gz4;
    pOut[ 43 ]+= Sinval * Gx4 * Gy1 * Gz4;
    pOut[ 44 ]+= Sinval * Gx4 * Gy4 * Gz1;
    pOut[ 45 ]+= Sinval * Gx5 * Gy2 * Gz2;
    pOut[ 46 ]+= Sinval * Gx2 * Gy5 * Gz2;
    pOut[ 47 ]+= Sinval * Gx2 * Gy2 * Gz5;
    pOut[ 48 ]+= Sinval * Gx3 * Gy4 * Gz2;
    pOut[ 49 ]+= Sinval * Gx3 * Gy2 * Gz4;
    pOut[ 50 ]+= Sinval * Gx4 * Gy3 * Gz2;
    pOut[ 51 ]+= Sinval * Gx2 * Gy3 * Gz4;
    pOut[ 52 ]+= Sinval * Gx4 * Gy2 * Gz3;
    pOut[ 53 ]+= Sinval * Gx2 * Gy4 * Gz3;
    pOut[ 54 ]+= Sinval * Gx3 * Gy3 * Gz3;
    return std::max(std::abs(Gx9), std::max(std::abs(Gy9), std::abs(Gz9)));
    //return std::max(Gx9, std::max(Gy9, Gz9));
    
  }
  else if (l == 10) {
    double Gx0=1., Gx1, Gx2, Gx4, Gx3, Gx5, Gx6, Gx7, Gx8, Gx9, Gx10;
    double Gy0=1., Gy1, Gy2, Gy4, Gy3, Gy5, Gy6, Gy7, Gy8, Gy9, Gy10;
    double Gz0=1., Gz1, Gz2, Gz4, Gz3, Gz5, Gz6, Gz7, Gz8, Gz9, Gz10;
    double Cosval = -cos((Gx * Tx + Gy * Ty + Gz * Tz)) * ExpVal * Scale;
    Gx1 = Gx * Gx0; Gx2 = Gx * Gx1; Gx3 = Gx * Gx2; Gx4 = Gx * Gx3; Gx5 = Gx * Gx4; Gx6 = Gx * Gx5; Gx7 = Gx * Gx6; Gx8 = Gx * Gx7; Gx9 = Gx * Gx8; Gx10 = Gx * Gx9;
    Gy1 = Gy * Gy0; Gy2 = Gy * Gy1; Gy3 = Gy * Gy2; Gy4 = Gy * Gy3; Gy5 = Gy * Gy4; Gy6 = Gy * Gy5; Gy7 = Gy * Gy6; Gy8 = Gy * Gy7; Gy9 = Gy * Gy8; Gy10 = Gy * Gy9;
    Gz1 = Gz * Gz0; Gz2 = Gz * Gz1; Gz3 = Gz * Gz2; Gz4 = Gz * Gz3; Gz5 = Gz * Gz4; Gz6 = Gz * Gz5; Gz7 = Gz * Gz6; Gz8 = Gz * Gz7; Gz9 = Gz * Gz8; Gz10 = Gz * Gz9;
    pOut[ 0 ]+= Cosval * Gx10 * Gy0 * Gz0;
    pOut[ 1 ]+= Cosval * Gx0 * Gy10 * Gz0;
    pOut[ 2 ]+= Cosval * Gx0 * Gy0 * Gz10;
    pOut[ 3 ]+= Cosval * Gx9 * Gy1 * Gz0;
    pOut[ 4 ]+= Cosval * Gx1 * Gy9 * Gz0;
    pOut[ 5 ]+= Cosval * Gx9 * Gy0 * Gz1;
    pOut[ 6 ]+= Cosval * Gx1 * Gy0 * Gz9;
    pOut[ 7 ]+= Cosval * Gx0 * Gy9 * Gz1;
    pOut[ 8 ]+= Cosval * Gx0 * Gy1 * Gz9;
    pOut[ 9 ]+= Cosval * Gx8 * Gy2 * Gz0;
    pOut[ 10 ]+= Cosval * Gx8 * Gy0 * Gz2;
    pOut[ 11 ]+= Cosval * Gx2 * Gy8 * Gz0;
    pOut[ 12 ]+= Cosval * Gx2 * Gy0 * Gz8;
    pOut[ 13 ]+= Cosval * Gx0 * Gy8 * Gz2;
    pOut[ 14 ]+= Cosval * Gx0 * Gy2 * Gz8;
    pOut[ 15 ]+= Cosval * Gx7 * Gy3 * Gz0;
    pOut[ 16 ]+= Cosval * Gx3 * Gy7 * Gz0;
    pOut[ 17 ]+= Cosval * Gx7 * Gy0 * Gz3;
    pOut[ 18 ]+= Cosval * Gx3 * Gy0 * Gz7;
    pOut[ 19 ]+= Cosval * Gx0 * Gy7 * Gz3;
    pOut[ 20 ]+= Cosval * Gx0 * Gy3 * Gz7;
    pOut[ 21 ]+= Cosval * Gx6 * Gy4 * Gz0;
    pOut[ 22 ]+= Cosval * Gx6 * Gy0 * Gz4;
    pOut[ 23 ]+= Cosval * Gx4 * Gy6 * Gz0;
    pOut[ 24 ]+= Cosval * Gx4 * Gy0 * Gz6;
    pOut[ 25 ]+= Cosval * Gx0 * Gy6 * Gz4;
    pOut[ 26 ]+= Cosval * Gx0 * Gy4 * Gz6;
    pOut[ 27 ]+= Cosval * Gx5 * Gy5 * Gz0;
    pOut[ 28 ]+= Cosval * Gx5 * Gy0 * Gz5;
    pOut[ 29 ]+= Cosval * Gx0 * Gy5 * Gz5;
    pOut[ 30 ]+= Cosval * Gx1 * Gy1 * Gz8;
    pOut[ 31 ]+= Cosval * Gx1 * Gy8 * Gz1;
    pOut[ 32 ]+= Cosval * Gx8 * Gy1 * Gz1;
    pOut[ 33 ]+= Cosval * Gx7 * Gy1 * Gz2;
    pOut[ 34 ]+= Cosval * Gx1 * Gy7 * Gz2;
    pOut[ 35 ]+= Cosval * Gx7 * Gy2 * Gz1;
    pOut[ 36 ]+= Cosval * Gx1 * Gy2 * Gz7;
    pOut[ 37 ]+= Cosval * Gx2 * Gy7 * Gz1;
    pOut[ 38 ]+= Cosval * Gx2 * Gy1 * Gz7;
    pOut[ 39 ]+= Cosval * Gx3 * Gy1 * Gz6;
    pOut[ 40 ]+= Cosval * Gx1 * Gy3 * Gz6;
    pOut[ 41 ]+= Cosval * Gx3 * Gy6 * Gz1;
    pOut[ 42 ]+= Cosval * Gx1 * Gy6 * Gz3;
    pOut[ 43 ]+= Cosval * Gx6 * Gy3 * Gz1;
    pOut[ 44 ]+= Cosval * Gx6 * Gy1 * Gz3;
    pOut[ 45 ]+= Cosval * Gx5 * Gy1 * Gz4;
    pOut[ 46 ]+= Cosval * Gx1 * Gy5 * Gz4;
    pOut[ 47 ]+= Cosval * Gx5 * Gy4 * Gz1;
    pOut[ 48 ]+= Cosval * Gx1 * Gy4 * Gz5;
    pOut[ 49 ]+= Cosval * Gx4 * Gy5 * Gz1;
    pOut[ 50 ]+= Cosval * Gx4 * Gy1 * Gz5;
    pOut[ 51 ]+= Cosval * Gx6 * Gy2 * Gz2;
    pOut[ 52 ]+= Cosval * Gx2 * Gy6 * Gz2;
    pOut[ 53 ]+= Cosval * Gx2 * Gy2 * Gz6;
    pOut[ 54 ]+= Cosval * Gx5 * Gy3 * Gz2;
    pOut[ 55 ]+= Cosval * Gx3 * Gy5 * Gz2;
    pOut[ 56 ]+= Cosval * Gx5 * Gy2 * Gz3;
    pOut[ 57 ]+= Cosval * Gx3 * Gy2 * Gz5;
    pOut[ 58 ]+= Cosval * Gx2 * Gy5 * Gz3;
    pOut[ 59 ]+= Cosval * Gx2 * Gy3 * Gz5;
    pOut[ 60 ]+= Cosval * Gx4 * Gy4 * Gz2;
    pOut[ 61 ]+= Cosval * Gx4 * Gy2 * Gz4;
    pOut[ 62 ]+= Cosval * Gx2 * Gy4 * Gz4;
    pOut[ 63 ]+= Cosval * Gx3 * Gy3 * Gz4;
    pOut[ 64 ]+= Cosval * Gx3 * Gy4 * Gz3;
    pOut[ 65 ]+= Cosval * Gx4 * Gy3 * Gz3;
    return std::max(std::abs(Gx10), std::max(std::abs(Gy10), std::abs(Gz10)));
    //return std::max(Gx10, std::max(Gy10, Gz10));
    
  }
  else if (l == 11) {
    double Gx0=1., Gx1, Gx2, Gx4, Gx3, Gx5, Gx6, Gx7, Gx8, Gx9, Gx10, Gx11;
    double Gy0=1., Gy1, Gy2, Gy4, Gy3, Gy5, Gy6, Gy7, Gy8, Gy9, Gy10, Gy11;
    double Gz0=1., Gz1, Gz2, Gz4, Gz3, Gz5, Gz6, Gz7, Gz8, Gz9, Gz10, Gz11;
    double Sinval = sin((Gx * Tx + Gy * Ty + Gz * Tz)) * ExpVal * Scale;
    Gx1 = Gx * Gx0; Gx2 = Gx * Gx1; Gx3 = Gx * Gx2; Gx4 = Gx * Gx3; Gx5 = Gx * Gx4; Gx6 = Gx * Gx5; Gx7 = Gx * Gx6; Gx8 = Gx * Gx7; Gx9 = Gx * Gx8; Gx10 = Gx * Gx9; Gx11 = Gx * Gx10;
    Gy1 = Gy * Gy0; Gy2 = Gy * Gy1; Gy3 = Gy * Gy2; Gy4 = Gy * Gy3; Gy5 = Gy * Gy4; Gy6 = Gy * Gy5; Gy7 = Gy * Gy6; Gy8 = Gy * Gy7; Gy9 = Gy * Gy8; Gy10 = Gy * Gy9; Gy11 = Gy * Gy10;
    Gz1 = Gz * Gz0; Gz2 = Gz * Gz1; Gz3 = Gz * Gz2; Gz4 = Gz * Gz3; Gz5 = Gz * Gz4; Gz6 = Gz * Gz5; Gz7 = Gz * Gz6; Gz8 = Gz * Gz7; Gz9 = Gz * Gz8; Gz10 = Gz * Gz9; Gz11 = Gz * Gz10;
    pOut[ 0 ]+= Sinval * Gx11 * Gy0 * Gz0;
    pOut[ 1 ]+= Sinval * Gx0 * Gy11 * Gz0;
    pOut[ 2 ]+= Sinval * Gx0 * Gy0 * Gz11;
    pOut[ 3 ]+= Sinval * Gx1 * Gy10 * Gz0;
    pOut[ 4 ]+= Sinval * Gx1 * Gy0 * Gz10;
    pOut[ 5 ]+= Sinval * Gx10 * Gy1 * Gz0;
    pOut[ 6 ]+= Sinval * Gx0 * Gy1 * Gz10;
    pOut[ 7 ]+= Sinval * Gx10 * Gy0 * Gz1;
    pOut[ 8 ]+= Sinval * Gx0 * Gy10 * Gz1;
    pOut[ 9 ]+= Sinval * Gx9 * Gy2 * Gz0;
    pOut[ 10 ]+= Sinval * Gx9 * Gy0 * Gz2;
    pOut[ 11 ]+= Sinval * Gx2 * Gy9 * Gz0;
    pOut[ 12 ]+= Sinval * Gx0 * Gy9 * Gz2;
    pOut[ 13 ]+= Sinval * Gx2 * Gy0 * Gz9;
    pOut[ 14 ]+= Sinval * Gx0 * Gy2 * Gz9;
    pOut[ 15 ]+= Sinval * Gx3 * Gy8 * Gz0;
    pOut[ 16 ]+= Sinval * Gx3 * Gy0 * Gz8;
    pOut[ 17 ]+= Sinval * Gx8 * Gy3 * Gz0;
    pOut[ 18 ]+= Sinval * Gx0 * Gy3 * Gz8;
    pOut[ 19 ]+= Sinval * Gx8 * Gy0 * Gz3;
    pOut[ 20 ]+= Sinval * Gx0 * Gy8 * Gz3;
    pOut[ 21 ]+= Sinval * Gx7 * Gy4 * Gz0;
    pOut[ 22 ]+= Sinval * Gx7 * Gy0 * Gz4;
    pOut[ 23 ]+= Sinval * Gx4 * Gy7 * Gz0;
    pOut[ 24 ]+= Sinval * Gx0 * Gy7 * Gz4;
    pOut[ 25 ]+= Sinval * Gx4 * Gy0 * Gz7;
    pOut[ 26 ]+= Sinval * Gx0 * Gy4 * Gz7;
    pOut[ 27 ]+= Sinval * Gx5 * Gy6 * Gz0;
    pOut[ 28 ]+= Sinval * Gx5 * Gy0 * Gz6;
    pOut[ 29 ]+= Sinval * Gx6 * Gy5 * Gz0;
    pOut[ 30 ]+= Sinval * Gx0 * Gy5 * Gz6;
    pOut[ 31 ]+= Sinval * Gx6 * Gy0 * Gz5;
    pOut[ 32 ]+= Sinval * Gx0 * Gy6 * Gz5;
    pOut[ 33 ]+= Sinval * Gx9 * Gy1 * Gz1;
    pOut[ 34 ]+= Sinval * Gx1 * Gy9 * Gz1;
    pOut[ 35 ]+= Sinval * Gx1 * Gy1 * Gz9;
    pOut[ 36 ]+= Sinval * Gx1 * Gy8 * Gz2;
    pOut[ 37 ]+= Sinval * Gx1 * Gy2 * Gz8;
    pOut[ 38 ]+= Sinval * Gx8 * Gy1 * Gz2;
    pOut[ 39 ]+= Sinval * Gx2 * Gy1 * Gz8;
    pOut[ 40 ]+= Sinval * Gx8 * Gy2 * Gz1;
    pOut[ 41 ]+= Sinval * Gx2 * Gy8 * Gz1;
    pOut[ 42 ]+= Sinval * Gx7 * Gy3 * Gz1;
    pOut[ 43 ]+= Sinval * Gx7 * Gy1 * Gz3;
    pOut[ 44 ]+= Sinval * Gx3 * Gy7 * Gz1;
    pOut[ 45 ]+= Sinval * Gx3 * Gy1 * Gz7;
    pOut[ 46 ]+= Sinval * Gx1 * Gy7 * Gz3;
    pOut[ 47 ]+= Sinval * Gx1 * Gy3 * Gz7;
    pOut[ 48 ]+= Sinval * Gx1 * Gy6 * Gz4;
    pOut[ 49 ]+= Sinval * Gx1 * Gy4 * Gz6;
    pOut[ 50 ]+= Sinval * Gx6 * Gy1 * Gz4;
    pOut[ 51 ]+= Sinval * Gx4 * Gy1 * Gz6;
    pOut[ 52 ]+= Sinval * Gx6 * Gy4 * Gz1;
    pOut[ 53 ]+= Sinval * Gx4 * Gy6 * Gz1;
    pOut[ 54 ]+= Sinval * Gx5 * Gy5 * Gz1;
    pOut[ 55 ]+= Sinval * Gx5 * Gy1 * Gz5;
    pOut[ 56 ]+= Sinval * Gx1 * Gy5 * Gz5;
    pOut[ 57 ]+= Sinval * Gx7 * Gy2 * Gz2;
    pOut[ 58 ]+= Sinval * Gx2 * Gy7 * Gz2;
    pOut[ 59 ]+= Sinval * Gx2 * Gy2 * Gz7;
    pOut[ 60 ]+= Sinval * Gx3 * Gy6 * Gz2;
    pOut[ 61 ]+= Sinval * Gx3 * Gy2 * Gz6;
    pOut[ 62 ]+= Sinval * Gx6 * Gy3 * Gz2;
    pOut[ 63 ]+= Sinval * Gx2 * Gy3 * Gz6;
    pOut[ 64 ]+= Sinval * Gx6 * Gy2 * Gz3;
    pOut[ 65 ]+= Sinval * Gx2 * Gy6 * Gz3;
    pOut[ 66 ]+= Sinval * Gx5 * Gy4 * Gz2;
    pOut[ 67 ]+= Sinval * Gx5 * Gy2 * Gz4;
    pOut[ 68 ]+= Sinval * Gx4 * Gy5 * Gz2;
    pOut[ 69 ]+= Sinval * Gx2 * Gy5 * Gz4;
    pOut[ 70 ]+= Sinval * Gx4 * Gy2 * Gz5;
    pOut[ 71 ]+= Sinval * Gx2 * Gy4 * Gz5;
    pOut[ 72 ]+= Sinval * Gx5 * Gy3 * Gz3;
    pOut[ 73 ]+= Sinval * Gx3 * Gy5 * Gz3;
    pOut[ 74 ]+= Sinval * Gx3 * Gy3 * Gz5;
    pOut[ 75 ]+= Sinval * Gx3 * Gy4 * Gz4;
    pOut[ 76 ]+= Sinval * Gx4 * Gy3 * Gz4;
    pOut[ 77 ]+= Sinval * Gx4 * Gy4 * Gz3;
    return std::max(std::abs(Gx11), std::max(std::abs(Gy11), std::abs(Gz11)));
    //return std::max(Gx11, std::max(Gy11, Gz11));
    
  }
  else if (l == 12) {
    double Gx0=1., Gx1, Gx2, Gx4, Gx3, Gx5, Gx6, Gx7, Gx8, Gx9, Gx10, Gx11, Gx12;
    double Gy0=1., Gy1, Gy2, Gy4, Gy3, Gy5, Gy6, Gy7, Gy8, Gy9, Gy10, Gy11, Gy12;
    double Gz0=1., Gz1, Gz2, Gz4, Gz3, Gz5, Gz6, Gz7, Gz8, Gz9, Gz10, Gz11, Gz12;
    double Cosval = cos((Gx * Tx + Gy * Ty + Gz * Tz)) * ExpVal * Scale;
    Gx1 = Gx * Gx0; Gx2 = Gx * Gx1; Gx3 = Gx * Gx2; Gx4 = Gx * Gx3; Gx5 = Gx * Gx4; Gx6 = Gx * Gx5; Gx7 = Gx * Gx6; Gx8 = Gx * Gx7; Gx9 = Gx * Gx8; Gx10 = Gx * Gx9; Gx11 = Gx * Gx10; Gx12 = Gx * Gx11;
    Gy1 = Gy * Gy0; Gy2 = Gy * Gy1; Gy3 = Gy * Gy2; Gy4 = Gy * Gy3; Gy5 = Gy * Gy4; Gy6 = Gy * Gy5; Gy7 = Gy * Gy6; Gy8 = Gy * Gy7; Gy9 = Gy * Gy8; Gy10 = Gy * Gy9; Gy11 = Gy * Gy10; Gy12 = Gy * Gy11;
    Gz1 = Gz * Gz0; Gz2 = Gz * Gz1; Gz3 = Gz * Gz2; Gz4 = Gz * Gz3; Gz5 = Gz * Gz4; Gz6 = Gz * Gz5; Gz7 = Gz * Gz6; Gz8 = Gz * Gz7; Gz9 = Gz * Gz8; Gz10 = Gz * Gz9; Gz11 = Gz * Gz10; Gz12 = Gz * Gz11;

    pOut[ 0 ] += Cosval * Gx12 * Gy0 * Gz0;
    pOut[ 1 ] += Cosval * Gx0 * Gy12 * Gz0;
    pOut[ 2 ] += Cosval * Gx0 * Gy0 * Gz12;
    pOut[ 3 ] += Cosval * Gx11 * Gy1 * Gz0;
    pOut[ 4 ] += Cosval * Gx1 * Gy11 * Gz0;
    pOut[ 5 ] += Cosval * Gx11 * Gy0 * Gz1;
    pOut[ 6 ] += Cosval * Gx1 * Gy0 * Gz11;
    pOut[ 7 ] += Cosval * Gx0 * Gy11 * Gz1;
    pOut[ 8 ] += Cosval * Gx0 * Gy1 * Gz11;
    pOut[ 9 ] += Cosval * Gx10 * Gy2 * Gz0;
    pOut[ 10 ] += Cosval * Gx10 * Gy0 * Gz2;
    pOut[ 11 ] += Cosval * Gx2 * Gy10 * Gz0;
    pOut[ 12 ] += Cosval * Gx2 * Gy0 * Gz10;
    pOut[ 13 ] += Cosval * Gx0 * Gy10 * Gz2;
    pOut[ 14 ] += Cosval * Gx0 * Gy2 * Gz10;
    pOut[ 15 ] += Cosval * Gx9 * Gy3 * Gz0;
    pOut[ 16 ] += Cosval * Gx3 * Gy9 * Gz0;
    pOut[ 17 ] += Cosval * Gx9 * Gy0 * Gz3;
    pOut[ 18 ] += Cosval * Gx3 * Gy0 * Gz9;
    pOut[ 19 ] += Cosval * Gx0 * Gy9 * Gz3;
    pOut[ 20 ] += Cosval * Gx0 * Gy3 * Gz9;
    pOut[ 21 ] += Cosval * Gx8 * Gy4 * Gz0;
    pOut[ 22 ] += Cosval * Gx8 * Gy0 * Gz4;
    pOut[ 23 ] += Cosval * Gx4 * Gy8 * Gz0;
    pOut[ 24 ] += Cosval * Gx4 * Gy0 * Gz8;
    pOut[ 25 ] += Cosval * Gx0 * Gy8 * Gz4;
    pOut[ 26 ] += Cosval * Gx0 * Gy4 * Gz8;
    pOut[ 27 ] += Cosval * Gx7 * Gy5 * Gz0;
    pOut[ 28 ] += Cosval * Gx5 * Gy7 * Gz0;
    pOut[ 29 ] += Cosval * Gx7 * Gy0 * Gz5;
    pOut[ 30 ] += Cosval * Gx5 * Gy0 * Gz7;
    pOut[ 31 ] += Cosval * Gx0 * Gy7 * Gz5;
    pOut[ 32 ] += Cosval * Gx0 * Gy5 * Gz7;
    pOut[ 33 ] += Cosval * Gx6 * Gy6 * Gz0;
    pOut[ 34 ] += Cosval * Gx6 * Gy0 * Gz6;
    pOut[ 35 ] += Cosval * Gx0 * Gy6 * Gz6;
    pOut[ 36 ] += Cosval * Gx1 * Gy1 * Gz10;
    pOut[ 37 ] += Cosval * Gx1 * Gy10 * Gz1;
    pOut[ 38 ] += Cosval * Gx10 * Gy1 * Gz1;
    pOut[ 39 ] += Cosval * Gx9 * Gy1 * Gz2;
    pOut[ 40 ] += Cosval * Gx1 * Gy9 * Gz2;
    pOut[ 41 ] += Cosval * Gx9 * Gy2 * Gz1;
    pOut[ 42 ] += Cosval * Gx1 * Gy2 * Gz9;
    pOut[ 43 ] += Cosval * Gx2 * Gy9 * Gz1;
    pOut[ 44 ] += Cosval * Gx2 * Gy1 * Gz9;
    pOut[ 45 ] += Cosval * Gx3 * Gy1 * Gz8;
    pOut[ 46 ] += Cosval * Gx1 * Gy3 * Gz8;
    pOut[ 47 ] += Cosval * Gx3 * Gy8 * Gz1;
    pOut[ 48 ] += Cosval * Gx1 * Gy8 * Gz3;
    pOut[ 49 ] += Cosval * Gx8 * Gy3 * Gz1;
    pOut[ 50 ] += Cosval * Gx8 * Gy1 * Gz3;
    pOut[ 51 ] += Cosval * Gx7 * Gy1 * Gz4;
    pOut[ 52 ] += Cosval * Gx1 * Gy7 * Gz4;
    pOut[ 53 ] += Cosval * Gx7 * Gy4 * Gz1;
    pOut[ 54 ] += Cosval * Gx1 * Gy4 * Gz7;
    pOut[ 55 ] += Cosval * Gx4 * Gy7 * Gz1;
    pOut[ 56 ] += Cosval * Gx4 * Gy1 * Gz7;
    pOut[ 57 ] += Cosval * Gx5 * Gy1 * Gz6;
    pOut[ 58 ] += Cosval * Gx1 * Gy5 * Gz6;
    pOut[ 59 ] += Cosval * Gx5 * Gy6 * Gz1;
    pOut[ 60 ] += Cosval * Gx1 * Gy6 * Gz5;
    pOut[ 61 ] += Cosval * Gx6 * Gy5 * Gz1;
    pOut[ 62 ] += Cosval * Gx6 * Gy1 * Gz5;
    pOut[ 63 ] += Cosval * Gx8 * Gy2 * Gz2;
    pOut[ 64 ] += Cosval * Gx2 * Gy8 * Gz2;
    pOut[ 65 ] += Cosval * Gx2 * Gy2 * Gz8;
    pOut[ 66 ] += Cosval * Gx7 * Gy3 * Gz2;
    pOut[ 67 ] += Cosval * Gx3 * Gy7 * Gz2;
    pOut[ 68 ] += Cosval * Gx7 * Gy2 * Gz3;
    pOut[ 69 ] += Cosval * Gx3 * Gy2 * Gz7;
    pOut[ 70 ] += Cosval * Gx2 * Gy7 * Gz3;
    pOut[ 71 ] += Cosval * Gx2 * Gy3 * Gz7;
    pOut[ 72 ] += Cosval * Gx6 * Gy4 * Gz2;
    pOut[ 73 ] += Cosval * Gx6 * Gy2 * Gz4;
    pOut[ 74 ] += Cosval * Gx4 * Gy6 * Gz2;
    pOut[ 75 ] += Cosval * Gx4 * Gy2 * Gz6;
    pOut[ 76 ] += Cosval * Gx2 * Gy6 * Gz4;
    pOut[ 77 ] += Cosval * Gx2 * Gy4 * Gz6;
    pOut[ 78 ] += Cosval * Gx5 * Gy5 * Gz2;
    pOut[ 79 ] += Cosval * Gx5 * Gy2 * Gz5;
    pOut[ 80 ] += Cosval * Gx2 * Gy5 * Gz5;
    pOut[ 81 ] += Cosval * Gx3 * Gy3 * Gz6;
    pOut[ 82 ] += Cosval * Gx3 * Gy6 * Gz3;
    pOut[ 83 ] += Cosval * Gx6 * Gy3 * Gz3;
    pOut[ 84 ] += Cosval * Gx5 * Gy3 * Gz4;
    pOut[ 85 ] += Cosval * Gx3 * Gy5 * Gz4;
    pOut[ 86 ] += Cosval * Gx5 * Gy4 * Gz3;
    pOut[ 87 ] += Cosval * Gx3 * Gy4 * Gz5;
    pOut[ 88 ] += Cosval * Gx4 * Gy5 * Gz3;
    pOut[ 89 ] += Cosval * Gx4 * Gy3 * Gz5;
    pOut[ 90 ] += Cosval * Gx4 * Gy4 * Gz4;    
    return std::max(Gx12, std::max(Gy12, Gz12));
  }
  return -1.;
}

namespace ir{
void EvalSlmX_Deriv0(double * Out, double x, double y, double z, unsigned L);
};

double getSphReciprocal(int la, int lb, double* pOut,
                        double* pSpha, double* pSphb,
                        double Gx, double Gy, double Gz,
                        double Tx, double Ty, double Tz,
                        double exponentVal,
                        double Scale) {
  
  double ExpVal = exponentVal;//exp(-exponentVal)/exponentVal; 


  if (la == 0 && lb == 0) {
    pOut[0] += Scale * ExpVal * cos(Gx*Tx + Gy*Ty+Gz*Tz);
    return 1.0;
  }
  else if (la == 0 && lb == 1 || lb == 0 && la == 1) {
    double preFactor = Scale * ExpVal * -sin((Gx * Tx + Gy * Ty + Gz * Tz));
    pOut[0] += preFactor * Gx;
    pOut[1] += preFactor * Gy;
    pOut[2] += preFactor * Gz;
    return std::max(fabs(Gx), std::max(fabs(Gy),fabs(Gz)));
  }
  else if (la == 1 && lb == 1) {
    double preFactor = Scale * ExpVal * -cos((Gx * Tx + Gy * Ty + Gz * Tz));
    pOut[0] +=  preFactor * Gx * Gx;
    pOut[1] +=  preFactor * Gy * Gx;
    pOut[2] +=  preFactor * Gz * Gx;
    pOut[3] +=  preFactor * Gx * Gy;
    pOut[4] +=  preFactor * Gy * Gy;
    pOut[5] +=  preFactor * Gz * Gy;
    pOut[6] +=  preFactor * Gx * Gz;
    pOut[7] +=  preFactor * Gy * Gz;
    pOut[8] +=  preFactor * Gz * Gz;
    return std::max(Gx*Gx, std::max(Gy*Gy, Gz*Gz));
  }
  else {
    int L = la > lb ? la : lb;
    ir::EvalSlcX_Deriv0(pSpha, Gx, Gy, Gz, L);


    double preFactor = Scale * ExpVal;
    if ( (la+lb)%4 == 0)
      preFactor *= cos((Gx * Tx + Gy * Ty + Gz * Tz));
    else if ((la+lb)%4 == 1)
      preFactor *= -sin((Gx * Tx + Gy * Ty + Gz * Tz));
    else if ((la+lb)%4 == 2)
      preFactor *= -cos((Gx * Tx + Gy * Ty + Gz * Tz));
    else if ((la+lb)%4 == 3)
      preFactor *= sin((Gx * Tx + Gy * Ty + Gz * Tz));
    
    int Nb = 2*lb+1, Na = 2*la+1;
    //DGER(Na, Nb, preFactor, pSpha+la*la, 1, pSpha+lb*lb,1,pOut, Na);
    
    
    double *Amat = pSpha+la*la, *Bmat = pSpha+lb*lb;
    
    int index = 0;
    for (int b=0; b<Nb; b++) {
      for (int a=0; a<Na; a++) {
	pOut[index] = pOut[index] + preFactor * Amat[a] * Bmat[b];
	index ++;
      }
    }
    
    
    double maxG1 = 0.0, maxG2 = 0.0;
    for (int b=0; b<Nb; b++)
      maxG1 = std::max(maxG1, fabs(Bmat[b]));
    for (int a=0; a<Na; a++)
      maxG2 = std::max(maxG2, fabs(Amat[a]));
    
    return maxG1*maxG2;
  }
}


double getSphReciprocal3(int la, int lb, int lc, double* pOut,
			 double* pSpha, double* pSphb, double* pSphc,
			 double Gx, double Gy, double Gz,
			 double Fx, double Fy, double Fz,
			 double TABx, double TABy, double TABz,
			 double TACx, double TACy, double TACz,
			 double exponentVal,
			 double Scale) {
  
  double ExpVal = exponentVal;//exp(-exponentVal)/exponentVal; 

  ir::EvalSlcX_Deriv0(pSpha, -Gx-Fx, -Gy-Fy, -Gz-Fz, la);
  ir::EvalSlcX_Deriv0(pSphb,     Fx,     Fy,     Fz, lb);
  ir::EvalSlcX_Deriv0(pSphc,  Gx   ,  Gy   ,  Gz   , lc);


  double preFactor = Scale * ExpVal;
  double cosArg = (Gx * TACx + Gy * TACy + Gz * TACz) + (Fx * TABx + Fy * TABy + Fz * TABz);

  double sign = 1.;//la % 2 == 0 ? 1 : -1;
  if ( (la+lb+lc)%4 == 0)
    preFactor *= sign*cos(cosArg);
  else if ((la+lb+lc)%4 == 1)
    preFactor *= -sign*sin(cosArg);
  else if ((la+lb+lc)%4 == 2)
    preFactor *= -sign*cos(cosArg);
  else if ((la+lb+lc)%4 == 3)
    preFactor *= sign*sin(cosArg);

  int Nb = 2*lb+1, Na = 2*la+1, Nc = 2*lc+1;

  
  double *Amat = pSpha+la*la, *Bmat = pSphb+lb*lb, *Cmat = pSphc+lc*lc;

  int index = 0;
  for (int c=0; c<Nc; c++) {
    double cfact = preFactor * Cmat[c];
    
    for (int b=0; b<Nb; b++) {
      double bfact = cfact * Bmat[b];
      
      for (int a=0; a<Na; a++) {
	pOut[index] = pOut[index] + bfact * Amat[a];
	index ++;
      }
    }
  }

  double maxG1 = 0.0, maxG2 = 0.0, maxG3 = 0.0;
  for (int b=0; b<Nb; b++)
    maxG1 = std::max(maxG1, fabs(Bmat[b]));
  for (int a=0; a<Na; a++)
    maxG2 = std::max(maxG2, fabs(Amat[a]));
  for (int c=0; c<Nc; c++)
    maxG3 = std::max(maxG3, fabs(Cmat[c]));

  return maxG1*maxG2*maxG3;
}


double getSphReciprocal2(int la, int lb, double* pOut,
			 double* pSpha, double* pSphb, 
			 double Gx, double Gy, double Gz,
			 double Fx, double Fy, double Fz,
			 double TAx, double TAy, double TAz,
			 double TBx, double TBy, double TBz,
			 double exponentVal,
			 double Scale) {
  
  double ExpVal = exponentVal;//exp(-exponentVal)/exponentVal; 

  ir::EvalSlcX_Deriv0(pSpha, -Gx-Fx, -Gy-Fy, -Gz-Fz, la);
  ir::EvalSlcX_Deriv0(pSphb,     Fx,     Fy,     Fz, lb);


  double preFactor = Scale * ExpVal;
  double cosArg = (Gx * TAx + Gy * TAy + Gz * TAz)
    + (Fx * (TAx-TBx) + Fy * (TAy-TBy) + Fz * (TAz-TBz));

  if ( (la+lb)%4 == 0)
    preFactor *= cos(cosArg);
  else if ((la+lb)%4 == 1)
    preFactor *= -sin(cosArg);
  else if ((la+lb)%4 == 2)
    preFactor *= -cos(cosArg);
  else if ((la+lb)%4 == 3)
    preFactor *= sin(cosArg);

  int Nb = 2*lb+1, Na = 2*la+1;

  
  double *Amat = pSpha+la*la, *Bmat = pSphb+lb*lb;

  int index = 0;
  for (int b=0; b<Nb; b++) {
    double bfact = preFactor * Bmat[b];
      
    for (int a=0; a<Na; a++) {
      pOut[index] += bfact * Amat[a];
      index ++;
    }
  }

  double maxG1 = 0.0, maxG2 = 0.0, maxG3 = 0.0;
  for (int b=0; b<Nb; b++)
    maxG1 = std::max(maxG1, fabs(Bmat[b]));
  for (int a=0; a<Na; a++)
    maxG2 = std::max(maxG2, fabs(Amat[a]));

  return maxG1*maxG2;
}


double getSphReciprocal3cos(int la, int lb, double* pOut,
			    double* pSpha1, double* pSpha2,
			    double* pSphb1, double* pSphb2, 
			    double Gx, double Gy, double Gz,
			    double Fx, double Fy, double Fz,
			    double Ax, double Ay, double Az,
			    double Bx, double By, double Bz,
			    double exponentVal,
			    double Scale) {
  
  double ExpVal = exponentVal;//exp(-exponentVal)/exponentVal; 

  ir::EvalSlcX_Deriv0(pSpha1,  Gx+Fx,  Gy+Fy,  Gz+Fz, la);
  ir::EvalSlcX_Deriv0(pSphb1,    -Fx,    -Fy,    -Fz, lb);


  double preFactor = Scale * ExpVal;
  double Arg1 = ((Gx + Fx) * Ax + (Gy + Fy) * Ay + (Gz + Fz) * Az);
  double Arg2 = -(Fx * Bx + Fy * By + Fz * Bz);

  double Oscillatory1;
  if ( (la+lb)%4 == 0) 
    { Oscillatory1 = preFactor*cos(Arg1 + Arg2);}
  else if ((la+lb)%4 == 1)
    { Oscillatory1 = -preFactor*sin(Arg1 + Arg2);}
  else if ((la+lb)%4 == 2)
    { Oscillatory1 = -preFactor*cos(Arg1 + Arg2);}
  else if ((la+lb)%4 == 3)
    { Oscillatory1 = preFactor*sin(Arg1 + Arg2);}

  int Nb = 2*lb+1, Na = 2*la+1;

  
  double *Amat1 = pSpha1+la*la;
  double *Bmat1 = pSphb1+lb*lb;

  int index = 0;
  for (int b=0; b<Nb; b++) {
    double bfact1 = Oscillatory1 * Bmat1[b];
      
    for (int a=0; a<Na; a++) {
      pOut[index] += bfact1 * Amat1[a] ;
      index ++;
    }
  }

  double maxG1 = 0.0, maxG2 = 0.0, maxG3 = 0.0;
  for (int b=0; b<Nb; b++)
    maxG1 = std::max(maxG1, fabs(Bmat1[b]));
  for (int a=0; a<Na; a++)
    maxG2 = std::max(maxG2, fabs(Amat1[a]));

  return maxG1*maxG2;
}


double getSphReciprocal3sin(int la, int lb, double* pOut,
			    double* pSpha1, double* pSpha2,
			    double* pSphb1, double* pSphb2, 
			    double Gx, double Gy, double Gz,
			    double Fx, double Fy, double Fz,
			    double Ax, double Ay, double Az,
			    double Bx, double By, double Bz,
			    double exponentVal,
			    double Scale) {
  
  double ExpVal = exponentVal;//exp(-exponentVal)/exponentVal; 

  ir::EvalSlcX_Deriv0(pSpha1,  Gx+Fx,  Gy+Fy,  Gz+Fz, la);
  ir::EvalSlcX_Deriv0(pSphb1,    -Fx,    -Fy,    -Fz, lb);


  double preFactor = Scale * ExpVal;
  double Arg1 = ((Gx + Fx) * Ax + (Gy + Fy) * Ay + (Gz + Fz) * Az);
  double Arg2 = -(Fx * Bx + Fy * By + Fz * Bz);

  double Oscillatory1;
  if ( (la+lb)%4 == 0) 
    { Oscillatory1 = preFactor*sin(Arg1 + Arg2);}
  else if ((la+lb)%4 == 1)
    { Oscillatory1 = preFactor*cos(Arg1 + Arg2);}
  else if ((la+lb)%4 == 2)
    { Oscillatory1 = -preFactor*sin(Arg1 + Arg2);}
  else if ((la+lb)%4 == 3)
    { Oscillatory1 = -preFactor*cos(Arg1 + Arg2);}

  int Nb = 2*lb+1, Na = 2*la+1;

  
  double *Amat1 = pSpha1+la*la;
  double *Bmat1 = pSphb1+lb*lb;

  int index = 0;
  for (int b=0; b<Nb; b++) {
    double bfact1 = Oscillatory1 * Bmat1[b];
      
    for (int a=0; a<Na; a++) {
      pOut[index] += bfact1 * Amat1[a] ;
      index ++;
    }
  }

  double maxG1 = 0.0, maxG2 = 0.0, maxG3 = 0.0;
  for (int b=0; b<Nb; b++)
    maxG1 = std::max(maxG1, fabs(Bmat1[b]));
  for (int a=0; a<Na; a++)
    maxG2 = std::max(maxG2, fabs(Amat1[a]));

  return maxG1*maxG2;
}

void PolynomialsUsingRecursion(double* pCos, double* pSin, complex<double> &factorg,
			       complex<double>& factorh, double alpha,
			       complex<double> *g, complex<double> *h,
			       int l) {

  
  pCos[0] = 0.5*(g[0] + h[0] ).real();
  pSin[0] = 0.5*(g[0] - h[0] ).imag();

  if (l == 0) return;


  double factorial = 1.0;

  g[1] = -factorg * g[0];
  h[1] = -factorh * h[0];
  
  pCos[1] = 0.5 * factorial * ( g[1] + h[1] ).real();
  pSin[1] = 0.5 * factorial * ( g[1] - h[1] ).imag();

  if (l == 1) return;

  factorial *= 2.;
  g[2] = 1./2. * ( - factorg * g[1]  - 2. * alpha * g[0] );
  h[2] = 1./2. * ( - factorh * h[1]  - 2. * alpha * h[0] );
  
  pCos[2] = 0.5 * factorial * ( g[2] + h[2] ).real();
  pSin[2] = 0.5 * factorial * ( g[2] - h[2] ).imag();
  
  if (l == 2) return;

  factorial *= 3.;
  g[3] = 1./3. * ( - factorg * g[2]  - 2. * alpha * g[1] );
  h[3] = 1./3. * ( - factorh * h[2]  - 2. * alpha * h[1] );
  
  pCos[3] = 0.5 * factorial * ( g[3] + h[3] ).real();
  pSin[3] = 0.5 * factorial * ( g[3] - h[3] ).imag();
  
  if (l == 3) return;

  factorial *= 4.;
  g[4] = 1./4. * ( - factorg * g[3]  - 2. * alpha * g[2] );
  h[4] = 1./4. * ( - factorh * h[3]  - 2. * alpha * h[2] );
  
  pCos[4] = 0.5 * factorial * ( g[4] + h[4] ).real();
  pSin[4] = 0.5 * factorial * ( g[4] - h[4] ).imag();
  
  if (l == 4) return;

  factorial *= 5.;
  g[5] = 1./5. * ( - factorg * g[4]  - 2. * alpha * g[3] );
  h[5] = 1./5. * ( - factorh * h[4]  - 2. * alpha * h[3] );
  
  pCos[5] = 0.5 * factorial * ( g[5] + h[5] ).real();
  pSin[5] = 0.5 * factorial * ( g[5] - h[5] ).imag();
  
  if (l == 5) return;

  factorial *= 6.;
  g[6] = 1./6. * ( - factorg * g[5]  - 2. * alpha * g[4] );
  h[6] = 1./6. * ( - factorh * h[5]  - 2. * alpha * h[4] );
  
  pCos[6] = 0.5 * factorial * ( g[6] + h[6] ).real();
  pSin[6] = 0.5 * factorial * ( g[6] - h[6] ).imag();

  if (l == 6) return;

}

void getCartesianHarmonics(double* x, double* y, double* z,
			   double* cart, int l) {
  using namespace ir;
  if (l == 0) {
    cart[0] = x[0] * y[0] * z[0];
    return;
  }
  else if (l ==1) {
    cart[0] = x[1] * y[0] * z[0];
    cart[1] = x[0] * y[1] * z[0];
    cart[2] = x[0] * y[0] * z[1];
    return;
  }
  else if (l >= 2 && l <=6) {
    int startL = nCartX(l-1), endL = nCartX(l);
    for (int i = startL; i < endL; i++) 
      cart[i - startL] = x[iCartPow[i][0]] *  y[iCartPow[i][1]] *  z[iCartPow[i][2]] ;
  }
  else {
    cout << "l has to be less <= 6 but given "<<l<<endl;
    exit(0);
  }
}


void Recurse(complex<double>* pOut, complex<double>& Tx,
	     double alpha, int LmaxA, int LmaxB) {

  complex<double> Txc = -conj(Tx);
  for (int la = 1; la < LmaxA; la++)
    pOut[la] = Tx * pOut[la-1] + (la > 1 ? -2*alpha*(la-1)*pOut[la-2] : 0.);
  for (int lb = 1; lb < LmaxB; lb++)
    pOut[lb*LmaxA] = Txc * pOut[(lb-1)*LmaxA] + (lb > 1 ? -2*alpha*(lb-1)*pOut[lb-2] : 0.);
  
  for (int la = 1; la < LmaxA; la++)
  for (int lb = 1; lb < LmaxB; lb++) {
    pOut[la + lb * LmaxA] = Tx * pOut[la -1 + lb * LmaxA]
      + 2 * alpha * lb * pOut[la -1 + (lb-1)*LmaxA]
      + (la > 1 ? -2 * alpha * (la-1) * pOut[la -2, lb * LmaxA] : 0.); 
  }
}

double getSphRealRecursion(int la, int lb, double* pOutCos, double* pOutSin,
			   double Gx, double Gy, double Gz,
			   double Qx, double Qy, double Qz,
			   double Ax, double Ay, double Az,
			   double Bx, double By, double Bz,
			   double alpha, double a, double b,
			   double exponentVal,
			   double Scale, ct::FMemoryStack2& Mem) {
  double arg = ( Gx * (a * Ax + b * Bx + b * Qx)
		 +  Gy * (a * Ay + b * By + b * Qy)
		 +  Gz * (a * Az + b * Bz + b * Qz) )/ (a + b);
  
  if (la == 0 && lb == 0) {
    pOutCos[0] += exponentVal * Scale * cos(arg);
    pOutSin[0] += exponentVal * Scale * sin(arg);
    return 1.;
  }
  complex<double> Tx = complex<double>(-2.*alpha*(Ax-Bx-Qx), Gx * a /(a+b));
  complex<double> Ty = complex<double>(-2.*alpha*(Ay-By-Qy), Gy * a /(a+b));
  complex<double> Tz = complex<double>(-2.*alpha*(Az-Bz-Qz), Gz * a /(a+b));

  complex<double> *Xab, *Yab, *Zab;
  Mem.Alloc(Xab, (la+1)*(lb+1)); Mem.Alloc(Yab, (la+1)*(lb+1));
  Mem.Alloc(Zab, (la+1)*(lb+1));

  Xab[0] = complex<double>(1.,1.); Yab[0] = Xab[0]; Zab[0] = Xab[0];
  Recurse(Xab, Tx, alpha, la+1, lb+1);
  Recurse(Yab, Ty, alpha, la+1, lb+1);
  Recurse(Zab, Tz, alpha, la+1, lb+1);


  int ncarta = ir::nCartY(la), ncartb = ir::nCartY(lb);
  double *cosInts, *sinInts;
  Mem.Alloc(cosInts, ncarta * ncartb);
  Mem.Alloc(sinInts, ncarta * ncartb);

  complex<double> f = Xab[0];
  complex<double> prefactor = Scale * exponentVal * complex<double>(cos(arg), sin(arg))/f/f/f;
  
  int aoff = ir::nCartX(la-1), boff = ir::nCartX(lb-1);
  for (int i=0; i<ncarta; i++)
  for (int j=0; j<ncartb; j++)
  {
    int ax = ir::iCartPow[aoff+i][0], ay = ir::iCartPow[aoff+i][1], az = ir::iCartPow[aoff+i][2];
    int bx = ir::iCartPow[boff+j][0], by = ir::iCartPow[boff+j][1], bz = ir::iCartPow[boff+j][2];

    complex<double> expInts = Xab[ax + bx * (la+1)] *
      Yab[ay + by * (la+1)] *
      Zab[az + bz * (la+1)] *
      prefactor ;
    
    cosInts[i + j * ncarta] = expInts.real(); 
    sinInts[i + j * ncarta] = expInts.imag();
  }

  double *pR1; Mem.Alloc(pR1, (2*lb+1)*ncarta);
  int nterms = (2*la+1)*(2*lb+1);

  double *pFinal; Mem.Alloc(pFinal, nterms);
  ir::ShTrN(pR1, cosInts, ncarta, lb);
  ir::ShTrN_NN(pFinal, pR1, (2*lb + 1), la);


  for (int i=0; i<nterms; i++) pOutCos[i] += pFinal[i];
  
  ir::ShTrN(pR1, sinInts, ncarta, lb);
  ir::ShTrN_NN(pFinal, pR1, (2*lb + 1), la);

  for (int i=0; i<nterms; i++) pOutSin[i] += pFinal[i];

  Mem.Free(Xab);
  return 1.;
}


/*
  double getSphRealRecursion(int la, int lb, double* pOutCos, double* pOutSin,
  double* pSphaCos, double* pSphaSin,
  double* pSphbCos, double* pSphbSin, 
  double Gx, double Gy, double Gz,
  double Qx, double Qy, double Qz,
  double Ax, double Ay, double Az,
  double Bx, double By, double Bz,
  double alpha, double a, double b,
  double exponentVal,
  double Scale, ct::FMemoryStack2& Mem) {
  double arg = ( Gx * (a * Ax + b * Bx + b * Qx)
  +  Gy * (a * Ay + b * By + b * Qy)
  +  Gz * (a * Az + b * Bz + b * Qz) )/ (a + b);
  
  if (la == 0 && lb == 0) {
  pOutCos[0] += exponentVal * Scale * cos(arg);
  pOutSin[0] += exponentVal * Scale * sin(arg);
  return 1.;
  }
  else if (la == 0 && lb == 1) {
  pOutCos[0] +=  exponentVal * Scale * (-Gx*a/(a+b) * sin(arg) - 2 * alpha * (Ax - Bx - Qx) * cos(arg));
  pOutCos[1] +=  exponentVal * Scale * (-Gy*a/(a+b) * sin(arg) - 2 * alpha * (Ay - By - Qy) * cos(arg));
  pOutCos[2] +=  exponentVal * Scale * (-Gz*a/(a+b) * sin(arg) - 2 * alpha * (Az - Bz - Qz) * cos(arg));
  return 1.;
  }
  else if (la == 1 && lb == 0) {
  pOutCos[0] +=  exponentVal * Scale * (-Gx*b/(a+b) * sin(arg) + 2 * alpha * (Ax - Bx - Qx) * cos(arg));
  pOutCos[1] +=  exponentVal * Scale * (-Gy*b/(a+b) * sin(arg) + 2 * alpha * (Ay - By - Qy) * cos(arg));
  pOutCos[2] +=  exponentVal * Scale * (-Gz*b/(a+b) * sin(arg) + 2 * alpha * (Az - Bz - Qz) * cos(arg));

  pOutSin[0] +=  exponentVal * Scale * (Gx*b/(a+b) * cos(arg) + 2 * alpha * (Ax - Bx - Qx) * sin(arg));
  pOutSin[1] +=  exponentVal * Scale * (Gy*b/(a+b) * cos(arg) + 2 * alpha * (Ay - By - Qy) * sin(arg));
  pOutSin[2] +=  exponentVal * Scale * (Gz*b/(a+b) * cos(arg) + 2 * alpha * (Az - Bz - Qz) * sin(arg));
  cout << pOutSin[0]<<endl;
  pOutSin[0] = 0.0;
  pOutSin[1] = 0.0;
  pOutSin[2] = 0.0;
  //return 1.;
  }
  int L = max(la, lb);
  complex<double>* pOut; Mem.Alloc(pOut, ir::nCartX(L));
  double *CartesianCos, *CartesianSin;
  Mem.Alloc(CartesianCos, 2*ir::nCartY(L));
  CartesianSin = CartesianCos + ir::nCartY(L);
  
  pOut[0] = complex<double>(cos(arg), sin(arg));
  complex<double> Tx = complex<double>(-2.*alpha*(Ax-Bx-Qx), Gx * a /(a+b));
  complex<double> Ty = complex<double>(-2.*alpha*(Ay-By-Qy), Gy * a /(a+b));
  complex<double> Tz = complex<double>(-2.*alpha*(Az-Bz-Qz), Gz * a /(a+b));

  double factorial = 1.0;
  Recurse(&pOut[1], &pOut[0], &pOut[0], Tx, Ty, Tz, alpha, 1); 
  for (int l=2; l<=la; l++) {
  int lindex2 = ir::nCartX(l-1); 
  int lindex1 = ir::nCartX(l-2); 
  int lindex0 = ir::nCartX(l-3);
  Recurse(&pOut[lindex2], &pOut[lindex1], &pOut[lindex0], Tx, Ty, Tz, alpha, l);
  factorial *= l;
  }

  //make cartesian components
  int ncartL = ir::nCartY(la), ncartLcum = ir::nCartX(la-1);
  for (int i=0; i<ncartL; i++) {
  CartesianCos[i] = pOut[i+ncartLcum].real() * factorial;
  CartesianSin[i] = pOut[i+ncartLcum].imag() * factorial;
  }

  //make spherical components
  ir::ShTrN(pSphaCos, CartesianCos, 1, la);
  ir::ShTrN(pSphaSin, CartesianSin, 1, la);


  Tx = complex<double>(2.*alpha*(Ax-Bx-Qx), Gx * b /(a+b));
  Ty = complex<double>(2.*alpha*(Ay-By-Qy), Gy * b /(a+b));
  Tz = complex<double>(2.*alpha*(Az-Bz-Qz), Gz * b /(a+b));

  factorial = 1.0;
  Recurse(&pOut[1], &pOut[0], &pOut[0], Tx, Ty, Tz, alpha, 1); 
  for (int l=2; l<=lb; l++) {
  int lindex2 = ir::nCartX(l-1); 
  int lindex1 = ir::nCartX(l-2); 
  int lindex0 = ir::nCartX(l-3);
  Recurse(&pOut[lindex2], &pOut[lindex1], &pOut[lindex0], Tx, Ty, Tz, alpha, l);
  factorial *= l;
  }

  //make cartesian components
  ncartL = ir::nCartY(lb); ncartLcum = ir::nCartX(lb-1);
  for (int i=0; i<ncartL; i++) {
  cout << i<<"  "<<ncartL<<"  "<<ncartLcum<<"  "<<pOut[i+ncartLcum]<<endl;
  CartesianCos[i] = pOut[i+ncartLcum].real() * factorial;
  CartesianSin[i] = pOut[i+ncartLcum].imag() * factorial;
  }

  //make spherical components
  ir::ShTrN(pSphbCos, CartesianCos, 1, lb);
  ir::ShTrN(pSphbSin, CartesianSin, 1, lb);


  
  //now tensor multiply a,b
  int Na = 2*la+1, Nb = 2*lb+1;
  int index = 0;
  double factor = exponentVal * Scale ;

  for (int b=0; b<Nb; b++) {
  double bfact1 = factor * pSphbCos[b];
      
  for (int a=0; a<Na; a++) {
  pOutCos[index] += bfact1 * pSphaCos[a] ;
  index ++;
  }
  }


  index = 0;
  for (int b=0; b<Nb; b++) {
  double bfact1 = factor * pSphbSin[b];
      
  for (int a=0; a<Na; a++) {
  pOutSin[index] += bfact1 * pSphaSin[a] ;
  cout << index<<"  "<<pSphbSin[b]<<"  "<<pSphaSin[a]<<"  "<<a<<"  "<<b<<endl;
  index ++;
  }
  }
  cout << factor<<"  "<<pSphbSin[0]<<"  "<<pSphaSin[0]<<"  "<<pOutSin[0]<<endl;
  exit(0);
  Mem.Free(pOut);
  return 1.;
  }
*/


