#include <iostream>

#include "primitives.h"
#include "workArray.h"

using namespace Eigen;

//actually this calculates phi_i(r, a, A) phi_0(r,tau, r') phi_j(r', b, B)
//where phi_0 is one of the gaussians approximating 1/r
double calcCoulombIntegral(int LA, double Ax, double Ay, double Az,
                           double expA, double normA,
                           double expG, double wtG,
                           int LB, double Bx, double By, double Bz,
                           double expB, double normB,
                           MatrixXd& Int) {

  double mu = expG * expB / (expG + expB);
  double prefactor = sqrt(M_PI/(expG + expB)) * normA * normB;
                         

  calc1DOvlp(LA, Ax, expA, LB, Bx, mu, Sx);  
  calc1DOvlp(LA, Ay, expA, LB, By, mu, Sy);  
  calc1DOvlp(LA, Az, expA, LB, Bz, mu, Sz);

  Sx2.block(0,0,LA+1, LB+1).setZero();
  Sy2.block(0,0,LA+1, LB+1).setZero();
  Sz2.block(0,0,LA+1, LB+1).setZero();
  
  double commonFactor = sqrt(M_PI/(expB+expG));
  double f = sqrt(2. * (expB+expG));
  
  for (int j=0; j <= LB; j++) {

    for (int n = 0; n <= j; n++) {
      if ( (j-n)%2 != 0) continue;

      double factor = nChoosek(j, n) * pow(expG/(expG+expB), n)
          * doubleFact(j - n - 1)/pow(f, 1.*j-1.*n) * commonFactor;

      for (int i=0; i <= LA; i++) {
        Sx2(i, j) += factor * Sx(i, n);
        Sy2(i, j) += factor * Sy(i, n);
        Sz2(i, j) += factor * Sz(i, n);
      }
    }
  }


  int IA0 = (LA)*(LA+1)*(LA+2)/6;
  int IB0 = (LB)*(LB+1)*(LB+2)/6;

  normA *= 1.0/sqrt( doubleFact(2*LA-1)/pow(4*expA, LA))
      / pow(M_PI/2./expA, 0.75);// * sqrt(4* M_PI/ (2*LA+1));
  if (LA >=2) normA = normA * sqrt(4* M_PI/ (2*LA+1));
  normB *= 1.0/sqrt( doubleFact(2*LB-1)/pow(4*expB, LB))
      / pow(M_PI/2./expB, 0.75);// * sqrt(4 * M_PI / (2*LB+1));
  if (LB >=2) normB = normB * sqrt(4 * M_PI / (2*LB+1));

  for (int i=0; i<(LA+1)*(LA+2)/2; i++)
    for (int j=0; j<(LB+1)*(LB+2)/2; j++) {
      auto ipow = CartOrder[IA0 + i];
      auto jpow = CartOrder[IB0 + j];

      Int(i, j) +=   Sx2(ipow[0], jpow[0]) *
          Sy2(ipow[1], jpow[1]) *
          Sz2(ipow[2], jpow[2]) * normA * normB * wtG;
    }  
}


double calcOvlpMatrix(int LA, double Ax, double Ay, double Az, double expA,
                     int LB, double Bx, double By, double Bz, double expB,
                     MatrixXd& S) {


  calc1DOvlp(LA, Ax, expA, LB, Bx, expB, Sx);
  calc1DOvlp(LA, Ay, expA, LB, By, expB, Sy);
  calc1DOvlp(LA, Az, expA, LB, Bz, expB, Sz);

  int IA0 = (LA)*(LA+1)*(LA+2)/6;
  int IB0 = (LB)*(LB+1)*(LB+2)/6;


  double normA = 1.0/sqrt( doubleFact(2*LA-1)/pow(4*expA, LA)) / pow(M_PI/2./expA, 0.75);
  if (LA >=2) normA = normA * sqrt(4* M_PI/ (2*LA+1));
  double normB = 1.0/sqrt( doubleFact(2*LB-1)/pow(4*expB, LB)) / pow(M_PI/2./expB, 0.75);
  if (LB >=2) normB = normB * sqrt(4 * M_PI / (2*LB+1));

  for (int i=0; i<(LA+1)*(LA+2)/2; i++)
    for (int j=0; j<(LB+1)*(LB+2)/2; j++) {
      auto ipow = CartOrder[IA0 + i];
      auto jpow = CartOrder[IB0 + j];

      S(i, j) =   Sx(ipow[0], jpow[0]) *
                  Sy(ipow[1], jpow[1]) *
                  Sz(ipow[2], jpow[2]) * normA * normB;
    }
    
}



double calcCoulombIntegralPeriodic(int LA, double Ax, double Ay, double Az,
				   double expA, double normA,
				   double expG, double wtG,
				   int LB, double Bx, double By, double Bz,
				   double expB, double normB,
				   MatrixXd& Int) {

  //assume that LA and LB = 0
  double t = expG * expA * expB / (expG*expA + expG*expB + expA*expB);

  double prefactor = pow(M_PI*M_PI*M_PI/(expG*expA*expB), 1.5) * normA * normB;

  double Xfactor = JacobiTheta(M_PI*(Ax - Bx), t, false) ;
  double Yfactor = JacobiTheta(M_PI*(Ay - By), t, false) ;
  double Zfactor = JacobiTheta(M_PI*(Az - Bz), t, false) ;
  Int(0,0) += wtG * prefactor *
      (Xfactor * Yfactor * Zfactor +
       Yfactor * Zfactor + Xfactor * Zfactor + Xfactor * Yfactor +
       Xfactor + Yfactor + Zfactor) * normA * normB;
  return Int(0,0);

  /*
  double t = expG * expA * expB / (expG*expA + expG*expB + expA*expB);

  double mu = expG * expB / (expG + expB);

  calc1DOvlpPeriodic(LA, Ax, expA, LB, Bx, mu, Sx);
  calc1DOvlpPeriodic(LA, Ay, expA, LB, By, mu, Sy);
  calc1DOvlpPeriodic(LA, Az, expA, LB, Bz, mu, Sz);

  Sx2.block(0,0,LA+1, LB+1).setZero();
  Sy2.block(0,0,LA+1, LB+1).setZero();
  Sz2.block(0,0,LA+1, LB+1).setZero();
  
  double commonFactor = sqrt(M_PI/(expB+expG));
  double f = sqrt(2. * (expB+expG));

  for (int j=0; j <= LB; j++) {

    for (int n = 0; n <= j; n++) {
      if ( (j-n)%2 != 0) continue;

      double factor = nChoosek(j, n) * pow(expG/(expG+expB), n)
          * doubleFact(j - n - 1)/pow(f, 1.*j-1.*n) * commonFactor;
      //cout << j <<"  "<<factor<<"  "<<commonFactor<<endl;
      for (int i=0; i <= LA; i++) {
        Sx2(i, j) += factor * Sx(i, n);
        Sy2(i, j) += factor * Sy(i, n);
        Sz2(i, j) += factor * Sz(i, n);
      }
    }
  }
  //exit(0);

  int IA0 = (LA)*(LA+1)*(LA+2)/6;
  int IB0 = (LB)*(LB+1)*(LB+2)/6;

  for (int i=0; i<(LA+1)*(LA+2)/2; i++)
    for (int j=0; j<(LB+1)*(LB+2)/2; j++) {
      auto ipow = CartOrder[IA0 + i];
      auto jpow = CartOrder[IB0 + j];

      double Xfactor = Sx2(ipow[0], jpow[0])/commonFactor;
      double Yfactor = Sy2(ipow[1], jpow[1])/commonFactor;
      double Zfactor = Sz2(ipow[2], jpow[2])/commonFactor;

      Int(i, j) +=   wtG *
          (Xfactor * Yfactor * Zfactor +
           Yfactor * Zfactor + Xfactor * Zfactor + Xfactor * Yfactor +
           Xfactor + Yfactor + Zfactor) * normA * normB 
          *  pow(M_PI*M_PI*M_PI/(expG*expA*expB), 1.5);
          
    }
  */
}


double calcCoulombPotentialPeriodic(int LA, double Ax, double Ay, double Az,
                                    double expA, double normA,
                                    double expG, double wtG,
                                    double yx, double yy, double yz,
                                    MatrixXd& Int) {

  //assume that LA and LB = 0
  double t = expG * expA / (expG + expA);

  double prefactor = pow( M_PI*M_PI/ (expG*expA), 1.5) * normA;

  Int(0,0) += wtG * prefactor * 
      (   JacobiTheta(M_PI*(Ax - yx), t) 
        * JacobiTheta(M_PI*(Ay - yy), t)
        * JacobiTheta(M_PI*(Az - yz), t)  - 1.0);
  return Int(0,0);
}
