#include <iostream>
#include <cmath>
#include <string.h>

#include "primitives.h"
#include "workArray.h"



double calcOvlpMatrix(int LA, double Ax, double Ay, double Az, double expA,
                     int LB, double Bx, double By, double Bz, double expB,
                     tensor& S_2d) {


  calc1DOvlp(LA, Ax, expA, LB, Bx, expB, Sx_2d);
  calc1DOvlp(LA, Ay, expA, LB, By, expB, Sy_2d);
  calc1DOvlp(LA, Az, expA, LB, Bz, expB, Sz_2d);

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

      S_2d(i, j) =   Sx_2d(ipow[0], jpow[0]) *
                  Sy_2d(ipow[1], jpow[1]) *
                  Sz_2d(ipow[2], jpow[2]) * normA * normB;
    }
    
}



void calcCoulombIntegralPeriodic(int LA, double Ax, double Ay, double Az,
                                 double expA, double normA,
                                 int LB, double Bx, double By, double Bz,
                                 double expB, double normB,
                                 double Lx, double Ly, double Lz,
                                 tensor& Int, bool includeNorm) {

  if (includeNorm ) {
    normA *= 1.0/sqrt( doubleFact(2*LA-1)/pow(4*expA, LA))
        / pow(M_PI/2./expA, 0.75);// * sqrt(4* M_PI/ (2*LA+1));
    if (LA >=2) normA = normA * sqrt(4* M_PI/ (2*LA+1));
    normB *= 1.0/sqrt( doubleFact(2*LB-1)/pow(4*expB, LB))
        / pow(M_PI/2./expB, 0.75);// * sqrt(4 * M_PI / (2*LB+1));
    if (LB >=2) normB = normB * sqrt(4 * M_PI / (2*LB+1));
  }

  powExpA.dimensions = {LA+1}; powExpB.dimensions = {LB+1};
  powPIOverLx.dimensions = {LA+LB+1}; powPIOverLy.dimensions = {LA+LB+1}; powPIOverLz.dimensions = {LA+LB+1};
  for (int i=0; i<=LA; i++)
    powExpA(i) = pow(1./expA, i);
  for (int j=0; j<=LB; j++)
    powExpB(j) = pow(1./expB, j);
  for (int k=0; k<=LA+LB; k++) {
    powPIOverLx(k) = pow(M_PI/Lx, k);
    powPIOverLy(k) = pow(M_PI/Ly, k);
    powPIOverLz(k) = pow(M_PI/Lz, k);
  }

  Sx_2d.dimensions ={LA+1, LB+1}; Sy_2d.dimensions ={LA+1, LB+1}; Sz_2d.dimensions ={LA+1, LB+1}; 
  Sx2_2d.dimensions={LA+1, LB+1}; Sy2_2d.dimensions={LA+1, LB+1}; Sz2_2d.dimensions={LA+1, LB+1};
  //Sx_2d.dimensions ={LA+1, LA+1}; Sy_2d.dimensions ={LA+1, LA+1}; Sz_2d.dimensions ={LA+1, LA+1}; 
  //Sx2_2d.dimensions={LA+1, LA+1}; Sy2_2d.dimensions={LA+1, LA+1}; Sz2_2d.dimensions={LA+1, LA+1};
  
  double prevt = -1.0;
  Coulomb_14_14_8& coulomb = coulomb_14_14_8;
  for (int i=0; i<coulomb.exponents.size(); i++) {
    double expG = coulomb.exponents[i], wtG = coulomb.weights[i];
    
    //assume that LA and LB = 0
    double t = expG * expA * expB / (expG*expA + expG*expB + expA*expB);
    double prefactor = pow(M_PI*M_PI*M_PI/(expG*expA*expB), 1.5) * normA * normB / (Lx * Ly * Lz);

    //if (abs(t) < 1.e-6) continue;
    if (abs(prevt - t) > 1.e-6) {
      calc1DOvlpPeriodic(LA, Ax, expA, LB, Bx, expB, t, 0, Lx, Sx_2d, &workArray[0], powPIOverLx, powExpA, powExpB);
      calc1DOvlpPeriodic(LA, Ay, expA, LB, By, expB, t, 0, Ly, Sy_2d, &workArray[0], powPIOverLy, powExpA, powExpB);
      calc1DOvlpPeriodic(LA, Az, expA, LB, Bz, expB, t, 0, Lz, Sz_2d, &workArray[0], powPIOverLz, powExpA, powExpB);

      calc1DOvlpPeriodic(LA, Ax, expA, LB, Bx, expB, 0, 0, Lx, Sx2_2d, &workArray[0], powPIOverLx, powExpA, powExpB);
      calc1DOvlpPeriodic(LA, Ay, expA, LB, By, expB, 0, 0, Ly, Sy2_2d, &workArray[0], powPIOverLy, powExpA, powExpB);
      calc1DOvlpPeriodic(LA, Az, expA, LB, Bz, expB, 0, 0, Lz, Sz2_2d, &workArray[0], powPIOverLz, powExpA, powExpB);
      
    }
    prevt = t;

    
    int IA0 = (LA)*(LA+1)*(LA+2)/6;
    int IB0 = (LB)*(LB+1)*(LB+2)/6;

    
    //now calculate the 3d integrals.
    for (int i=0; i<(LA+1)*(LA+2)/2; i++)
      for (int j=0; j<(LB+1)*(LB+2)/2; j++) {
        vector<int>& ipow = CartOrder[IA0 + i];
        vector<int>& jpow = CartOrder[IB0 + j];

        /*
        double k0Termx = DerivativeToPolynomial(ipow[0], 0)
            * DerivativeToPolynomial(jpow[0], 0)
            * powExpA(ipow[0]/2)  * powExpB(jpow[0]/2);
        double k0Termy = DerivativeToPolynomial(ipow[1], 0)
            * DerivativeToPolynomial(jpow[1], 0)
            * powExpA(ipow[1]/2)  * powExpB(jpow[1]/2);
        //* pow(1./expA, ipow[1]/2) * pow(1./expB, jpow[1]/2);
        double k0Termz = DerivativeToPolynomial(ipow[2], 0)
            * DerivativeToPolynomial(jpow[2], 0)
            * powExpA(ipow[2]/2)  * powExpB(jpow[2]/2);
        //* pow(1./expA, ipow[2]/2) * pow(1./expB, jpow[2]/2);
        */
      
        Int(i, j) +=  prefactor * ( (Sx_2d(ipow[0], jpow[0])) *
                                    (Sy_2d(ipow[1], jpow[1])) *
                                    (Sz_2d(ipow[2], jpow[2])) -
                                    (Sx2_2d(ipow[0], jpow[0])) *
                                    (Sy2_2d(ipow[1], jpow[1])) *
                                    (Sz2_2d(ipow[2], jpow[2]))) * wtG;
        //- k0Termx*k0Termy*k0Termz) * wtG ;
      }


  }
}


void calcKineticIntegralPeriodic(int LA, double Ax, double Ay, double Az,
                                 double expA, double normA,
                                 int LB, double Bx, double By, double Bz,
                                 double expB, double normB,
                                 double Lx, double Ly, double Lz,
                                 tensor& Int, bool Includenorm) {
  if (Includenorm) {
    normA *= 1.0/sqrt( doubleFact(2*LA-1)/pow(4*expA, LA))
        / pow(M_PI/2./expA, 0.75);// * sqrt(4* M_PI/ (2*LA+1));
    if (LA >=2) normA = normA * sqrt(4* M_PI/ (2*LA+1));
    normB *= 1.0/sqrt( doubleFact(2*LB-1)/pow(4*expB, LB))
        / pow(M_PI/2./expB, 0.75);// * sqrt(4 * M_PI / (2*LB+1));
    if (LB >=2) normB = normB * sqrt(4 * M_PI / (2*LB+1));
  }
  
  //assume that LA and LB = 0
  double prefactor = pow(M_PI*M_PI/(expA*expB), 1.5) * normA * normB / (Lx * Ly * Lz);
  double t = expA*expB/(expA + expB);

  powExpA.dimensions = {LA+3}; powExpB.dimensions = {LB+3};
  powPIOverLx.dimensions = {LA+LB+3}; powPIOverLy.dimensions = {LA+LB+3}; powPIOverLz.dimensions = {LA+LB+3};
  for (int i=0; i<=LA+2; i++)
    powExpA(i) = pow(1./expA, i);
  for (int j=0; j<=LB+2; j++)
    powExpB(j) = pow(1./expB, j);
  for (int k=0; k<=LA+LB+3; k++) {
    powPIOverLx(k) = pow(M_PI/Lx, k);
    powPIOverLy(k) = pow(M_PI/Ly, k);
    powPIOverLz(k) = pow(M_PI/Lz, k);
  }
  
  Sx_2d.dimensions ={LA+3, LB+3}; Sy_2d.dimensions ={LA+3, LB+3}; Sz_2d.dimensions ={LA+3, LB+3}; 
  Sx2_2d.dimensions={LA+3, LB+3}; Sy2_2d.dimensions={LA+3, LB+3}; Sz2_2d.dimensions={LA+3, LB+3};
  
  calc1DOvlpPeriodic(LA, Ax, expA, LB, Bx, expB, t, 2, Lx, Sx_2d, &workArray[0], powPIOverLx, powExpA, powExpB);
  calc1DOvlpPeriodic(LA, Ay, expA, LB, By, expB, t, 2, Ly, Sy_2d, &workArray[0], powPIOverLy, powExpA, powExpB);
  calc1DOvlpPeriodic(LA, Az, expA, LB, Bz, expB, t, 2, Lz, Sz_2d, &workArray[0], powPIOverLz, powExpA, powExpB);
  

  int IA0 = (LA)*(LA+1)*(LA+2)/6;
  int IB0 = (LB)*(LB+1)*(LB+2)/6;

  //now calculate the 3d integrals.
  for (int i=0; i<(LA+1)*(LA+2)/2; i++)
    for (int j=0; j<(LB+1)*(LB+2)/2; j++) {
      vector<int>& ipow = CartOrder[IA0 + i];
      vector<int>& jpow = CartOrder[IB0 + j];

      double derivX = 4*expB*expB * Sx_2d(ipow[0], jpow[0]+2)
          - 2. * expB * (2 * jpow[0] + 1) * Sx_2d(ipow[0], jpow[0])          
          + (jpow[0] >= 2 ? jpow[0]*(jpow[0]-1) * Sx_2d(ipow[0], jpow[0]-2) : 0.0);

      double derivY = 4*expB*expB * Sy_2d(ipow[1], jpow[1]+2)
          - 2. * expB * (2 * jpow[1] + 1) * Sy_2d(ipow[1], jpow[1])          
          + (jpow[1] >= 2 ? jpow[1]*(jpow[1]-1) * Sy_2d(ipow[1], jpow[1]-2) : 0.0);

      double derivZ = 4*expB*expB * Sz_2d(ipow[2], jpow[2]+2)
          - 2. * expB * (2 * jpow[2] + 1) * Sz_2d(ipow[2], jpow[2])          
          + (jpow[2] >= 2 ? jpow[2]*(jpow[2]-1) * Sz_2d(ipow[2], jpow[2]-2) : 0.0);

      Int(i, j) +=  -0.5 * prefactor * ( derivX * Sy_2d(ipow[1], jpow[1]) * Sz_2d(ipow[2], jpow[2]) +
                                         derivY * Sx_2d(ipow[0], jpow[0]) * Sz_2d(ipow[2], jpow[2]) +
                                         derivZ * Sy_2d(ipow[1], jpow[1]) * Sx_2d(ipow[0], jpow[0]) );
    }
}

void calcOverlapIntegralPeriodic(int LA, double Ax, double Ay, double Az,
                                 double expA, double normA,
                                 int LB, double Bx, double By, double Bz,
                                 double expB, double normB,
                                 double Lx, double Ly, double Lz,
                                 tensor& Int, bool Includenorm) {

  if (Includenorm) {
    normA *= 1.0/sqrt( doubleFact(2*LA-1)/pow(4*expA, LA))
        / pow(M_PI/2./expA, 0.75);// * sqrt(4* M_PI/ (2*LA+1));
    if (LA >=2) normA = normA * sqrt(4* M_PI/ (2*LA+1));
    normB *= 1.0/sqrt( doubleFact(2*LB-1)/pow(4*expB, LB))
        / pow(M_PI/2./expB, 0.75);// * sqrt(4 * M_PI / (2*LB+1));
    if (LB >=2) normB = normB * sqrt(4 * M_PI / (2*LB+1));
  }
  
  //assume that LA and LB = 0
  double prefactor = pow(M_PI*M_PI/(expA*expB), 1.5) * normA * normB / (Lx * Ly * Lz);
  double t = expA*expB/(expA + expB);

  powExpA.dimensions = {LA+1}; powExpB.dimensions = {LB+1};
  powPIOverLx.dimensions = {LA+LB+1}; powPIOverLy.dimensions = {LA+LB+1}; powPIOverLz.dimensions = {LA+LB+1};
  for (int i=0; i<=LA; i++)
    powExpA(i) = pow(1./expA, i);
  for (int j=0; j<=LB; j++)
    powExpB(j) = pow(1./expB, j);
  for (int k=0; k<=LA+LB; k++) {
    powPIOverLx(k) = pow(M_PI/Lx, k);
    powPIOverLy(k) = pow(M_PI/Ly, k);
    powPIOverLz(k) = pow(M_PI/Lz, k);
  }
  
  Sx_2d.dimensions ={LA+1, LB+1}; Sy_2d.dimensions ={LA+1, LB+1}; Sz_2d.dimensions ={LA+1, LB+1}; 
  Sx2_2d.dimensions={LA+1, LB+1}; Sy2_2d.dimensions={LA+1, LB+1}; Sz2_2d.dimensions={LA+1, LB+1};
  
  calc1DOvlpPeriodic(LA, Ax, expA, LB, Bx, expB, t, 0, Lx, Sx_2d, &workArray[0], powPIOverLx, powExpA, powExpB);
  calc1DOvlpPeriodic(LA, Ay, expA, LB, By, expB, t, 0, Ly, Sy_2d, &workArray[0], powPIOverLy, powExpA, powExpB);
  calc1DOvlpPeriodic(LA, Az, expA, LB, Bz, expB, t, 0, Lz, Sz_2d, &workArray[0], powPIOverLz, powExpA, powExpB);
  
  
  int IA0 = (LA)*(LA+1)*(LA+2)/6;
  int IB0 = (LB)*(LB+1)*(LB+2)/6;

  
  //now calculate the 3d integrals.
  for (int i=0; i<(LA+1)*(LA+2)/2; i++)
    for (int j=0; j<(LB+1)*(LB+2)/2; j++) {
      vector<int>& ipow = CartOrder[IA0 + i];
      vector<int>& jpow = CartOrder[IB0 + j];
        
      Int(i, j) +=  prefactor * ( (Sx_2d(ipow[0], jpow[0])) *
                                  (Sy_2d(ipow[1], jpow[1])) *
                                  (Sz_2d(ipow[2], jpow[2])) );
    }
}



void calcShellIntegral_2c(char* Name, double* integrals, int sh1, int sh2, int* ao_loc,
                          int* atm, int natm, int* bas, int nbas,
                          double* env, double* Lattice) {

  double Lx = Lattice[0], Ly = Lattice[4], Lz = Lattice[8];
  
  int nbas1 = bas[8* sh1 + 3], nbas2 = bas[8*sh2 +3];
  int LA = bas[8 * sh1 +1], LB = bas[8 * sh2 +1];
  int nLA = (LA+1)*(LA+2)/2,  nLB = (LB+1)*(LB+2)/2;
  int Dim1 = nbas1 * nLA,
      Dim2 = nbas2 * nLB;

  int stride1 = Dim2;
  int stride2 = 1;

  teri.dimensions = {nLA, nLB};

  std::fill(integrals, integrals+Dim1*Dim2, 0.0);

  double scaleLA = LA < 2 ? sqrt( (2*LA+1)/(4*M_PI)) : 1.0;
  double scaleLB = LB < 2 ? sqrt( (2*LB+1)/(4*M_PI)) : 1.0;
  double scaleLz = scaleLA * scaleLB ;

  
  int start1 = bas[8 * sh1 + 5], end1 = bas[8 * sh1 + 6];
  int sz1 = end1 - start1;
  for (int ia = start1; ia<end1; ia++) {
    int atmIndex = bas[8*sh1];
    int LA = bas[8 * sh1 +1], pos =  atm[6*atmIndex + 1];//20 + 4*bas[8*sh1];
    double Ax = env[pos],  Ay = env[pos + 1],  Az = env[pos + 2];
    double expA = env[ ia ];
    
    
    int start2 = bas[8 * sh2 + 5], end2 = bas[8 * sh2 + 6];
    int sz2 = end2 - start2;
    for (int ja = start2; ja<end2; ja++) {
      int atmIndex = bas[8*sh2];
      int LB = bas[8 * sh2 +1], pos =  atm[6*atmIndex + 1];//20 + 4*bas[8*sh2];
      double Bx = env[pos],  By = env[pos + 1],  Bz = env[pos + 2];
      double expB = env[ ja ];
      
      
      teri.setZero();
      if (strncmp(Name, "ovl", 3)==0) {
          calcOverlapIntegralPeriodic(
              LA, Ax, Ay, Az, expA, 1.0,
              LB, Bx, By, Bz, expB, 1.0,
              Lx, Ly, Lz, teri, false);
      }      
      else if (strncmp(Name, "kin", 3)==0) {
          calcKineticIntegralPeriodic(
              LA, Ax, Ay, Az, expA, 1.0,
              LB, Bx, By, Bz, expB, 1.0,
              Lx, Ly, Lz, teri, false);
      }      
      else if (strncmp(Name, "pot", 3)==0) {
          calcCoulombIntegralPeriodic(
              LA, Ax, Ay, Az, expA, 1.0,
              LB, Bx, By, Bz, expB, 1.0,
              Lx, Ly, Lz, teri, false);
      }      
      else 
        cout << "Integral "<<Name<<" not supported"<<endl;
      
      //now put teri in the correct location in the integrals
      for (int ii = 0; ii<nbas1; ii++)
      for (int jj = 0; jj<nbas2; jj++)
      {
        double scale = env[bas[8* sh1 +5] + (ii+1)*sz1 + (ia-start1)] *
            env[bas[8* sh2 +5] + (jj+1)*sz2 + (ja-start2)] ;
        
        for (int i=0; i<nLA; i++)
        for (int j=0; j<nLB; j++) {
          integrals[ (ii * nLA + i)*stride1 +
                     (jj * nLB + j)*stride2 ] += teri(i, j)*scale * scaleLz;
        }
                
      }
    } //ja
  }//ia
  


}

void calcIntegral_2c(char* Name, double* integrals, int* shls, int* ao_loc,
                     int* atm, int natm, int* bas, int nbas,
                     double* env, double* Lattice) {
  int dim = 0;
  for (int i=0; i<nbas; i++)
    if (ao_loc[i+1] - ao_loc[i] > dim)
      dim = ao_loc[i+1] - ao_loc[i];
  
  tensor teri( {dim, dim});

  int n1 = ao_loc[shls[1]] - ao_loc[shls[0]],
      n2 = ao_loc[shls[3]] - ao_loc[shls[2]];
  
  for(int sh1 = shls[0]; sh1 < shls[1]; sh1++) 
  for(int sh2 = shls[2]; sh2 < shls[3]; sh2++) {

    calcShellIntegral_2c(Name, teri.vals, sh1, sh2, ao_loc, atm, natm, bas, nbas, env, Lattice);

    teri.dimensions = {ao_loc[sh1+1] - ao_loc[sh1], ao_loc[sh2+1]-ao_loc[sh2]};
    
    for (int i = ao_loc[sh1] - ao_loc[shls[0]]; i < ao_loc[sh1+1] - ao_loc[shls[0]]; i++) 
    for (int j = ao_loc[sh2] - ao_loc[shls[2]]; j < ao_loc[sh2+1] - ao_loc[shls[2]]; j++) {
      int I = i - (ao_loc[sh1] - ao_loc[shls[0]]), J = j - (ao_loc[sh2] - ao_loc[shls[2]]);
      integrals[i*n2 + j] = teri(I, J);
    }

    
  }//sh2

}


//actually this calculates phi_i(r, a, A) phi_0(r,tau, r') phi_j(r', b, B)
//where phi_0 is one of the gaussians approximating 1/r
/*
double calcCoulombIntegral(int LA, double Ax, double Ay, double Az,
                           double expA, double normA,
                           double expG, double wtG,
                           int LB, double Bx, double By, double Bz,
                           double expB, double normB,
                           tensor& Int) {

  double mu = expG * expB / (expG + expB);
  double prefactor = sqrt(M_PI/(expG + expB)) * normA * normB;
                         

  calc1DOvlp(LA, Ax, expA, LB, Bx, mu, Sx);  
  calc1DOvlp(LA, Ay, expA, LB, By, mu, Sy_2d);  
  calc1DOvlp(LA, Az, expA, LB, Bz, mu, Sz);

  Sx2.block(0,0,LA+1, LB+1).setZero();
  Sy_2d2.block(0,0,LA+1, LB+1).setZero();
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
        Sy_2d2(i, j) += factor * Sy_2d(i, n);
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
          Sy_2d2(ipow[1], jpow[1]) *
          Sz2(ipow[2], jpow[2]) * normA * normB * wtG;
    }  
}
*/
