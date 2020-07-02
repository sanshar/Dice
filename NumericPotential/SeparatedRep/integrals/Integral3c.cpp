#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

#include "primitives.h"
#include "workArray.h"

using namespace std;
using namespace std::chrono;

void initArraySize(vector<int> coeffSize, vector<int> S2size,
                   vector<int> Ssize) {
  Coeffx_3d.dimensions = coeffSize; Coeffy_3d.dimensions = coeffSize; Coeffz_3d.dimensions = coeffSize;

  Sx_3d.dimensions = S2size; Sy_3d.dimensions = S2size; Sz_3d.dimensions = S2size;
  Sx2_3d.dimensions = S2size; Sy2_3d.dimensions = S2size; Sz2_3d.dimensions = S2size;

  Sx_2d.dimensions = Ssize; Sy_2d.dimensions = Ssize; Sz_2d.dimensions = Ssize;
  Sx2_2d.dimensions = Ssize; Sy2_2d.dimensions = Ssize; Sz2_2d.dimensions = Ssize;
}

void generateCoefficientMatrix(int LA, int LB, double expA, double expB,
                               double ABx, double p, tensor& Coeff_3d) {

  expaPow[0] = 1.;
  for (int i=1; i<= LB; i++)
    expaPow[i] = expaPow[i-1] * expA * ABx/p;
  
  expbPow[0] = 1.;
  for (int i=1; i<= LA; i++)
    expbPow[i] = expbPow[i-1] * expB * ABx/p;

  std::fill(Coeff_3d.vals, Coeff_3d.vals+(LA+1)*(LB+1)*(LA+LB+1), 0.0);
  //these are the 3d coefficients that take you from xP^{i+j} -> xA^i xB^j
  for (int i=0; i<=LA; i++)
    for (int j=0; j<=LB; j++) {
      
      for (int m=0; m<=i; m++)
        for (int l=0; l<=j; l++) {
          //double prefactor = pow(-1, i-m) * nChoosek(i, m) * nChoosek(j, l);
          double prefactor = ((i-m)%2 == 0 ? 1. : -1.) * nChoosek(i, m) * nChoosek(j, l);
          Coeff_3d(i, j, m+l) += prefactor * expbPow[i-m] * expaPow[j-l];
          //Coeff_3d(i, j, m+l) += prefactor * pow(expB * ABx/p, i-m) * pow(expA * ABx/p, j-l);
        }
    }
}

double calc1DOvlpPeriodicSumB(int LA, double Ax, double expA,
                              int LB, double Bx, double expB,
                              int LC, double Cx, double expC,
                              double expG, int AdditionalDeriv,
                              double Lx, tensor& Sx_3d, tensor& Sx2_3d,
                              tensor& powPIOverLx) 
{


  //THREE CASES
  if ( expA > 0.3/Lx/Lx && expB > 0.3/Lx/Lx) { //then explicit sums have to be performed
    double p = expA + expB;
    double t = expG * p * expC / (expG*p + expG*expC + p*expC);
    double mu = expA * expB / (expA + expB);
    
    double beta = 0.0;
    for (int nx = 0; nx < 30; nx++) {
      {
        double Px = (expA * Ax + expB * (Bx+nx*Lx))/(expA + expB);
        double ABx = Ax - (Bx + nx * Lx);
        double productfactor = exp(-mu * ABx * ABx);
        
        generateCoefficientMatrix(LA, LB, expA, expB, ABx, p, Coeffx_3d);
        calc1DOvlpPeriodic(LA+LB, Px, p, LC, Cx, expC, t, 0, Lx, Sx_2d, &workArray[0], powPIOverLx, powExpAPlusExpB, powExpC);
        calc1DOvlpPeriodic(LA+LB, Px, p, LC, Cx, expC, 0, 0, Lx, Sx2_2d, &workArray[0], powPIOverLx, powExpAPlusExpB, powExpC);
        
        contract_IJK_IJL_LK( &Sx_3d, &Coeffx_3d,  &Sx_2d, productfactor, beta); 
        contract_IJK_IJL_LK(&Sx2_3d, &Coeffx_3d, &Sx2_2d, productfactor, beta);
      }
      
      if (nx != 0)
      {
        double Px = (expA * Ax + expB * (Bx-nx*Lx))/(expA + expB);
        double ABx = Ax - (Bx - nx * Lx);
        double productfactor = exp(-mu * ABx * ABx);
        
        generateCoefficientMatrix(LA, LB, expA, expB, ABx, p, Coeffx_3d);
        calc1DOvlpPeriodic(LA+LB, Px, p, LC, Cx, expC, t, 0, Lx, Sx_2d, &workArray[0], powPIOverLx, powExpAPlusExpB, powExpC);
        calc1DOvlpPeriodic(LA+LB, Px, p, LC, Cx, expC, 0, 0, Lx, Sx2_2d, &workArray[0], powPIOverLx, powExpAPlusExpB, powExpC);
        
        contract_IJK_IJL_LK(&Sx_3d, &Coeffx_3d, &Sx_2d, productfactor, 1.0); 
        contract_IJK_IJL_LK(&Sx2_3d, &Coeffx_3d, &Sx2_2d, productfactor, 1.0); 
      }
      else 
        beta = 1.0;

      double max = min (abs(Ax - (Bx + (nx+1) * Lx)), abs(Ax - (Bx - (nx+1) * Lx )));
      if (exp(-mu*max*max) < 1.e-10) break;
    }
  }
  else if (expB <= 0.3/Lx/Lx) {//then theta function A is just a constant
    double t = expG * expA * expC / (expG*expA + expG*expC + expA*expC);
    calc1DOvlpPeriodic(LA, Ax, expA, LC, Cx, expC, t, 0, Lx, Sx_2d, &workArray[0], powPIOverLx, powExpA, powExpC);
    calc1DOvlpPeriodic(LA, Ax, expA, LC, Cx, expC, 0, 0, Lx, Sx2_2d, &workArray[0], powPIOverLx, powExpA, powExpC);

    double Bfactor = sqrt(M_PI/(expB*Lx*Lx) * ((expA+expB)/expA));
    for (int j=0; j<= (LB+1)*(LB+2)/2; j++) {
      double factor = DerivativeToPolynomial(j, 0) * powExpB(j/2);
      for (int i=0; i<= (LA+1)*(LA+2)/2; i++)
        for (int k=0; k<= (LC+1)*(LC+2)/2; k++) {
          Sx_3d(i,j,k) = Bfactor * factor * Sx_2d(i,k) ;
          Sx2_3d(i,j,k) = Bfactor * factor * Sx2_2d(i,k) ;
        }
    }
    
  }
  else if (expA <= 0.3/Lx/Lx) {//then theta function A is just a constant
    double t = expG * expB * expC / (expG*expB + expG*expC + expB*expC);
    calc1DOvlpPeriodic(LB, Bx, expB, LC, Cx, expC, t, 0, Lx, Sx_2d, &workArray[0], powPIOverLx, powExpB, powExpC);
    calc1DOvlpPeriodic(LB, Bx, expB, LC, Cx, expC, 0, 0, Lx, Sx2_2d, &workArray[0], powPIOverLx, powExpB, powExpC);

    double Afactor = sqrt( (M_PI/expA/Lx/Lx) * ((expA+expB)/expB) );
    for (int i=0; i<= (LA+1)*(LA+2)/2; i++) {
      double factor = DerivativeToPolynomial(i, 0)  * powExpA(i/2);
      for (int j=0; j<= (LB+1)*(LB+2)/2; j++) 
        for (int k=0; k<= (LC+1)*(LC+2)/2; k++) {
          Sx_3d(i,j,k)  = Afactor * factor * Sx_2d(j,k) ;
          Sx2_3d(i,j,k) = Afactor * factor * Sx2_2d(j,k) ;
        }
    }
    
  }
}



//calculate primitive cartesian integrals with no A,B, C translations
//assuming periodized coulomb kernel
void calcCoulombIntegralPeriodic_noTranslations(
    int LA, double Ax, double Ay, double Az,
    double expA, double normA,
    int LB, double Bx, double By, double Bz,
    double expB, double normB,
    int LC, double Cx, double Cy, double Cz,
    double expC, double normC,
    double Lx, double Ly, double Lz,
    tensor& Int3d, Coulomb& coulomb,
    bool normalize) {

  if (normalize) {
    normA *= 1.0/sqrt( doubleFact(2*LA-1)/pow(4*expA, LA))
        / pow(M_PI/2./expA, 0.75);// * sqrt(4* M_PI/ (2*LA+1));
    if (LA >=2) normA = normA * sqrt(4* M_PI/ (2*LA+1));

    normB *= 1.0/sqrt( doubleFact(2*LB-1)/pow(4*expB, LB))
        / pow(M_PI/2./expB, 0.75);// * sqrt(4 * M_PI / (2*LB+1));
    if (LB >=2) normB = normB * sqrt(4 * M_PI / (2*LB+1));

    normC *= 1.0/sqrt( doubleFact(2*LC-1)/pow(4*expC, LC))
        / pow(M_PI/2./expC, 0.75);// * sqrt(4 * M_PI / (2*LB+1));
    if (LC >=2) normC = normC * sqrt(4 * M_PI / (2*LC+1));
  }


  double Px = (expA * Ax + expB * Bx)/(expA + expB),
         Py = (expA * Ay + expB * By)/(expA + expB),
         Pz = (expA * Az + expB * Bz)/(expA + expB);
  double p = (expA + expB), ABx = Ax-Bx, ABy = Ay-By, ABz = Az-Bz;
  double mu = expA * expB / (expA + expB);
  double productfactor = exp(-mu *(ABx*ABx + ABy*ABy + ABz*ABz)) * normA * normB;

  initArraySize({LA+1, LB+1, LA+LB+1}, {LA+1, LB+1, LC+1}, {LA+LB+1, LC+1});

  generateCoefficientMatrix(LA, LB, expA, expB, ABx, p, Coeffx_3d);
  generateCoefficientMatrix(LA, LB, expA, expB, ABy, p, Coeffy_3d);
  generateCoefficientMatrix(LA, LB, expA, expB, ABz, p, Coeffz_3d);

  powExpAPlusExpB.dimensions = {LA+LB+1}; powExpC.dimensions = {LC+1}; powExpA.dimensions = {LA+1}; powExpB.dimensions = {LB+1};
  powPIOverLx.dimensions = {LA+LB+1}; powPIOverLy.dimensions = {LA+LB+1}; powPIOverLz.dimensions = {LA+LB+1};
  for (int i=0; i<=LA+LB; i++)
    powExpAPlusExpB(i) = pow(1./p, i);
  for (int i=0; i<=LA; i++)
    powExpA(i) = pow(1./expA, i);
  for (int i=0; i<=LB; i++)
    powExpB(i) = pow(1./expB, i);
  for (int j=0; j<=LC; j++)
    powExpC(j) = pow(1./expC, j);
  for (int k=0; k<=LA+LB; k++) {
    powPIOverLx(k) = pow(M_PI/Lx, k);
    powPIOverLy(k) = pow(M_PI/Ly, k);
    powPIOverLz(k) = pow(M_PI/Lz, k);
  }
  
  int IA0 = (LA)*(LA+1)*(LA+2)/6;
  int IB0 = (LB)*(LB+1)*(LB+2)/6;
  int IC0 = (LC)*(LC+1)*(LC+2)/6;

  double prevt = -1.0;
  
  for (int i=0; i<coulomb.exponents.size(); i++) {
    double expG = coulomb.exponents[i], wtG = coulomb.weights[i];
    
    //assume that LA and LB = 0
    double t = expG * p * expC / (expG*p + expG*expC + p*expC);
    double prefactor = pow(M_PI*M_PI*M_PI/(expG*p*expC), 1.5) * normC / (Lx * Ly * Lz);
    
    
    if (abs(prevt - t) > 1.e-6) {
      calc1DOvlpPeriodic(LA+LB, Px, p, LC, Cx, expC, t, 0, Lx, Sx_2d, &workArray[0], powPIOverLx, powExpAPlusExpB, powExpC);
      calc1DOvlpPeriodic(LA+LB, Py, p, LC, Cy, expC, t, 0, Ly, Sy_2d, &workArray[0], powPIOverLy, powExpAPlusExpB, powExpC);
      calc1DOvlpPeriodic(LA+LB, Pz, p, LC, Cz, expC, t, 0, Lz, Sz_2d, &workArray[0], powPIOverLz, powExpAPlusExpB, powExpC);
      
      //bkground terms
      calc1DOvlpPeriodic(LA+LB, Px, p, LC, Cx, expC, 0, 0, Lx, Sx2_2d, &workArray[0], powPIOverLx, powExpAPlusExpB, powExpC);
      calc1DOvlpPeriodic(LA+LB, Py, p, LC, Cy, expC, 0, 0, Ly, Sy2_2d, &workArray[0], powPIOverLy, powExpAPlusExpB, powExpC);
      calc1DOvlpPeriodic(LA+LB, Pz, p, LC, Cz, expC, 0, 0, Lz, Sz2_2d, &workArray[0], powPIOverLz, powExpAPlusExpB, powExpC);
    }
    prevt = t;
    
    contract_IJK_IJL_LK(&Sx_3d, &Coeffx_3d, &Sx_2d); 
    contract_IJK_IJL_LK(&Sy_3d, &Coeffy_3d, &Sy_2d); 
    contract_IJK_IJL_LK(&Sz_3d, &Coeffz_3d, &Sz_2d);

    contract_IJK_IJL_LK(&Sx2_3d, &Coeffx_3d, &Sx2_2d); 
    contract_IJK_IJL_LK(&Sy2_3d, &Coeffy_3d, &Sy2_2d); 
    contract_IJK_IJL_LK(&Sz2_3d, &Coeffz_3d, &Sz2_2d);

    for (int a = 0; a< (LA+1)*(LA+2)/2; a++)
    for (int b = 0; b< (LB+1)*(LB+2)/2; b++)
    for (int c = 0; c< (LC+1)*(LC+2)/2; c++)
    {
      vector<int>& apow = CartOrder[IA0 + a];
      vector<int>& bpow = CartOrder[IB0 + b];
      vector<int>& cpow = CartOrder[IC0 + c];
      
      Int3d(a,b,c) += productfactor * prefactor * wtG *
  (Sx_3d(apow[0],bpow[0],cpow[0]) * Sy_3d(apow[1],bpow[1],cpow[1]) * Sz_3d(apow[2],bpow[2],cpow[2]) -
   Sx2_3d(apow[0],bpow[0],cpow[0]) * Sy2_3d(apow[1],bpow[1],cpow[1]) * Sz2_3d(apow[2],bpow[2],cpow[2]));
      
    }
  }
  
}


//calculate primitive cartesian integrals with no B translations
void calcCoulombIntegralPeriodic_BTranslations(
    int LA, double Ax, double Ay, double Az,
    double expA, double normA,
    int LB, double Bx, double By, double Bz,
    double expB, double normB,
    int LC, double Cx, double Cy, double Cz,
    double expC, double normC,
    double Lx, double Ly, double Lz,
    tensor& Int3d, Coulomb& coulomb,
    bool normalize) {

  if (normalize) {
    normA *= 1.0/sqrt( doubleFact(2*LA-1)/pow(4*expA, LA))
        / pow(M_PI/2./expA, 0.75);// * sqrt(4* M_PI/ (2*LA+1));
    if (LA >=2) normA = normA * sqrt(4* M_PI/ (2*LA+1));

    normB *= 1.0/sqrt( doubleFact(2*LB-1)/pow(4*expB, LB))
        / pow(M_PI/2./expB, 0.75);// * sqrt(4 * M_PI / (2*LB+1));
    if (LB >=2) normB = normB * sqrt(4 * M_PI / (2*LB+1));

    normC *= 1.0/sqrt( doubleFact(2*LC-1)/pow(4*expC, LC))
        / pow(M_PI/2./expC, 0.75);// * sqrt(4 * M_PI / (2*LB+1));
    if (LC >=2) normC = normC * sqrt(4 * M_PI / (2*LC+1));
  }

  powExpAPlusExpB.dimensions = {LA+LB+1}; powExpC.dimensions = {LC+1}; powExpA.dimensions = {LA+1}; powExpB.dimensions = {LB+1};
  powPIOverLx.dimensions = {LA+LB+1}; powPIOverLy.dimensions = {LA+LB+1}; powPIOverLz.dimensions = {LA+LB+1};
  for (int i=0; i<=LA+LB; i++)
    powExpAPlusExpB(i) = pow(1./(expA+expB), i);
  for (int j=0; j<=LC; j++)
    powExpC(j) = pow(1./expC, j);
  for (int j=0; j<=LA; j++)
    powExpA(j) = pow(1./expA, j);
  for (int j=0; j<=LB; j++)
    powExpB(j) = pow(1./expB, j);
  for (int k=0; k<=LA+LB; k++) {
    powPIOverLx(k) = pow(M_PI/Lx, k);
    powPIOverLy(k) = pow(M_PI/Ly, k);
    powPIOverLz(k) = pow(M_PI/Lz, k);
  }

  initArraySize({LA+1, LB+1, LA+LB+1}, {LA+1, LB+1, LC+1}, {LA+LB+1, LC+1});
  Sx_3d.setZero(); Sy_3d.setZero(); Sz_3d.setZero();

  //cout << (LA+1)*(LA+2)/2<<"  "<< (LB+1)*(LB+2)/2<<"  "<< (LC+1)*(LC+2)/2<<"  "<<(LA+1)*(LA+2)*(LB+1)*(LB+2) * (LC+1)*(LC+2)/8<<"  "<<endl;

  double prevt = -1.0;
  int nterms = 0;
  //for (int i=155; i<156; i++) {
  for (int i=0; i<coulomb.exponents.size(); i++) {
    double expG = coulomb.exponents[i], wtG = coulomb.weights[i];
    double p = (expA + expB); //ABx = Ax-Bx, ABy = Ay-By, ABz = Az-Bz;
    double mu = expA * expB / (expA + expB);

    double t = expG * p * expC / (expG*p + expG*expC + p*expC);
    double prefactor = pow(M_PI*M_PI*M_PI/(expG*p*expC), 1.5) * normA * normB * normC / (Lx * Ly * Lz);

    if (t < 0.4/Lx/Lx && t < 0.4/Ly/Ly && t < 0.4/Lz/Lz) continue;
    //if (abs(prefactor*wtG) < 1.e-10) continue;
    if ( abs(prevt - t) > 1.e-6) {
      //generate summation over x-traslations
      calc1DOvlpPeriodicSumB(LA, Ax, expA, LB, Bx, expB, LC, Cx, expC, expG, 0, Lx, Sx_3d, Sx2_3d, powPIOverLx);
      calc1DOvlpPeriodicSumB(LA, Ay, expA, LB, By, expB, LC, Cy, expC, expG, 0, Ly, Sy_3d, Sy2_3d, powPIOverLy);
      calc1DOvlpPeriodicSumB(LA, Az, expA, LB, Bz, expB, LC, Cz, expC, expG, 0, Lz, Sz_3d, Sz2_3d, powPIOverLz);
    }
    if (abs(prefactor * wtG) < 1.e-12) break;
    prevt = t;
    
    int IA0 = (LA)*(LA+1)*(LA+2)/6;
    int IB0 = (LB)*(LB+1)*(LB+2)/6;
    int IC0 = (LC)*(LC+1)*(LC+2)/6;

    //double maxInt = 0.0;
    for (int a = 0; a< (LA+1)*(LA+2)/2; a++)
    for (int b = 0; b< (LB+1)*(LB+2)/2; b++)
    for (int c = 0; c< (LC+1)*(LC+2)/2; c++)
    {
      vector<int>& apow = CartOrder[IA0 + a];
      vector<int>& bpow = CartOrder[IB0 + b];
      vector<int>& cpow = CartOrder[IC0 + c];
      
      Int3d(a,b,c) += prefactor * wtG *
  (Sx_3d (apow[0],bpow[0],cpow[0]) * Sy_3d (apow[1],bpow[1],cpow[1]) * Sz_3d (apow[2],bpow[2],cpow[2]) -
   Sx2_3d(apow[0],bpow[0],cpow[0]) * Sy2_3d(apow[1],bpow[1],cpow[1]) * Sz2_3d(apow[2],bpow[2],cpow[2]));
      
    }
    nterms++;
    //cout << i<<"  "<<prefactor<<"  "<<wtG<<"  "<<prefactor*wtG<<"  "<<t<<"  "<<Int3d(0,0,0)<<endl;
  }
  //cout << nterms<<" "<<coulomb.exponents.size()<<endl;
  //exit(0);
}

void calcShellIntegral(double* integrals, int sh1, int sh2, int sh3, int* ao_loc,
                       int* atm, int natm, int* bas, int nbas,
                       double* env, double* Lattice) {
  double Lx = Lattice[0], Ly = Lattice[4], Lz = Lattice[8];
  
  int nbas1 = bas[8* sh1 + 3], nbas2 = bas[8*sh2 +3], nbas3 = bas[8*sh3 +3];
  int LA = bas[8 * sh1 +1], LB = bas[8 * sh2 +1], LC = bas[8 * sh3 +1];
  int nLA = (LA+1)*(LA+2)/2,  nLB = (LB+1)*(LB+2)/2,  nLC = (LC+1)*(LC+2)/2;
  int Dim1 = nbas1 * nLA,
      Dim2 = nbas2 * nLB,
      Dim3 = nbas3 * nLC;
  int stride1 = Dim2 * Dim3;
  int stride2 = Dim3;
  int stride3 = 1;

  teri.dimensions = {nLA, nLB, nLC};

  std::fill(integrals, integrals+Dim1*Dim2*Dim3, 0.0);
  double scaleLA = LA < 2 ? sqrt( (2*LA+1)/(4*M_PI)) : 1.0;
  double scaleLB = LB < 2 ? sqrt( (2*LB+1)/(4*M_PI)) : 1.0;
  double scaleLC = LC < 2 ? sqrt( (2*LC+1)/(4*M_PI)) : 1.0;
  double scaleLz = scaleLA * scaleLB * scaleLC;
  
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
      
      int start3 = bas[8 * sh3 + 5], end3 = bas[8 * sh3 + 6];
      int sz3 = end3 - start3;
      for (int ka = start3; ka<end3; ka++) {
        int atmIndex = bas[8*sh3];
        int LC = bas[8 * sh3 +1], pos = atm[6*atmIndex + 1];
        
        double Cx = env[pos],  Cy = env[pos + 1],  Cz = env[pos + 2];
        double expC = env[ ka ];

        teri.setZero();

        calcCoulombIntegralPeriodic_BTranslations(
            LA, Ax, Ay, Az, expA, 1.0,
            LB, Bx, By, Bz, expB, 1.0,
            LC, Cx, Cy, Cz, expC, 1.0,
            Lx, Ly, Lz, teri, coulomb_14_14_8, false);

        
        //now put teri in the correct location in the integrals
        for (int ii = 0; ii<nbas1; ii++)
        for (int jj = 0; jj<nbas2; jj++)
        for (int kk = 0; kk<nbas3; kk++)
        {
          double scale = env[bas[8* sh1 +5] + (ii+1)*sz1 + (ia-start1)] *
              env[bas[8* sh2 +5] + (jj+1)*sz2 + (ja-start2)] *
              env[bas[8* sh3 +5] + (kk+1)*sz3 + (ka-start3)];
          
          for (int i=0; i<nLA; i++)
          for (int j=0; j<nLB; j++)
          for (int k=0; k<nLC; k++) {
            integrals[ (ii * nLA + i)*stride1 +
                       (jj * nLB + j)*stride2 +
                       (kk * nLC + k)*stride3 ] += teri(i, j, k) * scale * scaleLz;
          }
        }        
      } //ka
    } //ja
  }//ia
  
}


void calcIntegral_3c(double* integrals, int* shls, int* ao_loc,
                     int* atm, int natm, int* bas, int nbas,
                     double* env, double* Lattice) {

  int dim = 0;
  for (int i=0; i<nbas; i++)
    if (ao_loc[i+1] - ao_loc[i] > dim)
      dim = ao_loc[i+1] - ao_loc[i];
  
  tensor teri( {dim, dim, dim});

  int n1 = ao_loc[shls[1]] - ao_loc[shls[0]],
      n2 = ao_loc[shls[3]] - ao_loc[shls[2]],
      n3 = ao_loc[shls[5]] - ao_loc[shls[4]];

  
  for(int sh1 = shls[0]; sh1 < shls[1]; sh1++) {
    cout << sh1<<"  "<<shls[1]<<endl;
  for(int sh2 = shls[2]; sh2 <= sh1; sh2++) 

    //for(int sh2 = shls[2]; sh2 < shls[3]; sh2++) 
    for(int sh3 = shls[4]; sh3 < shls[5]; sh3++) {
      //cout << sh1<<"  "<<shls[1]<<"  "<<sh2<<"  "<<shls[3]<<"  "<<sh3<<"  "<<shls[5]<<endl;
      calcShellIntegral(teri.vals, sh1, sh2, sh3, ao_loc, atm, natm, bas, nbas, env, Lattice);
      
      teri.dimensions = {ao_loc[sh1+1] - ao_loc[sh1],
                         ao_loc[sh2+1] - ao_loc[sh2],
                         ao_loc[sh3+1] - ao_loc[sh3]};
      
      for (int i = ao_loc[sh1] - ao_loc[shls[0]]; i < ao_loc[sh1+1] - ao_loc[shls[0]]; i++) 
      for (int j = ao_loc[sh2] - ao_loc[shls[2]]; j < ao_loc[sh2+1] - ao_loc[shls[2]]; j++) 
      for (int k = ao_loc[sh3] - ao_loc[shls[4]]; k < ao_loc[sh3+1] - ao_loc[shls[4]]; k++) {
        integrals[i*n2*n3 + j*n3 + k] = teri(i - (ao_loc[sh1] - ao_loc[shls[0]]),
                                             j - (ao_loc[sh2] - ao_loc[shls[2]]),
                                             k - (ao_loc[sh3] - ao_loc[shls[4]]));
      }
    }//sh
  }
}




void calcShellNuclearIntegral(double* integrals, int sh1, int sh2, int* ao_loc,
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

  teri.dimensions = {nLA, nLB, 1};

  std::fill(integrals, integrals+Dim1*Dim2, 0.0);
  //double scaleLz = sqrt( (2*LA+1) * (2*LB+1))/(4*M_PI);
  double scaleLA = LA < 2 ? sqrt( (2*LA+1)/(4*M_PI)) : 1.0;
  double scaleLB = LB < 2 ? sqrt( (2*LB+1)/(4*M_PI)) : 1.0;
  double scaleLz = scaleLA * scaleLB ;

  
  int start1 = bas[8 * sh1 + 5], end1 = bas[8 * sh1 + 6];
  int sz1 = end1 - start1;
  for (int ia = start1; ia<end1; ia++) {
    int atmIndex = bas[8*sh1];
    int LA = bas[8 * sh1 +1], pos =  atm[6*atmIndex + 1];//20 + 4*bas[8*sh1];
    //int LA = bas[8 * sh1 +1], pos = 20 + 4*bas[8*sh1];
    double Ax = env[pos],  Ay = env[pos + 1],  Az = env[pos + 2];
    double expA = env[ ia ];
    
    
    int start2 = bas[8 * sh2 + 5], end2 = bas[8 * sh2 + 6];
    int sz2 = end2 - start2;
    for (int ja = start2; ja<end2; ja++) {
      int atmIndex = bas[8*sh2];
      int LB = bas[8 * sh2 +1], pos =  atm[6*atmIndex + 1];//20 + 4*bas[8*sh2];
      //int LB = bas[8 * sh2 +1], pos = 20 + 4*bas[8*sh2];
      double Bx = env[pos],  By = env[pos + 1],  Bz = env[pos + 2];
      double expB = env[ ja ];
      
      
      teri.setZero();
      for (int ka = 0; ka<natm; ka++) {
        //int atmIndex = bas[8*sh3];
        int LC = 0, pos = atm[6*ka + 1];
        double Cx = env[pos],  Cy = env[pos + 1],  Cz = env[pos + 2];
        double zEta = 1.e11, Z = -atm[6*ka];

        //teri(0,0,0) = 0.0;
        calcCoulombIntegralPeriodic_BTranslations(
            LA, Ax, Ay, Az, expA, 1.0,
            LB, Bx, By, Bz, expB, 1.0,
            LC, Cx, Cy, Cz, zEta, Z*pow(zEta/M_PI, 1.5), //it uses sqrt(norm)
            Lx, Ly, Lz, teri, coulomb_14_14_8, false);

      }
      
      //now put teri in the correct location in the integrals
      for (int ii = 0; ii<nbas1; ii++)
      for (int jj = 0; jj<nbas2; jj++)
      {
        double scale = env[bas[8* sh1 +5] + (ii+1)*sz1 + (ia-start1)] *
            env[bas[8* sh2 +5] + (jj+1)*sz2 + (ja-start2)] ;
        
        for (int i=0; i<nLA; i++)
        for (int j=0; j<nLB; j++)
          integrals[ (ii * nLA + i)*stride1 +
                     (jj * nLB + j)*stride2 ] += teri(i, j, 0)*scale * scaleLz;

      }
    } //ja
  }//ia
  
}


void calcNuclearIntegral(double* integrals, int* shls, int* ao_loc,
                         int* atm, int natm, int* bas, int nbas,
                         double* env, double* Lattice) {

  int dim = 0;
  for (int i=0; i<nbas; i++)
    if (ao_loc[i+1] - ao_loc[i] > dim)
      dim = ao_loc[i+1] - ao_loc[i];
  
  tensor teri( {dim, dim, 1});

  int n1 = ao_loc[shls[1]] - ao_loc[shls[0]],
      n2 = ao_loc[shls[3]] - ao_loc[shls[2]];

  for(int sh1 = shls[0]; sh1 < shls[1]; sh1++) 
  for(int sh2 = shls[2]; sh2 < shls[3]; sh2++) {

    calcShellNuclearIntegral(teri.vals, sh1, sh2, ao_loc, atm, natm, bas, nbas, env, Lattice);

    teri.dimensions = {ao_loc[sh1+1] - ao_loc[sh1], ao_loc[sh2+1]-ao_loc[sh2], 1};
    
    for (int i = ao_loc[sh1] - ao_loc[shls[0]]; i < ao_loc[sh1+1] - ao_loc[shls[0]]; i++) 
    for (int j = ao_loc[sh2] - ao_loc[shls[2]]; j < ao_loc[sh2+1] - ao_loc[shls[2]]; j++) {
      int I = i - (ao_loc[sh1] - ao_loc[shls[0]]), J = j - (ao_loc[sh2] - ao_loc[shls[2]]);
      integrals[i*n2 + j] = teri(I, J, 0);
    }

    
  }//sh2

}

