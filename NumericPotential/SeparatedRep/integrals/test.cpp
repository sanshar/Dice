#include <iostream>
#include <ctime>
#include "math.h"
#include <chrono>

#include "Integral2c.h"
#include "Integral3c.h"
#include "workArray.h"
#include "primitives.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char** argv) {
  initWorkArray(); //initializes some arrays
  
  cout.precision(14);

  double Lx = 1.0, Ly = 1.0, Lz = 1.0;
  double Ax=0.11*Lx, Ay=0.12*Ly, Az=0.13*Lz, expA =  0.17; //gaussian center and exponent
  double Bx=0.60*Lx, By=0.61*Ly, Bz=0.62*Lz, expB =  0.7; //gaussian center and exponent
  double Cx=0.5*Lx, Cy=0.5*Ly, Cz=0.5*Lz, expC =  60; //gaussian center and exponent

  //the order of polynomial on bra and ket N = i+j+k (i,j,k are polynomial order)
  int NA, NB, NC; 
  S_2d.setZero(); NA=0; NB = 1; NC = 0;

  double zEta = 1.e11, Z = -4.0;

  auto start = high_resolution_clock::now();

  /*
  vector<double> work(100,0);
  JacobiThetaDerivative(0.0, 9.6313216766119, &work[0], 6, true);
  cout << work[0]<<"  "<<work[1]<<"  "<<work[2]<<"  "<<work[3]<<endl;
  work[0] = 0.0;
  JacobiThetaDerivative(-1.0e-12, 9.6313216766119, &work[0], 6, true);
  cout << work[0]<<"  "<<work[1]<<"  "<<work[2]<<"  "<<work[3]<<endl;
  work[0] = 0.0;
  JacobiThetaDerivative(2.*M_PI-1.0e-12, 9.6313216766119, &work[0], 6, true);
  cout << work[0]<<"  "<<work[1]<<"  "<<work[2]<<"  "<<work[3]<<endl;
  exit(0);
  */
  
  S_3d.setZero();
  /*
  calcCoulombIntegralPeriodic_BTranslations
      (NA, Ax, Ay, Az, expA, 1.0,
       NA, Ax, Ay, Az, expA, 1.0,
       NC, Ax, Ay, Az, expA, 1.0, //it uses sqrt(norm)
       Lx, Ly, Lz, S_3d, coulomb_14_14_8, true);
  */
  calcCoulombIntegralPeriodic_BTranslations
      (NA, Ax, Ay, Az, expA, 1.0,
       NB, Ax, Ay, Az, expB, 1.0,
       NC, Bx, By, Bz, zEta, Z*pow(zEta/M_PI/2., 0.75), //it uses sqrt(norm)
       Lx, Ly, Lz, S_3d, coulomb_14_14_8, true);

  cout <<"-->  "<< S_3d(0,0,0)<<endl;

  for (int a = 0; a< (NA+1)*(NA+2)/2; a++) {
    for (int b = 0; b< (NB+1)*(NB+2)/2; b++)
      for (int c = 0; c< (NC+1)*(NC+2)/2; c++)
        cout <<"-> "<< S_3d(a,b,c)<<"   ";

    cout << endl;
  }


  NA = 1; NB = 1;
  S_2d.setZero(); Lx = 3.5668/0.529177208;
  calcCoulombIntegralPeriodic(1, 1.68506879, 1.68506879, 1.68506879, 0.3376720, 1.0,
                              1, 0, 0, 0, 13.5472892, 1.0,
                              Lx, Lx, Lx, S_2d, false);
  /*
  calcCoulombIntegralPeriodic(1, 0, 0, 0, 13.5472892, 1.0,
                              1, 1.68506879, 1.68506879, 1.68506879, 0.3376720, 1.0,
                              Lx, Lx, Lx, S_2d, false);
  */

  for (int a = 0; a< (NA+1)*(NA+2)/2; a++) {
    for (int b = 0; b< (NB+1)*(NB+2)/2; b++)
      cout << S_2d(a,b)<<"   ";

    cout << endl;
  }
  
  /*
  calcCoulombIntegralPeriodic(NA, Ax, Ay, Az, expA, 1.0,
                              NB, Ax, Ay, Az, expA, 1.0,
                              Lx, Ly, Lz, S_2d);
  */
  cout << S_2d(1,1)<<endl;
  
  exit(0);
  /*
  for (int i=0; i<1000*100; i++) {
    calcCoulombIntegralPeriodic
        (NA, Ax, Ay, Az, expA, 1.0,
         NB, Bx, By, Bz, expB, 1.0,
         Lx, Ly, Lz, S_2d);  

  }
  
  for (int a = 0; a< (NA+1)*(NA+2)/2; a++) {
    for (int b = 0; b< (NB+1)*(NB+2)/2; b++)
      cout << S_2d(a,b)<<"   ";

    cout << endl;
  }
  */
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  cout <<"Executation time: "<< duration.count()/1e6 << endl;

  S_3d.setZero();
  for (int i=0; i<10000; i++) {
    calcCoulombIntegralPeriodic_BTranslations
        (NA, Ax, Ay, Az, expA, 1.0,
         NB, Bx, By, Bz, expB, 1.0,
         NC, Ax, Ay, Az, zEta, Z*pow(zEta/M_PI/2., 0.75), //it uses sqrt(norm)
         Lx, Ly, Lz, S_3d, coulomb_14_14_8, true);  
  }
  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  cout <<"Executation time: "<< duration.count()/1e6 << endl;


}




  /*
    calcCoulombIntegralPeriodic(NA, Ax, Ay, Az, 2*expA, 1.0,
    NB, Bx, By, Bz, expB, 1.0,
    Lx, Ly, Lz, S, false);
    cout << S(0,0)<<endl;
    
    
    calcKineticIntegralPeriodic(NA, Ax, Ay, Az, expA, 1.0,
                              NB, Bx, By, Bz, expB, 1.0,
                              Lx, Ly, Lz, S);
  */


  /*
  double zEta = 5.e5, Z = 2.0;
  S.setZero();
  calcCoulombIntegralPeriodic(0, Ax, Ay, Az, expA, 1.0,
                              0, Ax, Ay, Az, expA, 1.0,
                              0, Ax, Ay, Az, zEta, Z*pow(zEta/M_PI/2., 0.75), //it uses sqrt(norm)
                              Lx, Ly, Lz, S, true);
  cout << S(0,0)<<endl;
  */

