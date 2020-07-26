#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

#include "interface.h"
#include "CxMemoryStack.h"
#include "Integral2c_Boys.h"
#include "IrBoysFn.h"

using namespace std;
using namespace std::chrono;


template<typename T>
void readFile(vector<T>& vec, char* fname) {
  streampos begin,end;
  ifstream myfile (fname, ios::binary);
  begin = myfile.tellg();
  myfile.seekg (0, ios::end);
  end = myfile.tellg();

  vec.resize(end/sizeof(T));
  myfile.seekg (0, ios::beg);
  myfile.read (reinterpret_cast<char*>(&vec[0]), end);
  myfile.close();

}


int main(int argc, char** argv) {
  cout.precision(12);
  size_t RequiredMem = 1e9;
  ct::FMemoryStack2
      Mem(RequiredMem);

  vector<int> atm, bas, shls, ao_loc;
  vector<double> env, Lattice;

  readFile(atm, "atm");
  readFile(bas, "bas");
  readFile(shls, "shls");
  readFile(ao_loc, "aoloc");
  readFile(env, "env");
  readFile(Lattice, "Lattice");


  //shls[0] = shls[4]; shls[1] = shls[5]; shls[2] = shls[4]; shls[3] = shls[5];
  int n1 = ao_loc[shls[1]] - ao_loc[shls[0]]; 
  int n2 = ao_loc[shls[3]] - ao_loc[shls[2]]; 
  int n3 = ao_loc[shls[5]] - ao_loc[shls[4]];
  cout << n1<<"  "<<n2<<"  "<<n3<<endl;
  
  vector<double> integrals(n1*n2, 0.0);
  auto start = high_resolution_clock::now();
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);


  initPeriodic(&shls[0], &ao_loc[0], &atm[0], atm.size()/6,
               &bas[0], bas.size()/8/2, &env[0],
               &Lattice[0]);
  
  //basis.PrintAligned(cout, 8);
  cout << Lattice[0]<<"  "<<Lattice[1]<<"  "<<Lattice[2]<<endl;
  LatticeSum latsum(&Lattice[0]);
  cout << latsum.KLattice[0]<<"  "<<latsum.KLattice[3]<<"  "<<latsum.KLattice[4]<<endl;

  int sh1 = 0, sh2 = 0;
  //cout << basis.BasisShells[sh1].exponents[0]<<endl;
  //basis.BasisShells[sh1].exponents[0] = 20.; basis.BasisShells[sh2].exponents[0] = 20.;
  int nbas1 = basis.BasisShells[sh1].nCo * (2 * basis.BasisShells[sh1].l + 1);
  int nbas2 = basis.BasisShells[sh2].nCo * (2 * basis.BasisShells[sh2].l + 1);

  //CoulombKernel kernel;
  OverlapKernel kernel;
  //KineticKernel kernel;
  
  EvalInt2e2c(&integrals[0], 1, nbas1, &basis.BasisShells[sh1],
              &basis.BasisShells[sh2], 1.0, false, &kernel, latsum, Mem);

  for (int i=0; i<nbas1; i++) {
    for (int j=0; j<nbas2; j++)
      printf("%13.8f  ", integrals[i + j * nbas1]);
    //cout << integrals[i + j * nbas1]<<"  ";
    //cout << boost::format("%9.4e" %(integrals[i + j * nbas1]))<<"  ";
    cout << endl;
  }
  //exit(0);
  start = high_resolution_clock::now();
  int nbas = 0;
  for (int i = 0 ; i <basis.BasisShells.size(); i++) {
    nbas += basis.BasisShells[i].nCo * (2 * basis.BasisShells[i].l + 1);
  }

  int inbas = 0, jnbas = 0;
  for (int i = 0 ; i <basis.BasisShells.size(); i++) {
    sh1 = i;
    nbas1 = basis.BasisShells[sh1].nCo * (2 * basis.BasisShells[sh1].l + 1);
    jnbas = 0;
    for (int j = 0 ; j <=i; j++) {
      sh2 = j;
      nbas2 = basis.BasisShells[sh2].nCo * (2 * basis.BasisShells[sh2].l + 1);
      EvalInt2e2c(&integrals[inbas + jnbas * nbas], 1, nbas, &basis.BasisShells[sh1],
                  &basis.BasisShells[sh2], 1.0, false, &kernel, latsum, Mem);
      jnbas += nbas2;
    }
    inbas += nbas1;
  }

  for (int i=0; i<nbas; i++)
    for (int j=i+1; j<nbas; j++) {
      integrals[i + j*nbas] = integrals[j +i*nbas];
    }
  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  cout <<"Executation time: "<< duration.count()/1e6 << endl;

  vector<double> ovlp;
  readFile(ovlp, "kin.npy");
  exit(0);
  /*
  */
  //TAKE 1 with tmin
  {
    double omega = 0.2, tmin = 0.2;
    double rval = 0.0, umin = tmin/sqrt(tmin*tmin+10.), eta = omega/sqrt(omega*omega + 10.), rvalloc1 = 0.0, rvalloc2 = 0.0;
    cout << omega<<" omega  eta "<<eta<<endl;
    int Nterm = 100;
    for (int nx=-Nterm; nx<=Nterm; nx++) {
      for (int ny=-Nterm; ny<=Nterm; ny++)
        for (int nz=-Nterm; nz<=Nterm; nz++)
        {
          rvalloc1 = 0.0; rvalloc2 = 0.0;
          ir::IrBoysFn(&rvalloc1, 10. * (nx*nx + ny*ny + nz*nz), 0, 1.0);
          ir::IrBoysFn(&rvalloc2, eta * eta * 10. * (nx*nx + ny*ny + nz*nz), 0, 1.0);
          
          //cout <<eta<<"  "<< rvalloc1<<"  "<<eta * rvalloc2 <<"  "<<rvalloc1 - eta * rvalloc2<<endl;
          rval += pow(M_PI, 2.5) * 2. / pow(400., 1.5) * sqrt(10.) * (rvalloc1 - eta * rvalloc2);
        }
      //cout <<nx<<" "<< rval <<endl;
    }
    
    int Kterm = 10; double kval = 0.0;
    for (int nx=-Kterm; nx<=Kterm; nx++) {
      for (int ny=-Kterm; ny<=Kterm; ny++)
        for (int nz=-Kterm; nz<=Kterm; nz++)
        {
          if (nx == 0 && ny == 0 && nz == 0) continue;
          double G2 = M_PI*M_PI*(nx*nx + ny*ny + nz*nz);
          kval += M_PI*M_PI*M_PI*M_PI/pow(400, 1.5) * (exp(-G2/10./eta/eta) - exp(-G2/10./umin/umin))/G2;
        }
      //cout <<nx<<"   "<< kval <<endl;
    }
    
    double bkgrnd = pow(M_PI, 4)/pow(400, 1.5) *(1./tmin/tmin - 1./10/umin/umin + 1./10/eta/eta);
    cout << pow(M_PI, 4)/pow(400, 1.5) /10/omega/omega<<endl;
    cout <<kval<<" k  "<< rval <<" r  "<<bkgrnd<<" b  "<<rval + kval - bkgrnd<<endl;
  }


  //TAKE 2 WITHOUT Tmin
  {
    double omega = 3.873;
    double a = 20, b = 20; double p = (a*b)/(a+b);
    double rval = 0.0, eta = omega/sqrt(omega*omega + p), rvalloc1 = 0.0, rvalloc2 = 0.0;
    cout << omega<<" omega  eta "<<eta<<endl;
    int Nterm = 1;
    for (int nx=-Nterm; nx<=Nterm; nx++) {
      for (int ny=-Nterm; ny<=Nterm; ny++)
        for (int nz=-Nterm; nz<=Nterm; nz++)
        {
          rvalloc1 = 0.0; rvalloc2 = 0.0;
          ir::IrBoysFn(&rvalloc1, p * (nx*nx + ny*ny + nz*nz), 0, 1.0);
          ir::IrBoysFn(&rvalloc2, eta * eta * p * (nx*nx + ny*ny + nz*nz), 0, 1.0);
          
          rval += pow(M_PI, 2.5) * 2. / pow(a*b, 1.5) * sqrt(p) * (rvalloc1 - eta * rvalloc2);
        }
      //cout <<nx<<" "<< rval <<endl;
    }
    
    int Kterm = 4; double kval = 0.0;
    for (int nx=-Kterm; nx<=Kterm; nx++) {
      for (int ny=-Kterm; ny<=Kterm; ny++)
        for (int nz=-Kterm; nz<=Kterm; nz++)
        {
          if (nx == 0 && ny == 0 && nz == 0) continue;
          double G2 = M_PI*M_PI*(nx*nx + ny*ny + nz*nz);
          kval += M_PI*M_PI*M_PI*M_PI/pow(a*b, 1.5) * (exp(-G2/p/eta/eta))/G2;
        }
      //cout <<nx<<"   "<< kval <<endl;
    }
    
    double bkgrnd = pow(M_PI, 4)/pow(a*b, 1.5) / omega/omega;
    //pow(M_PI, 4)/pow(400, 1.5) *(1./tmin/tmin - 1./10/umin/umin + 1./10/eta/eta);
    //cout << 
    cout <<kval<<" k  "<< rval <<" r  "<<bkgrnd<<" b  "<<rval + kval - bkgrnd<<endl;

  }
    exit(0);
  
  for (int i=0; i<nbas1; i++) {
    for (int j=0; j<nbas2; j++)
      cout << integrals[i + j * nbas1]<<"  ";
    cout << endl;
  }
  exit(0);  
  double nbasis = 0.;
  for (int i=0; i<basis.BasisShells.size(); i++)
    nbasis += basis.BasisShells[i].nCo * (2 * basis.BasisShells[i].l +1);
  cout << nbasis <<endl;

  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  cout <<"Executation time: "<< duration.count()/1e6 << endl;

}
