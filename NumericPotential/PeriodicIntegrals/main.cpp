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
#include "LatticeSum.h"
#include "timer.h"

using namespace std;
using namespace std::chrono;

cumulTimer realSumTime, kSumTime;

double testIntegral(Kernel& kernel, int nbas, vector<double>& Lattice, ct::FMemoryStack2& Mem) ;

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


  int n1 = ao_loc[shls[1]] - ao_loc[shls[0]]; 
  int n2 = ao_loc[shls[3]] - ao_loc[shls[2]]; 
  
  vector<double> integrals(n1*n2, 0.0);


  initPeriodic(&shls[0], &ao_loc[0], &atm[0], atm.size()/6,
               &bas[0], bas.size()/8/2, &env[0],
               &Lattice[0]);  
  //basis.PrintAligned(cout, 0);
  //exit(0);
  LatticeSum latsum(&Lattice[0], 4,13,100.,1.e-14);
  latsum.printLattice();
  cout << "n Basis: "<<basis.getNbas()<<endl;


  int sh1 = 0, sh2 = 0;
  //cout << basis.BasisShells[sh1].exponents[0]<<endl;
  //basis.BasisShells[sh1].exponents[0] = 20.; basis.BasisShells[sh2].exponents[0] = 20.;
  int nbas1 = basis.BasisShells[sh1].nCo * (2 * basis.BasisShells[sh1].l + 1);
  int nbas2 = basis.BasisShells[sh2].nCo * (2 * basis.BasisShells[sh2].l + 1);

  CoulombKernel ckernel;
  OverlapKernel okernel;
  KineticKernel kkernel;


  EvalInt2e2c(&integrals[0], 1, nbas1, &basis.BasisShells[sh1],
              &basis.BasisShells[sh2], 1.0, false, &ckernel, latsum, Mem);

  /*
  for (int i=0; i<nbas1; i++) {
    for (int j=0; j<nbas2; j++)
      printf("%13.8f  ", integrals[i + j * nbas1]);
    cout << endl;
  }
  */
  {
    vector<double> integralsNew(n1*n2, 0.0);
    LatticeSum latsum(&Lattice[0], 20,20,200.,1.e-12);
    EvalInt2e2c(&integralsNew[0], 1, nbas1, &basis.BasisShells[sh1],
                &basis.BasisShells[sh2], 1.0, false, &ckernel, latsum, Mem);

    double error = 0.0, maxError = 0.0; int maxInd = 0;
    for (int i=0; i<nbas1*nbas2; i++) {
      error += pow(integralsNew[i] - integrals[i], 2);
      if (maxError < pow(integrals[i] - integralsNew[i], 2)) {
        maxError = pow(integrals[i] - integralsNew[i], 2);
        maxInd = i;
      }
    }
    cout << "Total error: "<<sqrt(error)<<endl<<"Max error:  "<<sqrt(maxError)<<endl;
    cout <<maxInd<<"  "<< integrals[maxInd]<<"  "<<integralsNew[maxInd]<<endl;
    cout << maxInd/nbas1<<"  "<<maxInd%nbas1<<endl;
  }
  cout << endl<<endl;
  //exit(0);

  //testIntegral(okernel, basis.getNbas(), Lattice, Mem); cout << endl;
  //testIntegral(kkernel, basis.getNbas(), Lattice, Mem); cout << endl;
  testIntegral(ckernel, basis.getNbas(), Lattice, Mem); cout << endl;

}


double testIntegral(Kernel& kernel, int nbas, vector<double>& Lattice, ct::FMemoryStack2& Mem) {
  vector<double> integrals(nbas*nbas, 0.0);

  {
    LatticeSum latsum(&Lattice[0], 2, 20, 100., 8.0, 1.e-12);
    auto start = high_resolution_clock::now();
    int inbas = 0, jnbas = 0, nbas1, nbas2;
    for (int sh1 = 0 ; sh1 <basis.BasisShells.size(); sh1++) {
      nbas1 = basis.BasisShells[sh1].nCo * (2 * basis.BasisShells[sh1].l + 1);
      jnbas = 0;
      for (int sh2 = 0 ; sh2 <=sh1; sh2++) {
        nbas2 = basis.BasisShells[sh2].nCo * (2 * basis.BasisShells[sh2].l + 1);
        EvalInt2e2c(&integrals[inbas + jnbas * nbas], 1, nbas, &basis.BasisShells[sh1],
                    &basis.BasisShells[sh2], 1.0, false, &kernel, latsum, Mem);
        jnbas += nbas2;
      }
      inbas += nbas1;
    }
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout <<"Executation time overlap--->: "<< duration.count()/1e6 << endl;
    cout <<"spacetime: "<< realSumTime<<endl<<"kspacetime: "<<kSumTime<<endl;
    
    //do it again with larger thresholds
    string name = "coul_ref";
    if (kernel.getname() == coulombKernel) name = "coul_ref";
    if (kernel.getname() == overlapKernel) name = "ovlp_ref";
    if (kernel.getname() == kineticKernel) name = "kin_ref";
    ofstream file(name.c_str(), ios::binary);
    file.write(reinterpret_cast<char*>(&integrals[0]), integrals.size()*sizeof(double));
    file.close();
  }

  {
    auto start = high_resolution_clock::now();
    LatticeSum latsum(&Lattice[0], 5, 15, 200.0, 1.e-16);
    int inbas = 0, jnbas = 0, nbas1, nbas2;
    for (int sh1 = 0 ; sh1 <basis.BasisShells.size(); sh1++) {
      nbas1 = basis.BasisShells[sh1].nCo * (2 * basis.BasisShells[sh1].l + 1);
      jnbas = 0;
      for (int sh2 = 0 ; sh2 <=sh1; sh2++) {
        nbas2 = basis.BasisShells[sh2].nCo * (2 * basis.BasisShells[sh2].l + 1);
        EvalInt2e2c(&integrals[inbas + jnbas * nbas], 1, nbas, &basis.BasisShells[sh1],
                    &basis.BasisShells[sh2], 1.0, false, &kernel, latsum, Mem);
        jnbas += nbas2;
      }
      inbas += nbas1;
    }
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout <<"Executation time: "<< duration.count()/1e6 << endl;
    
    vector<double> intRef;
    if (kernel.getname() == coulombKernel) readFile(intRef, "coul_ref"); 
    if (kernel.getname() == overlapKernel) readFile(intRef, "ovlp_ref"); 
    if (kernel.getname() == kineticKernel) readFile(intRef, "kin_ref" ); 
    
    double error = 0.0, maxError = 0.0; int maxInd = 0;
    for (int i=0; i<integrals.size(); i++) {
      error += pow(integrals[i] - intRef[i], 2);
      if (maxError < pow(integrals[i] - intRef[i], 2)) {
        maxError = pow(integrals[i] - intRef[i], 2);
        maxInd = i;
      }
    }
    cout << "Total error: "<<sqrt(error)<<endl<<"Max error:  "<<sqrt(maxError)<<endl;
    cout <<maxInd<<"  "<< integrals[maxInd]<<"  "<<intRef[maxInd]<<endl;
    cout << maxInd/nbas<<"  "<<maxInd%nbas<<endl;
  }  

}
