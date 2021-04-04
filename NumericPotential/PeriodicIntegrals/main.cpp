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
#include "Integral3c_Boys.h"
#include "IrBoysFn.h"
#include "LatticeSum.h"
#include "timer.h"
#include <boost/format.hpp>

using namespace std;
using namespace std::chrono;
using namespace boost;

cumulTimer realSumTime, kSumTime, ksumTime1, ksumTime2, ksumKsum;
cumulTimer pairRTime, pairKTime, coulombContractTime;
size_t add2;

void testIntegral(Kernel& kernel, vector<int>& shls, vector<double>& Lattice, ct::FMemoryStack2& Mem) ;
void FormIntIJA(double *pIntFai, vector<int>& shls, Kernel &IntKernel, LatticeSum& latsum, ct::FMemoryStack2 &Mem);

void Symmetrize(vector<double>& array, int nbas) {
  for (int i=0; i<nbas; i++)
    for (int j=0; j<i; j++) {
      if (fabs(array[i*nbas+j]) > 1.e-14 && fabs(array[j*nbas+i]) > 1.e-14) continue; 
      array[i*nbas+j] = array[i*nbas+j] + array[j*nbas+i];
      array[j*nbas+i] = array[i*nbas+j];
    }
}

template<typename T>
void readFile(vector<T>& vec, string fname) {
  streampos begin,end;
  ifstream myfile (fname.c_str(), ios::binary);
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
  size_t RequiredMem = 1e10;
  ct::FMemoryStack2
      Mem(RequiredMem);

  vector<int> atm, bas, shls;
  vector<double> env, Lattice;

  readFile(atm    , "atm");
  readFile(bas    , "bas");
  readFile(shls   , "shls");
  readFile(env    , "env");
  readFile(Lattice, "Lattice");

  initPeriodic(&shls[0], &atm[0], atm.size()/6,
               &bas[0], bas.size()/8, &env[0],
               &Lattice[0]);


  
  CoulombKernel ckernel;
  OverlapKernel okernel;
  KineticKernel kkernel;
  //basis.PrintAligned(cout,0);
  cout << "n Basis: "<<basis.getNbas()<<endl;

  {
    size_t nAuxbas = basis.getNbas(shls[5]) - basis.getNbas(shls[4]);
    size_t nbas = basis.getNbas(shls[1]) - basis.getNbas(shls[0]);
    vector<double> Integral3c(nbas*nbas*nAuxbas,0.0);
    LatticeSum latsum(&Lattice[0], 3, 4, Mem, basis, 1., 30., 1.e-10, 1e-10, false, false);
    auto start = high_resolution_clock::now();
    //Int2e3cRR(&Integral3c[0], shls, basis, latsum, Mem);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout <<"Executation time 3c2e--->: "<< duration.count()/1e6 << endl;

    string name = "int2e3cR";
    ofstream file(name.c_str(), ios::binary);
    file.write(reinterpret_cast<char*>(&Integral3c[0]), Integral3c.size()*sizeof(double));
    file.close();
  }
  
  //for (int i=0; i<10; i++) 
  basis.basisnormTodensitynorm(shls[4], shls[5]);
    ThreeCenterIntegrals(shls, basis, Lattice, Mem);
  basis.densitynormTobasisnorm(shls[4], shls[5]);

  //latsum.printLattice();
  /*
  {
    size_t nbas = basis.getNbas();
    cout <<"nbas "<< nbas<<endl;
    vector<double> threeInt(nbas*nbas*nbas,0.0);
    vector<double> twoInt(nbas*nbas, 0.0);
    LatticeSum latsum(&Lattice[0], 10, 10, Mem, basis, 1., 100., 1.e-14, 1e-14);

    BasisShell trans; trans.Xcoord = 0.0; trans.Ycoord = 0.0; trans.Zcoord = 0.0;
    trans.l = 0; trans.nFn = 1; trans.nCo = 1;
    trans.exponents.resize(1, 0.05);
    
    trans.contractions = MatrixXd::Zero(1,1);
    trans.contractions(0,0) = 1.0;

    EvalInt2e2c(&twoInt[0], 1, nbas, &basis.BasisShells[0], &basis.BasisShells[0],
		1.0, true, &ckernel, latsum, Mem);

    //EvalInt2e2c(&twoInt[0], 1, nbas, &trans, &trans,
    //1.0, true, &ckernel, latsum, Mem);

    cout <<"j2c "<< twoInt[0]<<endl;
    
    auto start = high_resolution_clock::now();
    /*
    {
      twoInt[0] = 0.0;
      BasisShell *pbas = &basis.BasisShells[0];

      BasisShell trans; trans.Xcoord = 0.0; trans.Ycoord = 0.0; trans.Zcoord = 0.0;
      trans.l = 0; trans.nFn = 1; trans.nCo = 1;
      trans.exponents.resize(1, 2*pbas->exponents[0]);
      
      trans.contractions = MatrixXd::Zero(1,1);
      trans.contractions(0,0) = pow(pbas->contractions(0,0),2);

      double old = 0.0;
      for (int r=0; r<latsum.Rdist.size(); r++) {
	double factor = exp(- pbas->exponents[0]/2. * latsum.Rdist[r]);
	trans.Xcoord = latsum.Rcoord[3*r+0]/2.;
	trans.Ycoord = latsum.Rcoord[3*r+1]/2.;
	trans.Zcoord = latsum.Rcoord[3*r+2]/2.;
	
	EvalInt2e2c(&twoInt[0], 1, nbas, &trans, &basis.BasisShells[0],
		    factor, true, &ckernel, latsum, Mem);
	if (fabs(twoInt[0] - old)<1.e-11 ) break;
	old = twoInt[0];
      }
      cout <<" FR "<< twoInt[0]<<" contraction "<<pbas->contractions(0,0)<<endl;
    }
    /

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout <<"Executation time overlap--->: "<< duration.count()/1e6 << endl;

    start = high_resolution_clock::now();
    /*
    {

      threeInt[0] = 0.0;
      EvalInt2e3cKKsum(&threeInt[0], &basis.BasisShells[shls[0]], &basis.BasisShells[shls[0]],
		       &basis.BasisShells[shls[0]], 1.0, &ckernel, latsum, Mem);
      //&basis.BasisShells[0], 1.0, &ckernel, latsum, Mem);
      cout <<"Int3c "<< threeInt[0]<<endl;
    }
      //FormIntIJA(&threeInt[0], shls, ckernel, latsum, Mem);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout <<"Executation time overlap--->: "<< duration.count()/1e6 << endl;
    exit(0);
    cout <<"3-center "<< threeInt[0]<<endl;
    ofstream file("nuc_ref", ios::binary);
    file.write(reinterpret_cast<char*>(&threeInt[0]), threeInt.size()*sizeof(double));
    file.close();

    /
  }
  */


  
  cout <<endl;
  cout << "Testing Overlap"<<endl;
  testIntegral(okernel, shls, Lattice, Mem); cout << endl;
  cout << "Testing Kinetic"<<endl;
  testIntegral(kkernel, shls, Lattice, Mem); cout << endl;
  cout << "Testing Coulomb"<<endl;
  
  basis.basisnormTodensitynorm(shls[4], shls[5]);
  testIntegral(ckernel, shls, Lattice, Mem); cout << endl;
  basis.densitynormTobasisnorm(shls[4], shls[5]);
}


void testIntegral(Kernel& kernel, vector<int>&shls, vector<double>& Lattice, ct::FMemoryStack2& Mem) {
  
  int nbas = basis.getNbas(shls[5]) - basis.getNbas(shls[4]);
  vector<double> integrals(nbas*nbas);

  {
    realSumTime = cumulTimer();
    kSumTime = cumulTimer();
    ksumTime1 = cumulTimer();
    ksumTime2 = cumulTimer();

    auto start = high_resolution_clock::now();
    LatticeSum latsum(&Lattice[0], 6, 20, Mem, basis, 5., 8.0, 1.e-10, 1e-11, true, false);

    {
      auto stop = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>(stop - start);
      cout <<"lattice sum "<< duration.count()/1e6<<endl;
    }
    
    int inbas = 0, jnbas = 0, nbas1, nbas2;
    for (int sh1 = shls[4] ; sh1 <shls[5]; sh1++) {
      nbas1 = basis.BasisShells[sh1].nCo * (2 * basis.BasisShells[sh1].l + 1);
      jnbas = 0;
      for (int sh2 = shls[4] ; sh2 <=sh1; sh2++) {
        nbas2 = basis.BasisShells[sh2].nCo * (2 * basis.BasisShells[sh2].l + 1);

        EvalInt2e2c(&integrals[inbas + jnbas * nbas], 1, nbas, &basis.BasisShells[sh1],
                    &basis.BasisShells[sh2], 1.0, false, &kernel, latsum, Mem);
        jnbas += nbas2;
      }
      inbas += nbas1;
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "num add2: "<<add2<<endl;
    cout <<format("Executation time     : %10.5f\n") % (duration.count()/1e6);
    cout <<format("Real space summation : %10.5f\n") % (realSumTime);
    cout <<format("K    space summation : %10.5f\n") % (kSumTime);
    
    //do it again with larger thresholds
    string name = "coul_ref";
    if (kernel.getname() == coulombKernel) name = "coul_ref";
    if (kernel.getname() == overlapKernel) name = "ovlp_ref";
    if (kernel.getname() == kineticKernel) name = "kin_ref";
    Symmetrize(integrals, nbas);
    ofstream file(name.c_str(), ios::binary);
    file.write(reinterpret_cast<char*>(&integrals[0]), integrals.size()*sizeof(double));
    file.close();
    cout << integrals[0]<<"  "<<integrals[1]<<"  "<<integrals[2]<<endl;
  }

  {
    vector<double> integrals(nbas*nbas, 0.0);
    auto start = high_resolution_clock::now();
    LatticeSum latsum(&Lattice[0], 6, 20, Mem, basis, 10.0, 10.0, 1e-16, 1.e-16, true, false);
    //if (kernel.getname() == coulombKernel) latsum.makeKsum(basis);

    int inbas = 0, jnbas = 0, nbas1, nbas2;
    for (int sh1 = shls[4] ; sh1 <shls[5]; sh1++) {
      nbas1 = basis.BasisShells[sh1].nCo * (2 * basis.BasisShells[sh1].l + 1);
      jnbas = 0;
      for (int sh2 = shls[4] ; sh2 <=sh1; sh2++) {
        nbas2 = basis.BasisShells[sh2].nCo * (2 * basis.BasisShells[sh2].l + 1);
        EvalInt2e2c(&integrals[inbas + jnbas * nbas], 1, nbas, &basis.BasisShells[sh1],
                    &basis.BasisShells[sh2], 1.0, false, &kernel, latsum, Mem);
        jnbas += nbas2;
      }
      inbas += nbas1;
    }
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    //cout <<"Executation time: "<< duration.count()/1e6 << endl;
    
    Symmetrize(integrals, nbas);
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
    cout << format("Total error: %10.4g\n") % (sqrt(error));
    cout << format("Max error  : %10.4g\n") % (sqrt(maxError));
    //cout <<maxInd<<"  "<< integrals[maxInd]<<"  "<<intRef[maxInd]<<endl;
    //cout << maxInd/nbas<<"  "<<maxInd%nbas<<endl;
  }  

}

