#include <fstream>
#include "global.h"
#include "Determinants.h"
#include "integral.h"
#include "Hmult.h"
#include "CIPSIbasics.h"
#include "Davidson.h"
#include <Eigen/Dense>

using namespace Eigen;
int Determinant::norbs = 1; //spin orbitals

void readInput(string input, std::vector<int>& occupied, double& eps1, double& eps2, double& tol) {
  ifstream dump(input.c_str());
  int nocc; dump >>nocc;
  occupied.resize(nocc);
  for (int i=0; i<nocc; i++)
    dump >> occupied[i];
  dump >> eps1 >>eps2>>tol;
}

int main(int argc, char* argv[]) {
  std::cout.precision(15);
  twoInt I2; oneInt I1; int nelec; int norbs; double coreE;
  readIntegrals("FCIDUMP", I2, I1, nelec, norbs, coreE);
  norbs *=2;
  Determinant::norbs = norbs; //spin orbitals

  std::vector<int> HFoccupied; double epsilon1, epsilon2, tol;
  readInput("input.dat", HFoccupied, epsilon1, epsilon2, tol);
  //make HF determinant
  Determinant d;
  std::cout << norbs<<std::endl;
  for (int i=0; i<HFoccupied.size(); i++) {
    std::cout <<i<<"  "<< HFoccupied[i]<<std::endl;
    d.setocc(HFoccupied[i], true);
  }


  char detchar[norbs]; d.getRepArray(detchar);
  double EHF= Energy(detchar,norbs,I1,I2,coreE);
  std::cout << "HF = "<<EHF<<std::endl;

  char closed[nelec], open[norbs-nelec];
  int o = d.getOpenClosed(open, closed); int v=norbs-o;
  std::vector<Determinant> Dets(1,d), prevDets(1,d);
  MatrixXd ci(1,1); ci(0,0) = 1.0;

  int niter = 5;
  double E0 = EHF;
  std::vector<char> detChar(norbs); d.getRepArray(&detChar[0]);
  MatrixXd diagOld(1,1); diagOld(0,0) = EHF;
  int prevSize = 0;

  //omp_set_num_threads(20);
  int num_thrds = omp_get_max_threads();
  std::cout << "max thrds "<<num_thrds<<std::endl;
  int iter = 0;
  //do the variational bit
  while(true){

    //#pragma omp parallel for 
    for (int i=0; i<prevDets.size(); i++) {
    //for (int i=prevSize; i<prevDets.size(); i++) {
      std::vector<Determinant> newDets;
      getDeterminants(prevDets[i], abs(epsilon1/ci(i,0)), I1, I2, coreE, E0, newDets);

      //#pragma omp critical 
      {
      for (int k=0; k<newDets.size(); k++) {
	if (find(Dets.begin(), Dets.end(), newDets[k]) == Dets.end())
	  Dets.push_back(newDets[k]);
      }
      }
    }

    //now diagonalize the hamiltonian
    detChar.resize(norbs* Dets.size()); 
    MatrixXd X0(Dets.size(), 1); X0 *= 0.0; X0.block(0,0,ci.rows(),1) = 1.*ci;
    MatrixXd diag(Dets.size(), 1); diag.block(0,0,ci.rows(),1)= 1.*diagOld;

    for (int k=prevDets.size(); k<Dets.size(); k++) {
      Dets[k].getRepArray(&detChar[norbs*k]);
      diag(k,0) = Energy(&detChar[norbs*k], norbs, I1, I2, coreE);
    }

    Hmult H(&detChar[0], norbs, I1, I2, coreE);
    E0 = davidson(H, X0, diag, 5, tol, true);
    std::cout <<iter<<"  "<<Dets.size()<<"  "<< E0 <<std::endl;iter++;
    ci.resize(Dets.size(),1); ci = 1.0*X0;
    diagOld.resize(Dets.size(),1); diagOld = 1.0*diag;

    if (1.*(Dets.size()-prevDets.size())/prevDets.size() < 0.01)  {
      break;
    }
    prevSize = prevDets.size();

    prevDets.clear();prevDets=Dets;
  }

  //now do the perturbative bit
  double energyEN = 0.0;
  std::vector<Determinant> Psi1;
  char psiArray[norbs];

  for (int i=0; i<Dets.size(); i++) {
    std::vector<Determinant> newDets;
    getDeterminants(Dets[i], abs(epsilon2/ci(i,0)), I1, I2, coreE, E0, newDets);
    
    for (int j=0; j<newDets.size(); j++) {
      if ( (find(Dets.begin(), Dets.end(), newDets[j]) == Dets.end()) &&
	   (find(Psi1.begin(), Psi1.end(), newDets[j]) == Psi1.end()) ) {
	Psi1.push_back(newDets[j]);
	
	double integral = 0.0;
	newDets[j].getRepArray(psiArray);
	//#pragma omp parallel for reduction(+:integral)
	for (int k=0; k<Dets.size(); k++) {
	  integral += Hij(&detChar[k*norbs], psiArray, norbs, I1, I2, coreE)*ci(k,0); 
	}
	energyEN += integral*integral/(Energy(psiArray, norbs, I1, I2, coreE)-E0);
      }
    }     

    if (i%100 == 0)
      cout << "done "<<i <<"  "<<energyEN<<endl;
  }
  cout <<energyEN<<"  "<< -energyEN+E0<<endl;
  
  return 0;
}
