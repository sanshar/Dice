#include "global.h"
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include "Determinants.h"
#include "input.h"
#include "integral.h"
#include "Hmult.h"
#include "HCIbasics.h"
#include "Davidson.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <set>
#include <list>
#include <tuple>
#include "boost/format.hpp"
#include "new_anglib.h"
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "communicate.h"

using namespace Eigen;
using namespace boost;
int HalfDet::norbs = 1; //spin orbitals
int Determinant::norbs = 1; //spin orbitals
int Determinant::EffDetLen = 1;
Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> Determinant::LexicalOrder ;
//get the current time
double getTime() {
  struct timeval start;
  gettimeofday(&start, NULL);
  return start.tv_sec + 1.e-6*start.tv_usec;
}
double startofCalc = getTime();

boost::interprocess::shared_memory_object int2Segment;
boost::interprocess::mapped_region regionInt2;
boost::interprocess::shared_memory_object int2SHMSegment;
boost::interprocess::mapped_region regionInt2SHM;


void readInput(string input, vector<std::vector<int> >& occupied, schedule& schd);

void calculateMatrixElements(int spin1, int spin2, int Sz, int rowIndex1, int rowIndex2,
			     const MatrixXx& c1, const MatrixXx& c2, vector<vector<int> >& connections,
			     vector<vector<CItype> >& Helements, MatrixXx& Hsubspace, vector<Determinant>& Dets, int norbs, vector<Determinant>::iterator& beginS0, vector<Determinant>::iterator& beginSp, vector<Determinant>::iterator& beginSm);


int main(int argc, char* argv[]) {

#ifndef SERIAL
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
#endif

  //Read the input file
  std::vector<std::vector<int> > HFoccupied;
  schedule schd;
  if (mpigetrank() == 0) readInput("input.dat", HFoccupied, schd);
  int nelec = HFoccupied[0].size();

#ifndef SERIAL
  mpi::broadcast(world, HFoccupied, 0);
  mpi::broadcast(world, schd, 0);
#endif


  //set the random seed
  startofCalc=getTime();
  srand(schd.randomSeed+world.rank());


  int num_thrds;  

  int norbs = readNorbs("FCIDUMP");
  norbs *=2; //spin orbitals
  Determinant::norbs = norbs; //spin orbitals
  HalfDet::norbs = norbs; //spin orbitals
  Determinant::EffDetLen = norbs/64+1;
  Determinant::initLexicalOrder(nelec);
  cout << "NORBS "<<norbs<<endl;
  oneInt I1; I1.store.resize(norbs*norbs, 0.0); I1.norbs = norbs;
  readSOCIntegrals(I1, norbs, "SOC");

  
  //have the dets, ci coefficient and diagnoal on all processors
  vector<MatrixXd> ci; 
  std::vector<Determinant> Dets, SortedDets;
  std::vector<double> Energy;
  MatrixXd diag;
  for (int i=0; i<schd.prefix.size(); i++)
  {
    char file [5000];
    sprintf (file, "%s/%d-variational.bkp" , schd.prefix[i].c_str(), world.rank() );
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);

    vector<MatrixXd> Localci; 
    std::vector<Determinant> LocalDets, LocalSortedDets;
    std::vector<double> LocalEnergy;
    MatrixXd Localdiag;

    int iter;
    load >> iter >> LocalDets >> LocalSortedDets ;
    LocalSortedDets.size();
    int diaglen;
    load >> diaglen;
    Localdiag.resize(diaglen,1);
    for (int j=0; j<Localdiag.rows(); j++)
      load >> Localdiag(j,0);
    load >> Localci;
    load >> LocalEnergy;

    //We assume that the states from different prefixes are of different
    //symmetry
    if (i == 0) {
      ci = Localci;
      diag = Localdiag;
    }
    else {
      std::vector<MatrixXd> bkpci = ci;
      ci.clear();
      ci.resize(bkpci.size()+Localci.size(), MatrixXd::Zero(Localci[0].rows()+bkpci[0].rows(),1));
      for (int j=0; j<bkpci.size(); j++)
	ci[j].block(0,0,bkpci[0].rows(),1) = 1.*bkpci[j];
      for (int j=0; j<Localci.size(); j++) {
	ci[j+bkpci.size()].block(bkpci[0].rows(),0,Localci[0].rows(),1) = 1.*Localci[j];
      }
    }
    Dets.insert(Dets.end(), LocalDets.begin(), LocalDets.end());
    Energy.insert(Energy.end(), LocalEnergy.begin(), LocalEnergy.end());
  }
  mpi::broadcast(world, ci, 0);
  mpi::broadcast(world, Dets, 0);
  mpi::broadcast(world, diag, 0);
  mpi::broadcast(world, Energy, 0);


  //Locate all determinants that are S+ |dets>  and S- |dets>
  //and add them to determinants. Rougly increases the space 3 times.

  //vector<Determinant> VecSplusDets; 
  vector<vector<size_t> > SplusConnections; 
  {
    //Now find out the Ms and S^2 
    std::map<Determinant, vector<size_t> > SplusDets;
    std::map<Determinant, vector<size_t> > SminusDets;
    
    for (int i=0; i<Dets.size(); i++) { //loop over all determinants
      for (int j=0; j<norbs/2 ; j++) {  //loop over spatial orbitals
	if (Dets[i].getocc(2*j+1) && 
	    !Dets[i].getocc(2*j  )   ) {//if beta-filled and alpha-empty
	  
	  Determinant d = Dets[i];
	  d.setocc(2*j+1, false); //empty the beta
	  d.setocc(2*j  , true);  //fill up the alpha
	  
	  SplusDets[d].push_back(i);
	}//if
      }// j (orbitals)
    }//i dets
    
    
    SplusConnections.reserve(SplusDets.size());
    //add the Splus determinants to Dets
    for (auto it = SplusDets.begin(); it != SplusDets.end(); it++) {
      Dets.push_back(it->first);
      SplusConnections.push_back(it->second);
      it->second.clear();
    }


    for (int i=0; i<ci[0].rows(); i++) { //loop over all determinants
      for (int j=0; j<norbs/2 ; j++) {  //loop over spatial orbitals
	if (Dets[i].getocc(2*j) && 
	    !Dets[i].getocc(2*j+1  )   ) {//if beta-filled and alpha-empty
	  
	  Determinant d = Dets[i];
	  d.setocc(2*j, false); //empty the beta
	  d.setocc(2*j+1, true);  //fill up the alpha
	  
	  SminusDets[d].push_back(i);
	}//if
      }// j (orbitals)
    }//i dets
    
    //add S- dets to the Determinants
    for (auto it = SminusDets.begin(); it != SminusDets.end(); it++) {
      Dets.push_back(it->first);
    }

  }


  //beings0<---->beginSp are original dets
  //beginSp<---->beginSm are the S+ dets
  //beginSm<---->Dets.end() are the S- dets
  auto beginS0 = Dets.begin();
  auto beginSp = Dets.begin()+ci[0].rows();
  auto beginSm = Dets.begin()+ci[0].rows()+SplusConnections.size();


  int Sz = Dets[0].Nalpha()-Dets[0].Nbeta();
  //Now calculate S^2 for each state
  std::vector<double> SpinSquare(ci.size(), 0.0);

  //Sz(Sz+1)
  for (int j=0; j<ci.size(); j++) {
    for (int i=0; i<ci[0].rows(); i++) {
      int M = Dets[i].Nalpha() - Dets[i].Nbeta();
      if (M != Sz) {
	cout << "Sz is not the same for all determinants."<<endl;
	exit(0);
      }
      SpinSquare[j] += ci[j](i,0)*ci[j](i,0)*M*0.5*(M*0.5+1);
    }
  }

  int hsubspaceSize = 0;
  //S+S-
  for (int detIndex = ci[0].rows(); detIndex<ci[0].rows()+SplusConnections.size(); detIndex++) {
    std::vector<size_t>& connections = SplusConnections[detIndex-ci[0].rows()];
    for (int k=0; k<ci.size(); k++) {

      double afterSplus = 0.0; 
      //sum up all the contributions toward determinant it->first
      //\sum_j S+c_j|Dj>  -> sum_j c_j |it->first>
      for (int i=0; i<connections.size(); i++) {
	int I = connections[i];
	afterSplus += ci[k](I,0);
      }

      for (int j=0; j<connections.size(); j++) {
	int J = connections[j];
	SpinSquare[k] += ci[k](J,0)*afterSplus;
      }

    }
  }

  //No we have S^2 we want 2S from it
  vector<int> Spin(ci.size(), 0);
  vector<int> rowIndex(ci.size(), 0);
  //calculate S from <S^2> = S(S+1)
  for (int j=0; j<ci.size(); j++) {
    int s = static_cast<int>( (-1.0 + pow(1+4*SpinSquare[j], 0.5)) + 0.5); //2S
    Spin[j] = s;
    hsubspaceSize += s+1; //2s+1
    if (j != 0)
      rowIndex[j] = rowIndex[j-1]+Spin[j-1]+1;
  }

  for (int j=0; j<ci.size(); j++)     
    cout <<"State "<< j<<"  Spin:"<<Spin[j]<<"  S^2:"<<SpinSquare[j]<<"  "<<rowIndex[j]<<"  "<<hsubspaceSize<<endl;

  MatrixXx Hsubspace = MatrixXx::Zero(hsubspaceSize, hsubspaceSize);

  for (int i=0; i<Spin.size(); i++)
    for (int row=rowIndex[i]; row < rowIndex[i]+Spin[i]+1; row++)
      Hsubspace(row, row) = (Energy[i]-Energy[0])*219470.;


  //update the connections
  //this will just update the connections and Helements with the SOC integrals
  std::vector<std::vector<int> > connections; connections.resize(Dets.size());
  std::vector<std::vector<CItype> > Helements;Helements.resize(Dets.size());
  HCIbasics::updateSOCconnections(Dets, 0, connections, Helements, norbs, I1, nelec); 

  // calculate the matrix elements <S1|SOC|S2>
  for (int i=0; i<ci.size(); i++) {
    for (int j=i+1; j<ci.size(); j++) {      
      int s1 = Spin[i], s2=Spin[j];
      if (s2>=s1)
	calculateMatrixElements(s1, s2, Sz, rowIndex[i], rowIndex[j], ci[i], ci[j], connections, Helements, Hsubspace, Dets, norbs, beginS0, beginSp, beginSm);
      else
	calculateMatrixElements(s2, s1, Sz, rowIndex[j], rowIndex[i], ci[j], ci[i], connections, Helements, Hsubspace, Dets, norbs, beginS0, beginSp, beginSm);
    }
  }

  cout << Hsubspace<<endl;
  SelfAdjointEigenSolver<MatrixXx> eigensolver(Hsubspace);
  if (eigensolver.info() != Success) abort();
  for (int i=0; i<Hsubspace.rows(); i++)
    cout << "Root: "<<i<<" -> Energy: "<<eigensolver.eigenvalues()[i]<<endl;
  return 0;
}


void getSplus(const MatrixXx& c2, MatrixXx& c2splus, vector<Determinant>& Dets, vector<Determinant>::iterator& beginS0, vector<Determinant>::iterator& beginSp, vector<Determinant>::iterator& beginSm, int norbs) 
{
  for (int i=0; i<c2.rows(); i++) { //loop over all determinants
    for (int j=0; j<norbs/2 ; j++) {  //loop over spatial orbitals
      if (Dets[i].getocc(2*j+1) && 
	  !Dets[i].getocc(2*j  )   ) {//if beta-filled and alpha-empty
	
	Determinant d = Dets[i];
	d.setocc(2*j+1, false); //empty the beta
	d.setocc(2*j  , true);  //fill up the alpha
	
	auto location = lower_bound(beginSp, beginSm, d);
	c2splus(location-Dets.begin(), 0) += c2(i,0);
      }//if
    }// j (orbitals)
  }//i dets
}

void getSminus(const MatrixXx& c2, MatrixXx& c2sminus, vector<Determinant>& Dets, vector<Determinant>::iterator& beginS0, vector<Determinant>::iterator& beginSp, vector<Determinant>::iterator& beginSm, int norbs) 
{
  for (int i=0; i<c2.rows(); i++) { //loop over all determinants
    for (int j=0; j<norbs/2 ; j++) {  //loop over spatial orbitals
      if (Dets[i].getocc(2*j) && 
	  !Dets[i].getocc(2*j +1 )   ) {//if beta-filled and alpha-empty
	
	Determinant d = Dets[i];
	d.setocc(2*j+1, true); //fill up the beta
	d.setocc(2*j  , false);  //empty the alpha
	
	auto location = lower_bound(beginSm, Dets.end(), d);
	c2sminus(location-Dets.begin(), 0) += c2(i,0);
      }//if
    }// j (orbitals)
  }//i dets
}


void calculateMatrixElements(int spin1, int spin2, int Sz, int rowIndex1, int rowIndex2,
			     const MatrixXx& c1, const MatrixXx& c2, vector<vector<int> >& connections,
			     vector<vector<CItype> >& Helements, MatrixXx& Hsubspace, 
			     vector<Determinant>& Dets, int norbs, 
			     vector<Determinant>::iterator& beginS0, 
			     vector<Determinant>::iterator& beginSp, 
			     vector<Determinant>::iterator& beginSm) {
  
  Hmult2 H(connections, Helements);

  int orbM = 0;
  vector<CItype> RME (3,0.0);
  int Sz2list [3] = {Sz, Sz+2, Sz-2};
  int Sz1list [3] = {Sz, Sz+2, Sz-2};
  int Jop = 2;
  for (int Mop = -2; Mop<=2; Mop+=2) {

    for (int ij=0; ij<9; ij++) { //Sz2 Sz1
      int i=ij/3, j = ij%3;

      int Sz2 = Sz2list[i];
      int Sz1 = Sz1list[j];
      if (Sz2 > spin2 || Sz2 <-spin2) continue;
      if (Sz1 > spin1 || Sz1 <-spin1) continue;
      double cg = clebsch(spin2, Sz2, Jop, Mop, spin1, Sz1) ;

      if (abs(cg) < 1.e-8) continue;

      MatrixXx c2extended = MatrixXx::Zero(Dets.size(), 1);
      MatrixXx c1extended = MatrixXx::Zero(Dets.size(), 1);
      MatrixXx Hc2 = MatrixXx::Zero(Dets.size(), 1);
      if (i == 0) {
	c2extended.block(0,0,c1.rows(),1) = 1.*c2;
      }
      else if (i==1) {
	getSplus(c2, c2extended, Dets, beginS0, beginSp, beginSm, norbs);
      }
      else if (i == 2) {
	getSminus(c2, c2extended, Dets, beginS0, beginSp, beginSm, norbs);
      }

      if (j == 0) {
	c1extended.block(0,0,c1.rows(),1) = 1.*c1;
      }
      else if (j==1) {
	getSplus(c1, c1extended, Dets, beginS0, beginSp, beginSm, norbs);
      }
      else if (j == 2) {
	getSminus(c1, c1extended, Dets, beginS0, beginSp, beginSm, norbs);
      }

      H(c2extended, Hc2);
      CItype element = (c1extended.adjoint()*Hc2)(0,0)/c2extended.norm()/c1extended.norm();
      RME[ (Mop+2)/2] = element/cg;

      break;
    }//for ij Sz1 Sz2
  }//Mop

  //now populate all elements of hsubspace
  for (int Sz1 = -spin1; Sz1 <=spin1; Sz1+=2)
    for (int Sz2 = -spin2; Sz2 <=spin2; Sz2+=2) {
      for (int Mop = -2; Mop<=2; Mop+=2) {
	double cg = clebsch(spin2, Sz2, Jop, Mop, spin1, Sz1);
	Hsubspace(rowIndex1 + (Sz1+spin1)/2, rowIndex2 + (Sz2+spin2)/2) += RME[ (Mop+2)/2]*cg*219470.;
	Hsubspace(rowIndex2 + (Sz2+spin2)/2, rowIndex1 + (Sz1+spin1)/2) += conj(RME[(Mop+2)/2]*cg)*219470.;
      }
    }
  return;
}

