#include "integral.h"
#include <fstream>
#include "string.h"
#include <boost/algorithm/string.hpp>
#include <algorithm>
#include "math.h"
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "communicate.h"

using namespace boost;
bool myfn(double i, double j) { return fabs(i)<fabs(j); }

#ifdef Complex
void readSOCIntegrals(oneInt& I1, int norbs) {
  if (mpigetrank() == 0) {
    vector<string> tok;
    string msg;

    //Read SOC.X
    {
      ifstream dump("SOC.X");
      int N;
      dump >> N;
      if (N != norbs) {
	cout << "number of orbitals in SOC.X should be equal to norbs in the input file."<<endl;
	exit(0);
      }
      
      //I1soc[1].store.resize(N*(N+1)/2, 0.0);
      while(!dump.eof()) {
	std::getline(dump, msg);
	trim(msg);
	boost::split(tok, msg, is_any_of(", \t="), token_compress_on);
	if (tok.size() != 3)
	  continue;
	
	double integral = atof(tok[0].c_str());
	int a=atoi(tok[1].c_str()), b=atoi(tok[2].c_str());
	I1(2*(a-1), 2*(b-1)+1) += integral/2.;  //alpha beta
	I1(2*(a-1)+1, 2*(b-1)) += integral/2.;  //beta alpha
      }      
    }

    //Read SOC.Y
    {
      ifstream dump("SOC.Y");
      int N;
      dump >> N;
      if (N != norbs) {
	cout << "number of orbitals in SOC.Y should be equal to norbs in the input file."<<endl;
	exit(0);
      }
      
      //I1soc[2].store.resize(N*(N+1)/2, 0.0);
      while(!dump.eof()) {
	std::getline(dump, msg);
	trim(msg);
	boost::split(tok, msg, is_any_of(", \t="), token_compress_on);
	if (tok.size() != 3)
	  continue;
	
	double integral = atof(tok[0].c_str());
	int a=atoi(tok[1].c_str()), b=atoi(tok[2].c_str());
	I1(2*(a-1), 2*(b-1)+1) += std::complex<double>(0, integral/2.);  //alpha beta
	I1(2*(a-1)+1, 2*(b-1)) += std::complex<double>(0, -integral/2.);  //beta alpha
      }      
    }


    //Read SOC.Z
    {
      ifstream dump("SOC.Z");
      int N;
      dump >> N;
      if (N != norbs) {
	cout << "number of orbitals in SOC.Z should be equal to norbs in the input file."<<endl;
	exit(0);
      }
      
      //I1soc[3].store.resize(N*(N+1)/2, 0.0);
      while(!dump.eof()) {
	std::getline(dump, msg);
	trim(msg);
	boost::split(tok, msg, is_any_of(", \t="), token_compress_on);
	if (tok.size() != 3)
	  continue;
	
	double integral = atof(tok[0].c_str());
	int a=atoi(tok[1].c_str()), b=atoi(tok[2].c_str());
	I1(2*(a-1), 2*(b-1)) += integral; //alpha, alpha
	I1(2*(a-1)+1, 2*(b-1)+1) += -integral; //beta, beta
      }      
    }

  }

}
#endif

void readIntegrals(string fcidump, twoInt& I2, oneInt& I1, int& nelec, int& norbs, double& coreE,
		   std::vector<int>& irrep) {

  if (mpigetrank() == 0) {

    I2.ksym = false;
    ifstream dump(fcidump.c_str());
    bool startScaling = false;
    norbs = -1;
    nelec = -1;
    
    int index = 0;
    vector<string> tok;
    string msg;
    while(!dump.eof()) {
      std::getline(dump, msg);
      trim(msg);
      boost::split(tok, msg, is_any_of(", \t="), token_compress_on);
      
      if (startScaling == false && tok.size() == 1 && (boost::iequals(tok[0],"&END") || boost::iequals(tok[0], "/"))) {
	startScaling = true;
	index += 1;
	break;
      }
      else if(startScaling == false) {
	if (boost::iequals(tok[0].substr(0,4),"&FCI")) {
	  if (boost::iequals(tok[1].substr(0,4), "NORB"))
	    norbs = atoi(tok[2].c_str());
	  
	  if (boost::iequals(tok[3].substr(0,5), "NELEC"))
	    nelec = atoi(tok[4].c_str());
	}
	else if (boost::iequals(tok[0].substr(0,4),"ISYM"))
	  continue;
	else if (boost::iequals(tok[0].substr(0,4),"KSYM"))
	  I2.ksym = true;
	else if (boost::iequals(tok[0].substr(0,6),"ORBSYM")) {
	  for (int i=1;i<tok.size(); i++)
	    irrep.push_back(atoi(tok[i].c_str()));
	}
	else {
	  for (int i=0;i<tok.size(); i++)
	    irrep.push_back(atoi(tok[i].c_str()));
	}
	
	index += 1;
      }
    }
    
    if (norbs == -1 || nelec == -1) {
      std::cout << "could not read the norbs or nelec"<<std::endl;
      exit(0);
    }
    irrep.resize(norbs);
    
    long npair = norbs*(norbs+1)/2;
    //I2.ksym = true;
    if (I2.ksym) {
      npair = norbs*norbs;
    }
    I2.norbs = norbs;
    I2.store.resize( npair*(npair+1)/2, 0.0);
    //I2.store.resize( npair*npair, npair*npair);
    I1.store.resize(2*norbs*(2*norbs+1)/2,0.0);
    coreE = 0.0;

    while(!dump.eof()) {
      std::getline(dump, msg);
      trim(msg);
      boost::split(tok, msg, is_any_of(", \t"), token_compress_on);
      if (tok.size() != 5)
	continue;
      
      double integral = atof(tok[0].c_str());int a=atoi(tok[1].c_str()), b=atoi(tok[2].c_str()), 
					       c=atoi(tok[3].c_str()), d=atoi(tok[4].c_str());
      
      if(a==b&&b==c&&c==d&&d==0)
	coreE = integral;
      else if (b==c&&c==d&&d==0)
	continue;//orbital energy
      else if (c==d&&d==0) {
	I1(2*(a-1),2*(b-1)) += integral; //alpha,alpha
	I1(2*(a-1)+1,2*(b-1)+1) += integral; //beta,beta
      }
      else
	I2(2*(a-1),2*(b-1),2*(c-1),2*(d-1)) = integral;
    }

    //exit(0);
    I2.maxEntry = *std::max_element(&I2.store[0], &I2.store[0]+I2.store.size(),myfn);
    I2.Direct = MatrixXd::Zero(norbs, norbs); I2.Direct *= 0.;
    I2.Exchange = MatrixXd::Zero(norbs, norbs); I2.Exchange *= 0.;
    
    for (int i=0; i<norbs; i++)
      for (int j=0; j<norbs; j++) {
	I2.Direct(i,j) = I2(2*i,2*i,2*j,2*j);
	I2.Exchange(i,j) = I2(2*i,2*j,2*j,2*i);
      }
  }

#ifndef SERIAL
  boost::mpi::communicator world;
  mpi::broadcast(world, I1, 0);

  size_t i2size = I2.store.size();
  mpi::broadcast(world, i2size, 0);
  if (mpigetrank() != 0)
    I2.store.resize(i2size);

  long intdim = I2.store.size();
  long  maxint = 26843540; //mpi cannot transfer more than these number of doubles
  long maxIter = intdim/maxint; 
  for (int i=0; i<maxIter; i++) {
    MPI::COMM_WORLD.Bcast(&I2.store[i*maxint], maxint, MPI_DOUBLE, 0);
  }
  MPI::COMM_WORLD.Bcast(&I2.store[(maxIter)*maxint], I2.store.size() - maxIter*maxint, MPI_DOUBLE, 0);

  mpi::broadcast(world, I2.ksym, 0);
  mpi::broadcast(world, I2.maxEntry, 0);
  mpi::broadcast(world, I2.Direct, 0);
  mpi::broadcast(world, I2.Exchange, 0);
  mpi::broadcast(world, I2.norbs, 0);
  mpi::broadcast(world, I2.zero, 0);

  mpi::broadcast(world, nelec, 0);
  mpi::broadcast(world, norbs, 0);
  mpi::broadcast(world, coreE, 0);
  mpi::broadcast(world, irrep, 0);
#endif

  return;
    

}


