#include "Determinants.h"
#include "SHCIbasics.h"
#include "SHCIgetdeterminants.h"
#include "SHCIsampledeterminants.h"
#include "SHCIrdm.h"
#include "SHCISortMpiUtils.h"
#include "input.h"
#include "integral.h"
#include <vector>
#include "math.h"
#include "Hmult.h"
#include <tuple>
#include <map>
#include "Davidson.h"
#include "boost/format.hpp"
#include <fstream>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "communicate.h"

using namespace std;
using namespace Eigen;
using namespace boost;
using namespace SHCISortMpiUtils;

void SHCIrdm::UpdateRDMPerturbativeDeterministic(vector<Determinant>& Dets, MatrixXx& ci, double& E0, 
						 oneInt& I1, twoInt& I2, schedule& schd, 
						 double coreE, int nelec, int norbs,
						 std::vector<StitchDEH>& uniqueDEH, int root) {

  int nSpatOrbs = norbs/2;
  
  //Read the variational RDM
  MatrixXx s2RDM, twoRDM;
  if (schd.DoSpinRDM ){
    if (mpigetrank() == 0) {
      char file [5000];
      sprintf (file, "%s/%d-spinRDM.bkp" , schd.prefix[0].c_str(), root );
      std::ifstream ifs(file, std::ios::binary);
      boost::archive::binary_iarchive load(ifs);
      load >> twoRDM;
      ComputeEnergyFromSpinRDM(norbs, nelec, I1, I2, coreE, twoRDM);
    }
    else
      twoRDM.setZero(norbs*(norbs+1)/2, norbs*(norbs+1)/2);
  }

  if (mpigetrank() == 0) {
    char file [5000];
    sprintf (file, "%s/%d-spatialRDM.bkp" , schd.prefix[0].c_str(), root );
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    load >> s2RDM;
    ComputeEnergyFromSpatialRDM(nSpatOrbs, nelec, I1, I2, coreE, s2RDM);
  }
  else
    s2RDM.setZero(nSpatOrbs*nSpatOrbs, nSpatOrbs*nSpatOrbs);


  int num_thrds = omp_get_max_threads();
  for (int thrd = 0; thrd <num_thrds; thrd++) {
    
    vector<Determinant>& uniqueDets = *uniqueDEH[thrd].Det;
    vector<double>& uniqueEnergy = *uniqueDEH[thrd].Energy;
    vector<CItype>& uniqueNumerator = *uniqueDEH[thrd].Num;
    vector<vector<int> >& uniqueVarIndices = *uniqueDEH[thrd].var_indices;
    vector<vector<size_t> >& uniqueOrbDiff = *uniqueDEH[thrd].orbDifference;
    
    for (size_t k=0; k<uniqueDets.size();k++) {
      for (size_t i=0; i<uniqueVarIndices[k].size(); i++){
	int d0=uniqueOrbDiff[k][i]%norbs, c0=(uniqueOrbDiff[k][i]/norbs)%norbs;
	
	if (uniqueOrbDiff[k][i]/norbs/norbs == 0) { // single excitation
	  vector<int> closed(nelec, 0);
	  vector<int> open(norbs-nelec,0);
	  Dets[uniqueVarIndices[k][i]].getOpenClosed(open, closed);
	  for (int n1=0;n1<nelec; n1++) {
	    double sgn = 1.0;
	    int a=max(closed[n1],c0), b=min(closed[n1],c0), I=max(closed[n1],d0), J=min(closed[n1],d0);
	    if (closed[n1] == d0) continue;
	    Dets[uniqueVarIndices[k][i]].parity(min(d0,c0), max(d0,c0),sgn);
	    if (!( (closed[n1] > c0 && closed[n1] > d0) || (closed[n1] < c0 && closed[n1] < d0))) sgn *=-1.;
	    if (schd.DoSpinRDM) {
	      twoRDM(a*(a+1)/2+b, I*(I+1)/2+J) += 0.5*sgn*uniqueNumerator[k]*ci(uniqueVarIndices[k][i],0)/(E0-uniqueEnergy[k]);
	      twoRDM(I*(I+1)/2+J, a*(a+1)/2+b) += 0.5*sgn*uniqueNumerator[k]*ci(uniqueVarIndices[k][i],0)/(E0-uniqueEnergy[k]);
	    }
	    populateSpatialRDM(a, b, I, J, s2RDM, 0.5*sgn*uniqueNumerator[k]*ci(uniqueVarIndices[k][i],0)/(E0-uniqueEnergy[k]), nSpatOrbs);
	    populateSpatialRDM(I, J, a, b, s2RDM, 0.5*sgn*uniqueNumerator[k]*ci(uniqueVarIndices[k][i],0)/(E0-uniqueEnergy[k]), nSpatOrbs);
	  } // for n1
	}  // single
	else { // double excitation
	  int d1=(uniqueOrbDiff[k][i]/norbs/norbs)%norbs, c1=(uniqueOrbDiff[k][i]/norbs/norbs/norbs)%norbs ;
	  double sgn = 1.0;
	  Dets[uniqueVarIndices[k][i]].parity(d1,d0,c1,c0,sgn);
	  int P = max(c1,c0), Q = min(c1,c0), R = max(d1,d0), S = min(d1,d0);
	  if (P != c0)  sgn *= -1;
	  if (Q != d0)  sgn *= -1;
	  
	  if (schd.DoSpinRDM) {
	    twoRDM(P*(P+1)/2+Q, R*(R+1)/2+S) += 0.5*sgn*uniqueNumerator[k]*ci(uniqueVarIndices[k][i],0)/(E0-uniqueEnergy[k]);
	    twoRDM(R*(R+1)/2+S, P*(P+1)/2+Q) += 0.5*sgn*uniqueNumerator[k]*ci(uniqueVarIndices[k][i],0)/(E0-uniqueEnergy[k]);
	  }
	  
	  populateSpatialRDM(P, Q, R, S, s2RDM, 0.5*sgn*uniqueNumerator[k]*ci(uniqueVarIndices[k][i],0)/(E0-uniqueEnergy[k]), nSpatOrbs);
	  populateSpatialRDM(R, S, P, Q, s2RDM, 0.5*sgn*uniqueNumerator[k]*ci(uniqueVarIndices[k][i],0)/(E0-uniqueEnergy[k]), nSpatOrbs);
	}// If
      } // i in variational connections to PT det k
    } // k in PT dets
  } //thrd in num_thrds
  
  if (schd.DoSpinRDM)
    MPI_Allreduce(MPI_IN_PLACE, &twoRDM(0,0), twoRDM.rows()*twoRDM.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &s2RDM(0,0), s2RDM.rows()*s2RDM.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  
  
  if(mpigetrank() == 0) {
    char file [5000];
    sprintf (file, "%s/spatialRDM.%d.%d.txt" , schd.prefix[0].c_str(), root, root );
    std::ofstream ofs(file, std::ios::out);
    ofs << nSpatOrbs<<endl;
    
    for (int n1=0; n1<nSpatOrbs; n1++)
      for (int n2=0; n2<nSpatOrbs; n2++)
	for (int n3=0; n3<nSpatOrbs; n3++)
	  for (int n4=0; n4<nSpatOrbs; n4++)
	    {
	      if (abs(s2RDM(n1*nSpatOrbs+n2, n3*nSpatOrbs+n4))  > 1.e-6)
		ofs << str(boost::format("%3d   %3d   %3d   %3d   %10.8g\n") % n1 % n2 % n3 % n4 % s2RDM(n1*nSpatOrbs+n2, n3*nSpatOrbs+n4));
	    }
    ofs.close();
    
    
    if (schd.DoSpinRDM) {
      char file [5000];
      sprintf (file, "%s/%d-spinRDM.bkp" , schd.prefix[0].c_str(), root );
      std::ofstream ofs(file, std::ios::binary);
      boost::archive::binary_oarchive save(ofs);
      save << twoRDM;
      ComputeEnergyFromSpinRDM(norbs, nelec, I1, I2, coreE, twoRDM);
    }
    
    {
      char file [5000];
      sprintf (file, "%s/%d-spatialRDM.bkp" , schd.prefix[0].c_str(), root );
      std::ofstream ofs(file, std::ios::binary);
      boost::archive::binary_oarchive save(ofs);
      save << s2RDM;
      ComputeEnergyFromSpatialRDM(nSpatOrbs, nelec, I1, I2, coreE, s2RDM);
    }
  }
  
}

void SHCIrdm::populateSpatialRDM(int& i, int& j, int& k, int& l, MatrixXx& s2RDM, 
				 CItype value, int& nSpatOrbs) {
  //we assume i != j  and  k != l
  int I = i/2, J=j/2, K=k/2, L=l/2;
  if (i%2 == l%2 && j%2 == k%2) {
    s2RDM(I*nSpatOrbs+J,L*nSpatOrbs+K) -= value;
    s2RDM(J*nSpatOrbs+I,K*nSpatOrbs+L) -= value;
  }

  if (i%2 == k%2 && l%2 == j%2 ) {
    s2RDM(I*nSpatOrbs+J,K*nSpatOrbs+L) += value;
    s2RDM(J*nSpatOrbs+I,L*nSpatOrbs+K) += value;
  }

}

void SHCIrdm::EvaluateAndStoreRDM(vector<vector<int> >& connections, vector<Determinant>& Dets, 
				  MatrixXx& ci,
				  vector<vector<size_t> >& orbDifference, int nelec, 
				  schedule& schd, int root, bool update) {
  boost::mpi::communicator world;

  size_t norbs = Dets[0].norbs;
  int nSpatOrbs = norbs/2;

  //RDM(i,j,k,l) = a_i^\dag a_j^\dag a_l a_k
  //also i>=j and k>=l
  MatrixXx twoRDM;
  if (schd.DoSpinRDM ){
    if (update) {
      if (mpigetrank() == 0) {
	char file [5000];
	sprintf (file, "%s/%d-spinRDM.bkp" , schd.prefix[0].c_str(), root );
	std::ifstream ifs(file, std::ios::binary);
	boost::archive::binary_iarchive load(ifs);
	load >> twoRDM;
      }
      mpi::broadcast(world, twoRDM, 0);
    }
    else
      twoRDM.setZero(norbs*(norbs+1)/2, norbs*(norbs+1)/2);
  }

  MatrixXx s2RDM;
  if (update) {
    if (mpigetrank() == 0) {
      char file [5000];
      sprintf (file, "%s/%d-spatialRDM.bkp" , schd.prefix[0].c_str(), root );
      std::ifstream ifs(file, std::ios::binary);
      boost::archive::binary_iarchive load(ifs);
      load >> s2RDM;
    }
    mpi::broadcast(world, s2RDM, 0);
  }
  else
    s2RDM.setZero(nSpatOrbs*nSpatOrbs, nSpatOrbs*nSpatOrbs);

  int num_thrds = omp_get_max_threads();

  //#pragma omp parallel for schedule(dynamic)
  for (int i=0; i<Dets.size(); i++) {
    if ((i/num_thrds)%world.size() != world.rank()) continue;

    vector<int> closed(nelec, 0);
    vector<int> open(norbs-nelec,0);
    Dets[i].getOpenClosed(open, closed);

    //<Di| Gamma |Di>
    for (int n1=0; n1<nelec; n1++) {
      for (int n2=0; n2<n1; n2++) {
	int orb1 = closed[n1], orb2 = closed[n2];
	if (schd.DoSpinRDM)
	  twoRDM(orb1*(orb1+1)/2 + orb2, orb1*(orb1+1)/2+orb2) += ci(i,0)*ci(i,0);
	populateSpatialRDM(orb1, orb2, orb1, orb2, s2RDM, ci(i,0)*ci(i,0), nSpatOrbs);
      }
    }


    for (int j=1; j<connections[i].size(); j++) {
      int d0=orbDifference[i][j]%norbs, c0=(orbDifference[i][j]/norbs)%norbs ;

      if (orbDifference[i][j]/norbs/norbs == 0) { //only single excitation
	for (int n1=0;n1<nelec; n1++) {
	  double sgn = 1.0;
	  int a=max(closed[n1],c0), b=min(closed[n1],c0), I=max(closed[n1],d0), J=min(closed[n1],d0);
	  if (closed[n1] == d0) continue;
	  Dets[i].parity(min(d0,c0), max(d0,c0),sgn);
	  if (!( (closed[n1] > c0 && closed[n1] > d0) || (closed[n1] < c0 && closed[n1] < d0))) sgn *=-1.;
	  if (schd.DoSpinRDM) {
	    twoRDM(a*(a+1)/2+b, I*(I+1)/2+J) += sgn*ci(connections[i][j],0)*ci(i,0);
	    twoRDM(I*(I+1)/2+J, a*(a+1)/2+b) += sgn*ci(connections[i][j],0)*ci(i,0);
	  }

	  populateSpatialRDM(a, b, I, J, s2RDM, sgn*ci(connections[i][j],0)*ci(i,0), nSpatOrbs);
	  populateSpatialRDM(I, J, a, b, s2RDM, sgn*ci(connections[i][j],0)*ci(i,0), nSpatOrbs);

	}
      }
      else {
	int d1=(orbDifference[i][j]/norbs/norbs)%norbs, c1=(orbDifference[i][j]/norbs/norbs/norbs)%norbs ;
	double sgn = 1.0;

	Dets[i].parity(d1,d0,c1,c0,sgn);

	if (schd.DoSpinRDM) {
	  twoRDM(c1*(c1+1)/2+c0, d1*(d1+1)/2+d0) += sgn*ci(connections[i][j],0)*ci(i,0);
	  twoRDM(d1*(d1+1)/2+d0, c1*(c1+1)/2+c0) += sgn*ci(connections[i][j],0)*ci(i,0);
	}

	populateSpatialRDM(c1, c0, d1, d0, s2RDM, sgn*ci(connections[i][j],0)*ci(i,0), nSpatOrbs);
	populateSpatialRDM(d1, d0, c1, c0, s2RDM, sgn*ci(connections[i][j],0)*ci(i,0), nSpatOrbs);
      }
    }
  }

  if (schd.DoSpinRDM)
    MPI_Allreduce(MPI_IN_PLACE, &twoRDM(0,0), twoRDM.rows()*twoRDM.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &s2RDM(0,0), s2RDM.rows()*s2RDM.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if(mpigetrank() == 0) {
    char file [5000];
    sprintf (file, "%s/spatialRDM.%d.%d.txt" , schd.prefix[0].c_str(), root, root );
    std::ofstream ofs(file, std::ios::out);
    ofs << nSpatOrbs<<endl;

    for (int n1=0; n1<nSpatOrbs; n1++)
      for (int n2=0; n2<nSpatOrbs; n2++)
	for (int n3=0; n3<nSpatOrbs; n3++)
	  for (int n4=0; n4<nSpatOrbs; n4++)
	    {
	      if (abs(s2RDM(n1*nSpatOrbs+n2, n3*nSpatOrbs+n4))  > 1.e-6)
		ofs << str(boost::format("%3d   %3d   %3d   %3d   %10.8g\n") % n1 % n2 % n3 % n4 % s2RDM(n1*nSpatOrbs+n2, n3*nSpatOrbs+n4));
	    }
    ofs.close();



    if (schd.DoSpinRDM) {
      char file [5000];
      sprintf (file, "%s/%d-spinRDM.bkp" , schd.prefix[0].c_str(), root );
      std::ofstream ofs(file, std::ios::binary);
      boost::archive::binary_oarchive save(ofs);
      save << twoRDM;
    }

    {
      char file [5000];
      sprintf (file, "%s/%d-spatialRDM.bkp" , schd.prefix[0].c_str(), root );
      std::ofstream ofs(file, std::ios::binary);
      boost::archive::binary_oarchive save(ofs);
      save << s2RDM;
    }
  }

}



void SHCIrdm::ComputeEnergyFromSpinRDM(int norbs, int nelec, oneInt& I1, twoInt& I2, 
				       double coreE, MatrixXx& twoRDM) {

  //RDM(i,j,k,l) = a_i^\dag a_j^\dag a_l a_k
  //also i>=j and k>=l
  double energy = coreE;
  double onebody = 0.0;
  double twobody = 0.0;
  //if (mpigetrank() == 0)  cout << "Core energy= " << energy << endl;

  MatrixXx oneRDM = MatrixXx::Zero(norbs, norbs);
#pragma omp parallel for schedule(dynamic)
  for (int p=0; p<norbs; p++)
    for (int q=0; q<norbs; q++)
      for (int r=0; r<norbs; r++) {
	int P = max(p,r), R1 = min(p,r);
	int Q = max(q,r), R2 = min(q,r);
	double sgn = 1.;
	if (P != p)  sgn *= -1;
	if (Q != q)  sgn *= -1;

	oneRDM(p,q) += sgn*twoRDM(P*(P+1)/2+R1,Q*(Q+1)/2+R2)/(nelec-1.);
      }

#pragma omp parallel for reduction(+ : onebody)
  for (int p=0; p<norbs; p++)
    for (int q=0; q<norbs; q++)
#ifdef Complex
      onebody += (I1(p, q)*oneRDM(p,q)).real();
#else
  onebody += I1(p, q)*oneRDM(p,q);
#endif

#pragma omp parallel for reduction(+ : twobody)
  for (int p=0; p<norbs; p++){
    for (int q=0; q<norbs; q++){
      for (int r=0; r<norbs; r++){
	for (int s=0; s<norbs; s++){
	  //if (p%2 != r%2 || q%2 != s%2)  continue; // This line is not necessary
	  int P = max(p,q), Q = min(p,q);
	  int R = max(r,s), S = min(r,s);
	  double sgn = 1;
	  if (P != p)  sgn *= -1;
	  if (R != r)  sgn *= -1;
#ifdef Complex
	  twobody += (sgn * 0.5 * twoRDM(P*(P+1)/2+Q, R*(R+1)/2+S) * I2(p,r,q,s)).real(); // 2-body term
#else
	  twobody += sgn * 0.5 * twoRDM(P*(P+1)/2+Q, R*(R+1)/2+S) * I2(p,r,q,s); // 2-body term
#endif
	}
      }
    }
  }

  //if (mpigetrank() == 0)  cout << "One-body from 2RDM: " << onebody << endl;
  //if (mpigetrank() == 0)  cout << "Two-body from 2RDM: " << twobody << endl;

  energy += onebody + twobody;
  if (mpigetrank() == 0)  cout << "E from 2RDM: " << energy << endl;

}


void SHCIrdm::ComputeEnergyFromSpatialRDM(int norbs, int nelec, oneInt& I1, twoInt& I2, 
					  double coreE, MatrixXx& twoRDM) {

  double energy = coreE;
  double onebody = 0.0;
  double twobody = 0.0;

  MatrixXx oneRDM = MatrixXx::Zero(norbs, norbs);
#pragma omp parallel for schedule(dynamic)
  for (int p=0; p<norbs; p++)
    for (int q=0; q<norbs; q++)
      for (int r=0; r<norbs; r++)
	oneRDM(p,q) += twoRDM(p*norbs+r, q*norbs+ r)/(1.*nelec-1.);


#pragma omp parallel for reduction(+ : onebody)
  for (int p=0; p<norbs; p++)
    for (int q=0; q<norbs; q++)
#ifdef Complex
      onebody += (I1(2*p, 2*q)*oneRDM(p,q)).real();
#else
  onebody += I1(2*p, 2*q)*oneRDM(p,q);
#endif

#pragma omp parallel for reduction(+ : twobody)
  for (int p=0; p<norbs; p++)
    for (int q=0; q<norbs; q++)
      for (int r=0; r<norbs; r++)
	for (int s=0; s<norbs; s++)
#ifdef Complex
	  twobody +=  (0.5 * twoRDM(p*norbs+q,r*norbs+s) * I2(2*p,2*r,2*q,2*s)).real(); // 2-body term
#else
  twobody +=  0.5*twoRDM(p*norbs+q,r*norbs+s) * I2(2*p,2*r,2*q,2*s); // 2-body term
#endif

  energy += onebody + twobody;
  if (mpigetrank() == 0)  cout << "E from 2RDM: " << energy << endl;

}



void SHCIrdm::printRDM(int norbs, schedule& schd, int root, MatrixXx& twoRDM) {

  boost::mpi::communicator world;

  //RDM(i,j,k,l) = a_i^\dag a_j^\dag a_l a_k
  //also i>=j and k>=l
  int num_thrds = omp_get_max_threads();

  int nSpatOrbs = norbs/2;


  if(mpigetrank() == 0) {
    MatrixXx s2RDM(nSpatOrbs*nSpatOrbs, nSpatOrbs*nSpatOrbs);
    s2RDM *= 0.0;

    char file [5000];
    sprintf (file, "%s/spatialRDM.%d.%d.txt" , schd.prefix[0].c_str(), root, root );
    std::ofstream ofs(file, std::ios::out);
    ofs << nSpatOrbs<<endl;

#pragma omp parallel for schedule(static)
    for (int n1=0; n1<nSpatOrbs; n1++)
      for (int n2=0; n2<nSpatOrbs; n2++)
	for (int n3=0; n3<nSpatOrbs; n3++)
	  for (int n4=0; n4<nSpatOrbs; n4++)
	    {
	      double sgn = 1.0;
	      int N1 = 2*max(n1,n2), N2=2*min(n1,n2), N3=2*max(n3,n4), N4=2*min(n3,n4);
	      if(( (n1>=n2 && n3<n4) || (n1<n2 && n3>=n4))) sgn = -1.0;
	      s2RDM(n1*nSpatOrbs+n2, n3*nSpatOrbs+n4) += sgn*twoRDM( N1*(N1+1)/2+N2, N3*(N3+1)/2+N4);

	      sgn = 1.0;
	      N1 = max(2*n1+1,2*n2); N2=min(2*n1+1,2*n2); N3=max(2*n3+1,2*n4); N4=min(2*n3+1,2*n4);
	      if(!( (2*n1+1>2*n2 && 2*n3+1>2*n4) || (2*n1+1<2*n2 && 2*n3+1<2*n4))) sgn = -1.0;
	      s2RDM(n1*nSpatOrbs+n2, n3*nSpatOrbs+n4) += sgn*twoRDM( N1*(N1+1)/2+N2, N3*(N3+1)/2+N4);

	      sgn = 1.0;
	      N1 = max(2*n1,2*n2+1); N2=min(2*n1,2*n2+1); N3=max(2*n3,2*n4+1); N4=min(2*n3,2*n4+1);
	      if(!( (2*n1>=2*n2+1 && 2*n3>=2*n4+1) || (2*n1<2*n2+1 && 2*n3<2*n4+1))) sgn = -1.0;
	      s2RDM(n1*nSpatOrbs+n2, n3*nSpatOrbs+n4) += sgn*twoRDM( N1*(N1+1)/2+N2, N3*(N3+1)/2+N4);

	      sgn = 1.0;
	      N1 = 2*max(n1,n2)+1; N2=2*min(n1,n2)+1; N3=2*max(n3,n4)+1; N4=2*min(n3,n4)+1;
	      if(( (n1>=n2 && n3<n4) || (n1<n2 && n3>=n4))) sgn = -1.0;
	      s2RDM(n1*nSpatOrbs+n2, n3*nSpatOrbs+n4) += sgn*twoRDM( N1*(N1+1)/2+N2, N3*(N3+1)/2+N4);

	    }

    for (int n1=0; n1<nSpatOrbs; n1++)
      for (int n2=0; n2<nSpatOrbs; n2++)
	for (int n3=0; n3<nSpatOrbs; n3++)
	  for (int n4=0; n4<nSpatOrbs; n4++)
	    {
	      if (abs(s2RDM(n1*nSpatOrbs+n2, n3*nSpatOrbs+n4))  > 1.e-6)
		ofs << str(boost::format("%3d   %3d   %3d   %3d   %10.8g\n") % n1 % n2 % n3 % n4 % s2RDM(n1*nSpatOrbs+n2, n3*nSpatOrbs+n4));
	    }



    {
      char file [5000];
      sprintf (file, "%s/%d-spinRDM.bkp" , schd.prefix[0].c_str(), root );
      std::ofstream ofs(file, std::ios::binary);
      boost::archive::binary_oarchive save(ofs);
      save << twoRDM;
    }

    {
      char file [5000];
      sprintf (file, "%s/%d-spatialRDM.bkp" , schd.prefix[0].c_str(), root );
      std::ofstream ofs(file, std::ios::binary);
      boost::archive::binary_oarchive save(ofs);
      save << s2RDM;
    }
  }
}

