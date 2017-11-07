/*
  Developed by Sandeep Sharma with contributions from James E. T. Smith and Adam A. Holmes, 2017
  Copyright (c) 2017, Sandeep Sharma
  
  This file is part of DICE.
  
  This program is free software: you can redistribute it and/or modify it under the terms
  of the GNU General Public License as published by the Free Software Foundation, 
  either version 3 of the License, or (at your option) any later version.
  
  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
  See the GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License along with this program. 
  If not, see <http://www.gnu.org/licenses/>.
*/
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

namespace localConj {
  CItype conj(CItype a) {
#ifdef Complex
    return std::conj(a);
#else
    return a;
#endif
  }
};



void SHCIrdm::makeRDM(int* &AlphaMajorToBetaLen, 
		      vector<int* > &AlphaMajorToBeta   ,
		      vector<int* > &AlphaMajorToDet    ,
		      int*          &BetaMajorToAlphaLen, 
		      vector<int* > &BetaMajorToAlpha   ,
		      vector<int* > &BetaMajorToDet     ,
		      int*          &SinglesFromAlphaLen, 
		      vector<int* > &SinglesFromAlpha   ,
		      int*          &SinglesFromBetaLen , 
		      vector<int* > &SinglesFromBeta    ,
		      Determinant* Dets,
		      int DetsSize,
		      int Norbs, int nelec, CItype* cibra, 
		      CItype* ciket,
		      MatrixXx& s2RDM) {


  int proc=commrank, nprocs=commsize;

  size_t norbs = Norbs;
  int nSpatOrbs = norbs/2;

  int EndIndex = DetsSize;

  //diagonal element
  for (size_t k=0; k<EndIndex; k++) {
    if (k%(nprocs) != proc ) continue;

    vector<int> closed(nelec, 0);
    vector<int> open(norbs-nelec,0);
    Dets[k].getOpenClosed(open, closed);

    for (int n1=0; n1<nelec; n1++) {
      for (int n2=0; n2<n1; n2++) {
	int orb1 = closed[n1], orb2 = closed[n2];
	//if (schd.DoSpinRDM)
	//twoRDM(orb1*(orb1+1)/2 + orb2, orb1*(orb1+1)/2+orb2) += localConj::conj(cibra[i])*ciket[i];
	populateSpatialRDM(orb1, orb2, orb1, orb2, s2RDM, localConj::conj(cibra[k])*ciket[k], nSpatOrbs);
      }
    }
  }
    
  //alpha-beta excitation
  for (int i=0; i<AlphaMajorToBeta.size(); i++) {
    
    for (int ii=0; ii<AlphaMajorToBetaLen[i]; ii++) {
      
      int Astring = i, 
	Bstring = AlphaMajorToBeta[i][ii], 
	DetI    = AlphaMajorToDet [i][ii];

      
      if ((std::abs(DetI)-1)%nprocs != proc ) 
	continue;
      
      vector<int> closed(nelec, 0);
      vector<int> open(norbs-nelec,0);
      Dets[abs(DetI)-1].getOpenClosed(open, closed);
      Determinant di = Dets[abs(DetI)-1];


      int maxBToA = BetaMajorToAlpha[Bstring][BetaMajorToAlphaLen[Bstring]-1];
      //singles from Astring
      for (int j=0; j<SinglesFromAlphaLen[Astring]; j++) {
	int Asingle = SinglesFromAlpha[Astring][j];
	
	int index = binarySearch ( &BetaMajorToAlpha[Bstring][0] , 
				   0                             , 
				   BetaMajorToAlphaLen[Bstring]-1, 
				   Asingle                       );
	if (index != -1 ) {
	  int DetJ = BetaMajorToDet[Bstring][index];

	  if (std::abs(DetJ) >=  std::abs(DetI)) continue;
	  Determinant dj = Dets[abs(DetJ)-1];
	  size_t orbDiff;
	  getOrbDiff(dj, di, orbDiff);
	  int d0 = orbDiff%norbs, c0= (orbDiff/norbs)%norbs;

	  for (int n1=0;n1<nelec; n1++) {
	    double sgn = 1.0;
	    int a=max(closed[n1],c0), b=min(closed[n1],c0), I=max(closed[n1],d0), J=min(closed[n1],d0);
	    if (closed[n1] == d0) continue;
	    di.parity(min(d0,c0), max(d0,c0),sgn);
	    if (!( (closed[n1] > c0 && closed[n1] > d0) || (closed[n1] < c0 && closed[n1] < d0)))
	      sgn *=-1.;
	    populateSpatialRDM(a, b, I, J, s2RDM, sgn*localConj::conj(cibra[abs(DetJ)-1])*ciket[abs(DetI)-1], nSpatOrbs);
	    populateSpatialRDM(I, J, a, b, s2RDM, sgn*localConj::conj(ciket[abs(DetJ)-1])*cibra[abs(DetI)-1], nSpatOrbs);
	  }

	}
      }
	
      //single Alpha and single Beta
      for (int j=0; j<SinglesFromAlphaLen[Astring]; j++) {
	int Asingle = SinglesFromAlpha[Astring][j];
	
	int SearchStartIndex = 0, AlphaToBetaLen = AlphaMajorToBetaLen[Asingle],
	  SinglesFromBLen  = SinglesFromBetaLen[Bstring];
	int maxAToB = AlphaMajorToBeta[Asingle][AlphaMajorToBetaLen[Asingle]-1];
	for (int k=0; k<SinglesFromBLen; k++) {
	  int& Bsingle = SinglesFromBeta[Bstring][k];
	  
	  if (SearchStartIndex >= AlphaToBetaLen) break;
	  
	  int index=SearchStartIndex;
	  for (; index <AlphaToBetaLen && AlphaMajorToBeta[Asingle][index] < Bsingle; index++) {}
	  
	  SearchStartIndex = index;
	  if (index <AlphaToBetaLen && AlphaMajorToBeta[Asingle][index] == Bsingle) {
	    
	    int DetJ = AlphaMajorToDet[Asingle][SearchStartIndex];
	    //if (std::abs(DetJ) < max(offSet, StartIndex) && std::abs(DetI) < max(offSet, StartIndex)) continue;
	    if (std::abs(DetJ) >=  std::abs(DetI)) continue;
	    Determinant dj = Dets[abs(DetJ)-1];
	    size_t orbDiff;
	    getOrbDiff(dj, di, orbDiff);
	    //CItype hij = Hij(Dets[std::abs(DetJ)], Dets[std::abs(DetI)], I1, I2, coreE, orbDiff);

	    int d0=orbDiff%norbs, c0=(orbDiff/norbs)%norbs ;
	    int d1=(orbDiff/norbs/norbs)%norbs, c1=(orbDiff/norbs/norbs/norbs)%norbs ;
	    double sgn = 1.0;
	    
	    di.parity(d1,d0,c1,c0,sgn);
	    populateSpatialRDM(c1, c0, d1, d0, s2RDM, sgn*localConj::conj(cibra[abs(DetJ)-1])*ciket[abs(DetI)-1], nSpatOrbs);
	    populateSpatialRDM(d1, d0, c1, c0, s2RDM, sgn*localConj::conj(ciket[abs(DetJ)-1])*cibra[abs(DetI)-1], nSpatOrbs);
	  }
	}
      }
      
	//singles from Bstring
      int maxAtoB = AlphaMajorToBeta[Astring][AlphaMajorToBetaLen[Astring]-1];
      for (int j=0; j< SinglesFromBetaLen[Bstring]; j++) {
	int Bsingle =  SinglesFromBeta   [Bstring][j];
	
	//if (Bsingle > maxAtoB) break;
	int index = binarySearch( &AlphaMajorToBeta[Astring][0] , 
				  0                             , 
				  AlphaMajorToBetaLen[Astring]-1, 
				  Bsingle                        );
	
	if (index != -1 ) {
	  int DetJ = AlphaMajorToDet[Astring][index];
	  if (std::abs(DetJ) >= std::abs(DetI)) continue;
	  Determinant dj = Dets[abs(DetJ)-1];

	  size_t orbDiff;
	  getOrbDiff(dj, di, orbDiff);
	  //CItype hij = Hij(Dets[std::abs(DetJ)], Dets[std::abs(DetI)], I1, I2, coreE, orbDiff);
	  int d0 = orbDiff%norbs, c0= (orbDiff/norbs)%norbs;

	  for (int n1=0;n1<nelec; n1++) {
	    double sgn = 1.0;
	    int a=max(closed[n1],c0), b=min(closed[n1],c0), I=max(closed[n1],d0), J=min(closed[n1],d0);
	    if (closed[n1] == d0) continue;
	    di.parity(min(d0,c0), max(d0,c0),sgn);
	    if (!( (closed[n1] > c0 && closed[n1] > d0) || (closed[n1] < c0 && closed[n1] < d0)))
	      sgn *=-1.;
	    populateSpatialRDM(a, b, I, J, s2RDM, sgn*localConj::conj(cibra[abs(DetJ)-1])*ciket[abs(DetI)-1], nSpatOrbs);
	    populateSpatialRDM(I, J, a, b, s2RDM, sgn*localConj::conj(ciket[abs(DetJ)-1])*cibra[abs(DetI)-1], nSpatOrbs);
	  }

	}
      }
      
      
      //double beta excitation
      for (int j=0; j< AlphaMajorToBetaLen[i]; j++) {
	int DetJ     = AlphaMajorToDet    [i][j];
	//if (std::abs(DetJ) < StartIndex) continue;
	//if (std::abs(DetJ) < max(offSet, StartIndex) && std::abs(DetI) < max(offSet, StartIndex)) continue;
	if (std::abs(DetJ) >= std::abs(DetI)) continue;
	Determinant dj = Dets[abs(DetJ)-1];

	if (dj.ExcitationDistance(di) == 2 ) {
	  size_t orbDiff;
	  getOrbDiff(dj, di, orbDiff);
	  //CItype hij = Hij(Dets[std::abs(DetJ)], Dets[std::abs(DetI)], I1, I2, coreE, orbDiff);
	  int d0=orbDiff%norbs, c0=(orbDiff/norbs)%norbs ;
	  int d1=(orbDiff/norbs/norbs)%norbs, c1=(orbDiff/norbs/norbs/norbs)%norbs ;
	  double sgn = 1.0;
	  
	  di.parity(d1,d0,c1,c0,sgn);
	  populateSpatialRDM(c1, c0, d1, d0, s2RDM, sgn*localConj::conj(cibra[abs(DetJ)-1])*ciket[abs(DetI)-1], nSpatOrbs);
	  populateSpatialRDM(d1, d0, c1, c0, s2RDM, sgn*localConj::conj(ciket[abs(DetJ)-1])*cibra[abs(DetI)-1], nSpatOrbs);
	}
      }
      
      //double Alpha excitation
      for (int j=0; j < BetaMajorToAlphaLen[Bstring]; j++) {
	int DetJ      = BetaMajorToDet     [Bstring][j];
	//if (std::abs(DetJ) < StartIndex) continue;
	//if (std::abs(DetJ) < max(offSet, StartIndex) && std::abs(DetI) < max(offSet, StartIndex)) continue;
	if (std::abs(DetJ) >= std::abs(DetI)) continue;

	Determinant dj = Dets[abs(DetJ)-1];
	if (di.ExcitationDistance(dj) == 2) {
	  size_t orbDiff;
	  getOrbDiff(dj, di, orbDiff);
	  //CItype hij = Hij(Dets[std::abs(DetJ)], Dets[std::abs(DetI)], I1, I2, coreE, orbDiff);
	  int d0=orbDiff%norbs, c0=(orbDiff/norbs)%norbs ;
	  int d1=(orbDiff/norbs/norbs)%norbs, c1=(orbDiff/norbs/norbs/norbs)%norbs ;
	  double sgn = 1.0;
	  
	  di.parity(d1,d0,c1,c0,sgn);
	  populateSpatialRDM(c1, c0, d1, d0, s2RDM, sgn*localConj::conj(cibra[abs(DetJ)-1])*ciket[abs(DetI)-1], nSpatOrbs);
	  populateSpatialRDM(d1, d0, c1, c0, s2RDM, sgn*localConj::conj(ciket[abs(DetJ)-1])*cibra[abs(DetI)-1], nSpatOrbs);
	}
      }
      
    }
  }

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &s2RDM(0,0), s2RDM.rows()*s2RDM.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  
}


void SHCIrdm::loadRDM(schedule& schd, MatrixXx& s2RDM, MatrixXx& twoRDM, int root) {
  int norbs = twoRDM.rows();
  int nSpatOrbs = pow(s2RDM.rows(),0.5);
  if (schd.DoSpinRDM ){
    if (commrank == 0) {
      char file [5000];
      sprintf (file, "%s/%d-spinRDM.bkp" , schd.prefix[0].c_str(), root );
      std::ifstream ifs(file, std::ios::binary);
      boost::archive::binary_iarchive load(ifs);
      load >> twoRDM;
      //ComputeEnergyFromSpinRDM(norbs, nelec, I1, I2, coreE, twoRDM);
    }
    else
      twoRDM.setZero(norbs*(norbs+1)/2, norbs*(norbs+1)/2);
  }

  if (commrank == 0) {
    char file [5000];
    sprintf (file, "%s/%d-spatialRDM.bkp" , schd.prefix[0].c_str(), root );
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    load >> s2RDM;
    //ComputeEnergyFromSpatialRDM(nSpatOrbs, nelec, I1, I2, coreE, s2RDM);
  }
  else
    s2RDM.setZero(nSpatOrbs*nSpatOrbs, nSpatOrbs*nSpatOrbs);

}

void SHCIrdm::saveRDM(schedule& schd, MatrixXx& s2RDM, MatrixXx& twoRDM, int root) {
  int nSpatOrbs = pow(s2RDM.rows(),0.5);
  if(commrank == 0) {
    char file [5000];
    sprintf (file, "%s/spatialRDM.%d.%d.txt" , schd.prefix[0].c_str(), root, root );
    std::ofstream ofs(file, std::ios::out);
    ofs << nSpatOrbs<<endl;

    for (int n1=0; n1<nSpatOrbs; n1++)
      for (int n2=0; n2<nSpatOrbs; n2++)
	for (int n3=0; n3<nSpatOrbs; n3++)
	  for (int n4=0; n4<nSpatOrbs; n4++)
	    {
	      if (fabs(s2RDM(n1*nSpatOrbs+n2, n3*nSpatOrbs+n4))  > 1.e-6)
		ofs << str(boost::format("%3d   %3d   %3d   %3d   %10.8g\n") % n1 % n2 % n3 % n4 % s2RDM(n1*nSpatOrbs+n2, n3*nSpatOrbs+n4));
	    }
    ofs.close();


    if (schd.DoSpinRDM) {
      char file [5000];
      sprintf (file, "%s/%d-spinRDM.bkp" , schd.prefix[0].c_str(), root );
      std::ofstream ofs(file, std::ios::binary);
      boost::archive::binary_oarchive save(ofs);
      save << twoRDM;
      //ComputeEnergyFromSpinRDM(norbs, nelec, I1, I2, coreE, twoRDM);
    }

    {
      char file [5000];
      sprintf (file, "%s/%d-spatialRDM.bkp" , schd.prefix[0].c_str(), root );
      std::ofstream ofs(file, std::ios::binary);
      boost::archive::binary_oarchive save(ofs);
      save << s2RDM;
      //ComputeEnergyFromSpatialRDM(nSpatOrbs, nelec, I1, I2, coreE, s2RDM);
    }
  }

}




void SHCIrdm::UpdateRDMResponsePerturbativeDeterministic(Determinant *Dets, int DetsSize, CItype *ci, double& E0,
							 oneInt& I1, twoInt& I2, schedule& schd,
							 double coreE, int nelec, int norbs,
							 StitchDEH& uniqueDEH, int root,
							 double& Psi1Norm, MatrixXx& s2RDM, MatrixXx& twoRDM) {

  s2RDM *= (1.-Psi1Norm);

  int nSpatOrbs = norbs/2;

  vector<Determinant>& uniqueDets = *uniqueDEH.Det;
  vector<double>& uniqueEnergy = *uniqueDEH.Energy;
  vector<CItype>& uniqueNumerator = *uniqueDEH.Num;
  vector<vector<int> >& uniqueVarIndices = *uniqueDEH.var_indices;
  vector<vector<size_t> >& uniqueOrbDiff = *uniqueDEH.orbDifference;


  for (size_t i=0; i<uniqueDets.size();i++) 
  {
    vector<int> closed(nelec, 0);
    vector<int> open(norbs-nelec,0);
    uniqueDets[i].getOpenClosed(open, closed);
    
    CItype coeff = uniqueNumerator[i]/(E0-uniqueEnergy[i]);
    //<Di| Gamma |Di>
    for (int n1=0; n1<nelec; n1++) {
      for (int n2=0; n2<n1; n2++) {
	int orb1 = closed[n1], orb2 = closed[n2];
	if (schd.DoSpinRDM)
#ifdef Complex
	  twoRDM(orb1*(orb1+1)/2 + orb2, orb1*(orb1+1)/2+orb2) += conj(coeff)*coeff;
#else
	  twoRDM(orb1*(orb1+1)/2 + orb2, orb1*(orb1+1)/2+orb2) += coeff*coeff;
#endif

#ifdef Complex
	populateSpatialRDM(orb1, orb2, orb1, orb2, s2RDM, conj(coeff)*coeff, nSpatOrbs);
#else
	populateSpatialRDM(orb1, orb2, orb1, orb2, s2RDM, coeff*coeff, nSpatOrbs);
#endif
      }
    }
    
  }

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
	  uniqueDets[k].parity(min(d0,c0), max(d0,c0), sgn);
	  //Dets[uniqueVarIndices[k][i]].parity(min(d0,c0), max(d0,c0),sgn);
	  if (!( (closed[n1] > c0 && closed[n1] > d0) || (closed[n1] < c0 && closed[n1] < d0))) sgn *=-1.;
	  if (schd.DoSpinRDM) {
	    twoRDM(a*(a+1)/2+b, I*(I+1)/2+J) += 1.0*sgn*uniqueNumerator[k]*ci[uniqueVarIndices[k][i]]/(E0-uniqueEnergy[k]);
	    twoRDM(I*(I+1)/2+J, a*(a+1)/2+b) += 1.0*sgn*uniqueNumerator[k]*ci[uniqueVarIndices[k][i]]/(E0-uniqueEnergy[k]);
	  }
	  populateSpatialRDM(a, b, I, J, s2RDM, 1.0*sgn*uniqueNumerator[k]*ci[uniqueVarIndices[k][i]]/(E0-uniqueEnergy[k]), nSpatOrbs);
	  populateSpatialRDM(I, J, a, b, s2RDM, 1.0*sgn*uniqueNumerator[k]*ci[uniqueVarIndices[k][i]]/(E0-uniqueEnergy[k]), nSpatOrbs);
	} // for n1
      }  // single
      else { // double excitation
	int d1=(uniqueOrbDiff[k][i]/norbs/norbs)%norbs, c1=(uniqueOrbDiff[k][i]/norbs/norbs/norbs)%norbs ;
	double sgn = 1.0;
	uniqueDets[k].parity(c1,c0,d1,d0,sgn);
	//Dets[uniqueVarIndices[k][i]].parity(d1,d0,c1,c0,sgn);
	int P = max(c1,c0), Q = min(c1,c0), R = max(d1,d0), S = min(d1,d0);
	if (P != c0)  sgn *= -1;
	if (Q != d0)  sgn *= -1;
	
	if (schd.DoSpinRDM) {
	  twoRDM(P*(P+1)/2+Q, R*(R+1)/2+S) += 1.0*sgn*uniqueNumerator[k]*ci[uniqueVarIndices[k][i]]/(E0-uniqueEnergy[k]);
	  twoRDM(R*(R+1)/2+S, P*(P+1)/2+Q) += 1.0*sgn*uniqueNumerator[k]*ci[uniqueVarIndices[k][i]]/(E0-uniqueEnergy[k]);
	}
	
	populateSpatialRDM(P, Q, R, S, s2RDM, 1.0*sgn*uniqueNumerator[k]*ci[uniqueVarIndices[k][i]]/(E0-uniqueEnergy[k]), nSpatOrbs);
	populateSpatialRDM(R, S, P, Q, s2RDM, 1.0*sgn*uniqueNumerator[k]*ci[uniqueVarIndices[k][i]]/(E0-uniqueEnergy[k]), nSpatOrbs);
      }// If
    } // i in variational connections to PT det k
  } // k in PT dets


  
#ifndef SERIAL
  if (schd.DoSpinRDM)
    MPI_Allreduce(MPI_IN_PLACE, &twoRDM(0,0), twoRDM.rows()*twoRDM.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &s2RDM(0,0), s2RDM.rows()*s2RDM.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  
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

void SHCIrdm::EvaluateRDM(vector<vector<int> >& connections, Determinant *Dets, int DetsSize,
			  CItype *cibra, CItype *ciket,
			  vector<vector<size_t> >& orbDifference, int nelec,
			  schedule& schd, int root, MatrixXx& twoRDM, MatrixXx& s2RDM) {

  size_t norbs = Dets[0].norbs;
  int nSpatOrbs = norbs/2;



  for (int i=0; i<DetsSize; i++) {
    if (i%commsize != commrank) continue;

    vector<int> closed(nelec, 0);
    vector<int> open(norbs-nelec,0);
    Dets[i].getOpenClosed(open, closed);

    //<Di| Gamma |Di>
    for (int n1=0; n1<nelec; n1++) {
      for (int n2=0; n2<n1; n2++) {
	int orb1 = closed[n1], orb2 = closed[n2];
	if (schd.DoSpinRDM)
	  twoRDM(orb1*(orb1+1)/2 + orb2, orb1*(orb1+1)/2+orb2) += localConj::conj(cibra[i])*ciket[i];
	populateSpatialRDM(orb1, orb2, orb1, orb2, s2RDM, localConj::conj(cibra[i])*ciket[i], nSpatOrbs);
      }
    }

    for (int j=1; j<connections[i/commsize].size(); j++) {
      //if (i == connections[i/commsize][j]) continue;
      int d0=orbDifference[i/commsize][j]%norbs, c0=(orbDifference[i/commsize][j]/norbs)%norbs ;

      if (orbDifference[i/commsize][j]/norbs/norbs == 0) { //only single excitation
	for (int n1=0;n1<nelec; n1++) {
	  double sgn = 1.0;
	  int a=max(closed[n1],c0), b=min(closed[n1],c0), I=max(closed[n1],d0), J=min(closed[n1],d0);
	  if (closed[n1] == d0) continue;
	  Dets[i].parity(min(d0,c0), max(d0,c0),sgn);
	  if (!( (closed[n1] > c0 && closed[n1] > d0) || (closed[n1] < c0 && closed[n1] < d0))) sgn *=-1.;
	  if (schd.DoSpinRDM) {
	    twoRDM(a*(a+1)/2+b, I*(I+1)/2+J) += sgn*localConj::conj(cibra[connections[i/commsize][j]])*ciket[i];
	    twoRDM(I*(I+1)/2+J, a*(a+1)/2+b) += sgn*localConj::conj(ciket[connections[i/commsize][j]])*cibra[i];
	  }

	  populateSpatialRDM(a, b, I, J, s2RDM, sgn*localConj::conj(cibra[connections[i/commsize][j]])*ciket[i], nSpatOrbs);
	  populateSpatialRDM(I, J, a, b, s2RDM, sgn*localConj::conj(ciket[connections[i/commsize][j]])*cibra[i], nSpatOrbs);

	}
      }
      else {
	int d1=(orbDifference[i/commsize][j]/norbs/norbs)%norbs, c1=(orbDifference[i/commsize][j]/norbs/norbs/norbs)%norbs ;
	double sgn = 1.0;

	Dets[i].parity(d1,d0,c1,c0,sgn);
	if (schd.DoSpinRDM) {
	  twoRDM(c1*(c1+1)/2+c0, d1*(d1+1)/2+d0) += sgn*localConj::conj(cibra[connections[i/commsize][j]])*ciket[i];
	  twoRDM(d1*(d1+1)/2+d0, c1*(c1+1)/2+c0) += sgn*localConj::conj(ciket[connections[i/commsize][j]])*cibra[i];
	}

	populateSpatialRDM(c1, c0, d1, d0, s2RDM, sgn*localConj::conj(cibra[connections[i/commsize][j]])*ciket[i], nSpatOrbs);
	populateSpatialRDM(d1, d0, c1, c0, s2RDM, sgn*localConj::conj(ciket[connections[i/commsize][j]])*cibra[i], nSpatOrbs);

      }
    }

  }

#ifndef SERIAL
  if (schd.DoSpinRDM)
    MPI_Allreduce(MPI_IN_PLACE, &twoRDM(0,0), twoRDM.rows()*twoRDM.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &s2RDM(0,0), s2RDM.rows()*s2RDM.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
}


void SHCIrdm::EvaluateOneRDM(vector<vector<int> >& connections, vector<Determinant>& Dets,
			     MatrixXx& cibra, MatrixXx& ciket,
			     vector<vector<size_t> >& orbDifference, int nelec,
			     schedule& schd, int root, MatrixXx& s1RDM) {
#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  size_t norbs = Dets[0].norbs;
  int nSpatOrbs = norbs/2;




  //#pragma omp parallel for schedule(dynamic)
  for (int i=0; i<Dets.size(); i++) {
    if (i%commsize != commrank) continue;

    vector<int> closed(nelec, 0);
    vector<int> open(norbs-nelec,0);
    Dets[i].getOpenClosed(open, closed);

    //<Di| Gamma |Di>
    for (int n1=0; n1<nelec; n1++) {
      int orb1 = closed[n1];
      s1RDM(orb1, orb1) += localConj::conj(cibra(i,0))*ciket(i,0);
    }

    for (int j=1; j<connections[i/commsize].size(); j++) {
      int d0=orbDifference[i/commsize][j]%norbs, c0=(orbDifference[i/commsize][j]/norbs)%norbs ;
      if (orbDifference[i/commsize][j]/norbs/norbs == 0) { //only single excitation
	double sgn = 1.0;
	Dets[i].parity(min(c0,d0), max(c0,d0),sgn);
	s1RDM(c0, d0)+= sgn*localConj::conj(cibra(connections[i/commsize][j],0))*ciket(i,0);
      }
    }
  }

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &s1RDM(0,0), s1RDM.rows()*s1RDM.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
}



double SHCIrdm::ComputeEnergyFromSpinRDM(int norbs, int nelec, oneInt& I1, twoInt& I2,
				       double coreE, MatrixXx& twoRDM) {

  //RDM(i,j,k,l) = a_i^\dag a_j^\dag a_l a_k
  //also i>=j and k>=l
  double energy = coreE;
  double onebody = 0.0;
  double twobody = 0.0;
  //if (commrank == 0)  cout << "Core energy= " << energy << endl;

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

  //if (commrank == 0)  cout << "One-body from 2RDM: " << onebody << endl;
  //if (commrank == 0)  cout << "Two-body from 2RDM: " << twobody << endl;

  energy += onebody + twobody;
  if (commrank == 0)  cout << "E from 2RDM: " << energy << endl;
  return energy;
}


double SHCIrdm::ComputeEnergyFromSpatialRDM(int norbs, int nelec, oneInt& I1, twoInt& I2,
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
    for (int q=0; q<norbs; q++) {
#ifdef Complex
      onebody += (I1(2*p, 2*q)*oneRDM(p,q)).real();
#else
      onebody += I1(2*p, 2*q)*oneRDM(p,q);
#endif
    }

#pragma omp parallel for reduction(+ : twobody)
  for (int p=0; p<norbs; p++)
    for (int q=0; q<norbs; q++)
      for (int r=0; r<norbs; r++)
	for (int s=0; s<norbs; s++) {
#ifdef Complex
	  twobody +=  (0.5 * twoRDM(p*norbs+q,r*norbs+s) * I2(2*p,2*r,2*q,2*s)).real(); // 2-body term
#else
	  twobody +=  0.5*twoRDM(p*norbs+q,r*norbs+s) * I2(2*p,2*r,2*q,2*s); // 2-body term
#endif
	}
  energy += onebody + twobody;
  pout << onebody<<"  "<<twobody<<endl;
  if (commrank == 0)  cout << "E from 2RDM: " << energy << endl;
  return energy;
}
