/*
Developed by Sandeep Sharma with contributions from James E. Smith and Adam A. Homes, 2017
Copyright (c) 2017, Sandeep Sharma

This file is part of DICE.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef HMULT_HEADER_H
#define HMULT_HEADER_H
#include "omp.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "communicate.h"
#include "global.h"
#include "Determinants.h"
#include <algorithm>
#include "SHCISortMpiUtils.h"
#include "SHCImakeHamiltonian.h"

using namespace Eigen;
using namespace std;
using namespace SHCISortMpiUtils;
using namespace SHCImakeHamiltonian;

std::complex<double> sumComplex(const std::complex<double>& a, const std::complex<double>& b) ;

namespace SHCISortMpiUtils{
  int ipow(int base, int exp);
};

struct Hmult2 {
  SparseHam& sparseHam;

Hmult2(SparseHam& p_sparseHam) : sparseHam(p_sparseHam) {}

  void operator()(CItype *x, CItype *y) {

#ifndef SERIAL
    boost::mpi::communicator world;
#endif
    int size = mpigetsize(), rank = mpigetrank();

    for (int batch=0; batch<sparseHam.Nbatches; batch++) {
      int offset = 0;
      if (sparseHam.diskio) {
	sparseHam.readBatch(batch);
	offset = batch*sparseHam.BatchSize;
      }

      for (int i=0; i<sparseHam.connections.size(); i++) {
	for (int j=0; j<sparseHam.connections[i].size(); j++) {
	  CItype hij = sparseHam.Helements[i][j];
	  int J = sparseHam.connections[i][j];
	  y[i*size+rank + offset] += hij*x[J];
	}
      }

      if (sparseHam.diskio)
	sparseHam.resize(0);
    }
  }

};

struct HmultDirect {
  int*          &AlphaMajorToBetaLen; 
  vector<int* > &AlphaMajorToBeta   ;
  vector<int* > &AlphaMajorToDet    ;
  int*          &BetaMajorToAlphaLen; 
  vector<int* > &BetaMajorToAlpha   ;
  vector<int* > &BetaMajorToDet     ;
  int*          &SinglesFromAlphaLen; 
  vector<int* > &SinglesFromAlpha   ;
  int*          &SinglesFromBetaLen ; 
  vector<int* > &SinglesFromBeta    ;
  Determinant *&Dets;
  int DetsSize;
  int StartIndex;
  int Norbs;
  oneInt& I1;
  twoInt& I2;
  double& coreE;
  MatrixXx& diag;

  HmultDirect(SHCImakeHamiltonian::HamHelpers2& helpers2, 
		     Determinant* &pDets,
         	     int pDetsSize,
		     int pStartIndex,
		     int pNorbs,
		     oneInt& pI1,
		     twoInt& pI2,
		     double& pcoreE,
		     MatrixXx& pDiag) : 
    AlphaMajorToBetaLen(helpers2.AlphaMajorToBetaLen),
    AlphaMajorToBeta   (helpers2.AlphaMajorToBetaSM ),
    AlphaMajorToDet    (helpers2.AlphaMajorToDetSM  ),
    BetaMajorToAlphaLen(helpers2.BetaMajorToAlphaLen),
    BetaMajorToAlpha   (helpers2.BetaMajorToAlphaSM ),
    BetaMajorToDet     (helpers2.BetaMajorToDetSM   ),
    SinglesFromAlphaLen(helpers2.SinglesFromAlphaLen),
    SinglesFromAlpha   (helpers2.SinglesFromAlphaSM ),
    SinglesFromBetaLen (helpers2.SinglesFromBetaLen ),
    SinglesFromBeta    (helpers2.SinglesFromBetaSM  ),
    Dets               (pDets               ),
    DetsSize           (pDetsSize           ),
    StartIndex         (pStartIndex         ),
    Norbs              (pNorbs              ),
    I1                 (pI1                 ),
    I2                 (pI2                 ),
    coreE              (pcoreE              ),
    diag               (pDiag               ) {};


  HmultDirect(  int*          &pAlphaMajorToBetaLen, 
		     vector<int* > &pAlphaMajorToBeta   ,
		     vector<int* > &pAlphaMajorToDet    ,
		     int*          &pBetaMajorToAlphaLen, 
		     vector<int* > &pBetaMajorToAlpha   ,
		     vector<int* > &pBetaMajorToDet     ,
		     int*          &pSinglesFromAlphaLen, 
		     vector<int* > &pSinglesFromAlpha   ,
		     int*          &pSinglesFromBetaLen , 
		     vector<int* > &pSinglesFromBeta    ,
		     Determinant* &pDets,
         	     int pDetsSize,
		     int pStartIndex,
		     int pNorbs,
		     oneInt& pI1,
		     twoInt& pI2,
		     double& pcoreE,
		     MatrixXx& pDiag) : 
    AlphaMajorToBetaLen(pAlphaMajorToBetaLen),
    AlphaMajorToBeta   (pAlphaMajorToBeta   ),
    AlphaMajorToDet    (pAlphaMajorToDet    ),
    BetaMajorToAlphaLen(pBetaMajorToAlphaLen),
    BetaMajorToAlpha   (pBetaMajorToAlpha   ),
    BetaMajorToDet     (pBetaMajorToDet     ),
    SinglesFromAlphaLen(pSinglesFromAlphaLen),
    SinglesFromAlpha   (pSinglesFromAlpha   ),
    SinglesFromBetaLen (pSinglesFromBetaLen ),
    SinglesFromBeta    (pSinglesFromBeta    ),
    Dets               (pDets               ),
    DetsSize           (pDetsSize           ),
    StartIndex         (pStartIndex         ),
    Norbs              (pNorbs              ),
    I1                 (pI1                 ),
    I2                 (pI2                 ),
    coreE              (pcoreE              ),
    diag               (pDiag               ) {};

  void operator()(CItype *x, CItype *y) {
    if (StartIndex >= DetsSize) return;
#ifndef SERIAL
    boost::mpi::communicator world;
#endif
    int nprocs = mpigetsize(), proc = mpigetrank();
    
    size_t norbs = Norbs;

    //diagonal element
    for (size_t k=StartIndex; k<DetsSize; k++) {
      if (k%(nprocs) != proc) continue;
      CItype hij = Dets[k].Energy(I1, I2, coreE);
      y[k] += hij*x[k];
    }

    
    //alpha-beta excitation
    for (int i=0; i<AlphaMajorToBeta.size(); i++) {
      
      for (int ii=0; ii<AlphaMajorToBetaLen[i]; ii++) {
	
	if (AlphaMajorToDet[i][ii]         < StartIndex ||
	    AlphaMajorToDet[i][ii]%nprocs != proc         ) 
	  continue;
	
	int Astring = i, 
	  Bstring = AlphaMajorToBeta[i][ii], 
	  DetI    = AlphaMajorToDet [i][ii];
	
	int maxBToA = BetaMajorToAlpha[Bstring][BetaMajorToAlphaLen[Bstring]-1];
	//singles from Astring
	for (int j=0; j<SinglesFromAlphaLen[Astring]; j++) {
	  int Asingle = SinglesFromAlpha[Astring][j];
	  
	  if (Asingle > maxBToA) break;
	  int index = binarySearch ( &BetaMajorToAlpha[Bstring][0] , 
				     0                             , 
				     BetaMajorToAlphaLen[Bstring]-1, 
				     Asingle                       );
	  if (index != -1 ) {
	    int DetJ = BetaMajorToDet[Bstring][index];
	    size_t orbDiff;
	    CItype hij = Hij(Dets[DetI], Dets[DetJ], I1, I2, coreE, orbDiff);
	    y[DetI] += hij*x[DetJ];
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
	  /*
	  auto itb = lower_bound(&AlphaMajorToBeta[Asingle][SearchStartIndex],
				 &AlphaMajorToBeta[Asingle][AlphaToBetaLen]  ,
				 Bsingle);

	  if (itb != &AlphaMajorToBeta[Asingle][AlphaToBetaLen] && *itb == Bsingle) {
	    SearchStartIndex = itb - &AlphaMajorToBeta[Asingle][0];
	  */
	  int index=SearchStartIndex;
	  for (; index <AlphaToBetaLen && AlphaMajorToBeta[Asingle][index] < Bsingle; index++) {}

	  SearchStartIndex = index;
	  if (index <AlphaToBetaLen && AlphaMajorToBeta[Asingle][index] == Bsingle) {

	    int DetJ = AlphaMajorToDet[Asingle][SearchStartIndex];
	    size_t orbDiff;
	    CItype hij = Hij(Dets[DetI], Dets[DetJ], I1, I2, coreE, orbDiff);
	    y[DetI] += hij*x[DetJ];
	  } //*itb == Bsingle
	} //k 0->SinglesFromBeta
      } //j singles fromAlpha


      
      //singles from Bstring
      int maxAtoB = AlphaMajorToBeta[Astring][AlphaMajorToBetaLen[Astring]-1];
      for (int j=0; j< SinglesFromBetaLen[Bstring]; j++) {
	int Bsingle =  SinglesFromBeta   [Bstring][j];

	if (Bsingle > maxAtoB) break;
	int index = binarySearch( &AlphaMajorToBeta[Astring][0] , 
				  0                             , 
				  AlphaMajorToBetaLen[Astring]-1, 
				  Bsingle                        );
	
	if (index != -1 ) {
	  int DetJ = AlphaMajorToDet[Astring][index];
	  size_t orbDiff;
	  CItype hij = Hij(Dets[DetI], Dets[DetJ], I1, I2, coreE, orbDiff);
	  y[DetI] += hij*x[DetJ];
	}
      }
      
      
      //double beta excitation
      for (int j=0; j< AlphaMajorToBetaLen[i]; j++) {
	int DetJ     = AlphaMajorToDet    [i][j];
	
	  if (Dets[DetJ].ExcitationDistance(Dets[DetI]) == 2) {
	    size_t orbDiff;
	    CItype hij = Hij(Dets[DetI], Dets[DetJ], I1, I2, coreE, orbDiff);
	    y[DetI] += hij*x[DetJ];
	  }
      }
      
      //double Alpha excitation
      for (int j=0; j < BetaMajorToAlphaLen[Bstring]; j++) {
	int DetJ      = BetaMajorToDet     [Bstring][j];
	
	  if (Dets[DetJ].ExcitationDistance(Dets[DetI]) == 2) {
	    size_t orbDiff;
	    CItype hij = Hij(Dets[DetI], Dets[DetJ], I1, I2, coreE, orbDiff);
	    y[DetI] += hij*x[DetJ];
	  }
      }
      
      }
    }
    
    
  };
};

#endif
