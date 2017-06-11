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

using namespace Eigen;
using namespace std;
using namespace SHCISortMpiUtils;

std::complex<double> sumComplex(const std::complex<double>& a, const std::complex<double>& b) ;

namespace SHCISortMpiUtils{
  int ipow(int base, int exp);
};

struct Hmult2 {
  std::vector<std::vector<int> >& connections;
  std::vector<std::vector<CItype> >& Helements;

  Hmult2(std::vector<std::vector<int> >& connections_, std::vector<std::vector<CItype> >& Helements_)
  : connections(connections_), Helements(Helements_) {}

  template <typename Derived>
  void operator()(MatrixBase<Derived>& x, MatrixBase<Derived>& y) {

#ifndef SERIAL
    boost::mpi::communicator world;
#endif
    int size = mpigetsize(), rank = mpigetrank();

    int num_thrds = omp_get_max_threads();
    if (num_thrds >1) {
      std::vector<MatrixXx> yarray(num_thrds);

#pragma omp parallel
      {
	int ithrd = omp_get_thread_num();
	int nthrd = omp_get_num_threads();

	yarray[ithrd] = MatrixXx::Zero(y.rows(),1);

	for (int i=0; i<connections.size(); i++) {
	  if ((i%(nthrd * size)
	       != rank*nthrd + ithrd)) continue;
	  for (int j=0; j<connections[i].size(); j++) {
	    CItype hij = Helements[i][j];
	    int J = connections[i][j];
	    yarray[ithrd](J,0) += hij*x(i,0);
#ifdef Complex
	    if (i!= J) yarray[ithrd](i,0) += conj(hij)*x(J,0);
#else
	    if (i!= J) yarray[ithrd](i,0) += hij*x(J,0);
#endif
	  }
	}

	int start = (x.rows()/nthrd)*ithrd;
	int end = ithrd == nthrd-1 ? x.rows() : (x.rows()/nthrd)*(ithrd+1);
#pragma omp barrier
        for(int i=start; i<end; i++) {
	  for (int thrd = 1; thrd<nthrd; thrd++) {
	    yarray[0](i,0) += yarray[thrd](i,0);
	  }
	}

      }

#ifndef SERIAL
#ifndef Complex
      MPI_Reduce(&yarray[0](0,0), &y(0,0), y.rows(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      //MPI_Bcast(&(y(0,0)), y.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      //boost::mpi::all_reduce(world, &yarray[0](0,0), y.rows(), &y(0,0), plus<double>());
#else
      MPI_Reduce(&yarray[0](0,0), &y(0,0), 2*y.rows(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      //boost::mpi::all_reduce(world, &yarray[0](0,0), y.rows(), &y(0,0), sumComplex);
#endif
#endif
    }
    else {
      for (int i=rank; i<connections.size(); i+=size) {
	//if (i%size != rank) continue;
	for (int j=0; j<connections[i].size(); j++) {
	  CItype hij = Helements[i][j];
	  int J = connections[i][j];
	  y(J,0) += hij*x(i,0);

#ifdef Complex
	  if (i!= J) y(i,0) += conj(hij)*x(J,0);
#else
	  if (i!= J) y(i,0) += hij*x(J,0);
#endif
	}
      }

      CItype* startptr;
      MatrixXx ycopy;
      if (rank == 0) {ycopy = MatrixXx(y.rows(), 1); ycopy=1.*y; startptr = &ycopy(0,0);}
      else {startptr = &y(0,0);}

#ifndef SERIAL
#ifndef Complex
      MPI_Reduce(startptr, &y(0,0), y.rows(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#else
      MPI_Reduce(startptr, &y(0,0), 2*y.rows(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      //boost::mpi::all_reduce(world, &y(0,0), y.rows(), &y(0,0), sumComplex);
#endif
#endif

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

  template <typename Derived>
    void operator()(MatrixBase<Derived>& x, MatrixBase<Derived>& y) {
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
      y(k,0) += hij*x(k,0);
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
	  
	  //if (Asingle > maxBToA) break;
	  int index = binarySearch ( &BetaMajorToAlpha[Bstring][0] , 
				     0                             , 
				     BetaMajorToAlphaLen[Bstring]-1, 
				     Asingle                       );
	  if (index != -1 ) {
	    int DetJ = BetaMajorToDet[Bstring][index];
	    
	    if (DetJ < DetI) {
	      size_t orbDiff;
	      CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
	      y(DetJ, 0) += hij*x(DetI,0);
#ifdef Complex
	      y(DetI,0) += conj(hij)*x(DetJ,0);
#else
	      y(DetI,0) += hij*x(DetJ,0);
#endif
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

	    if (DetJ < DetI) {
	      size_t orbDiff;
	      CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
	      y(DetJ, 0) += hij*x(DetI,0);
#ifdef Complex
	      y(DetI,0) += conj(hij)*x(DetJ,0);
#else
	      y(DetI,0) += hij*x(DetJ,0);
#endif
	    } //DetJ <Det I
	  } //*itb == Bsingle
	} //k 0->SinglesFromBeta
      } //j singles fromAlpha


      
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
	  
	  if (DetJ < DetI) {
	    size_t orbDiff;
	    CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
	    y(DetJ, 0) += hij*x(DetI,0);
#ifdef Complex
	    y(DetI,0) += conj(hij)*x(DetJ,0);
#else
	    y(DetI,0) += hij*x(DetJ,0);
#endif
	  }
	}
      }
      
      
      //double beta excitation
      for (int j=0; j< AlphaMajorToBetaLen[i]; j++) {
	int DetJ     = AlphaMajorToDet    [i][j];
	
	if (DetJ < DetI && Dets[DetJ].ExcitationDistance(Dets[DetI]) == 2) {
	  size_t orbDiff;
	  CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
	  y(DetJ, 0) += hij*x(DetI,0);
#ifdef Complex
	  y(DetI,0) += conj(hij)*x(DetJ,0);
#else
	  y(DetI,0) += hij*x(DetJ,0);
#endif
	}
      }
      
      //double Alpha excitation
      for (int j=0; j < BetaMajorToAlphaLen[Bstring]; j++) {
	int DetJ      = BetaMajorToDet     [Bstring][j];
	
	if (DetJ < DetI && Dets[DetJ].ExcitationDistance(Dets[DetI]) == 2) {
	  size_t orbDiff;
	  CItype hij = Hij(Dets[DetJ], Dets[DetI], I1, I2, coreE, orbDiff);
	  y(DetJ, 0) += hij*x(DetI,0);
#ifdef Complex
	  y(DetI,0) += conj(hij)*x(DetJ,0);
#else
	  y(DetI,0) += hij*x(DetJ,0);
#endif
	}
      }
      
      }
    }
    
    
  };
};

#endif
