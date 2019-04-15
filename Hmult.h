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
#ifndef HMULT_HEADER_H
#define HMULT_HEADER_H
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
#include <chrono>
#include "SHCISortMpiUtils.h"
#include "SHCImake4cHamiltonian.h"

using namespace Eigen;
using namespace std;
using namespace SHCISortMpiUtils;
using namespace SHCImake4cHamiltonian;

std::complex<double> sumComplex(const std::complex<double>& a, const std::complex<double>& b) ;

namespace SHCISortMpiUtils{
  int ipow(int base, int exp);
};

struct Hmult2 {
  SparseHam &sparseHam;

  Hmult2(SparseHam &p_sparseHam) : sparseHam(p_sparseHam) {}

  //=============================================================================
  void operator()(CItype *x, CItype *y) {
    //-----------------------------------------------------------------------------
    /*!
    Calculate y = H.x from the sparse Hamiltonian
    :Inputs:
        CItype *x:
            The vector to multiply by H
        CItype *y:
            The vector H.x (output)
    */
    //-----------------------------------------------------------------------------

#ifndef SERIAL
    boost::mpi::communicator world;
#endif
    int size = commsize, rank = commrank;

    int numDets = sparseHam.connections.size(),
        localDets = sparseHam.connections.size();
#ifndef SERIAL
    MPI_Allreduce(&localDets, &numDets, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

    // TODO
    auto start = std::chrono::system_clock::now();

    vector<CItype> ytemp(numDets, 0);
    int i, j;
#pragma omp declare reduction(vec_CItype_plus : std::vector<CItype> : \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<CItype>())) \
                    initializer(omp_priv = omp_orig)

#pragma omp parallel for private(i, j) reduction(vec_CItype_plus : ytemp)
    for (i = 0; i < sparseHam.connections.size(); i++) {
      for (j = 0; j < sparseHam.connections[i].size(); j++) {
        CItype hij = sparseHam.Helements[i][j];
        int J = sparseHam.connections[i][j];
        int I = i * size + rank;

	ytemp[I] += hij * x[J];

        if (J != I) ytemp[J] += std::conj(hij) * x[I];
      }
    }
    if (rank == 0) {
      auto end = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed_seconds = end - start;
      pout << "Time cost of Hmult " << elapsed_seconds.count() << '\n';
    }
    /*start = std::chrono::system_clock::now();
    ytemp = vector<CItype>(numDets, 0);
    for (int i = 0; i < sparseHam.connections.size(); i++) {
      for (int j = 0; j < sparseHam.connections[i].size(); j++) {
        CItype hij = sparseHam.Helements[i][j];
        int J = sparseHam.connections[i][j];
        int I = i * size + rank;

	ytemp[I] += hij * x[J];

        if (J != I) {
          ytemp[J] += hij * x[i * size + rank];
        }
      }
    }
      auto end = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed_seconds = end - start;
      pout << "Time cost of Hmult on single thread " << elapsed_seconds.count() << endl;*/
// Reduce result vector y (if using distributed memory)
#ifndef SERIAL
#ifndef Complex
    if (localrank == 0) {
      MPI_Reduce(MPI_IN_PLACE, &ytemp[0], numDets, MPI_DOUBLE, MPI_SUM, 0,
                 localcomm);
      for (int j = 0; j < numDets; j++) y[j] = ytemp[j];
    } else {
      MPI_Reduce(&ytemp[0], &ytemp[0], numDets, MPI_DOUBLE, MPI_SUM, 0,
                 localcomm);
    }
#else
    if (localrank == 0) {
      MPI_Reduce(MPI_IN_PLACE, &ytemp[0], 2 * numDets, MPI_DOUBLE, MPI_SUM, 0,
                 localcomm);
      for (int j = 0; j < numDets; j++) y[j] = ytemp[j];
    } else {
      MPI_Reduce(&ytemp[0], &ytemp[0], 2 * numDets, MPI_DOUBLE, MPI_SUM, 0,
                 localcomm);
    }
#endif
    MPI_Barrier(MPI_COMM_WORLD);
#else
    for (int j = 0; j < numDets; j++) y[j] = ytemp[j];
#endif
    // TODO
    if (rank == 0) {
      auto end = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed_seconds = end - start;
      pout << "Time cost of Hmult " << elapsed_seconds.count() << '\n';
    }
  }  // operator
};

/*struct HmultDirect {
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

  HmultDirect(
          SHCImakeHamiltonian::HamHelpers2& helpers2, 
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
    Dets               (pDets                       ),
    DetsSize           (pDetsSize                   ),
    StartIndex         (pStartIndex                 ),
    Norbs              (pNorbs                      ),
    I1                 (pI1                         ),
    I2                 (pI2                         ),
    coreE              (pcoreE                      ),
    diag               (pDiag                       ) {};


  HmultDirect(
          int*          &pAlphaMajorToBetaLen, 
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
    int nprocs = commsize, proc = commrank;
    
    size_t norbs = Norbs;

    //diagonal element
    for (size_t k=StartIndex; k<DetsSize; k++) {
      if (k%(nprocs) != proc) continue;
      CItype hij = Dets[k].Energy(I1, I2, coreE);
      size_t orbDiff;
      
      /*if (Determinant::Trev != 0) 
        updateHijForTReversal(hij, Dets[k], Dets[k], I1, I2, coreE, orbDiff);
      y[k] += hij*x[k];
    }
    
    //alpha-beta excitation
    for (int i=0; i<AlphaMajorToBeta.size(); i++) {
      for (int ii=0; ii<AlphaMajorToBetaLen[i]; ii++) {

        int Astring = i, 
        Bstring = AlphaMajorToBeta[i][ii], 
        DetI    = AlphaMajorToDet [i][ii];
        
        if (AlphaMajorToDet[i][ii]-1         < StartIndex ||
            (AlphaMajorToDet[i][ii]-1)%nprocs != proc       ||
            DetI < 0) continue;
        
        int maxBToA = BetaMajorToAlpha[Bstring][BetaMajorToAlphaLen[Bstring]-1];

        //singles from Astring
        for (int j=0; j<SinglesFromAlphaLen[Astring]; j++) {
          int Asingle = SinglesFromAlpha[Astring][j];
        
          if (Asingle > maxBToA) break;
          int index = binarySearch (
                  &BetaMajorToAlpha[Bstring][0] , 
                  0, 
                  BetaMajorToAlphaLen[Bstring]-1, 
                  Asingle);
          if (index != -1 ) {
            int DetJ = BetaMajorToDet[Bstring][index];
            if (abs(DetJ) == abs(DetI) ) continue;
            size_t orbDiff;
            CItype hij = Hij(Dets[abs(DetI)-1], Dets[abs(DetJ)-1], I1, I2, coreE, orbDiff);
            fixForTreversal(Dets, DetI, DetJ, I1, I2, coreE, orbDiff, hij);
            y[abs(DetI)-1] += hij*x[abs(DetJ)-1];
          }
        }

        //single Alpha and single Beta
        for (int j=0; j<SinglesFromAlphaLen[Astring]; j++) {
          int Asingle = SinglesFromAlpha[Astring][j];
          
          int SearchStartIndex = 0,
              AlphaToBetaLen = AlphaMajorToBetaLen[Asingle],
              SinglesFromBLen  = SinglesFromBetaLen[Bstring];
          int maxAToB = AlphaMajorToBeta[Asingle][AlphaMajorToBetaLen[Asingle]-1];
          for (int k=0; k<SinglesFromBLen; k++) {
            int& Bsingle = SinglesFromBeta[Bstring][k];
            
            if (SearchStartIndex >= AlphaToBetaLen) break;
            /*
            auto itb = lower_bound(
                 &AlphaMajorToBeta[Asingle][SearchStartIndex],
                 &AlphaMajorToBeta[Asingle][AlphaToBetaLen]  ,
                 Bsingle);
            if (itb != &AlphaMajorToBeta[Asingle][AlphaToBetaLen] && *itb == Bsingle)
              SearchStartIndex = itb - &AlphaMajorToBeta[Asingle][0];
            */
            /*int index=SearchStartIndex;
            for (; index <AlphaToBetaLen && AlphaMajorToBeta[Asingle][index] < Bsingle; index++) {}
            
            SearchStartIndex = index;
            if (index <AlphaToBetaLen && AlphaMajorToBeta[Asingle][index] == Bsingle) {
            
              int DetJ = AlphaMajorToDet[Asingle][SearchStartIndex];
              if (abs(DetJ) == abs(DetI) ) continue;
              size_t orbDiff;
              CItype hij = Hij(Dets[abs(DetI)-1], Dets[abs(DetJ)-1], I1, I2, coreE, orbDiff);
              fixForTreversal(Dets, DetI, DetJ, I1, I2, coreE, orbDiff, hij);
              y[abs(DetI)-1] += hij*x[abs(DetJ)-1];
            } //*itb == Bsingle
          } //k 0->SinglesFromBeta
        } //j singles fromAlpha

        //singles from Bstring
        int maxAtoB = AlphaMajorToBeta[Astring][AlphaMajorToBetaLen[Astring]-1];
        for (int j=0; j< SinglesFromBetaLen[Bstring]; j++) {
          int Bsingle =  SinglesFromBeta   [Bstring][j];
          if (Bsingle > maxAtoB) break;
          int index = binarySearch(
                  &AlphaMajorToBeta[Astring][0] , 
                  0                             , 
                  AlphaMajorToBetaLen[Astring]-1, 
                  Bsingle                       );
          
          if (index != -1 ) {
            int DetJ = AlphaMajorToDet[Astring][index];
            if (abs(DetJ) == abs(DetI) ) continue;
            size_t orbDiff;
            CItype hij = Hij(Dets[abs(DetI)-1], Dets[abs(DetJ)-1], I1, I2, coreE, orbDiff);
            fixForTreversal(Dets, DetI, DetJ, I1, I2, coreE, orbDiff, hij);
            y[abs(DetI)-1] += hij*x[abs(DetJ)-1];
          }
        }
        
        //double beta excitation
        for (int j=0; j< AlphaMajorToBetaLen[i]; j++) {
          int DetJ     = AlphaMajorToDet    [i][j];
          if (abs(DetJ) == abs(DetI) ) continue;
          Determinant dj = Dets[abs(DetJ)-1];
          if (DetJ <0) dj.flipAlphaBeta();
          if (dj.ExcitationDistance(Dets[DetI-1]) == 2) {
            size_t orbDiff;
            CItype hij = Hij(Dets[abs(DetI)-1], Dets[abs(DetJ)-1], I1, I2, coreE, orbDiff);
            fixForTreversal(Dets, DetI, DetJ, I1, I2, coreE, orbDiff, hij);
            y[abs(DetI)-1] += hij*x[abs(DetJ)-1];
          }
        }
        
        //double Alpha excitation
        for (int j=0; j < BetaMajorToAlphaLen[Bstring]; j++) {
          int DetJ      = BetaMajorToDet     [Bstring][j];
          if (abs(DetJ) == abs(DetI) ) continue;
          Determinant dj = Dets[std::abs(DetJ)-1];
          if (DetJ <0) dj.flipAlphaBeta();
          if (Dets[DetI-1].ExcitationDistance(dj) == 2 ) {
            size_t orbDiff;
            CItype hij = Hij(Dets[abs(DetI)-1], Dets[abs(DetJ)-1], I1, I2, coreE, orbDiff);
            fixForTreversal(Dets, DetI, DetJ, I1, I2, coreE, orbDiff, hij);
            y[abs(DetI)-1] += hij*x[abs(DetJ)-1];
          }
        }
      
      } // ii
    } // i
  }; // end operator
};*/

#endif
