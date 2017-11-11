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
#include "SHCImakeHamiltonian.h"
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
#include "SHCIshm.h"
#include "LCC.h"
#include "communicate.h"

using namespace std;
using namespace Eigen;
using namespace boost;
using namespace SHCISortMpiUtils;



//=============================================================================
double LCC::doLCC(
        Determinant *Dets, CItype *ci, int DetsSize,
        double& E0, oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2HB,
        vector<int>& irrep, schedule& schd, double coreE, int nelec, int root) {
//-----------------------------------------------------------------------------
    /*!
    BM_description

    :Inputs:

        Determinant *Dets:
            The determinants of the basis
        CItype *ci:
            The reference CI coefficient c_i
        int DetsSize:
            Number of determinants in the basis
        double& E0:
            The current variational energy
        oneInt& I1:
            One-electron tensor of the Hamiltonian
        twoInt& I2:
            Two-electron tensor of the Hamiltonian
        twoIntHeatBathSHM& I2HB:
            The sorted two-electron integrals to choose the bi-excited determinants
        vector<int>& irrep:
            Irrep of the orbitals
        schedule& schd:
            The schedule
        double coreE:
            The core energy
        int nelec:
            Number of electrons
        int root:
            (unused)

    :Returns:

        double ept:
            Perturbation energy
    */
//-----------------------------------------------------------------------------
#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  int norbs = Determinant::norbs;
  int size = commsize, rank = commrank;
  vector<size_t> all_to_all(size*size,0);

  // Prepare SortedDets
  Determinant* SortedDets;
  std::vector<Determinant> SortedDetsvec;
  if (commrank == 0 ) {
    for (int i=0; i<DetsSize; i++)
      SortedDetsvec.push_back(Dets[i]);
    std::sort(SortedDetsvec.begin(), SortedDetsvec.end());
  }
#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  SHMVecFromVecs(SortedDetsvec, SortedDets, shciSortedDets, SortedDetsSegment, regionSortedDets);
  SortedDetsvec.clear();

  // Accumulate the LCC determinants
  StitchDEH uniqueDEH;
  for (int i=0; i<DetsSize; i++) {
    if ((i%size != rank)) continue;
    LCC::getDeterminantsLCC(
            Dets[i], abs(schd.epsilon2/ci[i]), ci[i], 0.0,
            I1, I2, I2HB, irrep, coreE, E0,
            *uniqueDEH.Det,
            *uniqueDEH.Num,
            *uniqueDEH.Energy,
            schd,0, nelec);
  }

  // Unique ones (via merge, etc...), and Communications
  uniqueDEH.MergeSortAndRemoveDuplicates();
  uniqueDEH.RemoveDetsPresentIn(SortedDets, DetsSize);

  for (int level = 0; level <ceil(log2(size)); level++) {
    if (rank%ipow(2, level+1) == 0 && rank + ipow(2, level) < size) {
      int getproc = rank+ipow(2,level);
      long numDets = 0;
      long oldSize = uniqueDEH.Det->size();
      long maxint = 26843540;
      MPI_Recv(&numDets, 1, MPI_DOUBLE, getproc, getproc,
               MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      long totalMemory = numDets*DetLen;

      if (totalMemory != 0) {
        uniqueDEH.Det->resize(oldSize+numDets);
        uniqueDEH.Num->resize(oldSize+numDets);
        uniqueDEH.Energy->resize(oldSize+numDets);
        for (int i=0; i<(totalMemory/maxint); i++)
          MPI_Recv(&(uniqueDEH.Det->at(oldSize).repr[0])+i*maxint,
                   maxint, MPI_DOUBLE, getproc, getproc,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(uniqueDEH.Det->at(oldSize).repr[0])+(totalMemory/maxint)*maxint,
                 totalMemory-(totalMemory/maxint)*maxint, MPI_DOUBLE,
                 getproc, getproc, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i=0; i<(numDets/maxint); i++)
          MPI_Recv(&(uniqueDEH.Num->at(oldSize))+i*maxint,
                   maxint, MPI_DOUBLE, getproc, getproc,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(uniqueDEH.Num->at(oldSize))+(numDets/maxint)*maxint,
                 numDets-(numDets/maxint)*maxint, MPI_DOUBLE,
                 getproc, getproc, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i=0; i<(numDets/maxint); i++)
          MPI_Recv(&(uniqueDEH.Energy->at(oldSize))+i*maxint,
                   maxint, MPI_DOUBLE, getproc, getproc,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(uniqueDEH.Energy->at(oldSize))+(numDets/maxint)*maxint,
                 numDets-(numDets/maxint)*maxint, MPI_DOUBLE,
                 getproc, getproc, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        uniqueDEH.MergeSortAndRemoveDuplicates();
      }
    } else if ( rank%ipow(2, level+1) == 0 && rank + ipow(2, level) >= size) {
      continue ;
    } else if ( rank%ipow(2, level) == 0) {
      int toproc = rank-ipow(2,level);
      int proc = commrank;
      long numDets = uniqueDEH.Det->size();
      long maxint = 26843540;
      long totalMemory = numDets*DetLen;
      MPI_Send(&numDets, 1, MPI_DOUBLE, toproc, proc, MPI_COMM_WORLD);

      if (totalMemory != 0) {
        for (int i=0; i<(totalMemory/maxint); i++)
          MPI_Send(&(uniqueDEH.Det->at(0).repr[0])+i*maxint,
                   maxint, MPI_DOUBLE, toproc, proc, MPI_COMM_WORLD);
        MPI_Send(&(uniqueDEH.Det->at(0).repr[0])+(totalMemory/maxint)*maxint,
                 totalMemory-(totalMemory/maxint)*maxint, MPI_DOUBLE,
                 toproc, proc, MPI_COMM_WORLD);
        for (int i=0; i<(numDets/maxint); i++)
          MPI_Send(&(uniqueDEH.Num->at(0))+i*maxint,
                   maxint, MPI_DOUBLE, toproc, proc, MPI_COMM_WORLD);
        MPI_Send(&(uniqueDEH.Num->at(0))+(numDets/maxint)*maxint,
                 numDets-(numDets/maxint)*maxint, MPI_DOUBLE,
                 toproc, proc, MPI_COMM_WORLD);
        for (int i=0; i<(numDets/maxint); i++)
          MPI_Send(&(uniqueDEH.Energy->at(0))+i*maxint,
                   maxint, MPI_DOUBLE, toproc, proc, MPI_COMM_WORLD);
        MPI_Send(&(uniqueDEH.Energy->at(0))+(numDets/maxint)*maxint,
                 numDets-(numDets/maxint)*maxint, MPI_DOUBLE,
                 toproc, proc, MPI_COMM_WORLD);
        uniqueDEH.clear();
      }
    } // rank
  } // level
  // communications end here

  // This is Vpsi0
  vector<Determinant>& Vpsi0Dets   = *uniqueDEH.Det;
  vector<CItype>&      Vpsi0       = *uniqueDEH.Num;
  boost::mpi::broadcast(world, Vpsi0Dets, 0);
  boost::mpi::broadcast(world, Vpsi0, 0);

  // Make helpers
  SHCImakeHamiltonian::HamHelpers2 helper2;
  if (commrank == 0) {
    helper2.PopulateHelpers(&Vpsi0Dets[0], Vpsi0Dets.size(), 0);
  }
  helper2.MakeSHMHelpers();

  // Make sparseHam and H
  SHCImakeHamiltonian::SparseHam sparseHam;
  if (schd.DavidsonType != DIRECT)
    sparseHam.makeFromHelper(helper2, &Vpsi0Dets[0], 0, Vpsi0Dets.size(), norbs, I1, I2, coreE, false);
  Hmult2 H(sparseHam);

  // Prepare Psi1, Vpsi and proj
  MatrixXx Psi1 = MatrixXx::Zero(Vpsi0Dets.size(), 1);
  MatrixXx Vpsi = MatrixXx::Zero(Vpsi0Dets.size(), 1);
  for (int i=0; i<Vpsi0Dets.size(); i++)
    Vpsi(i, 0) = Vpsi0[i];
  std::vector<CItype*> proj;

  // Solve (H0-E0).Psi1 = Vpsi
  double ept=LinearSolver(H, E0, Psi1, Vpsi, proj, 1.e-5, false);

  return ept;
}



//=============================================================================
void LCC::getDeterminantsLCC(
        Determinant& d, double epsilon, CItype ci1, CItype ci2,
        oneInt& int1, twoInt& int2, twoIntHeatBathSHM& I2hb,
        vector<int>& irreps, double coreE, double E0,
        std::vector<Determinant>& dets, std::vector<CItype>& numerator, std::vector<double>& energy,
        schedule& schd, int Nmc, int nelec) {
//-----------------------------------------------------------------------------
    /*!
    BM_description

    :Inputs:

        Determinant& d:
            The reference |D_i>
        double epsilon:
            The criterion for chosing new determinants (understood as epsilon/c_i)
        CItype ci1:
            The reference CI coefficient c_i
        CItype ci2:
            The reference CI coefficient c_i
        oneInt& int1:
            One-electron tensor of the Hamiltonian
        twoInt& int2:
            Two-electron tensor of the Hamiltonian
        twoIntHeatBathSHM& I2hb:
            The sorted two-electron integrals to choose the bi-excited determinants
        vector<int>& irreps:
            Irrep of the orbitals
        double coreE:
            The core energy
        double E0:
            The current variational energy
        std::vector<Determinant>& dets:
            The determinants' determinant
        std::vector<CItype>& numerator:
            The determinants' numerator
        std::vector<double>& energy:
            The determinants' energy
        schedule& schd:
            The schedule
        int Nmc:
            BM_description
        int nelec:
            Number of electrons

    */
//-----------------------------------------------------------------------------

  // initialize stuff
  int norbs   = d.norbs;
  int nclosed = nelec;
  int nopen   = norbs-nclosed;
  vector<int> closed(nelec,0);
  vector<int> open(norbs-nelec,0);
  d.getOpenClosed(open, closed);
  //d.getRepArray(detArray);
  double Energyd = d.Energy(int1, int2, coreE);

  // mono-excited determinants
  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    //CItype integral = d.Hij_1Excite(closed[i],open[a],int1,int2);
    if (closed[i]%2 != open[a]%2 || irreps[closed[i]/2] != irreps[open[a]/2]) continue;
    CItype integral = Hij_1Excite(open[a],closed[i],int1,int2, &closed[0], nclosed);

    if (fabs(integral) > epsilon ) {
      dets.push_back(d);
      Determinant& di = *dets.rbegin();
      di.setocc(open[a], true); di.setocc(closed[i],false);

      // numerator and energy
      numerator.push_back(integral*ci1);
      double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, open[a], Energyd);
      energy.push_back(E);
    }
  } // ia

  // bi-excitated determinants
  //#pragma omp parallel for schedule(dynamic)
  if (fabs(int2.maxEntry) <epsilon) return;
  // for all pairs of closed
  for (int ij=0; ij<nclosed*nclosed; ij++) {
    int i=ij/nclosed, j = ij%nclosed;
    if (i<=j) continue;
    int I = closed[i]/2, J = closed[j]/2;
    int X = max(I, J), Y = min(I, J);

    int pairIndex = X*(X+1)/2+Y;
    size_t start = closed[i]%2==closed[j]%2 ? I2hb.startingIndicesSameSpin[pairIndex]   : I2hb.startingIndicesOppositeSpin[pairIndex];
    size_t end   = closed[i]%2==closed[j]%2 ? I2hb.startingIndicesSameSpin[pairIndex+1] : I2hb.startingIndicesOppositeSpin[pairIndex+1];
    float* integrals  = closed[i]%2==closed[j]%2 ?  I2hb.sameSpinIntegrals : I2hb.oppositeSpinIntegrals;
    short* orbIndices = closed[i]%2==closed[j]%2 ?  I2hb.sameSpinPairs     : I2hb.oppositeSpinPairs;

    // for all HCI integrals
    for (size_t index=start; index<end; index++) {
      // if we are going below the criterion, break
      if (fabs(integrals[index]) <epsilon) break;

      // otherwise: generate the determinant corresponding to the current excitation
      int a = 2*orbIndices[2*index] + closed[i]%2, b = 2*orbIndices[2*index+1]+closed[j]%2;
      if (!(d.getocc(a) || d.getocc(b))) {
        dets.push_back(d);
        Determinant& di = *dets.rbegin();
        di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);

        // sgn
        double sgn = 1.0;
        di.parity(a, b, closed[i], closed[j], sgn);

        // numerator and energy
        numerator.push_back(integrals[index]*sgn*ci1);
        double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, a, j, b, Energyd);
        energy.push_back(E);

      }
    } // heatbath integrals
  } // ij
}



