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



#ifndef Complex
//=============================================================================
void LCC::doLCC(
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
    */
//-----------------------------------------------------------------------------
#ifndef SERIAL
  boost::mpi::communicator world;
#endif
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

  // PT2  ================================================================

  cout<<"\nSecond-order PT ----------------------------------"<<endl;
  double totalpt=0;
  vector<Determinant>  Psi1Dets;
  vector<double>       Psi1Coef;
  vector<int>          Psi1nDets(8);
  int class_cor[8] = {-2, -2, -1, -2,  0, -1,  0, -1};
  int class_act[8] = { 0,  1, -1,  2, -2,  0, -1,  1};
  int class_vir[8] = { 2,  1,  2,  0,  2,  1,  1,  0};

  // Loop for classes
  for (int iclass=0; iclass<8; iclass++){
    double tA=getTime();

    // Accumulate the LCC determinants
    StitchDEH uniqueDEH; uniqueDEH.clear();
    for (int i=0; i<DetsSize; i++) {
      if ((i%size != rank)) continue;
      LCC::getDeterminantsLCC(
              Dets[i], abs(schd.epsilon2/ci[i]), ci[i], 0.0,
              I1, I2, I2HB, irrep, coreE, E0,
              *uniqueDEH.Det,
              *uniqueDEH.Num,
              *uniqueDEH.Energy,
              schd,0, nelec,
              class_cor[iclass],class_act[iclass],class_vir[iclass]);
    }

    // Unique ones (via merge, etc...)
    uniqueDEH.MergeSortAndRemoveDuplicates();
    uniqueDEH.RemoveDetsPresentIn(SortedDets, DetsSize);

#ifndef SERIAL
    // (communications) -------------------------------------------------------
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
    // (communications) -------------------------------------------------------
    #endif


    // Prepare Dets, Psi1, VPsi0 (and proj)
    vector<Determinant> Dets= *uniqueDEH.Det;
    int nDets = Dets.size();
    #ifndef SERIAL
    boost::mpi::broadcast(world, Dets, 0);
    #endif
    MatrixXx Psi1  = MatrixXx::Zero(nDets, 1);
    MatrixXx VPsi0 = MatrixXx::Zero(nDets, 1);
    for (int i=0; i<nDets; i++)
      VPsi0(i, 0) = uniqueDEH.Num->at(i);
    std::vector<CItype*> proj;

    // Make Helpers
    SHCImakeHamiltonian::HamHelpers2 helpers;
    if (commrank == 0) {
      helpers.PopulateHelpers(&Dets[0], nDets, 0);
    }
    helpers.MakeSHMHelpers();

    // Make sparseHab and Hab
    SHCImakeHamiltonian::SparseHam sparseHab;
    if (schd.DavidsonType != DIRECT)
      sparseHab.makeFromHelper(helpers,
              &Dets[0], 0, nDets,
              Determinant::norbs, I1, I2, coreE, false);
    Hmult2 Hab(sparseHab);

    // Solve (Hab-E0).Psi1 = VPsi0
    double ept=LinearSolver(Hab, E0, Psi1, VPsi0, proj, 1.e-5, false);
    vector<double> Coef(nDets);
    for (int i=0; i<nDets; i++) Coef[i]=Psi1(i,0);
    totalpt+=-ept;

    // Save for PT3
    Psi1Dets.insert(Psi1Dets.end(), Dets.begin(), Dets.end());
    Psi1Coef.insert(Psi1Coef.end(), Coef.begin(), Coef.end());
    Psi1nDets[iclass]=nDets;

    // Print out
    double tB=getTime();
    if (commrank == 0)
      cout<<"Class "<<iclass+1
          <<format(" [%3i %3i %3i] %20.9e    (%8i determinants, %7.2fsec)")
                   %(class_cor[iclass]) %(class_act[iclass]) %(class_vir[iclass])
                   %(-ept)
                   %(nDets) %(tB-tA)
          <<endl;

  } // iclass
  cout<<"Total PT              "<<format("%20.9e") %(totalpt)<<endl;

  // PT3  ================================================================

  cout<<"\nThird-order PT -----------------------------------"<<endl;
  double totalpt3;
  int nDets = Psi1Dets.size();
  double tA=getTime();

  // Make Helpers
  SHCImakeHamiltonian::HamHelpers2 helpers;
  if (commrank == 0) {
    helpers.PopulateHelpers(&Psi1Dets[0], nDets, 0);
  }
  helpers.MakeSHMHelpers();

  // Make sparseHab and Hab
  SHCImakeHamiltonian::SparseHam sparseHab;
  if (schd.DavidsonType != DIRECT)
    sparseHab.makeFromHelper(helpers,
            &Psi1Dets[0], 0, nDets,
            Determinant::norbs, I1, I2, coreE, false);
  Hmult2 Hab(sparseHab);

  // Scenario1: show the different contributions
  cout<<"Dets/class:";
  for (int iclass=0; iclass<8; iclass++)
    cout<<format("%8i") %(Psi1nDets[iclass]);
  cout<<endl;

  int istart=0;
  int istop =Psi1nDets[0];
  for (int iclass=0; iclass<8; iclass++){
    cout<<format("Class: %5i %8i %8i ==============================") %(iclass+1) %(istart) %(istop)<<endl;

    MatrixXx CoefA = MatrixXx::Zero(nDets, 1);
    for (int i=istart; i<istop; i++)
      CoefA(i,0)=Psi1Coef[i];

    MatrixXx HPsi1Class = MatrixXx::Zero(nDets, 1);
    Hab(&CoefA(0,0),&HPsi1Class(0,0));

    int jstart=0;
    int jstop =Psi1nDets[0];
    for (int jclass=0; jclass<8; jclass++){
      if (jclass>iclass){
        cout<<format("  Class: %3i %8i %8i") %(jclass+1) %(jstart) %(jstop);

        MatrixXx CoefB = MatrixXx::Zero(nDets, 1);
        for (int i=jstart; i<jstop; i++)
          CoefB(i,0)=Psi1Coef[i];
    
        double dotProduct = 0.0;
        for (int i=0; i<nDets; i++)
          dotProduct += 2.0*CoefB(i,0)*HPsi1Class(i,0);
        totalpt3+=dotProduct;
        cout<<format("  Contrib: %20.9e") %(dotProduct)<<endl;
      }

      jstart+=Psi1nDets[jclass];
      jstop +=Psi1nDets[jclass+1];
    } // jclass

    istart+=Psi1nDets[iclass];
    istop +=Psi1nDets[iclass+1];
  } // iclass

  double tB=getTime();
  cout<<"\nTotal PT3             "<<format("%20.9e  (%7.2fsec)") %(totalpt3) %(tB-tA)<<endl;


  // Scenario2: all-in-one
  {
  double tA=getTime();

  // Make sparseHab and Hab
  SHCImakeHamiltonian::SparseHam sparseHab;
  if (schd.DavidsonType != DIRECT)
    sparseHab.makeFromHelper(helpers,
            &Psi1Dets[0], 0, nDets,
            Determinant::norbs, I1, I2, coreE, false);

  // Remove H0
  int start=0;
  int stop =Psi1nDets[0];
  for (int iclass=0; iclass<8; iclass++){
    for (int i=start; i<stop; i++)
      for (int j=0; j<sparseHab.connections[i].size(); j++){
        if (sparseHab.connections[i][j]>=start && sparseHab.connections[i][j]<stop)
          sparseHab.Helements[i][j]=0.0;
    }
    start+=Psi1nDets[iclass];
    stop+=Psi1nDets[iclass+1];
  };
  Hmult2 Hab(sparseHab);

  MatrixXx CoefA = MatrixXx::Zero(nDets, 1);
  for (int i=0; i<nDets; i++)
    CoefA(i,0)=Psi1Coef[i];

  MatrixXx HPsi1Class = MatrixXx::Zero(nDets, 1);
  Hab(&CoefA(0,0),&HPsi1Class(0,0));

  double totalpt3 = 0.0;
  for (int i=0; i<nDets; i++)
    totalpt3 += CoefA(i,0)*HPsi1Class(i,0);

  double tB=getTime();
  cout<<"\nTotal PT3             "<<format("%20.9e  (%7.2fsec)") %(totalpt3) %(tB-tA)<<endl;
  }

}



//=============================================================================
void LCC::getDeterminantsLCC(
        Determinant& d, double epsilon, CItype ci1, CItype ci2,
        oneInt& int1, twoInt& int2, twoIntHeatBathSHM& I2hb,
        vector<int>& irreps, double coreE, double E0,
        std::vector<Determinant>& dets, std::vector<CItype>& numerator, std::vector<double>& energy,
        schedule& schd, int Nmc, int nelec, 
        int class_cor, int class_act, int class_vir) {
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
  int nclosed = nelec;
  int nopen   = d.norbs-nclosed;
  vector<int> closed(nclosed,0);
  vector<int> open(nopen,0);
  d.getOpenClosed(open, closed);
  //d.getRepArray(detArray);
  double Energyd = d.Energy(int1, int2, coreE);
  int d_cor=0, d_act=0, d_vir=0;

  // mono-excited determinants
  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    LCC::get_landscape(closed[i],open[a],&d_cor,&d_act,&d_vir, schd);
    if (d_cor!=class_cor || d_act!=class_act || d_vir!=class_vir) {
      //cout<<format("BM i: %3i %3i a: %3i %3i") %(i) %(closed[i]) %(a) %(open[a]);
      //    <<format(" landscape: %3i %3i %3i") %(d_cor) %(d_act) %(d_vir);
      //    <<" pass"<<endl;
      continue;
    }else{
      //cout<<format("BM i: %3i %3i a: %3i %3i") %(i) %(closed[i]) %(a) %(open[a]);
      //    <<format(" landscape: %3i %3i %3i") %(d_cor) %(d_act) %(d_vir);
      //    <<" take"<<endl;
    }
    //CItype integral = d.Hij_1Excite(closed[i],open[a],int1,int2);
    if (closed[i]%2 != open[a]%2 || irreps[closed[i]/2] != irreps[open[a]/2]) continue;
    CItype integral = Hij_1Excite(open[a],closed[i],int1,int2, &closed[0], nclosed);

    if (fabs(integral) > epsilon ) {
      dets.push_back(d);
      Determinant& di = *dets.rbegin();
      di.setocc(open[a], true); di.setocc(closed[i],false);
      //cout << "BM |D_a> "<<di<<" "<<format("%3i %3i %3i") %(d_cor) %(d_act) %(d_vir)<<endl;

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
      LCC::get_landscape(closed[i],closed[j],a,b,&d_cor,&d_act,&d_vir, schd);
      if (d_cor!=class_cor || d_act!=class_act || d_vir!=class_vir) {
        //cout<<format("BM i: %3i %3i j: %3i %3i a: %3i b: %3i") %(i) %(closed[i]) %(j) %(closed[j]) %(a) %(b);
        //    <<format(" landscape: %3i %3i %3i") %(d_cor) %(d_act) %(d_vir);
        //    <<" pass"<<endl;
        continue;
      }else{
        //cout<<format("BM i: %3i %3i j: %3i %3i a: %3i b: %3i") %(i) %(closed[i]) %(j) %(closed[j]) %(a) %(b);
        //    <<format(" landscape: %3i %3i %3i") %(d_cor) %(d_act) %(d_vir);
        //cout<<" take"<<endl;
      }
      if (!(d.getocc(a) || d.getocc(b))) {
        dets.push_back(d);
        Determinant& di = *dets.rbegin();
        di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);
        //cout << "BM |D_a> "<<di<<" "<<format("%3i %3i %3i") %(d_cor) %(d_act) %(d_vir)<<endl;

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


void LCC::get_landscape(
        int i,int a,
        int* d_cor,int* d_act,int* d_vir,
        schedule schd){
  *d_cor=0; *d_act=0; *d_vir=0;
  if      (i< 2*schd.ncore && a< 2*(schd.ncore+schd.nact)) {
   *d_cor=-1;
   *d_act=+1;
  }else if(i< 2*schd.ncore && a>=2*(schd.ncore+schd.nact)) {
   *d_cor=-1;
   *d_vir=+1;
  }else if(i>=2*schd.ncore && a< 2*(schd.ncore+schd.nact)) {
   //nothing
  }else if(i>=2*schd.ncore && a>=2*(schd.ncore+schd.nact)) {
   *d_act=-1;
   *d_vir=+1;
  }else{
   cout<<"BM: what?? "<<i<<" "<<a<<" "<<schd.ncore<<" "<<schd.nact<<endl;
  }
}



void LCC::get_landscape(
        int i,int j,int a,int b,
        int* d_cor,int* d_act,int* d_vir,
        schedule schd){
  *d_cor=0; *d_act=0; *d_vir=0;
  // a act and b act
  if       (i< 2*schd.ncore              && 
            j< 2*schd.ncore              && 
            a< 2*(schd.ncore+schd.nact)  &&
            b< 2*(schd.ncore+schd.nact)) {
    *d_cor=-2;
    *d_act=+2;
  }else if (i>=2*schd.ncore              && 
            j< 2*schd.ncore              && 
            a< 2*(schd.ncore+schd.nact)  &&
            b< 2*(schd.ncore+schd.nact)) {
    *d_cor=-1;
    *d_act=+1;
  }else if (i>=2*schd.ncore              && 
            j>=2*schd.ncore              && 
            a< 2*(schd.ncore+schd.nact)  &&
            b< 2*(schd.ncore+schd.nact)) {
    //nothing

  // a act and b vir
  }else if (i< 2*schd.ncore              && 
            j< 2*schd.ncore              && 
            a< 2*(schd.ncore+schd.nact)  &&
            b>=2*(schd.ncore+schd.nact)) {
    *d_cor=-2;
    *d_act=+1;
    *d_vir=+1;
  }else if (i>=2*schd.ncore              && 
            j< 2*schd.ncore              && 
            a< 2*(schd.ncore+schd.nact)  &&
            b>=2*(schd.ncore+schd.nact)) {
    *d_cor=-1;
    *d_vir=+1;
  }else if (i>=2*schd.ncore              && 
            j>=2*schd.ncore              && 
            a< 2*(schd.ncore+schd.nact)  &&
            b>=2*(schd.ncore+schd.nact)) {
    *d_act=-1;
    *d_vir=+1;

  // a vir and b act
  }else if (i< 2*schd.ncore              && 
            j< 2*schd.ncore              && 
            a>=2*(schd.ncore+schd.nact)  &&
            b< 2*(schd.ncore+schd.nact)) {
    *d_cor=-2;
    *d_act=+1;
    *d_vir=+1;
  }else if (i>=2*schd.ncore              && 
            j< 2*schd.ncore              && 
            a>=2*(schd.ncore+schd.nact)  &&
            b< 2*(schd.ncore+schd.nact)) {
    *d_cor=-1;
    *d_vir=+1;
  }else if (i>=2*schd.ncore              && 
            j>=2*schd.ncore              && 
            a>=2*(schd.ncore+schd.nact)  &&
            b< 2*(schd.ncore+schd.nact)) {
    *d_act=-1;
    *d_vir=+1;

  // a vir and b vir
  }else if (i< 2*schd.ncore              && 
            j< 2*schd.ncore              && 
            a>=2*(schd.ncore+schd.nact)  &&
            b>=2*(schd.ncore+schd.nact)) {
    *d_cor=-2;
    *d_vir=+2;
  }else if (i>=2*schd.ncore              && 
            j< 2*schd.ncore              && 
            a>=2*(schd.ncore+schd.nact)  &&
            b>=2*(schd.ncore+schd.nact)) {
    *d_cor=-1;
    *d_act=-1;
    *d_vir=+2;
  }else if (i>=2*schd.ncore              && 
            j>=2*schd.ncore              && 
            a>=2*(schd.ncore+schd.nact)  &&
            b>=2*(schd.ncore+schd.nact)) {
    *d_act=-2;
    *d_vir=+2;
  }else{
   cout<<"BM: what?? "<<i<<" "<<j<<" "<<a<<" "<<b<<" "<<schd.ncore<<" "<<schd.nact<<endl;
  }
}
#endif
