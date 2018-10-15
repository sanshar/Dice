/*
  Developed by Sandeep Sharma
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

#include <fstream>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif

#include "Determinants.h"
#include "integral.h"
#include "CPS.h"
#include "HFWalker.h"
#include "CPSSlater.h"
#include "global.h"
#include "input.h"
#include "Profile.h"
#include "workingArray.h"
#include "Slater.h"

using namespace Eigen;

CPSSlater::CPSSlater() {
  //cps, slater will read their respective default values
  ;}

void CPSSlater::initWalker(HFWalker &walk)
{
  slater.initWalker(walk);
}

void CPSSlater::initWalker(HFWalker &walk, Determinant &d)
{
  slater.initWalker(walk, d);
}

double CPSSlater::Overlap(HFWalker &walk)
{
  return cps.Overlap(walk.d) * slater.Overlap(walk);
}

double CPSSlater::getOverlapFactor(HFWalker& walk, Determinant& dcopy, bool doparity) {
  double ovlpdetcopy;
  int excitationDistance = dcopy.ExcitationDistance(walk.d);
  
  if (excitationDistance == 0)
  {
    ovlpdetcopy = 1.0;
  }
  else if (excitationDistance == 1)
  {
    int I, A;
    getDifferenceInOccupation(walk.d, dcopy, I, A);
    ovlpdetcopy = getOverlapFactor(I, A, walk, doparity);
  }
  else if (excitationDistance == 2)
  {
    int I, J, A, B;
    getDifferenceInOccupation(walk.d, dcopy, I, J, A, B);
    ovlpdetcopy = getOverlapFactor(I, J, A, B, walk, doparity);
  }
  else
  {
    cout << "higher than triple excitation not yet supported." << endl;
    exit(0);
  }
  return ovlpdetcopy;
}

double CPSSlater::getOverlapFactor(int i, int a, HFWalker& walk, bool doparity) {
  Determinant dcopy = walk.d;
  dcopy.setocc(i, false);
  dcopy.setocc(a, true);
  return cps.OverlapRatio(i/2, a/2, dcopy, walk.d) * slater.OverlapRatio(i, a, walk, doparity); 
}

double CPSSlater::getOverlapFactor(int i, int a, HFWalker& walk,
                                   BigDeterminant& dbig,
                                   BigDeterminant& dbigcopy,
                                   bool doparity) {

  dbigcopy[i] = 0; dbigcopy[a] = 1;
  double ovlpRatio = slater.OverlapRatio(i, a, walk, doparity);
  ovlpRatio *= cps.OverlapRatio(i/2, a/2, dbigcopy, dbig);
  dbigcopy[i] = 1; dbigcopy[a] = 0;

  return ovlpRatio;
}

double CPSSlater::getOverlapFactor(int I, int J, int A, int B, HFWalker& walk, bool doparity) {
  //singleexcitation
  if (J == 0 && B == 0) return getOverlapFactor(I, A, walk, doparity);
  
  Determinant dcopy = walk.d;
  dcopy.setocc(I, false);
  dcopy.setocc(J, false);
  dcopy.setocc(A, true);
  dcopy.setocc(B, true);
  return cps.OverlapRatio(I/2, J/2, A/2, B/2, dcopy, walk.d)
      * slater.OverlapRatio(I, J, A, B, walk, doparity);
}

double CPSSlater::getOverlapFactor(int I, int J, int A, int B, HFWalker& walk,
                                   BigDeterminant& dbig,
                                   BigDeterminant& dbigcopy,
                                   bool doparity) {
  //singleexcitation
  if (J == 0 && B == 0) return getOverlapFactor(I, A, walk, dbig, dbigcopy, doparity);
  
  dbigcopy[I] = 0; dbigcopy[A] = 1; dbigcopy[J] = 0; dbigcopy[B] = 1;
  double ovlpRatio = slater.OverlapRatio(I, J, A, B, walk, false);
  ovlpRatio *= cps.OverlapRatio(I/2, J/2, A/2, B/2, dbigcopy, dbig);
  dbigcopy[I] = 1; dbigcopy[A] = 0; dbigcopy[J] = 1; dbigcopy[B] = 0;

  return ovlpRatio;
}

void CPSSlater::OverlapWithGradient(HFWalker &walk,
                                    double &ovlp,
                                    VectorXd &grad)
{
  double factor = 1.0;
  cps.OverlapWithGradient(walk.d, grad, factor);

  Eigen::VectorBlock<VectorXd> gradtail = grad.tail(grad.rows()-cps.getNumVariables());
  slater.OverlapWithGradient(walk, ovlp, gradtail);
}


void CPSSlater::printVariables()
{
  cps.printVariables();
  slater.printVariables();
}

void CPSSlater::updateVariables(Eigen::VectorXd &v)
{
  cps.updateVariables(v);
  Eigen::VectorBlock<VectorXd> vtail = v.tail(v.rows()-cps.getNumVariables());
  slater.updateVariables(vtail);
}

void CPSSlater::getVariables(Eigen::VectorXd &v)
{
  if (v.rows() != getNumVariables())
    v = VectorXd::Zero(getNumVariables());

  cps.getVariables(v);
  Eigen::VectorBlock<VectorXd> vtail = v.tail(v.rows()-cps.getNumVariables());
  slater.getVariables(vtail);
}


long CPSSlater::getNumJastrowVariables()
{
  return cps.getNumVariables();
}
//factor = <psi|w> * prefactor;

long CPSSlater::getNumVariables()
{
  int norbs = Determinant::norbs;
  long numVars = 0;
  numVars += getNumJastrowVariables();
  numVars += slater.getNumVariables();

  return numVars;
}

void CPSSlater::writeWave()
{
  if (commrank == 0)
  {
    char file[5000];
    //sprintf (file, "wave.bkp" , schd.prefix[0].c_str() );
    sprintf(file, "cpsslaterwave.bkp");
    std::ofstream outfs(file, std::ios::binary);
    boost::archive::binary_oarchive save(outfs);
    save << *this;
    outfs.close();
  }
}

void CPSSlater::readWave()
{
  if (commrank == 0)
  {
    char file[5000];
    //sprintf (file, "wave.bkp" , schd.prefix[0].c_str() );
    sprintf(file, "cpsslaterwave.bkp");
    std::ifstream infs(file, std::ios::binary);
    boost::archive::binary_iarchive load(infs);
    load >> *this;
    infs.close();
  }
#ifndef SERIAL
  boost::mpi::communicator world;
  boost::mpi::broadcast(world, *this, 0);
#endif
}


//<psi_t| (H-E0) |D>

void CPSSlater::HamAndOvlp(HFWalker &walk,
                           double &ovlp, double &ham, 
			   workingArray& work, bool fillExcitations)
{
  int norbs = Determinant::norbs;

  ovlp = Overlap(walk);
  ham = walk.d.Energy(I1, I2, coreE); 

  BigDeterminant dbig(walk.d);
  BigDeterminant dbigcopy = dbig;
 
  work.setCounterToZero();
  generateAllScreenedSingleExcitation(walk.d, schd.epsilon, schd.screen,
                                      work, false);  
  generateAllScreenedDoubleExcitation(walk.d, schd.epsilon, schd.screen,
                                        work, false);  
  
  //loop over all the screened excitations
  for (int i=0; i<work.nExcitations; i++) {
    int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
    double tia = work.HijElement[i];
    
    int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
    int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;

    //double ovlpRatio = getOverlapFactor(I, J, A, B, walk, false);
    double ovlpRatio = getOverlapFactor(I, J, A, B, walk, dbig, dbigcopy, false);

    /*
    double ovlpRatio = 1.0;
    if (ex2 != 0) {
      //Determinant dcopy = walk.d;
      //dcopy.setocc(I, false); dcopy.setocc(A, true);
      //dcopy.setocc(J, false); dcopy.setocc(B, true);
      dbigcopy[I] = 0; dbigcopy[A] = 1; dbigcopy[J] = 0; dbigcopy[B] = 1;
      ovlpRatio = slater.OverlapRatio(I, J, A, B, walk, false);
      ovlpRatio *= cps.OverlapRatio(I/2, J/2, A/2, B/2, dbigcopy, dbig);
      dbigcopy[I] = 1; dbigcopy[A] = 0; dbigcopy[J] = 1; dbigcopy[B] = 0;
    }
    else {
      //Determinant dcopy = walk.d;
      dbigcopy[I] = 0; dbigcopy[A] = 1;
      //dcopy.setocc(I, false); dcopy.setocc(A, true);
      ovlpRatio = slater.OverlapRatio(I, A, walk, false);
      ovlpRatio *= cps.OverlapRatio(I/2, A/2, dbigcopy, dbig);
      dbigcopy[I] = 1; dbigcopy[A] = 0;
    }
    */
    //add contribution to the hamiltonian value
    ham += tia * ovlpRatio;

    work.ovlpRatio[i] = ovlpRatio;
  }
}

void CPSSlater::HamAndOvlpLanczos(HFWalker &walk,
                           Eigen::VectorXd &lanczosCoeffsSample, double &ovlpSample, 
			   workingArray& work, workingArray& moreWork, double &alpha)
{
  work.setCounterToZero();
  int norbs = Determinant::norbs;

  double el0 = 0., el1 = 0., ovlp0 = 0., ovlp1 = 0.;
  HamAndOvlp(walk, ovlp0, el0, work);
  std::vector<double> ovlp{0., 0., 0.};
  ovlp[0] = ovlp0;
  ovlp[1] = el0 * ovlp0;
  ovlp[2] = ovlp[0] + alpha * ovlp[1];
  lanczosCoeffsSample[0] = ovlp[0] * ovlp[0] * el0 / (ovlp[2] * ovlp[2]); 
  lanczosCoeffsSample[1] = ovlp[0] * ovlp[1] * el0 / (ovlp[2] * ovlp[2]);
  el1 = walk.d.Energy(I1, I2, coreE);

  //workingArray work1;
  //cout << "E0  " << el1 << endl;
  //loop over all the screened excitations
  for (int i=0; i<work.nExcitations; i++) {
    double tia = work.HijElement[i];
    HFWalker walkCopy = walk;
    walkCopy.updateWalker(slater, work.excitation1[i], work.excitation2[i], false);
    moreWork.setCounterToZero();
    HamAndOvlp(walkCopy, ovlp0, el0, moreWork);
    ovlp1 = el0 * ovlp0;
    //cout << walkCopy;
    el1 += tia * ovlp1 / ovlp[1];
    //cout << "ovlp0  " << ovlp0 << "  el0  " << el0 << "  ovlp1  " << ovlp1 << "  " << tia << endl;
    work.ovlpRatio[i] = (ovlp0 + alpha * ovlp1) / ovlp[2];
  }

  lanczosCoeffsSample[2] = ovlp[1] * ovlp[1] * el1 / (ovlp[2] * ovlp[2]);
  lanczosCoeffsSample[3] = ovlp[0] * ovlp[0] / (ovlp[2] * ovlp[2]);
  ovlpSample = ovlp[2];
}


void CPSSlater::derivativeOfLocalEnergy (HFWalker &walk,
                                          double &factor, VectorXd& hamRatio)
{
  //NEEDS TO BE IMPLEMENTED
}

//This is expensive and not recommended
//double CPSSlater::Overlap(Determinant &d)
//{
//  return cps.Overlap(d)  * slater.Overlap(d);
//}

