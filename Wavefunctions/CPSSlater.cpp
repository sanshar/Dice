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

#include "integral.h"
#include "CPS.h"
#include "HFWalker.h"
#include "CPSSlater.h"
#include "global.h"
#include "input.h"
#include "Profile.h"
#include "workingArray.h"
#include "Slater.h"
#include <fstream>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif

using namespace Eigen;

CPSSlater::CPSSlater() {
  //cps, slater will read their respective default values
  ;}

void CPSSlater::initWalker(HFWalker& walk) {

  slater.initWalker(walk);
}

void CPSSlater::initWalker(HFWalker& walk, Determinant& d) {
  slater.initWalker(walk, d);
}


//This is expensive and not recommended
double CPSSlater::Overlap(Determinant &d)
{
  return cps.Overlap(d)  * slater.Overlap(d);
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
  work.setCounterToZero();
  int norbs = Determinant::norbs;

  ovlp = Overlap(walk);
  ham = walk.d.Energy(I1, I2, coreE); 


  generateAllScreenedSingleExcitation(walk.d, schd.screen, schd.epsilon,
                                      work, false);  
  generateAllScreenedDoubleExcitation(walk.d, schd.screen, schd.epsilon,
                                      work, false);  

  //loop over all the screened excitations
  for (int i=0; i<work.nExcitations; i++) {
    int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
    double tia = work.HijElement[i];
    
    int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
    int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;

    //calculate the ovlpRatio
    double ovlpRatio = getOverlapFactor(I, J, A, B, walk, false);

    //add contribution to the hamiltonian value
    ham += tia * ovlpRatio;

    work.ovlpRatio[i] = ovlpRatio;
  }
}



void CPSSlater::derivativeOfLocalEnergy (HFWalker &walk,
                                          double &factor, VectorXd& hamRatio)
{
  //NEEDS TO BE IMPLEMENTED
}


