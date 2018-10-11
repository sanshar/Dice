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
#ifndef CIWavefunction_HEADER_H
#define CIWavefunction_HEADER_H
#include <vector>
#include <set>
#include "Determinants.h"
#include "workingArray.h"
#include "excitationOperators.h"
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <utility>

using namespace std;

class oneInt;
class twoInt;
class twoIntHeatBathSHM;
class CPSSlaterWalker;

class Slater;


/**
 * This is the wavefunction, that extends a given wavefunction by doing
 * a CI expansion on it
 * |Psi> = \sum_i d_i O_i ||phi>
 * where d_i are the coefficients and O_i are the list of operators
 * these operators O_i can be of type Operators or SpinFreeOperators
 */
template <typename Wfn, typename Walker, typename OpType>
  class CIWavefunction
{
 private:
  friend class boost::serialization::access;
  template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
      ar  & wave
	& oplist
	& ciCoeffs;
    }

 public:
  Wfn wave;
  std::vector<OpType> oplist;
  std::vector<double> ciCoeffs;

  CIWavefunction() {
    wave.readWave();
    oplist.resize(1);
    ciCoeffs.resize(1, 1.0);
  }

 CIWavefunction(Wfn &w1, std::vector<OpType> &pop) : wave(w1), oplist(pop)
  {
    ciCoeffs.resize(oplist.size(), 0.0);
    ciCoeffs[0] = 1.0;
  }

  Slater& getRef() { return wave.getRef(); }

  void appendSinglesToOpList()
  {
    OpType::populateSinglesToOpList(oplist);
    ciCoeffs.resize(oplist.size(), 0.0);
  }

  void appendScreenedDoublesToOpList(double screen)
  {
    OpType::populateScreenedDoublesToOpList(oplist, screen);
    ciCoeffs.resize(oplist.size(), 0.0);
  }

  void getVariables(VectorXd& vars) {
    if (vars.rows() != getNumVariables())
      vars = VectorXd::Zero(getNumVariables());
    for (int i=0; i<ciCoeffs.size(); i++)
      vars[i] = ciCoeffs[i];
  }

  void printVariables() {
    for (int i=0;i<oplist.size(); i++)
      cout << oplist[i]<<"  "<<ciCoeffs[i]<<endl;
  }

  void updateVariables(VectorXd& vars) {
    for (int i=0; i<vars.rows(); i++)
      ciCoeffs[i] = vars[i];
  }

  long getNumVariables() {
    return ciCoeffs.size();
  }

  double getOverlapFactor(int I, int J, int A, int B, Walker& walk, bool doparity=false) {
    int norbs = Determinant::norbs;
    if (J == 0 && B == 0) {
      Walker walkcopy = walk;
      walkcopy.exciteWalker(wave.getRef(), I*2*norbs+A, 0, norbs);
      return Overlap(walkcopy)/Overlap(walk);
    }
    else {
      Walker walkcopy = walk;
      walkcopy.exciteWalker(wave.getRef(), I*2*norbs+A, J*2*norbs+B, norbs);
      return Overlap(walkcopy)/Overlap(walk);
    }
  }
  
  double calculateOverlapWithUnderlyingWave(Walker &walk, Determinant &dcopy)
  {
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
	ovlpdetcopy = wave.getOverlapFactor(I, A, walk, false);
      }
    else if (excitationDistance == 2)
      {
	int I, J, A, B;
	getDifferenceInOccupation(walk.d, dcopy, I, J, A, B);
	bool doparity = false;

	//cout << I<<"  "<<J<<"  "<<A<<"  "<<B<<endl;
	ovlpdetcopy = wave.getOverlapFactor(I, J, A, B, walk, doparity);
      }
    else
      {
	cout << "higher than triple excitation not yet supported." << endl;
	exit(0);
      }
    return ovlpdetcopy;
  }

  double Overlap(Walker& walk) {
    double totalovlp = 0.0;
    double ovlp0 = wave.Overlap(walk);

    for (int i = 0; i < oplist.size(); i++)
      {
	for (int j=0; j<oplist[i].nops; j++) {
	  Determinant dcopy = walk.d;
	  bool valid = oplist[i].apply(dcopy, j);

	  if (valid) {
	    double ovlpdetcopy = calculateOverlapWithUnderlyingWave(walk, dcopy);
	    totalovlp += ciCoeffs[i] * ovlpdetcopy * ovlp0;
	  }
	}
      }
    return totalovlp;
  }

  double OverlapWithGradient(Walker &walk,
			     double &factor,
			     Eigen::VectorXd &grad)
  {
    VectorXd gradcopy = grad;
    gradcopy.setZero();

    double ovlp0 = wave.Overlap(walk);

    for (int i = 0; i < oplist.size(); i++)
      {
	for (int j=0; j<oplist[i].nops; j++) {
	  Determinant dcopy = walk.d;
	  
	  bool valid = oplist[i].apply(dcopy, j);

	  if (valid) {
	    double ovlpdetcopy = calculateOverlapWithUnderlyingWave(walk, dcopy);
	    gradcopy[i] += ovlpdetcopy;
	  }
	}
      }

    double totalOvlp = 0.0;
    for (int i=0; i<grad.rows(); i++) {
      totalOvlp += ciCoeffs[i] * gradcopy[i];
    }
    for (int i=0; i<grad.rows(); i++) {
      grad[i] += gradcopy[i]/totalOvlp;
    }
    return totalOvlp;
  }

  
  void HamAndOvlp(Walker &walk,
		  double &ovlp, double &ham,
		  workingArray& work, bool fillExcitations = true)
  {
    work.setCounterToZero();

    double TINY = schd.screen;
    double THRESH = schd.epsilon;

    Determinant &d = walk.d;

    int norbs = Determinant::norbs;
    vector<int> closed;
    vector<int> open;
    d.getOpenClosed(open, closed);

    //noexcitation
    {
      double E0 = d.Energy(I1, I2, coreE);
      ovlp = Overlap(walk);
      ham = E0;
    }

    //Single alpha-beta excitation
    {
      double time = getTime();
      for (int i = 0; i < closed.size(); i++)
	{
	  for (int a = 0; a < open.size(); a++)
	    {
	      if (closed[i] % 2 == open[a] % 2 && abs(I2hb.Singles(closed[i], open[a])) > THRESH)
		{
		  prof.SinglesCount++;

		  int I = closed[i] / 2, A = open[a] / 2;
		  double tia = 0;
		  Determinant dcopy = d;
		  bool Alpha = closed[i] % 2 == 0 ? true : false;

		  bool doparity = true;
		  if (schd.Hamiltonian == HUBBARD)
		    {
		      tia = I1(2 * A, 2 * I);
		      double sgn = 1.0;
		      if (Alpha)
			sgn *= d.parityA(A, I);
		      else
			sgn *= d.parityB(A, I);
		      tia *= sgn;
		    }
		  else
		    {
		      tia = I1(2 * A, 2 * I);
		      int X = max(I, A), Y = min(I, A);
		      int pairIndex = X * (X + 1) / 2 + Y;
		      size_t start = I2hb.startingIndicesSingleIntegrals[pairIndex];
		      size_t end = I2hb.startingIndicesSingleIntegrals[pairIndex + 1];
		      float *integrals = I2hb.singleIntegrals;
		      short *orbIndices = I2hb.singleIntegralsPairs;
		      for (size_t index = start; index < end; index++)
			{
			  if (fabs(integrals[index]) < TINY)
			    break;
			  int j = orbIndices[2 * index];
			  if (closed[i] % 2 == 1 && j % 2 == 1)
			    j--;
			  else if (closed[i] % 2 == 1 && j % 2 == 0)
			    j++;

			  if (d.getocc(j))
			    {
			      tia += integrals[index];
			    }
			  //cout << tia<<"  "<<a<<"  "<<integrals[index]<<endl;
			}
		      double sgn = 1.0;
		      if (Alpha)
			sgn *= d.parityA(A, I);
		      else
			sgn *= d.parityB(A, I);
		      tia *= sgn;
		    }

		  double localham = 0.0;
		  if (abs(tia) > THRESH)
		    {
		      Walker walkcopy = walk;
		      walkcopy.exciteWalker(wave.getRef(), closed[i]*2*norbs+open[a], 0, norbs);
		      double ovlpdetcopy = Overlap(walkcopy);
		      ham += ovlpdetcopy * tia / ovlp;

		      if (fillExcitations) {
                        //cout << closed[i]/2<<"  "<<open[a]/2<<"  0  0  "<<ovlpdetcopy<<"  "<<tia<<endl;
			work.appendValue(ovlpdetcopy/ovlp, closed[i] * 2 * norbs + open[a],
					 0, tia);
                      }
                    }
                }
            }
        }
      prof.SinglesTime += getTime() - time;
    }

    if (schd.Hamiltonian == HUBBARD)
      return;

    //Double excitation
    {
      double time = getTime();

      int nclosed = closed.size();
      for (int ij = 0; ij < nclosed * nclosed; ij++)
	{
	  int i = ij / nclosed, j = ij % nclosed;
	  if (i <= j)
	    continue;
	  int I = closed[i] / 2, J = closed[j] / 2;
	  int X = max(I, J), Y = min(I, J);

	  int pairIndex = X * (X + 1) / 2 + Y;
	  size_t start = closed[i] % 2 == closed[j] % 2 ? I2hb.startingIndicesSameSpin[pairIndex] : I2hb.startingIndicesOppositeSpin[pairIndex];
	  size_t end = closed[i] % 2 == closed[j] % 2 ? I2hb.startingIndicesSameSpin[pairIndex + 1] : I2hb.startingIndicesOppositeSpin[pairIndex + 1];
	  float *integrals = closed[i] % 2 == closed[j] % 2 ? I2hb.sameSpinIntegrals : I2hb.oppositeSpinIntegrals;
	  short *orbIndices = closed[i] % 2 == closed[j] % 2 ? I2hb.sameSpinPairs : I2hb.oppositeSpinPairs;

	  // for all HCI integrals
	  for (size_t index = start; index < end; index++)
	    {
	      // if we are going below the criterion, break
	      if (fabs(integrals[index]) < THRESH)
		break;

	      // otherwise: generate the determinant corresponding to the current excitation
	      int a = 2 * orbIndices[2 * index] + closed[i] % 2, b = 2 * orbIndices[2 * index + 1] + closed[j] % 2;
	      if (!(d.getocc(a) || d.getocc(b)))
		{
		  int A = a / 2, B = b / 2;
		  double tiajb = integrals[index];

		  Walker walkcopy = walk;
		  walkcopy.exciteWalker(wave.getRef(), closed[i] * 2 * norbs + a, closed[j] * 2 * norbs + b, norbs);
		  double ovlpdetcopy = Overlap(walkcopy);

		  double parity = 1.0;
		  if (closed[i] % 2 == closed[j] % 2 && closed[i] % 2 == 0)
		    parity = walk.d.parityAA(I, J, A, B); 
		  else if (closed[i] % 2 == closed[j] % 2 && closed[i] % 2 == 1)
		    parity = walk.d.parityBB(I, J, A, B); 
		  else if (closed[i] % 2 != closed[j] % 2 && closed[i] % 2 == 0)
		    {parity = walk.d.parityA(A, I) * walk.d.parityB(B, J);}
		  else
                    {parity = walk.d.parityB(A, I) * walk.d.parityA(B, J);}
            
		  ham += ovlpdetcopy * tiajb * parity / ovlp;

		  if (fillExcitations) {
                    cout << closed[i]/2<<"  "<<a/2<<"  "<<closed[j]/2<<"  "<<b/2<<"  "<<ovlpdetcopy<<"  "<<tiajb<<endl;
		    work.appendValue(ovlpdetcopy/ovlp, closed[i] * 2 * norbs + a,
				     closed[j] * 2 * norbs + b , tiajb);
                  }
                }
	    }
	}
      prof.DoubleTime += getTime() - time;
    }
  }

  void initWalker(Walker& walk)
  {
    wave.initWalker(walk);
  }
  
  void initWalker(Walker& walk, Determinant& d)
  {
    wave.initWalker(walk, d);
  }

  void writeWave()
  {
    if (commrank == 0)
      {
	char file[5000];
	//sprintf (file, "wave.bkp" , schd.prefix[0].c_str() );
	sprintf(file, "ciwave.bkp");
	std::ofstream outfs(file, std::ios::binary);
	boost::archive::binary_oarchive save(outfs);
	save << *this;
	outfs.close();
      }
  }

  void readWave()
  {
    if (commrank == 0)
      {
	char file[5000];
	//sprintf (file, "wave.bkp" , schd.prefix[0].c_str() );
	sprintf(file, "ciwave.bkp");
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
};

//template <typename Wfn, typename Walker>
//  class Lanczos : public CIWavefunction<Wfn, Walker, NormalOperator>
//{
// private:
//  friend class boost::serialization::access;
//  template <class Archive>
//    void serialize(Archive &ar, const unsigned int version)
//    {
//      ar  & boost::serialization::base_object<CIWavefunction<Wfn, Walker, NormalOperator>>(*this)
//	& alpha;
//    }
//
// public:
//  double alpha;
//
//  Lanczos()
//  {
//    CIWavefunction<Wfn, Walker, NormalOperator>();
//    this->ciCoeffs.push_back(coreE);
//    this->oplist.push_back(NormalOperator());
//    alpha = 0.1;
//  }
//
//  Lanczos(Wfn &w1, std::vector<NormalOperator> &pop, double alpha0) : alpha(alpha0)
//  {
//    CIWavefunction<Wfn, Walker, NormalOperator>(w1, pop);
//    this->ciCoeffs.push_back(coreE);
//    this->oplist.push_back(NormalOperator());
//  }
//
//  void appendSinglesToOpList()
//  {
//    Operator::populateSinglesToOpList(this->oplist, this->ciCoeffs);
//    //ciCoeffs.resize(oplist.size(), 0.0);
//  }
//
//  void appendScreenedDoublesToOpList(double screen)
//  {
//    Operator::populateScreenedDoublesToOpList(this->oplist, this->ciCoeffs, screen);
//    //ciCoeffs.resize(oplist.size(), 0.0);
//  }
//
//  //void initWalker(Walker& walk) {
//  //  this->wave.initWalker(walk);
//  //}  
//
//  //void initWalker(Walker& walk, Determinant& d) {
//  //  this->wave.initWalker(walk, d);
//  //}  
//
//  void getVariables(VectorXd& vars) 
//  {
//    if (vars.rows() != getNumVariables()) 
//      vars = VectorXd::Zero(getNumVariables());
//    vars[0] = alpha;
//  }
//
//  void printVariables() 
//  {
//    cout << "alpha  " << alpha << endl;
//    cout << "ciCoeffs\n";
//    for (int i=0; i<this->oplist.size(); i++)
//      cout << this->oplist[i] << "  " << this->ciCoeffs[i] << endl;
//    cout << endl;
//  }
//
//  void updateVariables(VectorXd& vars) 
//  {
//    alpha = vars[0];
//  }
//
//  long getNumVariables() 
//  {
//    return 1;
//  }
// 
//  double Overlap1(Walker &walk, double &ovlp0)
//  {
//    double totalovlp = 0.0;
//    //cout << "totalovlp   " << totalovlp << endl;
//
//    for (int i = 1; i < this->oplist.size(); i++) {
//      for (int j = 0; j < this->oplist[i].nops; j++) {
//	    Determinant dcopy = walk.d;
//	    bool valid = this-> oplist[i].apply(dcopy, j);
//
//	    if (valid) {
//	      //cout << i<<" -  "<<oplist[i].ops[j]<<"  "<<walk.d<<"  "<<dcopy<<endl;
//	      double ovlpdetcopy = calculateOverlapWithUnderlyingWave(walk, dcopy);
//	      cout << i << " -  " << this->oplist[i] << "  " << dcopy << endl;
//          cout << this->ciCoeffs[i] << "   " << ovlpdetcopy << "  " << ovlp0 << endl;
//	      totalovlp += this->ciCoeffs[i] * ovlpdetcopy * ovlp0;
//	    }
//      }
//    }
//    return totalovlp;
//  }
//   
//
//  double Overlap(Walker& walk) 
//  {
//    double ovlp0 = this->wave.Overlap(walk);
//    double ovlp1 = Overlap1(walk, ovlp0);
//    double totalOvlp = ovlp0 + alpha * ovlp1;
//    return totalovlp;
//  }
//
//  double OverlapWithGradient(Walker &walk,
//			     double &factor,
//			     Eigen::VectorXd &grad)
//  {
//
//    double num = 0.; 
//    for (int i=1; i<this->oplist.size(); i++)
//      {
//	for (int j=0; j<this->oplist[i].nops; j++) {
//	  Determinant dcopy = walk.d;
//	  
//	  bool valid = this->oplist[i].apply(dcopy, j);
//
//	  if (valid) {
//	    double ovlpdetcopy = calculateOverlapWithUnderlyingWave(walk, dcopy);
//	    num += this->ciCoeffs[i] * ovlpdetcopy;
//	  }
//	}
//      }
//
//    grad[0] += num/(alpha * num + 1);
//    return (alpha * num + 1);
//  }
//  
//  void HamAndOvlp(Walker &walk,
//    	  std::vector<double> &ovlp, std::vector<double> &ham,
//    	  workingArray& work, bool fillExcitations = true)
//  {
//    work.setCounterToZero();
//
//    double TINY = schd.screen;
//    double THRESH = schd.epsilon;
//
//    Determinant &d = walk.d;
//
//    int norbs = Determinant::norbs;
//    double el0 = 0., el1 = 0.;
//    
//    double E0 = d.Energy(I1, I2, coreE);
//    ovlp[0] = this->wave.Overlap(walk); 
//    ovlp[1] = Overlap1(walk, ovlp[0]);
//    ovlp[2] = ovlp[0] + alpha * ovlp[1];
//    cout << "hamandovlp\n" << ovlp[0] << "  " << ovlp[1] << "  " << ovlp[2] << "  " << E0 << endl << endl;
//    el0 = E0;
//    el1 = E0;
//    
//    generateAllScreenedSingleExcitation(walk.d, schd.epsilon, schd.screen,
//                                        work, false);  
//    generateAllScreenedDoubleExcitation(walk.d, schd.epsilon, schd.screen,
//                                      work, false);  
//
//    //loop over all the screened excitations
//    for (int i=0; i<work.nExcitations; i++) {
//      int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
//      double tia = work.HijElement[i];
//      
//      int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
//      int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
//
//      //double ovlpRatio = getOverlapFactor(I, J, A, B, walk, false);
//      double ovlpRatio0 = this->wave.getOverlapFactor(I, J, A, B, walk, false);
//
//      ham += tia * ovlpRatio;
//
//      work.ovlpRatio[i] = ovlpRatio;
//    }
//    
//    ham[0] = el0 * ovlp[0] * ovlp[0] / (ovlp[2] * ovlp[2]);
//    ham[1] = el0 * ovlp[0] * ovlp[1] / (ovlp[2] * ovlp[2]);
//    ham[2] = el1 * ovlp[1] * ovlp[1] / (ovlp[2] * ovlp[2]);
//  }
//
//
//  void writeWave()
//  {
//    if (commrank == 0)
//      {
//	char file[5000];
//	//sprintf (file, "wave.bkp" , schd.prefix[0].c_str() );
//	sprintf(file, "lanczoswave.bkp");
//	std::ofstream outfs(file, std::ios::binary);
//	boost::archive::binary_oarchive save(outfs);
//	save << *this;
//	outfs.close();
//      }
//  }
//
//  void readWave()
//  {
//    if (commrank == 0)
//      {
//	char file[5000];
//	//sprintf (file, "wave.bkp" , schd.prefix[0].c_str() );
//	sprintf(file, "lanczoswave.bkp");
//	std::ifstream infs(file, std::ios::binary);
//	boost::archive::binary_iarchive load(infs);
//	load >> *this;
//	infs.close();
//      }
//#ifndef SERIAL
//    boost::mpi::communicator world;
//    boost::mpi::broadcast(world, *this, 0);
//#endif
//  }
//};
#endif
