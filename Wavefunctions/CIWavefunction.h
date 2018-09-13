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
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <utility>

using namespace std;

class oneInt;
class twoInt;
class twoIntHeatBathSHM;
class CPSSlaterWalker;

class Operator {
 private:
  friend class boost::serialization::access;
  template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
      ar & cre
	& des
	& n
	& nops;
    }

 public:
  
  std::array<short, 4> cre;
  std::array<short, 4> des;
  int n;
  int nops;

 Operator() : cre ({0,0,0,0}), des ({0,0,0,0}) { 
    n = 0;
    nops = 1;
  }

  //a1^\dag i1
 Operator(short a1, short i1) : cre ({a1}), des ({i1}) {
    n = 1;
    nops = 1;
  }

  //a2^\dag i2 a1^\dag i1
 Operator(short a1, short a2, short i1, short i2) :cre ({a2, a1}), des ({i2, i1}) {
    n = 2;
    nops = 1;
  }

  friend ostream& operator << (ostream& os, Operator& o) {
    for (int i=0; i<o.n; i++)
      os<<o.cre[i]<<" "<<o.des[i]<<"    ";
    return os;
  }

  bool apply(Determinant &dcopy, int op)
  {
    bool valid = true;
    for (int j = 0; j < n; j++)
      {
	if (dcopy.getocc(cre[j]) == true)
	  dcopy.setocc(cre[j], false);
	else
	  return false;

	if (dcopy.getocc(des[j]) == false)
	  dcopy.setocc(des[j], true);
	else
	  return false;
      }
    return valid;
  }

  static void populateSinglesToOpList(vector<Operator>& oplist) {
    int norbs = Determinant::norbs;
    for (int i = 0; i < 2 * norbs; i++)
      for (int j = 0; j < 2 * norbs; j++)
	{
	  //if (I2hb.Singles(i, j) > schd.epsilon )
	  if (i % 2 == j % 2)
	    {
	      oplist.push_back(Operator(i, j));
	    }
	}
  }

  static void populateScreenedDoublesToOpList(vector<Operator>& oplist, double screen) {
    int norbs = Determinant::norbs;
    for (int i = 0; i < 2 * norbs; i++)
      {
	for (int j = i + 1; j < 2 * norbs; j++)
	  {
	    int pair = (j / 2) * (j / 2 + 1) / 2 + i / 2;

	    size_t start = i % 2 == j % 2 ? I2hb.startingIndicesSameSpin[pair] : I2hb.startingIndicesOppositeSpin[pair];
	    size_t end = i % 2 == j % 2 ? I2hb.startingIndicesSameSpin[pair + 1] : I2hb.startingIndicesOppositeSpin[pair + 1];
	    float *integrals = i % 2 == j % 2 ? I2hb.sameSpinIntegrals : I2hb.oppositeSpinIntegrals;
	    short *orbIndices = i % 2 == j % 2 ? I2hb.sameSpinPairs : I2hb.oppositeSpinPairs;

	    for (size_t index = start; index < end; index++)
	      {
		if (fabs(integrals[index]) < screen)
		  break;
		int a = 2 * orbIndices[2 * index] + i % 2, b = 2 * orbIndices[2 * index + 1] + j % 2;
		//cout << i<<"  "<<j<<"  "<<a<<"  "<<b<<"  spin orbs "<<integrals[index]<<endl;

		oplist.push_back(Operator(i, j, a, b));
	      }
	  }
      }

  }
};


class SpinFreeOperator {
 private:
  friend class boost::serialization::access;
  template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
      ar & ops
	& nops;
    }

 public:

  vector<Operator> ops;
  int nops;

  SpinFreeOperator() { 
    ops.push_back(Operator());
    nops = 1;
  }

  //a1^\dag i1
  SpinFreeOperator(short a1, short i1) {
    ops.push_back(Operator(2*a1, 2*i1));
    ops.push_back(Operator(2*a1+1, 2*i1+1));
    nops = 2;
  }

  //a2^\dag i2 a1^\dag i1
  SpinFreeOperator(short a1, short a2, short i1, short i2) {
    if (a1 == a2 && a1 == i1 && a1 == i2) {
      ops.push_back(Operator(2*a1+1, 2*a2, 2*i1+1, 2*i2));
      ops.push_back(Operator(2*a1, 2*a2+1, 2*i1, 2*i2+1));
      nops = 2;
    }
    else {
      ops.push_back(Operator(2*a1, 2*a2, 2*i1, 2*i2));
      ops.push_back(Operator(2*a1+1, 2*a2, 2*i1+1, 2*i2));
      ops.push_back(Operator(2*a1, 2*a2+1, 2*i1, 2*i2+1));
      ops.push_back(Operator(2*a1+1, 2*a2+1, 2*i1+1, 2*i2+1));
      nops = 4;
    }
  }

  friend ostream& operator << (ostream& os, const SpinFreeOperator& o) {
    for (int i=0; i<o.ops[0].n; i++)
      os<<o.ops[0].cre[i]/2<<" "<<o.ops[0].des[i]/2<<"    ";
    return os;
  }

  bool apply(Determinant &dcopy, int op)
  {
    return ops[op].apply(dcopy, op);
  }

  static void populateSinglesToOpList(vector<SpinFreeOperator>& oplist) {
    int norbs = Determinant::norbs;
    for (int i = 0; i <  norbs; i++)
      for (int j = 0; j <  norbs; j++)
	{
	  oplist.push_back(SpinFreeOperator(i, j));
	}
  }

  static void populateScreenedDoublesToOpList(vector<SpinFreeOperator>& oplist, double screen) {
    int norbs = Determinant::norbs;
    for (int i = 0; i < norbs; i++)
      {
	for (int j = i; j < norbs; j++)
	  {

	    /*
	    for (int a = 0; a<norbs; a++)
	      for (int b = 0; b<norbs; b++) 
		oplist.push_back(SpinFreeOperator(i, a, j, b));
	    */

	    int pair = (j) * (j + 1) / 2 + i ;

	    set<std::pair<int, int> > UniqueSpatialIndices;
	    if (j != i) { //same spin
	      size_t start = I2hb.startingIndicesSameSpin[pair] ;
	      size_t end = I2hb.startingIndicesSameSpin[pair + 1];
	      float *integrals = I2hb.sameSpinIntegrals ;
	      short *orbIndices = I2hb.sameSpinPairs ;
	      
	      for (size_t index = start; index < end; index++)
		{
		  if (fabs(integrals[index]) < screen)
		    break;
		  int a = orbIndices[2 * index], b = orbIndices[2 * index + 1];
		  //cout << i<<"  "<<j<<"  "<<a<<"  "<<b<<"  same spin "<<integrals[index]<<endl;
		  UniqueSpatialIndices.insert(std::pair<int, int>(a,b));
		}
	    }

	    { //opposite spin
	      size_t start = I2hb.startingIndicesOppositeSpin[pair];
	      size_t end = I2hb.startingIndicesOppositeSpin[pair + 1];
	      float *integrals = I2hb.oppositeSpinIntegrals;
	      short *orbIndices = I2hb.oppositeSpinPairs;
	      
	      for (size_t index = start; index < end; index++)
		{
		  if (fabs(integrals[index]) < screen)
		    break;
		  int a = orbIndices[2 * index], b = orbIndices[2 * index + 1];
		  //cout << i<<"  "<<j<<"  "<<a<<"  "<<b<<"  opp spin "<<integrals[index]<<endl;
		  UniqueSpatialIndices.insert(std::pair<int, int>(a,b));
		}		
	    }

	    for (auto it = UniqueSpatialIndices.begin(); 
		 it != UniqueSpatialIndices.end(); it++) {

	      int a = it->first, b = it->second;
	      oplist.push_back(SpinFreeOperator(a, b, i, j));

	    }

	  }
      }
  }

};


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


  void initWalker(Walker& walk) {
    wave.initWalker(walk);
  }  

  void initWalker(Walker& walk, Determinant& d) {
    wave.initWalker(walk, d);
  }  

  void getVariables(VectorXd& vars) {
    if (vars.rows() != getNumVariables())
      {
	vars = VectorXd::Zero(getNumVariables());
      }
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

  double getovlpRatio(Walker &walk, Determinant &dcopy)
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
	getOrbDiff(walk.d, dcopy, I, A);
	ovlpdetcopy = wave.getOverlapFactor(I, A, walk, false);
      }
    else if (excitationDistance == 2)
      {
	int I, J, A, B;
	getOrbDiff(walk.d, dcopy, I, J, A, B);
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
	    //cout << i<<" -  "<<oplist[i].ops[j]<<"  "<<walk.d<<"  "<<dcopy<<endl;
	    double ovlpdetcopy = getovlpRatio(walk, dcopy);
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
	    double ovlpdetcopy = getovlpRatio(walk, dcopy);
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
    work.init();

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
			d.parityA(A, I, sgn);
		      else
			d.parityB(A, I, sgn);
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
			d.parityA(A, I, sgn);
		      else
			d.parityB(A, I, sgn);
		      tia *= sgn;
		    }

		  double localham = 0.0;
		  if (abs(tia) > THRESH)
		    {
		      Walker walkcopy = walk;
		      walkcopy.exciteWalker(wave, closed[i]*2*norbs+open[a], 0, norbs);
		      double ovlpdetcopy = Overlap(walkcopy);
		      ham += ovlpdetcopy * tia / ovlp;

		      if (fillExcitations)
			work.appendValue(ovlpdetcopy/ovlp, closed[i] * 2 * norbs + open[a],
					 0, tia);
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
		  walkcopy.exciteWalker(wave, closed[i] * 2 * norbs + a, closed[j] * 2 * norbs + b, norbs);
		  double ovlpdetcopy = Overlap(walkcopy);

		  double parity = 1.0;
		  if (closed[i] % 2 == closed[j] % 2 && closed[i] % 2 == 0)
		    walk.d.parityAA(I, J, A, B, parity); 
		  else if (closed[i] % 2 == closed[j] % 2 && closed[i] % 2 == 1)
		    walk.d.parityBB(I, J, A, B, parity); 
		  else if (closed[i] % 2 != closed[j] % 2 && closed[i] % 2 == 0)
		    {walk.d.parityA(A, I, parity) ; walk.d.parityB(B, J, parity);}
		  else
		    {walk.d.parityB(A, I,parity); walk.d.parityA(B, J, parity);}
            
		  ham += ovlpdetcopy * tiajb * parity / ovlp;

		  if (fillExcitations)
		    work.appendValue(ovlpdetcopy/ovlp, closed[i] * 2 * norbs + a,
				     closed[j] * 2 * norbs + b , tiajb);
		}
	    }
	}
      prof.DoubleTime += getTime() - time;
    }
  }

  vector<Determinant> &getDeterminants() { return wave.getDeterminants(); }
  vector<double> &getciExpansion() { return wave.getciExpansion(); }

  MatrixXd& getHforbsA() {return wave.getHforbsA();}
  MatrixXd& getHforbsB() {return wave.getHforbsB();}
  MatrixXd& getGHFOrbs() {return wave.getGHFOrbs();}

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

#endif
