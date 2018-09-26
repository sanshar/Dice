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
#ifndef ExcitationOperators_HEADER_H
#define ExcitationOperators_HEADER_H
#include <vector>
#include <set>
#include "Determinants.h"
#include "workingArray.h"
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <utility>

using namespace std;

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
  
  //used in Lanczos
  static void populateSinglesToOpList(vector<Operator>& oplist, vector<double>& hamElements) {
    int norbs = Determinant::norbs;
    for (int i = 0; i < 2 * norbs; i++)
      for (int j = 0; j < 2 * norbs; j++)
	{
	  //if (I2hb.Singles(i, j) > schd.epsilon )
	  if (i % 2 == j % 2)
	    {
	      oplist.push_back(Operator(i, j));
	      hamElements.push_back(I1(i, j));
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

  //used in Lanczos
  static void populateScreenedDoublesToOpList(vector<Operator>& oplist, vector<double>& hamElements, double screen) {
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
		hamElements.push_back(integrals[index]);
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

#endif
