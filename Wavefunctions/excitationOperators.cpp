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
#include "excitationOperators.h"
#include "Determinants.h"
#include "integral.h"
#include "input.h"

Operator::Operator() : cre ({0,0,0,0}), des ({0,0,0,0}) { 
  n = 0;
  nops = 1;
}

//a1^\dag i1
Operator::Operator(short a1, short i1) : cre ({a1}), des ({i1}) {
  n = 1;
  nops = 1;
}

//a2^\dag i2 a1^\dag i1
Operator::Operator(short a1, short a2, short i1, short i2) :cre ({a2, a1}), des ({i2, i1}) {
  n = 2;
  nops = 1;
}

ostream& operator << (ostream& os, Operator& o) {
  for (int i=0; i<o.n; i++)
    os<<o.cre[i]<<" "<<o.des[i]<<"    ";
  return os;
}

bool Operator::apply(Determinant &dcopy, int op)
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

//used in active space calcs, excitedOrbs contains excited orbs in dcopy
bool Operator::apply(Determinant &dcopy, const unordered_set<int> &excitedOrbs)
{
  if (excitedOrbs.size() == 2) {
    for (int j = 0; j < n; j++) {
      //if ((cre[j] >= 2*schd.nciAct) || (excitedOrbs.find(des[j]) == excitedOrbs.end())) return false;
      //if (cre[j] >= 2*schd.nciAct) return false;
      if (dcopy.getocc(des[j]) == true)
        dcopy.setocc(des[j], false);
      else
        return false;
      if (dcopy.getocc(cre[j]) == false)
        dcopy.setocc(cre[j], true);
      else
        return false;
    }
    return true;
  }
  
  else if (excitedOrbs.size() == 1) {
    //bool valid = false;
    bool valid = true;
    for (int j = 0; j < n; j++) {
      //if (cre[j] >= 2*schd.nciAct) return false;
      //if (excitedOrbs.find(des[j]) == excitedOrbs.end()) valid = true;
      if (dcopy.getocc(des[j]) == true)
        dcopy.setocc(des[j], false);
      else
        return false;
      if (dcopy.getocc(cre[j]) == false)
        dcopy.setocc(cre[j], true);
      else
        return false;
    }
    return valid;
  }

  else {
    bool valid = true;
    for (int j = 0; j < n; j++) {
      //if ((des[j] >= 2*schd.nciAct) || (cre[j] >= 2*schd.nciAct)) return false;
      if (des[j] >= 2*schd.nciAct) return false;
      if (dcopy.getocc(des[j]) == true)
        dcopy.setocc(des[j], false);
      else
        return false;
      if (dcopy.getocc(cre[j]) == false)
        dcopy.setocc(cre[j], true);
      else
        return false;
    }
    return valid;
  }
}
  
//used in Lanczos
void Operator::populateSinglesToOpList(vector<Operator>& oplist, vector<double>& hamElements, double screen) {
  int norbs = Determinant::norbs;
  for (int i = 0; i < 2 * norbs; i++)
    for (int j = 0; j < 2 * norbs; j++)
    {
      //if (I2hb.Singles(i, j) > schd.epsilon )
      if (i % 2 == j % 2 && abs(I2hb.Singles(i,j)) > screen)
      {
        oplist.push_back(Operator(i, j));
        hamElements.push_back(I1(i, j));
      }
    }
}


//used in Lanczos
void Operator::populateScreenedDoublesToOpList(vector<Operator>& oplist, vector<double>& hamElements, double screen) {
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


//used in Lanczos
void Operator::populateDoublesToOpList(vector<Operator>& oplist, vector<double>& hamElements) {
  int norbs = Determinant::norbs;
  for (int i = 0; i < 2 * norbs; i++)
  {
    for (int j = i + 1; j < 2 * norbs; j++)
    {
      for (int a = 0; a < 2 * norbs ; a++)
        for (int b = a+1; b < 2* norbs; b++) {
          oplist.push_back(Operator(i, j, a, b));
          hamElements.push_back(0.0);
        }
    }
  }
}



SpinFreeOperator::SpinFreeOperator() { 
  ops.push_back(Operator());
  nops = 1;
}

//a1^\dag i1
SpinFreeOperator::SpinFreeOperator(short a1, short i1) {
  ops.push_back(Operator(2*a1, 2*i1));
  ops.push_back(Operator(2*a1+1, 2*i1+1));
  nops = 2;
}

//a2^\dag i2 a1^\dag i1
SpinFreeOperator::SpinFreeOperator(short a1, short a2, short i1, short i2) {
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

ostream& operator << (ostream& os, const SpinFreeOperator& o) {
  for (int i=0; i<o.ops[0].n; i++)
    os<<o.ops[0].cre[i]/2<<" "<<o.ops[0].des[i]/2<<"    ";
  return os;
}

bool SpinFreeOperator::apply(Determinant &dcopy, int op)
{
  return ops[op].apply(dcopy, op);
}


void SpinFreeOperator::populateSinglesToOpList(vector<SpinFreeOperator>& oplist, vector<double>& hamElements, double screen) {
  int norbs = Determinant::norbs;
  for (int i = 0; i < norbs; i++)
    for (int j = 0; j < norbs; j++)
    {
      if (abs(I2hb.Singles(2*i, 2*j)) < screen) continue;
      oplist.push_back(SpinFreeOperator(i, j));
      hamElements.push_back(I1(2*i, 2*j));
    }
}

void SpinFreeOperator::populateScreenedDoublesToOpList(vector<SpinFreeOperator>& oplist, vector<double>& hamElements, double screen) {
  int norbs = Determinant::norbs;
  for (int i = 0; i < norbs; i++)
  {
    for (int j = i; j < norbs; j++)
    {
      int pair = (j) * (j + 1) / 2 + i ;
      
      set<std::pair<int, int> > UniqueSpatialIndices;
      vector<double> uniqueHamElements;
      
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
          uniqueHamElements.push_back(integrals[index]);
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
          uniqueHamElements.push_back(integrals[index]);
        }		
      }
      
      int index = 0;
      for (auto it = UniqueSpatialIndices.begin(); 
           it != UniqueSpatialIndices.end(); it++) {
        
        int a = it->first, b = it->second;
        oplist.push_back(SpinFreeOperator(a, b, i, j));
        hamElements.push_back(uniqueHamElements[index]);
        index++;
      }
      
    }
  }
}

