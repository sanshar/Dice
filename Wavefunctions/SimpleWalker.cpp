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
#include "SimpleWalker.h"
  
void SimpleWalker::updateA(int i, int a)
{
  d.setoccA(i, false);
  d.setoccA(a, true);
}

void SimpleWalker::updateA(int i, int j, int a, int b)
{
  d.setoccA(i, false);
  d.setoccA(a, true);
  d.setoccA(j, false);
  d.setoccA(b, true);
}

void SimpleWalker::updateB(int i, int a)
{
  d.setoccB(i, false);
  d.setoccB(a, true);
}

void SimpleWalker::updateB(int i, int j, int a, int b)
{
  d.setoccB(i, false);
  d.setoccB(a, true);
  d.setoccB(j, false);
  d.setoccB(b, true);
}

void SimpleWalker::updateEnergyIntermediate(const oneInt& I1, const twoInt& I2, int I, int A) 
{
  int norbs = Determinant::norbs;
  for (int i = 0; i < norbs; i++) {
    energyIntermediates[0][i] += (I2.Direct(i, A/2) - I2.Direct(i, I/2));
    energyIntermediates[1][i] += (I2.Direct(i, A/2) - I2.Direct(i, I/2));
    energyIntermediates[I%2][i] -= (I2.Exchange(i, A/2) - I2.Exchange(i, I/2));
  }
}

void SimpleWalker::updateEnergyIntermediate(const oneInt& I1, const twoInt& I2, int I, int A, int J, int B) 
{
  int norbs = Determinant::norbs;
  for (int i = 0; i < norbs; i++) {
    energyIntermediates[0][i] += (I2.Direct(i, A/2) - I2.Direct(i, I/2) + I2.Direct(i, B/2) - I2.Direct(i, J/2));
    energyIntermediates[1][i] += (I2.Direct(i, A/2) - I2.Direct(i, I/2) + I2.Direct(i, B/2) - I2.Direct(i, J/2));
    energyIntermediates[I%2][i] -= (I2.Exchange(i, A/2) - I2.Exchange(i, I/2));
    energyIntermediates[J%2][i] -= (I2.Exchange(i, B/2) - I2.Exchange(i, J/2));
  }
}

//assumes valid excitations
//the energyIntermediates should only be updated for outer walker updates
void SimpleWalker::updateWalker(const Determinant &ref, const Determinant &corr, int ex1, int ex2, bool updateIntermediate)
{
  int norbs = Determinant::norbs;
  int I = ex1 / (2 * norbs), A = ex1 % (2 * norbs);

  if (I < 2*schd.nciCore) excitedHoles.insert(I);
  if (A < 2*schd.nciCore) excitedHoles.erase(A);

  if (I >= 2*(schd.nciCore+schd.nciAct)) excitedOrbs.erase(I);
  if (A >= 2*(schd.nciCore+schd.nciAct)) excitedOrbs.insert(A);

  if (I % 2 == 0) {
    updateA(I / 2, A / 2);
  }
  else {
    updateB(I / 2, A / 2);
  }
  
  //if (ex2 == 0 && updateIntermediate) {
  //  updateEnergyIntermediate(I1, I2, I, A);
  //}
  
  if (ex2 != 0)
  {
    int J = ex2 / (2 * norbs), B = ex2 % (2 * norbs);

    if (J < 2*schd.nciCore) excitedHoles.insert(J);
    if (B < 2*schd.nciCore) excitedHoles.erase(B);

    if (J >= 2*(schd.nciCore+schd.nciAct)) excitedOrbs.erase(J);
    if (B >= 2*(schd.nciCore+schd.nciAct)) excitedOrbs.insert(B);
    
    if (J % 2 == 0)
      updateA(J / 2, B / 2);
    else
      updateB(J / 2, B / 2);
    //if (updateIntermediate) {
    //  updateEnergyIntermediate(I1, I2, I, A, J, B);
    //}
  }

  // Find the excitation class
  if (excitedHoles.size() == 0) {
    // 0 core orbitals
    if (excitedOrbs.size() == 0) {
      excitation_class = 0;
    } else if (excitedOrbs.size() == 1) {
      excitation_class = 1;
    } else if (excitedOrbs.size() == 2) {
      excitation_class = 2;
    } else {
      excitation_class = -1;
    }

  } else if (excitedHoles.size() == 1) {
    // 1 core orbitals
    if (excitedOrbs.size() == 0) {
      excitation_class = 3;
    } else if (excitedOrbs.size() == 1) {
      excitation_class = 4;
    } else if (excitedOrbs.size() == 2) {
      excitation_class = 5;
    } else {
      excitation_class = -1;
    }

  } else if (excitedHoles.size() == 2) {
    // 2 core orbitals
    if (excitedOrbs.size() == 0) {
      excitation_class = 6;
    } else if (excitedOrbs.size() == 1) {
      excitation_class = 7;
    } else if (excitedOrbs.size() == 2) {
      excitation_class = 8;
    } else {
      excitation_class = -1;
    }

  // More than 2 core orbitals
  } else {
    excitation_class = -1;
  }

}

//not implemented for SimpleWalker
void SimpleWalker::exciteWalker(const Determinant &ref, const Determinant &corr, int excite1, int excite2, int norbs)
{
  int I1 = excite1 / (2 * norbs), A1 = excite1 % (2 * norbs);

  if (I1 % 2 == 0) {
    updateA(I1 / 2, A1 / 2);
  }
  else {
    updateB(I1 / 2, A1 / 2);
  }
  
  if (excite2 != 0)
  {
    int I2 = excite2 / (2 * norbs), A2 = excite2 % (2 * norbs);
    
    if (I2 % 2 == 0)
      updateA(I2 / 2, A2 / 2);
    else
      updateB(I2 / 2, A2 / 2);
  }
}

