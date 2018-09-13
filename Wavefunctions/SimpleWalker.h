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
#ifndef SimpleWalker_HEADER_H
#define SimpleWalker_HEADER_H

#include "Determinants.h"
#include <boost/serialization/serialization.hpp>
#include <Eigen/Dense>
#include "input.h"

using namespace Eigen;

/**
* Is essentially a single determinant used in the VMC/DMC simulation
* At each step in VMC one need to be able to calculate the following
* quantities
* a. The local energy = <walker|H|Psi>/<walker|Psi>
*
*/

class SimpleWalker
{

public:
  Determinant d;                      //The current determinant

  // The constructor
  SimpleWalker(Determinant &pd)
      : d(pd) {};
  SimpleWalker(){};

  
  template <typename Wfn>
  void updateWalker(Wfn &w, int ex1, int ex2)
  {
    int norbs = Determinant::norbs;

    int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
    int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;

    //updateElecHoleAfterDes(I); updateElecHoleAfterCre(A);
    //updateElecHoleAfterDes(J); updateElecHoleAfterCre(B);
    d.setocc(I, false); d.setocc(A, true);
    d.setocc(J, false); d.setocc(B, true);
  }

  Determinant &getDet() { return d; }

  //these are not absolute orbital indices, but instead the
  //ith occupied and ath unoccupied


  template <typename Wfn>
  void updateA(int i, int a, Wfn &w)
  {

    //updateElecHoleAfterDes(2*i); updateElecHoleAfterCre(2*a);
    d.setoccA(i, false);
    d.setoccA(a, true);
  }

  template <typename Wfn>
  void updateA(int i, int j, int a, int b, Wfn &w)
  {
    //updateElecHoleAfterDes(2*i); updateElecHoleAfterCre(2*a);
    //updateElecHoleAfterDes(2*j); updateElecHoleAfterCre(2*b);
    d.setoccA(i, false);
    d.setoccA(a, true);
    d.setoccA(j, false);
    d.setoccA(b, true);
  }

  template <typename Wfn>
  void updateB(int i, int a, Wfn &w)
  {
    //updateElecHoleAfterDes(2*i+1); updateElecHoleAfterCre(2*a+1);
    d.setoccB(i, false);
    d.setoccB(a, true);
  }

  template <typename Wfn>
  void updateB(int i, int j, int a, int b, Wfn &w)
  {
    //updateElecHoleAfterDes(2*i+1); updateElecHoleAfterCre(2*a+1);
    //updateElecHoleAfterDes(2*j+1); updateElecHoleAfterCre(2*b+1);
    d.setoccB(i, false);
    d.setoccB(a, true);
    d.setoccB(j, false);
    d.setoccB(b, true);
  }

  bool operator<(const SimpleWalker &w) const
  {
    return d < w.d;
  }

  bool operator==(const SimpleWalker &w) const
  {
    return d == w.d;
  }

  template <typename Wfn>
  void exciteTo(Wfn& w, Determinant& dcopy) {
    d = dcopy;
  }
  
  template <typename Wfn>
  void exciteWalker(Wfn &w, int excite1, int excite2, int norbs)
  {
    int I1 = excite1 / (2 * norbs), A1 = excite1 % (2 * norbs);

    //updateElecHoleAfterDes(I1); updateElecHoleAfterCre(A1);
    if (I1 % 2 == 0) {
      updateA(I1 / 2, A1 / 2, w);
    }
    else {
      updateB(I1 / 2, A1 / 2, w);
    }
    
    if (excite2 != 0)
    {
      int I2 = excite2 / (2 * norbs), A2 = excite2 % (2 * norbs);
      //updateElecHoleAfterDes(I2); updateElecHoleAfterCre(A2);
      if (I2 % 2 == 0)
        updateA(I2 / 2, A2 / 2, w);
      else
        updateB(I2 / 2, A2 / 2, w);
    }
  }

  
};

#endif
