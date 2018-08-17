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
#include "CPS.h"
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <array>

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
       & n;
  }

  public:

  std::array<short, 4> cre;
  std::array<short, 4> des;
  int n;

  Operator() { 
    cre = {0,0,0,0};
    des = {0,0,0,0};
    n = 0;
  }

  //a1^\dag i1
  Operator(int a1, int i1) {
    n = 1;
    cre = {a1};
    des = {i1};
  }

  //a2^\dag i2 a1^\dag i1
  Operator(int a1, int a2, int i1, int i2) {
    n = 2;
    cre = {a2, a1};
    des = {i2, i1};
  }

}

/**
* This is the wavefunction, that extends a given wavefunction by doing
* a CI expansion on it
* |Psi> = \sum_i d_i O_i ||phi>
* where d_i are the coefficients and O_i are the list of operators
*/
template <typename Wfn>
class CIWavefunction
{
private:
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar &Wfn
        &oplist;
  }

public:
  Wfn wave;
  std::vector<Operator> oplist;

  CIWavefunction(Wfn &w1, std::vector<Operator> &pop) : wave(w1), oplist(pop){};

  void OverlapWithGradient(CPSSlaterWalker &,
                           double &factor,
                           Eigen::VectorXd &grad);

  void HamAndOvlp(CPSSlaterWalker &walk,
                  double &ovlp, double &ham,
                  vector<double> &ovlpRatio, vector<size_t> &excitation1,
                  vector<size_t> &excitation2, vector<double> &HijElement,
                  int &nExcitations, bool fillExcitations = true);
};

#endif
