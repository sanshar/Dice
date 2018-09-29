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
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <utility>

class Determinant;

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

  Operator();

  //a1^\dag i1
  Operator(short a1, short i1);

  //a2^\dag i2 a1^\dag i1
  Operator(short a1, short a2, short i1, short i2);

  friend ostream& operator << (ostream& os, Operator& o);

  bool apply(Determinant &dcopy, int op);

  
  //used in Lanczos
  static void populateSinglesToOpList(vector<Operator>& oplist, vector<double>& hamElements);


  //used in Lanczos
  static void populateScreenedDoublesToOpList(vector<Operator>& oplist, vector<double>& hamElements, double screen);


    //used in Lanczos
  static void populateDoublesToOpList(vector<Operator>& oplist, vector<double>& hamElements);

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

  SpinFreeOperator();

  //a1^\dag i1
  SpinFreeOperator(short a1, short i1);

  //a2^\dag i2 a1^\dag i1
  SpinFreeOperator(short a1, short a2, short i1, short i2);

  friend ostream& operator << (ostream& os, const SpinFreeOperator& o);

  bool apply(Determinant &dcopy, int op);

  static void populateSinglesToOpList(vector<SpinFreeOperator>& oplist, vector<double>& hamElements);
  
  static void populateScreenedDoublesToOpList(vector<SpinFreeOperator>& oplist, vector<double>& hamElements, double screen);
  
};

#endif
