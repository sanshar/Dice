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

  static void populateSinglesToOpList(vector<Operator>& oplist, vector<double>& hamElements, double screen=0.0);

  static void populateScreenedDoublesToOpList(vector<Operator>& oplist, vector<double>& hamElements, double screen);

  static void populateDoublesToOpList(vector<Operator>& oplist, vector<double>& hamElements);
};

////normal ordered
//class NormalOperator {
//  private:
//    friend class boost::serialization::access;
//    template <class Archive>
//      void serialize(Archive &ar, const unsigned int version)
//      {
//        ar & cre
//      & des
//      & n
//      & nops;
//      }
//
//  public:
//   
//    std::array<short, 4> cre;
//    std::array<short, 4> des;
//    int n;
//    int nops;
//
//    NormalOperator() : cre ({0,0,0,0}), des ({0,0,0,0}) 
//    {
//      n = 0;
//      nops = 1;
//    }
//
//    //a1^\dag i1
//    NormalOperator(short a1, short i1) : cre ({a1}), des ({i1}) 
//    {
//      n = 1;
//      nops = 1;
//    }
//
//    //a2^\dag a1^\dag i2 i1
//    NormalOperator(short a2, short a1, short i2, short i1) :cre ({a2, a1}), des ({i2, i1}) 
//    {
//      n = 2;
//      nops = 1;
//    }
//
//    friend ostream& operator << (ostream& os, Operator& o) 
//    {
//      for (int i = 0; i < o.n; i++)
//        os << o.cre[i] << "  ";
//      os << "; ";
//      for (int i = 0; i < o.n; i++)
//        os << o.des[i] << "  ";
//      os << endl;
//      return os;
//    }
//
//    //apply to a bra: < dcopy | NormalOperator
//    bool apply(Determinant &dcopy, int op)
//    {
//      for (int j = 0; j < n; j++) {
//        if (dcopy.getocc(cre[j]) == true)
//          dcopy.setocc(cre[j], false);
//        else
//          return false;
//      }
//
//      for (int j = 0; j < n; j++) {
//        if (dcopy.getocc(des[j]) == false)
//          dcopy.setocc(des[j], true);
//        else
//          return false;
//      }
//      
//      return true;
//    }
//
//    static void populateSinglesToOpList(vector<NormalOperator>& oplist)
//    {
//      int norbs = Determinant::norbs;
//      for (int i = 0; i < 2 * norbs; i++) {
//        for (int j = 0; j < 2 * norbs; j++) {
//          //if (I2hb.Singles(i, j) > schd.epsilon )
//          if (i % 2 == j % 2)
//            oplist.push_back(NormalOperator(i, j));
//        }
//      }
//    }
//    
//    //used in Lanczos
//    static void populateSinglesToOpList(vector<NormalOperator>& oplist, vector<double>& hamElements) {
//      int norbs = Determinant::norbs;
//      for (int i = 0; i < 2 * norbs; i++) {
//        for (int j = 0; j < 2 * norbs; j++) {
//          //if (I2hb.Singles(i, j) > schd.epsilon )
//          if (i % 2 == j % 2) {
//            oplist.push_back(NormalOperator(i, j));
//            hamElements.push_back(I1(i, j));
//          }
//        }
//      }
//    }
//
//    static void populateScreenedDoublesToOpList(vector<NormalOperator>& oplist, double screen) 
//    {
//      int norbs = Determinant::norbs;
//      for (int p = 0; p < 2 * norbs; i++) {
//        for (int q = 0; q < 2 * norbs; j++) {
//          for (int r = 0; r < 2 * norbs; k++) { 
//            for (int s = 0; s < 2 * norbs; k++) {
//              if (p != r && s != q) {
//                double hamCoeff = I2(p, q, r, s);
//                if (hamCoeff > screen) 
//                  oplist.push_back(NormalOperator(p, r, s, q));
//              }
//            }
//          }
//        }
//      }
//    }
//    
//    //used in Lanczos
//    static void populateScreenedDoublesToOpList(vector<NormalOperator>& oplist, vector<double>& hamElements, double screen) 
//    {
//      int norbs = Determinant::norbs;
//      for (int p = 0; p < 2 * norbs; i++) {
//        for (int q = 0; q < 2 * norbs; j++) {
//          for (int r = 0; r < 2 * norbs; k++) { 
//            for (int s = 0; s < 2 * norbs; k++) {
//              if (p != r && s != q) {
//                double hamCoeff = I2(p, q, r, s);
//                if (hamCoeff > screen) {
//                  oplist.push_back(NormalOperator(p, r, s, q));
//                  hamElements.push_back(hamCoeff);
//                }
//              }
//            }
//          }
//        }
//      }
//    }
//
//};

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

  static void populateSinglesToOpList(vector<SpinFreeOperator>& oplist, vector<double>& hamElements, double screen =0.0);
  
  static void populateScreenedDoublesToOpList(vector<SpinFreeOperator>& oplist, vector<double>& hamElements, double screen);
  
};

#endif
