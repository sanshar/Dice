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
#ifndef workingArray_HEADER_H
#define workingArray_HEADER_H
#include <vector>


//this is a simple class that just stores the set of 
//overlaps and hij matix elements whenever local energy is
//calculated
struct workingArray {

  vector<double> ovlpRatio;
  vector<size_t> excitation1;
  vector<size_t> excitation2;
  vector<double> HijElement;
  int nExcitations;

  workingArray(size_t initialSize = 1000000) {
    nExcitations = 0;
    ovlpRatio.resize(initialSize);
    excitation1.resize(initialSize);
    excitation2.resize(initialSize);
    HijElement.resize(initialSize);
  }

  void incrementSize(size_t size) {
    size_t newSize = ovlpRatio.size()+size;
    ovlpRatio.resize(newSize);
    excitation1.resize(newSize);
    excitation2.resize(newSize);
    HijElement.resize(newSize);
  }
  
  void appendValue(double ovlp, size_t ex1, size_t ex2, double hij) {
    int ovlpsize = ovlpRatio.size();
    if (ovlpsize <= nExcitations) 
      incrementSize(1000000);

    ovlpRatio[nExcitations] = ovlp;
    excitation1[nExcitations] = ex1;
    excitation2[nExcitations] = ex2;
    HijElement[nExcitations] = hij;
    nExcitations++;
  }

  void init() {
    nExcitations = 0;
  }

  void clear() {
    ovlpRatio.clear();
    excitation1.clear();
    excitation2.clear();
    HijElement.clear();
    nExcitations = 0;
  }
};


#endif
