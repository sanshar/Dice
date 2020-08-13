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

#include "Determinants.h"
#include "global.h"

// This is hash_combine closely based on that used in boost. We use this
// instead of boost::hash_combined directly, because the latter is
// inconsistent across different versions of boost, meaning that tests fail
// depending on the boost version used.
template <class T>
inline void hash_combine_proc(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

// Get unique processor label for this determinant
int getProc(const Determinant& d, int DetLenLocal) {
  std::size_t h_tot = 0;
  for (int i=0; i<DetLenLocal; i++) {
    hash_combine_proc(h_tot, d.reprA[i]);
    hash_combine_proc(h_tot, d.reprB[i]);
  }
  return h_tot % commsize;
}
