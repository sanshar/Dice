/*
   Developed by Sandeep Sharma with contributions from James E. Smith
   and Adam A. Homes, 2017
   Copyright (c) 2017, Sandeep Sharma

   This file is part of DICE.
   This program is free software: you can redistribute it and/or modify it under
   the terms of the GNU General Public License as published by the Free Software
   Foundation, either version 3 of the License, or (at your option) any later
   version.

   This program is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
   details.

   You should have received a copy of the GNU General Public License along with
   this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "symmetry.h"

#include <stdlib.h>

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "Determinants.h"
#include "communicate.h"
#include "global.h"
#include "integral.h"

using namespace Eigen;
using namespace std;

//
// Helper Functions
//
bool compareForSortingEnergies(const pair<double, int>& a,
                               const pair<double, int>& b) {
  return a.first < b.first;
}

/**
 * @brief Convert the string code/representation for a given irreducible
 * representation to the corresponding integer code. These codes match the
 * MOLPRO conventions and they're indexing starts at 1. This function sets the
 * targetIrrepString instance variable.
 *
 * @param pg The string code for the point group, e.g. "d2h".
 * @param irrep The string code for the target irrep, e.g. "B3u".
 * @return int The numerical code for the target irrep.
 */
int symmetry::convertStringIrrepToInt(string pg, string irrep) {
  if (irrep == (string) "None") {
    irrep = pg_irreps[pg][0];
    targetIrrepString = irrep;
  }

  vector<string> irreps;
  if (pg_irreps.find(pg) != pg_irreps.end()) {
    irreps = pg_irreps[pg];
  } else {
    pout << "WARNING: Irrep " << irrep << " not supported for point group "
         << pg << endl;
    init_success = false;
    return -1;  // Failure
  }

  vector<string>::iterator it = find(irreps.begin(), irreps.end(), irrep);

  // Make sure it's found
  if (it == irreps.end()) {
    pout << "WARNING: Irrep " << irrep << " not supported for point group "
         << pg << endl;
    pout << "Try one of the following irreducible representations: ";
    for (auto ir : irreps) {
      pout << ir << " ";
    }
    pout << endl;
    init_success = false;
    return -1;  // Failure
  }
  // Return the 1-indexed value so it matches the IDs from the FCIDUMP
  return distance(irreps.begin(), it) + 1;
}

// Constructor
/**
 * @brief Construct a new symmetry::symmetry object. This is done once at the
 * beginning of every Dice calculation. The full product table is resized
 * based on the point group (with D2h being the full table). The irreps are
 * checked to make sure they match the specified point group.
 *
 * @param pg Point group for the molecule in all lower case letters.
 * @param mol_irreps A vector of the irreps for the active space orbitals.
 */
symmetry::symmetry(string pg, vector<int>& mol_irreps, string targetIrrepStr) {
  init_pg_irreps();
  // Assign correct product table for point group. Note that even though the
  // indices of the columns are one less than the MOLPRO notation the irrep
  // returned is again in the MOLPRO notation.
  init_success = true;
  moIrreps = mol_irreps;

  // D2h Ag:1, B1g:4, B2g:6, B3g:7, Au:8, B1u:5, B2u:3, B3u: 2 in MOLPRO.
  // Row & col indices: Ag:0, B1g:3, B2g:5, B3g:6, Au:7, B1u:4, B2u:2, B3u:1
  /*
  d2h_table = [
    ["Ag", "B1g", "B2g", "B3g", "Au", "B1u", "B2u", "B3u"],
    ["B1g", "Ag", "B3g", "B2g", "B1u", "Au", "B3u", "B2u"],
    ["B2g", "B3g", "Ag", "B1g", "B2u", "B3u", "Au", "B1u"],
    ["B3g", "B2g", "B1g", "Ag", "B3u", "B2u", "B1u", "Au"],
    ["Au", "B1u", "B2u", "B3u", "Ag", "B1g", "B2g", "B3g"],
    ["B1u", "Au", "B3u", "B2u", "B1g", "Ag", "B3g", "B2g"],
    ["B2u", "B3u", "Au", "B1u", "B2g", "B3g", "Ag", "B1g"],
    ["B3u", "B2u", "B1u", "Au", "B3g", "B2g", "B1g", "Ag"],
  ]
  */

  fullD2hTable.resize(8, 8);

  // clang-format off
  fullD2hTable << 
    1, 2, 3, 4, 5, 6, 7, 8, 
    2, 1, 4, 3, 6, 5, 8, 7, 
    3, 4, 1, 2, 7, 8, 5, 6, 
    4, 3, 2, 1, 8, 7, 6, 5,
    5, 6, 7, 8, 1, 2, 3, 4, 
    6, 5, 8, 7, 2, 1, 4, 3, 
    7, 8, 5, 6, 3, 4, 1, 2, 
    8, 7, 6, 5, 4, 3, 2, 1;
  // clang-format on

  if (pg == (string) "d2h") {
    pointGroup = pg;
    product_table = fullD2hTable;
  }

  else if (pg == (string) "c2v") {
    // C2v A1: 1, B1:2, B2:3, A2:4 in MOLPRO. Indices here: A1: 0, B1:1, B2:2,
    // A2:3
    pointGroup = pg;
    product_table.resize(4, 4);
    product_table = (fullD2hTable.topRows(4)).leftCols(4);
  }

  else if (pg == (string) "c2h") {
    // C2h Ag: 1, Au: 2, Bu: 3, Bg: 4 in Molrpro. Indices here: Ag: 0, Au: 1,
    // Bu: 2, Bg: 3
    pointGroup = pg;
    product_table.resize(4, 4);
    product_table = (fullD2hTable.topRows(4)).leftCols(4);
  }

  else if (pg == (string) "d2") {
    // D2 A:1, B3:2, B2: 3, B1: 4 in Molrpro. Indices here: A:0, B3:1, B2: 2,
    // B1: 3
    pointGroup = pg;
    product_table.resize(4, 4);
    product_table = (fullD2hTable.topRows(4)).leftCols(4);
  }

  else if (pg == (string) "cs") {
    // Cs A':1, A'':2 in MOLPRO. Indices here A':0, A'':1
    pointGroup = pg;
    product_table.resize(2, 2);
    product_table = (fullD2hTable.topRows(2)).leftCols(2);
  }

  else if (pg == (string) "c2") {
    // C2 A:1, B:2 in MOLPRO. Indices here: A:0, B:1
    pointGroup = pg;
    product_table.resize(2, 2);
    product_table = (fullD2hTable.topRows(2)).leftCols(2);
  }

  else if (pg == (string) "ci") {
    // Ci Ag:1, Au:2 in MOLPRO. Indices here: Ag:0, Au:1
    pointGroup = pg;
    product_table.resize(2, 2);
    product_table = (fullD2hTable.topRows(2)).leftCols(2);
  }

  else if (pg == (string) "c1") {
    pointGroup = pg;
    product_table.resize(1, 1);
    product_table = (fullD2hTable.topRows(1)).leftCols(1);
  } else {
    pout << "WARNING: Dice doesn't support the point group " << pg
         << " when selecting a reference determinant." << endl;
    init_success = false;
    return;
  }

  // Check irreps to make sure they match the point group
  // If this fails, HF will be used as the ref determinant.
  for (auto ir : mol_irreps) {
    if (ir - 1 > product_table.rows()) {
      pout << "WARNING: Irrep " << ir
           << " doesn't match the specified point group symmetry " << pg
           << endl;
      init_success = false;
      break;
    }
  }

  // Set the numerical code for target irrep, if none set to 1.
  targetIrrepString = targetIrrepStr;
  targetIrrep = convertStringIrrepToInt(pointGroup, targetIrrepStr);
}

int symmetry::getProduct(int irrep1, int irrep2) {
  // Note here that the irrep and column differ by 1 because the indexing
  // starts at 0 for the product_table object.
  return symmetry::product_table(irrep1 - 1, irrep2 - 1);
}

int symmetry::getProduct(vector<int>& irreps) {  // TODO test this
  // For more than two irreps.
  if (irreps.size() > 2) {
    int irrep = irreps.back();
    irreps.pop_back();
    return symmetry::getProduct(symmetry::getProduct(irreps), irrep);
  }

  else {
    return symmetry::getProduct(irreps.at(0), irreps.at(1));
  }
}

int symmetry::getDetSymmetry(Determinant det) {
  int n_spin_orbs = moIrreps.size();
  int det_irrep = 1;

  for (int i = 0; i < n_spin_orbs; i++) {
    if (det.getocc(i)) {
      det_irrep = getProduct(det_irrep, moIrreps.at(i / 2));
    }
  }
  return det_irrep;
}

void symmetry::estimateLowestEnergyDet(int spin, oneInt I1, vector<int>& irrep,
                                       vector<int>& occupied,
                                       Determinant& Det) {
  // This method is for a single determinant and should be placed inside a
  // loop over all input desired determinants.

  vector<pair<double, int> > sort1Body(0);  // Double: E, int: original idx
#ifndef Complex
  for (int i = 0; i < I1.norbs; i++) {
    sort1Body.push_back(make_pair(I1(i, i), i));
  }
#else
  for (int i = 0; i < I1.norbs; i++) {
    sort1Body.push_back(make_pair(I1(i, i).real(), i));
  }
#endif
  sort(sort1Body.begin(), sort1Body.end(), compareForSortingEnergies);

  // Populate doubly occupied orbitals and ensure all others are empty
  int nDOrbs = occupied.size() - spin;
  int nSpinOrbs = I1.norbs;
  int nDOElec = occupied.size() - spin;

  for (int i = 0; i < I1.norbs; i++) {
    Det.setocc(i, false);
  }

  string error_message =
      "Given spin (" + to_string(spin) + ") targetting " + targetIrrepString +
      " irrep not possible for current active space ..." +
      "\nFilling determinants may lead to the targetting of undesired states.";

  // Spin dependent population of singly occupied orbitals.
  // Do this first and then fill doubly occupied orbs
  if (spin == 0) {
    // If we aren't targetting the totally symmetric irrep
    if (targetIrrep != 1) {
      nDOElec -= 2;  // Take two electrons and upair them so we can find the
                     // right irrep
      bool found_irrep = false;
      // Find lowest energy orbitals with the appropriate symmetry
      for (int i = 0; i < I1.norbs && !found_irrep; i++) {
        int A = sort1Body.at(i).second / 2;
        int a = sort1Body.at(i).second;
        for (int j = i + 1; j < I1.norbs; j++) {
          int B = sort1Body.at(j).second / 2;
          int b = sort1Body.at(j).second;

          int det_irrep = getProduct(irrep.at(A), irrep.at(B));

          // Not the most efficient, but it keeps the code clean
          bool same_spin = a % 2 != b % 2;

          if (det_irrep == targetIrrep && same_spin) {
            Det.setocc(a, true);
            Det.setocc(b, true);
            found_irrep = true;
            break;
          }
        }
      }

      if (!found_irrep) {
        nDOElec += 2;  // Return DO elec
        for (int i = 0; i < nDOElec; i++) {
          Det.setocc(i, true);
        }
        pout << error_message << endl;
        return;
      }
    }

    int I = 0;
    while (I < I1.norbs / 2 && nDOElec > 0) {
      int a = sort1Body.at(2 * I).second;
      int b = sort1Body.at(2 * I + 1).second;
      if (!Det.getocc(a) && !Det.getocc(b)) {
        Det.setocc(a, true);
        Det.setocc(b, true);
        nDOElec -= 2;
      }

      I++;
    }
    return;
  } else if (spin == 1) {
    bool found_irrep = false;

    // Find lowest energy orbitals with the appropriate symmetry
    for (int i = 0; i < I1.norbs; i++) {
      int A = sort1Body.at(i).second / 2;
      int a = sort1Body.at(i).second;
      if (irrep.at(A) == targetIrrep && (Det.getocc(a) == false)) {
        Det.setocc(a, true);
        found_irrep = true;
        break;
      }
    }

    if (!found_irrep) {
      for (int i = 0; i < nDOElec; i++) {
        Det.setocc(i, true);
      }
      for (int i = 0; i < spin; i++) {
        Det.setocc(nDOElec + 2 * i, true);
      }
      pout << error_message << endl;
    }

    int I = 0;
    while (I < I1.norbs / 2 && nDOElec > 0) {
      int a = sort1Body.at(2 * I).second;
      int b = sort1Body.at(2 * I + 1).second;
      if (!Det.getocc(a) && !Det.getocc(b)) {
        Det.setocc(a, true);
        Det.setocc(b, true);
        nDOElec -= 2;
      }

      I++;
    }
    return;
  } else if (spin == 2) {
    bool found_irrep = false;
    // Find lowest energy orbitals with the appropriate symmetry
    for (int i = 0; i < I1.norbs && !found_irrep; i++) {
      int A = sort1Body.at(i).second / 2;
      int a = sort1Body.at(i).second;
      for (int j = i + 1; j < I1.norbs; j++) {
        int B = sort1Body.at(j).second / 2;
        int b = sort1Body.at(j).second;

        int det_irrep = getProduct(irrep.at(A), irrep.at(B));

        // Not the most efficient, but it keeps the code clean
        bool same_spin = a % 2 == b % 2;

        if (det_irrep == targetIrrep && same_spin) {
          Det.setocc(a, true);
          Det.setocc(b, true);
          found_irrep = true;
          break;
        }
      }
    }

    if (!found_irrep) {
      for (int i = 0; i < nDOElec; i++) {
        Det.setocc(i, true);
      }
      for (int i = 0; i < spin; i++) {
        Det.setocc(nDOElec + 2 * i, true);
      }
      pout << error_message << endl;
    }

    int I = 0;
    while (I < I1.norbs / 2 && nDOElec > 0) {
      int a = sort1Body.at(2 * I).second;
      int b = sort1Body.at(2 * I + 1).second;
      if (!Det.getocc(a) && !Det.getocc(b)) {
        Det.setocc(a, true);
        Det.setocc(b, true);
        nDOElec -= 2;
      }

      I++;
    }
    return;
  } else if (spin == 3) {
    bool found_irrep = false;
    // Find lowest energy orbitals with the appropriate symmetry
    for (int i = 0; i < I1.norbs && !found_irrep; i++) {
      int A = sort1Body.at(i).second / 2;
      int a = sort1Body.at(i).second;
      for (int j = i + 1; j < I1.norbs && !found_irrep; j++) {
        int B = sort1Body.at(j).second / 2;
        int b = sort1Body.at(j).second;
        for (int k = j + 1; k < I1.norbs; k++) {
          int C = sort1Body.at(k).second / 2;
          int c = sort1Body.at(k).second;

          int det_irrep = getProduct(irrep.at(A), irrep.at(B));
          det_irrep = getProduct(det_irrep, irrep.at(C));

          // Not the most efficient, but it keeps the code clean
          bool same_spin = (a % 2 == b % 2 && b % 2 == c % 2 && a % 2 == 0);

          if (det_irrep == targetIrrep && same_spin) {
            Det.setocc(a, true);
            Det.setocc(b, true);
            Det.setocc(c, true);
            found_irrep = true;
            break;
          }
        }
      }
    }

    if (!found_irrep) {
      for (int i = 0; i < nDOElec; i++) {
        Det.setocc(i, true);
      }
      for (int i = 0; i < spin; i++) {
        Det.setocc(nDOElec + 2 * i, true);
      }
      pout << error_message << endl;
    }

    int I = 0;
    while (I < I1.norbs / 2 && nDOElec > 0) {
      int a = sort1Body.at(2 * I).second;
      int b = sort1Body.at(2 * I + 1).second;
      if (!Det.getocc(a) && !Det.getocc(b)) {
        Det.setocc(a, true);
        Det.setocc(b, true);
        nDOElec -= 2;
      }

      I++;
    }
    return;
  } else if (spin == 4) {
    bool found_irrep = false;

    // Find lowest energy orbitals with the appropriate symmetry
    for (int i = 0; i < I1.norbs && !found_irrep; i++) {
      int A = sort1Body.at(i).second / 2;
      int a = sort1Body.at(i).second;
      for (int j = i + 1; j < I1.norbs && !found_irrep; j++) {
        int B = sort1Body.at(j).second / 2;
        int b = sort1Body.at(j).second;
        for (int k = j + 1; k < I1.norbs && !found_irrep; k++) {
          int C = sort1Body.at(k).second / 2;
          int c = sort1Body.at(k).second;
          for (int l = k + 1; l < I1.norbs; l++) {
            int D = sort1Body.at(l).second / 2;
            int d = sort1Body.at(l).second;

            int det_irrep = getProduct(irrep.at(A), irrep.at(B));
            det_irrep = getProduct(det_irrep, irrep.at(C));
            det_irrep = getProduct(det_irrep, irrep.at(D));

            // Not the most efficient, but it keeps the code clean
            bool same_spin = (a % 2 == b % 2 && b % 2 == c % 2 &&
                              c % 2 == d % 2 && a % 2 == 0);

            if (det_irrep == targetIrrep && same_spin) {
              Det.setocc(a, true);
              Det.setocc(b, true);
              Det.setocc(c, true);
              Det.setocc(d, true);
              found_irrep = true;
              break;
            }
          }
        }
      }
    }

    if (!found_irrep) {
      for (int i = 0; i < nDOElec; i++) {
        Det.setocc(i, true);
      }
      for (int i = 0; i < spin; i++) {
        Det.setocc(nDOElec + 2 * i, true);
      }
      pout << error_message << endl;
    }

    int I = 0;
    while (I < I1.norbs / 2 && nDOElec > 0) {
      int a = sort1Body.at(2 * I).second;
      int b = sort1Body.at(2 * I + 1).second;
      if (!Det.getocc(a) && !Det.getocc(b)) {
        Det.setocc(a, true);
        Det.setocc(b, true);
        nDOElec -= 2;
      }

      I++;
    }
    return;
  } else {
    pout << "WARNING: Finding the lowest energy determinant not supported for "
            "spin="
         << spin << " using the specified determinant." << endl;

    for (auto i : occupied) {
      Det.setocc(i, true);
    }
    return;
  }
};
