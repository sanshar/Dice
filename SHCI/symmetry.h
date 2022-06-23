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
#ifndef symmetry_HEADER_H
#define symmetry_HEADER_H

#include <stdio.h>
#include <stdlib.h>

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "Determinants.h"
#include "SHCIgetdeterminants.h"
#include "global.h"

using namespace Eigen;
using namespace std;

class oneInt;

bool compareForSortingEnergies(const pair<double, int>& a,
                               const pair<double, int>& b);

class symmetry {
 private:
  // Full table for d2h and all subgroups
  MatrixXd fullD2hTable;

  // The irreps for each point group
  map<string, vector<string>> pg_irreps;
  void init_pg_irreps() {
    pg_irreps["d2h"] = {"Ag", "B3u", "B2u", "B1g", "B1u", "B2g", "B3g", "Au"};
    pg_irreps["c2v"] = {"A1", "B1", "B2", "A2"};
    pg_irreps["cs"] = {"A'", "A''"};
    pg_irreps["c2"] = {"A", "B"};
    pg_irreps["ci"] = {"Ag", "Au"};
    pg_irreps["c1"] = {"A"};
  };

 public:
  bool init_success;  // Basically, are we going to search for the lowest energy
                      // determinant, by irrep and spin
  static MatrixXd product_table;  // product table for a given symmetry
  string pointGroup;              // eg. d2h, c2
  int targetIrrep;                // Numerical code for target irrep
  string targetIrrepString;       // String code for the target irrep
  vector<int> moIrreps;

  symmetry(string pointGroup, vector<int>& mol_irreps,
           string targetIrrepStr);  // product and sym is initialized
  int getProduct(int, int);
  int getProduct(vector<int>&);
  int getDetSymmetry(Determinant det);
  void estimateLowestEnergyDet(int spin, oneInt I1, vector<int>& irrep,
                               vector<int>& occupied, Determinant& Det);
  int convertStringIrrepToInt(string pg, string irrep);

  void checkTargetStates(vector<Determinant>& Dets, int spin) {
    // Check HF dets to see if they all have the same irrep and spin.
    int spin_det0 = Dets[0].Nalpha() - Dets[0].Nbeta();
    int irrep_det0 = getDetSymmetry(Dets[0]);
    bool spin_pars_match = true;
    bool irrep_pars_match = true;

    for (auto d : Dets) {
      if (spin_det0 != (d.Nalpha() - d.Nbeta())) {
        spin_pars_match = false;
      }
      if (irrep_det0 != getDetSymmetry(d)) {
        irrep_pars_match = false;
      }
    }

    // If either don't match skip all reference determinant search stuff and
    // just use the given starting dets.
    if (!spin_pars_match || !irrep_pars_match) {
      cout << "WARNING: Multiple spin/irreps used in given reference "
              "determinants."
           << endl;
      init_success = false;
    }
  };
};

#endif
