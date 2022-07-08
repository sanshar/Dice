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

#ifndef excitFCIQMC_HEADER_H
#define excitFCIQMC_HEADER_H

// Map two integers to a single integer by a one-to-one mapping,
// using the triangular indexing approach
inline int triInd(const int p, const int q)
{
  int Q = min(p,q);
  int P = max(p,q);
  return P*(P-1)/2 + Q;
}

// This class holds the arrays needed to use heat bath excitation generators.
// This was described in J. Chem. Theory Comput., 2016, 12 (4), 1561.
// Arrays are labelled by the same names as in the above paper.
class heatBathFCIQMC {
  public:
    // Used to construct probabilities for selecting the first electron, p.
    vector<double> S_p;

    // Used to construct probabilities for selecting the second electron, q,
    // given that p has already been chosen.
    vector<double> D_pq;

    // The probabilities for selecting the first unoccupied orbital, r,
    // given that p and q have been chosen already.
    // Depending on whether p and q have the same or opposite spin, use the
    // 'same' or 'opp' objects.
    // Note that r is chosen such that it has the same spin as p.
    vector<double> P_same_r_pq;
    vector<double> P_opp_r_pq;
    // Cumulative arrays
    vector<double> P_same_r_pq_cum;
    vector<double> P_opp_r_pq_cum;

    // The probabilities for selecting the second unoccupied orbital, s,
    // given that p, q and r have been chosen already.
    // Depending on whether p and q have the same or opposite spin, use the
    // 'same' or 'opp' objects.
    vector<double> P_same_s_pqr;
    vector<double> P_opp_s_pqr;
    // Cumulative arrays
    vector<double> P_same_s_pqr_cum;
    vector<double> P_opp_s_pqr_cum;

    // These objects are used to decide whether to generate a single or double
    // excitation, after first choosing p, q and r.
    vector<double> H_tot_same;
    vector<double> H_tot_opp;

    // Constructors
    heatBathFCIQMC() {}
    heatBathFCIQMC(int norbs, const twoInt& I2);

    void createArrays(int norbs, const twoInt& I2);
};


void generateExcitation(const heatBathFCIQMC& hb, const oneInt& I1, const twoInt& I2,
                        const Determinant& parentDet, const int nel, Determinant& childDet,
                        Determinant& childDet2, double& pGen, double& pGen2, int& ex1, int& ex2);

void generateExcitationSingDoub(const heatBathFCIQMC& hb, const oneInt& I1, const twoInt& I2,
                                const Determinant& parentDet, const int nel, Determinant& childDet,
                                Determinant& childDet2, double& pgen, double& pgen2,
                                int& ex1, int& ex2);

void generateSingleExcit(const Determinant& parentDet, Determinant& childDet, double& pgen_ia, int& ex1);

void generateDoubleExcit(const Determinant& parentDet, Determinant& childDet, double& pgen_ijab,
                         int& ex1, int& ex2);

void pickROrbitalHB(const heatBathFCIQMC& hb, const int norbs, const int p, const int q, int& r,
                    double& rProb, double& H_tot_pqr);

void pickSOrbitalHB(const heatBathFCIQMC& hb, const int norbs, const int p, const int q, const int r,
                    int& s, double& sProb);

double calcSinglesProb(const heatBathFCIQMC& hb, const oneInt& I1, const twoInt& I2, const int norbs,
                       const vector<int>& closed, const double pProb, const double D_pq_tot,
                       const double hSingAbs, const int p, const int r);

double calcProbDouble(const Determinant& parentDet, const oneInt& I1, const twoInt& I2,
                      const double H_tot_pqr, const int p, const int r);

void calcProbsForRAndS(const heatBathFCIQMC& hb, const oneInt& I1, const twoInt& I2, const int norbs,
                       const Determinant& parentDet, const int p, const int q, const int r, const int s,
                       double& rProb, double& sProb, double& doubleProb, const bool calcDoubleProb);

void generateExcitHB(const heatBathFCIQMC& hb, const oneInt& I1, const twoInt& I2, const Determinant& parentDet,
                     const int nel, Determinant& childDet, Determinant& childDet2, double& pGen, double& pGen2,
                     int& ex1, int& ex2, const bool attemptSingleExcit);

#endif
