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

class heatBathFCIQMC {
  public:
    vector<double> D_pq;

    vector<double> S_p;

    vector<double> P_same_r_pq;
    vector<double> P_opp_r_pq;

    vector<double> P_same_s_pqr;
    vector<double> P_opp_s_pqr;

    vector<double> H_tot_same_rpq;
    vector<double> H_tot_opp_rpq;


    heatBathFCIQMC(int norbs, int nalpha) {

      // Size of the D_pq array
      int size_1 = norbs*(norbs-1)/2;
      // Size of the P_same_r_pq array
      int size_2 = nalpha * nalpha*(nalpha-1)/2;
      // Size of the P_opp_r_pq array
      int size_3 = pow(nalpha,3);
      // Size of the P_same_s_pqr array
      int size_4 = pow(nalpha,2) * nalpha*(nalpha-1)/2;
      // Size of the P_opp_s_pqr array
      int size_5 = pow(nalpha,4);

      D_pq.resize(size_1, 0.0);
      S_p.resize(norbs, 0.0);
      P_same_r_pq.resize(size_2, 0.0);
      P_opp_r_pq.resize(size_3, 0.0);
      P_same_s_pqr.resize(size_4, 0.0);
      P_opp_s_pqr.resize(size_5, 0.0);
      H_tot_same_rpq.resize(size_2);
      H_tot_opp_rpq.resize(size_3);

      // Set up D_pq
      for (int p=1; p<norbs; p++) {
        for (int q=0; q<p; q++) {
          int ind = p*(p-1)/2 + q;
          D_pq.at(ind) = 0.0;

          for (int r=0; r<norbs; r++) {
            for (int s=0; s<norbs; s++) {
              D_pq.at(ind) += fabs( I2(r, s, p, q) );
            }
          }

        }
      }

      // Set up S_p
      for (int p=0; p<norbs; p++) {
        S_p.at(p) = 0.0;

        for (int q=0; q<norbs; q++) {
          if (p == q) continue;

          int Q = min(p,q);
          int P = max(p,q);
          int ind = P*(P-1)/2 + Q;

          S_p.at(p) += D_pq.at(ind);
        }
      }

      // Set up P_same_r_pq
      for (int p=1; p<nalpha; p++) {
        for (int q=0; q<p; q++) {

          int ind_pq = p*(p-1)/2 + q;

          for (int r=0; r<nalpha; r++) {

            int ind = nalpha*ind_pq + r;
            P_same_r_pq.at(ind) = 0.0;

            for (int s=0; s<nalpha; s++) {
              if (r == s) continue;

              if (r != p && s != q) {
                if (r != q && s != p) {
                  P_same_r_pq.at(ind) += fabs( I2(2*r, 2*s, 2*p, 2*q) );
                }
              }

            } // Loop over s

            int ind_Dpq = (2*p)*(2*p-1)/2 + 2*q;
            P_same_r_pq.at(ind) /= D_pq.at(ind_Dpq);

          } // Loop over r

        } // Loop over q
      } // Loop over p


      // Set up P_opp_r_pq
      for (int p=0; p<nalpha; p++) {
        for (int q=0; q<nalpha; q++) {

          int ind_pq = p*nalpha + q;

          for (int r=0; r<nalpha; r++) {

            int ind = nalpha*ind_pq + r;
            P_opp_r_pq.at(ind) = 0.0;

            for (int s=0; s<nalpha; s++) {
              if (r == s) continue;

              if (r != p && s != q) {
                P_opp_r_pq.at(ind) += fabs( I2(2*r, 2*s+1, 2*p, 2*q+1) );
              }

            } // Loop over s

            int ind_Dpq = (2*p)*(2*p-1)/2 + 2*q+1;
            P_opp_r_pq.at(ind) /= D_pq.at(ind_Dpq);

          } // Loop over r

        } // Loop over q
      } // Loop over p


      // Set up P_same_s_pqr
      for (int p=1; p<nalpha; p++) {
        for (int q=0; q<p; q++) {

          int ind_pq = p*(p-1)/2 + q;

          for (int r=0; r<nalpha; r++) {

            double tot_sum = 0.0;

            for (int s=0; s<nalpha; s++) {
              if (r == s) continue;

              int ind = pow(nalpha,2) * ind_pq + nalpha*r + s;

              if (r != p && s != q) {
                if (r != q && s != p) {
                  P_same_s_pqr.at(ind) = fabs( I2(2*r, 2*s, 2*p, 2*q) );
                  tot_sum += fabs( I2(2*r, 2*s, 2*p, 2*q) );
                }
              }

            } // Loop over s

            // Normalize probability
            for (int s=0; s<nalpha; s++) {
              int ind = pow(nalpha,2) * ind_pq + nalpha*r + s;
              P_same_s_pqr.at(ind) /= tot_sum;
            }

            int ind_tot = nalpha*ind_pq + r;
            H_tot_same_rpq.at(ind_tot) = tot_sum;

          } // Loop over r

        } // Loop over q
      } // Loop over p


      // Set up P_opp_s_pqr
      for (int p=0; p<nalpha; p++) {
        for (int q=0; q<nalpha; q++) {

          int ind_pq = p*nalpha + q;

          for (int r=0; r<nalpha; r++) {

            double tot_sum = 0.0;

            for (int s=0; s<nalpha; s++) {
              if (r == s) continue;

              int ind = pow(nalpha,2) * ind_pq + nalpha*r + s;

              if (r != p && s != q) {
                P_opp_s_pqr.at(ind) = fabs( I2(2*r, 2*s+1, 2*p, 2*q+1) );
                tot_sum += fabs( I2(2*r, 2*s+1, 2*p, 2*q+1) );
              }

            } // Loop over s

            // Normalize probability
            for (int s=0; s<nalpha; s++) {
              int ind = pow(nalpha,2) * ind_pq + nalpha*r + s;
              P_opp_s_pqr.at(ind) /= tot_sum;
            }

            int ind_tot = nalpha*ind_pq + r;
            H_tot_opp_rpq.at(ind_tot) = tot_sum;

          } // Loop over r

        } // Loop over q
      } // Loop over p

    } // End of contrusctor

};

void generateExcitation(const Determinant& parentDet, Determinant& childDet, double& pgen);
void generateSingleExcit(const Determinant& parentDet, Determinant& childDet, double& pgen_ia);
void generateDoubleExcit(const Determinant& parentDet, Determinant& childDet, double& pgen_ijab);

// Generate a random single or double excitation, and also return the
// probability that it was generated
void generateExcitation(const Determinant& parentDet, Determinant& childDet, double& pgen)
{
  double pSingle = 0.05;
  double pgen_ia, pgen_ijab;

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));

  if (random() < pSingle) {
    generateSingleExcit(parentDet, childDet, pgen_ia);
    pgen = pSingle * pgen_ia;
  } else {
    generateDoubleExcit(parentDet, childDet, pgen_ijab);
    pgen = (1 - pSingle) * pgen_ijab;
  }
}

// Generate a random single excitation, and also return the probability that
// it was generated
void generateSingleExcit(const Determinant& parentDet, Determinant& childDet, double& pgen_ia)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));

  vector<int> AlphaOpen;
  vector<int> AlphaClosed;
  vector<int> BetaOpen;
  vector<int> BetaClosed;

  parentDet.getOpenClosedAlphaBeta(AlphaOpen, AlphaClosed, BetaOpen, BetaClosed);

  childDet   = parentDet;
  int nalpha = AlphaClosed.size();
  int nbeta  = BetaClosed.size();
  int norbs  = Determinant::norbs;

  // Pick a random occupied orbital
  int i = floor(random() * (nalpha + nbeta));
  double pgen_i = 1.0/(nalpha + nbeta);

  // Pick an unoccupied orbital
  if (i < nalpha) // i is alpha
  {
    int a = floor(random() * (norbs - nalpha));
    int I = AlphaClosed[i];
    int A = AlphaOpen[a];

    childDet.setoccA(I, false);
    childDet.setoccA(A, true);
    pgen_ia = pgen_i / (norbs - nalpha);
  }
  else // i is beta
  {
    i = i - nalpha;
    int a = floor( random() * (norbs - nbeta));
    int I = BetaClosed[i];
    int A = BetaOpen[a];

    childDet.setoccB(I, false);
    childDet.setoccB(A, true);
    pgen_ia = pgen_i / (norbs - nbeta);
  }

  //cout << "parent:  " << parentDet << endl;
  //cout << "child:   " << childDet << endl;
}

// Generate a random double excitation, and also return the probability that
// it was generated
void generateDoubleExcit(const Determinant& parentDet, Determinant& childDet, double& pgen_ijab)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));

  vector<int> AlphaOpen;
  vector<int> AlphaClosed;
  vector<int> BetaOpen;
  vector<int> BetaClosed;

  int i, j, a, b, I, J, A, B;

  parentDet.getOpenClosedAlphaBeta(AlphaOpen, AlphaClosed, BetaOpen, BetaClosed);

  childDet   = parentDet;
  int nalpha = AlphaClosed.size();
  int nbeta  = BetaClosed.size();
  int norbs  = Determinant::norbs;
  int nel    = nalpha + nbeta;

  // Pick a combined ij index
  int ij = floor( random() * (nel*(nel-1))/2 ) + 1;
  // The probability of having picked this pair
  double pgen_ij = 2.0 / (nel * (nel-1));

  // Use triangular indexing scheme to obtain (i,j), with j>i
  j = floor(1.5 + sqrt(2*ij - 1.75)) - 1;
  i = ij - (j * (j - 1))/2 - 1;

  bool iAlpha = i < nalpha;
  bool jAlpha = j < nalpha;
  bool sameSpin = iAlpha == jAlpha;

  // Pick a and b
  if (sameSpin) {
    int nvirt;
    if (iAlpha)
    {
      nvirt = norbs - nalpha;
      // Pick a combined ab index
      int ab = floor( random() * (nvirt*(nvirt-1))/2 ) + 1;

      // Use triangular indexing scheme to obtain (a,b), with b>a
      b = floor(1.5 + sqrt(2*ab - 1.75)) - 1;
      a = ab - (b * (b - 1))/2 - 1;

      I = AlphaClosed[i];
      J = AlphaClosed[j];
      A = AlphaOpen[a];
      B = AlphaOpen[b];
    }
    else
    {
      i = i - nalpha;
      j = j - nalpha;

      nvirt = norbs - nbeta;
      // Pick a combined ab index
      int ab = floor( random() * (nvirt * (nvirt-1))/2 ) + 1;

      // Use triangular indexing scheme to obtain (a,b), with b>a
      b = floor(1.5 + sqrt(2*ab - 1.75)) - 1;
      a = ab - (b * (b - 1))/2 - 1;

      I = BetaClosed[i];
      J = BetaClosed[j];
      A = BetaOpen[a];
      B = BetaOpen[b];
    }
    pgen_ijab = pgen_ij * 2.0 / (nvirt * (nvirt-1));
  }
  else
  { // Opposite spin
    if (iAlpha) {
      a = floor(random() * (norbs - nalpha));
      I = AlphaClosed[i];
      A = AlphaOpen[a];

      j = j - nalpha;
      b = floor( random() * (norbs - nbeta));
      J = BetaClosed[j];
      B = BetaOpen[b];
    }
    else
    {
      i = i - nalpha;
      a = floor( random() * (norbs - nbeta));
      I = BetaClosed[i];
      A = BetaOpen[a];

      b = floor(random() * (norbs - nalpha));
      J = AlphaClosed[j];
      B = AlphaOpen[b];
    }
    pgen_ijab = pgen_ij / ( (norbs - nalpha) * (norbs - nbeta) );
  }

  if (iAlpha) {
    childDet.setoccA(I, false);
    childDet.setoccA(A, true);
  } else {
    childDet.setoccB(I, false);
    childDet.setoccB(A, true);
  }

  if (jAlpha) {
    childDet.setoccA(J, false);
    childDet.setoccA(B, true);
  } else {
    childDet.setoccB(J, false);
    childDet.setoccB(B, true);
  }

  //cout << "parent:  " << parentDet << endl;
  //cout << "child:   " << childDet << endl;
}
