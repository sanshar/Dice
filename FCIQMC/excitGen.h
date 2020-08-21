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
    double S_p_tot;

    vector<double> P_same_r_pq;
    vector<double> P_opp_r_pq;
    // Cumulative arrays
    vector<double> P_same_r_pq_cum;
    vector<double> P_opp_r_pq_cum;

    vector<double> P_same_s_pqr;
    vector<double> P_opp_s_pqr;
    // Cumulative arrays
    vector<double> P_same_s_pqr_cum;
    vector<double> P_opp_s_pqr_cum;

    vector<double> H_tot_same_rpq;
    vector<double> H_tot_opp_rpq;


    heatBathFCIQMC(int norbs) {

      int nSpinOrbs = 2*norbs;

      // Size of the D_pq array
      int size_1 = nSpinOrbs*(nSpinOrbs-1)/2;
      // Size of the P_same_r_pq array
      int size_2 = norbs * norbs*(norbs-1)/2;
      // Size of the P_opp_r_pq array
      int size_3 = pow(norbs,3);
      // Size of the P_same_s_pqr array
      int size_4 = pow(norbs,2) * norbs*(norbs-1)/2;
      // Size of the P_opp_s_pqr array
      int size_5 = pow(norbs,4);

      D_pq.resize(size_1, 0.0);
      S_p.resize(nSpinOrbs, 0.0);

      P_same_r_pq.resize(size_2, 0.0);
      P_opp_r_pq.resize(size_3, 0.0);
      P_same_r_pq_cum.resize(size_2, 0.0);
      P_opp_r_pq_cum.resize(size_3, 0.0);

      P_same_s_pqr.resize(size_4, 0.0);
      P_opp_s_pqr.resize(size_5, 0.0);
      P_same_s_pqr_cum.resize(size_4, 0.0);
      P_opp_s_pqr_cum.resize(size_5, 0.0);

      H_tot_same_rpq.resize(size_2, 0.0);
      H_tot_opp_rpq.resize(size_3, 0.0);

      // Set up D_pq
      for (int p=1; p<nSpinOrbs; p++) {
        for (int q=0; q<p; q++) {
          int ind = p*(p-1)/2 + q;
          D_pq.at(ind) = 0.0;

          for (int r=0; r<nSpinOrbs; r++) {
            for (int s=0; s<nSpinOrbs; s++) {
              D_pq.at(ind) += fabs( I2(r, p, s, q) );
            }
          }

        }
      }

      // Set up S_p
      S_p_tot = 0.0;
      for (int p=0; p<nSpinOrbs; p++) {
        S_p.at(p) = 0.0;

        for (int q=0; q<nSpinOrbs; q++) {
          if (p == q) continue;

          int Q = min(p,q);
          int P = max(p,q);
          int ind = P*(P-1)/2 + Q;

          S_p.at(p) += D_pq.at(ind);
        }
        S_p_tot += S_p.at(p);
      }

      // Set up P_same_r_pq
      for (int p=1; p<norbs; p++) {
        for (int q=0; q<p; q++) {

          int ind_pq = p*(p-1)/2 + q;
          double tot = 0.0;

          for (int r=0; r<norbs; r++) {

            int ind = norbs*ind_pq + r;
            P_same_r_pq.at(ind) = 0.0;

            for (int s=0; s<norbs; s++) {
              if (r == s) continue;

              if (r != p && s != q && r != q && s != p) {
                P_same_r_pq.at(ind) += fabs( I2(2*r, 2*p, 2*s, 2*q) );
              }

            } // Loop over s

            tot += P_same_r_pq.at(ind);

          } // Loop over r

          // Normalize probabilities
          for (int r=0; r<norbs; r++) {
            int ind = norbs*ind_pq + r;
            P_same_r_pq.at(ind) /= tot;
          }

        } // Loop over q
      } // Loop over p

      // Set up the cumulative arrays for P_same_r_pq
      for (int p=1; p<norbs; p++) {
        for (int q=0; q<p; q++) {

          int ind_pq = p*(p-1)/2 + q;
          double tot = 0.0;

          for (int r=0; r<norbs; r++) {

            int ind = norbs*ind_pq + r;
            tot += P_same_r_pq.at(ind);
            P_same_r_pq_cum.at(ind) = tot;

          } // Loop over r
          //cout << "Check: p: " << p << "  q: " << q << "  tot: " << tot << endl;
        } // Loop over q
      } // Loop over p

      // Set up P_opp_r_pq
      for (int p=0; p<norbs; p++) {
        for (int q=0; q<norbs; q++) {

          int ind_pq = p*norbs + q;
          double tot = 0.0;

          for (int r=0; r<norbs; r++) {

            int ind = norbs*ind_pq + r;
            P_opp_r_pq.at(ind) = 0.0;

            for (int s=0; s<norbs; s++) {
              if (r == s) continue;

              if (r != p && s != q) {
                P_opp_r_pq.at(ind) += fabs( I2(2*r, 2*p, 2*s+1, 2*q+1) );
              }

            } // Loop over s

            tot += P_opp_r_pq.at(ind);

          } // Loop over r

          // Normalize probabilities
          for (int r=0; r<norbs; r++) {
            int ind = norbs*ind_pq + r;
            P_opp_r_pq.at(ind) /= tot;
          }

        } // Loop over q
      } // Loop over p

      // Set up the cumulative arrays for P_opp_r_pq
      for (int p=0; p<norbs; p++) {
        for (int q=0; q<norbs; q++) {

          int ind_pq = p*norbs + q;
          double tot = 0.0;

          for (int r=0; r<norbs; r++) {

            int ind = norbs*ind_pq + r;
            tot += P_opp_r_pq.at(ind);
            P_opp_r_pq_cum.at(ind) = tot;

          } // Loop over r
          //cout << "Check: p: " << p << "  q: " << q << "  tot: " << tot << endl;
        } // Loop over q
      } // Loop over p


      // Set up P_same_s_pqr
      for (int p=1; p<norbs; p++) {
        for (int q=0; q<p; q++) {

          int ind_pq = p*(p-1)/2 + q;

          for (int r=0; r<norbs; r++) {

            double tot_sum = 0.0;

            for (int s=0; s<norbs; s++) {
              if (r == s) continue;

              int ind = pow(norbs,2) * ind_pq + norbs*r + s;

              if (r != p && s != q && r != q && s != p) {
                P_same_s_pqr.at(ind) = fabs( I2(2*r, 2*p, 2*s, 2*q) );
                tot_sum += fabs( I2(2*r, 2*p, 2*s, 2*q) );
              }

            } // Loop over s

            // Normalize probability
            for (int s=0; s<norbs; s++) {
              if (r != p && s != q && r != q && s != p) {
                int ind = pow(norbs,2) * ind_pq + norbs*r + s;
                P_same_s_pqr.at(ind) /= tot_sum;
              }
            }

            int ind_tot = norbs*ind_pq + r;
            H_tot_same_rpq.at(ind_tot) = tot_sum;

          } // Loop over r

        } // Loop over q
      } // Loop over p

      // Set up the cumulative arrays for P_same_s_pqr
      for (int p=1; p<norbs; p++) {
        for (int q=0; q<p; q++) {

          int ind_pq = p*(p-1)/2 + q;

          for (int r=0; r<norbs; r++) {
            double tot = 0.0;

            for (int s=0; s<norbs; s++) {
              int ind = pow(norbs,2) * ind_pq + norbs*r + s;
              tot += P_same_s_pqr.at(ind);
              P_same_s_pqr_cum.at(ind) = tot;
            }
            //cout << "Check: p: " << p << "  q: " << q << "  r: " << r << "  tot: " << tot << endl;
          } // Loop over r
        } // Loop over q
      } // Loop over p


      // Set up P_opp_s_pqr
      for (int p=0; p<norbs; p++) {
        for (int q=0; q<norbs; q++) {

          int ind_pq = p*norbs + q;

          for (int r=0; r<norbs; r++) {

            double tot_sum = 0.0;

            for (int s=0; s<norbs; s++) {
              if (r == s) continue;

              int ind = pow(norbs,2) * ind_pq + norbs*r + s;

              if (r != p && s != q) {
                P_opp_s_pqr.at(ind) = fabs( I2(2*r, 2*p, 2*s+1, 2*q+1) );
                tot_sum += fabs( I2(2*r, 2*p, 2*s+1, 2*q+1) );
              }

            } // Loop over s

            // Normalize probability
            for (int s=0; s<norbs; s++) {
              if (r != p && s != q) {
                int ind = pow(norbs,2) * ind_pq + norbs*r + s;
                P_opp_s_pqr.at(ind) /= tot_sum;
              }
            }

            int ind_tot = norbs*ind_pq + r;
            H_tot_opp_rpq.at(ind_tot) = tot_sum;

          } // Loop over r

        } // Loop over q
      } // Loop over p

      // Set up the cumulative arrays for P_opp_s_pqr
      for (int p=0; p<norbs; p++) {
        for (int q=0; q<norbs; q++) {

          int ind_pq = p*norbs + q;

          for (int r=0; r<norbs; r++) {
            double tot = 0.0;

            for (int s=0; s<norbs; s++) {
              int ind = pow(norbs,2) * ind_pq + norbs*r + s;
              tot += P_opp_s_pqr.at(ind);
              P_opp_s_pqr_cum.at(ind) = tot;
            }
            cout << "Check: p: " << p << "  q: " << q << "  r: " << r << "  tot: " << tot << endl;
          } // Loop over r
        } // Loop over q
      } // Loop over p

    } // End of contrusctor

};

void generateExcitation(const Determinant& parentDet, Determinant& childDet, double& pgen);
void generateSingleExcit(const Determinant& parentDet, Determinant& childDet, double& pgen_ia);
void generateDoubleExcit(const Determinant& parentDet, Determinant& childDet, double& pgen_ijab);

void pickROrbitalHB(heatBathFCIQMC& hb, const int norbs, const int p, const int q, int& r, double& rProb);
void pickSOrbitalHB(heatBathFCIQMC& hb, const int norbs, const int p, const int q, const int r, int& s, double& sProb);
void generateDoubleExcitHB(const Determinant& parentDet, Determinant& childDet, double& pgen_pqrs);

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

void pickROrbitalHB(heatBathFCIQMC& hb, const int norbs, const int p, const int q, int& r, double& rProb)
{
  int ind, ind_pq, rSpatial;

  bool sameSpin = (p%2 == q%2);
  int pSpatial = p/2;
  int qSpatial = q/2;

  // Pick a spin-orbital r from P(r|pq), such that r and p have the same spin
  if (sameSpin) {

    int Q = min(pSpatial, qSpatial);
    int P = max(pSpatial, qSpatial);
    ind_pq = P*(P-1)/2 + Q;

    // The first index for pair (p,q)
    int ind_pq_low = ind_pq*norbs;
    // The last index for pair (p,q)
    int ind_pq_high = ind_pq_low + norbs - 1;

    double rRand = random();
    rSpatial = std::lower_bound((hb.P_same_r_pq_cum.begin() + ind_pq_low),
                                (hb.P_same_r_pq_cum.begin() + ind_pq_high), rRand)
                                - hb.P_same_r_pq_cum.begin();

    // The probability that this electron was chosen
    ind = norbs*ind_pq + rSpatial;
    double rProb = hb.P_same_r_pq.at(ind);
  } else {

    ind_pq = pSpatial*norbs + qSpatial;
    // The first index for pair (p,q)
    int ind_pq_low = ind_pq*norbs;
    // The last index for pair (p,q)
    int ind_pq_high = ind_pq_low + norbs - 1;

    double rRand = random();
    rSpatial = std::lower_bound((hb.P_opp_r_pq_cum.begin() + ind_pq_low),
                                (hb.P_opp_r_pq_cum.begin() + ind_pq_high), rRand)
                                - hb.P_opp_r_pq_cum.begin();

    // The probability that this electron was chosen
    ind = norbs*ind_pq + rSpatial;
    double rProb = hb.P_opp_r_pq.at(ind);
  }

  // Get the spin orbital index (r and p have the same spin)
  r = 2*rSpatial + p%2;
}

void pickSOrbitalHB(heatBathFCIQMC& hb, const int norbs, const int p, const int q, const int r,
                    int& s, double& sProb)
{
  int ind, ind_pq, sSpatial;

  bool sameSpin = (p%2 == q%2);
  int pSpatial = p/2;
  int qSpatial = q/2;

  // Pick a spin-orbital r from P(r|pq), such that r and p have the same spin
  if (sameSpin) {

    int Q = min(pSpatial, qSpatial);
    int P = max(pSpatial, qSpatial);
    ind_pq = P*(P-1)/2 + Q;

    // The first index for triplet (p,q,r)
    int ind_pqr_low = pow(norbs,2) * ind_pq + norbs*r;
    // The last index for triplet (p,q,r)
    int ind_pqr_high = ind_pqr_low + norbs - 1;

    double sRand = random();
    sSpatial = std::lower_bound((hb.P_same_s_pqr_cum.begin() + ind_pqr_low),
                                (hb.P_same_s_pqr_cum.begin() + ind_pqr_high), sRand)
                                - hb.P_same_s_pqr_cum.begin();

    // The probability that this electron was chosen
    ind = pow(norbs,2) * ind_pq + norbs*r + sSpatial;
    double sProb = hb.P_same_s_pqr.at(ind);
  } else {

    ind_pq = pSpatial*norbs + qSpatial;
    // The first index for triplet (p,q,r)
    int ind_pqr_low = pow(norbs,2) * ind_pq + norbs*r;
    // The last index for triplet (p,q,r)
    int ind_pqr_high = ind_pqr_low + norbs - 1;

    double sRand = random();
    sSpatial = std::lower_bound((hb.P_opp_s_pqr_cum.begin() + ind_pqr_low),
                                (hb.P_opp_s_pqr_cum.begin() + ind_pqr_high), sRand)
                                - hb.P_opp_s_pqr_cum.begin();

    // The probability that this electron was chosen
    ind = pow(norbs,2) * ind_pq + norbs*r + sSpatial;
    double sProb = hb.P_opp_s_pqr.at(ind);
  }

  // Get the spin orbital index (s and q have the same spin)
  s = 2*sSpatial + q%2;
}

void generateDoubleExcitHB(const Determinant& parentDet, Determinant& childDet, double& pgen_pqrs, heatBathFCIQMC& hb)
{
  int P, Q, ind, ind_pq;
  int norbs = Determinant::norbs;
  int nSpinOrbs = 2*norbs;
  double rProb2, sProb2;

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));

  // Pick the first electron with probability P(p) = S_p / sum_p' S_p'
  // For this, we need to calculate the cumulative array
  vector<double> S_p_cum(nSpinOrbs, 0.0);
  for (int p=0; p<nSpinOrbs; p++) {
    S_p_cum.at(p) = S_p_cum.at(p-1) + hb.S_p.at(p);
  }

  // Pick the first electron
  double pRand = random() * hb.S_p_tot;
  int pFinal = std::lower_bound(S_p_cum.begin(), (S_p_cum.begin() + nSpinOrbs), pRand) - S_p_cum.begin();
  // The probability that this electron was chosen
  double pProb = hb.S_p.at(pFinal) / hb.S_p_tot;

  // Pick the second electron with probability D_pq / sum_q' D_pq'
  // We again need the relevant cumulative array
  vector<double> D_pq_cum(nSpinOrbs, 0.0);
  for (int q=0; q<nSpinOrbs; q++) {
    if (q == pFinal) {
      D_pq_cum.at(q) = D_pq_cum.at(q-1);
    } else {
      Q = min(pFinal,q);
      P = max(pFinal,q);
      ind = P*(P-1)/2 + Q;
      D_pq_cum.at(q) = D_pq_cum.at(q-1) + hb.D_pq.at(ind);
    }
  }

  // Pick the second electron
  double D_pq_tot = D_pq_cum.at(nSpinOrbs);
  double qRand = random() * D_pq_tot;
  int qFinal = std::lower_bound(D_pq_cum.begin(), (D_pq_cum.begin() + nSpinOrbs), qRand) - D_pq_cum.begin();
  // The probability that this electron was chosen
  Q = min(pFinal,qFinal);
  P = max(pFinal,qFinal);
  ind = P*(P-1)/2 + Q;
  double qProb = hb.D_pq.at(ind) / D_pq_tot;

  // We also need to know the probability that the same two electrons were
  // picked in the opposite order.
  // The probability that q was picked first:
  int qProb2 = hb.S_p.at(qFinal) / hb.S_p_tot;
  // The probability that p was picked second, given that p was picked first:
  double D_qp_tot = 0.0;
  for (int p=0; p<nSpinOrbs; p++) {
    if (p != qFinal) {
      Q = min(p,qFinal);
      P = max(p,qFinal);
      ind = P*(P-1)/2 + Q;
      D_qp_tot += hb.D_pq.at(ind);
    }
  }
  Q = min(pFinal,qFinal);
  P = max(pFinal,qFinal);
  ind = P*(P-1)/2 + Q;
  int pProb2 = hb.D_pq.at(ind) / D_qp_tot;

  int rFinal;
  double rProb;
  pickROrbitalHB(hb, norbs, pFinal, qFinal, rFinal, rProb);

  // If the orbital r is already occupied, return a null excitation
  if (parentDet.getocc(rFinal)) {
    pgen_pqrs = 0.0;
    childDet = parentDet;
  }
  
  // Now pick the final spin-orbital, s, from P(s|pqr), such that s and q
  // have the same spin
  int sFinal;
  double sProb;
  pickSOrbitalHB(hb, norbs, pFinal, qFinal, rFinal, sFinal, sProb);

  // If the orbital s is already occupied, return a null excitation
  if (parentDet.getocc(sFinal)) {
    pgen_pqrs = 0.0;
    childDet = parentDet;
  }

  // Find probabilities of selecting r and s the other way around
  if (sFinal%2 == pFinal%2) {
    int pSpatial = pFinal/2;
    int qSpatial = qFinal/2;

    if (pFinal%2 == qFinal%2) {
      // Same spin for p and q
      Q = min(pSpatial, qSpatial);
      P = max(pSpatial, qSpatial);
      ind_pq = P*(P-1)/2 + Q;

      int ind_pqs = norbs * ind_pq + (sFinal/2);
      sProb2 = hb.P_same_r_pq.at(ind_pqs);

      int ind_pqsr = pow(norbs,2)*ind_pq + norbs*(sFinal/2) + (rFinal/2);
      rProb2 = hb.P_same_s_pqr.at(ind_pqsr);

    } else {
      // Opposite spin for p and q
      ind_pq = pSpatial*norbs + qSpatial;

      int ind_pqs = norbs * ind_pq + (sFinal/2);
      sProb2 = hb.P_opp_r_pq.at(ind_pqs);

      int ind_pqsr = pow(norbs,2)*ind_pq + norbs*(sFinal/2) + (rFinal/2);
      rProb2 = hb.P_opp_s_pqr.at(ind_pqsr);
    }

  } else {
    // The first empty spin orbital is chosen such that its spin is the
    // same as p. So the probability of selecting s is 0 in this case.
    // Note, if s and p have different spins, then r and q have different
    // spins also
    sProb2 = 0.0;
    rProb2 = 0.0;
  }

  childDet = parentDet;
  childDet.setocc(pFinal, false);
  childDet.setocc(qFinal, false);
  childDet.setocc(rFinal, true);
  childDet.setocc(sFinal, true);

  pgen_pqrs = pProb  * qProb  * ( rProb * sProb + sProb2 * rProb2 ) +
              qProb2 * pProb2 * ( rProb * sProb + sProb2 * rProb2 );
}

