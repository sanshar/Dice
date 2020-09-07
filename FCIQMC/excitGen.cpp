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
#include "input.h"
#include "integral.h"
#include "excitGen.h"

heatBathFCIQMC::heatBathFCIQMC(int norbs, const twoInt& I2) {
  createArrays(norbs, I2);
}

// Set up arrays used by the heat bath excitation generator
void heatBathFCIQMC::createArrays(int norbs, const twoInt& I2) {

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

  H_tot_same.resize(size_2, 0.0);
  H_tot_opp.resize(size_3, 0.0);

  // Set up D_pq
  for (int p=1; p<nSpinOrbs; p++) {
    for (int q=0; q<p; q++) {
      int ind = p*(p-1)/2 + q;
      D_pq.at(ind) = 0.0;

      for (int r=0; r<nSpinOrbs; r++) {
        if (r != p && r != q) {
          for (int s=0; s<nSpinOrbs; s++) {
            if (r != s && s != p && s != q) {
              D_pq.at(ind) += fabs( I2(r, p, s, q) - I2(r, q, s, p) );
            }
          }
        }
      }

    }
  }

  // Set up S_p
  for (int p=0; p<nSpinOrbs; p++) {
    S_p.at(p) = 0.0;

    for (int q=0; q<nSpinOrbs; q++) {
      if (p == q) continue;
      int ind = triInd(p,q);
      S_p.at(p) += D_pq.at(ind);
    }
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
            P_same_r_pq.at(ind) += fabs( I2(2*r, 2*p, 2*s, 2*q) - I2(2*r, 2*q, 2*s, 2*p) );
          }

        } // Loop over s

        tot += P_same_r_pq.at(ind);

      } // Loop over r

      // Normalize probabilities
      if (abs(tot) > 1.e-15) {
        for (int r=0; r<norbs; r++) {
          int ind = norbs*ind_pq + r;
          P_same_r_pq.at(ind) /= tot;
        }
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

          if (r != p && s != q) {
            P_opp_r_pq.at(ind) += fabs( I2(2*r, 2*p, 2*s+1, 2*q+1) );
          }

        } // Loop over s

        tot += P_opp_r_pq.at(ind);

      } // Loop over r

      // Normalize probabilities
      if (abs(tot) > 1.e-15) {
        for (int r=0; r<norbs; r++) {
          int ind = norbs*ind_pq + r;
          P_opp_r_pq.at(ind) /= tot;
        }
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
            P_same_s_pqr.at(ind) = fabs( I2(2*r, 2*p, 2*s, 2*q) - I2(2*r, 2*q, 2*s, 2*p) );
            tot_sum += fabs( I2(2*r, 2*p, 2*s, 2*q) - I2(2*r, 2*q, 2*s, 2*p) );
          }

        } // Loop over s

        // Normalize probability
        if (abs(tot_sum) > 1.e-15) {
          for (int s=0; s<norbs; s++) {
            int ind = pow(norbs,2) * ind_pq + norbs*r + s;
            P_same_s_pqr.at(ind) /= tot_sum;
          }
        }

        int ind_tot = norbs*ind_pq + r;
        H_tot_same.at(ind_tot) = tot_sum;

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

          int ind = pow(norbs,2) * ind_pq + norbs*r + s;

          if (r != p && s != q) {
            P_opp_s_pqr.at(ind) = fabs( I2(2*r, 2*p, 2*s+1, 2*q+1) );
            tot_sum += fabs( I2(2*r, 2*p, 2*s+1, 2*q+1) );
          }

        } // Loop over s

        // Normalize probability
        if (abs(tot_sum) > 1.e-15) {
          for (int s=0; s<norbs; s++) {
            int ind = pow(norbs,2) * ind_pq + norbs*r + s;
            P_opp_s_pqr.at(ind) /= tot_sum;
          }
        }

        int ind_tot = norbs*ind_pq + r;
        H_tot_opp.at(ind_tot) = tot_sum;

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
      } // Loop over r
    } // Loop over q
  } // Loop over p

} // End of createArrays


// Wrapper function to call the appropriate excitation generator
void generateExcitation(const heatBathFCIQMC& hb, const oneInt& I1, const twoInt& I2, const Determinant& parentDet,
                        const int& nel, Determinant& childDet, Determinant& childDet2, double& pGen, double& pGen2)
{
  if (schd.uniformExGen || schd.heatBathUniformSingExGen) {
    generateExcitationSingDoub(hb, I1, I2, parentDet, nel, childDet, childDet2, pGen, pGen2);
  } else if (schd.heatBathExGen) {
    generateExcitHB(hb, I1, I2, parentDet, nel, childDet, childDet2, pGen, pGen2, true);
  }
}

// Generate a random single or double excitation, and also return the
// probability that it was generated. A single excitation is returned using
// childDet and pgen. A double excitation is returned using chilDet2 and pGen2.
void generateExcitationSingDoub(const heatBathFCIQMC& hb, const oneInt& I1, const twoInt& I2,
                                const Determinant& parentDet, const int& nel, Determinant& childDet,
                                Determinant& childDet2, double& pgen, double& pgen2)
{
  double pSingle = 0.05;
  double pgen_ia, pgen_ijab;

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

  if (random() < pSingle) {
    generateSingleExcit(parentDet, childDet, pgen_ia);
    pgen = pSingle * pgen_ia;

    // No double excitation is generated here:
    pgen2 = 0.0;
  } else {
    if (schd.uniformExGen) {
      generateDoubleExcit(parentDet, childDet2, pgen_ijab);
    } else if (schd.heatBathUniformSingExGen) {
      // Pass in attemptSingleExcit = false to indicate that a double
      // excitation must be generated. pgen will return as 0.
      generateExcitHB(hb, I1, I2, parentDet, nel, childDet, childDet2, pgen, pgen_ijab, false);
    }
    pgen2 = (1.0 - pSingle) * pgen_ijab;

    // No single excitation is generated here:
    pgen = 0.0;
  }
}

// Generate a random single excitation, and also return the probability that
// it was generated
void generateSingleExcit(const Determinant& parentDet, Determinant& childDet, double& pgen_ia)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

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
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

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

// Pick the 'r' orbital when using the heat bath excitation generator.
// This is the first unoccupied orbital, after both of the occupied orbitals
// (p and q) have been picked. Also, calculate a return the value H_tot_pqr,
// which is used to decide whether to generate a single or double excitation.
// This function also returns the probability that r was picked, given p and
// q were picked already, which is rProb.
void pickROrbitalHB(const heatBathFCIQMC& hb, const int norbs, const int p, const int q, int& r,
                    double& rProb, double& H_tot_pqr)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

  bool sameSpin = (p%2 == q%2);
  int ind, ind_pq, rSpatial;
  int pSpatial = p/2, qSpatial = q/2;

  // Pick a spin-orbital r from P(r|pq), such that r and p have the same spin
  if (sameSpin) {

    ind_pq = triInd(pSpatial, qSpatial);
    // The first index for pair (p,q)
    int ind_pq_low = ind_pq*norbs;
    // The last index for pair (p,q)
    int ind_pq_high = ind_pq_low + norbs - 1;

    double rRand = random();
    rSpatial = std::lower_bound((hb.P_same_r_pq_cum.begin() + ind_pq_low),
                                (hb.P_same_r_pq_cum.begin() + ind_pq_high), rRand)
                                - hb.P_same_r_pq_cum.begin() - ind_pq_low;

    // The probability that this electron was chosen
    ind = norbs*ind_pq + rSpatial;
    rProb = hb.P_same_r_pq.at(ind);
    // For choosing single excitation
    H_tot_pqr = hb.H_tot_same.at(ind);
  } else {

    ind_pq = pSpatial*norbs + qSpatial;
    // The first index for pair (p,q)
    int ind_pq_low = ind_pq*norbs;
    // The last index for pair (p,q)
    int ind_pq_high = ind_pq_low + norbs - 1;

    double rRand = random();
    rSpatial = std::lower_bound((hb.P_opp_r_pq_cum.begin() + ind_pq_low),
                                (hb.P_opp_r_pq_cum.begin() + ind_pq_high), rRand)
                                - hb.P_opp_r_pq_cum.begin() - ind_pq_low;

    // The probability that this electron was chosen
    ind = norbs*ind_pq + rSpatial;
    rProb = hb.P_opp_r_pq.at(ind);
    // For choosing single excitation
    H_tot_pqr = hb.H_tot_opp.at(ind);
  }

  // Get the spin orbital index (r and p have the same spin)
  r = 2*rSpatial + p%2;
}

// Pick the 's' orbital when using the heat bath excitation generator.
// This is the second unoccupied orbital, after both of the occupied orbitals
// (p and q) and the first unoccupied orbital (r) have been picked.
// This function also returns the probability that s was picked, given p, q
// and r were picked already, which is sProb.
void pickSOrbitalHB(const heatBathFCIQMC& hb, const int norbs, const int p, const int q, const int r,
                    int& s, double& sProb)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

  int ind, ind_pq, sSpatial;

  bool sameSpin = (p%2 == q%2);
  int pSpatial = p/2, qSpatial = q/2, rSpatial = r/2;

  // Pick a spin-orbital r from P(r|pq), such that r and p have the same spin
  if (sameSpin) {

    ind_pq = triInd(pSpatial, qSpatial);
    // The first index for triplet (p,q,r)
    int ind_pqr_low = pow(norbs,2) * ind_pq + norbs*rSpatial;
    // The last index for triplet (p,q,r)
    int ind_pqr_high = ind_pqr_low + norbs - 1;

    double sRand = random();
    sSpatial = std::lower_bound((hb.P_same_s_pqr_cum.begin() + ind_pqr_low),
                                (hb.P_same_s_pqr_cum.begin() + ind_pqr_high), sRand)
                                - hb.P_same_s_pqr_cum.begin() - ind_pqr_low;

    // The probability that this electron was chosen
    ind = pow(norbs,2) * ind_pq + norbs*rSpatial + sSpatial;
    sProb = hb.P_same_s_pqr.at(ind);
  } else {

    ind_pq = pSpatial*norbs + qSpatial;
    // The first index for triplet (p,q,r)
    int ind_pqr_low = pow(norbs,2) * ind_pq + norbs*rSpatial;
    // The last index for triplet (p,q,r)
    int ind_pqr_high = ind_pqr_low + norbs - 1;

    double sRand = random();
    sSpatial = std::lower_bound((hb.P_opp_s_pqr_cum.begin() + ind_pqr_low),
                                (hb.P_opp_s_pqr_cum.begin() + ind_pqr_high), sRand)
                                - hb.P_opp_s_pqr_cum.begin() - ind_pqr_low;

    // The probability that this electron was chosen
    ind = pow(norbs,2) * ind_pq + norbs*rSpatial + sSpatial;
    sProb = hb.P_opp_s_pqr.at(ind);
  }

  // Get the spin orbital index (s and q have the same spin)
  s = 2*sSpatial + q%2;
}

// Calculate the probability of choosing the excitation p to r, where p and r
// are both spin-orbital labels. pProb is the probability that p is chosen as
// the first electron. hSingAbs is the absolute value of the the Hamiltonian
// element between the original and singly excited determinants.
double calcSinglesProb(const heatBathFCIQMC& hb, const oneInt& I1, const twoInt& I2, const int norbs,
                       const vector<int>& closed, const double pProb, const double D_pq_tot,
                       const double hSingAbs, const int p, const int r)
{
  int nel = closed.size();
  int pSpatial = p/2, rSpatial = r/2;

  double pGen = 0.0;

  // Need to loop over all possible orbitals q that could have been chosen
  // as the second electron
  for (int q=0; q<nel; q++) {
    int qOrb = closed.at(q);
    int qSpatial = qOrb/2;
    if (qOrb != p) {
      int ind = triInd(p, qOrb);
      double qProb = hb.D_pq.at(ind) / D_pq_tot;

      double rProb, H_tot_pqr;
      if (p%2 == qOrb%2) {
        // Same spin
        int ind_pq = triInd(pSpatial, qSpatial);
        int ind_pqr = ind_pq*norbs + rSpatial;
        rProb = hb.P_same_r_pq.at(ind_pqr);
        H_tot_pqr = hb.H_tot_same.at(ind_pqr);
      } else {
        // Opposite spin
        int ind_pq = pSpatial*norbs + qSpatial;
        int ind_pqr = ind_pq*norbs + rSpatial;
        rProb = hb.P_opp_r_pq.at(ind_pqr);
        H_tot_pqr = hb.H_tot_opp.at(ind_pqr);
      }

      // The probability of generating a single excitation, rather than a
      // double excitation
      double singProb;
      if (hSingAbs < H_tot_pqr) {
        singProb = hSingAbs / ( H_tot_pqr + hSingAbs );
      } else {
        // If hSingAbs >= Htot_pqr, always attempt to generate both a
        // single and double excitation
        singProb = 1.0;
      }
      pGen += pProb * qProb * rProb * singProb;
    }
  }

  return pGen;
}

// This function returns the probability of choosing a double rather than
// a single excitation, given that orbitals p and q have been chosen to
// excite from, and orbital r has been chosen to excite to
double calcProbDouble(const Determinant& parentDet, const oneInt& I1, const twoInt& I2,
                      const double& H_tot_pqr, const int& p, const int& r) {

  double hSing = parentDet.Hij_1Excite(p, r, I1, I2);
  double hSingAbs = abs(hSing);

  double doubleProb;
  if (hSingAbs < H_tot_pqr) {
    doubleProb = 1.0 - hSingAbs / ( H_tot_pqr + hSingAbs );
  } else {
    doubleProb = 1.0;
  }
  return doubleProb;
}

// Calculate and return the probabilities of selecting the final two orbitals
// (labelled r and s in our convention). These are the two unoccupied orbitals.
// Also, return the probability of choosing a double, in the case that p, q, r
// are the first three orbitals chosen. This is only calculated if calcDoubleProb
// is true.
void calcProbsForRAndS(const heatBathFCIQMC& hb, const oneInt& I1, const twoInt& I2, const int& norbs,
                       const Determinant& parentDet, const int& p, const int& q, const int& r, const int& s,
                       double& rProb, double& sProb, double& doubleProb, const bool& calcDoubleProb)
{
  if (r%2 == p%2) {
    double H_tot;
    int pSpatial = p/2, qSpatial = q/2;
    int rSpatial = r/2, sSpatial = s/2;

    // Same spin for p and q:
    if (p%2 == q%2) {
      int ind1 = triInd(pSpatial, qSpatial);
      int ind2 = norbs * ind1 + rSpatial;
      // Probability of picking r first
      rProb = hb.P_same_r_pq.at(ind2);
      H_tot = hb.H_tot_same.at(ind2);
      int ind3 = pow(norbs,2)*ind1 + norbs*rSpatial + sSpatial;
      // Probability of picking s second, after picking r first
      sProb = hb.P_same_s_pqr.at(ind3);

    // Opposite spin for p and q:
    } else {
      int ind1 = pSpatial*norbs + qSpatial;
      int ind2 = norbs * ind1 + rSpatial;
      rProb = hb.P_opp_r_pq.at(ind2);
      H_tot = hb.H_tot_opp.at(ind2);
      int ind3 = pow(norbs,2)*ind1 + norbs*rSpatial + sSpatial;
      sProb = hb.P_opp_s_pqr.at(ind3);
    }

    // The probability of generating a double, rather than a single,
    // if s had been chosen first instead of r
    if (calcDoubleProb) {
      doubleProb = calcProbDouble(parentDet, I1, I2, H_tot, p, r);
    }

  } else {
    rProb = 0.0;
    sProb = 0.0;
    doubleProb = 0.0;
  }
}

// Use the heat bath algorithm to generate both the single and
// double excitations
void generateExcitHB(const heatBathFCIQMC& hb, const oneInt& I1, const twoInt& I2, const Determinant& parentDet,
                     const int& nel, Determinant& childDet, Determinant& childDet2, double& pGen, double& pGen2,
                     const bool& attemptSingleExcit)
{
  int norbs = Determinant::norbs, ind;
  int nSpinOrbs = 2*norbs;

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

  childDet = parentDet;
  childDet2 = parentDet;

  vector<int> closed(nel, 0);
  parentDet.getClosedAllocated(closed);

  // Pick the first electron with probability P(p) = S_p / sum_p' S_p'
  // For this, we need to calculate the cumulative array, summed over
  // occupied electrons only

  // Set up the cumulative array
  double S_p_tot = 0.0;
  vector<double> S_p_cum(nel, 0.0);
  for (int p=0; p<nel; p++) {
    int orb = closed.at(p);
    S_p_tot += hb.S_p.at(orb);
    S_p_cum.at(p) = S_p_tot;
  }

  // Pick the first electron
  double pRand = random() * S_p_tot;
  int pInd = std::lower_bound(S_p_cum.begin(), (S_p_cum.begin() + nel), pRand) - S_p_cum.begin();
  // The actual orbital being excited from:
  int pFinal = closed.at(pInd);
  // The probability that this electron was chosen
  double pProb = hb.S_p.at(pFinal) / S_p_tot;

  // Pick the second electron with probability D_pq / sum_q' D_pq'
  // We again need the relevant cumulative array, summed over
  // remaining occupied electrons, q'

  // Set up the cumulative array
  double D_pq_tot = 0.0;
  vector<double> D_pq_cum(nel, 0.0);
  for (int q=0; q<nel; q++) {
    if (q == pInd) {
      D_pq_cum.at(q) = D_pq_tot;
    } else {
      int orb = closed.at(q);
      int ind_pq = triInd(pFinal, orb);
      D_pq_tot += hb.D_pq.at(ind_pq);
      D_pq_cum.at(q) = D_pq_tot;
    }
  }

  // Pick the second electron
  double qRand = random() * D_pq_tot;
  int qInd = std::lower_bound(D_pq_cum.begin(), (D_pq_cum.begin() + nel), qRand) - D_pq_cum.begin();
  // The actual orbital being excited from:
  int qFinal = closed.at(qInd);
  // The probability that this electron was chosen
  ind = triInd(pFinal, qFinal);
  double qProb_p = hb.D_pq.at(ind) / D_pq_tot;

  if (pFinal == qFinal) cout << "ERROR: p = q in excitation generator";

  // We also need to know the probability that the same two electrons were
  // picked in the opposite order.
  // The probability that q was picked first:
  double qProb = hb.S_p.at(qFinal) / S_p_tot;
  // The probability that p was picked second, given that p was picked first:
  // Need to calculate the new normalizing factor in D_qp / sum_p' D_qp':
  double D_qp_tot = 0.0;
  for (int p=0; p<nel; p++) {
    int orb = closed.at(p);
    if (p != qInd) {
      int ind_pq = triInd(orb, qFinal);
      D_qp_tot += hb.D_pq.at(ind_pq);
    }
  }
  ind = triInd(pFinal, qFinal);
  double pProb_q = hb.D_pq.at(ind) / D_qp_tot;

  // Pick spin-orbital r from P(r|pq), such that r and p have the same spin
  int rFinal;
  double rProb_pq, H_tot_pqr;
  pickROrbitalHB(hb, norbs, pFinal, qFinal, rFinal, rProb_pq, H_tot_pqr);

  // If the orbital r is already occupied, return null excitations
  if (parentDet.getocc(rFinal)) {
    pGen = 0.0;
    pGen2 = 0.0;
    return;
  }

  // If attemptSingleExcit, then decide whether to generate a single
  // or double excitation. If a single excitation is chosen, then generate
  // it here and then return.
  double singProb_pqr, doubProb_pqr;
  if (attemptSingleExcit) {
    // Calculate the Hamiltonian element for a single excitation, p to r
    double hSing = parentDet.Hij_1Excite(pFinal, rFinal, I1, I2);
    double hSingAbs = abs(hSing);

    // If this condition is met then we generate either a single or a double
    // If it is not then, then we generate both a single and double excitation
    if (hSingAbs < H_tot_pqr) {
      singProb_pqr = hSingAbs / ( H_tot_pqr + hSingAbs );
      doubProb_pqr = 1.0 - singProb_pqr;

      double rand = random();
      if (rand < singProb_pqr) {
        // Generate a single excitation from p to r:
        childDet.setocc(pFinal, false);
        childDet.setocc(rFinal, true);
        pGen = calcSinglesProb(hb, I1, I2, norbs, closed, pProb, D_pq_tot, hSingAbs, pFinal, rFinal);

        // Return a null double excitation
        pGen2 = 0.0;
        return;
      }
      // If here, then we generate a double excitation instead of a single
      // Set pGen=0 to indicate that a single excitation is not generate
      pGen = 0.0;

    } else {
      // In this case, generate both a single and double excitation
      doubProb_pqr = 1.0;
      // The single excitation:
      childDet.setocc(pFinal, false);
      childDet.setocc(rFinal, true);
      pGen = calcSinglesProb(hb, I1, I2, norbs, closed, pProb, D_pq_tot, hSingAbs, pFinal, rFinal);
    }

  // If not attempting to generate a single:
  } else {
    pGen = 0.0;
    doubProb_pqr = 1.0;
  }


  // Pick the final spin-orbital, s, with probability P(s|pqr), such
  // that s and q have the same spin
  int sFinal;
  double sProb_pqr;
  pickSOrbitalHB(hb, norbs, pFinal, qFinal, rFinal, sFinal, sProb_pqr);

  if (rFinal == sFinal) cout << "ERROR: r = s in excitation generator";

  // If the orbital s is already occupied, return a null excitation
  if (parentDet.getocc(sFinal)) {
    pGen = 0.0;
    pGen2 = 0.0;
    return;
  }

  // Above we chose p first, then q, then r, then s. In calculating the
  // probability that this excited determinant was chosen, we need to
  // consider the probability that p and q were chosen the opposite way
  // around, and the same for r and s.

  // Find the probabilities for choosing p first, then q, then s, then r
  double rProb_pqs, sProb_pq, doubProb_pqs = 1.0;
  calcProbsForRAndS(hb, I1, I2, norbs, parentDet, pFinal, qFinal, sFinal, rFinal,
                    sProb_pq, rProb_pqs, doubProb_pqs, attemptSingleExcit);

  // Find the probabilities for choosing q first, then p, then r, then s
  double rProb_qp, sProb_qpr, doubProb_qpr = 1.0;
  calcProbsForRAndS(hb, I1, I2, norbs, parentDet, qFinal, pFinal, rFinal, sFinal,
                    rProb_qp, sProb_qpr, doubProb_qpr, attemptSingleExcit);

  // Now calculate the probabilities for choosing q first, then p, then s, then r
  double sProb_qp, rProb_qps, doubProb_qps = 1.0;
  calcProbsForRAndS(hb, I1, I2, norbs, parentDet, qFinal, pFinal, sFinal, rFinal,
                    sProb_qp, rProb_qps, doubProb_qps, attemptSingleExcit);

  // Generate the final doubly excited determinant...
  childDet2.setocc(pFinal, false);
  childDet2.setocc(qFinal, false);
  childDet2.setocc(rFinal, true);
  childDet2.setocc(sFinal, true);

  // ...and the probability that it was generated.
  pGen2 = pProb*qProb_p * ( doubProb_pqr*rProb_pq*sProb_pqr + doubProb_pqs*sProb_pq*rProb_pqs ) +
          qProb*pProb_q * ( doubProb_qpr*rProb_qp*sProb_qpr + doubProb_qps*sProb_qp*rProb_qps );
}
