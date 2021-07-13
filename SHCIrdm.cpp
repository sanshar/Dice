/*
  Developed by Sandeep Sharma with contributions from James E. T. Smith and Adam
  A. Holmes, 2017 Copyright (c) 2017, Sandeep Sharma

  This file is part of DICE.

  This program is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or (at your option) any later
  version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with
  this program. If not, see <http://www.gnu.org/licenses/>.
*/
#include "SHCIrdm.h"
#include <algorithm>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include <fstream>
#include <map>
#include <tuple>
#include <vector>
#include "Davidson.h"
#include "Determinants.h"
#include "Hmult.h"
#include "SHCISortMpiUtils.h"
#include "SHCIbasics.h"
#include "SHCIgetdeterminants.h"
#include "SHCIsampledeterminants.h"
#include "boost/format.hpp"
#include "input.h"
#include "integral.h"
#include "math.h"

#ifndef SERIAL
#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#endif
#include "communicate.h"

using namespace std;
using namespace Eigen;
using namespace boost;
using namespace SHCISortMpiUtils;

namespace localConj {
CItype conj(CItype a) {
#ifdef Complex
  return std::conj(a);
#else
  return a;
#endif
}
};  // namespace localConj

//=============================================================================
// TODO Ask Sandeep about this.
void SHCIrdm::makeRDM(int *&AlphaMajorToBetaLen,
                      vector<int *> &AlphaMajorToBeta,
                      vector<int *> &AlphaMajorToDet, int *&BetaMajorToAlphaLen,
                      vector<int *> &BetaMajorToAlpha,
                      vector<int *> &BetaMajorToDet, int *&SinglesFromAlphaLen,
                      vector<int *> &SinglesFromAlpha, int *&SinglesFromBetaLen,
                      vector<int *> &SinglesFromBeta, Determinant *Dets,
                      int DetsSize, int Norbs, int nelec, CItype *cibra,
                      CItype *ciket, MatrixXx &s2RDM) {
  /*!

  Efficient creation of 2RDM using MPI.

  :Arguments:

      int* & AlphaMajorToBetaLen:

      vector<int* > & AlphaMajorToBeta   :

      vector<int* > & AlphaMajorToDet    :

      int*          & BetaMajorToAlphaLen:

      vector<int* > & BetaMajorToAlpha   :

      vector<int* > & BetaMajorToDet     :

      int*          & SinglesFromAlphaLen:

      vector<int* > & SinglesFromAlpha   :

      int*          & SinglesFromBetaLen :

      vector<int* > & SinglesFromBeta    :

      Determinant* Dets:

      int DetsSize:

      int Norbs, int nelec, CItype* cibra:

      CItype* ciket:

      MatrixXx& s2RDM:

  */

  int proc = commrank, nprocs = commsize;

  size_t norbs = Norbs;
  int nSpatOrbs = norbs / 2;

  int EndIndex = DetsSize;

  // diagonal element
  for (size_t k = 0; k < EndIndex; k++) {
    if (k % (nprocs) != proc) continue;

    vector<int> closed(nelec, 0);
    vector<int> open(norbs - nelec, 0);
    Dets[k].getOpenClosed(open, closed);

    for (int n1 = 0; n1 < nelec; n1++) {
      for (int n2 = 0; n2 < n1; n2++) {
        int orb1 = closed[n1], orb2 = closed[n2];
        // if (schd.DoSpinRDM)
        // twoRDM(orb1*(orb1+1)/2 + orb2, orb1*(orb1+1)/2+orb2) +=
        // localConj::conj(cibra[i])*ciket[i];
        populateSpatialRDM(orb1, orb2, orb1, orb2, s2RDM,
                           localConj::conj(cibra[k]) * ciket[k], nSpatOrbs);
      }
    }
  }

  // alpha-beta excitation
  for (int i = 0; i < AlphaMajorToBeta.size(); i++) {
    for (int ii = 0; ii < AlphaMajorToBetaLen[i]; ii++) {
      int Astring = i, Bstring = AlphaMajorToBeta[i][ii],
          DetI = AlphaMajorToDet[i][ii];

      if ((std::abs(DetI) - 1) % nprocs != proc) continue;

      vector<int> closed(nelec, 0);
      vector<int> open(norbs - nelec, 0);
      Dets[abs(DetI) - 1].getOpenClosed(open, closed);
      Determinant di = Dets[abs(DetI) - 1];

      int maxBToA = BetaMajorToAlpha[Bstring][BetaMajorToAlphaLen[Bstring] - 1];
      // singles from Astring
      for (int j = 0; j < SinglesFromAlphaLen[Astring]; j++) {
        int Asingle = SinglesFromAlpha[Astring][j];

        int index = binarySearch(&BetaMajorToAlpha[Bstring][0], 0,
                                 BetaMajorToAlphaLen[Bstring] - 1, Asingle);
        if (index != -1) {
          int DetJ = BetaMajorToDet[Bstring][index];

          if (std::abs(DetJ) >= std::abs(DetI)) continue;
          Determinant dj = Dets[abs(DetJ) - 1];
          size_t orbDiff;
          getOrbDiff(dj, di, orbDiff);
          int d0 = orbDiff % norbs, c0 = (orbDiff / norbs) % norbs;

          for (int n1 = 0; n1 < nelec; n1++) {
            double sgn = 1.0;
            int a = max(closed[n1], c0), b = min(closed[n1], c0),
                I = max(closed[n1], d0), J = min(closed[n1], d0);
            if (closed[n1] == d0) continue;
            di.parity(min(d0, c0), max(d0, c0), sgn);
            if (!((closed[n1] > c0 && closed[n1] > d0) ||
                  (closed[n1] < c0 && closed[n1] < d0)))
              sgn *= -1.;
            populateSpatialRDM(a, b, I, J, s2RDM,
                               sgn * localConj::conj(cibra[abs(DetJ) - 1]) *
                                   ciket[abs(DetI) - 1],
                               nSpatOrbs);
            populateSpatialRDM(I, J, a, b, s2RDM,
                               sgn * localConj::conj(ciket[abs(DetJ) - 1]) *
                                   cibra[abs(DetI) - 1],
                               nSpatOrbs);
          }
        }
      }

      // single Alpha and single Beta
      for (int j = 0; j < SinglesFromAlphaLen[Astring]; j++) {
        int Asingle = SinglesFromAlpha[Astring][j];

        int SearchStartIndex = 0, AlphaToBetaLen = AlphaMajorToBetaLen[Asingle],
            SinglesFromBLen = SinglesFromBetaLen[Bstring];
        int maxAToB =
            AlphaMajorToBeta[Asingle][AlphaMajorToBetaLen[Asingle] - 1];
        for (int k = 0; k < SinglesFromBLen; k++) {
          int &Bsingle = SinglesFromBeta[Bstring][k];

          if (SearchStartIndex >= AlphaToBetaLen) break;

          int index = SearchStartIndex;
          for (; index < AlphaToBetaLen &&
                 AlphaMajorToBeta[Asingle][index] < Bsingle;
               index++) {
          }

          SearchStartIndex = index;
          if (index < AlphaToBetaLen &&
              AlphaMajorToBeta[Asingle][index] == Bsingle) {
            int DetJ = AlphaMajorToDet[Asingle][SearchStartIndex];
            // if (std::abs(DetJ) < max(offSet, StartIndex) && std::abs(DetI) <
            // max(offSet, StartIndex)) continue;
            if (std::abs(DetJ) >= std::abs(DetI)) continue;
            Determinant dj = Dets[abs(DetJ) - 1];
            size_t orbDiff;
            getOrbDiff(dj, di, orbDiff);
            // CItype hij = Hij(Dets[std::abs(DetJ)], Dets[std::abs(DetI)], I1,
            // I2, coreE, orbDiff);

            int d0 = orbDiff % norbs, c0 = (orbDiff / norbs) % norbs;
            int d1 = (orbDiff / norbs / norbs) % norbs,
                c1 = (orbDiff / norbs / norbs / norbs) % norbs;
            double sgn = 1.0;

            di.parity(d1, d0, c1, c0, sgn);
            populateSpatialRDM(c1, c0, d1, d0, s2RDM,
                               sgn * localConj::conj(cibra[abs(DetJ) - 1]) *
                                   ciket[abs(DetI) - 1],
                               nSpatOrbs);
            populateSpatialRDM(d1, d0, c1, c0, s2RDM,
                               sgn * localConj::conj(ciket[abs(DetJ) - 1]) *
                                   cibra[abs(DetI) - 1],
                               nSpatOrbs);
          }
        }
      }

      // singles from Bstring
      int maxAtoB = AlphaMajorToBeta[Astring][AlphaMajorToBetaLen[Astring] - 1];
      for (int j = 0; j < SinglesFromBetaLen[Bstring]; j++) {
        int Bsingle = SinglesFromBeta[Bstring][j];

        // if (Bsingle > maxAtoB) break;
        int index = binarySearch(&AlphaMajorToBeta[Astring][0], 0,
                                 AlphaMajorToBetaLen[Astring] - 1, Bsingle);

        if (index != -1) {
          int DetJ = AlphaMajorToDet[Astring][index];
          if (std::abs(DetJ) >= std::abs(DetI)) continue;
          Determinant dj = Dets[abs(DetJ) - 1];

          size_t orbDiff;
          getOrbDiff(dj, di, orbDiff);
          // CItype hij = Hij(Dets[std::abs(DetJ)], Dets[std::abs(DetI)], I1,
          // I2, coreE, orbDiff);
          int d0 = orbDiff % norbs, c0 = (orbDiff / norbs) % norbs;

          for (int n1 = 0; n1 < nelec; n1++) {
            double sgn = 1.0;
            int a = max(closed[n1], c0), b = min(closed[n1], c0),
                I = max(closed[n1], d0), J = min(closed[n1], d0);
            if (closed[n1] == d0) continue;
            di.parity(min(d0, c0), max(d0, c0), sgn);
            if (!((closed[n1] > c0 && closed[n1] > d0) ||
                  (closed[n1] < c0 && closed[n1] < d0)))
              sgn *= -1.;
            populateSpatialRDM(a, b, I, J, s2RDM,
                               sgn * localConj::conj(cibra[abs(DetJ) - 1]) *
                                   ciket[abs(DetI) - 1],
                               nSpatOrbs);
            populateSpatialRDM(I, J, a, b, s2RDM,
                               sgn * localConj::conj(ciket[abs(DetJ) - 1]) *
                                   cibra[abs(DetI) - 1],
                               nSpatOrbs);
          }
        }
      }

      // double beta excitation
      for (int j = 0; j < AlphaMajorToBetaLen[i]; j++) {
        int DetJ = AlphaMajorToDet[i][j];
        // if (std::abs(DetJ) < StartIndex) continue;
        // if (std::abs(DetJ) < max(offSet, StartIndex) && std::abs(DetI) <
        // max(offSet, StartIndex)) continue;
        if (std::abs(DetJ) >= std::abs(DetI)) continue;
        Determinant dj = Dets[abs(DetJ) - 1];

        if (dj.ExcitationDistance(di) == 2) {
          size_t orbDiff;
          getOrbDiff(dj, di, orbDiff);
          // CItype hij = Hij(Dets[std::abs(DetJ)], Dets[std::abs(DetI)], I1,
          // I2, coreE, orbDiff);
          int d0 = orbDiff % norbs, c0 = (orbDiff / norbs) % norbs;
          int d1 = (orbDiff / norbs / norbs) % norbs,
              c1 = (orbDiff / norbs / norbs / norbs) % norbs;
          double sgn = 1.0;

          di.parity(d1, d0, c1, c0, sgn);
          populateSpatialRDM(c1, c0, d1, d0, s2RDM,
                             sgn * localConj::conj(cibra[abs(DetJ) - 1]) *
                                 ciket[abs(DetI) - 1],
                             nSpatOrbs);
          populateSpatialRDM(d1, d0, c1, c0, s2RDM,
                             sgn * localConj::conj(ciket[abs(DetJ) - 1]) *
                                 cibra[abs(DetI) - 1],
                             nSpatOrbs);
        }
      }

      // double Alpha excitation
      for (int j = 0; j < BetaMajorToAlphaLen[Bstring]; j++) {
        int DetJ = BetaMajorToDet[Bstring][j];
        // if (std::abs(DetJ) < StartIndex) continue;
        // if (std::abs(DetJ) < max(offSet, StartIndex) && std::abs(DetI) <
        // max(offSet, StartIndex)) continue;
        if (std::abs(DetJ) >= std::abs(DetI)) continue;

        Determinant dj = Dets[abs(DetJ) - 1];
        if (di.ExcitationDistance(dj) == 2) {
          size_t orbDiff;
          getOrbDiff(dj, di, orbDiff);
          // CItype hij = Hij(Dets[std::abs(DetJ)], Dets[std::abs(DetI)], I1,
          // I2, coreE, orbDiff);
          int d0 = orbDiff % norbs, c0 = (orbDiff / norbs) % norbs;
          int d1 = (orbDiff / norbs / norbs) % norbs,
              c1 = (orbDiff / norbs / norbs / norbs) % norbs;
          double sgn = 1.0;

          di.parity(d1, d0, c1, c0, sgn);
          populateSpatialRDM(c1, c0, d1, d0, s2RDM,
                             sgn * localConj::conj(cibra[abs(DetJ) - 1]) *
                                 ciket[abs(DetI) - 1],
                             nSpatOrbs);
          populateSpatialRDM(d1, d0, c1, c0, s2RDM,
                             sgn * localConj::conj(ciket[abs(DetJ) - 1]) *
                                 cibra[abs(DetI) - 1],
                             nSpatOrbs);
        }
      }
    }
  }

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &s2RDM(0, 0), s2RDM.rows() * s2RDM.cols(),
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
}

//=============================================================================
void SHCIrdm::save1RDM(schedule &schd, MatrixXx &s1RDM, MatrixXx &oneRDM,
                       int root) {
  /*!

    Writes the spatial 1RDM to text.

    :Arguments:

        schedule& schd:
            Schedule object that stores Dice parameters.
        MatrixXx& s1RDM:
            Spatial 1RDM.
        MatrixXx& oneRDM:
            Spin 1RDM.
        int root:
            Index of wavefunction to save.
  */

  int nSpatOrbs = s1RDM.rows();
  int norbs = oneRDM.rows();

  if (commrank == 0) {
    char file[5000];
    sprintf(file, "%s/spatial1RDM.%d.%d.txt", schd.prefix[0].c_str(), root,
            root);
    std::ofstream ofs(file, std::ios::out);
    ofs << nSpatOrbs << endl;

    for (int n1 = 0; n1 < nSpatOrbs; n1++) {
      for (int n2 = 0; n2 < nSpatOrbs; n2++) {
        if (fabs(s1RDM(n1, n2)) > 1.e-15) {
          ofs << str(boost::format("%3d   %3d   %16.12g\n") % n1 % n2 %
                     s1RDM(n1, n2));
        }
      }
    }
    ofs.close();

    if (schd.DoSpinOneRDM) {
      char file2[5000];
      sprintf(file2, "%s/spin1RDM.%d.%d.txt", schd.prefix[0].c_str(), root,
              root);
      std::ofstream ofs2(file2, std::ios::out);
      ofs2 << norbs << endl;

      for (int n1 = 0; n1 < norbs; n1++) {
        for (int n2 = 0; n2 < norbs; n2++) {
          if (fabs(oneRDM(n1, n2)) > 1.e-6) {
            ofs2 << str(boost::format("%3d   %3d   %10.8g\n") % n1 % n2 %
                        oneRDM(n1, n2));
          }
        }
      }
      ofs2.close();
    }
  }  // end if commrank
}

//=============================================================================
void SHCIrdm::loadRDM(schedule &schd, MatrixXx &s2RDM, MatrixXx &twoRDM,
                      int root) {
  /*!

  Loads the spatial 2RDM and the spinRDM (if the DoSpinRDM keyword was used).

  :Arguments:

      schedule& schd:
          Schedule object that stores Dice parameters.
      MatrixXx& s2RDM:
          Spatial 2RDM, *changed in this function*.
      MatrixXx& twoRDM:
          Spin 2RDM, *changed in this function*.
      int root:
          Index of wavefunction to load.

  */
  int norbs = twoRDM.rows();
  int nSpatOrbs = pow(s2RDM.rows(), 0.5);

  if (schd.DoSpinRDM) {
    if (commrank == 0) {
      char file[5000];
      sprintf(file, "%s/%d-spinRDM.bkp", schd.prefix[0].c_str(), root);
      std::ifstream ifs(file, std::ios::binary);
      boost::archive::binary_iarchive load(ifs);
      load >> twoRDM;
      // ComputeEnergyFromSpinRDM(norbs, nelec, I1, I2, coreE, twoRDM);
    } else {
      twoRDM.setZero(norbs * (norbs + 1) / 2, norbs * (norbs + 1) / 2);
    }
  }

  if (commrank == 0) {
    char file[5000];
    sprintf(file, "%s/%d-spatialRDM.bkp", schd.prefix[0].c_str(), root);
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    load >> s2RDM;
    // ComputeEnergyFromSpatialRDM(nSpatOrbs, nelec, I1, I2, coreE, s2RDM);
  } else {
    s2RDM.setZero(nSpatOrbs * nSpatOrbs, nSpatOrbs * nSpatOrbs);
  }
}

void SHCIrdm::load3RDM(schedule &schd, MatrixXx &s3RDM, int root) {
  // TODO 3RDM is currently only for the spatial 3RDM not spin.
  int nSpatOrbs = pow(s3RDM.rows(), 1 / 3);
  int nSpatOrbs2 = nSpatOrbs * nSpatOrbs;

  if (commrank == 0) {
    char file[5000];
    sprintf(file, "%s/%d-spatial3RDM.bkp", schd.prefix[0].c_str(), root);
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    load >> s3RDM;
  } else {
    s3RDM.setZero(nSpatOrbs * nSpatOrbs2, nSpatOrbs * nSpatOrbs2);
  }
}

//=============================================================================
void SHCIrdm::saveRDM(schedule &schd, MatrixXx &s2RDM, MatrixXx &twoRDM,
                      int root) {
  /*!
  Saves the spatial 2RDM and the spinRDM (if the DoSpinRDM keyword was used).
  The spatial RDM is saves as "%s/spatialRDM.%d.%d.txt" where %s is the user
  determined prefix and %d is the root.

  :Arguments:

      schedule& schd:
          Schedule object that stores Dice parameters.
      MatrixXx& s2RDM:
          Spatial 2RDM.
      MatrixXx& twoRDM:
          Spin 2RDM.
      int root:
          Index of wavefunction to save.

  */
  int nSpatOrbs = pow(s2RDM.rows(), 0.5);
  if (commrank == 0) {
    {
      char file[5000];
      sprintf(file, "%s/spatialRDM.%d.%d.txt", schd.prefix[0].c_str(), root,
              root);
      std::ofstream ofs(file, std::ios::out);
      ofs << nSpatOrbs << endl;
      for (int n1 = 0; n1 < nSpatOrbs; n1++)
        for (int n2 = 0; n2 < nSpatOrbs; n2++)
          for (int n3 = 0; n3 < nSpatOrbs; n3++)
            for (int n4 = 0; n4 < nSpatOrbs; n4++) {
              if (fabs(s2RDM(n1 * nSpatOrbs + n2, n3 * nSpatOrbs + n4)) >
                  1.e-15)
                ofs << str(boost::format("%3d   %3d   %3d   %3d   %16.12g\n") %
                           n1 % n2 % n3 % n4 %
                           s2RDM(n1 * nSpatOrbs + n2, n3 * nSpatOrbs + n4));
            }
      ofs.close();
    }

    if (schd.DoSpinRDM) {
      // Original writing to binary
      char file_bin[5000];
      sprintf(file_bin, "%s/%d-spinRDM.bkp", schd.prefix[0].c_str(), root);
      std::ofstream ofs_bin(file_bin, std::ios::binary);
      boost::archive::binary_oarchive save(ofs_bin);
      save << twoRDM;
      ofs_bin.close();
      // ComputeEnergyFromSpinRDM(norbs, nelec, I1, I2, coreE, twoRDM);

      //
      // (New 06/21/21) Writing to text file
      //
      const int norbs = 2* nSpatOrbs;
      char file[5000];
      sprintf(file, "%s/spin2RDM.%d.%d.txt", schd.prefix[0].c_str(), root,
              root);
      std::ofstream ofs(file, std::ios::out);
      ofs << norbs << endl;

      for (int p = 0; p < norbs; p++)
        for (int q = 0; q < norbs; q++)
          for (int r = 0; r < norbs; r++)
            for (int s = 0; s < norbs; s++) {
              int P = max(p, q), Q = min(p, q);
              int R = max(r, s), S = min(r, s);
              double sgn = 1.;
              if (P != p) sgn *= -1;
              if (R != r) sgn *= -1;
              double value = sgn * twoRDM(P * (P + 1) / 2 + Q, R * (R + 1) / 2 + S);


              if (fabs(value) > 1.e-15)
                ofs << str(boost::format("%3d   %3d   %3d   %3d   %16.12g\n") %
                           p % q % r % s % value);
            }
      ofs.close();
    }

    {
      char file[5000];
      sprintf(file, "%s/%d-spatialRDM.bkp", schd.prefix[0].c_str(), root);
      std::ofstream ofs(file, std::ios::binary);
      boost::archive::binary_oarchive save(ofs);
      save << s2RDM;
      // ComputeEnergyFromSpatialRDM(nSpatOrbs, nelec, I1, I2, coreE, s2RDM);
    }

  }  // end commrank
}

void SHCIrdm::save3RDM(schedule &schd, MatrixXx &threeRDM, MatrixXx &s3RDM,
                       int root, size_t norbs) {
  int nSpatOrbs = norbs / 2;  // pow(s3RDM.rows(),1/3.0);
  int nSpatOrbs2 = nSpatOrbs * nSpatOrbs;

  if (commrank == 0) {
    // TXT
    {
      pout << "(save txt file)" << endl;
      pout << std::flush;
      char file[5000];
      sprintf(file, "%s/spatial3RDM.%d.%d.txt", schd.prefix[0].c_str(), root,
              root);
      std::ofstream ofs(file, std::ios::out);
      ofs << nSpatOrbs << endl;
      for (int n0 = 0; n0 < nSpatOrbs; n0++)
        for (int n1 = 0; n1 < nSpatOrbs; n1++)
          for (int n2 = 0; n2 < nSpatOrbs; n2++)
            for (int n3 = 0; n3 < nSpatOrbs; n3++)
              for (int n4 = 0; n4 < nSpatOrbs; n4++)
                for (int n5 = 0; n5 < nSpatOrbs; n5++) {
                  if (abs(s3RDM(n0 * nSpatOrbs2 + n1 * nSpatOrbs + n2,
                                n3 * nSpatOrbs2 + n4 * nSpatOrbs + n5)) >
                      1.e-12)
                    ofs << str(
                        boost::format(
                            "%3d   %3d   %3d   %3d   %3d   %3d   %20.14e\n") %
                        n0 % n1 % n2 % n3 % n4 % n5 %
                        s3RDM(n0 * nSpatOrbs2 + n1 * nSpatOrbs + n2,
                              n3 * nSpatOrbs2 + n4 * nSpatOrbs + n5));
                }
      ofs.close();
    }

    // BIN
    {
      pout << "(save bin file)" << endl;
      pout << std::flush;
      char file[5000];
      sprintf(file, "%s/spatial3RDM.%d.%d.bin", schd.prefix[0].c_str(), root,
              root);
      std::ofstream ofs(file, std::ios::binary);
      boost::archive::binary_oarchive save(ofs);
      save << s3RDM;
    }

    // SpinRDM
    if (schd.DoSpinRDM) {
      pout << "(save bkp file)" << endl;
      pout << std::flush;
      char file[5000];
      sprintf(file, "%s/%d-spin3RDM.bkp", schd.prefix[0].c_str(), root);
      std::ofstream ofs(file, std::ios::binary);
      boost::archive::binary_oarchive save(ofs);
      save << threeRDM;
    }

  }  // commrank
}

void SHCIrdm::save4RDM(schedule &schd, MatrixXx &fourRDM, MatrixXx &s4RDM,
                       int root, int norbs) {
  int n = norbs / 2;
  int n2 = n * n;
  int n3 = n2 * n;

  if (commrank == 0) {
    // TXT
    {
      pout << "(save txt file)" << endl;
      pout << std::flush;
      char file[5000];
      sprintf(file, "%s/spatial4RDM.%d.%d.txt", schd.prefix[0].c_str(), root,
              root);
      std::ofstream ofs(file, std::ios::out);
      ofs << n << endl;
      for (int c0 = 0; c0 < n; c0++)
        for (int c1 = 0; c1 < n; c1++)
          for (int c2 = 0; c2 < n; c2++)
            for (int c3 = 0; c3 < n; c3++)
              for (int d0 = 0; d0 < n; d0++)
                for (int d1 = 0; d1 < n; d1++)
                  for (int d2 = 0; d2 < n; d2++)
                    for (int d3 = 0; d3 < n; d3++) {
                      if (abs(s4RDM(n3 * c0 + n2 * c1 + n * c2 + c3,
                                    n3 * d0 + n2 * d1 + n * d2 + d3)) > 1.e-12)
                        ofs << str(
                            boost::format("%3d   %3d   %3d   %3d   %3d   %3d   "
                                          "%3d   %3d   %20.14e\n") %
                            c0 % c1 % c2 % c3 % d0 % d1 % d2 % d3 %
                            s4RDM(n3 * c0 + n2 * c1 + n * c2 + c3,
                                  n3 * d0 + n2 * d1 + n * d2 + d3));
                    }
      ofs.close();
    }

    // SpinRDM
    if (schd.DoSpinRDM) {
      pout << "(save bkp file)" << endl;
      pout << std::flush;
      char file[5000];
      sprintf(file, "%s/%d-spin4RDM.bkp", schd.prefix[0].c_str(), root);
      std::ofstream ofs(file, std::ios::binary);
      boost::archive::binary_oarchive save(ofs);
      save << fourRDM;
    }

    // BIN
    {
      pout << "(save bin file)" << endl;
      pout << std::flush;
      char file[5000];
      sprintf(file, "%s/spatial4RDM.%d.%d.bin", schd.prefix[0].c_str(), root,
              root);
      std::ofstream ofs(file, std::ios::binary);
      boost::archive::binary_oarchive save(ofs);
      save << s4RDM;
    }

  }  // commrank
}

//=============================================================================
void SHCIrdm::UpdateRDMResponsePerturbativeDeterministic(
    Determinant *Dets, int DetsSize, CItype *ci, double &E0, oneInt &I1,
    twoInt &I2, schedule &schd, double coreE, int nelec, int norbs,
    StitchDEH &uniqueDEH, int root, double &Psi1Norm, MatrixXx &s2RDM,
    MatrixXx &twoRDM) {
  /*!

  Update variational 2RDMs with perturbative contributions.

  :Arguments:

      Determinant *Dets:
          Pointer to determinants in wavefunction.
      int DetsSize:
          Number of determinants in wavefunction.
      CItype *ci:
          Pointer to ci coefficients for wavefunction.
      double& E0:
          Variational energy.
      oneInt& I1:
         One body integrals.
      twoInt& I2:
         Two body integrals.
      schedule& schd:
          Schedule that holds the parameters used throughout Dice.
      double coreE:
          Core energy.
      int nelec:
          Number of electrons.
      int norbs:
          Number of orbitals in active space.
      StitchDEH& uniqueDEH:
          TODO
      int root:
          Index of the wavefunction to save.
      double& Psi1Norm:
          Normalization constant for :math:`\Psi_1`. TODO
      MatrixXx& s2RDM:
          Spatial 2RDM.
      MatrixXx& twoRDM:
          Spin 2RDM.

  */

  s2RDM *= (1. - Psi1Norm);

  int nSpatOrbs = norbs / 2;

  vector<Determinant> &uniqueDets = *uniqueDEH.Det;
  vector<double> &uniqueEnergy = *uniqueDEH.Energy;
  vector<CItype> &uniqueNumerator = *uniqueDEH.Num;
  vector<vector<int>> &uniqueVarIndices = *uniqueDEH.var_indices;
  vector<vector<size_t>> &uniqueOrbDiff = *uniqueDEH.orbDifference;

  for (size_t i = 0; i < uniqueDets.size(); i++) {
    vector<int> closed(nelec, 0);
    vector<int> open(norbs - nelec, 0);
    uniqueDets[i].getOpenClosed(open, closed);

    CItype coeff = uniqueNumerator[i] / (E0 - uniqueEnergy[i]);
    //<Di| Gamma |Di>
    for (int n1 = 0; n1 < nelec; n1++) {
      for (int n2 = 0; n2 < n1; n2++) {
        int orb1 = closed[n1], orb2 = closed[n2];
        if (schd.DoSpinRDM)
#ifdef Complex
          twoRDM(orb1 * (orb1 + 1) / 2 + orb2, orb1 * (orb1 + 1) / 2 + orb2) +=
              conj(coeff) * coeff;
#else
          twoRDM(orb1 * (orb1 + 1) / 2 + orb2, orb1 * (orb1 + 1) / 2 + orb2) +=
              coeff * coeff;
#endif

#ifdef Complex
        populateSpatialRDM(orb1, orb2, orb1, orb2, s2RDM, conj(coeff) * coeff,
                           nSpatOrbs);
#else
        populateSpatialRDM(orb1, orb2, orb1, orb2, s2RDM, coeff * coeff,
                           nSpatOrbs);
#endif
      }
    }
  }

  for (size_t k = 0; k < uniqueDets.size(); k++) {
    for (size_t i = 0; i < uniqueVarIndices[k].size(); i++) {
      int d0 = uniqueOrbDiff[k][i] % norbs,
          c0 = (uniqueOrbDiff[k][i] / norbs) % norbs;

      if (uniqueOrbDiff[k][i] / norbs / norbs == 0) {  // single excitation
        vector<int> closed(nelec, 0);
        vector<int> open(norbs - nelec, 0);
        Dets[uniqueVarIndices[k][i]].getOpenClosed(open, closed);
        for (int n1 = 0; n1 < nelec; n1++) {
          double sgn = 1.0;
          int a = max(closed[n1], c0), b = min(closed[n1], c0),
              I = max(closed[n1], d0), J = min(closed[n1], d0);
          if (closed[n1] == d0) continue;
          uniqueDets[k].parity(min(d0, c0), max(d0, c0), sgn);
          // Dets[uniqueVarIndices[k][i]].parity(min(d0,c0), max(d0,c0),sgn);
          if (!((closed[n1] > c0 && closed[n1] > d0) ||
                (closed[n1] < c0 && closed[n1] < d0)))
            sgn *= -1.;
          if (schd.DoSpinRDM) {
            twoRDM(a * (a + 1) / 2 + b, I * (I + 1) / 2 + J) +=
                1.0 * sgn * uniqueNumerator[k] * ci[uniqueVarIndices[k][i]] /
                (E0 - uniqueEnergy[k]);
            twoRDM(I * (I + 1) / 2 + J, a * (a + 1) / 2 + b) +=
                1.0 * sgn * uniqueNumerator[k] * ci[uniqueVarIndices[k][i]] /
                (E0 - uniqueEnergy[k]);
          }
          populateSpatialRDM(a, b, I, J, s2RDM,
                             1.0 * sgn * uniqueNumerator[k] *
                                 ci[uniqueVarIndices[k][i]] /
                                 (E0 - uniqueEnergy[k]),
                             nSpatOrbs);
          populateSpatialRDM(I, J, a, b, s2RDM,
                             1.0 * sgn * uniqueNumerator[k] *
                                 ci[uniqueVarIndices[k][i]] /
                                 (E0 - uniqueEnergy[k]),
                             nSpatOrbs);
        }     // for n1
      }       // single
      else {  // double excitation
        int d1 = (uniqueOrbDiff[k][i] / norbs / norbs) % norbs,
            c1 = (uniqueOrbDiff[k][i] / norbs / norbs / norbs) % norbs;
        double sgn = 1.0;
        uniqueDets[k].parity(c1, c0, d1, d0, sgn);
        // Dets[uniqueVarIndices[k][i]].parity(d1,d0,c1,c0,sgn);
        int P = max(c1, c0), Q = min(c1, c0), R = max(d1, d0), S = min(d1, d0);
        if (P != c0) sgn *= -1;
        if (Q != d0) sgn *= -1;

        if (schd.DoSpinRDM) {
          twoRDM(P * (P + 1) / 2 + Q, R * (R + 1) / 2 + S) +=
              1.0 * sgn * uniqueNumerator[k] * ci[uniqueVarIndices[k][i]] /
              (E0 - uniqueEnergy[k]);
          twoRDM(R * (R + 1) / 2 + S, P * (P + 1) / 2 + Q) +=
              1.0 * sgn * uniqueNumerator[k] * ci[uniqueVarIndices[k][i]] /
              (E0 - uniqueEnergy[k]);
        }

        populateSpatialRDM(P, Q, R, S, s2RDM,
                           1.0 * sgn * uniqueNumerator[k] *
                               ci[uniqueVarIndices[k][i]] /
                               (E0 - uniqueEnergy[k]),
                           nSpatOrbs);
        populateSpatialRDM(R, S, P, Q, s2RDM,
                           1.0 * sgn * uniqueNumerator[k] *
                               ci[uniqueVarIndices[k][i]] /
                               (E0 - uniqueEnergy[k]),
                           nSpatOrbs);
      }  // If
    }    // i in variational connections to PT det k
  }      // k in PT dets

#ifndef SERIAL
  if (schd.DoSpinRDM)
    MPI_Allreduce(MPI_IN_PLACE, &twoRDM(0, 0), twoRDM.rows() * twoRDM.cols(),
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &s2RDM(0, 0), s2RDM.rows() * s2RDM.cols(),
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
}

//=============================================================================
void SHCIrdm::populateSpatialRDM(int &i, int &j, int &k, int &l,
                                 MatrixXx &s2RDM, CItype value,
                                 int &nSpatOrbs) {
  // we assume i != j  and  k != l
  int I = i / 2, J = j / 2, K = k / 2, L = l / 2;
  if (i % 2 == l % 2 && j % 2 == k % 2) {
    s2RDM(I * nSpatOrbs + J, L * nSpatOrbs + K) -= value;
    s2RDM(J * nSpatOrbs + I, K * nSpatOrbs + L) -= value;
  }

  if (i % 2 == k % 2 && l % 2 == j % 2) {
    s2RDM(I * nSpatOrbs + J, K * nSpatOrbs + L) += value;
    s2RDM(J * nSpatOrbs + I, L * nSpatOrbs + K) += value;
  }
}

//=============================================================================
void SHCIrdm::EvaluateRDM(vector<vector<int>> &connections, Determinant *Dets,
                          int DetsSize, CItype *cibra, CItype *ciket,
                          vector<vector<size_t>> &orbDifference, int nelec,
                          schedule &schd, int root, MatrixXx &twoRDM,
                          MatrixXx &s2RDM) {
  /*!

  Calculates the spatial (and spin if specified) 2RDMs based on the ci
  coefficient in cibra and ciket.

  :Arguments:

      vector<vector<int> >& connections:
          Linked list showing determinants that are connected to each other.
      Determinant *Dets:
          Pointer to determinants in wavefunction.
      int DetsSize:
          Number of determinants in wavefunction.
      CItype *cibr:
          Pointer to the ci coefficients for the bra.
      CItype *ciket:
          Pointer to the ci coefficients for the ket.
      vector<vector<size_t> >& orbDifference:
          Linked list that stores the orbital difference between determinants.
      int nelec:
          Number of electrons.
      schedule& schd:
          Schedule that holds the parameters used throughout Dice.
      int root:
          Index of the wavefunction to save.
      MatrixXx& s2RDM:
          Spatial 2RDM.
      MatrixXx& twoRDM:
          Spin 2RDM.
   */
  size_t norbs = Dets[0].norbs;
  int nSpatOrbs = norbs / 2;

  for (int i = 0; i < DetsSize; i++) {
    if (i % commsize != commrank) continue;

    vector<int> closed(nelec, 0);
    vector<int> open(norbs - nelec, 0);
    Dets[i].getOpenClosed(open, closed);

    //<Di| Gamma |Di>
    for (int n1 = 0; n1 < nelec; n1++)
      for (int n2 = 0; n2 < n1; n2++) {
        int orb1 = closed[n1], orb2 = closed[n2];
        if (schd.DoSpinRDM)
          twoRDM(orb1 * (orb1 + 1) / 2 + orb2, orb1 * (orb1 + 1) / 2 + orb2) +=
              localConj::conj(cibra[i]) * ciket[i];
        populateSpatialRDM(orb1, orb2, orb1, orb2, s2RDM,
                           localConj::conj(cibra[i]) * ciket[i], nSpatOrbs);
      }  // end n1 n2

    for (int j = 1; j < connections[i / commsize].size(); j++) {
      // if (i == connections[i/commsize][j]) continue;
      int d0 = orbDifference[i / commsize][j] % norbs;
      int c0 = (orbDifference[i / commsize][j] / norbs) % norbs;

      if (orbDifference[i / commsize][j] / norbs / norbs ==
          0) {  // only single excitation
        for (int n1 = 0; n1 < nelec; n1++) {
          double sgn = 1.0;
          int a = max(closed[n1], c0), b = min(closed[n1], c0),
              I = max(closed[n1], d0), J = min(closed[n1], d0);
          if (closed[n1] == d0) continue;
          if (closed[n1] == c0) continue;  // TODO REMOVE
          Dets[i].parity(min(d0, c0), max(d0, c0), sgn);
          if (!((closed[n1] > c0 && closed[n1] > d0) ||
                (closed[n1] < c0 && closed[n1] < d0)))
            sgn *= -1.;
          if (schd.DoSpinRDM) {
            twoRDM(a * (a + 1) / 2 + b, I * (I + 1) / 2 + J) +=
                sgn * localConj::conj(cibra[connections[i / commsize][j]]) *
                ciket[i];
            twoRDM(I * (I + 1) / 2 + J, a * (a + 1) / 2 + b) +=
                sgn * localConj::conj(ciket[connections[i / commsize][j]]) *
                cibra[i];
          }
          populateSpatialRDM(
              a, b, I, J, s2RDM,
              sgn * localConj::conj(cibra[connections[i / commsize][j]]) *
                  ciket[i],
              nSpatOrbs);
          populateSpatialRDM(
              I, J, a, b, s2RDM,
              sgn * localConj::conj(ciket[connections[i / commsize][j]]) *
                  cibra[i],
              nSpatOrbs);
        }  // end n1

      } else {
        int d1 = (orbDifference[i / commsize][j] / norbs / norbs) % norbs;
        int c1 =
            (orbDifference[i / commsize][j] / norbs / norbs / norbs) % norbs;
        double sgn = 1.0;

        Dets[i].parity(d1, d0, c1, c0, sgn);
        if (schd.DoSpinRDM) {
          twoRDM(c1 * (c1 + 1) / 2 + c0, d1 * (d1 + 1) / 2 + d0) +=
              sgn * localConj::conj(cibra[connections[i / commsize][j]]) *
              ciket[i];
          twoRDM(d1 * (d1 + 1) / 2 + d0, c1 * (c1 + 1) / 2 + c0) +=
              sgn * localConj::conj(ciket[connections[i / commsize][j]]) *
              cibra[i];
        }
        populateSpatialRDM(
            c1, c0, d1, d0, s2RDM,
            sgn * localConj::conj(cibra[connections[i / commsize][j]]) *
                ciket[i],
            nSpatOrbs);
        populateSpatialRDM(
            d1, d0, c1, c0, s2RDM,
            sgn * localConj::conj(ciket[connections[i / commsize][j]]) *
                cibra[i],
            nSpatOrbs);
      }  // end if

    }  // end j
  }    // end i

#ifndef SERIAL
  if (schd.DoSpinRDM)
    MPI_Allreduce(MPI_IN_PLACE, &twoRDM(0, 0), twoRDM.rows() * twoRDM.cols(),
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &s2RDM(0, 0), s2RDM.rows() * s2RDM.cols(),
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
}

//=============================================================================
void SHCIrdm::EvaluateOneRDM(vector<vector<int>> &connections,
                             Determinant *Dets, int DetsSize, CItype *cibra,
                             CItype *ciket,
                             vector<vector<size_t>> &orbDifference, int nelec,
                             schedule &schd, int root, MatrixXx &oneRDM,
                             MatrixXx &s1RDM) {
  /*!
  Calculates *just* the spatial 1RDM using cibra and ciket.

  :Arguments:

      vector<vector<int> >& connections:
          Linked list showing determinants that are connected to each other.
      Determinant * Dets:
          Pointer to determinants in wavefunction.
      int DetsSize  :
          Number of determinants.
      CItype *cibra:
          Pointer to the ci coefficients for the bra.
      CItype *ciket:
          Pointer to the ci coefficients for the ket.
      vector<vector<size_t> >& orbDifference:
          Linked list that stores the orbital difference between determinants.
      int nelec:
          Number of electrons.
      schedule& schd:
          Schedule that holds the parameters used throughout Dice.
      int root:
          Index of the wavefunction to save.
      MatrixXx& oneRDM:
          Spin 1RDM.
      MatrixXx& s1RDM:
          Spatial 1RDM.

  */

#ifndef SERIAL
  boost::mpi::communicator world;
#endif

  size_t norbs = Dets[0].norbs;
  int nSpatOrbs = norbs / 2;

  //#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < DetsSize; i++) {
    if (i % commsize != commrank) continue;

    vector<int> closed(nelec, 0);
    vector<int> open(norbs - nelec, 0);
    Dets[i].getOpenClosed(open, closed);

    //<Di| Gamma |Di>
    for (int n1 = 0; n1 < nelec; n1++) {
      int orb1 = closed[n1];
      oneRDM(orb1, orb1) += localConj::conj(cibra[i]) * ciket[i];
      s1RDM(orb1 / 2, orb1 / 2) += localConj::conj(cibra[i]) * ciket[i];
    }

    for (int j = 1; j < connections[i / commsize].size(); j++) {
      int d0 = orbDifference[i / commsize][j] % norbs,
          c0 = (orbDifference[i / commsize][j] / norbs) % norbs;
      if (orbDifference[i / commsize][j] / norbs / norbs ==
          0) {  // only single excitation
        double sgn = 1.0;
        Dets[i].parity(min(c0, d0), max(c0, d0), sgn);

        oneRDM(c0, d0) += sgn *
                          localConj::conj(cibra[connections[i / commsize][j]]) *
                          ciket[i];
        oneRDM(d0, c0) += sgn *
                          localConj::conj(cibra[connections[i / commsize][j]]) *
                          ciket[i];

        s1RDM(c0 / 2, d0 / 2) +=
            sgn * localConj::conj(cibra[connections[i / commsize][j]]) *
            ciket[i];
        s1RDM(d0 / 2, c0 / 2) +=
            sgn * localConj::conj(cibra[connections[i / commsize][j]]) *
            ciket[i];
      }
    }
  }

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &oneRDM(0, 0), oneRDM.rows() * oneRDM.cols(),
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &s1RDM(0, 0), s1RDM.rows() * s1RDM.cols(),
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  // Testing TODO
  // double test_elec = 0;
  // for (int i = 0; i < nSpatOrbs; i++) {
  //   test_elec += s1RDM(i, i);
  // }
  // cout << "Test electrons: " << test_elec << endl;

  // MatrixXx test_1RDM = MatrixXx::Zero(nSpatOrbs, nSpatOrbs);
  // for (int r = 0; r < norbs; r++)
  //   for (int c = 0; c < norbs; c++) {
  //     test_1RDM(r / 2, c / 2) += oneRDM(r, c);
  //   }

  // test_1RDM -= s1RDM;
  // cout << "Norm of difference in 1RDM: " << test_1RDM.norm() << endl;
}

//=============================================================================
double SHCIrdm::ComputeEnergyFromSpinRDM(int norbs, int nelec, oneInt &I1,
                                         twoInt &I2, double coreE,
                                         MatrixXx &twoRDM) {
  /*!
  Compute the energy of wavefunction from spin 2RDM.

  :Arguments:

      int norbs:
          Number of orbitals in the active space.
      int nelec:
          Number of electrons in the active space.
      oneInt& I1:
         One body integrals.
      twoInt& I2:
         Two body integrals.
      double coreE:
          Core energy.
      MatrixXx& twoRDM:
          Spin 2RDM.

  */

  // RDM(i,j,k,l) = a_i^\dag a_j^\dag a_l a_k
  // also i>=j and k>=l
  double energy = coreE;
  double onebody = 0.0;
  double twobody = 0.0;
  // if (commrank == 0)  cout << "Core energy= " << energy << endl;

  MatrixXx oneRDM = MatrixXx::Zero(norbs, norbs);
#pragma omp parallel for schedule(dynamic)
  for (int p = 0; p < norbs; p++)
    for (int q = 0; q < norbs; q++)
      for (int r = 0; r < norbs; r++) {
        int P = max(p, r), R1 = min(p, r);
        int Q = max(q, r), R2 = min(q, r);
        double sgn = 1.;
        if (P != p) sgn *= -1;
        if (Q != q) sgn *= -1;
        oneRDM(p, q) += sgn *
                        twoRDM(P * (P + 1) / 2 + R1, Q * (Q + 1) / 2 + R2) /
                        (nelec - 1.);
      }

#pragma omp parallel for reduction(+ : onebody)
  for (int p = 0; p < norbs; p++)
    for (int q = 0; q < norbs; q++)
#ifdef Complex
      onebody += (I1(p, q) * oneRDM(p, q)).real();
#else
      onebody += I1(p, q) * oneRDM(p, q);
#endif

#pragma omp parallel for reduction(+ : twobody)
  for (int p = 0; p < norbs; p++)
    for (int q = 0; q < norbs; q++)
      for (int r = 0; r < norbs; r++)
        for (int s = 0; s < norbs; s++) {
          // if (p%2 != r%2 || q%2 != s%2)  continue; // This line is not
          // necessary
          int P = max(p, q), Q = min(p, q);
          int R = max(r, s), S = min(r, s);
          double sgn = 1;
          if (P != p) sgn *= -1;
          if (R != r) sgn *= -1;
#ifdef Complex
          twobody +=
              (sgn * 0.5 * twoRDM(P * (P + 1) / 2 + Q, R * (R + 1) / 2 + S) *
               I2(p, r, q, s))
                  .real();  // 2-body term
#else
          twobody += sgn * 0.5 *
                     twoRDM(P * (P + 1) / 2 + Q, R * (R + 1) / 2 + S) *
                     I2(p, r, q, s);  // 2-body term
#endif
        }

  energy += onebody + twobody;
  pout << format("E(one-body) from 2RDM: %18.10f") % (onebody) << endl;
  pout << format("E(two-body) from 2RDM: %18.10f") % (twobody) << endl;
  pout << format("E from 2RDM:           %18.10f") % (energy) << endl;
  return energy;
}

//=============================================================================
double SHCIrdm::ComputeEnergyFromSpatialRDM(int norbs, int nelec, oneInt &I1,
                                            twoInt &I2, double coreE,
                                            MatrixXx &twoRDM) {
  /*!
  Computes the energy of the wavefunction by contracting the one and two body
  integrals with the spatial 1/2-RDMs.

  :Arguments:

      int norbs:
          Number of orbitals in the active space.
      int nelec:
          Number of electrons in the active space.
      oneInt& I1:
         One body integrals.
      twoInt& I2:
         Two body integrals.
      double coreE:
          Core energy.
      MatrixXx& twoRDM:
          Spatial 2RDM. TODO make the name of this input consistent.
  */

  double energy = coreE;
  double onebody = 0.0;
  double twobody = 0.0;

  MatrixXx oneRDM = MatrixXx::Zero(norbs, norbs);
#pragma omp parallel for schedule(dynamic)
  for (int p = 0; p < norbs; p++)
    for (int q = 0; q < norbs; q++)
      for (int r = 0; r < norbs; r++)
        oneRDM(p, q) +=
            twoRDM(p * norbs + r, q * norbs + r) / (1. * nelec - 1.);

#pragma omp parallel for reduction(+ : onebody)
  for (int p = 0; p < norbs; p++)
    for (int q = 0; q < norbs; q++)
#ifdef Complex
      onebody += (I1(2 * p, 2 * q) * oneRDM(p, q)).real();
#else
      onebody += I1(2 * p, 2 * q) * oneRDM(p, q);
#endif

#pragma omp parallel for reduction(+ : twobody)
  for (int p = 0; p < norbs; p++)
    for (int q = 0; q < norbs; q++)
      for (int r = 0; r < norbs; r++)
        for (int s = 0; s < norbs; s++)
#ifdef Complex
          twobody += (0.5 * twoRDM(p * norbs + q, r * norbs + s) *
                      I2(2 * p, 2 * r, 2 * q, 2 * s))
                         .real();  // 2-body term
#else
          twobody += 0.5 * twoRDM(p * norbs + q, r * norbs + s) *
                     I2(2 * p, 2 * r, 2 * q, 2 * s);  // 2-body term
#endif

  energy += onebody + twobody;
  pout << format("E(one-body) from 2RDM: %18.10f") % (onebody) << endl;
  pout << format("E(two-body) from 2RDM: %18.10f") % (twobody) << endl;
  pout << format("E from 2RDM:           %18.10f") % (energy) << endl;
  return energy;
}

/*
 *** 3-RDM Methods
 */
inline void SHCIrdm::getUniqueIndices(Determinant &bra, Determinant &ket,
                                      vector<int> &cs, vector<int> &ds) {
  // Appends two lists of creation and annihilation operator indices.
  for (int i = 0; i < bra.norbs; i++) {
    if (bra.getocc(i) == ket.getocc(i)) {
      continue;
    } else {
      if (ket.getocc(i) == 0)
        cs.push_back(i);
      else
        ds.push_back(i);
    }
  }
  return;
}

int genIdx(const int &a, const int &b, const int &c, const size_t &norbs) {
  return a * norbs * norbs + b * norbs + c;
}

void popSpin3RDM(vector<int> &cs, vector<int> &ds, CItype value, size_t &norbs,
                 MatrixXx &threeRDM) {
  // d2->c0    d1->c1    d2->c0
  int cI[] = {0, 1, 2};
  int dI[] = {0, 1, 2};
  int ctr = 0;
  vector<double> pars(6);
  pars[0] = 1.;
  pars[1] = -1.;
  pars[2] = -1.;
  pars[3] = 1.;
  pars[4] = 1.;
  pars[5] = -1.;
  double par;

  do {
    do {
      par = pars[ctr / 6] * pars[ctr % 6];
      ctr++;
      threeRDM(genIdx(cs[cI[0]], cs[cI[1]], cs[cI[2]], norbs),
               genIdx(ds[dI[0]], ds[dI[1]], ds[dI[2]], norbs)) += par * value;
    } while (next_permutation(dI, dI + 3));
  } while (next_permutation(cI, cI + 3));
  return;
}

void SHCIrdm::popSpatial3RDM(vector<int> &cs, vector<int> &ds, CItype value,
                             size_t &norbs, MatrixXx &s3RDM) {
  // d2->c0    d1->c1    d2->c0
  int cI[] = {0, 1, 2};
  int dI[] = {0, 1, 2};
  int ctr = 0;
  vector<double> pars(6);
  pars[0] = 1.;
  pars[1] = -1.;
  pars[2] = -1.;
  pars[3] = 1.;
  pars[4] = 1.;
  pars[5] = -1.;
  double par;

  do {
    do {
      par = pars[ctr / 6] * pars[ctr % 6];
      ctr++;
      if (cs[cI[0]] % 2 == ds[dI[2]] % 2 && cs[cI[1]] % 2 == ds[dI[1]] % 2 &&
          cs[cI[2]] % 2 == ds[dI[0]] % 2) {
        s3RDM(genIdx(cs[cI[0]] / 2, cs[cI[1]] / 2, cs[cI[2]] / 2, norbs / 2),
              genIdx(ds[dI[0]] / 2, ds[dI[1]] / 2, ds[dI[2]] / 2, norbs / 2)) +=
            par * value;
      }
    } while (next_permutation(dI, dI + 3));
  } while (next_permutation(cI, cI + 3));
  return;
}

void SHCIrdm::Evaluate3RDM(Determinant *Dets, int DetsSize, CItype *cibra,
                           CItype *ciket, int nelec, schedule &schd, int root,
                           MatrixXx &threeRDM, MatrixXx &s3RDM) {
  /*
    TODO optimize speed and memory
    TODO Add second instance of pop*RDM functions that switches cs as ds
  */
#ifndef SERIAL
  boost::mpi::communicator world;
#endif

  size_t norbs = Dets[0].norbs;
  int nSpatOrbs = norbs / 2;
  int nSpatOrbs2 = nSpatOrbs * nSpatOrbs;

  // Pairs of determinants
  for (int b = 0; b < DetsSize; b++) {
    if (b % commsize != commrank) continue;
    Determinant DetsB = Dets[b];  // Necessary for MPI
    for (int k = 0; k < DetsSize; k++) {
      Determinant DetsK = Dets[k];  // Necessary for MPI

      // Distances and unique indexes
      int dist = DetsB.ExcitationDistance(DetsK);
      if (dist > 3) {
        continue;
      }
      vector<int> cs(0), ds(0);
      getUniqueIndices(DetsB, DetsK, cs, ds);

      // D=3
      if (dist == 3) {
        double sgn = 1.0;
        DetsK.parity(cs[0], cs[1], cs[2], ds[0], ds[1], ds[2], sgn);
        if (schd.DoSpinRDM)
          popSpin3RDM(cs, ds, sgn * real(conj(cibra[b]) * ciket[k]), norbs,
                      threeRDM);
        popSpatial3RDM(cs, ds, sgn * real(conj(cibra[b]) * ciket[k]), norbs,
                       s3RDM);
      }

      // D=2
      else if (dist == 2) {
        vector<int> closed(nelec, 0);
        vector<int> open(norbs - nelec, 0);
        DetsK.getOpenClosed(open, closed);

        // Initialize the final spot in operator arrays and move d ops toward
        // ket so inner two c/d operators are paired
        cs.push_back(0);
        ds.push_back(0);
        ds[2] = ds[1];
        ds[1] = ds[0];

        for (int x = 0; x < nelec; x++) {
          cs[2] = closed[x];
          ds[0] = closed[x];
          if (closed[x] == ds[1] || closed[x] == ds[2]) continue;
          if (closed[x] == cs[0] || closed[x] == cs[1])
            continue;  // TODO CHECK W/ SS

          // Gamma = c0 c1 c2 d0 d1 d2
          double sgn = 1.0;
          DetsK.parity(ds[2], ds[1], cs[0], cs[1], sgn);  // TOOD

          if (schd.DoSpinRDM)
            popSpin3RDM(cs, ds, sgn * real(conj(cibra[b]) * ciket[k]), norbs,
                        threeRDM);
          popSpatial3RDM(cs, ds, sgn * real(conj(cibra[b]) * ciket[k]), norbs,
                         s3RDM);
        }  // end x
      }

      // D=1
      else if (dist == 1) {
        vector<int> closed(nelec, 0);
        vector<int> open(norbs - nelec, 0);
        DetsK.getOpenClosed(open, closed);

        cs.push_back(0);
        cs.push_back(0);
        ds.push_back(0);
        ds.push_back(0);
        ds[2] = ds[0];

        for (int x = 0; x < nelec; x++) {
          cs[1] = closed[x];
          ds[1] = closed[x];
          if (closed[x] == cs[0] || closed[x] == ds[2]) continue;
          for (int y = 0; y < x; y++) {
            cs[2] = closed[y];
            ds[0] = closed[y];
            if (closed[y] == ds[1] || closed[y] == ds[2]) continue;
            if (closed[y] == cs[0] || closed[y] == cs[1])
              continue;  // TODO CHECK W/ SS

            // Gamma = c0 c1 c2 d0 d1 d2
            double sgn = 1.0;
            DetsK.parity(min(cs[0], ds[2]), max(cs[0], ds[2]),
                         sgn);  // TODO Update repop order

            if (schd.DoSpinRDM)
              popSpin3RDM(cs, ds, sgn * real(conj(cibra[b]) * ciket[k]), norbs,
                          threeRDM);
            popSpatial3RDM(cs, ds, sgn * real(conj(cibra[b]) * ciket[k]), norbs,
                           s3RDM);
          }  // end y
        }    // end x
      }

      // D=0
      else if (dist == 0) {
        vector<int> closed(nelec, 0);
        vector<int> open(norbs - nelec, 0);
        DetsK.getOpenClosed(open, closed);

        cs.push_back(0);
        cs.push_back(0);
        cs.push_back(0);
        ds.push_back(0);
        ds.push_back(0);
        ds.push_back(0);

        for (int x = 0; x < nelec; x++) {
          cs[0] = closed[x];
          ds[2] = closed[x];
          for (int y = 0; y < x; y++) {
            cs[1] = closed[y];
            ds[1] = closed[y];
            for (int z = 0; z < y; z++) {
              cs[2] = closed[z];
              ds[0] = closed[z];

              if (schd.DoSpinRDM)
                popSpin3RDM(cs, ds, real(conj(cibra[b]) * ciket[k]), norbs,
                            threeRDM);
              popSpatial3RDM(cs, ds, real(conj(cibra[b]) * ciket[k]), norbs,
                             s3RDM);
            }  // end z
          }    // end y
        }      // end x

      }  // end if dist
    }    // end k
  }      // end b

#ifndef SERIAL
  world.barrier();
  if (schd.DoSpinRDM)
    MPI_Allreduce(MPI_IN_PLACE, &threeRDM(0, 0),
                  threeRDM.rows() * threeRDM.cols(), MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &s3RDM(0, 0), s3RDM.rows() * s3RDM.cols(),
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
}

/*
 *** 4-RDM Methods
 */
int gen4Idx(const int &a, const int &b, const int &c, const int &d,
            int &norbs) {
  return a * norbs * norbs * norbs + b * norbs * norbs + c * norbs + d;
}

void popSpin4RDM(vector<int> &cs, vector<int> &ds, CItype value, int &norbs,
                 MatrixXx &fourRDM) {
  int cI[] = {0, 1, 2, 3};
  int dI[] = {0, 1, 2, 3};
  int ctr = 0;
  double pars[] = {1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1,
                   1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1};
  double par = 1.0;

  do {
    do {
      par = pars[ctr / 24] * pars[ctr % 24];
      fourRDM(gen4Idx(cs[cI[0]], cs[cI[1]], cs[cI[2]], cs[cI[3]], norbs),
              gen4Idx(ds[dI[0]], ds[dI[1]], ds[dI[2]], ds[dI[3]], norbs)) +=
          par * value;
      ctr++;
    } while (next_permutation(dI, dI + 4));
  } while (next_permutation(cI, cI + 4));
  return;
}

void SHCIrdm::popSpatial4RDM(vector<int> &cs, vector<int> &ds, CItype value,
                             int &nSOs, MatrixXx &s4RDM) {
  int cI[] = {0, 1, 2, 3};
  int dI[] = {0, 1, 2, 3};
  int ctr = 0;
  double pars[] = {1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1,
                   1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1};
  double par = 1.0;

  do {
    do {
      par = pars[ctr / 24] * pars[ctr % 24];
      if (cs[cI[0]] % 2 == ds[dI[3]] % 2 && cs[cI[1]] % 2 == ds[dI[2]] % 2 &&
          cs[cI[2]] % 2 == ds[dI[1]] % 2 && cs[cI[3]] % 2 == ds[dI[0]] % 2) {
        s4RDM(gen4Idx(cs[cI[0]] / 2, cs[cI[1]] / 2, cs[cI[2]] / 2,
                      cs[cI[3]] / 2, nSOs),
              gen4Idx(ds[dI[0]] / 2, ds[dI[1]] / 2, ds[dI[2]] / 2,
                      ds[dI[3]] / 2, nSOs)) += par * value;
      }
      ctr++;
    } while (next_permutation(dI, dI + 4));
  } while (next_permutation(cI, cI + 4));
  return;
}

void SHCIrdm::Evaluate4RDM(Determinant *Dets, int DetsSize, CItype *cibra,
                           CItype *ciket, int nelec, schedule &schd, int root,
                           MatrixXx &fourRDM, MatrixXx &s4RDM) {
  /*
     TODO optimize speed and memory
  */
#ifndef SERIAL
  boost::mpi::communicator world;
#endif

  int norbs = Dets[0].norbs;
  int nSOs = norbs / 2;  // Number of spatial orbitals

  // Pairs of determinants
  for (int b = 0; b < DetsSize; b++) {
    if (b % commsize != commrank) continue;
    Determinant DetsB = Dets[b];
    for (int k = 0; k < DetsSize; k++) {
      Determinant DetsK = Dets[k];

      // Distances and unique indexes
      int dist = DetsB.ExcitationDistance(DetsK);
      if (dist > 4) {
        continue;
      }
      vector<int> cs(0), ds(0);
      getUniqueIndices(DetsB, DetsK, cs, ds);

      // D=4
      if (dist == 4) {
        double sgn = 1.0;
        DetsK.parity(cs[0], cs[1], cs[2], cs[3], ds[0], ds[1], ds[2], ds[3],
                     sgn);
        // popSpin4RDM(cs,ds,sgn*conj(cibra(b,0))*ciket(k,0),norbs,fourRDM);
        if (schd.DoSpinRDM)
          popSpin4RDM(cs, ds, sgn * real(conj(cibra[b]) * ciket[k]), norbs,
                      fourRDM);
        popSpatial4RDM(cs, ds, sgn * real(conj(cibra[b]) * ciket[k]), nSOs,
                       s4RDM);
      }

      // D=3
      else if (dist == 3) {
        vector<int> closed(nelec, 0);
        vector<int> open(norbs - nelec, 0);
        DetsK.getOpenClosed(open, closed);

        cs.push_back(0);
        ds.push_back(0);
        ds[3] = ds[2];
        ds[2] = ds[1];
        ds[1] = ds[0];  // Keep notation

        for (int w = 0; w < nelec; w++) {
          cs[3] = closed[w];
          ds[0] = closed[w];
          if (closed[w] == cs[0] || closed[w] == cs[1] || closed[w] == cs[2])
            continue;
          if (closed[w] == ds[3] || closed[w] == ds[2] || closed[w] == ds[1])
            continue;

          double sgn = 1.0;
          DetsK.parity(cs[0], cs[1], cs[2], ds[1], ds[2], ds[3], sgn);

          if (schd.DoSpinRDM)
            popSpin4RDM(cs, ds, sgn * real(conj(cibra[b]) * ciket[k]), norbs,
                        fourRDM);
          popSpatial4RDM(cs, ds, sgn * real(conj(cibra[b]) * ciket[k]), nSOs,
                         s4RDM);
        }  // end w
      }

      // D=2
      else if (dist == 2) {
        vector<int> closed(nelec, 0);
        vector<int> open(norbs - nelec, 0);
        DetsK.getOpenClosed(open, closed);

        cs.push_back(0);
        cs.push_back(0);
        ds.push_back(0);
        ds.push_back(0);
        ds[3] = ds[1];
        ds[2] = ds[0];  // Keep notation

        for (int w = 0; w < nelec; w++) {
          cs[2] = closed[w];
          ds[1] = closed[w];
          if (closed[w] == cs[0] || closed[w] == cs[1]) continue;
          if (closed[w] == ds[3] || closed[w] == ds[2]) continue;
          for (int x = 0; x < w; x++) {
            cs[3] = closed[x];
            ds[0] = closed[x];
            if (closed[x] == cs[0] || closed[x] == cs[1]) continue;
            if (closed[x] == ds[3] || closed[x] == ds[2]) continue;

            double sgn = 1.0;
            DetsK.parity(ds[3], ds[2], cs[0], cs[1], sgn);  // SS notation

            if (schd.DoSpinRDM)
              popSpin4RDM(cs, ds, sgn * real(conj(cibra[b]) * ciket[k]), norbs,
                          fourRDM);
            popSpatial4RDM(cs, ds, sgn * real(conj(cibra[b]) * ciket[k]), nSOs,
                           s4RDM);
          }  // end x
        }    // end w
      }

      // D=1
      else if (dist == 1) {
        vector<int> closed(nelec, 0);
        vector<int> open(norbs - nelec, 0);
        DetsK.getOpenClosed(open, closed);

        cs.push_back(0);
        cs.push_back(0);
        cs.push_back(0);
        ds.push_back(0);
        ds.push_back(0);
        ds.push_back(0);
        ds[3] = ds[0];  // Keep notation

        for (int w = 0; w < nelec; w++) {
          cs[1] = closed[w];
          ds[2] = closed[w];
          if (closed[w] == cs[0] || closed[w] == ds[3]) continue;
          for (int x = 0; x < w; x++) {
            cs[2] = closed[x];
            ds[1] = closed[x];
            if (closed[x] == cs[0] || closed[x] == ds[3]) continue;
            for (int y = 0; y < x; y++) {
              cs[3] = closed[y];
              ds[0] = closed[y];
              if (closed[y] == cs[0] || closed[y] == ds[3]) continue;

              double sgn = 1.0;
              DetsK.parity(min(ds[3], cs[0]), max(ds[3], cs[0]),
                           sgn);  // SS notation

              if (schd.DoSpinRDM)
                popSpin4RDM(cs, ds, sgn * real(conj(cibra[b]) * ciket[k]),
                            norbs, fourRDM);
              popSpatial4RDM(cs, ds, sgn * real(conj(cibra[b]) * ciket[k]),
                             nSOs, s4RDM);
            }  // end y
          }    // end x
        }      // end w
      }

      // D=0
      else if (dist == 0) {
        vector<int> closed(nelec, 0);
        vector<int> open(norbs - nelec, 0);
        DetsK.getOpenClosed(open, closed);

        cs.push_back(0);
        cs.push_back(0);
        cs.push_back(0);
        cs.push_back(0);
        ds.push_back(0);
        ds.push_back(0);
        ds.push_back(0);
        ds.push_back(0);
        for (int w = 0; w < nelec; w++) {
          cs[0] = closed[w];
          ds[3] = closed[w];
          for (int x = 0; x < w; x++) {
            cs[1] = closed[x];
            ds[2] = closed[x];
            for (int y = 0; y < x; y++) {
              cs[2] = closed[y];
              ds[1] = closed[y];
              for (int z = 0; z < y; z++) {
                cs[3] = closed[z];
                ds[0] = closed[z];

                if (schd.DoSpinRDM)
                  popSpin4RDM(cs, ds, real(conj(cibra[b]) * ciket[k]), norbs,
                              fourRDM);
                popSpatial4RDM(cs, ds, real(conj(cibra[b]) * ciket[k]), nSOs,
                               s4RDM);
              }  // end z
            }    // end y
          }      // end x
        }        // end w

      }  // end if dist
    }    // end k
  }      // end b

#ifndef SERIAL
  world.barrier();
  if (schd.DoSpinRDM)
    MPI_Allreduce(MPI_IN_PLACE, &fourRDM(0, 0), fourRDM.rows() * fourRDM.cols(),
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &s4RDM(0, 0), s4RDM.rows() * s4RDM.cols(),
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
}
