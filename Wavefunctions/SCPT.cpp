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

#include "SCPT.h"
#include "SelectedCI.h"
#include "igl/slice.h"
#include "igl/slice_into.h"

class oneInt;
class twoInt;
class twoIntHeatBathSHM;

template<typename Wfn>
SCPT<Wfn>::SCPT()
{
  wave.readWave();

  // Find which excitation classes are being considered. The classes are
  // labelled by integers from 0 to 8, and defined in SimpleWalker.h
  if (schd.nciCore == 0) {
    classesUsed[0] = true;
    classesUsed[1] = true;
    classesUsed[2] = true;
  } else {
    classesUsed[0] = true;
    classesUsed[1] = true;
    classesUsed[2] = true;
    classesUsed[3] = true;
    classesUsed[4] = true;
    //classesUsed[5] = true;
    classesUsed[6] = true;
    //classesUsed[7] = true;
    if (!schd.determCCVV)
      classesUsed[8] = true;
    else
      classesUsedDeterm[8] = true;

    // AAVV class
    normsDeterm[2] = true;
    // CAAV class
    normsDeterm[4] = true;
    // CCAA class
    normsDeterm[6] = true;
  }

  int numCore = schd.nciCore;
  int numVirt = Determinant::norbs - schd.nciCore - schd.nciAct;

  // The number of coefficients in each excitation class:
  // 0 holes, 0 particles:
  numCoeffsPerClass[0] = 1;
  // 0 holes, 1 particle:
  numCoeffsPerClass[1] = 2*numVirt;
  // 0 holes, 2 particles:
  numCoeffsPerClass[2] = 2*numVirt * (2*numVirt - 1) / 2;
  // 1 hole, 0 particles:
  numCoeffsPerClass[3] = 2*numCore;
  // 1 hole, 1 particle:
  numCoeffsPerClass[4] = (2*numCore) * (2*numVirt);
  // 1 hole, 2 particle:
  //numCoeffsPerClass[5] = (2*numCore) * (2*numVirt * (2*numVirt - 1) / 2);
  // 2 hole, 0 particles:
  numCoeffsPerClass[6] = 2*numCore * (2*numCore - 1) / 2;
  // 2 hole, 1 particle:
  //numCoeffsPerClass[7] = (2*numCore * (2*numCore - 1) / 2) * (2*numVirt);
  
  // Class 5 (2 holes, 1 particle), class 7 (1 holes, 2 particles) and
  // class 8 (2 holes, 2 particles) are more complicated. They are set up here:
  //createClassIndMap(numCoeffsPerClass[5], numCoeffsPerClass[7],  numCoeffsPerClass[8]);

  cumNumCoeffs[0] = 0;
  for (int i = 1; i < 9; i++)
  {
    // If the previous class (labelled i-1) is being used, add it to the
    // cumulative counter.

    if (classesUsed[i-1]) {
      cumNumCoeffs[i] = cumNumCoeffs[i-1] + numCoeffsPerClass[i-1];
    }
    else {
      cumNumCoeffs[i] = cumNumCoeffs[i-1];
    }
  }

  numCoeffs = 0;
  for (int i = 0; i < 9; i++)
  {
    // The total number of coefficients. Only include a class if that
    // class is being used.
    if (classesUsed[i]) numCoeffs += numCoeffsPerClass[i];
  }

  // Resize coeffs
  coeffs = VectorXd::Zero(numCoeffs);
  moEne = VectorXd::Zero(Determinant::norbs);

  //coeffs order: phi0, singly excited (spin orb index), doubly excited (spin orb pair index)

  // Use a constant amplitude for each contracted state except
  // the CASCI wave function.
  double amp = -std::min(0.5, 5.0/std::sqrt(numCoeffs));

  coeffs(0) = 1.0;
  for (int i=1; i < numCoeffs; i++) {
    coeffs(i) = amp;
  }
  
  
  char file[5000];
  sprintf(file, "ciCoeffs.txt");
  ifstream ofile(file);
  if (ofile) {
    for (int i = 0; i < coeffs.size(); i++) {
      ofile >> coeffs(i);
    }
  }
  
  char filem[5000];
  sprintf(filem, "moEne.txt");
  ifstream ofilem(filem);
  if (ofilem) {
    for (int i = 0; i < Determinant::norbs; i++) {
      ofilem >> moEne(i);
    }
  }
  else {
    if (commrank == 0) cout << "moEne.txt not found!\n";
    exit(0);
  }
}

template<typename Wfn>
void SCPT<Wfn>::createClassIndMap(int& numStates_1h2p, int& numStates_2h1p, int& numStates_2h2p) {
  // Loop over all combinations of vore and virtual pairs.
  // For each, see if the corresponding Hamiltonian element is non-zero.
  // If so, give it an index and put that index in a hash table for
  // later access. If not, then we do not want to consider the corresponding
  // internally contracted state.

  int norbs = Determinant::norbs;
  int first_virtual = 2*(schd.nciCore + schd.nciAct);

  // Class 5 (1 holes, 2 particles)
  numStates_1h2p = 0;
  for (int i = first_virtual+1; i < 2*norbs; i++) {
    for (int j = first_virtual; j < i; j++) {
      for (int a = 0; a < 2*schd.nciCore; a++) {
        // If this condition is not met, then it is not possible for this
        // internally contracted state to be accessed by a single
        // application of the Hamiltonian (due to spin conservation).
        if (i%2 == a%2 || j%2 == a%2) {
          std::array<int,3> inds = {i, j, a};
          class_1h2p_ind[inds] = numStates_1h2p;
          numStates_1h2p += 1;
        }
      }
    }
  }

  // Class 7 (2 holes, 1 particles)
  numStates_2h1p = 0;
  for (int i = first_virtual; i < 2*norbs; i++) {
    for (int a = 1; a < 2*schd.nciCore; a++) {
      for (int b = 0; b < a; b++) {
        // If this condition is not met, then it is not possible for this
        // internally contracted state to be accessed by a single
        // application of the Hamiltonian (due to spin conservation).
        if (i%2 == a%2 || i%2 == b%2) {
          std::array<int,3> inds = {i, a, b};
          class_2h1p_ind[inds] = numStates_2h1p;
          numStates_2h1p += 1;
        }
      }
    }
  }

  // Class 8 (2 holes, 2 particles)
  numStates_2h2p = 0;
  for (int i = first_virtual+1; i < 2*norbs; i++) {
    for (int j = first_virtual; j < i; j++) {
      for (int a = 1; a < 2*schd.nciCore; a++) {
        for (int b = 0; b < a; b++) {
          morework.setCounterToZero();
          generateAllScreenedExcitationsCAS_2h2p(schd.epsilon, morework, i, j, a, b);
          if (morework.nExcitations > 0) {
            std::array<int,4> inds = {i, j, a, b};
            class_2h2p_ind[inds] = numStates_2h2p;
            numStates_2h2p += 1;
          }
        }
      }
    }
  }

}

template<typename Wfn>
typename Wfn::ReferenceType& SCPT<Wfn>::getRef() { return wave.getRef(); }

template<typename Wfn>
typename Wfn::CorrType& SCPT<Wfn>::getCorr() { return wave.getCorr(); }

template<typename Wfn>
template<typename Walker>
void SCPT<Wfn>::initWalker(Walker& walk) {
  this->wave.initWalker(walk);
}

template<typename Wfn>
template<typename Walker>
void SCPT<Wfn>::initWalker(Walker& walk, Determinant& d) {
  this->wave.initWalker(walk, d);
}

//void initWalker(Walker& walk) {
//  this->wave.initWalker(walk);
//}

template<typename Wfn>
void SCPT<Wfn>::getVariables(Eigen::VectorXd& vars) {
  vars = coeffs;
}

template<typename Wfn>
void SCPT<Wfn>::printVariables() {
  cout << "ci coeffs\n" << coeffs << endl;
}

template<typename Wfn>
void SCPT<Wfn>::updateVariables(Eigen::VectorXd& vars) {
  coeffs = vars;
}

template<typename Wfn>
long SCPT<Wfn>::getNumVariables() {
  return coeffs.size();
}

template<typename Wfn>
template<typename Walker>
int SCPT<Wfn>::coeffsIndex(Walker& walk) {
  int norbs = Determinant::norbs;

  if (walk.excitation_class == 0) {
    // CAS det (0 holes, 0 particles)
    return 0;
  }
  else if (walk.excitation_class == 1) {
    // 0 holes, 1 particle
    return cumNumCoeffs[1] + *walk.excitedOrbs.begin() - 2*schd.nciCore - 2*schd.nciAct;
  }
  else if (walk.excitation_class == 2) {
    // 0 holes, 2 particles
    int a = *walk.excitedOrbs.begin() - 2*schd.nciCore - 2*schd.nciAct;
    int b = *(std::next(walk.excitedOrbs.begin())) - 2*schd.nciCore - 2*schd.nciAct;
    int A = max(a,b) - 1, B = min(a,b);

    return cumNumCoeffs[2] + A*(A+1)/2 + B;
  }
  else if (walk.excitation_class == 3) {
    // 1 hole, 0 particles
    return cumNumCoeffs[3] + *walk.excitedHoles.begin();
  }
  else if (walk.excitation_class == 4) {
    // 1 hole, 1 particles
    int i = *walk.excitedHoles.begin();
    int a = *walk.excitedOrbs.begin() - 2*schd.nciCore - 2*schd.nciAct;

    int numVirt = norbs - schd.nciCore - schd.nciAct;

    return cumNumCoeffs[4] + 2*numVirt*i + a;
  }
  else if (walk.excitation_class == 5) {
    // 1 hole, 2 particles
    int i = *walk.excitedHoles.begin();

    int a = *walk.excitedOrbs.begin();
    int b = *(std::next(walk.excitedOrbs.begin()));
    int A = max(a,b), B = min(a,b);

    std::array<int,3> inds = {A, B, i};

    auto it1 = class_1h2p_ind.find(inds);
    if (it1 != class_1h2p_ind.end())
      return cumNumCoeffs[5] + it1->second;
    else
      return -1;
  }
  else if (walk.excitation_class == 6) {
    // 2 hole, 0 particles
    int i = *walk.excitedHoles.begin();
    int j = *(std::next(walk.excitedHoles.begin()));
    int I = max(i,j) - 1, J = min(i,j);

    return cumNumCoeffs[6] + I*(I+1)/2 + J;
  }
  else if (walk.excitation_class == 7) {
    // 2 holes, 1 particles
    int i = *walk.excitedHoles.begin();
    int j = *(std::next(walk.excitedHoles.begin()));
    int I = max(i,j), J = min(i,j);

    int a = *walk.excitedOrbs.begin();

    std::array<int,3> inds = {a, I, J};

    auto it1 = class_2h1p_ind.find(inds);
    if (it1 != class_2h1p_ind.end())
      return cumNumCoeffs[7] + it1->second;
    else
      return -1;
  }
  else if (walk.excitation_class == 8) {
    // 2 holes, 2 particles
    int i = *walk.excitedHoles.begin();
    int j = *(std::next(walk.excitedHoles.begin()));
    int I = max(i,j), J = min(i,j);

    int a = *walk.excitedOrbs.begin();
    int b = *(std::next(walk.excitedOrbs.begin()));
    int A = max(a,b), B = min(a,b);

    std::array<int,4> inds = {A, B, I, J};

    auto it1 = class_2h2p_ind.find(inds);
    if (it1 != class_2h2p_ind.end())
      return cumNumCoeffs[8] + it1->second;
    else
      return -1;
  }
  else return -1;
}

// This perform the inverse of the coeffsIndex function: given the
// index of a perturber, return the external (non-active) orbtials
// involved. This only works for the main perturber types used -
// V, VV, C, CV, CC. These only have one or two external orbitals,
// which this function will return.
template<typename Wfn>
void SCPT<Wfn>::getOrbsFromIndex(const int index, int& i, int& j)
{
  int norbs = Determinant::norbs;
  int firstVirt = 2*(schd.nciCore + schd.nciAct);
  int numVirt = 2*(norbs - schd.nciCore - schd.nciAct);

  i = -1;
  j = -1;

  if (index >= cumNumCoeffs[1] && index < cumNumCoeffs[2]) {
    // AAAV perturber
    i = index - cumNumCoeffs[1] + firstVirt;

  }
  else if (index >= cumNumCoeffs[2] && index < cumNumCoeffs[3]) {
    // AAVV perturber
    int index2 = index - cumNumCoeffs[2];
    j = (int) floor(-0.5 + pow(0.25 + 2*index2, 0.5));
    i = index2 - j*(j+1)/2;

    i += firstVirt;
    j += firstVirt + 1;

  }
  else if (index >= cumNumCoeffs[3] && index < cumNumCoeffs[4]) {
    // CAAA perturber
    i = index - cumNumCoeffs[3];

  }
  else if (index >= cumNumCoeffs[4] && index < cumNumCoeffs[5]) {
    // CAAV perturber
    int index2 = index - cumNumCoeffs[4];
    j = index2 % numVirt;
    i = floor(index2 / numVirt);
    j += firstVirt;

  }
  else if (index >= cumNumCoeffs[6] && index < cumNumCoeffs[7]) {
    // CCAA perturber
    int index2 = index - cumNumCoeffs[6];
    j = (int) floor(-0.5 + pow(0.25 + 2*index2, 0.5));
    i = index2 - j*(j+1)/2;
    j += 1;

  }
}

// Take two orbital indices i and j, and convert them to a string.
// This is intended for use with two orbital obtained from the
// getOrbsFromIndex function, which are then to be output to
// pt2_energies files.
template<typename Wfn>
string SCPT<Wfn>::formatOrbString(const int i, const int j) {
  string str;

  if (i >= 0 && j == -1) {
    string tempStr = '(' + to_string(i) + ')';
    int tempLen = tempStr.length();
    str = string(12 - tempLen, ' ') + tempStr;
  }
  else if (i >= 0 && j >= 0 ) {
    string tempStr = '(' + to_string(i) + ',' + to_string(j) + ')';
    int tempLen = tempStr.length();
    str = string(12 - tempLen, ' ') + tempStr;
  }
  else {
    str = string(12, ' ');
  }

  return str;
}

template<typename Wfn>
template<typename Walker>
double SCPT<Wfn>::getOverlapFactor(int i, int a, const Walker& walk, bool doparity) const
{
  return 1.;
} // not used

template<typename Wfn>
template<typename Walker>
double SCPT<Wfn>::getOverlapFactor(int I, int J, int A, int B, const Walker& walk, bool doparity) const
{
  return 1.;
} // not used

template<typename Wfn>
template<typename Walker>
bool SCPT<Wfn>::checkWalkerExcitationClass(Walker &walk) {
  if (!classesUsed[walk.excitation_class]) return false;
  int coeffsIndex = this->coeffsIndex(walk);
  if (coeffsIndex == -1)
    return false;
  else
    return true;
}

// ham is a sample of the diagonal element of the Dyall ham
template<typename Wfn>
template<typename Walker>
void SCPT<Wfn>::HamAndOvlp(Walker &walk, double &ovlp, double &locEne, double &ham,
                           double &norm, int coeffsIndex, workingArray& work, bool fillExcitations)
{
  int norbs = Determinant::norbs;

  double ciCoeff = coeffs(coeffsIndex);
  morework.setCounterToZero();
  double ovlp0, ham0;

  if (coeffsIndex == 0) {
    wave.HamAndOvlp(walk, ovlp0, ham0, morework, true);
    ovlp = ciCoeff * ovlp0;
  }
  else {
    wave.HamAndOvlp(walk, ovlp0, ham0, morework, false);
    ovlp = ciCoeff * ovlp0;
  }

  ovlp_current = ovlp;

  if (ovlp == 0.) return; //maybe not necessary

  if (abs(ciCoeff) < 1.e-5) norm = 0;
  else norm = 1 / ciCoeff / ciCoeff;

  locEne = walk.d.Energy(I1, I2, coreE);

  // Get the diagonal Dyall Hamiltonian element from this determinant
  Determinant dAct = walk.d;
  double ene_h = 0.0, ene_hh = 0.0, ene_p = 0.0, ene_pp = 0.0;

  if (walk.excitedOrbs.size() > 0) {
    dAct.setocc(*walk.excitedOrbs.begin(), false);
    ene_p = moEne((*walk.excitedOrbs.begin())/2);
  }
  if (walk.excitedOrbs.size() == 2) {
    dAct.setocc(*(std::next(walk.excitedOrbs.begin())), false);
    ene_pp = moEne((*(std::next(walk.excitedOrbs.begin())))/2);
  }
  if (walk.excitedHoles.size() > 0) {
    dAct.setocc(*walk.excitedHoles.begin(), true);
    ene_h = moEne((*walk.excitedHoles.begin())/2);
  }
  if (walk.excitedHoles.size() == 2) {
    dAct.setocc(*(std::next(walk.excitedHoles.begin())), true);
    ene_hh = moEne((*(std::next(walk.excitedHoles.begin())))/2);
  }
  ham = (dAct.Energy(I1, I2, coreE) + ene_p + ene_pp - ene_h - ene_hh) / ciCoeff / ciCoeff;

  // Generate all excitations (after screening)
  work.setCounterToZero();
  generateAllScreenedSingleExcitationsDyallOld(walk.d, dAct, schd.epsilon, schd.screen, work, false);
  generateAllScreenedDoubleExcitationsDyallOld(walk.d, schd.epsilon, schd.screen, work, false);

  //loop over all the screened excitations
  //cout << endl << "m dets\n" << endl << endl;
  for (int i=0; i<work.nExcitations; i++) {
    double tia = work.HijElement[i];
    double tiaD = work.HijElement[i];

    int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
    int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
    int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;

    double isDyall = 0.;
    if (work.ovlpRatio[i] != 0.) {
      isDyall = 1.;
      if (ex2 == 0) tiaD = work.ovlpRatio[i];
      work.ovlpRatio[i] = 0.;
    }
    
    auto walkCopy = walk;
    double parity = 1.;
    Determinant dcopy = walkCopy.d;
    walkCopy.updateWalker(wave.getRef(), wave.getCorr(), work.excitation1[i], work.excitation2[i], false);

    // Is this excitation class being used? If not, then move to the next excitation.
    if (!classesUsed[walkCopy.excitation_class]) continue;
    int coeffsCopyIndex = this->coeffsIndex(walkCopy);
    if (coeffsCopyIndex == -1) continue;

    //if (walkCopy.excitedOrbs.size() > 2) continue;
    parity *= dcopy.parity(A/2, I/2, I%2);
    if (ex2 != 0) {
      dcopy.setocc(I, false);
      dcopy.setocc(A, true);
      parity *= dcopy.parity(B/2, J/2, J%2);
    }
    
    morework.setCounterToZero();

    if (coeffsCopyIndex == 0) {
      wave.HamAndOvlp(walkCopy, ovlp0, ham0, morework, true);
      ham += isDyall * parity * tiaD * ovlp0 / ciCoeff / ovlp;
      locEne += parity * tia * ovlp0 * coeffs(coeffsCopyIndex) / ovlp;
      work.ovlpRatio[i] = ovlp0 * coeffs(coeffsCopyIndex) / ovlp;
    }
    else {
      wave.HamAndOvlp(walkCopy, ovlp0, ham0, morework, false);
      ham += isDyall * parity * tiaD * ham0 / ciCoeff / ovlp;
      locEne += parity * tia * ham0 * coeffs(coeffsCopyIndex) / ovlp;
      work.ovlpRatio[i] = ham0 * coeffs(coeffsCopyIndex) / ovlp;
    }
  }

}

// ham is a sample of the diagonal element of the Dyall Hamiltonian
template<typename Wfn>
template<typename Walker>
void SCPT<Wfn>::FastHamAndOvlp(Walker &walk, double &ovlp, double &ham, workingArray& work, bool fillExcitations)
{
  double ovlp0, ham0;
  double tiaD, parity;
  int norbs = Determinant::norbs;

  work.setCounterToZero();
  morework.setCounterToZero();

  //if (coeffsIndex == 0)
  //  cout << "ERROR: FastHamAndOvlp should not be used within the CASCI space." << endl;
  //else
  wave.HamAndOvlp(walk, ovlp, ham0, morework, false);
  ovlp_current = ovlp;

  if (ovlp == 0.) return;

  // Get the diagonal Dyall Hamiltonian element from this determinant
  Determinant dAct = walk.d;
  double ene_h = 0.0, ene_hh = 0.0, ene_p = 0.0, ene_pp = 0.0;

  if (walk.excitedOrbs.size() > 0) {
    dAct.setocc(*walk.excitedOrbs.begin(), false);
    ene_p = moEne((*walk.excitedOrbs.begin())/2);
  }
  if (walk.excitedOrbs.size() == 2) {
    dAct.setocc(*(std::next(walk.excitedOrbs.begin())), false);
    ene_pp = moEne((*(std::next(walk.excitedOrbs.begin())))/2);
  }
  if (walk.excitedHoles.size() > 0) {
    dAct.setocc(*walk.excitedHoles.begin(), true);
    ene_h = moEne((*walk.excitedHoles.begin())/2);
  }
  if (walk.excitedHoles.size() == 2) {
    dAct.setocc(*(std::next(walk.excitedHoles.begin())), true);
    ene_hh = moEne((*(std::next(walk.excitedHoles.begin())))/2);
  }
  ham = (dAct.Energy(I1, I2, coreE) + ene_p + ene_pp - ene_h - ene_hh);

  // Generate all excitations (after screening)
  work.setCounterToZero();
  generateAllScreenedSingleExcitationsDyall(walk.d, dAct, schd.epsilon, schd.screen, work, false);
  generateAllScreenedDoubleExcitationsDyall(walk.d, schd.epsilon, schd.screen, work, false);

  // loop over all the screened excitations
  for (int i=0; i<work.nExcitations; i++) {
    double tiaD = work.HijElement[i];

    int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
    int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
    int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;

    // If this is true, then this is a valid excitation for the Dyall
    // Hamiltonian (i.e., using the active space Hamiltonian only).
    if (work.ovlpRatio[i] != 0.) {
      if (ex2 == 0) tiaD = work.ovlpRatio[i];
      work.ovlpRatio[i] = 0.;
    } else {
      continue;
    }

    auto walkCopy = walk;
    Determinant dcopy = walkCopy.d;
    walkCopy.updateWalker(wave.getRef(), wave.getCorr(), work.excitation1[i], work.excitation2[i], false);

    parity = 1.;
    parity *= dcopy.parity(A/2, I/2, I%2);
    if (ex2 != 0) {
      dcopy.setocc(I, false);
      dcopy.setocc(A, true);
      parity *= dcopy.parity(B/2, J/2, J%2);
    }

    morework.setCounterToZero();
    wave.HamAndOvlp(walkCopy, ovlp0, ham0, morework, false);
    ham += parity * tiaD * ham0 / ovlp;
    work.ovlpRatio[i] = ham0 / ovlp;
  }
}

template<typename Wfn>
template<typename Walker>
void SCPT<Wfn>::HamAndSCNorms(Walker &walk, double &ovlp, double &ham, Eigen::VectorXd &normSamples,
                   vector<Determinant>& initDets, vector<double>& largestCoeffs,
                   workingArray& work, bool calcExtraNorms)
{

  // which excitation classes should we consider?
  bool useAAVV;
  bool useCAAV;
  bool useCAVV;
  bool useCCAA;
  bool useCCAV;
  bool useCCVV;

  if (calcExtraNorms) {
    useAAVV = classesUsed[2];
    useCAAV = classesUsed[4];
    useCAVV = classesUsed[5];
    useCCAA = classesUsed[6];
    useCCAV = classesUsed[7];
    useCCVV = classesUsed[8];
  } else {
    // only consider these classes if we're not calculating the norms separately
    useAAVV = classesUsed[2] && !normsDeterm[2];
    useCAAV = classesUsed[4] && !normsDeterm[4];
    useCAVV = classesUsed[5] && !normsDeterm[5];
    useCCAA = classesUsed[6] && !normsDeterm[6];
    useCCAV = classesUsed[7] && !normsDeterm[7];
    useCCVV = classesUsed[8] && !normsDeterm[8];
  }

  int norbs = Determinant::norbs;
  double ham0;

  morework.setCounterToZero();

  // Get the WF overlap with the walker, ovlp
  wave.HamAndOvlp(walk, ovlp, ham0, morework, true);
  ovlp_current = ovlp;

  if (ovlp == 0.) return;
  ham = walk.d.Energy(I1, I2, coreE);

  // Generate all screened excitations
  work.setCounterToZero();

  size_t nExcitationsCASCI = 0;

  int nSpinCore = 2*schd.nciCore;
  int firstSpinVirt = 2*(schd.nciCore + schd.nciAct);

  vector<int> closed;
  vector<int> open;
  walk.d.getOpenClosed(open, closed);


  // single excitations
  for (int i = 0; i < closed.size(); i++) {
    bool iCore = closed[i] < nSpinCore;
    for (int a = 0; a < open.size(); a++) {
      if (closed[i] % 2 == open[a] % 2 &&
          abs(I2hb.Singles(closed[i], open[a])) > schd.epsilon)
      {
        bool aVirt = open[a] >= firstSpinVirt;

        bool caavExcit = iCore && aVirt;
        if (!useCAAV && caavExcit) continue;

        int ex1 = closed[i] * 2 * norbs + open[a];
        int ex2 = 0.0;
        double tia = walk.d.Hij_1ExciteScreened(open[a], closed[i], I2hb,
                                                      schd.screen, false);
        AddSCNormsContrib(walk, ovlp, ham, normSamples, initDets, largestCoeffs,
                          work, calcExtraNorms, ex1, ex2, tia, nExcitationsCASCI);
      }
    }
  }

  // double excitations
  int nclosed = closed.size();
  for (int i = 0; i<nclosed; i++) {
    bool iCore = closed[i] < nSpinCore;
    for (int j = 0; j<i; j++) {
      bool jCore = closed[j] < nSpinCore;

      const float *integrals; const short* orbIndices;
      size_t numIntegrals;
      I2hb.getIntegralArray(closed[i], closed[j], integrals, orbIndices, numIntegrals);
      size_t numLargeIntegrals = std::lower_bound(integrals, integrals + numIntegrals, schd.epsilon, [](const float &x, float val){ return fabs(x) > val; }) - integrals;

      // for all HCI integrals
      for (size_t index = 0; index < numLargeIntegrals; index++)
      {
        // otherwise: generate the determinant corresponding to the current excitation
        int a = 2 * orbIndices[2 * index] + closed[i] % 2,
            b = 2 * orbIndices[2 * index + 1] + closed[j] % 2;

        if (walk.d.getocc(a) || walk.d.getocc(b)) continue;

        bool aVirt = a >= firstSpinVirt;
        bool bVirt = b >= firstSpinVirt;

        bool ccvvExcit = iCore && aVirt && bVirt;
        if (!useCCVV && ccvvExcit) continue;
        bool cavvExcit = jCore && (!iCore) && aVirt && bVirt;
        if (!useCAVV && cavvExcit) continue;
        bool ccavExcit = iCore && ((aVirt && (!bVirt)) || ((!aVirt) && bVirt));
        if (!useCCAV && ccavExcit) continue;
        bool aavvExcit = (!jCore) && aVirt && bVirt;
        if (!useAAVV && aavvExcit) continue;
        bool caavExcit = jCore && (!iCore) && ( (aVirt && (!bVirt)) || ((bVirt && (!aVirt))) );
        if (!useCAAV && caavExcit) continue;
        bool ccaaExcit = iCore && (!aVirt) && (!bVirt);
        if (!useCCAA && ccaaExcit) continue;

        int ex1 = closed[i] * 2 * norbs + a;
        int ex2 = closed[j] * 2 * norbs + b;
        double tia = integrals[index];

        AddSCNormsContrib(walk, ovlp, ham, normSamples, initDets, largestCoeffs,
                          work, calcExtraNorms, ex1, ex2, tia, nExcitationsCASCI);
      }
    }
  }

  // For the CTMC algorithm, only need excitations within the CASCI space.
  // Update the number of excitations to reflect this
  work.nExcitations = nExcitationsCASCI;
}

template<typename Wfn>
template<typename Walker>
void SCPT<Wfn>::AddSCNormsContrib(Walker &walk, double &ovlp, double &ham, Eigen::VectorXd &normSamples,
                       vector<Determinant>& initDets, vector<double>& largestCoeffs,
                       workingArray& work, bool calcExtraNorms, int& ex1, int& ex2,
                       double& tia, size_t& nExcitationsCASCI)
{
  // This is called for each excitations from a determinant in the CASCI
  // space (walk.d)
  // If the excitations is within the CASCI space, store the overlap factor
  // in the work array, for the CTMC algorithm to make a move.
  // Otherwise, add contributions to the estimates of the norms.
  // Also, if the given excited determinant has the largest coefficient found,
  // then store the determinant and the coefficient in initDets and largestCoeffs.

  int norbs = Determinant::norbs;
  double ovlp0, ham0;

  int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
  int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;

  auto walkCopy = walk;
  double parity = 1.0;
  Determinant dcopy = walkCopy.d;
  walkCopy.updateWalker(wave.getRef(), wave.getCorr(), ex1, ex2, false);
  parity *= dcopy.parity(A/2, I/2, I%2);
  if (ex2 != 0) {
    dcopy.setocc(I, false);
    dcopy.setocc(A, true);
    parity *= dcopy.parity(B/2, J/2, J%2);
  }
  //work.ovlpRatio[i] = 0.0;
  if (walkCopy.excitation_class == 0) {
    // For the CTMC algorithm (which is performed within the CASCI space, when
    // calculating the SC norms), we need the ratio of overlaps for connected
    // determinants within the CASCI space. Store these in the work array, and
    // override other excitations, which we won't need any more.
    double ovlpRatio = wave.Overlap(walkCopy.d) / ovlp;
    work.excitation1[nExcitationsCASCI] = ex1;
    work.excitation2[nExcitationsCASCI] = ex2;
    work.ovlpRatio[nExcitationsCASCI] = ovlpRatio;
    ham += parity * tia * ovlpRatio;
    nExcitationsCASCI += 1;
    return;
  } else if (!classesUsed[walkCopy.excitation_class]) {
    // Is this excitation class being used? If not, then move to the next excitation
    return;
  } else if (normsDeterm[walkCopy.excitation_class]) {
    // Is the norm for this class being calculated exactly? If so, move to the next excitation
    // The exception is if we want to record the maximum coefficient size (calcExtraNorms == true)
    if (!calcExtraNorms) return;
  }
  int ind = this->coeffsIndex(walkCopy);
  if (ind == -1) return;

  morework.setCounterToZero();
  wave.HamAndOvlp(walkCopy, ovlp0, ham0, morework, false);
  normSamples(ind) += parity * tia * ham0 / ovlp;

  // If this is the determinant with the largest coefficient found within
  // the SC space so far, then store it.
  if (abs(ham0) > largestCoeffs[ind]) {
    initDets[ind] = walkCopy.d;
    largestCoeffs[ind] = abs(ham0);
  }
}

// this is a version of HamAndSCNorms, optimized for the case where only
// classes AAAV (class 1) and CAAA (class 3) are needed
template<typename Wfn>
template<typename Walker>
void SCPT<Wfn>::HamAndSCNormsCAAA_AAAV(Walker &walk, double &ovlp, double &ham, Eigen::VectorXd &normSamples,
                   vector<Determinant>& initDets, vector<double>& largestCoeffs,
                   workingArray& work, bool calcExtraNorms)
{
  int norbs = Determinant::norbs;
  double ham0;

  morework.setCounterToZero();

  // Get the WF overlap with the walker, ovlp
  wave.HamAndOvlp(walk, ovlp, ham0, morework, true);
  ovlp_current = ovlp;

  if (ovlp == 0.) return;
  ham = walk.d.Energy(I1, I2, coreE);

  // Generate all screened excitations
  work.setCounterToZero();

  size_t nExcitationsCASCI = 0;

  int nSpinCore = 2*schd.nciCore;
  int firstSpinVirt = 2*(schd.nciCore + schd.nciAct);

  vector<int> closed;
  vector<int> open;
  walk.d.getOpenClosed(open, closed);

  auto ub_1 = upper_bound(open.begin(), open.end(), firstSpinVirt - 1);
  int indActOpen = std::distance(open.begin(), ub_1);

  auto ub_2 = upper_bound(closed.begin(), closed.end(), nSpinCore - 1);
  int indCoreClosed = std::distance(closed.begin(), ub_2);

  // single excitations within the CASCI space
  // loop over all occupied orbitals in the active space
  for (int i = indCoreClosed; i < closed.size(); i++) {
    int closedOrb = closed[i];
    int closedOffset = closedOrb * 2 * norbs;

    // loop over all unoccupied orbitals in the active space
    for (int a = 0; a < indActOpen; a++) {
      if (closed[i] % 2 == open[a] % 2 &&
          abs(I2hb.Singles(closed[i], open[a])) > schd.epsilon)
      {
        int ex1 = closedOffset + open[a];
        int ex2 = 0;

        double tia = walk.d.Hij_1ExciteScreened(open[a], closedOrb, I2hb,
                                                      schd.screen, false);
        AddSCNormsContrib(walk, ovlp, ham, normSamples, initDets, largestCoeffs,
                          work, calcExtraNorms, ex1, ex2, tia, nExcitationsCASCI);
      }
    }
  }

  // double excitations within the CASCI space
  // loop over all closed orbitals in the active space
  for (int i = indCoreClosed; i < closed.size(); i++) {
    int closedOrb_i = closed[i];
    int closedOffset_i = closedOrb_i * 2 * norbs;

    // loop over all closed orbitals in the active space s.t. j<i
    for (int j = indCoreClosed; j<i; j++) {
      int closedOrb_j = closed[j];
      int closedOffset_j = closedOrb_j * 2 * norbs;

      const float *integrals; const short* orbIndices;
      size_t numIntegrals;
      I2hbCAS.getIntegralArrayCAS(closedOrb_i, closed[j], integrals, orbIndices, numIntegrals);
      size_t numLargeIntegrals = std::lower_bound(integrals, integrals + numIntegrals, schd.epsilon, [](const float &x, float val){ return fabs(x) > val; }) - integrals;

      // for all HCI integrals
      for (size_t index = 0; index < numLargeIntegrals; index++)
      {
        // otherwise: generate the determinant corresponding to the current excitation
        int a = 2 * orbIndices[2 * index] + closedOrb_i % 2,
            b = 2 * orbIndices[2 * index + 1] + closedOrb_j % 2;

        if (walk.d.getocc(a) || walk.d.getocc(b)) continue;

        int ex1 = closedOffset_i + a;
        int ex2 = closedOffset_j + b;
        double tia = integrals[index];

        AddSCNormsContrib(walk, ovlp, ham, normSamples, initDets, largestCoeffs,
                          work, calcExtraNorms, ex1, ex2, tia, nExcitationsCASCI);
      }
    }
  }

  // single excitations for CAAA class
  // loop over all core orbitals (which will all be occupied)
  for (int i = 0; i < indCoreClosed; i++) {
    int closedOrb = closed[i];
    int closedOffset = closedOrb * 2 * norbs;

    // loop over all open orbitals in the active space
    for (int a = 0; a < indActOpen; a++) {
      if (closed[i] % 2 == open[a] % 2 &&
          abs(I2hb.Singles(closed[i], open[a])) > schd.epsilon)
      {
        int ex1 = closedOffset + open[a];
        int ex2 = 0;
        double tia = walk.d.Hij_1ExciteScreened(open[a], closedOrb, I2hb,
                                                      schd.screen, false);
        AddSCNormsContrib(walk, ovlp, ham, normSamples, initDets, largestCoeffs,
                          work, calcExtraNorms, ex1, ex2, tia, nExcitationsCASCI);
      }
    }
  }

  // double excitations for CAAA class
  // loop over all closed orbitals in the active space
  for (int i = indCoreClosed; i < closed.size(); i++) {
    int closedOrb_i = closed[i];
    int closedOffset_i = closedOrb_i * 2 * norbs;

    // loop over all core orbitals (which will all be occupied)
    for (int j = 0; j < indCoreClosed; j++) {

      int closedOrb_j = closed[j];
      int closedOffset_j = closedOrb_j * 2 * norbs;

      const float *integrals; const short* orbIndices;
      size_t numIntegrals;
      I2hbCAS.getIntegralArrayCAS(closedOrb_i, closed[j], integrals, orbIndices, numIntegrals);
      size_t numLargeIntegrals = std::lower_bound(integrals, integrals + numIntegrals, schd.epsilon, [](const float &x, float val){ return fabs(x) > val; }) - integrals;

      // for all HCI integrals
      for (size_t index = 0; index < numLargeIntegrals; index++)
      {
        int a = 2 * orbIndices[2 * index] + closedOrb_i % 2,
            b = 2 * orbIndices[2 * index + 1] + closedOrb_j % 2;

        if (walk.d.getocc(a) || walk.d.getocc(b)) continue;

        int ex1 = closedOffset_i + a;
        int ex2 = closedOffset_j + b;
        double tia = integrals[index];

        AddSCNormsContrib(walk, ovlp, ham, normSamples, initDets, largestCoeffs,
                          work, calcExtraNorms, ex1, ex2, tia, nExcitationsCASCI);
      }
    }
  }

  // single excitations for AAAV class
  // loop over all occupied orbitals in the active space
  for (int i = indCoreClosed; i < closed.size(); i++) {
    int closedOrb = closed[i];
    int closedOffset = closedOrb * 2 * norbs;

    // loop over all virtual orbitals (which will all be unoccupied)
    for (int a = indActOpen; a < open.size(); a++) {
      if (closed[i] % 2 == open[a] % 2 &&
          abs(I2hb.Singles(closed[i], open[a])) > schd.epsilon)
      {
        int ex1 = closedOffset + open[a];
        int ex2 = 0;

        double tia = walk.d.Hij_1ExciteScreened(open[a], closedOrb, I2hb,
                                                      schd.screen, false);
        AddSCNormsContrib(walk, ovlp, ham, normSamples, initDets, largestCoeffs,
                          work, calcExtraNorms, ex1, ex2, tia, nExcitationsCASCI);
      }
    }
  }

  // double excitations for AAAV class
  // loop over all virtual orbitals
  for (int a = firstSpinVirt; a < 2 * norbs; a++) {

    // loop over all open orbitals in the active space
    for (int b = 0; b < indActOpen; b++) {

      int openOrb_b = open[b];

      const float *integrals; const short* orbIndices;
      size_t numIntegrals;
      I2hbCAS.getIntegralArrayCAS(a, open[b], integrals, orbIndices, numIntegrals);
      size_t numLargeIntegrals = std::lower_bound(integrals, integrals + numIntegrals, schd.epsilon, [](const float &x, float val){ return fabs(x) > val; }) - integrals;

      // for all HCI integrals
      for (size_t index = 0; index < numLargeIntegrals; index++)
      {
        int i = 2 * orbIndices[2 * index] + a % 2,
            j = 2 * orbIndices[2 * index + 1] + openOrb_b % 2;

        if ( (!walk.d.getocc(i)) || (!walk.d.getocc(j)) ) continue;

        int ex1 = i * 2 * norbs + a;
        int ex2 = j * 2 * norbs + openOrb_b;
        double tia = integrals[index];

        AddSCNormsContrib(walk, ovlp, ham, normSamples, initDets, largestCoeffs,
                          work, calcExtraNorms, ex1, ex2, tia, nExcitationsCASCI);
      }
    }
  }

  // For the CTMC algorithm, only need excitations within the CASCI space.
  // Update the number of excitations to reflect this
  work.nExcitations = nExcitationsCASCI;
}


template<typename Wfn>
template<typename Walker>
double SCPT<Wfn>::doNEVPT2_CT(Walker& walk) {

  int norbs = Determinant::norbs;
  
  // add noise to avoid zero coeffs
  if (commrank == 0) {
    //cout << "starting sampling at " << setprecision(4) << getTime() - startofCalc << endl; 
    auto random = std::bind(std::uniform_real_distribution<double>(0., 1.e-6), std::ref(generator));
    for (int i=0; i < coeffs.size(); i++) {
      if (coeffs(i) == 0) coeffs(i) = random();
    }
  }

#ifndef SERIAL
  MPI_Bcast(coeffs.data(), coeffs.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

  // sampling
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));

  double ovlp = 0., normSample = 0., hamSample = 0., locEne = 0., waveEne = 0.;
  VectorXd ham = VectorXd::Zero(coeffs.size()), norm = VectorXd::Zero(coeffs.size());
  workingArray work;

  int coeffsIndex = this->coeffsIndex(walk);
  HamAndOvlp(walk, ovlp, locEne, hamSample, normSample, coeffsIndex, work);

  int iter = 0;
  double cumdeltaT = 0.;
  int printMod = max(1, schd.stochasticIter / 10);

  while (iter < schd.stochasticIter) {
    double cumovlpRatio = 0;
    for (int i = 0; i < work.nExcitations; i++) {
      cumovlpRatio += abs(work.ovlpRatio[i]);
      work.ovlpRatio[i] = cumovlpRatio;
    }
    double deltaT = 1.0 / (cumovlpRatio);
    double nextDetRandom = random() * cumovlpRatio;
    int nextDet = std::lower_bound(work.ovlpRatio.begin(), (work.ovlpRatio.begin() + work.nExcitations),
                                   nextDetRandom) - work.ovlpRatio.begin();
    cumdeltaT += deltaT;
    double ratio = deltaT / cumdeltaT;
    norm *= (1 - ratio);
    norm(coeffsIndex) += ratio * normSample;
    ham *= (1 - ratio);
    ham(coeffsIndex) += ratio * hamSample;
    waveEne *= (1 - ratio);
    waveEne += ratio * locEne;

    walk.updateWalker(wave.getRef(), wave.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);

    // Make sure that the walker is within one of the classes being sampled, after this move
    if (!classesUsed[walk.excitation_class]) continue;
    coeffsIndex = this->coeffsIndex(walk);
    if (coeffsIndex == -1) continue;

    HamAndOvlp(walk, ovlp, locEne, hamSample, normSample, coeffsIndex, work);

    iter++;
    if (commrank == 0 && iter % printMod == 1) cout << "iter  " << iter << "  t  " << setprecision(4) << getTime() - startofCalc << endl; 
  }

  norm *= cumdeltaT;
  ham *= cumdeltaT;
  waveEne *= cumdeltaT;

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, norm.data(), coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, ham.data(), coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(cumdeltaT), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(waveEne), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  norm /= cumdeltaT;
  ham /= cumdeltaT;
  waveEne /= cumdeltaT;
  
  std::vector<int> largeNormIndices;
  int counter = 0;
  for (int i = 0; i < coeffs.size(); i++) {
    if (norm(i) > schd.overlapCutoff) {
      largeNormIndices.push_back(i);
    }
  }
  Map<VectorXi> largeNormSlice(&largeNormIndices[0], largeNormIndices.size());
  VectorXd largeNorms;
  igl::slice(norm, largeNormSlice, largeNorms);
  VectorXd largeHam;
  igl::slice(ham, largeNormSlice, largeHam);
  VectorXd ene = (largeHam.array() / largeNorms.array()).matrix();
  double ene2 = 0.;
  coeffs.setZero();
  coeffs(0) = 1.;
  for (int i = 1; i < largeNorms.size(); i++) {
    ene2 += largeNorms(i) / largeNorms(0) / (ene(0) - ene(i));
    coeffs(largeNormIndices[i]) = 1 / (ene(0) - ene(i));
  }

  if (commrank == 0) {
    cout << "ref energy:  " << setprecision(12) << ene(0) << endl;
    cout << "stochastic nevpt2 energy:  " << ene(0) + ene2 << endl;
    cout << "stochastic waveEne:  " << waveEne << endl;

    // If any classes are to be obtained deterministically, then do this now
    if (any_of(classesUsedDeterm.begin(), classesUsedDeterm.end(), [](bool i){return i;}) ) {
      double energy_ccvv = 0.0;
      if (classesUsedDeterm[8]) {
        energy_ccvv = get_ccvv_energy();
        cout << "deterministic CCVV energy:  " << setprecision(12) << energy_ccvv << endl;
      }
      cout << "total nevpt2 energy:  " << ene(0) + ene2 + energy_ccvv << endl;
    }

    if (schd.printVars) cout << endl << "ci coeffs\n" << coeffs << endl; 
  }
}

// Output the header for the "norms" file, which will output the norms of
// the strongly contracted (SC) states, divided by the norm of the CASCI
// state (squared)
template<typename Wfn>
double SCPT<Wfn>::outputNormFileHeader(FILE* out_norms)
{
  fprintf(out_norms, "# 1. iteration");
  fprintf(out_norms, "         # 2. residence_time");
  fprintf(out_norms, "            # 3. casci_energy");
  for (int ind = 1; ind < numCoeffs; ind++) {
    int label = ind + 3;

    string header;
    header.append(to_string(label));
    header.append(". weighted_norm_");
    header.append(to_string(ind));
    stringstream fixed_width;
    // fix the width of each column to 28 characters
    fixed_width << setw(28) << header;
    string fixed_width_str = fixed_width.str();
    fprintf(out_norms, fixed_width_str.c_str());
  }
  fprintf(out_norms, "\n");
}

// Create directories where the norm files will be stored
template<typename Wfn>
void SCPT<Wfn>::createDirForSCNorms()
{
  // create ./norms
  boost::filesystem::path pathNorms( boost::filesystem::current_path() / "norms" );

  if (commrank == 0) {
    if (!boost::filesystem::exists(pathNorms))
      boost::filesystem::create_directory(pathNorms);

    // create ./norms/exact
    boost::filesystem::path pathNormsExact( boost::filesystem::current_path() / "norms/exact" );

    if (!boost::filesystem::exists(pathNormsExact))
      boost::filesystem::create_directory(pathNormsExact);
  }

  // wait for process 0 to create directory
  MPI_Barrier(MPI_COMM_WORLD);

  // create ./norms/proc_i
  boost::filesystem::path pathNormsProc( boost::filesystem::current_path() / "norms/proc_" );
  pathNormsProc += to_string(commrank);

  if (!boost::filesystem::exists(pathNormsProc))
    boost::filesystem::create_directory(pathNormsProc);

  return;
}

// Create directories where the init_dets files will be stored
template<typename Wfn>
void SCPT<Wfn>::createDirForInitDets()
{
  // create ./init_dets
  boost::filesystem::path pathInitDets( boost::filesystem::current_path() / "init_dets" );

  if (commrank == 0) {
    if (!boost::filesystem::exists(pathInitDets))
      boost::filesystem::create_directory(pathInitDets);
  }

  // wait for process 0 to create directory
  MPI_Barrier(MPI_COMM_WORLD);

  // create ./init_dets/proc_i
  boost::filesystem::path pathInitDetsProc( boost::filesystem::current_path() / "init_dets/proc_" );
  pathInitDetsProc += to_string(commrank);

  if (!boost::filesystem::exists(pathInitDetsProc))
    boost::filesystem::create_directory(pathInitDetsProc);

  return;
}

// Create directories where the norm files will be stored
template<typename Wfn>
void SCPT<Wfn>::createDirForNormsBinary()
{
  if (commrank == 0) {
    // create ./norms
    boost::filesystem::path pathNorms( boost::filesystem::current_path() / "norm_data" );

    if (!boost::filesystem::exists(pathNorms))
      boost::filesystem::create_directory(pathNorms);
  }

  // wait for process 0 to create directory
  MPI_Barrier(MPI_COMM_WORLD);

  // create ./norms/proc_i
  boost::filesystem::path pathNormsProc( boost::filesystem::current_path() / "norm_data/proc_" );
  pathNormsProc += to_string(commrank);

  if (!boost::filesystem::exists(pathNormsProc))
    boost::filesystem::create_directory(pathNormsProc);

  return;
}

// Print norms to output files.
// If determClasses is true, only print the norms from classes where the
// norms are being found exactly.
// Otherwise, print the norms calculated stochastically, summed up until
// the current iteration. Also print the current estimate of the the
// CASCI energy, and the residence time.
template<typename Wfn>
void SCPT<Wfn>::printSCNorms(int& iter, double& deltaT_Tot, double& energyCAS_Tot, Eigen::VectorXd& norms_Tot, bool determClasses)
{
  if (! determClasses) {
    // First print the CASCI energy estimate
    string energyFileName = "norms/proc_" + to_string(commrank) + "/cas_energy_" + to_string(commrank) + ".dat";
    ofstream out_energy;
    out_energy.open(energyFileName);

    out_energy << "Iteration: " << iter << endl;
    out_energy.precision(12);
    out_energy << scientific;
    out_energy << "Total summed numerator for the CASCI energy: " << energyCAS_Tot << endl;
    out_energy << "Total summed residence time: " << deltaT_Tot << endl;

    out_energy.close();
  }

  // Next print the norms of each of the classes
  for (int i=1; i<9; i++)
  {
    if (classesUsed[i]) {

      if ((determClasses && normsDeterm[i]) || (!determClasses && !normsDeterm[i])) {
        string fileName;
        if (!determClasses)
          fileName = "norms/proc_" + to_string(commrank) + "/norms_" + classNames[i] + "_" + to_string(commrank) + ".dat";
        else
          fileName = "norms/exact/norms_" + classNames[i] + "_exact.dat";

        ofstream out_norms;
        out_norms.open(fileName);

        if (!determClasses)
          out_norms << "Iteration: " << iter << endl;
        else
          out_norms << "Norms calculated deterministically" << endl;

        out_norms.precision(12);
        out_norms << scientific;

        if (!determClasses)
          out_norms << "Total summed residence time: " << deltaT_Tot << endl;

        for (int ind = cumNumCoeffs[i]; ind < cumNumCoeffs[i]+numCoeffsPerClass[i]; ind++)
          out_norms << norms_Tot(ind) << endl;

        out_norms.close();
      }

    }
  }
}

template<typename Wfn>
void SCPT<Wfn>::readStochNorms(double& deltaT_Tot, double& energyCAS_Tot, Eigen::VectorXd& norms_Tot)
{
  std::string line;
  vector<string> tok;

  // Read CASCI energy
  string energyFileName = "norms/proc_" + to_string(commrank) + "/cas_energy_" + to_string(commrank) + ".dat";

  ifstream dump;
  dump.open(energyFileName);

  // First line contains the iteration number
  std::getline(dump, line);

  // Second line contains the energy as the final element
  std::getline(dump, line);
  boost::trim_if(line, boost::is_any_of(", \t\n"));
  boost::split(tok, line, boost::is_any_of(", \t\n"), boost::token_compress_on);
  energyCAS_Tot = stod(tok.back());

  // Third line contains the residence time as the final element
  std::getline(dump, line);
  boost::trim_if(line, boost::is_any_of(", \t\n"));
  boost::split(tok, line, boost::is_any_of(", \t\n"), boost::token_compress_on);
  deltaT_Tot = stod(tok.back());

  dump.close();

  // Read in the stochastically-sampled norms
  for (int i=1; i<9; i++)
  {
    if (classesUsed[i]) {
      string fileName = "norms/proc_" + to_string(commrank) + "/norms_" +
                        classNames[i] + "_" + to_string(commrank) + ".dat";

      ifstream dump;
      dump.open(fileName);

      // First two lines are not needed
      std::getline(dump, line);
      std::getline(dump, line);

      int ind = cumNumCoeffs[i];

      // Data starts on line 3
      std::getline(dump, line);
      while (dump.good())
      {
        boost::trim_if(line, boost::is_any_of(", \t\n"));
        boost::split(tok, line, boost::is_any_of(", \t\n"), boost::token_compress_on);
        norms_Tot[ind] = stod(tok[0]);

        std::getline(dump, line);
        ind++;
      }

    }
  }

}

template<typename Wfn>
void SCPT<Wfn>::readDetermNorms(Eigen::VectorXd& norms_Tot)
{
  std::string line;
  vector<string> tok;

  // Read in the exactly-calculated norms
  for (int i=1; i<9; i++)
  {
    if (classesUsed[i] && normsDeterm[i]) {
      string fileName = "norms/exact/norms_" + classNames[i] + "_exact.dat";

      ifstream dump;
      dump.open(fileName);

      // First line is not needed
      std::getline(dump, line);

      int ind = cumNumCoeffs[i];

      // Data starts on line 3
      std::getline(dump, line);
      while (dump.good())
      {
        boost::trim_if(line, boost::is_any_of(", \t\n"));
        boost::split(tok, line, boost::is_any_of(", \t\n"), boost::token_compress_on);
        norms_Tot[ind] = stod(tok[0]);

        std::getline(dump, line);
        ind++;
      }

    }
  }

}

// Print initial determinants to output files
// We only need to print out the occupations in the active spaces
// The occupations of the core and virtual orbitals are determined
// from the label of the SC state, which is fixed by the deterministic
// ordering (the same as used in coeffsIndex).
template<typename Wfn>
void SCPT<Wfn>::printInitDets(vector<Determinant>& initDets, vector<double>& largestCoeffs)
{
  // Loop over all classes
  for (int i=1; i<9; i++)
  {
    if (classesUsed[i]) {

      string fileName;
      fileName = "init_dets/proc_" + to_string(commrank) + "/init_dets_" +
                 classNames[i] + "_" + to_string(commrank) + ".dat";

      ofstream out_init_dets;
      out_init_dets.open(fileName);

      for (int ind = cumNumCoeffs[i]; ind < cumNumCoeffs[i]+numCoeffsPerClass[i]; ind++) {
        out_init_dets << setprecision(12) << largestCoeffs[ind] << "  ";
        initDets[ind].printActive(out_init_dets);
        out_init_dets << endl;
      }

      out_init_dets.close();

    }
  }
}

// read determinants in to the initDets array from previously output files
template<typename Wfn>
void SCPT<Wfn>::readInitDets(vector<Determinant>& initDets, vector<double>& largestCoeffs)
{
  string fileName;
  std::string line;
  double coeff;

  int norbs = Determinant::norbs;
  int first_virtual = schd.nciCore + schd.nciAct;
  int numVirt = norbs - first_virtual;

  // For each class, loop over all perturbers in the same order that they
  // are printed in the init_dets files (which is the same ordering used
  // in coeffsIndex)

  // Currently this will only read in classes:
  // AAAV, AAVV, CAAA, CAAV and CCAA

  // First, create a determinant with all core orbitals doubly occupied:
  Determinant detCAS;
  for (int i=0; i<schd.nciCore; i++) {
    detCAS.setoccA(i, true);
    detCAS.setoccB(i, true);
  }

  // AAAV
  fileName = "init_dets/proc_" + to_string(commrank) + "/init_dets_AAAV_" + to_string(commrank) + ".dat";
  ifstream dump;
  dump.open(fileName);
  for (int r=2*first_virtual; r<2*norbs; r++) {
    int ind = cumNumCoeffs[1] + r - 2*first_virtual;

    std::getline(dump, line);
    Determinant d(detCAS);
    readDetActive(line, d, coeff);

    d.setocc(r, true);
    initDets[ind] = d;
    largestCoeffs[ind] = coeff;
  }
  dump.close();

  // AAVV
  fileName = "init_dets/proc_" + to_string(commrank) + "/init_dets_AAVV_" + to_string(commrank) + ".dat";
  dump.open(fileName);
  for (int r=2*first_virtual+1; r<2*norbs; r++) {
    for (int s=2*first_virtual; s<r; s++) {
      int R = r - 2*first_virtual - 1;
      int S = s - 2*first_virtual;
      int ind = cumNumCoeffs[2] + R*(R+1)/2 + S;

      std::getline(dump, line);
      Determinant d(detCAS);
      readDetActive(line, d, coeff);

      d.setocc(r, true);
      d.setocc(s, true);
      initDets[ind] = d;
      largestCoeffs[ind] = coeff;
    }
  }
  dump.close();

  // CAAA
  fileName = "init_dets/proc_" + to_string(commrank) + "/init_dets_CAAA_" + to_string(commrank) + ".dat";
  dump.open(fileName);
  for (int i=0; i<2*schd.nciCore; i++) {
    int ind = cumNumCoeffs[3] + i;

    std::getline(dump, line);
    Determinant d(detCAS);
    readDetActive(line, d, coeff);

    d.setocc(i, false);
    initDets[ind] = d;
    largestCoeffs[ind] = coeff;
  }
  dump.close();

  // CAAV
  fileName = "init_dets/proc_" + to_string(commrank) + "/init_dets_CAAV_" + to_string(commrank) + ".dat";
  dump.open(fileName);
  for (int i=0; i<2*schd.nciCore; i++) {
    for (int r=2*first_virtual; r<2*norbs; r++) {
      int ind = cumNumCoeffs[4] + 2*numVirt*i + (r - 2*schd.nciCore - 2*schd.nciAct);

      std::getline(dump, line);
      Determinant d(detCAS);
      readDetActive(line, d, coeff);

      d.setocc(i, false);
      d.setocc(r, true);
      initDets[ind] = d;
      largestCoeffs[ind] = coeff;
    }
  }
  dump.close();

  // CCAA
  fileName = "init_dets/proc_" + to_string(commrank) + "/init_dets_CCAA_" + to_string(commrank) + ".dat";
  dump.open(fileName);
  for (int i=1; i<2*schd.nciCore; i++) {
    for (int j=0; j<i; j++) {
      int ind = cumNumCoeffs[6] + (i-1)*i/2 + j;

      std::getline(dump, line);
      Determinant d(detCAS);
      readDetActive(line, d, coeff);

      d.setocc(i, false);
      d.setocc(j, false);
      initDets[ind] = d;
      largestCoeffs[ind] = coeff;
    }
  }
  dump.close();
}

// From a given line of an output file, containing only the occupations of
// orbitals within the active space, construct the corresponding determinant
// (with all core obritals occupied, all virtual orbitals unocuppied).
// This is specifically used by for readInitDets
template<typename Wfn>
void SCPT<Wfn>::readDetActive(string& line, Determinant& det, double& coeff)
{
  boost::trim_if(line, boost::is_any_of(", \t\n"));

  vector<string> tok;
  boost::split(tok, line, boost::is_any_of(", \t\n"), boost::token_compress_on);

  coeff = stod(tok[0]);

  int offset = schd.nciCore;

  for (int i=0; i<schd.nciAct; i++)
  {
    if (boost::iequals(tok[i+1], "2"))
    {
      det.setoccA(i+offset, true);
      det.setoccB(i+offset, true);
    }
    else if (boost::iequals(tok[i+1], "a"))
    {
      det.setoccA(i+offset, true);
      det.setoccB(i+offset, false);
    }
    if (boost::iequals(tok[i+1], "b"))
    {
      det.setoccA(i+offset, false);
      det.setoccB(i+offset, true);
    }
    if (boost::iequals(tok[i+1], "0"))
    {
      det.setoccA(i+offset, false);
      det.setoccB(i+offset, false);
    }
  }

}

template<typename Wfn>
void SCPT<Wfn>::printNormDataBinary(vector<Determinant>& initDets, vector<double>& largestCoeffs,
                         double& energyCAS_Tot, Eigen::VectorXd& norms_Tot, double& deltaT_Tot)
{
  if (commrank == 0) cout << "About to print norm data..." << endl;

  string name_1 = "norm_data/proc_" + to_string(commrank) + "/norms_" + to_string(commrank) + ".bkp";
  ofstream file_1(name_1, std::ios::binary);
  boost::archive::binary_oarchive oa1(file_1);
  oa1 << norms_Tot;
  file_1.close();

  string name_2 = "norm_data/proc_" + to_string(commrank) + "/init_dets_" + to_string(commrank) + ".bkp";
  ofstream file_2(name_2, std::ios::binary);
  boost::archive::binary_oarchive oa2(file_2);
  oa2 << initDets;
  file_2.close();

  string name_3 = "norm_data/proc_" + to_string(commrank) + "/coeffs_" + to_string(commrank) + ".bkp";
  ofstream file_3(name_3, std::ios::binary);
  boost::archive::binary_oarchive oa3(file_3);
  oa3 << largestCoeffs;
  file_3.close();

  string name_4 = "norm_data/proc_" + to_string(commrank) + "/energy_cas_" + to_string(commrank) + ".bkp";
  ofstream file_4(name_4, std::ios::binary);
  boost::archive::binary_oarchive oa4(file_4);
  oa4 << energyCAS_Tot;
  file_4.close();

  string name_5 = "norm_data/proc_" + to_string(commrank) + "/delta_t_" + to_string(commrank) + ".bkp";
  ofstream file_5(name_5, std::ios::binary);
  boost::archive::binary_oarchive oa5(file_5);
  oa5 << deltaT_Tot;
  file_5.close();

  if (commrank == 0) cout << "Printing complete." << endl << endl;
}

template<typename Wfn>
void SCPT<Wfn>::readNormDataBinary(vector<Determinant>& initDets, vector<double>& largestCoeffs,
                        double& energyCAS_Tot, Eigen::VectorXd& norms_Tot, double& deltaT_Tot, bool readDeltaT)
{
  if (commrank == 0) cout << "About to read norm data..." << endl;

  string name_1 = "norm_data/proc_" + to_string(commrank) + "/norms_" + to_string(commrank) + ".bkp";
  ifstream file_1(name_1, std::ios::binary);
  boost::archive::binary_iarchive oa1(file_1);
  oa1 >> norms_Tot;
  file_1.close();

  string name_2 = "norm_data/proc_" + to_string(commrank) + "/init_dets_" + to_string(commrank) + ".bkp";
  ifstream file_2(name_2, std::ios::binary);
  boost::archive::binary_iarchive oa2(file_2);
  oa2 >> initDets;
  file_2.close();

  string name_3 = "norm_data/proc_" + to_string(commrank) + "/coeffs_" + to_string(commrank) + ".bkp";
  ifstream file_3(name_3, std::ios::binary);
  boost::archive::binary_iarchive oa3(file_3);
  oa3 >> largestCoeffs;
  file_3.close();

  string name_4 = "norm_data/proc_" + to_string(commrank) + "/energy_cas_" + to_string(commrank) + ".bkp";
  ifstream file_4(name_4, std::ios::binary);
  boost::archive::binary_iarchive oa4(file_4);
  oa4 >> energyCAS_Tot;
  file_4.close();

  if (readDeltaT) {
    string name_5 = "norm_data/proc_" + to_string(commrank) + "/delta_t_" + to_string(commrank) + ".bkp";
    ifstream file_5(name_5, std::ios::binary);
    boost::archive::binary_iarchive oa5(file_5);
    oa5 >> deltaT_Tot;
    file_5.close();
  }

  if (commrank == 0) cout << "Reading complete." << endl << endl;
}

/*
template<typename Wfn>
void SCPT<Wfn>::readNormDataText(vector<Determinant>& initDets, vector<double>& largestCoeffs,
                      double& energyCAS_Tot, Eigen::VectorXd& norms_Tot, double& deltaT_Tot)
{
  // Read in the stochastically-sampled norms
  if (commrank == 0) cout << "About to read stochastically-sampled norms..." << endl;
  readStochNorms(deltaT_Tot, energyCAS_Tot, norms_Tot);
  if (commrank == 0) cout << "Reading complete." << endl;

  energyCAS_Tot /= deltaT_Tot;
  norms_Tot /= deltaT_Tot;

  if (commrank == 0) cout << "About to read exactly-calculated norms..." << endl;
  readDetermNorms(norms_Tot);
  if (commrank == 0) cout << "Reading complete." << endl;

  // Read in the exactly-calculated norms
  if (commrank == 0) cout << "About to read initial determinants..." << endl;
  readInitDets(initDets, largestCoeffs);
  if (commrank == 0) cout << "Reading complete." << endl;

  if (commrank == 0) cout << endl;
}
*/
template<typename Wfn>
template<typename Walker>
double SCPT<Wfn>::doNEVPT2_CT_Efficient(Walker& walk) {

  double energy_ccvv = 0.0;
  if (commrank == 0) {
    cout << "Integrals and wave function preparation finished in " << getTime() - startofCalc << " s\n";
    if (any_of(classesUsedDeterm.begin(), classesUsedDeterm.end(), [](bool i){return i;}) ) {
      if (classesUsedDeterm[8])
      {
        energy_ccvv = get_ccvv_energy();
        cout << endl << "Deterministic CCVV energy:  " << setprecision(12) << energy_ccvv << endl << endl;
      }
    }
  }

  if (schd.SCNormsBurnIn >= schd.stochasticIterNorms) {
    if (commrank == 0) {
      cout << "WARNING: The number of sampling iterations for N_l^k estimation is"
              " not larger than the number of burn-in iterations. Setting the number"
              " of burn-in iterations to 0." << endl << endl;
    }
    schd.SCNormsBurnIn = 0;
  }

  if (schd.SCEnergiesBurnIn >= schd.stochasticIterEachSC) {
    if (commrank == 0) {
      cout << "WARNING: The number of sampling iterations for E_l^k estimation is"
              " not larger than the number of burn-in iterations. Setting the number"
              " of burn-in iterations to 0." << endl << endl;
    }
    schd.SCEnergiesBurnIn = 0;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (commrank == 0) cout << "About to sample the norms of the strongly contracted states..." << endl << endl;
  double timeNormsInit = getTime();

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));
  workingArray work;

  double ham = 0., hamSample = 0., ovlp = 0.;
  double deltaT = 0., deltaT_Tot = 0.;
  double energyCAS = 0., energyCAS_Tot = 0.;

  if (commrank == 0) {
    cout << "About to allocate arrays for sampling..." << endl;
    cout << "Total number of strongly contracted states to sample: " << numCoeffs << endl;
  }

  VectorXd normSamples = VectorXd::Zero(coeffs.size());
  VectorXd norms_Tot = VectorXd::Zero(coeffs.size());

  if (schd.printSCNorms) createDirForNormsBinary();

  // As we calculate the SC norms, we will simultaneously find the determinants
  // within each SC space that have the highest coefficient, as found during
  // the sampling. These quantities are searched for in the following arrays.
  vector<Determinant> initDets;
  initDets.resize(numCoeffs, walk.d);
  vector<double> largestCoeffs;
  largestCoeffs.resize(numCoeffs, 0.0);

  MPI_Barrier(MPI_COMM_WORLD);
  if (commrank == 0) cout << "Allocation of sampling arrays now finished." << endl << endl;

  if (schd.continueSCNorms) {
    readNormDataBinary(initDets, largestCoeffs, energyCAS_Tot, norms_Tot, deltaT_Tot, true);
    energyCAS_Tot *= deltaT_Tot;
    norms_Tot *= deltaT_Tot;
  }

  // If we are generating the norms stochastically here, rather than
  // reading them back in:
  if (!schd.readSCNorms)
  {
    if (commrank == 0) cout << "About to call first instance of HamAndSCNorms..." << endl;
    HamAndSCNorms(walk, ovlp, hamSample, normSamples, initDets, largestCoeffs, work, false);
    if (commrank == 0) cout << "First instance of HamAndSCNorms complete." << endl;

    int iter = 1;

    int printMod;
    if (schd.fixedResTimeNEVPT_Norm) {
      printMod = 10;
    } else {
      printMod = max(1, schd.stochasticIterNorms / 10);
    }

    if (commrank == 0) {
      cout << "iter: 0" << "  t: " << setprecision(6) << getTime() - startofCalc << endl;
    }

    while (true) {

      // Condition for exiting loop:
      // This depends on what 'mode' we are running in - constant
      // residence time, or constant iteration count
      if (schd.fixedResTimeNEVPT_Norm) {
        if (deltaT_Tot >= schd.resTimeNEVPT_Norm)
          break;
      } else {
        if (iter > schd.stochasticIterNorms)
          break;
      }

      double cumovlpRatio = 0;
      for (int i = 0; i < work.nExcitations; i++) {
        cumovlpRatio += abs(work.ovlpRatio[i]);
        work.ovlpRatio[i] = cumovlpRatio;
      }

      if (iter > schd.SCNormsBurnIn)
      {
        double deltaT = 1.0 / (cumovlpRatio);

        energyCAS = deltaT * hamSample;
        normSamples *= deltaT;

        // These hold the running totals
        deltaT_Tot += deltaT;
        energyCAS_Tot += energyCAS;
        norms_Tot += normSamples;
      }

      //if (schd.printSCNorms && iter % schd.printSCNormFreq == 0)
      //  printSCNorms(iter, deltaT_Tot, energyCAS_Tot, norms_Tot, false);

      // Pick the next determinant by the CTMC algorithm
      double nextDetRandom = random() * cumovlpRatio;
      int nextDet = std::lower_bound(work.ovlpRatio.begin(), (work.ovlpRatio.begin() + work.nExcitations),
                                     nextDetRandom) - work.ovlpRatio.begin();

      walk.updateWalker(wave.getRef(), wave.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);

      if (walk.excitation_class != 0) cout << "ERROR: walker not in CASCI space!" << endl;

      normSamples.setZero();

      // For schd.nIterFindInitDets iterations, we want to sample the
      // norms for *all* S_l^k, including those for which we can find
      // N_l^k exactly (this allows us to generate initital determinants).
      // This is the condition for the iterations in which we do this:
      bool sampleAllNorms;
      if (schd.fixedResTimeNEVPT_Norm) {
        // In this case, we use the first schd.nIterFindInitDets iterations
        // *after* the burn-in period.
        sampleAllNorms = (iter >= schd.SCNormsBurnIn) && (iter < schd.SCNormsBurnIn + schd.nIterFindInitDets);
      } else {
        // In this case, we just use the final schd.nIterFindInitDets iterations.
        sampleAllNorms = schd.stochasticIterNorms - schd.nIterFindInitDets < iter;
      }

      HamAndSCNorms(walk, ovlp, hamSample, normSamples, initDets, largestCoeffs, work, sampleAllNorms);

      if (commrank == 0 && (iter % printMod == 0 || iter == 1))
        cout << "iter: " << iter << "  t: " << setprecision(6) << getTime() - startofCalc << endl;
      iter++;
    }
    int samplingIters = iter - schd.SCNormsBurnIn - 1;
    //cout << "proc: " << commrank << "  samplingIters: " << samplingIters << endl;

    energyCAS_Tot /= deltaT_Tot;
    norms_Tot /= deltaT_Tot;

    if (any_of(normsDeterm.begin(), normsDeterm.end(), [](bool i){return i;}) )
    {
      MatrixXd oneRDM, twoRDM;
      readSpinRDM(oneRDM, twoRDM);

      if (normsDeterm[2])
        calc_AAVV_NormsFromRDMs(twoRDM, norms_Tot);
      if (normsDeterm[4])
        calc_CAAV_NormsFromRDMs(oneRDM, twoRDM, norms_Tot);
      if (normsDeterm[6])
        calc_CCAA_NormsFromRDMs(oneRDM, twoRDM, norms_Tot);

      if (schd.printSCNorms && commrank == 0)
        printSCNorms(iter, deltaT_Tot, energyCAS_Tot, norms_Tot, true);
    }

    if (commrank == 0)
    {
      cout << endl << "Calculation of strongly contracted norms complete." << endl;
      cout << "Total time for norms calculation:  " << getTime() - timeNormsInit << endl << endl;
    }
  }

  if (schd.readSCNorms) {
    readNormDataBinary(initDets, largestCoeffs, energyCAS_Tot, norms_Tot, deltaT_Tot, false);
  }

  if (schd.printSCNorms && (!schd.readSCNorms)) {
    printNormDataBinary(initDets, largestCoeffs, energyCAS_Tot, norms_Tot, deltaT_Tot);
  }

  if (schd.sampleNEVPT2Energy) {

    if (commrank == 0) {
      cout << "Now sampling the NEVPT2 energy..." << endl;
    }

    // Next we calculate the SC state energies and the final PT2 energy estimate
    double timeEnergyInit = getTime();
    double ene2;
    if (schd.efficientNEVPT_2) {
      ene2 = sampleSCEnergies(walk, initDets, largestCoeffs, energyCAS_Tot, norms_Tot, work);
    }
    if (schd.efficientNEVPT || schd.exactE_NEVPT) {
      ene2 = sampleAllSCEnergies(walk, initDets, largestCoeffs, energyCAS_Tot, norms_Tot, work);
    }
    if (schd.NEVPT_writeE || schd.NEVPT_readE) {
      ene2 = calcAllSCEnergiesExact(walk, initDets, largestCoeffs, energyCAS_Tot, norms_Tot, work);
    }

    if (commrank == 0) {
      cout << "Sampling complete." << endl << endl;

      cout << "Total time for energy sampling " << getTime() - timeEnergyInit << " seconds" << endl;
      cout << "CAS energy: " << setprecision(10) << energyCAS_Tot << endl;
      cout << "SC-NEVPT2(s) second-order energy: " << setprecision(10) << ene2 << endl;
      cout << "Total SC-NEVPT(s) energy: " << setprecision(10) << energyCAS_Tot + ene2 << endl;

      if (any_of(classesUsedDeterm.begin(), classesUsedDeterm.end(), [](bool i){return i;}) ) {
        if (classesUsedDeterm[8])
        {
          cout << "SC-NEVPT2(s) second-order energy with CCVV:  " << energy_ccvv + ene2 << endl;
          cout << "Total SC-NEVPT2(s) energy with CCVV:  " << energyCAS_Tot + ene2 + energy_ccvv << endl;
        }
      }

    }

  } // If sampling the energy
}

template<typename Wfn>
template<typename Walker>
double SCPT<Wfn>::sampleSCEnergies(Walker& walk, vector<Determinant>& initDets, vector<double>& largestCoeffs,
                        double& energyCAS_Tot, Eigen::VectorXd& norms_Tot, workingArray& work)
{
  vector<double> cumNorm;
  cumNorm.resize(numCoeffs, 0.0);
  vector<int> indexMap;
  indexMap.resize(numCoeffs, 0);

  int numCoeffsToSample = 0;
  double totCumNorm = 0.;
  for (int i = 1; i < numCoeffs; i++) {
    if (norms_Tot(i) > schd.overlapCutoff) {
      totCumNorm += norms_Tot(i);
      cumNorm[numCoeffsToSample] = totCumNorm;
      initDets[numCoeffsToSample] = initDets[i];
      largestCoeffs[numCoeffsToSample] = largestCoeffs[i];
      indexMap[numCoeffsToSample] = i;
      numCoeffsToSample += 1;
    }
  }

  if (commrank == 0) {
    cout << "Total cumulative squared norm (process 0): " << totCumNorm << endl << endl;
  }

  double energySample = 0., energyTot = 0, biasTot = 0;
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

  FILE * pt2_out;
  string pt2OutName = "pt2_energies_";
  pt2OutName.append(to_string(commrank));
  pt2OutName.append(".dat");

  pt2_out = fopen(pt2OutName.c_str(), "w");
  fprintf(pt2_out, "# 1. iter    2. energy            3. E_0 - E_l^k        4. E_l^K variance    "
                   "5. Bias correction  6. class   7. C/V orbs  8. niters   9. time\n");

  double timeInTotal = getTime();
  int orbi, orbj;

  int iter = 0;
  while (iter < schd.numSCSamples) {
    double timeIn = getTime();

    double nextSCRandom = random() * totCumNorm;
    int nextSC = std::lower_bound(cumNorm.begin(), (cumNorm.begin() + numCoeffsToSample), nextSCRandom) - cumNorm.begin();

    if (abs(largestCoeffs[nextSC]) < 1.e-15) cout << "Error: no initial determinant found:  " << setprecision(20) << largestCoeffs[nextSC] << endl;

    this->wave.initWalker(walk, initDets[nextSC]);

    double SCHam, SCHamVar;
    int samplingIters;

    if (schd.printSCEnergies) {
      SCHam = doSCEnergyCTMCPrint(walk, work, iter, schd.nWalkSCEnergies);
    } else {
      doSCEnergyCTMC(walk, work, SCHam, SCHamVar, samplingIters);
    }

    // If this same SC sector is sampled again, start from the final
    // determinant from this time:
    if (schd.continueMarkovSCPT) initDets[nextSC] = walk.d;

    energySample = totCumNorm / (energyCAS_Tot - SCHam);
    double biasCorr = - totCumNorm * ( SCHamVar / pow( energyCAS_Tot - SCHam, 3) );

    energyTot += energySample;
    biasTot += biasCorr;

    double timeOut = getTime();
    double eDiff = energyCAS_Tot - SCHam;

    // Get the external orbs of the perturber, for printing
    getOrbsFromIndex(indexMap[nextSC], orbi, orbj);
    string orbString = formatOrbString(orbi, orbj);

    fprintf(pt2_out, "%9d   %.12e   %.12e   %.12e   %.12e      %4s  %12s   %8d   %.4e\n",
            iter, energySample, eDiff, SCHamVar, biasCorr,
            classNames2[walk.excitation_class].c_str(),
            orbString.c_str(), samplingIters, timeOut-timeIn);
    fflush(pt2_out);

    iter++;
  }

  double timeOutTotal = getTime();
  double timeTotal = timeOutTotal - timeInTotal;

  fclose(pt2_out);

  energyTot /= iter;
  biasTot /= iter;

  // Average over MPI processes
  double energyFinal = 0., biasFinal = 0.;
#ifndef SERIAL
  // Check how long processes have to wait for other MPI processes to finish
  double timeIn = getTime();
  MPI_Barrier(MPI_COMM_WORLD);
  double timeOut = getTime();
  if (commrank == 0) cout << "MPI Barrier time: " << timeOut-timeIn << " seconds" << endl;

  // Gather and print energy estimates from all MPI processes
  double energyTotAll[commsize];
  MPI_Gather(&(energyTot), 1, MPI_DOUBLE, &(energyTotAll), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  double biasTotAll[commsize];
  MPI_Gather(&(biasTot), 1, MPI_DOUBLE, &(biasTotAll), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  // Gather the total time on each process
  double timeTotalAll[commsize];
  MPI_Gather(&(timeTotal), 1, MPI_DOUBLE, &(timeTotalAll), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (commrank == 0) {
    // Print final estimates from each process
    FILE * mpi_out;
    mpi_out = fopen("pt2_energies_avg.dat", "w");
    fprintf(mpi_out, "# 1. proc label     2. energy            "
                     "3. bias correction   4. time\n");
    for (int i=0; i<commsize; i++) {
      fprintf(mpi_out, "%15d    %.12e   %.12e   %.6e\n",
              i, energyTotAll[i], biasTotAll[i], timeTotalAll[i]);
    }
    fclose(mpi_out);

    // Calculate the energy and bias averaged over all processes
    for (int i=0; i<commsize; i++) {
      energyFinal += energyTotAll[i];
      biasFinal += biasTotAll[i];
    }
    energyFinal /= commsize;
    biasFinal /= commsize;

    // Calculate the standard error for the energy and bias estimates
    double stdDevEnergy = 0., stdDevBias = 0.;
    for (int i=0; i<commsize; i++) {
      stdDevEnergy += pow( energyTotAll[i] - energyFinal, 2 );
      stdDevBias += pow( biasTotAll[i] - biasFinal, 2 );
    }
    stdDevEnergy /= commsize - 1;
    stdDevBias /= commsize - 1;

    cout.precision(12);
    cout << "Energy error estimate: " << sqrt(stdDevEnergy / commsize) << endl;
    cout << "Bias correction error estimate: " << sqrt(stdDevBias / commsize) << endl;
  }
  MPI_Bcast(&energyFinal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&biasFinal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
  energyFinal = energyTot;
  biasFinal = biasTot;
#endif

  // Final energy returned includes the bias correction estimate
  return energyFinal + biasFinal;
}

template<typename Wfn>
template<typename Walker>
void SCPT<Wfn>::doSCEnergyCTMC(Walker& walk, workingArray& work, double& final_ham, double& var, int& samplingIters)
{
  double ham = 0., hamSample = 0., ovlp = 0.;
  double numerator = 0., numerator_Tot = 0.;
  double deltaT = 0., deltaT_Tot = 0.;

  int nSampleIters = schd.stochasticIterEachSC - schd.SCEnergiesBurnIn;

  vector<double> x, w;

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

  //int coeffsIndexCopy = this->coeffsIndex(walk);
  //if (coeffsIndexCopy != ind) cout << "ERROR at 1: " << ind << "    " << coeffsIndexCopy << endl;

  // Now, sample the SC energy in this space
  FastHamAndOvlp(walk, ovlp, hamSample, work);

  int iter = 0;

  while (true) {

    // Condition for exiting loop:
    // This depends on what 'mode' we are running in - constant
    // residence time, or constant iteration count
    if (schd.fixedResTimeNEVPT_Ene) {
      if (deltaT_Tot >= schd.resTimeNEVPT_Ene)
        break;
    } else {
      if (iter >= schd.stochasticIterEachSC)
        break;
    }

    double cumovlpRatio = 0.;
    for (int i = 0; i < work.nExcitations; i++) {
      cumovlpRatio += abs(work.ovlpRatio[i]);
      work.ovlpRatio[i] = cumovlpRatio;
    }

    // Only start accumulating data if beyond the burn-in period
    if (iter >= schd.SCEnergiesBurnIn) {
      deltaT = 1.0 / (cumovlpRatio);
      numerator = deltaT*hamSample;

      x.push_back(hamSample);
      w.push_back(deltaT);

      numerator_Tot += numerator;
      deltaT_Tot += deltaT;
    }

    double nextDetRandom = random() * cumovlpRatio;
    int nextDet = std::lower_bound(work.ovlpRatio.begin(), (work.ovlpRatio.begin() + work.nExcitations),
                                   nextDetRandom) - work.ovlpRatio.begin();

    walk.updateWalker(wave.getRef(), wave.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);

    //int coeffsIndexCopy = this->coeffsIndex(walk);
    //if (coeffsIndexCopy != ind) cout << "ERROR at 2: " << ind << "    " << coeffsIndexCopy << endl;

    FastHamAndOvlp(walk, ovlp, hamSample, work);
    iter++;
  }

  samplingIters = iter - schd.SCEnergiesBurnIn;
  final_ham = numerator_Tot/deltaT_Tot;

  // Estimate the error on final_ham
  var = SCEnergyVar(x, w);
}

// Estimate the variance of the weighted mean used to estimate E_l^k
template<typename Wfn>
double SCPT<Wfn>::SCEnergyVar(vector<double>& x, vector<double>& w)
{
  // This uses blocks of data, to account for the serial correlation
  int n = x.size();

  // If we only have one sample, we can't estimate the variance, so
  // just return 0 instead
  if (n == 1)
  {
    return 0.0;
  }

  int block_size = min(n/5, 16);
  if (block_size < 1) {
    block_size = 1;
  }

  int nblocks = n/block_size;

  vector<double> x_1, w_1;
  x_1.assign(nblocks, 0.0);
  w_1.assign(nblocks, 0.0);

  for (int i = 0; i < nblocks; i++)
  {
    for (int j = 0; j < block_size; j++)
    {
      w_1[i] += w[block_size*i + j];
      x_1[i] += w[block_size*i + j] * x[block_size*i + j];
    }
    x_1[i] /= w_1[i];
  }
  n = nblocks;
  x = x_1;
  w = w_1;

  double x_bar = 0.0, W = 0.0;
  for (int i = 0; i < n; i++)
  {
    x_bar += w[i] * x[i];
    W += w[i];
  }
  x_bar /= W;

  double var = 0.;
  for (int i = 0; i < n; i++)
  {
    var += pow(w[i]*x[i] - W*x_bar, 2)
         - 2*x_bar*(w[i] - W)*(w[i]*x[i] - W*x_bar)
         + pow(x_bar, 2) * pow(w[i] - W, 2);
  }

  var *= n / (n - 1.0);
  var /= pow(W, 2);

  return var;
}

template<typename Wfn>
template<typename Walker>
double SCPT<Wfn>::doSCEnergyCTMCPrint(Walker& walk, workingArray& work, int sampleIter, int nWalk)
{
  double ham = 0., hamSample = 0., ovlp = 0.;
  double numerator = 0., numerator_Tot = 0.;
  double deltaT = 0., deltaT_Tot = 0.;

  vector<double> numerator_Avg(schd.stochasticIterEachSC, 0.0);
  vector<double> deltaT_Avg(schd.stochasticIterEachSC, 0.0);

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

  Determinant initDet = walk.d;

  FILE * out;
  string outputFile = "sc_energies.";
  outputFile.append(to_string(commrank));
  outputFile.append(".");
  outputFile.append(to_string(sampleIter));

  out = fopen(outputFile.c_str(), "w");
  fprintf(out, "# 1. iteration     2. weighted_energy     3. residence_time\n");

  //int coeffsIndexCopy = this->coeffsIndex(walk);
  //if (coeffsIndexCopy != ind) cout << "ERROR at 1: " << ind << "    " << coeffsIndexCopy << endl;

  for (int i=0; i<nWalk; i++) {

    // Reinitialize the walker
    this->wave.initWalker(walk, initDet);

    // Now, sample the SC energy in this space
    FastHamAndOvlp(walk, ovlp, hamSample, work);

    int iter = 0;
    while (iter < schd.stochasticIterEachSC) {
      double cumovlpRatio = 0.;
      for (int i = 0; i < work.nExcitations; i++) {
        cumovlpRatio += abs(work.ovlpRatio[i]);
        work.ovlpRatio[i] = cumovlpRatio;
      }

      deltaT = 1.0 / (cumovlpRatio);
      numerator = deltaT*hamSample;

      numerator_Avg[iter] += numerator;
      deltaT_Avg[iter] += deltaT;

      numerator_Tot += numerator;
      deltaT_Tot += deltaT;

      double nextDetRandom = random() * cumovlpRatio;
      int nextDet = std::lower_bound(work.ovlpRatio.begin(), (work.ovlpRatio.begin() + work.nExcitations),
                                     nextDetRandom) - work.ovlpRatio.begin();

      walk.updateWalker(wave.getRef(), wave.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);

      //int coeffsIndexCopy = this->coeffsIndex(walk);
      //if (coeffsIndexCopy != ind) cout << "ERROR at 2: " << ind << "    " << coeffsIndexCopy << endl;

      FastHamAndOvlp(walk, ovlp, hamSample, work);
      iter++;
    }
  }

  double final_ham = numerator_Tot/deltaT_Tot;

  for (int i=0; i<schd.stochasticIterEachSC; i++) {
    fprintf(out, "%14d    %.12e    %.12e\n", i, numerator_Avg[i], deltaT_Avg[i]);
  }
  fclose(out);

  return final_ham;
}

// Wrapper function for calling doSCEnergyCTMC, which estimates E_l^k,
// given the appropriate input information, and then to print this info
// to the provded pt2_out file.
// This is designed to be called by sampleAllSCEnergies.
template<typename Wfn>
template<typename Walker>
double SCPT<Wfn>::SCEnergyWrapper(Walker& walk, int iter, FILE * pt2_out, Determinant& det,
                       double& energyCAS_Tot, double norm, int orbi, int orbj,
                       bool exactCalc, bool exactRead, double& SCHam, workingArray& work) {

  double SCHamVar = 0.;
  int samplingIters = 0;

  double timeIn = getTime();

  this->wave.initWalker(walk, det);

  // If we have already read in SCHam, then we don't need to
  // calculate or sample it here.
  if (!exactRead) {
    if (exactCalc) {
      doSCEnergyExact(walk, work, SCHam, SCHamVar, samplingIters);
    }
    else {
      doSCEnergyCTMC(walk, work, SCHam, SCHamVar, samplingIters);
    }
  }

  double energySample = norm / (energyCAS_Tot - SCHam);
  double biasCorr = - norm * ( SCHamVar / pow( energyCAS_Tot - SCHam, 3) );
  double eDiff = energyCAS_Tot - SCHam;
  string orbString = formatOrbString(orbi, orbj);

  double timeOut = getTime();

  fprintf(pt2_out, "%9d   %.12e   %.12e   %.12e   %.12e      %4s  %12s   %8d   %.4e\n",
          iter, energySample, eDiff, SCHamVar, biasCorr,
          classNames2[walk.excitation_class].c_str(),
          orbString.c_str(), samplingIters, timeOut-timeIn);
  fflush(pt2_out);

  return energySample;
}

// Loop over *all* S_l^k subspaces (for the classes AAAV, AAVV, CAAA,
// CAAV and CCAA) for which the calculated norm is above the threshold,
// and sample E_l^k for each. The final PT2 energy is then output as a
// sum over all of these spaces.
template<typename Wfn>
template<typename Walker>
double SCPT<Wfn>::sampleAllSCEnergies(Walker& walk, vector<Determinant>& initDets, vector<double>& largestCoeffs,
                           double& energyCAS_Tot, Eigen::VectorXd& norms_Tot, workingArray& work)
{
  int norbs = Determinant::norbs;
  int first_virtual = schd.nciCore + schd.nciAct;
  int numVirt = norbs - first_virtual;

  double ene2 = 0., SCHam = 0.;

  FILE * pt2_out;
  string pt2OutName = "pt2_energies_";
  pt2OutName.append(to_string(commrank));
  pt2OutName.append(".dat");

  pt2_out = fopen(pt2OutName.c_str(), "w");
  fprintf(pt2_out, "# 1. iter    2. energy            3. E_0 - E_l^k        4. E_l^K variance    "
                   "5. Bias correction  6. class   7. C/V orbs  8. niters   9. time\n");

  int iter = 0;

  // AAAV
  for (int r=2*first_virtual; r<2*norbs; r++) {
    int ind = cumNumCoeffs[1] + r - 2*first_virtual;
    if (norms_Tot(ind) > schd.overlapCutoff) {
      if (largestCoeffs[ind] == 0.0) cout << "Error: No initial determinant found. " << r << endl;

      int orb1 = r;
      int orb2 = -1;
      ene2 += SCEnergyWrapper(walk, iter, pt2_out, initDets[ind], energyCAS_Tot,
                              norms_Tot(ind), orb1, orb2, schd.exactE_NEVPT, false, SCHam, work);
      iter++;
    }
  }

  // AAVV
  for (int r=2*first_virtual+1; r<2*norbs; r++) {
    for (int s=2*first_virtual; s<r; s++) {
      int R = r - 2*first_virtual - 1;
      int S = s - 2*first_virtual;
      int ind = cumNumCoeffs[2] + R*(R+1)/2 + S;

      if (norms_Tot(ind) > schd.overlapCutoff) {
        if (largestCoeffs[ind] == 0.0) {
          cout << "Warning: No initial determinant found. " << r << "  " << s << endl;
          continue;
        }
        int orb1 = r;
        int orb2 = s;
        ene2 += SCEnergyWrapper(walk, iter, pt2_out, initDets[ind], energyCAS_Tot,
                                norms_Tot(ind), orb1, orb2, schd.exactE_NEVPT, false, SCHam, work);
        iter++;
      }
    }
  }

  // CAAA
  for (int i=0; i<2*schd.nciCore; i++) {
    int ind = cumNumCoeffs[3] + i;

    if (norms_Tot(ind) > schd.overlapCutoff) {
      if (largestCoeffs[ind] == 0.0) cout << "Error: No initial determinant found. " << i << endl;

      int orb1 = i;
      int orb2 = -1;
      ene2 += SCEnergyWrapper(walk, iter, pt2_out, initDets[ind], energyCAS_Tot,
                              norms_Tot(ind), orb1, orb2, schd.exactE_NEVPT, false, SCHam, work);
      iter++;
    }
  }

  // CAAV
  for (int i=0; i<2*schd.nciCore; i++) {
    for (int r=2*first_virtual; r<2*norbs; r++) {
      int ind = cumNumCoeffs[4] + 2*numVirt*i + (r - 2*schd.nciCore - 2*schd.nciAct);

      if (norms_Tot(ind) > schd.overlapCutoff) {
        if (largestCoeffs[ind] == 0.0) {
          cout << "Warning: No initial determinant found. " << i << "  " << r << endl;
          continue;
        }

        int orb1 = i;
        int orb2 = r;
        ene2 += SCEnergyWrapper(walk, iter, pt2_out, initDets[ind], energyCAS_Tot,
                                norms_Tot(ind), orb1, orb2, schd.exactE_NEVPT, false, SCHam, work);
        iter++;
      }
    }
  }

  // CCAA
  for (int i=1; i<2*schd.nciCore; i++) {
    for (int j=0; j<i; j++) {
      int ind = cumNumCoeffs[6] + (i-1)*i/2 + j;

      if (norms_Tot(ind) > schd.overlapCutoff) {
        if (largestCoeffs[ind] == 0.0) {
          cout << "Warning: No initial determinant found. " << i << "  " << j << endl;
          continue;
        }

        int orb1 = i;
        int orb2 = j;
        ene2 += SCEnergyWrapper(walk, iter, pt2_out, initDets[ind], energyCAS_Tot,
                                norms_Tot(ind), orb1, orb2, schd.exactE_NEVPT, false, SCHam, work);
        iter++;
      }
    }
  }

  fclose(pt2_out);

  return ene2;
}

// Loop over *all* S_l^k subspaces (for the classes AAAV, AAVV, CAAA,
// CAAV and CCAA), and either exactly calculate of read in E_l^k for each.
// The final PT2 energy is then output as a sum over all of these spaces.
//
// The difference between this and sampleAllSCEnergies is that *all*
// S_l^k are considered, even if the norm was calculated as zero, if run
// in write mode, and all E_l^k are then written out. If run in read mode,
// then all E_l^k are read in from this file instead, for quick calculation.
template<typename Wfn>
template<typename Walker>
double SCPT<Wfn>::calcAllSCEnergiesExact(Walker& walk, vector<Determinant>& initDets, vector<double>& largestCoeffs,
                              double& energyCAS_Tot, Eigen::VectorXd& norms_Tot, workingArray& work)
{

  vector<double> exactEnergies;

  if (schd.NEVPT_readE) {
    string name = "exact_energies.bkp";
    ifstream file(name, std::ios::binary);
    boost::archive::binary_iarchive oa(file);
    oa >> exactEnergies;
    file.close();
  }
  else {
    exactEnergies.resize(numCoeffs, 0.0);
  }

  int norbs = Determinant::norbs;
  int first_virtual = schd.nciCore + schd.nciAct;
  int numVirt = norbs - first_virtual;

  double ene2 = 0., SCHam = 0.;

  FILE * pt2_out;
  string pt2OutName = "pt2_energies_";
  pt2OutName.append(to_string(commrank));
  pt2OutName.append(".dat");

  pt2_out = fopen(pt2OutName.c_str(), "w");
  fprintf(pt2_out, "# 1. iter    2. energy            3. E_0 - E_l^k        4. E_l^K variance    "
                   "5. Bias correction  6. class   7. C/V orbs  8. niters   9. time\n");

  int iter = 0;

  // AAAV
  for (int r=2*first_virtual; r<2*norbs; r++) {
    int ind = cumNumCoeffs[1] + r - 2*first_virtual;

    int orb1 = r;
    int orb2 = -1;
    Determinant initDet = generateInitDet(orb1, orb2);
    if (schd.NEVPT_readE) SCHam = exactEnergies.at(ind);
    ene2 += SCEnergyWrapper(walk, iter, pt2_out, initDet, energyCAS_Tot,
                            norms_Tot(ind), orb1, orb2, schd.NEVPT_writeE, schd.NEVPT_readE, SCHam, work);
    if (schd.NEVPT_writeE) exactEnergies.at(ind) = SCHam;
    iter++;
  }

  // AAVV
  for (int r=2*first_virtual+1; r<2*norbs; r++) {
    for (int s=2*first_virtual; s<r; s++) {
      int R = r - 2*first_virtual - 1;
      int S = s - 2*first_virtual;
      int ind = cumNumCoeffs[2] + R*(R+1)/2 + S;

      int orb1 = r;
      int orb2 = s;
      Determinant initDet = generateInitDet(orb1, orb2);
      if (schd.NEVPT_readE) SCHam = exactEnergies.at(ind);
      ene2 += SCEnergyWrapper(walk, iter, pt2_out, initDet, energyCAS_Tot,
                              norms_Tot(ind), orb1, orb2, schd.NEVPT_writeE, schd.NEVPT_readE, SCHam, work);
      if (schd.NEVPT_writeE) exactEnergies.at(ind) = SCHam;
      exactEnergies.at(ind) = SCHam;
      iter++;
    }
  }

  // CAAA
  for (int i=0; i<2*schd.nciCore; i++) {
    int ind = cumNumCoeffs[3] + i;

    int orb1 = i;
    int orb2 = -1;
    Determinant initDet = generateInitDet(orb1, orb2);
    if (schd.NEVPT_readE) SCHam = exactEnergies.at(ind);
    ene2 += SCEnergyWrapper(walk, iter, pt2_out, initDet, energyCAS_Tot,
                            norms_Tot(ind), orb1, orb2, schd.NEVPT_writeE, schd.NEVPT_readE, SCHam, work);
    if (schd.NEVPT_writeE) exactEnergies.at(ind) = SCHam;
    exactEnergies.at(ind) = SCHam;
    iter++;
  }

  // CAAV
  for (int i=0; i<2*schd.nciCore; i++) {
    for (int r=2*first_virtual; r<2*norbs; r++) {
      int ind = cumNumCoeffs[4] + 2*numVirt*i + (r - 2*schd.nciCore - 2*schd.nciAct);

      int orb1 = i;
      int orb2 = r;
      Determinant initDet = generateInitDet(orb1, orb2);
      if (schd.NEVPT_readE) SCHam = exactEnergies.at(ind);
      ene2 += SCEnergyWrapper(walk, iter, pt2_out, initDet, energyCAS_Tot,
                              norms_Tot(ind), orb1, orb2, schd.NEVPT_writeE, schd.NEVPT_readE, SCHam, work);
      if (schd.NEVPT_writeE) exactEnergies.at(ind) = SCHam;
      exactEnergies.at(ind) = SCHam;
      iter++;
    }
  }

  // CCAA
  for (int i=1; i<2*schd.nciCore; i++) {
    for (int j=0; j<i; j++) {
      int ind = cumNumCoeffs[6] + (i-1)*i/2 + j;

      int orb1 = i;
      int orb2 = j;
      Determinant initDet = generateInitDet(orb1, orb2);
      if (schd.NEVPT_readE) SCHam = exactEnergies.at(ind);
      ene2 += SCEnergyWrapper(walk, iter, pt2_out, initDet, energyCAS_Tot,
                              norms_Tot(ind), orb1, orb2, schd.NEVPT_writeE, schd.NEVPT_readE, SCHam, work);
      if (schd.NEVPT_writeE) exactEnergies.at(ind) = SCHam;
      exactEnergies.at(ind) = SCHam;
      iter++;
    }
  }

  fclose(pt2_out);

  if (schd.NEVPT_writeE) {
    string name = "exact_energies.bkp";
    ofstream file(name, std::ios::binary);
    boost::archive::binary_oarchive oa(file);
    oa << exactEnergies;
    file.close();
  }

  return ene2;
}

template<typename Wfn>
template<typename Walker>
double SCPT<Wfn>::doSCEnergyCTMCSync(Walker& walk, int& ind, workingArray& work, string& outputFile)
{
  FILE * out;
  if (commrank == 0) {
    out = fopen(outputFile.c_str(), "w");
    fprintf(out, "# 1. iteration     2. weighted_energy     3. residence_time\n");
  }

  double ham = 0., hamSample = 0., ovlp = 0.;
  double numerator = 0., numerator_MPI = 0., numerator_Tot = 0.;
  double deltaT = 0., deltaT_Tot = 0., deltaT_MPI = 0.;

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

  int coeffsIndexCopy = this->coeffsIndex(walk);
  if (coeffsIndexCopy != ind) cout << "ERROR at 1: " << ind << "    " << coeffsIndexCopy << endl;

  // Now, sample the SC energy in this space
  FastHamAndOvlp(walk, ovlp, hamSample, work);

  int iter = 0;
  while (iter < schd.stochasticIterEachSC) {
    double cumovlpRatio = 0.;
    for (int i = 0; i < work.nExcitations; i++) {
      cumovlpRatio += abs(work.ovlpRatio[i]);
      work.ovlpRatio[i] = cumovlpRatio;
    }
    deltaT = 1.0 / (cumovlpRatio);

    numerator = deltaT*hamSample;

#ifndef SERIAL
    MPI_Allreduce(&(numerator), &(numerator_MPI), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&(deltaT), &(deltaT_MPI), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
    numerator_MPI = numerator;
    deltaT_MPI = deltaT;
#endif

    if (commrank == 0) fprintf(out, "%14d    %.12e    %.12e\n", iter, numerator_MPI, deltaT_MPI);

    numerator_Tot += numerator_MPI;
    deltaT_Tot += deltaT_MPI;

    double nextDetRandom = random() * cumovlpRatio;
    int nextDet = std::lower_bound(work.ovlpRatio.begin(), (work.ovlpRatio.begin() + work.nExcitations),
                                   nextDetRandom) - work.ovlpRatio.begin();

    walk.updateWalker(wave.getRef(), wave.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);

    int coeffsIndexCopy = this->coeffsIndex(walk);
    if (coeffsIndexCopy != ind) cout << "ERROR at 2: " << ind << "    " << coeffsIndexCopy << endl;

    FastHamAndOvlp(walk, ovlp, hamSample, work);
    iter++;
  }

  double final_ham = numerator_Tot/deltaT_Tot;
  if (commrank == 0) cout << "ham: " << setprecision(10) << final_ham << endl;
  if (commrank == 0) fclose(out);

  return final_ham;
}

template<typename Wfn>
template<typename Walker>
void SCPT<Wfn>::doSCEnergyExact(Walker& walk, workingArray& work, double& SCHam, double& SCHamVar, int& samplingIters) {

  int firstActive = schd.nciCore;
  int firstVirtual = schd.nciCore + schd.nciAct;

  Determinant dExternal = walk.d;

  // Remove all electrons from the active space
  for (int i = firstActive; i<firstVirtual; i++) {
    dExternal.setoccA(i, false);
    dExternal.setoccB(i, false);
  }

  int nalpha = Determinant::nalpha - dExternal.Nalpha();
  int nbeta = Determinant::nbeta - dExternal.Nbeta();

  // Generate all determinants in the appropriate S_l^k
  vector<Determinant> allDets;
  generateAllDeterminantsActive(allDets, dExternal, schd.nciCore, schd.nciAct, nalpha, nbeta);

  double hamTot = 0., normTot = 0.;
  double largestOverlap = 0.;
  Determinant bestDet;

  int nDets = allDets.size();

  for (int i = 0; i < allDets.size(); i++) {
    wave.initWalker(walk, allDets[i]);
    if (schd.debug) {
      cout << "walker\n" << walk << endl;
    }

    double ovlp = 0., ham = 0.;
    FastHamAndOvlp(walk, ovlp, ham, work);

    hamTot += ovlp * ovlp * ham;
    normTot += ovlp * ovlp;

    if (abs(ovlp) > largestOverlap) {
      largestOverlap = abs(ovlp);
      bestDet = allDets[i];
    }
  }

  if (abs(normTot) > 1.e-12) {
    SCHam = hamTot / normTot;
  } else {
    SCHam = 0.;
  }

  SCHamVar = 0.;
  samplingIters = 0;
}

// Generate an initial determinant with external orbitals orb1 and orb2
// set as appropriate, and all orbitals in the active space unoccupied.
template<typename Wfn>
Determinant SCPT<Wfn>::generateInitDet(int orb1, int orb2) {

  // The inital determinant at this point, taken from
  // Wfn.bestDeterminant, has all core orbitals doubly occupied and
  // all virtual orbitals unoccupied.
  Determinant initDet = this->wave.bestDeterminant;

  int firstActive = schd.nciCore;
  int firstVirtual = schd.nciCore + schd.nciAct;

  // Unset all active orbitals
  for (int i = firstActive; i<firstVirtual; i++) {
    initDet.setoccA(i, false);
    initDet.setoccB(i, false);
  }

  // Set the core and virtual orbitals as appropriate
  for (int i=0; i<2; i++) {
    int orb;

    if (i == 0) {
      orb = orb1;
    } else {
      orb = orb2;
    }

    // orb == -1 indicates that this orbital is not in use (i.e. we
    // only have a single excitation).
    if (orb == -1) continue;

    if (orb < 2*firstActive) {
      if (orb % 2 == 0) {
        initDet.setoccA(orb/2, false);
      } else {
        initDet.setoccB(orb/2, false);
      }
    } else if (orb >= 2*firstVirtual) {
      if (orb % 2 == 0) {
        initDet.setoccA(orb/2, true);
      } else {
        initDet.setoccB(orb/2, true);
      }
    }
  }

  return initDet;
}

template<typename Wfn>
template<typename Walker>
double SCPT<Wfn>::compareStochPerturberEnergy(Walker& walk, int orb1, int orb2, double CASEnergy, int nsamples) {

  // First, calculate the *exact* perturber energy

  int firstActive = schd.nciCore;
  int firstVirtual = schd.nciCore + schd.nciAct;

  Determinant dExternal = walk.d;

  // Set the core and virtual orbitals as appropriate
  for (int i=0; i<2; i++) {
    int orb;

    if (i == 0) {
      orb = orb1;
    } else {
      orb = orb2;
    }

    // orb == -1 indicates that this orbital is not in use (i.e. we
    // only have a single excitation).
    if (orb == -1) continue;

    if (orb < 2*firstActive) {
      if (orb % 2 == 0) {
        dExternal.setoccA(orb/2, false);
      } else {
        dExternal.setoccB(orb/2, false);
      }
    } else if (orb >= 2*firstVirtual) {
      if (orb % 2 == 0) {
        dExternal.setoccA(orb/2, true);
      } else {
        dExternal.setoccB(orb/2, true);
      }
    }
  }

  // Construct a determinant with all active orbitals unoccupied
  // (but core and virtual occupations are the same)
  for (int i = firstActive; i<firstVirtual; i++) {
    dExternal.setoccA(i, false);
    dExternal.setoccB(i, false);
  }

  // Get the number of alpha and beta electrons in the active space
  int nalpha = Determinant::nalpha - dExternal.Nalpha();
  int nbeta = Determinant::nbeta - dExternal.Nbeta();

  // Generate all determinants in the appropriate S_l^k
  vector<Determinant> allDets;
  generateAllDeterminantsActive(allDets, dExternal, schd.nciCore, schd.nciAct, nalpha, nbeta);

  workingArray work;
  double hamTot = 0., normTot = 0.;
  double largestOverlap = 0.;
  Determinant bestDet;

  int nDets = allDets.size();

  for (int i = 0; i < allDets.size(); i++) {
    wave.initWalker(walk, allDets[i]);
    if (schd.debug) {
      cout << "walker\n" << walk << endl;
    }

    double ovlp = 0., ham = 0.;
    FastHamAndOvlp(walk, ovlp, ham, work);

    hamTot += ovlp * ovlp * ham;
    normTot += ovlp * ovlp;

    if (abs(ovlp) > largestOverlap) {
      largestOverlap = abs(ovlp);
      bestDet = allDets[i];
    }
  }

  double perturberEnergy = hamTot / normTot;
  cout << "Exact perturber energy, E_l^k: " << setprecision(12) << perturberEnergy << endl;

  // Now generate stochastic samples
  FILE * pt2_out;
  string pt2OutName = "stoch_samples_";
  pt2OutName.append(to_string(commrank));
  pt2OutName.append(".dat");

  pt2_out = fopen(pt2OutName.c_str(), "w");
  fprintf(pt2_out, "# 1. iter     2. E_l^k              3. E_l^K variance     4. Bias correction     "
                   "5. E_0 - E_l^k         6. 1/(E_0 - E_l^k)   7. niters  8. time\n");

  for (int iter=0; iter<nsamples; iter++) {
    double SCHam = 0., SCHamVar = 0.;
    int samplingIters;

    double timeIn = getTime();

    this->wave.initWalker(walk, bestDet);
    doSCEnergyCTMC(walk, work, SCHam, SCHamVar, samplingIters);

    double timeOut = getTime();

    double eDiff = CASEnergy - SCHam;
    double energySample = 1.0/eDiff;
    double biasCorr = - SCHamVar / pow( CASEnergy - SCHam, 3);

    // Get the external orbs of the perturber, for printing
    string orbString = formatOrbString(orb1, orb2);

    fprintf(pt2_out, "%9d    %.12e    %.12e    %.12e    %.12e    %.12e   %8d   %.4e\n",
            iter, SCHam, SCHamVar, biasCorr, eDiff, energySample, samplingIters, timeOut-timeIn);
    fflush(pt2_out);
  }

  fclose(pt2_out);

}

template<typename Wfn>
template<typename Walker>
double SCPT<Wfn>::doNEVPT2_Deterministic(Walker& walk) {

  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;
  vector<Determinant> allDets;
  generateAllDeterminantsFOIS(allDets, norbs, nalpha, nbeta);

  workingArray work;
  double overlapTot = 0., overlapTotCASCI = 0.;
  VectorXd ham = VectorXd::Zero(coeffs.size()), norm = VectorXd::Zero(coeffs.size());
  double waveEne = 0.;
  //w.printVariables();

  VectorXd normSamples = VectorXd::Zero(coeffs.size()), SCNorm = VectorXd::Zero(coeffs.size());

  vector<Determinant> initDets;
  initDets.resize(numCoeffs, walk.d);
  vector<double> largestCoeffs;
  largestCoeffs.resize(numCoeffs, 0.0);

  for (int i = commrank; i < allDets.size(); i += commsize) {
    wave.initWalker(walk, allDets[i]);
    if (schd.debug) {
      cout << "walker\n" << walk << endl;
    }

    if (!classesUsed[walk.excitation_class]) continue;
    int coeffsIndex = this->coeffsIndex(walk);
    if (coeffsIndex == -1) continue;

    double ovlp = 0., normSample = 0., hamSample = 0., locEne = 0.;
    HamAndOvlp(walk, ovlp, locEne, hamSample, normSample, coeffsIndex, work);

    overlapTot += ovlp * ovlp;
    ham(coeffsIndex) += (ovlp * ovlp) * hamSample;
    norm(coeffsIndex) += (ovlp * ovlp) * normSample;
    waveEne += (ovlp * ovlp) * locEne;

    if (walk.excitation_class == 0) {
      normSamples.setZero();
      HamAndSCNorms(walk, ovlp, hamSample, normSamples, initDets, largestCoeffs, work, true);
      SCNorm += (ovlp * ovlp) * normSamples;
      overlapTotCASCI += ovlp * ovlp;
    }
  }

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(overlapTot), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(waveEne), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, ham.data(), coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, norm.data(), coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, SCNorm.data(), coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(overlapTotCASCI), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  waveEne = waveEne / overlapTot;
  ham = ham / overlapTot;
  norm = norm / overlapTot;
  SCNorm = SCNorm / overlapTotCASCI;

  // Print the exact squared norms
  //if (commrank == 0) {
  //  cout << "Exact norms:" << endl;
  //  int numVirt = norbs - schd.nciCore - schd.nciAct;
  //  int first_virtual = schd.nciCore + schd.nciAct;

  //  cout << "Class AAAV: " << endl;
  //  for (int r=2*first_virtual; r<2*norbs; r++) {
  //    int ind = cumNumCoeffs[1] + r - 2*first_virtual;
  //    cout << "r: " << r << "  norm: " << setprecision(12) << norm(ind) / norm(0) << endl;
  //  }

  //  cout << "Class AAVV: " << endl;
  //  for (int r=2*first_virtual+1; r<2*norbs; r++) {
  //    for (int s=2*first_virtual; s<r; s++) {
  //      int R = r - 2*first_virtual - 1;
  //      int S = s - 2*first_virtual;
  //      int ind = cumNumCoeffs[2] + R*(R+1)/2 + S;
  //      cout << "r: " << r << " s: " << s << " norm: " << setprecision(12) << norm(ind) / norm(0) << endl;
  //    }
  //  }

  //  cout << "Class CAAA: " << endl;
  //  for (int i=0; i<2*schd.nciCore; i++) {
  //    int ind = cumNumCoeffs[3] + i;
  //    cout << "i: " << i << "  norm: " << setprecision(12) << norm(ind) / norm(0) << endl;
  //  }

  //  cout << "Class CAAV: " << endl;
  //  for (int i=0; i<2*schd.nciCore; i++) {
  //    for (int r=2*first_virtual; r<2*norbs; r++) {
  //      int ind = cumNumCoeffs[4] + 2*numVirt*i + (r - 2*schd.nciCore - 2*schd.nciAct);
  //      cout << "i: " << i << " r: " << r << "  norm: " << setprecision(12) << norm(ind) / norm(0) << endl;
  //    }
  //  }

  //  cout << "Class CCAA: " << endl;
  //  for (int i=1; i<2*schd.nciCore; i++) {
  //    for (int j=0; j<i; j++) {
  //      int ind = cumNumCoeffs[6] + (i-1)*i/2 + j;
  //      cout << "i: " << i << " j: " << j << "  norm: " << setprecision(12) << norm(ind) / norm(0) << endl;
  //    }
  //  }
  //}

  std::vector<int> largeNormIndices;
  int counter = 0;
  for (int i = 0; i < coeffs.size(); i++) {
    if (norm(i) > 1.e-16) {
      largeNormIndices.push_back(i);
    }
  }

  Map<VectorXi> largeNormSlice(&largeNormIndices[0], largeNormIndices.size());
  VectorXd largeNorms;
  igl::slice(norm, largeNormSlice, largeNorms);
  VectorXd largeHam;
  igl::slice(ham, largeNormSlice, largeHam);
  VectorXd ene = (largeHam.array() / largeNorms.array()).matrix();
  double ene2 = 0.;
  for (int i = 1; i < largeNorms.size(); i++) {
    ene2 += largeNorms(i) / largeNorms(0) / (ene(0) - ene(i));
  }
  
  if (commrank == 0) {
    cout << "ref energy   " << setprecision(12) << ene(0) << endl;
    cout << "waveEne  " << waveEne << endl;
    cout << "nevpt2 energy  " << ene(0) + ene2 << endl;
  }
}

template<typename Wfn>
double SCPT<Wfn>::get_ccvv_energy() {

  double energy_ccvv = 0.0;

  int norbs = Determinant::norbs;
  int first_virtual = schd.nciCore + schd.nciAct;

  for (int j=1; j<2*schd.nciCore; j++) {
    for (int i=0; i<j; i++) {
      for (int s=2*first_virtual+1; s<2*norbs; s++) {
        for (int r=2*first_virtual; r<s; r++) {
          energy_ccvv -= pow( I2(r, j, s, i) - I2(r, i, s, j), 2) / ( moEne(r/2) + moEne(s/2) - moEne(i/2) - moEne(j/2) );
        }
      }
    }
  }
  return energy_ccvv;
}

template<typename Wfn>
void SCPT<Wfn>::readSpinRDM(Eigen::MatrixXd& oneRDM, Eigen::MatrixXd& twoRDM) {
  // Read a 2-RDM from the spin-RDM text file output by Dice
  // Also construct the 1-RDM at the same time

  int nSpinOrbsAct = 2*schd.nciAct;
  int nPairs = nSpinOrbsAct * nSpinOrbsAct;

  oneRDM = MatrixXd::Zero(nSpinOrbsAct, nSpinOrbsAct);
  twoRDM = MatrixXd::Zero(nPairs, nPairs);

  ifstream RDMFile("spinRDM.0.0.txt");
  string lineStr;
  while (getline(RDMFile, lineStr)) {
    string buf;
    stringstream ss(lineStr);
    vector<string> words;
    while (ss >> buf) words.push_back(buf);

    int a = stoi(words[0]);
    int b = stoi(words[1]);
    int c = stoi(words[2]);
    int d = stoi(words[3]);
    double elem = stod(words[4]);

    int ind1 = a * nSpinOrbsAct + b;
    int ind2 = c * nSpinOrbsAct + d;
    int ind3 = b * nSpinOrbsAct + a;
    int ind4 = d * nSpinOrbsAct + c;

    twoRDM(ind1, ind2) = elem;
    twoRDM(ind3, ind2) = -elem;
    twoRDM(ind1, ind4) = -elem;
    twoRDM(ind3, ind4) = elem;

    if (b == d) oneRDM(a,c) += elem;
    if (b == c) oneRDM(a,d) += -elem;
    if (a == d) oneRDM(b,c) += -elem;
    if (a == c) oneRDM(b,d) += elem;
  }

  int nelec_act = Determinant::nalpha + Determinant::nbeta - 2*schd.nciCore;

  // Normalize the 1-RDM
  for (int a = 0; a < nSpinOrbsAct; a++) {
    for (int b = 0; b < nSpinOrbsAct; b++) {
      oneRDM(a,b) /= nelec_act-1;
    }
  }
}

template<typename Wfn>
void SCPT<Wfn>::calc_AAVV_NormsFromRDMs(Eigen::MatrixXd& twoRDM, Eigen::VectorXd& norms) {

  if (commrank == 0) cout << "Calculating AAVV norms..." << endl;
  double timeIn = getTime();

  int norbs = Determinant::norbs;
  int first_virtual = schd.nciCore + schd.nciAct;

  int nSpinOrbs = 2*norbs;
  int nSpinOrbsCore = 2*schd.nciCore;
  int nSpinOrbsAct = 2*schd.nciAct;
  
  VectorXd normsLocal = 0. * norms;
  size_t numTerms = (nSpinOrbs - 2*first_virtual) * (nSpinOrbs - 2*first_virtual + 1) / 2;
  
  for (int r = 2*first_virtual+1; r < nSpinOrbs; r++) {
    for (int s = 2*first_virtual; s < r; s++) {
      int R = r - 2*first_virtual - 1;
      int S = s - 2*first_virtual;
      if ((R*(R+1)/2+S) % commsize != commrank) continue;
      double norm_rs = 0.0;
      for (int a = nSpinOrbsCore+1; a < 2*first_virtual; a++) {
        for (int b = nSpinOrbsCore; b < a; b++) {
          double int_ab = I2(r,a,s,b) - I2(r,b,s,a);
          for (int c = nSpinOrbsCore+1; c < 2*first_virtual; c++) {
            for (int d = nSpinOrbsCore; d < c; d++) {
              int ind1 = (a - nSpinOrbsCore) * nSpinOrbsAct + (b - nSpinOrbsCore);
              int ind2 = (c - nSpinOrbsCore) * nSpinOrbsAct + (d - nSpinOrbsCore);
              norm_rs += int_ab * twoRDM(ind1,ind2) * (I2(r,c,s,d) - I2(r,d,s,c));
            }
          }
        }
      }
      size_t ind = cumNumCoeffs[2] + R*(R+1)/2 + S;
      //double norm_old = norms(ind);
      normsLocal(ind) = norm_rs;
      //cout << r << "   " << s << "   " << setprecision(12) << norm_old << "   " << norm_rs << endl;
    }
  }

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, normsLocal.data(), normsLocal.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    
  for (int r = 2*first_virtual+1; r < nSpinOrbs; r++) {
    for (int s = 2*first_virtual; s < r; s++) {
      int R = r - 2*first_virtual - 1;
      int S = s - 2*first_virtual;
      size_t ind = cumNumCoeffs[2] + R*(R+1)/2 + S;
      norms(ind) = normsLocal(ind);
    }
  }

  double timeOut = getTime();
  if (commrank == 0) cout << "AAVV norms calculated. Time taken: " << timeOut - timeIn << endl;
}

template<typename Wfn>
void SCPT<Wfn>::calc_CAAV_NormsFromRDMs(Eigen::MatrixXd& oneRDM, Eigen::MatrixXd& twoRDM, Eigen::VectorXd& norms) {

  double timeIn = getTime();
  if (commrank == 0) cout << "Calculating CAAV norms..." << endl;

  int norbs = Determinant::norbs;
  int first_virtual = schd.nciCore + schd.nciAct;

  int nSpinOrbs = 2*norbs;
  int nSpinOrbsCore = 2*schd.nciCore;
  int nSpinVirtOrbs = 2*(norbs - schd.nciCore - schd.nciAct);
  int nSpinOrbsAct = 2*schd.nciAct;
  VectorXd normsLocal = 0. * norms;

  for (int i = commrank; i < nSpinOrbsCore; i+=commsize) {
    for (int r = 2*first_virtual; r < nSpinOrbs; r++) {
      double core_contrib = 0.0;
      for (int j = 0; j < nSpinOrbsCore; j++) {
        core_contrib += I2(i,r,j,j) - I2(i,j,j,r);
      }
      double norm_ir = (I1(i,r) + core_contrib) * (I1(i,r) + core_contrib);

      for (int a = nSpinOrbsCore; a < 2*first_virtual; a++) {
        for (int b = nSpinOrbsCore; b < 2*first_virtual; b++) {
          double int_ab = I2(i,r,b,a) - I2(i,a,b,r);

          norm_ir += 2*int_ab * oneRDM(b-nSpinOrbsCore, a-nSpinOrbsCore) * core_contrib;
          norm_ir += 2*int_ab * oneRDM(b-nSpinOrbsCore, a-nSpinOrbsCore) * I1(r,i);

          for (int c = nSpinOrbsCore; c < 2*first_virtual; c++) {

            norm_ir += int_ab * oneRDM(b-nSpinOrbsCore, c-nSpinOrbsCore) * (I2(r,i,a,c) - I2(r,c,a,i));

            for (int d = nSpinOrbsCore; d < 2*first_virtual; d++) {
              int ind1 = (b - nSpinOrbsCore) * nSpinOrbsAct + (c - nSpinOrbsCore);
              int ind2 = (a - nSpinOrbsCore) * nSpinOrbsAct + (d - nSpinOrbsCore);
              norm_ir += int_ab * twoRDM(ind1,ind2) * (I2(r,i,c,d) - I2(r,d,c,i));
            }
          }
        }
      }
      size_t ind = cumNumCoeffs[4] + nSpinVirtOrbs*i + (r - nSpinOrbsCore - nSpinOrbsAct);
      //double norm_old = norms(ind);
      normsLocal(ind) = norm_ir;
      //cout << i << "   " << r << "   " << setprecision(12) << norm_old << "   " << norm_ir << endl;
    }
  }

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, normsLocal.data(), normsLocal.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  for (int i = 0; i < nSpinOrbsCore; i++) {
    for (int r = 2*first_virtual; r < nSpinOrbs; r++) {
      size_t ind = cumNumCoeffs[4] + nSpinVirtOrbs*i + (r - nSpinOrbsCore - nSpinOrbsAct);
      norms(ind) = normsLocal(ind);
    }
  }

  double timeOut = getTime();
  if (commrank == 0) cout << "CAAV norms calculated. Time taken: " << timeOut - timeIn << endl;
}

template<typename Wfn>
void SCPT<Wfn>::calc_CCAA_NormsFromRDMs(Eigen::MatrixXd& oneRDM, Eigen::MatrixXd& twoRDM, Eigen::VectorXd& norms) {

  double timeIn = getTime();
  if (commrank == 0) cout << "Calculating CCAA norms..." << endl;

  int norbs = Determinant::norbs;
  int first_virtual = schd.nciCore + schd.nciAct;

  int nSpinOrbs = 2*norbs;
  int nSpinOrbsCore = 2*schd.nciCore;
  int nSpinOrbsAct = 2*schd.nciAct;

  int nPairs = nSpinOrbsAct * nSpinOrbsAct;

  // Construct auxiliary 2-RDM for CCAA class
  double *twoRDMAux = new double[(int)pow(nSpinOrbsAct, 4)];

  for (int a = 0; a < nSpinOrbsAct; a++) {
    for (int b = 0; b < nSpinOrbsAct; b++) {
      for (int c = 0; c < nSpinOrbsAct; c++) {
        for (int d = 0; d < nSpinOrbsAct; d++) {
          int ind1 = c * nSpinOrbsAct + d;
          int ind2 = a * nSpinOrbsAct + b;
          int ind3 = ((a*nSpinOrbsAct + b)*nSpinOrbsAct + c)*nSpinOrbsAct + d;
          twoRDMAux[ind3] = twoRDM(ind1, ind2);

          if (b == c) twoRDMAux[ind3] += oneRDM(d,a);
          if (a == d) twoRDMAux[ind3] += oneRDM(c,b);
          if (a == c) twoRDMAux[ind3] += -oneRDM(d,b);
          if (b == d) twoRDMAux[ind3] += -oneRDM(c,a);
          if (b == d && a == c) twoRDMAux[ind3] += 1;
          if (b == c && a == d) twoRDMAux[ind3] += -1;
        }
      }
    }
  }

  for (int i = 1; i < nSpinOrbsCore; i++) {
    for (int j = 0; j < i; j++) {
      double norm_ij = 0.0;
      for (int a = nSpinOrbsCore+1; a < 2*first_virtual; a++) {
        int a_shift = a - nSpinOrbsCore;
        for (int b = nSpinOrbsCore; b < a; b++) {
          double int_ij = I2(j,a,i,b) - I2(j,b,i,a);
          int b_shift = b - nSpinOrbsCore;
          for (int c = nSpinOrbsCore+1; c < 2*first_virtual; c++) {
            int c_shift = c - nSpinOrbsCore;
            for (int d = nSpinOrbsCore; d < c; d++) {
              int d_shift = d - nSpinOrbsCore;
              int ind = ((a_shift*nSpinOrbsAct + b_shift)*nSpinOrbsAct + c_shift)*nSpinOrbsAct + d_shift;
              norm_ij += int_ij * twoRDMAux[ind] * (I2(c,j,d,i) - I2(c,i,d,j));
            }
          }
        }
      }
      int norm_ind = cumNumCoeffs[6] + (i-1)*i/2 + j;
      norms(norm_ind) = norm_ij;
      //cout << i << "   " << j << "   " << setprecision(12) << norm_old << "   " << norm_ij << endl;
    }
  }

  delete []twoRDMAux;

  double timeOut = getTime();
  if (commrank == 0) cout << "CCAA norms calculated. Time taken: " << timeOut - timeIn << endl;
}

template<typename Wfn>
string SCPT<Wfn>::getfileName() const {
  return "scpt"+wave.getfileName();
}

template<typename Wfn>
void SCPT<Wfn>::writeWave()
{
  if (commrank == 0)
    {
char file[5000];
      sprintf(file, (getfileName()+".bkp").c_str() );
std::ofstream outfs(file, std::ios::binary);
boost::archive::binary_oarchive save(outfs);
save << *this;
outfs.close();
    }
}

template<typename Wfn>
void SCPT<Wfn>::readWave()
{
  if (commrank == 0)
    {
char file[5000];
      sprintf(file, (getfileName()+".bkp").c_str() );
std::ifstream infs(file, std::ios::binary);
boost::archive::binary_iarchive load(infs);
load >> *this;
infs.close();
    }
#ifndef SERIAL
  boost::mpi::communicator world;
  boost::mpi::broadcast(world, *this, 0);
#endif
}

// This is a wrapper function which is called during initialization.
// This is where the main NEVPT2 functions are called from.
void initNEVPT_Wrapper()
{
  schd.usingFOIS = true;

  SCPT<SelectedCI> wave;
  SimpleWalker walk;

  wave.initWalker(walk);

  if (schd.deterministic) {
    wave.doNEVPT2_Deterministic(walk);
  }
  else if (schd.exactPerturber) {
    wave.compareStochPerturberEnergy(walk, schd.perturberOrb1, schd.perturberOrb2,
                                     schd.CASEnergy, schd.numSCSamples);
  }
  else {
    bool new_NEVPT = schd.efficientNEVPT || schd.efficientNEVPT_2 || schd.exactE_NEVPT ||
                     schd.NEVPT_readE || schd.NEVPT_writeE;
    if (new_NEVPT) {
      wave.doNEVPT2_CT_Efficient(walk);
    } else {
      wave.doNEVPT2_CT(walk);
      wave.doNEVPT2_CT(walk);
    }
  }

}
