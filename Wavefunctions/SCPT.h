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

#ifndef SCPT_HEADER_H
#define SCPT_HEADER_H
#include <vector>
#include <set>
#include "Determinants.h"
#include "workingArray.h"
#include "excitationOperators.h"
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <Eigen/Eigenvalues>
#include <utility>
#include <iomanip>

#ifndef SERIAL
#include "mpi.h"
#endif

using namespace std;

class oneInt;
class twoInt;
class twoIntHeatBathSHM;


template<typename Wfn>
class SCPT
{
 private:
  friend class boost::serialization::access;
  template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
      ar  & wave
	& coeffs
    & moEne;
    }

 public:
  VectorXd coeffs;
  VectorXd moEne;
  Wfn wave; //reference wavefunction
  workingArray morework;

  double ovlp_current;
  
  // a list of the excitation classes being considered stochastically
  vector<int> classesUsed;
  // a list of the excitation classes being considered deterministically
  vector<int> classesUsedDeterm;
  // the total number of excitation classes (including the CAS itself, labelled as 0)
  static const int NUM_EXCIT_CLASSES = 9;
  // the number of coefficients in each excitation class
  int numCoeffsPerClass[NUM_EXCIT_CLASSES];
  // the cumulative sum of numCoeffsPerClass
  int cumNumCoeffs[NUM_EXCIT_CLASSES];
  // the total number of strongly contracted states (including the CASCI space itself)
  int numCoeffs;

  unordered_map<std::array<int,3>, int, boost::hash<std::array<int,3>> > class_1h2p_ind;
  unordered_map<std::array<int,3>, int, boost::hash<std::array<int,3>> > class_2h1p_ind;
  unordered_map<std::array<int,4>, int, boost::hash<std::array<int,4>> > class_2h2p_ind;

  SCPT()
  {
    wave.readWave();

    // Find which excitation classes are being considered. The classes are
    // labelled by integers from 0 to 8, and defined in SimpleWalker.h
    if (schd.nciCore == 0) {
      classesUsed.push_back(0);
      classesUsed.push_back(1);
      classesUsed.push_back(2);
    } else {
      classesUsed.push_back(0);
      classesUsed.push_back(1);
      //classesUsed.push_back(2);
      //classesUsed.push_back(3);
      //classesUsed.push_back(4);
      //classesUsed.push_back(5);
      //classesUsed.push_back(6);
      //classesUsed.push_back(7);
      //if (!schd.determCCVV)
      //  classesUsed.push_back(8);
      //classesUsed.push_back(8);

      // Classes to be treated by the deterministic formulas, rather than by
      // stochstic sampling
      //if (schd.determCCVV)
      //  classesUsedDeterm.push_back(8);
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
    createClassIndMap(numCoeffsPerClass[5], numCoeffsPerClass[7],  numCoeffsPerClass[8]);

    cumNumCoeffs[0] = 0;
    for (int i = 1; i < 9; i++)
    {
      // If the previous class (labelled i-1) is being used, add it to the
      // cumulative counter.
      if (std::find(classesUsed.begin(), classesUsed.end(), i-1) != classesUsed.end()) {
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
      if (std::find(classesUsed.begin(), classesUsed.end(), i) != classesUsed.end()) numCoeffs += numCoeffsPerClass[i];
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

  void createClassIndMap(int& numStates_1h2p, int& numStates_2h1p, int& numStates_2h2p) {
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

  typename Wfn::ReferenceType& getRef() { return wave.getRef(); }
  typename Wfn::CorrType& getCorr() { return wave.getCorr(); }

  template<typename Walker>
  void initWalker(Walker& walk) {
    this->wave.initWalker(walk);
  }
  
  template<typename Walker>
  void initWalker(Walker& walk, Determinant& d) {
    this->wave.initWalker(walk, d);
  }
  
  //void initWalker(Walker& walk) {
  //  this->wave.initWalker(walk);
  //}

  void getVariables(VectorXd& vars) {
    vars = coeffs;
  }

  void printVariables() {
    cout << "ci coeffs\n" << coeffs << endl;
  }

  void updateVariables(VectorXd& vars) {
    coeffs = vars;
  }

  long getNumVariables() {
    return coeffs.size();
  }

  template<typename Walker>
  int coeffsIndex(Walker& walk) {
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

      //int a = *walk.excitedOrbs.begin() - 2*schd.nciCore - 2*schd.nciAct;
      //int b = *(std::next(walk.excitedOrbs.begin())) - 2*schd.nciCore - 2*schd.nciAct;
      //int A = max(a,b) - 1, B = min(a,b);
      int a = *walk.excitedOrbs.begin();
      int b = *(std::next(walk.excitedOrbs.begin()));
      int A = max(a,b), B = min(a,b);

      // the number of *spatial* virtual orbitals
      //int numVirt = norbs - schd.nciCore - schd.nciAct;
      // Number of unique pairs of virtual orbitals
      //int numVirtPairs = 2*numVirt * (2*numVirt - 1) / 2;

      //return cumNumCoeffs[5] + numVirtPairs*i + A*(A+1)/2 + B;
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
      //int I = max(i,j) - 1, J = min(i,j);
      int I = max(i,j), J = min(i,j);

      //int a = *walk.excitedOrbs.begin() - 2*schd.nciCore - 2*schd.nciAct;
      int a = *walk.excitedOrbs.begin();

      // the number of *spatial* virtual orbitals
      //int numVirt = norbs - schd.nciCore - schd.nciAct;

      //return cumNumCoeffs[7] + (2*numVirt)*(I*(I+1)/2 + J) + a;
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
  
  template<typename Walker>
  double getOverlapFactor(int i, int a, const Walker& walk, bool doparity) const  
  {
    return 1.;
  }//not used
  
  template<typename Walker>
  double getOverlapFactor(int I, int J, int A, int B, const Walker& walk, bool doparity) const 
  {
    return 1.;
  }//not used

  // not implemented yet
  template<typename Walker>
  bool checkWalkerExcitationClass(Walker &walk) {
    if (std::find(classesUsed.begin(), classesUsed.end(), walk.excitation_class) == classesUsed.end()) return false;
    int coeffsIndex = this->coeffsIndex(walk);
    if (coeffsIndex == -1)
      return false;
    else
      return true;
  }
  
  //ham is a sample of the diagonal element of the Dyall ham
  template<typename Walker>
  void HamAndOvlp(Walker &walk,
                  double &ovlp, double &locEne, double &ham, double &norm, int coeffsIndex, 
                  workingArray& work, bool fillExcitations=true) 
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
    generateAllScreenedSingleExcitationsDyall(walk.d, dAct, schd.epsilon, schd.screen, work, false);
    generateAllScreenedDoubleExcitationsDyall(walk.d, schd.epsilon, schd.screen, work, false);

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
      if (std::find(classesUsed.begin(), classesUsed.end(), walkCopy.excitation_class) == classesUsed.end()) continue;
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
      //cout << endl;
    }
    //cout << endl << "n ham  " << ham << "  norm  " << norm << endl << endl;
  }
  
  //ham is a sample of the diagonal element of the Dyall ham
  template<typename Walker>
  void FastHamAndOvlp(Walker &walk, double &ovlp, double &ham, workingArray& work, bool fillExcitations=true)
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

  template<typename Walker>
  void HamAndSCNorms(Walker &walk, double &ovlp, double &ham, VectorXd &locSCNorms, workingArray& work)
  {
    int norbs = Determinant::norbs;
    double parity, tia;

    morework.setCounterToZero();
    double ovlp0, ham0;

    // Get the WF overlap with the walker, ovlp
    wave.HamAndOvlp(walk, ovlp, ham0, morework, true);
    ovlp_current = ovlp;

    if (ovlp == 0.) return;
    ham = walk.d.Energy(I1, I2, coreE);

    // Generate all screened excitations
    work.setCounterToZero();
    generateAllScreenedSingleExcitation(walk.d, schd.epsilon, schd.screen, work, false);
    generateAllScreenedDoubleExcitation(walk.d, schd.epsilon, schd.screen, work, false);

    int nExcitationsCASCI = 0;

    // loop over all the screened excitations
    for (int i=0; i<work.nExcitations; i++) {
      tia = work.HijElement[i];

      int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
      int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
      int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;

      auto walkCopy = walk;
      parity = 1.;
      Determinant dcopy = walkCopy.d;
      walkCopy.updateWalker(wave.getRef(), wave.getCorr(), work.excitation1[i], work.excitation2[i], false);

      parity *= dcopy.parity(A/2, I/2, I%2);
      if (ex2 != 0) {
        dcopy.setocc(I, false);
        dcopy.setocc(A, true);
        parity *= dcopy.parity(B/2, J/2, J%2);
      }

      work.ovlpRatio[i] = 0.0;
      if (walkCopy.excitation_class == 0) {
        // For the CTMC algorithm (which is performed within the CASCI space, when
        // calculating the SC norms), we need the ratio of overlaps for connected
        // determinants within the CASCI space. Store these in the work array, and
        // override other excitations, which we won't need any more.
        double ovlpRatio = wave.Overlap(walkCopy.d) / ovlp;
        work.excitation1[nExcitationsCASCI] = work.excitation1[i];
        work.excitation2[nExcitationsCASCI] = work.excitation2[i];
        work.ovlpRatio[nExcitationsCASCI] = ovlpRatio;
        ham += parity * tia * ovlpRatio;
        nExcitationsCASCI += 1;
        continue;
      } else if (std::find(classesUsed.begin(), classesUsed.end(), walkCopy.excitation_class) == classesUsed.end()) {
        // Is this excitation class being used? If not, then move to the next excitation
        continue;
      }
      int coeffsCopyIndex = this->coeffsIndex(walkCopy);
      if (coeffsCopyIndex == -1) continue;

      morework.setCounterToZero();
      wave.HamAndOvlp(walkCopy, ovlp0, ham0, morework, false);
      locSCNorms(coeffsCopyIndex) += parity * tia * ham0 / ovlp;
    }

    // For the CTMC algorithm, only need excitations within the CASCI space.
    // Update the number of excitations to reflect this
    work.nExcitations = nExcitationsCASCI;
  }

  template<typename Walker>
  double doNEVPT2_CT(Walker& walk) {

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
    int printMod = schd.stochasticIter / 5;

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
      if (std::find(classesUsed.begin(), classesUsed.end(), walk.excitation_class) == classesUsed.end()) continue;
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
      if (!classesUsedDeterm.empty()) {
        double energy_ccvv = 0.0;
        if (std::find(classesUsedDeterm.begin(), classesUsedDeterm.end(), 8) != classesUsedDeterm.end()) {
          energy_ccvv = get_ccvv_energy();
          cout << "deterministic CCVV energy:  " << energy_ccvv << endl;
        }
        cout << "total nevpt2 energy:  " << ene(0) + ene2 + energy_ccvv << endl;
      }

      if (schd.printVars) cout << endl << "ci coeffs\n" << coeffs << endl; 
    }
  }

  template<typename Walker>
  void findInitDets(Walker& walk, vector<pair<int,int>>& initDets, vector<double>& largestCoeffs, workingArray& work) {

    int norbs = Determinant::norbs;
    double ovlp0, ham0;

    // Generate all screened excitations
    work.setCounterToZero();
    generateAllScreenedSingleExcitation(walk.d, schd.epsilon, schd.screen, work, false);
    generateAllScreenedDoubleExcitation(walk.d, schd.epsilon, schd.screen, work, false);

    // loop over all the screened excitations
    for (int i=0; i<work.nExcitations; i++) {
      auto walkCopy = walk;
      walkCopy.updateWalker(wave.getRef(), wave.getCorr(), work.excitation1[i], work.excitation2[i], false);

      if (walkCopy.excitation_class == 0) {
        continue;
      } else if (std::find(classesUsed.begin(), classesUsed.end(), walkCopy.excitation_class) == classesUsed.end()) {
        // Is this excitation class being used? If not, then move to the next excitation
        continue;
      }
      int ind = this->coeffsIndex(walkCopy);
      if (ind == -1) continue;

      morework.setCounterToZero();
      wave.HamAndOvlp(walkCopy, ovlp0, ham0, morework, false);
      if (abs(ham0) > largestCoeffs[ind]) {
        initDets[ind] = make_pair(work.excitation1[i], work.excitation2[i]);
        largestCoeffs[ind] = abs(ham0);
      }
    }
  }
  
  template<typename Walker>
  double doNEVPT2_CT_Efficient(Walker& walk) {

    int norbs = Determinant::norbs;
    auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

    auto walkInit = walk;

    double ovlp = 0., hamSample = 0., energyCASCI = 0.;
    VectorXd locSCNormSamples = VectorXd::Zero(coeffs.size()), SCNorm = VectorXd::Zero(coeffs.size());
    workingArray work;

    HamAndSCNorms(walk, ovlp, hamSample, locSCNormSamples, work);

    int iter = 0;
    double cumdeltaT = 0.;
    int printMod = schd.stochasticIter / 5;

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
      energyCASCI *= (1 - ratio);
      energyCASCI += ratio * hamSample;
      SCNorm *= (1 - ratio);
      SCNorm += ratio * locSCNormSamples;

      walk.updateWalker(wave.getRef(), wave.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);

      if (walk.excitation_class != 0) cout << "ERROR: walker not in CASCI space!" << endl;

      locSCNormSamples.setZero();
      HamAndSCNorms(walk, ovlp, hamSample, locSCNormSamples, work);

      iter++;
      if (commrank == 0 && iter % printMod == 1) cout << "iter  " << iter << "  t  " << setprecision(4) << getTime() - startofCalc << endl; 
    }

    energyCASCI *= cumdeltaT;
    SCNorm *= cumdeltaT;

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(cumdeltaT), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(energyCASCI), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, SCNorm.data(), coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    energyCASCI /= cumdeltaT;
    SCNorm /= cumdeltaT;

    // Next we calculate the SC state energies, for the SC states with norms
    // above a certain threshold

    int first_virtual = schd.nciCore + schd.nciAct;

    // Find appropriate initial walkers for each SC sector
    pair<int,int> zero_excitation(0,0);
    vector<pair<int,int>> initDets;
    initDets.resize(numCoeffs, zero_excitation);
    vector<double> largestCoeffs;
    largestCoeffs.resize(numCoeffs, 0.0);

    findInitDets(walkInit, initDets, largestCoeffs, work);

    double ene2 = 0.;

    // Class 1 (0 holes, 1 particle)
    for (int r=2*first_virtual; r<2*norbs; r++) {
      if (commrank == 0) cout << "r: " << r << endl;

      int ind = cumNumCoeffs[1] + r - 2*first_virtual;
      if (SCNorm(ind) > schd.overlapCutoff) {

        if (largestCoeffs[ind] == 0.0) {
          cout << "Error: No initial determinant found. " << r << endl;
        }

        double SCHam = doSCEnergyCTMC(walkInit, ind, initDets[ind], work);
        ene2 += SCNorm(ind) / (energyCASCI - SCHam);
      }
    }

    if (commrank == 0) {
      cout << "PT2 energy:  " << setprecision(10) << ene2 << endl;
      cout << "stochastic nevpt2 energy:  " << setprecision(10) << energyCASCI + ene2 << endl;

      //cout << "CASCI energy: " << setprecision(10) << energyCASCI << endl;

      // Get the class 8 norms exactly
      //int ind;
      //double norm;
      //VectorXd SCNormExact = VectorXd::Zero(coeffs.size());

      //for (int j=1; j<2*schd.nciCore; j++) {
      //  for (int i=0; i<j; i++) {
      //    for (int s=2*first_virtual+1; s<2*norbs; s++) {
      //      for (int r=2*first_virtual; r<s; r++) {
      //        std::array<int,4> inds = {s, r, j, i};
      //        auto it1 = class_2h2p_ind.find(inds);
      //        if (it1 != class_2h2p_ind.end()) {
      //          ind = 1 + it1->second;
      //          SCNormExact(ind) = pow( I2(r, j, s, i) - I2(r, i, s, j), 2);
      //        }
      //      }
      //    }
      //  }
      //}
    }

  }

  template<typename Walker>
  double doSCEnergyCTMC(Walker& walkInit, int& ind, pair<int,int>& excitations, workingArray& work)
  {
    double ham = 0., ovlp = 0., hamSample = 0.;
    auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

    auto walk = walkInit;
    int ex1 = excitations.first;
    int ex2 = excitations.second;
    walk.updateWalker(wave.getRef(), wave.getCorr(), ex1, ex2, false);

    int coeffsIndexCopy = this->coeffsIndex(walk);
    if (coeffsIndexCopy != ind) cout << "ERROR at 1: " << ind << "    " << coeffsIndexCopy << endl;

    // Now, sample the SC energy in this space
    FastHamAndOvlp(walk, ovlp, hamSample, work);

    int iter = 0;
    double cumdeltaT = 0.;

    while (iter < 100) {
      double cumovlpRatio = 0.;
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
      ham *= (1 - ratio);
      ham += ratio * hamSample;

      walk.updateWalker(wave.getRef(), wave.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);

      int coeffsIndexCopy = this->coeffsIndex(walk);
      if (coeffsIndexCopy != ind) cout << "ERROR at 2: " << ind << "    " << coeffsIndexCopy << endl;

      FastHamAndOvlp(walk, ovlp, hamSample, work);
      iter++;
    }

    ham *= cumdeltaT;
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(ham), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(cumdeltaT), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    ham /= cumdeltaT;
    if (commrank == 0) cout << "ham: " << setprecision(10) << ham << endl;

    return ham;
  }

  template<typename Walker>
  double doNEVPT2_Deterministic(Walker& walk) {

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

    VectorXd locSCNormSamples = VectorXd::Zero(coeffs.size()), SCNorm = VectorXd::Zero(coeffs.size());

    for (int i = commrank; i < allDets.size(); i += commsize) {
      wave.initWalker(walk, allDets[i]);
      if (schd.debug) {
        cout << "walker\n" << walk << endl;
      }

      if (std::find(classesUsed.begin(), classesUsed.end(), walk.excitation_class) == classesUsed.end()) continue;
      int coeffsIndex = this->coeffsIndex(walk);
      if (coeffsIndex == -1) continue;

      double ovlp = 0., normSample = 0., hamSample = 0., locEne = 0.;
      HamAndOvlp(walk, ovlp, locEne, hamSample, normSample, coeffsIndex, work);

      overlapTot += ovlp * ovlp;
      ham(coeffsIndex) += (ovlp * ovlp) * hamSample;
      norm(coeffsIndex) += (ovlp * ovlp) * normSample;
      waveEne += (ovlp * ovlp) * locEne;

      if (walk.excitation_class == 0) {
        locSCNormSamples.setZero();
        HamAndSCNorms(walk, ovlp, hamSample, locSCNormSamples, work);
        SCNorm += (ovlp * ovlp) * locSCNormSamples;
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

    if (commrank == 0) {
      cout << "Old and new norms:" << endl;
      for (int i = 1; i < norm.size(); i++) {
        cout << i << "    " << setprecision(12) << norm(i) / norm(0) << "    " << SCNorm(i) << endl;
      }
    }

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
      cout << "nevpt2 energy  " << ene(0) + ene2 << endl;
      cout << "waveEne  " << waveEne << endl;
    }
  }

  double get_ccvv_energy() {

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

  string getfileName() const {
    return "scci"+wave.getfileName();
  }

  void writeWave()
  {
    if (commrank == 0)
      {
	char file[5000];
        sprintf(file, (getfileName()+".bkp").c_str() );
	//sprintf (file, "wave.bkp" , schd.prefix[0].c_str() );
	//sprintf(file, "lanczoscpswave.bkp");
	std::ofstream outfs(file, std::ios::binary);
	boost::archive::binary_oarchive save(outfs);
	save << *this;
	outfs.close();
      }
  }

  void readWave()
  {
    if (commrank == 0)
      {
	char file[5000];
        sprintf(file, (getfileName()+".bkp").c_str() );
	//sprintf (file, "wave.bkp" , schd.prefix[0].c_str() );
	//sprintf(file, "lanczoscpswave.bkp");
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
};
#endif
