/*
  Developed by Sandeep Sharma with contributions from James E. T. Smith and Adam A. Holmes, 2017
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
#ifndef MRCIWalker_HEADER_H
#define MRCIWalker_HEADER_H

#include "Determinants.h"
#include "Walker.h"
#include "integral.h"
#include <boost/serialization/serialization.hpp>
#include <Eigen/Dense>
#include "input.h"
#include <unordered_set>
#include <iterator>

using namespace Eigen;

/**
* Is essentially a single determinant used in the VMC/DMC simulation
* At each step in VMC one need to be able to calculate the following
* quantities
* a. The local energy = <walker|H|Psi>/<walker|Psi>
*
*/

template<typename Corr, typename Reference>
struct MRCIWalker
{
  Determinant d;                      //The current determinant n
  Walker<Corr, Reference> activeWalker;  //n_0
  unordered_set<int> excitedSpinOrbs; //redundant but useful
  std::array<unordered_set<int>, 2> excitedOrbs;     //spatial orbital indices of excited electrons (in virtual orbitals) in d 
  std::array<unordered_set<int>, 2> excitedHoles;    //spatial orbital indices of excited holes w.r.t. activeWalker (in active orbitals) in d 
  std::array<VectorXd, 2> energyIntermediates;       //would only be useful in lanczos
  //double parity;                          //parity between n_0 and n 

  //constructors
  //default
  MRCIWalker() {}
  
  //this is used in determinisitc calculations
  MRCIWalker(Corr &corr, const Reference &ref, Determinant &pd): d(pd) 
  {
    int norbs = Determinant::norbs;
    Determinant activeDet = d;
    excitedOrbs[0].clear();
    excitedOrbs[1].clear();
    excitedSpinOrbs.clear();
    for (int i = schd.nciAct; i < Determinant::norbs; i++) {
      if (d.getoccA(i)) {
        excitedOrbs[0].insert(i);
        excitedSpinOrbs.insert(2*i);
        activeDet.setoccA(i, 0);
      }
      if (d.getoccB(i)) {
        excitedOrbs[1].insert(i);
        excitedSpinOrbs.insert(2*i + 1);
        activeDet.setoccB(i, 0);
      }
    }
   
    vector<int> open;
    vector<int> closed;
    d.getOpenClosed(0, open, closed);
    excitedHoles[0].clear();
    excitedHoles[1].clear();
    for (int i = 0; i < excitedOrbs[0].size(); i++) {
      excitedHoles[0].insert(open[i]);
      activeDet.setoccA(open[i], 1);
    }
    
    open.clear(); closed.clear();
    d.getOpenClosed(1, open, closed);
    for (int i = 0; i < excitedOrbs[1].size(); i++) {
      excitedHoles[1].insert(open[i]);
      activeDet.setoccB(open[i], 1);
    }
    
    activeWalker = Walker<Corr, Reference>(corr, ref, activeDet);

    //parity = 1.;
    //for (int sz = 0; sz < 2; sz++) {//iterate over spins
    //  auto itFrom = excitedHoles[sz].begin();
    //  auto itTo = excitedOrbs[sz].begin();
    //  for (int n = 0; n < excitedHoles[sz].size(); n++) {//iterate over excitations
    //    int i = *itFrom, a = *itTo;
    //    parity *= activeDet.parity(a, i, sz);
    //    activeDet.setocc(i, sz, false);
    //    activeDet.setocc(a, sz, true);
    //    itFrom = std::next(itFrom); itTo = std::next(itTo);
    //  }
    //}

    //open.clear(); closed.clear()
    //d.getOpenClosed(open, closed);
    //energyIntermediates[0]= VectorXd::Zero(norbs);
    //energyIntermediates[1]= VectorXd::Zero(norbs);
    //for (int i = 0; i < norbs; i++) {
    //  for (int j = 0; j < closed.size(); j++) {
    //    energyIntermediates[0][i] += I2.Direct(i, closed[j]/2) + I1(2*i, 2*i);
    //    energyIntermediates[1][i] += I2.Direct(i, closed[j]/2) + I1(2*i, 2*i);
    //    energyIntermediates[closed[j] % 2][i] -= I2.Exchange(i, closed[j]/2);
    //  }
    //}
  }
  
  //this is used in stochastic calculations
  //it reads the best determinant (which should be in the CAS, so no excitedOrbs or holes) from a file
  MRCIWalker(Corr &corr, const Reference &ref) 
  {
    if (commrank == 0) {
      //char file[5000];
      //sprintf(file, "BestDeterminant.txt");
      //std::ifstream ifs(file, std::ios::binary);
      //boost::archive::binary_iarchive load(ifs);
      //load >> d;
      std::vector<Determinant> dets;
      std::vector<double> ci;
      readDeterminants(schd.determinantFile, dets, ci);
      d = dets[0];
    }
#ifndef SERIAL
    MPI_Bcast(&d.reprA, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d.reprB, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    int norbs = Determinant::norbs;
    
    activeWalker = Walker<Corr, Reference>(corr, ref, d);
    
    excitedOrbs[0].clear();
    excitedOrbs[1].clear();
    excitedHoles[0].clear();
    excitedHoles[1].clear();
    excitedSpinOrbs.clear();
    
    //parity = 1.;
    //vector<int> open;
    //vector<int> closed;
    //d.getOpenClosed(open, closed);
    //energyIntermediates[0]= VectorXd::Zero(norbs);
    //energyIntermediates[1]= VectorXd::Zero(norbs);
    //for (int i = 0; i < norbs; i++) {
    //  for (int j = 0; j < closed.size(); j++) {
    //    energyIntermediates[0][i] += I2.Direct(i, closed[j]/2) + I1(2*i, 2*i);
    //    energyIntermediates[1][i] += I2.Direct(i, closed[j]/2) + I1(2*i, 2*i);
    //    energyIntermediates[closed[j] % 2][i] -= I2.Exchange(i, closed[j]/2);
    //  }
    //}
  }

  //MRCIWalker(const MRCIWalker &w): d(w.d), excitedOrbs(w.excitedOrbs), excitedHoles(w.excitedHoles) {}

  Determinant getDet() { return d; }

  void update(int i, int a, bool sz, const Reference &ref, const Corr &corr) { return; }//to be defined for metropolis
  
  void updateEnergyIntermediate(const oneInt& I1, const twoInt& I2, int I, int A) 
  {
    int norbs = Determinant::norbs;
    for (int n = 0; n < norbs; n++) {
      energyIntermediates[0][n] += (I2.Direct(n, A/2) - I2.Direct(n, I/2));
      energyIntermediates[1][n] += (I2.Direct(n, A/2) - I2.Direct(n, I/2));
      energyIntermediates[I%2][n] -= (I2.Exchange(n, A/2) - I2.Exchange(n, I/2));
    }
  }
  
  void updateEnergyIntermediate(const oneInt& I1, const twoInt& I2, int I, int A, int J, int B)
  {
    int norbs = Determinant::norbs;
    for (int n = 0; n < norbs; n++) {
      energyIntermediates[0][n] += (I2.Direct(n, A/2) - I2.Direct(n, I/2) + I2.Direct(n, B/2) - I2.Direct(n, J/2));
      energyIntermediates[1][n] += (I2.Direct(n, A/2) - I2.Direct(n, I/2) + I2.Direct(n, B/2) - I2.Direct(n, J/2));
      energyIntermediates[I%2][n] -= (I2.Exchange(n, A/2) - I2.Exchange(n, I/2));
      energyIntermediates[J%2][n] -= (I2.Exchange(n, B/2) - I2.Exchange(n, J/2));
    }
  }

  //i and a are spatial orbital indices
  void updateDet(bool sz, int i, int a)
  {
    d.setocc(i, sz, false);
    d.setocc(a, sz, true);
  }

  //this is used for n -> n' MC updates
  //assumes valid excitations
  //the energyIntermediates should only be updated for outer walker updates
  void updateWalker(const Reference &ref, Corr &corr, int ex1, int ex2, bool updateIntermediates = true)
  {
    int norbs = Determinant::norbs;

    //spatial orb excitations for n -> n'
    std::array<std::vector<int>, 2> from, to;
    from[0].clear(); from[1].clear(); to[0].clear(); to[1].clear();
    int I = ex1 / (2 * norbs), A = ex1 % (2 * norbs); 
    from[I%2].push_back(I/2);
    to[I%2].push_back(A/2);
    //if (ex2 == 0) updateEnergyIntermediate(I1, I2, I, A);
    //else {
    if (ex2 != 0) {
      int J = ex2 / (2 * norbs), B = ex2 % (2 * norbs);
      from[J%2].push_back(J/2);
      to[J%2].push_back(B/2);
      //updateEnergyIntermediate(I1, I2, I, A, J, B);
    } 
    
    //spatial orb excitations for n0 -> n0'
    //std::array<std::vector<int>, 2> fromAct, toAct;
    //fromAct[0].clear(); fromAct[1].clear(); toAct[0].clear(); toAct[1].clear();
    std::vector<int> excAct;
    excAct.clear();

    for (int sz = 0; sz < 2; sz++) {// sz = 0, 1
      for (int n = 0; n < from[sz].size(); n++) {//loop over excitations
        int i = from[sz][n], a = to[sz][n];
        updateDet(sz, i, a);
        auto itOrbsi = excitedOrbs[sz].find(i); 
        auto itHolesa = excitedHoles[sz].find(a); 
        if (i < schd.nciAct) {//act ->
          if (a >= schd.nciAct) {//act -> virt, n0' = n0
            excitedOrbs[sz].insert(a);
            excitedHoles[sz].insert(i);
            excitedSpinOrbs.insert(2*a + sz);
          }
          else {//internal excitation, act -> act
            if (itHolesa == excitedHoles[sz].end()) {//no changes to excitedOrbs or holes
              //fromAct[sz].push_back(i); 
              //toAct[sz].push_back(a);
              int exc = (2 * norbs) * (2 * i + sz) + (2 * a + sz);
              excAct.push_back(exc);
            }
            else {//a is an excitedHole, n0' = n0
              excitedHoles[sz].erase(itHolesa);
              excitedHoles[sz].insert(i);
            }
          }
        }
        else {//virt ->
          if (a >= schd.nciAct) {//virt -> virt, n0' = n0
            excitedOrbs[sz].erase(itOrbsi);
            excitedOrbs[sz].insert(a);
            excitedSpinOrbs.erase(2*i + sz);
            excitedSpinOrbs.insert(2*a + sz);
          }
          else {//external excitation, virt -> act
            excitedOrbs[sz].erase(itOrbsi);
            excitedSpinOrbs.erase(2*i + sz);
            if (itHolesa == excitedHoles[sz].end()) {//one hole needs to be removed 
              int hole = *excitedHoles[sz].begin();
              excitedHoles[sz].erase(excitedHoles[sz].begin());
              //fromAct[sz].push_back(hole);
              //toAct[sz].push_back(a);
              int exc = (2 * norbs) * (2 * hole + sz) + (2 * a + sz);
              excAct.push_back(exc);
            }
            else {//a is an excited hole, n0' = n0
              excitedHoles[sz].erase(itHolesa);
            }
          }
        }
      }
    }
    
    if (excAct.size() == 1) activeWalker.updateWalker(ref, corr, excAct[0], 0); 
    else if (excAct.size() == 2) activeWalker.updateWalker(ref, corr, excAct[0], excAct[1]);
  }
  
  void exciteWalker(const Reference &ref, const Corr &corr, int excite1, int excite2, int norbs) { return; };//not used
  
  bool operator<(const MRCIWalker &w) const
  {
    return d < w.d;
  }

  bool operator==(const MRCIWalker &w) const
  {
    return d == w.d;
  }

  friend ostream& operator<<(ostream& os, const MRCIWalker& w) {
    os << w.d << endl;
    //os << "excited Orbs   " << w.excitedSpinOrbs << endl;
    os << "excitedSpinOrbs   ";
    copy(w.excitedSpinOrbs.begin(), w.excitedSpinOrbs.end(), ostream_iterator<int>(os, " "));
    os << endl << "excitedOrbs u  ";
    copy(w.excitedOrbs[0].begin(), w.excitedOrbs[0].end(), ostream_iterator<int>(os, " "));
    os << endl << "excitedOrbs d  ";
    copy(w.excitedOrbs[1].begin(), w.excitedOrbs[1].end(), ostream_iterator<int>(os, " "));
    os << endl << "excitedHoles u  ";
    copy(w.excitedHoles[0].begin(), w.excitedHoles[0].end(), ostream_iterator<int>(os, " "));
    os << endl << "excitedHoles d  ";
    copy(w.excitedHoles[1].begin(), w.excitedHoles[1].end(), ostream_iterator<int>(os, " "));
    os << endl << "activeWalker\n" << w.activeWalker << endl;
    return os;
  }
  //template <typename Wfn>
  //void exciteTo(Wfn& w, Determinant& dcopy) {
  //  d = dcopy;
  //}
  
};

#endif
