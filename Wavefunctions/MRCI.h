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

#ifndef MRCI_HEADER_H
#define MRCI_HEADER_H
#include <vector>
#include <set>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <utility>
#include <iomanip>
#include "MRCIWalker.h"
#include "CorrelatedWavefunction.h"

#ifndef SERIAL
#include "mpi.h"
#endif

using namespace std;

class oneInt;
class twoInt;
class twoIntHeatBathSHM;


template<typename Corr, typename Reference>
class MRCI
{
 private:
  friend class boost::serialization::access;
  template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
      ar  & wave
	  & coeffs;
    }

 public:
  using CorrType = Corr;
  using ReferenceType = Reference;
 
  VectorXd coeffs;
  CorrelatedWavefunction<Corr, Reference> wave; //reference wavefunction
  workingArray morework;
  

  MRCI()
  {
    //wave = CorrelatedWavefunction<Corr, Reference>();
    wave.readWave();
    
    //cout << "JS vars\n";  wave.printVariables();
    // Resize coeffs
    int numVirt = Determinant::norbs - schd.nciAct;
    int numCoeffs = 2 + 2*numVirt + (2*numVirt * (2*numVirt - 1) / 2);
    coeffs = VectorXd::Zero(numCoeffs);

    //coeffs order: phi0, singly excited (spin orb index), doubly excited (spin orb pair index)

    if (commrank == 0) {
      auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                              std::ref(generator));

      coeffs(0) = -0.5;
      //coeffs(1) = -0.5;
      for (int i=1; i < numCoeffs; i++) {
        coeffs(i) = 0.2*random() - 0.1;
      }
    }

#ifndef SERIAL
  MPI_Bcast(coeffs.data(), coeffs.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    
    char file[5000];
    sprintf(file, "ciCoeffs.txt");
    ifstream ofile(file);
    if (ofile) {
      for (int i = 0; i < coeffs.size(); i++) {
        ofile >> coeffs(i);
      }
    }
    //cout << "ciCoeffs\n" << coeffs << endl;

    
  }

  Reference& getRef() { return wave.getRef(); }
  Corr& getCorr() { return wave.getCorr(); }

  void initWalker(MRCIWalker<Corr, Reference> &walk)  
  {
    walk = MRCIWalker<Corr, Reference>(wave.corr, wave.ref);
  }
  
  void initWalker(MRCIWalker<Corr, Reference> &walk, Determinant &d) 
  {
    walk = MRCIWalker<Corr, Reference>(wave.corr, wave.ref, d);
  }
  
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
  
  //returns the s^(k)_l space index the walker with the given excitedOrbs belongs to
  int coeffsIndex(unordered_set<int> &excitedOrbs) {
    int norbs = Determinant::norbs;
    if (excitedOrbs.size() == 2) {
      int i = *excitedOrbs.begin() - 2*schd.nciAct;
      int j = *(std::next(excitedOrbs.begin())) - 2*schd.nciAct;
      int I = max(i, j) - 1, J = min(i,j);
      int numVirt = norbs - schd.nciAct;
      return 2 + 2*numVirt + I*(I+1)/2 + J;
    }
    else if (excitedOrbs.size() == 1) {
      return *excitedOrbs.begin() - 2*schd.nciAct + 2;
    }
    else if (excitedOrbs.size() == 0) return 0;
    else return -1;
  }

  //this could be expensive
  //returns the s^(k)_l space index the walker with the given excitedOrbs belongs to
  int coeffsIndex(MRCIWalker<Corr, Reference>& walk) {
    int norbs = Determinant::norbs;
    if (walk.excitedSpinOrbs.size() == 2) {
      int i = *walk.excitedSpinOrbs.begin() - 2*schd.nciAct;
      int j = *(std::next(walk.excitedSpinOrbs.begin())) - 2*schd.nciAct;
      int I = max(i, j) - 1, J = min(i,j);
      int numVirt = norbs - schd.nciAct;
      return 2 + 2*numVirt + I*(I+1)/2 + J;
    }
    else if (walk.excitedSpinOrbs.size() == 1) {
      return *walk.excitedSpinOrbs.begin() - 2*schd.nciAct + 2;
    }
    else if (walk.excitedSpinOrbs.size() == 0) return 0;
    else return -1;
  }
  
  double getOverlapFactor(int i, int a, const MRCIWalker<Corr, Reference>& walk, bool doparity) const  
  {
    return 1.;
  }//not used
  
  double getOverlapFactor(int I, int J, int A, int B, const MRCIWalker<Corr, Reference>& walk, bool doparity) const 
  {
    return 1.;
  }//not used
  
  void OverlapWithGradient(MRCIWalker<Corr, Reference> &walk,
			     double &factor,
			     Eigen::VectorXd &grad)
  {
    int coeffsIndex = this->coeffsIndex(walk);
    double ciCoeff = coeffs(coeffsIndex);
    if (coeffsIndex < 0 || ciCoeff == 0) return;
    if (coeffsIndex == 0) {
      double ham = 0., ovlp = 0.;
      morework.setCounterToZero();
      wave.HamAndOvlp(walk.activeWalker, ovlp, ham, morework);
      grad[0] += 1 / (coeffs(0) + coeffs(1) * ham);
      grad[1] += ham / (coeffs(0) + coeffs(1) * ham);
    }
    else {
      int norbs = Determinant::norbs;
      //if (abs(ciCoeff) <= 1.e-8) return;
      grad(coeffsIndex) += 1 / ciCoeff;
    }
  }

  //updates from and to by combining with excitations ex1 and ex2, also updates excitedOrbs
  //used for n -> m
  //ex1 assumed to be nonzero and ex2 = 0 for single excitations
  void combineExcitations(int ex1, int ex2, unordered_set<int> &excitedOrbs, std::array<unordered_set<int>, 2> &from, std::array<unordered_set<int>, 2> &to)
  {
    int norbs = Determinant::norbs;
    int I = ex1 / (2 * norbs), A = ex1 % (2 * norbs);
    if (A >= 2*schd.nciAct) excitedOrbs.insert(A);
    if (I >= 2*schd.nciAct) excitedOrbs.erase(I);
    int i = I / 2, a = A / 2;
    bool sz = I%2;
    auto itOrbsi = to[sz].find(i); 
    auto itHolesa = from[sz].find(a); 
    if (itOrbsi == to[sz].end()) from[sz].insert(i);
    else to[sz].erase(itOrbsi);
    if (itHolesa == from[sz].end()) to[sz].insert(a);
    else from[sz].erase(itHolesa);

    if (ex2 != 0) {
      int J = ex2 / (2 * norbs), B = ex2 % (2 * norbs);
      if (B >= 2*schd.nciAct) excitedOrbs.insert(B);
      if (J >= 2*schd.nciAct) excitedOrbs.erase(J);
      int j = J / 2, b = B / 2;
      sz = J%2;
      auto itOrbsj = to[sz].find(j); 
      auto itHolesb = from[sz].find(b); 
      if (itOrbsj == to[sz].end()) from[sz].insert(j);
      else to[sz].erase(itOrbsj);
      if (itHolesb == from[sz].end()) to[sz].insert(b);
      else from[sz].erase(itHolesb);
    }
  }
  
  //updates from and to by combining with excitations ex1 and ex2
  //used for m -> m'
  //ex1 assumed to be nonzero and ex2 = 0 for single excitations
  void combineExcitations(int ex1, int ex2, std::array<unordered_set<int>, 2> &from, std::array<unordered_set<int>, 2> &to)
  {
    int norbs = Determinant::norbs;
    int I = ex1 / (2 * norbs), A = ex1 % (2 * norbs);
    int i = I / 2, a = A / 2;
    bool sz = I%2;
    auto itOrbsi = to[sz].find(i); 
    auto itHolesa = from[sz].find(a); 
    if (itOrbsi == to[sz].end()) from[sz].insert(i);
    else to[sz].erase(itOrbsi);
    if (itHolesa == from[sz].end()) to[sz].insert(a);
    else from[sz].erase(itHolesa);

    if (ex2 != 0) {
      int J = ex2 / (2 * norbs), B = ex2 % (2 * norbs);
      int j = J / 2, b = B / 2;
      sz = J%2;
      auto itOrbsj = to[sz].find(j); 
      auto itHolesb = from[sz].find(b); 
      if (itOrbsj == to[sz].end()) from[sz].insert(j);
      else to[sz].erase(itOrbsj);
      if (itHolesb == from[sz].end()) to[sz].insert(b);
      else from[sz].erase(itHolesb);
    }
  }
  
  //returns < m | psi > / < n_0 | phi > 
  //the walker n contains n_0 and its helpers
  //from and to are excitations from n_0 to m
  //mEne would be required if the Lanczos term is included
  //this is essentially HamAndOvlp with the reference
  double ovlpRatio(MRCIWalker<Corr, Reference> &n, Determinant &m, int &coeffsIndex, unordered_set<int> &excitedOrbs, std::array<unordered_set<int>, 2> &from, std::array<unordered_set<int>, 2> &to, double mEne, workingArray &work)
  {
    int norbs = Determinant::norbs;
    double ham = 0.; //ovlp sans coeff 
    if (coeffsIndex == 0) {// < m | psi > = c_0 < m | phi > + c_0^(0) < m | H | phi >
      ham += m.Energy(I1, I2, coreE) * wave.getOverlapFactor(n.activeWalker, from, to);
    }
    work.setCounterToZero();
    if (excitedOrbs.size() == 2) {
      generateAllScreenedExcitationsCAS_0h2p(m, schd.epsilon, work, *excitedOrbs.begin(), *std::next(excitedOrbs.begin())); 
    }
    else if (excitedOrbs.size() == 1) {
      generateAllScreenedSingleExcitationsCAS_0h1p(m, schd.epsilon, schd.screen,
                                                   work, *excitedOrbs.begin(), false); 
      generateAllScreenedDoubleExcitationsCAS_0h1p(m, schd.epsilon, work, *excitedOrbs.begin());
    }
    else {
      generateAllScreenedSingleExcitationsCAS_0h0p(m, schd.epsilon, schd.screen,
                                                   work, false);
      generateAllScreenedDoubleExcitationsCAS_0h0p(m, schd.epsilon, work);
    }

    //if (schd.debug) cout << "phi0  d.energy  " << ham / ovlp << endl;
    //loop over all the screened excitations
    //cout << "m  " << walk.d << endl;
    //cout << "eloc excitations" << endl;
    std::array<unordered_set<int>, 2> fromNew = from;
    std::array<unordered_set<int>, 2> toNew = to;
    //cout << "\ngenerating mp's\n\n";
    for (int i=0; i<work.nExcitations; i++) {
      int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
      double tia = work.HijElement[i];
      //cout << "ex1  " << ex1 << "  ex2  " << ex2 << endl; 
    
      int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
      int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
      double parity = 1.;
      Determinant dcopy = m;
      parity *= dcopy.parity(A/2, I/2, I%2);
      //if (A > I) parity *= -1. * dcopy.parity(A/2, I/2, I%2);
      //else parity *= dcopy.parity(A/2, I/2, I%2);
      if (ex2 != 0) {
        dcopy.setocc(I, false);  //move these insisde the next if
        dcopy.setocc(A, true);
        parity *= dcopy.parity(B/2, J/2, J%2);
        //dcopy.setocc(J, false);  //delete these
        //dcopy.setocc(B, true);
        //if (B > J) parity *= -1 * dcopy.parity(B/2, J/2, J%2);
        //else parity *= dcopy.parity(B/2, J/2, J%2);
      }
      //cout << "m'   " << dcopy << endl;
      //dcopy = n.activeWalker.d;
      combineExcitations(ex1, ex2, fromNew, toNew);
      //cout << "from u  ";
      //copy(fromNew[0].begin(), fromNew[0].end(), ostream_iterator<int>(cout, " "));
      //cout << endl << "from d  ";
      //copy(fromNew[1].begin(), fromNew[1].end(), ostream_iterator<int>(cout, " "));
      //cout << endl << "to u  ";
      //copy(toNew[0].begin(), toNew[0].end(), ostream_iterator<int>(cout, " "));
      //cout << endl << "to d  ";
      //copy(toNew[1].begin(), toNew[1].end(), ostream_iterator<int>(cout, " "));
      //cout << endl;
      ham += tia * wave.getOverlapFactor(n.activeWalker, fromNew, toNew) * parity;
      //ovlpRatio += tia * wave.getOverlapFactor(n.activeWalker, fromNew, toNew);
      //cout << "tia  " << tia << "  ovlpRatio0   " << wave.getOverlapFactor(n.activeWalker, dcopy, fromNew, toNew) << "  " << parity << endl << endl;
      //cout << "n.parity  " << n.parity << endl;
      fromNew = from;
      toNew = to;
      //if (schd.debug) cout << ex1 << "  " << ex2 << "  tia  " << tia << "  ovlpRatio  " << ovlpcopy * parity << endl;
      //work.ovlpRatio[i] = ovlp;
    }
    if (coeffsIndex == 0) return coeffs(0) * wave.getOverlapFactor(n.activeWalker, from, to) + coeffs(1) * ham;
    else return coeffs(coeffsIndex) * ham;
    //if (schd.debug) cout << "ham  " << ham << "  ovlp  " << ovlp << endl << endl;
  }

  // not used
  template<typename Walker>
  bool checkWalkerExcitationClass(Walker &walk) {
    return true;
  }

  void HamAndOvlp(MRCIWalker<Corr, Reference> &n,
                  double &ovlp, double &ham, 
                  workingArray& work, bool fillExcitations=true) 
  {
    if (n.excitedSpinOrbs.size() > 2) return;
    //cout << "\nin HamAndOvlp\n";
    int norbs = Determinant::norbs;
    int coeffsIndex = this->coeffsIndex(n);
    //cout << "coeffsIndex   " << coeffsIndex << endl;
    double ciCoeff = coeffs(coeffsIndex);
    double dEne = n.d.Energy(I1, I2, coreE);
    double ovlp0 = wave.Overlap(n.activeWalker); // < n_0 | phi > 
    morework.setCounterToZero();
    double ovlpRatio0 = ovlpRatio(n, n.d, coeffsIndex, n.excitedSpinOrbs, n.excitedHoles, n.excitedOrbs, dEne, morework); // < n | psi > / < n_0 | phi >
    ovlp = ovlpRatio0 * ovlp0; // < n | psi >
    //cout << "ovlp0  " << ovlp0 << "  ovlpRatio0  " << ovlpRatio0 << endl;
    if (ovlp == 0.) return; //maybe not necessary
    ham = dEne * ovlpRatio0;
    work.setCounterToZero();
    generateAllScreenedSingleExcitation(n.d, schd.epsilon, schd.screen,
                                        work, false);
    if (n.excitedSpinOrbs.size() == 0) {
      generateAllScreenedDoubleExcitation(n.d, schd.epsilon, schd.screen,
                                        work, false);
    }
    else {
      generateAllScreenedDoubleExcitationsFOIS(n.d, schd.epsilon, schd.screen,
                                        work, false);
    }
   
    //loop over all the screened excitations
    //cout << "\ngenerating m's\n\n";
    for (int i=0; i<work.nExcitations; i++) {
      double tia = work.HijElement[i];
      int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
      int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
      int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
      double parity = 1.;
      //cout << "ex1  " << ex1 << "  ex2  " << ex2 << endl; 
      unordered_set<int> excitedOrbs = n.excitedSpinOrbs;
      std::array<unordered_set<int> , 2> from = n.excitedHoles;
      std::array<unordered_set<int> , 2> to = n.excitedOrbs;
      combineExcitations(ex1, ex2, excitedOrbs, from, to);
      //cout << "excitedOrbs  ";
      //copy(excitedOrbs.begin(), excitedOrbs.end(), ostream_iterator<int>(cout, " "));
      //cout << endl << "from u  ";
      //copy(from[0].begin(), from[0].end(), ostream_iterator<int>(cout, " "));
      //cout << endl << "from d  ";
      //copy(from[1].begin(), from[1].end(), ostream_iterator<int>(cout, " "));
      //cout << endl << "to u  ";
      //copy(to[0].begin(), to[0].end(), ostream_iterator<int>(cout, " "));
      //cout << endl << "to d  ";
      //copy(to[1].begin(), to[1].end(), ostream_iterator<int>(cout, " "));
      //cout << endl;
      if (excitedOrbs.size() > 2) continue;
      Determinant m = n.d;
      parity *= m.parity(A/2, I/2, I%2);
      double mEne = 0.;
      //if (ex2 == 0) {
      //  mEne = dEne + n.energyIntermediates[A%2][A/2] - n.energyIntermediates[I%2][I/2] 
      //              - (I2.Direct(I/2, A/2) - I2.Exchange(I/2, A/2));
      //}
      //else {
      m.setocc(I, false);
      m.setocc(A, true);
      if (ex2 != 0) {
        parity *= m.parity(B/2, J/2, J%2);
        m.setocc(J, false);
        m.setocc(B, true);
        //bool sameSpin = (I%2 == J%2);
        //mEne = dEne + n.energyIntermediates[A%2][A/2] - n.energyIntermediates[I%2][I/2]
        //            + n.energyIntermediates[B%2][B/2] - n.energyIntermediates[J%2][J/2]
        //            + I2.Direct(A/2, B/2) - sameSpin * I2.Exchange(A/2, B/2)
        //            + I2.Direct(I/2, J/2) - sameSpin * I2.Exchange(I/2, J/2)
        //            - (I2.Direct(I/2, A/2) - I2.Exchange(I/2, A/2))
        //            - (I2.Direct(J/2, B/2) - I2.Exchange(J/2, B/2))
        //            - (I2.Direct(I/2, B/2) - sameSpin * I2.Exchange(I/2, B/2))
        //            - (I2.Direct(J/2, A/2) - sameSpin * I2.Exchange(J/2, A/2));
      } 
      int coeffsCopyIndex = this->coeffsIndex(excitedOrbs);
      //cout << "m  " << m << endl;
      //cout << "m index  " << coeffsCopyIndex << endl;
      morework.setCounterToZero();
      double ovlpRatiom = ovlpRatio(n, m, coeffsCopyIndex, excitedOrbs, from, to, mEne, morework);// < m | psi > / < n_0 | psi >
      work.ovlpRatio[i] = ovlpRatiom / ovlpRatio0;
      ham += parity * tia * ovlpRatiom;
      //ham += tia * ovlpRatiom;
      //cout << "hamSample   " << tia * ovlpRatiom << endl;
      //cout << "ovlpRatiom  " << ovlpRatiom <<  "  tia  " << tia << "  parity  " << parity << endl << endl;
    }
    ham /= ovlpRatio0;
  }

//  template<typename Walker>
//  double calcEnergy(Walker& walk) {
//    
//    //sampling
//    int norbs = Determinant::norbs;
//    auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
//                            std::ref(generator));
//    
//
//    double ovlp = 0., locEne = 0., ene = 0., correctionFactor = 0.;
//    int coeffsIndex = this->coeffsIndex(walk);
//    workingArray work;
//    HamAndOvlp(walk, ovlp, locEne, work);
//
//    int iter = 0;
//    double cumdeltaT = 0.;
//
//    while (iter < schd.stochasticIter) {
//      double cumovlpRatio = 0;
//      for (int i = 0; i < work.nExcitations; i++) {
//        cumovlpRatio += abs(work.ovlpRatio[i]);
//        work.ovlpRatio[i] = cumovlpRatio;
//      }
//      double deltaT = 1.0 / (cumovlpRatio);
//      double nextDetRandom = random() * cumovlpRatio;
//      int nextDet = std::lower_bound(work.ovlpRatio.begin(), (work.ovlpRatio.begin() + work.nExcitations),
//                                     nextDetRandom) - work.ovlpRatio.begin();
//      cumdeltaT += deltaT;
//      double ratio = deltaT / cumdeltaT;
//      ene *= (1 - ratio);
//      ene += ratio * locEne;
//      correctionFactor *= (1 - ratio);
//      if (coeffsIndex == 0) {
//        correctionFactor += ratio;
//      }
//      walk.updateWalker(wave.getRef(), wave.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
//      coeffsIndex = this->coeffsIndex(walk);
//      HamAndOvlp(walk, ovlp, locEne, work); 
//      iter++;
//    }
//  
//    ene *= cumdeltaT;
//    correctionFactor *= cumdeltaT;
//
//#ifndef SERIAL
//  MPI_Allreduce(MPI_IN_PLACE, &(cumdeltaT), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//  MPI_Allreduce(MPI_IN_PLACE, &(ene), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//  MPI_Allreduce(MPI_IN_PLACE, &(correctionFactor), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//#endif
//    
//    cumulativeTime = cumdeltaT;
//    ene /= cumdeltaT;
//    correctionFactor /= cumdeltaT;
//    if (commrank == 0) {
//      cout << "energy of sampling wavefunction   "  << setprecision(12) << ene << endl;
//      cout << "correctionFactor   " << correctionFactor << endl;
//      cout << "MRCI+Q energy = ene + (1 - correctionFactor) * (ene - ene0)" << endl;
//      if (schd.printVars) cout << endl << "ci coeffs\n" << coeffs << endl; 
//    }
//  }
  
  string getfileName() const {
    return "mrci"+wave.getfileName();
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
