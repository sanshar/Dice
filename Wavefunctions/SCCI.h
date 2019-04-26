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

#ifndef SCCI_HEADER_H
#define SCCI_HEADER_H
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
class SCCI
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
  VectorXd coeffs;
  Wfn wave; //reference wavefunction
  workingArray morework;

  SCCI()
  {
    wave.readWave();
    int numVirt = Determinant::norbs - schd.nciAct;
    //coeffs order: phi, singly excited (spin orb index), double excited (spin orb pair index)
    
    coeffs = VectorXd::Random(1 + 2*numVirt + (2*numVirt * (2*numVirt - 1) / 2)) / 10;
    coeffs(0) = -0.5;
    char file[5000];
    sprintf(file, "ciCoeffs.txt");
    ifstream ofile(file);
    if (ofile) {
      for (int i = 0; i < coeffs.size(); i++) {
        ofile >> coeffs(i);
      }
    }
  }

  typename Wfn::ReferenceType& getRef() { return wave.getRef(); }
  typename Wfn::CorrType& getCorr() { return wave.getCorr(); }

  template<typename Walker>
  void initWalker(Walker& walk) {
    this->wave.initWalker(walk);
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

  template<typename Walker>
  int coeffsIndex(Walker& walk) {
    int norbs = Determinant::norbs;
    if (walk.excitedOrbs.size() == 2) {
      int i = *walk.excitedOrbs.begin() - 2*schd.nciAct;
      int j = *(std::next(walk.excitedOrbs.begin())) - 2*schd.nciAct;
      int I = max(i, j) - 1, J = min(i,j);
      int numVirt = norbs - schd.nciAct;
      return 1 + 2*numVirt + I*(I+1)/2 + J;
    }
    else if (walk.excitedOrbs.size() == 1) {
      return *walk.excitedOrbs.begin() - 2*schd.nciAct + 1;
    }
    else if (walk.excitedOrbs.size() == 0) return 0;
    else return -1;
  }

  //this calculates <n|psi^k_l>, where k and l depend on the walker
  //<n|psi^k_l> = <h|H|phi> for singly and doubly excited walkers outside as, = <n|phi> for walkers inside as, = 0 otherwise
  //template<typename Walker>
  //double Overlap(Walker& walk) {
  //  if (walk.excitedOrbs.size() == 0) return wave.Overlap(walk);
  //  else if (walk.excitedOrbs.size() > 2) return 0.;
  //  double ham = 0., ovlp = 0.; 
  //  wave.HamAndOvlp(walk, ovlp, ham, morework, false) {
  //  return ham;
  //}

  //hamSample = <psi^k_l|n>/<psi|n> * <n|H|psi^k'_l'>/<n|psi>, ovlp = <n|psi>
  template<typename Walker>
  void HamAndOvlp(Walker &walk,
                  double &ovlp, double &normSample, double &locEne, VectorXd& hamSample, int coeffsIndex,
                  workingArray& work, bool fillExcitations = true)
  {
    //if (walk.excitedOrbs.size() > 2) return;
    int norbs = Determinant::norbs;
    double ciCoeff = coeffs(coeffsIndex);
    morework.setCounterToZero();
    double ovlp0, ham0;
    wave.HamAndOvlp(walk, ovlp0, ham0, morework, false);
    if (coeffsIndex == 0) ovlp = ciCoeff * ovlp0;
    else ovlp = ciCoeff * ham0;
    //cout << "nWalk   ind  " << coeffsIndex  << "  <n|H|phi>  " << ham0 << "  <n|phi>  " << ovlp0 << endl;
    //cout << walk << endl;
    if (ovlp == 0.) return; //maybe not necessary
    normSample = 1 / ciCoeff / ciCoeff;
    hamSample(coeffsIndex) += walk.d.Energy(I1, I2, coreE) / ciCoeff / ciCoeff;
    locEne = walk.d.Energy(I1, I2, coreE);
    //cout << "nEnergy  " << walk.d.Energy(I1, I2, coreE) << endl << endl;
    work.setCounterToZero();
    generateAllScreenedSingleExcitation(walk.d, schd.epsilon, schd.screen,
                                        work, false);
    generateAllScreenedDoubleExcitation(walk.d, schd.epsilon, schd.screen,
                                        work, false);

    //loop over all the screened excitations
    for (int i=0; i<work.nExcitations; i++) {
      double tia = work.HijElement[i];
      int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
      int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
      int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
      auto walkCopy = walk;
      double parity = 1.;
      Determinant dcopy = walkCopy.d;
      if (A > I) parity *= -1. * dcopy.parity(A/2, I/2, I%2);
      else parity *= dcopy.parity(A/2, I/2, I%2);
      dcopy.setocc(I, false);
      dcopy.setocc(A, true);
      if (ex2 != 0) {
        if (B > J) parity *= -1 * dcopy.parity(B/2, J/2, J%2);
        else parity *= dcopy.parity(B/2, J/2, J%2);
      }
      walkCopy.updateWalker(wave.getRef(), wave.getCorr(),
                            work.excitation1[i], work.excitation2[i], false);
      if (walkCopy.excitedOrbs.size() > 2) continue;
      int coeffsCopyIndex = this->coeffsIndex(walkCopy);
      morework.setCounterToZero();
      wave.HamAndOvlp(walkCopy, ovlp0, ham0, morework);
      //cout << "mWalk   ind  " << coeffsCopyIndex  << "  <m|H|phi>  " << ham0 << "  <m|phi>  " << ovlp0 << "  <m|H|n>  " << parity * tia << endl;
      //cout << walkCopy << endl;
      if (coeffsCopyIndex == 0) {
        hamSample(coeffsCopyIndex) += parity * tia * ovlp0 / ciCoeff / ovlp;
        locEne += parity * tia * ovlp0 * coeffs(coeffsCopyIndex) / ovlp;
        work.ovlpRatio[i] = ovlp0 * coeffs(coeffsCopyIndex) / ovlp;
      }
      else {
        hamSample(coeffsCopyIndex) += parity * tia * ham0 / ciCoeff / ovlp;
        locEne += parity * tia * ham0 * coeffs(coeffsCopyIndex) / ovlp;
        work.ovlpRatio[i] = ham0 * coeffs(coeffsCopyIndex) / ovlp;
      }
      //ham += tia * parity * ovlp0/ovlp;
    }
    //cout << "hamSample\n" << hamSample * ovlp * ovlp << endl << endl;
    //cout << "normSample   " << normSample * ovlp * ovlp << endl << endl;
  }
  
  template<typename Walker>
  void LocalEnergy(Walker &walk,
                  double &ovlp, double& locEne, int coeffsIndex,
                  workingArray& work, bool fillExcitations = true)
  {
    int norbs = Determinant::norbs;
    double ciCoeff = coeffs(coeffsIndex);
    morework.setCounterToZero();
    double ovlp0, ham0;
    wave.HamAndOvlp(walk, ovlp0, ham0, morework, false);
    if (coeffsIndex == 0) ovlp = ciCoeff * ovlp0;
    else ovlp = ciCoeff * ham0;
    if (ovlp == 0.) return; //maybe not necessary
    locEne = walk.d.Energy(I1, I2, coreE);
    work.setCounterToZero();
    generateAllScreenedSingleExcitation(walk.d, schd.epsilon, schd.screen,
                                        work, false);
    generateAllScreenedDoubleExcitation(walk.d, schd.epsilon, schd.screen,
                                        work, false);

    //loop over all the screened excitations
    for (int i=0; i<work.nExcitations; i++) {
      double tia = work.HijElement[i];
      int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
      int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
      int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
      auto walkCopy = walk;
      double parity = 1.;
      Determinant dcopy = walkCopy.d;
      if (A > I) parity *= -1. * dcopy.parity(A/2, I/2, I%2);
      else parity *= dcopy.parity(A/2, I/2, I%2);
      dcopy.setocc(I, false);
      dcopy.setocc(A, true);
      if (ex2 != 0) {
        if (B > J) parity *= -1 * dcopy.parity(B/2, J/2, J%2);
        else parity *= dcopy.parity(B/2, J/2, J%2);
      }
      walkCopy.updateWalker(wave.getRef(), wave.getCorr(),
                            work.excitation1[i], work.excitation2[i], false);
      if (walkCopy.excitedOrbs.size() > 2) continue;
      int coeffsCopyIndex = this->coeffsIndex(walkCopy);
      morework.setCounterToZero();
      wave.HamAndOvlp(walkCopy, ovlp0, ham0, morework);
      if (coeffsCopyIndex == 0) {
        locEne += parity * tia * ovlp0 * coeffs(coeffsCopyIndex) / ovlp;
        work.ovlpRatio[i] = ovlp0 * coeffs(coeffsCopyIndex) / ovlp;
      }
      else {
        locEne += parity * tia * ham0 * coeffs(coeffsCopyIndex) / ovlp;
        work.ovlpRatio[i] = ham0 * coeffs(coeffsCopyIndex) / ovlp;
      }
    }
  }

  template<typename Walker>
  double optimizeWaveCT(Walker& walk) {
    int norbs = Determinant::norbs;
    auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                            std::ref(generator));

    MatrixXd ciHam = MatrixXd::Zero(coeffs.size(), coeffs.size());
    //MatrixXd sMat = MatrixXd::Zero(coeffs.size(), coeffs.size()); 
    VectorXd ovlpDiag = VectorXd::Zero(coeffs.size());
    double ovlp = 0., normSample = 0., locEne = 0., ene = 0., correctionFactor = 0.;
    VectorXd hamSample = VectorXd::Zero(coeffs.size());
    int coeffsIndex = this->coeffsIndex(walk);
    workingArray work;
    Walker walkIn = walk;
    HamAndOvlp(walk, ovlp, normSample, locEne, hamSample, coeffsIndex, work);

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
      //sMat *= (1 - ratio);
      ovlpDiag *= (1 - ratio);
      //sMat(coeffsIndex, coeffsIndex) += ratio * normSample;
      ovlpDiag(coeffsIndex) += ratio * normSample;
      ciHam *= (1 - ratio);
      ciHam.row(coeffsIndex) += ratio * hamSample;
      ene *= (1 - ratio);
      ene += ratio * locEne;
      correctionFactor *= (1 - ratio);
      if (coeffsIndex == 0) correctionFactor += ratio;
      walk.updateWalker(wave.getRef(), wave.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
      coeffsIndex = this->coeffsIndex(walk);
      hamSample.setZero();
      HamAndOvlp(walk, ovlp, normSample, locEne, hamSample, coeffsIndex, work);
      iter++;
      if (commrank == 0 && iter % printMod == 1) cout << "iter  " << iter << "  t  " << getTime() - startofCalc << endl; 
    }
  
    ciHam *= cumdeltaT;
    //sMat *= cumdeltaT;
    ovlpDiag *= cumdeltaT;
    ene *= cumdeltaT;
    correctionFactor *= cumdeltaT;

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, ciHam.data(), coeffs.size() * coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  //MPI_Allreduce(MPI_IN_PLACE, sMat.data(), coeffs.size() * coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, ovlpDiag.data(), coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(cumdeltaT), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(ene), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(correctionFactor), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    
    if (commrank == 0) cout << "matrices built in  " << getTime() - startofCalc << endl; 
    
    double ene0 = 0.;
    if (commrank == 0) {
      ciHam /= cumdeltaT;
      //sMat /= cumdeltaT;
      ovlpDiag /= cumdeltaT;
      ene /= cumdeltaT;
      cout << "energy of sampling wavefunction   "  << ene << endl;
      correctionFactor /= cumdeltaT;
      cout << "correctionFactor   " << correctionFactor << endl;
      ciHam = (ciHam + ciHam.transpose().eval()) / 2;
      ene0 = ciHam(0, 0) / ovlpDiag(0);
      cout << "ref energy   " << ene0 << endl;
      //DiagonalMatrix<double, Dynamic> sMat(coeffs.size());
      //sMat.diagonal() = 1.e-7 + ovlpDiag.array();
      std::vector<int> largeNormIndices;
      for (int i = 0; i < coeffs.size(); i++) {
        if (ovlpDiag(i) > schd.overlapCutoff) largeNormIndices.push_back(i);
      }
      Map<VectorXi> largeNormSlice(&largeNormIndices[0], largeNormIndices.size());
      VectorXd largeNorms;
      igl::slice(ovlpDiag, largeNormSlice, largeNorms);
      DiagonalMatrix<double, Dynamic> normInv(coeffs.size());
      //normInv.diagonal() = (ovlpDiag.array() + 1.e-7).cwiseSqrt().cwiseInverse();
      normInv.diagonal() = largeNorms.cwiseSqrt().cwiseInverse();
      //normInv.diagonal() = largeNorms.cwiseInverse();
      MatrixXd largeHam;
      igl::slice(ciHam, largeNormSlice, largeNormSlice, largeHam);
      MatrixXd ciHamNorm = normInv * largeHam * normInv;
      //MatrixXd ciHamNorm = normInv * largeHam;
      SelfAdjointEigenSolver<MatrixXd> diag(ciHamNorm);
      //EigenSolver<MatrixXd> diag(ciHamNorm);
      //auto sMat = sMatDiag + 1.e-7 * MatrixXd::Identity(coeffs.size(), coeffs.size());
      //DiagonalMatrix<double, Dynamic> normInv(size);
      //normInv.diagonal() = diagVec.cwiseSqrt().cwiseInverse();
      //GeneralizedEigenSolver<MatrixXd> diag(ciHam, sMat);
      //cout << "ciHam\n" << ciHam << endl << endl;
      //cout << "sMat\n" << sMat << endl << endl;
      cout << "diagonaliztion in time  " << getTime() - startofCalc << endl; 
      VectorXd::Index minInd;
      double minEne = diag.eigenvalues().real().minCoeff(&minInd);
      cout << "energy eigenvalue   " << minEne << endl;
      VectorXd largeCoeffs = normInv * diag.eigenvectors().col(minInd).real();
      
      coeffs.setZero();
      for (int i = 0; i < largeNormIndices.size(); i++) coeffs(largeNormIndices[i]) = largeCoeffs(i);
      coeffs.normalize();
      cout << "SCCI+Q energy  " << minEne + (1 - correctionFactor) * (minEne - ene0) << endl;
      if (schd.printVars) cout << endl << "ci coeffs\n" << coeffs << endl; 
    }

#ifndef SERIAL
  MPI_Bcast(coeffs.data(), coeffs.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

    //ovlp = 0.;
    //double localEne = 0., ene = 0., correctionFactor = 0.;
    //walk = walkIn;
    //coeffsIndex = this->coeffsIndex(walk);
    //LocalEnergy(walk, ovlp, localEne, coeffsIndex, work);

    //iter = 0;
    //cumdeltaT = 0.;

    //while (iter < schd.stochasticIter) {
    //  double cumovlpRatio = 0;
    //  for (int i = 0; i < work.nExcitations; i++) {
    //    cumovlpRatio += abs(work.ovlpRatio[i]);
    //    work.ovlpRatio[i] = cumovlpRatio;
    //  }
    //  double deltaT = 1.0 / (cumovlpRatio);
    //  double nextDetRandom = random() * cumovlpRatio;
    //  int nextDet = std::lower_bound(work.ovlpRatio.begin(), (work.ovlpRatio.begin() + work.nExcitations),
    //                                 nextDetRandom) - work.ovlpRatio.begin();
    //  cumdeltaT += deltaT;
    //  double ratio = deltaT / cumdeltaT;
    //  //sMat *= (1 - ratio);
    //  ene *= (1 - ratio);
    //  //sMat(coeffsIndex, coeffsIndex) += ratio * normSample;
    //  ene += ratio * localEne;
    //  correctionFactor *= (1 - ratio);
    //  if (coeffsIndex == 0) correctionFactor += ratio;
    //  walk.updateWalker(wave.getRef(), wave.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
    //  coeffsIndex = this->coeffsIndex(walk);
    //  LocalEnergy(walk, ovlp, localEne, coeffsIndex, work);
    //  iter++;
    //  if (commrank == 0 && iter % printMod == 1) cout << "iter  " << iter << "  t  " << getTime() - startofCalc << endl; 
    //}
  
    //ene *= cumdeltaT;
    ////sMat *= cumdeltaT;
    //correctionFactor *= cumdeltaT;

//#ifndef SERIAL
//  MPI_Allreduce(MPI_IN_PLACE, &(ene), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//  //MPI_Allreduce(MPI_IN_PLACE, sMat.data(), coeffs.size() * coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//  MPI_Allreduce(MPI_IN_PLACE, &(correctionFactor), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//  MPI_Allreduce(MPI_IN_PLACE, &(cumdeltaT), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//#endif
    
    //if (commrank == 0) cout << "sampling finished in    " << getTime() - startofCalc << endl; 
    //if (commrank == 0) {
    //  ene /= cumdeltaT;
    //  correctionFactor /= cumdeltaT;
    //  cout << "sampled optimized energy   " << fixed << setprecision(5) << ene << endl;
    //  cout << "ref energy   " << fixed << setprecision(5) << ene0 << endl;
    //  cout << "correctionFac   " << correctionFactor << endl;
    //  cout << "SCCI+Q energy  " << ene + (1 - correctionFactor) * (ene - ene0) << endl;
    //}
  
  }
  
  template<typename Walker>
  double optimizeWaveDeterministic(Walker& walk) {

    int norbs = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta = Determinant::nbeta;
    vector<Determinant> allDets;
    generateAllDeterminants(allDets, norbs, nalpha, nbeta);

    workingArray work;
    double overlapTot = 0.; 
    VectorXd hamSample = VectorXd::Zero(coeffs.size());
    MatrixXd ciHam = MatrixXd::Zero(coeffs.size(), coeffs.size());
    MatrixXd sMat = MatrixXd::Zero(coeffs.size(), coeffs.size());// + 1.e-6 * MatrixXd::Identity(coeffs.size(), coeffs.size());
    //w.printVariables();

    for (int i = commrank; i < allDets.size(); i += commsize) {
      wave.initWalker(walk, allDets[i]);
      if (schd.debug) {
        cout << "walker\n" << walk << endl;
      }
      if (walk.excitedOrbs.size() > 2) continue;
      int coeffsIndex = this->coeffsIndex(walk);
      double ovlp = 0., normSample = 0., locEne = 0.;
      HamAndOvlp(walk, ovlp, normSample, locEne, hamSample, coeffsIndex, work);
      //cout << "ham  " << ham[0] << "  " << ham[1] << "  " << ham[2] << endl;
      //cout << "ovlp  " << ovlp[0] << "  " << ovlp[1] << "  " << ovlp[2] << endl << endl;
      
      overlapTot += ovlp * ovlp;
      ciHam.row(coeffsIndex) += (ovlp * ovlp) * hamSample;
      sMat(coeffsIndex, coeffsIndex) += (ovlp * ovlp) * normSample;
      hamSample.setZero();
    }

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(overlapTot), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, ciHam.data(), coeffs.size() * coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, sMat.data(), coeffs.size() * coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    
    double ene0 = 0.;
    if (commrank == 0) {
      ciHam = ciHam / overlapTot;
      sMat = sMat / overlapTot;
      ene0 = ciHam(0, 0) / sMat(0,0);
      //sMat += 1.e-8 * MatrixXd::Identity(coeffs.size(), coeffs.size());
      //cout << "ciHam\n" << ciHam << endl << endl;
      //cout << "sMat\n" << sMat << endl << endl; 
      GeneralizedEigenSolver<MatrixXd> diag(ciHam, sMat);
      VectorXd::Index minInd;
      double minEne = diag.eigenvalues().real().minCoeff(&minInd);
      coeffs = diag.eigenvectors().col(minInd).real();
      cout << "energy   " << fixed << setprecision(5) << minEne << endl;
      cout << "coeffs\n" << coeffs << endl;
    }
#ifndef SERIAL
  MPI_Bcast(coeffs.data(), coeffs.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //MPI_Bcast(&(ene0), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

    //davidson
    overlapTot = 0.; 
    double ene = 0., correctionFac = 0.;
    //w.printVariables();

    for (int i = commrank; i < allDets.size(); i += commsize) {
      wave.initWalker(walk, allDets[i]);
      if (schd.debug) {
        cout << "walker\n" << walk << endl;
      }
      if (walk.excitedOrbs.size() > 2) continue;
      int coeffsIndex = this->coeffsIndex(walk);
      double ovlp = 0., locEne = 0.;
      LocalEnergy(walk, ovlp, locEne, coeffsIndex, work);
      //cout << "ham  " << ham[0] << "  " << ham[1] << "  " << ham[2] << endl;
      //cout << "ovlp  " << ovlp[0] << "  " << ovlp[1] << "  " << ovlp[2] << endl << endl;
      
      if (coeffsIndex == 0) correctionFac += ovlp * ovlp;
      overlapTot += ovlp * ovlp;
      ene += (ovlp * ovlp) * locEne;
    }

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(overlapTot), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(correctionFac), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(ene), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (commrank == 0) {
      ene = ene / overlapTot;
      correctionFac = correctionFac / overlapTot;
      cout << "sampled optimized energy   " << fixed << setprecision(5) << ene << endl;
      cout << "ref energy   " << fixed << setprecision(5) << ene0 << endl;
      cout << "correctionFac   " << correctionFac << endl;
      cout << "SCCI+Q energy  " << ene + (1 - correctionFac) * (ene - ene0) << endl;
    }
  //if (commrank == 0) cout << "energies\n" << diag.eigenvalues() << endl;
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
