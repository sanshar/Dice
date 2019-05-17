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
  
  //intermediates used in direct methods
  vector<Eigen::VectorXd> largeHamSamples;
  vector<double> largeSampleTimes;
  vector<int> largeSampleIndices;
  vector<int> nestedIndices;
  DiagonalMatrix<double, Dynamic> largeNormInv;
  VectorXd largeHamDiag;
  double cumulativeTime; 


  SCCI()
  {
    wave.readWave();

    // Resize coeffs
    int numVirt = Determinant::norbs - schd.nciAct;
    int numCoeffs = 1 + 2*numVirt + (2*numVirt * (2*numVirt - 1) / 2);
    coeffs = VectorXd::Zero(numCoeffs);

    //coeffs order: phi0, singly excited (spin orb index), doubly excited (spin orb pair index)

    if (commrank == 0) {
      auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                              std::ref(generator));

      coeffs(0) = -0.5;
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

  //hamSample = <psi^k_l|n>/<psi|n> * <n|H|psi^k'_l'>/<n|psi>, ovlp = <n|psi>
  template<typename Walker>
  void HamAndOvlp(Walker &walk,
                  double &ovlp, double &normSample, double &locEne, VectorXd& hamSample, int coeffsIndex,
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
    normSample = 1 / ciCoeff / ciCoeff;
    locEne = walk.d.Energy(I1, I2, coreE);
    hamSample(coeffsIndex) += locEne / ciCoeff / ciCoeff;
    work.setCounterToZero();
    generateAllScreenedSingleExcitation(walk.d, schd.epsilon, schd.screen,
                                        work, false);
    if (walk.excitedOrbs.size() == 0) {
      generateAllScreenedDoubleExcitation(walk.d, schd.epsilon, schd.screen,
                                        work, false);
    }
    else {
      generateAllScreenedDoubleExcitationsFOIS(walk.d, schd.epsilon, schd.screen,
                                        work, false);
    }
    
    //loop over all the screened excitations
    for (int i=0; i<work.nExcitations; i++) {
      double tia = work.HijElement[i];
      int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
      int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
      int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
      auto walkCopy = walk;
      double parity = 1.;
      Determinant dcopy = walkCopy.d;
      walkCopy.updateWalker(wave.getRef(), wave.getCorr(),
                            work.excitation1[i], work.excitation2[i], false);
      if (walkCopy.excitedOrbs.size() > 2) continue;
      parity *= dcopy.parity(A/2, I/2, I%2);
      dcopy.setocc(I, false);
      dcopy.setocc(A, true);
      if (ex2 != 0) {
        parity *= dcopy.parity(B/2, J/2, J%2);
      }
      int coeffsCopyIndex = this->coeffsIndex(walkCopy);
      morework.setCounterToZero();
      wave.HamAndOvlp(walkCopy, ovlp0, ham0, morework);
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
    }
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
      //if (A > I) parity *= -1. * dcopy.parityCI(A/2, I/2, I%2);
      //else parity *= dcopy.parityCI(A/2, I/2, I%2);
      parity *= dcopy.parity(A/2, I/2, I%2);
      dcopy.setocc(I, false);
      dcopy.setocc(A, true);
      if (ex2 != 0) {
        //if (B > J) parity *= -1 * dcopy.parityCI(B/2, J/2, J%2);
        //else parity *= dcopy.parityCI(B/2, J/2, J%2);
        parity *= dcopy.parity(B/2, J/2, J%2);
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
      VectorXd initGuess = VectorXd::Random(largeNorms.size());
      initGuess(0) = 1.;
      double eigenVal; VectorXd eigenVec;
      cout << "ciHamNorm\n" << ciHamNorm << endl;
      cout << "non-direct davidson\n";
      davidsonMethod(ciHamNorm, initGuess, eigenVec, eigenVal, 200, 1.e-5); 
      cout << "energy eigenvalue of built ham   " << eigenVal << endl;
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

  }
 
  //builds hamiltonian using sampled vectors (used for debugging)
  void buildHamMat(MatrixXd &hamMat) {
    MatrixXd ham = 0 * hamMat;
    for (int j = 0; j < largeSampleIndices.size(); j++) {
      ham.row(nestedIndices[largeSampleIndices[j]]) += largeSampleTimes[j] * largeHamSamples[j];
    }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, ham.data(), ham.rows() * ham.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    ham /= cumulativeTime;
    hamMat = largeNormInv * ham * largeNormInv;
  }

  void davidsonMethod(MatrixXd &mat, VectorXd& initGuess, VectorXd& eigenvec, double& eigenval, const int maxIter, const double threshold)
  {
    initGuess.normalize();
    VectorXd vectorToBeAdded = initGuess;
    int fullDim = initGuess.size(), subspaceDim = 0;
    MatrixXd subspaceBasis = MatrixXd::Zero(1,1), hV = MatrixXd::Zero(1, 1), subspaceHam = MatrixXd::Zero(1, 1);
    for (int i = 0; i < maxIter; i++) {
      //add a vector to the subspace
      subspaceDim += 1;
      subspaceBasis.conservativeResize(fullDim, subspaceDim);
      subspaceBasis.col(subspaceDim - 1) = vectorToBeAdded;
      //cout << "subspaceBasis\n" << subspaceBasis << endl;

      //extend subspaceHam to include added vector
      VectorXd hVectorToBeAdded = mat * subspaceBasis.col(subspaceDim - 1);
      hV.conservativeResize(fullDim, subspaceDim);
      hV.col(subspaceDim - 1)  = hVectorToBeAdded;
      //subspaceHam = subspaceBasis.transpose() * hV;
      VectorXd newHamCol = hV.transpose() * subspaceBasis.col(subspaceDim - 1);
      subspaceHam.conservativeResize(subspaceDim, subspaceDim);
      subspaceHam.col(subspaceDim - 1) = newHamCol;
      subspaceHam.row(subspaceDim - 1) = newHamCol;
      //diagonalize subspaceHam
      SelfAdjointEigenSolver<MatrixXd> diag(subspaceHam);
      //EigenSolver<MatrixXd> diag(subspaceHam);
      eigenval = diag.eigenvalues()(0);
      //VectorXd::Index minInd;
      //eigenval = diag.eigenvalues().real().minCoeff(&minInd);
      //cout << "eigenval  " << eigenval << endl;
      eigenvec = subspaceBasis * diag.eigenvectors().col(0);
      //eigenvec = subspaceBasis * diag.eigenvectors().col(minInd).real();
      VectorXd resVec = mat * eigenvec - eigenval * eigenvec;
      //cout << "resVec\n" << resVec << endl;
      if (resVec.norm() < threshold) {
        cout << "Davidson converged in " << i+1 << " iterations" << endl;
        return;
      }
      else {//calculate vector to be added
        //calculate delta
        DiagonalMatrix<double, Dynamic> preconditioner;
        preconditioner.diagonal() = mat.diagonal() - VectorXd::Constant(fullDim, eigenval);
        VectorXd delta = preconditioner.inverse() * resVec;
        //cout << "delta\n" << delta << endl;
        //orthonormalize
        VectorXd overlaps = subspaceBasis.transpose() * delta;
        //cout << "overlaps\n" << overlaps << endl;
        //cout << "subspaceBasis * overlaps\n" << subspaceBasis * overlaps << endl;
        vectorToBeAdded = delta - (subspaceBasis * overlaps);
        //cout << "before normalize\n" << vectorToBeAdded << endl;
        vectorToBeAdded.normalize();
      }
    }
    cout << "Davidson did not converge in " << maxIter << " iterations" << endl;
  }

  //simulates hamiltonian multiplication
  void HMult(VectorXd& x, VectorXd& hX) {
    VectorXd sInvX = largeNormInv * x;
    VectorXd hSInvX = VectorXd::Zero(x.size());
    for (int j = 0; j < largeSampleIndices.size(); j++) {
      hSInvX(nestedIndices[largeSampleIndices[j]]) += largeSampleTimes[j] * largeHamSamples[j].dot(sInvX);
    }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, hSInvX.data(), hSInvX.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    hSInvX /= cumulativeTime;
    hX = largeNormInv * hSInvX;
  }

  void powerMethod(VectorXd& initGuess, VectorXd& eigenVec, double& eigenVal, const int& numPowerIter, const double& threshold) {
    VectorXd oldIterVecNormal = initGuess;
    VectorXd newIterVec(initGuess.size());
    for (int i = 0; i < numPowerIter; i++) {
      HMult(oldIterVecNormal, newIterVec);
      eigenVal = newIterVec.dot(oldIterVecNormal);
      VectorXd r = newIterVec - eigenVal * oldIterVecNormal;
      if (r.norm() < 1.e-6) {
        eigenVec = newIterVec / newIterVec.norm();
        if (commrank == 0) cout << "power method converged in " << i << " iterations\n";
        return;
      }
      oldIterVecNormal= newIterVec / newIterVec.norm();
    }
    eigenVec = oldIterVecNormal;
    if (commrank == 0) cout << "power method did not converge in " << numPowerIter << " iterations\n";
  }
  
  void powerMethod(MatrixXd& mat, VectorXd& initGuess, VectorXd& eigenVec, double& eigenVal, const int& numPowerIter, const double& threshold) {
    VectorXd oldIterVecNormal = initGuess;
    VectorXd newIterVec(initGuess.size());
    for (int i = 0; i < numPowerIter; i++) {
      newIterVec = mat * oldIterVecNormal;
      eigenVal = newIterVec.dot(oldIterVecNormal);
      VectorXd r = newIterVec - eigenVal * oldIterVecNormal;
      if (r.norm() < 1.e-6) {
        eigenVec = newIterVec / newIterVec.norm();
        if (commrank == 0) cout << "power method converged in " << i << " iterations\n";
        return;
      }
      oldIterVecNormal= newIterVec / newIterVec.norm();
    }
    eigenVec = oldIterVecNormal;
    if (commrank == 0) cout << "power method did not converge in " << numPowerIter << " iterations\n";
  }
  
  void davidsonMethod(VectorXd& initGuess, VectorXd& eigenvec, double& eigenval, const int& maxDavidsonIter, const double& threshold) {
    initGuess.normalize();
    VectorXd vectorToBeAdded = initGuess;
    int fullDim = initGuess.size(), subspaceDim = 0;
    MatrixXd subspaceBasis = MatrixXd::Zero(1,1), hV = MatrixXd::Zero(1, 1), subspaceHam = MatrixXd::Zero(1, 1);
    for (int i = 0; i < maxDavidsonIter; i++) {
      //add a vector to the subspace
      subspaceDim += 1;
      subspaceBasis.conservativeResize(fullDim, subspaceDim);
      subspaceBasis.col(subspaceDim - 1) = vectorToBeAdded;
      //cout << "subspaceBasis\n" << subspaceBasis << endl;

      //extend subspaceHam to include added vector
      //VectorXd hVectorToBeAdded = mat * subspaceBasis.col(subspaceDim - 1);
      VectorXd hVectorToBeAdded = VectorXd::Zero(fullDim);
      HMult(vectorToBeAdded, hVectorToBeAdded);
      hV.conservativeResize(fullDim, subspaceDim);
      hV.col(subspaceDim - 1)  = hVectorToBeAdded;
      subspaceHam = subspaceBasis.transpose() * hV;
      //VectorXd newHamCol = hV.transpose() * subspaceBasis.col(subspaceDim - 1);
      //subspaceHam.conservativeResize(subspaceDim, subspaceDim);
      //subspaceHam.col(subspaceDim - 1) = newHamCol;
      //subspaceHam.row(subspaceDim - 1) = newHamCol;

      //diagonalize subspaceHam
      //SelfAdjointEigenSolver<MatrixXd> diag(subspaceHam);
      //eigenval = diag.eigenvalues()(0);
      //cout << "eigenval  " << eigenval << endl;
      //eigenvec = subspaceBasis * diag.eigenvectors().col(0);
      //VectorXd resVec = mat * eigenvec - eigenval * eigenvec;
      EigenSolver<MatrixXd> diag(subspaceHam);
      VectorXd::Index minInd;
      eigenval = diag.eigenvalues().real().minCoeff(&minInd);
      eigenvec = subspaceBasis * diag.eigenvectors().col(minInd).real();
      VectorXd hEigenvec = VectorXd::Zero(fullDim);
      HMult(eigenvec, hEigenvec);
      VectorXd resVec = hEigenvec - eigenval * eigenvec;
      //cout << "resVec\n" << resVec << endl;
      if (resVec.norm() < threshold) {
        if (commrank == 0) cout << "Davidson converged in " << i << " iterations" << endl;
        return;
      }
      else {//calculate vector to be added
        //calculate delta
        DiagonalMatrix<double, Dynamic> preconditioner;
        //preconditioner.diagonal() = mat.diagonal() - VectorXd::Constant(fullDim, eigenval);
        preconditioner.diagonal() = largeHamDiag - VectorXd::Constant(fullDim, eigenval);
        //preconditioner.diagonal() = VectorXd::Constant(fullDim, largeHamDiag(0) - eigenval);
        VectorXd delta = preconditioner.inverse() * resVec;
        //VectorXd delta = resVec;
        //cout << "delta\n" << delta << endl;
        //orthonormalize
        VectorXd overlaps = subspaceBasis.transpose() * delta;
        //cout << "overlaps\n" << overlaps << endl;
        //cout << "subspaceBasis * overlaps\n" << subspaceBasis * overlaps << endl;
        vectorToBeAdded = delta - (subspaceBasis * overlaps);
        //cout << "before normalize\n" << vectorToBeAdded << endl;
        vectorToBeAdded.normalize();
      }
    }
    if (commrank == 0) cout << "Davidson did not converge in " << maxDavidsonIter << " iterations" << endl;
  }

  template<typename Walker>
  double optimizeWaveCTDirect(Walker& walk) {
    
    //sampling
    int norbs = Determinant::norbs;
    auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                            std::ref(generator));

    VectorXd ovlpDiag = VectorXd::Zero(coeffs.size());
    VectorXd hamDiag = VectorXd::Zero(coeffs.size());
    double ovlp = 0., normSample = 0., locEne = 0., ene = 0., ene0 = 0., correctionFactor = 0.;
    VectorXd hamSample = VectorXd::Zero(coeffs.size());
    vector<Eigen::VectorXd> hamSamples;
    vector<double> sampleTimes; vector<int> sampleIndices;
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
      ovlpDiag *= (1 - ratio);
      ovlpDiag(coeffsIndex) += ratio * normSample;
      hamDiag *= (1 - ratio);
      hamDiag(coeffsIndex) += ratio * hamSample(coeffsIndex);
      ene *= (1 - ratio);
      ene += ratio * locEne;
      correctionFactor *= (1 - ratio);
      ene0 *= (1 - ratio);
      if (coeffsIndex == 0) {
        correctionFactor += ratio;
        ene0 += ratio * hamSample(0);
      }
      hamSamples.push_back(hamSample);
      sampleTimes.push_back(deltaT);
      sampleIndices.push_back(coeffsIndex);
      walk.updateWalker(wave.getRef(), wave.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
      coeffsIndex = this->coeffsIndex(walk);
      hamSample.setZero();
      HamAndOvlp(walk, ovlp, normSample, locEne, hamSample, coeffsIndex, work);
      iter++;
      if (commrank == 0 && iter % printMod == 1) cout << "iter  " << iter << "  t  " << getTime() - startofCalc << endl; 
    }
  
    ovlpDiag *= cumdeltaT;
    hamDiag *= cumdeltaT;
    ene *= cumdeltaT;
    ene0 *= cumdeltaT;
    correctionFactor *= cumdeltaT;

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, ovlpDiag.data(), coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, hamDiag.data(), coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(cumdeltaT), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(ene), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(ene0), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(correctionFactor), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    
    cumulativeTime = cumdeltaT;
    ovlpDiag /= cumdeltaT;
    hamDiag /= cumdeltaT;
    ene /= cumdeltaT;
    ene0 /= (cumdeltaT * ovlpDiag(0));
    correctionFactor /= cumdeltaT;
    if (commrank == 0) {
      cout << "ref energy   " << ene0 << endl;
      cout << "energy of sampling wavefunction   "  << ene << endl;
      cout << "correctionFactor   " << correctionFactor << endl;
      cout << "sampling done in   " << getTime() - startofCalc << endl; 
    }
    
    //preparing for diagonalization
    std::vector<int> largeNormIndices;
    nestedIndices.resize(coeffs.size());
    int counter = 0;
    for (int i = 0; i < coeffs.size(); i++) {
      if (ovlpDiag(i) > schd.overlapCutoff) {
        largeNormIndices.push_back(i);
        nestedIndices[i] = counter;
        counter++;
      }
    }
    Map<VectorXi> largeNormSlice(&largeNormIndices[0], largeNormIndices.size());
    VectorXd largeNorms;
    igl::slice(ovlpDiag, largeNormSlice, largeNorms);
    largeNormInv.resize(coeffs.size());
    largeNormInv.diagonal() = largeNorms.cwiseSqrt().cwiseInverse();
    igl::slice(hamDiag, largeNormSlice, largeHamDiag);
    largeHamDiag = (largeNormInv.diagonal().cwiseProduct(largeHamDiag)); 
    largeHamDiag = (largeNormInv.diagonal().cwiseProduct(largeHamDiag)); 
    for (int i = 0; i < sampleIndices.size(); i++) {
      if (ovlpDiag(sampleIndices[i]) <= schd.overlapCutoff) continue;
      else {
        largeSampleTimes.push_back(sampleTimes[i]);
        largeSampleIndices.push_back(sampleIndices[i]);
        VectorXd largeHamSample;  
        igl::slice(hamSamples[i], largeNormSlice, largeHamSample);
        largeHamSamples.push_back(largeHamSample);
      }
    }
    hamSamples.clear(); sampleTimes.clear(); sampleIndices.clear();
   
    //diagonaliztion
    VectorXd initGuess = VectorXd::Unit(largeNorms.size(), 0);
    if (commrank == 0) {
      auto random = std::bind(std::uniform_real_distribution<double>(-0.1, 0.1),
                              std::ref(generator));
      for (int i=1; i < initGuess.size(); i++) {
        initGuess(i) = random();
      }
    }

#ifndef SERIAL
  MPI_Bcast(initGuess.data(), initGuess.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    double eigenVal;
    VectorXd eigenVec;
    //powerMethod(initGuess, eigenVec, eigenVal, 5000, 1.e-6); 
    davidsonMethod(initGuess, eigenVec, eigenVal, 100, 1.e-4); 
    VectorXd largeCoeffs = largeNormInv * eigenVec;
    coeffs.setZero();
    for (int i = 0; i < largeNormIndices.size(); i++) coeffs(largeNormIndices[i]) = largeCoeffs(i);

    if (commrank == 0) {
      cout << "diagonalization in time  " << getTime() - startofCalc << endl; 
      cout << "retained " << largeNorms.size() << " out of " << coeffs.size() << " states" << endl;
      cout << "energy eigenvalue   " << eigenVal << endl;
      cout << "SCCI+Q energy  " << eigenVal + (1 - correctionFactor) * (eigenVal - ene0) << endl;
      if (schd.printVars) cout << endl << "ci coeffs\n" << coeffs << endl; 
    }
    largeHamSamples.clear(); largeSampleTimes.clear(); largeSampleIndices.clear(), nestedIndices.clear();
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
      sMat += 1.e-8 * MatrixXd::Identity(coeffs.size(), coeffs.size());
      cout << "ciHam\n" << ciHam << endl << endl;
      cout << "sMat\n" << sMat << endl << endl; 
      GeneralizedEigenSolver<MatrixXd> diag(ciHam, sMat);
      VectorXd::Index minInd;
      double minEne = diag.eigenvalues().real().minCoeff(&minInd);
      coeffs = diag.eigenvectors().col(minInd).real();
      cout << "energy   " << fixed << setprecision(5) << minEne << endl;
      cout << "eigenvalues\n" << diag.eigenvalues() << endl;
      cout << "ciHam\n" << ciHam << endl;
      cout << "sMat\n" << sMat << endl;
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
  
  template<typename Walker>
  double optimizeWaveDeterministicDirect(Walker& walk) {

    int norbs = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta = Determinant::nbeta;
    vector<Determinant> allDets;
    generateAllDeterminants(allDets, norbs, nalpha, nbeta);

    workingArray work;
    double overlapTot = 0.; 
    VectorXd hamSample = VectorXd::Zero(coeffs.size());
    MatrixXd ciHam = MatrixXd::Zero(coeffs.size(), coeffs.size());
    vector<Eigen::VectorXd> hamSamples;
    vector<double> ovlpSqSamples; vector<int> sampleIndices;
    MatrixXd sMat = MatrixXd::Zero(coeffs.size(), coeffs.size());// + 1.e-6 * MatrixXd::Identity(coeffs.size(), coeffs.size());
    //w.printVariables();

    for (int i = commrank; i < allDets.size(); i += commsize) {
      wave.initWalker(walk, allDets[i]);
      if (walk.excitedOrbs.size() > 2) continue;
      int coeffsIndex = this->coeffsIndex(walk);
      double ovlp = 0., normSample = 0., locEne = 0.;
      HamAndOvlp(walk, ovlp, normSample, locEne, hamSample, coeffsIndex, work);
      //cout << "walker\n" << walk << endl;
      //cout << "hamSample\n" << hamSample << endl;
      //cout << "ovlp   " << ovlp << endl;
      ciHam.row(coeffsIndex) += (ovlp * ovlp) * hamSample;
      hamSamples.push_back(hamSample);
      ovlpSqSamples.push_back(ovlp * ovlp);
      sampleIndices.push_back(coeffsIndex);
      overlapTot += ovlp * ovlp;
      sMat(coeffsIndex, coeffsIndex) += (ovlp * ovlp) * normSample;
      hamSample.setZero();
    }

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(overlapTot), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, sMat.data(), coeffs.size() * coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, ciHam.data(), coeffs.size() * coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    double ene = 0.;
    sMat = sMat / overlapTot;
    ciHam = ciHam / overlapTot;
    //if (commrank == 0) {
    //  cout << "sMat\n" << sMat.diagonal() << endl; 
    //  cout << "ciHam\n" << ciHam << endl;
    //}
    sMat += 1.e-8 * MatrixXd::Identity(coeffs.size(), coeffs.size());
    MatrixXd sMatInv = sMat.inverse().cwiseSqrt();
    MatrixXd ciHamNorm = sMatInv * ciHam * sMatInv;
    EigenSolver<MatrixXd> diag(ciHamNorm);
    //cout << "ciHamNorm eigenvalues\n" << diag.eigenvalues() << endl;
    VectorXd initGuess = VectorXd::Unit(coeffs.size(), 0);
    VectorXd iterVec = 0. * initGuess;
    VectorXd iterVecNormal = initGuess;
    for (int i = 0; i < 20; i++) {
      VectorXd sInvIterVec = sMatInv * iterVecNormal;
      for (int j = 0; j < sampleIndices.size(); j ++) {
        iterVec(sampleIndices[j]) += ovlpSqSamples[j] * hamSamples[j].dot(sInvIterVec);
      }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, iterVec.data(), coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      iterVec /= overlapTot;
      VectorXd newIterVec = sMatInv * iterVec;
      VectorXd ciHamNormIterVec = ciHamNorm * iterVecNormal;
      //cout << "iter   " << i << endl;
      //cout << "direct iterVec\n" << newIterVec << endl;
      //cout << "non-direct iterVec\n" << ciHamNormIterVec << endl; 
      ene = iterVecNormal.dot(newIterVec);
      iterVecNormal= newIterVec / newIterVec.norm();
      iterVec.setZero();
    }
    
    coeffs = sMatInv * iterVecNormal;
    cout << "energy power method    " << ene << endl;
    //cout << "coeffs\n" << coeffs << endl;

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
