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
#include "Wfn.h"
#include "integral.h"
#include "CPS.h"
#include "MoDeterminants.h"
#include "Walker.h"
#include "Wfn.h"
#include "global.h"
#include "input.h"
#include "Profile.h"
#include <fstream>
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif

using namespace Eigen;

CPSSlater::CPSSlater(std::vector<Correlator> &pcpsArray,
                     std::vector<Determinant> &pdeterminants,
                     std::vector<double> &pciExpansion) : cpsArray(pcpsArray),
                                                          determinants(pdeterminants),
                                                          ciExpansion(pciExpansion)
{
  int maxCPSSize = 0;
  orbitalToCPS.resize(Determinant::norbs);
  for (int i = 0; i < cpsArray.size(); i++)
  {
    for (int j = 0; j < cpsArray[i].asites.size(); j++)
      orbitalToCPS[cpsArray[i].asites[j]].push_back(i);
  }

  for (int i = 0; i < orbitalToCPS.size(); i++)
    if (orbitalToCPS[i].size() > maxCPSSize)
      maxCPSSize = orbitalToCPS[i].size();
  workingVectorOfCPS.resize(4 * maxCPSSize);
};

void CPSSlater::getDetMatrix(Determinant &d, Eigen::MatrixXd &DetAlpha, Eigen::MatrixXd &DetBeta)
{
  //alpha and beta orbitals of the walker determinant d
  std::vector<int> alpha, beta;
  d.getAlphaBeta(alpha, beta);
  int nalpha = alpha.size(), nbeta = beta.size();

  //alpha and beta orbitals of the reference determinant
  std::vector<int> alphaRef, betaRef;
  determinants[0].getAlphaBeta(alphaRef, betaRef);

  DetAlpha = MatrixXd::Zero(nalpha, nalpha);
  DetBeta = MatrixXd::Zero(nbeta, nbeta);

  //<psi1 | psi2> = det(phi1^dag phi2)
  //in out case psi1 is a simple occupation number determ inant
  for (int i = 0; i < nalpha; i++)
    for (int j = 0; j < nalpha; j++)
      DetAlpha(i, j) = HforbsA(alpha[i], alphaRef[j]);

  for (int i = 0; i < nbeta; i++)
    for (int j = 0; j < nbeta; j++)
      DetBeta(i, j) = HforbsB(beta[i], betaRef[j]);

  return;
}

double CPSSlater::getOverlapWithDeterminants(Walker &walk)
{
  return walk.getDetOverlap(*this);
}

//This is expensive and not recommended
double CPSSlater::Overlap(Determinant &d)
{
  double ovlp = 1.0;
  for (int i = 0; i < cpsArray.size(); i++)
  {
    ovlp *= cpsArray[i].Overlap(d);
  }
  Eigen::MatrixXd DetAlpha, DetBeta;
  getDetMatrix(d, DetAlpha, DetBeta);
  return ovlp * DetAlpha.determinant() * DetBeta.determinant();
}

double CPSSlater::Overlap(Walker &walk)
{
  double ovlp = 1.0;

  //Overlap with all the Correlators
  for (int i = 0; i < cpsArray.size(); i++)
  {
    ovlp *= cpsArray[i].Overlap(walk.d);
  }

  return ovlp * getOverlapWithDeterminants(walk);
}

/*
double CPSSlater::getJastrowFactor(int i, int a, Determinant& dcopy, Determinant& d){
  double cpsFactor = 1.0;
  for (int n = 0; n < cpsArray.size(); n++)
    for (int j = 0; j < cpsArray[n].asites.size(); j++)
    {
      if (cpsArray[n].asites[j] == i ||
          cpsArray[n].asites[j] == a)
      {
        cpsFactor *= cpsArray[n].Overlap(dcopy) / cpsArray[n].Overlap(d);
        break;
      }
    }
  return cpsFactor;
}

double CPSSlater::getJastrowFactor(int i, int j, int a, int b, Determinant& dcopy, Determinant& d){
  double cpsFactor = 1.0;
  for (int n = 0; n < cpsArray.size(); n++)
    for (int k = 0; k < cpsArray[n].asites.size(); k++)
    {
      if (cpsArray[n].asites[k] == i ||
          cpsArray[n].asites[k] == a ||
          cpsArray[n].asites[k] == j ||
          cpsArray[n].asites[k] == b )
      {
        cpsFactor *= cpsArray[n].Overlap(dcopy) / cpsArray[n].Overlap(d);
        break;
      }
    }
  return cpsFactor;
}
*/

double CPSSlater::getJastrowFactor(int i, int a, Determinant &dcopy, Determinant &d)
{
  double cpsFactor = 1.0;

  int index = 0;
  for (int x = 0; x < orbitalToCPS[i].size(); x++)
  {
    workingVectorOfCPS[index] = orbitalToCPS[i][x];
    index++;
  }
  for (int x = 0; x < orbitalToCPS[a].size(); x++)
  {
    workingVectorOfCPS[index] = orbitalToCPS[a][x];
    index++;
  }
  sort(workingVectorOfCPS.begin(), workingVectorOfCPS.begin() + index);

  int prevIndex = -1;
  for (int x = 0; x < index; x++)
  {
    if (workingVectorOfCPS[x] != prevIndex)
    {
      //cpsFactor *= cpsArray[ workingVectorOfCPS[x] ].OverlapRatio(dcopy,d);
      cpsFactor *= cpsArray[workingVectorOfCPS[x]].Overlap(dcopy) / cpsArray[workingVectorOfCPS[x]].Overlap(d);
      prevIndex = workingVectorOfCPS[x];
    }
  }

  return cpsFactor;
}

double CPSSlater::getJastrowFactor(int i, int j, int a, int b, Determinant &dcopy, Determinant &d)
{
  double cpsFactor = 1.0;

  int index = 0;
  for (int x = 0; x < orbitalToCPS[i].size(); x++)
  {
    workingVectorOfCPS[index] = orbitalToCPS[i][x];
    index++;
  }
  for (int x = 0; x < orbitalToCPS[a].size(); x++)
  {
    workingVectorOfCPS[index] = orbitalToCPS[a][x];
    index++;
  }
  for (int x = 0; x < orbitalToCPS[j].size(); x++)
  {
    workingVectorOfCPS[index] = orbitalToCPS[j][x];
    index++;
  }
  for (int x = 0; x < orbitalToCPS[b].size(); x++)
  {
    workingVectorOfCPS[index] = orbitalToCPS[b][x];
    index++;
  }
  sort(workingVectorOfCPS.begin(), workingVectorOfCPS.begin() + index);

  int prevIndex = -1;
  for (int x = 0; x < index; x++)
  {
    if (workingVectorOfCPS[x] != prevIndex)
    {
      cpsFactor *= cpsArray[workingVectorOfCPS[x]].Overlap(dcopy) / cpsArray[workingVectorOfCPS[x]].Overlap(d);
      //cpsFactor *= cpsArray[ workingVectorOfCPS[x] ].OverlapRatio(dcopy,d);
      prevIndex = workingVectorOfCPS[x];
    }
  }
  return cpsFactor;
}

void CPSSlater::OverlapWithGradient(Walker &w,
                                    double &factor,
                                    VectorXd &grad)
{
  OverlapWithGradient(w.d, factor, grad);
}

void CPSSlater::OverlapWithGradient(Determinant &d,
                                    double &factor,
                                    VectorXd &grad)
{

  double ovlp = factor;
  long startIndex = 0;

  for (int i = 0; i < cpsArray.size(); i++)
  {
    cpsArray[i].OverlapWithGradient(d, grad,
                                    ovlp, startIndex);
    startIndex += cpsArray[i].Variables.size();
  }
}

void CPSSlater::printVariables()
{
  long numVars = 0;
  for (int i = 0; i < cpsArray.size(); i++)
  {
    for (int j = 0; j < cpsArray[i].Variables.size(); j++)
    {
      cout << "  " << cpsArray[i].Variables[j];
      numVars++;
    }
  }

  for (int i = 0; i < determinants.size(); i++)
  {
    cout << "  " << ciExpansion[i];
  }
  cout << endl;
}

void CPSSlater::updateVariables(Eigen::VectorXd &v)
{
  long numVars = 0;
  for (int i = 0; i < cpsArray.size(); i++)
  {
    for (int j = 0; j < cpsArray[i].Variables.size(); j++)
    {
      cpsArray[i].Variables[j] = v[numVars];
      numVars++;
    }
  }

  for (int i = 0; i < determinants.size(); i++)
  {
    ciExpansion[i] = v[numVars];
    numVars++;
  }
}

void orthogonalise(MatrixXd &m)
{

  for (int i = 0; i < m.cols(); i++)
  {
    for (int j = 0; j < i; j++)
    {
      double ovlp = m.col(i).transpose() * m.col(j);
      double norm = m.col(j).transpose() * m.col(j);
      m.col(i) = m.col(i) - ovlp / norm * m.col(j);
    }
    m.col(i) = m.col(i) / pow(m.col(i).transpose() * m.col(i), 0.5);
  }
}

void CPSSlater::getVariables(Eigen::VectorXd &v)
{
  long numVars = 0;
  for (int i = 0; i < cpsArray.size(); i++)
  {
    for (int j = 0; j < cpsArray[i].Variables.size(); j++)
    {
      v[numVars] = cpsArray[i].Variables[j];
      numVars++;
    }
  }
  for (int i = 0; i < determinants.size(); i++)
  {
    v[numVars] = ciExpansion[i];
    numVars++;
  }
}

long CPSSlater::getNumVariables()
{
  long numVars = 0;
  for (int i = 0; i < cpsArray.size(); i++)
    numVars += cpsArray[i].Variables.size();

  numVars += determinants.size();
  //numVars+=det.norbs*det.nalpha+det.norbs*det.nbeta;
  return numVars;
}

long CPSSlater::getNumJastrowVariables()
{
  long numVars = 0;
  for (int i = 0; i < cpsArray.size(); i++)
    numVars += cpsArray[i].Variables.size();

  return numVars;
}
//factor = <psi|w> * prefactor;

void CPSSlater::writeWave()
{
  if (commrank == 0)
  {
    char file[5000];
    //sprintf (file, "wave.bkp" , schd.prefix[0].c_str() );
    sprintf(file, "wave.bkp");
    std::ofstream outfs(file, std::ios::binary);
    boost::archive::binary_oarchive save(outfs);
    save << *this;
    outfs.close();
  }
}

void CPSSlater::readWave()
{
  if (commrank == 0)
  {
    char file[5000];
    //sprintf (file, "wave.bkp" , schd.prefix[0].c_str() );
    sprintf(file, "wave.bkp");
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

//for a determinant |N> it updates the grad ratio vector
// grad[i] += <N|Psi_i>/<N|Psi0> * factor
void CPSSlater::OvlpRatioCI(Walker &walk, VectorXd &gradRatio, int (*getIndex)(int, int, int),
                            oneInt &I1, twoInt &I2, vector<int> &SingleIndices,
                            twoIntHeatBathSHM &I2hb, double &coreE, double factor)
{
  //<d|I^dag A |Psi0>
  //|dcopy> = A^dag I|d>
  int norbs = Determinant::norbs;
  gradRatio[0] += factor;
  for (int i = 0; i < SingleIndices.size() / 2; i++)
  {
    int I = SingleIndices[2 * i], A = SingleIndices[2 * i + 1];
    int index = getIndex(A, I, norbs);

    Determinant d = walk.d;
    Determinant dcopy = walk.d;

    if (I == A)
    {
      if (d.getocc(I))
        gradRatio[index] += factor;
      continue;
    }

    if (!d.getocc(I))
      continue;
    dcopy.setocc(I, false);
    if (dcopy.getocc(A))
      continue;
    dcopy.setocc(A, true);

    bool doparity = false;
    double JastrowFactor = getJastrowFactor(I / 2, A / 2, dcopy, d);
    double ovlpdetcopy;

    if (I % 2 == 0)
      ovlpdetcopy = walk.getDetFactorA(I / 2, A / 2, *this, doparity) * JastrowFactor;
    else
      ovlpdetcopy = walk.getDetFactorB(I / 2, A / 2, *this, doparity) * JastrowFactor;

    gradRatio[index] += factor * ovlpdetcopy; //<n|a_i^dag a_a|Psi0>/<n|Psi0>
    
  }
  //exit(0);
}

//<psi_t| (H-E0) |D>
void CPSSlater::HamAndOvlpGradient(Walker &walk,
                                   double &ovlp, double &ham, VectorXd &grad,
                                   oneInt &I1, twoInt &I2,
                                   twoIntHeatBathSHM &I2hb, double &coreE,
                                   vector<double> &ovlpRatio, vector<size_t> &excitation1,
                                   vector<size_t> &excitation2, vector<double> &HijElements,
                                   int &nExcitations,
                                   bool doGradient, bool fillExcitations)
{
  //ovlpRatio.clear();
  //excitation1.clear();
  //excitation2.clear();
  int ovlpSize = ovlpRatio.size();
  nExcitations = 0;

  double TINY = schd.screen;
  double THRESH = schd.epsilon;
  //MatrixXd alphainv = walk.alphainv, betainv = walk.betainv;

  double detOverlap = walk.getDetOverlap(*this);
  Determinant &d = walk.d;

  int norbs = Determinant::norbs;
  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);

  size_t numJastrow = getNumJastrowVariables();
  VectorXd ciGrad0(ciExpansion.size());
  ciGrad0.setZero(); //this is <d|Psi_x>/<d|Psi> for x= ci-coeffs
  //noexcitation
  {
    double E0 = d.Energy(I1, I2, coreE);
    ovlp = detOverlap;
    //cout << ovlp <<endl;
    for (int i = 0; i < cpsArray.size(); i++)
    {
      ovlp *= cpsArray[i].Overlap(d);
      //cout << i<<"  "<<cpsArray[i].Overlap(d)<<endl;
    }
    ham = E0;

    //cout << E0<<"  "<<ovlp<<endl;exit(0);

    if (doGradient)
    {
      double factor = E0; //*detOverlap;
      OverlapWithGradient(walk, factor, grad);
      for (int i = 0; i < ciExpansion.size(); i++)
      {
        ciGrad0(i) = walk.alphaDet[i] * walk.betaDet[i] / detOverlap;
        grad(numJastrow + i) += E0 * ciGrad0(i);
      }
    }
  }
  //cout << ham<<endl;

  //Single alpha-beta excitation
  {
    double time = getTime();
    for (int i = 0; i < closed.size(); i++)
    {
      for (int a = 0; a < open.size(); a++)
      {
        if (closed[i] % 2 == open[a] % 2 && abs(I2hb.Singles(closed[i], open[a])) > THRESH)
        {
          prof.SinglesCount++;

          int I = closed[i] / 2, A = open[a] / 2;
          double tia = 0;
          Determinant dcopy = d;
          bool Alpha = closed[i] % 2 == 0 ? true : false;

          bool doparity = true;
          if (schd.Hamiltonian == HUBBARD)
          {
            tia = I1(2 * A, 2 * I);
            doparity = false;
          }
          else
          {
            tia = I1(2 * A, 2 * I);
            int X = max(I, A), Y = min(I, A);
            int pairIndex = X * (X + 1) / 2 + Y;
            size_t start = I2hb.startingIndicesSingleIntegrals[pairIndex];
            size_t end = I2hb.startingIndicesSingleIntegrals[pairIndex + 1];
            float *integrals = I2hb.singleIntegrals;
            short *orbIndices = I2hb.singleIntegralsPairs;
            for (size_t index = start; index < end; index++)
            {
              if (fabs(integrals[index]) < TINY)
                break;
              int j = orbIndices[2 * index];
              if (closed[i] % 2 == 1 && j % 2 == 1)
                j--;
              else if (closed[i] % 2 == 1 && j % 2 == 0)
                j++;

              if (d.getocc(j))
              {
                tia += integrals[index];
              }
              //cout << tia<<"  "<<a<<"  "<<integrals[index]<<endl;
            }
            double sgn = 1.0;
            if (Alpha)
              d.parityA(A, I, sgn);
            else
              d.parityB(A, I, sgn);
            tia *= sgn;
          }

          double localham = 0.0;
          if (abs(tia) > THRESH)
          {
            if (Alpha)
            {
              dcopy.setoccA(I, false);
              dcopy.setoccA(A, true);
            }
            else
            {
              dcopy.setoccB(I, false);
              dcopy.setoccB(A, true);
            }

            double JastrowFactor = getJastrowFactor(I, A, dcopy, d);
            double ovlpdetcopy;
            if (Alpha)
              ovlpdetcopy = walk.getDetFactorA(I, A, *this, doparity) * JastrowFactor;
            //localham += tia * walk.getDetFactorA(I, A, *this, doparity) * JastrowFactor;
            else
              ovlpdetcopy = walk.getDetFactorB(I, A, *this, doparity) * JastrowFactor;

            ham += ovlpdetcopy * tia;

            //double ovlpdetcopy = localham / tia;
            if (doGradient)
            {
              double factor = tia * ovlpdetcopy;
              OverlapWithGradient(dcopy, factor, grad);

              double parity = 1.0;
              int tableIndexi, tableIndexa;
              if (Alpha)
              {
                walk.d.parityA(A, I, parity);
                tableIndexi = std::lower_bound(walk.AlphaClosed.begin(), walk.AlphaClosed.end(), I) - walk.AlphaClosed.begin();
                tableIndexa = std::lower_bound(walk.AlphaOpen.begin(), walk.AlphaOpen.end(), A) - walk.AlphaOpen.begin();
              }
              else
              {
                walk.d.parityB(A, I, parity);
                tableIndexi = std::lower_bound(walk.BetaClosed.begin(), walk.BetaClosed.end(), I) - walk.BetaClosed.begin();
                tableIndexa = std::lower_bound(walk.BetaOpen.begin(), walk.BetaOpen.end(), A) - walk.BetaOpen.begin();
              }
              for (int i = 0; i < ciExpansion.size(); i++)
              {
                if (Alpha)
                {
                  grad(numJastrow + i) += tia * ciGrad0(i) * JastrowFactor * parity * walk.AlphaTable[i](tableIndexa, tableIndexi);
                }
                else
                {
                  grad(numJastrow + i) += tia * ciGrad0(i) * JastrowFactor * parity * walk.BetaTable[i](tableIndexa, tableIndexi);
                }
              }
            }

            if (fillExcitations)
            {
              if (ovlpSize <= nExcitations)
              {
                ovlpSize += 100000;
                //ovlpSize += 1000;
                ovlpRatio.resize(ovlpSize);
                excitation1.resize(ovlpSize);
                excitation2.resize(ovlpSize);
                HijElements.resize(ovlpSize);
              }

              ovlpRatio[nExcitations] = ovlpdetcopy;
              excitation1[nExcitations] = closed[i] * 2 * norbs + open[a];
              excitation2[nExcitations] = 0;
              HijElements[nExcitations] = tia;
              nExcitations++;

              //ovlpRatio  .push_back(ovlpdetcopy);
              //excitation1.push_back(closed[i] * 2 * norbs + open[a]);
              //excitation2.push_back(0);
              //HijElements.push_back(tia);
            }
          }
        }
      }
    }
    prof.SinglesTime += getTime() - time;
  }

  if (schd.Hamiltonian == HUBBARD)
    return;

  //Double excitation
  {
    double time = getTime();

    int nclosed = closed.size();
    for (int ij = 0; ij < nclosed * nclosed; ij++)
    {
      int i = ij / nclosed, j = ij % nclosed;
      if (i <= j)
        continue;
      int I = closed[i] / 2, J = closed[j] / 2;
      int X = max(I, J), Y = min(I, J);

      int pairIndex = X * (X + 1) / 2 + Y;
      size_t start = closed[i] % 2 == closed[j] % 2 ? I2hb.startingIndicesSameSpin[pairIndex] : I2hb.startingIndicesOppositeSpin[pairIndex];
      size_t end = closed[i] % 2 == closed[j] % 2 ? I2hb.startingIndicesSameSpin[pairIndex + 1] : I2hb.startingIndicesOppositeSpin[pairIndex + 1];
      float *integrals = closed[i] % 2 == closed[j] % 2 ? I2hb.sameSpinIntegrals : I2hb.oppositeSpinIntegrals;
      short *orbIndices = closed[i] % 2 == closed[j] % 2 ? I2hb.sameSpinPairs : I2hb.oppositeSpinPairs;

      // for all HCI integrals
      for (size_t index = start; index < end; index++)
      {
        // if we are going below the criterion, break
        if (fabs(integrals[index]) < THRESH)
          break;

        // otherwise: generate the determinant corresponding to the current excitation
        int a = 2 * orbIndices[2 * index] + closed[i] % 2, b = 2 * orbIndices[2 * index + 1] + closed[j] % 2;
        if (!(d.getocc(a) || d.getocc(b)))
        {
          prof.DoubleCount++;
          Determinant dcopy = d;
          double localham = 0.0;
          double tiajb = integrals[index];

          dcopy.setocc(closed[i], false);
          dcopy.setocc(a, true);
          dcopy.setocc(closed[j], false);
          dcopy.setocc(b, true);

          int A = a / 2, B = b / 2;
          int type = 0; //0 = AA, 1 = BB, 2 = AB, 3 = BA
          double JastrowFactor = getJastrowFactor(I, J, A, B, dcopy, d);
          if (closed[i] % 2 == closed[j] % 2 && closed[i] % 2 == 0)
            localham += tiajb * walk.getDetFactorA(I, J, A, B, *this, false) * JastrowFactor;
          else if (closed[i] % 2 == closed[j] % 2 && closed[i] % 2 == 1)
            localham += tiajb * walk.getDetFactorB(I, J, A, B, *this, false) * JastrowFactor;
          else if (closed[i] % 2 != closed[j] % 2 && closed[i] % 2 == 0)
            localham += tiajb * walk.getDetFactorAB(I, J, A, B, *this, false) * JastrowFactor;
          else
            localham += tiajb * walk.getDetFactorAB(J, I, B, A, *this, false) * JastrowFactor;

          ham += localham;

          double ovlpdetcopy = localham / tiajb;
          if (doGradient)
          {
            double factor = tiajb * ovlpdetcopy;
            OverlapWithGradient(dcopy, factor, grad);
            bool Alpha1 = closed[i] % 2 == 0, Alpha2 = closed[j] % 2 == 0;

            int tableIndexi, tableIndexa, tableIndexj, tableIndexb;
            if (Alpha1)
            {
              tableIndexi = std::lower_bound(walk.AlphaClosed.begin(), walk.AlphaClosed.end(), I) - walk.AlphaClosed.begin();
              tableIndexa = std::lower_bound(walk.AlphaOpen.begin(), walk.AlphaOpen.end(), A) - walk.AlphaOpen.begin();
            }
            else
            {
              tableIndexi = std::lower_bound(walk.BetaClosed.begin(), walk.BetaClosed.end(), I) - walk.BetaClosed.begin();
              tableIndexa = std::lower_bound(walk.BetaOpen.begin(), walk.BetaOpen.end(), A) - walk.BetaOpen.begin();
            }
            if (Alpha2)
            {
              tableIndexj = std::lower_bound(walk.AlphaClosed.begin(), walk.AlphaClosed.end(), J) - walk.AlphaClosed.begin();
              tableIndexb = std::lower_bound(walk.AlphaOpen.begin(), walk.AlphaOpen.end(), B) - walk.AlphaOpen.begin();
            }
            else
            {
              tableIndexj = std::lower_bound(walk.BetaClosed.begin(), walk.BetaClosed.end(), J) - walk.BetaClosed.begin();
              tableIndexb = std::lower_bound(walk.BetaOpen.begin(), walk.BetaOpen.end(), B) - walk.BetaOpen.begin();
            }
            for (int i = 0; i < ciExpansion.size(); i++)
            {
              if (Alpha1 && Alpha2)
              {
                double factor = (walk.AlphaTable[i](tableIndexa, tableIndexi) * walk.AlphaTable[i](tableIndexb, tableIndexj) - walk.AlphaTable[i](tableIndexb, tableIndexi) * walk.AlphaTable[i](tableIndexa, tableIndexj));
                grad(numJastrow + i) += tiajb * ciGrad0(i) * JastrowFactor * factor;
              }
              else if (Alpha1 && !Alpha2)
              {
                double factor = walk.AlphaTable[i](tableIndexa, tableIndexi) * walk.BetaTable[i](tableIndexb, tableIndexj);
                grad(numJastrow + i) += tiajb * ciGrad0(i) * JastrowFactor * factor;
              }
              else if (!Alpha1 && Alpha2)
              {
                double factor = walk.BetaTable[i](tableIndexa, tableIndexi) * walk.AlphaTable[i](tableIndexb, tableIndexj);
                grad(numJastrow + i) += tiajb * ciGrad0(i) * JastrowFactor * factor;
              }
              else
              {
                double factor = (walk.BetaTable[i](tableIndexa, tableIndexi) * walk.BetaTable[i](tableIndexb, tableIndexj) - walk.BetaTable[i](tableIndexb, tableIndexi) * walk.BetaTable[i](tableIndexa, tableIndexj));
                grad(numJastrow + i) += tiajb * ciGrad0(i) * JastrowFactor * factor;
              }
            }
          }

          if (fillExcitations)
          {
            if (ovlpSize <= nExcitations)
            {
              ovlpSize += 100000;
              ovlpRatio.resize(ovlpSize);
              excitation1.resize(ovlpSize);
              excitation2.resize(ovlpSize);
              HijElements.resize(ovlpSize);
            }

            ovlpRatio[nExcitations] = ovlpdetcopy;
            excitation1[nExcitations] = closed[i] * 2 * norbs + a;
            excitation2[nExcitations] = closed[j] * 2 * norbs + b;
            HijElements[nExcitations] = tiajb;
            nExcitations++;

            //ovlpRatio.push_back(ovlpdetcopy);
            //excitation1.push_back(closed[i] * 2 * norbs + a);
            //excitation2.push_back(closed[j] * 2 * norbs + b);
            //HijElements.push_back(tiajb);
          }
        }
      }
    }
    prof.DoubleTime += getTime() - time;
  }
}

void CPSSlater::HamAndOvlpStochastic(Walker &walk,
                                     double &ovlp, double &ham,
                                     oneInt &I1, twoInt &I2,
                                     twoIntHeatBathSHM &I2hb, double &coreE,
                                     int nterms,
                                     vector<Walker> &returnWalker,
                                     vector<double> &coeffWalker,
                                     bool fillWalker)
{
  /*
  double TINY = schd.screen;
  double THRESH = schd.epsilon;

  MatrixXd alphainv = walk.alphainv, betainv = walk.betainv;
  double alphaDet = walk.alphaDet, betaDet = walk.betaDet;
  double detOverlap = alphaDet*betaDet;
  ovlp = detOverlap;
  for (int i=0; i<cpsArray.size(); i++)
    ovlp *= cpsArray[i].Overlap(walk.d);

  {
    double E0 = walk.d.Energy(I1, I2, coreE);
    ham  += E0;

    if (fillWalker) {
      returnWalker.push_back(walk);
      coeffWalker .push_back(E0);
    }
  }


  vector<int> Isingle, Asingle;
  vector<int> Idouble, Adouble,
    Jdouble, Bdouble;

  vector<double> psingle,
    pdouble;


  sampleSingleDoubleExcitation(walk.d, I1, I2, I2hb,
			       nterms,
			       Isingle, Asingle,
			       Idouble, Adouble,
			       Jdouble, Bdouble,
			       psingle,
			       pdouble);


  //cout << Isingle.size()<<"  "<<Idouble.size()<<"  ";

  //Calculate the contribution to the matrix through the single
  double HijSingle = 0;
  int nsingles = Isingle.size();
  for (int ns=0; ns<nsingles; ns++)
  {
    Determinant dcopy = walk.d;
    dcopy.setocc(Isingle[ns], false); dcopy.setocc(Asingle[ns], true);
    int I = Isingle[ns]/2, A = Asingle[ns]/2;

    HijSingle = Isingle[ns]%2 == 0 ?
      walk.d.Hij_1ExciteA(A, I, I1, I2)*
      OverlapA(walk.d, I, A, alphainv, betainv) :

      walk.d.Hij_1ExciteB(A, I, I1, I2)*
      OverlapB(walk.d, I, A, alphainv, betainv) ;

    for (int n=0; n<cpsArray.size(); n++)
      if (std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), I) ||
	  std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), A) )
  	HijSingle *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(walk.d);

    ham += HijSingle/psingle[ns]/nterms;

    if (fillWalker) {
      Walker wcopy = walk;
      if (Isingle[ns]%2 == 0) wcopy.updateA(Isingle[ns]/2, Asingle[ns]/2, *this);
      else                    wcopy.updateB(Isingle[ns]/2, Asingle[ns]/2, *this);

      returnWalker.push_back(wcopy);
      coeffWalker .push_back(HijSingle/psingle[ns]/nterms);
    }
  }



  int ndoubles = Idouble.size();
  //Calculate the contribution to the matrix through the double
  for (int nd=0; nd<ndoubles; nd++) {
    double Hijdouble = (I2(Adouble[nd], Idouble[nd], Bdouble[nd], Jdouble[nd])
			- I2(Adouble[nd], Jdouble[nd], Bdouble[nd], Idouble[nd]));

    Determinant dcopy = walk.d;
    dcopy.setocc(Idouble[nd], false); dcopy.setocc(Adouble[nd], true);
    dcopy.setocc(Jdouble[nd], false); dcopy.setocc(Bdouble[nd], true);

    int I = Idouble[nd]/2, J = Jdouble[nd]/2, A = Adouble[nd]/2, B = Bdouble[nd]/2;

    if      (Idouble[nd]%2==Jdouble[nd]%2 && Idouble[nd]%2 == 0)
      Hijdouble *= det.OverlapAA(walk.d, I, J, A, B,  alphainv, betainv, false);

    else if (Idouble[nd]%2==Jdouble[nd]%2 && Idouble[nd]%2 == 1)
      Hijdouble *= det.OverlapBB(walk.d, I, J, A, B,  alphainv, betainv, false);

    else if (Idouble[nd]%2!=Jdouble[nd]%2 && Idouble[nd]%2 == 0)
      Hijdouble *= det.OverlapAB(walk.d, I, J, A, B,  alphainv, betainv, false);

    else
      Hijdouble *= det.OverlapAB(walk.d, J, I, B, A,  alphainv, betainv, false);

    for (int n=0; n<cpsArray.size(); n++)
      for (int j = 0; j<cpsArray[n].asites.size(); j++) {
	      if (cpsArray[n].asites[j] == I ||
	          cpsArray[n].asites[j] == J ||
	          cpsArray[n].asites[j] == A ||
	          cpsArray[n].asites[j] == B) {
	        Hijdouble *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(walk.d);
	        //localham *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(d);
        	break;
        }
      }

    ham += Hijdouble/pdouble[nd]/nterms;

    if (fillWalker) {
      Walker wcopy = walk;
      if (Idouble[nd]%2 == 0) wcopy.updateA(Idouble[nd]/2, Adouble[nd]/2, *this);
      else                wcopy.updateB(Idouble[nd]/2, Adouble[nd]/2, *this);
      if (Jdouble[nd]%2 == 0) wcopy.updateA(Jdouble[nd]/2, Bdouble[nd]/2, *this);
      else                wcopy.updateB(Jdouble[nd]/2, Bdouble[nd]/2, *this);

      returnWalker.push_back(wcopy);
      coeffWalker .push_back(Hijdouble/pdouble[nd]/nterms);
    }
  }
*/
}

void CPSSlater::PTcontribution2ndOrder(Walker &walk, double &E0,
                                       oneInt &I1, twoInt &I2,
                                       twoIntHeatBathSHM &I2hb, double &coreE,
                                       double &Aterm, double &Bterm, double &C,
                                       vector<double> &ovlpRatio, vector<size_t> &excitation1,
                                       vector<size_t> &excitation2, bool doGradient)
{
  /*
  auto random = std::bind(std::uniform_real_distribution<double>(0,1),
			  std::ref(generator));

  double TINY = schd.screen;
  double THRESH = schd.epsilon;


  double ovlp=0, ham=0;
  VectorXd grad;
  vector<Walker> firstRoundDets; vector<double> firstRoundCi; bool fillWalker = true;
  size_t nterms=schd.integralSampleSize;
  double coreEtmp = coreE - E0;


  firstRoundDets.push_back(walk); firstRoundCi.push_back(walk.d.Energy(I1, I2, coreEtmp));

  //Here we first calculate the loca energy of walker and then use that to select
  //important elements of V
  double inputWalkerHam = 0;
  {
    vector<double> HijElements;
    HamAndOvlpGradient(walk, ovlp, inputWalkerHam, grad, I1, I2, I2hb, coreEtmp,
      ovlpRatio, excitation1, excitation2, HijElements, doGradient);


    std::vector<size_t> idx(HijElements.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&HijElements](size_t i1, size_t i2)
    {return abs(HijElements[i1])<abs(HijElements[i2]);});

      //pick the first nterms-1 most important integrals
    for (int i=0; i<min(nterms-1, HijElements.size()); i++) {
      int I = idx[HijElements.size()-i-1] ; //the largest Hij matrix elements

      firstRoundDets.push_back(walk);
      firstRoundDets.rbegin()->exciteWalker(*this, excitation1[I], excitation2[I], Determinant::norbs);
      firstRoundCi.push_back(HijElements[I]*ovlpRatio[I]);

        //zero out the integral
      HijElements[I] = 0;
    }

      //include the last term stochastically
    if (HijElements.size() > nterms-1) {

      double cumHij = 0;
      vector<double> cumHijList(HijElements.size(), 0);
      for (int i=0; i<HijElements.size(); i++) {
        cumHij += abs(HijElements[i]);
        cumHijList[i] = cumHij;
      }
      int T  = std::lower_bound(cumHijList.begin(), cumHijList.end(),
      random()*cumHij) - cumHijList.begin();
      firstRoundDets.push_back(walk);
      firstRoundDets.rbegin()->exciteWalker(*this, excitation1[T], excitation2[T], Determinant::norbs);
      //exciteWalker(*firstRoundDets.rbegin(), excitation1[T], excitation2[T], Determinant::norbs);
      firstRoundCi.push_back(cumHij*ovlpRatio[T]*abs(HijElements[T])/HijElements[T]);
    }
  }


  //HamAndOvlpGradientStochastic(walk, ovlp, ham, grad, I1, I2, I2hb, coreEtmp,
  //nterms,
  //firstRoundDets, firstRoundCi, fillWalker);

  double Eidet = walk.d.Energy(I1, I2, coreE);
  C = 1.0/(Eidet-E0);
  Aterm = 0.0;

  for (int dindex =0; dindex<firstRoundDets.size(); dindex++) {
    double ovlp=0, ham=0;
    VectorXd grad;
    double Eidet = firstRoundDets[dindex].d.Energy(I1, I2, coreE);


    if (dindex == 0)  {
      ham = inputWalkerHam;
      //vector<double> HijElements;
      //HamAndOvlpGradient(firstRoundDets[dindex], ovlp, ham, grad, I1, I2, I2hb, coreEtmp,
      //ovlpRatio, excitation1, excitation2, HijElements, doGradient);
      Bterm = ham/(Eidet-E0);
    }
    else {
      vector<double> HijElements;
      vector<double> ovlpRatio; vector<size_t> excitation1, excitation2;
      HamAndOvlpGradient(firstRoundDets[dindex], ovlp, ham, grad, I1, I2, I2hb, coreEtmp,
			 ovlpRatio, excitation1, excitation2, HijElements, doGradient, false);
    }

    Aterm -= (ham)*firstRoundCi[dindex]/(Eidet - E0);
    //cout << firstRoundCi[dindex]<<"  "<<firstRoundDets[dindex].d<<"  "<<ham<<"  "<<Aterm<<endl;
  }
*/
}

void CPSSlater::PTcontribution3rdOrder(Walker &walk, double &E0,
                                       oneInt &I1, twoInt &I2,
                                       twoIntHeatBathSHM &I2hb, double &coreE,
                                       double &Aterm2, double &Bterm, double &C, double &Aterm3,
                                       vector<double> &ovlpRatio, vector<size_t> &excitation1,
                                       vector<size_t> &excitation2, bool doGradient)
{

  /*
  double TINY = schd.screen;
  double THRESH = schd.epsilon;


  double ovlp=0, ham=0;
  VectorXd grad;
  vector<Walker> firstRoundDets; vector<double> firstRoundCi; bool fillWalker = true;
  int nterms=schd.integralSampleSize;

  double coreEtmp = coreE - E0;
  HamAndOvlpStochastic(walk, ovlp, ham, I1, I2, I2hb, coreEtmp,
			       nterms,
			       firstRoundDets, firstRoundCi, fillWalker);

  double Eidet = walk.d.Energy(I1, I2, coreE);
  C = 1.0/(Eidet-E0);
  Aterm2 = 0.0; Aterm3 = 0.0;
  double sgn = ovlp/abs(ovlp);

  for (int dindex1 =0; dindex1<firstRoundDets.size(); dindex1++) {
    double Eidet = firstRoundDets[dindex1].d.Energy(I1, I2, coreE);
    firstRoundCi[dindex1] /= (Eidet- E0);

    vector<Walker> secondRoundDets; vector<double> secondRoundCi; bool fillWalker = true;
    HamAndOvlpStochastic(firstRoundDets[dindex1], ovlp, ham, I1, I2, I2hb, coreEtmp,
				 nterms, secondRoundDets, secondRoundCi, fillWalker);


    for (int dindex2 =0; dindex2<secondRoundDets.size(); dindex2++) {
      double ovlp=0, ham=0;
      double Eidet = secondRoundDets[dindex2].d.Energy(I1, I2, coreE);
      secondRoundCi[dindex2] *= firstRoundCi[dindex1] / (Eidet- E0);


      if (dindex2 == 0 && dindex1 == 0)  {
	vector<double> HijElements;
	HamAndOvlpGradient(secondRoundDets[dindex2], ovlp, ham, grad, I1, I2, I2hb, coreEtmp,
			   ovlpRatio, excitation1, excitation2, HijElements, doGradient);
	Bterm = ham/(Eidet-E0);
      }
      else {
	vector<double> HijElements;
	vector<double> ovlpRatio; vector<size_t> excitation1, excitation2;
	HamAndOvlpGradient(secondRoundDets[dindex2], ovlp, ham, grad, I1, I2, I2hb, coreEtmp,
			   ovlpRatio, excitation1, excitation2, HijElements, doGradient, false);
      }

      if (dindex2 == 0)
	Aterm2 -= (ham)*firstRoundCi[dindex1];

      Aterm3 -= (ham)*secondRoundCi[dindex2];
    }
  }
  */
}
