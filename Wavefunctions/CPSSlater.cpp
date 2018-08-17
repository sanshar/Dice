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

#include "integral.h"
#include "CPS.h"
#include "CPSSlaterWalker.h"
#include "CPSSlater.h"
#include "global.h"
#include "input.h"
#include "Profile.h"
#include <fstream>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif

using namespace Eigen;

CPSSlater::CPSSlater() {}

void CPSSlater::read() {
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

  HforbsA = MatrixXd::Zero(norbs, norbs);
  HforbsB = MatrixXd::Zero(norbs, norbs);
  readHF(HforbsA, HforbsB, schd.uhf);

  //Setup CPS wavefunctions
  std::vector<Correlator> nSiteCPS;
  for (auto it = schd.correlatorFiles.begin(); it != schd.correlatorFiles.end();
       it++)
  {
    readCorrelator(it->second, it->first, cpsArray);
  }

  //vector<Determinant> detList;
  //vector<double> ciExpansion;

  if (boost::iequals(schd.determinantFile, ""))
  {
    determinants.resize(1);
    ciExpansion.resize(1, 1.0);
    for (int i = 0; i < nalpha; i++)
      determinants[0].setoccA(i, true);
    for (int i = 0; i < nbeta; i++)
      determinants[0].setoccB(i, true);
  }
  else
  {
    readDeterminants(schd.determinantFile, determinants, ciExpansion);
  }

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

}

void CPSSlater::initWalker(CPSSlaterWalker& walk) {

  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

	//initialize the walker
	Determinant& d = walk.d;
	bool readDeterminant = false;
	char file[5000];

	sprintf(file, "BestDeterminant.txt");

	{
		ifstream ofile(file);
		if (ofile)
			readDeterminant = true;
	}
	//readDeterminant = false;

	if (!readDeterminant)
	{

		for (int i = 0; i < nalpha; i++)
		{
			int bestorb = 0;
			double maxovlp = 0;
			for (int j = 0; j < norbs; j++)
			{
				if (abs(HforbsA(i, j)) > maxovlp && !d.getoccA(j))
				{
					maxovlp = abs(HforbsA(i, j));
					bestorb = j;
				}
			}
			d.setoccA(bestorb, true);
		}
		for (int i = 0; i < nbeta; i++)
		{
			int bestorb = 0;
			double maxovlp = 0;
			for (int j = 0; j < norbs; j++)
			{
				if (abs(HforbsB(i, j)) > maxovlp && !d.getoccB(j))
				{
					bestorb = j;
					maxovlp = abs(HforbsB(i, j));
				}
			}
			d.setoccB(bestorb, true);
		}
	}
	else
	{
		if (commrank == 0)
		{
			std::ifstream ifs(file, std::ios::binary);
			boost::archive::binary_iarchive load(ifs);
			load >> d;
		}
#ifndef SERIAL
		MPI_Bcast(&d.reprA, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&d.reprB, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
	}

	walk.initUsingWave(*this);
}

void CPSSlater::initWalker(CPSSlaterWalker& walk, Determinant& d) {

  walk.d = d;
	walk.initUsingWave(*this);
}

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

double CPSSlater::getOverlapWithDeterminants(CPSSlaterWalker &walk)
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

double CPSSlater::Overlap(CPSSlaterWalker &walk)
{
  double ovlp = 1.0;

  //Overlap with all the Correlators
  for (int i = 0; i < cpsArray.size(); i++)
  {
    ovlp *= cpsArray[i].Overlap(walk.d);
  }

  return ovlp * getOverlapWithDeterminants(walk);
}


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

void CPSSlater::OverlapWithGradient(CPSSlaterWalker &walk,
                                    double &ovlp,
                                    VectorXd &grad)
{
  double factor = 1.0;
  OverlapWithGradient(walk.d, factor, grad);

  int numJastrowVariables = getNumJastrowVariables();
  double detovlp = walk.getDetOverlap(*this);
  for (int k = 0; k < ciExpansion.size(); k++)
  {
    grad(k + numJastrowVariables) += walk.alphaDet[k] * walk.betaDet[k] / detovlp;
  }

  if (determinants.size() <= 1 && schd.optimizeOrbs)
  {
    walk.OverlapWithGradient(*this, grad, detovlp);
  }
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
  cout << "CPS"<<endl;
  for (int i = 0; i < cpsArray.size(); i++)
  {
    for (int j = 0; j < cpsArray[i].Variables.size(); j++)
    {
      cout << "  " << cpsArray[i].Variables[j];
      numVars++;
    }
  }

  cout << endl<<"CI-expansion"<<endl;
  for (int i = 0; i < determinants.size(); i++)
  {
    cout << "  " << ciExpansion[i];
  }

  cout << endl<<"DeterminantA"<<endl;
  int norbs = Determinant::norbs;
  for (int i = 0; i < norbs; i++)
    for (int j = 0; j < norbs; j++)
      cout << "  " << HforbsA(i, j);

  if (schd.uhf)
  {
    cout << endl
         << "DeterminantB" << endl;
    for (int i = 0; i < norbs; i++)
      for (int j = 0; j < norbs; j++)
        cout << "  " << HforbsB(i, j);
  }

  cout << endl;
}

void CPSSlater::updateVariables(Eigen::VectorXd &v)
{
  int norbs = Determinant::norbs;

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

  for (int i = 0; i < norbs; i++)
  {
    for (int j = 0; j < norbs; j++)
    {
      if (!schd.uhf)
      {
        HforbsA(i, j) = v[numVars + i * norbs + j];
        HforbsB(i, j) = v[numVars + i * norbs + j];
      }
      else
      {
        HforbsA(i, j) = v[numVars + i * norbs + j];
        HforbsB(i, j) = v[numVars + norbs * norbs + i * norbs + j];
      }
    }
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
  int norbs = Determinant::norbs;
  if (v.rows() != getNumVariables())
  {
    v = VectorXd::Zero(getNumVariables());
  }
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

  for (int i = 0; i < norbs; i++)
  {
    for (int j = 0; j < norbs; j++)
    {
      if (!schd.uhf)
      {
        v[numVars + i * norbs + j] = HforbsA(i, j);
        v[numVars + i * norbs + j] = HforbsB(i, j);
      }
      else
      {
        v[numVars + i * norbs + j] = HforbsA(i, j);
        v[numVars + norbs * norbs + i * norbs + j] = HforbsB(i, j);
      }
    }
  }
}

long CPSSlater::getNumVariables()
{
  int norbs = Determinant::norbs;
  long numVars = 0;
  for (int i = 0; i < cpsArray.size(); i++)
    numVars += cpsArray[i].Variables.size();

  numVars += determinants.size();
  if (schd.uhf)
    numVars += 2 * norbs * norbs;
  else
    numVars += norbs * norbs;

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
void CPSSlater::OvlpRatioCI(CPSSlaterWalker &walk, VectorXd &gradRatio,
                            oneInt &I1, twoInt &I2, vector<int> &SingleIndices,
                            vector<int>& DoubleIndices, twoIntHeatBathSHM &I2hb, 
			    double &coreE, double factor)
{
  //<d|I^dag A |Psi0>
  //|dcopy> = A^dag I|d>
  int norbs = Determinant::norbs;
  gradRatio[0] += factor;
  int index = 0;
  for (int i = 0; i < SingleIndices.size() / 2; i++)
  {
    int I = SingleIndices[2 * i], A = SingleIndices[2 * i + 1];

    Determinant d = walk.d;
    Determinant dcopy = walk.d;
    index ++;

    if (!d.getocc(I))
      continue;
    dcopy.setocc(I, false);
    if (dcopy.getocc(A))
      continue;
    dcopy.setocc(A, true);

    if (d == dcopy)  {
      gradRatio[index] += factor;
      continue;
    }

    bool doparity = false;
    double JastrowFactor = getJastrowFactor(I / 2, A / 2, dcopy, d);
    double ovlpdetcopy;

    if (I % 2 == 0)
      ovlpdetcopy = walk.getDetFactorA(I / 2, A / 2, *this, doparity) * JastrowFactor;
    else
      ovlpdetcopy = walk.getDetFactorB(I / 2, A / 2, *this, doparity) * JastrowFactor;

    gradRatio[index] += factor * ovlpdetcopy; //<n|a_i^dag a_a|Psi0>/<n|Psi0>
  }

  //<n|  J^dag  B I^dag A|Psi0>/<n|Psi0>
  for (int i = 0; i < DoubleIndices.size() / 4; i++)
  {
    int I = DoubleIndices[4 * i],     A = DoubleIndices[4 * i + 2];
    int J = DoubleIndices[4 * i + 1], B = DoubleIndices[4 * i + 3];

    index++;

    Determinant d = walk.d;
    Determinant dcopy = walk.d;

    if (!d.getocc(J))
      continue;
    dcopy.setocc(J, false);
    if (dcopy.getocc(B))
      continue;
    dcopy.setocc(B, true);

    if (!d.getocc(I))
      continue;
    dcopy.setocc(I, false);
    if (dcopy.getocc(A))
      continue;
    dcopy.setocc(A, true);

    if (d == dcopy)  {
      gradRatio[index] += factor;
      continue;
    }

    bool doparity = false;
    double JastrowFactor = getJastrowFactor(I / 2, J / 2, A / 2, B / 2, dcopy, d);
    double ovlpdetcopy;

    if (I % 2 == J % 2 && I % 2 == 0)
      ovlpdetcopy = walk.getDetFactorA(I / 2, J / 2, A / 2, B / 2, *this, doparity) * JastrowFactor;
    else if (I % 2 == J % 2 && I % 2 == 1)
      ovlpdetcopy = walk.getDetFactorB(I / 2, J / 2, A / 2, B / 2, *this, doparity) * JastrowFactor;
    else if (I % 2 != J % 2 && I % 2 == 0)
      ovlpdetcopy = walk.getDetFactorAB(I / 2, J / 2, A / 2, B / 2, *this, doparity) * JastrowFactor;
    else
      ovlpdetcopy = walk.getDetFactorAB(J / 2, I / 2, B / 2, A / 2, *this, doparity) * JastrowFactor;

    gradRatio[index] += factor * ovlpdetcopy; //<n|a_i^dag a_a|Psi0>/<n|Psi0>

  }
  //exit(0);
}

//<psi_t| (H-E0) |D>
void CPSSlater::HamAndOvlp(CPSSlaterWalker &walk,
                           double &ovlp, double &ham, 
                           vector<double> &ovlpRatio, vector<size_t> &excitation1,
                           vector<size_t> &excitation2, vector<double> &HijElements,
                           int &nExcitations, bool fillExcitations)
{

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
    }
    ham = E0;

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

            if (fillExcitations)
            {
              if (ovlpSize <= nExcitations)
              {
                ovlpSize += 1000000;
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

          }
        }
      }
    }
    prof.DoubleTime += getTime() - time;
  }
}



void CPSSlater::derivativeOfLocalEnergy (CPSSlaterWalker &walk,
                                          double &factor, VectorXd& hamRatio)
{
  //NEEDS TO BE IMPLEMENTED
}


