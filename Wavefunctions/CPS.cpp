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
#include "CPS.h"
#include "Correlator.h"
#include "Determinants.h"


using namespace Eigen;


double CPS::Overlap(const Determinant &d)
{
  double ovlp = 1.0;

  //Overlap with all the Correlators
  for (int i = 0; i < cpsArray.size(); i++)
  {
    ovlp *= cpsArray[i].Overlap(d);
  }

  return ovlp;
}


double CPS::OverlapRatio (const Determinant &d1, const Determinant &d2) {
    double overlapRatio = 1.0;
    for (int i=0; i < cpsArray.size(); i++) {
        overlapRatio *= cpsArray[i].Overlap(d1)/cpsArray[i].Overlap(d2);
    };
    return overlapRatio;
}


double CPS::OverlapRatio(int i, int a, Determinant &dcopy, Determinant &d)
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

double CPS::OverlapRatio(int i, int j, int a, int b, Determinant &dcopy, Determinant &d)
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

void CPS::OverlapWithGradient(const Determinant& d, 
				     VectorXd& grad,
				     const double& ovlp) {
  
  long startIndex = 0;

  for (int i = 0; i < cpsArray.size(); i++)
  {
    cpsArray[i].OverlapWithGradient(d, grad,
                                    ovlp, startIndex);
    startIndex += cpsArray[i].Variables.size();
  }
}

long CPS::getNumVariables()
{
  long numVars = 0;
  for (int i = 0; i < cpsArray.size(); i++)
    numVars += cpsArray[i].Variables.size();

  return numVars;
}

void CPS::getVariables(Eigen::VectorXd &v)
{
  int numVars = 0;
  for (int i = 0; i < cpsArray.size(); i++)
  {
    for (int j = 0; j < cpsArray[i].Variables.size(); j++)
    {
      v[numVars] = cpsArray[i].Variables[j];
      numVars++;
    }
  }
}

void CPS::updateVariables(Eigen::VectorXd &v)
{
  int numVars = 0;
  for (int i = 0; i < cpsArray.size(); i++)
  {
    for (int j = 0; j < cpsArray[i].Variables.size(); j++)
    {
      cpsArray[i].Variables[j] = v[numVars];
      numVars++;
    }
  }
}

void CPS::printVariables()
{
  cout << "CPS"<< endl;
  for (int i = 0; i < cpsArray.size(); i++)
  {
    for (int j = 0; j < cpsArray[i].Variables.size(); j++)
    {
      cout << "  " << cpsArray[i].Variables[j] << endl;
    }
  }
}
