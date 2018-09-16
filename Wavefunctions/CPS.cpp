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

CPS::CPS () {    

  for(const auto& p : schd.correlatorFiles) readCorrelator(p, this->cpsArray);
  
  generateMapFromOrbitalToCorrelators();
};

CPS::CPS (std::vector<Correlator>& pcpsArray) : cpsArray(pcpsArray) {
  generateMapFromOrbitalToCorrelators();
};


void CPS::generateMapFromOrbitalToCorrelators() {

  for (int i = 0; i < cpsArray.size(); i++)
  {
    for (int j = 0; j < cpsArray[i].asites.size(); j++) {
      int orbital = cpsArray[i].asites[j];
      mapFromOrbitalToCorrelator[orbital].push_back(i);
    }
  } 
}

double CPS::Overlap(const Determinant &d) const
{
  double ovlp = 1.0;
  for (const auto& c : cpsArray) ovlp *= c.Overlap(d);
  return ovlp;
}


double CPS::OverlapRatio (const Determinant &d1, const Determinant &d2) const {
  double ovlp = 1.0;
  for (const auto& c : cpsArray) ovlp *= c.Overlap(d1)/c.Overlap(d2);
  return ovlp;
}


double CPS::OverlapRatio(int i, int a, const Determinant &dcopy, const Determinant &d) const
{
  vector<int> commonCorrelators;
  
  copy(mapFromOrbitalToCorrelator.find(i)->second.begin(),
       mapFromOrbitalToCorrelator.find(i)->second.end(),
       back_inserter(commonCorrelators));
  copy(mapFromOrbitalToCorrelator.find(a)->second.begin(),
       mapFromOrbitalToCorrelator.find(a)->second.end(),
       back_inserter(commonCorrelators));


  sort(commonCorrelators.begin(), commonCorrelators.end() );
  commonCorrelators.erase( unique( commonCorrelators.begin(), commonCorrelators.end() ),
                           commonCorrelators.end() );

  double ovlp = 1.0;
  for (const auto& i : commonCorrelators)
    ovlp *= cpsArray[i].Overlap(dcopy)/cpsArray[i].Overlap(d);
  return ovlp;
}

double CPS::OverlapRatio(int i, int j, int a, int b, const Determinant &dcopy, const Determinant &d) const
{
  vector<int> commonCorrelators;
  
  copy(mapFromOrbitalToCorrelator.find(i)->second.begin(),
       mapFromOrbitalToCorrelator.find(i)->second.end(),
       back_inserter(commonCorrelators));
  copy(mapFromOrbitalToCorrelator.find(a)->second.begin(),
       mapFromOrbitalToCorrelator.find(a)->second.end(),
       back_inserter(commonCorrelators));

  copy(mapFromOrbitalToCorrelator.find(j)->second.begin(),
       mapFromOrbitalToCorrelator.find(j)->second.end(),
       back_inserter(commonCorrelators));
  copy(mapFromOrbitalToCorrelator.find(b)->second.begin(),
       mapFromOrbitalToCorrelator.find(b)->second.end(),
       back_inserter(commonCorrelators));


  sort(commonCorrelators.begin(), commonCorrelators.end() );
  commonCorrelators.erase( unique( commonCorrelators.begin(), commonCorrelators.end() ),
                           commonCorrelators.end() );
  
  double ovlp = 1.0;
  for (const auto& i : commonCorrelators)
    ovlp *= cpsArray[i].Overlap(dcopy)/cpsArray[i].Overlap(d);
  return ovlp;
}

void CPS::OverlapWithGradient(const Determinant& d, 
                              VectorXd& grad,
                              const double& ovlp) const {
  
  long startIndex = 0;
  for (const auto& c : cpsArray) {
    c.OverlapWithGradient(d, grad, ovlp, startIndex);
    startIndex += c.Variables.size();
  }
}

long CPS::getNumVariables() const
{
  long numVars = 0;
  return std::accumulate(cpsArray.begin(), cpsArray.end(), numVars,
                         [](long a, const Correlator& c)
                         {return a + c.Variables.size();}
                         );
}


void CPS::getVariables(Eigen::VectorXd &v) const
{
  int numVars = 0;
  for (const auto& c : cpsArray) {
    std::copy(c.Variables.begin(), c.Variables.end(), &v[numVars]);
    numVars+= c.Variables.size();
  }
}

void CPS::updateVariables(const Eigen::VectorXd &v)
{
  int numVars = 0;
  for (auto& c : cpsArray) {
    for (int j=0; j<c.Variables.size(); j++)
      c.Variables[j] = v[numVars+j];
    numVars+= c.Variables.size();
  }
}

void CPS::printVariables() const
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
