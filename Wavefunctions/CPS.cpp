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
#include <boost/container/static_vector.hpp>

using namespace Eigen;

CPS::CPS () {    

  for(const auto& p : schd.correlatorFiles) readCorrelator(p, this->cpsArray);
  
  generateMapFromOrbitalToCorrelators();
};

CPS::CPS (std::vector<Correlator>& pcpsArray) : cpsArray(pcpsArray) {
  generateMapFromOrbitalToCorrelators();
};


void CPS::generateMapFromOrbitalToCorrelators() {

  int norbs = Determinant::norbs;
  mapFromOrbitalToCorrelator.resize(norbs);
  
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
  //boost::container::static_vector<int, 100> commonCorrelators;
  vector<int>& common = const_cast<vector<int>&>(commonCorrelators);
  common.resize(0);
  
  copy(mapFromOrbitalToCorrelator[i].begin(),
       mapFromOrbitalToCorrelator[i].end(),
       back_inserter(common));
  copy(mapFromOrbitalToCorrelator[a].begin(),
       mapFromOrbitalToCorrelator[a].end(),
       back_inserter(common));


  sort(common.begin(), common.end() );
  common.erase( unique( common.begin(), common.end() ),
                           common.end() );

  double ovlp = 1.0;
  for (const auto& i : common)
    ovlp *= cpsArray[i].Overlap(dcopy)/cpsArray[i].Overlap(d);
  return ovlp;

}

double CPS::OverlapRatio(int i, int j, int a, int b, const Determinant &dcopy, const Determinant &d) const
{
  //boost::container::static_vector<int, 100> common;
  vector<int>& common = const_cast<vector<int>&>(commonCorrelators);
  common.resize(0);
  
  copy(mapFromOrbitalToCorrelator[i].begin(),
       mapFromOrbitalToCorrelator[i].end(),
       back_inserter(common));
  copy(mapFromOrbitalToCorrelator[a].begin(),
       mapFromOrbitalToCorrelator[a].end(),
       back_inserter(common));

  copy(mapFromOrbitalToCorrelator[j].begin(),
       mapFromOrbitalToCorrelator[j].end(),
       back_inserter(common));
  copy(mapFromOrbitalToCorrelator[b].begin(),
       mapFromOrbitalToCorrelator[b].end(),
       back_inserter(common));


  sort(common.begin(), common.end() );
  common.erase( unique( common.begin(), common.end() ),
                           common.end() );
  
  double ovlp = 1.0;
  for (const auto& i : common)
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
