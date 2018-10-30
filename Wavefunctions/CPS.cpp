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
#include "input.h"

using namespace Eigen;

CPS::CPS () {    

  twoSiteOrSmaller = true;
  for(const auto& p : schd.correlatorFiles) {
    if (p.first > 2) twoSiteOrSmaller = false;
    readCorrelator(p, this->cpsArray);
  }
  
  generateMapFromOrbitalToCorrelators();
};

CPS::CPS (std::vector<Correlator>& pcpsArray) : cpsArray(pcpsArray) {
  twoSiteOrSmaller = true;
  for (const auto& p : pcpsArray)
    if (p.asites.size() > 2) twoSiteOrSmaller = false;
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

  for (int i = 0; i < norbs; i++)
    std::sort(mapFromOrbitalToCorrelator[i].begin(),
              mapFromOrbitalToCorrelator[i].end());
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
  
  merge(mapFromOrbitalToCorrelator[i].begin(),
        mapFromOrbitalToCorrelator[i].end(),
        mapFromOrbitalToCorrelator[a].begin(),
        mapFromOrbitalToCorrelator[a].end(),
        back_inserter(common));


  sort(common.begin(), common.end() );

  double ovlp = 1.0;
  int previ = -1;
  for (const auto& i : common)
    if (i != previ) {
      ovlp *= cpsArray[i].OverlapRatio(dcopy,d);
      previ = i;
    }

  return ovlp;
}

double CPS::OverlapRatio(int i, int j, int a, int b, const Determinant &dcopy, const Determinant &d) const
{
  //boost::container::static_vector<int, 100> common;
  vector<int>& common = const_cast<vector<int>&>(commonCorrelators);
  common.resize(0);
  
  merge(mapFromOrbitalToCorrelator[i].begin(),
        mapFromOrbitalToCorrelator[i].end(),
        mapFromOrbitalToCorrelator[a].begin(),
        mapFromOrbitalToCorrelator[a].end(),
        back_inserter(common));

  int middlesize = common.size();  
  merge(mapFromOrbitalToCorrelator[j].begin(),
        mapFromOrbitalToCorrelator[j].end(),
        mapFromOrbitalToCorrelator[b].begin(),
        mapFromOrbitalToCorrelator[b].end(),
        back_inserter(common));


  //The following is an ugly code but is used because it is faster than the alternative.
  //Essentially common has elements 0...middlesize sorted and elements
  //middlesize..end sorted as well. The loop below find unique elements
  //in the entire common vector.
  double ovlp = 1.0;
  int previ = -1;
  for (int i=0,j=middlesize; i<middlesize || j<common.size(); ) {
    if (common[i] < common[j] ) {
      if (common[i] <= previ) 
        i++;
      else {
        int I = common[i];
        ovlp *= cpsArray[I].OverlapRatio(dcopy,d);
        i++; previ = I;
      }
    }
    else {
      if (common[j] <= previ) 
        j++;
      else {
        int J = common[j];
        ovlp *= cpsArray[J].OverlapRatio(dcopy,d);
        j++; previ = J;
      }     
    }
  }
  return ovlp;

}


double CPS::OverlapRatio(int i, int a, const BigDeterminant &dcopy, const BigDeterminant &d) const
{
  //boost::container::static_vector<int, 100> commonCorrelators;
  vector<int>& common = const_cast<vector<int>&>(commonCorrelators);
  common.resize(0);
  
  merge(mapFromOrbitalToCorrelator[i].begin(),
        mapFromOrbitalToCorrelator[i].end(),
        mapFromOrbitalToCorrelator[a].begin(),
        mapFromOrbitalToCorrelator[a].end(),
        back_inserter(common));


  sort(common.begin(), common.end() );

  double ovlp = 1.0;
  int previ = -1;
  for (const auto& i : common)
    if (i != previ) {
      ovlp *= cpsArray[i].OverlapRatio(dcopy,d);
      previ = i;
    }

  return ovlp;

}

double CPS::OverlapRatio(int i, int j, int a, int b, const BigDeterminant &dcopy, const BigDeterminant &d) const
{
  //boost::container::static_vector<int, 100> common;
  vector<int>& common = const_cast<vector<int>&>(commonCorrelators);
  common.resize(0);
  
  merge(mapFromOrbitalToCorrelator[i].begin(),
        mapFromOrbitalToCorrelator[i].end(),
        mapFromOrbitalToCorrelator[a].begin(),
        mapFromOrbitalToCorrelator[a].end(),
        back_inserter(common));

  int middlesize = common.size();
  merge(mapFromOrbitalToCorrelator[j].begin(),
        mapFromOrbitalToCorrelator[j].end(),
        mapFromOrbitalToCorrelator[b].begin(),
        mapFromOrbitalToCorrelator[b].end(),
        back_inserter(common));

  //The following is an ugly code but is used because it is faster than the alternative.
  //Essentially common has elements 0...middlesize sorted and elements
  //middlesize..end sorted as well. The loop below find unique elements
  //in the entire common vector.
  double ovlp = 1.0;
  int previ = -1;
  for (int i=0,j=middlesize; i<middlesize || j<common.size(); ) {
    if (common[i] < common[j] ) {
      if (common[i] <= previ) 
        i++;
      else {
        int I = common[i];
        ovlp *= cpsArray[I].OverlapRatio(dcopy,d);
        i++; previ = I;
      }
    }
    else {
      if (common[j] <= previ) 
        j++;
      else {
        int J = common[j];
        ovlp *= cpsArray[J].OverlapRatio(dcopy,d);
        j++; previ = J;
      }     
    }
  }

  return ovlp;

}

void CPS::OverlapWithGradient(const Determinant& d, 
                              VectorXd& grad,
                              const double& ovlp) const {
  
  if (schd.optimizeCps) {
    long startIndex = 0;
    for (const auto& c : cpsArray) {
      c.OverlapWithGradient(d, grad, ovlp, startIndex);
      startIndex += c.Variables.size();
    }
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
