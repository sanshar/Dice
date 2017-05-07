/*
Developed by Sandeep Sharma with contributions from James E. Smith and Adam A. Homes, 2017
Copyright (c) 2017, Sandeep Sharma

This file is part of DICE.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef SHCI_MAKEHAMILTONIAN_H
#define SHCI_MAKEHAMILTONIAN_H
#include <vector>
#include <Eigen/Dense>
#include <set>
#include <list>
#include <tuple>
#include <map>
#include <boost/serialization/serialization.hpp>

using namespace std;
using namespace Eigen;
class Determinant;
class HalfDet;
class oneInt;
class twoInt;
class twoIntHeatBath;
class twoIntHeatBathSHM;
class schedule;

namespace SHCImakeHamiltonian {


  void regenerateH(std::vector<Determinant>& Dets,
		   std::vector<std::vector<int> >&connections,
		   std::vector<std::vector<CItype> >& Helements,
		   oneInt& I1,
		   twoInt& I2,
		   double& coreE);

  void PopulateHelperLists(std::map<HalfDet, std::vector<int> >& BetaN,
			   std::map<HalfDet, std::vector<int> >& AlphaNm1,
			   std::vector<Determinant>& Dets,
			   int StartIndex);


  void MakeHfromHelpers(std::map<HalfDet, std::vector<int> >& BetaN,
			std::map<HalfDet, std::vector<int> >& AlphaNm1,
			std::vector<Determinant>& Dets,
			int StartIndex,
			std::vector<std::vector<int> >&connections,
			std::vector<std::vector<CItype> >& Helements,
			int Norbs,
			oneInt& I1,
			twoInt& I2,
			double& coreE,
			std::vector<std::vector<size_t> >& orbDifference,
			bool DoRDM=false) ;

  void updateSOCconnections(vector<Determinant>& Dets, int prevSize,
			    vector<vector<int> >& connections,
			    vector<vector<size_t> >& orbDifference,
			    vector<vector<CItype> >& Helements, int norbs,
			    oneInt& int1, int nelec, bool includeSz=true);

};

#endif
