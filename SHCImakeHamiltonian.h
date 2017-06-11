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

  void PopulateHelperLists2(std::map<HalfDet, int >& BetaN,
			    std::map<HalfDet, int >& AlphaN,
			    std::map<HalfDet, vector<int> >& BetaNm1,
			    std::map<HalfDet, vector<int> >& AlphaNm1,
			    vector<vector<int> >& AlphaMajorToBeta,
			    vector<vector<int> >& AlphaMajorToDet,
			    vector<vector<int> >& BetaMajorToAlpha,
			    vector<vector<int> >& BetaMajorToDet,
			    vector< vector<int> >& SinglesFromAlpha,
			    vector< vector<int> >& SinglesFromBeta,
			    std::vector<Determinant>& Dets,
			    int StartIndex);

  void MakeHfromHelpers2(vector<vector<int> >& AlphaMajorToBeta,
			 vector<vector<int> >& AlphaMajorToDet,
			 vector<vector<int> >& BetaMajorToAlpha,
			 vector<vector<int> >& BetaMajorToDet,
			 vector<vector<int> >& SinglesFromAlpha,
			 vector<vector<int> >& SinglesFromBeta,
			 std::vector<Determinant>& Dets,
			 int StartIndex,
			 std::vector<std::vector<int> >&connections,
			 std::vector<std::vector<CItype> >& Helements,
			 int Norbs,
			 oneInt& I1,
			 twoInt& I2,
			 double& coreE,
			 std::vector<std::vector<size_t> >& orbDifference,
			 bool DoRDM) ;

  void MakeHfromSMHelpers2(int*          &AlphaMajorToBetaLen, 
			   vector<int* > &AlphaMajorToBeta   ,
			   vector<int* > &AlphaMajorToDet    ,
			   int*          &BetaMajorToAlphaLen, 
			   vector<int* > &BetaMajorToAlpha   ,
			   vector<int* > &BetaMajorToDet     ,
			   int*          &SinglesFromAlphaLen, 
			   vector<int* > &SinglesFromAlpha   ,
			   int*          &SinglesFromBetaLen , 
			   vector<int* > &SinglesFromBeta    ,
			   Determinant *Dets,
			   int StartIndex,
			   std::vector<std::vector<int> >&connections,
			   std::vector<std::vector<CItype> >& Helements,
			   int Norbs,
			   oneInt& I1,
			   twoInt& I2,
			   double& coreE,
			   std::vector<std::vector<size_t> >& orbDifference,
			   bool DoRDM);
  
  void MakeSMHelpers(vector<vector<int> >& AlphaMajorToBeta,
		     vector<vector<int> >& AlphaMajorToDet,
		     vector<vector<int> >& BetaMajorToAlpha,
		     vector<vector<int> >& BetaMajorToDet,
		     vector<vector<int> >& SinglesFromAlpha,
		     vector<vector<int> >& SinglesFromBeta,
		     int* &AlphaMajorToBetaLen, vector<int* >& AlphaMajorToBetaSM,
		     vector<int* >& AlphaMajorToDetSM,
		     int* &BetaMajorToAlphaLen, vector<int* >& BetaMajorToAlphaSM,
		     vector<int* >& BetaMajorToDetSM,
		     int* &SinglesFromAlphaLen, vector<int* >& SinglesFromAlphaSM,
		     int* &SinglesFromBetaLen, vector<int* >& SinglesFromBetaSM) ;
  
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

  void MakeSHMHelpers(std::map<HalfDet, std::vector<int> >& BetaN,
		      std::map<HalfDet, std::vector<int> >& AlphaNm1,
		      int* &BetaVecLen, vector<int*>& BetaVec,
		      int* &AlphaVecLen, vector<int*>& AlphaVec);

  void MakeHfromHelpers(int* &BetaVecLen, vector<int*> &BetaVec,
			int* &AlphaVecLen, vector<int*> &AlphaVec,
			Determinant *Dets,
			int StartIndex,
			std::vector<std::vector<int> >&connections,
			std::vector<std::vector<CItype> >& Helements,
			int Norbs,
			oneInt& I1,
			twoInt& I2,
			double& coreE,
			std::vector<std::vector<size_t> >& orbDifference,
			bool DoRDM=false) ;

  void updateSOCconnections(Determinant *Dets, int prevSize,
			    vector<vector<int> >& connections,
			    vector<vector<size_t> >& orbDifference,
			    vector<vector<CItype> >& Helements, int norbs,
			    oneInt& int1, int nelec, bool includeSz=true);

};

#endif
