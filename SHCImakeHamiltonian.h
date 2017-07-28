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

  struct HamHelpers2 {

    vector<vector<int> >                 AlphaMajorToBeta;
    vector<vector<int> >                 AlphaMajorToDet;
    vector<vector<int> >                 BetaMajorToAlpha;
    vector<vector<int> >                 BetaMajorToDet;
    vector<vector<int> >                 SinglesFromAlpha;
    vector<vector<int> >                 SinglesFromBeta;
    vector<vector<int> >                 DoublesFromAlpha;
    vector<vector<int> >                 DoublesFromBeta;

    map<HalfDet, int>                    BetaN;
    map<HalfDet, int>                    AlphaN;


    //Shared Memory stuff
    int          *AlphaMajorToBetaLen;
    int          *SinglesFromAlphaLen;
    int          *DoublesFromAlphaLen;
    int          *BetaMajorToAlphaLen;
    int          *SinglesFromBetaLen ;
    int          *DoublesFromBetaLen ;
    
    vector<int*> AlphaMajorToBetaSM;
    vector<int*> AlphaMajorToDetSM;
    vector<int*> SinglesFromAlphaSM;
    vector<int*> DoublesFromAlphaSM;
    vector<int*> BetaMajorToAlphaSM;
    vector<int*> BetaMajorToDetSM;
    vector<int*> SinglesFromBetaSM ;
    vector<int*> DoublesFromBetaSM ;

    
    void PopulateHelpers(Determinant* SHMDets, int DetsSize, int startIndex);
    void MakeSHMHelpers();
    void clear() {
      AlphaMajorToBeta.clear();
      AlphaMajorToDet .clear();
      BetaMajorToAlpha.clear();
      BetaMajorToDet  .clear();
      SinglesFromAlpha.clear();
      SinglesFromBeta .clear();
      DoublesFromAlpha.clear();
      DoublesFromBeta .clear();
      BetaN.clear();
      AlphaN.clear();
    }
  };

  struct SparseHam {
    std::vector<std::vector<int> > connections;  
    std::vector<std::vector<CItype> > Helements;
    std::vector<std::vector<size_t> > orbDifference;
    int Nbatches;
    int BatchSize;
    bool diskio;
    string prefix;
    void clear() {
      connections.clear();
      Helements.clear();
      orbDifference.clear();
    }
    void resize(int size) {
      connections.resize(size);
      Helements.resize(size);
      orbDifference.resize(size);
    }

    SparseHam() {
      diskio = false;
      Nbatches = 1;
    }
    
    void makeFromHelper(HamHelpers2& helper2, Determinant *SHMDets,
			int startIndex, int endIndex, int Norbs, oneInt& I1,
			twoInt& I2, double& coreE, bool DoRDM);
    void writeBatch(int batch);
    void readBatch (int batch);
    void setNbatches(int DetSize);
  };

  void fixForTreversal(Determinant* Dets, int DetI, int DetJ, 
		       oneInt& I1, twoInt& I2, 
		       double& coreE, size_t& orbDiff, CItype& hij) ;

  void regenerateH(std::vector<Determinant>& Dets,
		   std::vector<std::vector<int> >&connections,
		   std::vector<std::vector<CItype> >& Helements,
		   oneInt& I1,
		   twoInt& I2,
		   double& coreE);

  void PopulateHelperLists2(std::map<HalfDet, int >& BetaN,
			    std::map<HalfDet, int >& AlphaN,
			    vector<vector<int> >& AlphaMajorToBeta,
			    vector<vector<int> >& AlphaMajorToDet,
			    vector<vector<int> >& BetaMajorToAlpha,
			    vector<vector<int> >& BetaMajorToDet,
			    vector< vector<int> >& SinglesFromAlpha,
			    vector< vector<int> >& SinglesFromBeta,
			    vector< vector<int> >& DoublesFromAlpha,
			    vector< vector<int> >& DoublesFromBeta,
			    Determinant *Dets, int DetsSize,
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
			   int*          &DoublesFromAlphaLen, 
			   vector<int* > &DoublesFromAlpha   ,
			   int*          &DoublesFromBetaLen , 
			   vector<int* > &DoublesFromBeta    ,
			   Determinant *Dets,
			   int StartIndex,
			   int EndIndex, bool diskio,
			   SparseHam& sparseHam,
			   int Norbs,
			   oneInt& I1,
			   twoInt& I2,
			   double& coreE,
			   bool DoRDM);
  
  void MakeSMHelpers(vector<vector<int> >& AlphaMajorToBeta,
		     vector<vector<int> >& AlphaMajorToDet,
		     vector<vector<int> >& BetaMajorToAlpha,
		     vector<vector<int> >& BetaMajorToDet,
		     vector<vector<int> >& SinglesFromAlpha,
		     vector<vector<int> >& SinglesFromBeta,
		     vector<vector<int> >& DoublesFromAlpha,
		     vector<vector<int> >& DoublesFromBeta,
		     int* &AlphaMajorToBetaLen, vector<int* >& AlphaMajorToBetaSM,
		     vector<int* >& AlphaMajorToDetSM,
		     int* &BetaMajorToAlphaLen, vector<int* >& BetaMajorToAlphaSM,
		     vector<int* >& BetaMajorToDetSM,
		     int* &SinglesFromAlphaLen, vector<int* >& SinglesFromAlphaSM,
		     int* &SinglesFromBetaLen, vector<int* >& SinglesFromBetaSM,
		     int* &DoublesFromAlphaLen, vector<int* >& DoublesFromAlphaSM,
		     int* &DoublesFromBetaLen, vector<int* >& DoublesFromBetaSM) ;
  
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
			    int DetsSize, Determinant* SortedDets,
			    vector<vector<int> >& connections,
			    vector<vector<size_t> >& orbDifference,
			    vector<vector<CItype> >& Helements, int norbs,
			    oneInt& int1, int nelec, bool includeSz=true);

};

#endif
