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
			    vector<vector<CItype> >& Helements, int norbs, 
			    oneInt& int1, int nelec, bool includeSz=true);
  
};

#endif
