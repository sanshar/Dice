#ifndef SHCI_MAKE4CHAMILTONIAN_H
#define SHCI_MAKE4CHAMILTONIAN_H
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
class oneInt;
class twoInt;
class twoIntHeatBath;
class twoIntHeatBathSHM;
class schedule;

namespace SHCImake4cHamiltonian {
  
  struct HamHelper4c {
    map<Determinant, int> Nminus1; // The string with n-1 electrons
    map<Determinant, int> Nminus2; // The string with n-2 electrons
    vector<vector<int>> Nminus2ToDet; // The connection between n-2 string and n strings
    vector<int*> Nminus2ToDetSM;
    int* Nminus2ToDetLen;

    vector<vector<int>> Nminus1ToDet; // The connection between n-1 string and n strings
    //This helps to generate the single excitation
    vector<int*> Nminus1ToDetSM;
    int* Nminus1ToDetLen;

    // Consistent with the non-relativistic function name
    void PopulateHelpers(Determinant* SHMDets, int DetsSize, int startIndex);

    void MakeSHMHelpers();

    void clear() {
      Nminus2.clear();
      Nminus2ToDet.clear();
      Nminus1.clear();
      Nminus1ToDet.clear();
    }
  }; // HamHelper4c

// This should just be the same as SparseHam in SHCImakeHamiltonian.h
struct SparseHam {
    std::vector<std::vector<int> > connections;  
    std::vector<std::vector<CItype> > Helements;
    std::vector<std::vector<size_t> > orbDifference;
    int Nbatches;
    int BatchSize;
    bool diskio;
    string prefix;
    SparseHam() {
      diskio = false;
      Nbatches = 1;
    }


    // routines
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

    // This function is the only one in sparseHam that is modified
    void makeFromHelper(
            HamHelper4c& helper2, Determinant *SHMDets,
            int startIndex, int endIndex,
            int Norbs, oneInt& I1, twoInt& I2,
            schedule& schd, double& coreE, bool DoRDM);

    void writeBatch(int batch);

    void readBatch (int batch);

    void setNbatches(int DetSize);
  }; // SparseHam

// fixForTreversal and regenerateH is not implemented

void PopulateHelperLists(
map<Determinant, int>& Nminus1s, vector<vector<int>>& Nminus1ToDet,
map<Determinant, int>& Nminus2s, vector<vector<int>>& Nminus2ToDet, 
Determinant *Dets, int DetsSize, int StartIndex);

void MakeHfromSMHelpers(
  vector<int* > &Nminus1ToDetSM,
  int* Nminus1ToDetLen,
  vector<int* > &Nminus2ToDetSM,
  int* Nminus2ToDetLen,
  Determinant* Dets, int StartIndex, int EndIndex,
  bool diskio, SparseHam& sparseHam,
  int Norbs, oneInt& I1, twoInt& I2, schedule& schd, double& coreE,
  bool DoRDM);

void MakeSHMHelpers();

void MakeSMHelpers(
  map<Determinant, int>& Nminus1,
  vector<vector<int>>& Nminus1ToDet,
  int* &Nminus1ToDetLen,
  vector<int*>& Nminus1ToDetSM,
  map<Determinant, int>& Nminus2,
  vector<vector<int>>& Nminus2ToDet,
  int* &Nminus2ToDetLen,
  vector<int*>& Nminus2ToDetSM);

}
#endif
