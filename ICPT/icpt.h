/*                                                                                                                                                                          
Developed by Sandeep Sharma and Gerald Knizia, 2016                                                                                                                     
Copyright (c) 2016, Sandeep Sharma
*/
#ifndef SRCI_HEADER
#define SRCI_HEADER

#include <sys/time.h> // for gettimeofday()
#include <string.h> // for memset and strcmp
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>
#include <boost/format.hpp>
#include "CxPodArray.h"

#ifdef _OPENMP
#define omprank omp_get_thread_num()
#define numthrds std::max(atoi(std::getenv("OMP_NUM_THREADS")), 1)
#else
#define omprank 0
#define numthrds 1
#endif

void SplitStackmem(ct::FMemoryStack2 *Mem);
void MergeStackmem(ct::FMemoryStack2 *Mem);


namespace ct {
 class FMemoryStack2;
};

namespace srci {
  double GetTime();
};


unsigned const
   EQINFO_MaxRhs = 11;



enum FTensorUsage {
   USAGE_Amplitude,
   USAGE_Residual,
   USAGE_Hamiltonian,
   USAGE_Density,
   USAGE_Intermediate,
   USAGE_PlaceHolder
};

enum FTensorStorage {
  STORAGE_Memory,
  STORAGE_Disk
};

struct FTensorDecl
{
   char const
      *pName,
      *pDomain, // name:domain. E.g., 't:eecc' is a T^ij_ab T2 tensor.
//       *pSpin, // spin class of the slots: A (alpha), B (beta), _ (spatial)
      *pSymmetry; // ITF-style permutation symmetry declaration.
   FTensorUsage
      Usage;
   FTensorStorage
   Storage;
};

struct FDomainDecl
{
  char const *pName;
  char const *pRelatedTo;
  int (*f)(int) ; //this is a function pointer that takes the size of the
  //pRelatedTo Domain and returns the size of the current Domain
};
 
struct FEqInfo
{
   char const
      *pCoDecl; // <- note that these are only used to determine the
                //    permutations. The names of the indices do not matter.
   FScalar
      Factor;
   unsigned
      nTerms;
   int
      // first one is lhs, the others are rhs.
      iTerms[1+EQINFO_MaxRhs];
};

// FEqInfo a = { "ikac,kjcb->ijab", 1.000, 3, {4,2,4} };
struct FEqSet
{
   FEqInfo const
      *pEqs;
   size_t
      nEqs;
   char const
      *pDesc;

   FEqSet() {};
   FEqSet(FEqInfo const *pEqs_, size_t nEqs_, char const *pDesc_)
      : pEqs(pEqs_), nEqs(nEqs_), pDesc(pDesc_)
   {}
};

struct FMethodInfo
{
   char const
      *pName, *pSpinClass;
   char const
      *perturberClass;

   FDomainDecl const *pDomainDecls; //if we need other domains besides e,a,c
   uint nDomainDecls;

   FTensorDecl const
      *pTensorDecls;
   uint
      nTensorDecls;
   FEqSet
   EqsRes, Overlap, MakeS1, MakeS2, EqsHandCode, EqsHandCode2;
};



enum FOrbType
{
   ORBTYPE_Spatial,
   ORBTYPE_SpinOrb
};

struct FWfDecl
{
   uint
      nElec;
   uint 
      nActElec;
   uint
      nActOrb;
   int
      Ms2;
   FOrbType
      OrbType;
   uint nElecA() const { return (nElec + Ms2)/2; }
   uint nElecB() const { return (nElec - Ms2)/2; }
};

enum FMethodClass {
   METHOD_MinE,    // minimization of energy.
   METHOD_ProjSgl  // projected schroedinger equation
};

struct FJobData
{
   std::string
      MethodName;
   FWfDecl
      WfDecl;
   FScalar
      RefEnergy;
   ct::FArrayNpy
      Int1e_Fock,
      Int1e_CoreH,
      Int2e_4ix,
      Int1e_Fock_A,
      Int1e_Fock_B,
      Int2e_3ix, // Frs fitting integrals.
      Int2e_3ix_A,
      Int2e_3ix_B,
      Int2e_4ix_AA,
      Int2e_4ix_BB,
      Int2e_4ix_AB,
     Int2e_4ix_BA;
   bool
      MakeFockAfterIntegralTrafo;
   ct::FArrayNpy
      // orbital matrix supplied in input: integrals are transformed to/from
      // this.  (may need different ones for A/B)
      Orbs,
      // overlap matrix supplied in input. Used to transform stuff back to input
      // basis (e.g., RDMs). May be empty; if empty, basis is assumed
      // orthogonal.
      Overlap;
   FScalar
      ThrTrun,
      ThrDen,
      ThrVar,
      LevelShift,
      Shift;
   uint
      nOrb,
      MaxIt,
      nMaxDiis;
  size_t WorkSpaceMb;
   FMethodClass
      MethodClass;

  std::string orbitalFile;
  std::string resultOut;
  std::string guessInput;
   void ReadInputFile(char const *pArgFile);
   FJobData() ;
 };



template <class FMap>
typename FMap::mapped_type const &value(FMap const &Map, typename FMap::key_type const &Key)
{
   typename FMap::const_iterator it = Map.find(Key);
   if (it == Map.end())
      throw std::runtime_error("map value which was expected to exist actually didn't.");
   return it->second;
}

template <class FMap>
bool exists(typename FMap::key_type const &Key, FMap const &Map)
{
   typename FMap::const_iterator it = Map.find(Key);
   return it != Map.end();
}

typedef FNdArrayView
   FSrciTensor;

struct FAmpResPair {
   FSrciTensor
      *pRes, *pAmp;
   char const
      *pDomain;
   TArrayFix<FScalar const*,nMaxRank>
      // diagonal denominators for the domains. One for each slot.
      pEps;
};

// index domain. That is effectively a special case of the slot properties
// in ITF
typedef char
   FDomainName;
struct FDomainInfo
{
   uint
      // base offset and size in the full orbital space
      iBase, nSize;
  ct::TArray<FScalar>
      // orbital energies (for denominators)
      Eps;
};
typedef std::map<FDomainName, FDomainInfo>
   FDomainInfoMap;



static FScalar pow2(FScalar x) {
   return x*x;
}


struct FJobContext : FJobData
{
//    TArray<double>
//       // this is where the tensor data for the amplitude/residual/other
//       // tensors go.
//       AmpData, ResData, OtherData;
   FMethodInfo
      Method;

   void ReadMethodFile();
   void Run(ct::FMemoryStack2 *Mem);


   // print time elapsed since since t0 = GetTime() was set.
   void PrintTimingT0(char const *p, FScalar t0);
   // print time "t"
   void PrintTiming(char const *p, FScalar t);
   void PrintResult(std::string const &s, FScalar f, int iState, char const *pAnnotation=0);

   // look up a tensor by name and domain, and return a pointer to it.
   FSrciTensor *TND(std::string const &NameAndDomain);

   // look up a tensor by name and return a pointer to it.
   FSrciTensor *TN(std::string const &Name);
   FSrciTensor *TensorById(int iTensorRef);
   void DeleteData(ct::FMemoryStack2 &Mem);
   void readorbitalsfile(std::string& orbitalfile);
private:
   void Init(ct::FMemoryStack2 &Mem);
   void CreateTensors(ct::FMemoryStack2 &Mem);
   void InitDomains();
   void InitAmpResPairs();
   void ExecHandCoded(ct::FMemoryStack2 *Mem);
   void MakeOrthogonal(std::string s, std::string s2);
   void BackToNonOrthogonal(std::string s, std::string s2);
   void MakeOverlapAndOrthogonalBasis(ct::FMemoryStack2 *Mem);
   void InitAmplitudes(ct::FMemoryStack2 &Mem);
   void ClearTensors(size_t UsageFlags);

   void UpdateAmplitudes(FScalar LevelShift, ct::FMemoryStack2 *Mem);
   void ExecEquationSet(FEqSet const &Eqs, std::vector<FSrciTensor>& m_Tensors, ct::FMemoryStack2 &Mem);
   void CleanAmplitudes(std::string const &r_or_t);
   void FillData (int i, ct::FMemoryStack2 &Mem);
   void SaveToDisk(std::string file, std::string op);
   void DeAllocate(std::string op, ct::FMemoryStack2 &Mem);
   void Allocate(std::string op, ct::FMemoryStack2 &Mem);
   size_t m_TensorDataSize;
   double*
      // data of resident tensors is stored in here.
      m_TensorData;
   FDomainInfoMap
      m_Domains;
   std::vector<FSrciTensor>
      m_Tensors;

   typedef std::map<std::string, FSrciTensor* >
      FTensorByNameMap;
   FTensorByNameMap
     m_TensorsByNameAndDomain,
     m_TensorsByName;
};


#endif
