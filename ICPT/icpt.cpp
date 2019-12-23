/*                                                                                                                                                                          
Developed by Sandeep Sharma and Gerald Knizia, 2016                                                                                                                     
Copyright (c) 2016, Sandeep Sharma
*/

#include "CxNumpyArray.h"
#include "CxAlgebra.h"
#include "CxMemoryStack.h"
#include "CxPodArray.h"
#include "BlockContract.h"
#include "icpt.h"
#include "CxDiis.h"
#include <fstream>
#include "boost/format.hpp"

#include "E_NEV_aavv.inl"
#include "E_NEV_ccav.inl"
#include "E_NEV_ccvv.inl"
#include "E_NEV_acvv.inl"
#include "E_NEV_ccaa.inl"
#include "E_NEV_caav.inl"

#include "E_LCC_aavv.inl"
#include "E_LCC_acvv.inl"
#include "E_LCC_ccvv.inl"
#include "E_LCC_ccaa.inl"
#include "E_LCC_ccav.inl"
#include "E_LCC_caav.inl"
using ct::TArray;
using ct::FMemoryStack2;
using boost::format;
using namespace std;

void IrPrintMatrixGen(std::ostream &xout, FScalar *pData, unsigned nRows, unsigned iRowSt, unsigned nCols, unsigned iColSt, std::string const &Caption);
FNdArrayView ViewFromNpy(ct::FArrayNpy &A);

void SplitStackmem(ct::FMemoryStack2 *Mem)
{
  //now we have to distribute remaining memory equally among different threads
  long originalSize = Mem[0].m_Size;
  long remainingMem = Mem[0].m_Size - Mem[0].m_Pos;
  long memPerThrd = remainingMem/numthrds;
  Mem[0].m_Size = Mem[0].m_Pos+memPerThrd;
  for (int i=1; i<numthrds; i++) {
    Mem[i].m_pData = Mem[i-1].m_pData+Mem[i-1].m_Size;
    Mem[i].m_Pos = 0;
    Mem[i].m_Size = memPerThrd;
  }
  Mem[numthrds-1].m_Size += remainingMem%numthrds; 
}

void MergeStackmem(ct::FMemoryStack2 *Mem)
{
  //put all the memory again in the zeroth thrd
  for (int i=1; i<numthrds; i++) {
    Mem[0].m_Size += Mem[i].m_Size;
    Mem[i].m_pData = 0;
    Mem[i].m_Pos = 0;
    Mem[i].m_Size = 0;
  }
}



namespace srci {
#define NO_RT_LIB
  double GetTime() {
#ifdef NO_RT_LIB
    timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
#else
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    // ^- interesting trivia: CLOCK_MONOTONIC is not guaranteed
    //    to actually be monotonic.
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
#endif // NO_RT_LIB
    
  }
};



FJobData::FJobData()
  : WfDecl(FWfDecl()), RefEnergy(0.), ThrTrun(1e-4), ThrDen(1e-6), ThrVar(1e-14),  resultOut(""), guessInput(""),
     LevelShift(0.), nOrb(0), MaxIt(100), WorkSpaceMb(1024), nMaxDiis(6), MethodClass(METHOD_MinE)
{}



void FJobData::ReadInputFile(char const *pArgFile)
{
   // maybe I should use boost program options for this? it can deal with files
   // and stuff. But it might be helpful to keep the ci program completely
   // dependency free for inclusion in molpro and stuff.
   std::ifstream
      inpf(pArgFile);
   while (inpf.good()) {
      // behold the most marvellous elegance of C++ I/O handling.
      std::string
         Line, ArgName, ArgValue;
      std::getline(inpf, Line);
      std::stringstream
         inp(Line);
      inp >> ArgName >> ArgValue;
      // ^- this won't work if spaces are in the file name...
      if (inpf.eof())
         break;
      if (!ArgName.empty() and ArgName[0] == '#')
         // comment. skip.
         continue;
//       std::cout << format("arg: '%s' val: '%s'") % ArgName % ArgValue << std::endl;
      if (ArgName == "method")
         MethodName = ArgValue;
      else if (ArgName == "orb-type") {
         if (ArgValue == "spatial/MO")
            WfDecl.OrbType = ORBTYPE_Spatial;
         else if (ArgValue == "spinorb/MO")
            WfDecl.OrbType = ORBTYPE_SpinOrb;
         else
            throw std::runtime_error("orbital type '" + ArgValue + "' not recognized.");
      } else if (ArgName == "nelec")
         WfDecl.nElec = atoi(ArgValue.c_str());
      else if (ArgName == "nact")
	WfDecl.nActElec = atoi(ArgValue.c_str());
      else if (ArgName == "nactorb")
	WfDecl.nActOrb = atoi(ArgValue.c_str());
      else if (ArgName == "ms2")
         WfDecl.Ms2 = atoi(ArgValue.c_str());
      else if (ArgName == "ref-energy")
         RefEnergy = atof(ArgValue.c_str());
      else if (ArgName == "thr-den")
         ThrDen = atof(ArgValue.c_str());
      else if (ArgName == "orbitalFile")
	orbitalFile = ArgValue;
      else if (ArgName == "thr-var")
         ThrVar = atof(ArgValue.c_str());
      else if (ArgName == "thr-trunc")
         ThrTrun = atof(ArgValue.c_str());
      else if (ArgName == "load")
         guessInput = ArgValue;
      else if (ArgName == "save")
         resultOut = ArgValue;
      else if (ArgName == "shift")
	Shift = atof(ArgValue.c_str());
      else if (ArgName == "level-shift")
         LevelShift = atof(ArgValue.c_str());
      else if (ArgName == "max-diis")
         nMaxDiis = atof(ArgValue.c_str());
      else if (ArgName == "max-iter")
         MaxIt = atoi(ArgValue.c_str());
      else if (ArgName == "int1e/fock") 
	ct::ReadNpy(Int1e_Fock, ArgValue);
      else if (ArgName == "int1e/coreh") 
	ct::ReadNpy(Int1e_CoreH, ArgValue);
      else if (ArgName == "work-space-mb")
         WorkSpaceMb = atoi(ArgValue.c_str());
      else
         throw std::runtime_error("argument '" + ArgName + "' not recognized.");
   }
   if (Int1e_Fock.Rank() != 0) {
      if (Int1e_Fock.Rank() != 2 || Int1e_Fock.Shape[0] != Int1e_Fock.Shape[1])
         throw std::runtime_error("int1e/fock must specify a rank 2 square matrix.");
      nOrb = Int1e_Fock.Shape[0];
      MakeFockAfterIntegralTrafo = false;
   } else {
      if (Int1e_CoreH.Rank() != 2 || Int1e_CoreH.Shape[0] != Int1e_CoreH.Shape[1])
         throw std::runtime_error("int1e/coreh must specify a rank 2 square matrix.");
      nOrb = Int1e_CoreH.Shape[0];
      MakeFockAfterIntegralTrafo = false;
   }
   /*
   if (Int2e_4ix.Rank() != 0) {
      if (Int2e_3ix.Rank() != 0)
         throw std::runtime_error("int2e/3ix and int2e/4x interaction matrix elements cannot both be given.");
      if (Int2e_4ix.Rank() != 4 || Int2e_4ix.Shape[0] != Int2e_4ix.Shape[1] ||
         Int2e_4ix.Shape[0] != Int2e_4ix.Shape[2] || Int2e_4ix.Shape[0] != Int2e_4ix.Shape[3])
         throw std::runtime_error("int2e-4ix must be a rank 4 tensor with equal dimensions.");
      if (Int2e_4ix.Shape[0] != nOrb)
         throw std::runtime_error("nOrb derived from int2e_4ix not compatible with nOrb derived from Fock matrix.");
      // ^- I guess technically we might want to allow not specifying
      //    int2e_4ix at all.
   } else {
      if (Int2e_3ix.Rank() == 0)
         throw std::runtime_error("either int2e/3ix or int2e/4x interaction matrix elements must be given.");
      if (Int2e_4ix.Rank() != 3 || Int2e_4ix.Shape[1] != Int2e_4ix.Shape[2])
         throw std::runtime_error("int2e-3ix must be a rank 3 tensor (F|rs) with r,s having equal dimensions.");
      if (Int2e_3ix.Shape[1] != nOrb)
         throw std::runtime_error("nOrb derived from int2e_3ix not compatible with nOrb derived from Fock matrix.");
   }
   */
   if (Orbs.Rank() != 0) {
      if (Orbs.Rank() != 2 || Orbs.Shape[0] < Orbs.Shape[1])
         throw std::runtime_error("Input orbital matrix must have rank 2 and have at least as many orbitals as there are basis functions.");
      if (Int1e_Fock.Shape[0] != Orbs.Shape[0])
         throw std::runtime_error("nOrb derived from Fock not compatible with number of rows in orbital matrix.");
      nOrb = Orbs.Shape[1];
   }
   std::cout << format("*File '%s'  nOrb = %i%s  nElec = %i  Ms2 = %i  iWfSym = %i")
      % pArgFile % nOrb % "(R)" % WfDecl.nElec % WfDecl.Ms2 % 0 << std::endl;
//         % pArgFile % nOrb % (IntClass == INTCLASS_Spatial? " (R)": " (U)") % nElec % Ms2 % iWfSym << endl;
   if (2*nOrb < WfDecl.nElec)
      throw std::runtime_error("not enough orbitals for the given number of electrons.");
   if ((uint)std::abs(WfDecl.Ms2) > WfDecl.nElec)
      throw std::runtime_error("spin projection quantum number Ms2 lies outside of -nElec..+nElec");
   if (((uint)std::abs(WfDecl.Ms2) % 2) != (WfDecl.nElec % 2))
      throw std::runtime_error("Ms2 and nElec must have same parity.");
}




void FJobContext::Allocate(std::string op, FMemoryStack2 &Mem) {
  size_t len = TND(op)->nValues();
  double scale = 1.0;
  double *data = Mem.AllocN(len, scale);
  TND(op)->pData = data;
}

void FJobContext::DeAllocate(std::string op, FMemoryStack2 &Mem) {
  Mem.Free(TND(op)->pData);
  TND(op)->pData = NULL;
}

void FJobContext::SaveToDisk(std::string file, std::string op) {
  FILE *File = fopen(file.c_str(), "wb");
  fwrite(TND(op)->pData, sizeof(TND(op)->pData[0]), TND(op)->nValues(), File);
  fclose(File);
}

void FJobContext::Run(FMemoryStack2 *Mem)
{
   Init(Mem[0]);
   InitAmplitudes(Mem[0]);


   MakeOverlapAndOrthogonalBasis(Mem);
   MakeOrthogonal(std::string("b"), std::string("B"));  //b -> B
   if (guessInput.compare("") == 0)
     MakeOrthogonal(std::string("t"), std::string("T"));  //t -> T

   FScalar scale = 1.0;
   bool
      Converged = false,
      NoSing = false; // if set, explicitly set all singles amplitudes to zero.
   FScalar
     Energy = 0, LastEnergy = 0, Var2 = 0, Nrm2 = 0,
    tResid = 0, tRest = 0, tRold = 0;
   double tStart = srci::GetTime(), tMain = -srci::GetTime();

   Copy(*TN("P"), *TN("T"));         //P <- T
   
   //Allocate("p", Mem[0]);
   BackToNonOrthogonal("p", "P");  //p<- P
   TN("Ap")->ClearData();                //Ap clear
   ExecEquationSet(Method.EqsRes, m_Tensors, Mem[0]);      //Ap = A*p
   ExecHandCoded(Mem);
   //DeAllocate("p", Mem[0]);
   MakeOrthogonal("Ap", "AP");     //Ap -> AP
   ct::Add(TN("AP")->pData, TN("P")->pData, Shift, TN("AP")->nValues());   //AP = AP + shift * P

   Copy(*TN("R"), *TN("AP"));        //R <- AP

   ct::Scale(TN("R")->pData, -scale, TN("R")->nValues());                     //R = -1.0*R
   ct::Add(TN("R")->pData, TN("B")->pData, scale, TN("R")->nValues());    //R = R+B

 
   tRold = ct::Dot(TN("R")->pData, TN("R")->pData, TN("R")->nValues()); //<R|R>

   Copy(*TN("P"), *TN("R"));                                                // R = AP

   std::cout << format("\n Convergence thresholds:   THRDEN = %6.2e  THRVAR = %6.2e\n") % ThrDen % ThrVar;
   std::cout << "\n ITER.      SQ.NORM        ENERGY      ENERGY CHANGE     VAR       TIME     DIIS" << std::endl;

   for (uint iIt = 0; iIt < MaxIt; ++ iIt)
   {
     //Allocate("p", Mem[0]);
      BackToNonOrthogonal("p", "P");                          //p <- P
      TN("Ap")->ClearData();                                       // Ap->clear
      tResid -= srci::GetTime();
      ExecEquationSet(Method.EqsRes, m_Tensors, Mem[0]);
      ExecHandCoded(Mem);
      //DeAllocate("p", Mem[0]);
      tResid += srci::GetTime();

      MakeOrthogonal("Ap", "AP");                           //Ap -> AP
      ct::Add(TN("AP")->pData, TN("P")->pData, Shift, TN("AP")->nValues());   //AP = AP + shift * P

      FScalar alpha = tRold / ct::Dot(TN("P")->pData, TN("AP")->pData, TN("P")->nValues());   //<P|AP>

      ct::Add(TN("T")->pData, TN("P")->pData, alpha, TN("R")->nValues());                    //T = T+alph*P
      ct::Add(TN("R")->pData, TN("AP")->pData, -alpha, TN("R")->nValues());                  //R = R-alph*AP


      tResid = ct::Dot(TN("R")->pData, TN("R")->pData, TN("R")->nValues());                 // <R|R>
      Nrm2   = ct::Dot(TN("T")->pData, TN("T")->pData, TN("T")->nValues());                 // <T|T>

      Energy = -ct::Dot(TN("T")->pData, TN("B")->pData, TN("T")->nValues())                 // -<T|B> - <T|R>
	-ct::Dot(TN("T")->pData, TN("R")->pData, TN("T")->nValues());
      

      std::cout << format("%4i   %14.8f %14.8f %14.8f    %8.2e%10.2f\n")
         % (1+iIt) % Nrm2 % (Energy+RefEnergy) % (Energy-LastEnergy) % tResid
         % (srci::GetTime() - tStart) ;
      std::cout << std::flush;
      tStart = srci::GetTime();
      if (tResid < ThrVar) {
	Converged = true;
	break;
      }

      ct::Scale(TN("P")->pData, tResid/tRold, TN("P")->nValues());                              //P = (rnew/rold)*P
      ct::Add(TN("P")->pData, TN("R")->pData, scale, TN("R")->nValues());                     //P = P+R

      tRold = tResid;
      LastEnergy = Energy;

   }

   if ( !Converged ) {
      std::cout << format("\n*WARNING: No convergence for root %i."
                     " Stopped at NIT: %i  DEN: %.2e  VAR: %.2e")
               % 0 % MaxIt % (Energy - LastEnergy) % Var2 << std::endl;
   }
   std::cout << "\n";
   tMain += srci::GetTime();
   tRest = tMain - tResid;
   PrintTiming("main loop", tMain);
   PrintTiming("residual", tResid);
   PrintTiming("rest", tRest);

   FScalar c0 = 1./std::sqrt(Nrm2);
   std::cout << "\n";
   PrintResult("Coefficient of reference function", c0, -1);
   PrintResult("Reference energy", RefEnergy, -1);
   if (RefEnergy != 0.)
      PrintResult("Correlation energy", Energy, -1);

   std::cout << "\n";
   PrintResult("ENERGY", RefEnergy + Energy, 0);

   if (resultOut.compare("") != 0) {
     PrintResult("Writing result to "+resultOut, 0, 0);
     ct::FShapeNpy outshape = ct::MakeShape(TN("T")->Sizes[0], TN("T")->Sizes[1],TN("T")->Sizes[2],TN("T")->Sizes[3]);
     ct::WriteNpy(resultOut, TN("T")->pData, outshape);
   }
}


void FJobContext::CleanAmplitudes(std::string const &r_or_t)
{
}


void FJobContext::ExecEquationSet( FEqSet const &Set, std::vector<FSrciTensor>& m_Tensors, FMemoryStack2 &Mem)
{
//    std::cout << "! EXEC: " << Set.pDesc << std::endl;
   for (FEqInfo const *pEq = Set.pEqs; pEq != Set.pEqs + Set.nEqs; ++ pEq) {
     //std::cout << "****  "<<pEq->pCoDecl<<std::endl;
      if (pEq->nTerms != 3) {
//          throw std::runtime_error("expected input equations to be in binary contraction form. One isn't.");
         FNdArrayView
            **pTs;
         Mem.Alloc(pTs, pEq->nTerms);
         for (uint i = 0; i < pEq->nTerms; ++i) {
	   pTs[i] = &m_Tensors[pEq->iTerms[i]];//TensorById(pEq->iTerms[i]);
	   if (Method.pTensorDecls[pEq->iTerms[i]].Storage == STORAGE_Disk) {
	     FillData(pEq->iTerms[i], Mem);
	   }
	   if (i != 0 && pTs[i] == pTs[0])
	     throw std::runtime_error(boost::str(format("contraction %i has overlapping dest and source tensors."
							" Tensors may not contribute to contractions involving themselves.") % (pEq - Set.pEqs)));
         }
         ContractN(pTs, pEq->pCoDecl, pEq->Factor, true, Mem);
         Mem.Free(pTs);
      } else {
	void
	  *pBaseOfMemory = Mem.Alloc(0);
         FNdArrayView
	   *pD = &m_Tensors[pEq->iTerms[0]],
	   *pS = &m_Tensors[pEq->iTerms[1]],
	   *pT = &m_Tensors[pEq->iTerms[2]];//TensorById(pEq->iTerms[i]);
	 for (int j=0; j<3; j++) 
	   if (Method.pTensorDecls[pEq->iTerms[j]].Storage == STORAGE_Disk) {
	     FillData(pEq->iTerms[j], Mem);
	   }

	 //*pD = TensorById(pEq->iTerms[0]),
	 //*pS = TensorById(pEq->iTerms[1]),
	 //*pT = TensorById(pEq->iTerms[2]);
         if (pD == pS || pD == pT)
            throw std::runtime_error(boost::str(format("contraction %i has overlapping dest and source tensors."
               " Tensors may not contribute to contractions involving themselves.") % (pEq - Set.pEqs)));
         ContractBinary(*pD, pEq->pCoDecl, *pS, *pT, pEq->Factor, true, Mem);
	 Mem.Free(pBaseOfMemory);
      }
   }
};


void FJobContext::ReadMethodFile()
{
  if (MethodName == "NEVPT2_AAVV") 
    NEVPT2_AAVV::GetMethodInfo(Method);
  else if (MethodName == "NEVPT2_ACVV") 
    NEVPT2_ACVV::GetMethodInfo(Method);
  else if (MethodName == "NEVPT2_CCAV") 
    NEVPT2_CCAV::GetMethodInfo(Method);
  else if (MethodName == "NEVPT2_CCVV") 
    NEVPT2_CCVV::GetMethodInfo(Method);
  else if (MethodName == "NEVPT2_CCAA") 
    NEVPT2_CCAA::GetMethodInfo(Method);
  else if (MethodName == "NEVPT2_CAAV") 
    NEVPT2_CAAV::GetMethodInfo(Method);

  else if (MethodName == "MRLCC_AAVV") 
    MRLCC_AAVV::GetMethodInfo(Method);
  else if (MethodName == "MRLCC_ACVV") 
    MRLCC_ACVV::GetMethodInfo(Method);
  else if (MethodName == "MRLCC_CCVV") 
    MRLCC_CCVV::GetMethodInfo(Method);
  else if (MethodName == "MRLCC_CCAA") 
    MRLCC_CCAA::GetMethodInfo(Method);
  else if (MethodName == "MRLCC_CCAV") 
    MRLCC_CCAV::GetMethodInfo(Method);
  else if (MethodName == "MRLCC_CAAV") 
    MRLCC_CAAV::GetMethodInfo(Method);
  else
    throw std::runtime_error("Method '" + MethodName + "' not recognized.");
}

void FJobContext::Init(ct::FMemoryStack2 &Mem)
{
   ReadMethodFile();

   std::cout << format(" Performing a %s %s calculation with %s orbitals.\n")
      % Method.pSpinClass
      % MethodName
      % ((WfDecl.OrbType == ORBTYPE_Spatial)? "spatial" : "spin");

   InitDomains();
   CreateTensors(Mem);
   InitAmpResPairs();
}

static void CopyEps(FDomainInfo &Dom, FScalar EpsFactor, ct::FArrayNpy const &Fock)
{
     for (uint i = Dom.iBase; i != Dom.iBase + Dom.nSize; ++ i)
         Dom.Eps.push_back(EpsFactor * Fock(i,i));
}

void FJobContext::InitDomains()
{
   if (WfDecl.OrbType == ORBTYPE_Spatial) {
      FDomainInfo
         &Ext = m_Domains['e'],
         &Act = m_Domains['a'],
         &Clo = m_Domains['c'];
      Clo.iBase = 0;
      Clo.nSize = (WfDecl.nElec-WfDecl.nActElec)/2.;
      //CopyEps(Clo, -1., Int1e_Fock);

      Act.iBase = Clo.nSize;
      Act.nSize = WfDecl.nActOrb;
      //CopyEps(Act, -1., Int1e_Fock);

      Ext.iBase = Clo.nSize+Act.nSize;
      Ext.nSize = nOrb - Ext.iBase;
      //CopyEps(Ext, +1., Int1e_Fock);

      for (int d=0; d<Method.nDomainDecls; d++) {
        FDomainInfo &d1 = m_Domains[Method.pDomainDecls[d].pName[0]];
	FDomainInfo &Related = m_Domains[ Method.pDomainDecls[d].pRelatedTo[0]];
	d1.iBase = Related.iBase;
	d1.nSize = Method.pDomainDecls[d].f(Related.nSize);
      }
   }
}

void CreateTensorFromShape(FSrciTensor &Out, char const *pDomain, char const *pSymmetry, FDomainInfoMap const &DomainInfo)
{
   Out.pData = 0;
   FArrayOffset
      Stride = 1;
   for (char const *p = pDomain; *p != 0; ++ p) {
      Out.Sizes.push_back(value(DomainInfo, *p).nSize);
      Out.Strides.push_back(Stride);
      Stride *= Out.Sizes.back();
   }
}

void CopyDomianSubset(FSrciTensor &Out, FSrciTensor const &In, char const *pDomain, FDomainInfoMap &Domains)
{
   assert(Out.Rank() == In.Rank());
   FNdArrayView
      // view into the subset of the input array we are to copy.
      InSubset = In;
   for (uint i = 0; i < In.Rank(); ++ i){
      assert(pDomain[i] != 0);
      FDomainInfo const &d = value(Domains, pDomain[i]);
      InSubset.pData += InSubset.Strides[i] * d.iBase;
      assert(InSubset.Sizes[i] >= d.iBase + d.nSize);
      InSubset.Sizes[i] = d.nSize;
      assert(InSubset.Sizes[i] == Out.Sizes[i]);
   }
   assert(pDomain[In.Rank()] == 0);

   Copy(Out, InSubset);
}

// make a NdArrayView looking into an ArrayNpy object.
FNdArrayView ViewFromNpy(ct::FArrayNpy &A)
{
   FNdArrayView
      Out;
   Out.pData = &A.Data[0];
   for (uint i = 0; i < A.Rank(); ++i){
      Out.Strides.push_back(A.Strides[i]);
      Out.Sizes.push_back(A.Shape[i]);
   }
   return Out;
}

void FJobContext::FillData(int i, ct::FMemoryStack2 &Mem) {
  FTensorDecl const
    &Decl = Method.pTensorDecls[i];

  char const
    *pIntNames[] = {"f", "k"};  
  ct::FArrayNpy
    *pIntArrays[] = {&Int1e_Fock, &Int1e_CoreH};
  uint
    nIntTerms = sizeof(pIntNames)/sizeof(pIntNames[0]);

  double A = 1.0;

  if (Decl.Usage != USAGE_PlaceHolder) {
    double* tensorData = Mem.AllocN(m_Tensors[i].nValues(), A);
    m_Tensors[i].pData = tensorData;
  }
     
  if (Decl.Usage == USAGE_Density) {
    uint k;
    const char *delta = "delta";
    if (0 == strcmp(Decl.pName, delta)) {
      m_Tensors[i].ClearData();
      for (int i1=0; i1<m_Domains[Decl.pDomain[0]].nSize; i1++) 
	m_Tensors[i](i1,i1) = 1.0;
    }
    else if (Decl.pName[0] == 'S') {;}
    else {
      std::string filename = "int/"+string(Decl.pName)+".npy";
      m_Tensors[i].Sizes.clear();
      m_Tensors[i].Strides.clear();
      ct::ReadNpyData(m_Tensors[i], filename);
    }
  }
  else if (Decl.Usage == USAGE_Hamiltonian) {
    if (0 == strcmp(Decl.pName, "W") ){// && 0==strcmp( string(Decl.pDomain).c_str(), "caca") ) {
      std::string filename = "int/W:"+string(Decl.pDomain)+".npy";
      m_Tensors[i].Sizes.clear();
      m_Tensors[i].Strides.clear();
      ct::ReadNpyData(m_Tensors[i], filename);
    }
    else {
      uint k;
      for (k = 0; k < nIntTerms; ++k)
	if (0 == strcmp(Decl.pName, pIntNames[k])) {
	  CopyDomianSubset(m_Tensors[i], ViewFromNpy(*pIntArrays[k]), Decl.pDomain, m_Domains);
	  break;
	}
      if (k == nIntTerms)
	throw std::runtime_error("Hamiltonian term '" + std::string(Decl.pName) + "' not recognized.");
    }
  }
}

void FJobContext::CreateTensors(ct::FMemoryStack2 &Mem)
{
   m_Tensors.resize(Method.nTensorDecls);
   // create meta-information: shapes & sizes.
   size_t
      TotalSize = 0;
   for (uint i = 0; i != Method.nTensorDecls; ++i) {
      FTensorDecl const
         &Decl = Method.pTensorDecls[i];
      CreateTensorFromShape(m_Tensors[i], Decl.pDomain, Decl.pSymmetry, m_Domains);
      if (Decl.Usage != USAGE_PlaceHolder && Decl.Storage != STORAGE_Disk) //placeholders dont have their own data
	TotalSize += m_Tensors[i].nValues();


      // keep a link in case we need to look up the contents of
      // individual tensors.
      std::string NameAndDomain = boost::str(format("%s:%s") % Decl.pName % Decl.pDomain);
      m_TensorsByNameAndDomain[NameAndDomain] = &m_Tensors[i];
      m_TensorsByName[Decl.pName] = &m_Tensors[i];
   }


     // make actual data array and assign the tensors to their target position.
   double A = 1.0;
   //m_TensorData = Mem.AllocN(TotalSize, A);
   //memset(&m_TensorData[0], 0, TotalSize*sizeof(FScalar));
   //size_t
   //iDataOff = 0;
     
   for (uint i = 0; i != Method.nTensorDecls; ++i) {
     if (Method.pTensorDecls[i].Storage != STORAGE_Disk)
       FillData(i, Mem);
   }
   
   size_t nSizeVec;// = m_pResEnd - m_pRes;
   std::cout << format("\n Size of working set:           %12i mb\n"
		       " Size of Hamiltonian:           %12i mb\n"
		       " Size of CI amplitude vector:   %12i mb [%i entries/vec]")
     % (TotalSize*sizeof(FScalar)/1048576)
     % ((Int1e_Fock.Size() + Int2e_4ix.Size())*sizeof(FScalar)/1048576)
     % (nSizeVec*sizeof(FScalar)/1048576)
     % (nSizeVec)
	     << std::endl;
}

   
   
FSrciTensor *FJobContext::TND(std::string const &NameAndDomain)
{
  return value(m_TensorsByNameAndDomain, NameAndDomain);
}

FSrciTensor *FJobContext::TN(std::string const &Name)
{
  return value(m_TensorsByName, Name);
}

FSrciTensor *FJobContext::TensorById(int iTensorRef) {
   if (iTensorRef < 0 || (unsigned)iTensorRef >= m_Tensors.size())
      throw std::runtime_error(boost::str(format("referenced tensor id %i does not exist.") % iTensorRef));
   return &m_Tensors[(unsigned)iTensorRef];
}

void FJobContext::InitAmpResPairs()
{
   for (uint i = 0; i != Method.nTensorDecls; ++ i) {
      FTensorDecl const
         &iDecl = Method.pTensorDecls[i];
      if (iDecl.Usage != USAGE_Residual)
         break; // through with the residuals. all should have been assigned by now.
      for (uint j = i+1; j != Method.nTensorDecls; ++ j) {
         FTensorDecl const
            &jDecl = Method.pTensorDecls[j];
         if (jDecl.Usage != USAGE_Amplitude)
            continue;
         if (0 != strcmp(iDecl.pDomain, jDecl.pDomain))
            continue;
      }
   }
}

void FJobContext::DeleteData(ct::FMemoryStack2 &Mem) {
  Mem.Free(m_TensorData);
  m_Tensors.resize(0);
}

void FJobContext::ClearTensors(size_t UsageFlags)
{
   for (uint i = 0; i != Method.nTensorDecls; ++ i) {
      FTensorDecl const
         &iDecl = Method.pTensorDecls[i];
      if ((iDecl.Usage & UsageFlags) == 0)
         // not supposed to touch this. Leave it alone.
         continue;
//       std::cout << "clear:" << UsageFlags << "  wiping: iDecl i == " << i << ": " << iDecl.pName << std::endl;
      m_Tensors[i].ClearData();
   }
}


static void DenomScaleUpdateR(FScalar *pAmp, FScalar *pRes, FArrayOffset *pStride,
   FArrayOffset *pSize, FArrayOffset RankLeft, FScalar const **ppEps, FScalar CurEps)
{
   // note: this assumes that the strides for pAmp and pRes are equal.
   // If we create the tensors ourselves (which we do), they are. But
   // in general they might not be.
//    if (RankLeft > 0)
//       for (FArrayOffset i = 0; i < pSize[0]; ++ i)
//          DenomScaleUpdateR(pAmp + i*pStride[0], pRes + i*pStride[0], pStride-1, pSize-1, RankLeft-1, ppEps-1, CurEps + ppEps[0][i]);
//    else if (RankLeft == 2)
//       for (FArrayOffset j = 0; j < pSize[ 0]; ++ j)
//          for (FArrayOffset i = 0; i < pSize[-1]; ++ i) {
//             FArrayOffset ij = j*pStride[0] + i*pStride[-1];
//             pAmp[ij] -= pRes[ij] / (CurEps + ppEps[0][j] + ppEps[-1][i]);
//          }
//    // ^- note: this case is technically redundant.
//    else
//       pAmp[0] -= pRes[0] / CurEps;

   if (RankLeft > 0)
      for (FArrayOffset i = 0; i < pSize[0]; ++ i)
         DenomScaleUpdateR(pAmp + i*pStride[0], pRes + i*pStride[0], pStride-1, pSize-1, RankLeft-1, ppEps-1, CurEps + ppEps[0][i]);
   else if (RankLeft == 2)
      for (FArrayOffset j = 0; j < pSize[ 0]; ++ j)
         for (FArrayOffset i = 0; i < pSize[-1]; ++ i) {
            pRes[j*pStride[0] + i*pStride[-1]] /= (CurEps + ppEps[0][j] + ppEps[-1][i]);
         }
   // ^- note: this case is technically redundant.
   else
      pRes[0] /= CurEps;
}





void FJobContext::PrintTiming(char const *p, FScalar t)
{
   std::cout << format(" Time for %-35s%10.2f sec")
      % (std::string(p) + ":") % t << std::endl;
}

void FJobContext::PrintTimingT0(char const *p, FScalar t0)
{
   PrintTiming(p, srci::GetTime() - t0);
}

void FJobContext::PrintResult(std::string const &s, FScalar v, int iState, char const *pAnnotation)
{
   std::stringstream
      Caption;
   if ( iState >= 0 ) {
      Caption << format("!%s STATE %i %s") % MethodName % (iState + 1) % s;
   } else {
      Caption << " " + s;
   }
//    std::cout << format("%-23s%20.14f") % Caption.str() % v << std::endl;
   std::cout << format("%-35s%20.14f") % Caption.str() % v;
   if (pAnnotation)
      std::cout << "  " << pAnnotation;
   std::cout << std::endl;
}



