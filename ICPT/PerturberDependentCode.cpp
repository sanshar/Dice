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
#ifdef _OPENMP
#include <omp.h>
#endif

using ct::TArray;
using ct::FMemoryStack2;
using boost::format;
using namespace std;

FNdArrayView ViewFromNpy(ct::FArrayNpy &A);

void FJobContext::ExecHandCoded(FMemoryStack2 *Mem)
{
  if (0 == strcmp(MethodName.c_str(), "MRLCC_AAVV") ) {

    FArrayOffset nact = TensorById(23)->Sizes[0];
    FArrayOffset nvirt = TensorById(6)->Sizes[0];

    void
      *pBaseOfMemory = Mem[omprank].Alloc(0);
    //{"abcd,abef,efgh,cdgh", 2.0  , 4, {6,20,4,22}},		//2.0 Ap[abcd] W[abef] p[efgh] E2[cdgh] 

    FScalar scaleA = 2.0, scaleB = 1.0, scaleC=0.0;
    FScalar *intermediateD = Mem[0].AllocN(nvirt*nvirt*nact*nact, scaleA);
    char T='t', N='n';
    FArrayOffset nvirtsq = nvirt*nvirt, nactsq = nact*nact;
#ifdef _SINGLE_PRECISION     
    sgemm_(N, T, nvirtsq, nactsq, nactsq,
	   scaleA, TensorById(4)->pData, nvirtsq, TensorById(22)->pData, nactsq, scaleC, intermediateD, nvirtsq);
    sgemm_(T, N, nvirtsq, nactsq, nvirtsq,
	   scaleB, TensorById(20)->pData, nvirtsq, intermediateD, nvirtsq, scaleB, TensorById(6)->pData, nvirtsq);
#else

    dgemm_(N, T, nactsq, nvirtsq, nactsq,
	   scaleA, TensorById(22)->pData, nactsq, TensorById(4)->pData, nvirtsq, scaleC, intermediateD, nactsq);

    SplitStackmem(Mem);
#pragma omp parallel 
    {
       void
	 *pBaseOfMemorylocal = Mem[omprank].Alloc(0);
      double *Wslice = Mem[omprank].AllocN(nvirt*nvirt*nvirt, scaleA);
      for (int b=0; b<nvirt; b++) {
	if (b%omp_get_num_threads() != omprank) continue;
	char filename[700];
	sprintf(filename, "int/W:eeee%04d", b);
	FILE* f = fopen(filename, "rb");
	fread(Wslice, sizeof(scaleA), nvirtsq*nvirt, f);
	dgemm_(N, T, nvirt, nactsq, nvirtsq,
	       scaleB, Wslice, nvirt, intermediateD, nactsq, scaleB, &TensorById(6)->operator()(0,b,0,0), nvirtsq);
	fclose(f);
      }
      Mem[omprank].Free(pBaseOfMemorylocal);
    }

    MergeStackmem(Mem);

#endif
    Mem[omprank].Free(pBaseOfMemory);

    FillData(16, Mem[0]);
    FillData(17, Mem[0]);
    //Allocate("W:aeae", Mem[0]);
    //Allocate("W:aeea", Mem[0]);
    SplitStackmem(Mem);
#pragma omp parallel
     {
       char Header0[10], Header[256];
       FILE *f = fopen("int/E3.npy", "rb");
       fread(&Header0[0], 1, 10, f);
       uint16_t HeaderSize = (uint16_t)Header0[8] + (((uint16_t)Header0[9]) << 8);
       //char Header0[10], Header[256];
       //FILE *f = fopen("int/node0/spatial_threepdm.0.0.bin.unpack", "rb");
       FNdArrayView E3slice, Apslice;
       E3slice.pData=0;
       E3slice.Sizes.clear(); E3slice.Strides.clear();
       FArrayOffset Stride = 1;
       for (int i=0; i<4; i++) {
	 E3slice.Sizes.push_back(nact);
	 E3slice.Strides.push_back(Stride);
	 Stride *= E3slice.Sizes.back();
       }
       
       Stride = 1;
       Apslice.Sizes.clear(); Apslice.Strides.clear();
       for (int i=0; i<2; i++) {
	 Apslice.Sizes.push_back(nvirt);
	 Apslice.Strides.push_back(Stride);
	 Stride *= Apslice.Sizes.back();
       }
       
       void
	 *pBaseOfMemory = Mem[omprank].Alloc(0);
       FNdArrayView **pTs1; Mem[omprank].Alloc(pTs1, 4);
       pTs1[0] = &Apslice;
       pTs1[1] = TensorById(16);
       pTs1[2] = TensorById(4);
       pTs1[3] = &E3slice;
       FNdArrayView **pTs2; Mem[omprank].Alloc(pTs2, 4);
       pTs2[0] = &Apslice;
       pTs2[1] = TensorById(17);
       pTs2[2] = TensorById(4);
       pTs2[3] = &E3slice;
       FNdArrayView **pTs3; Mem[omprank].Alloc(pTs3, 4);
       pTs3[0] = &Apslice;
       pTs3[1] = TensorById(19);
       pTs3[2] = TensorById(4);
       pTs3[3] = &E3slice;

       double temp=1.;
       size_t nact4 = nact*nact*nact*nact, nact2=nact*nact;
       double * E3slicedata = Mem[omprank].AllocN(nact4, temp);

       for (size_t cd=0; cd<nact*nact; cd++) {
	 if (cd%omp_get_num_threads() != omprank) continue;
	 size_t c=cd/nact, d=cd%nact;

	 fseek(f, HeaderSize+10+(d+c*nact)*nact4*sizeof(double), SEEK_SET);
	 fread(E3slicedata, sizeof(double), nact4, f);
	 //E3slice.pData = &TensorById(23)->operator()(0,0,0,0,d,c) ;
	 E3slice.pData = E3slicedata;
	 Apslice.pData = &TensorById(6)->operator()(0,0,c,d) ;
	 FScalar scale = 4.0;
	 ContractN(pTs3, "ab,efgh,abei,higf", -scale, true, Mem[omprank]);
	 ContractN(pTs2, "ab,eafg,bfhi,ihge", scale, true, Mem[omprank]);
	 ContractN(pTs1, "ab,eafg,bghi,fhie", scale, true, Mem[omprank]);
       }       
       fclose(f);
       Mem[omprank].Free(pBaseOfMemory);
       
     }
     MergeStackmem(Mem);
     DeAllocate("W:aeea", Mem[0]);
     DeAllocate("W:aeae", Mem[0]);

  }
  else if (0 == strcmp(MethodName.c_str(), "NEVPT2_AAVV") ) {

    FArrayOffset nact = TensorById(23)->Sizes[0];
    FArrayOffset nvirt = TensorById(6)->Sizes[0];

    SplitStackmem(Mem);
#pragma omp parallel
     {
       char Header0[10], Header[256];
       FILE *f = fopen("int/E3.npy", "rb");
       fread(&Header0[0], 1, 10, f);
       uint16_t HeaderSize = (uint16_t)Header0[8] + (((uint16_t)Header0[9]) << 8);
       //char Header0[10], Header[256];
       //FILE *f = fopen("int/node0/spatial_threepdm.0.0.bin.unpack", "rb");
       FNdArrayView E3slice, Apslice;
       E3slice.pData=0;
       E3slice.Sizes.clear(); E3slice.Strides.clear();
       FArrayOffset Stride = 1;
       for (int i=0; i<4; i++) {
	 E3slice.Sizes.push_back(nact);
	 E3slice.Strides.push_back(Stride);
	 Stride *= E3slice.Sizes.back();
       }
       
       Stride = 1;
       Apslice.Sizes.clear(); Apslice.Strides.clear();
       for (int i=0; i<2; i++) {
	 Apslice.Sizes.push_back(nvirt);
	 Apslice.Strides.push_back(Stride);
	 Stride *= Apslice.Sizes.back();
       }
       
       void
	 *pBaseOfMemory = Mem[omprank].Alloc(0);
       FNdArrayView **pTs3; Mem[omprank].Alloc(pTs3, 4);
       pTs3[0] = &Apslice;
       pTs3[1] = TensorById(19);
       pTs3[2] = TensorById(4);
       pTs3[3] = &E3slice;

       double temp=1.;
       size_t nact4 = nact*nact*nact*nact, nact2=nact*nact;
       double * E3slicedata = Mem[omprank].AllocN(nact4, temp);

       for (size_t cd=0; cd<nact*nact; cd++) {
	 if (cd%omp_get_num_threads() != omprank) continue;
	 size_t c=cd/nact, d=cd%nact;

	 fseek(f, HeaderSize+10+(d+c*nact)*nact4*sizeof(double), SEEK_SET);
	 fread(E3slicedata, sizeof(double), nact4, f);
	 E3slice.pData = E3slicedata;
	 Apslice.pData = &TensorById(6)->operator()(0,0,c,d) ;
	 FScalar scale = 4.0;
	 ContractN(pTs3, "ab,efgh,abei,higf", -scale, true, Mem[omprank]);
       }       
       fclose(f);
       Mem[omprank].Free(pBaseOfMemory);
       
     }
     MergeStackmem(Mem);
  }
  else if (0 == strcmp(MethodName.c_str(), "MRLCC_CAAV") ) {

    FArrayOffset nact = TensorById(22)->Sizes[2];
    FArrayOffset ncore = TensorById(22)->Sizes[0];
    FArrayOffset nvirt = TensorById(22)->Sizes[3];

    SplitStackmem(Mem);
#pragma omp parallel 
    {
      void
	 *pBaseOfMemorylocal = Mem[omprank].Alloc(0);

      char AHeader0[10], AHeader[256];
      FILE *fA = fopen("int/E3B.npy", "rb");
      fread(&AHeader0[0], 1, 10, fA);
      size_t AHeaderSize = (uint16_t)AHeader0[8] + (((uint16_t)AHeader0[9]) << 8);
      
      char BHeader0[10], BHeader[256];
      FILE *fB = fopen("int/E3C.npy", "rb");
      fread(&BHeader0[0], 1, 10, fB);
      size_t BHeaderSize = (uint16_t)BHeader0[8] + (((uint16_t)BHeader0[9]) << 8);

      std::vector<FSrciTensor> localTensors;
      localTensors.resize(47);
      for (int i=0; i<47; i++) {
	localTensors[i].Sizes = m_Tensors[i].Sizes; 
	localTensors[i].Strides = m_Tensors[i].Strides;
	if (i < 39)
	  localTensors[i].pData = m_Tensors[i].pData;
      } 

      double scale = 1.0;
      double* E3slicedata = Mem[omprank].AllocN(nact*nact*nact*nact, scale);
      double* Ap1data = Mem[omprank].AllocN(ncore*nvirt*nact*nact, scale);
      double* Ap2data = Mem[omprank].AllocN(ncore*nvirt*nact*nact, scale);
      for (size_t i=0; i<nact*nact*ncore*nvirt; i++) 
	Ap1data[i] = 0.0;
      for (size_t i=0; i<nact*nact*ncore*nvirt; i++) 
	Ap2data[i] = 0.0;

      localTensors[42].pData = Ap1data;
      localTensors[43].pData = Ap2data;
      
      for (size_t RS=0; RS<nact*nact; RS++)
      {
	if (RS%omp_get_num_threads() != omprank) continue;
	size_t R = (RS)/nact;
	size_t S = (RS)%nact;

	fseek(fA, AHeaderSize+10+(R+S*nact)*nact*nact*nact*nact*sizeof(double), SEEK_SET);
	fread(E3slicedata, sizeof(double), nact*nact*nact*nact, fA);
	localTensors[39].pData = &localTensors[42].operator()(0,0,R,S);
	localTensors[44].pData = &localTensors[43].operator()(0,0,R,S);
	localTensors[40].pData = E3slicedata;
	ExecEquationSet(Method.EqsHandCode, localTensors, Mem[omprank]);      //Ap = A*p      

	fseek(fB, BHeaderSize+10+(R+S*nact)*nact*nact*nact*nact*sizeof(double), SEEK_SET);
	fread(E3slicedata, sizeof(double), nact*nact*nact*nact, fB);
	localTensors[41].pData = E3slicedata;
	ExecEquationSet(Method.EqsHandCode2, localTensors, Mem[omprank]);      //Ap = A*p      

      }
#pragma omp critical
      {
	for (size_t i=0; i<ncore; i++) 
	for (size_t j=0; j<nact; j++) 
	for (size_t k=0; k<nact; k++) 
	for (size_t l=0; l<nvirt; l++) {
	  localTensors[21].operator()(i,j,k,l) += localTensors[42].operator()(i,l,j,k);
	  localTensors[22].operator()(i,j,k,l) += localTensors[43].operator()(i,l,j,k);
	}
      }
      localTensors.clear();
      fclose(fA);
      fclose(fB);
      Mem[omprank].Free(pBaseOfMemorylocal);      
    }
    MergeStackmem(Mem);

  }
  else if (0 == strcmp(MethodName.c_str(), "NEVPT2_CAAV") ) {

    FArrayOffset nact = TensorById(22)->Sizes[2];
    FArrayOffset ncore = TensorById(22)->Sizes[0];
    FArrayOffset nvirt = TensorById(22)->Sizes[3];

    SplitStackmem(Mem);
#pragma omp parallel 
    {
      void
	 *pBaseOfMemorylocal = Mem[omprank].Alloc(0);

      char AHeader0[10], AHeader[256];
      FILE *fA = fopen("int/E3B.npy", "rb");
      fread(&AHeader0[0], 1, 10, fA);
      size_t AHeaderSize = (uint16_t)AHeader0[8] + (((uint16_t)AHeader0[9]) << 8);
      
      char BHeader0[10], BHeader[256];
      FILE *fB = fopen("int/E3C.npy", "rb");
      fread(&BHeader0[0], 1, 10, fB);
      size_t BHeaderSize = (uint16_t)BHeader0[8] + (((uint16_t)BHeader0[9]) << 8);

      std::vector<FSrciTensor> localTensors;
      localTensors.resize(47);
      for (int i=0; i<47; i++) {
	localTensors[i].Sizes = m_Tensors[i].Sizes; 
	localTensors[i].Strides = m_Tensors[i].Strides;
	if (i < 39)
	  localTensors[i].pData = m_Tensors[i].pData;
      } 

      double scale = 1.0;
      double* E3slicedata = Mem[omprank].AllocN(nact*nact*nact*nact, scale);
      double* Ap1data = Mem[omprank].AllocN(ncore*nvirt*nact*nact, scale);
      double* Ap2data = Mem[omprank].AllocN(ncore*nvirt*nact*nact, scale);
      for (size_t i=0; i<nact*nact*ncore*nvirt; i++) 
	Ap1data[i] = 0.0;
      for (size_t i=0; i<nact*nact*ncore*nvirt; i++) 
	Ap2data[i] = 0.0;

      localTensors[42].pData = Ap1data;
      localTensors[43].pData = Ap2data;
      
      for (size_t RS=0; RS<nact*nact; RS++)
      {
	if (RS%omp_get_num_threads() != omprank) continue;
	size_t R = (RS)/nact;
	size_t S = (RS)%nact;

	fseek(fA, AHeaderSize+10+(R+S*nact)*nact*nact*nact*nact*sizeof(double), SEEK_SET);
	fread(E3slicedata, sizeof(double), nact*nact*nact*nact, fA);
	localTensors[39].pData = &localTensors[42].operator()(0,0,R,S);
	localTensors[44].pData = &localTensors[43].operator()(0,0,R,S);
	localTensors[40].pData = E3slicedata;
	ExecEquationSet(Method.EqsHandCode, localTensors, Mem[omprank]);      //Ap = A*p      

	fseek(fB, BHeaderSize+10+(R+S*nact)*nact*nact*nact*nact*sizeof(double), SEEK_SET);
	fread(E3slicedata, sizeof(double), nact*nact*nact*nact, fB);
	localTensors[41].pData = E3slicedata;
	ExecEquationSet(Method.EqsHandCode2, localTensors, Mem[omprank]);      //Ap = A*p      

      }
#pragma omp critical
      {
	for (size_t i=0; i<ncore; i++) 
	for (size_t j=0; j<nact; j++) 
	for (size_t k=0; k<nact; k++) 
	for (size_t l=0; l<nvirt; l++) {
	  localTensors[20].operator()(i,j,k,l) += localTensors[42].operator()(i,l,j,k);
	  localTensors[21].operator()(i,j,k,l) += localTensors[43].operator()(i,l,j,k);
	}
      }
      localTensors.clear();
      fclose(fA);
      fclose(fB);
      Mem[omprank].Free(pBaseOfMemorylocal);      
    }
    MergeStackmem(Mem);

  }
  else if (0 == strcmp(MethodName.c_str(), "MRLCC_CCAA") ) {
    //{"KLRS,IaKb,PQaRSb,ILPQ", -1.0  , 4, {22,5,16,21}},		//Ap[KLRS] += -1.0 W[IaKb] E3[PQaRSb] p[ILPQ]
    //{"KLSR,IaLb,PQaRSb,IKPQ", -1.0  , 4, {22,5,16,21}},		//Ap[KLRS] += -1.0 W[IaLb] E3[PQaSRb] p[IKPQ]
    //{"KLSR,JaKb,PQaRSb,LJPQ", -1.0  , 4, {22,5,16,21}},		//Ap[KLRS] += -1.0 W[JaKb] E3[PQaSRb] p[LJPQ]
    //{"KLRS,JaLb,PQaRSb,KJPQ", -1.0  , 4, {22,5,16,21}},		//Ap[KLRS] += -1.0 W[JaLb] E3[PQaRSb] p[KJPQ]
    //{"KLbS,KIaR,PQaRSb,ILPQ", -1.0  , 4, {22,26,16,21}},		//Ap[KLRS] += -1.0 W[IabK] E3[PQabSR] p[ILPQ]
    //{"KLSb,LIaR,PQaRSb,IKPQ", -1.0  , 4, {22,26,16,21}},		//Ap[KLRS] += -1.0 W[IabL] E3[PQabRS] p[IKPQ]
    //{"KLbR,KJaS,PQaRSb,LJPQ", -1.0  , 4, {22,26,16,21}},		//Ap[KLRS] += -1.0 W[JabK] E3[PQaSbR] p[LJPQ]
    //{"KLRb,LJaS,PQaRSb,KJPQ", -1.0  , 4, {22,26,16,21}},		//Ap[KLRS] += -1.0 W[JabL] E3[PQaRbS] p[KJPQ]
    //{"KLcR,PabS,QabRSc,KLPQ", 1.0  , 4, {22,12,16,21}},		//Ap[KLRS] += 1.0 W[Pabc] E3[QabScR] p[KLPQ]
    //{"KLRc,PabS,QabRSc,LKPQ", 1.0  , 4, {22,12,16,21}},		//Ap[KLRS] += 1.0 W[Pabc] E3[QabRcS] p[LKPQ]
    //{"KLRc,QabS,PabRSc,KLPQ", 1.0  , 4, {22,12,16,21}},		//Ap[KLRS] += 1.0 W[Qabc] E3[PabRcS] p[KLPQ]
    //{"KLcR,QabS,PabRSc,LKPQ", 1.0  , 4, {22,12,16,21}},		//Ap[KLRS] += 1.0 W[Qabc] E3[PabScR] p[LKPQ]

    FArrayOffset nact = TensorById(22)->Sizes[2];
    FArrayOffset ncore = TensorById(22)->Sizes[0];
    SplitStackmem(Mem);
#pragma omp parallel 
    {
      void
	 *pBaseOfMemorylocal = Mem[omprank].Alloc(0);
      char Header0[10], Header[256];
      FILE *f = fopen("int/E3.npy", "rb");
      fread(&Header0[0], 1, 10, f);
      uint16_t HeaderSize = (uint16_t)Header0[8] + (((uint16_t)Header0[9]) << 8);
      //FILE *f = fopen("scratch/node0/spatial_threepdm.0.0.bin.unpack", "rb");
      std::vector<FSrciTensor> localTensors;
      localTensors.resize(14);
      for (int i=0; i<12; i++) {
	localTensors[i].Sizes = m_Tensors[i+30].Sizes; 
	localTensors[i].Strides = m_Tensors[i+30].Strides;
      } 
      localTensors[12].Sizes = m_Tensors[21].Sizes;
      localTensors[12].Strides = m_Tensors[21].Strides;
      localTensors[12].pData = m_Tensors[21].pData;

      localTensors[13].Sizes = m_Tensors[42].Sizes;
      localTensors[13].Strides = m_Tensors[42].Strides;

      double scale = 1.0;
      double * E3slicedata = Mem[omprank].AllocN(nact*nact*nact, scale);
      double* Ap2data = Mem[omprank].AllocN(nact*nact*ncore*ncore, scale);
      for (size_t i=0; i<nact*nact*ncore*ncore; i++)
	Ap2data[i] = 0.0;
      localTensors[13].pData = Ap2data;
      
      for (size_t RSB=0; RSB<nact*nact*nact; RSB++)
      {
	if (RSB%omp_get_num_threads() != omprank) continue;
	size_t R = RSB/nact/nact;
	size_t S = (RSB%(nact*nact))/nact;
	size_t b = (RSB%(nact*nact))%nact;

	fseek(f, HeaderSize+10+(R+S*nact+b*nact*nact)*nact*nact*nact*sizeof(double), SEEK_SET);
	fread(E3slicedata, sizeof(double), nact*nact*nact, f);
	localTensors[0].pData = &localTensors[13].operator()(0,0,R,S); 
	localTensors[1].pData = &TensorById(5)->operator()(0,0,0,b); 
	localTensors[2].pData = E3slicedata;//&TensorById(16)->operator()(0,0,0,R,S,b); 
	localTensors[4].pData = &TensorById(26)->operator()(0,0,0,R); 
	localTensors[5].pData = &TensorById(12)->operator()(0,0,0,S); 
	localTensors[6].pData = &localTensors[13].operator()(0,0,S,R); 
	localTensors[7].pData = &localTensors[13].operator()(0,0,b,S); 
	localTensors[8].pData = &localTensors[13].operator()(0,0,S,b); 
	localTensors[9].pData = &localTensors[13].operator()(0,0,b,R); 
	localTensors[10].pData = &localTensors[13].operator()(0,0,R,b); 
	localTensors[11].pData = &TensorById(26)->operator()(0,0,0,S); 
	ExecEquationSet(Method.EqsHandCode, localTensors, Mem[omprank]);      //Ap = A*p      
      }
#pragma omp critical
      {
	for (size_t i=0; i<nact*nact*ncore*ncore; i++)
	  TensorById(22)->pData[i] += localTensors[13].pData[i];
      }
      localTensors.clear();
      Mem[omprank].Free(pBaseOfMemorylocal);      
    }
    MergeStackmem(Mem);
  }
  else if (0 == strcmp(MethodName.c_str(), "NEVPT2_CCAA") ) {

    FArrayOffset nact = TensorById(21)->Sizes[2];
    FArrayOffset ncore = TensorById(21)->Sizes[0];

    SplitStackmem(Mem);
#pragma omp parallel 
    {
      void
	 *pBaseOfMemorylocal = Mem[omprank].Alloc(0);

      char AHeader0[10], AHeader[256];
      FILE *fA = fopen("int/E3.npy", "rb");
      fread(&AHeader0[0], 1, 10, fA);
      size_t AHeaderSize = (uint16_t)AHeader0[8] + (((uint16_t)AHeader0[9]) << 8);
      

      std::vector<FSrciTensor> localTensors;
      localTensors.resize(32);
      for (int i=0; i<32; i++) {
	localTensors[i].Sizes = m_Tensors[i].Sizes; 
	localTensors[i].Strides = m_Tensors[i].Strides;
	if (i < 29)
	  localTensors[i].pData = m_Tensors[i].pData;
      } 

      double scale = 1.0;
      double* E3slicedata = Mem[omprank].AllocN(nact*nact*nact*nact, scale);
      
      for (size_t RS=0; RS<nact*nact; RS++)
      {
	if (RS%omp_get_num_threads() != omprank) continue;
	size_t R = (RS)/nact;
	size_t S = (RS)%nact;
	fseek(fA, AHeaderSize+10+(S+R*nact)*nact*nact*nact*nact*sizeof(double), SEEK_SET);
	fread(E3slicedata, sizeof(double), nact*nact*nact*nact, fA);
	localTensors[29].pData = &localTensors[21].operator()(0,0,R,S);
	localTensors[30].pData = &localTensors[21].operator()(0,0,S,R);
	localTensors[31].pData = E3slicedata;
	ExecEquationSet(Method.EqsHandCode, localTensors, Mem[omprank]);      //Ap = A*p      

      }
      localTensors.clear();
      fclose(fA);
      Mem[omprank].Free(pBaseOfMemorylocal);      
    }
    MergeStackmem(Mem);

  }
  else if (0 == strcmp(MethodName.c_str(), "MRLCC_CCVV") ) {
    //{"abcd,abef,efcd", 8.0  , 3, {22,13,21}},		//8.0 Ap[abcd] W[abef] p[efcd] 
    //{"abcd,abef,efdc", -4.0  , 3, {22,13,21}},		//-4.0 Ap[abcd] W[abef] p[efdc] 

    void
      *pBaseOfMemory = Mem[omprank].Alloc(0);
    int ncore = TensorById(22)->Sizes[2];
    int nvirt = TensorById(22)->Sizes[0];
    FScalar scaleA = -4.0, scaleB = 1.0, scaleC=8.0;
    char T='t', N='n';
    FArrayOffset nvirtsq = nvirt*nvirt, ncoresq = ncore*ncore;
    FScalar *intermediateD = Mem[0].AllocN(nvirt*nvirt*ncore*ncore, scaleA);
    for (int i=0; i<ncore; i++)
      for (int j=0; j<ncore; j++)
	for (int b=0; b<nvirt; b++)
	  for (int a=0; a<nvirt; a++)
	    intermediateD[a+b*nvirt+j*nvirt*nvirt+i*nvirt*nvirt*ncore] = TensorById(21)->operator()(a,b,i,j);

    SplitStackmem(Mem);
#pragma omp parallel 
    {
      void
	*pBaseOfMemorylocal = Mem[omprank].Alloc(0);
      double *Wslice = Mem[omprank].AllocN(nvirt*nvirt*nvirt, scaleA);
      for (int b=0; b<nvirt; b++) {
	if (b%omp_get_num_threads() != omprank) continue;
	char filename[700];
	sprintf(filename, "int/W:eeee%04d", b);
	FILE* f = fopen(filename, "rb");
	fread(Wslice, sizeof(scaleA), nvirtsq*nvirt, f);
	
	dgemm_(N, N, nvirt, ncoresq, nvirtsq,
	       scaleC, Wslice, nvirt, TensorById(21)->pData, nvirtsq, scaleB, &TensorById(22)->operator()(0,b,0,0), nvirtsq);
	dgemm_(N, N, nvirt, ncoresq, nvirtsq,
	       scaleA, Wslice, nvirt, intermediateD, nvirtsq, scaleB, &TensorById(22)->operator()(0,b,0,0), nvirtsq);

	fclose(f);
      }
      Mem[omprank].Free(pBaseOfMemorylocal);
    }
    MergeStackmem(Mem);

    //dgemm_(N, N, nvirtsq, ncoresq, nvirtsq,
    //scaleC, TensorById(13)->pData, nvirtsq, TensorById(21)->pData, nvirtsq, scaleB, TensorById(22)->pData, nvirtsq);
    //dgemm_(N, N, nvirtsq, ncoresq, nvirtsq,
    //scaleA, TensorById(13)->pData, nvirtsq, intermediateD, nvirtsq, scaleB, TensorById(22)->pData, nvirtsq);

    Mem[omprank].Free(pBaseOfMemory);

  }
  else if (0 == strcmp(MethodName.c_str(), "MRLCC_ACVV") ) {
  //{"CDJR,ABCD,RQ,ABJQ", 2.0  , 4, {22,13,14,21}},		//Ap[CDJR] += 2.0 W[ABCD] E1[RQ] p[ABJQ]
  //{"CDJR,ABDC,RQ,ABJQ", -1.0  , 4, {22,13,14,21}},		//Ap[CDJR] += -1.0 W[ABDC] E1[RQ] p[ABJQ]
    void
      *pBaseOfMemory = Mem[omprank].Alloc(0);
    FArrayOffset ncore = TensorById(22)->Sizes[2];
    FArrayOffset nact = TensorById(22)->Sizes[3];
    FArrayOffset nvirt = TensorById(22)->Sizes[0];
    FScalar scaleA = -1.0, scaleB = 1.0, scaleC=2.0, scaleD=0.0;
    char T='t', N='n';
    FArrayOffset nvirtsq = nvirt*nvirt, 
      ncoreact = ncore*nact,
      nvvc = nvirt*nvirt*ncore;

    FScalar *intermediateD = Mem[0].AllocN(nvirt*nvirt*ncore*nact, scaleA);

    dgemm_(N, N, nvvc, nact, nact,
	   scaleB, TensorById(21)->pData, nvvc, TensorById(14)->pData, nact, scaleD, TensorById(30)->pData, nvvc);
    
    for (int r=0; r<nact; r++)
      for (int j=0; j<ncore; j++)
	for (int b=0; b<nvirt; b++)
	  for (int a=0; a<nvirt; a++)
	    intermediateD[b+a*nvirt+j*nvirt*nvirt+r*nvirt*nvirt*ncore] = TensorById(30)->operator()(a,b,j,r);
    
    SplitStackmem(Mem);
#pragma omp parallel 
    {
      void
	*pBaseOfMemorylocal = Mem[omprank].Alloc(0);
      double *Wslice = Mem[omprank].AllocN(nvirt*nvirt*nvirt, scaleA);
      for (int b=0; b<nvirt; b++) {
	if (b%omp_get_num_threads() != omprank) continue;
	char filename[700];
	sprintf(filename, "int/W:eeee%04d", b);
	FILE* f = fopen(filename, "rb");
	fread(Wslice, sizeof(scaleA), nvirtsq*nvirt, f);
	
	dgemm_(N, N, nvirt, ncoreact, nvirtsq,
	       scaleC, Wslice, nvirt, TensorById(30)->pData, nvirtsq, scaleB, &TensorById(22)->operator()(0,b,0,0), nvirtsq);
	dgemm_(N, N, nvirt, ncoreact, nvirtsq,
	       scaleA, Wslice, nvirt, intermediateD, nvirtsq, scaleB, &TensorById(22)->operator()(0,b,0,0), nvirtsq);

	fclose(f);
      }
      Mem[omprank].Free(pBaseOfMemorylocal);
    }
    MergeStackmem(Mem);
    Mem[omprank].Free(pBaseOfMemory);
  }

  //zero out all intermediates
  for (uint i = 0; i != Method.nTensorDecls; ++i) {
    FTensorDecl const
      &Decl = Method.pTensorDecls[i];
    if (Decl.Usage == USAGE_Intermediate)
      memset(m_Tensors[i].pData, 0, m_Tensors[i].nValues() *sizeof(m_Tensors[i].pData[0]));
  }
}


void FJobContext::InitAmplitudes(FMemoryStack2 &Mem)
{
   // set all amplitudes to zero.
   ClearTensors(USAGE_Amplitude);

   if (0 == strcmp(Method.perturberClass, "AAVV")) {
     ExecEquationSet(Method.Overlap, m_Tensors, Mem);

   }
   else if (0 == strcmp(Method.perturberClass, "CCAA")) {
     ExecEquationSet(Method.Overlap, m_Tensors, Mem);
   }
   else if (0 == strcmp(Method.perturberClass, "ACVV")) {
     ExecEquationSet(Method.Overlap, m_Tensors, Mem);

   }
   else if (0 == strcmp(Method.perturberClass, "CCAV")) {
     ExecEquationSet(Method.Overlap, m_Tensors, Mem);
     FNdArrayView *b = TN("b");     
   }
   else if (0 == strcmp(Method.perturberClass, "CCVV")) {
     ExecEquationSet(Method.Overlap, m_Tensors, Mem);
   }
   else if (0 == strcmp(Method.perturberClass, "CAAV")) {
     FNdArrayView *p = TN("p");
     FNdArrayView *Ap = TN("Ap");
     FNdArrayView *bt = TN("b");
     FNdArrayView *Bt = TN("B");
     bt->pData = TN("b1")->pData;    //just make the pointer
     p->pData = TN("p1")->pData;    //just make the pointer
     Ap->pData = TN("Ap1")->pData;  //just make the pointer
     Bt->pData = TN("B1")->pData;  //just make the pointer
     p->Sizes.push_back(TN("p1")->nValues()*2);
     p->Strides.push_back(1);
     Ap->Sizes.push_back(TN("Ap1")->nValues()*2);
     Ap->Strides.push_back(1);
     bt->Sizes.push_back(TN("b1")->nValues()*2);
     bt->Strides.push_back(1);
     Bt->Sizes.push_back(TN("B1")->nValues()*2);
     Bt->Strides.push_back(1);


     //FillData(31, Mem);
     //FillData(32, Mem);
     FNdArrayView *t = TN("t");
     FNdArrayView *w1 = TND("W:eaca");
     FNdArrayView *w2 = TND("W:aeca");
     FNdArrayView *f1 = TND("f:ec");

     int ncore=t->Sizes[0], nvirt=t->Sizes[3];
     int nact = t->Sizes[2];
     for (int i=0; i<ncore; i++)
     for (int p=0; p<nact;  p++)
     for (int a=0; a<nvirt; a++) 
       w1->operator()(a,p,i,p) += f1->operator()(a,i)/(WfDecl.nActElec);

     ExecEquationSet(Method.Overlap, m_Tensors, Mem);
     //DeAllocate("W:aeca", Mem);
     //DeAllocate("W:eaca", Mem);
   }

   ct::FArrayNpy guess;
   if (guessInput.compare("") != 0) {
     FScalar zero = 0.0;
     PrintResult("Reading guess from "+guessInput, zero, 0);
     ct::ReadNpy( guess ,guessInput);
     Copy(*TN("T"), ViewFromNpy(guess));
   }
   else
     Copy(*TN("t"), *TN("b"));

}

void FJobContext::MakeOverlapAndOrthogonalBasis(ct::FMemoryStack2 *Mem) {
   //make S and Shalf
   if (0 == strcmp(Method.perturberClass, "AAVV")) {
     FNdArrayView *S1 = TN("S1");
     Copy(*TND("S2:aaaa"), *TND("E2:aaaa"));
     FNdArrayView *a = TND("S2:aaaa"), *b=TND("E2:aaaa");
     int nact = a->Sizes[0];

     for (int k=0; k<a->Sizes[0]; k++)
       for (int l=0; l<a->Sizes[0]; l++)
	 for (int j=0; j<a->Sizes[0]; j++)
	   for (int i=0; i<a->Sizes[0]; i++) {
	     a->operator()(i,j,k,l) += b->operator()(i,j,l,k);

	     S1->operator()(i,j,k,l) += b->operator()(i,j,k,l);
	     S1->operator()(i,j+nact,k,l+nact) += b->operator()(i,j,k,l);
	     S1->operator()(i,j+nact,k,l) += b->operator()(i,j,l,k);
	     S1->operator()(i,j,k,l+nact) += b->operator()(i,j,l,k);
	   }

      //make X(rs,mu) and Xhalf(rs,mu)  which are stored in place of overlap
     FScalar *eigen1 = (FScalar*)::malloc(sizeof(FScalar)*S1->Strides[2]);
     FScalar *eigen2 = (FScalar*)::malloc(sizeof(FScalar)*a->Strides[2]);
     ct::Diagonalize(eigen1, TN("S1")->pData, S1->Strides[2], S1->Strides[2]);
     ct::Diagonalize(eigen2, TND("S2:aaaa")->pData, a->Strides[2], a->Strides[2]);
     
     //FNdArrayView *b = TND("W:eecc");
     size_t aa = S1->Strides[2];
     
     for (int i=0; i<aa; i++) {     
       if (fabs(eigen1[i]) < ThrTrun) {
	 for (int j=0; j<aa; j++)
	   S1->pData[j+aa*i] = 0.0;
       }
       else {
	 for (int j=0; j<aa; j++)
	   S1->pData[j+i*aa] = S1->pData[j+i*aa]/(pow(eigen1[i],0.5));
       }
     }
       
     aa = a->Strides[2];
     for (int i=0; i<aa; i++) {     
       if (fabs(eigen2[i]) < ThrTrun) {
	 for (int j=0; j<aa; j++)
	   TND("S2:aaaa")->pData[j+aa*i] = 0.0;
       }
       else {
	 for (int j=0; j<aa; j++) {
	   TND("S2:aaaa")->pData[j+i*aa] = TND("S2:aaaa")->pData[j+i*aa]/(pow(eigen2[i],0.5));
	 }
       }
     }
 
     ::free(eigen2);
     ::free(eigen1);
   }
   else if (0 == strcmp(Method.perturberClass, "CCAA")) {

     FNdArrayView *a = TND("S2:aaaa"), *b=TND("E2:aaaa");
     FNdArrayView *S1 = TND("S1:aaaa"); S1->ClearData(); a->ClearData();
     ExecEquationSet(Method.MakeS1, m_Tensors, Mem[0]);
     ExecEquationSet(Method.MakeS2, m_Tensors, Mem[0]);
     
     //make X(rs,mu) and Xhalf(rs,mu)  which are stored in place of overlap
     FScalar *eigen1 = (FScalar*)::malloc(sizeof(FScalar)*a->Strides[2]);
     FScalar *eigen2 = (FScalar*)::malloc(sizeof(FScalar)*a->Strides[2]);
     ct::Diagonalize(eigen1, TND("S1:aaaa")->pData, a->Strides[2], a->Strides[2]);
     ct::Diagonalize(eigen2, TND("S2:aaaa")->pData, a->Strides[2], a->Strides[2]);
     
     size_t aa = a->Strides[2];
     
     for (int i=0; i<aa; i++) {     
       if (fabs(eigen1[i]) < ThrTrun) {
	 for (int j=0; j<aa; j++)
	   TND("S1:aaaa")->pData[j+aa*i] = 0.0;
       }
       else {
	 for (int j=0; j<aa; j++)
	   TND("S1:aaaa")->pData[j+i*aa] = TND("S1:aaaa")->pData[j+i*aa]/(pow(eigen1[i],0.5));
       }
       
       if (fabs(eigen2[i]) < ThrTrun) {
	 for (int j=0; j<aa; j++)
	   TND("S2:aaaa")->pData[j+aa*i] = 0.0;
       }
       else {
	 for (int j=0; j<aa; j++)
	   TND("S2:aaaa")->pData[j+i*aa] = TND("S2:aaaa")->pData[j+i*aa]/(pow(eigen2[i],0.5));
       }
     }
     ::free(eigen2);
     ::free(eigen1);
   }
   else if (0 == strcmp(Method.perturberClass, "ACVV")) {
     FNdArrayView *S1 = TND("S1:AA"), *E=TND("E1:aa");
     FNdArrayView *S2 = TND("S2:aa");
     int nact = E->Sizes[0];
     for (int k=0; k<nact; k++)
       for (int l=0; l<nact; l++) {
	 S1->operator()(k,l) = 2.*E->operator()(k,l);
	 S1->operator()(k+nact,l+nact) = 2.*E->operator()(k,l);
	 S1->operator()(k+nact,l) = -1.*E->operator()(k,l);
	 S1->operator()(k,l+nact) = -1.*E->operator()(k,l);
	 S2->operator()(k,l) = 1.*E->operator()(k,l);
       }     

     //make X(rs,mu) and Xhalf(rs,mu)  which are stored in place of overlap
     FScalar *eigen1 = (FScalar*)::malloc(sizeof(FScalar)*S1->Strides[1]);
     FScalar *eigen2 = (FScalar*)::malloc(sizeof(FScalar)*S2->Strides[1]);
     ct::Diagonalize(eigen1, S1->pData, S1->Strides[1], S1->Strides[1]);
     ct::Diagonalize(eigen2, S2->pData, S2->Strides[1], S2->Strides[1]);
     
     for (int i=0; i<2*nact; i++) {     
       if (fabs(eigen1[i]) < ThrTrun) {
	 for (int j=0; j<2*nact; j++)
	   S1->pData[j+2*nact*i] = 0.0;
       }
       else {
	 for (int j=0; j<2*nact; j++)
	   S1->pData[j+i*2*nact] = S1->pData[j+i*2*nact]/(pow(eigen1[i],0.5));
       }
     }
     for (int i=0; i<nact; i++) {            
       if (fabs(eigen2[i]) < ThrTrun) {
	 for (int j=0; j<nact; j++)
	   S2->pData[j+nact*i] = 0.0;
       }
       else {
	 for (int j=0; j<nact; j++)
	   S2->pData[j+i*nact] = S2->pData[j+i*nact]/(pow(eigen2[i],0.5));
       }
     }
     ::free(eigen2);
     ::free(eigen1);
   }
   else if (0 == strcmp(Method.perturberClass, "CCAV")) {
     FNdArrayView *S1 = TND("S1:AA"), *E=TND("E1:aa");
     FNdArrayView *S2 = TND("S2:aa");
     int nact = E->Sizes[0];
     for (int k=0; k<nact; k++)
       for (int l=0; l<nact; l++) {
	 S2->operator()(k,l) = -1.*E->operator()(k,l);
	 S1->operator()(k,l) = -2.*E->operator()(k,l);
	 S1->operator()(k+nact,l+nact) = -2.*E->operator()(k,l);
	 S1->operator()(k+nact,l) =  1.*E->operator()(k,l);
	 S1->operator()(k,l+nact) =  1.*E->operator()(k,l);
	 if (k == l) {
	   S1->operator()(k,k) +=  4.;
	   S1->operator()(k+nact,k) +=  -2.;
	   S1->operator()(k,k+nact) +=  -2.;
	   S1->operator()(k+nact,k+nact) +=  4.;
	   S2->operator()(k,k) +=  2.;
	 }
       }     

     //std::cout << "Norm "<<S1->nValues()<<" "<< ct::Dot(S1->pData, S1->pData, S1->nValues())<<std::endl;
     //std::cout << "Norm "<<S2->nValues()<<" "<< ct::Dot(S2->pData, S2->pData, S2->nValues())<<std::endl;

     //make X(rs,mu) and Xhalf(rs,mu)  which are stored in place of overlap
     FScalar *eigen1 = (FScalar*)::malloc(sizeof(FScalar)*S1->Strides[1]);
     FScalar *eigen2 = (FScalar*)::malloc(sizeof(FScalar)*S2->Strides[1]);
     ct::Diagonalize(eigen1, S1->pData, S1->Strides[1], S1->Strides[1]);
     ct::Diagonalize(eigen2, S2->pData, S2->Strides[1], S2->Strides[1]);
     
     for (int i=0; i<2*nact; i++) {     
       if (fabs(eigen1[i]) < ThrTrun) {
	 for (int j=0; j<2*nact; j++)
	   S1->pData[j+2*nact*i] = 0.0;
       }
       else {
	 for (int j=0; j<2*nact; j++)
	   S1->pData[j+i*2*nact] = S1->pData[j+i*2*nact]/(pow(eigen1[i],0.5));
       }
     }
     for (int i=0; i<nact; i++) {            
       if (fabs(eigen2[i]) < ThrTrun) {
	 for (int j=0; j<nact; j++)
	   S2->pData[j+nact*i] = 0.0;
       }
       else {
	 for (int j=0; j<nact; j++)
	   S2->pData[j+i*nact] = S2->pData[j+i*nact]/(pow(eigen2[i],0.5));
       }
     }
     ::free(eigen2);
     ::free(eigen1);
   }
   else if (0 == strcmp(Method.perturberClass, "CAAV")) {
     FNdArrayView *S1 = TN("S1"), *E1=TN("E1"), *E2=TN("E2");
     S1->ClearData();     
     int nact = E1->Sizes[0];


     S1->Sizes.resize(0);S1->Strides.resize(0);
     S1->Sizes.push_back(2*nact*nact); S1->Sizes.push_back(2*nact*nact);
     S1->Strides.push_back(1); S1->Strides.push_back(2*nact*nact);

     int fullrange = nact*nact;
     for (int r=0; r<nact; r++)
     for (int s=0; s<nact; s++) 
     for (int q=0; q<nact; q++)
     for (int p=0; p<nact; p++){
       if (p==r)
	 S1->operator()(p+q*nact,r+s*nact) += 2.*E1->operator()(s,q);
       S1->operator()(p+q*nact,r+s*nact)   += 2.*E2->operator()(p,s,q,r);


       if (p==r)
	 S1->operator()(p+q*nact+fullrange, r+s*nact) += -1.*E1->operator()(s,q);
       S1->operator()(p+q*nact+fullrange, r+s*nact)   += -1.*E2->operator()(p,s,q,r);


       if (p==r)
	 S1->operator()(r+s*nact, p+q*nact+fullrange) += -1.*E1->operator()(s,q);
       S1->operator()(r+s*nact, p+q*nact+fullrange)   += -1.*E2->operator()(p,s,q,r);


       if (p==r)
	 S1->operator()(p+q*nact+fullrange, r+s*nact+fullrange) += 2.*E1->operator()(s,q);
       S1->operator()(p+q*nact+fullrange, r+s*nact+fullrange)   += -1.*E2->operator()(p,s,r,q);
 
     }

     //make X(rs,mu) and Xhalf(rs,mu)  which are stored in place of overlap
     FScalar *eigen1 = (FScalar*)::malloc(sizeof(FScalar)*2*fullrange);
     ct::Diagonalize(eigen1, S1->pData, 2*fullrange, 2*fullrange);

     for (int i=0; i<2*nact*nact; i++) {     
       if (fabs(eigen1[i]) < ThrTrun) {
	 for (int j=0; j<2*nact*nact; j++)
	   S1->pData[j+2*nact*nact*i] = 0.0;
       }
       else {
	 for (int j=0; j<2*nact*nact; j++)
	   S1->pData[j+i*2*nact*nact] = S1->pData[j+i*2*nact*nact]/(pow(eigen1[i],0.5));
       }
     }
     ::free(eigen1);

   }
}
void FJobContext::MakeOrthogonal(std::string StringA, std::string StringB) {
   if (0 == strcmp(Method.perturberClass, "AAVV")) {
     FNdArrayView *t = TN(StringA); 
     FNdArrayView *Ta = TN(StringB); Ta->ClearData();
     FNdArrayView *S1 = TN("S1");
     FNdArrayView *S2 = TN("S2");

     int nact = t->Sizes[2];
#pragma omp parallel for schedule(dynamic)
     for (int rs=0; rs<t->Sizes[2]*t->Sizes[2]; rs++)
       for (int p=0; p<t->Sizes[2]; p++)
	 for (int q=0; q<t->Sizes[2]; q++) 
	   for (int c=0; c<t->Sizes[0]; c++)
	     for (int b=c; b<t->Sizes[0]; b++)
	       {
		 int r = rs/t->Sizes[2], s=rs%t->Sizes[2];
		 if (b == c) 
		   Ta->operator()(b,c,r,s) += S2->operator()(p,q,r,s)*t->operator()(b,c,p,q);
		 else {
		   Ta->operator()(b,c,r,s) += S1->operator()(p,q,r,s)     *t->operator()(b,c,p,q) + S1->operator()(p,q+nact,r,s)     *t->operator()(c,b,p,q);
		   Ta->operator()(c,b,r,s) += S1->operator()(p,q,r,s+nact)*t->operator()(b,c,p,q) + S1->operator()(p,q+nact,r,s+nact)*t->operator()(c,b,p,q);
		 }
	       }
     /*
     for (int p=0; p<t->Sizes[2]; p++)
       for (int q=0; q<t->Sizes[2]; q++)
	 for (int r=0; r<t->Sizes[2]; r++)
	   for (int s=0; s<t->Sizes[2]; s++) 
	     for (int c=0; c<t->Sizes[0]; c++)
	       for (int b=c; b<t->Sizes[0]; b++)
		 {
		   if (b == c) 
		     Ta->operator()(b,c,r,s) += S2->operator()(p,q,r,s)*t->operator()(b,c,p,q);
		   else {
		     Ta->operator()(b,c,r,s) += S1->operator()(p,q,r,s)     *t->operator()(b,c,p,q) + S1->operator()(p,q+nact,r,s)     *t->operator()(c,b,p,q);
		     Ta->operator()(c,b,r,s) += S1->operator()(p,q,r,s+nact)*t->operator()(b,c,p,q) + S1->operator()(p,q+nact,r,s+nact)*t->operator()(c,b,p,q);
		   }
		 }
     */
   }
   else if (0 == strcmp(Method.perturberClass, "CCVV")) {
     FNdArrayView *t = TN(StringA); 
     FNdArrayView *Ta = TN(StringB); Ta->ClearData();
     
     for (int i=0; i<t->Sizes[2]; i++)
       for (int j=i; j<t->Sizes[2]; j++) 
	 for (int c=0; c<t->Sizes[0]; c++)
	   for (int b=c; b<t->Sizes[0]; b++)
	     {
	       if (b == c) {
		 Ta->operator()(b,c,i,j) += (0.*t->operator()(b,c,i,j) + 0.*t->operator()(b,c,j,i))/pow(8,0.5);
		 Ta->operator()(b,c,j,i) += (   t->operator()(b,c,i,j) +    t->operator()(b,c,j,i))/pow(8,0.5);
	       }
	       else {
		 Ta->operator()(b,c,i,j) += ( t->operator()(b,c,i,j)*0.  + t->operator()(b,c,j,i)*0.  + t->operator()(c,b,i,j)*0.  + t->operator()(c,b,j,i)*0. ) /pow( 4.0,0.5);
		 Ta->operator()(b,c,j,i) += ( t->operator()(b,c,i,j)*0.  + t->operator()(b,c,j,i)*0.  + t->operator()(c,b,i,j)*0.  + t->operator()(c,b,j,i)*0. ) /pow( 4.0,0.5);
		 Ta->operator()(c,b,i,j) += ( t->operator()(b,c,i,j)*0.5 + t->operator()(b,c,j,i)*0.5 + t->operator()(c,b,i,j)*0.5 + t->operator()(c,b,j,i)*0.5) /pow( 4.0,0.5);
		 Ta->operator()(c,b,j,i) += (-t->operator()(b,c,i,j)*0.5 + t->operator()(b,c,j,i)*0.5 + t->operator()(c,b,i,j)*0.5 - t->operator()(c,b,j,i)*0.5) /pow(12.0,0.5);
	       }
	     }
   }
   else if (0 == strcmp(Method.perturberClass, "CCAA")) {
     FNdArrayView *t = TN(StringA); 
     FNdArrayView *Ta = TN(StringB); Ta->ClearData();
     FNdArrayView *S1 = TND("S1:aaaa");
     FNdArrayView *S2 = TND("S2:aaaa");
     
     for (int p=0; p<t->Sizes[2]; p++)
       for (int q=0; q<t->Sizes[2]; q++)
	 for (int r=0; r<t->Sizes[2]; r++)
	   for (int s=0; s<t->Sizes[2]; s++) 
	     for (int j=0; j<t->Sizes[0]; j++)
	       for (int i=j; i<t->Sizes[0]; i++)
		 {
		   if (i == j) 
		     Ta->operator()(i,j,r,s) += S2->operator()(p,q,r,s)*t->operator()(i,j,p,q);
		   else {
		     Ta->operator()(i,j,r,s) += S1->operator()(p,q,r,s)*t->operator()(i,j,p,q);
		     Ta->operator()(j,i,r,s)  = 0.0;
		   }
		 }
     //std::cout << "Norm "<<S2->nValues()<<" "<< ct::Dot(S2->pData, S2->pData, S2->nValues())<<std::endl;
     //std::cout << "Norm "<<t->nValues()<<" "<< ct::Dot(t->pData, t->pData, t->nValues())<<std::endl;
     //std::cout << "Norm "<<Ta->nValues()<<" "<< ct::Dot(Ta->pData, Ta->pData, Ta->nValues())<<std::endl;
   }
   else if (0 == strcmp(Method.perturberClass, "CAAV")) {
     FNdArrayView *t = TN(StringA); 
     FNdArrayView *Ta = TN(StringB); Ta->ClearData();
     FNdArrayView *S1 = TN("S1");

     int ncore = m_Domains['c'].nSize, nact = m_Domains['a'].nSize, nvirt = m_Domains['e'].nSize;

     int fullrange = ncore*nact*nact*nvirt;
     for (int p=0; p<nact; p++)
       for (int q=0; q<nact; q++)
	 for (int r=0; r<nact; r++)
	   for (int s=0; s<nact; s++) 
	     for (int a=0; a<nvirt; a++)
	       for (int i=0; i<ncore; i++)
		 {
		   Ta->pData[i+p*ncore+q*ncore*nact+a*ncore*nact*nact] += t->pData[i+r*ncore+s*ncore*nact+a*ncore*nact*nact]*S1->operator()(r+s*nact,p+q*nact) 
		                                            +t->pData[i+r*ncore+s*ncore*nact+a*ncore*nact*nact + fullrange]* S1->operator()(r+s*nact+nact*nact,p+q*nact);
		   Ta->pData[i+p*ncore+q*ncore*nact+a*ncore*nact*nact+fullrange] += t->pData[i+r*ncore+s*ncore*nact+a*ncore*nact*nact]*S1->operator()(r+s*nact,p+q*nact+nact*nact)
                                                            +t->pData[i+r*ncore+s*ncore*nact+a*ncore*nact*nact + fullrange]* S1->operator()(r+s*nact+nact*nact,p+q*nact+nact*nact);
		   //std::cout<<t->pData[i+r*ncore+s*ncore*nact+a*ncore*nact*nact]<<"  "<<S1->operator()(r,s,p,q)<<"  "<<Ta->pData[i+p*ncore+q*ncore*nact+a*ncore*nact*nact]<<"  "<<i+p*ncore+q*ncore*nact+a*ncore*nact*nact<<"  "<<Ta->nValues()<<std::endl;
		 }

   }
   else if (0 == strcmp(Method.perturberClass, "ACVV")) {
     FNdArrayView *t = TN(StringA); 
     FNdArrayView *Ta = TN(StringB); Ta->ClearData();
     FNdArrayView *S1 = TN("S1");
     FNdArrayView *S2 = TN("S2");
     int nact = t->Sizes[3];
     for (int p=0; p<t->Sizes[3]; p++)
       for (int q=0; q<t->Sizes[3]; q++)
	 for (int i=0; i<t->Sizes[2]; i++) 
	   for (int c=0; c<t->Sizes[0]; c++)
	     for (int b=c; b<t->Sizes[0]; b++)
	     {
	       if (b == c) 
		 Ta->operator()(b,c,i,q) += S2->operator()(p,q)*t->operator()(b,c,i,p);
	       else {
		 Ta->operator()(b,c,i,q) += S1->operator()(p,q)          *t->operator()(b,c,i,p);
		 Ta->operator()(b,c,i,q) += S1->operator()(p+nact,q)     *t->operator()(c,b,i,p);
		 Ta->operator()(c,b,i,q) += S1->operator()(p,q+nact)     *t->operator()(b,c,i,p);
		 Ta->operator()(c,b,i,q) += S1->operator()(p+nact,q+nact)*t->operator()(c,b,i,p);
	       }
	     }

   }
   else if (0 == strcmp(Method.perturberClass, "CCAV")) {
     FNdArrayView *t = TN(StringA); 
     FNdArrayView *Ta = TN(StringB); Ta->ClearData();
     FNdArrayView *S1 = TN("S1");
     FNdArrayView *S2 = TN("S2");
     int nact = t->Sizes[2];
     for (int i=0; i<t->Sizes[0]; i++)
       for (int j=0; j<i+1; j++)
	 for (int a=0; a<t->Sizes[3]; a++) 
	   for (int p=0; p<t->Sizes[2]; p++)
	     for (int q=0; q<t->Sizes[2]; q++)
	     {
	       if (i == j) 
		 Ta->operator()(i,j,p,a) += S2->operator()(q,p)*t->operator()(i,j,q,a);
	       else {
		 Ta->operator()(i,j,p,a) += S1->operator()(q,p)*t->operator()(i,j,q,a);
		 Ta->operator()(i,j,p,a) += S1->operator()(q+nact,p)*t->operator()(j,i,q,a);
		 Ta->operator()(j,i,p,a) += S1->operator()(q,p+nact)*t->operator()(i,j,q,a);
		 Ta->operator()(j,i,p,a) += S1->operator()(q+nact,p+nact)*t->operator()(j,i,q,a);
	       }
	     }

   }
     //nothing to do

}

   
void FJobContext::BackToNonOrthogonal(std::string StringA, std::string StringB) {

  if (0 == strcmp(Method.perturberClass, "AAVV")) {
    FNdArrayView *t = TN(StringA); t->ClearData();
    FNdArrayView *Ta = TN(StringB); 
    FNdArrayView *S1 = TN("S1");
    FNdArrayView *S2 = TN("S2");

    int nact = t->Sizes[2];
#pragma omp parallel for schedule(dynamic)
    for (int pq=0; pq<t->Sizes[2]*t->Sizes[2]; pq++)
      //for (int q=0; q<t->Sizes[2]; q++)
	for (int r=0; r<t->Sizes[2]; r++)
	  for (int s=0; s<t->Sizes[2]; s++) 
	    for (int c=0; c<t->Sizes[0]; c++)
	      for (int b=c; b<t->Sizes[0]; b++)
		{
		  int p=pq/t->Sizes[2], q=pq%t->Sizes[2];
		  if (b == c) 
		    t->operator()(b,c,p,q) += S2->operator()(p,q,r,s)*Ta->operator()(b,c,r,s);
		  else {
		    t->operator()(b,c,p,q) += S1->operator()(p,q,r,s)     *Ta->operator()(b,c,r,s) + S1->operator()(p,q,r,s+nact)     *Ta->operator()(c,b,r,s);
		    t->operator()(c,b,p,q) += S1->operator()(p,q+nact,r,s)*Ta->operator()(b,c,r,s) + S1->operator()(p,q+nact,r,s+nact)*Ta->operator()(c,b,r,s);
		  }
		}

    //in the equations we assume that t[a,b,p,q,] = t[b,a,q,p]
    //this equality might be lost in single precision
    //so for numerical reasons it is important to explicitly symmetrize
#pragma omp parallel for schedule(dynamic)
    for (int p=0; p<t->Sizes[2]; p++)
      for (int q=p; q<t->Sizes[2]; q++)
	for (int c=0; c<t->Sizes[0]; c++)
	  for (int b=c+1; b<t->Sizes[0]; b++)
	    {
	      float x = t->operator()(b,c,p,q), y = t->operator()(c,b,q,p);
	      t->operator()(b,c,p,q) = 0.5*(x+y);
	      t->operator()(c,b,q,p) = 0.5*(x+y);

	      x = t->operator()(b,c,q,p); y = t->operator()(c,b,p,q);
	      t->operator()(b,c,q,p) = 0.5*(x+y);
	      t->operator()(c,b,p,q) = 0.5*(x+y);
	    }
  }
  if (0 == strcmp(Method.perturberClass, "CCAA")) {
    FNdArrayView *t = TN(StringA); t->ClearData();
    FNdArrayView *Ta = TN(StringB); 
    FNdArrayView *S1 = TND("S1:aaaa");
    FNdArrayView *S2 = TND("S2:aaaa");
    
    for (int p=0; p<t->Sizes[2]; p++)
      for (int q=0; q<t->Sizes[2]; q++)
	for (int r=0; r<t->Sizes[2]; r++)
	  for (int s=0; s<t->Sizes[2]; s++) 
	    for (int j=0; j<t->Sizes[0]; j++)
	      for (int i=j; i<t->Sizes[0]; i++)
		{
		  if (i == j) 
		    t->operator()(i,j,p,q) += S2->operator()(p,q,r,s)*Ta->operator()(i,j,r,s);
		  else {
		    t->operator()(i,j,p,q) += S1->operator()(p,q,r,s)*Ta->operator()(i,j,r,s);
		    t->operator()(j,i,p,q)  = 0.0;
		  }
		}
    //std::cout << "Norm "<<TND("T:eeaa")->nValues()<<" "<< ct::Dot(TND("T:eeaa")->pData, TND("T:eeaa")->pData, TND("t:eeaa")->nValues())<<std::endl;
  }
  else if (0 == strcmp(Method.perturberClass, "ACVV")) {
     FNdArrayView *t = TN(StringA); t->ClearData();
     FNdArrayView *Ta = TN(StringB); 
     FNdArrayView *S1 = TN("S1");
     FNdArrayView *S2 = TN("S2");
     int nact = t->Sizes[3];
     for (int p=0; p<t->Sizes[3]; p++)
       for (int q=0; q<t->Sizes[3]; q++)
	 for (int i=0; i<t->Sizes[2]; i++) 
	   for (int c=0; c<t->Sizes[0]; c++)
	     for (int b=c; b<t->Sizes[0]; b++)
	     {
	       if (b == c) 
		 t->operator()(b,c,i,p) += S2->operator()(p,q)*Ta->operator()(b,c,i,q);
	       else {
		 t->operator()(b,c,i,p) += S1->operator()(p,q)          *Ta->operator()(b,c,i,q);
		 t->operator()(b,c,i,p) += S1->operator()(p,q+nact)     *Ta->operator()(c,b,i,q);
		 t->operator()(c,b,i,p) += S1->operator()(p+nact,q)     *Ta->operator()(b,c,i,q);
		 t->operator()(c,b,i,p) += S1->operator()(p+nact,q+nact)*Ta->operator()(c,b,i,q);
	       }
	     }

   }
   else if (0 == strcmp(Method.perturberClass, "CCAV")) {
     FNdArrayView *t = TN(StringA);  t->ClearData();
     FNdArrayView *Ta = TN(StringB);
     FNdArrayView *S1 = TN("S1");
     FNdArrayView *S2 = TN("S2");
     int nact = t->Sizes[2];
     for (int i=0; i<t->Sizes[0]; i++)
       for (int j=0; j<i+1; j++)
	 for (int a=0; a<t->Sizes[3]; a++) 
	   for (int p=0; p<t->Sizes[2]; p++)
	     for (int q=0; q<t->Sizes[2]; q++)
	     {
	       if (i == j) 
		 t->operator()(i,j,q,a) += S2->operator()(q,p)*Ta->operator()(i,j,p,a);
	       else {
		 t->operator()(i,j,q,a) += S1->operator()(q,p)*Ta->operator()(i,j,p,a);
		 t->operator()(i,j,q,a) += S1->operator()(q,p+nact)*Ta->operator()(j,i,p,a);
		 t->operator()(j,i,q,a) += S1->operator()(q+nact,p)*Ta->operator()(i,j,p,a);
		 t->operator()(j,i,q,a) += S1->operator()(q+nact,p+nact)*Ta->operator()(j,i,p,a);
	       }
	     }

   }
   else if (0 == strcmp(Method.perturberClass, "CCVV")) {
     FNdArrayView *t = TN(StringA); t->ClearData();
     FNdArrayView *Ta = TN(StringB); 
     
     for (int i=0; i<t->Sizes[2]; i++)
       for (int j=i; j<t->Sizes[2]; j++) 
	 for (int c=0; c<t->Sizes[0]; c++)
	   for (int b=c; b<t->Sizes[0]; b++)
	     {
	       if (b == c) {
		 t->operator()(b,c,i,j) += (0.*Ta->operator()(b,c,i,j) +    Ta->operator()(b,c,j,i))/pow(8,0.5);
		 t->operator()(b,c,j,i) += (0.*Ta->operator()(b,c,i,j) +    Ta->operator()(b,c,j,i))/pow(8,0.5);
	       }
	       else {
		 t->operator()(b,c,i,j) +=   Ta->operator()(b,c,i,j)*0.0 + Ta->operator()(b,c,j,i)*0.0 + Ta->operator()(c,b,i,j)*0.5/2. - Ta->operator()(c,b,j,i)*0.5/pow(12.0,0.5);
		 t->operator()(b,c,j,i) +=  -Ta->operator()(b,c,i,j)*0.0 + Ta->operator()(b,c,j,i)*0.0 + Ta->operator()(c,b,i,j)*0.5/2. + Ta->operator()(c,b,j,i)*0.5/pow(12.0,0.5);
		 t->operator()(c,b,i,j) +=   Ta->operator()(b,c,i,j)*0.0 + Ta->operator()(b,c,j,i)*0.0 + Ta->operator()(c,b,i,j)*0.5/2. + Ta->operator()(c,b,j,i)*0.5/pow(12.0,0.5);
		 t->operator()(c,b,j,i) +=   Ta->operator()(b,c,i,j)*0.0 + Ta->operator()(b,c,j,i)*0.0 + Ta->operator()(c,b,i,j)*0.5/2. - Ta->operator()(c,b,j,i)*0.5/pow(12.0,0.5);
	       }

	     }
   }
   else if (0 == strcmp(Method.perturberClass, "CAAV")) {
     FNdArrayView *t = TN(StringA); t->ClearData();
     FNdArrayView *Ta = TN(StringB); 
     FNdArrayView *S1 = TN("S1");

     int ncore = m_Domains['c'].nSize, nact = m_Domains['a'].nSize, nvirt = m_Domains['e'].nSize;

     int fullrange = ncore*nact*nact*nvirt;
     for (int p=0; p<nact; p++)
       for (int q=0; q<nact; q++)
	 for (int r=0; r<nact; r++)
	   for (int s=0; s<nact; s++) 
	     for (int a=0; a<nvirt; a++)
	       for (int i=0; i<ncore; i++)
		 {
		   t->pData[i+p*ncore+q*ncore*nact+a*ncore*nact*nact] += Ta->pData[i+r*ncore+s*ncore*nact+a*ncore*nact*nact]*S1->operator()(p+q*nact,r+s*nact) 
		                                            +Ta->pData[i+r*ncore+s*ncore*nact+a*ncore*nact*nact + fullrange]* S1->operator()(p+q*nact,r+s*nact+nact*nact);
		   t->pData[i+p*ncore+q*ncore*nact+a*ncore*nact*nact+fullrange] += Ta->pData[i+r*ncore+s*ncore*nact+a*ncore*nact*nact]*S1->operator()(p+q*nact+nact*nact,r+s*nact)
                                                            +Ta->pData[i+r*ncore+s*ncore*nact+a*ncore*nact*nact + fullrange]* S1->operator()(p+q*nact+nact*nact,r+s*nact+nact*nact);
		   //std::cout<<t->pData[i+r*ncore+s*ncore*nact+a*ncore*nact*nact]<<"  "<<S1->operator()(r,s,p,q)<<"  "<<Ta->pData[i+p*ncore+q*ncore*nact+a*ncore*nact*nact]<<"  "<<i+p*ncore+q*ncore*nact+a*ncore*nact*nact<<"  "<<Ta->nValues()<<std::endl;
		 }


     //std::cout << "Norm "<<S1->Strides[0]<<" "<< ct::Dot(S1->pData, S1->pData, S1->nValues())<<std::endl;
     //exit(0);
   }

}
  
