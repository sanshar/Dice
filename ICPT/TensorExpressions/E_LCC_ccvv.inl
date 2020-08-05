/*                                                                                                                                                                          
Developed by Sandeep Sharma and Gerald Knizia, 2016                                                                                                                     
Copyright (c) 2016, Sandeep Sharma
*/
namespace MRLCC_CCVV {

	FTensorDecl TensorDecls[31] = {
		/*  0*/{"t", "eecc", "",USAGE_Amplitude, STORAGE_Memory},
		/*  1*/{"R", "eecc", "",USAGE_Residual, STORAGE_Memory},
		/*  2*/{"k", "cc", "",USAGE_Hamiltonian, STORAGE_Memory},
		/*  3*/{"k", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/*  4*/{"k", "ee", "",USAGE_Hamiltonian, STORAGE_Memory},
		/*  5*/{"W", "caca", "",USAGE_Hamiltonian, STORAGE_Disk},
		/*  6*/{"W", "caac", "",USAGE_Hamiltonian, STORAGE_Disk},
		/*  7*/{"W", "cece", "",USAGE_Hamiltonian, STORAGE_Disk},
		/*  8*/{"W", "ceec", "",USAGE_Hamiltonian, STORAGE_Disk},
		/*  9*/{"W", "aeae", "",USAGE_Hamiltonian, STORAGE_Disk},
		/* 10*/{"W", "aeea", "",USAGE_Hamiltonian, STORAGE_Disk},
		/* 11*/{"W", "cccc", "",USAGE_Hamiltonian, STORAGE_Disk},
		/* 12*/{"W", "aaaa", "",USAGE_Hamiltonian, STORAGE_Disk},
		/* 13*/{"W", "e", "",USAGE_Intermediate, STORAGE_Memory},
		/* 14*/{"E1", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 15*/{"E2", "aaaa", "",USAGE_Density, STORAGE_Memory},
		/* 16*/{"E3", "a", "",USAGE_Intermediate, STORAGE_Memory},
		/* 17*/{"S1", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 18*/{"S2", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 19*/{"T", "eecc", "",USAGE_Amplitude, STORAGE_Memory},
		/* 20*/{"b", "eecc", "",USAGE_Amplitude, STORAGE_Memory},
		/* 21*/{"p", "eecc", "",USAGE_Amplitude, STORAGE_Memory},
		/* 22*/{"Ap", "eecc", "",USAGE_Amplitude, STORAGE_Memory},
		/* 23*/{"P", "eecc", "",USAGE_Amplitude, STORAGE_Memory},
		/* 24*/{"AP", "eecc", "",USAGE_Amplitude, STORAGE_Memory},
		/* 25*/{"B", "eecc", "",USAGE_Amplitude, STORAGE_Memory},
		/* 26*/{"W", "eecc", "",USAGE_Hamiltonian, STORAGE_Disk},
		/* 27*/{"delta", "cc", "",USAGE_Density, STORAGE_Memory},
		/* 28*/{"delta", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 29*/{"delta", "ee", "",USAGE_Density, STORAGE_Memory},
		/* 30*/{"t1", "eecc", "",USAGE_Amplitude, STORAGE_Memory},
	};

//Number of terms :  32
	FEqInfo EqsRes[30] = {

		{"abcd,ce,abde", 8.0  , 3, {22,2,21}},		//8.0 Ap[abcd] k[ce] p[abde] 
		{"abcd,ce,abed", -16.0  , 3, {22,2,21}},		//-16.0 Ap[abcd] k[ce] p[abed] 
		{"abcd,ae,becd", -8.0  , 3, {22,4,21}},		//-8.0 Ap[abcd] k[ae] p[becd] 
		{"abcd,ae,bedc", 16.0  , 3, {22,4,21}},		//16.0 Ap[abcd] k[ae] p[bedc] 
		{"abcd,cdef,abef", 8.0  , 3, {22,11,21}},		//8.0 Ap[abcd] W[cdef] p[abef] 
		{"abcd,cdef,abfe", -4.0  , 3, {22,11,21}},		//-4.0 Ap[abcd] W[cdef] p[abfe] 
		{"abcd,cegf,abdf,ge", -8.0  , 4, {22,11,21,27}},		//-8.0 Ap[abcd] W[cegf] p[abdf] delta[ge] 
		{"abcd,cegf,abfd,ge", 16.0  , 4, {22,11,21,27}},		//16.0 Ap[abcd] W[cegf] p[abfd] delta[ge] 
		{"abcd,cefg,abdf,ge", 16.0  , 4, {22,11,21,27}},		//16.0 Ap[abcd] W[cefg] p[abdf] delta[ge] 
		{"abcd,cefg,abfd,ge", -32.0  , 4, {22,11,21,27}},		//-32.0 Ap[abcd] W[cefg] p[abfd] delta[ge] 
		{"abcd,caef,bfde", -16.0  , 3, {22,7,21}},		//-16.0 Ap[abcd] W[caef] p[bfde] 
		{"abcd,caef,bfed", 8.0  , 3, {22,7,21}},		//8.0 Ap[abcd] W[caef] p[bfed] 
		{"abcd,cbef,afde", 8.0  , 3, {22,7,21}},		//8.0 Ap[abcd] W[cbef] p[afde] 
		{"abcd,cbef,afed", -16.0  , 3, {22,7,21}},		//-16.0 Ap[abcd] W[cbef] p[afed] 
		{"abcd,eagf,bfcd,ge", -16.0  , 4, {22,7,21,27}},		//-16.0 Ap[abcd] W[eagf] p[bfcd] delta[ge] 
		{"abcd,eagf,bfdc,ge", 32.0  , 4, {22,7,21,27}},		//32.0 Ap[abcd] W[eagf] p[bfdc] delta[ge] 
		{"abcd,eafc,bfde", 32.0  , 3, {22,8,21}},		//32.0 Ap[abcd] W[eafc] p[bfde] 
		{"abcd,eafc,bfed", -16.0  , 3, {22,8,21}},		//-16.0 Ap[abcd] W[eafc] p[bfed] 
		{"abcd,eafd,bfce", -16.0  , 3, {22,8,21}},		//-16.0 Ap[abcd] W[eafd] p[bfce] 
		{"abcd,eafd,bfec", 8.0  , 3, {22,8,21}},		//8.0 Ap[abcd] W[eafd] p[bfec] 
		{"abcd,eafg,bfcd,ge", 8.0  , 4, {22,8,21,27}},		//8.0 Ap[abcd] W[eafg] p[bfcd] delta[ge] 
		{"abcd,eafg,bfdc,ge", -16.0  , 4, {22,8,21,27}},		//-16.0 Ap[abcd] W[eafg] p[bfdc] delta[ge] 
		//{"abcd,abef,efcd", 8.0  , 3, {22,13,21}},		//8.0 Ap[abcd] W[abef] p[efcd] 
		//{"abcd,abef,efdc", -4.0  , 3, {22,13,21}},		//-4.0 Ap[abcd] W[abef] p[efdc] 
		{"abcd,eafg,bgcd,ef", -8.0  , 4, {22,9,21,14}},		//-8.0 Ap[abcd] W[eafg] p[bgcd] E1[ef] 
		{"abcd,eafg,bgdc,ef", 16.0  , 4, {22,9,21,14}},		//16.0 Ap[abcd] W[eafg] p[bgdc] E1[ef] 
		{"abcd,eafg,bfcd,eg", 4.0  , 4, {22,10,21,14}},		//4.0 Ap[abcd] W[eafg] p[bfcd] E1[eg] 
		{"abcd,eafg,bfdc,eg", -8.0  , 4, {22,10,21,14}},		//-8.0 Ap[abcd] W[eafg] p[bfdc] E1[eg] 
		{"abcd,cefg,abdf,eg", 8.0  , 4, {22,5,21,14}},		//8.0 Ap[abcd] W[cefg] p[abdf] E1[eg] 
		{"abcd,cefg,abfd,eg", -16.0  , 4, {22,5,21,14}},		//-16.0 Ap[abcd] W[cefg] p[abfd] E1[eg] 
		{"abcd,efgc,abde,fg", -4.0  , 4, {22,6,21,14}},		//-4.0 Ap[abcd] W[efgc] p[abde] E1[fg] 
		{"abcd,efgc,abed,fg", 8.0  , 4, {22,6,21,14}},		//8.0 Ap[abcd] W[efgc] p[abed] E1[fg] 

	};
	int f(int i) {
		return 2*i;
	}
	FDomainDecl DomainDecls[1] = {
		{"A", "a", f}
	};
	FEqInfo Overlap[4] = {
		{"CDKL,LM,CDKM", 2.0, 3, {20, 27, 26}},
		{"CDKL,LM,DCKM",-1.0, 3, {20, 27, 26}},
		{"CDKL,LM,CDMK",-1.0, 3, {20, 27, 26}},
		{"CDKL,LM,DCMK", 2.0, 3, {20, 27, 26}},
	};
	static void GetMethodInfo(FMethodInfo &Out) {
		Out = FMethodInfo();
		Out.pName = "MRLCC_CCVV";
		Out.perturberClass = "CCVV";
		Out.pSpinClass = "restricted";
		Out.pTensorDecls = &TensorDecls[0];
		Out.nTensorDecls = 31;
		Out.pDomainDecls = &DomainDecls[0];
		Out.nDomainDecls = 0;
		Out.EqsRes = FEqSet(&EqsRes[0], 30, "MRLCC_CCVV/Res");
		Out.Overlap = FEqSet(&Overlap[0], 4, "MRLCC_CCVV/Overlap");
	};
};
