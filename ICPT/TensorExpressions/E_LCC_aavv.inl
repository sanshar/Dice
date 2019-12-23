/*                                                                                                                                                                          
Developed by Sandeep Sharma and Gerald Knizia, 2016                                                                                                                     
Copyright (c) 2016, Sandeep Sharma
*/
namespace MRLCC_AAVV {

	FTensorDecl TensorDecls[31] = {
		/*  0*/{"b", "eeaa", "",USAGE_Amplitude, STORAGE_Memory},
		/*  1*/{"B", "eeaa", "",USAGE_Residual, STORAGE_Memory},
		/*  2*/{"t", "eeaa", "",USAGE_Amplitude, STORAGE_Memory},
		/*  3*/{"T", "eeaa", "",USAGE_Amplitude, STORAGE_Memory},
		/*  4*/{"p", "eeaa", "",USAGE_Amplitude, STORAGE_Memory},
		/*  5*/{"P", "eeaa", "",USAGE_Amplitude, STORAGE_Memory},
		/*  6*/{"Ap", "eeaa", "",USAGE_Amplitude, STORAGE_Memory},
		/*  7*/{"AP", "eeaa", "",USAGE_Amplitude, STORAGE_Memory},
		/*  8*/{"R", "eeaa", "",USAGE_Amplitude, STORAGE_Memory},
		/*  9*/{"k", "cc", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 10*/{"k", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 11*/{"k", "ee", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 12*/{"W", "caca", "",USAGE_Hamiltonian, STORAGE_Disk},
		/* 13*/{"W", "caac", "",USAGE_Hamiltonian, STORAGE_Disk},
		/* 14*/{"W", "cece", "",USAGE_Hamiltonian, STORAGE_Disk},
		/* 15*/{"W", "ceec", "",USAGE_Hamiltonian, STORAGE_Disk},
		/* 16*/{"W", "aeae", "",USAGE_Hamiltonian, STORAGE_Disk},
		/* 17*/{"W", "aeea", "",USAGE_Hamiltonian, STORAGE_Disk},
		/* 18*/{"W", "cccc", "",USAGE_Hamiltonian, STORAGE_Disk},
		/* 19*/{"W", "aaaa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 20*/{"Inter", "e", "",USAGE_Intermediate, STORAGE_Memory},
		/* 21*/{"E1", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 22*/{"E2", "aaaa", "",USAGE_Density, STORAGE_Memory},
		/* 23*/{"S3", "a", "",USAGE_Density, STORAGE_Memory},
		/* 24*/{"S1", "aAaA", "",USAGE_Density, STORAGE_Memory},
		/* 25*/{"S2", "aaaa", "",USAGE_Density, STORAGE_Memory},
		/* 26*/{"W", "eeaa", "",USAGE_Hamiltonian, STORAGE_Disk},
		/* 27*/{"delta", "cc", "",USAGE_Density, STORAGE_Memory},
		/* 28*/{"delta", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 29*/{"delta", "ee", "",USAGE_Density, STORAGE_Memory},
		/* 30*/{"Inter", "ee", "", USAGE_Intermediate, STORAGE_Memory},
	};

//Number of terms :  11
	FEqInfo EqsRes[8] = {

		{"abcd,ef,abeg,cdfg", -4.0  , 4, {6,10,4,22}},		//-4.0 Ap[abcd] k[ef] p[abeg] E2[cdfg] 
		{"abcd,ae,befg,cdgf", 4.0  , 4, {6,11,4,22}},		//4.0 Ap[abcd] k[ae] p[befg] E2[cdgf] 
		{"abcd,efgh,abef,cdgh", -2.0  , 4, {6,19,4,22}},		//-2.0 Ap[abcd] W[efgh] p[abef] E2[cdgh] 
		{"abcd,efig,abfh,cdgh,ie", -8.0  , 5, {6,12,4,22,27}},		//-8.0 Ap[abcd] W[efig] p[abfh] E2[cdgh] delta[ie] 
		{"abcd,efgi,abfh,cdgh,ie", 4.0  , 5, {6,13,4,22,27}},		//4.0 Ap[abcd] W[efgi] p[abfh] E2[cdgh] delta[ie] 

		//the next commented two are replaced by the following 3
		//{"abcd,eaif,bfgh,cdhg,ie", 8.0  , 5, {6,14,4,22,27}},		//8.0 Ap[abcd] W[eaif] p[bfgh] E2[cdhg] delta[ie] 
		//{"abcd,eafi,bfgh,cdhg,ie", -4.0  , 5, {6,15,4,22,27}},		//-4.0 Ap[abcd] W[eafi] p[bfgh] E2[cdhg] delta[ie] 
		{"af,eaif,ie", 8.0, 3, {30,14,27}},
		{"af,eafi,ie", -4.0, 3, {30,15,27}},
		{"abcd,af,bfgh,cdhg", 1.0  , 4, {6,30,4,22}},		
		//{"abcd,abef,efgh,cdgh", 2.0  , 4, {6,20,4,22}},		//2.0 Ap[abcd] W[abef] p[efgh] E2[cdgh] 
		//{"abcd,efgh,abei,higfdc", -4.0  , 4, {6,19,4,23}},		//-4.0 Ap[abcd] W[efgh] p[abei] E3[cdfgih] 
		//{"abcd,eafg,bfhi,ihgedc", 4.0  , 4, {6,17,4,23}},		//4.0 Ap[abcd] W[eafg] p[bfhi] E3[cdeghi] 
		//{"abcd,eafg,bghi,fhiedc", 4.0  , 4, {6,16,4,23}},		//4.0 Ap[abcd] W[eafg] p[bghi] E3[cdeihf] 
	};
	int f(int i) {
		return 2*i;
	}
	FDomainDecl DomainDecls[1] = {
		{"A", "a", f}
	};
	FEqInfo Overlap[2] = {
		{"ABRS,ABPQ,RSPQ", 0.5, 3, {0, 26, 22}},
		{"BARS,ABPQ,RSQP", 0.5, 3, {0, 26, 22}}
	};
	static void GetMethodInfo(FMethodInfo &Out) {
		Out = FMethodInfo();
		Out.pName = "MRLCC_AAVV";
		Out.perturberClass = "AAVV";
		Out.pSpinClass = "restricted";
		Out.pTensorDecls = &TensorDecls[0];
		Out.nTensorDecls = 31;
		Out.pDomainDecls = &DomainDecls[0];
		Out.nDomainDecls = 1;
		Out.EqsRes = FEqSet(&EqsRes[0], 8, "MRLCC_AAVV/Res");
		Out.Overlap = FEqSet(&Overlap[0], 2, "MRLCC_AAVV/Overlap");
	};
};
