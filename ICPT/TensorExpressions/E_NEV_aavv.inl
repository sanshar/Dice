/*                                                                                                                                                                          
Developed by Sandeep Sharma and Gerald Knizia, 2016                                                                                                                     
Copyright (c) 2016, Sandeep Sharma
*/
namespace NEVPT2_AAVV {

	FTensorDecl TensorDecls[29] = {
		/*  0*/{"b", "eeaa", "",USAGE_Amplitude, STORAGE_Memory},
		/*  1*/{"B", "eeaa", "",USAGE_Residual, STORAGE_Memory},
		/*  2*/{"t", "eeaa", "",USAGE_Amplitude, STORAGE_Memory},
		/*  3*/{"T", "eeaa", "",USAGE_Amplitude, STORAGE_Memory},
		/*  4*/{"p", "eeaa", "",USAGE_Amplitude, STORAGE_Memory},
		/*  5*/{"P", "eeaa", "",USAGE_Amplitude, STORAGE_Memory},
		/*  6*/{"Ap", "eeaa", "",USAGE_Amplitude, STORAGE_Memory},
		/*  7*/{"AP", "eeaa", "",USAGE_Amplitude, STORAGE_Memory},
		/*  8*/{"R", "eeaa", "",USAGE_Amplitude, STORAGE_Memory},
		/*  9*/{"f", "cc", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 10*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 11*/{"f", "ee", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 12*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 13*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 14*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 15*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 16*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 17*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 18*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 19*/{"W", "aaaa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 20*/{"E1", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 21*/{"E2", "aaaa", "",USAGE_Density, STORAGE_Memory},
		/* 22*/{"S3", "a", "",USAGE_Density, STORAGE_Memory},
		/* 23*/{"S1", "aAaA", "",USAGE_Density, STORAGE_Memory},
		/* 24*/{"S2", "aaaa", "",USAGE_Density, STORAGE_Memory},
		/* 25*/{"W", "eeaa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 26*/{"delta", "cc", "",USAGE_Density, STORAGE_Memory},
		/* 27*/{"delta", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 28*/{"delta", "ee", "",USAGE_Density, STORAGE_Memory},
	};

//Number of terms :  4
	FEqInfo EqsRes[3] = {

		{"abcd,ef,abeg,cdfg", -4.0  , 4, {6,10,4,21}},		//-4.0 Ap[abcd] f[ef] p[abeg] E2[cdfg] 
		{"abcd,ae,befg,cdgf", 4.0  , 4, {6,11,4,21}},		//4.0 Ap[abcd] f[ae] p[befg] E2[cdgf] 
		{"abcd,efgh,abef,cdgh", -2.0  , 4, {6,19,4,21}},		//-2.0 Ap[abcd] W[efgh] p[abef] E2[cdgh] 
		//{"abcd,efgh,abei,cdfgih", -4.0  , 4, {6,19,4,22}},		//-4.0 Ap[abcd] W[efgh] p[abei] E3[cdfgih] 

	};
	int f(int i) {
		return 2*i;
	}
	FDomainDecl DomainDecls[1] = {
		{"A", "a", f}
	};
	FEqInfo Overlap[2] = {
		{"ABRS,ABPQ,RSPQ", 0.5, 3, {0, 25, 21}},
		{"BARS,ABPQ,RSQP", 0.5, 3, {0, 25, 21}}
	};
	static void GetMethodInfo(FMethodInfo &Out) {
		Out = FMethodInfo();
		Out.pName = "NEVPT2_AAVV";
		Out.perturberClass = "AAVV";
		Out.pSpinClass = "restricted";
		Out.pTensorDecls = &TensorDecls[0];
		Out.nTensorDecls = 29;
		Out.pDomainDecls = &DomainDecls[0];
		Out.nDomainDecls = 1;
		Out.EqsRes = FEqSet(&EqsRes[0], 3, "NEVPT2_AAVV/Res");
		Out.Overlap = FEqSet(&Overlap[0], 2, "NEVPT2_AAVV/Overlap");
	};
};
