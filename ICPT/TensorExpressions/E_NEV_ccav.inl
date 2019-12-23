/*                                                                                                                                                                          
Developed by Sandeep Sharma and Gerald Knizia, 2016                                                                                                                     
Copyright (c) 2016, Sandeep Sharma
*/
namespace NEVPT2_CCAV {

	FTensorDecl TensorDecls[30] = {
		/*  0*/{"t", "ccae", "",USAGE_Amplitude, STORAGE_Memory},
		/*  1*/{"R", "ccae", "",USAGE_Residual, STORAGE_Memory},
		/*  2*/{"f", "cc", "",USAGE_Hamiltonian, STORAGE_Memory},
		/*  3*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/*  4*/{"f", "ee", "",USAGE_Hamiltonian, STORAGE_Memory},
		/*  5*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/*  6*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/*  7*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/*  8*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/*  9*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 10*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 11*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 12*/{"W", "aaaa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 13*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 14*/{"E1", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 15*/{"E2", "aaaa", "",USAGE_Density, STORAGE_Memory},
		/* 16*/{"f", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 17*/{"S1", "AA", "",USAGE_Density, STORAGE_Memory},
		/* 18*/{"S2", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 19*/{"T", "ccae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 20*/{"b", "ccae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 21*/{"p", "ccae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 22*/{"Ap", "ccae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 23*/{"P", "ccae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 24*/{"AP", "ccae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 25*/{"B", "ccae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 26*/{"W", "ccae", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 27*/{"delta", "cc", "",USAGE_Density, STORAGE_Memory},
		/* 28*/{"delta", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 29*/{"delta", "ee", "",USAGE_Density, STORAGE_Memory},
	};

//Number of terms :  22
	FEqInfo EqsRes[22] = {

		{"KLQB,PQ,KLPB", 4.0  , 3, {22,3,21}},		//Ap[KLQB] += 4.0 f[PQ] p[KLPB]
		{"KLQB,PQ,LKPB", -2.0  , 3, {22,3,21}},		//Ap[KLQB] += -2.0 f[PQ] p[LKPB]
		{"KLQB,IK,ILQB", -4.0  , 3, {22,2,21}},		//Ap[KLQB] += -4.0 f[IK] p[ILQB]
		{"KLQB,IL,IKQB", 2.0  , 3, {22,2,21}},		//Ap[KLQB] += 2.0 f[IL] p[IKQB]
		{"KLQB,JK,LJQB", 2.0  , 3, {22,2,21}},		//Ap[KLQB] += 2.0 f[JK] p[LJQB]
		{"KLQB,JL,KJQB", -4.0  , 3, {22,2,21}},		//Ap[KLQB] += -4.0 f[JL] p[KJQB]
		{"KLQB,AB,KLQA", 4.0  , 3, {22,4,21}},		//Ap[KLQB] += 4.0 f[AB] p[KLQA]
		{"KLQB,AB,LKQA", -2.0  , 3, {22,4,21}},		//Ap[KLQB] += -2.0 f[AB] p[LKQA]
		{"KLQB,IK,PQ,ILPB", 2.0  , 4, {22,2,14,21}},		//Ap[KLQB] += 2.0 f[IK] E1[PQ] p[ILPB]
		{"KLQB,IL,PQ,IKPB", -1.0  , 4, {22,2,14,21}},		//Ap[KLQB] += -1.0 f[IL] E1[PQ] p[IKPB]
		{"KLQB,JK,PQ,LJPB", -1.0  , 4, {22,2,14,21}},		//Ap[KLQB] += -1.0 f[JK] E1[PQ] p[LJPB]
		{"KLQB,JL,PQ,KJPB", 2.0  , 4, {22,2,14,21}},		//Ap[KLQB] += 2.0 f[JL] E1[PQ] p[KJPB]
		{"KLQB,AB,PQ,KLPA", -2.0  , 4, {22,4,14,21}},		//Ap[KLQB] += -2.0 f[AB] E1[PQ] p[KLPA]
		{"KLQB,AB,PQ,LKPA", 1.0  , 4, {22,4,14,21}},		//Ap[KLQB] += 1.0 f[AB] E1[PQ] p[LKPA]
		{"KLQB,Pa,aQ,KLPB", -2.0  , 4, {22,3,14,21}},		//Ap[KLQB] += -2.0 f[Pa] E1[aQ] p[KLPB]
		{"KLQB,Pa,aQ,LKPB", 1.0  , 4, {22,3,14,21}},		//Ap[KLQB] += 1.0 f[Pa] E1[aQ] p[LKPB]
		{"KLQB,PQab,ab,KLPB", -2.0  , 4, {22,12,14,21}},		//Ap[KLQB] += -2.0 W[PQab] E1[ab] p[KLPB]
		{"KLQB,PQab,ab,LKPB", 1.0  , 4, {22,12,14,21}},		//Ap[KLQB] += 1.0 W[PQab] E1[ab] p[LKPB]
		{"KLQB,PaQb,ab,KLPB", 4.0  , 4, {22,12,14,21}},		//Ap[KLQB] += 4.0 W[PaQb] E1[ab] p[KLPB]
		{"KLQB,PaQb,ab,LKPB", -2.0  , 4, {22,12,14,21}},		//Ap[KLQB] += -2.0 W[PaQb] E1[ab] p[LKPB]
		{"KLQB,Pabc,abcQ,KLPB", -2.0  , 4, {22,12,15,21}},		//Ap[KLQB] += -2.0 W[Pabc] E2[abcQ] p[KLPB]
		{"KLQB,Pabc,abcQ,LKPB", 1.0  , 4, {22,12,15,21}},		//Ap[KLQB] += 1.0 W[Pabc] E2[abcQ] p[LKPB]

	};
	int f(int i) {
		return 2*i;
	}
	FDomainDecl DomainDecls[1] = {
		{"A", "a", f}
	};
	FEqInfo Overlap[4] = {
		{"IJPA,RP,IJRA", 4.0, 3, {20, 28, 26}},
		{"IJPA,RP,JIRA",-2.0, 3, {20, 28, 26}},
		{"IJPA,RP,IJRA",-2.0, 3, {20, 14, 26}},
		{"IJPA,RP,JIRA", 1.0, 3, {20, 14, 26}}
	};
	static void GetMethodInfo(FMethodInfo &Out) {
		Out = FMethodInfo();
		Out.pName = "NEVPT2_CCAV";
		Out.perturberClass = "CCAV";
		Out.pSpinClass = "restricted";
		Out.pTensorDecls = &TensorDecls[0];
		Out.nTensorDecls = 30;
		Out.pDomainDecls = &DomainDecls[0];
		Out.nDomainDecls = 1;
		Out.EqsRes = FEqSet(&EqsRes[0], 22, "NEVPT2_CCAV/Res");
		Out.Overlap = FEqSet(&Overlap[0], 4, "NEVPT2_CCAV/Overlap");
	};
};
