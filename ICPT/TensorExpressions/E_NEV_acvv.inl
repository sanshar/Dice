/*                                                                                                                                                                          
Developed by Sandeep Sharma and Gerald Knizia, 2016                                                                                                                     
Copyright (c) 2016, Sandeep Sharma
*/
namespace NEVPT2_ACVV {

	FTensorDecl TensorDecls[29] = {
		/*  0*/{"t", "eeca", "",USAGE_Amplitude, STORAGE_Memory},
		/*  1*/{"R", "eeca", "",USAGE_Residual, STORAGE_Memory},
		/*  2*/{"f", "cc", "",USAGE_Hamiltonian, STORAGE_Memory},
		/*  3*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/*  4*/{"f", "ee", "",USAGE_Hamiltonian, STORAGE_Memory},
		/*  5*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory}, //dummy
		/*  6*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory}, //dummy
		/*  7*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory}, //dummy
		/*  8*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory}, //dummy
		/*  9*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory}, //dummy
		/* 10*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 11*/{"f", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 12*/{"W", "aaaa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 13*/{"E1", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 14*/{"E2", "aaaa", "",USAGE_Density, STORAGE_Memory},
		/* 15*/{"E1", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 16*/{"S1", "AA", "",USAGE_Density, STORAGE_Memory},
		/* 17*/{"S2", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 18*/{"T", "eeca", "",USAGE_Amplitude, STORAGE_Memory},
		/* 19*/{"b", "eeca", "",USAGE_Amplitude, STORAGE_Memory},
		/* 20*/{"p", "eeca", "",USAGE_Amplitude, STORAGE_Memory},
		/* 21*/{"Ap", "eeca", "",USAGE_Amplitude, STORAGE_Memory},
		/* 22*/{"P", "eeca", "",USAGE_Amplitude, STORAGE_Memory},
		/* 23*/{"AP", "eeca", "",USAGE_Amplitude, STORAGE_Memory},
		/* 24*/{"B", "eeca", "",USAGE_Amplitude, STORAGE_Memory},
		/* 25*/{"W", "eeca", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 26*/{"delta", "cc", "",USAGE_Density, STORAGE_Memory},
		/* 27*/{"delta", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 28*/{"delta", "ee", "",USAGE_Density, STORAGE_Memory},
	};

//Number of terms :  10
	FEqInfo EqsRes[10] = {

		{"CDJR,IJ,RQ,CDIQ", -2.0  , 4, {21,2,13,20}},		//Ap[CDJR] += -2.0 f[IJ] E1[RQ] p[CDIQ]
		{"CDJR,IJ,RQ,DCIQ", 1.0  , 4, {21,2,13,20}},		//Ap[CDJR] += 1.0 f[IJ] E1[RQ] p[DCIQ]
		{"CDJR,AC,RQ,ADJQ", 2.0  , 4, {21,4,13,20}},		//Ap[CDJR] += 2.0 f[AC] E1[RQ] p[ADJQ]
		{"CDJR,AD,RQ,ACJQ", -1.0  , 4, {21,4,13,20}},		//Ap[CDJR] += -1.0 f[AD] E1[RQ] p[ACJQ]
		{"CDJR,BC,RQ,DBJQ", -1.0  , 4, {21,4,13,20}},		//Ap[CDJR] += -1.0 f[BC] E1[RQ] p[DBJQ]
		{"CDJR,BD,RQ,CBJQ", 2.0  , 4, {21,4,13,20}},		//Ap[CDJR] += 2.0 f[BD] E1[RQ] p[CBJQ]
		{"CDJR,Qa,Ra,CDJQ", -2.0  , 4, {21,3,13,20}},		//Ap[CDJR] += -2.0 f[Qa] E1[Ra] p[CDJQ]
		{"CDJR,Qa,Ra,DCJQ", 1.0  , 4, {21,3,13,20}},		//Ap[CDJR] += 1.0 f[Qa] E1[Ra] p[DCJQ]
		{"CDJR,Qabc,Rabc,CDJQ", -2.0  , 4, {21,12,14,20}},		//Ap[CDJR] += -2.0 W[Qabc] E2[Rabc] p[CDJQ]
		{"CDJR,Qabc,Rabc,DCJQ", 1.0  , 4, {21,12,14,20}},		//Ap[CDJR] += 1.0 W[Qabc] E2[Rabc] p[DCJQ]

	};
	int f(int i) {
		return 2*i;
	}
	FDomainDecl DomainDecls[1] = {
		{"A", "a", f}
	};
	FEqInfo Overlap[2] = {
		{"ABIP,RP,ABIR", 2.0, 3, {19, 13, 25}},
		{"ABIP,RP,BAIR",-1.0, 3, {19, 13, 25}}
	};
	static void GetMethodInfo(FMethodInfo &Out) {
		Out = FMethodInfo();
		Out.pName = "NEVPT2_ACVV";
		Out.perturberClass = "ACVV";
		Out.pSpinClass = "restricted";
		Out.pTensorDecls = &TensorDecls[0];
		Out.nTensorDecls = 29;
		Out.pDomainDecls = &DomainDecls[0];
		Out.nDomainDecls = 1;
		Out.EqsRes = FEqSet(&EqsRes[0], 10, "NEVPT2_ACVV/Res");
		Out.Overlap = FEqSet(&Overlap[0], 2, "NEVPT2_ACVV/Overlap");
	};
};
