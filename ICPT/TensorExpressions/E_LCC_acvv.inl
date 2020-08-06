/*                                                                                                                                                                          
Developed by Sandeep Sharma and Gerald Knizia, 2016                                                                                                                     
Copyright (c) 2016, Sandeep Sharma
*/
namespace MRLCC_ACVV {

	FTensorDecl TensorDecls[31] = {
		/*  0*/{"t", "eeca", "",USAGE_Amplitude, STORAGE_Memory},
		/*  1*/{"R", "eeca", "",USAGE_Residual, STORAGE_Memory},
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
		/* 13*/{"k", "e", "",USAGE_Intermediate, STORAGE_Memory},
		/* 14*/{"E1", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 15*/{"E2", "aaaa", "",USAGE_Density, STORAGE_Memory},
		/* 16*/{"S3", "a", "",USAGE_Density, STORAGE_Memory},
		/* 17*/{"S1", "AA", "",USAGE_Density, STORAGE_Memory},
		/* 18*/{"S2", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 19*/{"T", "eeca", "",USAGE_Amplitude, STORAGE_Memory},
		/* 20*/{"b", "eeca", "",USAGE_Amplitude, STORAGE_Memory},
		/* 21*/{"p", "eeca", "",USAGE_Amplitude, STORAGE_Memory},
		/* 22*/{"Ap", "eeca", "",USAGE_Amplitude, STORAGE_Memory},
		/* 23*/{"P", "eeca", "",USAGE_Amplitude, STORAGE_Memory},
		/* 24*/{"AP", "eeca", "",USAGE_Amplitude, STORAGE_Memory},
		/* 25*/{"B", "eeca", "",USAGE_Amplitude, STORAGE_Memory},
		/* 26*/{"W", "eeca", "",USAGE_Hamiltonian, STORAGE_Disk},
		/* 27*/{"delta", "cc", "",USAGE_Density, STORAGE_Memory},
		/* 28*/{"delta", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 29*/{"delta", "ee", "",USAGE_Density, STORAGE_Memory},
		/* 30*/{"Inter", "eeca", "",USAGE_Intermediate, STORAGE_Memory},
	};

//Number of terms :  48
	FEqInfo EqsRes[46] = {

		{"CDJR,IAJC,RQ,ADIQ", -2.0  , 4, {22,7,14,21}},		//Ap[CDJR] += -2.0 W[IAJC] E1[RQ] p[ADIQ]
		{"CDJR,IAJD,RQ,ACIQ", 1.0  , 4, {22,7,14,21}},		//Ap[CDJR] += 1.0 W[IAJD] E1[RQ] p[ACIQ]
		{"CDJR,IBJC,RQ,DBIQ", 1.0  , 4, {22,7,14,21}},		//Ap[CDJR] += 1.0 W[IBJC] E1[RQ] p[DBIQ]
		{"CDJR,IBJD,RQ,CBIQ", -2.0  , 4, {22,7,14,21}},		//Ap[CDJR] += -2.0 W[IBJD] E1[RQ] p[CBIQ]
		{"CDJR,ICAJ,RQ,ADIQ", 4.0  , 4, {22,8,14,21}},		//Ap[CDJR] += 4.0 W[ICAJ] E1[RQ] p[ADIQ]
		{"CDJR,ICBJ,RQ,DBIQ", -2.0  , 4, {22,8,14,21}},		//Ap[CDJR] += -2.0 W[ICBJ] E1[RQ] p[DBIQ]
		{"CDJR,IDAJ,RQ,ACIQ", -2.0  , 4, {22,8,14,21}},		//Ap[CDJR] += -2.0 W[IDAJ] E1[RQ] p[ACIQ]
		{"CDJR,IDBJ,RQ,CBIQ", 1.0  , 4, {22,8,14,21}},		//Ap[CDJR] += 1.0 W[IDBJ] E1[RQ] p[CBIQ]
		//{"CDJR,ABCD,RQ,ABJQ", 2.0  , 4, {22,13,14,21}},		//Ap[CDJR] += 2.0 W[ABCD] E1[RQ] p[ABJQ]
		//{"CDJR,ABDC,RQ,ABJQ", -1.0  , 4, {22,13,14,21}},		//Ap[CDJR] += -1.0 W[ABDC] E1[RQ] p[ABJQ]
		{"CDJR,IJ,RQ,CDIQ", -2.0  , 4, {22,2,14,21}},		//Ap[CDJR] += -2.0 k[IJ] E1[RQ] p[CDIQ]
		{"CDJR,IJ,RQ,DCIQ", 1.0  , 4, {22,2,14,21}},		//Ap[CDJR] += 1.0 k[IJ] E1[RQ] p[DCIQ]
		{"CDJR,AC,RQ,ADJQ", 2.0  , 4, {22,4,14,21}},		//Ap[CDJR] += 2.0 k[AC] E1[RQ] p[ADJQ]
		{"CDJR,AD,RQ,ACJQ", -1.0  , 4, {22,4,14,21}},		//Ap[CDJR] += -1.0 k[AD] E1[RQ] p[ACJQ]
		{"CDJR,BC,RQ,DBJQ", -1.0  , 4, {22,4,14,21}},		//Ap[CDJR] += -1.0 k[BC] E1[RQ] p[DBJQ]
		{"CDJR,BD,RQ,CBJQ", 2.0  , 4, {22,4,14,21}},		//Ap[CDJR] += 2.0 k[BD] E1[RQ] p[CBJQ]
		{"CDJR,IJab,RQ,ba,CDIQ", 2.0  , 5, {22,11,14,27,21}},		//Ap[CDJR] += 2.0 W[IJab] E1[RQ] delta[ba] p[CDIQ]
		{"CDJR,IJab,RQ,ba,DCIQ", -1.0  , 5, {22,11,14,27,21}},		//Ap[CDJR] += -1.0 W[IJab] E1[RQ] delta[ba] p[DCIQ]
		{"CDJR,IaJb,RQ,ba,CDIQ", -4.0  , 5, {22,11,14,27,21}},		//Ap[CDJR] += -4.0 W[IaJb] E1[RQ] delta[ba] p[CDIQ]
		{"CDJR,IaJb,RQ,ba,DCIQ", 2.0  , 5, {22,11,14,27,21}},		//Ap[CDJR] += 2.0 W[IaJb] E1[RQ] delta[ba] p[DCIQ]
		{"CDJR,aAbC,RQ,ba,ADJQ", 4.0  , 5, {22,7,14,27,21}},		//Ap[CDJR] += 4.0 W[aAbC] E1[RQ] delta[ba] p[ADJQ]
		{"CDJR,aAbD,RQ,ba,ACJQ", -2.0  , 5, {22,7,14,27,21}},		//Ap[CDJR] += -2.0 W[aAbD] E1[RQ] delta[ba] p[ACJQ]
		{"CDJR,aBbC,RQ,ba,DBJQ", -2.0  , 5, {22,7,14,27,21}},		//Ap[CDJR] += -2.0 W[aBbC] E1[RQ] delta[ba] p[DBJQ]
		{"CDJR,aBbD,RQ,ba,CBJQ", 4.0  , 5, {22,7,14,27,21}},		//Ap[CDJR] += 4.0 W[aBbD] E1[RQ] delta[ba] p[CBJQ]
		{"CDJR,aCAb,RQ,ba,ADJQ", -2.0  , 5, {22,8,14,27,21}},		//Ap[CDJR] += -2.0 W[aCAb] E1[RQ] delta[ba] p[ADJQ]
		{"CDJR,aCBb,RQ,ba,DBJQ", 1.0  , 5, {22,8,14,27,21}},		//Ap[CDJR] += 1.0 W[aCBb] E1[RQ] delta[ba] p[DBJQ]
		{"CDJR,aDAb,RQ,ba,ACJQ", 1.0  , 5, {22,8,14,27,21}},		//Ap[CDJR] += 1.0 W[aDAb] E1[RQ] delta[ba] p[ACJQ]
		{"CDJR,aDBb,RQ,ba,CBJQ", -2.0  , 5, {22,8,14,27,21}},		//Ap[CDJR] += -2.0 W[aDBb] E1[RQ] delta[ba] p[CBJQ]
		{"CDJR,Qa,Ra,CDJQ", -2.0  , 4, {22,3,14,21}},		//Ap[CDJR] += -2.0 k[Qa] E1[Ra] p[CDJQ]
		{"CDJR,Qa,Ra,DCJQ", 1.0  , 4, {22,3,14,21}},		//Ap[CDJR] += 1.0 k[Qa] E1[Ra] p[DCJQ]
		{"CDJR,aQcb,Rb,ca,CDJQ", -4.0  , 5, {22,5,14,27,21}},		//Ap[CDJR] += -4.0 W[aQcb] E1[Rb] delta[ca] p[CDJQ]
		{"CDJR,aQcb,Rb,ca,DCJQ", 2.0  , 5, {22,5,14,27,21}},		//Ap[CDJR] += 2.0 W[aQcb] E1[Rb] delta[ca] p[DCJQ]
		{"CDJR,aQbc,Rb,ca,CDJQ", 2.0  , 5, {22,6,14,27,21}},		//Ap[CDJR] += 2.0 W[aQbc] E1[Rb] delta[ca] p[CDJQ]
		{"CDJR,aQbc,Rb,ca,DCJQ", -1.0  , 5, {22,6,14,27,21}},		//Ap[CDJR] += -1.0 W[aQbc] E1[Rb] delta[ca] p[DCJQ]
		{"CDJR,aAbC,RaQb,ADJQ", 2.0  , 4, {22,9,15,21}},		//Ap[CDJR] += 2.0 W[aAbC] E2[RaQb] p[ADJQ]
		{"CDJR,aAbD,RaQb,ACJQ", -1.0  , 4, {22,9,15,21}},		//Ap[CDJR] += -1.0 W[aAbD] E2[RaQb] p[ACJQ]
		{"CDJR,aBbC,RaQb,DBJQ", -1.0  , 4, {22,9,15,21}},		//Ap[CDJR] += -1.0 W[aBbC] E2[RaQb] p[DBJQ]
		{"CDJR,aBbD,RaQb,CBJQ", 2.0  , 4, {22,9,15,21}},		//Ap[CDJR] += 2.0 W[aBbD] E2[RaQb] p[CBJQ]
		{"CDJR,aCAb,RaQb,ADJQ", -1.0  , 4, {22,10,15,21}},		//Ap[CDJR] += -1.0 W[aCAb] E2[RaQb] p[ADJQ]
		{"CDJR,aCBb,RabQ,DBJQ", -1.0  , 4, {22,10,15,21}},		//Ap[CDJR] += -1.0 W[aCBb] E2[RabQ] p[DBJQ]
		{"CDJR,aDAb,RabQ,ACJQ", -1.0  , 4, {22,10,15,21}},		//Ap[CDJR] += -1.0 W[aDAb] E2[RabQ] p[ACJQ]
		{"CDJR,aDBb,RabQ,CBJQ", 2.0  , 4, {22,10,15,21}},		//Ap[CDJR] += 2.0 W[aDBb] E2[RabQ] p[CBJQ]
		{"CDJR,IaJb,RaQb,CDIQ", -2.0  , 4, {22,5,15,21}},		//Ap[CDJR] += -2.0 W[IaJb] E2[RaQb] p[CDIQ]
		{"CDJR,IaJb,RaQb,DCIQ", 1.0  , 4, {22,5,15,21}},		//Ap[CDJR] += 1.0 W[IaJb] E2[RaQb] p[DCIQ]
		{"CDJR,IabJ,RaQb,CDIQ", 1.0  , 4, {22,6,15,21}},		//Ap[CDJR] += 1.0 W[IabJ] E2[RaQb] p[CDIQ]
		{"CDJR,IabJ,RabQ,DCIQ", 1.0  , 4, {22,6,15,21}},		//Ap[CDJR] += 1.0 W[IabJ] E2[RabQ] p[DCIQ]
		{"CDJR,Qabc,Rabc,CDJQ", -2.0  , 4, {22,12,15,21}},		//Ap[CDJR] += -2.0 W[Qabc] E2[Rabc] p[CDJQ]
		{"CDJR,Qabc,Rabc,DCJQ", 1.0  , 4, {22,12,15,21}},		//Ap[CDJR] += 1.0 W[Qabc] E2[Rabc] p[DCJQ]

	};
	int f(int i) {
		return 2*i;
	}
	FDomainDecl DomainDecls[1] = {
		{"A", "a", f}
	};
	FEqInfo Overlap[2] = {
		{"ABIP,RP,ABIR", 2.0, 3, {20, 14, 26}},
		{"ABIP,RP,BAIR",-1.0, 3, {20, 14, 26}}
	};
	static void GetMethodInfo(FMethodInfo &Out) {
		Out = FMethodInfo();
		Out.pName = "MRLCC_ACVV";
		Out.perturberClass = "ACVV";
		Out.pSpinClass = "restricted";
		Out.pTensorDecls = &TensorDecls[0];
		Out.nTensorDecls = 31;
		Out.pDomainDecls = &DomainDecls[0];
		Out.nDomainDecls = 1;
		Out.EqsRes = FEqSet(&EqsRes[0], 46, "MRLCC_ACVV/Res");
		Out.Overlap = FEqSet(&Overlap[0], 2, "MRLCC_ACVV/Overlap");
	};
};
