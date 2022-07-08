/*                                                                                                                                                                          
Developed by Sandeep Sharma and Gerald Knizia, 2016                                                                                                                     
Copyright (c) 2016, Sandeep Sharma
*/
namespace NEVPT2_CAAV {

	FTensorDecl TensorDecls[47] = {
		/*  0*/{"t", "cAae", "",USAGE_Amplitude, STORAGE_Memory},
		/*  1*/{"R", "cAae", "",USAGE_Residual, STORAGE_Memory},
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
		/* 13*/{"E1", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 14*/{"E2", "aaaa", "",USAGE_Density, STORAGE_Memory},
		/* 15*/{"S3", "a", "",USAGE_Density, STORAGE_Memory},
		/* 16*/{"S1", "AaAa", "",USAGE_Density, STORAGE_Memory},
		/* 17*/{"T", "cAae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 18*/{"p1", "caae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 19*/{"p2", "caae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 20*/{"Ap1", "caae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 21*/{"Ap2", "caae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 22*/{"b1", "caae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 23*/{"b2", "caae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 24*/{"b", "", "",USAGE_Amplitude, STORAGE_Memory},
		/* 25*/{"P", "cAae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 26*/{"AP", "cAae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 27*/{"B1", "caae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 28*/{"B2", "caae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 29*/{"B", "", "",USAGE_Amplitude, STORAGE_Memory},
		/* 30*/{"W", "eaca", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 31*/{"W", "aeca", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 32*/{"delta", "cc", "",USAGE_Density, STORAGE_Memory},
		/* 33*/{"delta", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 34*/{"delta", "ee", "",USAGE_Density, STORAGE_Memory},
		/* 35*/{"p", "", "",USAGE_Amplitude, STORAGE_Memory},
		/* 36*/{"Ap", "", "",USAGE_Amplitude, STORAGE_Memory},
		/* 37*/{"f", "ec", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 38*/{"I0", "ec", "",USAGE_PlaceHolder, STORAGE_Memory},
		/* 39*/{"I1", "ce", "", USAGE_PlaceHolder, STORAGE_Memory}, //21 SB
		/* 40*/{"I2", "aaaa", "", USAGE_PlaceHolder, STORAGE_Memory}, //16 S
		/* 41*/{"I3", "aaaa", "", USAGE_PlaceHolder, STORAGE_Memory}, //16 notused
		/* 42*/{"I4", "ceaa", "", USAGE_PlaceHolder, STORAGE_Memory}, //10 b
		/* 43*/{"I5", "ceaa", "", USAGE_PlaceHolder, STORAGE_Memory}, //22 SB
		/* 44*/{"I6", "ce", "", USAGE_PlaceHolder, STORAGE_Memory}, //19 B
		/* 45*/{"I7", "caa", "", USAGE_PlaceHolder, STORAGE_Memory}, //20 B
		/* 46*/{"I8", "aea", "", USAGE_PlaceHolder, STORAGE_Memory}, //9 B
	};

//Number of terms :  8
	FEqInfo EqsHandCode[6] = {
		{"JB,Pabc,caQb,JPQB", 2.0  , 4, {39,12,40,18}},		//Ap1[JRSB] += 2.0 W[Pabc] E3[SabRcQ] p1[JPQB]
		{"JB,Qabc,bPca,JPQB", -2.0  , 4, {39,12,40,18}},		//Ap1[JRSB] += -2.0 W[Qabc] E3[PSabRc] p1[JPQB]
//
		{"JB,Pabc,caQb,JPQB", -1.0  , 4, {39,12,40,19}},		//Ap1[JRSB] += -1.0 W[Pabc] E3[SabRcQ] p2[JPQB]
		{"JB,Qabc,bPca,JPQB", 1.0  , 4, {39,12,40,19}},		//Ap1[JRSB] += 1.0 W[Qabc] E3[PSabRc] p2[JPQB]
//
		{"JB,Pabc,caQb,JPQB", -1.0  , 4, {44,12,40,18}},		//Ap2[JRSB] += -1.0 W[Pabc] E3[SabRcQ] p1[JPQB]
		{"JB,Qabc,bPca,JPQB", 1.0  , 4, {44,12,40,18}},		//Ap2[JRSB] += 1.0 W[Qabc] E3[PSabRc] p1[JPQB]
//
        };
//Number of terms :  8
	FEqInfo EqsHandCode2[2] = {
		{"JB,Qabc,Pcab,JPQB", 1.0  , 4, {44,12,41,19}},		//Ap2[JRSB] += 1.0 W[Qabc] E3[PSaRbc] p2[JPQB]
		{"JB,Pabc,bcaQ,JPQB", -1.0  , 4, {44,12,41,19}},		//Ap2[JRSB] += -1.0 W[Pabc] E3[SabQcR] p2[JPQB]
        };
//Number of terms :  56
	FEqInfo EqsRes[48] = {

		{"JRSB,PR,SQ,JPQB", 2.0  , 4, {20,3,13,18}},		//Ap1[JRSB] += 2.0 f[PR] E1[SQ] p1[JPQB]
		{"JRSB,IJ,SQ,IRQB", -2.0  , 4, {20,2,13,18}},		//Ap1[JRSB] += -2.0 f[IJ] E1[SQ] p1[IRQB]
		{"JRSB,AB,SQ,JRQA", 2.0  , 4, {20,4,13,18}},		//Ap1[JRSB] += 2.0 f[AB] E1[SQ] p1[JRQA]
		{"JRSB,Qa,Sa,JRQB", -2.0  , 4, {20,3,13,18}},		//Ap1[JRSB] += -2.0 f[Qa] E1[Sa] p1[JRQB]
		{"JRSB,IJ,PSQR,IPQB", -2.0  , 4, {20,2,14,18}},		//Ap1[JRSB] += -2.0 f[IJ] E2[PSQR] p1[IPQB]
		{"JRSB,AB,PSQR,JPQA", 2.0  , 4, {20,4,14,18}},		//Ap1[JRSB] += 2.0 f[AB] E2[PSQR] p1[JPQA]
		{"JRSB,Pa,SaRQ,JPQB", 2.0  , 4, {20,3,14,18}},		//Ap1[JRSB] += 2.0 f[Pa] E2[SaRQ] p1[JPQB]
		{"JRSB,Qa,PSaR,JPQB", -2.0  , 4, {20,3,14,18}},		//Ap1[JRSB] += -2.0 f[Qa] E2[PSaR] p1[JPQB]
		{"JRSB,PRab,SabQ,JPQB", 2.0  , 4, {20,12,14,18}},		//Ap1[JRSB] += 2.0 W[PRab] E2[SabQ] p1[JPQB]
		{"JRSB,PaRb,SaQb,JPQB", 2.0  , 4, {20,12,14,18}},		//Ap1[JRSB] += 2.0 W[PaRb] E2[SaQb] p1[JPQB]
		{"JRSB,QRab,PSab,JPQB", -2.0  , 4, {20,12,14,18}},		//Ap1[JRSB] += -2.0 W[QRab] E2[PSab] p1[JPQB]
		{"JRSB,Qabc,Sabc,JRQB", -2.0  , 4, {20,12,14,18}},		//Ap1[JRSB] += -2.0 W[Qabc] E2[Sabc] p1[JRQB]
		//{"JRSB,Pabc,SabRcQ,JPQB", 2.0  , 4, {20,12,15,18}},		//Ap1[JRSB] += 2.0 W[Pabc] E3[SabRcQ] p1[JPQB]
		//{"JRSB,Qabc,PSabRc,JPQB", -2.0  , 4, {20,12,15,18}},		//Ap1[JRSB] += -2.0 W[Qabc] E3[PSabRc] p1[JPQB]
//
		{"JRSB,PR,SQ,JPQB", -1.0  , 4, {21,3,13,18}},		//Ap2[JRSB] += -1.0 f[PR] E1[SQ] p1[JPQB]
		{"JRSB,IJ,SQ,IRQB", 1.0  , 4, {21,2,13,18}},		//Ap2[JRSB] += 1.0 f[IJ] E1[SQ] p1[IRQB]
		{"JRSB,AB,SQ,JRQA", -1.0  , 4, {21,4,13,18}},		//Ap2[JRSB] += -1.0 f[AB] E1[SQ] p1[JRQA]
		{"JRSB,Qa,Sa,JRQB", 1.0  , 4, {21,3,13,18}},		//Ap2[JRSB] += 1.0 f[Qa] E1[Sa] p1[JRQB]
		{"JRSB,IJ,PSQR,IPQB", 1.0  , 4, {21,2,14,18}},		//Ap2[JRSB] += 1.0 f[IJ] E2[PSQR] p1[IPQB]
		{"JRSB,AB,PSQR,JPQA", -1.0  , 4, {21,4,14,18}},		//Ap2[JRSB] += -1.0 f[AB] E2[PSQR] p1[JPQA]
		{"JRSB,Pa,SaRQ,JPQB", -1.0  , 4, {21,3,14,18}},		//Ap2[JRSB] += -1.0 f[Pa] E2[SaRQ] p1[JPQB]
		{"JRSB,Qa,PSaR,JPQB", 1.0  , 4, {21,3,14,18}},		//Ap2[JRSB] += 1.0 f[Qa] E2[PSaR] p1[JPQB]
		{"JRSB,PRab,SabQ,JPQB", -1.0  , 4, {21,12,14,18}},		//Ap2[JRSB] += -1.0 W[PRab] E2[SabQ] p1[JPQB]
		{"JRSB,PaRb,SaQb,JPQB", -1.0  , 4, {21,12,14,18}},		//Ap2[JRSB] += -1.0 W[PaRb] E2[SaQb] p1[JPQB]
		{"JRSB,QRab,PSab,JPQB", 1.0  , 4, {21,12,14,18}},		//Ap2[JRSB] += 1.0 W[QRab] E2[PSab] p1[JPQB]
		{"JRSB,Qabc,Sabc,JRQB", 1.0  , 4, {21,12,14,18}},		//Ap2[JRSB] += 1.0 W[Qabc] E2[Sabc] p1[JRQB]
		//{"JRSB,Pabc,SabRcQ,JPQB", -1.0  , 4, {21,12,15,18}},		//Ap2[JRSB] += -1.0 W[Pabc] E3[SabRcQ] p1[JPQB]
		//{"JRSB,Qabc,PSabRc,JPQB", 1.0  , 4, {21,12,15,18}},		//Ap2[JRSB] += 1.0 W[Qabc] E3[PSabRc] p1[JPQB]
//
		{"JRSB,PR,SQ,JPQB", -1.0  , 4, {20,3,13,19}},		//Ap1[JRSB] += -1.0 f[PR] E1[SQ] p2[JPQB]
		{"JRSB,IJ,SQ,IRQB", 1.0  , 4, {20,2,13,19}},		//Ap1[JRSB] += 1.0 f[IJ] E1[SQ] p2[IRQB]
		{"JRSB,AB,SQ,JRQA", -1.0  , 4, {20,4,13,19}},		//Ap1[JRSB] += -1.0 f[AB] E1[SQ] p2[JRQA]
		{"JRSB,Qa,Sa,JRQB", 1.0  , 4, {20,3,13,19}},		//Ap1[JRSB] += 1.0 f[Qa] E1[Sa] p2[JRQB]
		{"JRSB,IJ,PSQR,IPQB", 1.0  , 4, {20,2,14,19}},		//Ap1[JRSB] += 1.0 f[IJ] E2[PSQR] p2[IPQB]
		{"JRSB,AB,PSQR,JPQA", -1.0  , 4, {20,4,14,19}},		//Ap1[JRSB] += -1.0 f[AB] E2[PSQR] p2[JPQA]
		{"JRSB,Pa,SaRQ,JPQB", -1.0  , 4, {20,3,14,19}},		//Ap1[JRSB] += -1.0 f[Pa] E2[SaRQ] p2[JPQB]
		{"JRSB,Qa,PSaR,JPQB", 1.0  , 4, {20,3,14,19}},		//Ap1[JRSB] += 1.0 f[Qa] E2[PSaR] p2[JPQB]
		{"JRSB,PRab,SabQ,JPQB", -1.0  , 4, {20,12,14,19}},		//Ap1[JRSB] += -1.0 W[PRab] E2[SabQ] p2[JPQB]
		{"JRSB,PaRb,SaQb,JPQB", -1.0  , 4, {20,12,14,19}},		//Ap1[JRSB] += -1.0 W[PaRb] E2[SaQb] p2[JPQB]
		{"JRSB,QRab,PSab,JPQB", 1.0  , 4, {20,12,14,19}},		//Ap1[JRSB] += 1.0 W[QRab] E2[PSab] p2[JPQB]
		{"JRSB,Qabc,Sabc,JRQB", 1.0  , 4, {20,12,14,19}},		//Ap1[JRSB] += 1.0 W[Qabc] E2[Sabc] p2[JRQB]
		//{"JRSB,Pabc,SabRcQ,JPQB", -1.0  , 4, {20,12,15,19}},		//Ap1[JRSB] += -1.0 W[Pabc] E3[SabRcQ] p2[JPQB]
		//{"JRSB,Qabc,PSabRc,JPQB", 1.0  , 4, {20,12,15,19}},		//Ap1[JRSB] += 1.0 W[Qabc] E3[PSabRc] p2[JPQB]
//
		{"JRSB,PR,SQ,JPQB", 2.0  , 4, {21,3,13,19}},		//Ap2[JRSB] += 2.0 f[PR] E1[SQ] p2[JPQB]
		{"JRSB,IJ,SQ,IRQB", -2.0  , 4, {21,2,13,19}},		//Ap2[JRSB] += -2.0 f[IJ] E1[SQ] p2[IRQB]
		{"JRSB,AB,SQ,JRQA", 2.0  , 4, {21,4,13,19}},		//Ap2[JRSB] += 2.0 f[AB] E1[SQ] p2[JRQA]
		{"JRSB,Qa,Sa,JRQB", -2.0  , 4, {21,3,13,19}},		//Ap2[JRSB] += -2.0 f[Qa] E1[Sa] p2[JRQB]
		{"JRSB,IJ,PSRQ,IPQB", 1.0  , 4, {21,2,14,19}},		//Ap2[JRSB] += 1.0 f[IJ] E2[PSRQ] p2[IPQB]
		{"JRSB,AB,PSRQ,JPQA", -1.0  , 4, {21,4,14,19}},		//Ap2[JRSB] += -1.0 f[AB] E2[PSRQ] p2[JPQA]
		{"JRSB,Pa,SaQR,JPQB", -1.0  , 4, {21,3,14,19}},		//Ap2[JRSB] += -1.0 f[Pa] E2[SaQR] p2[JPQB]
		{"JRSB,Qa,PSRa,JPQB", 1.0  , 4, {21,3,14,19}},		//Ap2[JRSB] += 1.0 f[Qa] E2[PSRa] p2[JPQB]
		{"JRSB,PRab,SaQb,JPQB", -1.0  , 4, {21,12,14,19}},		//Ap2[JRSB] += -1.0 W[PRab] E2[SaQb] p2[JPQB]
		{"JRSB,PaRb,SaQb,JPQB", 2.0  , 4, {21,12,14,19}},		//Ap2[JRSB] += 2.0 W[PaRb] E2[SaQb] p2[JPQB]
		{"JRSB,QRab,PSba,JPQB", 1.0  , 4, {21,12,14,19}},		//Ap2[JRSB] += 1.0 W[QRab] E2[PSba] p2[JPQB]
		{"JRSB,Qabc,Sabc,JRQB", -2.0  , 4, {21,12,14,19}},		//Ap2[JRSB] += -2.0 W[Qabc] E2[Sabc] p2[JRQB]
		//{"JRSB,Pabc,SabQcR,JPQB", -1.0  , 4, {21,12,15,19}},		//Ap2[JRSB] += -1.0 W[Pabc] E3[SabQcR] p2[JPQB]
		//{"JRSB,Qabc,PSaRbc,JPQB", 1.0  , 4, {21,12,15,19}},		//Ap2[JRSB] += 1.0 W[Qabc] E3[PSaRbc] p2[JPQB]

	};
	int f(int i) {
		return 2*i;
	}
	FDomainDecl DomainDecls[1] = {
		{"A", "a", f}
	};
	FEqInfo Overlap[8] = {
		{"IPSA,QS,APIQ",   2.0, 3, {22, 13, 30}},
		{"IRSA,PSQR,APIQ", 2.0, 3, {22, 14, 30}},
		{"IPSA,QS,APIQ",  -1.0, 3, {23, 13, 30}},
		{"IRSA,PSQR,APIQ",-1.0, 3, {23, 14, 30}},
		{"IPSA,QS,PAIQ",  -1.0, 3, {22, 13, 31}},
		{"IRSA,PSQR,PAIQ",-1.0, 3, {22, 14, 31}},
		{"IPSA,QS,PAIQ",   2.0, 3, {23, 13, 31}},
		{"IRSA,PSRQ,PAIQ",-1.0, 3, {23, 14, 31}},
	};
	static void GetMethodInfo(FMethodInfo &Out) {
		Out = FMethodInfo();
		Out.pName = "NEVPT2_CAAV";
		Out.perturberClass = "CAAV";
		Out.pSpinClass = "restricted";
		Out.pTensorDecls = &TensorDecls[0];
		Out.nTensorDecls = 47;
		Out.pDomainDecls = &DomainDecls[0];
		Out.nDomainDecls = 1;
		Out.EqsHandCode = FEqSet(&EqsHandCode[0], 6, "MRLCC_CAAV/Res");
		Out.EqsHandCode2 = FEqSet(&EqsHandCode2[0], 2, "MRLCC_CAAV/Res");
		Out.EqsRes = FEqSet(&EqsRes[0], 48, "NEVPT2_CAAV/Res");
		Out.Overlap = FEqSet(&Overlap[0], 8, "NEVPT2_CAAV/Overlap");
	};
};
