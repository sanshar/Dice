/*                                                                                                                                                                          
Developed by Sandeep Sharma and Gerald Knizia, 2016                                                                                                                     
Copyright (c) 2016, Sandeep Sharma
*/
namespace MRLCC_CAAV {

	FTensorDecl TensorDecls[47] = {
		/*  0*/{"t", "cAae", "",USAGE_Amplitude, STORAGE_Memory},
		/*  1*/{"R", "cAae", "",USAGE_Residual, STORAGE_Memory},
		/*  2*/{"k", "cc", "",USAGE_Hamiltonian, STORAGE_Memory},
		/*  3*/{"k", "aa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/*  4*/{"k", "ee", "",USAGE_Hamiltonian, STORAGE_Memory},
		/*  5*/{"W", "caca", "",USAGE_Hamiltonian, STORAGE_Memory},
		/*  6*/{"W", "caac", "",USAGE_Hamiltonian, STORAGE_Memory},
		/*  7*/{"W", "cece", "",USAGE_Hamiltonian, STORAGE_Memory},
		/*  8*/{"W", "ceec", "",USAGE_Hamiltonian, STORAGE_Memory},
		/*  9*/{"W", "aeae", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 10*/{"W", "aeea", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 11*/{"W", "cccc", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 12*/{"W", "aaaa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 13*/{"W", "e", "",USAGE_Intermediate, STORAGE_Memory},
		/* 14*/{"E1", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 15*/{"E2", "aaaa", "",USAGE_Density, STORAGE_Memory},
		/* 16*/{"S3", "a", "",USAGE_Density, STORAGE_Memory},
		/* 17*/{"S1", "AaAa", "",USAGE_Density, STORAGE_Memory},
		/* 18*/{"T", "cAae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 19*/{"p1", "caae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 20*/{"p2", "caae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 21*/{"Ap1", "caae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 22*/{"Ap2", "caae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 23*/{"b1", "caae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 24*/{"b2", "caae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 25*/{"b", "", "",USAGE_PlaceHolder, STORAGE_Memory},
		/* 26*/{"P", "cAae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 27*/{"AP", "cAae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 28*/{"B1", "caae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 29*/{"B2", "caae", "",USAGE_Amplitude, STORAGE_Memory},
		/* 30*/{"B", "", "",USAGE_PlaceHolder, STORAGE_Memory},
		/* 31*/{"W", "eaca", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 32*/{"W", "aeca", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 33*/{"delta", "cc", "",USAGE_Density, STORAGE_Memory},
		/* 34*/{"delta", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 35*/{"delta", "ee", "",USAGE_Density, STORAGE_Memory},
		/* 36*/{"p", "", "",USAGE_PlaceHolder, STORAGE_Memory},
		/* 37*/{"Ap", "", "",USAGE_PlaceHolder, STORAGE_Memory},
		/* 38*/{"f", "ec", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 39*/{"I1", "ce", "", USAGE_PlaceHolder, STORAGE_Memory}, //21 SB
		/* 40*/{"I2", "aaaa", "", USAGE_PlaceHolder, STORAGE_Memory}, //16 S
		/* 41*/{"I3", "aaaa", "", USAGE_PlaceHolder, STORAGE_Memory}, //16 notused
		/* 42*/{"I4", "ceaa", "", USAGE_PlaceHolder, STORAGE_Memory}, //10 b
		/* 43*/{"I5", "ceaa", "", USAGE_PlaceHolder, STORAGE_Memory}, //22 SB
		/* 44*/{"I6", "ce", "", USAGE_PlaceHolder, STORAGE_Memory}, //19 B
		/* 45*/{"I7", "caa", "", USAGE_PlaceHolder, STORAGE_Memory}, //20 B
		/* 46*/{"I8", "aea", "", USAGE_PlaceHolder, STORAGE_Memory}, //9 B
	};

//Number of terms :  24
/*
	FEqInfo EqsHandCode2[8] = {
		{"JRSB,aAbB,QPbaRS,JPQA", 2.0  , 4, {21,9,16,19}},		//Ap1[JRSB] += 2.0 W[aAbB] E3[PSaQRb] p1[JPQA]
		{"JRSB,aBAb,PQSRab,JPQA", -1.0  , 4, {21,10,16,19}},		//Ap1[JRSB] += -1.0 W[aBAb] E3[PSaQRb] p1[JPQA]
		{"JRSB,IaJb,QPbaRS,IPQB", -2.0  , 4, {21,5,16,19}},		//Ap1[JRSB] += -2.0 W[IaJb] E3[PSaQRb] p1[IPQB]
		{"JRSB,IabJ,QPbaRS,IPQB", 1.0  , 4, {21,6,16,19}},		//Ap1[JRSB] += 1.0 W[IabJ] E3[PSaQRb] p1[IPQB]
		{"JRSB,Pabc,caQbRS,JPQB", 2.0  , 4, {21,12,16,19}},		//Ap1[JRSB] += 2.0 W[Pabc] E3[SabRcQ] p1[JPQB]
		{"JRSB,Qabc,bPcaRS,JPQB", -2.0  , 4, {21,12,16,19}},		//Ap1[JRSB] += -2.0 W[Qabc] E3[PSabRc] p1[JPQB]
//
		{"JRSB,aAbB,QPbaRS,JPQA", -1.0  , 4, {21,9,16,20}},		//Ap1[JRSB] += -1.0 W[aAbB] E3[PSaQRb] p2[JPQA]
		{"JRSB,aBAb,bPQaRS,JPQA", -1.0  , 4, {21,10,16,20}},		//Ap1[JRSB] += -1.0 W[aBAb] E3[PSabRQ] p2[JPQA]
		{"JRSB,IaJb,QPbaRS,IPQB", 1.0  , 4, {21,5,16,20}},		//Ap1[JRSB] += 1.0 W[IaJb] E3[PSaQRb] p2[IPQB]
		{"JRSB,IabJ,bPQaRS,IPQB", 1.0  , 4, {21,6,16,20}},		//Ap1[JRSB] += 1.0 W[IabJ] E3[PSabRQ] p2[IPQB]
		{"JRSB,Pabc,caQbRS,JPQB", -1.0  , 4, {21,12,16,20}},		//Ap1[JRSB] += -1.0 W[Pabc] E3[SabRcQ] p2[JPQB]
		{"JRSB,Qabc,bPcaRS,JPQB", 1.0  , 4, {21,12,16,20}},		//Ap1[JRSB] += 1.0 W[Qabc] E3[PSabRc] p2[JPQB]
//
		{"JRSB,IaJb,QPbaRS,IPQB", 1.0  , 4, {22,5,16,19}},		//Ap2[JRSB] += 1.0 W[IaJb] E3[PSaQRb] p1[IPQB]
		{"JRSB,Pabc,caQbRS,JPQB", -1.0  , 4, {22,12,16,19}},		//Ap2[JRSB] += -1.0 W[Pabc] E3[SabRcQ] p1[JPQB]
		{"JRSB,Qabc,bPcaRS,JPQB", 1.0  , 4, {22,12,16,19}},		//Ap2[JRSB] += 1.0 W[Qabc] E3[PSabRc] p1[JPQB]
		{"JRSB,aAbB,QPbaRS,JPQA", -1.0  , 4, {22,9,16,19}},		//Ap2[JRSB] += -1.0 W[aAbB] E3[PSaQRb] p1[JPQA]
		{"JRSB,IabJ,QPRabS,IPQB", 1.0  , 4, {22,6,16,19}},		//Ap2[JRSB] += 1.0 W[IabJ] E3[PSaQbR] p1[IPQB]
		{"JRSB,aBAb,PQaRSb,JPQA", -1.0  , 4, {22,10,16,19}},		//Ap2[JRSB] += -1.0 W[aBAb] E3[PSaQbR] p1[JPQA]
//
		{"JRSB,aAbB,RPbaQS,JPQA", -1.0  , 4, {22,9,16,20}},		//Ap2[JRSB] += -1.0 W[aAbB] E3[PSaRQb] p2[JPQA]
		{"JRSB,aBAb,RPQabS,JPQA", -1.0  , 4, {22,10,16,20}},		//Ap2[JRSB] += -1.0 W[aBAb] E3[PSaRbQ] p2[JPQA]
		{"JRSB,IaJb,RPbaQS,IPQB", 1.0  , 4, {22,5,16,20}},		//Ap2[JRSB] += 1.0 W[IaJb] E3[PSaRQb] p2[IPQB]
		{"JRSB,IabJ,bPRaQS,IPQB", 1.0  , 4, {22,6,16,20}},		//Ap2[JRSB] += 1.0 W[IabJ] E3[PSabQR] p2[IPQB]
		{"JRSB,Pabc,caRbQS,JPQB", -1.0  , 4, {22,12,16,20}},		//Ap2[JRSB] += -1.0 W[Pabc] E3[SabQcR] p2[JPQB]
		{"JRSB,Qabc,RPcabS,JPQB", 1.0  , 4, {22,12,16,20}},		//Ap2[JRSB] += 1.0 W[Qabc] E3[PSaRbc] p2[JPQB]
        };
*/

//Number of terms :  24
	FEqInfo EqsHandCode[24] = {
		{"JB,aAbB,QPba,JPQA", 2.0  , 4, {39,9,40,19}},		//Ap1[JRSB] += 2.0 W[aAbB] E3[PSaQRb] p1[JPQA]
		{"JB,aBAb,QPba,JPQA", -1.0  , 4, {39,10,40,19}},		//Ap1[JRSB] += -1.0 W[aBAb] E3[PSaQRb] p1[JPQA]
		{"JB,IaJb,QPba,IPQB", -2.0  , 4, {39,5,40,19}},		//Ap1[JRSB] += -2.0 W[IaJb] E3[PSaQRb] p1[IPQB]
		{"JB,IabJ,QPba,IPQB", 1.0  , 4, {39,6,40,19}},		//Ap1[JRSB] += 1.0 W[IabJ] E3[PSaQRb] p1[IPQB]
		{"JB,Pabc,caQb,JPQB", 2.0  , 4, {39,12,40,19}},		//Ap1[JRSB] += 2.0 W[Pabc] E3[SabRcQ] p1[JPQB]
		{"JB,Qabc,bPca,JPQB", -2.0  , 4, {39,12,40,19}},		//Ap1[JRSB] += -2.0 W[Qabc] E3[PSabRc] p1[JPQB]
//
		{"JB,aAbB,QPba,JPQA", -1.0  , 4, {39,9,40,20}},		//Ap1[JRSB] += -1.0 W[aAbB] E3[PSaQRb] p2[JPQA]
		{"JB,aBAb,bPQa,JPQA", -1.0  , 4, {39,10,40,20}},		//Ap1[JRSB] += -1.0 W[aBAb] E3[PSabRQ] p2[JPQA]
		{"JB,IaJb,QPba,IPQB", 1.0  , 4, {39,5,40,20}},		//Ap1[JRSB] += 1.0 W[IaJb] E3[PSaQRb] p2[IPQB]
		{"JB,IabJ,bPQa,IPQB", 1.0  , 4, {39,6,40,20}},		//Ap1[JRSB] += 1.0 W[IabJ] E3[PSabRQ] p2[IPQB]
		{"JB,Pabc,caQb,JPQB", -1.0  , 4, {39,12,40,20}},		//Ap1[JRSB] += -1.0 W[Pabc] E3[SabRcQ] p2[JPQB]
		{"JB,Qabc,bPca,JPQB", 1.0  , 4, {39,12,40,20}},		//Ap1[JRSB] += 1.0 W[Qabc] E3[PSabRc] p2[JPQB]
//
		{"JB,IaJb,QPba,IPQB", 1.0  , 4, {44,5,40,19}},		//Ap2[JRSB] += 1.0 W[IaJb] E3[PSaQRb] p1[IPQB]
		{"JB,Pabc,caQb,JPQB", -1.0  , 4, {44,12,40,19}},		//Ap2[JRSB] += -1.0 W[Pabc] E3[SabRcQ] p1[JPQB]
		{"JB,Qabc,bPca,JPQB", 1.0  , 4, {44,12,40,19}},		//Ap2[JRSB] += 1.0 W[Qabc] E3[PSabRc] p1[JPQB]
		{"JB,aAbB,QPba,JPQA", -1.0  , 4, {44,9,40,19}},		//Ap2[JRSB] += -1.0 W[aAbB] E3[PSaQRb] p1[JPQA]
       };

       FEqInfo EqsHandCode2[8] = {
		{"JB,IabJ,aQPb,IPQB", 1.0  , 4, {44,6,41,19}},		//Ap2[JRSB] += 1.0 W[IabJ] E3[PSaQbR] p1[IPQB]
		{"JB,aBAb,aQPb,JPQA", -1.0  , 4, {44,10,41,19}},		//Ap2[JRSB] += -1.0 W[aBAb] E3[PSaQbR] p1[JPQA]
//
		{"JB,aAbB,PbaQ,JPQA", -1.0  , 4, {44,9,41,20}},		//Ap2[JRSB] += -1.0 W[aAbB] E3[PSaRQb] p2[JPQA]
		{"JB,aBAb,PQab,JPQA", -1.0  , 4, {44,10,41,20}},		//Ap2[JRSB] += -1.0 W[aBAb] E3[PSaRbQ] p2[JPQA]
		{"JB,IaJb,PbaQ,IPQB", 1.0  , 4, {44,5,41,20}},		//Ap2[JRSB] += 1.0 W[IaJb] E3[PSaRQb] p2[IPQB]
		{"JB,Qabc,Pcab,JPQB", 1.0  , 4, {44,12,41,20}},		//Ap2[JRSB] += 1.0 W[Qabc] E3[PSaRbc] p2[JPQB]
		{"JB,IabJ,abPQ,IPQB", 1.0  , 4, {44,6,41,20}},		//Ap2[JRSB] += 1.0 W[IabJ] E3[PSabQR] p2[IPQB]
		{"JB,Pabc,bcaQ,JPQB", -1.0  , 4, {44,12,41,20}},		//Ap2[JRSB] += -1.0 W[Pabc] E3[SabQcR] p2[JPQB]
        };


//Number of terms :  192
	FEqInfo EqsRes[192] = {

		{"JRSB,PARB,SQ,JPQA", 2.0  , 4, {21,9,14,19}},		//Ap1[JRSB] += 2.0 W[PARB] E1[SQ] p1[JPQA]
		{"JRSB,RBAP,SQ,JPQA", -1.0  , 4, {21,10,14,19}},		//Ap1[JRSB] += -1.0 W[RBAP] E1[SQ] p1[JPQA]
		{"JRSB,IPJR,SQ,IPQB", -2.0  , 4, {21,5,14,19}},		//Ap1[JRSB] += -2.0 W[IPJR] E1[SQ] p1[IPQB]
		{"JRSB,IRPJ,SQ,IPQB", 1.0  , 4, {21,6,14,19}},		//Ap1[JRSB] += 1.0 W[IRPJ] E1[SQ] p1[IPQB]
		{"JRSB,IAJB,SQ,IRQA", -2.0  , 4, {21,7,14,19}},		//Ap1[JRSB] += -2.0 W[IAJB] E1[SQ] p1[IRQA]
		{"JRSB,IBAJ,SQ,IRQA", 4.0  , 4, {21,8,14,19}},		//Ap1[JRSB] += 4.0 W[IBAJ] E1[SQ] p1[IRQA]
		{"JRSB,PR,SQ,JPQB", 2.0  , 4, {21,3,14,19}},		//Ap1[JRSB] += 2.0 k[PR] E1[SQ] p1[JPQB]
		{"JRSB,IJ,SQ,IRQB", -2.0  , 4, {21,2,14,19}},		//Ap1[JRSB] += -2.0 k[IJ] E1[SQ] p1[IRQB]
		{"JRSB,AB,SQ,JRQA", 2.0  , 4, {21,4,14,19}},		//Ap1[JRSB] += 2.0 k[AB] E1[SQ] p1[JRQA]
		{"JRSB,IJab,SQ,ba,IRQB", 2.0  , 5, {21,11,14,33,19}},		//Ap1[JRSB] += 2.0 W[IJab] E1[SQ] delta[ba] p1[IRQB]
		{"JRSB,IaJb,SQ,ba,IRQB", -4.0  , 5, {21,11,14,33,19}},		//Ap1[JRSB] += -4.0 W[IaJb] E1[SQ] delta[ba] p1[IRQB]
		{"JRSB,aPbR,SQ,ba,JPQB", 4.0  , 5, {21,5,14,33,19}},		//Ap1[JRSB] += 4.0 W[aPbR] E1[SQ] delta[ba] p1[JPQB]
		{"JRSB,aRPb,SQ,ba,JPQB", -2.0  , 5, {21,6,14,33,19}},		//Ap1[JRSB] += -2.0 W[aRPb] E1[SQ] delta[ba] p1[JPQB]
		{"JRSB,aAbB,SQ,ba,JRQA", 4.0  , 5, {21,7,14,33,19}},		//Ap1[JRSB] += 4.0 W[aAbB] E1[SQ] delta[ba] p1[JRQA]
		{"JRSB,aBAb,SQ,ba,JRQA", -2.0  , 5, {21,8,14,33,19}},		//Ap1[JRSB] += -2.0 W[aBAb] E1[SQ] delta[ba] p1[JRQA]
		{"JRSB,Qa,Sa,JRQB", -2.0  , 4, {21,3,14,19}},		//Ap1[JRSB] += -2.0 k[Qa] E1[Sa] p1[JRQB]
		{"JRSB,aQcb,Sb,ca,JRQB", -4.0  , 5, {21,5,14,33,19}},		//Ap1[JRSB] += -4.0 W[aQcb] E1[Sb] delta[ca] p1[JRQB]
		{"JRSB,aQbc,Sb,ca,JRQB", 2.0  , 5, {21,6,14,33,19}},		//Ap1[JRSB] += 2.0 W[aQbc] E1[Sb] delta[ca] p1[JRQB]
		{"JRSB,IAJB,PSQR,IPQA", -2.0  , 4, {21,7,15,19}},		//Ap1[JRSB] += -2.0 W[IAJB] E2[PSQR] p1[IPQA]
		{"JRSB,IBAJ,PSQR,IPQA", 4.0  , 4, {21,8,15,19}},		//Ap1[JRSB] += 4.0 W[IBAJ] E2[PSQR] p1[IPQA]
		{"JRSB,IJ,PSQR,IPQB", -2.0  , 4, {21,2,15,19}},		//Ap1[JRSB] += -2.0 k[IJ] E2[PSQR] p1[IPQB]
		{"JRSB,AB,PSQR,JPQA", 2.0  , 4, {21,4,15,19}},		//Ap1[JRSB] += 2.0 k[AB] E2[PSQR] p1[JPQA]
		{"JRSB,PAaB,SaRQ,JPQA", 2.0  , 4, {21,9,15,19}},		//Ap1[JRSB] += 2.0 W[PAaB] E2[SaRQ] p1[JPQA]
		{"JRSB,RAaB,PSQa,JPQA", 2.0  , 4, {21,9,15,19}},		//Ap1[JRSB] += 2.0 W[RAaB] E2[PSQa] p1[JPQA]
		{"JRSB,RBAa,PSQa,JPQA", -1.0  , 4, {21,10,15,19}},		//Ap1[JRSB] += -1.0 W[RBAa] E2[PSQa] p1[JPQA]
		{"JRSB,aBAP,SaRQ,JPQA", -1.0  , 4, {21,10,15,19}},		//Ap1[JRSB] += -1.0 W[aBAP] E2[SaRQ] p1[JPQA]
		{"JRSB,IJab,PSQR,ba,IPQB", 2.0  , 5, {21,11,15,33,19}},		//Ap1[JRSB] += 2.0 W[IJab] E2[PSQR] delta[ba] p1[IPQB]
		{"JRSB,IaJb,PSQR,ba,IPQB", -4.0  , 5, {21,11,15,33,19}},		//Ap1[JRSB] += -4.0 W[IaJb] E2[PSQR] delta[ba] p1[IPQB]
		{"JRSB,IPJa,SaRQ,IPQB", -2.0  , 4, {21,5,15,19}},		//Ap1[JRSB] += -2.0 W[IPJa] E2[SaRQ] p1[IPQB]
		{"JRSB,IRJa,PSQa,IPQB", -2.0  , 4, {21,5,15,19}},		//Ap1[JRSB] += -2.0 W[IRJa] E2[PSQa] p1[IPQB]
		{"JRSB,IRaJ,PSQa,IPQB", 1.0  , 4, {21,6,15,19}},		//Ap1[JRSB] += 1.0 W[IRaJ] E2[PSQa] p1[IPQB]
		{"JRSB,IaPJ,SaRQ,IPQB", 1.0  , 4, {21,6,15,19}},		//Ap1[JRSB] += 1.0 W[IaPJ] E2[SaRQ] p1[IPQB]
		{"JRSB,aAbB,PSQR,ba,JPQA", 4.0  , 5, {21,7,15,33,19}},		//Ap1[JRSB] += 4.0 W[aAbB] E2[PSQR] delta[ba] p1[JPQA]
		{"JRSB,aBAb,PSQR,ba,JPQA", -2.0  , 5, {21,8,15,33,19}},		//Ap1[JRSB] += -2.0 W[aBAb] E2[PSQR] delta[ba] p1[JPQA]
		{"JRSB,Pa,SaRQ,JPQB", 2.0  , 4, {21,3,15,19}},		//Ap1[JRSB] += 2.0 k[Pa] E2[SaRQ] p1[JPQB]
		{"JRSB,Qa,PSaR,JPQB", -2.0  , 4, {21,3,15,19}},		//Ap1[JRSB] += -2.0 k[Qa] E2[PSaR] p1[JPQB]
		{"JRSB,PRab,SabQ,JPQB", 2.0  , 4, {21,12,15,19}},		//Ap1[JRSB] += 2.0 W[PRab] E2[SabQ] p1[JPQB]
		{"JRSB,PaRb,SaQb,JPQB", 2.0  , 4, {21,12,15,19}},		//Ap1[JRSB] += 2.0 W[PaRb] E2[SaQb] p1[JPQB]
		{"JRSB,QRab,PSab,JPQB", -2.0  , 4, {21,12,15,19}},		//Ap1[JRSB] += -2.0 W[QRab] E2[PSab] p1[JPQB]
		{"JRSB,aAbB,SaQb,JRQA", 2.0  , 4, {21,9,15,19}},		//Ap1[JRSB] += 2.0 W[aAbB] E2[SaQb] p1[JRQA]
		{"JRSB,aBAb,SaQb,JRQA", -1.0  , 4, {21,10,15,19}},		//Ap1[JRSB] += -1.0 W[aBAb] E2[SaQb] p1[JRQA]
		{"JRSB,IaJb,SaQb,IRQB", -2.0  , 4, {21,5,15,19}},		//Ap1[JRSB] += -2.0 W[IaJb] E2[SaQb] p1[IRQB]
		{"JRSB,aPcb,SbRQ,ca,JPQB", 4.0  , 5, {21,5,15,33,19}},		//Ap1[JRSB] += 4.0 W[aPcb] E2[SbRQ] delta[ca] p1[JPQB]
		{"JRSB,aQcb,PSbR,ca,JPQB", -4.0  , 5, {21,5,15,33,19}},		//Ap1[JRSB] += -4.0 W[aQcb] E2[PSbR] delta[ca] p1[JPQB]
		{"JRSB,IabJ,SaQb,IRQB", 1.0  , 4, {21,6,15,19}},		//Ap1[JRSB] += 1.0 W[IabJ] E2[SaQb] p1[IRQB]
		{"JRSB,aQbc,PSbR,ca,JPQB", 2.0  , 5, {21,6,15,33,19}},		//Ap1[JRSB] += 2.0 W[aQbc] E2[PSbR] delta[ca] p1[JPQB]
		{"JRSB,abPc,SbRQ,ca,JPQB", -2.0  , 5, {21,6,15,33,19}},		//Ap1[JRSB] += -2.0 W[abPc] E2[SbRQ] delta[ca] p1[JPQB]
		{"JRSB,Qabc,Sabc,JRQB", -2.0  , 4, {21,12,15,19}},		//Ap1[JRSB] += -2.0 W[Qabc] E2[Sabc] p1[JRQB]
//
		{"JRSB,PARB,SQ,JPQA", -1.0  , 4, {22,9,14,19}},		//Ap2[JRSB] += -1.0 W[PARB] E1[SQ] p1[JPQA]
		{"JRSB,RBAP,SQ,JPQA", 2.0  , 4, {22,10,14,19}},		//Ap2[JRSB] += 2.0 W[RBAP] E1[SQ] p1[JPQA]
		{"JRSB,IPJR,SQ,IPQB", 1.0  , 4, {22,5,14,19}},		//Ap2[JRSB] += 1.0 W[IPJR] E1[SQ] p1[IPQB]
		{"JRSB,IRPJ,SQ,IPQB", -2.0  , 4, {22,6,14,19}},		//Ap2[JRSB] += -2.0 W[IRPJ] E1[SQ] p1[IPQB]
		{"JRSB,IAJB,SQ,IRQA", 1.0  , 4, {22,7,14,19}},		//Ap2[JRSB] += 1.0 W[IAJB] E1[SQ] p1[IRQA]
		{"JRSB,IBAJ,SQ,IRQA", -2.0  , 4, {22,8,14,19}},		//Ap2[JRSB] += -2.0 W[IBAJ] E1[SQ] p1[IRQA]
		{"JRSB,PR,SQ,JPQB", -1.0  , 4, {22,3,14,19}},		//Ap2[JRSB] += -1.0 k[PR] E1[SQ] p1[JPQB]
		{"JRSB,IJ,SQ,IRQB", 1.0  , 4, {22,2,14,19}},		//Ap2[JRSB] += 1.0 k[IJ] E1[SQ] p1[IRQB]
		{"JRSB,AB,SQ,JRQA", -1.0  , 4, {22,4,14,19}},		//Ap2[JRSB] += -1.0 k[AB] E1[SQ] p1[JRQA]
		{"JRSB,IJab,SQ,ba,IRQB", -1.0  , 5, {22,11,14,33,19}},		//Ap2[JRSB] += -1.0 W[IJab] E1[SQ] delta[ba] p1[IRQB]
		{"JRSB,IaJb,SQ,ba,IRQB", 2.0  , 5, {22,11,14,33,19}},		//Ap2[JRSB] += 2.0 W[IaJb] E1[SQ] delta[ba] p1[IRQB]
		{"JRSB,aPbR,SQ,ba,JPQB", -2.0  , 5, {22,5,14,33,19}},		//Ap2[JRSB] += -2.0 W[aPbR] E1[SQ] delta[ba] p1[JPQB]
		{"JRSB,aRPb,SQ,ba,JPQB", 1.0  , 5, {22,6,14,33,19}},		//Ap2[JRSB] += 1.0 W[aRPb] E1[SQ] delta[ba] p1[JPQB]
		{"JRSB,aAbB,SQ,ba,JRQA", -2.0  , 5, {22,7,14,33,19}},		//Ap2[JRSB] += -2.0 W[aAbB] E1[SQ] delta[ba] p1[JRQA]
		{"JRSB,aBAb,SQ,ba,JRQA", 1.0  , 5, {22,8,14,33,19}},		//Ap2[JRSB] += 1.0 W[aBAb] E1[SQ] delta[ba] p1[JRQA]
		{"JRSB,Qa,Sa,JRQB", 1.0  , 4, {22,3,14,19}},		//Ap2[JRSB] += 1.0 k[Qa] E1[Sa] p1[JRQB]
		{"JRSB,aQcb,Sb,ca,JRQB", 2.0  , 5, {22,5,14,33,19}},		//Ap2[JRSB] += 2.0 W[aQcb] E1[Sb] delta[ca] p1[JRQB]
		{"JRSB,aQbc,Sb,ca,JRQB", -1.0  , 5, {22,6,14,33,19}},		//Ap2[JRSB] += -1.0 W[aQbc] E1[Sb] delta[ca] p1[JRQB]
		{"JRSB,IAJB,PSQR,IPQA", 1.0  , 4, {22,7,15,19}},		//Ap2[JRSB] += 1.0 W[IAJB] E2[PSQR] p1[IPQA]
		{"JRSB,IBAJ,PSQR,IPQA", -2.0  , 4, {22,8,15,19}},		//Ap2[JRSB] += -2.0 W[IBAJ] E2[PSQR] p1[IPQA]
		{"JRSB,IJ,PSQR,IPQB", 1.0  , 4, {22,2,15,19}},		//Ap2[JRSB] += 1.0 k[IJ] E2[PSQR] p1[IPQB]
		{"JRSB,AB,PSQR,JPQA", -1.0  , 4, {22,4,15,19}},		//Ap2[JRSB] += -1.0 k[AB] E2[PSQR] p1[JPQA]
		{"JRSB,PAaB,SaRQ,JPQA", -1.0  , 4, {22,9,15,19}},		//Ap2[JRSB] += -1.0 W[PAaB] E2[SaRQ] p1[JPQA]
		{"JRSB,RAaB,PSQa,JPQA", -1.0  , 4, {22,9,15,19}},		//Ap2[JRSB] += -1.0 W[RAaB] E2[PSQa] p1[JPQA]
		{"JRSB,RBAa,PSQa,JPQA", 2.0  , 4, {22,10,15,19}},		//Ap2[JRSB] += 2.0 W[RBAa] E2[PSQa] p1[JPQA]
		{"JRSB,aBAP,SaQR,JPQA", -1.0  , 4, {22,10,15,19}},		//Ap2[JRSB] += -1.0 W[aBAP] E2[SaQR] p1[JPQA]
		{"JRSB,IJab,PSQR,ba,IPQB", -1.0  , 5, {22,11,15,33,19}},		//Ap2[JRSB] += -1.0 W[IJab] E2[PSQR] delta[ba] p1[IPQB]
		{"JRSB,IaJb,PSQR,ba,IPQB", 2.0  , 5, {22,11,15,33,19}},		//Ap2[JRSB] += 2.0 W[IaJb] E2[PSQR] delta[ba] p1[IPQB]
		{"JRSB,IPJa,SaRQ,IPQB", 1.0  , 4, {22,5,15,19}},		//Ap2[JRSB] += 1.0 W[IPJa] E2[SaRQ] p1[IPQB]
		{"JRSB,IRJa,PSQa,IPQB", 1.0  , 4, {22,5,15,19}},		//Ap2[JRSB] += 1.0 W[IRJa] E2[PSQa] p1[IPQB]
		{"JRSB,IRaJ,PSQa,IPQB", -2.0  , 4, {22,6,15,19}},		//Ap2[JRSB] += -2.0 W[IRaJ] E2[PSQa] p1[IPQB]
		{"JRSB,IaPJ,SaQR,IPQB", 1.0  , 4, {22,6,15,19}},		//Ap2[JRSB] += 1.0 W[IaPJ] E2[SaQR] p1[IPQB]
		{"JRSB,aAbB,PSQR,ba,JPQA", -2.0  , 5, {22,7,15,33,19}},		//Ap2[JRSB] += -2.0 W[aAbB] E2[PSQR] delta[ba] p1[JPQA]
		{"JRSB,aBAb,PSQR,ba,JPQA", 1.0  , 5, {22,8,15,33,19}},		//Ap2[JRSB] += 1.0 W[aBAb] E2[PSQR] delta[ba] p1[JPQA]
		{"JRSB,Pa,SaRQ,JPQB", -1.0  , 4, {22,3,15,19}},		//Ap2[JRSB] += -1.0 k[Pa] E2[SaRQ] p1[JPQB]
		{"JRSB,Qa,PSaR,JPQB", 1.0  , 4, {22,3,15,19}},		//Ap2[JRSB] += 1.0 k[Qa] E2[PSaR] p1[JPQB]
		{"JRSB,PRab,SabQ,JPQB", -1.0  , 4, {22,12,15,19}},		//Ap2[JRSB] += -1.0 W[PRab] E2[SabQ] p1[JPQB]
		{"JRSB,PaRb,SaQb,JPQB", -1.0  , 4, {22,12,15,19}},		//Ap2[JRSB] += -1.0 W[PaRb] E2[SaQb] p1[JPQB]
		{"JRSB,QRab,PSab,JPQB", 1.0  , 4, {22,12,15,19}},		//Ap2[JRSB] += 1.0 W[QRab] E2[PSab] p1[JPQB]
		{"JRSB,aAbB,SaQb,JRQA", -1.0  , 4, {22,9,15,19}},		//Ap2[JRSB] += -1.0 W[aAbB] E2[SaQb] p1[JRQA]
		{"JRSB,aBAb,SabQ,JRQA", -1.0  , 4, {22,10,15,19}},		//Ap2[JRSB] += -1.0 W[aBAb] E2[SabQ] p1[JRQA]
		{"JRSB,IaJb,SaQb,IRQB", 1.0  , 4, {22,5,15,19}},		//Ap2[JRSB] += 1.0 W[IaJb] E2[SaQb] p1[IRQB]
		{"JRSB,aPcb,SbRQ,ca,JPQB", -2.0  , 5, {22,5,15,33,19}},		//Ap2[JRSB] += -2.0 W[aPcb] E2[SbRQ] delta[ca] p1[JPQB]
		{"JRSB,aQcb,PSbR,ca,JPQB", 2.0  , 5, {22,5,15,33,19}},		//Ap2[JRSB] += 2.0 W[aQcb] E2[PSbR] delta[ca] p1[JPQB]
		{"JRSB,IabJ,SabQ,IRQB", 1.0  , 4, {22,6,15,19}},		//Ap2[JRSB] += 1.0 W[IabJ] E2[SabQ] p1[IRQB]
		{"JRSB,aQbc,PSbR,ca,JPQB", -1.0  , 5, {22,6,15,33,19}},		//Ap2[JRSB] += -1.0 W[aQbc] E2[PSbR] delta[ca] p1[JPQB]
		{"JRSB,abPc,SbRQ,ca,JPQB", 1.0  , 5, {22,6,15,33,19}},		//Ap2[JRSB] += 1.0 W[abPc] E2[SbRQ] delta[ca] p1[JPQB]
		{"JRSB,Qabc,Sabc,JRQB", 1.0  , 4, {22,12,15,19}},		//Ap2[JRSB] += 1.0 W[Qabc] E2[Sabc] p1[JRQB]
//
		{"JRSB,PARB,SQ,JPQA", -1.0  , 4, {21,9,14,20}},		//Ap1[JRSB] += -1.0 W[PARB] E1[SQ] p2[JPQA]
		{"JRSB,RBAP,SQ,JPQA", 2.0  , 4, {21,10,14,20}},		//Ap1[JRSB] += 2.0 W[RBAP] E1[SQ] p2[JPQA]
		{"JRSB,IPJR,SQ,IPQB", 1.0  , 4, {21,5,14,20}},		//Ap1[JRSB] += 1.0 W[IPJR] E1[SQ] p2[IPQB]
		{"JRSB,IRPJ,SQ,IPQB", -2.0  , 4, {21,6,14,20}},		//Ap1[JRSB] += -2.0 W[IRPJ] E1[SQ] p2[IPQB]
		{"JRSB,IAJB,SQ,IRQA", 1.0  , 4, {21,7,14,20}},		//Ap1[JRSB] += 1.0 W[IAJB] E1[SQ] p2[IRQA]
		{"JRSB,IBAJ,SQ,IRQA", -2.0  , 4, {21,8,14,20}},		//Ap1[JRSB] += -2.0 W[IBAJ] E1[SQ] p2[IRQA]
		{"JRSB,PR,SQ,JPQB", -1.0  , 4, {21,3,14,20}},		//Ap1[JRSB] += -1.0 k[PR] E1[SQ] p2[JPQB]
		{"JRSB,IJ,SQ,IRQB", 1.0  , 4, {21,2,14,20}},		//Ap1[JRSB] += 1.0 k[IJ] E1[SQ] p2[IRQB]
		{"JRSB,AB,SQ,JRQA", -1.0  , 4, {21,4,14,20}},		//Ap1[JRSB] += -1.0 k[AB] E1[SQ] p2[JRQA]
		{"JRSB,IJab,SQ,ba,IRQB", -1.0  , 5, {21,11,14,33,20}},		//Ap1[JRSB] += -1.0 W[IJab] E1[SQ] delta[ba] p2[IRQB]
		{"JRSB,IaJb,SQ,ba,IRQB", 2.0  , 5, {21,11,14,33,20}},		//Ap1[JRSB] += 2.0 W[IaJb] E1[SQ] delta[ba] p2[IRQB]
		{"JRSB,aPbR,SQ,ba,JPQB", -2.0  , 5, {21,5,14,33,20}},		//Ap1[JRSB] += -2.0 W[aPbR] E1[SQ] delta[ba] p2[JPQB]
		{"JRSB,aRPb,SQ,ba,JPQB", 1.0  , 5, {21,6,14,33,20}},		//Ap1[JRSB] += 1.0 W[aRPb] E1[SQ] delta[ba] p2[JPQB]
		{"JRSB,aAbB,SQ,ba,JRQA", -2.0  , 5, {21,7,14,33,20}},		//Ap1[JRSB] += -2.0 W[aAbB] E1[SQ] delta[ba] p2[JRQA]
		{"JRSB,aBAb,SQ,ba,JRQA", 1.0  , 5, {21,8,14,33,20}},		//Ap1[JRSB] += 1.0 W[aBAb] E1[SQ] delta[ba] p2[JRQA]
		{"JRSB,Qa,Sa,JRQB", 1.0  , 4, {21,3,14,20}},		//Ap1[JRSB] += 1.0 k[Qa] E1[Sa] p2[JRQB]
		{"JRSB,aQcb,Sb,ca,JRQB", 2.0  , 5, {21,5,14,33,20}},		//Ap1[JRSB] += 2.0 W[aQcb] E1[Sb] delta[ca] p2[JRQB]
		{"JRSB,aQbc,Sb,ca,JRQB", -1.0  , 5, {21,6,14,33,20}},		//Ap1[JRSB] += -1.0 W[aQbc] E1[Sb] delta[ca] p2[JRQB]
		{"JRSB,IAJB,PSQR,IPQA", 1.0  , 4, {21,7,15,20}},		//Ap1[JRSB] += 1.0 W[IAJB] E2[PSQR] p2[IPQA]
		{"JRSB,IBAJ,PSQR,IPQA", -2.0  , 4, {21,8,15,20}},		//Ap1[JRSB] += -2.0 W[IBAJ] E2[PSQR] p2[IPQA]
		{"JRSB,IJ,PSQR,IPQB", 1.0  , 4, {21,2,15,20}},		//Ap1[JRSB] += 1.0 k[IJ] E2[PSQR] p2[IPQB]
		{"JRSB,AB,PSQR,JPQA", -1.0  , 4, {21,4,15,20}},		//Ap1[JRSB] += -1.0 k[AB] E2[PSQR] p2[JPQA]
		{"JRSB,PAaB,SaRQ,JPQA", -1.0  , 4, {21,9,15,20}},		//Ap1[JRSB] += -1.0 W[PAaB] E2[SaRQ] p2[JPQA]
		{"JRSB,RAaB,PSQa,JPQA", -1.0  , 4, {21,9,15,20}},		//Ap1[JRSB] += -1.0 W[RAaB] E2[PSQa] p2[JPQA]
		{"JRSB,RBAa,PSaQ,JPQA", -1.0  , 4, {21,10,15,20}},		//Ap1[JRSB] += -1.0 W[RBAa] E2[PSaQ] p2[JPQA]
		{"JRSB,aBAP,SaRQ,JPQA", 2.0  , 4, {21,10,15,20}},		//Ap1[JRSB] += 2.0 W[aBAP] E2[SaRQ] p2[JPQA]
		{"JRSB,IJab,PSQR,ba,IPQB", -1.0  , 5, {21,11,15,33,20}},		//Ap1[JRSB] += -1.0 W[IJab] E2[PSQR] delta[ba] p2[IPQB]
		{"JRSB,IaJb,PSQR,ba,IPQB", 2.0  , 5, {21,11,15,33,20}},		//Ap1[JRSB] += 2.0 W[IaJb] E2[PSQR] delta[ba] p2[IPQB]
		{"JRSB,IPJa,SaRQ,IPQB", 1.0  , 4, {21,5,15,20}},		//Ap1[JRSB] += 1.0 W[IPJa] E2[SaRQ] p2[IPQB]
		{"JRSB,IRJa,PSQa,IPQB", 1.0  , 4, {21,5,15,20}},		//Ap1[JRSB] += 1.0 W[IRJa] E2[PSQa] p2[IPQB]
		{"JRSB,IRaJ,PSaQ,IPQB", 1.0  , 4, {21,6,15,20}},		//Ap1[JRSB] += 1.0 W[IRaJ] E2[PSaQ] p2[IPQB]
		{"JRSB,IaPJ,SaRQ,IPQB", -2.0  , 4, {21,6,15,20}},		//Ap1[JRSB] += -2.0 W[IaPJ] E2[SaRQ] p2[IPQB]
		{"JRSB,aAbB,PSQR,ba,JPQA", -2.0  , 5, {21,7,15,33,20}},		//Ap1[JRSB] += -2.0 W[aAbB] E2[PSQR] delta[ba] p2[JPQA]
		{"JRSB,aBAb,PSQR,ba,JPQA", 1.0  , 5, {21,8,15,33,20}},		//Ap1[JRSB] += 1.0 W[aBAb] E2[PSQR] delta[ba] p2[JPQA]
		{"JRSB,Pa,SaRQ,JPQB", -1.0  , 4, {21,3,15,20}},		//Ap1[JRSB] += -1.0 k[Pa] E2[SaRQ] p2[JPQB]
		{"JRSB,Qa,PSaR,JPQB", 1.0  , 4, {21,3,15,20}},		//Ap1[JRSB] += 1.0 k[Qa] E2[PSaR] p2[JPQB]
		{"JRSB,PRab,SabQ,JPQB", -1.0  , 4, {21,12,15,20}},		//Ap1[JRSB] += -1.0 W[PRab] E2[SabQ] p2[JPQB]
		{"JRSB,PaRb,SaQb,JPQB", -1.0  , 4, {21,12,15,20}},		//Ap1[JRSB] += -1.0 W[PaRb] E2[SaQb] p2[JPQB]
		{"JRSB,QRab,PSab,JPQB", 1.0  , 4, {21,12,15,20}},		//Ap1[JRSB] += 1.0 W[QRab] E2[PSab] p2[JPQB]
		{"JRSB,aAbB,SaQb,JRQA", -1.0  , 4, {21,9,15,20}},		//Ap1[JRSB] += -1.0 W[aAbB] E2[SaQb] p2[JRQA]
		{"JRSB,aBAb,SabQ,JRQA", -1.0  , 4, {21,10,15,20}},		//Ap1[JRSB] += -1.0 W[aBAb] E2[SabQ] p2[JRQA]
		{"JRSB,IaJb,SaQb,IRQB", 1.0  , 4, {21,5,15,20}},		//Ap1[JRSB] += 1.0 W[IaJb] E2[SaQb] p2[IRQB]
		{"JRSB,aPcb,SbRQ,ca,JPQB", -2.0  , 5, {21,5,15,33,20}},		//Ap1[JRSB] += -2.0 W[aPcb] E2[SbRQ] delta[ca] p2[JPQB]
		{"JRSB,aQcb,PSbR,ca,JPQB", 2.0  , 5, {21,5,15,33,20}},		//Ap1[JRSB] += 2.0 W[aQcb] E2[PSbR] delta[ca] p2[JPQB]
		{"JRSB,IabJ,SabQ,IRQB", 1.0  , 4, {21,6,15,20}},		//Ap1[JRSB] += 1.0 W[IabJ] E2[SabQ] p2[IRQB]
		{"JRSB,aQbc,PSbR,ca,JPQB", -1.0  , 5, {21,6,15,33,20}},		//Ap1[JRSB] += -1.0 W[aQbc] E2[PSbR] delta[ca] p2[JPQB]
		{"JRSB,abPc,SbRQ,ca,JPQB", 1.0  , 5, {21,6,15,33,20}},		//Ap1[JRSB] += 1.0 W[abPc] E2[SbRQ] delta[ca] p2[JPQB]
		{"JRSB,Qabc,Sabc,JRQB", 1.0  , 4, {21,12,15,20}},		//Ap1[JRSB] += 1.0 W[Qabc] E2[Sabc] p2[JRQB]
//
		{"JRSB,PARB,SQ,JPQA", 2.0  , 4, {22,9,14,20}},		//Ap2[JRSB] += 2.0 W[PARB] E1[SQ] p2[JPQA]
		{"JRSB,RBAP,SQ,JPQA", -1.0  , 4, {22,10,14,20}},		//Ap2[JRSB] += -1.0 W[RBAP] E1[SQ] p2[JPQA]
		{"JRSB,IPJR,SQ,IPQB", -2.0  , 4, {22,5,14,20}},		//Ap2[JRSB] += -2.0 W[IPJR] E1[SQ] p2[IPQB]
		{"JRSB,IRPJ,SQ,IPQB", 4.0  , 4, {22,6,14,20}},		//Ap2[JRSB] += 4.0 W[IRPJ] E1[SQ] p2[IPQB]
		{"JRSB,IAJB,SQ,IRQA", -2.0  , 4, {22,7,14,20}},		//Ap2[JRSB] += -2.0 W[IAJB] E1[SQ] p2[IRQA]
		{"JRSB,IBAJ,SQ,IRQA", 1.0  , 4, {22,8,14,20}},		//Ap2[JRSB] += 1.0 W[IBAJ] E1[SQ] p2[IRQA]
		{"JRSB,PR,SQ,JPQB", 2.0  , 4, {22,3,14,20}},		//Ap2[JRSB] += 2.0 k[PR] E1[SQ] p2[JPQB]
		{"JRSB,IJ,SQ,IRQB", -2.0  , 4, {22,2,14,20}},		//Ap2[JRSB] += -2.0 k[IJ] E1[SQ] p2[IRQB]
		{"JRSB,AB,SQ,JRQA", 2.0  , 4, {22,4,14,20}},		//Ap2[JRSB] += 2.0 k[AB] E1[SQ] p2[JRQA]
		{"JRSB,IJab,SQ,ba,IRQB", 2.0  , 5, {22,11,14,33,20}},		//Ap2[JRSB] += 2.0 W[IJab] E1[SQ] delta[ba] p2[IRQB]
		{"JRSB,IaJb,SQ,ba,IRQB", -4.0  , 5, {22,11,14,33,20}},		//Ap2[JRSB] += -4.0 W[IaJb] E1[SQ] delta[ba] p2[IRQB]
		{"JRSB,aPbR,SQ,ba,JPQB", 4.0  , 5, {22,5,14,33,20}},		//Ap2[JRSB] += 4.0 W[aPbR] E1[SQ] delta[ba] p2[JPQB]
		{"JRSB,aRPb,SQ,ba,JPQB", -2.0  , 5, {22,6,14,33,20}},		//Ap2[JRSB] += -2.0 W[aRPb] E1[SQ] delta[ba] p2[JPQB]
		{"JRSB,aAbB,SQ,ba,JRQA", 4.0  , 5, {22,7,14,33,20}},		//Ap2[JRSB] += 4.0 W[aAbB] E1[SQ] delta[ba] p2[JRQA]
		{"JRSB,aBAb,SQ,ba,JRQA", -2.0  , 5, {22,8,14,33,20}},		//Ap2[JRSB] += -2.0 W[aBAb] E1[SQ] delta[ba] p2[JRQA]
		{"JRSB,Qa,Sa,JRQB", -2.0  , 4, {22,3,14,20}},		//Ap2[JRSB] += -2.0 k[Qa] E1[Sa] p2[JRQB]
		{"JRSB,aQcb,Sb,ca,JRQB", -4.0  , 5, {22,5,14,33,20}},		//Ap2[JRSB] += -4.0 W[aQcb] E1[Sb] delta[ca] p2[JRQB]
		{"JRSB,aQbc,Sb,ca,JRQB", 2.0  , 5, {22,6,14,33,20}},		//Ap2[JRSB] += 2.0 W[aQbc] E1[Sb] delta[ca] p2[JRQB]
		{"JRSB,IAJB,PSRQ,IPQA", 1.0  , 4, {22,7,15,20}},		//Ap2[JRSB] += 1.0 W[IAJB] E2[PSRQ] p2[IPQA]
		{"JRSB,IBAJ,PSQR,IPQA", 1.0  , 4, {22,8,15,20}},		//Ap2[JRSB] += 1.0 W[IBAJ] E2[PSQR] p2[IPQA]
		{"JRSB,IJ,PSRQ,IPQB", 1.0  , 4, {22,2,15,20}},		//Ap2[JRSB] += 1.0 k[IJ] E2[PSRQ] p2[IPQB]
		{"JRSB,AB,PSRQ,JPQA", -1.0  , 4, {22,4,15,20}},		//Ap2[JRSB] += -1.0 k[AB] E2[PSRQ] p2[JPQA]
		{"JRSB,PAaB,SaQR,JPQA", -1.0  , 4, {22,9,15,20}},		//Ap2[JRSB] += -1.0 W[PAaB] E2[SaQR] p2[JPQA]
		{"JRSB,RAaB,PSaQ,JPQA", -1.0  , 4, {22,9,15,20}},		//Ap2[JRSB] += -1.0 W[RAaB] E2[PSaQ] p2[JPQA]
		{"JRSB,RBAa,PSQa,JPQA", -1.0  , 4, {22,10,15,20}},		//Ap2[JRSB] += -1.0 W[RBAa] E2[PSQa] p2[JPQA]
		{"JRSB,aBAP,SaRQ,JPQA", -1.0  , 4, {22,10,15,20}},		//Ap2[JRSB] += -1.0 W[aBAP] E2[SaRQ] p2[JPQA]
		{"JRSB,IJab,PSRQ,ba,IPQB", -1.0  , 5, {22,11,15,33,20}},		//Ap2[JRSB] += -1.0 W[IJab] E2[PSRQ] delta[ba] p2[IPQB]
		{"JRSB,IaJb,PSRQ,ba,IPQB", 2.0  , 5, {22,11,15,33,20}},		//Ap2[JRSB] += 2.0 W[IaJb] E2[PSRQ] delta[ba] p2[IPQB]
		{"JRSB,IPJa,SaQR,IPQB", 1.0  , 4, {22,5,15,20}},		//Ap2[JRSB] += 1.0 W[IPJa] E2[SaQR] p2[IPQB]
		{"JRSB,IRJa,PSaQ,IPQB", 1.0  , 4, {22,5,15,20}},		//Ap2[JRSB] += 1.0 W[IRJa] E2[PSaQ] p2[IPQB]
		{"JRSB,IRaJ,PSaQ,IPQB", -2.0  , 4, {22,6,15,20}},		//Ap2[JRSB] += -2.0 W[IRaJ] E2[PSaQ] p2[IPQB]
		{"JRSB,IaPJ,SaQR,IPQB", -2.0  , 4, {22,6,15,20}},		//Ap2[JRSB] += -2.0 W[IaPJ] E2[SaQR] p2[IPQB]
		{"JRSB,aAbB,PSRQ,ba,JPQA", -2.0  , 5, {22,7,15,33,20}},		//Ap2[JRSB] += -2.0 W[aAbB] E2[PSRQ] delta[ba] p2[JPQA]
		{"JRSB,aBAb,PSRQ,ba,JPQA", 1.0  , 5, {22,8,15,33,20}},		//Ap2[JRSB] += 1.0 W[aBAb] E2[PSRQ] delta[ba] p2[JPQA]
		{"JRSB,Pa,SaQR,JPQB", -1.0  , 4, {22,3,15,20}},		//Ap2[JRSB] += -1.0 k[Pa] E2[SaQR] p2[JPQB]
		{"JRSB,Qa,PSRa,JPQB", 1.0  , 4, {22,3,15,20}},		//Ap2[JRSB] += 1.0 k[Qa] E2[PSRa] p2[JPQB]
		{"JRSB,PRab,SaQb,JPQB", -1.0  , 4, {22,12,15,20}},		//Ap2[JRSB] += -1.0 W[PRab] E2[SaQb] p2[JPQB]
		{"JRSB,PaRb,SaQb,JPQB", 2.0  , 4, {22,12,15,20}},		//Ap2[JRSB] += 2.0 W[PaRb] E2[SaQb] p2[JPQB]
		{"JRSB,QRab,PSba,JPQB", 1.0  , 4, {22,12,15,20}},		//Ap2[JRSB] += 1.0 W[QRab] E2[PSba] p2[JPQB]
		{"JRSB,aAbB,SaQb,JRQA", 2.0  , 4, {22,9,15,20}},		//Ap2[JRSB] += 2.0 W[aAbB] E2[SaQb] p2[JRQA]
		{"JRSB,aBAb,SabQ,JRQA", 2.0  , 4, {22,10,15,20}},		//Ap2[JRSB] += 2.0 W[aBAb] E2[SabQ] p2[JRQA]
		{"JRSB,IaJb,SaQb,IRQB", -2.0  , 4, {22,5,15,20}},		//Ap2[JRSB] += -2.0 W[IaJb] E2[SaQb] p2[IRQB]
		{"JRSB,aPcb,SbQR,ca,JPQB", -2.0  , 5, {22,5,15,33,20}},		//Ap2[JRSB] += -2.0 W[aPcb] E2[SbQR] delta[ca] p2[JPQB]
		{"JRSB,aQcb,PSRb,ca,JPQB", 2.0  , 5, {22,5,15,33,20}},		//Ap2[JRSB] += 2.0 W[aQcb] E2[PSRb] delta[ca] p2[JPQB]
		{"JRSB,IabJ,SaQb,IRQB", 1.0  , 4, {22,6,15,20}},		//Ap2[JRSB] += 1.0 W[IabJ] E2[SaQb] p2[IRQB]
		{"JRSB,aQbc,PSRb,ca,JPQB", -1.0  , 5, {22,6,15,33,20}},		//Ap2[JRSB] += -1.0 W[aQbc] E2[PSRb] delta[ca] p2[JPQB]
		{"JRSB,abPc,SbQR,ca,JPQB", 1.0  , 5, {22,6,15,33,20}},		//Ap2[JRSB] += 1.0 W[abPc] E2[SbQR] delta[ca] p2[JPQB]
		{"JRSB,Qabc,Sabc,JRQB", -2.0  , 4, {22,12,15,20}},		//Ap2[JRSB] += -2.0 W[Qabc] E2[Sabc] p2[JRQB]

	};
	int f(int i) {
		return 2*i;
	}
	FDomainDecl DomainDecls[1] = {
		{"A", "a", f}
	};
	FEqInfo Overlap[8] = {
		{"IPSA,QS,APIQ",   2.0, 3, {23, 14, 31}},
		{"IRSA,PSQR,APIQ", 2.0, 3, {23, 15, 31}},
		{"IPSA,QS,APIQ",  -1.0, 3, {24, 14, 31}},
		{"IRSA,PSQR,APIQ",-1.0, 3, {24, 15, 31}},
		{"IPSA,QS,PAIQ",  -1.0, 3, {23, 14, 32}},
		{"IRSA,PSQR,PAIQ",-1.0, 3, {23, 15, 32}},
		{"IPSA,QS,PAIQ",   2.0, 3, {24, 14, 32}},
		{"IRSA,PSRQ,PAIQ",-1.0, 3, {24, 15, 32}},
	};
	static void GetMethodInfo(FMethodInfo &Out) {
		Out = FMethodInfo();
		Out.pName = "MRLCC_CAAV";
		Out.perturberClass = "CAAV";
		Out.pSpinClass = "restricted";
		Out.pTensorDecls = &TensorDecls[0];
		Out.nTensorDecls = 47;
		Out.pDomainDecls = &DomainDecls[0];
		Out.nDomainDecls = 1;
		Out.EqsHandCode = FEqSet(&EqsHandCode[0], 16, "MRLCC_CAAV/Res");
		Out.EqsHandCode2 = FEqSet(&EqsHandCode2[0], 8, "MRLCC_CAAV/Res");
		Out.EqsRes = FEqSet(&EqsRes[0], 192, "MRLCC_CAAV/Res");
		Out.Overlap = FEqSet(&Overlap[0], 8, "MRLCC_CAAV/Overlap");
	};
};
