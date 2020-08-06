/*                                                                                                                                                                          
Developed by Sandeep Sharma and Gerald Knizia, 2016                                                                                                                     
Copyright (c) 2016, Sandeep Sharma
*/
namespace MRLCC_CCAV {

	FTensorDecl TensorDecls[30] = {
		/*  0*/{"t", "ccae", "",USAGE_Amplitude, STORAGE_Memory},
		/*  1*/{"R", "ccae", "",USAGE_Residual, STORAGE_Memory},
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

//Number of terms :  134
	FEqInfo EqsRes[134] = {

		{"KLQB,PAQB,KLPA", 4.0  , 3, {22,9,21}},		//Ap[KLQB] += 4.0 W[PAQB] p[KLPA]
		{"KLQB,PAQB,LKPA", -2.0  , 3, {22,9,21}},		//Ap[KLQB] += -2.0 W[PAQB] p[LKPA]
		{"KLQB,QBAP,KLPA", -2.0  , 3, {22,10,21}},		//Ap[KLQB] += -2.0 W[QBAP] p[KLPA]
		{"KLQB,QBAP,LKPA", 4.0  , 3, {22,10,21}},		//Ap[KLQB] += 4.0 W[QBAP] p[LKPA]
		{"KLQB,IJKL,IJQB", 4.0  , 3, {22,11,21}},		//Ap[KLQB] += 4.0 W[IJKL] p[IJQB]
		{"KLQB,IJLK,IJQB", -2.0  , 3, {22,11,21}},		//Ap[KLQB] += -2.0 W[IJLK] p[IJQB]
		{"KLQB,IPKQ,ILPB", -4.0  , 3, {22,5,21}},		//Ap[KLQB] += -4.0 W[IPKQ] p[ILPB]
		{"KLQB,IPLQ,IKPB", 2.0  , 3, {22,5,21}},		//Ap[KLQB] += 2.0 W[IPLQ] p[IKPB]
		{"KLQB,JPKQ,LJPB", 2.0  , 3, {22,5,21}},		//Ap[KLQB] += 2.0 W[JPKQ] p[LJPB]
		{"KLQB,JPLQ,KJPB", -4.0  , 3, {22,5,21}},		//Ap[KLQB] += -4.0 W[JPLQ] p[KJPB]
		{"KLQB,IQPK,ILPB", 8.0  , 3, {22,6,21}},		//Ap[KLQB] += 8.0 W[IQPK] p[ILPB]
		{"KLQB,IQPL,IKPB", -4.0  , 3, {22,6,21}},		//Ap[KLQB] += -4.0 W[IQPL] p[IKPB]
		{"KLQB,JQPK,LJPB", -4.0  , 3, {22,6,21}},		//Ap[KLQB] += -4.0 W[JQPK] p[LJPB]
		{"KLQB,JQPL,KJPB", 2.0  , 3, {22,6,21}},		//Ap[KLQB] += 2.0 W[JQPL] p[KJPB]
		{"KLQB,IAKB,ILQA", -4.0  , 3, {22,7,21}},		//Ap[KLQB] += -4.0 W[IAKB] p[ILQA]
		{"KLQB,IALB,IKQA", 2.0  , 3, {22,7,21}},		//Ap[KLQB] += 2.0 W[IALB] p[IKQA]
		{"KLQB,JAKB,LJQA", 2.0  , 3, {22,7,21}},		//Ap[KLQB] += 2.0 W[JAKB] p[LJQA]
		{"KLQB,JALB,KJQA", -4.0  , 3, {22,7,21}},		//Ap[KLQB] += -4.0 W[JALB] p[KJQA]
		{"KLQB,IBAK,ILQA", 2.0  , 3, {22,8,21}},		//Ap[KLQB] += 2.0 W[IBAK] p[ILQA]
		{"KLQB,IBAL,IKQA", -4.0  , 3, {22,8,21}},		//Ap[KLQB] += -4.0 W[IBAL] p[IKQA]
		{"KLQB,JBAK,LJQA", -4.0  , 3, {22,8,21}},		//Ap[KLQB] += -4.0 W[JBAK] p[LJQA]
		{"KLQB,JBAL,KJQA", 8.0  , 3, {22,8,21}},		//Ap[KLQB] += 8.0 W[JBAL] p[KJQA]
		{"KLQB,PQ,KLPB", 4.0  , 3, {22,3,21}},		//Ap[KLQB] += 4.0 k[PQ] p[KLPB]
		{"KLQB,PQ,LKPB", -2.0  , 3, {22,3,21}},		//Ap[KLQB] += -2.0 k[PQ] p[LKPB]
		{"KLQB,IK,ILQB", -4.0  , 3, {22,2,21}},		//Ap[KLQB] += -4.0 k[IK] p[ILQB]
		{"KLQB,IL,IKQB", 2.0  , 3, {22,2,21}},		//Ap[KLQB] += 2.0 k[IL] p[IKQB]
		{"KLQB,JK,LJQB", 2.0  , 3, {22,2,21}},		//Ap[KLQB] += 2.0 k[JK] p[LJQB]
		{"KLQB,JL,KJQB", -4.0  , 3, {22,2,21}},		//Ap[KLQB] += -4.0 k[JL] p[KJQB]
		{"KLQB,AB,KLQA", 4.0  , 3, {22,4,21}},		//Ap[KLQB] += 4.0 k[AB] p[KLQA]
		{"KLQB,AB,LKQA", -2.0  , 3, {22,4,21}},		//Ap[KLQB] += -2.0 k[AB] p[LKQA]
		{"KLQB,IKab,ba,ILQB", 4.0  , 4, {22,11,27,21}},		//Ap[KLQB] += 4.0 W[IKab] delta[ba] p[ILQB]
		{"KLQB,ILab,ba,IKQB", -2.0  , 4, {22,11,27,21}},		//Ap[KLQB] += -2.0 W[ILab] delta[ba] p[IKQB]
		{"KLQB,IaKb,ba,ILQB", -8.0  , 4, {22,11,27,21}},		//Ap[KLQB] += -8.0 W[IaKb] delta[ba] p[ILQB]
		{"KLQB,IaLb,ba,IKQB", 4.0  , 4, {22,11,27,21}},		//Ap[KLQB] += 4.0 W[IaLb] delta[ba] p[IKQB]
		{"KLQB,JKab,ba,LJQB", -2.0  , 4, {22,11,27,21}},		//Ap[KLQB] += -2.0 W[JKab] delta[ba] p[LJQB]
		{"KLQB,JLab,ba,KJQB", 4.0  , 4, {22,11,27,21}},		//Ap[KLQB] += 4.0 W[JLab] delta[ba] p[KJQB]
		{"KLQB,JaKb,ba,LJQB", 4.0  , 4, {22,11,27,21}},		//Ap[KLQB] += 4.0 W[JaKb] delta[ba] p[LJQB]
		{"KLQB,JaLb,ba,KJQB", -8.0  , 4, {22,11,27,21}},		//Ap[KLQB] += -8.0 W[JaLb] delta[ba] p[KJQB]
		{"KLQB,aPbQ,ba,KLPB", 8.0  , 4, {22,5,27,21}},		//Ap[KLQB] += 8.0 W[aPbQ] delta[ba] p[KLPB]
		{"KLQB,aPbQ,ba,LKPB", -4.0  , 4, {22,5,27,21}},		//Ap[KLQB] += -4.0 W[aPbQ] delta[ba] p[LKPB]
		{"KLQB,aQPb,ba,KLPB", -4.0  , 4, {22,6,27,21}},		//Ap[KLQB] += -4.0 W[aQPb] delta[ba] p[KLPB]
		{"KLQB,aQPb,ba,LKPB", 2.0  , 4, {22,6,27,21}},		//Ap[KLQB] += 2.0 W[aQPb] delta[ba] p[LKPB]
		{"KLQB,aAbB,ba,KLQA", 8.0  , 4, {22,7,27,21}},		//Ap[KLQB] += 8.0 W[aAbB] delta[ba] p[KLQA]
		{"KLQB,aAbB,ba,LKQA", -4.0  , 4, {22,7,27,21}},		//Ap[KLQB] += -4.0 W[aAbB] delta[ba] p[LKQA]
		{"KLQB,aBAb,ba,KLQA", -4.0  , 4, {22,8,27,21}},		//Ap[KLQB] += -4.0 W[aBAb] delta[ba] p[KLQA]
		{"KLQB,aBAb,ba,LKQA", 2.0  , 4, {22,8,27,21}},		//Ap[KLQB] += 2.0 W[aBAb] delta[ba] p[LKQA]
		{"KLQB,IJKL,PQ,IJPB", -2.0  , 4, {22,11,14,21}},		//Ap[KLQB] += -2.0 W[IJKL] E1[PQ] p[IJPB]
		{"KLQB,IJLK,PQ,IJPB", 1.0  , 4, {22,11,14,21}},		//Ap[KLQB] += 1.0 W[IJLK] E1[PQ] p[IJPB]
		{"KLQB,IAKB,PQ,ILPA", 2.0  , 4, {22,7,14,21}},		//Ap[KLQB] += 2.0 W[IAKB] E1[PQ] p[ILPA]
		{"KLQB,IALB,PQ,IKPA", -1.0  , 4, {22,7,14,21}},		//Ap[KLQB] += -1.0 W[IALB] E1[PQ] p[IKPA]
		{"KLQB,JAKB,PQ,LJPA", -1.0  , 4, {22,7,14,21}},		//Ap[KLQB] += -1.0 W[JAKB] E1[PQ] p[LJPA]
		{"KLQB,JALB,PQ,KJPA", 2.0  , 4, {22,7,14,21}},		//Ap[KLQB] += 2.0 W[JALB] E1[PQ] p[KJPA]
		{"KLQB,IBAK,PQ,ILPA", -1.0  , 4, {22,8,14,21}},		//Ap[KLQB] += -1.0 W[IBAK] E1[PQ] p[ILPA]
		{"KLQB,IBAL,PQ,IKPA", 2.0  , 4, {22,8,14,21}},		//Ap[KLQB] += 2.0 W[IBAL] E1[PQ] p[IKPA]
		{"KLQB,JBAK,PQ,LJPA", 2.0  , 4, {22,8,14,21}},		//Ap[KLQB] += 2.0 W[JBAK] E1[PQ] p[LJPA]
		{"KLQB,JBAL,PQ,KJPA", -4.0  , 4, {22,8,14,21}},		//Ap[KLQB] += -4.0 W[JBAL] E1[PQ] p[KJPA]
		{"KLQB,IK,PQ,ILPB", 2.0  , 4, {22,2,14,21}},		//Ap[KLQB] += 2.0 k[IK] E1[PQ] p[ILPB]
		{"KLQB,IL,PQ,IKPB", -1.0  , 4, {22,2,14,21}},		//Ap[KLQB] += -1.0 k[IL] E1[PQ] p[IKPB]
		{"KLQB,JK,PQ,LJPB", -1.0  , 4, {22,2,14,21}},		//Ap[KLQB] += -1.0 k[JK] E1[PQ] p[LJPB]
		{"KLQB,JL,PQ,KJPB", 2.0  , 4, {22,2,14,21}},		//Ap[KLQB] += 2.0 k[JL] E1[PQ] p[KJPB]
		{"KLQB,AB,PQ,KLPA", -2.0  , 4, {22,4,14,21}},		//Ap[KLQB] += -2.0 k[AB] E1[PQ] p[KLPA]
		{"KLQB,AB,PQ,LKPA", 1.0  , 4, {22,4,14,21}},		//Ap[KLQB] += 1.0 k[AB] E1[PQ] p[LKPA]
		{"KLQB,PAaB,aQ,KLPA", -2.0  , 4, {22,9,14,21}},		//Ap[KLQB] += -2.0 W[PAaB] E1[aQ] p[KLPA]
		{"KLQB,PAaB,aQ,LKPA", 1.0  , 4, {22,9,14,21}},		//Ap[KLQB] += 1.0 W[PAaB] E1[aQ] p[LKPA]
		{"KLQB,QAaB,Pa,KLPA", -2.0  , 4, {22,9,14,21}},		//Ap[KLQB] += -2.0 W[QAaB] E1[Pa] p[KLPA]
		{"KLQB,QAaB,Pa,LKPA", 1.0  , 4, {22,9,14,21}},		//Ap[KLQB] += 1.0 W[QAaB] E1[Pa] p[LKPA]
		{"KLQB,QBAa,Pa,KLPA", 1.0  , 4, {22,10,14,21}},		//Ap[KLQB] += 1.0 W[QBAa] E1[Pa] p[KLPA]
		{"KLQB,QBAa,Pa,LKPA", -2.0  , 4, {22,10,14,21}},		//Ap[KLQB] += -2.0 W[QBAa] E1[Pa] p[LKPA]
		{"KLQB,aBAP,aQ,KLPA", 1.0  , 4, {22,10,14,21}},		//Ap[KLQB] += 1.0 W[aBAP] E1[aQ] p[KLPA]
		{"KLQB,aBAP,aQ,LKPA", -2.0  , 4, {22,10,14,21}},		//Ap[KLQB] += -2.0 W[aBAP] E1[aQ] p[LKPA]
		{"KLQB,IKab,PQ,ba,ILPB", -2.0  , 5, {22,11,14,27,21}},		//Ap[KLQB] += -2.0 W[IKab] E1[PQ] delta[ba] p[ILPB]
		{"KLQB,ILab,PQ,ba,IKPB", 1.0  , 5, {22,11,14,27,21}},		//Ap[KLQB] += 1.0 W[ILab] E1[PQ] delta[ba] p[IKPB]
		{"KLQB,IaKb,PQ,ba,ILPB", 4.0  , 5, {22,11,14,27,21}},		//Ap[KLQB] += 4.0 W[IaKb] E1[PQ] delta[ba] p[ILPB]
		{"KLQB,IaLb,PQ,ba,IKPB", -2.0  , 5, {22,11,14,27,21}},		//Ap[KLQB] += -2.0 W[IaLb] E1[PQ] delta[ba] p[IKPB]
		{"KLQB,JKab,PQ,ba,LJPB", 1.0  , 5, {22,11,14,27,21}},		//Ap[KLQB] += 1.0 W[JKab] E1[PQ] delta[ba] p[LJPB]
		{"KLQB,JLab,PQ,ba,KJPB", -2.0  , 5, {22,11,14,27,21}},		//Ap[KLQB] += -2.0 W[JLab] E1[PQ] delta[ba] p[KJPB]
		{"KLQB,JaKb,PQ,ba,LJPB", -2.0  , 5, {22,11,14,27,21}},		//Ap[KLQB] += -2.0 W[JaKb] E1[PQ] delta[ba] p[LJPB]
		{"KLQB,JaLb,PQ,ba,KJPB", 4.0  , 5, {22,11,14,27,21}},		//Ap[KLQB] += 4.0 W[JaLb] E1[PQ] delta[ba] p[KJPB]
		{"KLQB,IPKa,aQ,ILPB", 2.0  , 4, {22,5,14,21}},		//Ap[KLQB] += 2.0 W[IPKa] E1[aQ] p[ILPB]
		{"KLQB,IPLa,aQ,IKPB", -1.0  , 4, {22,5,14,21}},		//Ap[KLQB] += -1.0 W[IPLa] E1[aQ] p[IKPB]
		{"KLQB,IQKa,Pa,ILPB", 2.0  , 4, {22,5,14,21}},		//Ap[KLQB] += 2.0 W[IQKa] E1[Pa] p[ILPB]
		{"KLQB,IQLa,Pa,IKPB", -1.0  , 4, {22,5,14,21}},		//Ap[KLQB] += -1.0 W[IQLa] E1[Pa] p[IKPB]
		{"KLQB,JPKa,aQ,LJPB", -1.0  , 4, {22,5,14,21}},		//Ap[KLQB] += -1.0 W[JPKa] E1[aQ] p[LJPB]
		{"KLQB,JPLa,aQ,KJPB", 2.0  , 4, {22,5,14,21}},		//Ap[KLQB] += 2.0 W[JPLa] E1[aQ] p[KJPB]
		{"KLQB,JQKa,Pa,LJPB", -1.0  , 4, {22,5,14,21}},		//Ap[KLQB] += -1.0 W[JQKa] E1[Pa] p[LJPB]
		{"KLQB,JQLa,Pa,KJPB", 2.0  , 4, {22,5,14,21}},		//Ap[KLQB] += 2.0 W[JQLa] E1[Pa] p[KJPB]
		{"KLQB,IQaK,Pa,ILPB", -4.0  , 4, {22,6,14,21}},		//Ap[KLQB] += -4.0 W[IQaK] E1[Pa] p[ILPB]
		{"KLQB,IQaL,Pa,IKPB", 2.0  , 4, {22,6,14,21}},		//Ap[KLQB] += 2.0 W[IQaL] E1[Pa] p[IKPB]
		{"KLQB,IaPK,aQ,ILPB", -4.0  , 4, {22,6,14,21}},		//Ap[KLQB] += -4.0 W[IaPK] E1[aQ] p[ILPB]
		{"KLQB,IaPL,aQ,IKPB", 2.0  , 4, {22,6,14,21}},		//Ap[KLQB] += 2.0 W[IaPL] E1[aQ] p[IKPB]
		{"KLQB,JQaK,Pa,LJPB", 2.0  , 4, {22,6,14,21}},		//Ap[KLQB] += 2.0 W[JQaK] E1[Pa] p[LJPB]
		{"KLQB,JQaL,Pa,KJPB", -1.0  , 4, {22,6,14,21}},		//Ap[KLQB] += -1.0 W[JQaL] E1[Pa] p[KJPB]
		{"KLQB,JaPK,aQ,LJPB", 2.0  , 4, {22,6,14,21}},		//Ap[KLQB] += 2.0 W[JaPK] E1[aQ] p[LJPB]
		{"KLQB,JaPL,aQ,KJPB", -1.0  , 4, {22,6,14,21}},		//Ap[KLQB] += -1.0 W[JaPL] E1[aQ] p[KJPB]
		{"KLQB,aAbB,PQ,ba,KLPA", -4.0  , 5, {22,7,14,27,21}},		//Ap[KLQB] += -4.0 W[aAbB] E1[PQ] delta[ba] p[KLPA]
		{"KLQB,aAbB,PQ,ba,LKPA", 2.0  , 5, {22,7,14,27,21}},		//Ap[KLQB] += 2.0 W[aAbB] E1[PQ] delta[ba] p[LKPA]
		{"KLQB,aBAb,PQ,ba,KLPA", 2.0  , 5, {22,8,14,27,21}},		//Ap[KLQB] += 2.0 W[aBAb] E1[PQ] delta[ba] p[KLPA]
		{"KLQB,aBAb,PQ,ba,LKPA", -1.0  , 5, {22,8,14,27,21}},		//Ap[KLQB] += -1.0 W[aBAb] E1[PQ] delta[ba] p[LKPA]
		{"KLQB,Pa,aQ,KLPB", -2.0  , 4, {22,3,14,21}},		//Ap[KLQB] += -2.0 k[Pa] E1[aQ] p[KLPB]
		{"KLQB,Pa,aQ,LKPB", 1.0  , 4, {22,3,14,21}},		//Ap[KLQB] += 1.0 k[Pa] E1[aQ] p[LKPB]
		{"KLQB,PQab,ab,KLPB", -2.0  , 4, {22,12,14,21}},		//Ap[KLQB] += -2.0 W[PQab] E1[ab] p[KLPB]
		{"KLQB,PQab,ab,LKPB", 1.0  , 4, {22,12,14,21}},		//Ap[KLQB] += 1.0 W[PQab] E1[ab] p[LKPB]
		{"KLQB,PaQb,ab,KLPB", 4.0  , 4, {22,12,14,21}},		//Ap[KLQB] += 4.0 W[PaQb] E1[ab] p[KLPB]
		{"KLQB,PaQb,ab,LKPB", -2.0  , 4, {22,12,14,21}},		//Ap[KLQB] += -2.0 W[PaQb] E1[ab] p[LKPB]
		{"KLQB,aAbB,ab,KLQA", 4.0  , 4, {22,9,14,21}},		//Ap[KLQB] += 4.0 W[aAbB] E1[ab] p[KLQA]
		{"KLQB,aAbB,ab,LKQA", -2.0  , 4, {22,9,14,21}},		//Ap[KLQB] += -2.0 W[aAbB] E1[ab] p[LKQA]
		{"KLQB,aBAb,ab,KLQA", -2.0  , 4, {22,10,14,21}},		//Ap[KLQB] += -2.0 W[aBAb] E1[ab] p[KLQA]
		{"KLQB,aBAb,ab,LKQA", 1.0  , 4, {22,10,14,21}},		//Ap[KLQB] += 1.0 W[aBAb] E1[ab] p[LKQA]
		{"KLQB,IaKb,ab,ILQB", -4.0  , 4, {22,5,14,21}},		//Ap[KLQB] += -4.0 W[IaKb] E1[ab] p[ILQB]
		{"KLQB,IaLb,ab,IKQB", 2.0  , 4, {22,5,14,21}},		//Ap[KLQB] += 2.0 W[IaLb] E1[ab] p[IKQB]
		{"KLQB,JaKb,ab,LJQB", 2.0  , 4, {22,5,14,21}},		//Ap[KLQB] += 2.0 W[JaKb] E1[ab] p[LJQB]
		{"KLQB,JaLb,ab,KJQB", -4.0  , 4, {22,5,14,21}},		//Ap[KLQB] += -4.0 W[JaLb] E1[ab] p[KJQB]
		{"KLQB,aPcb,bQ,ca,KLPB", -4.0  , 5, {22,5,14,27,21}},		//Ap[KLQB] += -4.0 W[aPcb] E1[bQ] delta[ca] p[KLPB]
		{"KLQB,aPcb,bQ,ca,LKPB", 2.0  , 5, {22,5,14,27,21}},		//Ap[KLQB] += 2.0 W[aPcb] E1[bQ] delta[ca] p[LKPB]
		{"KLQB,IabK,ab,ILQB", 2.0  , 4, {22,6,14,21}},		//Ap[KLQB] += 2.0 W[IabK] E1[ab] p[ILQB]
		{"KLQB,IabL,ab,IKQB", -1.0  , 4, {22,6,14,21}},		//Ap[KLQB] += -1.0 W[IabL] E1[ab] p[IKQB]
		{"KLQB,JabK,ab,LJQB", -1.0  , 4, {22,6,14,21}},		//Ap[KLQB] += -1.0 W[JabK] E1[ab] p[LJQB]
		{"KLQB,JabL,ab,KJQB", 2.0  , 4, {22,6,14,21}},		//Ap[KLQB] += 2.0 W[JabL] E1[ab] p[KJQB]
		{"KLQB,abPc,bQ,ca,KLPB", 2.0  , 5, {22,6,14,27,21}},		//Ap[KLQB] += 2.0 W[abPc] E1[bQ] delta[ca] p[KLPB]
		{"KLQB,abPc,bQ,ca,LKPB", -1.0  , 5, {22,6,14,27,21}},		//Ap[KLQB] += -1.0 W[abPc] E1[bQ] delta[ca] p[LKPB]
		{"KLQB,aAbB,PaQb,KLPA", -2.0  , 4, {22,9,15,21}},		//Ap[KLQB] += -2.0 W[aAbB] E2[PaQb] p[KLPA]
		{"KLQB,aAbB,PaQb,LKPA", 1.0  , 4, {22,9,15,21}},		//Ap[KLQB] += 1.0 W[aAbB] E2[PaQb] p[LKPA]
		{"KLQB,aBAb,PaQb,KLPA", 1.0  , 4, {22,10,15,21}},		//Ap[KLQB] += 1.0 W[aBAb] E2[PaQb] p[KLPA]
		{"KLQB,aBAb,PabQ,LKPA", 1.0  , 4, {22,10,15,21}},		//Ap[KLQB] += 1.0 W[aBAb] E2[PabQ] p[LKPA]
		{"KLQB,IaKb,PaQb,ILPB", 2.0  , 4, {22,5,15,21}},		//Ap[KLQB] += 2.0 W[IaKb] E2[PaQb] p[ILPB]
		{"KLQB,IaLb,PaQb,IKPB", -1.0  , 4, {22,5,15,21}},		//Ap[KLQB] += -1.0 W[IaLb] E2[PaQb] p[IKPB]
		{"KLQB,JaKb,PaQb,LJPB", -1.0  , 4, {22,5,15,21}},		//Ap[KLQB] += -1.0 W[JaKb] E2[PaQb] p[LJPB]
		{"KLQB,JaLb,PaQb,KJPB", 2.0  , 4, {22,5,15,21}},		//Ap[KLQB] += 2.0 W[JaLb] E2[PaQb] p[KJPB]
		{"KLQB,IabK,PabQ,ILPB", 2.0  , 4, {22,6,15,21}},		//Ap[KLQB] += 2.0 W[IabK] E2[PabQ] p[ILPB]
		{"KLQB,IabL,PabQ,IKPB", -1.0  , 4, {22,6,15,21}},		//Ap[KLQB] += -1.0 W[IabL] E2[PabQ] p[IKPB]
		{"KLQB,JabK,PabQ,LJPB", -1.0  , 4, {22,6,15,21}},		//Ap[KLQB] += -1.0 W[JabK] E2[PabQ] p[LJPB]
		{"KLQB,JabL,PaQb,KJPB", -1.0  , 4, {22,6,15,21}},		//Ap[KLQB] += -1.0 W[JabL] E2[PaQb] p[KJPB]
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
		{"IJPA,RP,JIRA", 1.0, 3, {20, 14, 26}},
	};
	static void GetMethodInfo(FMethodInfo &Out) {
		Out = FMethodInfo();
		Out.pName = "MRLCC_CCAV";
		Out.perturberClass = "CCAV";
		Out.pSpinClass = "restricted";
		Out.pTensorDecls = &TensorDecls[0];
		Out.nTensorDecls = 30;
		Out.pDomainDecls = &DomainDecls[0];
		Out.nDomainDecls = 1;
		Out.EqsRes = FEqSet(&EqsRes[0], 134, "MRLCC_CCAV/Res");
		Out.Overlap = FEqSet(&Overlap[0], 4, "MRLCC_CCAV/Overlap");
	};
};
