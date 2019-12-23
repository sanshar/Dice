/*                                                                                                                                                                          
Developed by Sandeep Sharma and Gerald Knizia, 2016                                                                                                                     
Copyright (c) 2016, Sandeep Sharma
*/
namespace NEVPT2_CCAA {

	FTensorDecl TensorDecls[32] = {
		/*  0*/{"t", "ccaa", "",USAGE_Amplitude, STORAGE_Memory},
		/*  1*/{"R", "ccaa", "",USAGE_Residual, STORAGE_Memory},
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
		/* 16*/{"S1", "aaaa", "",USAGE_Density, STORAGE_Memory},
		/* 17*/{"S2", "aaaa", "",USAGE_Density, STORAGE_Memory},
		/* 18*/{"T", "ccaa", "",USAGE_Amplitude, STORAGE_Memory},
		/* 19*/{"b", "ccaa", "",USAGE_Amplitude, STORAGE_Memory},
		/* 20*/{"p", "ccaa", "",USAGE_Amplitude, STORAGE_Memory},
		/* 21*/{"Ap", "ccaa", "",USAGE_Amplitude, STORAGE_Memory},
		/* 22*/{"P", "ccaa", "",USAGE_Amplitude, STORAGE_Memory},
		/* 23*/{"AP", "ccaa", "",USAGE_Amplitude, STORAGE_Memory},
		/* 24*/{"B", "ccaa", "",USAGE_Amplitude, STORAGE_Memory},
		/* 25*/{"W", "ccaa", "",USAGE_Hamiltonian, STORAGE_Memory},
		/* 26*/{"delta", "cc", "",USAGE_Density, STORAGE_Memory},
		/* 27*/{"delta", "aa", "",USAGE_Density, STORAGE_Memory},
		/* 28*/{"delta", "ee", "",USAGE_Density, STORAGE_Memory},
		/* 29*/{"Ap12a", "cc", "", USAGE_PlaceHolder, STORAGE_Memory},
		/* 30*/{"Ap12b", "cc", "", USAGE_PlaceHolder, STORAGE_Memory},
		/* 31*/{"E312", "aaaa", "", USAGE_PlaceHolder, STORAGE_Memory}, //SR
	};

//Number of terms :  122
	FEqInfo EqsRes[122] = {

		{"KLRS,PQRS,KLPQ", 4.0  , 3, {21,12,20}},		//Ap[KLRS] += 4.0 W[PQRS] p[KLPQ]
		{"KLRS,PQRS,LKPQ", -2.0  , 3, {21,12,20}},		//Ap[KLRS] += -2.0 W[PQRS] p[LKPQ]
		{"KLRS,PQSR,KLPQ", -2.0  , 3, {21,12,20}},		//Ap[KLRS] += -2.0 W[PQSR] p[KLPQ]
		{"KLRS,PQSR,LKPQ", 4.0  , 3, {21,12,20}},		//Ap[KLRS] += 4.0 W[PQSR] p[LKPQ]
		{"KLRS,PR,KLPS", 4.0  , 3, {21,3,20}},		//Ap[KLRS] += 4.0 f[PR] p[KLPS]
		{"KLRS,PR,LKPS", -2.0  , 3, {21,3,20}},		//Ap[KLRS] += -2.0 f[PR] p[LKPS]
		{"KLRS,PS,KLPR", -2.0  , 3, {21,3,20}},		//Ap[KLRS] += -2.0 f[PS] p[KLPR]
		{"KLRS,PS,LKPR", 4.0  , 3, {21,3,20}},		//Ap[KLRS] += 4.0 f[PS] p[LKPR]
		{"KLRS,QR,KLSQ", -2.0  , 3, {21,3,20}},		//Ap[KLRS] += -2.0 f[QR] p[KLSQ]
		{"KLRS,QR,LKSQ", 4.0  , 3, {21,3,20}},		//Ap[KLRS] += 4.0 f[QR] p[LKSQ]
		{"KLRS,QS,KLRQ", 4.0  , 3, {21,3,20}},		//Ap[KLRS] += 4.0 f[QS] p[KLRQ]
		{"KLRS,QS,LKRQ", -2.0  , 3, {21,3,20}},		//Ap[KLRS] += -2.0 f[QS] p[LKRQ]
		{"KLRS,IK,ILRS", -4.0  , 3, {21,2,20}},		//Ap[KLRS] += -4.0 f[IK] p[ILRS]
		{"KLRS,IK,ILSR", 2.0  , 3, {21,2,20}},		//Ap[KLRS] += 2.0 f[IK] p[ILSR]
		{"KLRS,IL,IKRS", 2.0  , 3, {21,2,20}},		//Ap[KLRS] += 2.0 f[IL] p[IKRS]
		{"KLRS,IL,IKSR", -4.0  , 3, {21,2,20}},		//Ap[KLRS] += -4.0 f[IL] p[IKSR]
		{"KLRS,JK,LJRS", 2.0  , 3, {21,2,20}},		//Ap[KLRS] += 2.0 f[JK] p[LJRS]
		{"KLRS,JK,LJSR", -4.0  , 3, {21,2,20}},		//Ap[KLRS] += -4.0 f[JK] p[LJSR]
		{"KLRS,JL,KJRS", -4.0  , 3, {21,2,20}},		//Ap[KLRS] += -4.0 f[JL] p[KJRS]
		{"KLRS,JL,KJSR", 2.0  , 3, {21,2,20}},		//Ap[KLRS] += 2.0 f[JL] p[KJSR]
		{"KLRS,PR,QS,KLPQ", -2.0  , 4, {21,3,13,20}},		//Ap[KLRS] += -2.0 f[PR] E1[QS] p[KLPQ]
		{"KLRS,PR,QS,LKPQ", 1.0  , 4, {21,3,13,20}},		//Ap[KLRS] += 1.0 f[PR] E1[QS] p[LKPQ]
		{"KLRS,PS,QR,KLPQ", 1.0  , 4, {21,3,13,20}},		//Ap[KLRS] += 1.0 f[PS] E1[QR] p[KLPQ]
		{"KLRS,PS,QR,LKPQ", -2.0  , 4, {21,3,13,20}},		//Ap[KLRS] += -2.0 f[PS] E1[QR] p[LKPQ]
		{"KLRS,QR,PS,KLPQ", 1.0  , 4, {21,3,13,20}},		//Ap[KLRS] += 1.0 f[QR] E1[PS] p[KLPQ]
		{"KLRS,QR,PS,LKPQ", -2.0  , 4, {21,3,13,20}},		//Ap[KLRS] += -2.0 f[QR] E1[PS] p[LKPQ]
		{"KLRS,QS,PR,KLPQ", -2.0  , 4, {21,3,13,20}},		//Ap[KLRS] += -2.0 f[QS] E1[PR] p[KLPQ]
		{"KLRS,QS,PR,LKPQ", 1.0  , 4, {21,3,13,20}},		//Ap[KLRS] += 1.0 f[QS] E1[PR] p[LKPQ]
		{"KLRS,IK,QS,ILRQ", 2.0  , 4, {21,2,13,20}},		//Ap[KLRS] += 2.0 f[IK] E1[QS] p[ILRQ]
		{"KLRS,IK,QR,ILSQ", -1.0  , 4, {21,2,13,20}},		//Ap[KLRS] += -1.0 f[IK] E1[QR] p[ILSQ]
		{"KLRS,IK,PS,ILPR", -1.0  , 4, {21,2,13,20}},		//Ap[KLRS] += -1.0 f[IK] E1[PS] p[ILPR]
		{"KLRS,IK,PR,ILPS", 2.0  , 4, {21,2,13,20}},		//Ap[KLRS] += 2.0 f[IK] E1[PR] p[ILPS]
		{"KLRS,IL,QS,IKRQ", -1.0  , 4, {21,2,13,20}},		//Ap[KLRS] += -1.0 f[IL] E1[QS] p[IKRQ]
		{"KLRS,IL,QR,IKSQ", 2.0  , 4, {21,2,13,20}},		//Ap[KLRS] += 2.0 f[IL] E1[QR] p[IKSQ]
		{"KLRS,IL,PS,IKPR", 2.0  , 4, {21,2,13,20}},		//Ap[KLRS] += 2.0 f[IL] E1[PS] p[IKPR]
		{"KLRS,IL,PR,IKPS", -1.0  , 4, {21,2,13,20}},		//Ap[KLRS] += -1.0 f[IL] E1[PR] p[IKPS]
		{"KLRS,JK,QS,LJRQ", -1.0  , 4, {21,2,13,20}},		//Ap[KLRS] += -1.0 f[JK] E1[QS] p[LJRQ]
		{"KLRS,JK,QR,LJSQ", 2.0  , 4, {21,2,13,20}},		//Ap[KLRS] += 2.0 f[JK] E1[QR] p[LJSQ]
		{"KLRS,JK,PS,LJPR", 2.0  , 4, {21,2,13,20}},		//Ap[KLRS] += 2.0 f[JK] E1[PS] p[LJPR]
		{"KLRS,JK,PR,LJPS", -1.0  , 4, {21,2,13,20}},		//Ap[KLRS] += -1.0 f[JK] E1[PR] p[LJPS]
		{"KLRS,JL,QS,KJRQ", 2.0  , 4, {21,2,13,20}},		//Ap[KLRS] += 2.0 f[JL] E1[QS] p[KJRQ]
		{"KLRS,JL,QR,KJSQ", -1.0  , 4, {21,2,13,20}},		//Ap[KLRS] += -1.0 f[JL] E1[QR] p[KJSQ]
		{"KLRS,JL,PS,KJPR", -1.0  , 4, {21,2,13,20}},		//Ap[KLRS] += -1.0 f[JL] E1[PS] p[KJPR]
		{"KLRS,JL,PR,KJPS", 2.0  , 4, {21,2,13,20}},		//Ap[KLRS] += 2.0 f[JL] E1[PR] p[KJPS]
		{"KLRS,PQRa,aS,KLPQ", -2.0  , 4, {21,12,13,20}},		//Ap[KLRS] += -2.0 W[PQRa] E1[aS] p[KLPQ]
		{"KLRS,PQRa,aS,LKPQ", 1.0  , 4, {21,12,13,20}},		//Ap[KLRS] += 1.0 W[PQRa] E1[aS] p[LKPQ]
		{"KLRS,PQSa,aR,KLPQ", 1.0  , 4, {21,12,13,20}},		//Ap[KLRS] += 1.0 W[PQSa] E1[aR] p[KLPQ]
		{"KLRS,PQSa,aR,LKPQ", -2.0  , 4, {21,12,13,20}},		//Ap[KLRS] += -2.0 W[PQSa] E1[aR] p[LKPQ]
		{"KLRS,PQaR,aS,KLPQ", 1.0  , 4, {21,12,13,20}},		//Ap[KLRS] += 1.0 W[PQaR] E1[aS] p[KLPQ]
		{"KLRS,PQaR,aS,LKPQ", -2.0  , 4, {21,12,13,20}},		//Ap[KLRS] += -2.0 W[PQaR] E1[aS] p[LKPQ]
		{"KLRS,PQaS,aR,KLPQ", -2.0  , 4, {21,12,13,20}},		//Ap[KLRS] += -2.0 W[PQaS] E1[aR] p[KLPQ]
		{"KLRS,PQaS,aR,LKPQ", 1.0  , 4, {21,12,13,20}},		//Ap[KLRS] += 1.0 W[PQaS] E1[aR] p[LKPQ]
		{"KLRS,PRSa,Qa,KLPQ", 1.0  , 4, {21,12,13,20}},		//Ap[KLRS] += 1.0 W[PRSa] E1[Qa] p[KLPQ]
		{"KLRS,PRSa,Qa,LKPQ", -2.0  , 4, {21,12,13,20}},		//Ap[KLRS] += -2.0 W[PRSa] E1[Qa] p[LKPQ]
		{"KLRS,PSRa,Qa,KLPQ", -2.0  , 4, {21,12,13,20}},		//Ap[KLRS] += -2.0 W[PSRa] E1[Qa] p[KLPQ]
		{"KLRS,PSRa,Qa,LKPQ", 1.0  , 4, {21,12,13,20}},		//Ap[KLRS] += 1.0 W[PSRa] E1[Qa] p[LKPQ]
		{"KLRS,QRSa,Pa,KLPQ", -2.0  , 4, {21,12,13,20}},		//Ap[KLRS] += -2.0 W[QRSa] E1[Pa] p[KLPQ]
		{"KLRS,QRSa,Pa,LKPQ", 1.0  , 4, {21,12,13,20}},		//Ap[KLRS] += 1.0 W[QRSa] E1[Pa] p[LKPQ]
		{"KLRS,QSRa,Pa,KLPQ", 1.0  , 4, {21,12,13,20}},		//Ap[KLRS] += 1.0 W[QSRa] E1[Pa] p[KLPQ]
		{"KLRS,QSRa,Pa,LKPQ", -2.0  , 4, {21,12,13,20}},		//Ap[KLRS] += -2.0 W[QSRa] E1[Pa] p[LKPQ]
		{"KLRS,Pa,aS,KLPR", 1.0  , 4, {21,3,13,20}},		//Ap[KLRS] += 1.0 f[Pa] E1[aS] p[KLPR]
		{"KLRS,Pa,aS,LKPR", -2.0  , 4, {21,3,13,20}},		//Ap[KLRS] += -2.0 f[Pa] E1[aS] p[LKPR]
		{"KLRS,Pa,aR,KLPS", -2.0  , 4, {21,3,13,20}},		//Ap[KLRS] += -2.0 f[Pa] E1[aR] p[KLPS]
		{"KLRS,Pa,aR,LKPS", 1.0  , 4, {21,3,13,20}},		//Ap[KLRS] += 1.0 f[Pa] E1[aR] p[LKPS]
		{"KLRS,Qa,aS,KLRQ", -2.0  , 4, {21,3,13,20}},		//Ap[KLRS] += -2.0 f[Qa] E1[aS] p[KLRQ]
		{"KLRS,Qa,aS,LKRQ", 1.0  , 4, {21,3,13,20}},		//Ap[KLRS] += 1.0 f[Qa] E1[aS] p[LKRQ]
		{"KLRS,Qa,aR,KLSQ", 1.0  , 4, {21,3,13,20}},		//Ap[KLRS] += 1.0 f[Qa] E1[aR] p[KLSQ]
		{"KLRS,Qa,aR,LKSQ", -2.0  , 4, {21,3,13,20}},		//Ap[KLRS] += -2.0 f[Qa] E1[aR] p[LKSQ]
		{"KLRS,PRab,ab,KLPS", -2.0  , 4, {21,12,13,20}},		//Ap[KLRS] += -2.0 W[PRab] E1[ab] p[KLPS]
		{"KLRS,PRab,ab,LKPS", 1.0  , 4, {21,12,13,20}},		//Ap[KLRS] += 1.0 W[PRab] E1[ab] p[LKPS]
		{"KLRS,PSab,ab,KLPR", 1.0  , 4, {21,12,13,20}},		//Ap[KLRS] += 1.0 W[PSab] E1[ab] p[KLPR]
		{"KLRS,PSab,ab,LKPR", -2.0  , 4, {21,12,13,20}},		//Ap[KLRS] += -2.0 W[PSab] E1[ab] p[LKPR]
		{"KLRS,PaRb,ab,KLPS", 4.0  , 4, {21,12,13,20}},		//Ap[KLRS] += 4.0 W[PaRb] E1[ab] p[KLPS]
		{"KLRS,PaRb,ab,LKPS", -2.0  , 4, {21,12,13,20}},		//Ap[KLRS] += -2.0 W[PaRb] E1[ab] p[LKPS]
		{"KLRS,PaSb,ab,KLPR", -2.0  , 4, {21,12,13,20}},		//Ap[KLRS] += -2.0 W[PaSb] E1[ab] p[KLPR]
		{"KLRS,PaSb,ab,LKPR", 4.0  , 4, {21,12,13,20}},		//Ap[KLRS] += 4.0 W[PaSb] E1[ab] p[LKPR]
		{"KLRS,QRab,ab,KLSQ", 1.0  , 4, {21,12,13,20}},		//Ap[KLRS] += 1.0 W[QRab] E1[ab] p[KLSQ]
		{"KLRS,QRab,ab,LKSQ", -2.0  , 4, {21,12,13,20}},		//Ap[KLRS] += -2.0 W[QRab] E1[ab] p[LKSQ]
		{"KLRS,QSab,ab,KLRQ", -2.0  , 4, {21,12,13,20}},		//Ap[KLRS] += -2.0 W[QSab] E1[ab] p[KLRQ]
		{"KLRS,QSab,ab,LKRQ", 1.0  , 4, {21,12,13,20}},		//Ap[KLRS] += 1.0 W[QSab] E1[ab] p[LKRQ]
		{"KLRS,QaRb,ab,KLSQ", -2.0  , 4, {21,12,13,20}},		//Ap[KLRS] += -2.0 W[QaRb] E1[ab] p[KLSQ]
		{"KLRS,QaRb,ab,LKSQ", 4.0  , 4, {21,12,13,20}},		//Ap[KLRS] += 4.0 W[QaRb] E1[ab] p[LKSQ]
		{"KLRS,QaSb,ab,KLRQ", 4.0  , 4, {21,12,13,20}},		//Ap[KLRS] += 4.0 W[QaSb] E1[ab] p[KLRQ]
		{"KLRS,QaSb,ab,LKRQ", -2.0  , 4, {21,12,13,20}},		//Ap[KLRS] += -2.0 W[QaSb] E1[ab] p[LKRQ]
		{"KLRS,IK,PQRS,ILPQ", -1.0  , 4, {21,2,14,20}},		//Ap[KLRS] += -1.0 f[IK] E2[PQRS] p[ILPQ]
		{"KLRS,IL,PQSR,IKPQ", -1.0  , 4, {21,2,14,20}},		//Ap[KLRS] += -1.0 f[IL] E2[PQSR] p[IKPQ]
		{"KLRS,JK,PQSR,LJPQ", -1.0  , 4, {21,2,14,20}},		//Ap[KLRS] += -1.0 f[JK] E2[PQSR] p[LJPQ]
		{"KLRS,JL,PQRS,KJPQ", -1.0  , 4, {21,2,14,20}},		//Ap[KLRS] += -1.0 f[JL] E2[PQRS] p[KJPQ]
		{"KLRS,Pa,QaSR,KLPQ", 1.0  , 4, {21,3,14,20}},		//Ap[KLRS] += 1.0 f[Pa] E2[QaSR] p[KLPQ]
		{"KLRS,Pa,QaRS,LKPQ", 1.0  , 4, {21,3,14,20}},		//Ap[KLRS] += 1.0 f[Pa] E2[QaRS] p[LKPQ]
		{"KLRS,Qa,PaRS,KLPQ", 1.0  , 4, {21,3,14,20}},		//Ap[KLRS] += 1.0 f[Qa] E2[PaRS] p[KLPQ]
		{"KLRS,Qa,PaSR,LKPQ", 1.0  , 4, {21,3,14,20}},		//Ap[KLRS] += 1.0 f[Qa] E2[PaSR] p[LKPQ]
		{"KLRS,PQab,abRS,KLPQ", 1.0  , 4, {21,12,14,20}},		//Ap[KLRS] += 1.0 W[PQab] E2[abRS] p[KLPQ]
		{"KLRS,PQab,abSR,LKPQ", 1.0  , 4, {21,12,14,20}},		//Ap[KLRS] += 1.0 W[PQab] E2[abSR] p[LKPQ]
		{"KLRS,PRab,QaSb,KLPQ", 1.0  , 4, {21,12,14,20}},		//Ap[KLRS] += 1.0 W[PRab] E2[QaSb] p[KLPQ]
		{"KLRS,PRab,QabS,LKPQ", 1.0  , 4, {21,12,14,20}},		//Ap[KLRS] += 1.0 W[PRab] E2[QabS] p[LKPQ]
		{"KLRS,PSab,QabR,KLPQ", 1.0  , 4, {21,12,14,20}},		//Ap[KLRS] += 1.0 W[PSab] E2[QabR] p[KLPQ]
		{"KLRS,PSab,QaRb,LKPQ", 1.0  , 4, {21,12,14,20}},		//Ap[KLRS] += 1.0 W[PSab] E2[QaRb] p[LKPQ]
		{"KLRS,PaRb,QaSb,KLPQ", -2.0  , 4, {21,12,14,20}},		//Ap[KLRS] += -2.0 W[PaRb] E2[QaSb] p[KLPQ]
		{"KLRS,PaRb,QaSb,LKPQ", 1.0  , 4, {21,12,14,20}},		//Ap[KLRS] += 1.0 W[PaRb] E2[QaSb] p[LKPQ]
		{"KLRS,PaSb,QaRb,KLPQ", 1.0  , 4, {21,12,14,20}},		//Ap[KLRS] += 1.0 W[PaSb] E2[QaRb] p[KLPQ]
		{"KLRS,PaSb,QaRb,LKPQ", -2.0  , 4, {21,12,14,20}},		//Ap[KLRS] += -2.0 W[PaSb] E2[QaRb] p[LKPQ]
		{"KLRS,QRab,PabS,KLPQ", 1.0  , 4, {21,12,14,20}},		//Ap[KLRS] += 1.0 W[QRab] E2[PabS] p[KLPQ]
		{"KLRS,QRab,PaSb,LKPQ", 1.0  , 4, {21,12,14,20}},		//Ap[KLRS] += 1.0 W[QRab] E2[PaSb] p[LKPQ]
		{"KLRS,QSab,PaRb,KLPQ", 1.0  , 4, {21,12,14,20}},		//Ap[KLRS] += 1.0 W[QSab] E2[PaRb] p[KLPQ]
		{"KLRS,QSab,PabR,LKPQ", 1.0  , 4, {21,12,14,20}},		//Ap[KLRS] += 1.0 W[QSab] E2[PabR] p[LKPQ]
		{"KLRS,QaRb,PaSb,KLPQ", 1.0  , 4, {21,12,14,20}},		//Ap[KLRS] += 1.0 W[QaRb] E2[PaSb] p[KLPQ]
		{"KLRS,QaRb,PaSb,LKPQ", -2.0  , 4, {21,12,14,20}},		//Ap[KLRS] += -2.0 W[QaRb] E2[PaSb] p[LKPQ]
		{"KLRS,QaSb,PaRb,KLPQ", -2.0  , 4, {21,12,14,20}},		//Ap[KLRS] += -2.0 W[QaSb] E2[PaRb] p[KLPQ]
		{"KLRS,QaSb,PaRb,LKPQ", 1.0  , 4, {21,12,14,20}},		//Ap[KLRS] += 1.0 W[QaSb] E2[PaRb] p[LKPQ]
		{"KLRS,Pabc,abcS,KLPR", 1.0  , 4, {21,12,14,20}},		//Ap[KLRS] += 1.0 W[Pabc] E2[abcS] p[KLPR]
		{"KLRS,Pabc,abcS,LKPR", -2.0  , 4, {21,12,14,20}},		//Ap[KLRS] += -2.0 W[Pabc] E2[abcS] p[LKPR]
		{"KLRS,Pabc,abcR,KLPS", -2.0  , 4, {21,12,14,20}},		//Ap[KLRS] += -2.0 W[Pabc] E2[abcR] p[KLPS]
		{"KLRS,Pabc,abcR,LKPS", 1.0  , 4, {21,12,14,20}},		//Ap[KLRS] += 1.0 W[Pabc] E2[abcR] p[LKPS]
		{"KLRS,Qabc,abcS,KLRQ", -2.0  , 4, {21,12,14,20}},		//Ap[KLRS] += -2.0 W[Qabc] E2[abcS] p[KLRQ]
		{"KLRS,Qabc,abcS,LKRQ", 1.0  , 4, {21,12,14,20}},		//Ap[KLRS] += 1.0 W[Qabc] E2[abcS] p[LKRQ]
		{"KLRS,Qabc,abcR,KLSQ", 1.0  , 4, {21,12,14,20}},		//Ap[KLRS] += 1.0 W[Qabc] E2[abcR] p[KLSQ]
		{"KLRS,Qabc,abcR,LKSQ", -2.0  , 4, {21,12,14,20}},		//Ap[KLRS] += -2.0 W[Qabc] E2[abcR] p[LKSQ]
		//{"KLRS,Pabc,aQbcSR,KLPQ", 1.0  , 4, {21,12,15,20}},		//Ap[KLRS] += 1.0 W[Pabc] E3[QabScR] p[KLPQ]
		//{"KLSR,Pabc,aQbcSR,LKPQ", 1.0  , 4, {21,12,15,20}},		//Ap[KLRS] += 1.0 W[Pabc] E3[QabRcS] p[LKPQ]
		//{"KLSR,Qabc,aPbcSR,KLPQ", 1.0  , 4, {21,12,15,20}},		//Ap[KLRS] += 1.0 W[Qabc] E3[PabRcS] p[KLPQ]
		//{"KLRS,Qabc,aPbcSR,LKPQ", 1.0  , 4, {21,12,15,20}},		//Ap[KLRS] += 1.0 W[Qabc] E3[PabScR] p[LKPQ]

	};
//Number of terms : 12
	FEqInfo EqsHandCode[4] = {
		{"KL,Pabc,aQbc,KLPQ", 1.0  , 4, {29,12,31,20}},		//Ap[KLRS] += 1.0 W[Pabc] E3[QabScR] p[KLPQ]
		{"KL,Pabc,aQbc,LKPQ", 1.0  , 4, {30,12,31,20}},		//Ap[KLRS] += 1.0 W[Pabc] E3[QabRcS] p[LKPQ]
		{"KL,Qabc,aPbc,KLPQ", 1.0  , 4, {30,12,31,20}},		//Ap[KLRS] += 1.0 W[Qabc] E3[PabRcS] p[KLPQ]
		{"KL,Qabc,aPbc,LKPQ", 1.0  , 4, {29,12,31,20}},		//Ap[KLRS] += 1.0 W[Qabc] E3[PabScR] p[LKPQ]
       };
	FEqInfo Overlap[14] = {

		{"KLRS,PR,KLPS", 2.0  , 3, {19,27,25}},		//b[KLRS] += 2.0 delta[PR] V[KLPS]
		{"KLRS,PR,LKPS", -1.0  , 3, {19,27,25}},		//b[KLRS] += -1.0 delta[PR] V[LKPS]
		{"KLRS,PS,KLPR", -1.0  , 3, {19,27,25}},		//b[KLRS] += -1.0 delta[PS] V[KLPR]
		{"KLRS,PS,LKPR", 2.0  , 3, {19,27,25}},		//b[KLRS] += 2.0 delta[PS] V[LKPR]
		{"KLRS,QS,KLRQ", -1.0  , 3, {19,13,25}},		//b[KLRS] += -1.0 E1[QS] V[KLRQ]
		{"KLRS,QS,LKRQ", 0.5  , 3, {19,13,25}},		//b[KLRS] += 0.5 E1[QS] V[LKRQ]
		{"KLRS,QR,KLSQ", 0.5  , 3, {19,13,25}},		//b[KLRS] += 0.5 E1[QR] V[KLSQ]
		{"KLRS,QR,LKSQ", -1.0  , 3, {19,13,25}},		//b[KLRS] += -1.0 E1[QR] V[LKSQ]
		{"KLRS,PS,KLPR", 0.5  , 3, {19,13,25}},		//b[KLRS] += 0.5 E1[PS] V[KLPR]
		{"KLRS,PS,LKPR", -1.0  , 3, {19,13,25}},		//b[KLRS] += -1.0 E1[PS] V[LKPR]
		{"KLRS,PR,KLPS", -1.0  , 3, {19,13,25}},		//b[KLRS] += -1.0 E1[PR] V[KLPS]
		{"KLRS,PR,LKPS", 0.5  , 3, {19,13,25}},		//b[KLRS] += 0.5 E1[PR] V[LKPS]
		{"KLRS,PQRS,KLPQ", 0.5  , 3, {19,14,25}},		//b[KLRS] += 0.5 E2[PQRS] V[KLPQ]
		{"KLRS,PQSR,LKPQ", 0.5  , 3, {19,14,25}},		//b[KLRS] += 0.5 E2[PQSR] V[LKPQ]

	};
        FEqInfo MakeS1[7] = {
                {"PQRS,PR,QS", 4.0  , 3, {16,27,27}},           //S1[PQRS] += 4.0 delta[PR] delta[QS] delta[IK] delta[JL] []
                {"PQRS,PS,QR", -2.0  , 3, {16,27,27}},          //S1[PQRS] += -2.0 delta[PR] delta[QS] delta[IL] delta[JK] []
                {"PQRS,QS,PR", -2.0  , 3, {16,13,27}},          //S1[PQRS] += -2.0 E1[QS] delta[PR] delta[IK] delta[JL] []
                {"PQRS,QR,PS", 1.0  , 3, {16,13,27}},           //S1[PQRS] += 1.0 E1[PR] delta[QS] delta[IL] delta[JK] []
                {"PQRS,PS,QR", 1.0  , 3, {16,13,27}},           //S1[PQRS] += 1.0 E1[QS] delta[PR] delta[IL] delta[JK] []
                {"PQRS,PR,QS", -2.0  , 3, {16,13,27}},          //S1[PQRS] += -2.0 E1[PR] delta[QS] delta[IK] delta[JL] []
                {"PQRS,PQRT,TS", 1.0  , 3, {16,14,27}},                 //S1[PQRS] += 1.0 E2[PQRS] delta[IK] delta[JL] []
        };
        FEqInfo MakeS2[14] = {
                {"PQRS,PR,QS", 4.0  , 5, {17,27,27}},           //S1[PQRS] += 4.0 delta[PR] delta[QS] delta[IK] delta[JL] []
                {"PQRS,PR,QS", -2.0  , 5, {17,27,27}},          //S1[PQRS] += -2.0 delta[PR] delta[QS] delta[IL] delta[JK] []
                {"PQRS,PS,QR", -2.0  , 5, {17,27,27}},          //S1[PQRS] += -2.0 delta[PS] delta[QR] delta[IK] delta[JL] []
                {"PQRS,PS,QR", 4.0  , 5, {17,27,27}},           //S1[PQRS] += 4.0 delta[PS] delta[QR] delta[IL] delta[JK] []
                {"PQRS,QS,PR", -2.0  , 5, {17,13,27}},          //S1[PQRS] += -2.0 E1[QS] delta[PR] delta[IK] delta[JL] []
                {"PQRS,QS,PR", 1.0  , 5, {17,13,27}},           //S1[PQRS] += 1.0 E1[QS] delta[PR] delta[IL] delta[JK] []
                {"PQRS,QR,PS", 1.0  , 5, {17,13,27}},           //S1[PQRS] += 1.0 E1[QR] delta[PS] delta[IK] delta[JL] []
                {"PQRS,QR,PS", -2.0  , 5, {17,13,27}},          //S1[PQRS] += -2.0 E1[QR] delta[PS] delta[IL] delta[JK] []
                {"PQRS,PS,QR", 1.0  , 5, {17,13,27}},           //S1[PQRS] += 1.0 E1[PS] delta[QR] delta[IK] delta[JL] []
                {"PQRS,PS,QR", -2.0  , 5, {17,13,27}},          //S1[PQRS] += -2.0 E1[PS] delta[QR] delta[IL] delta[JK] []
                {"PQRS,PR,QS", -2.0  , 5, {17,13,27}},          //S1[PQRS] += -2.0 E1[PR] delta[QS] delta[IK] delta[JL] []
                {"PQRS,PR,QS", 1.0  , 5, {17,13,27}},           //S1[PQRS] += 1.0 E1[PR] delta[QS] delta[IL] delta[JK] []
                {"PQRS,PQRT,TS", 1.0  , 4, {17,14,27}},         //S1[PQRS] += 1.0 E2[PQRS] delta[IK] delta[JL] []
                {"PQRS,PQTR,TS", 1.0  , 4, {17,14,27}},         //S1[PQRS] += 1.0 E2[PQSR] delta[IL] delta[JK] []
        };
	static void GetMethodInfo(FMethodInfo &Out) {
		Out = FMethodInfo();
		Out.pName = "NEVPT2_CCAA";
		Out.perturberClass = "CCAA";
		Out.pSpinClass = "restricted";
		Out.pTensorDecls = &TensorDecls[0];
		Out.nTensorDecls = 32;
		Out.nDomainDecls = 0;
		Out.EqsHandCode = FEqSet(&EqsHandCode[0], 4, "NEVPT2_CCAA/Res");
		Out.EqsRes = FEqSet(&EqsRes[0], 118, "NEVPT2_CCAA/Res");
		Out.Overlap = FEqSet(&Overlap[0], 14, "NEVPT2_CCAA/Overlap");
		Out.MakeS1 = FEqSet(&MakeS1[0], 7, "NEVPT2_CCAA/Overlap");
		Out.MakeS2 = FEqSet(&MakeS2[0], 14, "NEVPT2_CCAA/Overlap");
	};
};
