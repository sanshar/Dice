namespace NEVPT2_CCAV {

  FTensorDecl TensorDecls[19] = {
    /*  0*/{"R"      , "ccae"     , "", USAGE_Residual    },
    /*  1*/{"t"      , "ccae"     , "", USAGE_Amplitude   },
    /*  2*/{"T"      , "ccae"     , "", USAGE_Amplitude   },
    /*  3*/{"b"      , "ccae"     , "", USAGE_Amplitude   },
    /*  4*/{"B"      , "ccae"     , "", USAGE_Amplitude   },
    /*  5*/{"p"      , "ccae"     , "", USAGE_Amplitude   },
    /*  6*/{"Ap"     , "ccae"     , "", USAGE_Amplitude   },
    /*  7*/{"P"      , "ccae"     , "", USAGE_Amplitude   },
    /*  8*/{"AP"     , "ccae"     , "", USAGE_Amplitude   },
    /*  9*/{"W"      , "ccae"     , "", USAGE_Hamiltonian },
    /* 10*/{"f"      , "aa"       , "", USAGE_Hamiltonian },
    /* 11*/{"W"      , "aaaa"     , "", USAGE_Hamiltonian },
    /* 12*/{"f"      , "cc"       , "", USAGE_Hamiltonian },
    /* 13*/{"f"      , "ee"       , "", USAGE_Hamiltonian },
    /* 14*/{"delta"  , "aa"       , "", USAGE_Density     },
    /* 15*/{"E1"     , "aa"       , "", USAGE_Density     },
    /* 16*/{"E2"     , "aaaa"     , "", USAGE_Density     },
    /* 17*/{"S1"     , "AA"       , "", USAGE_Density     },
    /* 18*/{"S2"     , "aa"       , "", USAGE_Density     },
  };

  FEqInfo EqsRes[22] = {
    {"abcd,ce,abed",          4.0,   3, { 6,10, 5}},     //   4.0 Ap[abcd] f[ce] p[abed] 
    {"abcd,ce,baed",         -2.0,   3, { 6,10, 5}},     //  -2.0 Ap[abcd] f[ce] p[baed] 
    {"abcd,ae,becd",          2.0,   3, { 6,12, 5}},     //   2.0 Ap[abcd] f[ae] p[becd] 
    {"abcd,ae,ebcd",         -4.0,   3, { 6,12, 5}},     //  -4.0 Ap[abcd] f[ae] p[ebcd] 
    {"abcd,be,aecd",         -4.0,   3, { 6,12, 5}},     //  -4.0 Ap[abcd] f[be] p[aecd] 
    {"abcd,be,eacd",          2.0,   3, { 6,12, 5}},     //   2.0 Ap[abcd] f[be] p[eacd] 
    {"abcd,de,abce",          4.0,   3, { 6,13, 5}},     //   4.0 Ap[abcd] f[de] p[abce] 
    {"abcd,de,bace",         -2.0,   3, { 6,13, 5}},     //  -2.0 Ap[abcd] f[de] p[bace] 
    {"abcd,ef,abed,fc",      -2.0,   4, { 6,10, 5,15}},  //  -2.0 Ap[abcd] f[ef] p[abed] E1[fc] 
    {"abcd,ef,baed,fc",       1.0,   4, { 6,10, 5,15}},  //   1.0 Ap[abcd] f[ef] p[baed] E1[fc] 
    {"abcd,cefg,abed,gf",    -2.0,   4, { 6,11, 5,15}},  //  -2.0 Ap[abcd] W[cefg] p[abed] E1[gf] 
    {"abcd,cefg,abfd,eg",     4.0,   4, { 6,11, 5,15}},  //   4.0 Ap[abcd] W[cefg] p[abfd] E1[eg] 
    {"abcd,cefg,baed,gf",     1.0,   4, { 6,11, 5,15}},  //   1.0 Ap[abcd] W[cefg] p[baed] E1[gf] 
    {"abcd,cefg,bafd,eg",    -2.0,   4, { 6,11, 5,15}},  //  -2.0 Ap[abcd] W[cefg] p[bafd] E1[eg] 
    {"abcd,ae,befd,fc",      -1.0,   4, { 6,12, 5,15}},  //  -1.0 Ap[abcd] f[ae] p[befd] E1[fc] 
    {"abcd,ae,ebfd,fc",       2.0,   4, { 6,12, 5,15}},  //   2.0 Ap[abcd] f[ae] p[ebfd] E1[fc] 
    {"abcd,be,aefd,fc",       2.0,   4, { 6,12, 5,15}},  //   2.0 Ap[abcd] f[be] p[aefd] E1[fc] 
    {"abcd,be,eafd,fc",      -1.0,   4, { 6,12, 5,15}},  //  -1.0 Ap[abcd] f[be] p[eafd] E1[fc] 
    {"abcd,de,abfe,fc",      -2.0,   4, { 6,13, 5,15}},  //  -2.0 Ap[abcd] f[de] p[abfe] E1[fc] 
    {"abcd,de,bafe,fc",       1.0,   4, { 6,13, 5,15}},  //   1.0 Ap[abcd] f[de] p[bafe] E1[fc] 
    {"abcd,efgh,abed,fghc",  -2.0,   4, { 6,11, 5,16}},  //  -2.0 Ap[abcd] W[efgh] p[abed] E2[fghc] 
    {"abcd,efgh,baed,fghc",   1.0,   4, { 6,11, 5,16}},  //   1.0 Ap[abcd] W[efgh] p[baed] E2[fghc] 
  };

  FEqInfo Overlap[4] = {
    {"IJPA,RP,IJRA",   4.0,   3, { 3,14, 9}}, //  4.0 b[IJPA] delta[RP] W[IJRA]
    {"IJPA,RP,IJRA",  -2.0,   3, { 3,15, 9}}, // -2.0 b[IJPA] E1[RP]    W[IJRA]
    {"IJPA,RP,JIRA",  -2.0,   3, { 3,14, 9}}, // -2.0 b[IJPA] delta[RP] W[JIRA]
    {"IJPA,RP,JIRA",   1.0,   3, { 3,15, 9}}, //  1.0 b[IJPA] E1[RP]    W[JIRA]
  };

  int f(int i) {
    return 2*i;
  }
  FDomainDecl DomainDecls[1] = {
    {"A", "a", f}
  };

  static void GetMethodInfo(FMethodInfo &Out) {
    Out = FMethodInfo();
    Out.pName = "NEVPT2_CCAV";
    Out.perturberClass = "CCAV";
    Out.pSpinClass = "restricted";
    Out.pTensorDecls = &TensorDecls[0];
    Out.nTensorDecls = 19;
    Out.pDomainDecls = &DomainDecls[0];
    Out.nDomainDecls = 1;
    Out.EqsRes = FEqSet(&EqsRes[0], 22, "NEVPT2_CCAV/Res");
    Out.Overlap = FEqSet(&Overlap[0], 4, "NEVPT2_CCAV/Overlap");
  };
};
