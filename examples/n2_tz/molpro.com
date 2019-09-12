        memory,3800,mw;
        basis=cc-pVTZ;
        bohr
        geometry={
            n1
            n2,n1,2.5
        }
        hf
        {casscf
        closed,1,0,0,0,1,0,0
        occ,3,1,1,0,3,1,1
        wf, 14,1,0}
        {caspt2
        closed,1,0,0,0,1,0,0
        occ,3,1,1,0,3,1,1
        wf, 14,1,0}
        {nevpt2
        closed,1,0,0,0,1,0,0
        occ,3,1,1,0,3,1,1
        wf, 14,1,0}
        {mrcic
        closed,1,0,0,0,1,0,0
        occ,3,1,1,0,3,1,1
        wf, 14,1,0}
