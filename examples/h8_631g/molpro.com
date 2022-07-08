        memory,500,mw;
        basis=6-31g;
        bohr
        symmetry,x,y,z
        orient,mass
        geometry={
          H  0.   0.   0. 
          H  2.   0.   0.
          H  4.   0.   0.
          H  6.   0.   0.
          H  8.   0.   0.
          H  10.  0.   0.
          H  12.  0.   0.
          H  14.  0.   0.
        }
        hf
        {casscf
        closed,0
        occ,4,,,,4
        wf,8,1,0}
        !{nevpt2
        !closed,0
        !occ,10
        !wf,10,1,0}
        {mrcic
        closed,0
        occ,4,,,,4
        wf,8,1,0}
