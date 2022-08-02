## RDM calculation with a Jastrow-Slater wave function

```
python -u rdm.py > rdm.out
```

This writes input files for VMC and runs the calculation, it will generate a bunch of output files including the RDM files. All RDM's are spin RDM's and they are all stored in matrices with spin orbital indices. Even orbitals are spin up and odd are spin down. Two RDM's are in the format T(pq, rs) = < a_p^dag a_q^dag a_s a_r > with p > q and r > s. A lot of elements will be zero because of spin symmetry. 
