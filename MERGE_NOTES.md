# Notes About All Things Dice During Merger w/ VMC

__Wish List__:
- [ ] CMake
- [ ] Doctest 
- [ ] Travis CI
- [ ] ZDice
- [ ] Compile with `-Wall` and clear all warnings


## 3/13/2020
- MPI even `mpirun -np 1` is causing problems with RDMS and energies
- I changed a bunch of argument types to non-reference version, which meant that if they were modified, this modification wasn't visible outside the scope of the method.

## 3/12/2020
- Dice using `Determinants::norbs` for spin orbitals while VMC uses it for spatial orbitals.
- I'm going to add `Determinants::n_spinorbs` and switch over all the Dice calls to this so VMC will continue to work as is.
