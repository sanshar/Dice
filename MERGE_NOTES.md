# Notes About All Things Dice During Merger w/ VMC

__Wish List__:
- [ ] CMake
- [ ] Doctest 
- [ ] Travis CI
- [ ] ZDice
- [ ] Compile with `-Wall` and clear all warnings
- [ ] Consolidate variable names: DetLen and Determinant::EffDetLen
- [ ] Differentiate between VMC-style and Dice-style functions in their names to make it easier to remember/use
- [ ] Consolidate Determinant::EffDetLen and DetLen

## 3/18/2020
- The MPI issue was coming from a bad memory allocation/sharing in SHCIbasics.cpp after adding determinants

## 3/16/2020
- The problem in energy after the fixes on 3/13/2020 came from an issue with the parity
- The first MPI issues I've uncovered are coming from a corruption of sharing the dets in `SHCIbasics.cpp`

## 3/13/2020
- MPI even `mpirun -np 1` is causing problems with RDMS and energies
- I changed a bunch of argument types to non-reference version, which meant that if they were modified, this modification wasn't visible outside the scope of the method.

## 3/12/2020
- Dice using `Determinants::norbs` for spin orbitals while VMC uses it for spatial orbitals.
- I'm going to add `Determinants::n_spinorbs` and switch over all the Dice calls to this so VMC will continue to work as is.
