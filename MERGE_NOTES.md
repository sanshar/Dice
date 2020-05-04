# Notes About All Things Dice During Merger w/ VMC

__Wish List__:
- [X] CMake
- [X] Doctest 
- [ ] Travis CI
- [ ] ZDice
- [ ] Compile with `-Wall` and clear all warnings
- [ ] Consolidate variable names: DetLen and Determinant::EffDetLen
- [ ] Differentiate between VMC-style and Dice-style functions in their names to make it easier to remember/use
- [ ] Trev should be constrained to 0 or 1
- [ ] Make separate testing repo
- [ ] Make separate benchmarking repo
- [ ] Differentiate between compile time variables (DetLen) and runtime variables Determinant::EffDetLen
- [ ] Add HDF5 for FCIDUMP and RDMs (see https://shankarkulumani.com/2018/09/hdf5.html)
- [ ] Add option to batch RDMs

## 4/3/2020

## 3/26/2020
- Trev problem is triggered by having more orbitals in the active space, it's triggered by Mn(Salen) and O2 w/ only 1s not in active space, might actually be CI vector size

## 3/25/2020
- Problem with the Trev restart test is still getting the better of me. The SIGSEGV error is popping up during the initial input, i.e. it's not even restarting. Not entirely sure why this error is cropping up now since the other TREV tests work fine.

## 3/23/2020
- Trev tests (except restart trev) are passing
  - The problem was in `isStandard()` and I was including more determinants than I should because too many were considered standard in my initial implementation.
- Some of the restart tests pass too
- Some of the ref determinants fail

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
