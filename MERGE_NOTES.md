# Notes About All Things Dice

__Wish List__:
- CMake
- Doctest 
- Travis CI
- ZDice
- 


## 3/12/2020
- Dice using `Determinants::norbs` for spin orbitals while VMC uses it for spatial orbitals.
- I'm going to add `Determinants::n_spinorbs` and switch over all the Dice calls to this so VMC will continue to work as is.
