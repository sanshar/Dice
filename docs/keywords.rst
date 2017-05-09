.. _dice-keywords:

Keywords
**************

The keywords are listed below can be added to the input script after the occupation of the orbitals is specified. To use a keyword, write the keyword followed by the type of value specified in parentheses e.g. to use davidsonTol write "davidsonTol 5e-5" in your input script. If no type of value is given in parentheses simply use th keyword itself to toggle the desired behavior, e.g. "DoRDM" in your input file will turn on the DoRDM behavior described below.

Basic Keywords
--------------

* davidsonTol
	(Double) Tolerance for the Davidson step. The default is 5e−5.


* epsilon2
	(Double) Lower limit for accepted value of :math:`|H_{ai}c_i|` used when adding determinants to the Fock space during the variational calculation. Here Hai is the hamiltonian matrix element between determinants i and a. ci is the projection of the wavefunction onto the ith determinant. Used with stochastic, deterministic, and semi-stochastic perturbative calculations.


* epsilon2Large
	(Double) This keyword and value cause the perturbative component to be calculated using a semistochastic approach. Epsilon2 specifices the lower limit for the stochastic portion of the perturbative calculation and epsilon2Large specifies the lower limit of the perturbative component that will be calculated deterministically.


* SampleN
	(Integer) Number of times the set of determinants outside the variational space is sampled in a given stochastic or semistochastic iteration.


* epsilon1
	(Array of doubles) Lower limit for accepted value of :math:`|H_{ai}c_i|` used when adding determinants to the Fock space during the variational calculation. Here Hai is the hamiltonian matrix element between determinants i and a. :math:`c_i` is the projection of the wavefunction onto the ith determinant.


* dE
	(Double) This is the energy convergence tolerance for the variational portion of the calculation.


* prefix
	(String) Path to scratch directory. Default is ".". To set your own path, simply write the absolute path in the input file without quotations.


* stochastic
	The default is true and this keyword should not be included in input file. True means that the the perturbative component will be calculated stochastically or semi-stochastically if epsilon2Large is used. To switch to deterministic write "deterministic" in input file.


* io
	Default is true and does not need to be included in input file. When true SHCI will write the variational results. To set to false write "noio" in input script.


* nroots
	(Integer) Number of eigenstates to solve for. Default is 1, i.e. ground state.


* nPTiter
	(Integer) Number of iterations for stochastic or semi-stochastic perturbative calculations.


* DoRDM
	If true, the program will save the spatial reduced density matrix as a text file. Default is False.
