Overview
**************
Algorithm
--------------
There are two key ideas that differentiate SHCI and accounts for its vastly higher efficiency. SHCI partitions the entire set of determinants in the Hilbert space into two sets, the first that are treated variationally are included in the set *V* and a second set *C* that are connected to the determinants in *V* by a non-zero Hamiltonian matrix element. We use indices *i, j* for determinants in the variational space *V*, and index *a* for determinants in the connected space *C*.

1. SHCI takes advantage of the fact that the 2-body excitation matrix elements depend only on the indices of the 4 orbitals whose occupations are changing and not on the other occupied orbitals. This allows SHCI to use a procedure in which only the important determinants are ever looked at! In addition, the usual SCI-PT algorithm promotes a determinant a to the V space if the criterion :math:`\frac{\sum_i H_{ai} c_i}{E_0 - H_{aa}} > \epsilon_1`, is satisfied for a user defined parameter :math:`\epsilon_1`. Instead SHCI uses a much simpler criterion in which a determinant *a* is promoted to *V* if :math:`|H_{ai}c_i| > \epsilon_1` for any *i* in *V*. Although using the SHCI criterion is much cheaper, the difference in calculated energies is negligibly small.

2. The above ideas speed up both the variational and the perturbative steps of the algorithm by several orders of magnitude. However, the perturbative step has a large memory requirement because all determinants that are connected to those in *V* must be stored. We have developed a semistochastic perturbative approach that both overcomes this memory bottleneck and is faster than the deterministic approach at the cost of having a small statistical error.

Features
--------
The resulting SHCI algorithm has several attractive features:

* There is no sign problem that plagues several quantum Monte Carlo methods. Instead of using Metropolis-Hastings sampling, we use the Alias method to directly sample determinants from the reference wavefunction, thus avoiding correlations between consecutive samples.

* In addition to removing the memory bottleneck, semistochastic HCI (SHCI) is faster than the deterministic variant for many systems if a stochastic error of 0.1 mHa is acceptable.

* The SHCI algorithm extends the range of applicability of the original algorithm, allowing us to calculate the correlation energy of very large active spaces.

License and how to cite
-----------------------
*Dice* is distributed under the GNU GPL license which is reproduced on the top of every source file. We would appreciate if you cite the following paper in publications resulting from the use of *Dice*.

* S. Sharma, A. A. Holmes, G. Jeanmairet, A. Alavi, C. J. Umrigar, `"Semistochastic Heat-bath Configuration Interaction method: selected configuration interaction with semistochastic perturbation theory." <http://pubs.acs.org/doi/abs/10.1021/acs.jctc.6b01028?journalCode=jctcce>`_ *Journal* *of* *Chemical* *Theory* *and* *Computations*, **2017**, 13, 1595. 