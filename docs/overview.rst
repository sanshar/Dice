Overview
**************
Algorithm
--------------
The Semistochastic Heat-Bath Configuration Interaction (SHCI) method is a Selection Configuration Interaction (SCI) method with perturbation theory (PT).
These methods partition the determinants in the Hilbert space into two sets, the determinants in the first set :math:`V` are used to expand the variational wavefunction and the second set :math:`C` is composed of the determinants that are connected to the determinants in :math:`V` by a non-zero Hamiltonian matrix element, and will be used to compute the perturbative correction to the variational energy.

.. In the following, we use indices :math:`i, j` for determinants in the variational space :math:`V`, and index :math:`a, b` for determinants in the connected space :math:`C`.

There are two key ideas that differentiate SHCI and accounts for its vastly higher efficiency:

1. Contrary to other SCI-PT algorithms, the SHCI algorithm uses a simpler and cheaper criterion in which a determinant :math:`D_a` is included in the space :math:`C` if :math:`|H_{ai}c_i| > \epsilon_1` is satisfied for any :math:`D_i` in :math:`V` and for a user-defined parameter :math:`\epsilon_1`. The same criterion is used to approximate the large summation involved in the second-order perturbative correction to the variational energy, with a second user-defined parameter :math:`\epsilon_2`.

.. (Notably due to the fact that the magnitude of the 2-body excitation matrix elements :math:`|H_{ai}|` between two determinants only depends on the indices of the 4 orbitals whose occupations are changing between the two determinants and not on the other occupied orbitals of the determinants.)

.. Whereas the usual SCI-PT algorithms includes a determinant :math:`D_{a}` in the space :math:`V` if the criterion :math:`\frac{\sum_i H_{ai} c_i}{E_0 - H_{aa}} > \epsilon_1`, is satisfied for a user-defined parameter :math:`\epsilon_1`,

..

  Firstly, this simpler criterion is cheaper to compute that usual SCI criterions (as a matter of fact all computations involved in the selection of determinants can be made at the beginning of the run, and not at each iteration) and secondly, the use of this simpler criterion allows SHCI to design a procedure in which only the important determinants are ever looked at, and no computational time is lost on determinant that are ultimatly not included in the spaces :math:`V` or :math:`C`!


2. The above ideas speed up both the variational and the perturbative steps of the algorithm by several orders of magnitude. However, the perturbative step has a large memory requirement because all determinants that are connected to those in :math:`V` must be stored. We have developed a semistochastic perturbative approach that both overcomes this memory bottleneck and is faster than the deterministic approach at the cost of having a small statistical error.
   
   The resulting semistochastic algorithm has several attractive features:


 * There is no sign problem that plagues quantum Monte Carlo methods. Furthermore, instead of using Metropolis-Hastings sampling, we use the Alias method to directly sample determinants from the reference wavefunction, thus avoiding correlations between consecutive samples.

 * In addition to removing the memory bottleneck, SHCI is faster than the deterministic variant for many systems if a stochastic error of 0.1 mHa is acceptable.

 * The SHCI algorithm extends the range of applicability of the original algorithm, allowing us to calculate the correlation energy of large active spaces.

Features
--------
* Wavefunction optimization and calculation of one and two reduced density matrices (RDMs) for multireference systems with large active spaces, between 30-100 orbitals. *Dice* can also calculate the one and two-body relaxed density matrices during the perturbative calculation.

* Exploitation of Abelian, :math:`D_{\infty h}`, and :math:`C_{\infty v}` symmetries.

* When combined with `PySCF <https://github.com/sunqm/pyscf/blob/master/README.md>`_, *Dice* can also calculate excited states using state averaging.

* Active space orbital optimization with the `PySCF <https://github.com/sunqm/pyscf/blob/master/README.md>`_.
  




License and how to cite
-----------------------
*Dice* is distributed under the GNU GPL license which is reproduced on the top of every source file. We would appreciate if you cite the following paper in publications resulting from the use of *Dice*.

* S. Sharma, A. A. Holmes, G. Jeanmairet, A. Alavi, C. J. Umrigar, `"Semistochastic Heat-bath Configuration Interaction method: selected configuration interaction with semistochastic perturbation theory." <http://pubs.acs.org/doi/abs/10.1021/acs.jctc.6b01028?journalCode=jctcce>`_ *Journal* *of* *Chemical* *Theory* *and* *Computations*, **2017**, 13, 1595.
