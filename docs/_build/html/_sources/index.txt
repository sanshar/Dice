.. Dice documentation master file, created by
   sphinx-quickstart on Sun May  7 23:12:23 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Dice's documentation!
================================
.. figure::  images/dice.jpg
   :width: 80px
   :align:   left
*Dice* implements the semistochastic heat bath configuration interaction (SHCI) algorithm for *ab initio* Hamiltonian of a quantum chemical system. Unlike full configuration interaction (FCI), SHCI can be used to treat active spaces containing 30-100 orbitals. SHCI is able to accomplish this by taking advantage of the fact that although the full Hilbert space may be enormous, only a small fraction of the determinants in the space have appreciable coefficients. Compared to other methods in its class SHCI is often not only orders of magnitude faster, it also does not suffer from a serious memory bottleneck that plauges such methods. The resulting algorithm as implemented in Dice allows us to treat large benchmark systems such as the Chromium dimer and Mn-Salen (a challenging bioinorganic cluster) at a cost that is often an order of magnitude faster than either density matrix renormalization group (DMRG) or full configuration interaction quantum Monte Carlo (FCIQMC). Thus if you are interested in performing multireference calculations with active space containing several tens to hundreds of orbitals, SHCI might be an ideal choice for you.


Contents:


.. toctree::
   :maxdepth: 2

   overview
   installation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

