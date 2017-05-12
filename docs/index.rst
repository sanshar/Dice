.. Dice documentation master file, created by
   sphinx-quickstart on Sun May  7 23:12:23 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Dice's documentation!
================================
.. figure::  images/dice_orange.png
   :width: 80px
   :align:   left
   
*Dice* implements the semistochastic heat bath configuration interaction (SHCI) algorithm for *ab initio* Hamiltonians of quantum chemical systems.

Unlike full configuration interaction (FCI), SHCI can be used to treat active spaces containing 30 to 100 orbitals. SHCI is able to accomplish this by taking advantage of the fact that although the full Hilbert space may be enormous, only a small fraction of the determinants in the space have appreciable coefficients.

Compared to other methods in its class, SHCI is often not only orders of magnitude faster, it also does not suffer from serious memory bottlenecks that plagues these methods. The resulting algorithm as implemented in *Dice* allows us to treat difficult benchmark systems such as the Chromium dimer and Mn-Salen (a challenging bioinorganic cluster) at a cost that is often an order of magnitude faster than density matrix renormalization group (DMRG) or full configuration interaction quantum Monte Carlo (FCIQMC).

Thus if you are interested in performing multireference calculations with active space containing several tens to hundreds of orbitals, *Dice* might be an ideal choice for you.


Contents:


.. toctree::
   :maxdepth: 2

   overview
   installation
   gettingstarted
   keywords
   benchmarking

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
