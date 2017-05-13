Getting Started
***************
Structure of the Input
----------------------
A typical input (this below) is broken up into three main sections:

* header (where the number of electrons that should be included in the active space is specified, and which orbitals they occupy in the reference determinants)

* variational part (schedule and settings)
  
* perturbative part (settings)

.. code-block:: none

	#reference determinant
	nocc 8
	0 1  4 5  8 9  14 15
	0 1  4 5  8 9  14 17
	end
	nroots 2  #eg nroots don't have to be the same as number of reference determinants

	#var keywords
	schedule
	0 9e-4
	3 5e-4
	end
	maxiter 12

	#PT keywords
	deterministic
	epsilon2 1.e-8
	sampleN 200


.. note::

	Although we have presented the input separated into three distinct sections, you do *not* have to write the keywords in any particular order, **but is is highly recommended** for clarity. For more details about the keywords used or other keywords please see the :ref:`dice-keywords` page.



Header
++++++
In the first part of the previous input file (between lines 1 and 4) we establish the number of electrons in the active space (8 in this case) and we specify which orbitals each of those eight electrons occupy in given reference determinants.

Variational Section
+++++++++++++++++++
The schedule specifies which :math:`\epsilon_1` values will be used at each iteration. In the example, an :math:`\epsilon_1` value of :math:`9*10^{-4}` will be used from the :math:`0^{th}` up to the :math:`3^{rd}` iteration, i.e. in iterations 0, 1, and 2, and an :math:`\epsilon_1` value of :math:`5*10^{-4}` will be used in subsequent iterations.

Perturbative Section
++++++++++++++++++++
*Dice* can calculate the second order perturbative contribution to the energy in three ways all of which involve the use of the :math:`\epsilon_2` keyword. This parameter controls how much of the connected space will be included in the perturbative calculation.

1. Deterministic:

   All determinants :math:`D_a` which statisfy :math:`|H_{ai}c_i| > \epsilon_2` for a determinant :math:`D_i` of the variational space are included in the following summation

	.. math:: \Delta E_2^D[\epsilon_2] = \sum_a \frac{1}{E^{(0)}-H_{aa}}\left(\sum_i^{\epsilon_2} H_{ai} c_i \right)^2
		:label: e_det_pt

   where :math:`H_{ai}` is the Hamiltonian matrix element connecting determinants :math:`D_i` and :math:`D_a`, :math:`c_i` is the coefficient of the determinant :math:`D_i`, and :math:`E^{(0)}` is the eigenvalue of the zeroth-order Hamiltonian.

2. Stochastic:
   
   A small number of determinants in the variational space (determined by the `SampleN` parameter, see :ref:`dice-keywords` for more details) are stochastically selected and used to evaluate the following equation:

	.. math:: \Delta E_2^S[\epsilon_2] = \frac{1}{N_d(N_d-1)} \Bigg \langle \sum_a \frac{1}{E^{(0)}-H_{aa}} \Bigg[ \bigg(\sum_i^{N_d^{diff}} \frac{\omega_i}{p_i} H_{ai} c_i\bigg)^2 + \sum_i^{N_d^{diff}} \bigg(\frac{\omega_i}{p_i} (N_d-1) - \frac{\omega_i^2}{p_i^2}\bigg) \bigg(H_{ai}c_i\bigg)^2 \Bigg] \Bigg \rangle
		:label: e_stoc_pt

   where :math:`N_d` is the total number of determinants sampled, :math:`N_d^{diff}` is the number of unique determinants sampled, :math:`\omega_i` is the number of times determinant :math:`D_i` was selected in the sample, and :math:`p_i = \frac{|c_i|}{\sum_i |c_i|}` is the probability of selecting determinant :math:`D_i`.
   
   The stochastic evaluation of the perturbative correction to the energy enables the use of a much smaller :math:`\epsilon_2` for large systems, while retaining the sub-mHa accuracy of the deterministic variant.

3. Semistochastic:
   
   In this variant *Dice* evaluates both eq. :eq:`e_det_pt` and :eq:`e_stoc_pt` and computes:

	.. math:: \Delta E_2[\epsilon_2] = \big( \Delta E_2^S[\epsilon_2] − \Delta E_2^S[\epsilon_{2Large}] \big) + \Delta E_2^D[\epsilon_{2Large}]
		:label: e_semi_pt

   where the energy correction using :eq:`e_stoc_pt` is computed for both :math:`\epsilon_2` and :math:`\epsilon_{2Large}` and the energy correction using :eq:`e_det_pt` is computed for :math:`\epsilon_{2Large}`. 
   The subtracting of both stochastic correction reduces the stochastic noise.



Using Dice as a stand-alone program
-----------------------------------
The *Dice* program can be used as a wavefunction solver without interfacing it with other programs (see also the :ref:`interfacing-with-pyscf` section for the use of *Dice* as an active space solver in a CASSCF calculation).

The program requires a `FCIDUMP` two-body integral file, which can be generated using whatever electronic structure package you prefer. 

An example `input.dat` file is shown below, for more input files see the test directory inside your main *Dice* directory.

.. code-block:: none

	nocc 8
	0 1  4 5  8 9  14 15
	end

	sampleN 200
	davidsontol 5.e-5
	dE 1.e-7
	DoRDM

	schedule
	0 9e-4
	3 5e-4
	end

	epsilon2 1.e-8
	deterministic
	noio
	maxiter 12


The *Dice* program is simply run in the directory with both `FCIDUMP` and `input.dat` files using the following command:

.. code-block:: bash

	mpirun -np 2 /path_to/Dice/Dice input.dat > output.dat

.. note::

	You can run *Dice* in parallel using OMP, MPI, or a hybrid of both, but we recommend that you use MPI. If you would like to run a hybrid scheme, please contact us to help you set it up.


This will execute your `input.dat` file and write all output to the `output.dat` file in your current working directory. An example of the output is shown below:

.. code-block:: none


	**************************************************************
	Dice  Copyright (C) 2017  Sandeep Sharma
	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
	See the GNU General Public License for more details.


	Author:       Sandeep Sharma
	Contributors: James E Smith, Adam A Holmes, Bastien Mussard
	For detailed documentation on Dice please visit
	https://sanshar.github.io/Dice/
	Please visit our group page for up to date information on other projects
	http://www.colorado.edu/lab/sharmagroup/
	**************************************************************


	**************************************************************
	Input file  :
	**************************************************************
	nocc 12
	#0 2 4 6 8 10 1 3 5 7 9 11
	epsilon2 1e-07
	sampleN 200
	davidsonTol 5e-05
	dE 1e-08
	DoRDM True
	schedule
	#0      0.01
	3       0.001
	end
	deterministic
	noio
	maxiter 9




	**************************************************************
	VARIATIONAL STEP
	**************************************************************
	Iter Root       Eps1   #Var. Det.               Energy     Time(s)
	 0    0    1.00e-02          27      -149.6693332442        0.01
	 1    0    1.00e-02          40      -149.6755619487        0.01
	 2    0    1.00e-02          47      -149.6762309265        0.01
	 3    0    1.00e-03         106      -149.6776013798        0.01
	 4    0    1.00e-03         106      -149.6776013798        0.01

	Exiting variational iterations
	Calculating RDM
	VARIATIONAL CALCULATION RESULT
	------------------------------
	Root             Energy     Time(s)
	   0     -149.6776013798        0.07


	Printing most important determinants
	 Det     weight  Determinant string
	State :0
	   0   9.27e-01  2 2 2 2 2   2 0 0
	   1   2.48e-01  2 2 2 2 2   0 2 0
	   2   2.12e-01  2 2 2 2 0   2 2 0
	   3   6.66e-02  2 2 2 a b   2 a b
	   4   6.66e-02  2 2 2 b a   2 b a
	   5   6.55e-02  2 2 2 0 2   2 0 2

	**************************************************************
	PERTURBATION THEORY STEP
	**************************************************************
	Deterministic PT calculation converged
	epsilon2: 1e-07
	PTEnergy: -149.677601562499
	Time(s):  0.107640981674194
	Now calculating PT RDM




.. _interfacing-with-pyscf:

Interfacing with PySCF in a CASSCF calculation
----------------------
*Dice* can also be used as an FCI solver in CASSCF calculation with a large number of active space orbitals and electrons. If `PySCF <https://github.com/sunqm/pyscf/blob/master/README.md>`_ is successfully installed (see `Installing PySCF <https://github.com/sunqm/pyscf/blob/master/README.md#installation>`_), you can call *Dice* from within a python input script. After you compile *Dice*, you must edit the two path variables shown below in the settings.py file in the shciscf module directory in PySCF.

.. code-block:: bash

	SHCIEXE = '/path_to/SHCI/SHCI'
	SHCIQDPTEXE = '/path_to/SHCI/QDPTSOC'


Once this is completed you can run call *Dice* from within `PySCF <https://github.com/sunqm/pyscf/blob/master/README.md>`_. An example input is shown below:

.. code-block:: python

	from pyscf import gto, scf
	from pyscf.future.shciscf import shci


	# Initialize O2 molecule
	b =  1.243
	mol = gto.Mole()
	mol.build(
	verbose = 5,
	output = None,
	atom = [
	    ['C',(  0.000000,  0.000000, -b/2)],
	    ['C',(  0.000000,  0.000000,  b/2)], ],
	basis = {'C': 'ccpvdz', },
	symmetry = True,
	symmetry_subgroup = 'D2h',
	spin = 0
	)

	# Create HF molecule
	mf = scf.RHF( mol )
	mf.conv_tol = 1e-9
	mf.scf()

	# Number of orbital and electrons
	norb = 26
	nelec = 8

	# Create SHCI molecule for just variational opt.
	# Active spaces chosen to reflect valence active space.
	mch = shci.SHCISCF( mf, norb, nelec )
	mch.fcisolver.mpiprefix = 'mpirun -np 2'
	mch.fcisolver.stochastic = True
	mch.fcisolver.nPTiter = 0
	mch.fcisolver.sweep_iter = [ 0, 3 ]
	mch.fcisolver.DoRDM = True
	mch.fcisolver.sweep_epsilon = [ 5e-3, 1e-3 ]
	e_shci = mch.mc1step()[0]


This script can be executed in the command line as follows:

.. code-block:: bash

	python test_c2.py
