Getting Started
***************
Structure of the Input
----------------------
The input is broken up into three main sections: first, the header specifies how many electrons should be included in the active space and which orbitals they occupy in the Hatree Fock (HF) determinant; second, the schedule and settings for the variational part of the calculation; finally, the settings for the perturbative calculation.

Header
++++++
In the first part of the input file, between lines 1 and 4, we establish the number of electrons in the active space (8 in this case) and then we specify which orbitals each of those eight electrons is in. The header can include more than one reference determinant as shown below in a minimal input file.

Variational Section
+++++++++++++++++++
The schedule specifies which :math:`\epsilon_1` values will be used at each iteration. In the example shown directly below, an :math:`\epsilon_1` value of :math:`9*10^{-4}` will be used from the :math:`0^{th}` up to the :math:`3^{rd}` iteration, i.e. in iterations 0, 1, and 2.

Perturbative Section
++++++++++++++++++++
*Dice* can calculate the second order perturbative contribution to the energy in three ways all of which involve the use of the :math:`epsilon_2` keyword. This parameter controls how much of the determinant space will be included in the perturbative calculation

1) Deterministic: All determinants connected to the variational space with :math:`|H_{ai}c_i| > \epsilon_2` are included in the following summation where :math:`\Delta E^{(2)}` is the second order perturbative correction to the energy, :math:`H_{ai}` is the Hamiltonian matrix element connecting determinants i and a, :math:`c_i` is the coefficient of the i :math:`^{th}` determinant, and :math:`E^{(0)}` is the eigenvalue of the zeroth-order Hamiltonian.

	.. math:: \Delta E^{(2)} \approx \sum_a \frac{\big(\sum_i^{\epsilon_2} H_{ai} c_i \big)^2}{E^{(0)}-H_{aa}}
		:label: e_det_pt

2) Stochastic: A small number of determinants in the variational space (determined by the SampleN parameter, see :ref:`dice-keywords` for more details) are stochastically selected and used evaluate equation :eq:`e_stoc_pt` where :math:`N_d` is the total number of determinants sampled, :math:`N_d^{diff}` is the number of unique determinants sampled, :math:`\omega_i` is the number of times determinant i was selected in this sample, and :math:`p_i = \frac{|c_i|}{\sum_i |c_i|}` or the probability of selecting determinant i. Stochastic evaluation of the energy correction enables the use of much smaller :math:`\epsilon_2` values for large systems, while retaining the sub mHa accuracy of the deterministic variant.

	.. math:: \Delta E^{(2)} \approx \frac{1}{N_d(N_d-1)} \Bigg \langle \sum_a \frac{1}{E^{(0)}-H_{aa}} \Bigg[ \bigg(\sum_i^{N_d^{diff}} \frac{\omega_i H_{ai} c_i}{p_i}\bigg)^2 + \sum_i^{N_d^{diff}} \bigg(\frac{\omega_i (N_d-1)}{p_i} - \frac{\omega_i^2}{p_i^2}\bigg) c_i^2 H_{ai}^2 \Bigg] \Bigg \rangle
		:label: e_stoc_pt

3) Semistochastic: In this variant of the peturbative calculation, *Dice* both eq. :eq:`e_det_pt` and :eq:`e_stoc_pt`. A new keyword called :math:`\epsilon_{2Large}` is needed to distinguish which determinants will be treated stochastically and which will not. *Dice* calculates the energy correction using :eq:`e_stoc_pt` for both :math:`\epsilon_2` and :math:`\epsilon_{2Large}` and using :eq:`e_det_pt` for :math:`\epsilon_{2Large}`. We then subtract the less strict stochastic correction (:math:`E_2^S[\epsilon_2^d]`) from the strict stochastic correction (:math:`\Delta E_2^S[\epsilon_2]`) to reduce the stochastic noise before adding the deterministic contribution (:math:`\Delta E_2^D[\epsilon_2^d]`).

	.. math:: \Delta E_2[\epsilon_2] = \big( \Delta E_2^S[\epsilon_2] − \Delta E_2^S[\epsilon_2^d] \big) + \Delta E_2^D[\epsilon_2^d]
		:label: e_semi_pt

.. note::

	Although we have presented the input separated into three distinct sections, you do *not* have to write the keywords in any particular order, **but is is highly recommended** for clarity. For more details about the keywords used or other keywords please see the :ref:`dice-keywords` page.

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




Using SHCI as a stand-alone program
-----------------------------------
You can choose to use SHCI as a wavefunction solver without interfacing it with other programs. This will require a two-body integral file in the FCIDUMP format, which can be generated using whatever electronic structure package you prefer. If you'd like to use it as an active space solver in a CASSCF calculation see the :ref:`interfacing-with-pyscf` page.

Once you have generated your FCIDUMP file, you must create an input file that contains all of the parameters that SHCI needs to run a calculation. An example input.dat file is shown below. For more input files see the test directory inside your main SHCI directory.

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


Once you have your input.dat and FCIDUMP file you can open a terminal and navigate to the directory with both files and use the following command:

.. note::

	You can run *Dice* in parallel using OMP, MPI, or a hybrid of both, but we recommend that you use MPI. If you would like to run a hybrid scheme, please contact us to help you set it up.

.. code-block:: bash

	mpirun -np 2 /path_to_dice/Dice input.dat > output.dat


This will execute your input.dat file and write all output to the output.dat file in your current working directory. An example of the output is shown below:

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
	State :State :0
	State :State :0State :0
	State :0
	State :0
	State :0
	State :0
	0

	State :State :0
	State :0
	State :0
	State :0
	State :0
	State :0State :0
	State :0
	State :0
	State :0
	State :0
	State :0
	State :0
	State :0
	State :0
	State :0
	State :0
	0
	0

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
