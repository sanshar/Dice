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

	#nocc 8
	#0 1 4 5 8 9 14 15
	#
	#sampleN 200
	#davidsontol 5.e-5
	#dE 1.e-7
	#DoRDM true
	#
	#schedule
	#0 9e-4
	#3 5e-4
	#end
	#
	#epsilon2 1.e-8
	#deterministic
	#noio
	#maxiter 12
	##nPTiter 0
	#
	#using seed: 1493622999
	#Making Helpers                                        0.00
	#HF = -75.3869032802436
	# 1  1  4
	#BetaN                                                 0.00
	#AlphaN-1                                              0.00
	#-------------Iter=   0---------------
	#Making Helpers                                        0.00
	# 177  57  100
	#BetaN                                                 0.00
	#AlphaN-1                                              0.00
	#niter:  0 root: -1 -> Energy :       -75.38690328
	#niter:  6 root:  0 -> Energy :        -75.4797232
	###########################################            0.01
	#-------------Iter=   1---------------
	#Initial guess(PT) :        -75.4797232
	#Making Helpers                                        0.01
	# 929  168  170
	#BetaN                                                 0.01
	#AlphaN-1                                              0.01
	#niter:  0 root: -1 -> Energy :        -75.4797232
	#niter:  7 root:  0 -> Energy :       -75.48399527
	###########################################            0.01
	#-------------Iter=   2---------------
	#Initial guess(PT) :       -75.48399527
	#Making Helpers                                        0.01
	# 959  168  170
	#BetaN                                                 0.01
	#AlphaN-1                                              0.01
	#niter:  0 root: -1 -> Energy :       -75.48399527
	#niter:  4 root:  0 -> Energy :       -75.48401296
	###########################################            0.02
	#-------------Iter=   3---------------
	#Initial guess(PT) :       -75.48401296
	#Making Helpers                                        0.02
	# 1691  234  198
	#BetaN                                                 0.02
	#AlphaN-1                                              0.02
	#niter:  0 root: -1 -> Energy :       -75.48401296
	#niter:  6 root:  0 -> Energy :       -75.48421962
	###########################################            0.03
	#-------------Iter=   4---------------
	#Initial guess(PT) :       -75.48421962
	#Making Helpers                                        0.03
	# 1705  234  198
	#BetaN                                                 0.03
	#AlphaN-1                                              0.03
	#niter:  0 root: -1 -> Energy :       -75.48421962
	#niter:  3 root:  0 -> Energy :       -75.48422229
	###########################################            0.03
	#-------------Iter=   5---------------
	#Initial guess(PT) :       -75.48422229
	#Making Helpers                                        0.03
	# 1705  234  198
	#BetaN                                                 0.03
	#AlphaN-1                                              0.03
	#niter:  0 root: -1 -> Energy :       -75.48422229
	#niter:  1 root:  0 -> Energy :       -75.48422229
	#Begin writing variational wf                          0.03
	#End   writing variational wf                          0.05
	E from 2RDM: -75.4842222865948
	### IMPORTANT DETERMINANTS FOR STATE: 0
	#0  -0.962972405251402  0.962972405251402  2 0 2 0 2   0 0 2 0 0   0 0
	#1  0.113332103493894  0.113332103493894  2 0 2 0 0   0 0 2 0 0   2 0
	#2  0.113332103493883  0.113332103493883  2 0 0 0 2   0 0 2 0 2   0 0
	#3  0.0778346380808918  0.0778346380808918  2 0 b 0 a   0 0 2 0 a   b 0
	#4  0.0778346380808912  0.0778346380808912  2 0 a 0 b   0 0 2 0 b   a 0
	#5  0.0620766498642539  0.0620766498642539  2 0 2 0 2   0 0 0 0 0   2 0
	### PERFORMING PERTURBATIVE CALCULATION
	# 0
	#Before hash 0.104489803314209
	#After hash 0.110838890075684
	#After all_to_all 0.163533926010132
	#After collecting 0.178984880447388
	#Unique determinants 0.178997993469238
	#Done energy -75.4844111804806  0.180866956710815
