## Guide to phaseless AFQMC calculations

This directory contains examples of phaseless AFQMC calculations in mean_field_trials and hci_trials. Preparation for and running of the calculations is handled in a single python script $system_name.py. Preparation consists of mean-field calculations, integral generation, and trial state preparation. HCI trial states are calculated using Dice. The script writes input files for Dice and DQMC binaries, executes them, and performs blocking analysis of AFQMC samples to produce an estimate of the energy and stochastic error. The following lines at the beginning of the script should be changed to point to the users' paths:

```
dice_binary = "/projects/anma2640/newDice/Dice/Dice"
vmc_root = "/projects/anma2640/VMC/master/VMC"
```

vmc_root/QMCUtils.py contains tools used for setting up QMC calculations and needs to be in the PYTHONPATH. Details of Dice input options specified in the script can be found in the [Dice documentaion](https://sanshar.github.io/Dice/). 

AFQMC calculations can be divided into two parts: propagation and measurement. Trial states affect both these parts; and their accuracy dictates the systematic error (phaseless error) and efficiency (statistical noise) of the AFQMC calculation. Following is a rough description of the relevant arguments accepted by the run_afqmc function (with default values in parentheses):

1. Required argument:
  * mf_or_mc: pyscf mean field object if using the mean field trial or a casscf object if using HCI trial
  
2. DQMC binary related arguments:
  * vmc_root (None): path for the VMC code, if not provided it will try to read the VMC_ROOT environment variable
  * nproc (None): number of MPI processes to be launched, if not provided all available cores are used
  * run_dir (None): directory where AFQMC calculations are performed, to be used to prevent overwriting in case of multiple calculations

3. System arguments:
  * norb_frozen (0): number of orbitals to freeze in AFQMC, these are not correlated but affect the one-body potential
  * chol_cut (1e-5): threshold used in calculating Cholesky integrals

4. Sampling arguments:
  * seed (None): random number seed to be able to repoduce results
  * dt (0.005): Trotter propagation time-step, the default is pretty conservative
  * nwalk_per_proc (5): number of walkers per process, there need to be enough of these in total to minimize population control bias
  * nblocks (1000): number of energy measurements performed for each walker, there should be enough of these to allow a blocking analysis for error estimation
  * steps_per_block (50): number of propagation steps per block
  * burn_in (50): number of equilibration steps to throw away, may need to be adjusted post-hoc for a separate blocking analysis

1. HCI trial related arguments:
  * ndets (100): number of leading dets to pick from the dets file, can be a list
  * nroot (0): for targetting excited states if corresponding det files have been generated

The three options "steps_per_block", "nwalk_per_proc", and "nblocks" determine the number of samples collected (= nproc * nwalk_per_proc * nblock) and the correlation length (inversely propportional to nsteps). Thus the noise and the cost of the calculation is dictated by these options. They need to be adjusted depending on the system. I usually use the default values noted above and increase the number of processes for bigger and more difficult systems. 


## Miscellaneous 
1. One useful thing to keep in mind when choosing sampling options is that since the noise in QMC scales as 1 / sqrt(number of samples), reducing the noise by a factor of two requires roughly four times as many samples!
2. For HCI trial states, using more dets leads to less noisy and usually less biased estimators but at a higher cost. We find it useful to pick determinants from a moderately sized active space from which converged energies can be obtained with less than ~100k dets.
3. Blocking analysis is performed with the script vmc_root/scripts/blocking.py to estimate stochastic errors from the serially correlated samples. To estimate the stochastic error, one needs to look for an intermediate plateau in the error as a function of the block size. Consider the following example:

    ```
    reading samples from samples_rhf.dat, ignoring first 50
    mean: -5.382689152884524
    blocked statistics:
    block size    # of blocks        mean                error
         1            450       -5.38268915e+00       5.397366e-04
         2            225       -5.38268915e+00       7.067836e-04
         5             90       -5.38268915e+00       1.029813e-03
        10             45       -5.38268915e+00       1.039668e-03
        20             22       -5.38266672e+00       1.036828e-03
        50              9       -5.38268915e+00       1.512644e-03
        70              6       -5.38272344e+00       1.395108e-03
       100              4       -5.38261626e+00       1.733287e-03
       200              2       -5.38261626e+00       3.212095e-03
    ```

The first 50 samples were discarded as equilibration steps. The error column shows increasing values of the sample variance as block size is increased indicating serial correlation. It is clear that there is a plateau in error values around 1.e-3 mH for block sizes of 5-20 after which the errors seem to diverge because of small number of blocks. So the energy in this example can be reported as -5.383(1) H.
4. All parallelization is done through MPI and thereading should be disabled so as not to interfere with MPI. The parallelization is not perfect because there needs to be communication for population control.
5. I use DQMC and AFQMC interchangeably in many places and not in others. For example, the dqmc examples in this directory refer to free projection, but the binary DQMC is used for both free projection AFQMC and phaseless AFQMC calculations.
