#ifndef dataFCIQMC_HEADER_H
#define dataFCIQMC_HEADER_H

class dataFCIQMC {

  public:
    // The total walker population (on this process only)
    vector<double> walkerPop;
    // The numerator of the projected energy (against the HF determinant)
    vector<double> EProj;
    // The walker population on the HF determinant
    vector<double> HFAmp;

    // The numerator of the trial-WF-based projected energy estimator
    vector<double> trialEProj;
    // The sum of all walker amplitudes (the denominator of the trial
    // WF-based projected energy estimator)
    vector<double> ampSum;

    // The total number of walkers annihilated (on this process only)
    vector<double> nAnnihil;

    // The shift to the diagonal of the Hamiltonian
    vector<double> Eshift;
    // True is the shift has started to vary (left the constant phase)
    vector<bool> varyShift;

    // All of the following quantities are summed over all processes:

    // The total walker population
    vector<double> walkerPopTot;
    // The total walker population from the previous iteration
    vector<double> walkerPopOldTot;
    // The numerator of the projected energy (against the HF determinant)
    vector<double> EProjTot;
    // The walker population on the HF determinant
    vector<double> HFAmpTot;
    // The numerator of the trial-WF-based projected energy
    vector<double> trialEProjTot;
    // The denominator of the trial-WF-based projected energy
    vector<double> ampSumTot;
    // The total number of walkers annihilated
    vector<double> nAnnihilTot;

    // Estimates requiring two replicas simulations:

    // The numerator and denominator of the variational energy
    double EVarNumAll;
    double EVarDenomAll;
    // The numerator of the EN2 perturbative correction
    double EN2All;


    dataFCIQMC(int nreplicas, double initEshift) {

      walkerPop.resize(nreplicas, 0.0);
      EProj.resize(nreplicas, 0.0);
      HFAmp.resize(nreplicas, 0.0);
      trialEProj.resize(nreplicas, 0.0);
      ampSum.resize(nreplicas, 0.0);

      Eshift.resize(nreplicas, initEshift);
      varyShift.resize(nreplicas, false);

      walkerPopTot.resize(nreplicas, 0.0);
      walkerPopOldTot.resize(nreplicas, 0.0);
      EProjTot.resize(nreplicas, 0.0);
      HFAmpTot.resize(nreplicas, 0.0);
      trialEProjTot.resize(nreplicas, 0.0);
      ampSumTot.resize(nreplicas, 0.0);

      nAnnihil.resize(nreplicas, 0.0);
      nAnnihilTot.resize(nreplicas, 0.0);

      EVarNumAll = 0.0;
      EVarDenomAll = 0.0;
      EN2All = 0.0;

    }

};

#endif
