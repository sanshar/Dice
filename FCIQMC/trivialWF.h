#ifndef TrivialWF_HEADER_H
#define TrivialWF_HEADER_H

// This is a trial wave function class for cases in FCIQMC where a
// trial wave function is not used (i.e. the original version of the
// algorithm). In this case, these functions are either not used or
// should return trivial results. This is a basic class to deal with
// this situation, together with the TrialWalk class.

#include "trivialWalk.h"

class TrivialWF {
 public:
  Determinant d;

  TrivialWF() {}
  TrivialWF(Determinant& det) : d(det) {}

  void HamAndOvlp(const TrivialWalk &walk, double &ovlp,
                  double &ham, workingArray& work, bool fillExcitations=true,
                  double epsilon=schd.epsilon) const {
    work.setCounterToZero();
    ovlp = 1.0;
    ham = 0.0;
  }

  void HamAndOvlpAndSVTotal(const TrivialWalk &walk, double &ovlp,
                            double &ham, double& SVTotal, workingArray& work,
                            const bool is, double epsilon=schd.epsilon) const {
    work.setCounterToZero();
    ovlp = 1.0;
    ham = 0.0;
    SVTotal = 0.0;
  }
  
  double parityFactor(Determinant& det, const int ex2, const int i,
                      const int j, const int a, const int b) const {
    return 1.0;
  }

  double getOverlapFactor(const TrivialWalk& walk, Determinant& dNew,
                          bool doparity) const {
    return 1.0;
  }

  double getOverlapFactor(int I, int J, int A, int B,
                          const TrivialWalk& walk, bool doparity) const {
    return 1.0;
  }

  double Overlap(const TrivialWalk& walk) const {
    return 1.0;
  }

  Determinant& getRef() {
    return d;
  }

  Determinant& getCorr() {
    return d;
  }

};

#endif
