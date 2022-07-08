#ifndef TrivialWalk_HEADER_H
#define TrivialWalk_HEADER_H

// This is a trial wave function walker for cases in FCIQMC where a
// trial wave function is not used (i.e. the original version of the
// algorithm). In this case, these functions are either not used or
// should return trivial results. This is a basic class to deal with
// this situation, together with the TrialWF class.

class TrivialWalk {
 public:
  Determinant d;

  TrivialWalk() {}

  template<typename Wave>
  TrivialWalk(Wave& wave, Determinant& det) : d(det) {}

  void updateWalker(const Determinant& ref, const Determinant& corr,
                    int ex1, int ex2, bool updateIntermediate=true) {}
};

#endif
