#include "global.h"
#include "walkersFCIQMC.h"

void stochastic_round(const double minPop, double& amp, bool& roundedUp) {
  auto random = bind(uniform_real_distribution<double>(0, 1), ref(generator));
  double pAccept = abs(amp)/minPop;
  if (random() < pAccept) {
    amp = copysign(minPop, amp);
    roundedUp = true;
  } else {
    amp = 0.0;
    roundedUp = false;
  }
}
