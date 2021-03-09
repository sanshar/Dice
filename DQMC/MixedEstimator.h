#ifndef MixedEstimator_HEADER_H
#define MixedEstimator_HEADER_H
#include "DQMCWalker.h"

void calcMixedEstimator(Wavefunction& waveLeft, Wavefunction& waveRight, DQMCWalker& walker, Hamiltonian& ham);

void calcMixedEstimatorNoProp(Wavefunction& waveLeft, Wavefunction& waveRight, DQMCWalker& walker, Hamiltonian& ham);

#endif
