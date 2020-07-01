#pragma once

#include <vector>

struct Coulomb {
  std::vector<double> exponents;
  std::vector<double> weights;
};

struct Coulomb_14_8_8 : public Coulomb {
  Coulomb_14_8_8();
};

struct Coulomb_14_14_8 : public Coulomb{
  Coulomb_14_14_8();
};
