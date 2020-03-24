#ifndef HAMILTONIAN_DENSE_HPP
#define HAMILTONIAN_DENSE_HPP

#include <iomanip>
#include <iostream>
#include "SHCImakeHamiltonian.h"
#include "communicate.h"
#include "global.h"

class HamiltonianDense {
 private:
  const size_t n_ci;
  MatrixXx ham_matrix;

 public:
  HamiltonianDense(SparseHam& sp_ham) : n_ci(sp_ham.Helements.size()) {
    ham_matrix.resize(n_ci, n_ci);
    for (size_t I = 0; I < sp_ham.connections.size(); I++) {
      for (size_t j = 0; j < sp_ham.connections[I].size(); j++) {
        CItype h_IJ = sp_ham.Helements[I][j];
        size_t J = sp_ham.connections[I][j];

        ham_matrix(I, J) = h_IJ;

#ifdef Complex
        ham_matrix(J, I) = std::conj(h_IJ);
#else
        ham_matrix(J, I) = h_IJ;

#endif
      }
    }
  }
  void print(int precision = 6) {
    std::cout << std::setprecision(precision);
    pout << ham_matrix << std::endl;
  }
  void diagonalize() {
    Eigen::SelfAdjointEigenSolver<MatrixXx> es(ham_matrix);
    // pout << es.eigenvalues() +
    //             Eigen::Matrix<CItype, Eigen::Dynamic, 1>::Ones(n_ci) * coreE
    //      << std::endl;
  }
};
#endif