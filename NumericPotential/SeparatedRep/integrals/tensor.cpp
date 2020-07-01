//#include "mkl.h"
#include "tensor.h"

int contract_IJK_IJL_LK(tensor *O, tensor *C, tensor *S, double scale, double beta) {
  /*
  int m = O->dimensions[0]*O->dimensions[1],
      n = S->dimensions[1],
      k = S->dimensions[0];
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
              scale, C->vals, k, S->vals, n, beta, O->vals, n); 
  */          

  int O1_dimension = (int)(O->dimensions[0]);
  int O2_dimension = (int)(O->dimensions[1]);
  int O3_dimension = (int)(O->dimensions[2]);
  double* O_vals = (double*)(O->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int C3_dimension = (int)(C->dimensions[2]);
  double* C_vals = (double*)(C->vals);
  int S1_dimension = (int)(S->dimensions[0]);
  int S2_dimension = (int)(S->dimensions[1]);
  double* S_vals = (double*)(S->vals);

  //#pragma omp parallel for schedule(static)
  for (int pO = 0; pO < ((O1_dimension * O2_dimension) * O3_dimension); pO++) {
    O_vals[pO] *= beta;
  }

  //#pragma omp parallel for schedule(runtime)
  for (int i = 0; i < C1_dimension; i++) {
    for (int j = 0; j < C2_dimension; j++) {
      int jO = i * O2_dimension + j;
      int jC = i * C2_dimension + j;
      for (int l = 0; l < S1_dimension; l++) {
        int lC = jC * C3_dimension + l;
        for (int k = 0; k < S2_dimension; k++) {
          int kO = jO * O3_dimension + k;
          int kS = l * S2_dimension + k;
          O_vals[kO] = O_vals[kO] + scale * C_vals[lC] * S_vals[kS];
        }
      }
    }
  }

  return 0;
}

