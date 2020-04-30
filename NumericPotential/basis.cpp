#include "pythonInterface.h"
#include <iostream>
#include <vector>
#include <omp.h>
#include <stdio.h>
#include <pvfmm.hpp>
#include <ctime>

using namespace std;

//the coords are nx3 matrix in "c" order but libcint requires it to be in
//"f" order
void getValuesAtGrid(const double* grid_cformat, int ngrid, double* out) {
  int nao = ao_loc[shls[1]] - ao_loc[shls[0]];

  
  if (aovals.size() < nao*ngrid) {
    aovals.resize(nao*ngrid, 0.0);
    intermediate.resize(nao*ngrid, 0.0);
  }
  if (grid_fformat.size() < 3*ngrid)
    grid_fformat.resize(ngrid*3);

  std::fill(aovals.begin(), aovals.end(), 0.);
  std::fill(out, out+ngrid, 0.);
  std::fill(&intermediate[0], &intermediate[0]+nao*ngrid, 0.);
  std::fill(&grid_fformat[0], &grid_fformat[0]+3*ngrid, 0.);

  long lena[2]   = {ngrid, nao}, stridea[2]   = {1, ngrid}; //aovals, intermediate
  long lenc[2]   = {3, ngrid}  , stridec[2]   = {1, 3};     //grid-cformat
  long lenf[2]   = {ngrid, 3}  , stridef[2]   = {1, ngrid}; //grid-fformat
  long lendm[2]  = {nao, nao}  , stridedm[2]  = {1, nao}; //density matrix
  long lenout[1] = {ngrid}     , strideout[1] = {1}; //density matrix

  tblis_init_tensor_d(&AOVALS,       2, lena,   &aovals[0],                            stridea);
  tblis_init_tensor_d(&INTERMEDIATE, 2, lena,   &intermediate[0],                      stridea);
  tblis_init_tensor_d(&GRID_C,       2, lenc,   const_cast<double*>(&grid_cformat[0]), stridec);
  tblis_init_tensor_d(&GRID_F,       2, lenf,   &grid_fformat[0],                      stridef);
  tblis_init_tensor_d(&DM,           2, lendm,  dm,                                   stridedm);
  tblis_init_tensor_d(&OUT,          1, lenout, out,                                 strideout);

  //make the f-format from c-format
  tblis_tensor_add(NULL, NULL, &GRID_C, "ij", &GRID_F, "ji");
  //now you have to scale the basis such that centroid is 0.5
  for (int i=0; i<ngrid; i++) 
    grid_fformat[i] = (grid_fformat[i]-0.5)*coordScale + centroid[0];
  for (int i=0; i<ngrid; i++) 
    grid_fformat[ngrid+i] = (grid_fformat[ngrid+i]-0.5)*coordScale + centroid[1];
  for (int i=0; i<ngrid; i++) 
    grid_fformat[2*ngrid+i] = (grid_fformat[2*ngrid+i]-0.5)*coordScale + centroid[2];


  int tabSize = (((ngrid + BLKSIZE-1)/BLKSIZE) * nbas)*10;
  if (non0tab.size() < tabSize)
    non0tab.resize(tabSize, 1);

  GTOval_cart(ngrid, shls, ao_loc, &aovals[0], &grid_fformat[0],
              &non0tab[0], atm, natm, bas, nbas, env);
  
  tblis_tensor_mult(NULL, NULL, &AOVALS, "gp", &DM, "pq", &INTERMEDIATE, "gq");
  tblis_tensor_mult(NULL, NULL, &INTERMEDIATE, "gp", &AOVALS, "gp", &OUT, "g");
  ncalls += ngrid;
  
}


void getPotential(int nTarget, double* coordsTarget, double* potential, double* pdm,
                  double* pcentroid, double pscale, double tol) {
  cout.precision(12);
  ncalls = 0;
  dm = pdm;
  centroid = pcentroid;
  coordScale = pscale;
  int rank;MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  time_t Initcurrent_time, current_time;
  Initcurrent_time = time(NULL);
  
  MPI_Comm comm = MPI_COMM_WORLD;
  int cheb_deg = 10, max_pts = 100, mult_order = 10;

  vector<double> trg_coord(3*nTarget), input_coord(3*nTarget);
  for (int i=0; i<nTarget; i++) {

    trg_coord[3*i+0] = (coordsTarget[3*i+0] - centroid[0])/coordScale + 0.5;
    trg_coord[3*i+1] = (coordsTarget[3*i+1] - centroid[1])/coordScale + 0.5;
    trg_coord[3*i+2] = (coordsTarget[3*i+2] - centroid[2])/coordScale + 0.5;
  }

  const pvfmm::Kernel<double>& kernel_fn=pvfmm::LaplaceKernel<double>::potential();

  auto* tree=ChebFMM_CreateTree(cheb_deg, kernel_fn.ker_dim[0],
                                getValuesAtGrid,
                                trg_coord, comm, tol, max_pts, pvfmm::FreeSpace);

  current_time = time(NULL);
  if (rank == 0) cout << "make tree "<<(current_time - Initcurrent_time)<<endl;
  // Load matrices.
  pvfmm::ChebFMM<double> matrices;
  matrices.Initialize(mult_order, cheb_deg, comm, &kernel_fn);

  // FMM Setup
  tree->SetupFMM(&matrices);
  current_time = time(NULL);
  if (rank == 0) cout << "fmm setup "<<(current_time - Initcurrent_time)<<endl;

  // Run FMM
  std::vector<double> trg_value;
  size_t n_trg=trg_coord.size()/PVFMM_COORD_DIM;
  pvfmm::ChebFMM_Evaluate(tree, trg_value, n_trg);

  current_time = time(NULL);

  if (rank == 0) cout << "evalstep setup "<<(current_time - Initcurrent_time)<<endl;

  for (int i=0; i<nTarget; i++)
    potential[i] = trg_value[i]*4*M_PI;

}

