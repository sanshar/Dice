#include "CalculateSphHarmonics.h"
#include <Eigen/Dense>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include "pythonInterface.h"
#include <boost/math/interpolators/cubic_b_spline.hpp>
#include <stdio.h>
#include <pvfmm.hpp>
#include <ctime>

//#### LIBCINT VARIABLES ######
int *shls, *ao_loc, *atm, *bas;
int natm, nbas, ncalls;
double *env, *dm, *centroid, coordScale;
std::vector<char> non0tab;
int BLKSIZE;
std::vector<double> aovals, intermediate, grid_fformat;
tblis_tensor AOVALS, INTERMEDIATE, GRID_C, GRID_F, DM, OUT;


//#### BECKE GRID VARIABLES ######
vector<RawDataOnGrid> densityYlm;
vector<RawDataOnGrid> potentialYlm;  
vector<SplineFit> splinePotential;
MatrixXdR RawDataOnGrid::lebdevgrid;  
MatrixXd RawDataOnGrid::SphericalCoords;
MatrixXd RawDataOnGrid::sphVals;
MatrixXd RawDataOnGrid::WeightedSphVals;
VectorXd RawDataOnGrid::densityStore;
int RawDataOnGrid::lmax;
vector<int> RawDataOnGrid::lebdevGridSize{ 1, 6, 14, 26, 38, 50, 74, 86, 110, 146, 170,
      194, 230, 266, 302, 350, 434, 590, 770, 974, 1202, 1454, 1730, 2030,
      2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810 };

void init(int* pshls, int *pao_loc, int *patm, int pnatm,
          int *pbas, int pnbas, double *penv) {
  BLKSIZE = 128;
  shls    = pshls;
  ao_loc  = pao_loc;
  atm     = patm;
  natm    = pnatm;
  bas     = pbas;
  nbas    = pnbas;
  env     = penv;
}

void getBeckePartition(double* coords, int ngrids, double* pbecke) {
  vector<double> fcoord(ngrids*3);
  vector<double> atmCoord(natm*3);

  for (int i=0; i<ngrids; i++) {
    fcoord[0*ngrids+i] = coords[i*3+0];
    fcoord[1*ngrids+i] = coords[i*3+1];
    fcoord[2*ngrids+i] = coords[i*3+2];
  }
  for (int i=0; i<natm; i++) {
    atmCoord[3*i+0] = env[atm[6*i+1] + 0];
    atmCoord[3*i+1] = env[atm[6*i+1] + 1];
    atmCoord[3*i+2] = env[atm[6*i+1] + 2];
  }

  VXCgen_grid(pbecke, &fcoord[0], &atmCoord[0], NULL, natm, ngrids);
}

void initAtomGrids() {
  int natom = natm;
  densityYlm.resize(natom);
  potentialYlm.resize(natom);
  splinePotential.resize(natom);
  for (int i=0; i<natom; i++) {
    densityYlm[i].coord(0) = env[atm[6*i+1] + 0];
    densityYlm[i].coord(1) = env[atm[6*i+1] + 1];
    densityYlm[i].coord(2) = env[atm[6*i+1] + 2];
    
    potentialYlm[i].coord = densityYlm[i].coord;
    splinePotential[i].coord = densityYlm[i].coord;
  }
}

void initAngularData(int lmax, int lebdevOrder){
  RawDataOnGrid::InitStaticVariables(lmax, lebdevOrder);
}

void initDensityRadialGrid(int ia, int rmax, double rm) {
  densityYlm[ia].InitRadialGrids(rmax, rm);
}

void fitDensity(int ia, int rindex) {
  int GridSize = RawDataOnGrid::lebdevgrid.rows();

  MatrixXdR grid(GridSize, 3);
  grid = RawDataOnGrid::lebdevgrid.block(0,0,GridSize, 3) * densityYlm[ia].radialGrid(rindex);

  for (int i=0; i<grid.rows(); i++) {
    grid(i,0) += densityYlm[ia].coord(0);
    grid(i,1) += densityYlm[ia].coord(1);
    grid(i,2) += densityYlm[ia].coord(2);
  }
  
  vector<double> density(GridSize, 0.0), becke(natm*GridSize, 0.0), beckenorm(GridSize,0.0);
  getValuesAtGrid(&grid(0,0), GridSize, &density[0]);
  getBeckePartition(&grid(0,0), GridSize, &becke[0]);

  for (int i = 0; i<GridSize; i++) {
    for (int a =0; a<natm; a++)
      beckenorm[i] += becke[a * GridSize + i];
  }
  
  for (int i = 0; i<GridSize; i++) {
    density[i] *= becke[ia*GridSize + i]/beckenorm[i];
  }
  densityYlm[ia].fit(rindex, &density[0]);
}

//void getDensityOnLebdevGridAtGivenR(int rindex, double* densityOut) {
//densityYlm.getValue(rindex, densityOut);
//}

void solvePoissonsEquation(int ia) {
  double totalQ = densityYlm[ia].wts.dot(densityYlm[ia].GridValues * densityYlm[ia].lebdevgrid.col(3));

  potentialYlm[ia] = densityYlm[ia];

  int nr = densityYlm[ia].radialGrid.size();
  double h = 1./(1.*nr+1.), rm = densityYlm[ia].rm;
  VectorXd drdz(nr), d2rdz2(nr), b(nr);
  VectorXd& z = densityYlm[ia].zgrid, &r = densityYlm[ia].radialGrid;

  for (int i=0; i<nr; i++) {
    drdz(i)   = (-2.*M_PI * rm * sin(M_PI * z(i))) / pow(1. - cos(M_PI * z(i)), 2);
    d2rdz2(i) = ( 2.* M_PI * M_PI * rm) *
        (2 * pow(sin(M_PI * z(i)), 2.) + pow(cos(M_PI * z(i)), 2) - cos(M_PI * z(i)))
        / pow(1. - cos(M_PI * z(i)), 3);
  }

  cout << ia<<"  "<<nr<<"  "<<totalQ<<endl;
  MatrixXd A1Der(nr, nr), A2Der(nr, nr), A(nr, nr);
  A1Der.setZero(); A2Der.setZero(); A.setZero();
  A2Der.block(0,0,1,6) <<  -147., -255.,  470., -285.,  93., -13.;
  A1Der.block(0,0,1,6) <<   -77.,  150., -100.,   50., -15.,   2.;

  A2Der.block(1,0,1,6) <<  228., -420.,  200.,   15., -12.,   2.; 
  A1Der.block(1,0,1,6) <<  -24.,  -35.,   80.,  -30.,   8.,  -1.;

  A2Der.block(2,0,1,6) <<  -27.,  270., -490.,  270., -27.,   2.;
  A1Der.block(2,0,1,6) <<    9.,  -45.,    0.,   45.,  -9.,   1.;

  for (int i=3; i<nr-3; i++) {
    A2Der.block(i,i-3,1,7) <<  2., -27.,  270., -490.,  270., -27.,   2.;
    A1Der.block(i,i-3,1,7) << -1.,   9.,  -45.,    0.,   45.,  -9.,   1.;
  }

  A2Der.block(nr-1, nr-6, 1, 6) <<  -13.,   93., -285.,  470.,-255.,-147.;
  A1Der.block(nr-1, nr-6, 1, 6) <<   -2.,   15.,  -50.,  100.,-150.,  77.;

  A2Der.block(nr-2, nr-6, 1, 6) <<   2.,  -12.,   15.,  200.,-420., 228.;
  A1Der.block(nr-2, nr-6, 1, 6) <<   1.,   -8.,   30.,  -80.,  35.,  24.;
  
  A2Der.block(nr-3, nr-6, 1, 6) <<  2., -27.,  270., -490.,  270., -27.;
  A1Der.block(nr-3, nr-6, 1, 6) << -1.,   9.,  -45.,    0.,   45.,  -9.;

  int lmax = densityYlm[ia].lmax;
  for (int l = 0; l < lmax; l++) {
    for (int m = -l; m < l+1; m++) {
      int lm = l * l + (l + m);

      b = -4 * M_PI * r.cwiseProduct(densityYlm[ia].CoeffsYlm.col(lm));

      if (l == 0) {
        b(0) += sqrt(4 * M_PI) * (-137. / (180. * h * h)/(drdz(0) * drdz(0))) * totalQ;
        b(1) += sqrt(4 * M_PI) * (  13. / (180. * h * h)/(drdz(1) * drdz(1))) * totalQ;
        b(2) += sqrt(4 * M_PI) * (  -2. / (180. * h * h)/(drdz(2) * drdz(2))) * totalQ;
        
        b(0) += sqrt(4 * M_PI) * (  10. / (60. * h)*(-d2rdz2(0) / pow(drdz(0),3))) * totalQ;
        b(1) += sqrt(4 * M_PI) * (  -2. / (60. * h)*(-d2rdz2(1) / pow(drdz(1),3))) * totalQ;
        b(2) += sqrt(4 * M_PI) * (   1. / (60. * h)*(-d2rdz2(2) / pow(drdz(2),3))) * totalQ;
      }
      
      for (int i=0; i<nr; i++) {
        A.row(i) = A2Der.row(i) / (180. * h * h)/(drdz(i) * drdz(i))
            + A1Der.row(i) / (60 * h) * (-d2rdz2(i) / pow(drdz(i), 3));
        
        A(i, i) += (-l * (l + 1) / r(i) / r(i)); 
      }

      potentialYlm[ia].CoeffsYlm.col(lm) = A.colPivHouseholderQr().solve(b);
    }
  }
}

void fitSplinePotential(int ia) {
  splinePotential[ia].Init(potentialYlm[ia]);
}

void getPotentialBeckeOfAtom(int ia, int ngrid, double* grid, double* potential) {
  splinePotential[ia].getPotential(ngrid, grid, potential);
}

void getPotentialBecke(int ngrid, double* grid, double* potential, int lmax,
                       int* nrad, double* rm, double* pdm) {
  coordScale = 1.0;
  centroid = new double[3];
  centroid[0] = 0.5; centroid[1] = 0.5; centroid[2] = 0.5;

  dm = pdm;
  int lebedevOrder = 2*lmax + 1;

  initAngularData(lmax, lebedevOrder);
  initAtomGrids();

  for (int ia=0; ia<natm; ia++) { 
    initDensityRadialGrid(ia, nrad[ia], rm[ia]);

    for (int rindex = 0; rindex < nrad[ia] ; rindex++)
      fitDensity(ia, rindex);

    solvePoissonsEquation(ia);
    fitSplinePotential(ia);

    getPotentialBeckeOfAtom(ia, ngrid, grid, potential);
  }
  delete [] centroid;
}

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


void getPotentialFMM(int nTarget, double* coordsTarget, double* potential, double* pdm,
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

