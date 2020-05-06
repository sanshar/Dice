#include "CalculateSphHarmonics.h"
#include <Eigen/Dense>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include "pythonInterface.h"
#include <boost/math/interpolators/cubic_b_spline.hpp>
#include <stdio.h>
#include <pvfmm.hpp>
#include <ctime>
#include <boost/math/quadrature/gauss.hpp>
#include "mkl.h"

//#### LIBCINT VARIABLES ######
int *shls, *ao_loc, *atm, *bas;
int natm, nbas, ncalls;
double *env, *dm, *centroid, coordScale;
std::vector<char> non0tab;
int BLKSIZE;
std::vector<double> aovals, intermediate, grid_fformat;

//#FMM TREE
pvfmm::ChebFMM_Tree<double>* tree;
vector<double> LegCoord, LegWts;
vector<double> AoValsAtLegCoord;

vector<double> ChebCoord;
vector<double> AoValsAtChebCoord;
vector<double> density;

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

void getAoValuesAtGrid(const double* grid_cformat, int ngrid, double* aovals) {
  int nao = ao_loc[shls[1]] - ao_loc[shls[0]];
  if (grid_fformat.size() < 3*ngrid)
    grid_fformat.resize(ngrid*3);

  std::fill(&grid_fformat[0], &grid_fformat[0]+3*ngrid, 0.);

  for (int i=0; i<ngrid; i++)
    for (int j=0; j<3; j++)
      grid_fformat[i + j*ngrid] = grid_cformat[j + i*3];

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


  std::fill(&aovals[0], &aovals[0]+nao*ngrid, 0.);
  std::fill(out, out+ngrid, 0.);

  for (int i=0; i<ngrid; i++)
    for (int j=0; j<3; j++)
      grid_fformat[i + j*ngrid] = grid_cformat[j + i*3];
  
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
  
  char N = 'N', T = 'T'; double alpha = 1.0, beta = 0.0; int LDA = 1, LDB = 1, LDC=1;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              ngrid, nao, nao, alpha, &aovals[0], ngrid, &dm[0], nao, beta,
              &intermediate[0], ngrid);

  for (int p = 0; p<nao; p++)
  for (int g = 0; g<ngrid; g++)
    out[g] += intermediate[p*ngrid + g] * aovals[p*ngrid + g];
  ncalls += ngrid;  
}

void getLegCoords(int dim, vector<double>& Coord,
                  vector<double>& Weights) {
  boost::math::quadrature::gauss<double, 7> gaussLegendre;
  auto abscissa = boost::math::quadrature::gauss<double, 7>::abscissa();
  auto weights = boost::math::quadrature::gauss<double, 7>::weights();

  vector<double> coords1D(7,0), weights1D(7, 0.0); 
  coords1D[0] = -abscissa[3]; coords1D[1] = -abscissa[2]; coords1D[2] = -abscissa[1];
  coords1D[3] =  abscissa[0];
  coords1D[4] =  abscissa[1]; coords1D[5] =  abscissa[2]; coords1D[6] =  abscissa[3];
  
  weights1D[0] =  weights[3]; weights1D[1] =  weights[2]; weights1D[2] =  weights[1];
  weights1D[3] =  weights[0];
  weights1D[4] =  weights[1]; weights1D[5] =  weights[2]; weights1D[6] =  weights[3];
  
  Coord.resize(7*7*7*3, 0.0); Weights.resize(7*7*7, 0.0);
  for (int x =0; x<7; x++)
  for (int y =0; y<7; y++)
  for (int z =0; z<7; z++)
  {
    Coord[3 * ( x * 49 + y * 7 + z) + 0] = 0.5*coords1D[x]+0.5;
    Coord[3 * ( x * 49 + y * 7 + z) + 1] = 0.5*coords1D[y]+0.5;
    Coord[3 * ( x * 49 + y * 7 + z) + 2] = 0.5*coords1D[z]+0.5;
    Weights[( x * 49 + y * 7 + z)] = weights1D[x] * weights1D[y] * weights1D[z];
  }
}

void initFMMGridAndTree(double* pdm, double* pcentroid, double pscale, double tol) {
  cout.precision(12);
  dm = pdm;
  centroid = pcentroid;
  coordScale = pscale;


  
  time_t Initcurrent_time, current_time;
  Initcurrent_time = time(NULL);
  
  MPI_Comm comm = MPI_COMM_WORLD;
  int cheb_deg = 10, max_pts = 10000, mult_order = 10;

  const pvfmm::Kernel<double>& kernel_fn=pvfmm::LaplaceKernel<double>::potential();

  int leafNodes = 0;

  //generate the gauss-legendre and gaiss-chebyshev coords for each leaf node
  //which will be used for integration

  vector<double> dummy(3,0.0);
  tree=ChebFMM_CreateTree(cheb_deg, kernel_fn.ker_dim[0],
                          getValuesAtGrid,
                          dummy, comm, tol, max_pts, pvfmm::FreeSpace);

  auto nodes = tree->GetNodeList();
  for (int i=0; i<nodes.size(); i++)
    if (nodes[i]->IsLeaf()) 
      leafNodes++;
    
  size_t nLeg  = 7*7*7;
  size_t nCheb = (cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);
  size_t numCoords = leafNodes * nLeg;
  size_t index = 0;
  double volume = 0.0;
  
  int legdeg = 7;
  vector<double> Legcoord, Legwts;
  getLegCoords(nodes[0]->Dim(), Legcoord, Legwts);
  LegCoord.resize(numCoords*3, 0.0); LegWts.resize(numCoords, 0.0);
  
  vector<double> nodeTrgCoords(nLeg*3, 0.0);

  //legendre coord for each leaf node and make them also the target coords
  //so that the potential is calculated at those points
  for (int i=0; i<nodes.size(); i++)
    if (nodes[i]->IsLeaf()) {
      double s=pvfmm::pow<double>(0.5,nodes[i]->Depth());
      double wt = s*s*s/8;
      volume += wt;
      for (int j=0; j<nLeg; j++) {
        LegCoord[index*3+0] = Legcoord[j*3+0]*s+nodes[i]->Coord()[0];
        LegCoord[index*3+1] = Legcoord[j*3+1]*s+nodes[i]->Coord()[1];
        LegCoord[index*3+2] = Legcoord[j*3+2]*s+nodes[i]->Coord()[2];
        
        LegWts[index] = Legwts[j] * wt;//j == 0 ? wt: 0.0;//wt;//s*s*s * scal;
        
        nodeTrgCoords[j*3+0] = LegCoord[index*3+0];
        nodeTrgCoords[j*3+1] = LegCoord[index*3+1];
        nodeTrgCoords[j*3+2] = LegCoord[index*3+2];
        
        nodes[i]->trg_coord = nodeTrgCoords;
        
        
        index++;
      }
    }

  
  //chebyshev coord
  ChebCoord.resize(leafNodes*nCheb*3, 0.0);
  index = 0;
  std::vector<double> Chebcoord=pvfmm::cheb_nodes<double>(cheb_deg,nodes[0]->Dim());
  for (int i=0; i<nodes.size(); i++)
    if (nodes[i]->IsLeaf()) {
      double s=pvfmm::pow<double>(0.5,nodes[i]->Depth());
      for (int j=0; j<nCheb; j++) {
        ChebCoord[index*3+0] = Chebcoord[j*3+0]*s+nodes[i]->Coord()[0];
        ChebCoord[index*3+1] = Chebcoord[j*3+1]*s+nodes[i]->Coord()[1];
        ChebCoord[index*3+2] = Chebcoord[j*3+2]*s+nodes[i]->Coord()[2];
        index++;
      }
    }

  int nao = ao_loc[shls[1]] - ao_loc[shls[0]];
  
  //calculate the ao values at the legendre coordinates
  AoValsAtLegCoord.resize(LegCoord.size() * nao/3, 0.0);
  getAoValuesAtGrid(&LegCoord[0], LegCoord.size()/3, &AoValsAtLegCoord[0]);

  //calculate the ao values at the chebyshev coordinates
  AoValsAtChebCoord.resize(ChebCoord.size() * nao/3, 0.0);
  getAoValuesAtGrid(&ChebCoord[0], ChebCoord.size()/3, &AoValsAtChebCoord[0]);
  
  
}

void writeNorm(double* mat, size_t size) {
  double norm = 0.0;
  for (int i=0; i<size; i++)
    norm += mat[i];
  cout << norm<<endl;
}

void getFockFMM(double* pdm, double* fock, double tol) {
  int rank;MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm comm = MPI_COMM_WORLD;
  const pvfmm::Kernel<double>& kernel_fn=pvfmm::LaplaceKernel<double>::potential();
  time_t Initcurrent_time, current_time;
  Initcurrent_time = time(NULL);

  
  //construct the density from density matrix and ao values stored on chebyshev grid
  int cheb_deg = 10, mult_order = 10;
  int nao = ao_loc[shls[1]] - ao_loc[shls[0]];

  {
    int ngrid = ChebCoord.size()/3;
    if (intermediate.size() < nao*ngrid) {
      intermediate.resize(nao*ngrid, 0.0);
    }

    double alpha = 1.0, beta = 0.0;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                ngrid, nao, nao, alpha, &AoValsAtChebCoord[0], ngrid, &pdm[0], nao, beta,
                &intermediate[0], ngrid);
    
    if (density.size() < ngrid) density.resize(ngrid, 0.0);
    std::fill(&density[0], &density[0]+ngrid, 0.0);
    
    for (int p = 0; p<nao; p++)
      for (int g = 0; g<ngrid; g++)
        density[g] += intermediate[p*ngrid + g] * AoValsAtChebCoord[p*ngrid + g];

  }


  //use the density on the chebyshev grid points to fit the
  //chebyshev polynomials of the leaf nodes
  int index = 0, nCheb = (cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);;
  auto nodes = tree->GetNodeList();
  for (int i=0; i<nodes.size(); i++)
    if (nodes[i]->IsLeaf()) {
      nodes[i]->ChebData().SetZero();
      pvfmm::cheb_approx<double,double>(&density[nCheb*index], cheb_deg, nodes[i]->DataDOF(), &(nodes[i]->ChebData()[0]));
      index++;
    }

  pvfmm::ChebFMM<double> matrices;
  matrices.Initialize(mult_order, cheb_deg, comm, &kernel_fn);
  tree->SetupFMM(&matrices);
  std::vector<double> trg_value;
  size_t n_trg=LegCoord.size()/PVFMM_COORD_DIM;
  pvfmm::ChebFMM_Evaluate(tree, trg_value, n_trg);

  //construct the fock matrix using potential and AOvals on the legendre grid
  {
    int ngrid = LegCoord.size()/3;
    if (intermediate.size() < nao*ngrid) 
      intermediate.resize(nao*ngrid, 0.0);

    vector<double>& aovals = AoValsAtLegCoord;

    double scale = pow(coordScale, 5);
    for (int p = 0; p<nao; p++)
    for (int g = 0; g<ngrid; g++)
      intermediate[p*ngrid + g] =  aovals[p*ngrid + g] * trg_value[g] * LegWts[g] * scale * 4 * M_PI;

    double alpha = 1.0, beta = 0.0;
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                nao, nao, ngrid, alpha, &intermediate[0], ngrid, &aovals[0], ngrid, beta,
                &fock[0], nao);

  }
  
  return;
  /*
  cout.precision(12);
  ncalls = 0;
  dm = pdm;
  centroid = pcentroid;
  coordScale = pscale;
  int rank;MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int ngrid = LegWts.size();
  
  time_t Initcurrent_time, current_time;
  Initcurrent_time = time(NULL);
  
  MPI_Comm comm = MPI_COMM_WORLD;
  int cheb_deg = 10, max_pts = 10000, mult_order = 10;

  const pvfmm::Kernel<double>& kernel_fn=pvfmm::LaplaceKernel<double>::potential();

  vector<double>& trg_coord = LegCoord;
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
  
  int nao = ao_loc[shls[1]] - ao_loc[shls[0]];
  if (intermediate.size() < nao*ngrid) 
    intermediate.resize(nao*ngrid, 0.0);

  tblis_tensor Fock;
  std::fill(fock, fock+nao*nao, 0.);
  vector<double>& aovals = AoValsAtLegCoord;
  long lena[2]   = {ngrid, nao}, stridea[2]   = {1, ngrid}; //aovals, intermediate
  long lendm[2]  = {nao, nao}  , stridedm[2]  = {1, nao}; //density matrix

  tblis_init_tensor_d(&AOVALS,       2, lena,   &aovals[0],                            stridea);
  tblis_init_tensor_d(&INTERMEDIATE, 2, lena,   &intermediate[0],                      stridea);
  tblis_init_tensor_d(&Fock,         2, lendm,  fock,                                 stridedm);

  double scale = pow(coordScale, 5);
  for (int p = 0; p<nao; p++)
  for (int g = 0; g<ngrid; g++)
    intermediate[p*ngrid + g] =  aovals[p*ngrid + g] * trg_value[g] * LegWts[g] * scale * 4 * M_PI;
    //intermediate[p*ngrid + g] =  aovals[p*ngrid + g] * LegWts[g] * scale;
  
  tblis_tensor_mult(NULL, NULL, &INTERMEDIATE, "gp", &AOVALS, "gq", &Fock, "pq");
  delete tree;
  */
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
  int cheb_deg = 10, max_pts = 10000, mult_order = 10;

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
  //if (rank == 0) cout << "make tree "<<(current_time - Initcurrent_time)<<endl;
  // Load matrices.
  pvfmm::ChebFMM<double> matrices;
  matrices.Initialize(mult_order, cheb_deg, comm, &kernel_fn);

  // FMM Setup
  tree->SetupFMM(&matrices);
  current_time = time(NULL);
  //if (rank == 0) cout << "fmm setup "<<(current_time - Initcurrent_time)<<endl;

  // Run FMM
  std::vector<double> trg_value;
  size_t n_trg=trg_coord.size()/PVFMM_COORD_DIM;
  pvfmm::ChebFMM_Evaluate(tree, trg_value, n_trg);

  current_time = time(NULL);

  //if (rank == 0) cout << "evalstep setup "<<(current_time - Initcurrent_time)<<endl;

  for (int i=0; i<nTarget; i++)
    potential[i] = trg_value[i]*4*M_PI;

  delete tree;
}
