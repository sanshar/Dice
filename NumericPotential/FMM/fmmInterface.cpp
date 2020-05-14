#include "pythonInterface.h"
#include <stdio.h>
#include <pvfmm.hpp>
#include <ctime>
#include <boost/math/quadrature/gauss.hpp>
#include "mkl.h"

//#FMM TREE
pvfmm::ChebFMM_Tree<double>* tree;
vector<double> LegCoord, LegWts;
vector<double> AoValsAtLegCoord;

vector<double> ChebCoord;
vector<double> AoValsAtChebCoord;
vector<double> density;

void writeNorm(double* mat, size_t size) {
  double norm = 0.0;
  for (int i=0; i<size; i++)
    norm += mat[i];
  cout << norm<<endl;
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

  int nt = omp_get_max_threads();
  omp_set_num_threads(1);
  
  vector<double> dummy(3,0.0);
  tree=ChebFMM_CreateTree(cheb_deg, kernel_fn.ker_dim[0],
                          getValuesAtGrid,
                          dummy, comm, tol, max_pts, pvfmm::FreeSpace);

  omp_set_num_threads(nt);
  
  auto nodes = tree->GetNodeList();
  for (int i=0; i<nodes.size(); i++)
    if (nodes[i]->IsLeaf()) 
      leafNodes++;
  cout << leafNodes<<endl;
  
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

  cout << LegCoord.size()<<endl;
  writeNorm(&AoValsAtLegCoord[0], AoValsAtLegCoord.size());
  
  //calculate the ao values at the chebyshev coordinates
  AoValsAtChebCoord.resize(ChebCoord.size() * nao/3, 0.0);
  getAoValuesAtGrid(&ChebCoord[0], ChebCoord.size()/3, &AoValsAtChebCoord[0]);
  
  cout << ChebCoord.size()<<endl;
  writeNorm(&AoValsAtChebCoord[0], AoValsAtChebCoord.size());
  
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
