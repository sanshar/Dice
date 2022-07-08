#include <stdio.h>
#include <pvfmm.hpp>
#include <ctime>
#include "mkl.h"
#include <finufft.h>
#include <Eigen/Dense>

#include "fmmInterface.h"
#include "pythonInterface.h"

using namespace Eigen;

//#FMM TREE
pvfmm::ChebFMM_Tree<double>* tree;
vector<double> LegCoord, LegWts;
vector<double> AoValsAtLegCoord;
vector<complex<double>> AoValsAtLegCoordComplex;

vector<double> ChebCoord;
vector<double> AoValsAtChebCoord;
vector<complex<double>> AoValsAtChebCoordComplex;
vector<double> density;

template<typename T>
void writeNorm(T* mat, size_t size) {
  T norm = 0.0;
  for (int i=0; i<size; i++)
    norm += mat[i];
  cout << norm<<endl;
}


void populateChebyshevCoordinates(int leafNodes, int cheb_deg) {
  size_t nCheb = (cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);
  size_t index = 0;
  double volume = 0.0;
  auto nodes = tree->GetNodeList();
  
  //chebyshev coord
  ChebCoord.resize(leafNodes*nCheb*3, 0.0);
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

}


void populateLegendreCoordinates(int leafNodes) {
  
  auto nodes = tree->GetNodeList();
  const int legdeg = 11;
  size_t nLeg  = legdeg * legdeg * legdeg;
  size_t numCoords = leafNodes * nLeg;
  size_t index = 0;
  double volume = 0.0;
  
  vector<double> Legcoord, Legwts;
  getLegCoords<legdeg>(nodes[0]->Dim(), Legcoord, Legwts);
  
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

}

int printTreeStats() {
  auto nodes = tree->GetNodeList();
  int leafNodes;
  MPI_Comm comm = MPI_COMM_WORLD;
  {
    int myrank;
    MPI_Comm_rank(comm, &myrank);
    std::vector<size_t> all_nodes(PVFMM_MAX_DEPTH+1,0);
    std::vector<size_t> leaf_nodes(PVFMM_MAX_DEPTH+1,0);
    for (int i=0; i<nodes.size(); i++) {
      auto n=nodes[i];
      if(!n->IsGhost()) all_nodes[n->Depth()]++;
      if(!n->IsGhost() && n->IsLeaf()) leaf_nodes[n->Depth()]++;
      if (nodes[i]->IsLeaf()) 
        leafNodes++;
    }

    if(!myrank) std::cout<<"All  Nodes: ";
    for(int i=0;i<PVFMM_MAX_DEPTH;i++){
      int local_size=all_nodes[i];
      int global_size;
      MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, comm);
      if(!myrank) std::cout<<global_size<<' ';
    }
    if(!myrank) std::cout<<'\n';

    if(!myrank) std::cout<<"Leaf Nodes: ";
    for(int i=0;i<PVFMM_MAX_DEPTH;i++){
      int local_size=leaf_nodes[i];
      int global_size;
      MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, comm);
      if(!myrank) std::cout<<global_size<<' ';
    }
    if(!myrank) std::cout<<'\n';
  }

  return leafNodes;
}

void initFMMGridAndTree(double* pdm, double* pcentroid, double pscale, double tol,
                        double eta, int Periodic) {
  
  int nao = ao_loc[shls[1]] - ao_loc[shls[0]];
  cout.precision(12);
  if (!Periodic)
    dm = pdm;
  else {
    dmComplex.resize(nkpts * nao * nao);
    for (int i=0; i<dmComplex.size(); i++)
      dmComplex[i] = pdm[i];
  }
  centroid = pcentroid;
  coordScale = pscale;


  //auto getDensity = getValuesAtGrid;
  auto getDensity  = Periodic == 0 ? getValuesAtGrid : getValuesAtGridPeriodic;
  
  time_t Initcurrent_time, current_time;
  Initcurrent_time = time(NULL);
  
  MPI_Comm comm = MPI_COMM_WORLD;
  int cheb_deg = 10, max_pts = 10000, mult_order = 10;

  const pvfmm::Kernel<double>& kernel_fn = (abs(eta)<1.e-10) ?
      pvfmm::LaplaceKernel<double>::potential() :
      pvfmm::ShortRangeCoulomb<double,1>::potential();
  
  int nt = omp_get_max_threads();
  omp_set_num_threads(1);
  
  vector<double> dummy(3,0.0);
  tree=ChebFMM_CreateTree(cheb_deg, kernel_fn.ker_dim[0],
                          getDensity,
                          //dummy, comm, tol, max_pts, pvfmm::FreeSpace);
                          dummy, comm, tol, max_pts, pvfmm::Periodic);

  current_time = time(NULL);
  cout << "time to make tree "<<current_time - Initcurrent_time<<endl;
  
  int leafNodes = printTreeStats();  
  omp_set_num_threads(nt);

  
  populateLegendreCoordinates(leafNodes);
  populateChebyshevCoordinates(leafNodes, cheb_deg);

  //calculate the ao values at the legendre coordinates
  if (Periodic) {
    AoValsAtLegCoordComplex.resize(LegCoord.size() * nao/3 * nkpts, 0.0);
    getAoValuesAtGridPeriodic(&LegCoord[0], LegCoord.size()/3, &AoValsAtLegCoordComplex[0]);

    current_time = time(NULL);
    cout << "legendre coordinates "<<current_time - Initcurrent_time<<"  "<<LegCoord.size()/3<<endl;

    AoValsAtChebCoordComplex.resize(ChebCoord.size() * nao/3 * nkpts, 0.0);
    getAoValuesAtGridPeriodic(&ChebCoord[0], ChebCoord.size()/3, &AoValsAtChebCoordComplex[0]);

    current_time = time(NULL);
    cout << "chebyshev coordinates "<<current_time - Initcurrent_time<<"  "<<ChebCoord.size()/3<<endl;
  }
  else {
    AoValsAtLegCoord.resize(LegCoord.size() * nao/3, 0.0);
    getAoValuesAtGrid(&LegCoord[0], LegCoord.size()/3, &AoValsAtLegCoord[0]);
    AoValsAtChebCoord.resize(ChebCoord.size() * nao/3, 0.0);
    getAoValuesAtGrid(&ChebCoord[0], ChebCoord.size()/3, &AoValsAtChebCoord[0]);
  }
  cout << "calculated density"<<endl;
  
}

void getFockFMMPeriodic(double* pdm, complex<double>* fock, double tol) {
  
  int nao = ao_loc[shls[1]] - ao_loc[shls[0]];

  for (int i=0; i<dmComplex.size(); i++)
    dmComplex[i] = pdm[i];


  getDensity(density, AoValsAtLegCoordComplex);
  vector<double> potential(density.size(), 0.0);
  getElectronicFFTPeriodic(&LegCoord[0], &LegWts[0], LegWts.size(), &density[0], &potential[0],
                           RLattice, 0.0, 80, 80, 80, tol);

  getFock(potential, AoValsAtLegCoordComplex, fock);
  
  getElectronicFFTPeriodicUniform(RLattice, potential, 0.0, 100, 100, 100);
  
  return;
  
  int rank;MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm comm = MPI_COMM_WORLD;
  //const pvfmm::Kernel<double>& kernel_fn = pvfmm::ShortRangeCoulomb<double,1>::potential();
  const pvfmm::Kernel<double>& kernel_fn = pvfmm::LaplaceKernel<double>::potential();
  time_t Initcurrent_time, current_time;
  Initcurrent_time = time(NULL);

  
  //construct the density from density matrix and ao values stored on chebyshev grid
  int cheb_deg = 10, mult_order = 10;

  {
    int ngrid = ChebCoord.size()/3;
    if (intermediateComplex.size() < nao*ngrid) {
      intermediateComplex.resize(nao*ngrid, 0.0);
    }

    complex<double> alpha = 1.0, beta = 0.0;
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                ngrid, nao, nao, &alpha, &AoValsAtChebCoordComplex[0], ngrid, &dmComplex[0], nao,&beta,
                &intermediateComplex[0], ngrid);
    
    if (density.size() < ngrid) density.resize(ngrid, 0.0);
    std::fill(&density[0], &density[0]+ngrid, 0.0);
    
    for (int p = 0; p<nao; p++)
      for (int g = 0; g<ngrid; g++)
        density[g] += (intermediateComplex[p*ngrid + g] * std::conj(AoValsAtChebCoordComplex[p*ngrid + g])).real();

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
    if (intermediateComplex.size() < nao*ngrid) 
      intermediateComplex.resize(nao*ngrid, 0.0);

    vector<complex<double>>& aovals = AoValsAtLegCoordComplex;

    cout << "before intermediate"<<endl;
    cout << aovals.size()<<"  "<<trg_value.size()<<"  "<<LegWts.size()<<"  "<<nao<<"  "<<ngrid<<endl;
    double scale = pow(coordScale, 5);
    for (int p = 0; p<nao; p++)
    for (int g = 0; g<ngrid; g++)
      intermediateComplex[p*ngrid + g] =  aovals[p*ngrid + g] * trg_value[g] * LegWts[g] * scale * 4 * M_PI;

    cout << "before zgemm"<<endl;
    complex<double> alpha = 1.0, beta = 0.0;
    cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                nao, nao, ngrid, &alpha, &intermediateComplex[0], ngrid, &aovals[0], ngrid, &beta,
                &fock[0], nao);
    cout << "after zgemm"<<endl;
  }
  
  return;
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
