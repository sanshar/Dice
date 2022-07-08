#include "pythonInterface.h"
#include "fittedFunctions.h"
#include <iostream>

void getSphericalCoords(double& x, double& y, double& z,
                        double& r, double& t, double& p) {
  double xy = x*x + y*y;
  r = sqrt(xy + z*z);
  t = atan2(sqrt(xy), z);
  p = atan2(y, x);
  if (p < 0.0)
    p += 2*M_PI;
}

void getSphericalCoords(MatrixXdR& grid, MatrixXd& SphericalCoords) {

  int ngridPts = grid.rows();
  for (int i=0; i<ngridPts; i++) {
    getSphericalCoords(grid(i,0), grid(i, 1), grid(i,2),
                       SphericalCoords(i,0), SphericalCoords(i,1), SphericalCoords(i,2));
    
  }

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


void SplineFit::Init(const RawDataOnGrid& potentialYlm){
  zgrid = potentialYlm.zgrid;
  rm    = potentialYlm.rm;
  lmax  = potentialYlm.lmax;

  int nlm = potentialYlm.CoeffsYlm.cols();
  int nr  = zgrid.size();
  
  CoeffsYlm.resize(nlm);
  CoeffsYlmFit.resize(nlm);

  vector<double> vals(nr+2,0.0);
  double h = 1.0/(1.*nr + 1.), t0 = 0; //t0 = 0 -> r= infty

  for (int lm = 0; lm < nlm; lm++) {
    
    for (int j=0; j<nr; j++)
      vals[j+1] = potentialYlm.CoeffsYlm(j, lm);
    vals[nr+1] = 0.0;  //at r-> 0  Ulm = 0;

    if (lm == 0) //l == 0
      vals[0] = vals[1]; //sqrt(r pi) q
    else
      vals[0] = 0.0;
          
    CoeffsYlmFit[lm] = boost::math::cubic_b_spline<double>(vals.begin(), vals.end(), t0, h);
  }
}

void SplineFit::getPotential(int ngrid, double* grid, double* potential) {

  #pragma omp parallel
  {    
    double sphCoord[3];
    CalculateSphHarmonics sph(lmax); 
    #pragma omp for
    for (int i=0; i<ngrid; i++) {
      double X = grid[3*i] - coord[0],
          Y = grid[3*i+1] - coord[1],
          Z = grid[3*i+2] - coord[2];
      getSphericalCoords(X, Y, Z, 
                         sphCoord[0], sphCoord[1], sphCoord[2]);
      
      sph.populate(sphCoord[1], sphCoord[2]); //sphcoords
      double r = sphCoord[0];
      double z = (1./M_PI) * acos( (r - rm)/(r+rm));
      
      for (int lm=0; lm<CoeffsYlmFit.size(); lm++)
        CoeffsYlm[lm] = CoeffsYlmFit[lm](z);
      
      potential[i] += CoeffsYlm.dot( sph.values )/r;
      
    }
  }
}



void RawDataOnGrid::InitStaticVariables(int plmax, int lebdevOrder) {
  
  //make lebdev grid of order lebdevorder
  lmax = plmax;
  int GridSize = lebdevGridSize[lebdevOrder/2];
  lebdevgrid.resize(GridSize,4);
  LebdevGrid::MakeAngularGrid(&lebdevgrid(0,0), GridSize);
  lebdevgrid.col(3) *= 4 * M_PI;
  
  //get spherical coordinates from it
  SphericalCoords.resize(GridSize, 3);
  getSphericalCoords(lebdevgrid, SphericalCoords);

  //get value of spherical harmonics on the lebdev grid
  sphVals.resize(GridSize, (lmax+1)*(lmax+1));
  WeightedSphVals.resize(GridSize, (lmax+1)*(lmax+1));

  CalculateSphHarmonics sph(lmax); //this is just used to calculated Ylm on the grid

  for (int i=0; i<GridSize; i++) {
    sph.populate(SphericalCoords(i,1), SphericalCoords(i,2));
    sphVals.row(i) = sph.values;
    WeightedSphVals.row(i) = sph.values * lebdevgrid(i,3) ;    
  }

  densityStore.resize(GridSize);
}

void RawDataOnGrid::InitRadialGrids(int rmax, double prm) {
  GridValues.resize(rmax, lebdevgrid.rows());
  CoeffsYlm.resize(rmax, (lmax+1)*(lmax+1));  //spherical harmonics on a radial grid
  radialGrid.resize(rmax);
  zgrid.resize(rmax);
  wts.resize(rmax);
  rm = prm;

  GridValues.setZero(); CoeffsYlm.setZero();
  
  for (int i=0; i<rmax; i++) {
    zgrid(i) = (1.*i+1.)/(1.*rmax+1.);
    double x = cos(M_PI*zgrid(i));
    radialGrid(i) = rm * ((1.+x)/(1.-x));
    //grid[i] = radialGrid(i);

    wts(i) = pow(radialGrid(i),2) * M_PI/(1.*rmax+1.) *
        pow( sin( (i+1.) * M_PI / (1.*rmax + 1.0)), 2.) *
        2 * rm/(1.-x)/(1.-x)/sqrt(1.-x*x);
    //radwts[i] = wts(i);
  }

}
  
//calculate the coefficients for spherical Harmonics for lebdev grid at radius r
void RawDataOnGrid::fit(int rindex, double* density) {
  int nang = densityStore.size();
  for (int i=0; i<densityStore.size(); i++)
    densityStore(i) = density[i];  

  GridValues.row(rindex) = densityStore;
  CoeffsYlm.row(rindex) = WeightedSphVals.transpose() * densityStore;
}

void RawDataOnGrid::getValue(int rindex, double* densityOut) {
  densityStore = CoeffsYlm.row(rindex) * sphVals.transpose();
  for (int i=0; i<densityStore.size(); i++)
    densityOut[i] = densityStore(i);
}

