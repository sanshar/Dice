#pragma once
#include <Eigen/Dense>
#include "CalculateSphHarmonics.h"
#include <boost/math/interpolators/cubic_b_spline.hpp>

using namespace boost;
using namespace Eigen;

using MatrixXdR = Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>;

void getSphericalCoords(MatrixXdR& grid, MatrixXd& SphericalCoords);

namespace LebdevGrid{
int MakeAngularGrid(double *Out, int nPoints);
};


//We use this to store the quantities rho^{i,n}_{lm}
//this is a four index quantity
struct RawDataOnGrid {
  Vector3d coord; //this is the coordinate of the atom
  MatrixXd GridValues; //r x lebdev
  MatrixXd CoeffsYlm;    //r x lm
  VectorXd radialGrid;   //r
  VectorXd zgrid; //r = rm*(1+cos(pi z))/(1-cos(pi z))
  VectorXd wts; //w
  double rm;
  
  //al radial points use the same lebdev grid and so this is only needed once
  static vector<int> lebdevGridSize;
  static int lmax;
  static MatrixXdR lebdevgrid;   //#lebdev x 4
  static MatrixXd SphericalCoords; //#lebdev x 3
  static MatrixXd sphVals;      //#lebdev x Ylm
  static MatrixXd WeightedSphVals; //#lebdev x Ylm
  static VectorXd densityStore; //#lebdev
  
  static void InitStaticVariables(int lmax, int lebdevOrder);
  void InitRadialGrids(int rmax, double rm);
  
  //calculate the coefficients for spherical Harmonics for lebdev grid at radius r
  void fit(int rindex,double* density);  
  void getValue(int rindex, double* densityOut);
  
};

struct SplineFit {
  Vector3d coord; //this is the coordinate of the atom
  vector<boost::math::cubic_b_spline<double> > CoeffsYlmFit; //length = #lm
  VectorXd zgrid; //r = rm*(1+cos(pi z))/(1-cos(pi z))  
  double rm;
  int lmax;
  VectorXd CoeffsYlm; //#lm (this is just used during claculations)

  void Init(const RawDataOnGrid& in);
  void getPotential(int ngrid, double* grid, double* potential);
};

