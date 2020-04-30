#include "CalculateSphHarmonics.h"
#include <Eigen/Dense>
#include <boost/math/special_functions/spherical_harmonic.hpp>

using namespace std;
using namespace Eigen;

double factorialRatio(int l, int m) {
  double ratio = 1.0;
  for (int k = -m + 1 ; k < m+1 ; k++)
    ratio *= 1./(1.*l + 1.*k);
  return ratio;
}

//calculate all the spherical harmonics at theta phi
void CalculateSphHarmonics::populate(double theta, double phi) {

  double x = cos(theta);

  //fill out all the legendre polynomial values for positive m
  for (int m = 0; m < lmax; m++) {
    double vallm1 = boost::math::legendre_p(m  , m, x);
    double vall   = boost::math::legendre_p(m+1, m, x); 
    getval(m  , m) = vallm1;
    getval(m+1, m) = vall;

      
    for (int l = m+1 ; l<lmax; l++) {
      double temp = boost::math::legendre_next(l, m, x, vall, vallm1);
      vallm1 = vall;
      vall = temp;
      getval(l+1, m) = vall;
    }
  }
  getval(lmax, lmax) = boost::math::legendre_p(lmax, lmax, x);

  //use the legendre polynomial values to get the spherical harmonics
  double sqrt2 = sqrt(2.0);
  for (int m=0; m<lmax+1; m++) {
    double sinphi = pow(-1., m) * sin(m*phi), cosphi = pow(-1., m) * cos(m*phi);
    for (int l=m; l<lmax+1; l++) {
      double leg = sqrt2 * sqrt( (2*l+1) * factorialRatio(l, m) / 4./M_PI) * getval(l, m);
      getval(l, m) = leg * cosphi;
      if (m != 0)
        getval(l,-m) = leg * sinphi;
      else
        getval(l, m) /= sqrt2;
    }
  }
}

/*

int main(int argc, char* argv[]) {

  double val[100];
  double theta = M_PI/2.3, phi = 1.3*M_PI;
  int l = 4;
  cin >> l ;

  CalculateSphHarmonics sph(l);

  for (int i=0; i<50000; i++)
    sph.populate(theta, phi);
  cout << sph.values[0]<<" - "<<sph.getval(l,0)<<endl;
  
  cout <<0<<"  "<< sph.getval(l, 0)<<"  "<< boost::math::spherical_harmonic(l, 0, theta, phi)<<endl;
  for (int m=1; m<l+1; m++) {
    double val1 = sph.getval(l,m),
        val2 = sqrt(2.0)* boost::math::spherical_harmonic(l, m, theta, phi).real() * pow(-1,m);
    double val3 = sph.getval(l, -m),
        val4 = sqrt(2.0)* boost::math::spherical_harmonic(l, m, theta, phi).imag() * pow(-1,m);
    
    if (abs(val1-val2) > 1.e-14)
      cout <<m<<"  "<< val1<<"  "<<val2<<endl;
    if (abs(val3-val4) > 1.e-14)
      cout <<-m<<"  "<< val3<<"  "<<val4<<endl;
  }
}
*/
