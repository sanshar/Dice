#ifndef HMULT_HEADER_H
#define HMULT_HEADER_H
#include <Eigen/Dense>
#include <Eigen/Core>
#include "integral.h"

using namespace Eigen;

double parity(char* d, int& sizeA, int& i);
double Energy(char* ket, int& sizeA, oneInt& I1, twoInt& I2, double& coreE);
double Hij(char* bra, char* ket, int& sizeA, oneInt& I1, twoInt& I2, double& coreE);

template <typename Derived>
void multiplyH(MatrixBase<Derived>& x, MatrixBase<Derived>& y, char* a, int& sizeA, oneInt& I1, twoInt& I2, double& coreE)
{
  y *= 0.0;

  int num_thrds = 1;//omp_get_max_threads();
  std::vector<MatrixXd> yarray(num_thrds);

  for (int i=0; i<num_thrds; i++) {
    yarray[i] = Eigen::MatrixXd(y.rows(),1);
    yarray[i] = 0.* y;
  }
  //std::cout << yarray[i].rows()<<std::endl;

  //#pragma omp parallel for 
  for (int i=0; i<x.rows(); i++) {
    for (int j=i; j<y.rows(); j++) {
        double hij = Hij(a+i*sizeA, a+j*sizeA, sizeA, I1, I2, coreE);
        yarray[omp_get_thread_num()](i,0) += hij*x(j,0);
        if (i!=j) yarray[omp_get_thread_num()](j,0) += hij*x(i,0);
        //y(i,0) += hij*x(j,0);
        //if (i!=j) y(j,0) += hij*x(i,0);
    }
  }

  for (int i=0; i<num_thrds; i++)
    y += yarray[i];
};

double Hij_1Excite(int i, int a, oneInt& I1, twoInt& I2, char* ket, int& sizeA);
double Hij_2Excite(int i, int j, int a, int b, twoInt& I2, char* ket, int& sizeA);

struct Hmult {
  char* a;
  int& sizeA;
  oneInt& I1;
  twoInt& I2;
  double& coreE;
  
Hmult(char* a_, int& sizeA_, oneInt& I1_, twoInt& I2_, double& coreE_) :
  a(a_), sizeA(sizeA_), I1(I1_), I2(I2_), coreE(coreE_) {}

  template <typename Derived>
  void operator()(MatrixBase<Derived>& x, MatrixBase<Derived>& y) {
    multiplyH(x,y,a,sizeA,I1,I2,coreE);
  };



};

struct Hmult2 {
  std::vector<std::vector<int> >& connections;
  std::vector<std::vector<double> >& Helements;
  
  Hmult2(std::vector<std::vector<int> >& connections_, std::vector<std::vector<double> >& Helements_)
  : connections(connections_), Helements(Helements_) {}

  template <typename Derived>
  void operator()(MatrixBase<Derived>& x, MatrixBase<Derived>& y) {
    y*=0.0;
    for (int i=0; i<x.rows(); i++) {
      for (int j=0; j<connections[i].size(); j++) {
	double hij = Helements[i][j];
	int J = connections[i][j];
	y(i,0) += hij*x(J,0);
	if (i!= J) y(J,0) += hij*x(i,0);
      }
    }
  }
};



#endif
