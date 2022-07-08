#pragma once
#include <vector>
#include "coulomb.h"
#include "tensor.h"

using namespace std;

extern vector<double> normA, normB;
extern const int Lmax;// = 7;
extern vector<vector<int>> CartOrder;//[(Lmax+1) * (Lmax+2) * (Lmax+3)/6][3];
extern vector<double> workArray; //only need 4*Lmax
extern tensor DerivativeToPolynomial;
extern Coulomb_14_8_8 coulomb_14_8_8;
extern Coulomb_14_14_8 coulomb_14_14_8;

extern tensor Sx_2d ;
extern tensor Sy_2d ;
extern tensor Sz_2d ;
extern tensor Sx2_2d;
extern tensor Sy2_2d;
extern tensor Sz2_2d;
extern tensor S_2d  ;
extern tensor Sx_3d;
extern tensor Sy_3d;
extern tensor Sz_3d;
extern tensor Sx2_3d;
extern tensor Sy2_3d;
extern tensor Sz2_3d;
extern tensor S_3d;
extern tensor Coeffx_3d;
extern tensor Coeffy_3d;
extern tensor Coeffz_3d;
extern tensor powExpAPlusExpB;
extern tensor powExpA;
extern tensor powExpB;
extern tensor powExpC;
extern tensor powPIOverLx;
extern tensor powPIOverLy;
extern tensor powPIOverLz;
extern vector<double> expbPow;
extern vector<double> expaPow;
extern tensor Ci;
extern tensor Cj;
extern tensor Ck;
extern tensor teri;
void initWorkArray();
