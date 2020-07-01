#include "workArray.h"


tensor Sx_2d({120,120});
tensor Sy_2d ({120,120});
tensor Sz_2d ({120,120});
tensor Sx2_2d({120,120});
tensor Sy2_2d({120,120});
tensor Sz2_2d({120,120});
tensor S_2d  ({120,120});
vector<double> normA(120), normB(120);
const int Lmax = 7;
vector<vector<int>> CartOrder((Lmax+1) * (Lmax+2) * (Lmax+3)/6, vector<int>(3));
Coulomb_14_8_8 coulomb_14_8_8;
Coulomb_14_14_8 coulomb_14_14_8;

vector<double> workArray(200); //only need 4*Lmax
tensor DerivativeToPolynomial({18, 18});

tensor Sx_3d({50,50,50});
tensor Sy_3d({50,50,50});
tensor Sz_3d({50,50,50});
tensor Sx2_3d({50,50,50});
tensor Sy2_3d({50,50,50});
tensor Sz2_3d({50,50,50});
tensor S_3d({50,50,50});
tensor Coeffx_3d({50,50,50});
tensor Coeffy_3d({50,50,50});
tensor Coeffz_3d({50,50,50});
tensor powExpAPlusExpB({1000});
tensor powExpA({1000});
tensor powExpB({1000});
tensor powExpC({1000});
tensor powPIOverLx({1000});
tensor powPIOverLy({1000});
tensor powPIOverLz({1000});
vector<double> expaPow(1000), expbPow(1000);
tensor Ci({1000});
tensor Cj({1000});
tensor Ck({1000});
tensor teri({1000, 1000, 1000});

void initWorkArray() {

  for (int L=0; L< Lmax; L++) {
    int index = L*(L+1)*(L+2)/6;
    for (int i=L; i>=0; i--) {
      for (int j=L-i; j>=0; j--) {
        CartOrder[index][0] = i;
        CartOrder[index][1] = j;
        CartOrder[index][2] = L-i-j;
        index++;
      }
    }
  }


  tensor& deriv = DerivativeToPolynomial;
  deriv(0 , 0) = 1.;  deriv(1 , 1) = 0.5;  deriv(2 , 0) = 0.5;  deriv(2 , 2) = 0.25;  deriv(3 , 1) = 0.75;  deriv(3 , 3) = 0.125;  deriv(4 , 0) = 0.75;  deriv(4 , 2) = 0.75;  deriv(4 , 4) = 0.0625;  deriv(5 , 1) = 1.875;  deriv(5 , 3) = 0.625;  deriv(5 , 5) = 0.03125;  deriv(6 , 0) = 1.875;  deriv(6 , 2) = 2.8125;  deriv(6 , 4) = 0.46875;  deriv(6 , 6) = 0.015625;  deriv(7 , 1) = 6.5625;  deriv(7 , 3) = 3.28125;  deriv(7 , 5) = 0.328125;  deriv(7 , 7) = 0.0078125;  deriv(8 , 0) = 6.5625;  deriv(8 , 2) = 13.125;  deriv(8 , 4) = 3.28125;  deriv(8 , 6) = 0.21875;  deriv(8 , 8) = 0.00390625;  deriv(9 , 1) = 29.5313;  deriv(9 , 3) = 19.6875;  deriv(9 , 5) = 2.95313;  deriv(9 , 7) = 0.140625;  deriv(9 , 9) = 0.00195313;  deriv(10 , 0) = 29.5313;  deriv(10 , 2) = 73.8281;  deriv(10 , 4) = 24.6094;  deriv(10 , 6) = 2.46094;  deriv(10 , 8) = 0.0878906;  deriv(10 , 10) = 0.000976563;  deriv(11 , 1) = 162.422;  deriv(11 , 3) = 135.352;  deriv(11 , 5) = 27.0703;  deriv(11 , 7) = 1.93359;  deriv(11 , 9) = 0.0537109;  deriv(11 , 11) = 0.000488281;  deriv(12 , 0) = 162.422;  deriv(12 , 2) = 487.266;  deriv(12 , 4) = 203.027;  deriv(12 , 6) = 27.0703;  deriv(12 , 8) = 1.4502;  deriv(12 , 10) = 0.0322266;  deriv(12 , 12) = 0.000244141;  deriv(13 , 1) = 1055.74;  deriv(13 , 3) = 1055.74;  deriv(13 , 5) = 263.936;  deriv(13 , 7) = 25.1367;  deriv(13 , 9) = 1.04736;  deriv(13 , 11) = 0.019043;  deriv(13 , 13) = 0.00012207;  deriv(14 , 0) = 1055.74;  deriv(14 , 2) = 3695.1;  deriv(14 , 4) = 1847.55;  deriv(14 , 6) = 307.925;  deriv(14 , 8) = 21.9946;  deriv(14 , 10) = 0.733154;  deriv(14 , 12) = 0.0111084;  deriv(14 , 14) = 0.0000610352;  deriv(15 , 1) = 7918.07;  deriv(15 , 3) = 9237.74;  deriv(15 , 5) = 2771.32;  deriv(15 , 7) = 329.919;  deriv(15 , 9) = 18.3289;  deriv(15 , 11) = 0.499878;  deriv(15 , 13) = 0.00640869;  deriv(15 , 15) = 0.0000305176;  deriv(16 , 0) = 7918.07;  deriv(16 , 2) = 31672.3;  deriv(16 , 4) = 18475.5;  deriv(16 , 6) = 3695.1;  deriv(16 , 8) = 329.919;  deriv(16 , 10) = 14.6631;  deriv(16 , 12) = 0.333252;  deriv(16 , 14) = 0.00366211;  deriv(16 , 16) = 0.0000152588;  deriv(17 , 1) = 67303.6;  deriv(17 , 3) = 89738.1;  deriv(17 , 5) = 31408.3;  deriv(17 , 7) = 4486.9;  deriv(17 , 9) = 311.591;  deriv(17 , 11) = 11.3306;  deriv(17 , 13) = 0.217896;  deriv(17 , 15) = 0.0020752;  deriv(17 , 17) = 0.00001525878906;
  /*
  deriv(0 , 0) = 1;
  deriv(1 , 1) = 0.5;
  deriv(2 , 0) = 0.5;
  deriv(2 , 2) = 0.25;
  deriv(3 , 1) = 0.75;
  deriv(3 , 3) = 0.125;
  deriv(4 , 0) = 0.75;
  deriv(4 , 2) = 0.75;
  deriv(4 , 4) = 0.0625;
  deriv(5 , 1) = 1.875;
  deriv(5 , 3) = 0.625;
  deriv(5 , 5) = 0.03125;
  deriv(6 , 0) = 1.875;
  deriv(6 , 2) = 2.8125;
  deriv(6 , 4) = 0.46875;
  deriv(6 , 6) = 0.015625;
  deriv(7 , 1) = 6.5625;
  deriv(7 , 3) = 3.28125;
  deriv(7 , 5) = 0.328125;
  deriv(7 , 7) = 0.0078125;
  deriv(8 , 0) = 6.5625;
  deriv(8 , 2) = 13.125;
  deriv(8 , 4) = 3.28125;
  deriv(8 , 6) = 0.21875;
  deriv(8 , 8) = 0.00390625;
  deriv(9 , 1) = 29.5313;
  deriv(9 , 3) = 19.6875;
  deriv(9 , 5) = 2.95313;
  deriv(9 , 7) = 0.140625;
  deriv(9 , 9) = 0.00195313;
  deriv(10 , 0) = 29.5313;
  deriv(10 , 2) = 73.8281;
  deriv(10 , 4) = 24.6094;
  deriv(10 , 6) = 2.46094;
  deriv(10 , 8) = 0.0878906;
  deriv(10 , 10) = 0.000976563;
  deriv(11 , 1) = 162.422;
  deriv(11 , 3) = 135.352;
  deriv(11 , 5) = 27.0703;
  deriv(11 , 7) = 1.93359;
  deriv(11 , 9) = 0.0537109;
  deriv(11 , 11) = 0.000488281;
  deriv(12 , 0) = 162.422;
  deriv(12 , 2) = 487.266;
  deriv(12 , 4) = 203.027;
  deriv(12 , 6) = 27.0703;
  deriv(12 , 8) = 1.4502;
  deriv(12 , 10) = 0.0322266;
  deriv(12 , 12) = 0.000244141;
  deriv(13 , 1) = 1055.74;
  deriv(13 , 3) = 1055.74;
  deriv(13 , 5) = 263.936;
  deriv(13 , 7) = 25.1367;
  deriv(13 , 9) = 1.04736;
  deriv(13 , 11) = 0.019043;
  deriv(13 , 13) = 0.00012207;
  */
}
