#include "SchmidtStates.h"
#include "global.h"
#include "Determinants.h"
using namespace Eigen;
using namespace std;

void getMatrix(std::vector<Determinant> &Dets, int norbs, int nocc, MatrixXx& ci, MatrixXd& A, MatrixXd& B,
            MatrixXd& AA, MatrixXd& BB, MatrixXd& AB, MatrixXi& AAidx, MatrixXi& BBidx, MatrixXi& ABidx)
{
  int nvirt = norbs - nocc;
  int noccPair = nocc*(nocc+1)/2, nvirtPair = nvirt*(nvirt+1)/2;
  //cout << nocc<<"  "<<nvirt<<"  "<<noccPair<<"  "<<nvirtPair<<endl;

  //singles
  A = MatrixXd::Zero(nocc, nvirt);
  B = MatrixXd::Zero(nocc, nvirt);

  //doubles
  AB = MatrixXd::Zero(nocc*nocc, nvirt*nvirt);
  AA = MatrixXd::Zero(noccPair, nvirtPair);
  BB = MatrixXd::Zero(noccPair, nvirtPair);

  char det[norbs];
  for (int i=0; i<Dets.size(); i++) {
    Dets[i].getRepArray(det);

    int occval1=-1, occval2 = -1;
    for (int j=0; j<nocc; j++) {
      if (det[2 * j] == false && det[2 * j + 1] == false) {
        occval1 = 2*j; 
        occval2 = 2*j+1;
      }
      else if (det[2 * j] == false && det[2 * j + 1] == true) {
        if (occval1 != -1) occval2 = 2*j;
        else occval1 = 2*j;
      }
      else if (det[2 * j] == true && det[2 * j + 1] == false) {
        if (occval1 != -1) occval2 = 2*j+1;
        else occval1 = 2*j+1;
      }
    }

    int virtval1=-1, virtval2 = -1;
    for (int j=nocc; j<norbs; j++) {
      if (det[2 * j] == true && det[2 * j + 1] == true) {
        virtval1 = 2*(j-nocc); 
        virtval2 = 2*(j-nocc)+1;
      }
      else if (det[2 * j] == true && det[2 * j + 1] == false) {
        if (virtval1 != -1) virtval2 = 2*(j-nocc);
        else virtval1 = 2*(j-nocc);
      }
      else if (det[2 * j] == false && det[2 * j + 1] == true) {
        if (virtval1 != -1) virtval2 = 2*(j-nocc)+1;
        else virtval1 = 2*(j-nocc)+1;
      }
    }

    //singles
    if (occval1 != -1 && occval2 == -1) {
      if (occval1%2 == 0) //alpha
        A(occval1/2, virtval1/2) = ci(i);
      else
        B(occval1/2, virtval1/2) = ci(i);
    }

    //doubles
    if (occval1 != -1 && occval2 != -1 && virtval1 != -1 && virtval2 != -1) {
      if (occval1%2 == 0 && occval2%2 == 0) {
        int I = max(occval1, occval2)/2, J = min(occval1, occval2)/2;
        int A = max(virtval1, virtval2)/2, B = min(virtval1, virtval2)/2;
        //cout <<I<<"  "<<J<<"  "<<A<<"  "<<B<<"  "<<"  "<<"  "<<Dets[i]<<endl;
        AA(I*(I+1)/2+J, A*(A+1)/2+B) = ci(i);
      }

      else if (occval1%2 == 1 && occval2%2 == 1) {
        int I = max(occval1, occval2)/2, J = min(occval1, occval2)/2;
        int A = max(virtval1, virtval2)/2, B = min(virtval1, virtval2)/2;
        //cout <<I<<"  "<<J<<"  "<<A<<"  "<<B<<"  "<<"  "<<"  "<<Dets[i]<<endl;
        BB(I*(I+1)/2+J, A*(A+1)/2+B) = ci(i);
      }

      else {
        int I = (occval1%2 == 0) ? occval1/2 : occval2/2;  //I is even 
        int J = (occval1%2 == 0) ? occval2/2 : occval1/2;  //J is odd
        int A = (virtval1%2 == 0) ? virtval1/2 : virtval2/2;  //A is even 
        int B = (virtval1%2 == 0) ? virtval2/2 : virtval1/2;  //B is odd
        //cout <<I<<"  "<<J<<"  "<<A<<"  "<<B<<"  "<<"  "<<"  "<<Dets[i]<<endl;
        AB(I*nocc+J, A*nvirt+B) = ci(i);
      }

      //cout << occval1<<"  "<<occval2<<"  "<<virtval1<<"  "<<virtval2<<"  "<<Dets[i]<<"  "<<ci[0](i)<<endl;
    }


  }
}