/*
  Developed by Sandeep Sharma with contributions from James E. T. Smith and Adam
  A. Holmes, 2017 Copyright (c) 2017, Sandeep Sharma

  This file is part of DICE.

  This program is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or (at your option) any later
  version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with
  this program. If not, see <http://www.gnu.org/licenses/>.
*/
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#define N 1000000

using namespace std;

int main() {
  double *v = new double[N];
  double *w = new double[N];
  int n = 0;
  void Correlation(double v[], double w[], double n);
  void Blocking(double v[], double w[], double n);
  void Histogram(double v[], double w[], double n);
  int dummy;
  cin >> dummy >> v[n];
  w[n] = 1.0;
  while (!cin.eof()) {
    n++;
    cin >> dummy >> v[n];
    w[n] = 1.0;
  };
  Correlation(v, w, n);
  Blocking(v, w, n);
  Histogram(v, w, n);
  delete[] v;
  delete[] w;
  return 0;
};

void Correlation(double v[], double w[], double n) {
  int l;
  double var, dev, rk, f;
  double aver = 0.0;
  double norm = 0.0;
  double neff;
  double c[2000];

  ofstream file_corr;
  for (int i = 0; i < n; i++) {
    aver += v[i] * w[i];
    norm += w[i];
  };
  aver = aver / norm;

  neff = 0.0;
  for (int i = 0; i < n; i++) {
    neff = neff + w[i] * w[i];
  };
  neff = norm * norm / neff;

  var = 0.0;
  for (int i = 0; i < n; i++) {
    var = var + w[i] * (v[i] - aver) * (v[i] - aver);
  };
  var = var / norm;
  var = var * neff / (neff - 1.0);
  dev = sqrt(var);

  file_corr.open("corr.out");
  l = min(int(n) - 1, 2000);
  for (int i = 1; i < l; i++) {
    c[i] = 0.0;
    norm = 0.0;
    for (int k = 0; k < n - i; k++) {
      c[i] = c[i] + sqrt(w[k] * w[k + i]) * (v[k] - aver) * (v[k + i] - aver);
      norm = norm + sqrt(w[k] * w[k + i]);
    };
    c[i] = c[i] / norm / var;
  };
  rk = 1.0;
  f = 1.0;
  for (int i = 1; i < l; i++) {
    file_corr << i << "\t" << c[i] << endl;
    if (c[i] < 0.0) f = 0.0;
    rk = rk + 2.0 * c[i] * f;
  };
  file_corr.close();
  rk = max(1.0, rk);
  cout << setprecision(10);
  cout << "\n";
  cout << "  Average :  " << aver << endl;
  cout << "  N       :  " << double(n) << endl;
  cout << "  Neff    :  " << neff << endl;
  cout << "  Variance:  " << var << endl;
  cout << "  Error   :  " << dev << endl;
  cout << "  T corr  :  " << rk << endl;
  cout << "  Neff    :  " << neff / rk << endl;
  cout << "  Error   :  " << sqrt(var * rk / neff) << endl;
  cout << "\n";
  return;
};

void Blocking(double v[], double w[], double n) {
  int large, minleft, nsizes, sstep, nblk, k;
  double aver, ave2, ab, err;
  minleft = 20;
  nsizes = 100;
  ofstream file_block;

  file_block.open("block.out");
  large = int(n / double(minleft));
  sstep = max(1, large / nsizes);
  for (int i = 1; i < large; i += sstep) {
    nblk = n / i;
    k = 0;
    aver = 0.0;
    ave2 = 0.0;
    for (int j = 0; j < nblk; j++) {
      ab = 0.0;
      for (int l = 0; l < i; l++) {
        ab += v[k];  //*w[k];
        k++;
      };
      ab = ab / double(i);
      aver += ab;
      ave2 += ab * ab;
    };
    aver = aver / double(nblk);
    ave2 = ave2 / double(nblk);

    err = sqrt((ave2 - aver * aver) / double(nblk - 1.0));
    file_block << i << "\t" << err << "\n";
  };
  file_block.close();
  return;
};

void Histogram(double v[], double w[], double n) {
  int nbin = 51, bin;
  double dmin, dmax, delta;
  double histo[nbin];
  ofstream file_out;
  for (int i = 0; i < nbin; i++) {
    histo[i] = 0;
  };
  dmin = v[0];
  dmax = v[0];
  for (int i = 1; i < n; i++) {
    dmin = min(dmin, v[i]);
    dmax = max(dmax, v[i]);
  };
  delta = (dmax - dmin) / double(nbin);
  for (int i = 0; i < n; i++) {
    bin = int(floor((v[i] - dmin) / delta) + .5);
    histo[bin] += w[i];
  };
  file_out.open("histo.out");
  for (int i = 0; i < nbin; i++) {
    file_out << dmin + (i + 0.5) * delta << "\t"
             << histo[i] / (double(n) * delta) << "\n";
  };
  file_out.close();
  return;
};
