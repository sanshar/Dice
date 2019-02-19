#include "statistics.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "global.h"
#ifndef SERIAL
#include "mpi.h"
#endif

using namespace std;

//random calcTcorr function, idk who wrote this it's pretty rough
double calcTcorr(vector<double> &v)
{
  //vector<double> w(v.size(), 1);
  int n = v.size();
  double norm, rk, f, neff;

  double aver = 0, var = 0;
  for (int i = 0; i < v.size(); i++)
  {
    aver += v[i] ;
    norm += 1.0 ;
  }
  aver = aver / norm;

  neff = 0.0;
  for (int i = 0; i < n; i++)
  {
    neff = neff + 1.0;
  };
  neff = norm * norm / neff;

  for (int i = 0; i < v.size(); i++)
  {
    var = var + (v[i] - aver) * (v[i] - aver);
  };
  var = var / norm;
  var = var * neff / (neff - 1.0);

  //double c[v.size()];
  vector<double> c(v.size(),0);
  //for (int i=0; i<v.size(); i++) c[i] = 0.0;
  int l = v.size() - 1;

  int i = commrank+1;
  for (; i < l; i+=commsize)
  //int i = 1;
  //for (; i < l; i++)
  {
    c[i] = 0.0;
    double norm = 0.0;
    for (int k = 0; k < n - i; k++)
    {
      c[i] = c[i] + (v[k] - aver) * (v[k + i] - aver);
      norm = norm + 1.0;
    };
    c[i] = c[i] / norm / var;
  };
 #ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &c[0], v.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  rk = 1.0;

  f = 1.0;

  i = 1;
  ofstream out("OldCorrFunc.txt");
  for (; i < l; i++)
  {

    if (commrank == 0)
    {
      out << c[i] << endl;
    }

    if (c[i] < 0.0)
      f = 0.0;
    rk = rk + 2.0 * c[i] * f;
  }
  out.close();
  
  return rk;
}

//various functions for calculating statistics of serially correlated data, functions overloaded for weighted and unweighted data sets

//average function
double average(vector<double> &x, vector<double> &w)
{
    double x_bar = 0.0, W = 0.0;
    int n = x.size();
    for (int i = 0; i < n; i++)
    {
        x_bar += w[i] * x[i];
        W += w[i];
    }
    x_bar /= W;
    return x_bar;
}

double average(vector<double> &x)
{
    vector<double> w(x.size(), 1.0);
    return average(x, w);
}

//calculates effective sample size for weighted data sets. For unweighted data, will just return the size of the data set
double neff(vector<double> &w)
{
    double n_eff, W = 0.0, W_2 = 0.0;
    int n = w.size();
    for (int i = 0; i < n; i++)
    {
        W += w[i];
        W_2 += w[i] * w[i];
    }
    n_eff = (W * W) / W_2;
    return n_eff;
}

//variance function, takes advantage of bessel's correction
double variance(vector<double> &x, vector<double> &w)
{
    double x_bar = 0.0, x_bar_2 = 0.0, n_eff, W = 0.0, W_2 = 0.0;
    int n = x.size();
    for (int i = 0; i < n; i++)
    {
        x_bar += w[i] * x[i];
        x_bar_2 += w[i] * x[i] * x[i];
        W += w[i];
        W_2 += w[i] * w[i];
    }
    x_bar /= W;
    x_bar_2 /= W;
    n_eff = (W * W) / W_2;

    double s_2, var;
    s_2 = x_bar_2 - x_bar * x_bar;
    var = (s_2 * n_eff) / (n_eff - 1.0);
    return var;
}

double variance(vector<double> &x)
{
    vector<double> w(x.size(), 1.0);
    return variance(x, w);
}

//correlation function: given a weighted/unweighted data set, calculates C(t) = <(x(i)-x_bar)(x(i+t)-x_bar)> / <(x(i)-x_bar)^2>
//Input: x, w if weighted; x if unweighted
//Output: c
void corrFunc(vector<double> &c, vector<double> &x, vector<double> &w)
{
    double x_bar = average(x,w);
    double var = variance(x,w);
    int n = x.size();
    int l = min(n - 1, 1000);
    double norm;
    c.assign(l, 0.0);
    c[0] = 1.0;
    for (int t = 1; t < l; t++)
    {
        norm = 0.0;
        for (int i = 0; i < n - t; i++)
        {
            c[t] += sqrt(w[i] * w[i+t]) * (x[i] - x_bar) * (x[i+t] - x_bar);
            norm += sqrt(w[i] * w[i+t]);
        }
        c[t] /= norm;
        c[t] /= var;
    }
}

void corrFunc(vector<double> &c, vector<double> &x)
{
    vector<double> w(x.size(), 1.0);
    corrFunc(c, x, w);
}

//autocorrelation time: given correlation function, calculates autocorrelation time: t = 1 + 2 \sum C(i)
double corrTime(vector<double> &c)
{
    double t = 1.0;
    for (int i = 1; i < c.size(); i++)
    {
        if (c[i] < 0.0 || c[i] < 0.01)
        {
            break;
        }
        t += 2 * c[i];
    }
    return t;
}

//writes correlation function to text file
void writeCorrFunc(vector<double> &c)
{
    ofstream ofile("CorrFunc.txt");
    for (int i = 0; i < c.size(); i++)
    {
        ofile << i << "\t" << c[i] << endl;
    }
    ofile.close();
}

//blocking function, given a wighted or unweighted data set, calculates autocorrelation time vs. block size
//Input: x, w if weighted; x if unweighted
//Output: b_size - block size per iteration, r_t - autocorrelation time per iteration
void block(vector<double> &b_size, vector<double> &r_t, vector<double> &x, vector<double> &w)
{
    b_size.clear();
    r_t.clear();
    double var = variance(x,w);
    vector<double> x_0, x_1, w_0, w_1;
    x_0 = x;
    w_0 = w;
    int N, n_0, n_1;
    double b_var, b_size_tok;
    N = x.size();
    for (int iter = 0; iter < 40; iter++)
    {
        b_size_tok = pow(2.0, (double)iter);
        if (b_size_tok > (N / 2))
        {
            break;
        }
        b_var = variance(x_0,w_0);
        b_size.push_back(b_size_tok);
        r_t.push_back((b_size_tok * b_var) / var);

        n_0 = x_0.size();
        n_1 = floor(n_0 / 2);
        x_1.assign(n_1, 0.0);
        w_1.assign(n_1, 0.0);
        for (int i = 0; i < n_1; i++)
        {
            w_1[i] = w_0[2*i] + w_0[(2*i)+1];
            x_1[i] = w_0[2*i] * x_0[2*i] + w_0[(2*i)+1] * x_0[(2*i)+1]; 
            x_1[i] /= w_1[i];
        }
        x_0 = x_1;
        w_0 = w_1;
    }
}

void block(vector<double> &b_size, vector<double> &r_t, vector<double> &x)
{
    vector<double> w(x.size(), 1.0);
    block(b_size, r_t, x, w);
}

//autocorrelation time: given blocking data, finds autocorrelation time based on the criteria: (block size)^3 > 2 * (number of original data points) * (autocorrelation time)^2
double corrTime(double n_original, vector<double> &b_size, vector<double> &r_t)
{
    double t = 0.0;
    for (int i = 0; i < r_t.size(); i++)
    {
        double B = b_size[i] * b_size[i] * b_size[i];
        double C = 2 * n_original * (r_t[i] * r_t[i]);
        if (B > C)
        {
            t = r_t[i];
            break;
        }
    }
    if (t == 0.0)
    {
        //throw runtime_error("Insufficient number of stochastic iterations");
        t = 1.;
    }
    if (t < 1.0)
    {
      t = 1.0;
    }
    return t;
}

//writes blocking data to file
void writeBlock(vector<double> &b_size, vector<double> &r_t)
{
    ofstream ofile("Block.txt");
    for (int i = 0; i < r_t.size(); i++)
    {
        ofile << b_size[i] << "\t" << r_t[i] << endl;
    }
    ofile.close();
}
