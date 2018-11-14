#ifndef STATS_HEADER_H
#define STATS_HEADER_H
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

//various functions for calculating statistics of serially correlated data, functions overloaded for weighted and unweighted data sets


//average function
//	weighted data
double average(vector<double> &x, vector<double> &w);


//	unweighted data
double average(vector<double> &x);


//calculates effective sample size for weighted data sets. For unweighted data, will just return the size of the data set
double n_eff(vector<double> &w);


//variance function, takes advantage of bessel's correction
//	weighted data
double variance(vector<double> &x, vector<double> &w);


//	unweighted data
double variance(vector<double> &x);


//correlation function: given a weighted/unweighted data set, calculates C(t) = <(x(i)-x_bar)(x(i+t)-x_bar)> / <(x(i)-x_bar)^2>
//Input: x, w if weighted; x if unweighted
//Output: c
void corr_func(vector<double> &c, vector<double> &x, vector<double> &w);


void corr_func(vector<double> &c, vector<double> &x);


//autocorrelation time: given correlation function, calculates autocorrelation time: t = 1 + 2 \sum C(i)
double corr_time(vector<double> &c);


//writes correlation function to text file
void write_corr_func(vector<double> &c);


//blocking function, given a wighted or unweighted data set, calculates autocorrelation time vs. block size
//Input: x, w if weighted; x if unweighted
//Output: b_size - block size per iteration, r_t - autocorrelation time per iteration
void blocking(vector<double> &b_size, vector<double> &r_t, vector<double> &x, vector<double> &w);

void blocking(vector<double> &b_size, vector<double> &r_t, vector<double> &x);


//autocorrelation time: given blocking data, finds autocorrelation time based on the criteria: (block size)^3 > 2 * (number of original data points) * (autocorrelation time)^2
double corr_time(double n_original, vector<double> &b_size, vector<double> &r_t);


//writes blocking data to file
void write_block(vector<double> &b_size, vector<double> &r_t);

#endif
