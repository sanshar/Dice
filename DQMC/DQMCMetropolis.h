#ifndef DQMCSampling_HEADER_H
#define DQMCSampling_HEADER_H
#include <Eigen/Dense>
#include <utility>
#include <vector>


// calculates energy of the imaginary time propagated wave function
// w/o jastrow
void calcEnergyMetropolis(double enuc, Eigen::MatrixXd& h1, Eigen::MatrixXd& h1Mod, std::vector<Eigen::MatrixXd>& chol);


// calculates variational energy estimator of the imaginary time propagated wave function
// w/ jastrow
void calcEnergyJastrowMetropolis(double enuc, Eigen::MatrixXd& h1, Eigen::MatrixXd& h1Mod, std::vector<Eigen::MatrixXd>& chol);
