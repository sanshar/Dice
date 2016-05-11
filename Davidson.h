#ifndef DAVIDSON_HEADER_H
#define DAVIDSON_HEADER_H
#include <Eigen/Dense>
#include <Eigen/Core>

class Hmult;

void precondition(Eigen::MatrixXd& r, Eigen::MatrixXd& diag, double& e);
double davidson(Hmult& H, Eigen::MatrixXd& x0, Eigen::MatrixXd& diag, int maxCopies, double tol, bool print=true);

#endif
