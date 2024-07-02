#pragma once
#include <Eigen/Dense>
#include <vector>

class Determinant;

void getMatrix(std::vector<Determinant> &Dets, int norbs, int nocc, Eigen::MatrixXd& ci, 
            Eigen::MatrixXd& A, Eigen::MatrixXd& B, Eigen::MatrixXd& AA, Eigen::MatrixXd& BB, Eigen::MatrixXd& AB, 
            Eigen::MatrixXi& AAidx, Eigen::MatrixXi& BBidx, Eigen::MatrixXi& ABidx);