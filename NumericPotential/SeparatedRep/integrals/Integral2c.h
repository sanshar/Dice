#pragma once

#include <Eigen/Dense>

using namespace Eigen;

double calcCoulombIntegral(int n1, double Ax, double Ay, double Az,
                           double expA, double normA,
                           double expG, double wtG,
                           int n2, double Bx, double By, double Bz,
                           double expB, double normB,
                           MatrixXd& Int);
double calcOvlpMatrix(int LA, double Ax, double Ay, double Az, double expA,
                     int LB, double Bx, double By, double Bz, double expB,
                     MatrixXd& S) ;


double calcCoulombIntegralPeriodic(int n1, double Ax, double Ay, double Az,
				   double expA, double normA,
				   double expG, double wtG,
				   int n2, double Bx, double By, double Bz,
				   double expB, double normB,
				   MatrixXd& Int);

double calcCoulombPotentialPeriodic(int LA, double Ax, double Ay, double Az,
                                    double expA, double normA,
                                    double expG, double wtG,
                                    double Bx, double By, double Bz,
                                    MatrixXd& Int);
