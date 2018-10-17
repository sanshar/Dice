/*
  Developed by Sandeep Sharma
  Copyright (c) 2017, Sandeep Sharma
  
  This file is part of DICE.
  
  This program is free software: you can redistribute it and/or modify it under the terms
  of the GNU General Public License as published by the Free Software Foundation, 
  either version 3 of the License, or (at your option) any later version.
  
  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
  See the GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License along with this program. 
  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef OPTIMIZERSR_HEADER_H
#define OPTIMIZERSR_HEADER_H
#include <Eigen/Dense>
#include <boost/serialization/serialization.hpp>
#include "iowrapper.h"
#include "global.h"
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>

#ifndef SERIAL
#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif

using namespace Eigen;
using namespace boost;

void conj_grad(MatrixXd &A, VectorXd &b, VectorXd &xguess, int n, VectorXd &x)
{
    VectorXd x0 = xguess;
    VectorXd r0 = b - A * x0;
    VectorXd p0 = r0;
    double r0_square = r0.adjoint() * r0;
    for (int i = 0; i < n; i++)
    {
	VectorXd Ap = A * p0;
	double pAp = p0.adjoint() * Ap;
	double alpha = r0_square / pAp;
	VectorXd x1 = x0 + alpha * p0;
	VectorXd r1 = r0 - alpha * Ap;
	double r1_square = r1.adjoint() * r1;
	double beta = r1_square / r0_square;
	VectorXd p1 = r1 + beta * p0;

	x0 = x1;
	r0 = r1;
	p0 = p1;
	r0_square = r1_square;
        x = x1;
    }
}

void pinv(MatrixXd &A, MatrixXd &Ainv)
{
    SelfAdjointEigenSolver<MatrixXd> es;
    es.compute(A);
    std::vector<int> cols;
    for (int m = 0; m < A.cols(); m++)
    {
      if (fabs(es.eigenvalues()(m)) > 1.e-10)
      {
        cols.push_back(m);
      }
    }
    MatrixXd U = MatrixXd::Zero(A.rows(), cols.size());
    MatrixXd eig_inv = MatrixXd::Zero(cols.size(),cols.size());
    for (int m = 0; m < cols.size(); m++)
    {
      int index = cols[m];
      U.col(m) = es.eigenvectors().col(index);
      double eigval = es.eigenvalues()(index);
      eig_inv(m,m) = 1.0 / eigval;
    }
    Ainv = U * eig_inv * U.adjoint();
}

class SR
{
  private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & stepsize
            & iter;
    }

  public:
    double stepsize;
    int maxIter;
    int iter;

    SR(double pstepsize=0.001, int pmaxIter=1000) : stepsize(pstepsize), maxIter(pmaxIter)
    {
        iter = 0;
    }

    void write(VectorXd& vars)
    {
        if (commrank == 0)
        {
            char file[5000];
            sprintf(file, "sr.bkp");
            std::ofstream ofs(file, std::ios::binary);
            boost::archive::binary_oarchive save(ofs);
            save << *this;
            save << vars;
            ofs.close();
        }
    }

    void read(VectorXd& vars)
    {
        if (commrank == 0)
        {
            char file[5000];
            sprintf(file, "sr.bkp");
            std::ifstream ifs(file, std::ios::binary);
            boost::archive::binary_iarchive load(ifs);
            load >> *this;
            load >> vars;
            ifs.close();
        }
    }

   template<typename Function>
    void optimize(VectorXd &vars, Function& getMetric, bool restart)
    {
        if (restart)
        {
            if (commrank == 0)
                read(vars);
#ifndef SERIAL
	    boost::mpi::communicator world;
	    boost::mpi::broadcast(world, *this, 0);
	    boost::mpi::broadcast(world, vars, 0);
#endif
        }

        VectorXd grad, b, x;
        MatrixXd s;
        double E0, stddev, rt;
        while (iter < maxIter)
        {
            E0 = 0.0;
            stddev = 0.0;
            rt = 0.0;
            grad.setZero(vars.rows());
	    b.setZero(vars.rows() + 1);
	    x.setZero(vars.rows() + 1);
	    s.setZero(vars.rows() + 1, vars.rows() + 1);
            
            getMetric(vars, grad, s, E0, stddev, rt);
            write(vars);

          if (commrank == 0) {  
            for (int l = 0; l < b.rows(); l++)
            {
                if (l == 0)
                {
                  b(l) = s(l,0);
                }
                else
                {
                  b(l) = s(l,0) - stepsize * grad(l-1);
                }
            }
/*
            s.block(1,1,vars.rows(),vars.rows()) += 0.1 * MatrixXd::Identity(vars.rows(), vars.rows());
            VectorXd xguess = VectorXd::Zero(b.rows());
            for (int i = 0; i < xguess.rows(); i++)
            {
                if (i == 0)
                {
                  xguess(i) = 1.0;
                }
                else 
                {
                  xguess(i) = vars(i-1);
                }
            }
            VectorXd x = VectorXd::Zero(b.rows());
            conj_grad(s,b,xguess,10,x);
*/

            MatrixXd s_inv = MatrixXd::Zero(vars.rows() + 1, vars.rows() + 1);
            pinv(s,s_inv);
	    x = s_inv * b; 

            for (int i = 0; i < vars.rows(); i++)
            {
               vars(i) += (x(i+1) / x(0));
            }
          }

#ifndef SERIAL
            MPI_Bcast(&vars[0], vars.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

            if (commrank == 0)
                std::cout << format("%5i %14.8f (%8.2e) %14.8f %8.1f %10i %8.2f\n") % iter % E0 % stddev % (grad.norm()) % (rt) % (schd.stochasticIter) % ((getTime() - startofCalc));
            iter++;
        }
    }
};

#endif
