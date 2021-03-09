#ifndef SERIAL
#include <iomanip>
#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include <boost/format.hpp>
#include <unsupported/Eigen/MatrixFunctions>
#include "input.h"
#include "global.h"
#include "Determinants.h"
#include "DQMCUtils.h"
#include "DQMCMatrixElements.h"
#include "DQMCMetropolis.h"
#include <iomanip> 

using namespace Eigen;
using namespace std;
using namespace boost;

using matPair = pair<MatrixXcd, MatrixXcd>;
using vecPair = pair<VectorXcd, VectorXcd>;
// calculates energy of the imaginary time propagated wave function
// w/o jastrow
void calcEnergyMetropolis(double enuc, MatrixXd& h1, MatrixXd& h1Mod, vector<MatrixXd>& chol)
{
  size_t norbs = Determinant::norbs;
  size_t nfields = chol.size();
  size_t nsweeps = schd.stochasticIter;
  size_t nsteps = schd.nsteps;
  size_t orthoSteps = schd.orthoSteps;
  double stepsize = schd.fieldStepsize;
  double dt = schd.dt;

  // prep and init
  // this is for mean field subtraction
  MatrixXd hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair rhf;
  rhf.first = hf.block(0, 0, norbs, Determinant::nalpha);
  rhf.second = hf.block(0, 0, norbs, Determinant::nbeta);
  
  vector<matPair> hsOperators;
  matPair oneBodyOperator;
  prepPropagatorHS(rhf, chol, hsOperators, oneBodyOperator);
  
  // this is the actual reference in J | ref >
  hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair ref;
  ref.first = hf.block(0, 0, norbs, Determinant::nalpha);
  ref.second = hf.block(0, 0, norbs, Determinant::nbeta);

  // the first and second are the same for now
  matPair expOneBodyOperator;
  expOneBodyOperator.first =  (-dt * (h1Mod - oneBodyOperator.first) / 2.).exp();
  expOneBodyOperator.second = (-dt * (h1Mod - oneBodyOperator.second) / 2.).exp();
  // normalizing for numerical stability
  JacobiSVD<MatrixXcd> svd1(expOneBodyOperator.first);
  JacobiSVD<MatrixXcd> svd2(expOneBodyOperator.second);
  expOneBodyOperator.first /= svd1.singularValues()(0);
  expOneBodyOperator.second /= svd2.singularValues()(0);

  // fields initialized to random values
  // arranged right to left
  vector<VectorXd> fields;
  uniform_real_distribution<double> uniformStep(-stepsize, stepsize);
  uniform_real_distribution<double> uniform(0., 1.);
  normal_distribution<double> normal(0., 1.);

  complex<double> rOrthoFac = 1., lOrthoFac = 1.;
  // right to left sweep
  vector<matPair> rn;
  rn.push_back(ref);
  for (int n = 0; n < nsteps; n++) {
    fields.push_back(VectorXd::Zero(nfields));
    matPair prop;
    prop.first = MatrixXcd::Zero(norbs, norbs);
    prop.second = MatrixXcd::Zero(norbs, norbs);
    for (int i = 0; i < nfields; i++) {
      fields[n](i) = normal(generator);//uniformStep(generator);
      prop.first += fields[n](i) * hsOperators[i].first;
      prop.second += fields[n](i) * hsOperators[i].second;
    }
    prop.first = (sqrt(dt) * prop.first).exp();
    prop.second = (sqrt(dt) * prop.second).exp();
    matPair rni;
    rni.first = expOneBodyOperator.first * prop.first * expOneBodyOperator.first * rn[n].first;
    rni.second = expOneBodyOperator.second * prop.second * expOneBodyOperator.second * rn[n].second;

    // orthogonalize for stability
    if (n % orthoSteps == 0) {
      orthogonalize(rni, rOrthoFac);
    }

    rn.push_back(rni);
  }

  // left to right sweep
  vector<matPair> ln;
  matPair refT;
  refT.first = ref.first.adjoint();
  refT.second = ref.second.adjoint();
  ln.push_back(refT);
  for (int n = 0; n < nsteps; n++) {
    matPair prop;
    prop.first = MatrixXcd::Zero(norbs, norbs);
    prop.second = MatrixXcd::Zero(norbs, norbs);
    for (int i = 0; i < nfields; i++) {
      prop.first += fields[nsteps - n - 1](i) * hsOperators[i].first;
      prop.second += fields[nsteps - n - 1](i) * hsOperators[i].second;
    }
    prop.first = (sqrt(dt) * prop.first).exp();
    prop.second = (sqrt(dt) * prop.second).exp();
    matPair lni;
    lni.first = ln[n].first * expOneBodyOperator.first * prop.first * expOneBodyOperator.first;
    lni.second = ln[n].second * expOneBodyOperator.second * prop.second * expOneBodyOperator.second;

    // orthogonalize for stability
    if (n % orthoSteps == 0) {
      matPair lniT; lniT.first = lni.first.adjoint(); lniT.second = lni.second.adjoint();
      orthogonalize(lniT, lOrthoFac);
      lni.first = lniT.first.adjoint(); lni.second = lniT.second.adjoint();
    }

    ln.push_back(lni);
  }

  complex<double> overlap = (ln[0].first * rn[nsteps].first).determinant() * (ln[0].second * rn[nsteps].second).determinant();
  VectorXcd overlaps = VectorXcd::Zero(2*nsweeps), num = VectorXcd::Zero(nsweeps), denom = VectorXcd::Zero(nsweeps);
  // metropolis sweep
  // moves proposed by varying all fields at a time slice
  size_t accepted = 0, counter = 0;
  auto iterTime = getTime();
  double expTime = 0., propBuildTime = 0., propTime = 0., energyTime = 0., detTime = 0.;
  for (int sweep = 0; sweep < 2*nsweeps; sweep++) {
    if (sweep % (2*nsweeps/5) == 0 && commrank == 0) cout << sweep/2 << "  " << getTime() - iterTime << endl;
    
    // right to left sweep
    if (sweep % 2 == 0) {
      for (int n = 0; n < nsteps; n++) {
        rOrthoFac = 1.0;
        // propose move
        double init = getTime();
        VectorXd proposedFields = VectorXd::Zero(nfields);
        matPair proposedProp;
        proposedProp.first = MatrixXcd::Zero(norbs, norbs);
        proposedProp.second = MatrixXcd::Zero(norbs, norbs);
        double expRatio = 1.;
        for (int i = 0; i < nfields; i++) {
          //proposedFields(i) = fields[n](i) + uniformStep(generator);
          //expRatio *= exp((fields[n](i) * fields[n](i) - proposedFields(i) * proposedFields(i))/2);
          proposedFields(i) = normal(generator);
          proposedProp.first += proposedFields(i) * hsOperators[i].first;
          proposedProp.second += proposedFields(i) * hsOperators[i].second;
        }
        propBuildTime += getTime() - init;
        init = getTime();
        proposedProp.first = (sqrt(dt) * proposedProp.first).exp();
        proposedProp.second = (sqrt(dt) * proposedProp.second).exp();
        expTime += getTime() - init;
        init = getTime();
        matPair rni;
        rni.first = expOneBodyOperator.first * proposedProp.first * expOneBodyOperator.first * rn[n].first;
        rni.second = expOneBodyOperator.second * proposedProp.second * expOneBodyOperator.second * rn[n].second;
        propTime += getTime() - init;
        complex<double> proposedOverlap = (ln[nsteps - n - 1].first * rni.first).determinant() * (ln[nsteps - n - 1].second * rni.second).determinant();

        //cout << proposedOverlap<<"  "<<overlap<<endl;
        // accept / reject
        if (expRatio * abs(proposedOverlap) / abs(overlap) >= uniform(generator)) {
          accepted++;
          fields[n] = proposedFields;
          overlap = proposedOverlap;
          rn[n + 1] = rni;
        }
        else {
          matPair prop;
          prop.first = MatrixXcd::Zero(norbs, norbs);
          prop.second = MatrixXcd::Zero(norbs, norbs);
          for (int i = 0; i < nfields; i++) {
            prop.first += fields[n](i) * hsOperators[i].first;
            prop.second += fields[n](i) * hsOperators[i].second;
          }
          prop.first = (sqrt(dt) * prop.first).exp();
          prop.second = (sqrt(dt) * prop.second).exp();
          rn[n + 1].first = expOneBodyOperator.first * prop.first * expOneBodyOperator.first * rn[n].first;
          rn[n + 1].second = expOneBodyOperator.second * prop.second * expOneBodyOperator.second * rn[n].second;
        }

        if (n % orthoSteps == 0) {
          orthogonalize(rn[n+1], rOrthoFac);
        }

      }
      
      matPair green;
      calcGreensFunction(ln[0], rn[nsteps], green);
      denom(sweep/2) = overlap / abs(overlap);
      num(sweep/2) = denom(sweep/2) * calcHamiltonianElement(green, enuc, h1, chol);
      //cout << denom(sweep/2)<<"  "<<num(sweep/2)<<"  "<<sweep/2<<"  "<<accepted<<endl;
    }
    
    // left to right sweep
    else {
      for (int n = 0; n < nsteps; n++) {
        lOrthoFac = 1.0;

        // propose move
        VectorXd proposedFields = VectorXd::Zero(nfields);
        matPair proposedProp;
        proposedProp.first = MatrixXcd::Zero(norbs, norbs);
        proposedProp.second = MatrixXcd::Zero(norbs, norbs);
        double expRatio = 1.;
        for (int i = 0; i < nfields; i++) {
          //proposedFields(i) = fields[nsteps - n - 1](i) + uniformStep(generator);
          //expRatio *= exp((fields[nsteps - n - 1](i) * fields[nsteps - n - 1](i) - proposedFields(i) * proposedFields(i))/2);
          proposedFields(i) = normal(generator);
          proposedProp.first += proposedFields(i) * hsOperators[i].first;
          proposedProp.second += proposedFields(i) * hsOperators[i].second;
        }
        proposedProp.first = (sqrt(dt) * proposedProp.first).exp();
        proposedProp.second = (sqrt(dt) * proposedProp.second).exp();
        matPair lni;
        lni.first = ln[n].first * expOneBodyOperator.first * proposedProp.first * expOneBodyOperator.first;
        lni.second = ln[n].second * expOneBodyOperator.second * proposedProp.second * expOneBodyOperator.second;
        complex<double> proposedOverlap = (lni.first * rn[nsteps - n - 1].first).determinant() * (lni.second * rn[nsteps - n - 1].second).determinant();

        // accept / reject
        if (expRatio * abs(proposedOverlap) / abs(overlap) >= uniform(generator)) {
          accepted++;
          fields[nsteps - n - 1] = proposedFields;
          overlap = proposedOverlap;
          ln[n + 1] = lni;
        }
        else {
          matPair prop;
          prop.first = MatrixXcd::Zero(norbs, norbs);
          prop.second = MatrixXcd::Zero(norbs, norbs);
          for (int i = 0; i < nfields; i++) {
            prop.first += fields[nsteps - n - 1](i) * hsOperators[i].first;
            prop.second += fields[nsteps - n - 1](i) * hsOperators[i].second;
          }
          prop.first = (sqrt(dt) * prop.first).exp();
          prop.second = (sqrt(dt) * prop.second).exp();
          ln[n + 1].first = ln[n].first * expOneBodyOperator.first * prop.first * expOneBodyOperator.first;
          ln[n + 1].second = ln[n].second * expOneBodyOperator.second * prop.second * expOneBodyOperator.second;
        }

        if (n % orthoSteps == 0) {
          matPair lniT; lniT.first = ln[n+1].first.adjoint(); lniT.second = ln[n+1].second.adjoint();
          orthogonalize(lniT, lOrthoFac);
          ln[n+1].first = lniT.first.adjoint(); ln[n+1].second = lniT.second.adjoint();
        }


      }
    }
    
    overlaps(sweep) = overlap;
  }

  if (commrank == 0) {
    cout << "expTime: " << expTime << ", propBuildTime: " << propBuildTime << ", propTime: " << propTime << ", energyTime: " << energyTime << endl;
    VectorXi binSizes(4);
    binSizes << 1, 10, 50, 100;
    VectorXd stdDev = VectorXd::Zero(binSizes.size());
    binning(num, stdDev, binSizes);
    cout << "binning\n" << binSizes.transpose() << endl;
    cout << stdDev.transpose() << endl << endl;
  }
  complex<double> numMean = num.mean();
  complex<double> denomMean = denom.mean();
  //cout << denom.mean()<<"  "<<numMean<<endl;
  complex<double> energyTotAll[commsize];
  complex<double> numTotAll[commsize];
  complex<double> denomTotAll[commsize];
  for (int i = 0; i < commsize; i++) {
    energyTotAll[i] = complex<double>(0., 0.);
    numTotAll[i] = complex<double>(0., 0.);
    denomTotAll[i] = complex<double>(0., 0.);
  }
  
  complex<double> energyProc = numMean / denomMean;
  MPI_Gather(&(energyProc), 1, MPI_DOUBLE_COMPLEX, &(energyTotAll), 1, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
  MPI_Gather(&(numMean), 1, MPI_DOUBLE_COMPLEX, &(numTotAll), 1, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
  MPI_Gather(&(denomMean), 1, MPI_DOUBLE_COMPLEX, &(denomTotAll), 1, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &energyProc, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &numMean, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &denomMean, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
  
  energyProc /= commsize;
  numMean /= commsize;
  denomMean /= commsize;
  double stddev = 0., stddev2 = 0.;
  for (int i = 0; i < commsize; i++) {
    stddev += pow(abs(energyTotAll[i] - energyProc), 2);
    stddev2 += pow(abs(energyTotAll[i] - energyProc), 4);
  }
  stddev /= (commsize - 1);
  stddev2 /= commsize;
  stddev2 = sqrt((stddev2 - (commsize - 3) * pow(stddev, 2) / (commsize - 1)) / commsize) / 2. / sqrt(stddev) / sqrt(sqrt(commsize));
  stddev = sqrt(stddev / commsize);

  double acceptanceRatio = accepted / (2. * nsweeps) / nsteps;

  if (commrank == 0) {
    cout << "\nAcceptance ratio:  " << acceptanceRatio << endl;
    cout << "Numerator:  " << numMean << ", Denominator:  " << denomMean << endl;
    cout << "Energy:  " << energyProc << " (" << stddev << ") " << " (" << stddev2 << ")\n";
  }
  
}


// calculates variational energy estimator of the imaginary time propagated wave function
// w/ jastrow
void calcEnergyJastrowMetropolis(double enuc, MatrixXd& h1, MatrixXd& h1Mod, vector<MatrixXd>& chol)
{
  size_t norbs = Determinant::norbs;
  size_t nfields = chol.size();
  size_t nsweeps = schd.stochasticIter;
  size_t nsteps = schd.nsteps;
  //size_t orthoSteps = schd.orthoSteps;
  double stepsize = schd.fieldStepsize;
  double dt = schd.dt;

  // prep and init
  // this is for mean field subtraction
  MatrixXd hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair rhf;
  rhf.first = hf.block(0, 0, norbs, Determinant::nalpha);
  rhf.second = hf.block(0, 0, norbs, Determinant::nbeta);
  
  // jastrow operators
  vector<vecPair> jhsOperators;
  vecPair joneBodyOperator;
  prepJastrowHS(rhf, jhsOperators, joneBodyOperator);
  size_t jnfields = jhsOperators.size();
 
  // propagator
  vector<matPair> hsOperators;
  matPair oneBodyOperator;
  prepPropagatorHS(rhf, chol, hsOperators, oneBodyOperator);
  
  // this is the actual reference in J | ref >
  hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "ref.txt");
  matPair ref;
  ref.first = hf.block(0, 0, norbs, Determinant::nalpha);
  ref.second = hf.block(0, 0, norbs, Determinant::nbeta);
  

  // the first and second are the same for now
  matPair expOneBodyOperator;
  expOneBodyOperator.first =  (-dt * (h1Mod - oneBodyOperator.first) / 2.).exp();
  expOneBodyOperator.second = (-dt * (h1Mod - oneBodyOperator.second) / 2.).exp();
  // normalizing for numerical stability
  JacobiSVD<MatrixXcd> svd1(expOneBodyOperator.first);
  JacobiSVD<MatrixXcd> svd2(expOneBodyOperator.second);
  expOneBodyOperator.first /= svd1.singularValues()(0);
  expOneBodyOperator.second /= svd2.singularValues()(0);

  
  // fields initialized to random values
  // arranged right to left
  vector<VectorXd> fields;
  
  // right to left sweep
  vector<matPair> rn;
  vecPair jexpOneBodyOperator;
  jexpOneBodyOperator.first = joneBodyOperator.first.array().exp();
  jexpOneBodyOperator.second = joneBodyOperator.second.array().exp();
  matPair jref;
  jref.first = jexpOneBodyOperator.first.asDiagonal() * ref.first;
  jref.second = jexpOneBodyOperator.second.asDiagonal() * ref.second;
  rn.push_back(jref);
  
  // right jastrow fields
  pair<VectorXd, VectorXd> jfields;
  jfields.first = stepsize * VectorXd::Random(jnfields);
  vecPair jpropRight;
  jpropRight.first = VectorXcd::Zero(norbs);
  jpropRight.second = VectorXcd::Zero(norbs);
  for (int i = 0; i < jnfields; i++) {
    jpropRight.first += jfields.first(i) * jhsOperators[i].first;
    jpropRight.second += jfields.first(i) * jhsOperators[i].second;
  }
  matPair jright;
  jright.first = jpropRight.first.array().exp().matrix().asDiagonal() * rn[0].first;
  jright.second = jpropRight.second.array().exp().matrix().asDiagonal() * rn[0].second;
  rn.push_back(jright);
  
  // propagator fields
  for (int n = 0; n < nsteps; n++) {
    fields.push_back(stepsize * VectorXd::Random(nfields));
    matPair prop;
    prop.first = MatrixXcd::Zero(norbs, norbs);
    prop.second = MatrixXcd::Zero(norbs, norbs);
    for (int i = 0; i < nfields; i++) {
      prop.first += fields[n](i) * hsOperators[i].first;
      prop.second += fields[n](i) * hsOperators[i].second;
    }
    prop.first = (sqrt(dt) * prop.first).exp();
    prop.second = (sqrt(dt) * prop.second).exp();
    matPair rni;
    rni.first = expOneBodyOperator.first * prop.first * expOneBodyOperator.first * rn[n + 1].first;
    rni.second = expOneBodyOperator.second * prop.second * expOneBodyOperator.second * rn[n + 1].second;
    rn.push_back(rni);
  }
  
 
  // left to right sweep
  vector<matPair> ln;
  matPair jrefT;
  jrefT.first = jref.first.adjoint();
  jrefT.second = jref.second.adjoint();
  ln.push_back(jrefT);
  
  // left jastrow
  jfields.second = stepsize * VectorXd::Random(jnfields);
  vecPair jpropLeft;
  jpropLeft.first = VectorXcd::Zero(norbs);
  jpropLeft.second = VectorXcd::Zero(norbs);
  for (int i = 0; i < jnfields; i++) {
    jpropLeft.first += jfields.second(i) * jhsOperators[i].first;
    jpropLeft.second += jfields.second(i) * jhsOperators[i].second;
  }
  matPair jleft;
  jleft.first = ln[0].first * jpropLeft.first.array().exp().matrix().asDiagonal();
  jleft.second = ln[0].second * jpropLeft.second.array().exp().matrix().asDiagonal();
  ln.push_back(jleft);

  // propagator
  for (int n = 0; n < nsteps; n++) {
    matPair prop;
    prop.first = MatrixXcd::Zero(norbs, norbs);
    prop.second = MatrixXcd::Zero(norbs, norbs);
    for (int i = 0; i < nfields; i++) {
      prop.first += fields[nsteps - n - 1](i) * hsOperators[i].first;
      prop.second += fields[nsteps - n - 1](i) * hsOperators[i].second;
    }
    prop.first = (sqrt(dt) * prop.first).exp();
    prop.second = (sqrt(dt) * prop.second).exp();
    matPair lni;
    lni.first = ln[n + 1].first * expOneBodyOperator.first * prop.first * expOneBodyOperator.first;
    lni.second = ln[n + 1].second * expOneBodyOperator.second * prop.second * expOneBodyOperator.second;
    ln.push_back(lni);
  }
  
  complex<double> overlap = (ln[1].first * rn[nsteps + 1].first).determinant() * (ln[1].second * rn[nsteps + 1].second).determinant();
  VectorXcd overlaps = VectorXcd::Zero(2*nsweeps), num = VectorXcd::Zero(nsweeps), denom = VectorXcd::Zero(nsweeps);
  

  // metropolis sweep
  // moves proposed by varying all fields at a time slice
  size_t accepted = 0;
  uniform_real_distribution<double> uniformStep(-stepsize, stepsize);
  uniform_real_distribution<double> uniform(0., 1.);
  auto iterTime = getTime();
  for (int sweep = 0; sweep < 2*nsweeps; sweep++) {
    if (sweep % (2*nsweeps/5) == 0 && commrank == 0) cout << sweep/2 << "  " << getTime() - iterTime << endl;
    
    // right to left sweep
    if (sweep%2 == 0) {
      // right jastrow
      VectorXd jproposedFields = VectorXd::Zero(jnfields);
      vecPair jproposedProp;
      jproposedProp.first = VectorXcd::Zero(norbs);
      jproposedProp.second = VectorXcd::Zero(norbs);
      double jexpRatio = 1.;

      // propose move
      for (int i = 0; i < jnfields; i++) {
        jproposedFields(i) = jfields.first(i) + uniformStep(generator);
        jexpRatio *= exp((jfields.first(i) * jfields.first(i) - jproposedFields(i) * jproposedFields(i))/2);
        jproposedProp.first += jproposedFields(i) * jhsOperators[i].first;
        jproposedProp.second += jproposedFields(i) * jhsOperators[i].second;
      }
      matPair jproposed;
      jproposed.first = jproposedProp.first.array().exp().matrix().asDiagonal() * rn[0].first;
      jproposed.second = jproposedProp.second.array().exp().matrix().asDiagonal() * rn[0].second;
      complex<double> jproposedOverlap = (ln[nsteps + 1].first * jproposed.first).determinant() * (ln[nsteps + 1].second * jproposed.second).determinant();

      // accept / reject
      if (jexpRatio * abs(jproposedOverlap) / abs(overlap) >= uniform(generator)) {
        accepted++;
        jfields.first = jproposedFields;
        overlap = jproposedOverlap;
        rn[1] = jproposed;
      }

      // propagator
      for (int n = 0; n < nsteps; n++) {
        // propose move
        VectorXd proposedFields = VectorXd::Zero(nfields);
        matPair proposedProp;
        proposedProp.first = MatrixXcd::Zero(norbs, norbs);
        proposedProp.second = MatrixXcd::Zero(norbs, norbs);
        double expRatio = 1.;
        for (int i = 0; i < nfields; i++) {
          proposedFields(i) = fields[n](i) + uniformStep(generator);
          expRatio *= exp((fields[n](i) * fields[n](i) - proposedFields(i) * proposedFields(i))/2);
          proposedProp.first += proposedFields(i) * hsOperators[i].first;
          proposedProp.second += proposedFields(i) * hsOperators[i].second;
        }
        proposedProp.first = (sqrt(dt) * proposedProp.first).exp();
        proposedProp.second = (sqrt(dt) * proposedProp.second).exp();
        matPair rni;
        rni.first = expOneBodyOperator.first * proposedProp.first * expOneBodyOperator.first * rn[n + 1].first;
        rni.second = expOneBodyOperator.second * proposedProp.second * expOneBodyOperator.second * rn[n + 1].second;
        complex<double> proposedOverlap = (ln[nsteps - n].first * rni.first).determinant() * (ln[nsteps - n].second * rni.second).determinant();

        // accept / reject
        if (expRatio * abs(proposedOverlap) / abs(overlap) >= uniform(generator)) {
          accepted++;
          fields[n] = proposedFields;
          overlap = proposedOverlap;
          rn[n + 2] = rni;
        }
        else {
          matPair prop;
          prop.first = MatrixXcd::Zero(norbs, norbs);
          prop.second = MatrixXcd::Zero(norbs, norbs);
          for (int i = 0; i < nfields; i++) {
            prop.first += fields[n](i) * hsOperators[i].first;
            prop.second += fields[n](i) * hsOperators[i].second;
          }
          prop.first = (sqrt(dt) * prop.first).exp();
          prop.second = (sqrt(dt) * prop.second).exp();
          rn[n + 2].first = expOneBodyOperator.first * prop.first * expOneBodyOperator.first * rn[n + 1].first;
          rn[n + 2].second = expOneBodyOperator.second * prop.second * expOneBodyOperator.second * rn[n + 1].second;
        }
      }
      
      matPair green;
      calcGreensFunction(ln[1], rn[nsteps + 1], green);
      denom(sweep/2) = overlap / abs(overlap);
      num(sweep/2) = denom(sweep/2) * calcHamiltonianElement(green, enuc, h1, chol);
    }
    
    // left to right sweep
    else {
      // left jastrow
      VectorXd jproposedFields = VectorXd::Zero(jnfields);
      vecPair jproposedProp;
      jproposedProp.first = VectorXcd::Zero(norbs);
      jproposedProp.second = VectorXcd::Zero(norbs);
      double jexpRatio = 1.;

      // propose move
      for (int i = 0; i < jnfields; i++) {
        jproposedFields(i) = jfields.second(i) + uniformStep(generator);
        jexpRatio *= exp((jfields.second(i) * jfields.second(i) - jproposedFields(i) * jproposedFields(i))/2);
        jproposedProp.first += jproposedFields(i) * jhsOperators[i].first;
        jproposedProp.second += jproposedFields(i) * jhsOperators[i].second;
      }
      matPair jproposed;
      jproposed.first = ln[0].first * jproposedProp.first.array().exp().matrix().asDiagonal();
      jproposed.second = ln[0].second * jproposedProp.second.array().exp().matrix().asDiagonal();
      complex<double> jproposedOverlap = (jproposed.first * rn[nsteps + 1].first).determinant() * (jproposed.second * rn[nsteps + 1].second).determinant();

      // accept / reject
      if (jexpRatio * abs(jproposedOverlap) / abs(overlap) >= uniform(generator)) {
        accepted++;
        jfields.second = jproposedFields;
        overlap = jproposedOverlap;
        ln[1] = jproposed;
      }
      
      // propagator
      for (int n = 0; n < nsteps; n++) {
        // propose move
        VectorXd proposedFields = VectorXd::Zero(nfields);
        matPair proposedProp;
        proposedProp.first = MatrixXcd::Zero(norbs, norbs);
        proposedProp.second = MatrixXcd::Zero(norbs, norbs);
        double expRatio = 1.;
        for (int i = 0; i < nfields; i++) {
          proposedFields(i) = fields[nsteps - n - 1](i) + uniformStep(generator);
          expRatio *= exp((fields[nsteps - n - 1](i) * fields[nsteps - n - 1](i) - proposedFields(i) * proposedFields(i))/2);
          proposedProp.first += proposedFields(i) * hsOperators[i].first;
          proposedProp.second += proposedFields(i) * hsOperators[i].second;
        }
        proposedProp.first = (sqrt(dt) * proposedProp.first).exp();
        proposedProp.second = (sqrt(dt) * proposedProp.second).exp();
        matPair lni;
        lni.first = ln[n + 1].first * expOneBodyOperator.first * proposedProp.first * expOneBodyOperator.first;
        lni.second = ln[n + 1].second * expOneBodyOperator.second * proposedProp.second * expOneBodyOperator.second;
        complex<double> proposedOverlap = (lni.first * rn[nsteps - n].first).determinant() * (lni.second * rn[nsteps - n].second).determinant();

        // accept / reject
        if (expRatio * abs(proposedOverlap) / abs(overlap) >= uniform(generator)) {
          accepted++;
          fields[nsteps - n - 1] = proposedFields;
          overlap = proposedOverlap;
          ln[n + 2] = lni;
        }
        else {
          matPair prop;
          prop.first = MatrixXcd::Zero(norbs, norbs);
          prop.second = MatrixXcd::Zero(norbs, norbs);
          for (int i = 0; i < nfields; i++) {
            prop.first += fields[nsteps - n - 1](i) * hsOperators[i].first;
            prop.second += fields[nsteps - n - 1](i) * hsOperators[i].second;
          }
          prop.first = (sqrt(dt) * prop.first).exp();
          prop.second = (sqrt(dt) * prop.second).exp();
          ln[n + 2].first = ln[n + 1].first * expOneBodyOperator.first * prop.first * expOneBodyOperator.first;
          ln[n + 2].second = ln[n + 1].second * expOneBodyOperator.second * prop.second * expOneBodyOperator.second;
        }
      }
    }
    
    overlaps(sweep) = overlap;
  }

  complex<double> numMean = num.mean();
  complex<double> denomMean = denom.mean();
  complex<double> energyTotAll[commsize];
  complex<double> numTotAll[commsize];
  complex<double> denomTotAll[commsize];
  for (int i = 0; i < commsize; i++) {
    energyTotAll[i] = complex<double>(0., 0.);
    numTotAll[i] = complex<double>(0., 0.);
    denomTotAll[i] = complex<double>(0., 0.);
  }
  complex<double> energyProc = numMean / denomMean;
  MPI_Gather(&(energyProc), 1, MPI_DOUBLE_COMPLEX, &(energyTotAll), 1, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
  MPI_Gather(&(numMean), 1, MPI_DOUBLE_COMPLEX, &(numTotAll), 1, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
  MPI_Gather(&(denomMean), 1, MPI_DOUBLE_COMPLEX, &(denomTotAll), 1, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &energyProc, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &numMean, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &denomMean, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
  
  energyProc /= commsize;
  numMean /= commsize;
  denomMean /= commsize;
  double stddev = 0., stddev2 = 0.;
  for (int i = 0; i < commsize; i++) {
    stddev += pow(abs(energyTotAll[i] - energyProc), 2);
    stddev2 += pow(abs(energyTotAll[i] - energyProc), 4);
  }
  stddev /= (commsize - 1);
  stddev2 /= commsize;
  stddev2 = sqrt((stddev2 - (commsize - 3) * pow(stddev, 2) / (commsize - 1)) / commsize) / 2. / sqrt(stddev) / sqrt(sqrt(commsize));
  stddev = sqrt(stddev / commsize);

  double acceptanceRatio = accepted / (2. * nsweeps) / nsteps;

  if (commrank == 0) {
    cout << "\nAcceptance ratio:  " << acceptanceRatio << endl;
    cout << "Numerator:  " << numMean << ", Denominator:  " << denomMean << endl;
    cout << "Energy:  " << energyProc << " (" << stddev << ") " << " (" << stddev2 << ")\n";
  }
  
}

