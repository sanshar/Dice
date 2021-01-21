#ifndef SERIAL
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
#include "DQMCStatistics.h"
#include "DQMCSampling.h"

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


void calcEnergyDirectVartiational(double enuc, MatrixXd& h1, MatrixXd& h1Mod, vector<MatrixXd>& chol)
{
  size_t norbs = Determinant::norbs;
  size_t nfields = chol.size();
  size_t nsweeps = schd.stochasticIter;
  size_t nsteps = schd.nsteps;
  size_t orthoSteps = schd.orthoSteps;
  double dt = schd.dt;
  vector<int> eneSteps = schd.eneSteps;

  // prep and init
  // this is for mean field subtraction
  MatrixXd hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair rhf;
  rhf.first = hf.block(0, 0, norbs, Determinant::nalpha);
  rhf.second = hf.block(0, 0, norbs, Determinant::nbeta);
  
  vector<matPair> hsOperators;
  matPair oneBodyOperator;
  complex<double> mfConst = prepPropagatorHS(rhf, chol, hsOperators, oneBodyOperator);

  matPair expOneBodyOperator;
  expOneBodyOperator.first =  (-dt * (h1Mod - oneBodyOperator.first) / 2.).exp();
  expOneBodyOperator.second = (-dt * (h1Mod - oneBodyOperator.second) / 2.).exp();

  matPair refAd = rhf; refAd.first = rhf.first.adjoint(); refAd.second = rhf.second.adjoint(); 
  // rotate cholesky vectors
  pair<vector<MatrixXcd>, vector<MatrixXcd>> rotChol;
  for (int i = 0; i < chol.size(); i++) {
    MatrixXcd rotUp = refAd.first * chol[i];
    MatrixXcd rotDn = refAd.second * chol[i];
    rotChol.first.push_back(rotUp);
    rotChol.second.push_back(rotDn);
  }

  vector<VectorXd> fields;
  normal_distribution<double> normal(0., 1.);
  
  //matPair green;
  //calcGreensFunction(refT, ref, green);
  //complex<double> refEnergy = calcHamiltonianElement(green, enuc, h1, chol);
  complex<double> refEnergy = calcHamiltonianElement(refAd, rhf, enuc, h1, chol);
  complex<double> ene0;
  if (schd.ene0Guess == 1.e10) ene0 = refEnergy;
  else ene0 = schd.ene0Guess;
  if (commrank == 0) {
    cout << "Initial state energy:  " << refEnergy << endl;
    cout << "Ground state energy guess:  " << ene0 << endl << endl; 
  }
  
  //vector<int> eneSteps = { int(0.2*nsteps) - 1, int(0.4*nsteps) - 1, int(0.6*nsteps) - 1, int(0.8*nsteps) - 1, int(nsteps - 1) };
  int nEneSteps = eneSteps.size();
  //DQMCStatistics stats(nEneSteps);
  DQMCStatistics stats(2);
  auto iterTime = getTime();
  double propTime = 0., eneTime = 0.;
  //ArrayXd iTime(nEneSteps);
  //for (int i = 0; i < nEneSteps; i++) iTime(i) = dt * (eneSteps[i] + 1);
  ArrayXd iTime(2);
  for (int i = 0; i < 2; i++) iTime(i) = dt * (eneSteps[0] + 1);
  if (commrank == 0) cout << "Starting sampling sweeps\n";

  cout << normal(generator)<<"  first normal "<<endl;
  int nstepsHalf = schd.nsteps/2 + schd.nsteps%2;
  //cout << dt<<"  "<<nsteps<<"  "<<nstepsHalf<<endl;
  for (int sweep = 0; sweep < nsweeps; sweep++) {
    if (sweep != 0 && sweep % (schd.printFrequency) == 0) {
      if (commrank == 0) {
        cout << "Sweep steps: " << sweep << endl << "Total walltime: " << getTime() - iterTime << " s\n";
        cout << "\nPropagation time:  " << propTime << " s\n";
        cout << "Energy evaluation time:  " << eneTime << " s\n\n";
      }
      stats.gatherAndPrintStatistics(iTime);
    }
    double init = getTime();
      
    complex<double> orthoFacr = complex<double>(1., 0.), orthoFacl = 1., orthoFac2 = 1.;
    int eneStepCounter = 0;
    //ArrayXcd numSampleA(nEneSteps), denomSampleA(nEneSteps);
    ArrayXcd numSampleA(2), denomSampleA(2);
    numSampleA.setZero(); denomSampleA.setZero();

    matPair rn = rhf, ln = rhf, rn2 = rhf;
    vector<MatrixXcd> rightPotMat(nstepsHalf, MatrixXcd::Zero(norbs, norbs)), 
                      leftPotMat(nstepsHalf, MatrixXcd::Zero(norbs, norbs));
    vector<MatrixXcd> rightPotProp(nstepsHalf, MatrixXcd::Zero(norbs, norbs)), 
                      leftPotProp(nstepsHalf, MatrixXcd::Zero(norbs, norbs));

    propTime += getTime() - init;
    for (int n=0; n<nstepsHalf; n++) {
     for (int i=0; i<nfields; i++) {
        rightPotMat[n] += normal(generator) * hsOperators[i].first;
      }
      rightPotProp[n] = (sqrt(dt) * rightPotMat[n]).exp(); 
    }
    for (int n=0; n<nstepsHalf; n++) {
      for (int i=0; i<nfields; i++) {
        leftPotMat[n] += normal(generator) * hsOperators[i].first;
      }
      leftPotProp[n] = (sqrt(dt) * leftPotMat[n]).exp();
    }

    for (int n=0; n<nstepsHalf; n++) {
      rn.first = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * (expOneBodyOperator.first * (rightPotProp[n] * (expOneBodyOperator.first * rn.first)));
      ln.first = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * (expOneBodyOperator.first * (leftPotProp[nstepsHalf - n -1].adjoint() * (expOneBodyOperator.first * ln.first)));
      rn.second = rn.first;
      ln.second = ln.first;

      if (n != 0 && n % orthoSteps == 0) {
        orthogonalize(rn, orthoFacr);
        orthogonalize(ln, orthoFacl);
      }
    }

    for (int n=0; n<nstepsHalf; n++) {
      rn2.first = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * (expOneBodyOperator.first * (rightPotProp[n] * (expOneBodyOperator.first * rn2.first)));
      rn2.second = rn2.first;
      if (n != 0 && (n) % orthoSteps == 0) {
        orthogonalize(rn2, orthoFac2);
      }
    }

    for (int n=0; n<nstepsHalf; n++) {
      rn2.first = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * (expOneBodyOperator.first * (leftPotProp[n] * (expOneBodyOperator.first * rn2.first)));
      rn2.second = rn2.first;
      if (n != 0 && (n) % orthoSteps == 0) {
        orthogonalize(rn2, orthoFac2);
      }
    }

    propTime += getTime() - init;

    init = getTime();
    matPair lnAd; lnAd.first = ln.first.adjoint(); lnAd.second = ln.second.adjoint();
    complex<double> overlap = std::conj(orthoFacl)*orthoFacr * (lnAd.first * rn.first).determinant() * (lnAd.second * rn.second).determinant();
    complex<double> numSample;
    numSample = overlap * calcHamiltonianElement(lnAd, rn, enuc, h1, chol); 
    numSampleA[eneStepCounter] = numSample;
    denomSampleA[eneStepCounter] = overlap;

    //if (Determinant::nalpha == Determinant::nbeta) numSample = overlap * calcHamiltonianElement(ln.first, rn.first, enuc, h1, rotChol.first);
    //else numSample = overlap * calcHamiltonianElement(refT, rn, enuc, h1, rotChol);

    //cout << std::conj(orthoFacl)*orthoFacr <<"  "<<overlap<<"  "<<numSample<<endl;
    //matPair green;
    //calcGreensFunction(lnAd, rn, green);
    //numSample = overlap * calcHamiltonianElement(green, enuc, h1, chol);

    overlap = orthoFac2 * (refAd.first * rn2.first).determinant() * (refAd.second * rn2.second).determinant();
    numSample = overlap * calcHamiltonianElement(refAd, rn2, enuc, h1, chol); 
    //cout << orthoFac2 <<"  "<<overlap<<"  "<<numSample<<endl;
    numSampleA[1] = numSample;
    denomSampleA[1] = overlap;
    eneTime += getTime() - init;
    stats.addSamples(numSampleA, denomSampleA);
  }
  stats.gatherAndPrintStatistics(iTime);

}


// calculates energy of the imaginary time propagated wave function using direct sampling of exponentials
// w/o jastrow
void calcEnergyDirect(double enuc, MatrixXd& h1, MatrixXd& h1Mod, vector<MatrixXd>& chol)
{
  size_t norbs = Determinant::norbs;
  size_t nfields = chol.size();
  size_t nsweeps = schd.stochasticIter;
  size_t nsteps = schd.nsteps;
  size_t orthoSteps = schd.orthoSteps;
  double dt = schd.dt;
  vector<int> eneSteps = schd.eneSteps;

  // prep and init
  // this is for mean field subtraction
  MatrixXd hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair rhf;
  rhf.first = hf.block(0, 0, norbs, Determinant::nalpha);
  rhf.second = hf.block(0, 0, norbs, Determinant::nbeta);
  
  vector<matPair> hsOperators;
  matPair oneBodyOperator;
  complex<double> mfConst = prepPropagatorHS(rhf, chol, hsOperators, oneBodyOperator);
  
  // this is the initial state
  matPair ref;
  hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  ref.first = hf.block(0, 0, norbs, Determinant::nalpha);
  ref.second = hf.block(0, 0, norbs, Determinant::nbeta);
 
  // this is the left state
  matPair refT;
  if (schd.hf == "rhf") {
    refT.first = ref.first.adjoint();
    refT.second = ref.second.adjoint();
  }
  if (schd.hf == "rhfc") {
    MatrixXcd hfc = MatrixXcd::Zero(norbs, norbs);
    readMat(hfc, "rhfC.txt");
    refT.first = hfc.block(0, 0, norbs, Determinant::nalpha).adjoint();
    refT.second = hfc.block(0, 0, norbs, Determinant::nbeta).adjoint();
  }
  else if (schd.hf == "uhf") {
    hf = MatrixXd::Zero(norbs, 2*norbs);
    readMat(hf, "uhf.txt");
    refT.first = hf.block(0, 0, norbs, Determinant::nalpha).adjoint();
    refT.second = hf.block(0, norbs, norbs, Determinant::nbeta).adjoint();
  }
  else if (schd.hf == "uhfc") {
    MatrixXcd hfc = MatrixXcd::Zero(norbs, 2*norbs);
    readMat(hfc, "uhfC.txt");
    refT.first = hfc.block(0, 0, norbs, Determinant::nalpha).adjoint();
    refT.second = hfc.block(0, norbs, norbs, Determinant::nbeta).adjoint();
  }
  
  matPair expOneBodyOperator;
  expOneBodyOperator.first =  (-dt * (h1Mod - oneBodyOperator.first) / 2.).exp();
  expOneBodyOperator.second = (-dt * (h1Mod - oneBodyOperator.second) / 2.).exp();

  // rotate cholesky vectors
  pair<vector<MatrixXcd>, vector<MatrixXcd>> rotChol;
  for (int i = 0; i < chol.size(); i++) {
    MatrixXcd rotUp = refT.first * chol[i];
    MatrixXcd rotDn = refT.second * chol[i];
    rotChol.first.push_back(rotUp);
    rotChol.second.push_back(rotDn);
  }
  //vector<MatrixXd> richol(1, chol[0]); //sri
  
  // Gaussian sampling
  // field values arranged right to left
  vector<VectorXd> fields;
  normal_distribution<double> normal(0., 1.);
  
  //matPair green;
  //calcGreensFunction(refT, ref, green);
  //complex<double> refEnergy = calcHamiltonianElement(green, enuc, h1, chol);
  complex<double> refEnergy = calcHamiltonianElement(refT, ref, enuc, h1, rotChol);
  complex<double> ene0;
  if (schd.ene0Guess == 1.e10) ene0 = refEnergy;
  else ene0 = schd.ene0Guess;
  if (commrank == 0) {
    cout << "Initial state energy:  " << refEnergy << endl;
    cout << "Ground state energy guess:  " << ene0 << endl << endl; 
  }
  
  //vector<int> eneSteps = { int(0.2*nsteps) - 1, int(0.4*nsteps) - 1, int(0.6*nsteps) - 1, int(0.8*nsteps) - 1, int(nsteps - 1) };
  int nEneSteps = eneSteps.size();
  DQMCStatistics stats(nEneSteps);
  auto iterTime = getTime();
  double propTime = 0., eneTime = 0.;
  ArrayXd iTime(nEneSteps);
  for (int i = 0; i < nEneSteps; i++) iTime(i) = dt * (eneSteps[i] + 1);
  if (commrank == 0) cout << "Starting sampling sweeps\n";
  cout << normal(generator)<<"  first normal "<<endl;
  for (int sweep = 0; sweep < nsweeps; sweep++) {
    if (sweep != 0 && sweep % (schd.printFrequency) == 0) {
      if (commrank == 0) {
        cout << "Sweep steps: " << sweep << endl << "Total walltime: " << getTime() - iterTime << " s\n";
        cout << "\nPropagation time:  " << propTime << " s\n";
        cout << "Energy evaluation time:  " << eneTime << " s\n\n";
      }
      stats.gatherAndPrintStatistics(iTime);
    }
    matPair rn;
    rn = ref;
    VectorXd fields = VectorXd::Zero(nfields);
    complex<double> orthoFac = complex<double>(1., 0.);
    int eneStepCounter = 0;
    ArrayXcd numSampleA(nEneSteps), denomSampleA(nEneSteps);
    numSampleA.setZero(); denomSampleA.setZero();

    for (int n = 0; n < nsteps; n++) {
      // sampling
      double init = getTime();
      matPair prop;
      prop.first = MatrixXcd::Zero(norbs, norbs);
      prop.second = MatrixXcd::Zero(norbs, norbs);
      for (int i = 0; i < nfields; i++) {
        double field_n_i = normal(generator);
        prop.first += field_n_i * hsOperators[i].first;
        //prop.second += field_n_i * hsOperators[i].second;
      }
      prop.first = (sqrt(dt) * prop.first).exp();
      rn.first = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * (expOneBodyOperator.first * (prop.first * (expOneBodyOperator.first * rn.first)));
      //cout << n<<"  "<<(refT.first * rn.first).determinant()<<endl;
      if (Determinant::nalpha == Determinant::nbeta) rn.second = rn.first;
      else {
        //prop.second = (sqrt(dt) * prop.second).exp();
        rn.second = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nbeta)) * expOneBodyOperator.second * prop.first * expOneBodyOperator.second * rn.second;
      }

      propTime += getTime() - init;
      
      // orthogonalize for stability
      if (n!= 0 && n % orthoSteps == 0) {
        //cout << "ortho"<<endl;
        orthogonalize(rn, orthoFac);
      }

      // measure
      init = getTime();
      if (n == eneSteps[eneStepCounter]) {
        complex<double> overlap = orthoFac * (refT.first * rn.first).determinant() * (refT.second * rn.second).determinant();
        complex<double> numSample;
        if (Determinant::nalpha == Determinant::nbeta) numSample = overlap * calcHamiltonianElement(refT.first, rn.first, enuc, h1, rotChol.first);
        else numSample = overlap * calcHamiltonianElement(refT, rn, enuc, h1, rotChol);
        //numSample = overlap * (refEnergy + calcHamiltonianElement_sRI(refT, rn, refT, ref, enuc, h1, chol, richol)); 
        //cout << orthoFac<<" "<<overlap<<"  "<<numSample<<endl;
        numSampleA[eneStepCounter] = numSample;
        denomSampleA[eneStepCounter] = overlap;
        eneStepCounter++;
      }
      eneTime += getTime() - init;
    }
    stats.addSamples(numSampleA, denomSampleA);
  }

  if (commrank == 0) {
    cout << "\nPropagation time:  " << propTime << " s\n";
    cout << "Energy evaluation time:  " << eneTime << " s\n\n";
  }

  stats.gatherAndPrintStatistics(iTime);
  if (schd.printLevel > 10) stats.writeSamples();
}

// calculates energy of the imaginary time propagated wave function using direct sampling of exponentials
// w/o jastrow
void calcEnergyDirectGHF(double enuc, MatrixXd& h1, MatrixXd& h1Mod, vector<MatrixXd>& chol)
{
  size_t norbs = Determinant::norbs;
  size_t nfields = chol.size();
  size_t nsweeps = schd.stochasticIter;
  size_t nsteps = schd.nsteps;
  size_t orthoSteps = schd.orthoSteps;
  size_t nalpha = Determinant::nalpha;
  size_t nbeta = Determinant::nbeta;
  size_t nelec = nalpha+nbeta;
  double dt = schd.dt;
  vector<int> eneSteps = schd.eneSteps;

  // prep and init
  // this is for mean field subtraction
  MatrixXd hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair rhf;
  rhf.first = hf.block(0, 0, norbs, Determinant::nalpha);
  rhf.second = hf.block(0, 0, norbs, Determinant::nbeta);
  
  vector<matPair> hsOperators;
  matPair oneBodyOperator;
  complex<double> mfConst = prepPropagatorHS(rhf, chol, hsOperators, oneBodyOperator);
  
  // this is the initial state
  matPair ref;
  hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  ref.first = hf.block(0, 0, norbs, Determinant::nalpha);
  ref.second = hf.block(0, 0, norbs, Determinant::nbeta);
 
  // this is the left state
  MatrixXcd refAd, refT;
  {
    MatrixXcd hf = MatrixXcd::Zero(2*norbs, 2*norbs);
    readMat(hf, "ghf.txt");
    refAd = 1.*hf.block(0,0,2*norbs, nelec).adjoint();// + 0.01*MatrixXcd::Random(2*norbs, nelec);
    refT = refAd.conjugate();
  }

// rotate cholesky vectors
  pair<vector<MatrixXcd>, vector<MatrixXcd>> rotCholAd;
  pair<vector<MatrixXcd>, vector<MatrixXcd>> rotCholT;
  for (int i = 0; i < chol.size(); i++) {
    MatrixXcd rotUp = refT.block(0,0,nelec,norbs) * chol[i];
    MatrixXcd rotDn = refT.block(0,norbs,nelec,norbs)  * chol[i];
    rotCholT.first.push_back(rotUp);
    rotCholT.second.push_back(rotDn);

    rotUp = refAd.block(0,0,nelec,norbs) * chol[i];
    rotDn = refAd.block(0,norbs,nelec,norbs)  * chol[i];
    rotCholAd.first.push_back(rotUp);
    rotCholAd.second.push_back(rotDn);

  }
  
  matPair expOneBodyOperator;
  expOneBodyOperator.first =  (-dt * (h1Mod - oneBodyOperator.first) / 2.).exp();
  expOneBodyOperator.second = (-dt * (h1Mod - oneBodyOperator.second) / 2.).exp();

   vector<VectorXd> fields;
  normal_distribution<double> normal(0., 1.);
  
  //matPair green;
  //calcGreensFunction(refAd, ref, green);
  //complex<double> refEnergy = calcHamiltonianElement(green, enuc, h1, chol);
  complex<double> ovlp1, ovlp2;
  complex<double> refEnergy1 = calcHamiltonianElement(refAd, ref, enuc, h1, chol, ovlp1);
  complex<double> refEnergy2 = calcHamiltonianElement(refT, ref, enuc, h1, chol, ovlp2);
  complex<double> refEnergy = (refEnergy1*ovlp1 + refEnergy2*ovlp2)/(ovlp1+ovlp2);

  complex<double> ene0;
  if (schd.ene0Guess == 1.e10) ene0 = refEnergy;
  else ene0 = schd.ene0Guess;
  if (commrank == 0) {
    cout << "Initial state energy:  " << refEnergy << endl;
    cout << "Ground state energy guess:  " << ene0 << endl << endl; 
  }
  
  //vector<int> eneSteps = { int(0.2*nsteps) - 1, int(0.4*nsteps) - 1, int(0.6*nsteps) - 1, int(0.8*nsteps) - 1, int(nsteps - 1) };
  int nEneSteps = eneSteps.size();
  DQMCStatistics stats(nEneSteps);
  auto iterTime = getTime();
  double propTime = 0., eneTime = 0., qrTime = 0.;
  if (commrank == 0) cout << "Starting sampling sweeps\n";
  ArrayXd iTime(nEneSteps);
  for (int i = 0; i < nEneSteps; i++) iTime(i) = dt * (eneSteps[i] + 1);

  for (int sweep = 0; sweep < nsweeps; sweep++) {
    if (sweep != 0 && sweep % (schd.printFrequency) == 0) {
      if (commrank == 0) {
        cout << "Sweep steps: " << sweep << endl << "Total walltime: " << getTime() - iterTime << " s\n";
        cout << "\nPropagation time:  " << propTime << " s\n";
        cout << "Energy evaluation time:  " << eneTime << " s\n\n";
      }
      stats.gatherAndPrintStatistics(iTime);
    }

    matPair rn;
    rn = ref;
    VectorXd fields = VectorXd::Zero(nfields);
    complex<double> orthoFac = complex<double>(1., 0.);
    int eneStepCounter = 0;
    ArrayXcd numSampleA(nEneSteps), denomSampleA(nEneSteps);
    numSampleA.setZero(); denomSampleA.setZero();
    for (int n = 0; n < nsteps; n++) {
      // sampling
      double init = getTime();
      matPair prop;
      prop.first = MatrixXcd::Zero(norbs, norbs);
      prop.second = MatrixXcd::Zero(norbs, norbs);
      for (int i = 0; i < nfields; i++) {
        double field_n_i = normal(generator);
        prop.first += field_n_i * hsOperators[i].first;
        //prop.second += field_n_i * hsOperators[i].second;
      }
      prop.first = (sqrt(dt) * prop.first).exp();
      rn.first = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * (expOneBodyOperator.first * (prop.first * (expOneBodyOperator.first * rn.first)));
      if (Determinant::nalpha == Determinant::nbeta) rn.second = rn.first;
      else {
        //prop.second = (sqrt(dt) * prop.second).exp();
        rn.second = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nbeta)) * expOneBodyOperator.second * prop.first * expOneBodyOperator.second * rn.second;
      }

      propTime += getTime() - init;
      
      // orthogonalize for stability
      if (n % orthoSteps == 0) {
        orthogonalize(rn, orthoFac);
      }

      // measure
      init = getTime();
      if (n == eneSteps[eneStepCounter]) {
        complex<double> num, den;
        complex<double> overlapAd, overlapT;
        //complex<double> numSample = calcHamiltonianElement(refAd, rn, enuc, h1, chol, overlapAd);
        complex<double> numSample = calcHamiltonianElement(refAd, rn, enuc, h1, rotCholAd, overlapAd);
        overlapAd *= orthoFac;
        numSample *= overlapAd;
        num = numSample; den = overlapAd;

        //numSample = calcHamiltonianElement(refT, rn, enuc, h1, chol, overlapT);
        numSample = calcHamiltonianElement(refT, rn, enuc, h1, rotCholT, overlapT);
        overlapT *= orthoFac;
        numSample *= overlapT;
        num += numSample; den += overlapT;

        numSampleA[eneStepCounter] = num;
        denomSampleA[eneStepCounter] = den;
        eneStepCounter++;
      }
      eneTime += getTime() - init;
    }
    stats.addSamples(numSampleA, denomSampleA);
  }

  if (commrank == 0) {
    cout << "\nPropagation time:  " << propTime << " s\n";
    cout << "Energy evaluation time:  " << eneTime << " s\n\n";
  }

  stats.gatherAndPrintStatistics(iTime);
  if (schd.printLevel > 10) stats.writeSamples();
}



// calculates energy of the imaginary time propagated wave function using direct sampling of exponentials
// w/o jastrow
void findDtCorrelatedSamplingGHF(double enuc, MatrixXd& h1, MatrixXd& h1Mod, vector<MatrixXd>& chol)
{
  size_t norbs = Determinant::norbs;
  size_t nfields = chol.size();
  size_t nsweeps = schd.stochasticIter;
  size_t nsteps = schd.nsteps;
  size_t orthoSteps = schd.orthoSteps;
  size_t nalpha = Determinant::nalpha;
  size_t nbeta = Determinant::nbeta;
  size_t nelec = nalpha+nbeta;
  //double dt = schd.dt;
  vector<int> eneSteps = schd.eneSteps;

  // prep and init
  // this is for mean field subtraction
  MatrixXd hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair rhf;
  rhf.first = hf.block(0, 0, norbs, Determinant::nalpha);
  rhf.second = hf.block(0, 0, norbs, Determinant::nbeta);
  
  vector<matPair> hsOperators;
  matPair oneBodyOperator;
  complex<double> mfConst = prepPropagatorHS(rhf, chol, hsOperators, oneBodyOperator);
  
  // this is the initial state
  matPair ref;
  hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  ref.first = hf.block(0, 0, norbs, Determinant::nalpha);
  ref.second = hf.block(0, 0, norbs, Determinant::nbeta);
 
  // this is the left state
  MatrixXcd refAd, refT;
  {
    MatrixXcd hf = MatrixXcd::Zero(2*norbs, 2*norbs);
    readMat(hf, "ghf.txt");
    refAd = 1.*hf.block(0,0,2*norbs, nelec).adjoint();// + 0.01*MatrixXcd::Random(2*norbs, nelec);
    refT = refAd.conjugate();
  }
  int nCorrelatedT = 3;
  vector<double> dt(3, schd.dt);

  //largest to smallest with factors of 2
  for (int j=0; j<nCorrelatedT; j++) {
    dt[j] = schd.dt/pow(2.,j);
    cout <<j<<"  "<<dt[j]<<"  "<<schd.dt<<endl;
  }

  vector<matPair> expOneBodyOperator(nCorrelatedT);
  for (int j=0; j<nCorrelatedT; j++) {
    expOneBodyOperator[j].first =  (-dt[j] * (h1Mod - oneBodyOperator.first) / 2.).exp();
    expOneBodyOperator[j].second = (-dt[j] * (h1Mod - oneBodyOperator.second) / 2.).exp();
  }

  vector<VectorXd> fields;
  normal_distribution<double> normal(0., 1.);
  
  //matPair green;
  //calcGreensFunction(refAd, ref, green);
  //complex<double> refEnergy = calcHamiltonianElement(green, enuc, h1, chol);
  complex<double> ovlp1, ovlp2;
  complex<double> refEnergy1 = calcHamiltonianElement(refAd, ref, enuc, h1, chol, ovlp1);
  complex<double> refEnergy2 = calcHamiltonianElement(refT, ref, enuc, h1, chol, ovlp2);
  complex<double> refEnergy = (refEnergy1*ovlp1 + refEnergy2*ovlp2)/(ovlp1+ovlp2);

  complex<double> ene0;
  if (schd.ene0Guess == 1.e10) ene0 = refEnergy;
  else ene0 = schd.ene0Guess;
  if (commrank == 0) {
    cout << "Initial state energy:  " << refEnergy << endl;
    cout << "Ground state energy guess:  " << ene0 << endl << endl; 
  }
  
  //vector<int> eneSteps = { int(0.2*nsteps) - 1, int(0.4*nsteps) - 1, int(0.6*nsteps) - 1, int(0.8*nsteps) - 1, int(nsteps - 1) };

  int nEneSteps = eneSteps.size();
  vector<DQMCStatistics> statsVec(nCorrelatedT, DQMCStatistics(nEneSteps)); //a separate statistics class for each dT
  auto iterTime = getTime();
  double propTime = 0., eneTime = 0., qrTime = 0.;
  if (commrank == 0) cout << "Starting sampling sweeps\n";
  ArrayXd iTime(nEneSteps);
  for (int i = 0; i < nEneSteps; i++) iTime(i) = dt[0] * (eneSteps[i] + 1);

  for (int sweep = 0; sweep < nsweeps; sweep++) {
    if (sweep != 0 && sweep % (schd.printFrequency) == 0) {
      if (commrank == 0) {
        cout <<"sweep steps: "<< sweep <<endl<<"Total walltime: " << getTime() - iterTime << " s\n";
        cout << "\nPropagation time:  " << propTime << " s\n";
        cout << "Energy evaluation time:  " << eneTime << " s\n\n";
      }
      for (int j=0; j<nCorrelatedT; j++) 
        statsVec[j].gatherAndPrintStatistics(iTime);
    }

    vector<matPair> rn(nCorrelatedT, ref);
    VectorXd fields = VectorXd::Zero(nfields);
    vector<complex<double>> orthoFac(nCorrelatedT, complex<double>(1., 0.));
    int eneStepCounter = 0;

    vector<ArrayXcd> numSampleA(nCorrelatedT, ArrayXcd(nEneSteps)), denomSampleA(nCorrelatedT, ArrayXcd(nEneSteps));
    for (int j=0; j<nCorrelatedT; j++) {
      numSampleA[j].setZero(); denomSampleA[j].setZero();
    }

    for (int n = 0; n < nsteps; n++) {
      // sampling
      double init = getTime();
      vector<matPair> prop(nCorrelatedT);

      MatrixXcd hsProp = MatrixXcd::Zero(norbs, norbs);
      for (int i = 0; i < nfields; i++) {
        double field_n_i = normal(generator);
        hsProp += field_n_i * hsOperators[i].first;
        //prop.second += field_n_i * hsOperators[i].second;
      }

      //for each dt apply the Hamiltonian as many times as needed such that total time dt[0] is propogated
      for (int j=0; j<nCorrelatedT; j++) {
        int nterms = pow(2,j);
        prop[j].first = exp((ene0 - enuc - mfConst) * dt[j] / (2. * Determinant::nalpha)) * (expOneBodyOperator[j].first * ( (sqrt(dt[j]/nterms) * hsProp).exp() * expOneBodyOperator[j].first));

        for (int napply = 0; napply < nterms; napply++) {
          rn[j].first =  prop[j].first * rn[j].first;
        }

        if (Determinant::nalpha == Determinant::nbeta) rn[j].second = rn[j].first;
        else {
          for (int napply = 0; napply < pow(2, j); napply++) {
            rn[j].second = prop[j].first * rn[j].second;
          }
        }
        //cout << rn[j].first<<endl<<endl<<endl;
      }
      //cout << n<<"  "<<(prop[0].first - prop[1].first*prop[1].first).norm()<<endl;
      //exit(0);

      propTime += getTime() - init;
      
      // orthogonalize for stability
      if (n % orthoSteps == 0) {
        for (int j=0; j<nCorrelatedT; j++)
          orthogonalize(rn[j], orthoFac[j]);
      }

      // measure
      init = getTime();
      if (n == eneSteps[eneStepCounter]) {
        for (int j=0; j<nCorrelatedT; j++) {
          complex<double> num, den;
          complex<double> overlapAd, overlapT;
          complex<double> numSample = calcHamiltonianElement(refAd, rn[j], enuc, h1, chol, overlapAd);
          overlapAd *= orthoFac[j];
          numSample *= overlapAd;
          num = numSample; den = overlapAd;

          numSample = calcHamiltonianElement(refT, rn[j], enuc, h1, chol, overlapT);
          overlapT *= orthoFac[j];
          numSample *= overlapT;
          num += numSample; den += overlapT;

          //cout << j<<" e "<<num<<"  "<<den<<"  "<<num/den<<"  "<<orthoFac[j]<<"  "<<nCorrelatedT<<endl;
          numSampleA[j][eneStepCounter] = num;
          denomSampleA[j][eneStepCounter] = den;
        }
        eneStepCounter++;
        //exit(0);
      }
      
      eneTime += getTime() - init;
    }
    for (int j=0; j<nCorrelatedT; j++) {
      statsVec[j].addSamples(numSampleA[j], denomSampleA[j]);
      //cout << j<<"  "<<numSampleA[j][0]<<"  "<<denomSampleA[j][0]<<endl;
    }
  }

  if (commrank == 0) {
    cout << "\nPropagation time:  " << propTime << " s\n";
    cout << "Energy evaluation time:  " << eneTime << " s\n\n";
  }

  for (int j=0; j<nCorrelatedT; j++)
    statsVec[j].gatherAndPrintStatistics(iTime);
  //if (schd.printLevel > 10) stats.writeSamples();
}




// calculates mixed energy estimator of the imaginary time propagated wave function
// w jastrow
void calcEnergyJastrowDirect(double enuc, MatrixXd& h1, MatrixXd& h1Mod, vector<MatrixXd>& chol)
{
  size_t norbs = Determinant::norbs;
  size_t nfields = chol.size();
  size_t nsweeps = schd.stochasticIter;
  size_t nsteps = schd.nsteps;
  size_t orthoSteps = schd.orthoSteps;
  double dt = schd.dt;

  // prep and init
  // this is for mean field subtraction
  MatrixXd hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair rhf;
  rhf.first = hf.block(0, 0, norbs, Determinant::nalpha);
  rhf.second = hf.block(0, 0, norbs, Determinant::nbeta);
  
  // jastrow operators
  matPair refAct;
  size_t numActOrbs;
  if (schd.nciAct < 0) numActOrbs = norbs;
  else numActOrbs = schd.nciAct;
  refAct.first = rhf.first.block(0, 0, numActOrbs, Determinant::nalpha);
  refAct.second = rhf.second.block(0, 0, numActOrbs, Determinant::nbeta);
  vector<vecPair> jhsOperators;
  vecPair joneBodyOperator;
  complex<double> jmfConst = prepJastrowHS(refAct, jhsOperators, joneBodyOperator);
  size_t jnfields = jhsOperators.size();
  
  // prop
  vector<matPair> hsOperators;
  matPair oneBodyOperator;
  complex<double> mfConst = prepPropagatorHS(rhf, chol, hsOperators, oneBodyOperator);
  
  // this is the initial state
  matPair ref;
  if (schd.hf == "rhf") {
    hf = MatrixXd::Zero(norbs, norbs);
    readMat(hf, "rhf.txt");
    ref.first = hf.block(0, 0, norbs, Determinant::nalpha);
    ref.second = hf.block(0, 0, norbs, Determinant::nbeta);
  }
  else if (schd.hf == "uhf") {
    hf = MatrixXd::Zero(norbs, 2*norbs);
    readMat(hf, "uhf.txt");
    ref.first = hf.block(0, 0, norbs, Determinant::nalpha);
    ref.second = hf.block(0, norbs, norbs, Determinant::nbeta);
  }

  matPair expOneBodyOperator;
  expOneBodyOperator.first =  (-dt * (h1Mod - oneBodyOperator.first) / 2.).exp();
  expOneBodyOperator.second = (-dt * (h1Mod - oneBodyOperator.second) / 2.).exp();
  

  vecPair jexpOneBodyOperator;
  jexpOneBodyOperator.first = joneBodyOperator.first.array().exp();
  jexpOneBodyOperator.second = joneBodyOperator.second.array().exp();
  matPair jref;
  jref.first = jexpOneBodyOperator.first.asDiagonal() * refAct.first;
  jref.second = jexpOneBodyOperator.second.asDiagonal() * refAct.second;

  // Gaussian sampling
  // field values arranged right to left
  vector<VectorXd> fields;
  normal_distribution<double> normal(0., 1.);
  matPair jrefT;
  jrefT.first = jref.first.adjoint();
  jrefT.second = jref.second.adjoint();
  matPair refT;
  refT.first = ref.first.adjoint();
  refT.second = ref.second.adjoint();  

  matPair green;
  calcGreensFunction(refT, ref, green);
  complex<double> refEnergy = calcHamiltonianElement(green, enuc, h1, chol);

  complex<double> jrefEnergy = calcHamiltonianElement(jrefT, ref,enuc, h1, chol);
  vector<MatrixXd> richol(1, chol[0]); //sri

  complex<double> ene0;
  if (schd.ene0Guess == 1.e10) ene0 = refEnergy;
  else ene0 = schd.ene0Guess;
  if (commrank == 0) {
    cout << "Initial state energy:  " << refEnergy << endl;
    cout << "Ground state energy guess:  " << ene0 << endl << endl; 
  }
  
  vector<int> eneSteps = { int(nsteps - 1) };
  int nEneSteps = eneSteps.size();
  DQMCStatistics stats(nEneSteps);
  auto iterTime = getTime();
  double propTime = 0., eneTime = 0.;
  if (commrank == 0) cout << "Starting sampling sweeps\n";
  for (int sweep = 0; sweep < nsweeps; sweep++) {
    if (sweep != 0 && sweep % (nsweeps/5) == 0 && commrank == 0) cout << sweep << "  " << getTime() - iterTime << " s\n";
    matPair rn = ref;
    complex<double> orthoFac = complex<double> (1., 0.);
    int eneStepCounter = 0;

    // sampling
    ArrayXcd numSampleA(nEneSteps), denomSampleA(nEneSteps);
    numSampleA.setZero(); denomSampleA.setZero();
    VectorXd fields = VectorXd::Zero(nfields);
    for (int n = 0; n < nsteps; n++) {
      // prop
      double init = getTime();
      matPair prop;
      prop.first = MatrixXcd::Zero(norbs, norbs);
      prop.second = MatrixXcd::Zero(norbs, norbs);
      for (int i = 0; i < nfields; i++) {
        double field_n_i = normal(generator);
        prop.first += field_n_i * hsOperators[i].first;
        prop.second += field_n_i * hsOperators[i].second;
      }
      prop.first = (sqrt(dt) * prop.first).exp();
      prop.second = (sqrt(dt) * prop.second).exp();
      rn.first = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * expOneBodyOperator.first * prop.first * expOneBodyOperator.first * rn.first;
      rn.second = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nbeta)) * expOneBodyOperator.second * prop.second * expOneBodyOperator.second * rn.second;
      propTime += getTime() - init;

      // orthogonalize for stability
      orthogonalize(rn , orthoFac);

      // measure
      init = getTime();
      if (n == eneSteps[eneStepCounter]) {
        // sample left jastrow
        size_t numJastrowSamples = schd.numJastrowSamples;
        complex<double> jOverlap(0., 0.), jLocalEnergy(0., 0.);
        for (int i = 0; i < numJastrowSamples; i++) {
          VectorXd jfields = VectorXd::Zero(jnfields);
          vecPair jpropLeft;
          jpropLeft.first = VectorXcd::Zero(numActOrbs);
          jpropLeft.second = VectorXcd::Zero(numActOrbs);
          for (int i = 0; i < jnfields; i++) {
            jfields(i) = normal(generator);
            jpropLeft.first += jfields(i) * jhsOperators[i].first;
            jpropLeft.second += jfields(i) * jhsOperators[i].second;
          }
          matPair ln;
          ln.first = exp(jmfConst / (2. * Determinant::nalpha)) * jrefT.first * jpropLeft.first.array().exp().matrix().asDiagonal();
          ln.second = exp(jmfConst / (2. * Determinant::nbeta)) * jrefT.second * jpropLeft.second.array().exp().matrix().asDiagonal();

          complex<double> overlapSample = orthoFac * (ln.first * rn.first.block(0, 0, numActOrbs, Determinant::nalpha)).determinant() 
                                        * (ln.second * rn.second.block(0, 0, numActOrbs, Determinant::nbeta)).determinant();
          jOverlap += overlapSample;
          jLocalEnergy += overlapSample * calcHamiltonianElement(ln, rn, enuc, h1, chol); 
          //jLocalEnergy += overlapSample * (calcHamiltonianElement_sRI(ln, rn, jrefT, ref, enuc, h1, chol, richol) + jrefEnergy); 
        }
        jOverlap /= numJastrowSamples;
        jLocalEnergy /= numJastrowSamples;
        numSampleA[eneStepCounter] = jLocalEnergy;
        denomSampleA[eneStepCounter] = jOverlap;
        eneStepCounter++;
      }
      eneTime += getTime() - init;
    }
    stats.addSamples(numSampleA, denomSampleA);
  }

  if (commrank == 0) {
    cout << "\nPropagation time:  " << propTime << " s\n";
    cout << "Energy evaluation time:  " << eneTime << " s\n\n";
  }

  ArrayXd iTime(nEneSteps);
  for (int i = 0; i < nEneSteps; i++) iTime(i) = dt * (eneSteps[i] + 1);
  stats.gatherAndPrintStatistics(iTime);
  if (schd.printLevel > 10) stats.writeSamples();
}


// calculates mixed energy estimator of the imaginary time propagated wave function
// w jastrow
void calcEnergyJastrowDirectVariational(double enuc, MatrixXd& h1, MatrixXd& h1Mod, vector<MatrixXd>& chol)
{
  size_t norbs = Determinant::norbs;
  size_t nfields = chol.size();
  size_t nsweeps = schd.stochasticIter;
  size_t nsteps = schd.nsteps;
  size_t orthoSteps = schd.orthoSteps;
  double dt = schd.dt;

  // prep and init
  // this is for mean field subtraction
  MatrixXd hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair rhf;
  rhf.first = hf.block(0, 0, norbs, Determinant::nalpha);
  rhf.second = hf.block(0, 0, norbs, Determinant::nbeta);
  
  // jastrow operators
  matPair refAct;
  size_t numActOrbs;
  if (schd.nciAct < 0) numActOrbs = norbs;
  else numActOrbs = schd.nciAct;
  refAct.first = rhf.first.block(0, 0, numActOrbs, Determinant::nalpha);
  refAct.second = rhf.second.block(0, 0, numActOrbs, Determinant::nbeta);
  vector<vecPair> jhsOperators;
  vecPair joneBodyOperator;
  complex<double> jmfConst = prepJastrowHS(refAct, jhsOperators, joneBodyOperator);
  size_t jnfields = jhsOperators.size();
  
  // prop
  vector<matPair> hsOperators;
  matPair oneBodyOperator;
  complex<double> mfConst = prepPropagatorHS(rhf, chol, hsOperators, oneBodyOperator);
  
  // this is the initial state
  matPair ref;
  if (schd.hf == "rhf") {
    hf = MatrixXd::Zero(norbs, norbs);
    readMat(hf, "rhf.txt");
    ref.first = hf.block(0, 0, norbs, Determinant::nalpha);
    ref.second = hf.block(0, 0, norbs, Determinant::nbeta);
  }
  else if (schd.hf == "uhf") {
    hf = MatrixXd::Zero(norbs, 2*norbs);
    readMat(hf, "uhf.txt");
    ref.first = hf.block(0, 0, norbs, Determinant::nalpha);
    ref.second = hf.block(0, norbs, norbs, Determinant::nbeta);
  }
  matPair refT;
  refT.first = ref.first.adjoint();
  refT.second = ref.second.adjoint();

  matPair expOneBodyOperator;
  expOneBodyOperator.first =  (-dt * (h1Mod - oneBodyOperator.first) / 2.).exp();
  expOneBodyOperator.second = (-dt * (h1Mod - oneBodyOperator.second) / 2.).exp();
  

  // rotate cholesky vectors
  pair<vector<MatrixXcd>, vector<MatrixXcd>> rotChol;
  for (int i = 0; i < chol.size(); i++) {
    MatrixXcd rotUp = refT.first * chol[i];
    MatrixXcd rotDn = refT.second * chol[i];
    rotChol.first.push_back(rotUp);
    rotChol.second.push_back(rotDn);
  }


  vecPair jexpOneBodyOperator;
  jexpOneBodyOperator.first = joneBodyOperator.first.array().exp();
  jexpOneBodyOperator.second = joneBodyOperator.second.array().exp();
  //cout << joneBodyOperator.first<<endl<<endl;
  //cout << joneBodyOperator.first.array()<<endl<<endl;
  //cout << joneBodyOperator.first.array().exp()<<endl<<endl;
  //exit(0);
  matPair jref;
  jref.first = jexpOneBodyOperator.first.asDiagonal() * refAct.first;
  jref.second = jexpOneBodyOperator.second.asDiagonal() * refAct.second;
  

  // Gaussian sampling
  // field values arranged right to left
  vector<VectorXd> fields;
  normal_distribution<double> normal(0., 1.);
  matPair jrefT;
  jrefT.first = jref.first.adjoint();
  jrefT.second = jref.second.adjoint();
  matPair green;
  calcGreensFunction(refT, ref, green);
  complex<double> refEnergy = calcHamiltonianElement(green, enuc, h1, chol);
  complex<double> ene0;
  if (schd.ene0Guess == 1.e10) ene0 = refEnergy;
  else ene0 = schd.ene0Guess;
  if (commrank == 0) {
    cout << "Initial state energy:  " << refEnergy << endl;
    cout << "Ground state energy guess:  " << ene0 << endl << endl; 
  }
  
  vector<int> eneSteps = { int(nsteps - 1) };
  int numEneSteps = eneSteps.size();
  vector<complex<double>> numMeanVar(numEneSteps, complex<double>(0., 0.)), denomMeanVar(numEneSteps, complex<double>(0., 0.)), denomAbsMeanVar(numEneSteps, complex<double>(0., 0.));
  vector<complex<double>> numMeanProj(numEneSteps, complex<double>(0., 0.)), denomMeanProj(numEneSteps, complex<double>(0., 0.)), denomAbsMeanProj(numEneSteps, complex<double>(0., 0.));
  auto iterTime = getTime();
  double qrTime = 0., propTime = 0., eneTime = 0.;
  if (commrank == 0) cout << "Starting sampling sweeps\n";

  MatrixXcd identity = MatrixXcd::Identity(norbs, norbs);
  for (int sweep = 0; sweep < nsweeps; sweep++) {
    if (sweep != 0 && sweep % (nsweeps/5) == 0 && commrank == 0) cout << sweep << "  " << getTime() - iterTime << " s\n";
    matPair rn;
    int eneStepCounter = 0;

    rn = ref;

    // sampling
    VectorXd fields = VectorXd::Zero(nfields);
    vector<MatrixXcd> expHam(nsteps/orthoSteps +1, identity);

    matPair prop;
    prop.first = MatrixXcd::Zero(norbs, norbs);
    prop.second = MatrixXcd::Zero(norbs, norbs);

    int hindex = 0;
    double init = getTime();
    //sample ham for time slice n
    for (int n = 0; n < nsteps; n++) {

      
      prop.first.setZero();  
      for (int i = 0; i < nfields; i++) {
        double field_n_i = normal(generator);
        prop.first += field_n_i * hsOperators[i].first;
      }

      prop.first = (sqrt(dt) * prop.first).exp();
      expHam[hindex] = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * expOneBodyOperator.first * prop.first * expOneBodyOperator.first * expHam[hindex];

      if (n % orthoSteps == 0) hindex++;
      
    }
    propTime += getTime() - init;
    
  
    size_t numJastrowSamples = schd.numJastrowSamples;
    complex<double> jOverlapVar(0., 0.), jLocalEnergyVar(0., 0.);
    complex<double> jOverlapProj(0., 0.), jLocalEnergyProj(0., 0.);

    //measure energy numJastrowSamples times
    for (int i = 0; i < numJastrowSamples; i++) {

      // measure energy
      init = getTime();
      matPair ln, rn;

      //sample left jastrow
      {
        VectorXd jfields = VectorXd::Zero(jnfields);
        vecPair jpropLeft;
        jpropLeft.first = VectorXcd::Zero(numActOrbs);
        jpropLeft.second = VectorXcd::Zero(numActOrbs);
        for (int i = 0; i < jnfields; i++) {
          jfields(i) = normal(generator);
          jpropLeft.first += jfields(i) * jhsOperators[i].first;
          jpropLeft.second += jfields(i) * jhsOperators[i].second;
        }
        ln.first = exp(jmfConst / (2. * Determinant::nalpha)) * jrefT.first * jpropLeft.first.array().exp().matrix().asDiagonal();
        ln.second = exp(jmfConst / (2. * Determinant::nbeta)) * jrefT.second * jpropLeft.second.array().exp().matrix().asDiagonal();
      }
      //cout << "left prep"<<endl;
      //sample right jastrow
      {
        VectorXd jfields = VectorXd::Zero(jnfields);
        vecPair jpropRight;
        jpropRight.first = VectorXcd::Zero(numActOrbs);
        jpropRight.second = VectorXcd::Zero(numActOrbs);
        for (int i = 0; i < jnfields; i++) {
          jfields(i) = normal(generator);
          jpropRight.first  += jfields(i) * jhsOperators[i].first;
          jpropRight.second += jfields(i) * jhsOperators[i].second;
        }
        rn.first = exp(jmfConst / (2. * Determinant::nalpha)) *  jpropRight.first.array().exp().matrix().asDiagonal() * jref.first;
        rn.second = exp(jmfConst / (2. * Determinant::nbeta)) *  jpropRight.second.array().exp().matrix().asDiagonal() * jref.second;
       }    
      //cout << "right prep"<<endl;

      //apply ham to the ln
      matPair rnHam, lnHamAd, rnRefHam, lnRefHamAd;
      rnHam = rn; lnHamAd.first = 1.*ln.first.adjoint(); lnHamAd.second = 1.*ln.second.adjoint();
      rnRefHam = ref; lnRefHamAd = ref;
      
      complex<double> orthoFacR = complex<double> (1., 0.);
      complex<double> orthoFacRRef = complex<double> (1., 0.);
      for (int i=0; i<hindex; i++) {
        rnHam.first = expHam[i]*rnHam.first; 
        rnHam.second = expHam[i]*rnHam.second;
        orthogonalize(rnHam, orthoFacR);

        rnRefHam.first = expHam[i]*rnRefHam.first; 
        rnRefHam.second = expHam[i]*rnRefHam.second;
        orthogonalize(rnRefHam, orthoFacRRef);

      } 

      complex<double> orthoFacL = complex<double> (1., 0.);
      complex<double> orthoFacLRef = complex<double> (1., 0.);
      //cout << "before for"<<endl;
      for (int i=0; i<hindex; i++) {
        lnHamAd.first = expHam[i]*lnHamAd.first; 
        lnHamAd.second = expHam[i]*lnHamAd.second;
        orthogonalize(lnHamAd, orthoFacL);

        lnRefHamAd.first = expHam[i]*lnRefHamAd.first; 
        lnRefHamAd.second = expHam[i]*lnRefHamAd.second;
        orthogonalize(lnRefHamAd, orthoFacLRef);
      } 
      //cout<< "after for"<<endl;
      matPair lnHam, lnRefHam;
      lnHam.first = 1.*lnHamAd.first.adjoint(); lnHam.second = 1.*lnHamAd.second.adjoint();
      lnRefHam.first = 1.*lnRefHamAd.first.adjoint(); lnRefHam.second = 1.*lnRefHamAd.second.adjoint();

      propTime += getTime() - init;

      init = getTime();

      //Projected energy
      complex<double> overlapSample =  orthoFacRRef * (ln.first * rnRefHam.first).determinant() 
                                    * (ln.second * rnRefHam.second).determinant();
      jOverlapProj += overlapSample;
      //calcGreensFunction(ln, rnRefHam, green);
      jLocalEnergyProj += overlapSample * calcHamiltonianElement(ln, rnRefHam, enuc, h1, chol);

      overlapSample =  orthoFacLRef * (lnRefHam.first * rn.first).determinant() 
                                    * (lnRefHam.second * rn.second).determinant();
      jOverlapProj += overlapSample;
      //calcGreensFunction(lnRefHam, rn, green);
      jLocalEnergyProj += overlapSample * calcHamiltonianElement(lnRefHam, rn, enuc, h1, chol);

      //variational energy
      complex<double> overlapSampleR = orthoFacR * (ln.first * rnHam.first).determinant() 
                               * (ln.second * rnHam.second).determinant();
      complex<double>overlapSampleL =  orthoFacL * (lnHam.first * rn.first).determinant() 
                               * (lnHam.second * rn.second).determinant();
      jOverlapVar += (overlapSampleR + overlapSampleL);
      
      //calcGreensFunction(ln, rnHam, green);
      jLocalEnergyVar += overlapSampleR * calcHamiltonianElement(ln, rnHam, enuc, h1, chol);
      //calcGreensFunction(lnHam, rn, green);
      jLocalEnergyVar += overlapSampleL * calcHamiltonianElement(lnHam, rn, enuc, h1, chol);
      eneTime += getTime() - init;
    }

    jOverlapVar /= numJastrowSamples;
    jLocalEnergyVar /= numJastrowSamples;
    jOverlapProj /= numJastrowSamples;
    jLocalEnergyProj /= numJastrowSamples;

    denomMeanVar[eneStepCounter] += (jOverlapVar - denomMeanVar[eneStepCounter]) / (1.*(sweep + 1));
    denomAbsMeanVar[eneStepCounter] += (abs(jOverlapVar) - denomAbsMeanVar[eneStepCounter]) / (1.*(sweep + 1));
    numMeanVar[eneStepCounter] += (jLocalEnergyVar - numMeanVar[eneStepCounter]) / (1.*(sweep + 1));

    denomMeanProj[eneStepCounter] += (jOverlapProj - denomMeanProj[eneStepCounter]) / (1.*(sweep + 1));
    denomAbsMeanProj[eneStepCounter] += (abs(jOverlapProj) - denomAbsMeanProj[eneStepCounter]) / (1.*(sweep + 1));
    numMeanProj[eneStepCounter] += (jLocalEnergyProj - numMeanProj[eneStepCounter]) / (1.*(sweep + 1));
  }

  if (commrank == 0) {
    cout << "\nPropagation time:  " << propTime << " s\n";
    cout << "Energy evaluation time:  " << eneTime << " s\n\n";
    cout << "          iTime                 Energy                     Energy error         Average phase\n";
  }

  for (int n = 0; n < numEneSteps; n++) {
    complex<double> energyAll[commsize];
    for (int i = 0; i < commsize; i++) energyAll[i] = complex<double>(0., 0.);
    
    complex<double> energyProc = numMeanVar[n] / denomMeanVar[n];
    complex<double> numProc = numMeanVar[n];
    complex<double> denomProc = denomMeanVar[n];
    complex<double> denomAbsProc = denomAbsMeanVar[n];
    MPI_Gather(&(energyProc), 1, MPI_DOUBLE_COMPLEX, &(energyAll), 1, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &energyProc, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &numProc, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &denomProc, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &denomAbsProc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    energyProc /= commsize;
    numProc /= commsize;
    denomProc /= commsize;
    denomAbsProc /= commsize;
    double stddev = 0., stddev2 = 0.;
    for (int i = 0; i < commsize; i++) {
      stddev += pow(abs(energyAll[i] - energyProc), 2);
      stddev2 += pow(abs(energyAll[i] - energyProc), 4);
    }
    stddev /= (commsize - 1);
    stddev2 /= commsize;
    stddev2 = sqrt((stddev2 - (commsize - 3) * pow(stddev, 2) / (commsize - 1)) / commsize) / 2. / sqrt(stddev) / sqrt(sqrt(commsize));
    stddev = sqrt(stddev / commsize);

    if (commrank == 0) {
      cout << "Variational Energy"<<endl;
      cout << format(" %14.2f   (%14.8f, %14.8f)   (%8.2e   (%8.2e))   (%3.3f, %3.3f) \n") % ((eneSteps[n] + 1) * dt) % energyProc.real() % energyProc.imag() % stddev % stddev2 % (denomProc / denomAbsProc).real() % (denomProc / denomAbsProc).imag(); 
    }
  }

  for (int n = 0; n < numEneSteps; n++) {
    complex<double> energyAll[commsize];
    for (int i = 0; i < commsize; i++) energyAll[i] = complex<double>(0., 0.);
    
    complex<double> energyProc = numMeanProj[n] / denomMeanProj[n];
    complex<double> numProc = numMeanProj[n];
    complex<double> denomProc = denomMeanProj[n];
    complex<double> denomAbsProc = denomAbsMeanProj[n];
    MPI_Gather(&(energyProc), 1, MPI_DOUBLE_COMPLEX, &(energyAll), 1, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &energyProc, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &numProc, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &denomProc, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &denomAbsProc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    energyProc /= commsize;
    numProc /= commsize;
    denomProc /= commsize;
    denomAbsProc /= commsize;
    double stddev = 0., stddev2 = 0.;
    for (int i = 0; i < commsize; i++) {
      stddev += pow(abs(energyAll[i] - energyProc), 2);
      stddev2 += pow(abs(energyAll[i] - energyProc), 4);
    }
    stddev /= (commsize - 1);
    stddev2 /= commsize;
    stddev2 = sqrt((stddev2 - (commsize - 3) * pow(stddev, 2) / (commsize - 1)) / commsize) / 2. / sqrt(stddev) / sqrt(sqrt(commsize));
    stddev = sqrt(stddev / commsize);

    if (commrank == 0) {
      cout << "Projected Energy"<<endl;
      cout << format(" %14.2f   (%14.8f, %14.8f)   (%8.2e   (%8.2e))   (%3.3f, %3.3f) \n") % ((eneSteps[n] + 1) * dt) % energyProc.real() % energyProc.imag() % stddev % stddev2 % (denomProc / denomAbsProc).real() % (denomProc / denomAbsProc).imag(); 
    }
  }

}

// calculates energy of the imaginary time propagated wave function using direct sampling of exponentials
void calcEnergyDirectMultiSlater(double enuc, MatrixXd& h1, MatrixXd& h1Mod, vector<MatrixXd>& chol)
{
  size_t norbs = Determinant::norbs;
  size_t nfields = chol.size();
  size_t nsweeps = schd.stochasticIter;
  size_t nsteps = schd.nsteps;
  size_t orthoSteps = schd.orthoSteps;
  double dt = schd.dt;
  vector<int> eneSteps = schd.eneSteps;

  // prep and init
  // this is for mean field subtraction
  MatrixXd hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  matPair rhf;
  rhf.first = hf.block(0, 0, norbs, Determinant::nalpha);
  rhf.second = hf.block(0, 0, norbs, Determinant::nbeta);
  
  vector<matPair> hsOperators;
  matPair oneBodyOperator;
  complex<double> mfConst = prepPropagatorHS(rhf, chol, hsOperators, oneBodyOperator);
  
  // this is the initial state
  matPair ref;
  hf = MatrixXd::Zero(norbs, norbs);
  readMat(hf, "rhf.txt");
  ref.first = hf.block(0, 0, norbs, Determinant::nalpha);
  ref.second = hf.block(0, 0, norbs, Determinant::nbeta);
 
  // this is the left state
  matPair refT;
  refT.first = ref.first.adjoint();
  refT.second = ref.second.adjoint();
  string fname = schd.determinantFile;
  std::array<std::vector<int>, 2> refDet; 
  std::array<std::vector<std::array<Eigen::VectorXi, 2>>, 2> ciExcitations;
  std::vector<double> ciParity; 
  std::vector<double> ciCoeffs;
  // if using text file name it "dets", if using binary use anything other than dets
  if (fname == "dets") readDeterminants(fname, refDet, ciExcitations, ciParity, ciCoeffs);
  else readDeterminantsBinary(fname, refDet, ciExcitations, ciParity, ciCoeffs);

  double cumulative;
  vector<int> alias; vector<double> prob;
  setUpAliasMethod(&ciCoeffs[1], ciCoeffs.size()-1, cumulative, alias, prob);

  std::array<std::vector<std::array<Eigen::VectorXi, 2>>, 2> ciExcitationsSample;
  std::vector<double> ciParitySample; 
  std::vector<double> ciCoeffsSample;

  matPair expOneBodyOperator;
  expOneBodyOperator.first =  (-dt * (h1Mod - oneBodyOperator.first) / 2.).exp();
  expOneBodyOperator.second = (-dt * (h1Mod - oneBodyOperator.second) / 2.).exp();

  // rotate cholesky vectors
  pair<vector<MatrixXcd>, vector<MatrixXcd>> rotChol;
  for (int i = 0; i < chol.size(); i++) {
    MatrixXcd rotUp = refT.first * chol[i];
    MatrixXcd rotDn = refT.second * chol[i];
    rotChol.first.push_back(rotUp);
    rotChol.second.push_back(rotDn);
  }
  
  // Gaussian sampling
  // field values arranged right to left
  vector<VectorXd> fields;
  normal_distribution<double> normal(0., 1.);
  
  //matPair green;
  //calcGreensFunction(refT, ref, green);
  //complex<double> refEnergy = calcHamiltonianElement(green, enuc, h1, chol);
  vector<double> times = { 0., 0.};
  auto refOverlapHam = calcHamiltonianElement(refT, ciExcitations, ciParity, ciCoeffs, ref, enuc, h1, chol);
  complex<double> refEnergy = refOverlapHam.second / refOverlapHam.first;
  //complex<double> refEnergy = calcHamiltonianElement(refT, ref, enuc, h1, rotChol);
  complex<double> ene0;
  if (schd.ene0Guess == 1.e10) ene0 = refEnergy;
  else ene0 = schd.ene0Guess;
  if (commrank == 0) {
    cout << "Initial state energy:  " << refEnergy << endl;
    cout << "Ground state energy guess:  " << ene0 << endl << endl; 
  }
  
  int nEneSteps = eneSteps.size();
  DQMCStatistics stats(nEneSteps);

  auto iterTime = getTime();
  double propTime = 0., eneTime = 0.;
  ArrayXd iTime(nEneSteps);
  for (int i = 0; i < nEneSteps; i++) iTime(i) = dt * (eneSteps[i] + 1);
  if (commrank == 0) cout << "Starting sampling sweeps\n";
  for (int sweep = 0; sweep < nsweeps; sweep++) {
    if (stats.isConverged()) break;
    if (sweep != 0 && sweep % (schd.printFrequency) == 0) {
      if (commrank == 0) {
        cout << "Sweep steps: " << sweep << endl << "Total walltime: " << getTime() - iterTime << " s\n";
        cout << "\nPropagation time:  " << propTime << " s\n";
        cout << "Energy evaluation time:  " << eneTime << " s\n\n";
      }
      stats.gatherAndPrintStatistics(iTime);
    }

    //sample the determinants
    if (schd.sampleDeterminants != -1)
    {
      int nSample = schd.sampleDeterminants;
      ciExcitationsSample[0].clear(); ciExcitationsSample[1].clear();
      ciParitySample.clear(); ciCoeffsSample.clear();

      vector<int> sample(nSample,-1); vector<double> wts(nSample,0.); 
      sample_N2_alias(&ciCoeffs[1], cumulative, sample, wts, alias, prob);
      ciExcitationsSample[0].push_back(ciExcitations[0][0]);
      ciExcitationsSample[1].push_back(ciExcitations[1][0]);
      ciCoeffsSample.push_back(ciCoeffs[0]);
      ciParitySample.push_back(ciParity[0]);

      for (int j=0; j<sample.size(); j++) {
        if (sample[j] == -1) break;
        ciExcitationsSample[0].push_back(ciExcitations[0][sample[j]+1]);
        ciExcitationsSample[1].push_back(ciExcitations[1][sample[j]+1]);
        ciCoeffsSample.push_back(wts[j]);
        ciParitySample.push_back(ciParity[sample[j]+1]);
      }

    }

    matPair rn;
    rn = ref;
    VectorXd fields = VectorXd::Zero(nfields);
    complex<double> orthoFac = complex<double>(1., 0.);
    int eneStepCounter = 0;
    ArrayXcd numSampleA(nEneSteps), denomSampleA(nEneSteps);
    numSampleA.setZero(); denomSampleA.setZero();
    for (int n = 0; n < nsteps; n++) {
      // sampling
      double init = getTime();
      matPair prop;
      prop.first = MatrixXcd::Zero(norbs, norbs);
      prop.second = MatrixXcd::Zero(norbs, norbs);
      for (int i = 0; i < nfields; i++) {
        double field_n_i = normal(generator);
        prop.first += field_n_i * hsOperators[i].first;
        //prop.second += field_n_i * hsOperators[i].second;
      }
      prop.first = (sqrt(dt) * prop.first).exp();
      rn.first = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nalpha)) * (expOneBodyOperator.first * (prop.first * (expOneBodyOperator.first * rn.first)));
      if (Determinant::nalpha == Determinant::nbeta) rn.second = rn.first;
      else {
        //prop.second = (sqrt(dt) * prop.second).exp();
        rn.second = exp((ene0 - enuc - mfConst) * dt / (2. * Determinant::nbeta)) * expOneBodyOperator.second * prop.first * expOneBodyOperator.second * rn.second;
      }

      propTime += getTime() - init;
      
      // orthogonalize for stability
      if (n % orthoSteps == 0) {
        orthogonalize(rn, orthoFac);
      }

      // measure
      init = getTime();
      if (n == eneSteps[eneStepCounter] ) { 
        if (stats.converged[eneStepCounter] == -1) {//-1 means this time slice has not yet converged
          pair<complex<double>, complex<double>> overlapHam;
          if (schd.sampleDeterminants != -1)
            overlapHam = calcHamiltonianElement(refT, ciExcitationsSample, ciParitySample, ciCoeffsSample, rn, enuc, h1, chol);
          else 
            overlapHam = calcHamiltonianElement(refT, ciExcitations, ciParity, ciCoeffs, rn, enuc, h1, chol);
          //auto overlapHam = calcHamiltonianElement_sRI(refT, ciExcitations, ciParity, ciCoeffs, rn, enuc, h1, chol, std::real(refEnergy));

          complex<double> overlap = orthoFac * overlapHam.first;
          complex<double> numSample = orthoFac * overlapHam.second;
          numSampleA[eneStepCounter] = numSample;
          denomSampleA[eneStepCounter] = overlap;
        }
        eneStepCounter++;
      }
      eneTime += getTime() - init;
    }
    stats.addSamples(numSampleA, denomSampleA);
  }

  if (commrank == 0) {
    cout << "\nPropagation time:  " << propTime << " s\n";
    cout << "Energy evaluation time:  " << eneTime << " s\n\n";
    //cout << "Intermediates:  " << times[0] << endl;
    //cout << "citer:  " << times[1] << endl << endl;
  }

  stats.gatherAndPrintStatistics(iTime);
  if (schd.printLevel > 10) stats.writeSamples();
}
