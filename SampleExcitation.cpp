/*
Developed by Sandeep Sharma with contributions from James E. Smith and Adam A. Homes, 2017
Copyright (c) 2017, Sandeep Sharma

This file is part of DICE.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "math.h"

int sample(std::vector<double>& wts, double& cumulative){
  double rand_no = ((double) rand() / (RAND_MAX))*cumulative;
  for (int i=0; i<wts.size(); i++) {
    if (rand_no <wts[i])
      return i;
    rand_no -= wts[i];
  }
}


//given "i" (an electron) we want to pick another electron "j" (hole "a")
//it is picked by weighted probability (ii|jj) or (ii|aa)
//the spin of "j"/"a" has to be predetermined
void MakeCumulativeArray(std::vector<int>& orbs, int OrbI, twoInt& I2, int spin,
			 std::vector<double>& wts_for_J, std::vector<int>& jindices, double& cumulative) {
  int index = 1;
  cumulative = 0.0;
  for (int j=0; j<orbs.size(); j++) {
    int J = orbs[j];
    if (orbs[j] == OrbI || J%2 != spin) continue ;
    double integral = I2(OrbI, OrbI, J, J);
    double cumulative += fabs(integral);
    wts_for_J.push_back(abs(integral));
    jindices.push_back(J);
  }
}

//given "i" (an electron) we want to pick another electron "j" (hole "a")
//it is picked by weighted probability (ii|jj) or (ii|aa)
//the spin of "j"/"a" has to be predetermined
void MakeCumulativeArray(std::vector<int>& orbs, int OrbI, oneInt& I1, int spin,
			 std::vector<double>& wts_for_J, std::vector<int>& jindices, double& cumulative) {
  int index = 1;
  cumulative = 0.0;
  for (int j=0; j<orbs.size(); j++) {
    int J = orbs[j];
    if (orbs[j] == OrbI || J%2 != spin) continue ;
    double integral = I1(OrbI, J);
    double cumulative += fabs(integral);
    wts_for_J.push_back(abs(integral));
    jindices.push_back(J);
  }
}



//given "i" and "j" (electrons) we want to pick the first hole "a"
//it is picked by weighted probability (ii|aa)+(jj|aa). Here it is assumed that "i" and "j" have the same spin
void MakeCumulativeArray(std::vector<int>& orbs, int OrbI, int OrbJ, twoInt& I2, int spin,
			 std::vector<double>& wts_for_J, std::vector<int>& jindices, double& cumulative) {
  int index = 1;
  cumulative = 0.0;
  for (int j=0; j<orbs.size(); j++) {
    int J = orbs[j];
    if (orbs[j] == OrbI || J%2 != spin) continue ;
    double integral = I2(OrbI, OrbI, J, J)+I2(OrbJ, OrbJ, J,J);
    double cumulative += fabs(integral);
    wts_for_J.push_back(abs(integral));
    jindices.push_back(J);
  }
}

//given electron i, j and hole "a", we want to pick the final hole "b"
//if "i" and "j" have the same spin then b is picked with probability (ia|jb)-(ib|ja)
//if "i" and "j" have different spins them b is picked with probability (ia|jb)
void MakeCumulativeArray(std::vector<int>& orbs, int OrbI, int OrbJ, int OrbA, twoInt& I2,
			 std::vector<double>& wts_for_J, std::vector<int>& jindices, double& cumulative) {
  bool SameSpin = OrbI%2 == OrbJ%2;
  int index = 1;
  cumulative = 0.0;
  for (int j=0; j<orbs.size(); j++) {
    int J = orbs[j];
    if (J == OrbA) continue;
    if (SameSpin &&  J%2 != OrbA%2) continue ;
    double integral = 0.0;
    if (SameSpin) integral = I2(OrbI, OrbA, OrbJ, J) - I2(OrbI, J, OrbJ, OrbA);
    else integral = I2(OrbI, OrbA, OrbJ, J);
    double cumulative += fabs(integral);
    wts_for_J.push_back(abs(integral));
    jindices.push_back(J);
  }
}


void getOneExcitation(closed, open, I2, I1, norbs)
    nclosed = size(closed)[1]

    pgen = 1.0
    #pic i randomly and update pgen

    i, pgen = StatsBase.sample(closed,1)[1], pgen/nclosed

    #make the cumulative list for a (i|a) and pick  a
    aints, awts, cumA = MakeCumulativeArray(open, i, I1, mod(i,2))
    a = StatsBase.sample(aints, StatsBase.WeightVec(awts))[1]
    pgen *= awts[locate(a,aints)]/cumA

    return i,a,pgen
end


function getTwoExcitation(closed, open, I2, norbs)
    #first decide same spin or opposite spin
    pgen = 1.0
    sameSpinThresh = 0.1
    sameSpin, pgen = false, 1.0-sameSpinThresh
    if (rand() < sameSpinThresh)
        sameSpin = true
        pgen = sameSpinThresh
    end

    nclosed = size(closed)[1]
    nopen = size(open)[1]

    #pick i randomly and update pgen
    i = StatsBase.sample(closed,1)[1]
    spinj = mod(i,2) #the same spin as i
    if (!sameSpin) spinj = mod(i+1,2) end #not the same spin as i

    #make the cumulative list for j (ii|jj) and pick  j
    jints, jwts, cumJ = MakeCumulativeArray(closed, i, I2, spinj)
    j = StatsBase.sample(jints, StatsBase.WeightVec(jwts))[1]
    pgenij = (1.0/nclosed)*jwts[locate(j,jints)]/cumJ

    #now correct the pgen for what it would have been ifyou have picked j first
    # and i second
    #correct the pgen
    iints_givenJ, iwts_givenJ, cumI_givenJ = MakeCumulativeArray(closed, j, I2, mod(i,2))
    pgenij += (1.0/nclosed)*(iwts_givenJ[locate(i,iints_givenJ)]/cumI_givenJ)


    pgenab = 1.0
    #make the cumulative list for a and pick a
    aints, awts, cumA = Int64[], Float64[], Float64
    if (sameSpin)
        aints, awts, cumA = MakeCumulativeArray(open, i, j, I2, mod(i,2))
    else
        aints, awts, cumA = MakeCumulativeArray(open, i, I2, mod(i,2))
    end
    a = StatsBase.sample(aints, StatsBase.WeightVec(awts), 1)[1]
    pgenab *= awts[locate(a,aints)]/cumA




    if (sameSpin)
        #make the cumulative list for b and pick b
        bints, bwts, cumB = MakeCumulativeArray(open, i, j, a, I2)
        b = StatsBase.sample(bints, StatsBase.WeightVec(bwts), 1)[1]
        pgenab *= bwts[locate(b,bints)]/cumB

        aints_givenbij, awts_givenbij, cumA_givenbij = MakeCumulativeArray(open, j, i, b, I2)
        pgenab += (awts[locate(b,aints)]/cumA)*(awts_givenbij[locate(a, aints_givenbij)]/cumA_givenbij)
      return i,j,a,b, pgen*pgenij*pgenab
    else
      #make the cumulative list for b and pick b
      bints, bwts, cumB = MakeCumulativeArray(open, j, I2, mod(j,2))
      b = StatsBase.sample(bints, StatsBase.WeightVec(bwts), 1)[1]
      pgenab *= bwts[locate(b,bints)]/cumB

      return i,j,a,b, pgen*pgenij*pgenab
    end
end


function getTwoExcitation_test(closed, open, I2, norbs)
    #first decide same spin or opposite spin
    pgen = 1.0
    nclosed = size(closed)[1]
    nopen = size(open)[1]

    #pick i randomly and update pgen
    #pgenij = 2.0/nclosed/(nclosed-1)
    #ij = StatsBase.sample(closed, 2, replace=false)
    #i,j=ij[1],ij[2]

    i = StatsBase.sample(closed,1)[1]
    jints, jwts, cumJ = MakeCumulativeArray(closed, i, I2, mod(i,2))
    j = StatsBase.sample(jints)[1]
    pgenij = (2.0/nclosed)*(1./length(jints))

    abints, abwts, cumJ = MakeCumulativeArray(open, i, j, I2, mod(i,2))
    #abwts = ones(length(abints))
    cumJ = sum(abwts)

    #=
    aindex = sample(abwts)
    awt, abwts[aindex] = abwts[aindex],0.0
    bindex = sample(abwts)
    pgenab = awt*abwts[bindex]*( 1.0/(cumJ-awt) +1.0/(cumJ-abwts[bindex]) )/cumJ
    a, b = abints[aindex], abints[bindex]
    =#

    ab = StatsBase.sample(1:length(abwts),StatsBase.WeightVec(abwts), 2,replace=false)
    a, b = abints[ab[1]], abints[ab[2]]
    pgenab = abwts[ab[1]]*abwts[ab[2]]*(1.0/(cumJ-abwts[ab[1]]) +1.0/(cumJ-abwts[ab[2]]))/cumJ

    return i,j,a,b, pgen*pgenij*pgenab

end

end
