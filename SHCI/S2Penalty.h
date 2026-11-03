/*
  S^2 (total spin) penalty kernel for DICE / SHCI.

  Implements the action of the total-spin operator on a CI vector expressed in
  DICE's determinant basis, for use as a level-shift penalty in the Davidson
  solver:

      (H + lambda * (S^2 - ss_target)) |Psi> = E |Psi>

  This mirrors the penalty form used by PySCF's `fci.addons.fix_spin_`, so that
  results are directly comparable between the two solvers.

  --------------------------------------------------------------------------
  Conventions (verified against DICE source, not documentation)
  --------------------------------------------------------------------------
  Spin orbitals are interleaved: spatial orbital p has
      alpha component at spin-orbital index 2*p      (even)
      beta  component at spin-orbital index 2*p + 1  (odd)
  This is confirmed by Determinant::Nalpha()/Nbeta(), which mask with
  0x5555... (even) and 0xAAAA... (odd) respectively, and by SHCI.cpp which
  populates determinants via setocc(orb*2, true) / setocc(orb*2+1, true).

  --------------------------------------------------------------------------
  Operator decomposition
  --------------------------------------------------------------------------
      S^2 = Sz^2 + Sz + S_minus * S_plus

  Diagonal part, for a single determinant |D>:

      <D|S^2|D> = Sz*(Sz+1) + n_beta - n_doubly_occupied
      where Sz = (n_alpha - n_beta)/2

  Off-diagonal part (S_minus * S_plus) connects |D> to |D'> where D' is
  obtained by two single spin-flips in *different* spatial orbitals p != q:

      S_plus  : beta in p  ->  alpha in p
      S_minus : alpha in q ->  beta  in q

  so |D> must have (beta occupied in p) and (alpha occupied in q), and the
  target must have both resulting slots empty. The magnitude of every such
  matrix element is exactly 1; only the fermionic sign varies.

  IMPORTANT: this sign is *not* the sign of the stored Hamiltonian element
  Helements[i][j] for the same determinant pair, so it must be computed
  independently. Getting it wrong yields a penalty that still converges, but
  to the wrong state.

  --------------------------------------------------------------------------
  Validation
  --------------------------------------------------------------------------
  The formulae above were validated in Python against
  pyscf.fci.spin_op.contract_ss over 7 (norb, na, nb) combinations spanning
  closed-shell, open-shell, and na != nb cases; worst deviation 1.78e-15.
  See s2_ref.py / BUILD_NOTES.md.

  Note when comparing against PySCF directly: DICE orders creation operators
  interleaved while PySCF groups all-alpha-then-all-beta. The two bases differ
  by a per-determinant phase (-1)^{#pairs (i<j) with i beta, j alpha}. That
  phase cancels in a pure-DICE calculation (it is a diagonal similarity
  transform, and S^2 is basis-independent under it), but it is required for
  any element-by-element cross-check against PySCF.
*/

#ifndef S2PENALTY_HEADER_H
#define S2PENALTY_HEADER_H

#include <algorithm>
#include <vector>

#include "Determinants.h"
#include "global.h"

namespace S2Penalty {

/*!
Number of doubly occupied spatial orbitals in a determinant.
*/
inline int numDoublyOccupied(const Determinant& d, int norbs) {
  int ndouble = 0;
  for (int p = 0; p < norbs; p++) {
    if (d.getocc(2 * p) && d.getocc(2 * p + 1)) ndouble++;
  }
  return ndouble;
}

/*!
Diagonal matrix element <D|S^2|D>.

  <D|S^2|D> = Sz(Sz+1) + n_beta - n_doubly_occupied

Equivalently (and this is the form that makes the spin-purity intuition
obvious) this equals Sz(Sz+1) + (number of singly-occupied beta orbitals).
*/
inline double diagonal(const Determinant& d, int norbs) {
  const int na = d.Nalpha();
  const int nb = d.Nbeta();
  const double sz = 0.5 * (na - nb);
  return sz * (sz + 1.0) + static_cast<double>(nb - numDoublyOccupied(d, norbs));
}

/*!
Number of occupied spin orbitals with index strictly below k.

This is the Jordan-Wigner string length that sets the fermionic sign of a
single creation/annihilation operator acting at position k.
*/
inline int numOccupiedBelow(const Determinant& d, int k) {
  int count = 0;
  const int nlong = k / 64;
  for (int i = 0; i < nlong; i++) {
    count += BitCount(d.repr[i]);
  }
  const int rem = k % 64;
  if (rem > 0) {
    const long one = 1;
    const long mask = (one << rem) - one;
    count += BitCount(d.repr[nlong] & mask);
  }
  return count;
}

/*!
Apply a^dag_create a_destroy to |d>.

Returns true on success and writes the resulting determinant to `out`,
multiplying `sgn` by the fermionic phase. Returns false if the excitation
annihilates the determinant (source empty or target already occupied), in
which case `out` and `sgn` are unspecified.
*/
inline bool applyExcitation(const Determinant& d, int create, int destroy,
                            Determinant& out, double& sgn) {
  if (!d.getocc(destroy)) return false;

  double phase = (numOccupiedBelow(d, destroy) % 2) ? -1.0 : 1.0;

  out = d;
  out.setocc(destroy, false);

  if (out.getocc(create)) return false;

  if (numOccupiedBelow(out, create) % 2) phase = -phase;
  out.setocc(create, true);

  sgn *= phase;
  return true;
}

/*!
Apply S_minus S_plus to determinant `d`, emitting each (determinant, coeff)
pair to the supplied callback.

The callback is invoked as `emit(const Determinant& target, double coeff)`.
Templating on the callback keeps this header free of any assumption about how
the caller maps determinants to vector indices -- in DICE that lookup differs
between the sparse (`Hmult2`) and direct (`HmultDirect`) paths.
*/
template <typename EmitFn>
inline void applySminusSplus(const Determinant& d, int norbs, EmitFn emit) {
  for (int p = 0; p < norbs; p++) {
    // S_plus needs a beta electron in p and an empty alpha slot in p
    if (!d.getocc(2 * p + 1)) continue;
    if (d.getocc(2 * p)) continue;

    Determinant d1;
    double sgn1 = 1.0;
    if (!applyExcitation(d, 2 * p, 2 * p + 1, d1, sgn1)) continue;

    for (int q = 0; q < norbs; q++) {
      if (q == p) continue;
      // S_minus needs an alpha electron in q and an empty beta slot in q
      if (!d1.getocc(2 * q)) continue;
      if (d1.getocc(2 * q + 1)) continue;

      Determinant d2;
      double sgn2 = sgn1;
      if (!applyExcitation(d1, 2 * q + 1, 2 * q, d2, sgn2)) continue;

      emit(d2, sgn2);
    }
  }
}

/*!
Diagonal contribution of the p == q terms of S_minus S_plus.

applySminusSplus deliberately skips q == p. Those terms are already folded
into `diagonal()` via the (n_beta - n_doubly_occupied) counting, so they must
not be added twice. This function exists only to document that fact and to
let the unit test assert the decomposition is complete.
*/
inline double numSinglyOccupiedBeta(const Determinant& d, int norbs) {
  int count = 0;
  for (int p = 0; p < norbs; p++) {
    if (d.getocc(2 * p + 1) && !d.getocc(2 * p)) count++;
  }
  return static_cast<double>(count);
}

/*!
Index for resolving a Determinant back to its position in the CI vector.

This exists because Dice's determinant array (`SHMDets`) is NOT sorted --
Dice keeps a separate `SortedDets` copy for its own searches, whose positions
deliberately do not correspond to CI-vector indices. A naive
`lower_bound(SHMDets, ...)` is therefore wrong.

We build a permutation `order` such that `dets[order[k]]` is ascending, then
binary search over that permutation and return the *original* index. Build
cost is O(n log n) once per Davidson solve; each lookup is O(log n).
*/
class DetIndex {
 public:
  DetIndex() : dets_(nullptr), size_(0) {}

  void build(Determinant* dets, int size) {
    dets_ = dets;
    size_ = size;
    order_.resize(size);
    for (int i = 0; i < size; i++) order_[i] = i;
    Determinant* d = dets;
    std::sort(order_.begin(), order_.end(),
              [d](int a, int b) { return d[a] < d[b]; });
  }

  bool ready() const { return dets_ != nullptr && (int)order_.size() == size_; }

  /*! Return the CI-vector index of `target`, or -1 if it is not present. */
  int find(const Determinant& target) const {
    int lo = 0, hi = size_;
    while (lo < hi) {
      const int mid = lo + (hi - lo) / 2;
      if (dets_[order_[mid]] < target) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    if (lo < size_ && dets_[order_[lo]] == target) return order_[lo];
    return -1;
  }

 private:
  Determinant* dets_;
  int size_;
  std::vector<int> order_;
};

/*!
Helper bundling everything needed to apply the S^2 level-shift penalty.

Mirrors PySCF's SpinPenaltyFCISolver::contract_2e, which has TWO regimes:

    if ss < sz*(sz+1) + 0.1:
        (S^2 - ss) |Psi>             # LINEAR
    else:
        (S^2 - ss)^2 |Psi>           # QUADRATIC

The distinction is essential, not cosmetic. The linear form is only valid when
`ss` is the LOWEST spin reachable in this Sz sector. Above that, states with
S^2 < ss receive a *negative* shift and are pushed BELOW the target, so the
Davidson solver happily converges to the wrong spin. (Observed: targeting
ss=2 in an Ms=0 O2 space with a linear-only implementation produced the
singlet at E - 0.4 Ha, i.e. the lowest eigenvalue of a malformed operator.)

The quadratic form (S^2 - ss)^2 is positive semi-definite and vanishes only on
the target spin, so every other spin is pushed strictly up.

Implementation note: because

    (S^2 - ss)^2 x = (S^2 - ss) [ (S^2 - ss) x ]

the quadratic case needs no new kernel -- it is `shiftedRow` applied twice,
with a full-vector exchange in between (the second pass reads the intermediate
at arbitrary indices, so under MPI it must be all-reduced first). The caller
owns that exchange because row ownership differs between Dice's two Davidson
functors.
*/
class Penalty {
 public:
  Penalty()
      : dets_(nullptr), size_(0), norbs_(0), lambda_(0.0), ss_(0.0), sz_(0.0) {}

  void init(Determinant* dets, int size, int norbs, double lambda, double ss) {
    dets_ = dets;
    size_ = size;
    norbs_ = norbs;
    lambda_ = lambda;
    ss_ = ss;
    sz_ = 0.0;
    if (dets != nullptr && size > 0) {
      sz_ = 0.5 * std::abs(dets[0].Nalpha() - dets[0].Nbeta());
    }
    if (lambda != 0.0 && dets != nullptr) index_.build(dets, size);
  }

  bool active() const { return lambda_ != 0.0 && dets_ != nullptr; }

  /*! True when the quadratic form is required (same test as PySCF). */
  bool quadratic() const { return !(ss_ < sz_ * (sz_ + 1.0) + 0.1); }

  double lambda() const { return lambda_; }
  double ssTarget() const { return ss_; }
  double sz() const { return sz_; }
  int size() const { return size_; }

  /*!
  Return [ (S^2 - ss) v ]_row.

  Reads v at arbitrary indices, writes nothing -- so the caller may add the
  result to whichever row it owns, under any MPI partitioning.
  */
  CItype shiftedRow(int row, const CItype* v) const {
    const Determinant& di = dets_[row];
    CItype acc = (diagonal(di, norbs_) - ss_) * v[row];
    applySminusSplus(di, norbs_, [&](const Determinant& target, double coeff) {
      const int j = index_.find(target);
      if (j >= 0) acc += coeff * v[j];
    });
    return acc;
  }

  /*! Return [ S^2 v ]_row (no shift). */
  CItype s2Row(int row, const CItype* v) const {
    const Determinant& di = dets_[row];
    CItype acc = diagonal(di, norbs_) * v[row];
    applySminusSplus(di, norbs_, [&](const Determinant& target, double coeff) {
      const int j = index_.find(target);
      if (j >= 0) acc += coeff * v[j];
    });
    return acc;
  }

 private:
  Determinant* dets_;
  int size_;
  int norbs_;
  double lambda_;
  double ss_;
  double sz_;
  DetIndex index_;
};

/*!
Expectation value <Psi|S^2|Psi> for a normalized CI vector.

Standalone (does not need a Penalty object) so <S^2> can be reported for
ordinary unpenalized runs too -- which is what
Qiskit/qiskit-addon-dice-solver#59 asks for.

Serial by design: it is called once per root at the end of the variational
solve, not inside the Davidson loop, so the O(ndet * norb^2) cost is
negligible and there is no partitioning to respect. `dets` need not be
sorted; a DetIndex permutation is built internally.

Returns <S^2> / <Psi|Psi>, so an unnormalized vector is handled correctly.
*/
inline double expectation(Determinant* dets, int size, int norbs,
                          const CItype* v) {
  if (dets == nullptr || size <= 0) return 0.0;

  DetIndex index;
  index.build(dets, size);

  double num = 0.0;  // <Psi|S^2|Psi>
  double den = 0.0;  // <Psi|Psi>
  for (int i = 0; i < size; i++) {
    const Determinant& di = dets[i];

    CItype acc = diagonal(di, norbs) * v[i];
    applySminusSplus(di, norbs, [&](const Determinant& target, double coeff) {
      const int j = index.find(target);
      if (j >= 0) acc += coeff * v[j];
    });

#ifdef Complex
    num += std::real(std::conj(v[i]) * acc);
    den += std::norm(v[i]);
#else
    num += v[i] * acc;
    den += v[i] * v[i];
#endif
  }
  return (den > 0.0) ? num / den : 0.0;
}

}  // namespace S2Penalty

#endif
