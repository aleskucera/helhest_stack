"""GPU estimator for the v2 attitude cost  L = sum_t (pitch_t^2 + roll_t^2).

Companion to `contact.py` (same node machinery, same rho1 separable belief
model, same register-resident Clark fold). Where the v1 scalar cost allowed
the variance to be one grid convolution, the attitude cost needs the joint
Gaussian of the stacked attitude vector x (2T rows): S_x = W C W'. That is
computed by direct pair contraction over folded nodes -- each attitude row
touches 12 nodes (3 wheels x 4 corners) of <= K candidates, and pairs of
rows further apart than the correlation table's reach contribute exactly
zero, so the work is banded.

The fold semantics match `studies/baseprod/v2_moments.fold_supports_fast`:
cross-node covariance by the lambda (SSTA) rule, Clark-exact variance on
the diagonal (carried as a per-node correction `dvar`), so the campaign's
frozen deficit correction applies unchanged.

Moments only (E[L], Var[L]); CVaR and the deficit correction are one
fused multiply-add on the host or in the MPPI cost kernel.

HOW THE PAIR CONTRACTION IS ORGANISED (rewritten 2026-08-31; the semantics are
unchanged and `studies/bench/v2_att_mirror_test.py` is the proof). The naive
shape -- a thread per step pair looping over 12x12 node pairs of K x K
candidate pairs -- does the right arithmetic in the wrong order. MEASURED on
an RTX A500, 90x90 map, P = 256, T = 40 (`studies/bench/v2_att_wall.py`,
corr_len 0.15 m at a 0.10 m cell): 446 -> 32 us/plan, a factor of 14. The
deterministic forward pass on the same scene is 0.3 us/plan, so propagating
the belief now costs 85x a rollout rather than 1030x, and the estimator is
29x cheaper than the sampler it matches the accuracy of rather than 2x.
Three structural facts and two mechanical ones get it there.

  1. THE CANDIDATE SET OF A WHEEL-STEP IS SMALLER THAN 4K. The four bilinear
     corners of one wheel share a yaw bin, so their structuring elements are
     the SAME element anchored at four cells that differ by (0/1, 0/1); their
     union is 14-18 distinct cells at the study's 0.10 m cell, not 4x7 = 28
     (20-26 rather than 4x13 = 52 at the node's 0.08 m). `k_pack_att` merges
     the four corners into one per-(plan, step, wheel) list of (cell,
     pitch-weight, roll-weight) -- the corner weight, the settle row weight
     and the lambda weight all multiplied in -- so the contraction loses the
     12x12 node loop and its scattered reads entirely. Merging two entries
     that name the same cell is exact: the covariance depends on a candidate
     only through its cell, so their weights add.

  2. REACH IS A PROPERTY OF THE CELLS, NOT OF THE TIME LAG. `rho1` is exactly
     zero at and beyond its table length, so a wheel-step pair whose candidate
     BOUNDING BOXES are that far apart in y or in x contributes exactly zero.
     `k_pack_att` emits the box and the pair kernel culls on it, per WHEEL
     pair -- which is where culling pays, since at a large step lag only one
     or two of the nine wheel pairs are still in reach. This is exact, not a
     tolerance, and it replaces guessing a band from a nominal step length:
     `band_t` survives only as a coarse cap on the step lag (and as the
     `band_m` knob), while the cell-level cull does the real work.

  3. THE 2x2 ATTITUDE BLOCK IS ONE CONTRACTION, NOT FOUR. For candidate e of
     the row step and f of the column step all four block entries share the
     factor rho1[|dy|] rho1[|dx|], so accumulating u = sum_f ap_f rho and
     v = sum_f ar_f rho in the inner loop and combining them with (ap_e, ar_e)
     once per e costs two FMAs per candidate pair instead of four multiplies
     and four adds.

  4. THE INNER LOOP IS BOUND BY LOAD ISSUE, NOT ARITHMETIC. So it carries six
     row-side candidates at once against one shared column-side candidate,
     and the two correlation lookups are pre-multiplied into one padded 2-D
     table (`rho2`) wide enough that the range test guarding them can go. The
     pair list is ordered by its COLUMN step, so a warp shares the list the
     innermost loop walks. Together: 11.3 -> 4.8 ms of the 8.5 ms call.

  5. LAYOUT. The fold's candidate arrays are SLOT-major, [slot][node] rather
     than [node][slot]: a warp writing slot j then writes one contiguous run
     instead of 32 four-byte pokes 4*MAX_K apart. That alone was 20% of the
     fold and 30% of the pack. The fold also carries five MAX_K-wide vectors
     rather than ten -- they are dynamically indexed, so they are stack
     traffic, and the stack is most of what the fold costs.

The diagonal Clark correction `node_dvar` lives on identical NODES, so it is
not part of the packed contraction at all: `k_att_diag` reduces it (and the
mean row) per step and the pair kernel adds it on the tr == ts blocks only.
`k_att_finish` was one thread per plan over a (2T)^2 matrix -- 256 threads for
the whole GPU; it is now a row pass and a reduction.

Nothing here is atomic or order-dependent: every kernel writes locations no
other thread touches, so two calls on the same input agree bit for bit, and
`moments()` is still capture-safe (launches only -- the one `zero_` the old
version needed per call is gone, since S is written in full on every listed
step pair and is exactly zero everywhere else for every input).
"""
from __future__ import annotations

import os

import numpy as np
import warp as wp

from ..engine.envelope import cyl_table, wheel_half_width
from .contact import MAX_K, VECF, VECI, k_nodes
from .settle import settle_map

# Compile-time bound on the DEDUPLICATED candidate count of one (plan, step, wheel):
# the union of the four corner-shifted copies of the structuring element. 18 at the
# study's 0.10 m cell, 26 at the perception node's 0.08 m, and never more than 4 *
# MAX_K by construction. It bounds three per-thread scratch vectors in `k_pack_att`
# and nothing else; the constructor computes the true requirement from the element
# tables and raises rather than truncating.
MAX_U = int(os.environ.get("HELHEST_RISK_MAX_U", "32"))
UVECF = wp.types.vector(length=MAX_U, dtype=wp.float32)
UVECI = wp.types.vector(length=MAX_U, dtype=wp.int32)
VEC4I = wp.types.vector(length=4, dtype=wp.int32)


@wp.kernel
def k_fold_att(
    belief: wp.array2d(dtype=wp.float32),
    sigma: wp.array2d(dtype=wp.float32),
    off_dy: wp.array2d(dtype=wp.int32),
    off_dx: wp.array2d(dtype=wp.int32),
    off_cap: wp.array2d(dtype=wp.float32),
    k_real: wp.array(dtype=wp.int32),
    rho_e: wp.array3d(dtype=wp.float32),
    anchor_y: wp.array(dtype=wp.int32),
    anchor_x: wp.array(dtype=wp.int32),
    node_bin: wp.array(dtype=wp.int32),
    ny: int, nx: int,
    node_mean: wp.array(dtype=wp.float32),
    node_dvar: wp.array(dtype=wp.float32),
    cand_iy: wp.array2d(dtype=wp.int32),
    cand_ix: wp.array2d(dtype=wp.int32),
    cand_wsig: wp.array2d(dtype=wp.float32),
):
    """Per node: gather, sort, fold (contact.py's recursion verbatim), then
    write the folded mean, the lambda-weighted candidates, and the
    Clark-vs-lambda variance gap.

    Two departures from `contact.k_fold_scatter`, both of them cost and neither
    of them semantics. (a) The candidates' correlation with each other is a
    property of the yaw BIN, not of the map, so `rho_e[b, j1, j2]` is built once
    on the host and the recursion's two `rho1` lookups and their product become
    one load. That is why the sort carries the element INDEX and not the
    offsets: the index is what `rho_e` and `off_d*` are keyed by, and one
    int vector is one less than two. (b) Five per-thread vectors, all of them
    MAX_K wide and dynamically indexed -- so they are stack traffic, and the
    stack is what the fold costs. The cells are recomputed from the offsets
    after the sort, `1 - phi` is recomputed rather than stored, and the
    backward weights are written over `m`, which is dead by then."""
    n = wp.tid()
    b = node_bin[n]
    ay = anchor_y[n]
    ax = anchor_x[n]
    kk = k_real[b]

    m = VECF(); sg = VECF(); ii = VECI()
    for j in range(kk):
        iy = wp.clamp(ay + off_dy[b, j], 0, ny - 1)
        ix = wp.clamp(ax + off_dx[b, j], 0, nx - 1)
        m[j] = belief[iy, ix] + off_cap[b, j]
        sg[j] = sigma[iy, ix]
        ii[j] = j
    for a in range(1, kk):
        km = m[a]; ks = sg[a]; ki = ii[a]
        q = a - 1
        while q >= 0 and m[q] < km:
            m[q+1] = m[q]; sg[q+1] = sg[q]; ii[q+1] = ii[q]
            q = q - 1
        m[q+1] = km; sg[q+1] = ks; ii[q+1] = ki

    base = m[0]
    for j in range(kk):
        m[j] = m[j] - base

    mean_run = m[0]
    var_run = sg[0] * sg[0]
    cov = VECF()
    for j in range(kk):
        cov[j] = rho_e[b, ii[0], ii[j]] * sg[0] * sg[j]
    phi_s = VECF()
    for i in range(1, kk):
        v2 = sg[i] * sg[i]
        a2 = var_run + v2 - 2.0 * cov[i]
        if a2 < 0.0:
            a2 = 0.0
        aa = wp.sqrt(a2)
        alpha = float(0.0)
        if aa < 1.0e-9:
            if mean_run - m[i] >= 0.0:
                alpha = 1.0e6
            else:
                alpha = -1.0e6
        else:
            alpha = (mean_run - m[i]) / aa
        ph = 0.5 * (1.0 + wp.erf(alpha * 0.70710678))
        pn = 1.0 - ph
        pd = 0.39894228 * wp.exp(-0.5 * alpha * alpha)
        new_mean = mean_run * ph + m[i] * pn + aa * pd
        new_ex2 = ((mean_run * mean_run + var_run) * ph + (m[i] * m[i] + v2) * pn
                   + (mean_run + m[i]) * aa * pd)
        var_run = wp.max(new_ex2 - new_mean * new_mean, 0.0)
        for j in range(kk):
            cov[j] = cov[j] * ph + rho_e[b, ii[i], ii[j]] * sg[i] * sg[j] * pn
        phi_s[i-1] = ph
        mean_run = new_mean

    # backward pass, written over `m` (dead since the fold ended)
    tail = float(1.0)
    for i in range(kk - 1, 0, -1):
        m[i] = (1.0 - phi_s[i-1]) * tail
        tail = tail * phi_s[i-1]
    m[0] = tail

    # lambda C lambda (this node with itself) for the diagonal correction; symmetric,
    # so only the upper triangle is summed
    lcl = float(0.0)
    for a in range(kk):
        wa_ = m[a] * sg[a]
        lcl += wa_ * wa_
        for bb in range(a + 1, kk):
            lcl += 2.0 * wa_ * m[bb] * sg[bb] * rho_e[b, ii[a], ii[bb]]
    node_mean[n] = mean_run + base
    node_dvar[n] = var_run - lcl
    # SLOT-MAJOR, not node-major. One thread writes one node's <= K candidates; laying
    # the arrays out [slot][node] makes each slot's write across a warp one contiguous
    # run instead of 32 four-byte pokes 4*MAX_K apart, which was costing more than the
    # fold's arithmetic. Only the weight is padded -- everything that reads these arrays
    # stops at a zero weight.
    for j in range(kk):
        jj = ii[j]
        cand_iy[j, n] = wp.clamp(ay + off_dy[b, jj], 0, ny - 1)
        cand_ix[j, n] = wp.clamp(ax + off_dx[b, jj], 0, nx - 1)
        cand_wsig[j, n] = m[j] * sg[j]
    for j in range(kk, MAX_K):
        cand_wsig[j, n] = 0.0


@wp.kernel
def k_pack_att(
    node_wgt: wp.array(dtype=wp.float32),
    node_bin: wp.array(dtype=wp.int32),
    k_real: wp.array(dtype=wp.int32),
    cand_iy: wp.array2d(dtype=wp.int32),
    cand_ix: wp.array2d(dtype=wp.int32),
    cand_wsig: wp.array2d(dtype=wp.float32),
    row_w: wp.array2d(dtype=wp.float32),
    n_t: int,
    pk_cnt: wp.array(dtype=wp.int32),
    pk_c: wp.array2d(dtype=wp.int32),
    pk_w: wp.array2d(dtype=wp.vec2f),
    pk_box: wp.array(dtype=VEC4I),
):
    """One thread per (plan, step, wheel): merge the four corners' folded
    candidates into one list keyed by CELL, carrying the pitch and roll
    weights (settle row x corner weight x lambda x sigma), and the list's
    bounding box in cells.

    The merge is exact: two candidates naming the same cell enter the
    covariance identically, so summing their weights and contracting once is
    the same number in exact arithmetic and fewer roundings in float32."""
    iw = wp.tid()
    w = iw % 3
    r = iw / 3
    t = r % n_t
    p = r / n_t
    cap = pk_c.shape[1]
    rw0 = row_w[0, w]
    rw1 = row_w[1, w]

    code = UVECI(); ap = UVECF(); ar = UVECF()
    cnt = int(0)
    ymin = int(1 << 24); ymax = int(-1); xmin = int(1 << 24); xmax = int(-1)
    for c in range(4):
        n = ((p * 3 + w) * n_t + t) * 4 + c
        g = node_wgt[n]
        if g == 0.0:
            continue
        kk = k_real[node_bin[n]]
        for j in range(kk):
            ws = cand_wsig[j, n]
            if ws == 0.0:
                continue
            iy = cand_iy[j, n]
            ix = cand_ix[j, n]
            cd = (iy << 16) | ix
            slot = cnt
            for e in range(cnt):
                if code[e] == cd:
                    slot = e
                    break
            if slot == cnt:
                if cnt >= cap:      # unreachable: the constructor sizes for the worst bin
                    continue
                code[cnt] = cd
                ap[cnt] = 0.0
                ar[cnt] = 0.0
                cnt += 1
                ymin = wp.min(ymin, iy); ymax = wp.max(ymax, iy)
                xmin = wp.min(xmin, ix); xmax = wp.max(xmax, ix)
            gw = g * ws
            ap[slot] = ap[slot] + rw0 * gw
            ar[slot] = ar[slot] + rw1 * gw

    pk_cnt[iw] = cnt
    for e in range(cnt):
        pk_c[iw, e] = code[e]
        pk_w[iw, e] = wp.vec2f(ap[e], ar[e])
    pk_box[iw] = VEC4I(ymin, ymax, xmin, xmax)


@wp.kernel
def k_att_diag(
    node_mean: wp.array(dtype=wp.float32),
    node_dvar: wp.array(dtype=wp.float32),
    node_wgt: wp.array(dtype=wp.float32),
    row_w: wp.array2d(dtype=wp.float32),
    n_t: int,
    mx: wp.array2d(dtype=wp.float32),
    dd: wp.array2d(dtype=wp.vec3f),
):
    """One thread per (plan, step): the attitude mean row, and the Clark
    variance correction, which is carried by identical NODES and so only ever
    lands on the tr == ts block."""
    it = wp.tid()
    t = it % n_t
    p = it / n_t
    mp = float(0.0); mr = float(0.0)
    d00 = float(0.0); d01 = float(0.0); d11 = float(0.0)
    for w in range(3):
        rw0 = row_w[0, w]
        rw1 = row_w[1, w]
        for c in range(4):
            n = ((p * 3 + w) * n_t + t) * 4 + c
            g = node_wgt[n]
            mp += rw0 * g * node_mean[n]
            mr += rw1 * g * node_mean[n]
            gg = g * g * node_dvar[n]
            d00 += rw0 * rw0 * gg
            d01 += rw0 * rw1 * gg
            d11 += rw1 * rw1 * gg
    mx[p, 2 * t] = mp
    mx[p, 2 * t + 1] = mr
    dd[p, t] = wp.vec3f(d00, d01, d11)


@wp.kernel
def k_att_cov(
    pk_cnt: wp.array(dtype=wp.int32),
    pk_c: wp.array2d(dtype=wp.int32),
    pk_w: wp.array2d(dtype=wp.vec2f),
    pk_box: wp.array(dtype=VEC4I),
    dd: wp.array2d(dtype=wp.vec3f),
    rho2: wp.array(dtype=wp.float32),
    reach: int, half: int,
    pair_tr: wp.array(dtype=wp.int32),
    pair_ts: wp.array(dtype=wp.int32),
    n_t: int, n_pair: int,
    S: wp.array3d(dtype=wp.float32),
):
    """One thread per (plan, step pair): the 2x2 attitude block, contracted
    over the two steps' packed candidate lists, wheel pair by wheel pair, with
    out-of-reach wheel pairs culled on their bounding boxes.

    The inner loop carries SIX row-side candidates at once. That is not
    cosmetic unrolling: the loop is bound by L1 load issue, not by arithmetic,
    and the six lanes share the one column-side candidate they are all
    multiplied against -- so the loads per candidate PAIR fall from three
    (cell, weight, correlation) to about one and a third.

    `rho2` is rho1 outer-producted with itself and padded with the zeros the
    hard cutoff implies, wide enough (`half = len(rho1) + 2 * element span`)
    that no lag surviving the bounding-box cull can index outside it. So the
    two correlation lookups become one, and the range test that guarded them
    disappears -- a zero from the table is the same zero the test produced."""
    tid = wp.tid()
    p = tid / n_pair
    q = tid % n_pair
    tr = pair_tr[q]
    ts = pair_ts[q]
    W = 2 * half - 1
    off = half - 1
    s00 = float(0.0); s01 = float(0.0); s10 = float(0.0); s11 = float(0.0)
    for wa in range(3):
        ia = (p * n_t + tr) * 3 + wa
        na = pk_cnt[ia]
        if na == 0:
            continue
        ba = pk_box[ia]
        for wb in range(3):
            ib = (p * n_t + ts) * 3 + wb
            nb = pk_cnt[ib]
            if nb == 0:
                continue
            bb = pk_box[ib]
            # exactly-zero test: rho1 is hard zero at and beyond its table length,
            # so boxes that far apart in y or in x contribute nothing at all
            if ba[0] - bb[1] >= reach or bb[0] - ba[1] >= reach:
                continue
            if ba[2] - bb[3] >= reach or bb[2] - ba[3] >= reach:
                continue
            ngrp = na / 6
            for gg in range(ngrp):
                e = 6 * gg
                a0 = pk_c[ia, e]; a1 = pk_c[ia, e+1]; a2 = pk_c[ia, e+2]
                a3 = pk_c[ia, e+3]; a4 = pk_c[ia, e+4]; a5 = pk_c[ia, e+5]
                y0 = (a0 >> 16) + off; x0 = (a0 & 65535) + off
                y1 = (a1 >> 16) + off; x1 = (a1 & 65535) + off
                y2 = (a2 >> 16) + off; x2 = (a2 & 65535) + off
                y3 = (a3 >> 16) + off; x3 = (a3 & 65535) + off
                y4 = (a4 >> 16) + off; x4 = (a4 & 65535) + off
                y5 = (a5 >> 16) + off; x5 = (a5 & 65535) + off
                g0 = pk_w[ia, e]; g1 = pk_w[ia, e+1]; g2 = pk_w[ia, e+2]
                g3 = pk_w[ia, e+3]; g4 = pk_w[ia, e+4]; g5 = pk_w[ia, e+5]
                u0 = float(0.0); v0 = float(0.0); u1 = float(0.0); v1 = float(0.0)
                u2 = float(0.0); v2 = float(0.0); u3 = float(0.0); v3 = float(0.0)
                u4 = float(0.0); v4 = float(0.0); u5 = float(0.0); v5 = float(0.0)
                for f in range(nb):
                    cb = pk_c[ib, f]
                    yb = cb >> 16
                    xb = cb & 65535
                    h = pk_w[ib, f]
                    r0 = rho2[(y0 - yb) * W + (x0 - xb)]
                    u0 += h[0] * r0; v0 += h[1] * r0
                    r1 = rho2[(y1 - yb) * W + (x1 - xb)]
                    u1 += h[0] * r1; v1 += h[1] * r1
                    r2 = rho2[(y2 - yb) * W + (x2 - xb)]
                    u2 += h[0] * r2; v2 += h[1] * r2
                    r3 = rho2[(y3 - yb) * W + (x3 - xb)]
                    u3 += h[0] * r3; v3 += h[1] * r3
                    r4 = rho2[(y4 - yb) * W + (x4 - xb)]
                    u4 += h[0] * r4; v4 += h[1] * r4
                    r5 = rho2[(y5 - yb) * W + (x5 - xb)]
                    u5 += h[0] * r5; v5 += h[1] * r5
                s00 += (g0[0] * u0 + g1[0] * u1 + g2[0] * u2
                        + g3[0] * u3 + g4[0] * u4 + g5[0] * u5)
                s01 += (g0[0] * v0 + g1[0] * v1 + g2[0] * v2
                        + g3[0] * v3 + g4[0] * v4 + g5[0] * v5)
                s10 += (g0[1] * u0 + g1[1] * u1 + g2[1] * u2
                        + g3[1] * u3 + g4[1] * u4 + g5[1] * u5)
                s11 += (g0[1] * v0 + g1[1] * v1 + g2[1] * v2
                        + g3[1] * v3 + g4[1] * v4 + g5[1] * v5)
            for e in range(ngrp * 6, na):    # the 0..5 candidates past the last group
                ca = pk_c[ia, e]
                ya = (ca >> 16) + off
                xa = (ca & 65535) + off
                ga = pk_w[ia, e]
                u = float(0.0); v = float(0.0)
                for f in range(nb):
                    cb = pk_c[ib, f]
                    rr = rho2[(ya - (cb >> 16)) * W + (xa - (cb & 65535))]
                    h = pk_w[ib, f]
                    u += h[0] * rr; v += h[1] * rr
                s00 += ga[0] * u; s01 += ga[0] * v
                s10 += ga[1] * u; s11 += ga[1] * v
    if tr == ts:
        d = dd[p, tr]
        s00 += d[0]
        s01 += d[1]
        s10 += d[1]
        s11 += d[2]
    r0_ = 2 * tr
    c0_ = 2 * ts
    S[p, r0_, c0_] = s00
    S[p, r0_, c0_ + 1] = s01
    S[p, r0_ + 1, c0_] = s10
    S[p, r0_ + 1, c0_ + 1] = s11
    S[p, c0_, r0_] = s00
    S[p, c0_ + 1, r0_] = s01
    S[p, c0_, r0_ + 1] = s10
    S[p, c0_ + 1, r0_ + 1] = s11


@wp.kernel
def k_att_rows(
    S: wp.array3d(dtype=wp.float32),
    mx: wp.array2d(dtype=wp.float32),
    e_row: wp.array2d(dtype=wp.float32),
    v_row: wp.array2d(dtype=wp.float32),
):
    """Row r of E = tr(S) + m'm and Var = 2 tr(S^2) + 4 m'S m."""
    p, r = wp.tid()
    n2 = S.shape[1]
    srm = float(0.0)
    vv = float(0.0)
    for s in range(n2):
        x = S[p, r, s]
        vv += 2.0 * x * x
        srm += x * mx[p, s]
    e_row[p, r] = S[p, r, r] + mx[p, r] * mx[p, r]
    v_row[p, r] = vv + 4.0 * mx[p, r] * srm


@wp.kernel
def k_att_finish(
    e_row: wp.array2d(dtype=wp.float32),
    v_row: wp.array2d(dtype=wp.float32),
    n2: int,
    e_l: wp.array(dtype=wp.float32),
    var_l: wp.array(dtype=wp.float32),
):
    p = wp.tid()
    E = float(0.0)
    V = float(0.0)
    for r in range(n2):
        E += e_row[p, r]
        V += v_row[p, r]
    e_l[p] = E
    var_l[p] = V


class WarpAttitudeEstimator:
    """E[L], Var[L] of the attitude cost per plan, device-resident."""

    def __init__(self, ny, nx, cell, origin_x, origin_y, rp, rho1, n_plans, n_t,
                 device="cuda:0", n_bins=None, band_m=None):
        from .contact import WarpContactEstimator
        # reuse the contact estimator's node/table construction wholesale
        self._c = WarpContactEstimator(ny, nx, cell, origin_x, origin_y, rp, rho1,
                                       n_plans, n_t, device=device,
                                       **({} if n_bins is None else {"n_bins": n_bins}))
        self.device = device
        self.n_plans, self.n_t = n_plans, n_t
        n_nodes = self._c.n_nodes
        n_ws = n_plans * n_t * 3
        R = settle_map(rp)[1:3, :].astype(np.float32)   # (pitch, roll) x wheel
        reach = (len(np.asarray(rho1)) * cell)
        # Coarse upper bound on the step lag that can still be in reach, at a nominal
        # 0.10 m of travel per step. It only bounds the pair LIST; what a pair costs is
        # decided by the exact bounding-box cull in `k_att_cov`.
        self.band_t = int(np.ceil((reach + 2 * 0.5) / 0.10)) if band_m is None \
            else int(np.ceil(band_m / 0.10))
        self.band_t = min(self.band_t, n_t - 1)

        r1f = np.asarray(rho1, np.float64).astype(np.float32)
        L = int(len(r1f))

        def _r2(dy, dx):
            """rho1[|dy|] * rho1[|dx|] in float32 -- the same two roundings, in the
            same order, the device did when it looked both up and multiplied them."""
            if abs(dy) >= L or abs(dx) >= L:
                return np.float32(0.0)
            return np.float32(r1f[abs(dy)] * r1f[abs(dx)])

        # per bin: the element's own correlation matrix, the deduplicated candidate
        # count (four corner-shifted copies of the element, unioned) and the widest
        # bounding box that union can have
        u_need = 0
        span = 0
        hw = wheel_half_width(rp)
        rho_e = np.zeros((self._c.n_bins, MAX_K, MAX_K), np.float32)
        for b in range(self._c.n_bins):
            a_, b_, _ = cyl_table(cell, rp.wheel_radius, hw, b, self._c.n_bins)
            for j1 in range(len(a_)):
                for j2 in range(len(a_)):
                    rho_e[b, j1, j2] = _r2(a_[j1] - a_[j2], b_[j1] - b_[j2])
            u_need = max(u_need, len({(int(dy) + cy, int(dx) + cx)
                                      for dy, dx in zip(a_, b_)
                                      for cy in (0, 1) for cx in (0, 1)}))
            span = max(span, int(np.ptp(np.asarray(a_))) + 1,
                       int(np.ptp(np.asarray(b_))) + 1)
        if u_need > MAX_U:
            raise ValueError(
                f"merged candidate count {u_need} exceeds MAX_U={MAX_U} at cell={cell} "
                f"radius={rp.wheel_radius}; raise HELHEST_RISK_MAX_U")
        self.u_max = u_need

        # rho1 outer-producted with itself, zero-padded past its hard cutoff. `half`
        # is the widest lag a wheel pair that survives the bounding-box cull can ask
        # for: the two boxes are within `reach` of each other and each is at most
        # `span` cells wide, so |lag| < reach + 2 * span.
        self.reach = L
        self.half = self.reach + 2 * int(span)
        lag = np.abs(np.arange(-(self.half - 1), self.half))
        col = np.where(lag < self.reach, r1f[np.minimum(lag, self.reach - 1)],
                       np.float32(0.0)).astype(np.float32)
        rho2 = (col[:, None] * col[None, :]).astype(np.float32)   # float32 product,
        # bit-identical to the two float32 lookups multiplied on the device

        # COLUMN-MAJOR pair order: consecutive threads share `ts`, hence share the
        # column-side candidate list that the innermost loop walks -- so that loop's
        # loads are broadcast across the warp rather than 32 separate lines. Measured
        # 8% of the covariance kernel, for a change of host-side loop order.
        pairs = [(tr, ts) for ts in range(n_t)
                 for tr in range(ts, min(n_t, ts + self.band_t + 1))]
        self.n_pair = len(pairs)
        with wp.ScopedDevice(device):
            self.row_w = wp.array(R, dtype=wp.float32)
            self.node_mean = wp.zeros(n_nodes, dtype=wp.float32)
            self.node_dvar = wp.zeros(n_nodes, dtype=wp.float32)
            self.cand_iy = wp.zeros((MAX_K, n_nodes), dtype=wp.int32)
            self.cand_ix = wp.zeros((MAX_K, n_nodes), dtype=wp.int32)
            self.cand_wsig = wp.zeros((MAX_K, n_nodes), dtype=wp.float32)
            self.node_wgt = wp.zeros(n_nodes, dtype=wp.float32)
            self.pk_cnt = wp.zeros(n_ws, dtype=wp.int32)
            self.pk_c = wp.zeros((n_ws, u_need), dtype=wp.int32)
            self.pk_w = wp.zeros((n_ws, u_need), dtype=wp.vec2f)
            self.rho2 = wp.array(rho2.reshape(-1), dtype=wp.float32)
            self.rho_e = wp.array(rho_e, dtype=wp.float32)
            self.pk_box = wp.zeros(n_ws, dtype=VEC4I)
            self.dd = wp.zeros((n_plans, n_t), dtype=wp.vec3f)
            self.pair_tr = wp.array(np.asarray([a for a, _ in pairs], np.int32),
                                    dtype=wp.int32)
            self.pair_ts = wp.array(np.asarray([b for _, b in pairs], np.int32),
                                    dtype=wp.int32)
            # S is written in full on every listed step pair (zero where the cull
            # fires), and entries outside the pair list are exactly zero for every
            # input, so this is the only time it is ever cleared.
            self.S = wp.zeros((n_plans, 2 * n_t, 2 * n_t), dtype=wp.float32)
            self.mx = wp.zeros((n_plans, 2 * n_t), dtype=wp.float32)
            self.e_row = wp.zeros((n_plans, 2 * n_t), dtype=wp.float32)
            self.v_row = wp.zeros((n_plans, 2 * n_t), dtype=wp.float32)
            self.e_l = wp.zeros(n_plans, dtype=wp.float32)
            self.var_l = wp.zeros(n_plans, dtype=wp.float32)
            ones = wp.array(np.ones(3, np.float32), dtype=wp.float32)
            self._ones_settle = ones

    def moments(self, belief, sigma, ctrl):
        c = self._c
        d = self.device
        n_ws = self.n_plans * self.n_t * 3
        # node pass with unit settle weights: node_ceff becomes the bare
        # bilinear corner weight
        wp.launch(k_nodes, dim=c.n_nodes, device=d,
                  inputs=[ctrl, c.wheel_xy, self._ones_settle, c.origin_x, c.origin_y,
                          c.cell, c.ny, c.nx, c.n_bins, c.n_t],
                  outputs=[c.anchor_y, c.anchor_x, c.node_bin, self.node_wgt])
        wp.launch(k_fold_att, dim=c.n_nodes, device=d,
                  inputs=[belief, sigma, c.off_dy, c.off_dx, c.off_cap, c.k_real,
                          self.rho_e, c.anchor_y, c.anchor_x, c.node_bin, c.ny, c.nx],
                  outputs=[self.node_mean, self.node_dvar,
                           self.cand_iy, self.cand_ix, self.cand_wsig])
        wp.launch(k_pack_att, dim=n_ws, device=d,
                  inputs=[self.node_wgt, c.node_bin, c.k_real, self.cand_iy,
                          self.cand_ix, self.cand_wsig, self.row_w, self.n_t],
                  outputs=[self.pk_cnt, self.pk_c, self.pk_w, self.pk_box])
        wp.launch(k_att_diag, dim=self.n_plans * self.n_t, device=d,
                  inputs=[self.node_mean, self.node_dvar, self.node_wgt, self.row_w,
                          self.n_t],
                  outputs=[self.mx, self.dd])
        wp.launch(k_att_cov, dim=self.n_plans * self.n_pair, device=d,
                  inputs=[self.pk_cnt, self.pk_c, self.pk_w, self.pk_box, self.dd,
                          self.rho2, self.reach, self.half,
                          self.pair_tr, self.pair_ts, self.n_t, self.n_pair],
                  outputs=[self.S])
        wp.launch(k_att_rows, dim=(self.n_plans, 2 * self.n_t), device=d,
                  inputs=[self.S, self.mx], outputs=[self.e_row, self.v_row])
        wp.launch(k_att_finish, dim=self.n_plans, device=d,
                  inputs=[self.e_row, self.v_row, 2 * self.n_t],
                  outputs=[self.e_l, self.var_l])
        return self.e_l, self.var_l
