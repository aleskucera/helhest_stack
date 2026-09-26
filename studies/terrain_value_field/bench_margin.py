"""Is `margin_to_z_kernel` laid out well, and would anything faster help?

Three questions, measured rather than argued:
  1. is the access pattern coalesced, or is the constraint-outermost layout hurting?
  2. what fraction of peak bandwidth does it reach -- i.e. is it bandwidth-bound at all?
  3. does fusing it with `classify_kernel` beat two passes?
"""

from __future__ import annotations

import time

import numpy as np
import warp as wp

from helhest.planning.terrain_value_field.margin import classify_kernel
from helhest.planning.terrain_value_field.margin import IGNORED
from helhest.planning.terrain_value_field.margin import margin_to_fields_kernel as margin_fused
from helhest.planning.terrain_value_field.margin import margin_to_z_kernel


@wp.kernel
def margin_to_z_transposed(
    margin: wp.array4d(dtype=wp.float32),  # [row, col, heading, constraint] -- constraint FASTEST
    sigma: wp.array4d(dtype=wp.float32),
    floor: wp.array(dtype=wp.float32),
    z: wp.array3d(dtype=wp.float32),
    z_certain: wp.array3d(dtype=wp.float32),
):
    """The obvious alternative: put the loop's own axis last so a thread reads contiguously.

    It trades one thing for another -- a thread's loop becomes sequential in memory, but the
    threads of a warp become strided by n_constraints instead of adjacent, which is the axis
    that actually governs coalescing.
    """
    r, c, t = wp.tid()
    best = float(IGNORED)
    best_certain = float(IGNORED)
    for i in range(margin.shape[3]):
        m = margin[r, c, t, i]
        if m >= IGNORED:
            continue
        f = floor[i]
        s = wp.max(sigma[r, c, t, i], f)
        best = wp.min(best, m / s)
        best_certain = wp.min(best_certain, m / f)
    z[r, c, t] = best
    z_certain[r, c, t] = best_certain


def timeit(fn, reps=60):
    fn()
    wp.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        wp.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def run(rows, cols, headings, n_con):
    shape4 = (n_con, rows, cols, headings)
    shape3 = (rows, cols, headings)
    rng = np.random.default_rng(0)
    m = wp.array(rng.uniform(-0.2, 0.5, shape4).astype(np.float32), dtype=wp.float32)
    s = wp.array(rng.uniform(0.01, 0.1, shape4).astype(np.float32), dtype=wp.float32)
    mt = wp.array(np.ascontiguousarray(m.numpy().transpose(1, 2, 3, 0)), dtype=wp.float32)
    st = wp.array(np.ascontiguousarray(s.numpy().transpose(1, 2, 3, 0)), dtype=wp.float32)
    f = wp.array(np.full(n_con, 0.02, np.float32), dtype=wp.float32)
    k = wp.array(np.array([2.0], np.float32), dtype=wp.float32)
    z, zc = wp.zeros(shape3, dtype=wp.float32), wp.zeros(shape3, dtype=wp.float32)
    bl, pen, db = (wp.zeros(shape3, dtype=wp.float32) for _ in range(3))

    def two_pass():
        wp.launch(margin_to_z_kernel, dim=shape3, inputs=[m, s, f], outputs=[z, zc])
        wp.launch(classify_kernel, dim=shape3, inputs=[z, zc, k, 4.0, 1.0], outputs=[bl, pen, db])

    def transposed():
        wp.launch(margin_to_z_transposed, dim=shape3, inputs=[mt, st, f], outputs=[z, zc])

    def fused():
        wp.launch(
            margin_fused, dim=shape3, inputs=[m, s, f, k, 4.0, 1.0], outputs=[z, zc, bl, pen, db]
        )

    def reduce_only():
        wp.launch(margin_to_z_kernel, dim=shape3, inputs=[m, s, f], outputs=[z, zc])

    n = rows * cols * headings
    # reduce-only traffic: read 2 arrays x n_con, write 2.
    bytes_reduce = 4 * n * (2 * n_con + 2)
    # two-pass adds: classify re-reads z, z_certain and writes 3.
    bytes_two = bytes_reduce + 4 * n * (2 + 3)
    bytes_fused = 4 * n * (2 * n_con + 5)

    t_red, t_two, t_tr, t_fu = (timeit(x) for x in (reduce_only, transposed, two_pass, fused))
    gb = lambda b, t: b / t / 1e9  # noqa: E731
    print(f"{n_con:2d} con  {rows}x{cols}x{headings} = {n/1e6:5.2f} M states")
    print(f"   reduce only   {t_red*1e3:7.3f} ms   {gb(bytes_reduce, t_red):6.1f} GB/s")
    print(
        f"   transposed    {t_tr*1e3:7.3f} ms   {gb(bytes_reduce, t_tr):6.1f} GB/s"
        f"   ({t_tr/t_red:.2f}x vs reduce)"
    )
    print(f"   two-pass      {t_two*1e3:7.3f} ms   {gb(bytes_two, t_two):6.1f} GB/s")
    print(
        f"   FUSED         {t_fu*1e3:7.3f} ms   {gb(bytes_fused, t_fu):6.1f} GB/s"
        f"   ({t_two/t_fu:.2f}x faster than two-pass)"
    )


if __name__ == "__main__":
    wp.init()
    dev = wp.get_device()
    print(f"device {dev}\n")
    # The deployed shape (16 m window, 0.24 m routing cell, 24 headings) and a bigger one.
    for rows, cols, headings, n_con in [
        (67, 67, 24, 2),
        (67, 67, 24, 5),
        (200, 200, 24, 2),
        (200, 200, 24, 5),
        (400, 400, 16, 2),
    ]:
        run(rows, cols, headings, n_con)
        print()


@wp.kernel
def _stream_copy(src: wp.array3d(dtype=wp.float32), dst: wp.array3d(dtype=wp.float32)):
    r, c, t = wp.tid()
    dst[r, c, t] = src[r, c, t]


def audit(rows=400, cols=400, headings=16, n_con=2):
    """Time each pass alone, and calibrate against a pure copy to get the real peak."""
    shape4, shape3 = (n_con, rows, cols, headings), (rows, cols, headings)
    rng = np.random.default_rng(0)
    m = wp.array(rng.uniform(-0.2, 0.5, shape4).astype(np.float32), dtype=wp.float32)
    s = wp.array(rng.uniform(0.01, 0.1, shape4).astype(np.float32), dtype=wp.float32)
    f = wp.array(np.full(n_con, 0.02, np.float32), dtype=wp.float32)
    k = wp.array(np.array([2.0], np.float32), dtype=wp.float32)
    z, zc = wp.zeros(shape3, dtype=wp.float32), wp.zeros(shape3, dtype=wp.float32)
    bl, pen, db = (wp.zeros(shape3, dtype=wp.float32) for _ in range(3))
    n = rows * cols * headings

    t_copy = timeit(lambda: wp.launch(_stream_copy, dim=shape3, inputs=[z], outputs=[zc]))
    peak = 4 * n * 2 / t_copy / 1e9
    t_red = timeit(
        lambda: wp.launch(margin_to_z_kernel, dim=shape3, inputs=[m, s, f], outputs=[z, zc])
    )
    t_cls = timeit(
        lambda: wp.launch(
            classify_kernel, dim=shape3, inputs=[z, zc, k, 4.0, 1.0], outputs=[bl, pen, db]
        )
    )

    def two():
        wp.launch(margin_to_z_kernel, dim=shape3, inputs=[m, s, f], outputs=[z, zc])
        wp.launch(classify_kernel, dim=shape3, inputs=[z, zc, k, 4.0, 1.0], outputs=[bl, pen, db])

    t_two = timeit(two)
    t_fu = timeit(
        lambda: wp.launch(
            margin_fused, dim=shape3, inputs=[m, s, f, k, 4.0, 1.0], outputs=[z, zc, bl, pen, db]
        )
    )
    print(f"AUDIT {rows}x{cols}x{headings}, {n_con} constraints, {n/1e6:.2f} M states")
    print(f"  pure copy (2 arrays)   {t_copy*1e3:7.3f} ms  -> measured peak ~{peak:.0f} GB/s")
    print(f"  reduce alone           {t_red*1e3:7.3f} ms   {4*n*(2*n_con+2)/t_red/1e9:6.1f} GB/s")
    print(f"  classify alone         {t_cls*1e3:7.3f} ms   {4*n*5/t_cls/1e9:6.1f} GB/s")
    print(f"  reduce + classify      {t_red*1e3+t_cls*1e3:7.3f} ms  (sum of the two above)")
    print(f"  two-pass measured      {t_two*1e3:7.3f} ms")
    print(f"  fused measured         {t_fu*1e3:7.3f} ms")


if __name__ == "__main__":
    audit()
