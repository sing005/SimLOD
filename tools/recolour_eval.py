"""Measurements behind tools/recolour_canopy.py, reproducible on any scan or sample.

  python tools/recolour_eval.py data/samples/column_3m.laz
  python tools/recolour_eval.py data/samples/column_3m.laz --output out.laz   # also check a recoloured file
  python tools/recolour_eval.py data/samples/column_3m.laz --blowout 0.15     # end to end, simulated blowout
  git show e246d28:tools/recolour_canopy.py > old.py                           # the previous tool, for comparison
  python tools/recolour_eval.py data/samples/column_3m.laz --blowout 0.15 --tool old.py

Colour differences are CIEDE2000 (Sharma, Wu & Dalal 2005). The report has
these parts:

  0. The scan's colours: hue of well-exposed points, the dark channel by
     height, and whether laser intensity follows visible lightness.
  1. The CIEDE2000 code against the 34 published test pairs.
  2. Borrowing. Well-exposed points are hidden in cubes and their colour is
     predicted from the well-exposed points outside, the way the tool fills a
     blown point. The error is reported against the distance to the nearest
     source, next to two baselines: the scan's median colour, and a palette
     draw (the colour of a random real point within 2 m). The tool's
     borrow_error() curve is fitted from these numbers.
  3. Greyed points. Greyed points lying within 2 point spacings of a
     well-exposed point imaged at another time give (greyed, well-exposed)
     pairs. The regression is fitted on half of them (alternate 0.5 m
     blocks, so no surface is in both halves) and tested on the other half,
     by dark-channel band. The tool's regression_error() table comes from
     these numbers.
  4. With --output: the share of points that are still blown or greyed by the
     tool's own tests, before and after.
  5. With --blowout CUBE: well-exposed points inside random cubes of that
     size are turned white, the tool (or --tool SCRIPT) is run on the result,
     and its colours for those points are scored against the true ones.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import laspy
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recolour_canopy as rc  # noqa: E402

# Sharma, Wu & Dalal (2005) Table 1: L1 a1 b1 L2 a2 b2 dE00
SHARMA = np.array([
    [50.0000, 2.6772, -79.7751, 50.0000, 0.0000, -82.7485, 2.0425],
    [50.0000, 3.1571, -77.2803, 50.0000, 0.0000, -82.7485, 2.8615],
    [50.0000, 2.8361, -74.0200, 50.0000, 0.0000, -82.7485, 3.4412],
    [50.0000, -1.3802, -84.2814, 50.0000, 0.0000, -82.7485, 1.0000],
    [50.0000, -1.1848, -84.8006, 50.0000, 0.0000, -82.7485, 1.0000],
    [50.0000, -0.9009, -85.5211, 50.0000, 0.0000, -82.7485, 1.0000],
    [50.0000, 0.0000, 0.0000, 50.0000, -1.0000, 2.0000, 2.3669],
    [50.0000, -1.0000, 2.0000, 50.0000, 0.0000, 0.0000, 2.3669],
    [50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0009, 7.1792],
    [50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0010, 7.1792],
    [50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0011, 7.2195],
    [50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0012, 7.2195],
    [50.0000, -0.0010, 2.4900, 50.0000, 0.0009, -2.4900, 4.8045],
    [50.0000, -0.0010, 2.4900, 50.0000, 0.0010, -2.4900, 4.8045],
    [50.0000, -0.0010, 2.4900, 50.0000, 0.0011, -2.4900, 4.7461],
    [50.0000, 2.5000, 0.0000, 50.0000, 0.0000, -2.5000, 4.3065],
    [50.0000, 2.5000, 0.0000, 73.0000, 25.0000, -18.0000, 27.1492],
    [50.0000, 2.5000, 0.0000, 61.0000, -5.0000, 29.0000, 22.8977],
    [50.0000, 2.5000, 0.0000, 56.0000, -27.0000, -3.0000, 31.9030],
    [50.0000, 2.5000, 0.0000, 58.0000, 24.0000, 15.0000, 19.4535],
    [50.0000, 2.5000, 0.0000, 50.0000, 3.1736, 0.5854, 1.0000],
    [50.0000, 2.5000, 0.0000, 50.0000, 3.2972, 0.0000, 1.0000],
    [50.0000, 2.5000, 0.0000, 50.0000, 1.8634, 0.5757, 1.0000],
    [50.0000, 2.5000, 0.0000, 50.0000, 3.2592, 0.3350, 1.0000],
    [60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644],
    [63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.2630],
    [61.2901, 3.7196, -5.3901, 61.4292, 2.2480, -4.9620, 1.8731],
    [35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645],
    [22.7233, 20.0904, -46.6940, 23.0331, 14.9730, -42.5619, 2.0373],
    [36.4612, 47.8580, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146],
    [90.8027, -2.0831, 1.4410, 91.1528, -1.6435, 0.0447, 1.4441],
    [90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381],
    [6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 0.6377],
    [2.0776, 0.0795, -1.1350, 0.9033, -0.0636, -0.5514, 0.9082],
])


def load(path, with_intensity=False):
    las = laspy.read(path)
    xyz = np.stack([las.x, las.y, las.z], 1)
    xyz = (xyz - xyz.min(0)).astype(np.float32)
    rgb = np.stack([las.red, las.green, las.blue], 1).astype(np.int32)
    if rgb.max() > 255:
        rgb >>= 8
    t = np.asarray(las.gps_time) if 'gps_time' in las.point_format.dimension_names else None
    if with_intensity:
        return xyz, rgb, t, np.asarray(las.intensity).astype(np.float64)
    return xyz, rgb, t


def part0(xyz, rgb, inten, a):
    blown, dc, w, trusted = rc.colour_state(rgb, a)
    lab = rc.srgb_to_lab(rgb)
    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    m = trusted & (chroma > 5)
    h10, h90 = np.percentile(hue[m], [10, 90])
    brown = np.mean((hue[m] >= 40) & (hue[m] <= 85))
    print(f'0. Well-exposed colours: 80% of hues between {h10:.0f} and {h90:.0f} degrees; {100 * brown:.1f}% brown (40-85 degrees)')
    # remove a white veil equal to the darkest channel (linear light) and look at the hue left
    lin = rc.srgb_to_linear(rgb)
    res = lin - lin.min(1, keepdims=True)
    lab_r = rc.srgb_to_lab(rc.linear_to_srgb(res / np.maximum(res.max(1, keepdims=True), 1e-6) * 0.2))
    hue_r = np.degrees(np.arctan2(lab_r[:, 2], lab_r[:, 1])) % 360
    g = ~blown & (dc >= 0.2)
    print(f'   hue left after removing the white: greyed points (dark channel >= 0.2) median {np.median(hue_r[g]):.0f} degrees, '
          f'well-exposed points median {np.median(hue_r[trusted]):.0f} degrees')
    z = xyz[:, 2]
    zb = np.percentile(z, [0, 25, 50, 75, 90, 100])
    print('   dark channel (linear) of points not blown out, median by height: ' +
          ', '.join(f'{lo:.1f}-{hi:.1f} m {np.median(dc[~blown & (z >= lo) & (z <= hi)]):.3f}' for lo, hi in zip(zb[:-1], zb[1:])))
    if inten is not None and inten.max() > inten.min():
        ti = np.flatnonzero(trusted & (inten > 0))
        rng = np.random.default_rng(0)
        q = rng.choice(ti, min(20_000, len(ti)), replace=False)
        nbrs = cKDTree(xyz[ti]).query_ball_point(xyz[q], r=0.10, workers=-1)
        cs = []
        for nb in nbrs:
            if len(nb) < 30:
                continue
            nb = np.asarray(nb[:2000])
            A = inten[ti[nb]]; B = lab[ti[nb], 0]
            if A.std() > 0 and B.std() > 0:
                cs.append(np.corrcoef(A, B)[0, 1])
        if cs:
            print(f'   laser intensity against lightness L* within 10 cm: median correlation {np.median(cs):.3f} ({len(cs):,} neighbourhoods)')


def part5(path, cube, tool, a, seed=7):
    las = laspy.read(path)
    xyz = np.stack([las.x, las.y, las.z], 1)
    rgb = np.stack([las.red, las.green, las.blue], 1).astype(np.int32)
    s16 = rgb.max() > 255
    if s16:
        rgb >>= 8
    _, _, _, trusted = rc.colour_state(rgb, a)
    rng = np.random.default_rng(seed)
    z = xyz[:, 2]
    zlo, zhi = np.percentile(z[trusted], [2, 60])
    ti = np.flatnonzero(trusted & (z >= zlo) & (z < zhi))
    n_cubes = max(20, int(round(300 * (0.15 / cube) ** 2)))
    cen = xyz[rng.choice(ti, min(n_cubes, len(ti)), replace=False)]
    d, _ = cKDTree(cen).query(xyz, k=1, p=np.inf, workers=-1)
    hid = np.flatnonzero((d <= cube / 2) & trusted)
    white = 245 * (257 if s16 else 1)
    las.red[hid] = white; las.green[hid] = white; las.blue[hid] = white
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, 'blowout.las'); out = os.path.join(tmp, 'out.las')
        las.write(src)
        subprocess.run([sys.executable, tool, src, out], check=True, stdout=subprocess.DEVNULL)
        o = laspy.read(out)
    xo = np.stack([o.x, o.y, o.z], 1)
    dd, idx = cKDTree(xo).query(xyz[hid], k=1, workers=-1)
    co = np.stack([o.red, o.green, o.blue], 1).astype(np.int32)[idx]
    if s16:
        co >>= 8
    de = rc.ciede2000(rc.srgb_to_lab(co), rc.srgb_to_lab(rgb[hid]))
    print(f'5. Simulated blowout, {cube} m cubes, {len(hid):,} well-exposed points turned white, recoloured by '
          f'{os.path.basename(tool)}: median error {np.median(de):.2f}, {100 * np.mean(de < 5):.1f}% under 5')


def part1():
    got = rc.ciede2000(SHARMA[:, 0:3], SHARMA[:, 3:6])
    err = np.abs(got - SHARMA[:, 6]).max()
    print(f'1. CIEDE2000 against Sharma, Wu & Dalal (2005), {len(SHARMA)} pairs: largest difference {err:.5f} '
          f'({"pass" if err < 1e-3 else "FAIL"})')


def part2(xyz, rgb, a, seed=1):
    blown, dc, w, trusted = rc.colour_state(rgb, a)
    lab = rc.srgb_to_lab(rgb)
    rng = np.random.default_rng(seed)
    ti = np.flatnonzero(trusted)
    # noise floor: each well-exposed point against its 8 nearest well-exposed neighbours
    q = rng.choice(ti, min(50_000, len(ti)), replace=False)
    d, ii = cKDTree(xyz[ti]).query(xyz[q], k=9, workers=-1)
    nf = rc.ciede2000(np.median(lab[ti[ii[:, 1:]]], 1), lab[q])
    print(f'2. Borrowing. Noise floor (a well-exposed point against its 8 nearest, {np.median(d[:, 8]) * 100:.1f} cm): '
          f'median {np.median(nf):.2f}')
    zlo, zhi = np.percentile(xyz[ti, 2], [2, 60])
    gmed = np.median(lab[ti], 0)
    rows = []
    print('   cube   n hidden   nearest source   borrowed   palette draw   scan median')
    for B in (0.05, 0.15, 0.4, 1.0):
        cand = ti[(xyz[ti, 2] >= zlo) & (xyz[ti, 2] < zhi)]
        cen = xyz[rng.choice(cand, 400 if B < 0.5 else 120, replace=False)]
        dd, _ = cKDTree(cen).query(xyz, k=1, p=np.inf, workers=-1)
        hidden = dd <= B / 2
        ev = np.flatnonzero(hidden & trusted)
        if len(ev) > 200_000:
            ev = rng.choice(ev, 200_000, replace=False)
        src = np.flatnonzero(trusted & ~hidden)
        pred, ok, dn = rc.gaussian_borrow(xyz[src], lab[src], None, xyz[ev], max(2.0, 2 * B), a.k, 1, a.batch)
        # the tool borrows with --fill-radius; beyond it, a Gaussian of radius 2 B shows what borrowing would give
        de_b = rc.ciede2000(pred, lab[ev])
        # palette draw: a random well-exposed point within 2 m of the hidden point, outside the cube
        dpal, ipal = cKDTree(xyz[src]).query(xyz[ev], k=64, distance_upper_bound=2.0, workers=-1)
        n_ok = np.isfinite(dpal).sum(1)
        j = np.minimum((rng.random(len(ev)) * n_ok).astype(int), np.maximum(n_ok - 1, 0))
        palc = lab[src[ipal[np.arange(len(ev)), j]]]
        de_p = rc.ciede2000(palc[n_ok > 0], lab[ev][n_ok > 0])
        de_g = rc.ciede2000(np.broadcast_to(gmed, (len(ev), 3)), lab[ev])
        print(f'   {B:4.2f} m {len(ev):9,d}   {np.median(dn) * 100:8.1f} cm     {np.median(de_b):6.2f}      {np.median(de_p):6.2f}         {np.median(de_g):6.2f}')
        rows.append((dn, de_b))
    # fit borrow_error(d) = A + B sqrt(d) on points whose nearest source is within 0.3 m
    dn = np.concatenate([r[0] for r in rows]); de = np.concatenate([r[1] for r in rows])
    edges = np.array([0, 0.005, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3])
    bc, bm = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dn >= lo) & (dn < hi)
        if m.sum() >= 200:
            bc.append(np.median(dn[m])); bm.append(np.median(de[m]))
    bc = np.array(bc); bm = np.array(bm)
    A, Bc = np.linalg.lstsq(np.stack([np.ones_like(bc), np.sqrt(bc)], 1), bm, rcond=None)[0]
    print('   median error by nearest-source distance: ' + ', '.join(f'{c * 100:.1f} cm {m:.2f}' for c, m in zip(bc, bm)))
    print(f'   fit: borrow_error(d) = {A:.2f} + {Bc:.2f} sqrt(d)   (the tool uses {rc._BORROW_A} + {rc._BORROW_B} sqrt(d))')


def part3(xyz, rgb, t, a):
    X, Y = rc.collect_pairs(xyz, rgb, t, a, cap=10 ** 9)
    if len(X) < 1000:
        print(f'3. Greyed points: only {len(X)} pairs; skipped')
        return
    # recover the pairs' positions for a spatial split: collect_pairs does not return them, so redo its selection
    blown, dc, w, trusted = rc.colour_state(rgb, a)
    p_idx = np.flatnonzero(~blown & (w > 0)); t_idx = np.flatnonzero(trusted)
    r = float(np.clip(2 * rc.point_spacing(xyz), 0.01, 0.03))
    d, ii = cKDTree(xyz[t_idx]).query(xyz[p_idx], k=8, distance_upper_bound=r, workers=-1)
    ok = np.isfinite(d); ii = np.where(ok, ii, 0)
    if t is not None:
        ok &= np.abs(t[t_idx[ii]] - t[p_idx][:, None]) > 1.0
    P = p_idx[ok.any(1)]
    assert len(P) == len(X)
    blk = (np.floor(xyz[P] / 0.5).astype(int).sum(1)) % 2
    tr, te = blk == 0, blk == 1
    W = rc.fit_pale_model(X[tr], Y[tr])
    pred = X[te] @ W
    lab_p = rc.srgb_to_lab(rgb[P[te]])
    de_r = rc.ciede2000(pred, Y[te]); de_0 = rc.ciede2000(lab_p, Y[te])
    de_c = rc.ciede2000(np.broadcast_to(np.median(Y[tr], 0), Y[te].shape), Y[te])
    print(f'3. Greyed points: {len(X):,} pairs within {r * 100:.1f} cm. Held out {te.sum():,}: median error unchanged {np.median(de_0):.2f}, '
          f'one colour for all {np.median(de_c):.2f}, regression {np.median(de_r):.2f}')
    dcp = dc[P[te]]
    print('   dark channel band   n        unchanged   regression')
    edges = [0.0, 0.10, 0.14, 0.18, 0.25, 0.32, 1.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dcp >= lo) & (dcp < hi)
        if m.sum() >= 200:
            print(f'   {lo:.2f}-{hi:.2f} (median {np.median(dcp[m]):.3f}) {m.sum():8,d}   {np.median(de_0[m]):6.2f}      {np.median(de_r[m]):6.2f}')


def part4(xyz, rgb, out_path, a):
    xo, ro, _ = load(out_path)
    # match points: the output is written tile by tile
    d, idx = cKDTree(xo).query(xyz, k=1, workers=-1)
    if d.max() > 1e-4:
        print('4. --output does not hold the same points; skipped')
        return
    b0, _, w0, _ = rc.colour_state(rgb, a)
    b1, _, w1, _ = rc.colour_state(ro[idx], a)
    print(f'4. Before: {100 * b0.mean():.2f}% blown, {100 * (~b0 & (w0 > 0.5)).mean():.2f}% greyed (weight > 0.5). '
          f'After: {100 * b1.mean():.2f}% blown, {100 * (~b1 & (w1 > 0.5)).mean():.2f}% greyed.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('--output', help='a file written by recolour_canopy.py from INPUT, to compare')
    ap.add_argument('--blowout', type=float, metavar='CUBE', help='also run the simulated blowout with cubes of this size, m')
    ap.add_argument('--tool', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recolour_canopy.py'),
                    help='the recolour script --blowout runs (default: this version)')
    e = ap.parse_args()
    a = rc.build_parser().parse_args(['in', 'out'])
    xyz, rgb, t, inten = load(e.input, with_intensity=True)
    part0(xyz, rgb, inten, a)
    part1()
    part2(xyz, rgb, a)
    part3(xyz, rgb, t, a)
    if e.output:
        part4(xyz, rgb, e.output, a)
    if e.blowout:
        part5(e.input, e.blowout, e.tool, a)


if __name__ == '__main__':
    main()
