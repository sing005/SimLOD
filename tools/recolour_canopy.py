"""Repair the blown-out and greyed canopy colour of a FARO scan, offline.

The scanner colours points from its panorama camera. In the sunlit canopy the
camera saturates, so leaves and branches there come out white, pale grey or
with a lilac/blue sky cast, and on the way up from the floor the colour fades
through pale grey-green. Positions and laser intensity are fine; only the
colour is lost or washed out. This rewrites the colour of those points and
leaves every other point, and every other attribute, untouched.

Each 16 m ground tile is processed with a halo of its neighbours' points:

  1. Detect. Blown out: white, or a sky cast (the thresholds below). Greyed:
     the darkest channel, in linear light, is raised. Well-exposed foliage and
     bark keep one channel dark, so a raised darkest channel measures how much
     white has been added to the colour (the dark channel of He, Sun & Tang
     2011). Points bright enough to judge but with almost no saturation are
     greyed too. The greyed correction is weighted from 0 to 1 across
     --pale-lo..--pale-hi and --grey-sat-hi..--grey-sat-lo, so there is no
     visible edge where it starts.
  2. Protect survey pages. A printed checkerboard page is flat, about A4, and
     achromatic: white paper with dark print. The print also returns little
     laser light, so the page shows a strong intensity contrast (Lichti et
     al. 2019). Its points keep their colour and are never used as a colour
     source. Page centres are written to OUT_pages.csv in the viewer's
     markers.csv format.
  3. Repair greyed points. A greyed point still holds much of its colour, so
     it is mapped back by a regression learnt from this scan: greyed points
     lying within a centimetre or two of a well-exposed point of the same
     surface, imaged in another photo, give pairs of (greyed, well-exposed)
     colour. The regression's estimate is averaged with the colour of
     well-exposed points nearby, each weighted by its measured error.
  4. Classify structure where there are blown points. A line of points is a
     trunk or branch; a less linear cluster is leaf material. Segments of
     points whose normals have similar verticality are judged by their own
     linearity and size over a range of thresholds (after LeWoS, Wang et al.
     2020); thin lines are found by multi-scale linearity (Weinmann et al.
     2014). The probability is averaged over neighbours, so labels come in
     connected pieces rather than single points.
  5. Fill blown points from usable colour within --fill-radius: well-exposed
     and repaired points, the same structure class first; a second pass
     reaches one radius further. Measured on the sample, borrowed colour gets
     less accurate with distance: at 24 cm from the nearest source its error
     is 5.8 against 7.0 for the scan's median colour, at 77 cm it is 7.7. So
     filling stops at about 30 cm.
  6. Palette. Remaining points take the colour of a real point of the same
     class nearby, one colour per cellular patch of about --patch metres
     (Worley 1996), so a branch or a cluster of leaves gets one consistent
     colour instead of a speckle of unrelated ones.

tools/recolour_eval.py reproduces the measurements behind these choices.

References
  He, Sun & Tang (2011) IEEE TPAMI 33(12): 2341-2353 (dark channel).
  Lichti, Glennie, Jahraus & Hartzell (2019) J. Surv. Eng. 145(3): 04019008 (target contrast).
  Weinmann, Jutzi & Mallet (2014) ISPRS Annals II-3: 181-188 (eigenvalue features).
  Wang, Momo Takoudjou & Casella (2020) Methods Ecol. Evol. 11(3): 376-389 (LeWoS).
  Worley (1996) SIGGRAPH '96: 291-294 (cellular texture).
  Sharma, Wu & Dalal (2005) Color Res. Appl. 30(1): 21-30 (CIEDE2000).

Usage:
  python recolour_canopy.py IN.laz OUT.laz                      # whole scan
  python recolour_canopy.py IN.laz OUT.laz --crop -10 -10 10 10 # test tile
  python recolour_canopy.py IN.laz OUT.laz --show classes       # structure classes as false colour
  python recolour_canopy.py IN.laz OUT.laz --workers 4          # tiles in parallel (memory x4)
"""
import argparse
import concurrent.futures
import multiprocessing
import os
import shutil
import tempfile
import time

import laspy
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


# ---------------------------------------------------------------- colour
# sRGB (IEC 61966-2-1) and CIELAB under D65. Colour differences are judged in
# CIELAB with CIEDE2000.

_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])
_E, _K = 216 / 24389, 24389 / 27


def srgb_to_linear(c8):
    c = np.asarray(c8, np.float32) / np.float32(255)
    return np.where(c <= 0.04045, c / np.float32(12.92), ((c + np.float32(0.055)) / np.float32(1.055)) ** np.float32(2.4))


def linear_to_srgb(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055) * 255


def srgb_to_lab(c8, batch=4_000_000):
    c8 = np.asarray(c8)
    out = np.empty((len(c8), 3), np.float32)
    A = (_M.T / _WHITE[None, :]).astype(np.float32)
    for b0 in range(0, len(c8), batch):
        xyz = srgb_to_linear(c8[b0:b0 + batch]) @ A
        f = np.where(xyz > _E, np.cbrt(xyz), (_K * xyz + 16) / 116)
        out[b0:b0 + batch, 0] = 116 * f[:, 1] - 16
        out[b0:b0 + batch, 1] = 500 * (f[:, 0] - f[:, 1])
        out[b0:b0 + batch, 2] = 200 * (f[:, 1] - f[:, 2])
    return out


def lab_to_srgb(lab, batch=4_000_000):
    lab = np.asarray(lab, np.float32)
    out = np.empty((len(lab), 3), np.float32)
    Minv = np.linalg.inv(_M).T.astype(np.float32)
    W = _WHITE.astype(np.float32)
    for b0 in range(0, len(lab), batch):
        L, a, b = lab[b0:b0 + batch, 0], lab[b0:b0 + batch, 1], lab[b0:b0 + batch, 2]
        fy = (L + 16) / 116; fx = fy + a / 500; fz = fy - b / 200
        x = np.where(fx ** 3 > _E, fx ** 3, (116 * fx - 16) / _K)
        y = np.where(L > _K * _E, fy ** 3, L / _K)
        z = np.where(fz ** 3 > _E, fz ** 3, (116 * fz - 16) / _K)
        out[b0:b0 + batch] = linear_to_srgb((np.stack([x, y, z], 1) * W) @ Minv)
    return out


def ciede2000(lab1, lab2):
    """CIEDE2000 colour difference, as written out by Sharma, Wu & Dalal (2005)."""
    L1, a1, b1 = (np.asarray(lab1, np.float64)[..., i] for i in range(3))
    L2, a2, b2 = (np.asarray(lab2, np.float64)[..., i] for i in range(3))
    Cb = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7)))
    a1p = (1 + G) * a1; a2p = (1 + G) * a2
    C1p = np.hypot(a1p, b1); C2p = np.hypot(a2p, b2)
    h1p = np.where((a1p == 0) & (b1 == 0), 0.0, np.degrees(np.arctan2(b1, a1p)) % 360)
    h2p = np.where((a2p == 0) & (b2 == 0), 0.0, np.degrees(np.arctan2(b2, a2p)) % 360)
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0.0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh) / 2)
    Lbp = (L1 + L2) / 2; Cbp = (C1p + C2p) / 2; hs = h1p + h2p
    hbp = np.where(C1p * C2p == 0, hs,
          np.where(np.abs(h1p - h2p) <= 180, hs / 2, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    RC = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    SL = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    SC = 1 + 0.045 * Cbp
    SH = 1 + 0.015 * Cbp * T
    RT = -np.sin(np.radians(2 * 30 * np.exp(-(((hbp - 275) / 25) ** 2)))) * RC
    dL = (L2 - L1) / SL; dC = (C2p - C1p) / SC; dH = dHp / SH
    return np.sqrt(dL ** 2 + dC ** 2 + dH ** 2 + RT * dC * dH)


# ---------------------------------------------------------------- detection

def blown_mask(r, g, b, a):
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    white = (mn >= a.white_level) & ((mx - mn) <= a.white_spread)
    cast = (b.astype(np.int32) > g.astype(np.int32) + a.sky_cast) & (mx >= 40) if a.sky_cast > 0 else np.zeros_like(white)
    return white | cast


def smoothstep(lo, hi, x):
    t = np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def colour_state(rgb, a):
    """Blown-out points, the dark channel, the greyed weight, and the points usable as colour sources.

    A point is greyed when its darkest channel is raised (pale), or when it is
    bright enough to judge and has almost no saturation (dull grey). Near the
    well-exposed forest floor of the sample scan only 0.2% of points are that
    unsaturated; in the washed-out zone 2-6 m up, 7-8% are."""
    blown = blown_mask(rgb[:, 0], rgb[:, 1], rgb[:, 2], a)
    mx = rgb.max(1)
    dc = srgb_to_linear(rgb.min(1))           # min and max commute with the monotone sRGB curve
    sat = 1 - dc / np.maximum(srgb_to_linear(mx), 1e-6)
    w = np.maximum(smoothstep(a.pale_lo, a.pale_hi, dc),
                   (1 - smoothstep(a.grey_sat_lo, a.grey_sat_hi, sat)) * (mx >= a.min_source)).astype(np.float32)
    w[blown] = 1
    trusted = ~blown & (w == 0) & (mx >= a.min_source)
    return blown, dc, w, trusted


# ---------------------------------------------------------------- geometry helpers

def cell_keys(xyz, s):
    q = np.floor(xyz / s).astype(np.int64)
    q -= q.min(0)
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


def voxelise(xyz, s, weight=None):
    """Centroids of the occupied voxels of size s, and each point's voxel (points may carry weights)."""
    uk, inv = np.unique(cell_keys(xyz, s), return_inverse=True)
    inv = inv.ravel()
    wt = np.ones(len(xyz)) if weight is None else weight
    cnt = np.bincount(inv, wt, len(uk))
    cen = np.stack([np.bincount(inv, xyz[:, j] * wt, len(uk)) for j in range(3)], 1) / cnt[:, None]
    return cen, inv


def point_spacing(xyz, cell=0.5, n_cells=40, seed=0):
    """Median distance to the 4 nearest neighbours, sampled in random occupied cells.

    Linking distances and the base voxel follow it, so the same settings work
    on the full scan and on thinned copies."""
    if len(xyz) < 10:
        return 0.02
    rng = np.random.default_rng(seed)
    key = cell_keys(xyz, cell)
    sel = np.isin(key, key[rng.choice(len(xyz), min(n_cells, len(xyz)), replace=False)])
    P = xyz[sel]
    if len(P) < 10:
        return 0.02
    q = P[rng.choice(len(P), min(20_000, len(P)), replace=False)]
    d, _ = cKDTree(P).query(q, k=5, workers=-1)
    return float(np.median(d[:, 1:]))


def knn_graph_components(cen, link, k=8):
    """Connected pieces of points joined to any of their k nearest neighbours within link."""
    n = len(cen)
    if n < 2:
        return n, np.zeros(n, np.int32)
    d, idx = cKDTree(cen).query(cen, k=min(k, n), distance_upper_bound=link, workers=-1)
    ok = np.isfinite(d) & (idx < n)
    src = np.repeat(np.arange(n), d.shape[1])[ok.ravel()]
    dst = idx.ravel()[ok.ravel()]
    g = coo_matrix((np.ones(len(src), np.int8), (src, dst)), shape=(n, n))
    return connected_components(g, directed=False)


def dilated_cells(xyz, mask, cell):
    """Points in the cells that hold a masked point, or touch one."""
    q = np.floor(xyz / cell).astype(np.int64)
    q -= q.min(0) - 1
    key = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
    occ = np.unique(q[mask], axis=0)
    off = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    nb = (occ[:, None, :] + off[None, :, :]).reshape(-1, 3)
    nkey = np.unique((nb[:, 0] << 42) | (nb[:, 1] << 21) | nb[:, 2])
    return np.isin(key, nkey)


# ---------------------------------------------------------------- survey pages

def find_pages(xyz, rgb, sp, a, inten=None):
    """Flat, A4-sized patches of achromatic points with both near-white and near-black parts.

    Achromatic follows the viewer's "Find targets" idea: a checkerboard is
    black, grey and white, so its channels are nearly equal, while foliage is
    green and bark green-brown. Here the allowed spread grows with brightness
    (--page-spread of the brightest channel, at least 8): with a fixed spread,
    dull dark olive passes too, and a page joins up with the trunk or post it
    is fixed to."""
    protected = np.zeros(len(xyz), bool)
    pages = []
    mx = rgb.max(1); mn = rgb.min(1)
    idx = np.flatnonzero((mx - mn) <= np.maximum(8, a.page_spread * mx))
    if len(idx) < a.page_min_points:
        return protected, pages
    link = float(np.clip(3 * sp, 0.01, 0.03))
    cen, inv = voxelise(xyz[idx], link / 2)
    nc, lab = knn_graph_components(cen, link)
    plab = lab[inv]
    cnt = np.bincount(plab, minlength=nc)
    lo = np.full((nc, 3), np.inf); hi = np.full((nc, 3), -np.inf)
    P = xyz[idx]
    np.minimum.at(lo, plab, P); np.maximum.at(hi, plab, P)
    diag = np.linalg.norm(hi - lo, axis=1)
    cand = np.flatnonzero((cnt >= a.page_min_points) & (diag >= 0.12) & (diag <= 0.6))
    if len(cand) == 0:
        return protected, pages
    order = np.argsort(plab, kind='stable')
    starts = np.searchsorted(plab[order], cand); ends = np.searchsorted(plab[order], cand, side='right')
    tree = None
    for b0, b1 in zip(starts, ends):
        # the achromatic piece (mostly the white paper) gives the page's plane and outline
        Q = xyz[idx[order[b0:b1]]].astype(np.float64)
        ctr = Q.mean(0)
        w, v = np.linalg.eigh(np.cov((Q - ctr).T))
        if np.sqrt(max(w[0], 0.0)) > a.page_thickness:
            continue
        u0, u1 = np.percentile((Q - ctr) @ v[:, 2], [2, 98]); s0, s1 = np.percentile((Q - ctr) @ v[:, 1], [2, 98])
        l1 = u1 - u0; l2 = s1 - s0
        # A4 is 29.7 x 21 cm; allow a page that is partly hidden, but not a long thin strip
        if not (0.18 <= l1 <= 0.36 and 0.12 <= l2 <= 0.26 and l1 / l2 <= 1.8):
            continue
        # The rest is measured on every point inside that outline and within 1.5 cm of the
        # plane, so printed squares count even when a colour cast stops them being achromatic.
        if tree is None:
            tree = cKDTree(xyz)
        near = np.asarray(tree.query_ball_point(ctr, 0.25 * np.sqrt(3)), np.int64)
        near = near[np.all(np.abs(xyz[near] - ctr) <= 0.25, 1)]
        R = xyz[near].astype(np.float64) - ctr
        u = R @ v[:, 2]; s = R @ v[:, 1]; dn = np.abs(R @ v[:, 0])
        on = (dn <= 0.015) & (u >= u0) & (u <= u1) & (s >= s0) & (s <= s1)
        pp = near[on]
        if len(pp) < a.page_min_points:
            continue
        # a page is a filled rectangle; a cluster of leaves is sparse and ragged inside its outline
        nu = max(1, int(np.ceil(l1 / 0.02))); ns = max(1, int(np.ceil(l2 / 0.02)))
        iu = np.minimum(((u[on] - u0) / 0.02).astype(int), nu - 1); js = np.minimum(((s[on] - s0) / 0.02).astype(int), ns - 1)
        cover = len(np.unique(iu * ns + js)) / (nu * ns)
        if cover < a.page_cover:
            continue
        # A 15 cm checkerboard printed on A4 is about 18% black; the rest is white paper.
        # Dark and light are judged against the page's own paper, so a page that is
        # overexposed (black washed to dark grey) or in shade still qualifies.
        paper = float(np.percentile(mx[pp], 90))
        if paper < a.page_paper:
            continue
        is_dark = mx[pp] <= 0.5 * paper; is_paper = mn[pp] >= 0.8 * paper
        dark = float(is_dark.mean()); bright = float(is_paper.mean())
        if not (0.10 <= dark <= 0.6 and bright >= 0.3):
            continue
        # Black print returns little laser light and white paper a lot, so a printed page shows a
        # strong intensity contrast (Lichti et al. 2019): about 0.95 on the two pages in the sample
        # scan, below zero for sunflecks on leaf litter. Skipped if the scan has no intensity.
        contrast = np.nan
        if inten is not None and inten[pp].max() > inten[pp].min():
            ip = np.median(inten[pp][is_paper]); idk = np.median(inten[pp][is_dark])
            contrast = float((ip - idk) / max(ip + idk, 1e-9))
            if contrast < a.page_contrast:
                continue
        # keep every point of the page: its outline grown by 3 cm, 1.5 cm either side of the plane
        grow = (dn <= 0.015) & (u >= u0 - 0.03) & (u <= u1 + 0.03) & (s >= s0 - 0.03) & (s <= s1 + 0.03)
        protected[near[grow]] = True
        centre = ctr + v[:, 2] * (u0 + u1) / 2 + v[:, 1] * (s0 + s1) / 2
        pages.append(dict(centre=centre, long=l1, short=l2, normal=v[:, 0], points=int(grow.sum()),
                          dark=dark, bright=bright, cover=cover, contrast=contrast))
    return protected, pages


# ---------------------------------------------------------------- structure

SCALES = (0.04, 0.08, 0.16, 0.32)   # voxel sizes for the thin-line test


def eig_knn(cen, k, batch=300_000):
    """Eigen-decomposition of each centroid's k-NN covariance: linearity, |z| of the normal, k-NN radius, neighbours."""
    n = len(cen); k = min(k, n)
    tree = cKDTree(cen)
    lin = np.zeros(n, np.float32); nz = np.zeros(n, np.float32); rad = np.zeros(n, np.float32)
    nbr = np.zeros((n, k), np.int64); dst = np.zeros((n, k), np.float32)
    for b0 in range(0, n, batch):
        d, idx = tree.query(cen[b0:b0 + batch], k=k, workers=-1)
        nb = cen[idx]; c = nb - nb.mean(1, keepdims=True)
        w, v = np.linalg.eigh(np.einsum('nki,nkj->nij', c, c) / k)
        lin[b0:b0 + batch] = (w[:, 2] - w[:, 1]) / np.maximum(w[:, 2], 1e-12)
        nz[b0:b0 + batch] = np.abs(v[:, 2, 0])
        rad[b0:b0 + batch] = d[:, -1]
        nbr[b0:b0 + batch] = idx; dst[b0:b0 + batch] = d
    return lin, nz, rad, nbr, dst


def thin_lines(cen0, cnt0, nb16, sp, a):
    """Base voxels on a line of points at some scale from 4 to 32 cm, in pieces at least 25 cm long."""
    v = np.zeros(len(cen0), np.float32)
    for s in SCALES:
        cen, inv = voxelise(cen0, s, cnt0)     # coarser voxels from the base centroids, weighted by their points
        if len(cen) < 16:
            continue
        lin, _, rad, _, _ = eig_knn(cen, 16)
        lin[rad > 3 * s] = 0          # too sparse to judge at this scale
        v = np.maximum(v, lin[inv])
    v = v[nb16].mean(1)
    cand = np.flatnonzero(v >= a.line_threshold)
    keep = np.zeros(len(cen0), bool)
    if len(cand) == 0:
        return keep
    nc, lab = knn_graph_components(cen0[cand], float(np.clip(3 * sp, 0.03, 0.15)))
    lo = np.full((nc, 3), np.inf); hi = np.full((nc, 3), -np.inf)
    np.minimum.at(lo, lab, cen0[cand]); np.maximum.at(hi, lab, cen0[cand])
    keep[cand] = (np.linalg.norm(hi - lo, axis=1) >= 0.25)[lab]
    return keep


def segment_shape(cen, lab, nc):
    """Linearity, size and spread along the main axis of each segment."""
    cnt = np.bincount(lab, minlength=nc).astype(np.float64)
    m = np.stack([np.bincount(lab, cen[:, j], nc) for j in range(3)], 1) / np.maximum(cnt, 1)[:, None]
    c = cen - m[lab]
    C = np.zeros((nc, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            C[:, i, j] = C[:, j, i] = np.bincount(lab, c[:, i] * c[:, j], nc) / np.maximum(cnt, 1)
    w = np.linalg.eigvalsh(C)
    lin = (w[:, 2] - w[:, 1]) / np.maximum(w[:, 2], 1e-12)
    return lin, cnt, np.sqrt(np.maximum(w[:, 2], 0))


def segment_probability(cen, nz, nbr, dst, a):
    """LeWoS-style wood probability of each base voxel (Wang, Momo Takoudjou & Casella 2020).

    Neighbours are joined only when their normals have similar verticality, so a
    trunk wall (normals all horizontal) stays one long segment while foliage
    falls apart into small ones. Segments are split again with a stricter
    verticality test while they are large and not linear. A point's
    probability is the share of linearity thresholds from 0.70 to 0.94 that
    its segment passes, as LeWoS sweeps thresholds instead of fixing one."""
    n, k = nbr.shape
    dmax = 3.0 * float(np.median(dst[:, 1])) if k > 1 else 0.0
    src = np.repeat(np.arange(n), k); dd = nbr.ravel(); ds = dst.ravel()
    base = (src != dd) & (ds <= dmax)
    best = np.zeros(n)
    active = np.ones(n, bool)
    thr = a.seg_vertical
    taus = np.linspace(0.70, 0.94, 7)
    for _ in range(4):
        keep = base & active[src] & active[dd] & (np.abs(nz[src] - nz[dd]) < thr)
        nc, lab = connected_components(coo_matrix((np.ones(keep.sum(), np.int8), (src[keep], dd[keep])), shape=(n, n)),
                                       directed=False)
        lin, cnt, sd = segment_shape(cen, lab, nc)
        ok = (cnt >= 10) & (sd >= 0.125)      # 10 voxels; a straight segment at least ~0.43 m long
        best = np.where(active, np.maximum(best, np.where(ok, lin, 0)[lab]), best)
        active &= ((lin < taus[0]) & (cnt >= 40))[lab]
        thr *= 0.6
        if not active.any():
            break
    return (best[:, None] >= taus[None, :]).mean(1)


def wood_probability(xyz, sp, a):
    """Probability that each point lies on a trunk or branch (a line of points)."""
    if len(xyz) < 50:
        return np.zeros(len(xyz))
    cen, inv = voxelise(xyz, max(0.01, sp))
    if len(cen) < 50:
        return np.zeros(len(xyz))
    _, nz, _, nbr, dst = eig_knn(cen, 10)
    q = segment_probability(cen, nz, nbr, dst, a)
    _, nb16 = cKDTree(cen).query(cen, k=min(16, len(cen)), workers=-1)
    if a.thin_lines:
        q = np.maximum(q, thin_lines(cen, np.bincount(inv, minlength=len(cen)).astype(np.float64), nb16, sp, a))
    for _ in range(2):          # average over neighbours, so labels come in pieces
        q = q[nb16].mean(1)
    return q[inv]


# ---------------------------------------------------------------- greyed-colour model

def pale_features(lab, dc):
    L = lab[:, 0] / 100.0; A = lab[:, 1] / 50.0; B = lab[:, 2] / 50.0; v = np.asarray(dc, np.float64) / 0.4
    return np.stack([np.ones_like(L), L, A, B, v, L * v, A * v, B * v, L * L, A * A, B * B, A * B, v * v], 1)


# Fitted on data/samples/column_3m.laz by fit_pale_model() with the default
# settings; used when a run is told not to fit its own (--fit-tiles 0) or
# finds too few pairs.
DEFAULT_PALE_MODEL = np.array([
    [ 17.56124,  -2.45456,   5.37702],
    [ 36.10189,  -2.01774,  -0.21131],
    [-17.78052,  22.64424,   0.14203],
    [  8.28403,   2.69819,  19.86426],
    [  9.24995,  -2.65738,   8.95432],
    [ -7.62288,  -0.55423,   1.93355],
    [  3.78745,   5.66259,  -0.01854],
    [-10.20143,   3.20160,   2.20488],
    [  9.47655,  -1.75778,  -0.82406],
    [ -0.13444,  -7.70555,  -1.04892],
    [  3.37894,  -3.98351,  13.56810],
    [ -1.72391,   9.50162, -15.46680],
    [-11.94266,  -0.66622,   1.84424]])

# Median CIEDE2000 error of each estimate, measured on data/samples/column_3m.laz
# by tools/recolour_eval.py. Borrowed colour against the distance to the
# nearest source (hidden blocks of well-exposed points); the regression
# against the dark channel (held-out pairs). They weight the two estimates.
_BORROW_A, _BORROW_B = 2.63, 7.75
_REG_DC = np.array([0.061, 0.114, 0.156, 0.205, 0.279, 0.356])
_REG_ERR = np.array([3.75, 4.12, 4.39, 5.09, 5.74, 7.01])


def borrow_error(d):
    return _BORROW_A + _BORROW_B * np.sqrt(np.maximum(d, 0))


def regression_error(dc):
    return np.interp(dc, _REG_DC, _REG_ERR)


def collect_pairs(xyz, rgb, t, a, cap, seed=0):
    """(greyed, well-exposed) colour pairs: a greyed point and the nearest well-exposed point
    within 2 point spacings (1-3 cm), captured more than a second apart."""
    blown, dc, w, trusted = colour_state(rgb, a)
    p_idx = np.flatnonzero(~blown & (w > 0)); t_idx = np.flatnonzero(trusted)
    if len(p_idx) == 0 or len(t_idx) < 8:
        return np.zeros((0, 13)), np.zeros((0, 3), np.float32)
    rng = np.random.default_rng(seed)
    if len(p_idx) > cap:
        p_idx = rng.choice(p_idx, cap, replace=False)
    r = float(np.clip(2 * point_spacing(xyz), 0.01, 0.03))
    d, ii = cKDTree(xyz[t_idx]).query(xyz[p_idx], k=8, distance_upper_bound=r, workers=-1)
    ok = np.isfinite(d)
    ii = np.where(ok, ii, 0)
    if t is not None:
        ok &= np.abs(t[t_idx[ii]] - t[p_idx][:, None]) > 1.0
    has = ok.any(1)
    P = p_idx[has]; Q = t_idx[ii[has, np.argmax(ok, 1)[has]]]
    return pale_features(srgb_to_lab(rgb[P]), dc[P]), srgb_to_lab(rgb[Q])


def fit_pale_model(X, Y, ridge=1e-3):
    lam = ridge * len(X)
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y.astype(np.float64))


# ---------------------------------------------------------------- borrowing and palette

def gaussian_borrow(src_xyz, src_lab, src_q, q_xyz, radius, k, min_n, batch):
    """Gaussian-weighted mean colour (CIELAB) of up to k sources within radius."""
    tree = cKDTree(src_xyz)
    k = min(k, len(src_xyz))
    out = np.zeros((len(q_xyz), 3), np.float32)
    ok = np.zeros(len(q_xyz), bool)
    dn = np.full(len(q_xyz), np.inf, np.float32)
    s2 = (radius / 2) ** 2
    for b0 in range(0, len(q_xyz), batch):
        d, idx = tree.query(q_xyz[b0:b0 + batch], k=k, distance_upper_bound=radius, workers=-1)
        if d.ndim == 1:
            d = d[:, None]; idx = idx[:, None]
        fin = np.isfinite(d)
        ii = np.where(fin, idx, 0)
        w = np.where(fin, np.exp(-np.where(fin, d, 0).astype(np.float32) ** 2 / np.float32(s2)), np.float32(0))
        if src_q is not None:
            w = w * src_q[ii]
        ws = w.sum(1)
        out[b0:b0 + batch] = np.einsum('nk,nkc->nc', w, src_lab[ii]) / np.maximum(ws, 1e-12)[:, None]
        ok[b0:b0 + batch] = fin.sum(1) >= min_n
        dn[b0:b0 + batch] = d[:, 0]
    return out, ok, dn


_GOLD = np.uint64(0x9E3779B97F4A7C15)


def _mix(z):
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def hash_ints(*cols):
    """Stable 64-bit hash of integer columns (splitmix64 finaliser)."""
    with np.errstate(over='ignore'):
        h = np.full(np.shape(cols[0]), _GOLD, np.uint64)
        for c in cols:
            h = _mix(h ^ (np.asarray(c).astype(np.int64).view(np.uint64) + _GOLD))
    return h


def unit(h):
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def patches(xyz, cell, salt):
    """Cellular patches (Worley 1996): each point joins the nearest of a jittered lattice of feature points."""
    base = np.floor(xyz / cell).astype(np.int64)
    best = np.full(len(xyz), np.inf)
    key = np.zeros(len(xyz), np.uint64)
    fpt = np.zeros((len(xyz), 3))
    with np.errstate(over='ignore'):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    c = base + np.array([dx, dy, dz])
                    h = hash_ints(c[:, 0], c[:, 1], c[:, 2], np.full(len(c), salt))
                    j = np.stack([unit(h), unit(_mix(h + _GOLD)), unit(_mix(h + _GOLD + _GOLD))], 1)
                    f = (c + j) * cell
                    d = ((xyz - f) ** 2).sum(1)
                    upd = d < best
                    best[upd] = d[upd]; key[upd] = h[upd]; fpt[upd] = f[upd]
    return key, fpt


def palette_colours(q_xyz, q_cls, s_xyz, s_lab, s_cls, gpal, a):
    """Colour each point from a real colour near its patch: same class within --palette-radius,
    then any class, then any colour in the tile, then the scan-wide palette."""
    out = np.zeros((len(q_xyz), 3), np.float32)
    salt = int(a.patch * 1000) + 7
    key, fpt = patches(q_xyz.astype(np.float64), a.patch, salt)
    key = key ^ hash_ints(q_cls.astype(np.int64))           # a branch and the leaves around it differ
    uk, first, inv = np.unique(key, return_index=True, return_inverse=True)
    inv = inv.ravel()
    pf = fpt[first]; pc = q_cls[first]
    pick = unit(_mix(uk + _GOLD))
    col = np.full((len(uk), 3), np.nan, np.float32)

    def choose(sel, src_mask):
        si = np.flatnonzero(src_mask)
        if len(si) == 0 or not sel.any():
            return
        k = min(a.palette_k, len(si))
        d, ii = cKDTree(s_xyz[si]).query(pf[sel], k=k, distance_upper_bound=a.palette_radius, workers=-1)
        if d.ndim == 1:
            d = d[:, None]; ii = ii[:, None]
        n_ok = np.isfinite(d).sum(1)
        has = n_ok > 0
        j = np.minimum((pick[sel] * n_ok).astype(np.int64), np.maximum(n_ok - 1, 0))
        rows = np.flatnonzero(sel)[has]
        col[rows] = s_lab[si[ii[has, j[has]]]]

    for c in np.unique(pc):
        choose((pc == c) & np.isnan(col[:, 0]), s_cls == c)
    choose(np.isnan(col[:, 0]), np.ones(len(s_xyz), bool))
    miss = np.isnan(col[:, 0])
    if miss.any():
        if len(s_lab):
            col[miss] = s_lab[(pick[miss] * len(s_lab)).astype(np.int64) % len(s_lab)]
        else:
            col[miss] = gpal[(pick[miss] * len(gpal)).astype(np.int64) % len(gpal)]
    out[:] = col[inv]
    # a little per-point variation, the size of the difference between neighbouring
    # well-exposed points (about 1.5 CIEDE2000), so patches are not flat
    mm = np.floor(q_xyz * 1000).astype(np.int64)
    h = hash_ints(mm[:, 0], mm[:, 1], mm[:, 2])
    out[:, 0] += (unit(h) - 0.5) * 2 * a.jitter
    out[:, 1] += (unit(_mix(h + _GOLD)) - 0.5) * a.jitter
    out[:, 2] += (unit(_mix(h + _GOLD + _GOLD)) - 0.5) * a.jitter
    return out


# ---------------------------------------------------------------- one tile

STAT_KEYS = ('points', 'blown', 'greyed', 'repaired', 'borrowed', 'palette', 'protected', 'line', 'leaf')

SHOW_COLOURS = {   # --show: false colours, 8-bit sRGB
    'line': (230, 140, 40), 'leaf': (60, 150, 80), 'page': (255, 0, 255),
    'repaired': (240, 220, 60), 'borrowed': (60, 200, 230), 'palette': (220, 60, 60),
}


def recolour_tile(pts, n_own, scale16, model, gpal, a, own_box=None):
    """New colours for the tile's own points (the first n_own of pts; the rest is halo)."""
    st = dict.fromkeys(STAT_KEYS, 0)
    X = np.asarray(pts.x); Y = np.asarray(pts.y); Z = np.asarray(pts.z)
    origin = np.array([X.min(), Y.min(), Z.min()])
    xyz = (np.stack([X, Y, Z], 1) - origin).astype(np.float32)   # float32 relative to the tile corner
    rgb = np.stack([np.asarray(pts.red), np.asarray(pts.green), np.asarray(pts.blue)], 1).astype(np.int32)
    if scale16:
        rgb >>= 8
    n = len(xyz)
    blown, dc, w, trusted = colour_state(rgb, a)
    sp = point_spacing(xyz)

    protected = np.zeros(n, bool); pages = []
    if a.pages:
        protected, pages = find_pages(xyz, rgb, sp, a, np.asarray(pts.intensity).astype(np.float32))
        for p in pages:
            p['centre'] = p['centre'] + origin
        if own_box is not None:
            x0, y0, x1, y1 = own_box
            pages = [p for p in pages if x0 <= p['centre'][0] < x1 and y0 <= p['centre'][1] < y1]
    blown &= ~protected
    trusted &= ~protected
    w = np.where(protected, 0, w)
    pale = ~blown & ~protected & (w > 0)

    st['points'] = n_own
    st['blown'] = int(blown[:n_own].sum()); st['greyed'] = int(pale[:n_own].sum())
    st['protected'] = int(protected[:n_own].sum())
    if not (blown[:n_own].any() or pale[:n_own].any()) and not a.show:
        return None, st, pages

    lab = srgb_to_lab(rgb)
    new = lab.copy()
    path = np.zeros(n, np.uint8)          # 0 untouched, 1 repaired, 2 borrowed, 3 palette

    # 3. greyed points: regression from their own colour, averaged with nearby well-exposed colour
    if pale.any():
        pi = np.flatnonzero(pale)
        reg = (pale_features(lab[pi], dc[pi]) @ model).astype(np.float32)
        comb = reg
        if trusted.sum() >= 3:
            ti = np.flatnonzero(trusted)
            bo, okb, dn = gaussian_borrow(xyz[ti], lab[ti], None, xyz[pi], a.fill_radius, a.k, 3, a.batch)
            wr = 1.0 / regression_error(dc[pi]) ** 2
            wb = np.where(okb, 1.0 / borrow_error(dn) ** 2, 0.0)
            comb = ((wr[:, None] * reg + wb[:, None] * bo) / (wr + wb)[:, None]).astype(np.float32)
        ww = w[pi][:, None]
        new[pi] = (1 - ww) * lab[pi] + ww * comb
        path[pi] = 1

    # 4-6. blown points
    cls = np.full(n, -1, np.int8)         # -1 not classified, 0 leaf (less linear cluster), 1 line (trunk or branch)
    if blown.any():
        roi = dilated_cells(xyz, blown, a.roi_cell)
        ri = np.flatnonzero(roi)
        cls[ri] = (wood_probability(xyz[ri], sp, a) >= a.wood_threshold).astype(np.int8)
        src = roi & ~blown & ~protected   # usable colour near the blown points: well-exposed or repaired
        srcq = np.where(trusted, 1.0, 0.5).astype(np.float32)
        todo = blown.copy()
        for _ in range(a.fill_passes):
            si = np.flatnonzero(src)
            if len(si) < 3:
                break
            done = np.zeros(n, bool)
            for c in (1, 0):              # the same structure class first
                tg = np.flatnonzero(todo & (cls == c)); sc = si[cls[si] == c]
                if len(tg) and len(sc) >= 3:
                    col, ok, _ = gaussian_borrow(xyz[sc], new[sc], srcq[sc], xyz[tg], a.fill_radius, a.k, 3, a.batch)
                    new[tg[ok]] = col[ok]; done[tg[ok]] = True
            tg = np.flatnonzero(todo & ~done)
            if len(tg):
                col, ok, _ = gaussian_borrow(xyz[si], new[si], srcq[si], xyz[tg], a.fill_radius, a.k, 3, a.batch)
                new[tg[ok]] = col[ok]; done[tg[ok]] = True
            path[done] = 2
            todo &= ~done
            src |= done; srcq[done] = 0.25   # filled points help the next pass, weakly
            if not todo.any():
                break
        if todo.any():
            ti = np.flatnonzero(todo)
            ps = np.flatnonzero(roi & ~blown & ~protected)
            new[ti] = palette_colours(xyz[ti], cls[ti], xyz[ps], new[ps], cls[ps], gpal, a)
            path[ti] = 3

    o = slice(0, n_own)
    st['repaired'] = int((path[o] == 1).sum()); st['borrowed'] = int((path[o] == 2).sum())
    st['palette'] = int((path[o] == 3).sum())
    st['line'] = int((blown[o] & (cls[o] == 1)).sum()); st['leaf'] = int((blown[o] & (cls[o] == 0)).sum())

    if a.show:
        grey = (0.3 * rgb[o, 0] + 0.59 * rgb[o, 1] + 0.11 * rgb[o, 2]) * 0.5
        out = np.repeat(grey[:, None], 3, 1)
        if a.show == 'classes':
            for name, m in (('line', cls[o] == 1), ('leaf', cls[o] == 0)):
                out[m & (blown[o] | pale[o])] = SHOW_COLOURS[name]
        else:
            for name, code in (('repaired', 1), ('borrowed', 2), ('palette', 3)):
                out[path[o] == code] = SHOW_COLOURS[name]
        out[protected[o]] = SHOW_COLOURS['page']
        return np.clip(np.rint(out), 0, 255).astype(np.int32), st, pages
    return np.clip(np.rint(lab_to_srgb(new[o])), 0, 255).astype(np.int32), st, pages


# ---------------------------------------------------------------- tiles on disk

SCRATCH_FORMAT = 'recolour scratch v2: t_*.rec own records, h_*.rec halo records\n'


def split_into_tiles(a, scratch):
    """Pass 1: stream the file into per-tile record files, plus each tile's halo of neighbours' records."""
    tiles = {}      # (ix, iy) -> open file of raw records
    halos = {}      # (ix, iy) -> open file of raw records of neighbours' points within the margin
    t = a.tile
    m = a.margin
    n_in = 0
    with laspy.open(a.input) as rd:
        hdr = rd.header
        total = hdr.point_count
        for chunk in rd.chunk_iterator(a.chunk):
            x = np.asarray(chunk.x); y = np.asarray(chunk.y)
            if a.crop is not None:
                x0, y0, x1, y1 = a.crop
                keep = (x >= x0 - m) & (x < x1 + m) & (y >= y0 - m) & (y < y1 + m)
                if not keep.any():
                    n_in += len(chunk); continue
                chunk = chunk[keep]; x = x[keep]; y = y[keep]
                inside = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
            else:
                inside = np.ones(len(x), bool)
            ix = np.floor(x / t).astype(np.int64); iy = np.floor(y / t).astype(np.int64)
            raw = chunk.array   # structured record array
            # own tile
            key = ix * 1_000_003 + iy
            order = np.argsort(key, kind='stable')
            bounds = np.flatnonzero(np.diff(key[order])) + 1
            for grp in np.split(order, bounds):
                if len(grp) == 0: continue
                g_in = grp[inside[grp]]
                if len(g_in) == 0: continue
                k = (int(ix[g_in[0]]), int(iy[g_in[0]]))
                f = tiles.get(k)
                if f is None:
                    f = open(os.path.join(scratch, f't_{k[0]}_{k[1]}.rec'), 'ab'); tiles[k] = f
                f.write(raw[g_in].tobytes())
            # halo: points near a tile edge also go to the neighbouring tile(s)
            fx = x / t - ix; fy = y / t - iy
            mt = m / t
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0: continue
                    sel = np.ones(len(x), bool)
                    if dx == -1: sel &= fx < mt
                    if dx == 1:  sel &= fx > 1 - mt
                    if dy == -1: sel &= fy < mt
                    if dy == 1:  sel &= fy > 1 - mt
                    if not sel.any(): continue
                    nix = ix[sel] + dx; niy = iy[sel] + dy
                    nk = nix * 1_000_003 + niy
                    rs = raw[sel]
                    o = np.argsort(nk, kind='stable')
                    b2 = np.flatnonzero(np.diff(nk[o])) + 1
                    for grp in np.split(o, b2):
                        if len(grp) == 0: continue
                        k = (int(nix[grp[0]]), int(niy[grp[0]]))
                        f = halos.get(k)
                        if f is None:
                            f = open(os.path.join(scratch, f'h_{k[0]}_{k[1]}.rec'), 'ab'); halos[k] = f
                        f.write(rs[grp].tobytes())
            n_in += len(raw)
            log(f'split {n_in / 1e6:.1f} M of {total / 1e6:.1f} M points, {len(tiles)} tiles')
            if len(tiles) + len(halos) > 400:   # stay under the open-file limit;
                for f in list(tiles.values()) + list(halos.values()): f.close()
                tiles = {}; halos = {}              # reopened in append mode on demand
    for f in list(tiles.values()) + list(halos.values()): f.close()
    with open(os.path.join(scratch, 'format.txt'), 'w') as f:
        f.write(SCRATCH_FORMAT)
    return hdr


def load_tile(scratch, key, hdr, with_halo=True):
    dtype = hdr.point_format.dtype()
    own = np.fromfile(os.path.join(scratch, f't_{key}.rec'), dtype=dtype)
    hp = os.path.join(scratch, f'h_{key}.rec')
    data = np.concatenate([own, np.fromfile(hp, dtype=dtype)]) if with_halo and os.path.exists(hp) else own
    return laspy.ScaleAwarePointRecord(data, hdr.point_format, hdr.scales, hdr.offsets), len(own)


def has_gps_time(hdr):
    return 'gps_time' in hdr.point_format.dimension_names


def fit_from_tiles(scratch, keys, hdr, scale16, a):
    """Fit the greyed-colour regression on pairs from a spread of tiles; also keep their
    well-exposed colours as the scan-wide fallback palette."""
    fallback_pal = np.array([[41.0, -12.0, 20.0]], np.float32)
    if a.fit_tiles <= 0:
        log('greyed-colour model: built in (--fit-tiles 0)')
        return DEFAULT_PALE_MODEL, fallback_pal
    sizes = np.array([os.path.getsize(os.path.join(scratch, f't_{k}.rec')) for k in keys])
    big = [k for k, s in zip(keys, sizes) if s >= np.median(sizes)] or list(keys)
    pick = [big[i] for i in np.unique(np.linspace(0, len(big) - 1, min(a.fit_tiles, len(big))).round().astype(int))]
    Xs, Ys = [], []
    cap = max(10_000, a.fit_pairs // len(pick))
    for i, key in enumerate(pick):
        pts, _ = load_tile(scratch, key, hdr, with_halo=False)
        if len(pts) > a.slab_points:      # a central slab is plenty for pairs, and bounds memory
            x = np.asarray(pts.x)
            f = a.slab_points / len(pts)
            lo, hi = np.quantile(x, [0.5 - f / 2, 0.5 + f / 2])
            pts = pts[(x >= lo) & (x <= hi)]
        xyz = np.stack([np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)], 1)
        xyz = (xyz - xyz.min(0)).astype(np.float32)
        rgb = np.stack([np.asarray(pts.red), np.asarray(pts.green), np.asarray(pts.blue)], 1).astype(np.int32)
        if scale16:
            rgb >>= 8
        t = np.asarray(pts.gps_time) if has_gps_time(hdr) else None
        X, Y = collect_pairs(xyz, rgb, t, a, cap, seed=i)
        Xs.append(X); Ys.append(Y)
        log(f'fit: tile {i + 1}/{len(pick)} gave {len(X):,} greyed/well-exposed pairs')
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    if len(X) < 2000:
        log(f'greyed-colour model: only {len(X)} pairs, using the built-in model')
        return DEFAULT_PALE_MODEL, fallback_pal
    W = fit_pale_model(X, Y)
    rng = np.random.default_rng(0)
    pal = Y[rng.choice(len(Y), min(20_000, len(Y)), replace=False)]
    log(f'greyed-colour model: fitted on {len(X):,} pairs')
    return W, pal.astype(np.float32)


def slabs(x_own, x_all, n_own, slab_points, margin):
    """Cut a dense tile into slabs along x of at most slab_points own points, each with its
    context: every other point of the tile or halo within margin. Memory is about 0.35 GB per
    million points processed at once (measured on the samples)."""
    n = max(1, int(np.ceil(n_own / slab_points)))
    if n == 1:
        yield np.arange(n_own), np.arange(n_own, len(x_all)), -np.inf, np.inf
        return
    edges = np.quantile(x_own, np.linspace(0, 1, n + 1)); edges[0] = -np.inf; edges[-1] = np.inf
    for x0, x1 in zip(edges[:-1], edges[1:]):
        own = np.flatnonzero((x_own >= x0) & (x_own < x1))
        ctx = np.flatnonzero((x_all >= x0 - margin) & (x_all < x1 + margin))
        yield own, ctx[(ctx >= n_own) | (x_all[ctx] < x0) | (x_all[ctx] >= x1)], x0, x1


def process_tile(job):
    """Recolour one tile and write its records to o_<key>.rec. Runs in a worker process when --workers > 1."""
    scratch, key, a, scale16, model, gpal = job
    with laspy.open(a.input) as rd:
        hdr = rd.header
    pts, n_own = load_tile(scratch, key, hdr)
    ix, iy = (int(v) for v in key.split('_'))
    X = np.asarray(pts.x)
    st = dict.fromkeys(STAT_KEYS, 0); pages = []
    res = None
    for own_i, ctx_i, x0, x1 in slabs(X[:n_own], X, n_own, a.slab_points, a.margin):
        box = (max(ix * a.tile, x0), iy * a.tile, min((ix + 1) * a.tile, x1), (iy + 1) * a.tile)
        sub = pts if len(ctx_i) == len(X) - n_own and len(own_i) == n_own else pts[np.concatenate([own_i, ctx_i])]
        r, s, p = recolour_tile(sub, len(own_i), scale16, model, gpal, a, own_box=box)
        for k in STAT_KEYS: st[k] += s[k]
        pages.extend(p)
        if r is not None:
            if res is None:
                res = np.stack([np.asarray(pts.red[:n_own]), np.asarray(pts.green[:n_own]), np.asarray(pts.blue[:n_own])], 1).astype(np.int32)
                if scale16:
                    res >>= 8
            res[own_i] = r
    own = pts[:n_own]
    if res is not None:
        mul = 257 if scale16 else 1
        dt = np.asarray(own.red).dtype
        own.red = (res[:, 0] * mul).astype(dt); own.green = (res[:, 1] * mul).astype(dt); own.blue = (res[:, 2] * mul).astype(dt)
    own.array.tofile(os.path.join(scratch, f'o_{key}.rec'))
    return key, st, pages


def write_pages_csv(path, pages):
    """Detected page centres in the viewer's markers.csv format (file coordinates)."""
    with open(path, 'w') as f:
        f.write('mic,label,x,y,z,sx,sy,sz,has_survey\n')
        for i, p in enumerate(pages):
            x, y, z = p['centre']
            f.write(f'{i + 1},page {i + 1},{x:.6f},{y:.6f},{z:.6f},0,0,0,0\n')


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input'); ap.add_argument('output')
    ap.add_argument('--crop', type=float, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'), help='only this rectangle, for a test')
    ap.add_argument('--tile', type=float, default=16.0, help='tile size in metres (default 16); smaller tiles use less memory')
    ap.add_argument('--margin', type=float, default=0.6, help='halo width in metres (default 0.6)')
    g = ap.add_argument_group('detection')
    g.add_argument('--white-level', type=int, default=170, help='blown out: darkest channel at least this (default 170) ...')
    g.add_argument('--white-spread', type=int, default=45, help='... and channels within this of each other (default 45)')
    g.add_argument('--sky-cast', type=int, default=6, help='blue above green by more than this is a sky cast; 0 = off (default 6)')
    g.add_argument('--pale-lo', type=float, default=0.10, help='greyed: correction starts at this dark channel, linear 0-1 (default 0.10, about 89 in 8 bit)')
    g.add_argument('--pale-hi', type=float, default=0.20, help='greyed: full correction from this dark channel (default 0.20, about 124 in 8 bit)')
    g.add_argument('--grey-sat-lo', type=float, default=0.15, help='greyed: full correction below this saturation, linear light (default 0.15)')
    g.add_argument('--grey-sat-hi', type=float, default=0.30, help='greyed: no correction from this saturation up (default 0.30)')
    g.add_argument('--min-source', type=int, default=40, help='brightest channel a well-exposed colour source needs (default 40)')
    g = ap.add_argument_group('survey pages')
    g.add_argument('--no-pages', dest='pages', action='store_false', help='do not look for checkerboard pages')
    g.add_argument('--page-spread', type=float, default=0.12, help='achromatic: channels within this share of the brightest one, at least 8 (default 0.12)')
    g.add_argument('--page-paper', type=int, default=110, help='the paper (brightest tenth of the page) at least this bright (default 110)')
    g.add_argument('--page-thickness', type=float, default=0.01, help='flat: standard deviation off the plane at most this, m (default 0.01)')
    g.add_argument('--page-contrast', type=float, default=0.3, help='laser intensity contrast between paper and print at least this (default 0.3)')
    g.add_argument('--page-cover', type=float, default=0.5, help='filled: share of the 2 cm cells inside the page holding points (default 0.5)')
    g.add_argument('--page-min-points', type=int, default=30)
    g = ap.add_argument_group('structure')
    g.add_argument('--wood-threshold', type=float, default=0.45, help='trunk/branch when the smoothed probability is at least this (default 0.45)')
    g.add_argument('--seg-vertical', type=float, default=0.15, help='LeWoS verticality threshold for joining neighbours (default 0.15)')
    g.add_argument('--line-threshold', type=float, default=0.6, help='thin lines: smoothed multi-scale linearity at least this (default 0.6)')
    g.add_argument('--no-thin-lines', dest='thin_lines', action='store_false', help='skip the thin-line test (faster; keeps trunks and big branches)')
    g.add_argument('--roi-cell', type=float, default=0.5, help='structure is classified in cells of this size around blown points, m (default 0.5)')
    g = ap.add_argument_group('colour')
    g.add_argument('--fill-radius', type=float, default=0.15, help='borrow colour from sources within this, m (default 0.15)')
    g.add_argument('--fill-passes', type=int, default=2, help='borrowing passes; each reaches one radius further (default 2)')
    g.add_argument('--k', type=int, default=16, help='sources per point at most (default 16)')
    g.add_argument('--patch', type=float, default=0.3, help='palette patch size, m (default 0.3)')
    g.add_argument('--palette-radius', type=float, default=2.0, help='palette colours come from real points within this of the patch, m (default 2)')
    g.add_argument('--palette-k', type=int, default=64)
    g.add_argument('--jitter', type=float, default=2.0, help='per-point lightness variation in palette patches, CIELAB L* (default 2)')
    g.add_argument('--fit-tiles', type=int, default=8, help='tiles the greyed-colour model is fitted on; 0 = built-in model (default 8)')
    g.add_argument('--fit-pairs', type=int, default=400_000, help='pairs to fit on at most (default 400000)')
    g = ap.add_argument_group('run')
    g.add_argument('--show', choices=('classes', 'paths'), default=None,
                   help='write false colour instead: classes = trunk/branch orange, leaf green; '
                        'paths = repaired yellow, borrowed cyan, palette red. Pages magenta, untouched points dim grey')
    g.add_argument('--workers', type=int, default=1, help='tiles processed in parallel (default 1); memory grows with each')
    g.add_argument('--slab-points', type=int, default=12_000_000,
                   help='denser tiles are processed in slabs of at most this many points (default 12 M, about 4.5 GB per worker)')
    g.add_argument('--chunk', type=int, default=5_000_000)
    g.add_argument('--batch', type=int, default=500_000, help='points queried at once (memory)')
    g.add_argument('--reuse-scratch', action='store_true', help='--scratch already holds the tiles of this input: skip the split')
    g.add_argument('--scratch', default=None, help='directory for tile files (default: next to the output)')
    return ap


def main():
    a = build_parser().parse_args()

    scratch = a.scratch or tempfile.mkdtemp(prefix='recolour_', dir=os.path.dirname(os.path.abspath(a.output)))
    os.makedirs(scratch, exist_ok=True)
    log(f'scratch {scratch}')
    t0 = time.time()

    with laspy.open(a.input) as rd:
        hdr = rd.header
        sample = next(rd.chunk_iterator(200_000))
    scale16 = max(int(np.max(sample.red)), int(np.max(sample.green)), int(np.max(sample.blue))) > 255
    log(f'colour is {"16" if scale16 else "8"}-bit')

    if a.reuse_scratch:
        fp = os.path.join(scratch, 'format.txt')
        if not os.path.exists(fp) or open(fp).read() != SCRATCH_FORMAT:
            raise SystemExit('--reuse-scratch: the scratch directory was written by an older version; run without it')
        log('reusing the tiles already in scratch')
    else:
        split_into_tiles(a, scratch)
        log(f'split done in {time.time() - t0:.0f} s')

    keys = sorted(f[2:-4] for f in os.listdir(scratch) if f.startswith('t_') and f.endswith('.rec'))
    if not keys:
        raise SystemExit('no points to process (is --crop outside the scan?)')
    biggest = max(os.path.getsize(os.path.join(scratch, f't_{k}.rec')) for k in keys) // hdr.point_format.size
    at_once = min(biggest, a.slab_points)
    log(f'{len(keys)} tiles; the largest holds {biggest / 1e6:.1f} M points. About {0.35 * at_once / 1e6 + 0.3:.1f} GB '
        f'per worker ({a.workers} worker{"s" if a.workers > 1 else ""}); --slab-points and --tile lower it')
    model, gpal = fit_from_tiles(scratch, keys, hdr, scale16, a)

    stats = dict.fromkeys(STAT_KEYS, 0)
    pages = []
    jobs = [(scratch, key, a, scale16, model, gpal) for key in keys]

    def done(i, key, st, pg):
        for k in STAT_KEYS: stats[k] += st[k]
        pages.extend(pg)
        if i % 10 == 0 or i == len(keys) - 1:
            log(f'tile {i + 1}/{len(keys)}: {stats["points"] / 1e6:.1f} M points, {stats["blown"] / 1e6:.2f} M blown, '
                f'{stats["greyed"] / 1e6:.2f} M greyed; {stats["repaired"] / 1e6:.2f} M repaired, '
                f'{stats["borrowed"] / 1e6:.2f} M borrowed, {stats["palette"] / 1e6:.2f} M palette')

    if a.workers > 1:
        ctx = multiprocessing.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
            for i, (key, st, pg) in enumerate(ex.map(process_tile, jobs)):
                done(i, key, st, pg)
    else:
        for i, job in enumerate(jobs):
            key, st, pg = process_tile(job)
            done(i, key, st, pg)

    out_hdr = laspy.LasHeader(point_format=hdr.point_format, version=hdr.version)
    out_hdr.scales = hdr.scales; out_hdr.offsets = hdr.offsets
    for v in hdr.vlrs: out_hdr.vlrs.append(v)
    dtype = hdr.point_format.dtype()
    with laspy.open(a.output, mode='w', header=out_hdr) as wr:
        for key in keys:
            data = np.fromfile(os.path.join(scratch, f'o_{key}.rec'), dtype=dtype)
            wr.write_points(laspy.ScaleAwarePointRecord(data, hdr.point_format, hdr.scales, hdr.offsets))
            os.remove(os.path.join(scratch, f'o_{key}.rec'))

    if pages:
        pp = os.path.splitext(a.output)[0] + '_pages.csv'
        write_pages_csv(pp, pages)
        for i, p in enumerate(pages):
            x, y, z = p['centre']
            log(f'page {i + 1}: centre {x:.3f} {y:.3f} {z:.3f}, {p["long"] * 100:.0f} x {p["short"] * 100:.0f} cm, '
                f'{p["points"]} points, {100 * p["cover"]:.0f}% filled, {100 * p["dark"]:.0f}% dark, '
                f'{100 * p["bright"]:.0f}% paper, intensity contrast {p["contrast"]:.2f}')
        log(f'{len(pages)} checkerboard pages left unchanged; centres in {pp}')
    if a.scratch is None:
        shutil.rmtree(scratch, ignore_errors=True)
    p = max(stats['points'], 1); b = max(stats['blown'], 1)
    log(f'done in {(time.time() - t0) / 60:.1f} min: {stats["points"]:,} points; '
        f'{100 * stats["blown"] / p:.2f}% blown, {100 * stats["greyed"] / p:.2f}% greyed. '
        f'Greyed repaired: {stats["repaired"]:,}. Blown filled from neighbours: {100 * stats["borrowed"] / b:.0f}%, '
        f'from the palette: {100 * stats["palette"] / b:.0f}%. Blown classed trunk/branch: {stats["line"]:,}, '
        f'leaf: {stats["leaf"]:,}. Page points kept: {stats["protected"]:,}')


if __name__ == '__main__':
    main()
