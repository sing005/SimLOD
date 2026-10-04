"""Repair the blown-out canopy colour of a FARO scan, offline.

The scanner colours points from its panorama camera. In the sunlit canopy the
camera saturates, so leaves and branches there come out white, pale grey or
with a lilac/blue sky cast. Their positions and laser intensity are fine;
only the colour is lost. This rewrites the colour of those points and
leaves every other point, and every other attribute, untouched.

For each blown-out point, its k nearest neighbours in the full cloud decide:
  1. If enough neighbours have usable colour, it takes their median colour
     (lifted to a minimum brightness): the scan's own colour, borrowed.
  2. Otherwise it is classed as wood or leaf and given a natural brown or
     green with per-point variation:
       - shape of the neighbourhood (PCA): bark lies on lines and cylinder
         walls (linear / planar), foliage is scattered;
       - laser intensity, which the sun does not affect: bark returns are
         stronger than leaves on this scanner (median ~9,800 vs ~6,900).

The cloud is split into ground tiles on disk first, each with a thin halo of
its neighbours' points, so the job runs in bounded memory at any size.

Usage:
  python recolour_canopy.py IN.laz OUT.laz                      # whole scan
  python recolour_canopy.py IN.laz OUT.laz --crop -10 -10 10 10 # test tile
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

import laspy
import numpy as np
from scipy.spatial import cKDTree


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def rgb8(points):
    r = np.asarray(points.red); g = np.asarray(points.green); b = np.asarray(points.blue)
    return r, g, b


def blown_mask(r, g, b, a):
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    white = (mn >= a.white_level) & ((mx - mn) <= a.white_spread)
    cast = (b.astype(np.int32) > g.astype(np.int32) + a.sky_cast) & (mx >= 40) if a.sky_cast > 0 else np.zeros_like(white)
    return white | cast


def hash01(x, y, z):
    """Stable per-point pseudo-random in [0, 1) from position."""
    h = (np.asarray(x * 1000, np.int64) * 73856093) ^ (np.asarray(y * 1000, np.int64) * 19349663) ^ (np.asarray(z * 1000, np.int64) * 83492791)
    h = (h ^ (h >> 13)) * 0x5bd1e995
    h = h ^ (h >> 15)
    return (h & 0xFFFFFF).astype(np.float64) / float(0x1000000)


# natural palettes, 8-bit sRGB; points pick one by hash and get +-variation
LEAF = np.array([[ 62, 102,  38], [ 78, 118,  44], [ 52,  88,  34],
                 [ 92, 128,  52], [ 70,  96,  40], [ 58, 110,  48]], np.float64)
WOOD = np.array([[ 92,  68,  48], [ 78,  58,  42], [104,  80,  58],
                 [ 70,  56,  44], [ 88,  72,  56]], np.float64)


def split_into_tiles(a, scale16, scratch):
    """Pass 1: stream the file into per-tile record files plus halo xyz."""
    tiles = {}      # (ix, iy) -> open file of raw records
    halos = {}      # (ix, iy) -> open file of float64 xyz, neighbours' points within the margin
    t = a.tile
    m = a.margin
    n_in = 0
    with laspy.open(a.input) as rd:
        hdr = rd.header
        rec_size = hdr.point_format.size
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
            z = np.asarray(chunk.z)
            ix = np.floor(x / t).astype(np.int64); iy = np.floor(y / t).astype(np.int64)
            raw = chunk.array   # structured record array
            # own tile
            key = ix * 1_000_003 + iy
            order = np.argsort(key, kind='stable')
            ks = key[order]
            bounds = np.flatnonzero(np.diff(ks)) + 1
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
                    nk = (ix[sel] + dx) * 1_000_003 + (iy[sel] + dy)
                    xyz = np.stack([x[sel], y[sel], z[sel]], 1)
                    nix = ix[sel] + dx; niy = iy[sel] + dy
                    o = np.argsort(nk, kind='stable')
                    b2 = np.flatnonzero(np.diff(nk[o])) + 1
                    for grp in np.split(o, b2):
                        if len(grp) == 0: continue
                        k = (int(nix[grp[0]]), int(niy[grp[0]]))
                        f = halos.get(k)
                        if f is None:
                            f = open(os.path.join(scratch, f'h_{k[0]}_{k[1]}.xyz'), 'ab'); halos[k] = f
                        f.write(xyz[grp].astype(np.float64).tobytes())
            n_in += len(raw)
            log(f'split {n_in / 1e6:.1f} M of {total / 1e6:.1f} M points, {len(tiles)} tiles')
            if len(tiles) + len(halos) > 400:   # stay under the open-file limit;
                for f in list(tiles.values()) + list(halos.values()): f.close()
                tiles = {}; halos = {}              # reopened in append mode on demand
    for f in list(tiles.values()) + list(halos.values()): f.close()
    return hdr, rec_size


def recolour_tile(pts, halo_xyz, scale16, a, stats):
    x = np.asarray(pts.x); y = np.asarray(pts.y); z = np.asarray(pts.z)
    r, g, b = rgb8(pts)
    if scale16:
        r = r >> 8; g = g >> 8; b = b >> 8
    r = r.astype(np.int32); g = g.astype(np.int32); b = b.astype(np.int32)
    blown = blown_mask(r, g, b, a)
    nb = int(blown.sum())
    stats['points'] += len(x); stats['blown'] += nb
    if nb == 0:
        return None

    n_own = len(x)
    mx = np.maximum(np.maximum(r, g), b)
    good = (~blown) & (mx >= a.min_source)
    inten = np.asarray(pts.intensity).astype(np.float64)
    lo, hi = np.percentile(inten, [10, 90]) if n_own > 10 else (0.0, 1.0)

    # float32, relative to the tile corner: half the memory of float64
    origin = np.array([x.min(), y.min(), z.min()])
    own_xyz = (np.stack([x, y, z], 1) - origin).astype(np.float32)
    halo = None if halo_xyz is None or len(halo_xyz) == 0 else (halo_xyz - origin).astype(np.float32)
    blown_idx = np.flatnonzero(blown)
    out = np.zeros((nb, 3), np.float64)
    pos_in_out = np.full(n_own, -1, np.int64); pos_in_out[blown_idx] = np.arange(nb)

    # Tiles beside the scanner hold tens of millions of points, too many for
    # one k-d tree. Cut them into slabs along x of at most slab_points each,
    # every slab with its own margin of context, and query in batches.
    n_slabs = max(1, int(np.ceil(n_own / a.slab_points)))
    edges = np.quantile(own_xyz[:, 0], np.linspace(0, 1, n_slabs + 1)) if n_slabs > 1 else np.array([-np.inf, np.inf])
    edges[0] = -np.inf; edges[-1] = np.inf
    m = np.float32(a.margin)
    for si in range(n_slabs):
        x0, x1 = edges[si], edges[si + 1]
        q_sel = blown_idx[(own_xyz[blown_idx, 0] >= x0) & (own_xyz[blown_idx, 0] < x1)]
        if len(q_sel) == 0: continue
        ctx_own = np.flatnonzero((own_xyz[:, 0] >= x0 - m) & (own_xyz[:, 0] < x1 + m))
        ctx = own_xyz[ctx_own]
        if halo is not None:
            hsel = (halo[:, 0] >= x0 - m) & (halo[:, 0] < x1 + m)
            ctx = np.vstack([ctx, halo[hsel]])
        n_ctx_own = len(ctx_own)
        good_ctx = np.zeros(len(ctx), bool); good_ctx[:n_ctx_own] = good[ctx_own]
        tree = cKDTree(ctx)
        k = min(a.k, len(ctx))
        for b0 in range(0, len(q_sel), a.batch):
            qi = q_sel[b0:b0 + a.batch]
            _, idx = tree.query(own_xyz[qi], k=k, workers=-1)
            if idx.ndim == 1: idx = idx[:, None]
            ng = good_ctx[idx]
            borrow = ng.sum(1) >= a.min_good
            oi = pos_in_out[qi]
            if borrow.any():
                src = ctx_own[np.where(idx[borrow] < n_ctx_own, idx[borrow], 0)]
                gg = ng[borrow]
                cr = np.where(gg, r[src], np.nan); cg_ = np.where(gg, g[src], np.nan); cb = np.where(gg, b[src], np.nan)
                out[oi[borrow]] = np.stack([np.nanmedian(cr, 1), np.nanmedian(cg_, 1), np.nanmedian(cb, 1)], 1)
            stats['borrowed'] += int(borrow.sum())

            rest = ~borrow
            if rest.any():
                nbr = ctx[idx[rest]].astype(np.float64)       # (n, k, 3)
                c = nbr - nbr.mean(1, keepdims=True)
                cov = np.einsum('nki,nkj->nij', c, c) / max(k - 1, 1)
                ev = np.linalg.eigvalsh(cov)[:, ::-1]         # l1 >= l2 >= l3
                l1 = np.maximum(ev[:, 0], 1e-12)
                shape = np.maximum((ev[:, 0] - ev[:, 1]) / l1, (ev[:, 1] - ev[:, 2]) / l1)   # bark: line or cylinder wall
                pr = qi[rest]
                irank = np.clip((inten[pr] - lo) / max(hi - lo, 1.0), 0, 1)
                wood = (a.w_shape * shape + (1 - a.w_shape) * irank) > a.wood_threshold
                stats['wood'] += int(wood.sum()); stats['leaf'] += int((~wood).sum())
                h = hash01(x[pr], y[pr], z[pr])
                h2 = hash01(y[pr] + 17.3, z[pr] - 5.1, x[pr] + 2.7)
                col = np.where(wood[:, None], WOOD[(h * len(WOOD)).astype(int) % len(WOOD)],
                                              LEAF[(h * len(LEAF)).astype(int) % len(LEAF)])
                # brightness: +-18% per point, plus a little from intensity
                col = col * (0.82 + 0.36 * h2)[:, None] * (0.9 + 0.2 * irank)[:, None]
                out[oi[rest]] = col
        del tree

    # brightness floor for borrowed colours
    luma = 0.299 * out[:, 0] + 0.587 * out[:, 1] + 0.114 * out[:, 2]
    lift = np.where((luma > 1) & (luma < a.min_luma), a.min_luma / np.maximum(luma, 1), 1.0)
    out = np.clip(out * lift[:, None], 0, 255)

    mul = 257 if scale16 else 1
    nr = np.asarray(pts.red).copy(); ngr = np.asarray(pts.green).copy(); nbl = np.asarray(pts.blue).copy()
    nr[blown] = (out[:, 0] * mul).astype(nr.dtype)
    ngr[blown] = (out[:, 1] * mul).astype(ngr.dtype)
    nbl[blown] = (out[:, 2] * mul).astype(nbl.dtype)
    return nr, ngr, nbl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input'); ap.add_argument('output')
    ap.add_argument('--crop', type=float, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'), help='only this rectangle, for a test')
    ap.add_argument('--tile', type=float, default=16.0, help='tile size in metres (default 16)')
    ap.add_argument('--margin', type=float, default=0.6, help='halo width in metres (default 0.6)')
    ap.add_argument('--k', type=int, default=24, help='neighbours per point (default 24)')
    ap.add_argument('--white-level', type=int, default=170)
    ap.add_argument('--white-spread', type=int, default=45)
    ap.add_argument('--sky-cast', type=int, default=6, help='blue above green by more than this is a sky cast; 0 = off')
    ap.add_argument('--min-source', type=int, default=60, help='brightest channel a colour source needs')
    ap.add_argument('--min-good', type=int, default=6, help='good neighbours needed to borrow colour')
    ap.add_argument('--min-luma', type=float, default=85.0)
    ap.add_argument('--w-shape', type=float, default=0.65, help='weight of shape vs intensity in the wood score')
    ap.add_argument('--wood-threshold', type=float, default=0.55)
    ap.add_argument('--chunk', type=int, default=5_000_000)
    ap.add_argument('--slab-points', type=int, default=12_000_000, help='max points per k-d tree (memory)')
    ap.add_argument('--batch', type=int, default=1_000_000, help='blown points queried at once (memory)')
    ap.add_argument('--reuse-scratch', action='store_true', help='--scratch already holds the tiles of this input: skip the split')
    ap.add_argument('--scratch', default=None, help='directory for tile files (default: next to the output)')
    a = ap.parse_args()

    scratch = a.scratch or tempfile.mkdtemp(prefix='recolour_', dir=os.path.dirname(os.path.abspath(a.output)))
    os.makedirs(scratch, exist_ok=True)
    log(f'scratch {scratch}')
    t0 = time.time()

    with laspy.open(a.input) as rd:
        sample = next(rd.chunk_iterator(200_000))
    scale16 = max(int(np.max(sample.red)), int(np.max(sample.green)), int(np.max(sample.blue))) > 255
    log(f'colour is {"16" if scale16 else "8"}-bit')

    if a.reuse_scratch:
        with laspy.open(a.input) as rd:
            hdr = rd.header
        log('reusing the tiles already in scratch')
    else:
        hdr, rec_size = split_into_tiles(a, scale16, scratch)
        log(f'split done in {time.time() - t0:.0f} s')

    out_hdr = laspy.LasHeader(point_format=hdr.point_format, version=hdr.version)
    out_hdr.scales = hdr.scales; out_hdr.offsets = hdr.offsets
    for v in hdr.vlrs: out_hdr.vlrs.append(v)
    dtype = hdr.point_format.dtype()

    stats = dict(points=0, blown=0, borrowed=0, wood=0, leaf=0)
    files = sorted(f for f in os.listdir(scratch) if f.startswith('t_'))
    with laspy.open(a.output, mode='w', header=out_hdr) as wr:
        for i, fn in enumerate(files):
            key = fn[2:-4]
            data = np.fromfile(os.path.join(scratch, fn), dtype=dtype)
            pts = laspy.ScaleAwarePointRecord(data, hdr.point_format, hdr.scales, hdr.offsets)
            hp = os.path.join(scratch, f'h_{key}.xyz')
            halo = np.fromfile(hp, dtype=np.float64).reshape(-1, 3) if os.path.exists(hp) else None
            res = recolour_tile(pts, halo, scale16, a, stats)
            if res is not None:
                pts.red, pts.green, pts.blue = res
            wr.write_points(pts)
            if i % 10 == 0 or i == len(files) - 1:
                log(f'tile {i + 1}/{len(files)}: {stats["points"] / 1e6:.1f} M points, '
                    f'{stats["blown"] / 1e6:.2f} M blown ({stats["borrowed"] / 1e6:.2f} M borrowed, '
                    f'{stats["wood"] / 1e6:.2f} M wood, {stats["leaf"] / 1e6:.2f} M leaf)')

    if a.scratch is None:
        shutil.rmtree(scratch, ignore_errors=True)
    p = max(stats['points'], 1)
    log(f'done in {(time.time() - t0) / 60:.1f} min: {stats["points"]:,} points, '
        f'{100 * stats["blown"] / p:.2f}% recoloured; of those '
        f'{100 * stats["borrowed"] / max(stats["blown"], 1):.0f}% borrowed neighbour colour, '
        f'{stats["wood"]:,} wood, {stats["leaf"]:,} leaf')


if __name__ == '__main__':
    main()
