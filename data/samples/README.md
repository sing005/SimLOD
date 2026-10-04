# Point-cloud samples for the canopy recolour

The full scans are 3–4 GB and stay on the local machine. These samples are cut from the scan with the whited-out canopy, so `tools/recolour_canopy.py` can be developed and tested anywhere, including cloud sessions.

They are ordinary git files, each under GitHub's 50 MB warning size. Cloud sessions cannot fetch Git LFS files, so LFS is not used.

| File | Contents | Points | Size |
|---|---|---|---|
| `column_3m.laz` | Every point in a 3 m square column (x 2.11..5.11, y −1.50..1.50), centred on the centre of the scan's points. All heights, from the forest floor (z −0.33) to the canopy above it (z 19.8). 9.6 % of its points are blown out, 63 % of those above 8 m. | 4,815,894 | 44.0 MB |
| `every100.laz` | Every 100th point of the whole scan, in file order. 9.6 % of its points are blown out, the same share as the full scan. | 3,458,367 | 39.3 MB |

## Source

- **File:** `24234f25-7605-45c1-956d-d24ec5ec9bc0_2025-11-21_16-35-47_clean_color-WhiteCanopy.laz`, kept locally in `C:\prog\SimLodMod\LiDAR\`.
- **Scan:** a FARO terrestrial laser scan of a forest plot at Rotokare, exported from FARO Connect. 345,836,698 points.
- **Records:** copied unchanged. Format, scale and offsets are as in the full scan.
- **Format:** LAZ, LAS 1.2 point format 3: XYZ, intensity, return number and count, classification, GPS time, RGB.
- **Coordinates:** metres in the scan's own frame (scale 0.0001, offset 0). The scan's points span x −47.3..54.6, y −55.6..55.6, z −4.7..27.3. The full scan's LAS header box is looser (213 × 182 × 77 m); each sample's header has tight bounds.
- **Colour:** 8-bit values (0–255) stored in the 16-bit RGB fields.

## The problem

The scanner colours each point from its panorama camera. In the sunlit canopy the camera saturates, so leaves and branches come out white or pale grey, some with a lilac, pink or blue sky cast. Positions and laser intensity are unaffected; only the colour is lost.

At the script's default thresholds, 9.6 % of the full scan's points are blown out. A point counts as blown out when either:
- its darkest channel is at least 170 and its channels are within 45 of each other, or
- its blue is more than 6 above its green (a sky cast) and its brightest channel is at least 40.

## The current tool

For each blown-out point, `tools/recolour_canopy.py` looks at its 24 nearest neighbours.
- If at least 6 of them have usable colour, the point takes their median colour, lifted to a minimum brightness.
- Otherwise it is classed as wood or leaf, from its neighbourhood's shape (PCA) and its laser intensity, and given a natural brown or green with per-point variation.

```
pip install -r tools/requirements-recolour.txt
python tools/recolour_canopy.py data/samples/every100.laz out.laz
```

On the full scan it runs in 16 m ground tiles with a 0.6 m halo, and takes about 15 minutes on a 20-thread laptop. Run `--help` for the thresholds.

Weaknesses seen in 4K renders of the full scan:
- Tree-fern fronds stay pale grey-green, because they fall below the white test.
- The wood/leaf classification leaves orange "wood" speckle through the crown.

## Neighbour density

Point density varies a great deal: near the scanner positions there are about 440,000 points per square metre of ground. A k-nearest-neighbour radius tuned on the full-density column is far smaller than one that suits `every100.laz`. Parameters given in metres (tile, halo) do not change with density; `k` and the neighbour counts do.
