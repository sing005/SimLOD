# Point-cloud samples for the canopy recolour

The full scans are 3–4 GB and stay on the local machine. These samples are cut from the scan with the whited-out canopy, so `tools/recolour_canopy.py` can be developed and tested anywhere, including Claude Code cloud sessions.

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

The scanner colours each point from its panorama camera. In the sunlit canopy the camera saturates, so leaves and branches come out white or pale grey, some with a lilac, pink or blue sky cast. Below the white canopy the colour fades with height through pale grey-green, on trunks and tree-fern fronds alike. Positions and laser intensity are unaffected; only the colour is lost or washed out.

At the tool's default thresholds:

| | `every100.laz` | `column_3m.laz` |
|---|---|---|
| Blown out (white or sky cast) | 9.6 % | 9.6 % |
| Greyed, corrected at least partly | 32.6 % | 31.2 % |
| Greyed, corrected more than half | 21.3 % | 22.9 % |

A point counts as blown out when either its darkest channel is at least 170 and its channels are within 45 of each other, or its blue is more than 6 above its green (a sky cast) and its brightest channel is at least 40. A point counts as greyed when its darkest channel, in linear light, is above 0.10 (about 89 in 8 bit), or when it is bright enough to judge and has almost no saturation.

## The tool

```
pip install -r tools/requirements-recolour.txt
python tools/recolour_canopy.py data/samples/every100.laz out.laz
python tools/recolour_canopy.py data/samples/every100.laz classes.laz --show classes   # false colour
python tools/recolour_canopy.py data/samples/every100.laz out.laz --texture 1          # natural variation of colour
python tools/recolour_eval.py data/samples/column_3m.laz --output out.laz              # the measurements
```

`tools/recolour_canopy.py` works in 16 m ground tiles with a 0.6 m halo. Its docstring (`--help`) gives the method; in short:

1. **Detect** blown-out and greyed points. The greyed correction is weighted from 0 to 1, so there is no visible edge where it starts.
2. **Protect survey pages.** Flat, A4-sized, achromatic patches with near-white paper and dark print, which also show the print's low laser intensity, keep their colour. Their centres go to `OUT_pages.csv`, in the viewer's `markers.csv` format.
3. **Repair greyed points** with a regression from the greyed colour to the well-exposed colour, learnt on each run from this scan, averaged with nearby well-exposed colour.
4. **Classify structure** around blown points: a line of points is trunk or branch, and a less linear cluster is leaf material. The method follows LeWoS (Wang et al. 2020), with multi-scale linearity for thin branches.
5. **Fill blown points** from well-exposed and repaired points within 15 cm, the same class first. A second pass reaches 30 cm.
6. **Palette.** The remaining points take the colours of real points of the same class nearby, one colour per 0.3 m cellular patch.
7. **Texture, optional (`--texture`).** Steps 3 and 5 average, so recoloured areas vary less than real foliage. This step adds back variation fitted to the gap measured on the scan. It is off by default (see *The texture option*).

Every attribute other than red, green and blue is left unchanged, and so are the points that need no change. `--show classes` writes the structure classes as false colour (orange trunk/branch, green leaf, magenta pages). `--show paths` shows which step coloured each point.

Time and memory, measured in a 4-core cloud session:

| | Points | 1 worker | 3 workers | Peak memory |
|---|---|---|---|---|
| `column_3m.laz` | 4.8 M | 64 s | | 1.3 GB |
| `every100.laz` | 3.5 M | 66 s | 33 s | 1.2 GB |

Memory grows by about 0.35 GB per million points processed at once. Tiles larger than `--slab-points` (12 M) are processed in slabs, so a worker needs at most about 4.5 GB. The log reports the largest tile after the split. The full scan has not been run with this version.

## What the samples showed

These measurements, made on `column_3m.laz`, set the defaults. `python tools/recolour_eval.py data/samples/column_3m.laz --blowout 0.15` reproduces them; item 6 also needs `--tool` (see the script's docstring). Colour differences are CIEDE2000; the code passes the 34 published test pairs of Sharma, Wu & Dalal (2005).

1. **The forest's own colours are olive to green.** 80 % of well-exposed points have a CIELAB hue between 102° and 123°; only 1.2 % are brown. The previous tool's fixed brown "wood" palette therefore did not belong to this forest; it was the orange speckle in the crowns.
2. **Greying is a gradient with height.** The darkest channel, in linear light, has a median of 0.04 in the lowest quarter of points (below 0.3 m) and 0.27 above 5.8 m. Once that white is subtracted, greyed points keep the forest's hue (item 0 of the script's report gives both medians).
3. **Laser intensity does not follow visible lightness.** The median correlation within 10 cm neighbourhoods is 0.009, and intensity does not separate green from brown points. It is not used for colour or for structure.
4. **Borrowing works at short range only.** The error rises from 3.2 at 1 cm from the nearest well-exposed point to 4.4 at 4 cm and 5.8 at 24 cm. At 77 cm it is 7.7, worse than the scan's median colour (7.0); the noise floor between neighbouring well-exposed points is 1.5.
5. **Greyed points still hold their colour.** Pairs of a greyed point and a well-exposed point of the same surface, under 1 cm apart and imaged at another time, give 500,000 examples. On held-out pairs the regression lowers the error from 7.0 (left unchanged) to 4.2. It improves every veil band, most of all the palest (from 24.6 to 7.0).
6. **Compared with the previous tool**, on well-exposed points hidden under a simulated blowout (turned white), with their true colours known:

   | Simulated blowout | Previous tool | This version |
   |---|---|---|
   | 15 cm cubes, 760,000 points | 15.9 (3.3 % under 5) | 5.77 (43.8 % under 5) |
   | 40 cm cubes, 769,000 points | 16.3 (2.2 % under 5) | 7.16 (33.8 % under 5) |

   The previous tool's 24 nearest neighbours all lie inside a blown region, so it falls back to its fixed palette; this version borrows from up to 30 cm away.

   After a run, by the tool's own tests, 0.00 % of the column's points are still blown out and 0.03 % greyed.

## Where the colours come from

Every replacement colour comes from points of this scan that the camera did not wash out. Most of those lie near the ground. The near-ground forest therefore supplies the colours, though not by copying them directly:

1. **Greyed points** are repaired by the regression. It learns from pairs of the same surface seen washed out and seen well exposed. The well-exposed points of those pairs lie mostly near the ground: a median of 1.6 m above it in `every100.laz` (2.3 m in the column), 83 % below 4 m and 0.6 % above 8 m.
   - The regression is applied to each canopy point's own washed-out colour. The hue, and the differences between canopy points, therefore come from the canopy. The overall lightness and saturation come from what the regression learnt lower down.
   - Well-exposed colour from within 15 cm is averaged in where there is any. That is rare in the canopy.
2. **Blown points** take their colour from their neighbours, mostly repaired greyed points at the same height. Where there are none within 30 cm, they take a real colour of the same class from within 2 m.

The result in `every100.laz`, by height above ground (`recolour_eval.py --output`):

| | L* (median) | L* interquartile range | Chroma C* | Hue |
|---|---|---|---|---|
| Well-exposed, below 4 m | 36.1 | 14.6 | 17.8 | 113° |
| Well-exposed, 8–12 m (17,883 points) | 46.2 | 11.4 | 23.3 | 118° |
| Recoloured, 8–12 m | 44.3 | 6.0 | 22.7 | 116° |
| Recoloured, above 12 m | 44.3 | 7.4 | 23.2 | 116° |

The recoloured canopy is lighter and more saturated than the near-ground forest. Its median is close to that of the few canopy points the camera did expose well. Its lightness varies about half as much as real foliage does, because both the regression and the borrowing average. The column gives the same picture:

| | L* (median) | L* interquartile range | Chroma C* |
|---|---|---|---|
| Well-exposed, near the ground | 35.3 | 13.6 | 17.1 |
| Well-exposed, 8–12 m | 48.1 | 10.5 | 26.5 |
| Recoloured, 8–12 m | 48.3 | 6.9 | 25.8 |

After a run, 0.02 % of `every100.laz` still counts as blown out. That is the white paper of the two protected pages.

## The texture option

Without it, the recoloured canopy varies much less than well-exposed foliage at the same height. The shortfall is greatest between near neighbours: between points under 2 cm apart, recoloured points keep 9–24 % of the variation, and at 16–64 cm about half. The regression pulls towards the mean, and borrowing takes averages of neighbours.

`--texture 1` adds variation back, measured on each run:

1. A few tiles are recoloured without texture. On points at least 4 m above a simple ground model, the semivariance of colour is measured for well-exposed foliage near the washed-out areas and for the recoloured points. Semivariance is half the mean squared difference between two points, against their distance (Cressie 1993).
2. The gap is fitted with a separate value for every point plus smooth lattice noise (after Perlin 1985) at 2, 5, 12 and 30 cm. Its colour direction is the gap's own, so lightness and chroma vary together, as they do in real foliage.
   - Matching the variation scale by scale follows Heeger & Bergen (1995).
   - Adding variation, instead of stretching the predicted colours, follows Maraun (2013), who shows that stretching misrepresents spatial structure.
3. Each recoloured point gets the variation in proportion to its correction. It takes the full amount, else half, else none, so that no point becomes greyed or blown out by the tool's own tests.

Fitting on all heights made the canopy two to three times too grainy, because understory foliage is grainier; hence the 4 m limit (`--texture-above`). The noise is computed in scan coordinates, so it continues across tile and slab boundaries, and the output is identical with any number of workers. A smaller value, such as `--texture 0.5`, restores that share of the missing variance.

Measured with `recolour_eval.py --output` and `--blowout`:

| | `column_3m.laz` | `every100.laz` |
|---|---|---|
| Lightness spread (IQR) at 8–12 m: default / `--texture 1` / well-exposed | 6.9 / 9.4 / 10.5 | 6.0 / 10.1 / 11.4 |
| Canopy variation by distance, as a share of well-exposed foliage: default | 9–51 % | 16–48 % |
| Same, with `--texture 1` | 69–128 % | 80–112 % |
| Still greyed after a run (tool's own test): default / `--texture 1` | 0.03 % / 0.02 % | 0.04 % / 0.04 % |
| Simulated blowout, 15 cm cubes, median error: default / `--texture 1` | 5.77 / 6.90 | |
| Simulated blowout, 40 cm cubes, median error: default / `--texture 1` | 7.16 / 7.78 | |
| Run time, column with 2 workers and thinned sample with 3: default / `--texture 1` | 0.7 / 1.5 min | 0.5 / 0.9 min |

The added variation is not predicted from the scan, so each point's colour is less accurate. The cost on the simulated blowouts is the difference shown; the previous tool's errors there were 15.9 and 16.3.

## Survey pages in `every100.laz`

Two pages pass every test in the thinned sample. Both are flat, about 29 × 22 cm, with paper returning a laser intensity of 43–53 and print about 1. The centres, in file coordinates, are:

| x | y | z |
|---|---|---|
| −2.347 | 4.208 | 0.375 |
| 6.575 | 12.904 | 1.710 |

They have not been checked against the survey. The column sample has none. Six synthetic A4 pages were also added to the column, each with a 15 cm printed checkerboard: well exposed, overexposed, in shade, sky-tinted, a third hidden and about half hidden. All six were found, and their points came out unchanged.

## Known limits

- **No labelled wood/leaf data for this forest.** The structure classes are judged by eye (`--show classes`). During development, with a simpler linearity score than the final classifier, adding the class changed neither the regression (4.62 against 4.63) nor borrowing. Well-exposed points of high and low linearity also had similar median colours (hue 113–119° against 109–115°). The script does not reproduce these two checks. In the thinned sample, trunks inside crowns are often classed as leaf; in the full-density column the trunk is found along its whole length.
- **Pages that touch other near-white or grey surfaces can be missed.** A page pinned flush to a pale, achromatic surface, or hanging among blown-out leaves, may join up with them.
- **Recoloured areas vary less than real foliage**, unless `--texture` is used. By default their lightness spread is about half that of well-exposed points at the same height. With `--texture 1` the spread comes close to real foliage, but each point's colour is less accurate (see *The texture option*).
- **The error tables and the built-in greyed-colour model were measured on `column_3m.laz`.** The model is refitted on each run. For another scan, run `recolour_eval.py` and update `_BORROW_A`, `_BORROW_B` and `_REG_ERR` if they differ much.

## Neighbour density

Point density varies a great deal: near the scanner positions there are about 440,000 points per square metre of ground. The tool therefore gives its distances in metres (fill radius, patch, halo) and sets its linking distances from the measured point spacing. The same defaults work on the full-density column and on `every100.laz`; `--k` only caps how many sources a point takes.
