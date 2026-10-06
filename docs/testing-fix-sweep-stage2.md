# Testing the new version of SimLOD and the recolour tool

Branch `fix/sweep-stage2` on GitHub. The code tested here is at commit `38737bd` (6 October 2026); the commit after it adds only this file.

This sheet lists what to check on the laptop before the new version replaces `main`. Most tests compare two versions:

- **before**: your current build of `main`, as you use it now;
- **after**: the new version, downloaded from GitHub (section 1).

The "before" version should be `main` as it is on GitHub now: `git log -1 --oneline` in your SimLodMod folder should show `0dd0c6d Merge pull request #8`. If it shows something older, run `git pull` and rebuild first.

Neither version changes your scans. Paths such as `C:\scans\WhiteCanopy.las` and `C:\compare` are examples; use your own. Any scan you know well will do for the viewer tests, and a `.las` loads much faster than a `.laz`.

Tests 1 to 8 take about an hour and a half, plus one unattended run of the recolour tool on a full scan.

## 1. Getting the new version onto the laptop

The new version is already on GitHub, so nothing needs to be uploaded from the laptop. The laptop does not have it until you download it. `main` stays as it is, on GitHub and on the laptop, until the new version is merged into it.

### The viewer, ready to run

1. Open <https://github.com/sing005/SimLodMod/actions/runs/37430393791>. This is the build of `fix/sweep-stage2` made on 6 October.
2. Under **Artifacts**, download **SimLOD-windows-x64**, a zip of about 4 MB. GitHub keeps it until 20 October.
3. Unzip it into a new folder, for example `C:\SimLOD-after`. The folder holds `SimLOD.exe`, `laszip.dll`, `SimLOD.pdb` and a `modules` folder.
4. Start `SimLOD.exe` from inside that folder: double-click it, or in a command window type `cd /d C:\SimLOD-after` first. Its run logs go into `C:\SimLOD-after\log`.

It needs the same CUDA set-up as your current build: CUDA 12.4, with `CUDA_PATH` set and CUDA's `bin` folder on `PATH`. Your current build is not changed.

### The recolour tool

Open <https://github.com/sing005/SimLodMod/blob/fix/sweep-stage2/tools/recolour_canopy.py>, click **Download raw file**, and save it in your SimLodMod folder as `recolour_new.py`. Your current `tools\recolour_canopy.py` is the "before" version.

### With git instead (optional)

In your SimLodMod folder:

```
git fetch origin
git worktree add ..\SimLodMod-after origin/fix/sweep-stage2
```

This puts the new version in a separate folder, `SimLodMod-after`, beside your current one, which stays on `main`. Build it there in the usual way.

## 2. Checklist

| # | Test | Time | Passed? |
|---|---|---|---|
| 1 | Starts, and starts sooner the second time | 5 min | |
| 2 | The same picture as before | 20 min | |
| 3 | Key light: shade under tall canopy | 10 min | |
| 4 | Less work on a still view | 10 min | |
| 5 | A damaged file is refused | 5 min | |
| 6 | Markers and calls | 10 min | |
| 7 | Closing | 1 min | |
| 8 | Recolour tool | 20 min, then a full run | |
| 9 | Overnight with insects (optional) | overnight | |

If a test fails, send terrestrial the run log (the newest file in the `log` folder) and a screenshot or a description of what you saw.

## 3. The tests

### Test 1. Starts, and starts sooner the second time

The new version keeps its compiled GPU programs between runs. The old one compiled them again at every start, which took about 8 seconds in the cloud test.

1. Start the new version, load a scan, and close it.
2. Start it again.
3. Open the two newest files in `C:\SimLOD-after\log`.

**Expected:** the first log has lines beginning `[kernels] compiled` and `[kernels] linked`. The second has lines containing `loaded from the cache` instead, and the window is ready sooner.

### Test 2. The same picture as before

Most of the speed changes are meant to leave the picture exactly as it was. Make the folder `C:\compare` first, then in a command window:

Before, from your SimLodMod folder:

```
build\Release\SimLOD.exe "C:\scans\WhiteCanopy.las" --render 3840 2160 --set adaptive=0 --screenshot C:\compare\before_1.bmp --exit
```

After:

```
cd /d C:\SimLOD-after
SimLOD.exe "C:\scans\WhiteCanopy.las" --render 3840 2160 --set adaptive=0 --screenshot C:\compare\after_1.bmp --exit
```

Then run the "before" command once more, with `before_2.bmp` in place of `before_1.bmp`.

Each run loads the scan, takes a 4K screenshot of the unedited view 3 seconds after loading has finished, and closes. `--set adaptive=0` keeps the level of detail fixed; without it, the faster version would show more detail.

Open the three screenshots and switch between them, for example in Photos with the arrow keys.

**Expected:** `after_1` looks the same as `before_1`, apart from two things:

- the outermost two rows and columns of pixels. The old version left out points whose centres fell there; the new one draws them.
- small scattered differences in distant, coarser areas. Compare them with the differences between `before_1` and `before_2`. If they are of the same kind, the changes did not cause them: the coarser levels keep the first point to reach each cell, and that order can vary from one run to the next.

Any other difference, such as a change of colour or brightness, or an area missing, is a fault. Send the three screenshots to terrestrial.

### Test 3. Key light: shade under tall canopy

With the key light on, the old version cut off shade from canopy high above a point. For the Rotokare scans with the default light direction, canopy more than about 28 m above a point gave it no shade at grid resolution 256 (the default), and more than about 14 m above at 512. The new version takes shade from any height in the scan.

1. Start both versions with the same scan, using the commands from test 2 without `--screenshot` and `--exit`. Both then start from the same camera position.
2. Under **Light**, tick **Key light on**, and set the same Azimuth, Elevation and Canopy density in both.
3. Look at the ground and the lower trunks under the tallest trees.
4. Set **Grid resolution** (under Scan clean-up) to 512 in both, and look again.

**Expected:**

- More shade under tall canopy in the new version.
- In the new version, the shade looks much the same at 256 and 512, only finer at 512. In the old one, raising the resolution removed the shade of the upper canopy.
- Changing the light makes the new version rebuild its shade from the full height of the scan. At 512, with the light low in the sky, this can take longer than before.
- With **Eye dome lighting** on (under Points), the shading now covers the whole frame. The old version left parts of the frame's edges, up to 15 pixels wide, without it.

### Test 4. Less work on a still view

The old version kept one processor core busy all the time, even with nothing loading, and drew a still view again on every frame. The new one does neither.

1. In the new version, load a scan and wait until loading has finished.
2. Leave the view still for a minute, with no insects, grain or calls playing.
3. Open Task Manager (Ctrl+Shift+Esc).

**Expected:**

- SimLOD's CPU use is well below the old version's. The old one used about one core's worth, around 5 % on a 20-thread processor.
- Under **Detail and speed**, beside **Skip unchanged frames**, the share of recent frames kept is near 100 %.
- GPU use (Performance tab) is lower than, or the same as, with the old version on the same still view.
- Moving the camera is at least as smooth as before.
- When the window is minimised, SimLOD uses almost no CPU.

Optional: the run log's `[host] stage ms:` lines give the time each step of a frame takes. Compare them between the two versions at the same view, after loading has finished.

### Test 5. A damaged file is refused

A `.laz` that is cut short, for example by an interrupted copy, crashed the old version. The new one refuses it and keeps the scan already on screen. Do not try this with the old version: a cut-short `.laz` can crash it.

1. Make a cut-short copy of the small sample. In PowerShell, in your SimLodMod folder:

   ```
   Copy-Item data\samples\every100.laz $env:TEMP\cut.laz
   $f = [IO.File]::Open("$env:TEMP\cut.laz", 'Open'); $f.SetLength(20MB); $f.Close()
   ```

2. In the new version, load any scan. Then drag `cut.laz` onto the window from your temporary folder (type `%TEMP%` in Explorer's address bar to open it).

**Expected:** for a few seconds the window shows `Not loaded - cut.laz: cut short: at least 18.3 MB missing from the end - keeping the current cloud`, and the scan you had stays on screen. The log has a line beginning `[load] not loading`.

### Test 6. Markers and calls

The new version looks for `markers.csv` and `calls.csv` in the folder it was started from. Copy yours into `C:\SimLOD-after` first.

1. Under **Markers and survey**, click **Save markers.csv**, then **Load markers.csv**. **Expected:** the same markers come back. The message after saving gives the file's full path.
2. If you use calls in survey coordinates: load your calls, then change one board's survey coordinates and fit the survey again. **Expected:** the calls move straight away. The old version left them where they were until `calls.csv` was loaded again.
3. Turn on **Show bounding box** (under Display and debug), and place a marker on a part of the scan that a box line crosses. **Expected:** the marker lands on the scan, not on the line.

### Test 7. Closing

Close the window in the usual way, a few times during these tests, including once while a scan is still loading.

**Expected:** it always closes without an error message. The old version could crash on closing while its loading threads were still running.

### Test 8. Recolour tool

**Same result as before, on a test patch (about 20 minutes).** These commands cut the patch into slabs of 1 million points, so the new way of sharing slabs between workers is used. In a command window, in your SimLodMod folder:

```
python tools\recolour_canopy.py "C:\scans\WhiteCanopy.laz" C:\compare\crop_before.laz --crop 8 -4 12 0 --slab-points 1000000
python recolour_new.py "C:\scans\WhiteCanopy.laz" C:\compare\crop_after.laz --crop 8 -4 12 0 --slab-points 1000000 --workers 3
fc.exe /b C:\compare\crop_before.laz C:\compare\crop_after.laz
```

**Expected:** `FC: no differences encountered`. If the runs also wrote `crop_before_pages.csv` and `crop_after_pages.csv`, compare those two in the same way. In PowerShell, type `fc.exe`, not `fc`.

**The full scan (unattended):**

```
python recolour_new.py "C:\scans\WhiteCanopy.laz" C:\out\WhiteCanopy_recoloured.laz --workers 2
```

- Choose the number of workers so that their memory stays well below the laptop's. Each worker needs about 10 GB for the full scan (an estimate), so use 2 workers with 32 GB of memory, or up to 4 with 64 GB. A few minutes into the run, after it has split the scan into tiles, the log gives the memory each worker needs and the disk space the rest of the run needs.
- The run stops at that point, with a message, if there is not enough disk space for its working files. WhiteCanopy needs about 15 GB.
- At the end the log gives the total time (`done in ... min`). Please note it. This version spreads the densest part of the scan over all the workers, which makes a full run faster only with 4 or more workers. With 2 or 3 workers it should take about as long as before (estimates from the thinned sample, not yet measured on a full scan).

**Expected:** the run finishes, and the result loads in SimLOD and looks as you expect at 4K unedited. Please send terrestrial the console text whether it passes or not: it holds the first measurements of this version on a full scan.

Optional, a few seconds: `python recolour_new.py data\samples\every100.laz data\samples\every100.laz` should stop at once with `error: the output would overwrite the input`. The old version would have overwritten the scan.

### Test 9. Overnight with insects (optional)

The old version's clock lost precision over long runs, so after a day or more the insects moved unevenly. Leave the new version running overnight with insects on (under **Insects**).

**Expected:** the next morning they still move smoothly.

## 4. After testing

Tell terrestrial which tests passed. If they all pass, the next step is to merge the new version into `main` on GitHub, through a pull request that terrestrial can open. After the merge, `git pull` in your SimLodMod folder brings `main` on the laptop up to date, and `C:\SimLOD-after` can be deleted. If you used git instead, remove the extra folder with `git worktree remove ..\SimLodMod-after`.
