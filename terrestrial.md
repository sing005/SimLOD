# terrestrial.md

This file provides guidance for working with code in this repository.

## What this is

A modified fork of [m-schuetz/SimLOD](https://github.com/m-schuetz/SimLOD) (MIT): a CUDA point-cloud viewer that builds an octree LOD on the GPU while loading LAS/LAZ/.simlod files and renders it in real time. Local work adds a fisheye lens, key light + volumetrics, bloom/"look" grading, an occupancy voxel grid (stray-point cull, insect placement, light transmittance), survey markers with CSV I/O and a Horn survey-frame fit, walk navigation, find/isolate, and call-schedule emission for the Rotokare acoustic project.

Git remotes: `upstream` = m-schuetz/SimLOD. The user's own repo is private. Work on feature branches; `main` holds the user's baseline, not upstream.

## Build and run (Windows, VS 2022, CUDA 12.x)

```
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build . --config Release
```

- The policy flag is needed with CMake 4.x, because the fetched GLFW declares `cmake_minimum_required` < 3.5.
- The CUDA `.cu` files cannot be syntax-checked with plain `nvcc`. NVRTC compiles them with device as the default execution space. To check them, run the built exe from the repo root for a few seconds and look for `=== COMPILING:` lines followed by errors in stdout.
- Without a GPU (for example in a cloud session), the PyPI wheels `nvidia-cuda-nvrtc-cu12`, `nvidia-cuda-runtime-cu12`, `nvidia-curand-cu12` and `nvidia-cuda-cccl-cu12` (all 12.4) supply NVRTC and the headers. Merge the wheels' `include/` folders into one directory, then call `nvrtcCompileProgram` (for example through Python `ctypes`) with the options listed in `CudaModularProgram::compile()`. This compiles the `.cu` files without a GPU.

- Build **Release** for normal use. Debug is much slower, but it is useful for host-side debugging. Release also writes `SimLOD.pdb`; keep it next to the exe so crash traces have line numbers.
- Requires CUDA ≥ 12.0 (links `CUDA::nvJitLink`) and CMake ≥ 3.26. At runtime `CUDA_PATH` must point to a CUDA 12 toolkit, because NVRTC compiles the kernels against its headers.
- There are no tests and no linter. Verification means building and running.
- Post-build copies `modules/` next to the exe. The `.cu` files are compiled **at runtime** with NVRTC, so CUDA edits do not need a C++ rebuild.
- CI (`.github/workflows/build.yml`) builds Release on Windows for each pull request and uploads a runnable bundle. It does not compile the `.cu` files.

**Running reproducibly** (run from the repo root so `./modules` resolves):

```
build\Release\SimLOD.exe cloud.las --view <yaw> <pitch> <radius> <tx> <ty> <tz> --set hqs=0 --set splat=8 --set adaptive=0 --set vsync=0
```

- Files load as if dropped on the window.
- `--view` places the camera. The target is in the renderer frame, i.e. relative to the cloud's minimum corner.
- `--set` accepts `hqs`, `splat`, `detail`, `adaptive`, `budget` and `vsync`, and for the haze and key light `haze`, `hazefall`, `hazefloor`, `hazesky`, `light` and `sunbeams`.
- Timings go to the run log as `[host] stage ms:` lines. Measure only after loading finishes: frame cost changes a lot while the octree is still being built.
- Kill with `timeout N`. Stdout is buffered and lost on kill, but the log file is flushed every line.
- `.laz` decompresses slowly (about 4 min for 54 M points). Convert to `.las` with `laspy` for repeated runs.

## Architecture

**Host and device split.** `modules/progressive_octree/main_progressive_octree.cpp` is the whole host application: `main()`, the ImGui UI, loader and uploader threads (`spawnLoader`, `spawnUploader`), GPU buffer allocation, and kernel launches. The GPU side consists of three runtime-compiled programs, each `CudaModularProgram` built from a `.cu` file plus `utils.cu`:

| Program | Kernel | Role |
|---|---|---|
| `progressive_octree_voxels.cu` | `kernel_construct` | inserts uploaded point batches into the octree; also fills the occupancy voxel grid |
| `render.cu` | `kernel_render` | visibility, point/voxel rasterisation, overlays, post-processing (EDL, volumetric, bloom, composite) |
| `reset.cu` | `kernel` | clears octree state on reset/drop |

**Kernel conventions.**
- All kernels are launched with `cuLaunchCooperativeKernel` and synchronise across the whole grid with `grid.sync()`. Any early `return` before a `grid.sync()` must be uniform across all threads.
- Work is split with `processRange` from `utils.h.cu`.

**Hot reload.** `CudaModularProgram` (`include/CudaModularProgram.h`) watches each `.cu` file and recompiles it when it changes. This only works if the working directory is the repo root, because the files are loaded from `./modules/...`. It compiles for the present GPU's compute capability.

**Shared host/device structs.**
- `HostDeviceInterface.h` defines `Uniforms` (per-frame settings passed **by value** as a kernel parameter) and `Stats` (counters and pick results written by the GPU).
- Both structs are compiled by MSVC and NVRTC, so field order and types must stay layout-compatible.
- `Uniforms` is about 2 KB against the 4 KB kernel-parameter limit.
- Fixed arrays in these structs, such as `MAX_EVENTS` and the 32-entry marker arrays, are bounds-checked on the host.

**Per-frame flow.**
1. `beginFrameState()` runs once per frame. It (re)allocates the voxel and light grids, settles the one-shot `g_gridRefill` / `g_lightDirty` requests into `g_frameGridRefill` / `g_frameLightDirty`, and updates adaptive detail.
2. `renderCUDA()` and `updateOctree()` launch the kernels. Each calls `getUniforms()`, which copies `settings` and frame state into `Uniforms`.
3. `Stats` is read back with an unsynchronised `cuMemcpyDtoHAsync`, so host reads lag by a frame or more and a single read can be half-copied. Results the host must not miss use a request id that the kernel answers once and leaves in place (see `pickSeq` / `pickSeqDone`).

`getUniforms()` runs for every kernel launch, so it must stay free of side effects such as allocation and flag clearing; those belong in `beginFrameState()`.

**Haze, sunbeams and the composite.**
- The haze (`fog*` uniforms) has a density that depends only on height, so `hazeOpticalDepth()` computes its optical depth along a ray in closed form. The composite applies its extinction and its ambient light exactly, at full resolution.
- The key light's share of the haze is shaded by the canopy, so it varies along the ray. The volumetric pass marches it through the light grid, alongside the calls' medium, at quarter resolution.
- A ray that hits no point (sky, or the void below the horizon) gets the haze, its ambient light and its sunbeams only with Haze over sky (`fogSky`) on. The march's dither is fixed per pixel; a per-frame one made the lit haze flicker.
- `volBuf` holds RGB plus the distance to the surface each ray sees (`SKY_DISTANCE` for sky). The composite upsamples it with weights from that distance (joint bilateral), so lit haze does not spread across silhouettes.
- The composite keeps the backdrop (sky, or the void below the horizon) apart from the scene light and grades only the scene. Anything added in front of the sky is graded the same as when it is in front of geometry.

**Adaptive detail.**
- `updateAdaptiveDetail()` steers `g_adaptReduce` from the render kernel's time against `settings.frameBudgetMs`.
- It scales both the LOD threshold and the splat-size cap.
- On dense scans viewed up close, the cost is splatting the nearest full-resolution leaves, which LOD alone cannot reduce.
- High-quality shading (on by default) draws every point twice, in a depth pass and a colour pass.

**Global state.** Host state lives in file-scope globals prefixed `g_` (grids, markers, call schedule, survey fit). The rendering settings struct is `settings` (anonymous struct at about line 124).

**Runtime outputs.** These are written to the working directory and are git-ignored:
- `log/simlod_<timestamp>.log`, written by `logLine()`.
  - Periodic status lines go through `logIfChanged()`.
  - Each run is capped at 20 MB, and only the newest 10 logs are kept.
  - `CrashHandler.cpp` appends a symbolised stack trace on any unhandled exception. Start any crash investigation there.
- `markers.csv`.

## Working rules for this repo

- Edit source files directly and commit to git. Do not use or extend the old text-replacement patch scripts (`../apply_*.ps1`, `../fix_*.ps1`). Do not create `.bak` copies, because git history replaces them.
- Strip provenance comments of the form "added by apply_xxx.ps1" when touching nearby code. Do not add new ones.
- Changes to `Uniforms` or `Stats` must be made in `HostDeviceInterface.h` only, with matching host fill code in `getUniforms()`.
- Point filtering exists twice in `render.cu`: in `drawPoint` for the standard path, and in `hqsAcceptPoint` for both high-quality passes. A change to which points are drawn must go into both.
- Files under `modules/` and `include/` use CRLF line endings. `sed -i` in Git Bash rewrites them as LF, so use the Edit tool, or restore CRLF before committing.
- Point-cloud data (`*.las`, `*.laz`) is never committed. Test data lives outside the repo in `../Laspy/` and `../rotokare_test_kit/`. The one exception is `data/samples/`: small cuts of a scan, each under 50 MB, kept in git so the canopy recolour can be developed in cloud sessions. Those sessions cannot fetch Git LFS files.

## Canopy recolour (`tools/recolour_canopy.py`)

An offline Python tool, separate from the viewer. It rewrites the colour of two kinds of point and leaves every other point and attribute unchanged:
- points the scanner's camera blew out in the sunlit canopy (white, or a lilac/blue sky cast);
- greyed points (pale or unsaturated), which fade from the floor up into the white canopy.

The docstring gives the method. In order, the steps are:
1. Detect blown-out and greyed points.
2. Keep printed survey pages unchanged.
3. Repair greyed points with a regression learnt per run.
4. Classify structure around blown points: a line of points is trunk or branch, a less linear cluster is leaf material.
5. Borrow colour from points within 15 cm.
6. Fill the rest with patches of real nearby colours.
7. Optionally (`--texture`, off by default) add back the colour variation that steps 3 and 5 average away. The amount is fitted per run, scale by scale, to the gap between well-exposed and recoloured points at least 4 m above ground.

- Install with `pip install -r tools/requirements-recolour.txt` (laspy with lazrs, numpy, scipy).
- Develop on `data/samples/`. Its README describes the files, their coordinates and the colour encoding, and records the measurements behind the defaults. For example: `python tools/recolour_canopy.py data/samples/every100.laz out.laz`.
- `tools/recolour_eval.py` reproduces those measurements. Run it after changing the method, for example `python tools/recolour_eval.py data/samples/column_3m.laz --output out.laz --blowout 0.15`.
  - It checks the CIEDE2000 code against the published test pairs.
  - It scores colour borrowing on hidden well-exposed points.
  - It scores the greyed-colour regression on held-out pairs.
  - It scores the whole tool on a simulated blowout. Use `--tool` to score another version, and `--tool-args "--texture 1"` to pass options.
  - The error tables `_BORROW_A`, `_BORROW_B` and `_REG_ERR` in the tool come from its output.
- Diagnostics:
  - `--show classes` writes the structure classes as false colour: orange trunk/branch, green leaf, magenta pages.
  - `--show paths` shows which step coloured each point.
  - Detected page centres go to `OUT_pages.csv`, in the viewer's `markers.csv` format.
- The full scans (3–4 GB, 346–455 M points) live only on the local machine.
  - The tool streams them through 16 m ground tiles with a 0.6 m halo. A tile over `--slab-points` (12 M own points) is read from disk one slab at a time.
  - Memory is about 0.32 GB per million points a worker holds at once: a slab's own points plus those within 0.6 m of it. Near the scanner the slabs are about 1 m wide and their context holds more points than they do, so the full scan needs up to about 27 M points, about 10 GB, per worker. The log reports this after the split.
  - A quick test on a full scan: `--crop 8 -4 12 0`, about 3.4 M points. A patch around the scanner such as `--crop -10 -10 10 10` holds half the scan.
  - `--workers N` processes tiles in parallel and gives identical output.
  - This version has not been run on a full scan. The samples take about a minute each with one worker in a 4-core cloud session; the previous version took about 15 minutes for a full scan on the 20-thread laptop.
- Judge the results in SimLOD at 4K, with Scan clean-up > "Fix blown-out canopy" off. That checkbox is a separate, on-the-fly repair inside the viewer.
