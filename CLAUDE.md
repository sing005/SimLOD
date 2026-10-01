# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A modified fork of [m-schuetz/SimLOD](https://github.com/m-schuetz/SimLOD) (MIT): a CUDA point-cloud viewer that builds an octree LOD on the GPU while loading LAS/LAZ/.simlod files and renders it in real time. Local work adds a fisheye lens, key light + volumetrics, bloom/"look" grading, an occupancy voxel grid (stray-point cull, insect placement, light transmittance), survey markers with CSV I/O and a Horn survey-frame fit, walk navigation, find/isolate, and call-schedule emission for the Rotokare acoustic project.

Git remotes: `upstream` = m-schuetz/SimLOD. The user's own repo is private. Work on feature branches; `main` holds the user's baseline, not upstream.

## Build and run (Windows, VS 2022, CUDA 12.x)

```
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

- Build **Release**. Debug currently fails (CMake adds `/O2` in every config, which clashes with `/RTC1`).
- Requires CUDA ≥ 12.0 (links `CUDA::nvJitLink`) and CMake ≥ 3.26, despite what `CMakeLists.txt` declares.
- There are no tests and no linter. Verification means building and running, then dropping a point cloud onto the window.
- Post-build copies `modules/` next to the exe. The `.cu` files are compiled **at runtime** with NVRTC, so CUDA edits do not need a C++ rebuild.

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

**Hot reload.** `CudaModularProgram` (`include/CudaModularProgram.h`) watches each `.cu` file and recompiles it when it changes. This only works if the working directory is the repo root, because the files are loaded from `./modules/...`. The GPU architecture is hardcoded to `compute_86` there.

**Shared host/device structs.**
- `HostDeviceInterface.h` defines `Uniforms` (per-frame settings passed **by value** as a kernel parameter) and `Stats` (counters and pick results written by the GPU).
- Both structs are compiled by MSVC and NVRTC, so field order and types must stay layout-compatible.
- `Uniforms` is about 2 KB against the 4 KB kernel-parameter limit.
- Fixed arrays in these structs, such as `MAX_EVENTS` and the 32-entry marker arrays, are bounds-checked on the host.

**Per-frame flow.**
1. `getUniforms()` copies `settings` into `Uniforms`. It also lazily (re)allocates the voxel and light grids via `ensureVoxelGrid` / `ensureLightGrid` and sets the `g_gridRefill` / `g_lightDirty` flags.
2. `updateOctree()` and `renderCUDA()` launch the kernels.
3. `Stats` is read back with `cuMemcpyDtoHAsync`, so host reads lag the GPU by a frame or more.

`getUniforms()` is called several times per frame, so side effects in it repeat.

**Global state.** Host state lives in file-scope globals prefixed `g_` (grids, markers, call schedule, survey fit). The rendering settings struct is `settings` (anonymous struct at about line 124).

**Runtime outputs.** These are written to the working directory and are git-ignored:
- `log/simlod_<timestamp>.log`, written by `logLine()`, which also installs signal handlers.
- `markers.csv`.

## Working rules for this repo

- Edit source files directly and commit to git. Do not use or extend the old text-replacement patch scripts (`../apply_*.ps1`, `../fix_*.ps1`). Do not create `.bak` copies, because git history replaces them.
- Strip provenance comments of the form "added by apply_xxx.ps1" when touching nearby code. Do not add new ones.
- Changes to `Uniforms` or `Stats` must be made in `HostDeviceInterface.h` only, with matching host fill code in `getUniforms()`.
- The high-quality point path in `render.cu` duplicates the point filtering (density, stray cull, find/isolate) from `drawPoint`. Change both copies together.
- Point-cloud data (`*.las`, `*.laz`) is never committed. Test data lives outside the repo in `../Laspy/` and `../rotokare_test_kit/`.
