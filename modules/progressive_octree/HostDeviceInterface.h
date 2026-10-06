// modules/progressive_octree/HostDeviceInterface.h

#pragma once

#include "builtin_types.h"

struct mat4{
	float4 rows[4];
};

#define MAX_EVENTS 16

// ---- buffer sizes shared by the host and the kernels ---------------------

// Upload ring: device slots of one batch (MAX_BATCH_SIZE points, 16 MB) each,
// filled by the uploader and drained by kernel_construct. The kernel adds
// about one batch per frame, so a few slots keep it fed; the loaders decode
// further ahead into pinned host memory (MAX_DECODED_AHEAD on the host).
// Was 50 slots, 800 MB of VRAM.
constexpr uint64_t BATCH_STREAM_SIZE = 8;

// kernel_construct's scratch (momentary) buffer. The capacities live here so
// the host allocates exactly what the kernel lays out; it was a fixed 1.2 GB
// of which 409 MB was used.
constexpr uint64_t CONSTRUCT_VOXEL_BACKLOG  = 10'000'000;   // voxel samples: a 16 B point and an 8 B node pointer each
constexpr uint64_t CONSTRUCT_SPILLING_NODES = 100'000;      // node pointers, 8 B
constexpr uint64_t CONSTRUCT_SPILLED_POINTS = 10'000'000;   // 12 B stored points; written unchecked, re-inserted only up to 3 M
constexpr uint64_t CONSTRUCT_SCRATCH_FIXED_BYTES =
	  CONSTRUCT_VOXEL_BACKLOG  * (16 + 8)
	+ CONSTRUCT_SPILLING_NODES * 8
	+ CONSTRUCT_SPILLED_POINTS * 12
	+ 4096;                                                  // counters and 16-byte alignment
// After the fixed part comes the chunk queue: one 8 B entry for every point
// chunk the octree pool can hold (pool / CHUNK_BYTES + 1), because chunks
// freed by a split are queued there for reuse. It was a fixed 1 M entries,
// which a pool over 16 GB could overflow. The host sizes it from total VRAM.
constexpr uint64_t CHUNK_BYTES = 12'016;                     // sizeof(Chunk): 1000 12-byte points + header; checked in structures.cuh

// kernel_render's scratch starts with a list of up to MAX_VISIBLE_NODES copies
// of struct Node. The host sizes that buffer from these (renderBufferBytesFor),
// so a field added to Node must update NODE_BYTES: when the LOD cross-fade
// grew Node from 152 to 168 B, every frame under ~2.1 Mpx overran the buffer
// and drew nothing. Fisheye disables frustum culling, so wide lenses inside a
// large cloud can exceed the cap; writes past it are dropped.
constexpr uint32_t MAX_VISIBLE_NODES = 100'000;
constexpr uint64_t NODE_BYTES        = 168;                  // sizeof(Node); checked in structures.cuh

struct Uniforms{
	float width;
	float height;
	float insectPhase[6];   // radians in [0, 2 pi): the insects' six motion phases (getUniforms)
	float fovy_rad;
	mat4 world;
	mat4 view;
	mat4 proj;
	mat4 transform;
	mat4 transform_updateBound;
	mat4 transformInv_updateBound;

	uint64_t persistentBufferCapacity;
	uint64_t momentaryBufferCapacity;
	uint64_t renderBufferCapacity;      // kernel_render draws nothing if its layout would not fit

	uint64_t frameCounter;
	
	float3 boxMin;
	float3 boxMax;

	bool showBoundingBox;
	bool showPoints;
	bool colorByNode;
	bool colorByLOD;
	bool useHighQualityShading;
	float minNodeSize;
	int pointSize;
	bool enableEDL;
	float edlStrength;

	// tier 1 shading
	bool  roundSplats;
	bool  enableFog;
	float fogDensity;        // haze extinction per metre, at and below fogBase
	float fogR;              // haze colour, linear (the host converts the picker's sRGB)
	float fogG;
	float fogB;
	float fogHeightFalloff;  // per metre: 0 = even haze; above 0 it thins with height
	float fogBase;           // metres above the lowest point: full density at and below
	int   fogSky;            // 1 = the haze also lies between the camera and the sky
	bool  enableBloom;
	float bloomStrength;
	float bloomRadius;

	float exposure;

	// excitation (inserted by apply_emission.ps1). Parallel arrays rather
	// than a struct array: no alignment surprises in the uniform block.
	int   numEvents;
	float emissionGain;
	int   numInsects;
	int   insectSize;
	float insectGlow;

	// voxel occupancy grid (apply_voxel_grid.ps1). One uint32 count per cell,
	// built on the GPU as points enter the octree. Passed as an integer so the
	// struct stays plain data on both sides of the host/device boundary.
	uint64_t voxelGridPtr;
	int      gridX;
	int      gridY;
	int      gridZ;
	float    cellSize;
	bool     showVoxelGrid;
	int      insectPlacement;   // 0 anywhere, 1 in air near foliage
	int      gridRefill;        // 1 = rebuild the occupancy counts from the octree
	int      numNodesTotal;     // nodes in the octree, for the refill walk
	bool     showEventMarkers;
	bool     useVoxelGrid;      // gates the cull, insect placement, grid view and medium; not the counting
	float    pointDensity;      // fraction of scanned points drawn, 0..1

	// volumetric medium (apply_volumetric.ps1)
	mat4     transformInv;      // clip -> world, for building view rays
	float3   camPos;
	float3   camDir;
	bool     enableVolumetric;
	float    volDensity;        // scattering coefficient of the air
	float    volHeightFalloff;  // >0 pools the medium near the ground
	float    volOcclusion;      // extra extinction inside occupied voxels
	int      volSteps;          // march steps, quarter resolution
	float    volMaxDist;
	float    volScatter;        // brightness, independent of density
	int      volShadowSteps;    // 0 = no shadowing; light passes through wood
	float    falloffPower;      // shape of the radial curve; 2 = the old fixed one
	float    coreBoost;         // extra inverse-square hot spot at the centre
	int      minNeighbours;     // cull points with fewer than this many in their cell
	int      cullReach;         // 0 own cell, 1 adds the six axis neighbours
	bool     showCulledOnly;    // invert: show ONLY what would be culled

	// colour grading + sky (apply_features.ps1). Applied in the composite in
	// linear light, after emission and bloom, before exposure and tone mapping.
	float    saturation;        // 1 = as scanned, 0 = greyscale, 2 = doubled
	float    temperature;       // -1 cool .. 0 neutral .. +1 warm
	float    tintR;             // tint colour, linear
	float    tintG;
	float    tintB;
	float    tintMix;           // 0 = scan colour only .. 1 = tint only
	float    skyR;              // background, sRGB 0..1
	float    skyG;
	float    skyB;

	// look refinement
	float    pointWorldSize;    // splat diameter in metres; 0 = fixed pixel size
	float    pointShading;      // 0 flat disc .. 1 fully shaded sphere
	float    focalPx;           // height / (2 tan(fovy/2)), for projecting sizes
	int      maxSplatPx;        // cap on projected splat size
	float    grain;             // film grain amplitude
	float    vignette;          // corner darkening 0..1
	float    toneCurve;         // 0 none (brightness limited at white, hue kept) .. 1 the ACES fit
	int      showUnedited;      // 1 = the unedited view: stored colours; no light, shading, repair, haze, grade, insects or cull

	// directional key light
	uint64_t lightGridPtr;      // per-cell transmittance toward the light, 0..1
	float    lightDirX;         // unit vector TOWARD the light
	float    lightDirY;
	float    lightDirZ;
	float    lightR;            // light colour, linear
	float    lightG;
	float    lightB;
	float    lightIntensity;
	float    ambient;           // floor, so shadowed points (and haze) are not black
	float    lightAbsorb;       // extinction per metre inside occupied cells
	int      lightSteps;        // cells marched toward the light
	int      lightDirty;        // 1 = rebuild the transmittance grid this frame
	float    sunbeams;          // key light scattered by the haze; 0 = haze gets ambient only
	float    sunAnisotropy;     // Henyey-Greenstein g: 0 even, toward 1 brighter facing the light

	// blown-out colour repair: the scanner's camera whites out the sunlit canopy
	uint64_t colourGridPtr;     // per occupancy cell, uint64: see render.cu, colour repair
	int      whiteFix;          // 1 = recolour blown-out points from their neighbours
	float    whiteLevel;        // darkest channel at or above this ...
	float    whiteSpread;       // ... and channels within this of each other = blown out
	int      whiteReach;        // cells searched for a colour when a cell has none
	float    tintMargin;        // blue above green by more than this = sky cast; 0 = off

	// torch: a spotlight from the camera, aimed through the mouse pixel
	int      torchOn;
	float    torchX;            // frame pixel the torch points through (y up, as pickY)
	float    torchY;
	float    torchIntensity;    // added light at the beam centre, 1 m away
	float    torchCosInner;     // cos of the half-angle of the full-strength core
	float    torchCosOuter;     // cos of the half-angle where the beam ends
	float    torchRange;        // metres at which the light has fallen to half

	// key light switch, near clip, channel gains
	int      lightOn;           // key light switch
	float    nearClip;          // metres: points nearer than this are not drawn
	float    mixR;              // channel gains, applied in the composite
	float    mixG;
	float    mixB;

	// lens (apply_fisheye.ps1)
	int      lensMode;          // 0 perspective, 1 fisheye
	float    fisheyeFov;        // degrees reaching the top and bottom edge
	float3   camRight;          // camera basis in world space, for building rays
	float3   camUp;

	// navigation and markers (apply_navigate.ps1)
	int      pickX;             // pixel to pick, -1 = none; held until answered
	int      pickY;
	int      pickSeq;           // request id; the kernel answers each id once
	int      numMarkers;
	int      showMarkers;       // apply_markers2.ps1: 0 hides the balls, markers are kept
	float    markerX[32];       // placed markers, renderer frame (apply_markers2.ps1: 32)
	float    markerY[32];
	float    markerZ[32];
	float    belowR;            // colour of the void below the horizon
	float    belowG;
	float    belowB;

	float    markerScale;       // marker cluster size, metres
	float    horizonSoftness;   // degrees the sky-to-void blend spans; 0 = hard edge
	float eventX[MAX_EVENTS];
	float eventY[MAX_EVENTS];
	float eventZ[MAX_EVENTS];
	float eventR[MAX_EVENTS];
	float eventI[MAX_EVENTS];
	float eventCR[MAX_EVENTS];
	float eventCG[MAX_EVENTS];
	float eventCB[MAX_EVENTS];
};

struct Stats{
	uint32_t frameID                    = 0;
	uint32_t numNodes                   = 0;
	uint32_t numInner                   = 0;
	uint32_t numLeaves                  = 0;
	uint32_t numNonemptyLeaves          = 0;
	uint32_t numPoints                  = 0;
	uint32_t numVoxels                  = 0;
	uint64_t allocatedBytes_momentary   = 0;
	uint64_t allocatedBytes_persistent  = 0;
	uint32_t numVisibleNodes            = 0;
	uint32_t numVisibleInner            = 0;
	uint32_t numVisibleLeaves           = 0;
	uint32_t numVisiblePoints           = 0;
	uint32_t numVisibleVoxels           = 0;
	uint32_t numEmitting                = 0;   // apply_insects.ps1
	uint32_t numDrawn                   = 0;   // apply_sweep_fixes.ps1
	uint64_t stageNs[8]                 = {0,0,0,0,0,0,0,0};   // apply_stage_timers.ps1: ns since kernel start at each stage boundary
	int      pickValid                  = 0;   // apply_navigate.ps1: a pick landed on geometry
	float    pickWX                     = 0;   // its position, renderer frame
	float    pickWY                     = 0;
	float    pickWZ                     = 0;
	int      pickSeqDone                = 0;   // id of the last pick request answered (hit or miss)
	uint32_t numChunksPoints            = 0;
	uint32_t numChunksVoxels            = 0;

	uint32_t batchletIndex              = 0;
	uint64_t numPointsProcessed         = 0;
	uint64_t numAllocatedChunks         = 0;
	uint64_t chunkPoolSize              = 0;
	uint32_t dbg                        = 0;
	uint32_t numDroppedPoints           = 0;   // points whose leaf had no chunk for them; not stored
	uint32_t numRingMismatches          = 0;   // upload-ring slots refilled before the kernel had finished their batch
	uint32_t numClampedPoints           = 0;   // points outside the box, stored on its faces (header bounds too small)

	bool memCapacityReached             = false;
};