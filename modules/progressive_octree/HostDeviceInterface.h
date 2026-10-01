// modules/progressive_octree/HostDeviceInterface.h

#pragma once

#include "builtin_types.h"

struct mat4{
	float4 rows[4];
};

#define MAX_EVENTS 16

struct Uniforms{
	float width;
	float height;
	float time;
	float fovy_rad;
	mat4 world;
	mat4 view;
	mat4 proj;
	mat4 transform;
	mat4 transform_updateBound;
	mat4 transformInv_updateBound;

	uint64_t persistentBufferCapacity;
	uint64_t momentaryBufferCapacity;

	uint64_t frameCounter;
	
	float3 boxMin;
	float3 boxMax;

	bool showBoundingBox;
	bool showPoints;
	bool colorByNode;
	bool colorByLOD;
	bool colorWhite;
	bool doUpdateVisibility;
	bool doProgressive;
	float LOD;
	bool useHighQualityShading;
	float minNodeSize;
	int pointSize;
	bool updateStats;
	bool enableEDL;
	float edlStrength;

	// tier 1 shading (inserted by apply_tier1_shading.ps1)
	bool  roundSplats;
	bool  enableFog;
	float fogDensity;
	float fogR;
	float fogG;
	float fogB;
	bool  enableBloom;
	float bloomThreshold;
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
	float insectSpeed;

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
	bool     useVoxelGrid;      // master switch: gates every use, not the counting
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

	// look refinement (apply_look.ps1)
	float    pointWorldSize;    // splat diameter in metres; 0 = fixed pixel size
	float    pointShading;      // 0 flat disc .. 1 fully shaded sphere
	float    focalPx;           // height / (2 tan(fovy/2)), for projecting sizes
	int      maxSplatPx;        // cap on projected splat size
	float    grain;             // film grain amplitude
	float    vignette;          // corner darkening 0..1

	// directional key light (apply_keylight.ps1)
	uint64_t lightGridPtr;      // per-cell transmittance toward the light, 0..1
	float    lightDirX;         // unit vector TOWARD the light
	float    lightDirY;
	float    lightDirZ;
	float    lightR;            // light colour, linear
	float    lightG;
	float    lightB;
	float    lightIntensity;
	float    ambient;           // floor, so shadowed points are not black
	float    lightAbsorb;       // extinction per occupied cell along the ray
	int      lightSteps;        // cells marched toward the light
	int      lightDirty;        // 1 = rebuild the transmittance grid this frame

	// regroup (apply_regroup.ps1)
	int      lightOn;           // key light switch
	float    nearClip;          // metres: points nearer than this are not drawn
	int      groundOn;          // paint the ground band
	float    groundHeight;      // metres above the lowest point
	float    groundR;           // ground colour, sRGB 0..1
	float    groundG;
	float    groundB;
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

	// find the checkerboards (fix_walk_targets.ps1)
	int      findTargets;       // 1 = paint bright points magenta
	float    findThreshold;     // sRGB byte value all three channels must exceed
	float    markerScale;       // marker cluster size, metres
	float    findGrey;          // max allowed spread between R, G and B
	int      findIsolate;       // 1 = draw ONLY matches, everything else hidden
	int      findBoost;         // extra splat pixels for a match
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
	uint32_t numTargets                 = 0;   // fix_find_targets.ps1: points matching the target test
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

	bool memCapacityReached             = false;
};