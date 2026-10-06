// modules/progressive_octree/render.cu
// Some code in this file, particularly frustum, ray and intersection tests, 
// is adapted from three.js. Three.js is licensed under the MIT license
// This file this follows the three.js licensing
// License: MIT https://github.com/mrdoob/three.js/blob/dev/LICENSE

#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "utils.h.cu"
#include "builtin_types.h"
#include "helper_math.h"
#include "HostDeviceInterface.h"

#include "math.cuh"
#include "rasterization.cuh"
#include "structures.cuh"

#include "../CudaPrint/CudaPrint.cuh"

namespace cg = cooperative_groups;

Allocator* allocator;
// The frame's settings, in constant memory: the host copies them to this
// symbol before each launch (renderCUDA). As a device global written by
// thread 0 they were reloaded from global memory after every atomic, two to
// five loads per splat pixel; constant memory is read-only on the device, so
// the compiler need not reload it.
__constant__ Uniforms uniforms;
Lines* _lines = nullptr;
float* _emission = nullptr;   // RGB per pixel, added by apply_emission.ps1
PointCodec g_codec;           // decodes the octree's stored points; set with uniforms each frame
uint32_t _numEmitting = 0;    // debug counter, apply_debug_log.ps1
uint32_t _numDrawn    = 0;    // points that survived the density gate, apply_sweep_fixes.ps1

// ---- warp-aggregated counting (apply_counter_opt.ps1) -------------------
// A plain atomicAdd(&counter, 1) per point sends 31 MILLION atomics to one
// global address every frame, and they serialise through L2. This elects one
// thread per coalesced group to add the group's size instead, cutting the
// atomic count by up to 32x. Same result, and it is measurement code, so it
// had no business costing render time.
inline void countUp(uint32_t* counter){
	auto g = cg::coalesced_threads();
	if(g.thread_rank() == 0){
		atomicAdd(counter, g.size());
	}
}

constexpr int VISIBLITY_DISJUNCT = 0;
constexpr int VISIBLITY_ADDING = 1;
constexpr int VISIBLITY_SKIP = 2;
constexpr int visiblityMethod = VISIBLITY_DISJUNCT;

constexpr int SPLAT_SIZE = 1;
constexpr uint32_t BACKGROUND_COLOR = 0x00332211;
// constexpr uint32_t BACKGROUND_COLOR = 0x00ffff00;
// constexpr uint32_t BACKGROUND_COLOR = 0x00ffffff;
// constexpr uint32_t BACKGROUND_COLOR = 0x000000ff;

// https://colorbrewer2.org/
uint32_t SPECTRAL[8] = {
	0x4f3ed5,
	0x436df4,
	0x61aefd,
	0x8be0fe,
	0x98f5e6,
	0xa4ddab,
	0xa5c266,
	0xbd8832,
};

uint32_t getLodColor(int level){
	int index = float(8 - level) * 1.8f;
	index = clamp(index, 0, 7);

	uint32_t color = SPECTRAL[index];

	// if(level == 4) return 0x000000ff;
	// if(level == 8) return 0x0000ffff;
	
	return color;
}


// ---- tier 1 shading helpers (inserted by apply_tier1_shading.ps1) ------
// sRGB 8-bit to linear. Stored colours are gamma encoded; all fog, bloom
// and exposure work has to happen in linear or the maths is wrong.
inline float srgbToLinear(float c){
	c = c / 255.0f;
	return (c <= 0.04045f) ? (c / 12.92f)
						   : __powf((c + 0.055f) / 1.055f, 2.4f);
}

inline float linearToSrgb(float c){
	c = c < 0.0f ? 0.0f : (c > 1.0f ? 1.0f : c);
	return (c <= 0.0031308f) ? (c * 12.92f)
							 : (1.055f * __powf(c, 1.0f / 2.4f) - 0.055f);
}

// Narkowicz 2015 ACES approximation.
inline float acesToneMap(float x){
	const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
	float y = (x * (a * x + b)) / (x * (c * x + d) + e);
	return y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
}

inline float luminance(float r, float g, float b){
	return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}
// ---- end tier 1 helpers --------------------------------------------------


// ---- shared emission evaluation (apply_insects.ps1) ----------------------
// Used by both scanned points and insects, so the two cannot drift apart.
// Emission is EVALUATED per point per frame, never stored: no decay pass and
// nothing written back into point data, so the cost is numEvents distance
// tests per DRAWN point and is the same at 3 million points as at 500 million.
// ---- radial falloff (apply_falloff.ps1) ---------------------------------
// Was fixed at (1 - d/r) squared: one curve, no control. Now two knobs.
//
//   falloffPower  shapes the whole curve. 1 is a broad linear ramp; 2 is the
//                 old behaviour; 6 concentrates light near the centre and
//                 leaves a long feathered tail out to the radius.
//   coreBoost     adds an inverse-square hot spot at the very centre, which is
//                 what makes a source read as a light rather than a patch of
//                 brightness. Falls to a twentieth of its peak by a fifth of
//                 the radius, so it brightens the core without widening it.
//
// Both reach exactly zero at d = r, so there is never a hard edge.
inline float eventFalloff(float d, float r){
	if(r <= 0.0f) return 0.0f;
	float x = d / r;
	if(x >= 1.0f) return 0.0f;

	// A feathered body plus a narrow gaussian core: bright at the source,
	// fading smoothly to nothing at the radius. The core used to be a broad
	// 1/(1+80x^2) lobe that still held a fifth of its strength at x = 0.1.
	float feather = __powf(1.0f - x, uniforms.falloffPower);
	float core    = uniforms.coreBoost * __expf(-x * x * 120.0f);

	return feather + core * 4.0f;
}

inline bool showUnedited();   // defined below, before cullActive

inline void evalEmission(float px, float py, float pz,
						 float& emR, float& emG, float& emB)
{
	emR = 0.0f; emG = 0.0f; emB = 0.0f;
	if(showUnedited()) return;   // the unedited view shows no glow, so none is accumulated
	for(int e = 0; e < uniforms.numEvents; e++){
		float dx = px - uniforms.eventX[e];
		float dy = py - uniforms.eventY[e];
		float dz = pz - uniforms.eventZ[e];
		float rr = uniforms.eventR[e];
		float d2 = dx*dx + dy*dy + dz*dz;
		if(d2 > rr * rr) continue;
		float v = uniforms.eventI[e] * uniforms.emissionGain
				* eventFalloff(sqrt(d2), rr);
		emR += v * uniforms.eventCR[e];
		emG += v * uniforms.eventCG[e];
		emB += v * uniforms.eventCB[e];
	}
}

inline uint32_t insectHash(uint32_t x){
	x ^= x >> 16; x *= 0x7feb352du;
	x ^= x >> 15; x *= 0x846ca68bu;
	x ^= x >> 16;
	return x;
}
inline float insectUnit(uint32_t x){ return float(x & 0x00ffffffu) / 16777215.0f; }


// ---- lens (apply_fisheye.ps1) --------------------------------------------
// Every point goes through this. Perspective is the original path through
// uniforms.transform. Fisheye is an equidistant projection: the angle between
// a point and the optical axis maps LINEARLY to its distance from the screen
// centre. A 180 degree lens puts the zenith at the centre and the horizon on
// a circle - the view from the forest floor that a perspective lens cannot
// hold, because past about 110 degrees it stretches the edges to nothing.
//
// Depth in fisheye mode is euclidean distance rather than view-plane depth,
// so points beside or behind the camera still sort correctly. Callers treat
// depth as "how far", which both definitions satisfy.
inline bool projectPoint(float px, float py, float pz, int& sx, int& sy, float& depth){
	if(uniforms.lensMode == 0){
		float4 ndc = uniforms.transform * float4{px, py, pz, 1.0f};
		depth = ndc.w;
		if(depth <= 0.0f) return false;
		ndc = ndc / ndc.w;
		sx = int((ndc.x * 0.5f + 0.5f) * uniforms.width);
		sy = int((ndc.y * 0.5f + 0.5f) * uniforms.height);
		return true;
	}
	// view space: camera at the origin looking down -Z, +Y up
	float4 v = uniforms.view * float4{px, py, pz, 1.0f};
	float r  = sqrtf(v.x * v.x + v.y * v.y);
	float theta = atan2f(r, -v.z);                 // 0 straight ahead, pi behind
	if(theta > 3.10f) return false;                // directly behind: direction undefined
	float halfFov = uniforms.fisheyeFov * 0.5f * 0.017453292f;
	float radPx = (theta / halfFov) * (0.5f * uniforms.height);   // FOV reaches top and bottom
	float ux = r > 1e-6f ? v.x / r : 0.0f;
	float uy = r > 1e-6f ? v.y / r : 0.0f;
	sx = int(uniforms.width  * 0.5f + ux * radPx);
	sy = int(uniforms.height * 0.5f + uy * radPx);
	depth = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
	return depth > 0.0f;
}

// ---- pixel to world ray ---------------------------------------------------
// The inverse of projectPoint: which direction does this pixel look along.
// Used by the volumetric medium, by the sky/below-horizon split, by the
// marker picker and by the torch.
//
// The point unprojected is at NDC depth 0, about 2 near-plane distances in
// front of the camera, not at the far plane. With near 0.1 m and far 2e6 m
// the far point's w is below 1e-6, the size of float rounding in the inverse
// matrix, so its sign was wrong for some pixels and the ray pointed backwards:
// 1925 of 15000 rays over 600 camera positions in a float32 test, against
// none at depth 0, with the same worst direction error (0.03 degrees).
inline bool pixelRayDir(float px, float py, float3& dir){
	float ndcx = (px / uniforms.width)  * 2.0f - 1.0f;
	float ndcy = (py / uniforms.height) * 2.0f - 1.0f;
	if(uniforms.lensMode == 0){
		float4 nearP = uniforms.transformInv * float4{ndcx, ndcy, 0.0f, 1.0f};
		if(!(nearP.w > 0.0f)) return false;
		float3 world = float3{nearP.x / nearP.w, nearP.y / nearP.w, nearP.z / nearP.w};
		dir = world - uniforms.camPos;
		float len = sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
		if(len < 1e-6f) return false;
		dir = dir / len;
		return true;
	}
	float dxp = ndcx * 0.5f * uniforms.width;
	float dyp = ndcy * 0.5f * uniforms.height;
	float radPx = sqrtf(dxp*dxp + dyp*dyp);
	float halfFov = uniforms.fisheyeFov * 0.5f * 0.017453292f;
	float theta = (radPx / (0.5f * uniforms.height)) * halfFov;
	if(theta > 3.10f) return false;
	float ux = radPx > 1e-6f ? dxp / radPx : 0.0f;
	float uy = radPx > 1e-6f ? dyp / radPx : 0.0f;
	float st = sinf(theta), ct = cosf(theta);
	float lx = st * ux, ly = st * uy, lz = ct;
	dir.x = uniforms.camRight.x * lx + uniforms.camUp.x * ly + uniforms.camDir.x * lz;
	dir.y = uniforms.camRight.y * lx + uniforms.camUp.y * ly + uniforms.camDir.y * lz;
	dir.z = uniforms.camRight.z * lx + uniforms.camUp.z * ly + uniforms.camDir.z * lz;
	return true;
}

// Stored depth to distance along the ray: perspective stores view-plane
// depth, fisheye stores euclidean distance.
inline float depthToDistance(float depth, const float3& dir){
	if(uniforms.lensMode != 0) return depth;
	float cosT = dir.x*uniforms.camDir.x + dir.y*uniforms.camDir.y + dir.z*uniforms.camDir.z;
	if(cosT < 0.05f) cosT = 0.05f;
	return depth / cosT;
}

// One insect. Almost black by default, so it is nearly invisible until a call
// lights it. That is the point: the air between the trees gets something to
// catch light on, without adding anything to the scan.
inline void drawInsect(float px, float py, float pz, uint32_t color, float glow, uint64_t* framebuffer){
	int x, y; float depth;
	if(!projectPoint(px, py, pz, x, y, depth)) return;   // apply_fisheye.ps1
	if(depth < uniforms.nearClip) return;
	if(x < 1 || x > uniforms.width  - 2) return;
	if(y < 1 || y > uniforms.height - 2) return;

	float emR = 0.0f, emG = 0.0f, emB = 0.0f;
	if(glow > 0.0f){   // markers and the grid view pass 0: skip the per-event loop
		evalEmission(px, py, pz, emR, emG, emB);
		emR *= glow;
		emG *= glow;
		emB *= glow;
	}
	bool emitting = (emR + emG + emB) > 0.0001f;

	int R = uniforms.insectSize < 1 ? 1 : uniforms.insectSize;
	float half = float(R - 1) * 0.5f;
	float rad2 = (float(R) * 0.5f) * (float(R) * 0.5f);

	for(int ox = 0; ox < R; ox++)
	for(int oy = 0; oy < R; oy++)
	{
		if(R > 2){
			float ddx = float(ox) - half;
			float ddy = float(oy) - half;
			if(ddx * ddx + ddy * ddy > rad2) continue;
		}
		uint32_t qx = clamp(int(x + ox - half), 0, int(uniforms.width)  - 1);
		uint32_t qy = clamp(int(y + oy - half), 0, int(uniforms.height) - 1);
		uint32_t pixelID = qx + int(uniforms.width) * qy;

		if(emitting && _emission != nullptr){
			atomicAdd(&_emission[3 * pixelID + 0], emR);
			atomicAdd(&_emission[3 * pixelID + 1], emG);
			atomicAdd(&_emission[3 * pixelID + 2], emB);
		}

		uint64_t udepth = *((uint32_t*)&depth);
		uint64_t encoded = (udepth << 32) | color;
		if(encoded < framebuffer[pixelID]){
			atomicMin(&framebuffer[pixelID], encoded);
		}
	}
}

// ---- voxel occupancy grid helpers (apply_voxel_grid.ps1) ---------------
inline uint32_t gridCountAt(int cx, int cy, int cz){
	if(!uniforms.useVoxelGrid) return 0;
	if(uniforms.voxelGridPtr == 0) return 0;
	if(cx < 0 || cy < 0 || cz < 0) return 0;
	if(cx >= uniforms.gridX || cy >= uniforms.gridY || cz >= uniforms.gridZ) return 0;
	const uint32_t* g = (const uint32_t*)uniforms.voxelGridPtr;
	return g[cx + uniforms.gridX * (cy + uniforms.gridY * cz)];
}
inline uint32_t gridCountAtPos(float x, float y, float z){
	float c = uniforms.cellSize;
	return gridCountAt(int(floorf(x / c)), int(floorf(y / c)), int(floorf(z / c)));
}
// True when the cell is empty but foliage lies within `reach` cells along any
// axis: in air, but not in open sky. Six axis rays, so 6*reach lookups.
inline bool gridIsAirNearFoliage(int cx, int cy, int cz, int reach){
	if(gridCountAt(cx, cy, cz) > 0) return false;
	for(int d = 1; d <= reach; d++){
		if(gridCountAt(cx + d, cy, cz) > 0) return true;
		if(gridCountAt(cx - d, cy, cz) > 0) return true;
		if(gridCountAt(cx, cy + d, cz) > 0) return true;
		if(gridCountAt(cx, cy - d, cz) > 0) return true;
		if(gridCountAt(cx, cy, cz + d) > 0) return true;
		if(gridCountAt(cx, cy, cz - d) > 0) return true;
	}
	return false;
}

// ---- the unedited view ---------------------------------------------------
// Show unedited draws the cloud as Reset to unedited leaves it: each point in
// the colour stored in the file, nothing added and nothing hidden. Every step
// that changes the representation checks this: the blown-out repair, the
// stray-point cull, the key light, the torch, splat shading, eye dome
// lighting and the insects, and in the composite haze, sunbeams, bloom, the
// grade, exposure and the tone curve. The host draws the points and leaves
// out the debug colours (getUniforms). Point size, splat shape and
// high-quality shading stay as set, so the points keep their size and shape.
inline bool showUnedited(){
	return uniforms.showUnedited != 0;
}

// ---- stray-point culling -------------------------------------------------
// Scanner noise - dust, flying insects, leaves moving between passes, range
// artefacts - lands as isolated points in open air. Real surfaces do not: a
// leaf or a trunk puts many points in the same small volume.
//
// The voxel occupancy grid already counts points per cell, so "how many
// neighbours does this point have" is a lookup rather than a search. Culling
// on that count removes strays and leaves surfaces intact.
//
// reach 0 counts the point's own cell only - one read, cheapest, and enough
// for most noise. reach 1 adds the six axis neighbours, seven reads, which
// keeps thin surfaces that straddle a cell boundary and would otherwise be
// thinned along the seam.

// The cull runs only with a threshold and a grid to count in. Without the
// grid no point is stray, so Show culled only must not then hide them all.
inline bool cullActive(){
	return uniforms.minNeighbours > 0 && uniforms.useVoxelGrid && uniforms.voxelGridPtr != 0
		&& !showUnedited();
}

inline bool isStrayPoint(float px, float py, float pz){
	if(!cullActive()) return false;

	float c = uniforms.cellSize;
	int cx = int(floorf(px / c)), cy = int(floorf(py / c)), cz = int(floorf(pz / c));

	uint32_t n = gridCountAt(cx, cy, cz);
	if(uniforms.cullReach > 0){
		n += gridCountAt(cx + 1, cy, cz) + gridCountAt(cx - 1, cy, cz)
		   + gridCountAt(cx, cy + 1, cz) + gridCountAt(cx, cy - 1, cz)
		   + gridCountAt(cx, cy, cz + 1) + gridCountAt(cx, cy, cz - 1);
	}
	return n < uint32_t(uniforms.minNeighbours);
}

// ---- directional key light (apply_keylight.ps1) -------------------------
// The reference render is BACKLIT: light comes down through the canopy from
// behind, so foliage glows where light passes through it and trunks read as
// silhouettes. Nothing in a raw scan carries that - the points hold the colour
// the scanner recorded, under its own flash, flat.
//
// This computes, ONCE per cell rather than per point per frame, how much light
// reaches that cell from the light direction: march through the occupancy grid
// until the ray leaves it, accumulating extinction. That is a one-off pass, on
// a light change or a grid rebuild; at draw time it is a single lookup.
inline float lightAt(float px, float py, float pz){
	if(uniforms.lightGridPtr == 0) return 1.0f;
	float c = uniforms.cellSize;
	int cx = int(floorf(px / c)), cy = int(floorf(py / c)), cz = int(floorf(pz / c));
	if(cx < 0 || cy < 0 || cz < 0) return 1.0f;
	if(cx >= uniforms.gridX || cy >= uniforms.gridY || cz >= uniforms.gridZ) return 1.0f;
	const float* g = (const float*)uniforms.lightGridPtr;

	// Trilinear between cell centres. One cell per point lit every point in a
	// 0.35 m cube identically, so light and shadow met in hard-edged blocks.
	float fx = px / c - 0.5f, fy = py / c - 0.5f, fz = pz / c - 0.5f;
	int x0 = int(floorf(fx)), y0 = int(floorf(fy)), z0 = int(floorf(fz));
	float tx = fx - float(x0), ty = fy - float(y0), tz = fz - float(z0);
	auto at = [&](int x, int y, int z){
		x = min(max(x, 0), uniforms.gridX - 1);
		y = min(max(y, 0), uniforms.gridY - 1);
		z = min(max(z, 0), uniforms.gridZ - 1);
		return g[x + uniforms.gridX * (y + uniforms.gridY * z)];
	};
	float c00 = at(x0, y0,     z0    ) * (1.0f - tx) + at(x0 + 1, y0,     z0    ) * tx;
	float c10 = at(x0, y0 + 1, z0    ) * (1.0f - tx) + at(x0 + 1, y0 + 1, z0    ) * tx;
	float c01 = at(x0, y0,     z0 + 1) * (1.0f - tx) + at(x0 + 1, y0,     z0 + 1) * tx;
	float c11 = at(x0, y0 + 1, z0 + 1) * (1.0f - tx) + at(x0 + 1, y0 + 1, z0 + 1) * tx;
	float c0  = c00 * (1.0f - ty) + c10 * ty;
	float c1  = c01 * (1.0f - ty) + c11 * ty;
	return c0 * (1.0f - tz) + c1 * tz;
}

// A lit colour back to the 8-bit sRGB a point is stored and blended in. Light
// is added in linear light, as the haze's is, and when the brightest channel
// passes 1 all three are scaled down together, so the colour keeps its hue.
// Lighting used to multiply the sRGB bytes and clip each channel at 255 on
// its own: on olive foliage red and green reached 255 while blue did not, and
// sunlit leaves turned yellow. In full sun at the default strength that was
// 42% of the foliage points in data/samples/every100.laz.
inline uint32_t packLitColour(float r, float g, float b, uint32_t alphaBits){
	float m = fmaxf(r, fmaxf(g, b));
	if(m > 1.0f){
		float s = 1.0f / m;
		r *= s; g *= s; b *= s;
	}
	uint32_t ri = uint32_t(linearToSrgb(r) * 255.0f + 0.5f);
	uint32_t gi = uint32_t(linearToSrgb(g) * 255.0f + 0.5f);
	uint32_t bi = uint32_t(linearToSrgb(b) * 255.0f + 0.5f);
	return alphaBits | (bi << 16) | (gi << 8) | ri;
}

// ---- torch -------------------------------------------------------------
// A spotlight held at the camera and aimed through the mouse pixel. It adds
// light in proportion to a point's own (repaired) colour, so a lit leaf stays
// a leaf colour, with a soft-edged cone and inverse-square-like falloff.
// No shadows: the beam lights what is inside the cone at any depth.
float3 _torchDir   = {0.0f, 0.0f, 1.0f};   // set once per frame by thread 0
int    _torchValid = 0;

inline uint32_t applyTorch(uint32_t lit, uint32_t base, float px, float py, float pz){
	if(showUnedited() || uniforms.torchOn == 0 || _torchValid == 0) return lit;
	float vx = px - uniforms.camPos.x, vy = py - uniforms.camPos.y, vz = pz - uniforms.camPos.z;
	float d2 = vx*vx + vy*vy + vz*vz;
	if(d2 < 1e-6f) return lit;
	float d = sqrtf(d2);
	float c = (vx * _torchDir.x + vy * _torchDir.y + vz * _torchDir.z) / d;
	if(c <= uniforms.torchCosOuter) return lit;
	float t = (c - uniforms.torchCosOuter) / fmaxf(uniforms.torchCosInner - uniforms.torchCosOuter, 1e-4f);
	t = fminf(t, 1.0f);
	float spot = t * t * (3.0f - 2.0f * t);
	float rr = d / fmaxf(uniforms.torchRange, 0.1f);
	float k = uniforms.torchIntensity * spot / (1.0f + rr * rr);
	// in linear light, limited without changing hue (packLitColour, above)
	float r = srgbToLinear(float((lit      ) & 0xff)) + k * srgbToLinear(float((base      ) & 0xff));
	float g = srgbToLinear(float((lit >>  8) & 0xff)) + k * srgbToLinear(float((base >>  8) & 0xff));
	float b = srgbToLinear(float((lit >> 16) & 0xff)) + k * srgbToLinear(float((base >> 16) & 0xff));
	return packLitColour(r, g, b, lit & 0xff000000u);
}

// ---- blown-out colour repair -------------------------------------------
// The scanner colours each point from its camera, and in the sunlit canopy
// the camera saturates: leaves and branches come out white or pale grey.
// Those points are recognisable - every channel high and the three nearly
// equal - and their geometry is fine, only the colour is lost.
//
// While the occupancy grid is rebuilt from the octree, each cell also sums
// the colour of its well-exposed points. Cells holding only blown points
// then take the average of the nearest cells that have colour, searched
// outward shell by shell, so a branch in the bright crown inherits the brown
// of the bark below it and a leaf cluster the green around it. At draw time
// a blown point takes its position's colour from the grid, interpolated
// between cells, with a little per-point variation so a patch keeps texture.
//
// Cell layout, uint64:
//   while summing:  R sum <<46 | G sum <<28 | B sum <<10 | count (10 bits)
//   finished:       COLOUR_OWN  | rgb   (from the cell's own points)
//                   COLOUR_FILL | rgb   (taken from neighbouring cells)
//
// Only a cell's first COLOUR_MAX_TICKETS points are offered as sources: the
// occupancy count's atomicAdd hands out the tickets, so the cap is exact at no
// extra cost and the 10-bit count and 18-bit sums (1023 x 255 < 2^18) cannot
// overflow. The old cap was a plain read before the add, which concurrent
// blocks raced past: the count wrapped into the blue sum and cells came out
// in garbage hues, magenta and teal specks in the repaired canopy.
constexpr uint64_t COLOUR_OWN  = 1ull << 63;
constexpr uint64_t COLOUR_FILL = 1ull << 62;
constexpr uint32_t COLOUR_MAX_TICKETS = 1023;
constexpr uint32_t COLOUR_MIN_SOURCE  = 60;    // brightest channel a source point needs
constexpr float    COLOUR_MIN_LUMA    = 85.0f; // repaired colours are lifted to at least this

inline bool isBlownOut(uint32_t c){
	if(uniforms.whiteFix == 0) return false;
	float r = float((c      ) & 0xff);
	float g = float((c >>  8) & 0xff);
	float b = float((c >> 16) & 0xff);
	float mx = fmaxf(r, fmaxf(g, b));
	float mn = fminf(r, fminf(g, b));
	if(mn >= uniforms.whiteLevel && (mx - mn) <= uniforms.whiteSpread) return true;
	// Sky cast: the same saturation leaves lilac, pink and blue points, the
	// sky bleeding into the camera's pixel. Bark (r > g > b) and foliage
	// (g highest) never have more blue than green, so that alone marks them.
	if(uniforms.tintMargin > 0.0f && mx >= 40.0f && b > g + uniforms.tintMargin) return true;
	return false;
}

inline void accumulateColour(uint64_t* cell, uint32_t c){
	// Shadowed points are not a colour source: with an aggressive white test
	// they were the only "good" points left, and the canopy came out black.
	uint32_t mx = max((c) & 0xff, max((c >> 8) & 0xff, (c >> 16) & 0xff));
	if(mx < COLOUR_MIN_SOURCE) return;
	uint64_t r = (c      ) & 0xff;
	uint64_t g = (c >>  8) & 0xff;
	uint64_t b = (c >> 16) & 0xff;
	atomicAdd((unsigned long long*)cell, (unsigned long long)((r << 46) | (g << 28) | (b << 10) | 1ull));
}

// Called by refillVoxelGrid once every point is counted. Two passes, so the
// fill reads only cells finished by the first and never one it is writing.
void finishColourGrid(){
	if(uniforms.colourGridPtr == 0) return;
	auto grid = cg::this_grid();
	uint64_t* cg64 = (uint64_t*)uniforms.colourGridPtr;
	const uint32_t* occ = (const uint32_t*)uniforms.voxelGridPtr;
	int numCells = uniforms.gridX * uniforms.gridY * uniforms.gridZ;

	processRangeStrided(numCells, [&](int i){
		uint64_t v = cg64[i];
		uint64_t n = v & 0x3ffull;
		if(n == 0){ cg64[i] = 0; return; }
		uint64_t r = ((v >> 46) & 0x3ffffull) / n;
		uint64_t g = ((v >> 28) & 0x3ffffull) / n;
		uint64_t b = ((v >> 10) & 0x3ffffull) / n;
		cg64[i] = COLOUR_OWN | (b << 16) | (g << 8) | r;
	});
	grid.sync();

	int reach = uniforms.whiteReach;
	processRangeStrided(numCells, [&](int i){
		if(cg64[i] != 0 || occ[i] == 0) return;   // has colour, or nothing to colour
		int x = i % uniforms.gridX;
		int y = (i / uniforms.gridX) % uniforms.gridY;
		int z = i / (uniforms.gridX * uniforms.gridY);
		for(int d = 1; d <= reach; d++){
			uint32_t sr = 0, sg = 0, sb = 0, n = 0;
			for(int dz = -d; dz <= d; dz++)
			for(int dy = -d; dy <= d; dy++)
			for(int dx = -d; dx <= d; dx++){
				if(max(abs(dx), max(abs(dy), abs(dz))) != d) continue;   // this shell only
				int qx = x + dx, qy = y + dy, qz = z + dz;
				if(qx < 0 || qy < 0 || qz < 0 || qx >= uniforms.gridX || qy >= uniforms.gridY || qz >= uniforms.gridZ) continue;
				uint64_t q = cg64[qx + uniforms.gridX * (qy + uniforms.gridY * qz)];
				if((q & COLOUR_OWN) == 0) continue;
				sr += (q      ) & 0xff;
				sg += (q >>  8) & 0xff;
				sb += (q >> 16) & 0xff;
				n++;
			}
			if(n > 0){
				cg64[i] = COLOUR_FILL | (uint64_t(sb / n) << 16) | (uint64_t(sg / n) << 8) | uint64_t(sr / n);
				return;
			}
		}
	});
	grid.sync();
}

// Colour for a blown-out point at this position, or the point's own colour
// if the grid has nothing nearby. Trilinear over the cells that have colour.
inline uint32_t repairColour(uint32_t c, float px, float py, float pz){
	if(showUnedited() || uniforms.colourGridPtr == 0 || !isBlownOut(c)) return c;
	const uint64_t* g = (const uint64_t*)uniforms.colourGridPtr;
	float cs = uniforms.cellSize;
	float fx = px / cs - 0.5f, fy = py / cs - 0.5f, fz = pz / cs - 0.5f;
	int x0 = int(floorf(fx)), y0 = int(floorf(fy)), z0 = int(floorf(fz));
	float tx = fx - float(x0), ty = fy - float(y0), tz = fz - float(z0);
	float sr = 0.0f, sg = 0.0f, sb = 0.0f, sw = 0.0f;
	for(int k = 0; k < 8; k++){
		int x = x0 + (k & 1), y = y0 + ((k >> 1) & 1), z = z0 + (k >> 2);
		if(x < 0 || y < 0 || z < 0 || x >= uniforms.gridX || y >= uniforms.gridY || z >= uniforms.gridZ) continue;
		uint64_t q = g[x + uniforms.gridX * (y + uniforms.gridY * z)];
		if((q & (COLOUR_OWN | COLOUR_FILL)) == 0) continue;
		float w = ((k & 1) ? tx : 1.0f - tx) * (((k >> 1) & 1) ? ty : 1.0f - ty) * ((k >> 2) ? tz : 1.0f - tz);
		sr += w * float((q      ) & 0xff);
		sg += w * float((q >>  8) & 0xff);
		sb += w * float((q >> 16) & 0xff);
		sw += w;
	}
	if(sw <= 1e-4f) return c;
	// per-point variation, +-12%, stable for a given position
	uint32_t h = insectHash(__float_as_uint(px) ^ insectHash(__float_as_uint(py) ^ insectHash(__float_as_uint(pz))));
	float v = (0.88f + 0.24f * insectUnit(h)) / sw;
	// brightness floor: a borrowed colour is never darker than a mid-tone leaf
	float luma = (0.299f * sr + 0.587f * sg + 0.114f * sb) / sw;
	if(luma > 1.0f && luma < COLOUR_MIN_LUMA) v *= COLOUR_MIN_LUMA / luma;
	uint32_t r  = uint32_t(fminf(sr * v, 255.0f));
	uint32_t gg = uint32_t(fminf(sg * v, 255.0f));
	uint32_t b  = uint32_t(fminf(sb * v, 255.0f));
	return (c & 0xff000000u) | (b << 16) | (gg << 8) | r;
}

// Combine the key light with an ambient floor. Backlit foliage goes toward the
// light colour; shadowed points keep their own colour at ambient strength.
// Both act on linear light, so a point in full shade keeps the ambient share
// of its light (0.109, about a third of its sRGB value), the level the haze
// in shade darkens to.
inline uint32_t applyKeyLight(uint32_t c, float trans){
	if(showUnedited() || uniforms.lightOn == 0 || uniforms.lightIntensity <= 0.0f) return c;
	// ambient floors the shadow; intensity scales the lit side; at most 4x
	float k = uniforms.ambient + (1.0f - uniforms.ambient) * trans * uniforms.lightIntensity;
	if(k > 4.0f) k = 4.0f;
	float r = srgbToLinear(float((c      ) & 0xff)) * k * uniforms.lightR;
	float g = srgbToLinear(float((c >>  8) & 0xff)) * k * uniforms.lightG;
	float b = srgbToLinear(float((c >> 16) & 0xff)) * k * uniforms.lightB;
	return packLitColour(r, g, b, c & 0xff000000u);
}

// ---- haze and sunbeams ------------------------------------------------------
// The haze is a medium whose extinction per metre is fogDensity at and below
// fogBase, falling off exponentially above it:
//
//   sigma(z) = a                        z <= h0
//   sigma(z) = a * exp(-b * (z - h0))   z >  h0
//
// with a = fogDensity, b = fogHeightFalloff and h0 = fogBase (renderer frame,
// z = 0 at the lowest point). b = 0 is the even haze this replaced. Along a
// straight ray the optical depth has a closed form, so the composite applies
// the haze to every pixel exactly, at full resolution, without marching.
//
// The haze is lit the way the key light lights a point: an ambient share that
// reaches all of it, and the key light's share where the canopy lets it
// through. The ambient share is the same everywhere, so it also has a closed
// form, colour * (1 - transmittance), applied in the composite. The key light's
// share changes along the ray with the canopy's shade, so the volumetric pass
// marches it through the light grid. Looking toward the light, the lit haze
// between the trunks shows as beams.

// The sky is taken to lie this far away, beyond any haze. Rays toward it are
// this long when the haze covers the sky, and the volumetric upsample uses it
// as the sky's distance.
constexpr float SKY_DISTANCE = 1.0e7f;

// The key light changes nothing unless it is on with some strength; the same
// test applyKeyLight makes.
inline bool keyLightActive(){
	return !showUnedited() && uniforms.lightOn != 0 && uniforms.lightIntensity > 0.0f;
}

inline float hazeDensityAt(float z){
	float a = uniforms.fogDensity;
	float b = uniforms.fogHeightFalloff;
	if(b <= 0.0f || z <= uniforms.fogBase) return a;
	return a * __expf(-b * (z - uniforms.fogBase));
}

// Optical depth of the haze along z(s) = z0 + dz * s, for s from 0 to len.
// The density depends only on height, so the integral is the same in either
// direction along the segment. It is taken from the lower end upward, where
// every exponent is zero or negative and nothing can overflow, even on the
// 1e7 m rays the composite uses for the sky.
inline float hazeOpticalDepth(float z0, float dz, float len){
	float a = uniforms.fogDensity;
	if(!(len > 0.0f) || a <= 0.0f) return 0.0f;
	float b = uniforms.fogHeightFalloff;
	if(b <= 0.0f) return a * len;

	float h0  = uniforms.fogBase;
	float z1  = z0 + dz * len;
	float zlo = fminf(z0, z1);
	float zhi = fmaxf(z0, z1);

	// the part of the segment at or below h0 has the full density
	float below = 0.0f;
	if(zhi <= h0)     below = len;
	else if(zlo < h0) below = len * (h0 - zlo) / (zhi - zlo);
	float tau = a * below;

	// above h0, starting at height zs and rising |dz| per metre of ray:
	//   integral of a * exp(-b * (zs + |dz| u - h0)) du over u from 0 to above
	//   = a * exp(-b * (zs - h0)) * above * (1 - exp(-x)) / x,   x = b |dz| above
	// with the last factor taken from its series where x is too small to divide by
	float above = len - below;
	if(above > 0.0f){
		float zs    = fmaxf(zlo, h0);
		float x     = b * fabsf(dz) * above;
		float shape = (x < 1e-3f) ? 1.0f - 0.5f * x + x * x * (1.0f / 6.0f)
								  : (1.0f - __expf(-x)) / x;
		tau += a * __expf(-b * (zs - h0)) * above * shape;
	}
	return tau;
}

// Beams need the key light, some haze, and some light left above the ambient
// floor for the beams to add.
inline bool sunbeamsOn(){
	return keyLightActive() && uniforms.enableFog && uniforms.fogDensity > 0.0f
		&& uniforms.sunbeams > 0.0f && uniforms.ambient < 1.0f;
}

// Henyey-Greenstein phase function (Henyey & Greenstein 1941), multiplied by
// 4 pi so that g = 0, scattering the same in every direction, gives 1.
// cosTheta is the cosine between the view ray and the direction toward the
// light: 1 when looking straight at the light, where g > 0 is brightest.
inline float phaseHG(float cosTheta, float g){
	float g2 = g * g;
	float d  = fmaxf(1.0f + g2 - 2.0f * g * cosTheta, 1e-6f);
	return (1.0f - g2) / (d * sqrtf(d));
}

// Light the haze sends toward the camera per unit of optical depth, from the
// ambient share. With the key light off this is the haze colour as picked;
// with it on, the haze in shade darkens with the points in shade, and is
// tinted by the light colour as they are.
inline float3 hazeAmbientColour(){
	float3 f = float3{uniforms.fogR, uniforms.fogG, uniforms.fogB};
	if(!keyLightActive()) return f;
	float k = uniforms.ambient;
	return float3{f.x * k * uniforms.lightR, f.y * k * uniforms.lightG, f.z * k * uniforms.lightB};
}

// The key light's share, before the canopy's shade and the phase function
// are applied. Zero unless beams are on.
inline float3 hazeSunColour(){
	if(!sunbeamsOn()) return float3{0.0f, 0.0f, 0.0f};
	float k = (1.0f - uniforms.ambient) * uniforms.lightIntensity * uniforms.sunbeams;
	return float3{uniforms.fogR * k * uniforms.lightR,
				  uniforms.fogG * k * uniforms.lightG,
				  uniforms.fogB * k * uniforms.lightB};
}

// One light grid cell. Outside the grid there is no canopy above, so full light.
inline float lightCell(int cx, int cy, int cz){
	if(cx < 0 || cy < 0 || cz < 0) return 1.0f;
	if(cx >= uniforms.gridX || cy >= uniforms.gridY || cz >= uniforms.gridZ) return 1.0f;
	return ((const float*)uniforms.lightGridPtr)[cx + uniforms.gridX * (cy + uniforms.gridY * cz)];
}

// lightAt, interpolated between cell centres. A point takes its own cell's
// value, but haze marched through whole cells comes out in cell-sized steps,
// which near the camera show on screen as blocks along the beams.
inline float lightAtSmooth(float px, float py, float pz){
	if(uniforms.lightGridPtr == 0) return 1.0f;
	float c  = uniforms.cellSize;
	float fx = px / c - 0.5f;
	float fy = py / c - 0.5f;
	float fz = pz / c - 0.5f;
	int   x0 = int(floorf(fx)), y0 = int(floorf(fy)), z0 = int(floorf(fz));
	float tx = fx - float(x0), ty = fy - float(y0), tz = fz - float(z0);

	float c00 = lightCell(x0, y0,     z0    ) * (1.0f - tx) + lightCell(x0 + 1, y0,     z0    ) * tx;
	float c10 = lightCell(x0, y0 + 1, z0    ) * (1.0f - tx) + lightCell(x0 + 1, y0 + 1, z0    ) * tx;
	float c01 = lightCell(x0, y0,     z0 + 1) * (1.0f - tx) + lightCell(x0 + 1, y0,     z0 + 1) * tx;
	float c11 = lightCell(x0, y0 + 1, z0 + 1) * (1.0f - tx) + lightCell(x0 + 1, y0 + 1, z0 + 1) * tx;
	float c0  = c00 * (1.0f - ty) + c10 * ty;
	float c1  = c01 * (1.0f - ty) + c11 * ty;
	return c0 * (1.0f - tz) + c1 * tz;
}

// ---- survey marker balls (apply_markers2.ps1) -----------------------------
// A picked checkerboard centre, drawn as a filled disc of a fixed WORLD
// diameter (markerScale, default 0.10 m) in magenta, which nothing in a
// forest scan is. Projected once, then one atomicMin per covered pixel. A
// floor of 3 px radius keeps a 10 cm ball visible from across the array;
// a ceiling of 64 px keeps one at the lens from costing thousands of atomics.
// Depth-tested like any point, so a ball behind a trunk is hidden by it, and
// written a hair nearer than its own depth so it wins against the very point
// it was picked from.
// Screen pixels per metre at view depth `depth` for the current lens. The
// fisheye is equidistant (radius proportional to angle), so its scale is
// (height/2)/halfFov pixels per radian; the perspective focal length would
// make fisheye splats several times too large.
inline float pxPerMetreAt(float depth){
	float d = fmaxf(depth, 0.05f);
	if(uniforms.lensMode == 0) return uniforms.focalPx / d;
	float halfFov = uniforms.fisheyeFov * 0.5f * 0.017453292f;
	return ((0.5f * uniforms.height) / halfFov) / d;
}

// One pixel (ox, oy) of a marker ball's square of side 2 MARKER_MAX_R + 1.
// Drawn one item a pixel (kernel_render), so a ball near the camera no longer
// keeps one thread looping over 16,641 pixels while the rest wait.
constexpr int MARKER_MAX_R = 64;
inline void drawMarkerBallPixel(float px, float py, float pz, int ox, int oy, uint64_t* framebuffer){
	int x, y; float depth;
	if(!projectPoint(px, py, pz, x, y, depth)) return;
	if(depth < uniforms.nearClip) return;
	int R = int(0.5f * uniforms.markerScale * pxPerMetreAt(depth) + 0.5f);
	if(R < 3)  R = 3;
	if(R > MARKER_MAX_R) R = MARKER_MAX_R;
	if(ox*ox + oy*oy > R*R) return;
	int qx = x + ox, qy = y + oy;
	if(qx < 0 || qx >= int(uniforms.width) || qy < 0 || qy >= int(uniforms.height)) return;
	float dz = depth * 0.999f;
	uint64_t udz = *((uint32_t*)&dz);
	uint64_t encoded = (udz << 32) | 0x00ff00ffull;   // aabbggrr: magenta
	uint32_t pixelID = qx + int(uniforms.width) * qy;
	if(encoded < framebuffer[pixelID]) atomicMin(&framebuffer[pixelID], encoded);
}

// ---- world-space point size and sphere shading (apply_look.ps1) ---------
// The reference render draws points as spheres of a fixed WORLD size: near
// ones large, distant ones shrinking until they merge into texture. A fixed
// pixel size flattens the forest into uniform noise. This projects a diameter
// in metres to pixels using the view depth, capped so a point at the lens
// cannot cost thousands of atomics.
//
// minWorldSize is the smallest splat, in metres, the sample may get: 0 for
// scanned points, and for a voxel its node's cell spacing (voxelSplatMetres),
// so that the voxels of an inner node close up into a surface. They used to
// get the point size, which leaves holes wherever cells are further apart on
// screen than a point splat is wide.
inline int splatRadiusPx(float depth, float minWorldSize){
	if(uniforms.pointWorldSize <= 0.0f && minWorldSize <= 0.0f) return uniforms.pointSize;
	float pxPerMetre = pxPerMetreAt(depth);
	// A voxel is drawn at least its cell spacing, so a coarse node has no
	// holes, but no larger than that: adding the spacing to the point size made
	// coarse nodes' splats far bigger than their neighbours' leaf points, and
	// every node boundary between two levels showed as a patch.
	float worldSize = fmaxf(uniforms.pointWorldSize > 0.0f ? uniforms.pointWorldSize : 0.0f, minWorldSize);
	int R = worldSize > 0.0f ? int(worldSize * pxPerMetre + 0.5f) : uniforms.pointSize;
	if(R < 1) R = 1;
	if(R > uniforms.maxSplatPx) R = uniforms.maxSplatPx;
	return R;
}

// Smallest splat for the voxels of a node of this size: one cell spacing, so
// neighbouring voxels close up into a surface.
constexpr float VOXEL_SPLAT_COVER = 1.0f;
inline float voxelSplatMetres(float nodeSize){
	return VOXEL_SPLAT_COVER * nodeSize / float(GRID_SIZE);
}

// Shade a splat pixel as a point on a sphere lit from the upper left front.
// (ox,oy) is the pixel's offset within the R x R splat. Returns a multiplier;
// 1.0 everywhere when shading is off or the splat is too small to show it.
inline float splatShade(int ox, int oy, int R, float half){
	if(showUnedited() || uniforms.pointShading <= 0.0f || R < 3) return 1.0f;
	float rr = float(R) * 0.5f;
	float u  = (float(ox) - half) / rr;
	float v  = (float(oy) - half) / rr;
	float nz2 = 1.0f - u*u - v*v;
	if(nz2 < 0.0f) nz2 = 0.0f;
	float nz = sqrtf(nz2);
	// light direction, normalised: up-left-towards viewer
	float ndotl = -0.35f * u + 0.50f * v + 0.79f * nz;
	if(ndotl < 0.0f) ndotl = 0.0f;
	// never fully black on the far limb: keep a floor so silhouettes read
	float lit = 0.18f + 0.82f * ndotl;
	return (1.0f - uniforms.pointShading) + uniforms.pointShading * lit;
}

// The pixels of an R x R splat centred on (x, y), row by row, as f(ox, oy,
// pixelID) with ox and oy from 0 to R - 1 (the offsets splatShade takes).
// a = 2 ox - (R - 1) and b = 2 oy - (R - 1) are offsets in half pixels, so the
// round test (ox - half)^2 + (oy - half)^2 <= (R/2)^2 is a^2 + b^2 <= R^2, and
// each row keeps |a| <= floor(sqrt(R^2 - b^2)) of the parity of R - 1. This
// visits exactly the pixels the column-by-column loop with the test inside
// visited, for every R from 1 to 64 (tested), with the test once a row.
// Pixels off the frame are skipped rather than moved onto its edge.
template<typename F>
inline void forEachSplatPixel(int x, int y, int R, F&& f){
	int W = int(uniforms.width), H = int(uniforms.height);
	bool clip = uniforms.roundSplats && R > 2;
	for(int oy = 0; oy < R; oy++){
		int b  = 2 * oy - (R - 1);
		int py = y + (b >> 1);
		if(py < 0 || py >= H) continue;
		int amax = R - 1;
		if(clip){
			int m = R * R - b * b;
			int r = int(sqrtf(float(m)));
			if((r + 1) * (r + 1) <= m) r++;
			if(r * r > m) r--;
			if(((r ^ (R - 1)) & 1) != 0) r--;
			if(r < 0) continue;
			amax = r;
		}
		int row = W * py;
		for(int a = -amax; a <= amax; a += 2){
			int px = x + (a >> 1);
			if(px < 0 || px >= W) continue;
			f((a + R - 1) >> 1, oy, uint32_t(px + row));
		}
	}
}

inline uint32_t shadeColour(uint32_t c, float s){
	if(s >= 0.999f) return c;
	uint32_t r = uint32_t(fminf(float((c      ) & 0xff) * s, 255.0f));
	uint32_t g = uint32_t(fminf(float((c >>  8) & 0xff) * s, 255.0f));
	uint32_t b = uint32_t(fminf(float((c >> 16) & 0xff) * s, 255.0f));
	return (c & 0xff000000u) | (b << 16) | (g << 8) | r;
}

// minWorldSize: see splatRadiusPx; 0 for scanned points
void drawPoint(Point point, Node* node, uint64_t* framebuffer, float minWorldSize){
	// Density is applied as a loop stride in drawNode now, so dropped points are
	// never read. The per-point hash that used to sit here ran for every one of
	// 90.7 M samples and rejected them AFTER the load, which is why turning
	// density down barely changed the frame time.

	int x, y; float depth;
	if(!projectPoint(point.x, point.y, point.z, x, y, depth)) return;   // apply_fisheye.ps1
	if(depth < uniforms.nearClip) return;

	// Drawn when its splat can reach the frame; forEachSplatPixel skips the
	// pixels off it. The test used to want the centre 2 px inside, which left
	// a border up to 2 px wide that small splats never reached, and moved the
	// off-frame pixels of larger splats onto the edge.
	int R = splatRadiusPx(depth, minWorldSize);   // world-sized (apply_look.ps1)
	if(x > -R && x < int(uniforms.width)  + R)
	if(y > -R && y < int(uniforms.height) + R)
	{
		// stray-point culling (apply_stray_cull.ps1). Up to seven grid reads,
		// so it runs only for points that land on screen; projection is
		// cheaper and rejects everything behind or beside the camera first.
		// "Show culled only" with the threshold at 0 used to blank the screen:
		// nothing was stray, so "only the strays" was nothing. Now the checkbox
		// does nothing until the cull can run (cullActive).
		bool stray = isStrayPoint(point.x, point.y, point.z);
		if(uniforms.showCulledOnly && cullActive()){ if(!stray) return; }
		else if(stray) return;
		countUp(&_numDrawn);     // on-screen points that pass density and the cull

		uint32_t color = point.color;
		if(uniforms.colorByNode){
			color = (node->getID() % 127) * 123456789ull;
		}else if(uniforms.colorByLOD){
			color = getLodColor(node->level);
		}else{
			color = repairColour(color, point.x, point.y, point.z);   // blown-out canopy
			uint32_t base = color;
			if(keyLightActive()) color = applyKeyLight(color, lightAt(point.x, point.y, point.z));   // gated: no grid read while off
			color = applyTorch(color, base, point.x, point.y, point.z);
		}

		// float w = y - 3.0 * x;
		// if(w > -3800.0){
		// 	color = point.color;
		// }else{
		// 	color = (node->getID() % 127) * 123456789ull;
		// 	color = getLodColor(node->level);
		// }

		// ---- excitation -------------------------------------------------
		float emR, emG, emB;
		evalEmission(point.x, point.y, point.z, emR, emG, emB);
		bool emitting = (emR + emG + emB) > 0.0001f;
		if(emitting) countUp(&_numEmitting);   // debug

		// Centred, optionally round splat, visited row by row
		// (forEachSplatPixel).
		float half = float(R - 1) * 0.5f;
		float* em = emitting ? _emission : nullptr;   // read once a point
		uint64_t udepth = *((uint32_t*)&depth);
		forEachSplatPixel(x, y, R, [&](int ox, int oy, uint32_t pixelID){
			// Additive, and deliberately outside the depth test: overlapping
			// excited points sum, which atomicMin cannot do. A point behind a
			// trunk still glows through it - move this inside the depth test
			// below to occlude instead.
			if(em != nullptr){
				atomicAdd(&em[3 * pixelID + 0], emR);
				atomicAdd(&em[3 * pixelID + 1], emG);
				atomicAdd(&em[3 * pixelID + 2], emB);
			}
			uint32_t shaded = shadeColour(color, splatShade(ox, oy, R, half));   // apply_look.ps1
			uint64_t encoded = (udepth << 32) | shaded;

			if(encoded < framebuffer[pixelID]){
				atomicMin(&framebuffer[pixelID], encoded);
			}
		});

	}
}

// LOD cross-fade (see the disjunct visibility pass): a stable per-sample
// dither, so the same samples stay drawn from frame to frame. A parent
// fading out keeps samples at or above the blend, its children fading in keep
// those below it, so together they cover the area at every blend.
inline bool lodKeep(const Node* node, int index){
	if(node->lodRole == 0) return true;
	uint32_t h = insectHash(uint32_t(index) * 0x9E3779B9u
	                      ^ node->X * 73856093u ^ node->Y * 19349663u ^ node->Z * 83492791u ^ node->level);
	float u = insectUnit(h);
	return node->lodRole == 1 ? (u >= node->lodBlend) : (u < node->lodBlend);
}

// Point density as a fractional loop stride: the k-th kept point is
// floor(k / d), with 1/d in 16.16 fixed point, so a node keeps the fraction d
// of its points whatever d is. An integer stride, 1/round(1/d), kept all of
// them for any d above two thirds and half of them from 0.4 to two thirds.
// Deterministic, so the same points are kept from frame to frame, and the
// depth and colour passes keep the same ones.
inline uint32_t densityStepFx(){
	if(uniforms.pointDensity >= 0.999f) return 65536u;
	float d = uniforms.pointDensity < 0.0005f ? 0.0005f : uniforms.pointDensity;
	return uint32_t(65536.0f / d + 0.5f);
}
inline int keptIndex(int k, uint32_t stepFx){ return int((uint64_t(k) * stepFx) >> 16); }

// ---- shared point test for the high-quality (HQS) passes ------------------
// The depth pass and the colour pass must agree exactly on which points exist,
// where they land and how large they are, or the depth test hides points the
// colour pass then cannot draw. The four loops used to repeat this inline and
// had drifted.
// minWorldSize: see splatRadiusPx; 0 for scanned points. Both passes must pass
// the same value for the same sample.
inline bool hqsAcceptPoint(const Point& point, Node* node, float minWorldSize,
                           int& x, int& y, float& depth, int& R, uint32_t& color){
	if(!projectPoint(point.x, point.y, point.z, x, y, depth)) return false;   // apply_fisheye.ps1
	if(!(depth > 0.0f) || depth < uniforms.nearClip) return false;
	R = splatRadiusPx(depth, minWorldSize);   // world-sized (apply_look.ps1)
	if(!(x > -R && x < int(uniforms.width)  + R)) return false;   // as drawPoint: the splat can reach the frame
	if(!(y > -R && y < int(uniforms.height) + R)) return false;

	// stray-point culling, after the cheaper on-screen test (apply_stray_cull.ps1)
	bool stray = isStrayPoint(point.x, point.y, point.z);
	if(uniforms.showCulledOnly && cullActive()){ if(!stray) return false; }
	else if(stray) return false;

	color = point.color;
	if(uniforms.colorByNode){
		color = (node->getID() % 127) * 123456789;
	}else if(uniforms.colorByLOD){
		color = getLodColor(node->level);
	}
	if(RIGHTSIDE_NODECOLORS && x > uniforms.width / 2){
		color = (node->getID() % 127) * 123456789;
	}

	return true;
}

// minWorldSize: see splatRadiusPx; 0 for a node's points, voxelSplatMetres
// for its voxels
void drawNode(Chunk* chunk, Node* node, uint32_t numElements, uint64_t* framebuffer, float minWorldSize){
	auto block = cg::this_thread_block();

	int chunkIndex = 0;

	// ---- density as a STRIDE (apply_stride_density.ps1) -------------------
	// Measured: at density 1.0, 31.1 M points drawn cost 62.2 ms; at density
	// 0.018, 0.32 M points drawn cost 56.3 ms. A 97-fold cut in drawn points
	// bought 9%. Splatting was never the cost - VISITING the points was. The
	// gate lived inside drawPoint, so every rejected point had already been
	// walked to and read from memory, and the hash then ran for all 90.7 M.
	//
	// Striding the loop means the skipped points are never touched: no chunk
	// walk, no 16-byte load, no hash. The traversal shrinks by the same factor
	// as the density. Deterministic, so the same points are kept frame to
	// frame, as with the hash.
	uint32_t stepFx = densityStepFx();

	for(int k = block.thread_rank(); ; k += block.num_threads()){
		int pointIndex = keptIndex(k, stepFx);
		if(pointIndex >= int(numElements)) break;
		int targetChunkIndex = pointIndex / POINTS_PER_CHUNK;

		// while, not if: with a stride the target can jump several chunks at
		// once, and the original single step would silently fall behind and
		// read from the wrong chunk.
		while(chunkIndex < targetChunkIndex && chunk->next != nullptr){
			chunk = chunk->next;
			chunkIndex++;
		}

		int pointInChunkIndex = pointIndex % POINTS_PER_CHUNK;
		if(!lodKeep(node, pointIndex)) continue;   // LOD cross-fade
		Point point = g_codec.decode(chunk->points[pointInChunkIndex]);

		drawPoint(point, node, framebuffer, minWorldSize);

		// debug pseudo-voxel
		// if(node->numPoints == 0)
		// { 
		// 	float s = nodeSize / 256.0;
		// 	float numSteps = 5;

		// 	for(float sx = -numSteps; sx <= numSteps; sx += 1.0f)
		// 	for(float sy = -numSteps; sy <= numSteps; sy += 1.0f)
		// 	{
		// 		float ox = s * sx / numSteps;
		// 		float oy = s * sy / numSteps;
		// 		Point p;
		// 		p.x = point.x + ox;
		// 		p.y = point.y + oy;
		// 		p.z = point.z + s;
		// 		p.color = point.color;

		// 		drawPoint(point);

		// 		p.x = point.x + ox;
		// 		p.y = point.y + s;
		// 		p.z = point.z + oy;
		// 		drawPoint(point);

		// 		p.x = point.x + s;
		// 		p.y = point.y + ox;
		// 		p.z = point.z + oy;
		// 		drawPoint(p);
		// 	}
		// }
	}
};

uint32_t _refillNextNode = 0;   // apply_grid_refill.ps1

// ---- rebuild the occupancy grid from the octree (apply_grid_refill.ps1) --
// The counts were only ever accumulated as points ENTERED the octree, in
// kernel_construct. So any reallocation of the grid - which the resolution
// slider does on every move - threw them all away and nothing put them back.
// The cull then read a grid that was almost entirely empty.
//
// The octree still holds every point, so the grid can be rebuilt from it
// without reloading the file. Same walk drawNodes uses, counting instead of
// drawing. Runs once, when the host says the grid is stale.
void countNodeIntoGrid(Chunk* chunk, uint32_t numElements){
	auto block = cg::this_thread_block();
	int chunkIndex = 0;
	uint32_t* g = (uint32_t*)uniforms.voxelGridPtr;
	float c = uniforms.cellSize;

	for(int pointIndex = block.thread_rank(); pointIndex < numElements; pointIndex += block.num_threads()){
		int targetChunkIndex = pointIndex / POINTS_PER_CHUNK;
		while(chunkIndex < targetChunkIndex && chunk->next != nullptr){
			chunk = chunk->next;
			chunkIndex++;
		}
		int pointInChunkIndex = pointIndex % POINTS_PER_CHUNK;
		Point point = g_codec.decode(chunk->points[pointInChunkIndex]);

		int cx = int(floorf(point.x / c));
		int cy = int(floorf(point.y / c));
		int cz = int(floorf(point.z / c));
		if(cx >= 0 && cx < uniforms.gridX && cy >= 0 && cy < uniforms.gridY && cz >= 0 && cz < uniforms.gridZ){
			uint32_t cell = cx + uniforms.gridX * (cy + uniforms.gridY * cz);
			uint32_t ticket = atomicAdd(&g[cell], 1u);   // also caps the colour samples
			if(uniforms.colourGridPtr != 0 && ticket < COLOUR_MAX_TICKETS && !isBlownOut(point.color)){
				accumulateColour(((uint64_t*)uniforms.colourGridPtr) + cell, point.color);
			}
		}
	}
}

void refillVoxelGrid(Node* nodes, Stats* stats){
	if(uniforms.gridRefill == 0) return;
	if(uniforms.voxelGridPtr == 0) return;

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	// zero first: a refill must not add to whatever was already there
	processRangeStrided(uniforms.gridX * uniforms.gridY * uniforms.gridZ, [&](int i){
		((uint32_t*)uniforms.voxelGridPtr)[i] = 0;
		if(uniforms.colourGridPtr != 0) ((uint64_t*)uniforms.colourGridPtr)[i] = 0;
	});
	grid.sync();

	// A device global, not allocator->alloc: this function returns early on
	// every frame that does not refill, so an allocation here would shift
	// the offset of every later buffer between frames.
	if(grid.thread_rank() == 0){ _refillNextNode = 0; }
	grid.sync();

	__shared__ int sh_n;
	while(true){
		block.sync();
		if(block.thread_rank() == 0){ sh_n = atomicAdd(&_refillNextNode, 1); }
		block.sync();
		// the device's own count: the host's copy of it lags by a frame or more
		if(sh_n >= int(stats->numNodes)) break;

		Node* node = &nodes[sh_n];
		// points only. Voxels are coarse summaries of the same surfaces, and
		// counting them as well would inflate the density of everything that
		// happens to be summarised, which is exactly what the cull compares.
		countNodeIntoGrid(node->points, node->numPoints);
		block.sync();
	}
	grid.sync();

	finishColourGrid();
}

void drawNodes(Node* nodes, uint32_t numNodes, uint64_t* framebuffer){

	if(!uniforms.showPoints) return;

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	float3 boxSize = uniforms.boxMax - uniforms.boxMin;
	float cubeSize = max(max(boxSize.x, boxSize.y), boxSize.z);
	float3 cubeMin = uniforms.boxMin;

	uint32_t& numNodesDrawn = *allocator->alloc<uint32_t*>(4);
	if(grid.thread_rank() == 0){
		numNodesDrawn = 0;
	}

	grid.sync();

	__shared__ int sh_nodeIndex;

	while(true){

		block.sync();

		if(block.thread_rank() == 0){
			sh_nodeIndex = atomicAdd(&numNodesDrawn, 1);
		}

		block.sync();

		if(sh_nodeIndex >= numNodes) break;

		Node* node = &nodes[sh_nodeIndex];

		float nodeSize = cubeSize / pow(2.0f, float(node->level));
		float scale = cubeSize / pow(2.0f, float(node->level));
		float3 nodeMin = {
			cubeMin.x + float(node->X + 0.0f) * scale,
			cubeMin.y + float(node->Y + 0.0f) * scale,
			cubeMin.z + float(node->Z + 0.0f) * scale,
		};

		drawNode(node->points, node, node->numPoints, framebuffer, 0.0f);
		// A leaf's voxels duplicate its own points. Only a root that never
		// spilled (a cloud of 50 k points or fewer) is a leaf with voxels, and
		// at cell size they would cover its points. Same test in both HQS passes.
		if(!node->isLeafFn()){
			drawNode(node->voxelChunks, node, node->numVoxels, framebuffer, voxelSplatMetres(nodeSize));
		}

		block.sync();
	}

	grid.sync();
}

// fb_depth (4 B a pixel) and fb_color (16 B: red, green, blue and a count)
// are allocated by kernel_render and passed in. Allocated here, through the
// allocator global, the compiler could not tell they were in global memory,
// and the depth test and colour sums used generic atomics.
void drawNodesHQS(Node* nodes, uint32_t numNodes, uint64_t* framebuffer, uint32_t* fb_depth, uint32_t* fb_color){

	if(!uniforms.showPoints) return;

	// density as a loop stride, as in drawNode: skipped points are never read,
	// and both passes step identically so they keep the same points
	uint32_t hqsStepFx = densityStepFx();

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	float3 boxSize = uniforms.boxMax - uniforms.boxMin;
	float cubeSize = max(max(boxSize.x, boxSize.y), boxSize.z);
	float3 cubeMin = uniforms.boxMin;


	uint32_t& numNodesDrawn = *allocator->alloc<uint32_t*>(4);
	if(grid.thread_rank() == 0){
		numNodesDrawn = 0;
	}

	int numPixels = int(uniforms.width) * int(uniforms.height);

	processRangeStrided(numPixels, [&](int pixelID){
		fb_depth[pixelID] = 0x7f800000;
	});
	processRangeStrided(numPixels, [&](int pixelID){
		fb_color[4 * pixelID + 0] = 0;
		fb_color[4 * pixelID + 1] = 0;
		fb_color[4 * pixelID + 2] = 0;
		fb_color[4 * pixelID + 3] = 0;
	});

	grid.sync();

	__shared__ int sh_nodeIndex;

	// DEPTH
	while(true){

		block.sync();

		if(block.thread_rank() == 0){
			sh_nodeIndex = atomicAdd(&numNodesDrawn, 1);
		}

		block.sync();

		if(sh_nodeIndex >= numNodes) break;

		Node* node = &nodes[sh_nodeIndex];

		float nodeSize = cubeSize / pow(2.0f, float(node->level));
		float scale = cubeSize / pow(2.0f, float(node->level));
		float3 nodeMin = {
			cubeMin.x + float(node->X + 0.0f) * scale,
			cubeMin.y + float(node->Y + 0.0f) * scale,
			cubeMin.z + float(node->Z + 0.0f) * scale,
		};

		{ // draw points
			Chunk* chunk = node->points;
			int chunkIndex = 0;

			for(int k = block.thread_rank(); ; k += block.num_threads()){
				int pointIndex = keptIndex(k, hqsStepFx);
				if(pointIndex >= int(node->numPoints)) break;

				int targetChunkIndex = pointIndex / POINTS_PER_CHUNK;

				while(chunkIndex < targetChunkIndex && chunk->next != nullptr){
					chunk = chunk->next;
					chunkIndex++;
				}

				int pointInChunkIndex = pointIndex % POINTS_PER_CHUNK;
				if(!lodKeep(node, pointIndex)) continue;   // LOD cross-fade
				Point point = g_codec.decode(chunk->points[pointInChunkIndex]);

				int x, y, R; float depth; uint32_t color;
				if(!hqsAcceptPoint(point, node, 0.0f, x, y, depth, R, color)) continue;
				{
					// HQS depth pass: the same pixels as the colour pass
					// below (forEachSplatPixel), or the depth test and the
					// colour disagree.
					uint32_t udepth = *((uint32_t*)&depth);
					forEachSplatPixel(x, y, R, [&](int ox, int oy, uint32_t pixelID){
						if(udepth < fb_depth[pixelID]){
							atomicMin(&fb_depth[pixelID], udepth);
						}
					});
				}
			}
		}

		block.sync();

		uint32_t childVisibleMask = 0;
		for(int i = 0; i < 8; i++){
			Node* child = node->children[i];
			uint32_t childExists = (child != nullptr) ? 1 : 0;
			uint32_t childIsVisible = 0;

			if(childExists){
				childIsVisible = child->visible ? 1 : 0;
			}

			childVisibleMask = childVisibleMask | (childIsVisible << i);
		}

		if(node->numVoxels > 0 && !node->isLeafFn())   // a leaf's voxels duplicate its points; as in drawNodes
		{// draw voxels
			Chunk* chunk = node->voxelChunks;
			int chunkIndex = 0;

			for(int k = block.thread_rank(); ; k += block.num_threads()){
				int pointIndex = keptIndex(k, hqsStepFx);
				if(pointIndex >= int(node->numVoxels)) break;

				int targetChunkIndex = pointIndex / POINTS_PER_CHUNK;

				while(chunkIndex < targetChunkIndex && chunk->next != nullptr){
					chunk = chunk->next;
					chunkIndex++;
				}

				int pointInChunkIndex = pointIndex % POINTS_PER_CHUNK;
				if(!lodKeep(node, pointIndex)) continue;   // LOD cross-fade
				Point point = g_codec.decode(chunk->points[pointInChunkIndex]);

				// cull voxels in areas where child nodes are visible
				float3 posuv = (float3{point.x, point.y, point.z} - nodeMin) / nodeSize;
				uint32_t cx = (posuv.x < 0.5f) ? 0 : 1;
				uint32_t cy = (posuv.y < 0.5f) ? 0 : 1;
				uint32_t cz = (posuv.z < 0.5f) ? 0 : 1;
				uint32_t childIndex = (cx << 2) | (cy << 1) | cz;

				bool childIsVisible = (childVisibleMask & (1 << childIndex)) != 0;

				if(visiblityMethod == VISIBLITY_ADDING){
					if(childIsVisible) continue;
				}
				// if(node->children[childIndex]){
				// 	if(node->children[childIndex]->visible) continue;
				// }

				int x, y, R; float depth; uint32_t color;
				if(!hqsAcceptPoint(point, node, voxelSplatMetres(nodeSize), x, y, depth, R, color)) continue;
				{

					// HQS depth pass: the same pixels as the colour pass
					// below (forEachSplatPixel), or the depth test and the
					// colour disagree.
					uint32_t udepth = *((uint32_t*)&depth);
					forEachSplatPixel(x, y, R, [&](int ox, int oy, uint32_t pixelID){
						if(udepth < fb_depth[pixelID]){
							atomicMin(&fb_depth[pixelID], udepth);
						}
					});

				}
			}
		}
	}

	grid.sync();

	if(grid.thread_rank() == 0){
		numNodesDrawn = 0;
	}

	grid.sync();

	// COLOR
	while(true){

		block.sync();

		if(block.thread_rank() == 0){
			sh_nodeIndex = atomicAdd(&numNodesDrawn, 1);
		}

		block.sync();

		if(sh_nodeIndex >= numNodes) break;

		Node* node = &nodes[sh_nodeIndex];

		float nodeSize = cubeSize / pow(2.0f, float(node->level));
		float scale = cubeSize / pow(2.0f, float(node->level));
		float3 nodeMin = {
			cubeMin.x + float(node->X + 0.0f) * scale,
			cubeMin.y + float(node->Y + 0.0f) * scale,
			cubeMin.z + float(node->Z + 0.0f) * scale,
		};

		{ // draw points
			Chunk* chunk = node->points;
			int chunkIndex = 0;

			for(int k = block.thread_rank(); ; k += block.num_threads()){
				int pointIndex = keptIndex(k, hqsStepFx);
				if(pointIndex >= int(node->numPoints)) break;

				int targetChunkIndex = pointIndex / POINTS_PER_CHUNK;

				while(chunkIndex < targetChunkIndex && chunk->next != nullptr){
					chunk = chunk->next;
					chunkIndex++;
				}

				// if(chunkIndex > (int(10 * uniforms.time) % 50)) break;

				int pointInChunkIndex = pointIndex % POINTS_PER_CHUNK;
				if(!lodKeep(node, pointIndex)) continue;   // LOD cross-fade
				Point point = g_codec.decode(chunk->points[pointInChunkIndex]);

				int x, y, R; float depth; uint32_t color;
				if(!hqsAcceptPoint(point, node, 0.0f, x, y, depth, R, color)) continue;
				{
					if(!uniforms.colorByNode && !uniforms.colorByLOD){   // debug colours stay unlit, as in drawPoint
						color = repairColour(color, point.x, point.y, point.z);   // blown-out canopy
						uint32_t base = color;
						if(keyLightActive()) color = applyKeyLight(color, lightAt(point.x, point.y, point.z));   // gated, HQS
						color = applyTorch(color, base, point.x, point.y, point.z);
					}
					uint8_t* rgba = (uint8_t*)&color;

					// HQS colour pass (apply_test_source.ps1): the same splat as
					// the depth pass above, plus emission. Emission uses the one
					// shared evalEmission, so HQS and the standard path light the
					// same points the same way. As in drawPoint it is accumulated
					// for every splat pixel, outside the depth tolerance, so
					// overlapping sources sum and a lit point behind a trunk
					// still glows through it.
					float emR, emG, emB;
					evalEmission(point.x, point.y, point.z, emR, emG, emB);
					bool emitting = (emR + emG + emB) > 0.0001f;
					if(emitting) countUp(&_numEmitting);
					countUp(&_numDrawn);   // colour pass only, once per point

					float half = float(R - 1) * 0.5f;
					float* em = emitting ? _emission : nullptr;   // read once a point
					forEachSplatPixel(x, y, R, [&](int ox, int oy, uint32_t pixelID){
						if(em != nullptr){
							atomicAdd(&em[3 * pixelID + 0], emR);
							atomicAdd(&em[3 * pixelID + 1], emG);
							atomicAdd(&em[3 * pixelID + 2], emB);
						}

						float fbDepth = *((float*)&fb_depth[pixelID]);

						if(depth < fbDepth * 1.01f)
						{
							float sh = splatShade(ox, oy, R, half);   // apply_look.ps1
							// The four sums as two 64-bit adds, (green << 32 | red)
							// and (count << 32 | blue): the same layout in memory.
							// A field carries into the next only after 16.8 M
							// contributions to one pixel, where the 32-bit sums
							// used to wrap.
							unsigned long long* c64 = (unsigned long long*)&fb_color[4 * pixelID];
							atomicAdd(&c64[0], (unsigned long long)uint32_t(float(rgba[0]) * sh) | ((unsigned long long)uint32_t(float(rgba[1]) * sh) << 32));
							atomicAdd(&c64[1], (unsigned long long)uint32_t(float(rgba[2]) * sh) | (1ull << 32));
						}
					});
				}
			}
		}

		block.sync();

		uint32_t childVisibleMask = 0;
		for(int i = 0; i < 8; i++){
			Node* child = node->children[i];
			uint32_t childExists = (child != nullptr) ? 1 : 0;
			uint32_t childIsVisible = 0;

			if(childExists){
				childIsVisible = child->visible ? 1 : 0;
			}

			childVisibleMask = childVisibleMask | (childIsVisible << i);
		}

		if(node->numVoxels > 0 && !node->isLeafFn())   // a leaf's voxels duplicate its points; as in drawNodes
		{// draw voxels
			Chunk* chunk = node->voxelChunks;
			int chunkIndex = 0;

			for(int k = block.thread_rank(); ; k += block.num_threads()){
				int pointIndex = keptIndex(k, hqsStepFx);
				if(pointIndex >= int(node->numVoxels)) break;

				int targetChunkIndex = pointIndex / POINTS_PER_CHUNK;

				while(chunkIndex < targetChunkIndex && chunk->next != nullptr){
					chunk = chunk->next;
					chunkIndex++;
				}

				// if(chunkIndex > (int(10 * uniforms.time) % 30)) break;

				int pointInChunkIndex = pointIndex % POINTS_PER_CHUNK;
				if(!lodKeep(node, pointIndex)) continue;   // LOD cross-fade
				Point point = g_codec.decode(chunk->points[pointInChunkIndex]);

				// cull voxels in areas where child nodes are visible
				float3 posuv = (float3{point.x, point.y, point.z} - nodeMin) / nodeSize;
				uint32_t cx = (posuv.x < 0.5f) ? 0 : 1;
				uint32_t cy = (posuv.y < 0.5f) ? 0 : 1;
				uint32_t cz = (posuv.z < 0.5f) ? 0 : 1;
				uint32_t childIndex = (cx << 2) | (cy << 1) | cz;
				// if(node->children[childIndex]){
				// 	if(node->children[childIndex]->visible) continue;
				// }

				bool childIsVisible = (childVisibleMask & (1 << childIndex)) != 0;

				if(visiblityMethod == VISIBLITY_ADDING){
					if(childIsVisible) continue;
				}

				int x, y, R; float depth; uint32_t color;
				if(!hqsAcceptPoint(point, node, voxelSplatMetres(nodeSize), x, y, depth, R, color)) continue;
				{

					if(!uniforms.colorByNode && !uniforms.colorByLOD){   // debug colours stay unlit, as in drawPoint
						color = repairColour(color, point.x, point.y, point.z);   // blown-out canopy
						uint32_t base = color;
						if(keyLightActive()) color = applyKeyLight(color, lightAt(point.x, point.y, point.z));   // gated, HQS
						color = applyTorch(color, base, point.x, point.y, point.z);
					}
					uint8_t* rgba = (uint8_t*)&color;

					// HQS colour pass (apply_test_source.ps1): the same splat as
					// the depth pass above, plus emission. Emission uses the one
					// shared evalEmission, so HQS and the standard path light the
					// same points the same way. As in drawPoint it is accumulated
					// for every splat pixel, outside the depth tolerance, so
					// overlapping sources sum and a lit point behind a trunk
					// still glows through it.
					float emR, emG, emB;
					evalEmission(point.x, point.y, point.z, emR, emG, emB);
					bool emitting = (emR + emG + emB) > 0.0001f;
					if(emitting) countUp(&_numEmitting);
					countUp(&_numDrawn);   // colour pass only, once per point

					float half = float(R - 1) * 0.5f;
					float* em = emitting ? _emission : nullptr;   // read once a point
					forEachSplatPixel(x, y, R, [&](int ox, int oy, uint32_t pixelID){
						if(em != nullptr){
							atomicAdd(&em[3 * pixelID + 0], emR);
							atomicAdd(&em[3 * pixelID + 1], emG);
							atomicAdd(&em[3 * pixelID + 2], emB);
						}

						float fbDepth = *((float*)&fb_depth[pixelID]);

						if(depth < fbDepth * 1.01f)
						{
							float sh = splatShade(ox, oy, R, half);   // apply_look.ps1
							// The four sums as two 64-bit adds, (green << 32 | red)
							// and (count << 32 | blue): the same layout in memory.
							// A field carries into the next only after 16.8 M
							// contributions to one pixel, where the 32-bit sums
							// used to wrap.
							unsigned long long* c64 = (unsigned long long*)&fb_color[4 * pixelID];
							atomicAdd(&c64[0], (unsigned long long)uint32_t(float(rgba[0]) * sh) | ((unsigned long long)uint32_t(float(rgba[1]) * sh) << 32));
							atomicAdd(&c64[1], (unsigned long long)uint32_t(float(rgba[2]) * sh) | (1ull << 32));
						}
					});

				}
			}
		}
	}

	grid.sync();

	// RESOLVE
	processRangeStrided(numPixels, [&](int pixelID){
		
		// one 16-byte load: the Allocator aligns every buffer to 16 bytes,
		// and a pixel's four sums take 16
		ulonglong2 v = ((const ulonglong2*)fb_color)[pixelID];
		uint32_t R = uint32_t(v.x), G = uint32_t(v.x >> 32), B = uint32_t(v.y), C = uint32_t(v.y >> 32);

		if(C == 0) return;

		float depth = *((float*)&fb_depth[pixelID]);

		uint32_t color = 0;
		uint8_t* rgba = (uint8_t*)&color;
		
		rgba[0] = R / C;
		rgba[1] = G / C;
		rgba[2] = B / C;
		rgba[3] = 255;

		uint64_t udepth = *((uint32_t*)&depth);
		uint64_t encoded = (udepth << 32) | color;
		

		framebuffer[pixelID] = encoded;

	});

	grid.sync();
}

void drawNodesBoundingBoxes(
	Node* nodes, uint32_t numNodes, 
	uint64_t* framebuffer,
	float3 cubeMin, float cubeSize,
	Lines* lines
){
	processRange(numNodes, [&](int index){
		Node* node = &nodes[index];

		if(node->numPoints == 0 && node->numVoxels == 0) return;

		float scale = cubeSize / pow(2.0f, float(node->level));

		float3 pos = {
			cubeMin.x + float(node->X + 0.5f) * scale,
			cubeMin.y + float(node->Y + 0.5f) * scale,
			cubeMin.z + float(node->Z + 0.5f) * scale,
		};

		uint32_t color = 0;
		if(node->isLeafFn()){
			color = 0x0000ff00;
		}else{
			color = 0x0000ffff;
		}

		// color = 0x00ffffff;
		// color = 0x00000000;
		color = 0x0000ffff;
		// color = SPECTRAL[(4 - node->level) % 8];
		if(node->level == 2){
			color = SPECTRAL[6];
		} else if(node->level == 3){
			color = SPECTRAL[4];
		}else if(node->level == 4){
			color = SPECTRAL[0];
		}else if(node->level == 5){
			color = SPECTRAL[6];
		}

		color = 0x0000ff00;

		drawBoundingBox(lines, pos, {scale, scale, scale}, color);   // once: a loop added three identical copies
	});
}

// The visibleNodes buffer allocated in kernel_render holds MAX_VISIBLE_NODES
// (HostDeviceInterface.h). Writes past the cap are dropped and the count is
// clamped after visibility.

void compute_visibility_disjunct(
	uint32_t& numVisibleNodes,
	uint32_t& numVisiblePoints,
	uint32_t& numVisibleVoxels,
	uint32_t& numVisibleInner,
	uint32_t& numVisibleLeaves,
	uint32_t& dbg_numSmallNodes,
	uint32_t& dbg_numSmallNodePoints,
	Stats* stats, Node* nodes, Node* visibleNodes,
	float cubeSize, float3 cubeMin
){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	auto min8 = [](float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7){

		float m0 = min(f0, f1);
		float m1 = min(f2, f3);
		float m2 = min(f4, f5);
		float m3 = min(f6, f7);

		float n0 = min(m0, m1);
		float n1 = min(m2, m3);

		return min(n0, n1);
	};

	auto max8 = [](float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7){

		float m0 = max(f0, f1);
		float m1 = max(f2, f3);
		float m2 = max(f4, f5);
		float m3 = max(f6, f7);

		float n0 = max(m0, m1);
		float n1 = max(m2, m3);

		return max(n0, n1);
	};

	float fwidth = uniforms.width;
	float fheight = uniforms.height;

	grid.sync();
	if(grid.thread_rank() == 0){
		numVisibleNodes  = 0;
		numVisiblePoints = 0;
		numVisibleVoxels = 0;
		numVisibleInner  = 0;
		numVisibleLeaves = 0;
		dbg_numSmallNodes = 0;
		dbg_numSmallNodePoints = 0;
	}
	grid.sync();

	auto makeVisible = [&](Node* node, uint32_t lodRole, float lodBlend){
		uint32_t visibleNodeIndex = atomicAdd(&numVisibleNodes, 1);
		if(visibleNodeIndex >= MAX_VISIBLE_NODES) return;
		visibleNodes[visibleNodeIndex] = *node;
		visibleNodes[visibleNodeIndex].lodRole  = lodRole;
		visibleNodes[visibleNodeIndex].lodBlend = lodBlend;

		if(node->numPoints > 0){
			atomicAdd(&numVisibleLeaves, 1);
			atomicAdd(&numVisiblePoints, node->numPoints);
		}else if(node->numVoxels > 0){
			atomicAdd(&numVisibleInner, 1);
			atomicAdd(&numVisibleVoxels, node->numVoxels);
		}

		// const char* name = (const char*)node->name;
		// printf("node: %s \n", name);
	};

	processRange(stats->numNodes, [&](int nodeIndex){
		Node* node = &nodes[nodeIndex];

		float nodeSize = cubeSize / pow(2.0f, float(node->level));
		float3 nodeMin = {
			cubeMin.x + float(node->X + 0.0f) * nodeSize,
			cubeMin.y + float(node->Y + 0.0f) * nodeSize,
			cubeMin.z + float(node->Z + 0.0f) * nodeSize,
		};
		float3 nodeMax = {
			cubeMin.x + float(node->X + 1.0f) * nodeSize,
			cubeMin.y + float(node->Y + 1.0f) * nodeSize,
			cubeMin.z + float(node->Z + 1.0f) * nodeSize,
		};
		float3 nodeCenter = {
			cubeMin.x + float(node->X + 0.5f) * nodeSize,
			cubeMin.y + float(node->Y + 0.5f) * nodeSize,
			cubeMin.z + float(node->Z + 0.5f) * nodeSize,
		};

		// compute node boundaries in screen space
		float4 p000 = {nodeMin.x, nodeMin.y, nodeMin.z, 1.0f};
		float4 p001 = {nodeMin.x, nodeMin.y, nodeMax.z, 1.0f};
		float4 p010 = {nodeMin.x, nodeMax.y, nodeMin.z, 1.0f};
		float4 p011 = {nodeMin.x, nodeMax.y, nodeMax.z, 1.0f};
		float4 p100 = {nodeMax.x, nodeMin.y, nodeMin.z, 1.0f};
		float4 p101 = {nodeMax.x, nodeMin.y, nodeMax.z, 1.0f};
		float4 p110 = {nodeMax.x, nodeMax.y, nodeMin.z, 1.0f};
		float4 p111 = {nodeMax.x, nodeMax.y, nodeMax.z, 1.0f};

		float4 ndc000 = uniforms.transform_updateBound * p000;
		float4 ndc001 = uniforms.transform_updateBound * p001;
		float4 ndc010 = uniforms.transform_updateBound * p010;
		float4 ndc011 = uniforms.transform_updateBound * p011;
		float4 ndc100 = uniforms.transform_updateBound * p100;
		float4 ndc101 = uniforms.transform_updateBound * p101;
		float4 ndc110 = uniforms.transform_updateBound * p110;
		float4 ndc111 = uniforms.transform_updateBound * p111;

		float4 s000 = ((ndc000 / ndc000.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s001 = ((ndc001 / ndc001.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s010 = ((ndc010 / ndc010.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s011 = ((ndc011 / ndc011.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s100 = ((ndc100 / ndc100.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s101 = ((ndc101 / ndc101.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s110 = ((ndc110 / ndc110.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s111 = ((ndc111 / ndc111.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};

		float smin_x = min8(s000.x, s001.x, s010.x, s011.x, s100.x, s101.x, s110.x, s111.x);
		float smin_y = min8(s000.y, s001.y, s010.y, s011.y, s100.y, s101.y, s110.y, s111.y);

		float smax_x = max8(s000.x, s001.x, s010.x, s011.x, s100.x, s101.x, s110.x, s111.x);
		float smax_y = max8(s000.y, s001.y, s010.y, s011.y, s100.y, s101.y, s110.y, s111.y);

		// screen-space size
		float dx = smax_x - smin_x;
		float dy = smax_y - smin_y;

		// Fisheye (fix_crash_instrument.ps1): with frustum culling off, nodes
		// beside or behind the eye project through the perspective matrix
		// with w near zero and come out enormous, so every one was refined to
		// its leaves - 28 M points drawn at 91 ms. In fisheye the projection
		// is linear in angle, so size a node by the angle it subtends:
		// pixels per radian is (height/2) / (fov/2).
		// apply_markers2.ps1: the same blow-up happens in PERSPECTIVE whenever a
		// node corner is behind the eye (w <= 0), which is every node around the
		// camera once it stands inside the cloud. Size those by angle as well.
		float wmin = min8(ndc000.w, ndc001.w, ndc010.w, ndc011.w, ndc100.w, ndc101.w, ndc110.w, ndc111.w);
		if(uniforms.lensMode != 0 || wmin <= 0.0f){
			float ddx = 0.5f * (nodeMin.x + nodeMax.x) - uniforms.camPos.x;
			float ddy = 0.5f * (nodeMin.y + nodeMax.y) - uniforms.camPos.y;
			float ddz = 0.5f * (nodeMin.z + nodeMax.z) - uniforms.camPos.z;
			float dist = sqrtf(ddx*ddx + ddy*ddy + ddz*ddz);
			if(dist < 0.5f * nodeSize) dist = 0.5f * nodeSize;   // inside the node: treat as large
			float halfFov = uniforms.fisheyeFov * 0.5f * 0.017453292f;
			float pxPerRad = (uniforms.lensMode != 0) ? (0.5f * fheight) / halfFov : uniforms.focalPx;   // apply_markers2.ps1
			float ang = 1.7320508f * nodeSize / dist;               // diagonal subtended, radians
			dx = ang * pxPerRad;
			dy = dx;
		}

		// {
		// 	float sx_x = abs(s100.x - s000.x);
		// 	float sx_y = abs(s100.y - s000.y);
		// 	float sy_x = abs(s010.x - s000.x);
		// 	float sy_y = abs(s010.y - s000.y);
		// 	float sz_x = abs(s001.x - s000.x);
		// 	float sz_y = abs(s001.y - s000.y);

		// 	smax_x = max(max(sx_x, sy_x), sz_x);
		// 	smax_y = max(max(sx_y, sy_y), sz_y);

		// 	dx = 1.0 * smax_x;
		// 	dy = 1.0 * smax_y;
		// }

		// {
		// 	float4 proj = uniforms.transform_updateBound * float4{nodeCenter.x, nodeCenter.y, nodeCenter.z, 1.0};
		// 	float d = proj.w;
		// 	float r = nodeSize;
		// 	float aspect = fwidth / fheight;
		// 	float pr = 1.0 / tan(uniforms.fovy_rad) * r / sqrt(d * d - r * r);
		// 	float pixels = pr * fheight * 2.0;
		// 	float projSize = aspect * fheight * nodeSize / proj.w;
		// 	if(proj.w > 0.0){
		// 		dx = projSize;
		// 		dy = projSize;

		// 		dx = pixels;
		// 		dy = pixels;
		// 	}
		// }

		bool isIntersectingFrustum = (uniforms.lensMode != 0) || intersectsFrustum(uniforms.transform_updateBound, nodeMin, nodeMax);   // fisheye: no frustum
		bool hasSamples = node->numPoints > 0 || node->numVoxels > 0;

		bool visible = true;

		if(!isIntersectingFrustum) visible = false;
		if(!hasSamples) visible = false;

		node->visible = visible;
		node->isLarge = dx > 2.0f * uniforms.minNodeSize || dy > 2.0f * uniforms.minNodeSize;
		node->screenSize = fmaxf(dx, dy);

		// if(strequal(node->name, "r")){

		// 	bool isLarge = false;
		// 	{
		// 		float sx_x = abs(s100.x - s000.x);
		// 		float sx_y = abs(s100.y - s000.y);
		// 		float sy_x = abs(s010.x - s000.x);
		// 		float sy_y = abs(s010.y - s000.y);
		// 		float sz_x = abs(s001.x - s000.x);
		// 		float sz_y = abs(s001.y - s000.y);

		// 		smax_x = max(max(sx_x, sy_x), sz_x);
		// 		smax_y = max(max(sx_y, sy_y), sz_y);

		// 		dx = 1.0 * smax_x;
		// 		dy = 1.0 * smax_y;

		// 		isLarge = dx > 2.0 * uniforms.minNodeSize || dy > 2.0 * uniforms.minNodeSize;
		// 	}

		// 	uint32_t color = 0x0000ffff;
		// 	if(isLarge) color = 0x00ffffff;

		// 	drawLine(_lines, make_float3(p000), make_float3(p100), color);
		// 	drawLine(_lines, make_float3(p000), make_float3(p010), color);
		// 	drawLine(_lines, make_float3(p000), make_float3(p001), color);

		// 	drawLine(_lines, make_float3(p111), make_float3(p011), 0xffffffff - color);
		// 	drawLine(_lines, make_float3(p111), make_float3(p101), 0xffffffff - color);
		// 	drawLine(_lines, make_float3(p111), make_float3(p110), 0xffffffff - color);

		// 	makeVisible(node);


		// }else{
		// 	node->visible = false;
		// }

	});

	grid.sync();

	// if(false)
	processRange(stats->numNodes, [&](int nodeIndex){
		Node* node = &nodes[nodeIndex];

		// Banyunibo 7 & 10
		// if(strequal(node->name, "r06226")){
		// if(strequal(node->name, "r062263")){
		// 	makeVisible(node);
		// 	printf("#samples: %i \n", node->numPoints);
		// }
		// return;


		if(node->isLarge && !node->isLeafFn()){

			for(int i = 0; i < 8; i++){
				Node* child = node->children[i];

				if(child == nullptr) continue;
				if(child->isLarge) continue;
				if(!child->visible) continue;

				// Cross-fade across the level switch. A drawn node refines once
				// it passes 2 x minNodeSize on screen. From 1 x upward its own
				// samples thin out while its children's fill in, so at the
				// switch it is already fully replaced and nothing pops, and
				// neighbouring nodes at different levels meet with matching
				// density instead of a hard edge.
				float t = 0.0f;
				if(!child->isLeafFn()){
					t = child->screenSize / uniforms.minNodeSize - 1.0f;
					t = fminf(fmaxf(t, 0.0f), 1.0f);
					t = t * t * (3.0f - 2.0f * t);
				}

				if(t <= 0.0f){
					makeVisible(child, 0, 0.0f);
					continue;
				}

				makeVisible(child, 1, t);
				for(int j = 0; j < 8; j++){
					Node* gc = child->children[j];
					if(gc == nullptr || !gc->visible) continue;
					if(gc->numPoints == 0 && gc->numVoxels == 0) continue;
					makeVisible(gc, 2, t);
				}
			}
		}else if(node->isLarge && node->isLeafFn() && node->visible){
			makeVisible(node, 0, 0.0f);
		}else if(nodeIndex == 0 && !node->isLarge && node->visible){
			// The root has no parent to draw it, so a cloud whose whole cube
			// is under 2 x minNodeSize on screen drew nothing.
			makeVisible(node, 0, 0.0f);
		}
	});
}

void compute_visibility_add(
	uint32_t& numVisibleNodes,
	uint32_t& numVisiblePoints,
	uint32_t& numVisibleVoxels,
	uint32_t& numVisibleInner,
	uint32_t& numVisibleLeaves,
	uint32_t& dbg_numSmallNodes,
	uint32_t& dbg_numSmallNodePoints,
	Stats* stats, Node* nodes, Node* visibleNodes,
	float cubeSize, float3 cubeMin
){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	auto min8 = [](float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7){

		float m0 = min(f0, f1);
		float m1 = min(f2, f3);
		float m2 = min(f4, f5);
		float m3 = min(f6, f7);

		float n0 = min(m0, m1);
		float n1 = min(m2, m3);

		return min(n0, n1);
	};

	auto max8 = [](float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7){

		float m0 = max(f0, f1);
		float m1 = max(f2, f3);
		float m2 = max(f4, f5);
		float m3 = max(f6, f7);

		float n0 = max(m0, m1);
		float n1 = max(m2, m3);

		return max(n0, n1);
	};

	float fwidth = uniforms.width;
	float fheight = uniforms.height;

	grid.sync();
	if(grid.thread_rank() == 0){
		numVisibleNodes  = 0;
		numVisiblePoints = 0;
		numVisibleVoxels = 0;
		numVisibleInner  = 0;
		numVisibleLeaves = 0;
		dbg_numSmallNodes = 0;
		dbg_numSmallNodePoints = 0;
	}
	grid.sync();

	processRange(stats->numNodes, [&](int nodeIndex){
		Node* node = &nodes[nodeIndex];

		float nodeSize = cubeSize / pow(2.0f, float(node->level));
		// node-center
		float ncx = (float(node->X) + 0.5f) * nodeSize + cubeMin.x;
		float ncy = (float(node->Y) + 0.5f) * nodeSize + cubeMin.y;
		float ncz = (float(node->Z) + 0.5f) * nodeSize + cubeMin.z;

		float scale = cubeSize / pow(2.0f, float(node->level));
		float3 nodeMin = {
			cubeMin.x + float(node->X + 0.0f) * scale,
			cubeMin.y + float(node->Y + 0.0f) * scale,
			cubeMin.z + float(node->Z + 0.0f) * scale,
		};
		float3 nodeMax = {
			cubeMin.x + float(node->X + 1.0f) * scale,
			cubeMin.y + float(node->Y + 1.0f) * scale,
			cubeMin.z + float(node->Z + 1.0f) * scale,
		};

		bool visible = true;

		// compute node boundaries in screen space
		float4 p000 = {nodeMin.x, nodeMin.y, nodeMin.z, 1.0f};
		float4 p001 = {nodeMin.x, nodeMin.y, nodeMax.z, 1.0f};
		float4 p010 = {nodeMin.x, nodeMax.y, nodeMin.z, 1.0f};
		float4 p011 = {nodeMin.x, nodeMax.y, nodeMax.z, 1.0f};
		float4 p100 = {nodeMax.x, nodeMin.y, nodeMin.z, 1.0f};
		float4 p101 = {nodeMax.x, nodeMin.y, nodeMax.z, 1.0f};
		float4 p110 = {nodeMax.x, nodeMax.y, nodeMin.z, 1.0f};
		float4 p111 = {nodeMax.x, nodeMax.y, nodeMax.z, 1.0f};

		float4 ndc000 = uniforms.transform_updateBound * p000;
		float4 ndc001 = uniforms.transform_updateBound * p001;
		float4 ndc010 = uniforms.transform_updateBound * p010;
		float4 ndc011 = uniforms.transform_updateBound * p011;
		float4 ndc100 = uniforms.transform_updateBound * p100;
		float4 ndc101 = uniforms.transform_updateBound * p101;
		float4 ndc110 = uniforms.transform_updateBound * p110;
		float4 ndc111 = uniforms.transform_updateBound * p111;

		float4 s000 = ((ndc000 / ndc000.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s001 = ((ndc001 / ndc001.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s010 = ((ndc010 / ndc010.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s011 = ((ndc011 / ndc011.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s100 = ((ndc100 / ndc100.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s101 = ((ndc101 / ndc101.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s110 = ((ndc110 / ndc110.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s111 = ((ndc111 / ndc111.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};

		float smin_x = min8(s000.x, s001.x, s010.x, s011.x, s100.x, s101.x, s110.x, s111.x);
		float smin_y = min8(s000.y, s001.y, s010.y, s011.y, s100.y, s101.y, s110.y, s111.y);

		float smax_x = max8(s000.x, s001.x, s010.x, s011.x, s100.x, s101.x, s110.x, s111.x);
		float smax_y = max8(s000.y, s001.y, s010.y, s011.y, s100.y, s101.y, s110.y, s111.y);

		// screen-space size
		float dx = smax_x - smin_x;
		float dy = smax_y - smin_y;

		// Fisheye (fix_crash_instrument.ps1): with frustum culling off, nodes
		// beside or behind the eye project through the perspective matrix
		// with w near zero and come out enormous, so every one was refined to
		// its leaves - 28 M points drawn at 91 ms. In fisheye the projection
		// is linear in angle, so size a node by the angle it subtends:
		// pixels per radian is (height/2) / (fov/2).
		// apply_markers2.ps1: the same blow-up happens in PERSPECTIVE whenever a
		// node corner is behind the eye (w <= 0), which is every node around the
		// camera once it stands inside the cloud. Size those by angle as well.
		float wmin = min8(ndc000.w, ndc001.w, ndc010.w, ndc011.w, ndc100.w, ndc101.w, ndc110.w, ndc111.w);
		if(uniforms.lensMode != 0 || wmin <= 0.0f){
			float ddx = 0.5f * (nodeMin.x + nodeMax.x) - uniforms.camPos.x;
			float ddy = 0.5f * (nodeMin.y + nodeMax.y) - uniforms.camPos.y;
			float ddz = 0.5f * (nodeMin.z + nodeMax.z) - uniforms.camPos.z;
			float dist = sqrtf(ddx*ddx + ddy*ddy + ddz*ddz);
			if(dist < 0.5f * nodeSize) dist = 0.5f * nodeSize;   // inside the node: treat as large
			float halfFov = uniforms.fisheyeFov * 0.5f * 0.017453292f;
			float pxPerRad = (uniforms.lensMode != 0) ? (0.5f * fheight) / halfFov : uniforms.focalPx;   // apply_markers2.ps1
			float ang = 1.7320508f * nodeSize / dist;               // diagonal subtended, radians
			dx = ang * pxPerRad;
			dy = dx;
		}

		if(uniforms.lensMode == 0 && !intersectsFrustum(uniforms.transform_updateBound, nodeMin, nodeMax)){   // fisheye: no frustum cull
			visible = false;
		}

		if(dx < uniforms.minNodeSize || dy < uniforms.minNodeSize){
			visible = false;
		}

		if(node->numPoints == 0 && node->numVoxels == 0){
			visible = false;
		}

		node->visible = visible;
		
		if(visible){
			uint32_t visibleNodeIndex = atomicAdd(&numVisibleNodes, 1);
			if(visibleNodeIndex >= MAX_VISIBLE_NODES) return;
			visibleNodes[visibleNodeIndex] = nodes[nodeIndex];

			if(node->numPoints > 0){
				atomicAdd(&numVisibleLeaves, 1);
				atomicAdd(&numVisiblePoints, node->numPoints);
			}else if(node->numVoxels > 0){
				atomicAdd(&numVisibleInner, 1);
				atomicAdd(&numVisibleVoxels, node->numVoxels);
			}
		}
	});

	grid.sync();
}

// The host launches this with 256-thread blocks (workgroupSize in renderCUDA),
// as many as fit on each SM, and every stage of the frame shares those
// threads. Two blocks fit only at 128 registers per thread or fewer. With the
// uniforms in constant memory the kernel needs 117 at compute 8.6 and 8.9
// (NVRTC 12.4) and spills nothing; the bound stays so that a later change
// cannot quietly halve the threads drawing points. Pascal ran one block
// (about 130 registers) and is left unbounded.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 && __CUDA_ARCH__ < 700
#define RENDER_LAUNCH_BOUNDS
#else
#define RENDER_LAUNCH_BOUNDS __launch_bounds__(256, 2)
#endif

extern "C" __global__
void RENDER_LAUNCH_BOUNDS kernel_render(
	uint32_t* buffer,
	const Uniforms _uniforms,
	Node* nodes,
	cudaSurfaceObject_t gl_colorbuffer,
	Stats* stats,
	uint64_t* frameStartTimestamp,
	CudaPrint* cudaprint
){

	auto tStart = nanotime();

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	if(grid.thread_rank() == 0){
		*frameStartTimestamp = nanotime();
	}

	// The uniforms are already in constant memory, copied by the host before
	// the launch. Thread 0 sets up the per-frame globals derived from them.
	if(grid.thread_rank() == 0){
		g_codec  = PointCodec::forBox(_uniforms.boxMin, _uniforms.boxMax);   // stored points
		float3 td;
		_torchValid = (uniforms.torchOn != 0 && pixelRayDir(uniforms.torchX + 0.5f, uniforms.torchY + 0.5f, td)) ? 1 : 0;
		if(_torchValid) _torchDir = td;
	}
	grid.sync();
	Allocator _allocator(buffer, 0);
	allocator = &_allocator;

	Node* visibleNodes                 = allocator->alloc<Node*>(uint64_t(MAX_VISIBLE_NODES) * sizeof(Node));
	uint32_t& numVisibleNodes          = *allocator->alloc<uint32_t*>(4);
	uint32_t& numVisiblePoints         = *allocator->alloc<uint32_t*>(4);
	uint32_t& numVisibleVoxels         = *allocator->alloc<uint32_t*>(4);
	uint32_t& numVisibleInner          = *allocator->alloc<uint32_t*>(4);
	uint32_t& numVisibleLeaves         = *allocator->alloc<uint32_t*>(4);
	uint32_t& dbg_numSmallNodes        = *allocator->alloc<uint32_t*>(4);
	uint32_t& dbg_numSmallNodePoints   = *allocator->alloc<uint32_t*>(4);

	Lines* lines = allocator->alloc<Lines*>(sizeof(Lines));
	lines->count = 0;
	lines->vertices = allocator->alloc<Point*>(int64_t(MAX_LINE_VERTICES) * sizeof(Point));
	_lines = lines;

	int framebufferSize = int(uniforms.width) * int(uniforms.height) * sizeof(uint64_t);
	uint64_t* framebuffer = allocator->alloc<uint64_t*>(framebufferSize);

	// half-resolution dimensions, used by the bloom composite sample
	int bloomW = int(uniforms.width)  / 2;
	int bloomH = int(uniforms.height) / 2;

	// full-resolution RGB emission accumulation
	_emission = allocator->alloc<float*>(int(uniforms.width) * int(uniforms.height) * 3 * sizeof(float));

	// Bloom mip chain. Six half-then-quarter levels sum to under a third of a
	// half-res buffer, so one full-res-sized allocation is ample.
	// A mip chain from half resolution sums to (W*H/4)*3 * (1 + 1/4 + 1/16 ...)
	// = W*H floats. The previous W*H*3 was three times what is used.
	float* bloomChain = allocator->alloc<float*>((int(uniforms.width) * int(uniforms.height) + 8192) * sizeof(float));

	// volumetric scattering, accumulated at quarter resolution. Volumetrics are
	// low frequency, so a quarter-res march upsampled is indistinguishable from
	// full res and sixteen times cheaper. Four floats per pixel: RGB, and the
	// distance to the surface the ray sees (SKY_DISTANCE for sky), which the
	// upsample matches against each full-res pixel's own.
	int volW = int(uniforms.width)  / 4; if(volW < 1) volW = 1;
	int volH = int(uniforms.height) / 4; if(volH < 1) volH = 1;
	float* volBuf = allocator->alloc<float*>(volW * volH * 4 * sizeof(float));

	// the high-quality passes' depth (4 B a pixel) and colour sums (16 B)
	uint32_t* hqsDepth = allocator->alloc<uint32_t*>(int64_t(uniforms.width) * int64_t(uniforms.height) * 4);
	uint32_t* hqsColor = allocator->alloc<uint32_t*>(int64_t(uniforms.width) * int64_t(uniforms.height) * 16);

	// The host sizes the buffer for the window (renderBufferBytesFor) and
	// skips the launch when it cannot. Should the frame's layout ever outgrow
	// it, draw nothing rather than write past its end: a fixed 200 MB used to
	// be overrun by any window above 3.8 Mpx. Only small allocations come
	// later in the frame, where returning is no longer safe (a counter in
	// drawNodes and drawNodesHQS); 1 KB covers them. Every thread computes
	// the same offset, so the return is uniform across the grid.
	int64_t laterBytes = 1024;
	if(allocator->offset + laterBytes > int64_t(uniforms.renderBufferCapacity)){
		return;
	}

	if(grid.thread_rank() == 0){ _numEmitting = 0; _numDrawn = 0; }

	// Recount the occupancy grid first: the light grid below is derived from
	// these counts, and the host requests both together after a reallocation.
	grid.sync();
	refillVoxelGrid(nodes, stats);   // only when the host says the counts are stale

	// ---- build the light transmittance grid (apply_keylight.ps1) ----------
	// Only when the host says it is stale: after a load, or when a light
	// control moves. Otherwise this is skipped entirely and the light costs
	// one texture-like lookup per point.
	if(uniforms.lightOn != 0 && uniforms.lightDirty != 0 && uniforms.lightGridPtr != 0 && uniforms.voxelGridPtr != 0){
		grid.sync();
		int nCells = uniforms.gridX * uniforms.gridY * uniforms.gridZ;
		processRangeStrided(nCells, [&](int i){
			int cx = i % uniforms.gridX;
			int cy = (i / uniforms.gridX) % uniforms.gridY;
			int cz = i / (uniforms.gridX * uniforms.gridY);

			float trans = 1.0f;
			for(int s = 1; s <= uniforms.lightSteps; s++){
				// round to nearest; int(x + 0.5f) truncates toward zero, which
				// shortened shadows for negative light-direction components
				int sx = cx + __float2int_rn(uniforms.lightDirX * float(s));
				int sy = cy + __float2int_rn(uniforms.lightDirY * float(s));
				int sz = cz + __float2int_rn(uniforms.lightDirZ * float(s));
				// left the grid: open sky from here on, stop accumulating
				if(sx < 0 || sy < 0 || sz < 0) break;
				if(sx >= uniforms.gridX || sy >= uniforms.gridY || sz >= uniforms.gridZ) break;
				const uint32_t* vg = (const uint32_t*)uniforms.voxelGridPtr;
				if(vg[sx + uniforms.gridX * (sy + uniforms.gridY * sz)] > 0){
					trans *= __expf(-uniforms.lightAbsorb * uniforms.cellSize);   // per METRE: independent of grid resolution
					if(trans < 0.004f) break;
				}
			}
			((float*)uniforms.lightGridPtr)[i] = trans;
		});
		grid.sync();
	}
	grid.sync();
	// the grid rebuilds end here; adaptive detail leaves them out
	if(grid.thread_rank() == 0) stats->stageNs[7] = nanotime() - tStart;

	// Only an event (a call playing, or the test source) puts anything in the
	// emission buffer: evalEmission adds nothing without one, nor in the
	// unedited view. Without one the buffer is neither cleared nor read, and
	// the bloom, whose only source it is, does not run: 24 B a pixel less
	// memory traffic, 199 MB a frame at 4K.
	const bool emissive = uniforms.numEvents > 0 && !showUnedited();

	// clear framebuffer
	// integer product: as a float it rounds above 2^24 pixels (--render 4097 4097)
	processRangeStrided(int(uniforms.width) * int(uniforms.height), [&](int pixelIndex){
		// depth:            7f800000 (Infinity)
		// background color: 00332211 (aabbggrr)
		// framebuffer[pixelIndex] = 0x7f800000'00332211ull;
		// sky colour from uniforms (apply_features.ps1) - replaces the constant
		{
			uint32_t sr = uint32_t(fminf(fmaxf(uniforms.skyR, 0.0f), 1.0f) * 255.0f + 0.5f);
			uint32_t sg = uint32_t(fminf(fmaxf(uniforms.skyG, 0.0f), 1.0f) * 255.0f + 0.5f);
			uint32_t sb = uint32_t(fminf(fmaxf(uniforms.skyB, 0.0f), 1.0f) * 255.0f + 0.5f);
			uint32_t sky = (sb << 16) | (sg << 8) | sr;
			framebuffer[pixelIndex] = (0x7f800000ull << 32) | uint64_t(sky);
		}
		if(emissive){
			_emission[3 * pixelIndex + 0] = 0.0f;
			_emission[3 * pixelIndex + 1] = 0.0f;
			_emission[3 * pixelIndex + 2] = 0.0f;
		}
	});

	grid.sync();

	float3 boxSize = uniforms.boxMax - uniforms.boxMin;
	float cubeSize = max(max(boxSize.x, boxSize.y), boxSize.z);
	float3 cubeMin = uniforms.boxMin;
	float3 cubeMax = cubeMin + cubeSize;
	float3 cubePosition = uniforms.boxMin + cubeSize * 0.5f;

	grid.sync();
	
	{
		if(visiblityMethod == VISIBLITY_DISJUNCT){
			compute_visibility_disjunct(
				numVisibleNodes,
				numVisiblePoints,
				numVisibleVoxels,
				numVisibleInner,
				numVisibleLeaves,
				dbg_numSmallNodes,
				dbg_numSmallNodePoints,
				stats, nodes, visibleNodes,
				cubeSize, cubeMin
			);
		}else if(visiblityMethod == VISIBLITY_ADDING){
			compute_visibility_add(
				numVisibleNodes,
				numVisiblePoints,
				numVisibleVoxels,
				numVisibleInner,
				numVisibleLeaves,
				dbg_numSmallNodes,
				dbg_numSmallNodePoints,
				stats, nodes, visibleNodes,
				cubeSize, cubeMin
			);
		}
	}

	// draw all nodes
	// visibleNodes = nodes;
	// numVisibleNodes = numNodes;

	grid.sync();

	// the counter keeps counting past the cap; draw only what was stored
	if(grid.thread_rank() == 0 && numVisibleNodes > MAX_VISIBLE_NODES){
		numVisibleNodes = MAX_VISIBLE_NODES;
	}
	grid.sync();

	{ // draw nodes, boxes, lines, ...
		if(grid.thread_rank() == 0) stats->stageNs[0] = nanotime() - tStart;   // apply_stage_timers.ps1
		if(uniforms.useHighQualityShading){
			drawNodesHQS(visibleNodes, numVisibleNodes, framebuffer, hqsDepth, hqsColor);
		}else{
			drawNodes(visibleNodes, numVisibleNodes, framebuffer);
		}

		grid.sync();

		// ---- pick -----------------------------------------------------------
		// After the scan is drawn but before anything else (bounding boxes and
		// the frustum, the voxel-grid view, event markers), so a pick always
		// lands on the scan: read the depth under the requested pixel, turn it
		// back into a world position, hand it to the host via Stats. Each
		// request id is answered exactly once and the answer is left in place,
		// so the host cannot miss it however far its asynchronous Stats copy
		// lags behind. It came after the bounding boxes and the frustum, so
		// with those shown a pick could land on one of their lines.
		if(grid.thread_rank() == 0 && uniforms.pickSeq != stats->pickSeqDone){
			int valid = 0;
			// Reject the background explicitly. The clear writes 0x7f800000, which
			// is +infinity; comparing a float against 1.0e30 relies on that bit
			// pattern decoding cleanly, and a NaN would pass every comparison.
			// Test the bits.
			if(uniforms.pickX >= 0 && uniforms.pickY >= 0
			   && uniforms.pickX < int(uniforms.width) && uniforms.pickY < int(uniforms.height)){
				uint64_t fbv = framebuffer[uniforms.pickX + int(uniforms.width) * uniforms.pickY];
				uint32_t dbits = uint32_t(fbv >> 32);
				float depth = *((float*)&dbits);
				float3 rd;
				bool hitGeometry = (dbits != 0x7f800000u)          // not the cleared background
								&& ((dbits & 0x7f800000u) != 0x7f800000u)   // not inf or NaN
								&& depth > 0.0f && depth < 1.0e6f;
				if(hitGeometry && pixelRayDir(float(uniforms.pickX) + 0.5f, float(uniforms.pickY) + 0.5f, rd)){
					float dist = depthToDistance(depth, rd);
					stats->pickWX = uniforms.camPos.x + rd.x * dist;
					stats->pickWY = uniforms.camPos.y + rd.y * dist;
					stats->pickWZ = uniforms.camPos.z + rd.z * dist;
					valid = 1;
				}
			}
			stats->pickValid = valid;
			__threadfence();
			stats->pickSeqDone = uniforms.pickSeq;   // written last
		}
		grid.sync();   // nothing may draw over the pixel while thread 0 reads it

		if(uniforms.showBoundingBox)
		if(grid.thread_rank() == 0)
		{// draw frustum

			auto project = [](float3 coordinate){
				
				float4 pos = uniforms.transformInv_updateBound * float4{coordinate.x, coordinate.y, coordinate.z, 1.0};

				pos.x = pos.x / pos.w;
				pos.y = pos.y / pos.w;
				pos.z = pos.z / pos.w;

				return make_float3(pos);
			};

			float fend = 0.99995;

			drawLine(lines, project({ 1.0,  1.0, -1.0}), project({ 1.0,  1.0, fend}), 0x000000ff);
			drawLine(lines, project({ 1.0, -1.0, -1.0}), project({ 1.0, -1.0, fend}), 0x000000ff);
			drawLine(lines, project({-1.0,  1.0, -1.0}), project({-1.0,  1.0, fend}), 0x000000ff);
			drawLine(lines, project({-1.0, -1.0, -1.0}), project({-1.0, -1.0, fend}), 0x000000ff);

			drawLine(lines, project({-1.0, -1.0, fend}), project({ 1.0, -1.0, fend}), 0x000000ff);
			drawLine(lines, project({-1.0,  1.0, fend}), project({ 1.0,  1.0, fend}), 0x000000ff);
			drawLine(lines, project({-1.0, -1.0, fend}), project({-1.0,  1.0, fend}), 0x000000ff);
			drawLine(lines, project({ 1.0, -1.0, fend}), project({ 1.0,  1.0, fend}), 0x000000ff);
		}

		grid.sync();

		if(uniforms.showBoundingBox){ 
			drawNodesBoundingBoxes(visibleNodes, numVisibleNodes, framebuffer, cubeMin, cubeSize, lines);
		}

		grid.sync();

		rasterizeLines(lines, framebuffer, uniforms.width, uniforms.height, uniforms.transform);

		grid.sync();
	}

	// if(grid.thread_rank() == 0){
	// 	double duration_ms = double(nanotime() - tStart) / 1'000'000.0;
	// 	printf("ms: %f \n", duration_ms);
	// }

	// UPDATE STATS
	if(grid.thread_rank() == 0){
		// printf("update stats! \n");
		stats->numVisibleNodes  = numVisibleNodes;
		stats->numVisibleInner  = numVisibleInner;
		stats->numVisibleLeaves = numVisibleLeaves;
		stats->numVisiblePoints = numVisiblePoints;
		stats->numVisibleVoxels = numVisibleVoxels;
		stats->frameID          = uniforms.frameCounter;
		stats->numEmitting      = _numEmitting;
		stats->numDrawn         = _numDrawn;
	}

	// if(false)
	if(grid.thread_rank() == 0) stats->stageNs[1] = nanotime() - tStart;   // apply_stage_timers.ps1

	// ---- voxel grid debug view (apply_voxel_grid.ps1) ------------------
	// Draws the centre of every occupied cell, coloured by count: blue for
	// sparse, through green, to red for dense. A verification view, like the
	// axes - it should sit exactly on the foliage and nowhere else.
	if(uniforms.useVoxelGrid && uniforms.showVoxelGrid && uniforms.voxelGridPtr != 0){
		int nCells = uniforms.gridX * uniforms.gridY * uniforms.gridZ;
		processRangeStrided(nCells, [&](int i){
			const uint32_t* g = (const uint32_t*)uniforms.voxelGridPtr;
			uint32_t n = g[i];
			if(n == 0) return;
			int cx = i % uniforms.gridX;
			int cy = (i / uniforms.gridX) % uniforms.gridY;
			int cz = i / (uniforms.gridX * uniforms.gridY);
			float c = uniforms.cellSize;
			float px = (float(cx) + 0.5f) * c;
			float py = (float(cy) + 0.5f) * c;
			float pz = (float(cz) + 0.5f) * c;
			// count -> colour. 1..8 blue-ish, 8..32 green, 32+ red
			float f = fminf(float(n) / 48.0f, 1.0f);
			uint32_t r = uint32_t(255.0f * fmaxf(0.0f, 2.0f * f - 1.0f));
			uint32_t gg = uint32_t(255.0f * (1.0f - fabsf(2.0f * f - 1.0f)));
			uint32_t b = uint32_t(255.0f * fmaxf(0.0f, 1.0f - 2.0f * f));
			uint32_t color = (b << 16) | (gg << 8) | r;
			drawInsect(px, py, pz, color, 0.0f, framebuffer);
		});
		grid.sync();
	}

	// ---- event markers (apply_event_markers.ps1) ---------------------
	// A small bright cluster at every active event position, in the event's
	// own colour, drawn with the depth test and without emission. So "where is
	// the source" is always answerable by looking, whatever the radius.
	if(uniforms.showEventMarkers && uniforms.numEvents > 0){
		processRange(0, uniforms.numEvents * 27, [&](int k){
			int e  = k / 27;
			int m  = k % 27;
			int ox = (m % 3) - 1;
			int oy = ((m / 3) % 3) - 1;
			int oz = (m / 9) - 1;
			float s = 0.25f;
			float px = uniforms.eventX[e] + ox * s;
			float py = uniforms.eventY[e] + oy * s;
			float pz = uniforms.eventZ[e] + oz * s;
			uint32_t r = uint32_t(255.0f * fminf(uniforms.eventCR[e] * 1.4f, 1.0f));
			uint32_t g = uint32_t(255.0f * fminf(uniforms.eventCG[e] * 1.4f, 1.0f));
			uint32_t b = uint32_t(255.0f * fminf(uniforms.eventCB[e] * 1.4f, 1.0f));
			uint32_t color = (b << 16) | (g << 8) | r;
			drawInsect(px, py, pz, color, 0.0f, framebuffer);
		});
		grid.sync();
	}

	// ---- placed markers (apply_navigate.ps1) ----------------------------
	// The checkerboard centres the user has picked. Drawn as a white cluster
	// with a small vertical spike so they read at any distance.
	// A solid cube of points, no spike. The spike was 18 points climbing by
	// 0.15 m each - fine in principle, but drawn as a line it made any bad
	// pick look like a beam running off to the horizon. A cube cannot do that.
	if(uniforms.numMarkers > 0 && uniforms.showMarkers != 0){   // apply_markers2.ps1
		constexpr int SIDE = 2 * MARKER_MAX_R + 1;
		processRangeStrided(uniforms.numMarkers * SIDE * SIDE, [&](int k){
			int m = k / (SIDE * SIDE), r = k - m * (SIDE * SIDE);
			drawMarkerBallPixel(uniforms.markerX[m], uniforms.markerY[m], uniforms.markerZ[m],
								r % SIDE - MARKER_MAX_R, r / SIDE - MARKER_MAX_R, framebuffer);
		});
		grid.sync();
	}

	// ---- insects (apply_insects.ps1) -------------------------------------
	// Sparse fliers in the empty volume between the trees. Position is a pure
	// function of index and time: a hash gives each one a home, a few sinusoids
	// give it a wander. Nothing is stored, initialised or updated, so changing
	// the count costs nothing but the drawing. Drawn after the forest, so they
	// depth-test against it correctly.
	if(uniforms.numInsects > 0 && !showUnedited()){
		processRange(0, uniforms.numInsects, [&](int idx){
			uint32_t h0 = insectHash(uint32_t(idx) * 2654435761u + 17u);
			uint32_t h1 = insectHash(h0 ^ 0x9e3779b9u);
			uint32_t h2 = insectHash(h1 ^ 0x85ebca6bu);
			uint32_t h3 = insectHash(h2 ^ 0xc2b2ae35u);

			float u0 = insectUnit(h0), u1 = insectUnit(h1);
			float u2 = insectUnit(h2), u3 = insectUnit(h3);

			float3 bx = uniforms.boxMax;
			// u2 squared biases them low: most insects fly in the understorey
			// and lower canopy rather than evenly through the whole column
			float hx = u0 * bx.x;
			float hy = u1 * bx.y;
			float hz = 0.8f + u2 * u2 * (bx.z * 0.85f);

			// With the grid available, reject homes that sit inside foliage or
			// out in open sky, retrying with fresh hashes. Deterministic per
			// index, so the same insects are always kept and always dropped.
			if(uniforms.useVoxelGrid && uniforms.insectPlacement == 1 && uniforms.voxelGridPtr != 0){
				bool ok = false;
				uint32_t hh = h3;
				for(int attempt = 0; attempt < 6 && !ok; attempt++){
					float c  = uniforms.cellSize;
					int cx = int(floorf(hx / c)), cy = int(floorf(hy / c)), cz = int(floorf(hz / c));
					ok = gridIsAirNearFoliage(cx, cy, cz, 3);
					if(!ok){
						hh = insectHash(hh ^ 0x27d4eb2fu); hx = insectUnit(hh) * bx.x;
						hh = insectHash(hh ^ 0x165667b1u); hy = insectUnit(hh) * bx.y;
						hh = insectHash(hh ^ 0xd3a2646cu); float u = insectUnit(hh);
						hz = 0.8f + u * u * (bx.z * 0.85f);
					}
				}
				if(!ok) return;
			}

			// insectPhase[k]: seconds x speed x rate k, reduced to [0, 2 pi)
			// on the host, for the rates 0.83, 2.7, 0.71, 3.1, 1.13 and 3.7
			float a = u3 * 6.2831853f;

			float px = hx + 2.2f * sin(uniforms.insectPhase[0] + a)        + 0.8f * sin(uniforms.insectPhase[1] + a * 3.1f);
			float py = hy + 2.2f * cos(uniforms.insectPhase[2] + a * 1.7f) + 0.8f * sin(uniforms.insectPhase[3] + a * 2.3f);
			float pz = hz + 1.1f * sin(uniforms.insectPhase[4] + a * 2.9f) + 0.4f * cos(uniforms.insectPhase[5] + a);

			drawInsect(px, py, pz, 0x00171c22u, uniforms.insectGlow, framebuffer);
		});
		grid.sync();
	}

	if(grid.thread_rank() == 0) stats->stageNs[2] = nanotime() - tStart;   // apply_stage_timers.ps1
	// Skip the whole pass when EDL is off or its strength is zero. It reads
	// four neighbouring pixels per pixel and was running regardless.
	if(uniforms.enableEDL && uniforms.edlStrength > 0.0f && !showUnedited())
	{ // tile-based EDL

		struct Pixel{
			uint32_t color;
			float depth;
		};

		Pixel* framebuffer_p = (Pixel*)framebuffer;

		int width = uniforms.width;
		int height = uniforms.height;

		// Every tile, the partial ones at the right and top edges included,
		// each block stepping through them. Only whole tiles used to be
		// counted, and only numTiles / numBlocks of them per block, so the
		// last few tiles and an edge strip up to 15 px wide went unshaded.
		// The pass reads depths and writes colours, so no barrier is needed
		// between tiles.
		uint32_t tileSize = 16;
		uint32_t numTiles_x = (width  + tileSize - 1) / tileSize;
		uint32_t numTiles_y = (height + tileSize - 1) / tileSize;
		uint32_t numTiles = numTiles_x * numTiles_y;

		for(uint32_t tileID = block.group_index().x; tileID < numTiles; tileID += grid.num_blocks()){

			int tileX = tileID % numTiles_x;
			int tileY = tileID / numTiles_x;
			int px = tileX * tileSize + block.thread_rank() % tileSize;
			int py = tileY * tileSize + block.thread_rank() / tileSize;
			if(px >= width || py >= height) continue;
			int pixelID = px + py * width;

			Pixel& pixel = framebuffer_p[pixelID];

			float PI = 3.1415;
			float numSamples = 50;
			float r = 1.5;
			// The EDL Strength slider and the "enable Eye Dome Lighting" checkbox
			// have never done anything: this local shadowed uniforms.edlStrength,
			// and uniforms.enableEDL was not read anywhere in render.cu at all.
			// Both are honoured now. Setting strength to 0, or unticking the
			// box, removes the dark ring around every splat.
			float edlStrength = uniforms.enableEDL ? uniforms.edlStrength : 0.0f;

			float sum = 0.0;
			// for(float u = 0.0; u < 2.0 * PI; u += stepsize)
			for(float u : {0.0f, PI / 2.0f, PI, 3.0f * PI / 2.0f})
			{
				// r = u;
				int dx = r * sin(u);
				int dy = r * cos(u);

				int index = pixelID + dx + width * dy;
				index = max(index, 0);
				index = min(index, width * height - 1);   // was one past the end

				Pixel neighbour = framebuffer_p[index];
				
				// single precision: the double literals made this, and the
				// shade below, double-precision arithmetic at every pixel
				sum = sum + fmaxf(log2f(pixel.depth) - log2f(neighbour.depth), 0.0f);
			}
			
			float response = sum / numSamples;
			float shade = __expf(-response * 300.0f * edlStrength);
			// shade = 1.0;

			uint32_t R = shade * ((pixel.color >>  0) & 0xff);
			uint32_t G = shade * ((pixel.color >>  8) & 0xff);
			uint32_t B = shade * ((pixel.color >> 16) & 0xff);
			uint32_t color = R | (G << 8) | (B << 16) | (255u << 24);

			pixel.color = color;
		}
		grid.sync();
	}

	grid.sync();

	// auto tEnd = nanotime();
	// double millies = double(tEnd - tStart) / 1'000'000.0;
	// PRINT("duration: %.1f ms \n", millies);

	// (device-side printf removed by apply_voxel_grid.ps1: the host run log
	//  carries every field, including the emitting count via Stats)

	if(grid.thread_rank() == 0) stats->stageNs[3] = nanotime() - tStart;   // apply_stage_timers.ps1
	// ---- volumetric light -------------------------------------------------
	// The reason light only landed on leaves and trunks: those were the only
	// things in the scene. Real forest light scatters off the AIR - that is
	// why a lamp in mist has a body rather than only illuminating what it
	// touches. This marches a ray per quarter-res pixel and accumulates light
	// scattered toward the camera from two kinds of source.
	//
	// Calls light the volumetric medium (Calls). The voxel grid does two
	// jobs there. Occupied cells add extinction, so trunks and canopy cast
	// shafts rather than glowing through. And the medium is suppressed inside
	// them, since solid wood is not air.
	//
	// The key light lights the haze (Depth haze), shaded by the canopy through
	// the light grid: the sunbeams. The haze's ambient share and its extinction
	// of whatever lies behind it are applied in closed form in the composite;
	// only the part that changes along the ray with the canopy's shade is
	// marched here.
	bool volEvents = !showUnedited() && uniforms.enableVolumetric && uniforms.numEvents > 0;
	bool volSun    = sunbeamsOn();
	if(volEvents || volSun){

		// constant over the frame; the phase function is constant along each ray
		float3 sunCol = hazeSunColour();

		processRangeStrided(volW * volH, [&](int i){
			int vx = i % volW;
			int vy = i / volW;

			// The composite reads every entry while the march runs, so clear
			// it before any early return (fisheye corners beyond the lens, or
			// nothing to march through).
			volBuf[4*i + 0] = 0.0f;
			volBuf[4*i + 1] = 0.0f;
			volBuf[4*i + 2] = 0.0f;
			volBuf[4*i + 3] = SKY_DISTANCE;

			// Ray through the centre of this quarter-res pixel.
			float3 dir;
			if(!pixelRayDir((float(vx) + 0.5f) / float(volW) * uniforms.width,
							(float(vy) + 0.5f) / float(volH) * uniforms.height, dir)) return;

			// Stop the march at the geometry already in the framebuffer, so the
			// medium sits in front of the forest and not through it. Stored
			// depth is view depth, so convert along the ray.
			int fx = min(4 * vx + 1, int(uniforms.width)  - 1);
			int fy = min(4 * vy + 1, int(uniforms.height) - 1);
			uint64_t fbv = framebuffer[fx + int(uniforms.width) * fy];
			uint32_t dbits = uint32_t(fbv >> 32);
			float viewDepth = *((float*)&dbits);

			// distance to what this ray sees: a surface, or the sky
			bool  sky   = !(viewDepth > 0.0f && viewDepth < 1.0e30f);
			float tSurf = sky ? SKY_DISTANCE : depthToDistance(viewDepth, dir);
			float tMax  = fminf(tSurf, uniforms.volMaxDist);
			volBuf[4*i + 3] = tSurf;
			if(tMax <= 0.0f) return;

			int   steps = uniforms.volSteps;
			float dt    = tMax / float(steps);

			// Dither the start point per pixel to trade banding for fine noise,
			// which the bilinear upsample then smooths away. Per pixel only: with
			// the frame number in the hash the noise changed every frame, and the
			// lit haze flickered in 4 x 4 blocks wherever rays run far through
			// the canopy, toward the horizon most of all.
			uint32_t h = insectHash(uint32_t(i) * 2654435761u);
			float jitter = insectUnit(h);

			float accR = 0.0f, accG = 0.0f, accB = 0.0f;
			float trans    = 1.0f;   // calls: transmittance through the medium so far
			float sunAcc   = 0.0f;   // key light: lit optical depth, before colour and phase
			float sunTrans = 1.0f;   // key light: transmittance through the haze so far
			bool  eventsLive = volEvents;
			// The sunbeams light the haze, and with Haze over sky off the haze
			// lies in front of points only, as in the composite: a ray that
			// hits none (sky, or the void below the horizon) gets no lit haze.
			// It was marched anyway, and below the ground, where the haze is
			// at full density and nothing shades it, the void filled with light.
			bool  sunLive    = volSun && (!sky || uniforms.fogSky != 0);

			for(int s = 0; s < steps && (eventsLive || sunLive); s++){
				float t = (float(s) + jitter) * dt;
				float px = uniforms.camPos.x + dir.x * t;
				float py = uniforms.camPos.y + dir.y * t;
				float pz = uniforms.camPos.z + dir.z * t;

				bool occupied = uniforms.useVoxelGrid && gridCountAtPos(px, py, pz) > 0;

				if(eventsLive){
					// medium density: uniform, optionally pooled near the ground
					float dens = uniforms.volDensity;
					if(uniforms.volHeightFalloff > 0.0f){
						dens *= __expf(-fmaxf(pz, 0.0f) * uniforms.volHeightFalloff);   // below z=0 this grew without bound
					}

					// occupied cells are solid: no medium, and they block
					float extinction = dens;
					if(occupied){
						dens = 0.0f;
						extinction += uniforms.volOcclusion;
					}

					if(dens > 0.0f){
						// Per event, with a SHADOW RAY. Without this the medium is
						// a featureless ball of glow: extinction along the view ray
						// alone never asks whether the path from the light to this
						// sample is blocked, so a point behind a trunk is lit as
						// brightly as one in the open. Marching a short ray back
						// towards the source is what produces shafts and dappling.
						//
						// Naturally sparse: samples outside every event radius skip
						// the whole thing, and most of the volume is outside.
						float sR = 0.0f, sG = 0.0f, sB = 0.0f;

						for(int e = 0; e < uniforms.numEvents; e++){
							float dx = uniforms.eventX[e] - px;
							float dy = uniforms.eventY[e] - py;
							float dz = uniforms.eventZ[e] - pz;
							float rr = uniforms.eventR[e];
							float d2 = dx*dx + dy*dy + dz*dz;
							if(d2 > rr * rr) continue;

							float d = sqrt(d2);
							if(d < 1e-4f) d = 1e-4f;
							float v = uniforms.eventI[e] * uniforms.emissionGain
									* eventFalloff(d, rr);

							int ss = uniforms.volShadowSteps;
							if(ss > 0 && uniforms.useVoxelGrid && uniforms.voxelGridPtr != 0){
								float sdt = d / float(ss + 1);
								float ux = dx / d, uy = dy / d, uz = dz / d;
								float shadow = 1.0f;
								for(int k = 1; k <= ss; k++){
									float q = float(k) * sdt;
									if(gridCountAtPos(px + ux*q, py + uy*q, pz + uz*q) > 0){
										shadow *= __expf(-uniforms.volOcclusion * sdt);
										if(shadow < 0.02f) break;
									}
								}
								v *= shadow;
							}

							sR += v * uniforms.eventCR[e];
							sG += v * uniforms.eventCG[e];
							sB += v * uniforms.eventCB[e];
						}

						if(sR + sG + sB > 0.0f){
							float w = trans * dens * uniforms.volScatter * dt;
							accR += sR * w;
							accG += sG * w;
							accB += sB * w;
						}
					}

					trans *= __expf(-extinction * dt);
					if(trans < 0.003f) eventsLive = false;   // fully occluded, nothing more to add
				}

				if(sunLive){
					// Haze at this height, lit by whatever the canopy lets through
					// to this sample. The step is integrated exactly for constant
					// density, (1 - exp(-ext dt)) / ext rather than dt, so dense haze
					// over a long step is not over-counted. Foliage along the view
					// ray that this pixel does not show dims what lies behind it,
					// at the same per-metre canopy density the light grid uses.
					float sigma = hazeDensityAt(pz);
					float ext   = sigma + (occupied ? uniforms.lightAbsorb : 0.0f);
					float att   = __expf(-ext * dt);
					float w     = ext > 1e-6f ? (1.0f - att) / ext : dt;
					float lit   = sunTrans * sigma * w;
					if(lit > 1e-6f) sunAcc += lit * lightAtSmooth(px, py, pz);   // skip 8 reads high above thinning haze
					sunTrans *= att;
					if(sunTrans < 0.003f) sunLive = false;
				}
			}

			if(volSun){
				// Past Medium reach there is no canopy left to shade the haze, so
				// the rest of the ray, out to the surface or, with Haze over sky,
				// to the sky, is lit throughout. Its share has a closed form:
				// what got through to the reach, times the part of that the haze
				// beyond scatters.
				float tEnd = sky ? (uniforms.fogSky != 0 ? SKY_DISTANCE : 0.0f) : tSurf;
				if(tEnd > tMax && sunLive){
					float tau = hazeOpticalDepth(uniforms.camPos.z + dir.z * tMax, dir.z, tEnd - tMax);
					sunAcc += sunTrans * (1.0f - __expf(-tau));
				}

				float cosT = dir.x * uniforms.lightDirX + dir.y * uniforms.lightDirY + dir.z * uniforms.lightDirZ;
				float s = sunAcc * phaseHG(cosT, uniforms.sunAnisotropy);
				accR += sunCol.x * s;
				accG += sunCol.y * s;
				accB += sunCol.z * s;
			}

			volBuf[4*i + 0] = accR;
			volBuf[4*i + 1] = accG;
			volBuf[4*i + 2] = accB;
		});
		grid.sync();

	}

	// ---- tier 1 shading composite ------------------------------------------
	{
		struct Pixel { uint32_t color; float depth; };
		Pixel* fb = (Pixel*)framebuffer;

		int W = int(uniforms.width);
		int H = int(uniforms.height);

		// ---- bloom: threshold to half resolution ----------------------
		// When an emission buffer exists, replace the luminance test with a
		// read from it. Everything downstream stays the same.
		if(grid.thread_rank() == 0) stats->stageNs[4] = nanotime() - tStart;   // apply_stage_timers.ps1
		// ---- bloom: progressive mip chain (apply_bloom_mip.ps1) --------
		// The previous version blurred once at half resolution with a gaussian
		// truncated at min(2.5*sigma, 24). Past sigma ~10 that clamp cuts the
		// kernel off at well under two sigma, which turns it into a box filter,
		// and a separable box filter is a SQUARE. It was also the reason wide
		// bloom cost so much: 49 taps per pixel per pass, twice.
		//
		// This instead downsamples to a chain of ever smaller levels, then
		// upsamples back up adding each level into the one below with a tent
		// filter. Every level costs a quarter of the one above, so the whole
		// chain is about 1.33x the half-res pixel count rather than 98x. The
		// result is round, smooth at any width, and roughly thirty times cheaper.
		// Bloom radius now chooses how many levels to use rather than a sigma.
		int mipW[8], mipH[8], mipOff[8];
		// spread 0 keeps only the half-resolution level: a tight halo
		int numMips = 1 + int(uniforms.bloomRadius);
		if(numMips > 7) numMips = 7;
		if(numMips < 1) numMips = 1;
		{
			int off = 0;
			for(int L = 0; L < numMips; L++){
				mipW[L]   = max(W >> (L + 1), 1);
				mipH[L]   = max(H >> (L + 1), 1);
				mipOff[L] = off;
				off += mipW[L] * mipH[L] * 3;
			}
		}

		if(uniforms.enableBloom && emissive){   // emissive also excludes the unedited view

			// level 0: 2x2 box from the emission buffer. Reads emission, not a
			// luminance threshold, so only actual sources bloom.
			processRangeStrided(mipW[0] * mipH[0], [&](int i){
				int x = i % mipW[0];
				int y = i / mipW[0];
				float sr = 0.0f, sg = 0.0f, sb = 0.0f;
				for(int oy = 0; oy < 2; oy++)
				for(int ox = 0; ox < 2; ox++){
					int sx = min(2*x + ox, W - 1);
					int sy = min(2*y + oy, H - 1);
					int s  = sx + W * sy;
					sr += _emission[3 * s + 0];
					sg += _emission[3 * s + 1];
					sb += _emission[3 * s + 2];
				}
				int d = mipOff[0] + 3 * i;
				bloomChain[d + 0] = sr * 0.25f;
				bloomChain[d + 1] = sg * 0.25f;
				bloomChain[d + 2] = sb * 0.25f;
			});
			grid.sync();

			// downsample: each level a 2x2 box of the one above
			for(int L = 1; L < numMips; L++){
				processRangeStrided(mipW[L] * mipH[L], [&](int i){
					int x = i % mipW[L];
					int y = i / mipW[L];
					float sr = 0.0f, sg = 0.0f, sb = 0.0f;
					for(int oy = 0; oy < 2; oy++)
					for(int ox = 0; ox < 2; ox++){
						int sx = min(2*x + ox, mipW[L-1] - 1);
						int sy = min(2*y + oy, mipH[L-1] - 1);
						int s  = mipOff[L-1] + 3 * (sx + mipW[L-1] * sy);
						sr += bloomChain[s + 0];
						sg += bloomChain[s + 1];
						sb += bloomChain[s + 2];
					}
					int d = mipOff[L] + 3 * i;
					bloomChain[d + 0] = sr * 0.25f;
					bloomChain[d + 1] = sg * 0.25f;
					bloomChain[d + 2] = sb * 0.25f;
				});
				grid.sync();
			}

			// upsample: bilinear from L+1, added into L. Accumulating on the way
			// down is what makes the falloff smooth rather than banded.
			for(int L = numMips - 2; L >= 0; L--){
				processRangeStrided(mipW[L] * mipH[L], [&](int i){
					int x = i % mipW[L];
					int y = i / mipW[L];

					float fx = (float(x) - 0.5f) * 0.5f;
					float fy = (float(y) - 0.5f) * 0.5f;
					int x0 = int(fx); x0 = x0 < 0 ? 0 : (x0 >= mipW[L+1] ? mipW[L+1]-1 : x0);
					int y0 = int(fy); y0 = y0 < 0 ? 0 : (y0 >= mipH[L+1] ? mipH[L+1]-1 : y0);
					int x1 = min(x0 + 1, mipW[L+1] - 1);
					int y1 = min(y0 + 1, mipH[L+1] - 1);
					// clamped: at the left/top edge fx is -0.25, which made one
					// weight negative and drew a dark fringe
					float tx = fminf(fmaxf(fx - float(x0), 0.0f), 1.0f);
					float ty = fminf(fmaxf(fy - float(y0), 0.0f), 1.0f);

					int i00 = mipOff[L+1] + 3 * (x0 + mipW[L+1]*y0);
					int i10 = mipOff[L+1] + 3 * (x1 + mipW[L+1]*y0);
					int i01 = mipOff[L+1] + 3 * (x0 + mipW[L+1]*y1);
					int i11 = mipOff[L+1] + 3 * (x1 + mipW[L+1]*y1);
					float w00 = (1.0f-tx)*(1.0f-ty), w10 = tx*(1.0f-ty);
					float w01 = (1.0f-tx)*ty,        w11 = tx*ty;

					int d = mipOff[L] + 3 * i;
					for(int c = 0; c < 3; c++){
						float up = bloomChain[i00+c]*w00 + bloomChain[i10+c]*w10
								 + bloomChain[i01+c]*w01 + bloomChain[i11+c]*w11;
						bloomChain[d + c] = bloomChain[d + c] * 0.6f + up * 0.6f;
					}
				});
				grid.sync();
			}
		}

		if(grid.thread_rank() == 0) stats->stageNs[5] = nanotime() - tStart;   // apply_stage_timers.ps1
		// ---- composite: haze, light, grade, exposure, tone map, encode ------
		// Each pixel has two parts. The scene is light from geometry and from
		// the air in front of it (haze, emission, volumetric light, bloom), and
		// it is graded. The backdrop is the sky, or the void below the horizon,
		// seen through any haze in front of it, and it keeps the exact colour
		// it was given. Every step of the grade is linear in RGB, so this is
		// the whole pixel graded with the backdrop left out: haze and beams in
		// front of the sky are graded the same as in front of the trunk beside
		// it, and nothing changes colour at the silhouette.
		bool volActive = volEvents || volSun;   // exactly when the march above filled volBuf
		// grid-stride: coalesced, and with the tone curve's amount the
		// contiguous form made this loop reload two spilled registers a pixel
		processRangeStrided(W * H, [&](int pixelIndex){

			int x = pixelIndex % W;
			int y = pixelIndex / W;

			Pixel p = fb[pixelIndex];

			float cr = srgbToLinear(float((p.color >>  0) & 0xff));
			float cg = srgbToLinear(float((p.color >>  8) & 0xff));
			float cb = srgbToLinear(float((p.color >> 16) & 0xff));

			// Background was cleared with depth = +infinity (0x7f800000).
			bool isGeometry = (p.depth > 0.0f) && (p.depth < 1.0e30f);

			// This pixel's view ray and, for geometry, the distance along it.
			// Perspective stores view-plane depth, which is shorter than the
			// path through the haze everywhere off the image centre.
			float3 rd;
			bool  haveRay = pixelRayDir(float(x) + 0.5f, float(y) + 0.5f, rd);
			float dist    = (isGeometry && haveRay) ? depthToDistance(p.depth, rd) : 0.0f;

			float r  = 0.0f, g  = 0.0f, b  = 0.0f;   // scene, graded
			float kr = 0.0f, kg = 0.0f, kb = 0.0f;   // backdrop, not graded
			if(isGeometry){
				r = cr; g = cg; b = cb;
			}else{
				kr = cr; kg = cg; kb = cb;

				// Below the horizon (apply_navigate.ps1): an empty pixel looking
				// downward is the void under the ground, not sky. Painted the
				// "below" colour, black by default, as a game engine would. A
				// narrow blend across the horizon so it does not cut.
				if(haveRay){
					// The blend used a fixed 0.5 - rd.z*25, which spans about
					// 1.2 degrees either side of level: at any field of view
					// that is a HARD EDGE, and it reads as a drawn line across
					// the frame rather than a horizon. Worse, the partial band
					// mixes sky and void and comes out lighter than either.
					//
					// Now the span is in DEGREES and smoothstepped, so the
					// gradient is monotonic - no band brighter than the sky it
					// came from - and wide enough to read as atmosphere.
					float softRad = uniforms.horizonSoftness * 0.017453292f;
					float f;
					if(softRad < 1e-4f){
						f = rd.z < 0.0f ? 1.0f : 0.0f;      // deliberate hard edge
					}else{
						float t = 0.5f - (asinf(fminf(fmaxf(rd.z, -1.0f), 1.0f)) / (2.0f * softRad));
						t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
						f = t * t * (3.0f - 2.0f * t);      // smoothstep
					}
					kr = kr * (1.0f - f) + srgbToLinear(uniforms.belowR * 255.0f) * f;
					kg = kg * (1.0f - f) + srgbToLinear(uniforms.belowG * 255.0f) * f;
					kb = kb * (1.0f - f) + srgbToLinear(uniforms.belowB * 255.0f) * f;
				}
			}

			// ---- the unedited view ------------------------------------
			// A point's pixel keeps the colour stored in the file, byte for
			// byte, and the backdrop the colour it was given: none of the haze,
			// light, bloom, grade, exposure or tone curve below.
			if(showUnedited()){
				uint32_t out;
				if(isGeometry){
					out = (p.color & 0x00ffffffu) | (255u << 24);
				}else{
					uint32_t R = uint32_t(linearToSrgb(kr) * 255.0f + 0.5f);
					uint32_t G = uint32_t(linearToSrgb(kg) * 255.0f + 0.5f);
					uint32_t B = uint32_t(linearToSrgb(kb) * 255.0f + 0.5f);
					out = R | (G << 8) | (B << 16) | (255u << 24);
				}
				surf2Dwrite(out, gl_colorbuffer, x * 4, y);
				return;
			}

			// ---- haze -------------------------------------------------
			// Exact for the height profile in hazeOpticalDepth, with the sky at
			// SKY_DISTANCE: a ray level with or below the camera never leaves
			// the haze and sees only haze, while a ray climbing out of haze
			// that thins with height sees the sky through what lies below it.
			// With Haze over sky off, the sky is left as it was, since even
			// haze would cover it completely. The key light's share is added
			// with the volumetric light below.
			if(uniforms.enableFog && haveRay && (isGeometry || uniforms.fogSky != 0)){
				float T = __expf(-hazeOpticalDepth(uniforms.camPos.z, rd.z, isGeometry ? dist : SKY_DISTANCE));
				float3 amb = hazeAmbientColour();
				r  = r  * T + amb.x * (1.0f - T);
				g  = g  * T + amb.y * (1.0f - T);
				b  = b  * T + amb.z * (1.0f - T);
				kr *= T; kg *= T; kb *= T;
			}

			// unblurred emission core, before the blurred halo
			if(emissive){
				r += _emission[3 * pixelIndex + 0];
				g += _emission[3 * pixelIndex + 1];
				b += _emission[3 * pixelIndex + 2];
			}

			// Volumetric light from quarter resolution. The march stopped at the
			// geometry under one pixel of each 4x4 block, so beside a silhouette
			// a neighbouring sample may have marched far past the trunk this
			// pixel shows, or stopped at a trunk this pixel sees past. Each of
			// the four bilinear neighbours is weighted by how closely the
			// distance it saw matches this pixel's own (joint bilateral
			// upsampling, Kopf et al. 2007, with distance as the guide), so lit
			// haze behind a trunk does not spread onto its edge. If none match,
			// the closest is used alone.
			if(volActive){
				float fvx = (float(x) + 0.5f) * 0.25f - 0.5f;
				float fvy = (float(y) + 0.5f) * 0.25f - 0.5f;
				int vx0 = int(fvx); vx0 = vx0 < 0 ? 0 : (vx0 >= volW ? volW-1 : vx0);
				int vy0 = int(fvy); vy0 = vy0 < 0 ? 0 : (vy0 >= volH ? volH-1 : vy0);
				int vx1 = min(vx0 + 1, volW - 1);
				int vy1 = min(vy0 + 1, volH - 1);
				float ux = fvx - float(vx0); ux = ux < 0.0f ? 0.0f : (ux > 1.0f ? 1.0f : ux);
				float uy = fvy - float(vy0); uy = uy < 0.0f ? 0.0f : (uy > 1.0f ? 1.0f : uy);

				// the distance to what this pixel sees, as the march records it
				float tPix = isGeometry ? dist : SKY_DISTANCE;
				float inv  = 1.0f / fmaxf(tPix, 0.01f);

				float vr = 0.0f, vg = 0.0f, vb = 0.0f, wSum = 0.0f;
				int   best = 0;
				float bestDiff = 1.0e30f;
				for(int k = 0; k < 4; k++){
					int   j    = 4 * (((k & 1) ? vx1 : vx0) + volW * ((k & 2) ? vy1 : vy0));
					float q    = ((k & 1) ? ux : 1.0f - ux) * ((k & 2) ? uy : 1.0f - uy);
					float diff = fabsf(volBuf[j + 3] - tPix) * inv;   // relative
					float w    = q * __expf(-50.0f * diff * diff);    // 10% apart: 0.6, 30%: 0.01
					vr   += w * volBuf[j + 0];
					vg   += w * volBuf[j + 1];
					vb   += w * volBuf[j + 2];
					wSum += w;
					if(diff < bestDiff){ bestDiff = diff; best = j; }
				}
				if(wSum > 1e-4f){
					float n = 1.0f / wSum;
					r += vr * n;
					g += vg * n;
					b += vb * n;
				}else if(isGeometry || uniforms.fogSky != 0 || volEvents){
					// No sample is near this pixel's distance: take the closest.
					// A sky pixel takes none while only the sunbeams light the
					// haze and Haze over sky is off, as a sky ray gets none
					// then; in a canopy gap narrower than the 4 px sample
					// spacing it used to take a leaf's beam light.
					r += volBuf[best + 0];
					g += volBuf[best + 1];
					b += volBuf[best + 2];
				}
			}

			if(uniforms.enableBloom && emissive){
				// bilinear upsample from half resolution
				float fx = (float(x) - 0.5f) * 0.5f;
				float fy = (float(y) - 0.5f) * 0.5f;
				int x0 = int(fx); x0 = x0 < 0 ? 0 : (x0 >= bloomW ? bloomW - 1 : x0);
				int y0 = int(fy); y0 = y0 < 0 ? 0 : (y0 >= bloomH ? bloomH - 1 : y0);
				int x1 = (x0 + 1 >= bloomW) ? bloomW - 1 : x0 + 1;
				int y1 = (y0 + 1 >= bloomH) ? bloomH - 1 : y0 + 1;
				float tx = fminf(fmaxf(fx - float(x0), 0.0f), 1.0f);   // see the mip upsample
				float ty = fminf(fmaxf(fy - float(y0), 0.0f), 1.0f);

				int i00 = mipOff[0] + 3*(x0 + bloomW*y0), i10 = mipOff[0] + 3*(x1 + bloomW*y0);
				int i01 = mipOff[0] + 3*(x0 + bloomW*y1), i11 = mipOff[0] + 3*(x1 + bloomW*y1);
				float w00 = (1.0f-tx)*(1.0f-ty), w10 = tx*(1.0f-ty);
				float w01 = (1.0f-tx)*ty,        w11 = tx*ty;

				float s = uniforms.bloomStrength;
				r += s * (bloomChain[i00+0]*w00 + bloomChain[i10+0]*w10 + bloomChain[i01+0]*w01 + bloomChain[i11+0]*w11);
				g += s * (bloomChain[i00+1]*w00 + bloomChain[i10+1]*w10 + bloomChain[i01+1]*w01 + bloomChain[i11+1]*w11);
				b += s * (bloomChain[i00+2]*w00 + bloomChain[i10+2]*w10 + bloomChain[i01+2]*w01 + bloomChain[i11+2]*w11);
			}

			// ---- colour grading -------------------------------------------
			// The scene only: the backdrop keeps the exact colour it was given,
			// and is added after. Emission has already been added, so light
			// sources are graded with the surfaces they sit on rather than
			// separately. On an empty pixel with nothing in front of the sky the
			// scene is zero and every step below leaves it zero.
			{
				// tint mix: pull the colour towards a chosen tint
				// channel mixer (apply_regroup.ps1): plain per-channel gain in
				// linear light, before everything else in the grade
				r *= uniforms.mixR;
				g *= uniforms.mixG;
				b *= uniforms.mixB;
				if(uniforms.tintMix > 0.0f){
					float m = uniforms.tintMix;
					float l = luminance(r, g, b);
					r = r * (1.0f - m) + uniforms.tintR * l * m;
					g = g * (1.0f - m) + uniforms.tintG * l * m;
					b = b * (1.0f - m) + uniforms.tintB * l * m;
				}
				// temperature: warm raises red and lowers blue, cool the reverse
				if(uniforms.temperature != 0.0f){
					float tt = uniforms.temperature * 0.35f;
					r *= (1.0f + tt);
					b *= (1.0f - tt);
					if(r < 0.0f) r = 0.0f;
					if(b < 0.0f) b = 0.0f;
				}
				// saturation: scale the distance from grey
				if(uniforms.saturation != 1.0f){
					float l = luminance(r, g, b);
					r = l + (r - l) * uniforms.saturation;
					g = l + (g - l) * uniforms.saturation;
					b = l + (b - l) * uniforms.saturation;
					if(r < 0.0f) r = 0.0f;
					if(g < 0.0f) g = 0.0f;
					if(b < 0.0f) b = 0.0f;
				}
			}

			// the backdrop, ungraded, behind everything graded above
			r += kr;
			g += kg;
			b += kb;

			// vignette (apply_look.ps1): quadratic falloff from the centre, in
			// linear light so it darkens rather than greys
			if(uniforms.vignette > 0.0f){
				float vx = (float(x) / float(W)) * 2.0f - 1.0f;
				float vy = (float(y) / float(H)) * 2.0f - 1.0f;
				float vr = (vx*vx + vy*vy) * 0.5f;          // 0 centre, 1 corner
				float vig = 1.0f - uniforms.vignette * vr * vr;
				r *= vig; g *= vig; b *= vig;
			}

			r *= uniforms.exposure;
			g *= uniforms.exposure;
			b *= uniforms.exposure;

			{
				// The curve acts on the brightest channel, the other two scaled
				// by the same factor, so a colour keeps its hue as it brightens
				// (tone mapping one brightness value and keeping the colour
				// ratios, as Reinhard et al. 2002 do with luminance). Applied
				// to each channel on its own, the curve compressed the largest
				// channels most: at exposure 4 sunlit foliage moved 6 degrees
				// of hue toward yellow, and bright glows went white.
				// Tone curve 0 leaves brightness as it is, the unedited
				// baseline; 1 is the ACES fit. Either way, a colour brighter
				// than white is brought down to it whole rather than clipped
				// channel by channel. A switch between two curves made this
				// per-pixel loop spill registers; one curve with an amount
				// does not.
				float m = fmaxf(r, fmaxf(g, b));
				if(m > 0.0f){
					float s = 1.0f + uniforms.toneCurve * (acesToneMap(m) / m - 1.0f);
					if(m * s > 1.0f) s = 1.0f / m;
					r *= s; g *= s; b *= s;
				}
			}

			// grain (apply_look.ps1): per pixel, per frame, applied after the
			// tone curve so it reads as texture on the image, not as noise in
			// the light. The reference has it across the whole frame.
			if(uniforms.grain > 0.0f){
				uint32_t gh = insectHash(uint32_t(pixelIndex) * 2654435761u
							+ uint32_t(uniforms.frameCounter) * 40503u);
				float gn = (insectUnit(gh) - 0.5f) * 2.0f * uniforms.grain;
				r = fminf(fmaxf(r + gn, 0.0f), 1.0f);
				g = fminf(fmaxf(g + gn, 0.0f), 1.0f);
				b = fminf(fmaxf(b + gn, 0.0f), 1.0f);
			}

			uint32_t R = uint32_t(linearToSrgb(r) * 255.0f + 0.5f);
			uint32_t G = uint32_t(linearToSrgb(g) * 255.0f + 0.5f);
			uint32_t B = uint32_t(linearToSrgb(b) * 255.0f + 0.5f);

			uint32_t out = R | (G << 8) | (B << 16) | (255u << 24);
			surf2Dwrite(out, gl_colorbuffer, x * 4, y);
		});
	}
	grid.sync();
	if(grid.thread_rank() == 0) stats->stageNs[6] = nanotime() - tStart;   // apply_stage_timers.ps1
	// ---- end tier 1 composite ----------------------------------------------

	// if(grid.thread_rank() == 0){
	// 	printf("allocator.offset: %llu \n", allocator->offset / 1'000'000);
	// }


	// if(grid.thread_rank() == 0){
	// 	auto tEnd = nanotime();
	// 	float ms = double(tEnd - tStart) / 1'000'000.0;
	// 	printf("%f ms\n", ms);
	// }
}

