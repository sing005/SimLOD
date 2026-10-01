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
Uniforms uniforms;
Lines* _lines = nullptr;
float* _emission = nullptr;   // RGB per pixel, added by apply_emission.ps1
uint32_t _numEmitting = 0;    // debug counter, apply_debug_log.ps1
uint32_t _numDrawn    = 0;    // points that survived the density gate, apply_sweep_fixes.ps1
uint32_t _numTargets  = 0;    // fix_find_targets.ps1

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

	float feather = __powf(1.0f - x, uniforms.falloffPower);
	float core    = uniforms.coreBoost / (1.0f + 80.0f * x * x);

	return feather * (1.0f + core);
}

inline void evalEmission(float px, float py, float pz,
						 float& emR, float& emG, float& emB)
{
	emR = 0.0f; emG = 0.0f; emB = 0.0f;
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

// ---- survey target test (fix_find_targets.ps1) --------------------------
// Brightness alone was a poor test: a sunlit leaf and a gap of sky both pass
// it, and a checkerboard in shade fails. What actually separates a survey
// target from a forest is that it is ACHROMATIC - black, grey and white, so
// its three channels are nearly equal - while foliage is strongly green and
// bark strongly red-brown. Testing the SPREAD between channels finds a target
// in shadow and rejects a bright leaf.
//
// Both halves of a checkerboard qualify: the white squares and the black ones
// are equally grey. Only the brightness floor decides which side is caught,
// so a low floor finds the whole board.
inline bool isSurveyTarget(uint32_t c){
	if(uniforms.findTargets == 0) return false;
	float r = float((c      ) & 0xff);
	float g = float((c >>  8) & 0xff);
	float b = float((c >> 16) & 0xff);
	float mx = fmaxf(r, fmaxf(g, b));
	float mn = fminf(r, fminf(g, b));
	if(mx < uniforms.findThreshold) return false;   // too dark to judge
	return (mx - mn) <= uniforms.findGrey;          // achromatic
}

// ---- pixel to world ray (apply_navigate.ps1) -----------------------------
// The inverse of projectPoint: which direction does this pixel look along.
// Used by the volumetric medium, by the sky/below-horizon split, and by the
// marker picker. Was inline in the volumetric code; now shared.
inline bool pixelRayDir(float px, float py, float3& dir){
	float ndcx = (px / uniforms.width)  * 2.0f - 1.0f;
	float ndcy = (py / uniforms.height) * 2.0f - 1.0f;
	if(uniforms.lensMode == 0){
		float4 farP = uniforms.transformInv * float4{ndcx, ndcy, 1.0f, 1.0f};
		if(farP.w == 0.0f) return false;
		float3 world = float3{farP.x / farP.w, farP.y / farP.w, farP.z / farP.w};
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
	return gridCountAt(int(x / c), int(y / c), int(z / c));
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

// ---- stray-point culling (apply_stray_cull.ps1) ------------------------
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
inline bool isStrayPoint(float px, float py, float pz){
	if(uniforms.minNeighbours <= 0) return false;
	if(!uniforms.useVoxelGrid || uniforms.voxelGridPtr == 0) return false;

	float c = uniforms.cellSize;
	int cx = int(px / c), cy = int(py / c), cz = int(pz / c);

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
// and accumulate extinction. 5.3 M cells at 32 steps is a one-off pass; at draw
// time it is a single lookup.
inline float lightAt(float px, float py, float pz){
	if(uniforms.lightGridPtr == 0) return 1.0f;
	float c = uniforms.cellSize;
	int cx = int(px / c), cy = int(py / c), cz = int(pz / c);
	if(cx < 0 || cy < 0 || cz < 0) return 1.0f;
	if(cx >= uniforms.gridX || cy >= uniforms.gridY || cz >= uniforms.gridZ) return 1.0f;
	const float* g = (const float*)uniforms.lightGridPtr;
	return g[cx + uniforms.gridX * (cy + uniforms.gridY * cz)];
}

// Combine the key light with an ambient floor. Backlit foliage goes toward the
// light colour; shadowed points keep their own colour at ambient strength.
inline uint32_t applyKeyLight(uint32_t c, float trans){
	if(uniforms.lightOn == 0 || uniforms.lightIntensity <= 0.0f) return c;
	// ambient floors the shadow; intensity scales the lit side. Capped so the
	// brightest fully-lit point cannot exceed 2x its own colour, which is
	// plenty and stops strength 4 turning foliage into flat yellow.
	float k = uniforms.ambient + (1.0f - uniforms.ambient) * trans * uniforms.lightIntensity;
	if(k > 2.0f) k = 2.0f;
	float r = float((c      ) & 0xff) * k * uniforms.lightR;
	float g = float((c >>  8) & 0xff) * k * uniforms.lightG;
	float b = float((c >> 16) & 0xff) * k * uniforms.lightB;
	uint32_t ri = uint32_t(fminf(r, 255.0f));
	uint32_t gi = uint32_t(fminf(g, 255.0f));
	uint32_t bi = uint32_t(fminf(b, 255.0f));
	return (c & 0xff000000u) | (bi << 16) | (gi << 8) | ri;
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
inline void drawMarkerBall(float px, float py, float pz, uint64_t* framebuffer){
	int x, y; float depth;
	if(!projectPoint(px, py, pz, x, y, depth)) return;
	if(depth < uniforms.nearClip) return;
	float pxPerMetre;
	if(uniforms.lensMode == 0){
		pxPerMetre = uniforms.focalPx / fmaxf(depth, 0.05f);
	}else{
		float halfFov = uniforms.fisheyeFov * 0.5f * 0.017453292f;
		pxPerMetre = ((0.5f * uniforms.height) / halfFov) / fmaxf(depth, 0.05f);
	}
	int R = int(0.5f * uniforms.markerScale * pxPerMetre + 0.5f);
	if(R < 3)  R = 3;
	if(R > 64) R = 64;
	float dz = depth * 0.999f;
	uint64_t udz = *((uint32_t*)&dz);
	uint64_t encoded = (udz << 32) | 0x00ff00ffull;   // aabbggrr: magenta
	for(int oy = -R; oy <= R; oy++){
		int qy = y + oy;
		if(qy < 0 || qy >= int(uniforms.height)) continue;
		for(int ox = -R; ox <= R; ox++){
			if(ox*ox + oy*oy > R*R) continue;
			int qx = x + ox;
			if(qx < 0 || qx >= int(uniforms.width)) continue;
			uint32_t pixelID = qx + int(uniforms.width) * qy;
			if(encoded < framebuffer[pixelID]) atomicMin(&framebuffer[pixelID], encoded);
		}
	}
}

// ---- world-space point size and sphere shading (apply_look.ps1) ---------
// The reference render draws points as spheres of a fixed WORLD size: near
// ones large, distant ones shrinking until they merge into texture. A fixed
// pixel size flattens the forest into uniform noise. This projects a diameter
// in metres to pixels using the view depth, capped so a point at the lens
// cannot cost thousands of atomics.
inline int splatRadiusPx(float depth){
	if(uniforms.pointWorldSize <= 0.0f) return uniforms.pointSize;
	float px = uniforms.pointWorldSize * uniforms.focalPx / fmaxf(depth, 0.05f);
	int R = int(px + 0.5f);
	if(R < 1) R = 1;
	if(R > uniforms.maxSplatPx) R = uniforms.maxSplatPx;
	return R;
}

// Shade a splat pixel as a point on a sphere lit from the upper left front.
// (ox,oy) is the pixel's offset within the R x R splat. Returns a multiplier;
// 1.0 everywhere when shading is off or the splat is too small to show it.
inline float splatShade(int ox, int oy, int R, float half){
	if(uniforms.pointShading <= 0.0f || R < 3) return 1.0f;
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

inline uint32_t shadeColour(uint32_t c, float s){
	if(s >= 0.999f) return c;
	uint32_t r = uint32_t(fminf(float((c      ) & 0xff) * s, 255.0f));
	uint32_t g = uint32_t(fminf(float((c >>  8) & 0xff) * s, 255.0f));
	uint32_t b = uint32_t(fminf(float((c >> 16) & 0xff) * s, 255.0f));
	return (c & 0xff000000u) | (b << 16) | (g << 8) | r;
}

void drawPoint(Point point, Node* node, uint64_t* framebuffer){
	// Density is applied as a loop stride in drawNode now, so dropped points are
	// never read. The per-point hash that used to sit here ran for every one of
	// 90.7 M samples and rejected them AFTER the load, which is why turning
	// density down barely changed the frame time.

	// stray-point culling (apply_stray_cull.ps1). Unlike density this cannot be
	// strided away - whether a point is a stray depends on where it is, so it
	// has to be read first. One grid lookup, and nearby points share a cell, so
	// it caches well.
	// "Show culled only" with the threshold at 0 used to blank the screen:
	// nothing was stray, so "only the strays" was nothing. Now the checkbox
	// does nothing until there is a threshold for it to show.
	bool stray = isStrayPoint(point.x, point.y, point.z);
	if(uniforms.showCulledOnly && uniforms.minNeighbours > 0){ if(!stray) return; }
	else if(stray) return;
	countUp(&_numDrawn);         // counted AFTER the density gate, so the Stats
								 // line shows exactly what density is doing

	int x, y; float depth;
	if(!projectPoint(point.x, point.y, point.z, x, y, depth)) return;   // apply_fisheye.ps1
	if(depth < uniforms.nearClip) return;

	if(x > 1 && x < uniforms.width  - 2.0)
	if(y > 1 && y < uniforms.height - 2.0)
	{

		bool isTarget = false;   // fix_find_targets.ps1: needed later for the size boost,
		                         // so it must live outside the colour if/else below
		uint32_t color = point.color;
		if(uniforms.colorByNode){
			color = (node->getID() % 127) * 123456789ull;
		}else if(uniforms.colorByLOD){
			color = getLodColor(node->level);
		}else{
			// ground band (apply_regroup.ps1): everything within groundHeight of
			// the lowest point takes the ground colour. z is in the translated
			// frame, so the lowest point is at 0.
			// (ground band removed by apply_navigate.ps1: points keep their scanned colour)
			// Find the checkerboards (fix_walk_targets.ps1): a survey target is
			// white, and nothing else in a forest is. Paint anything whose
			// three channels all clear the threshold in magenta, before the
			// light touches it, so it stays visible in shadow.
			isTarget = isSurveyTarget(color);
			if(isTarget) countUp(&_numTargets);
			if(uniforms.findIsolate != 0 && uniforms.findTargets != 0 && !isTarget) return;
			if(uniforms.lightOn != 0) color = applyKeyLight(color, lightAt(point.x, point.y, point.z));   // gated: no grid read while off
			if(isTarget) color = 0x00ff00ffu;   // magenta, full brightness
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

		// Centred, optionally circular splat. The original loop offset only
		// in +x/+y, so the splat sat below-right of the point. Clamp to
		// width-1/height-1 also closes a latent one-past-the-end write.
		int   R    = splatRadiusPx(depth);   // world-sized (apply_look.ps1)
		if(isTarget) R += uniforms.findBoost;   // fix_find_targets.ps1: a 15 cm board is sub-pixel from 30 m
		float half = float(R - 1) * 0.5f;
		float rad2 = (float(R) * 0.5f) * (float(R) * 0.5f);

		for(int ox = 0; ox < R; ox++)
		for(int oy = 0; oy < R; oy++)
		{
			if(uniforms.roundSplats && R > 2){
				float ddx = float(ox) - half;
				float ddy = float(oy) - half;
				if(ddx * ddx + ddy * ddy > rad2) continue;
			}

			uint32_t px = clamp(int(x + ox - half), 0, int(uniforms.width)  - 1);
			uint32_t py = clamp(int(y + oy - half), 0, int(uniforms.height) - 1);

			uint32_t pixelID = px + int(uniforms.width) * py;

			// Additive, and deliberately outside the depth test: overlapping
			// excited points sum, which atomicMin cannot do. A point behind a
			// trunk still glows through it - move this inside the depth test
			// below to occlude instead.
			if(emitting && _emission != nullptr){
				atomicAdd(&_emission[3 * pixelID + 0], emR);
				atomicAdd(&_emission[3 * pixelID + 1], emG);
				atomicAdd(&_emission[3 * pixelID + 2], emB);
			}
			uint64_t udepth = *((uint32_t*)&depth);
			uint32_t shaded = shadeColour(color, splatShade(ox, oy, R, half));   // apply_look.ps1
			uint64_t encoded = (udepth << 32) | shaded;

			if(encoded < framebuffer[pixelID]){
				atomicMin(&framebuffer[pixelID], encoded);
			}
		}

	}
}

void drawNode(Chunk* chunk, Node* node, uint32_t numElements, uint64_t* framebuffer){
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
	int step = 1;
	if(uniforms.pointDensity < 0.999f){
		float d = uniforms.pointDensity < 0.0005f ? 0.0005f : uniforms.pointDensity;
		step = int(1.0f / d + 0.5f);
		if(step < 1) step = 1;
	}

	for(
		int pointIndex = block.thread_rank() * step; 
		pointIndex < numElements; 
		pointIndex += block.num_threads() * step
	){
		int targetChunkIndex = pointIndex / POINTS_PER_CHUNK;

		// while, not if: with a stride the target can jump several chunks at
		// once, and the original single step would silently fall behind and
		// read from the wrong chunk.
		while(chunkIndex < targetChunkIndex && chunk->next != nullptr){
			chunk = chunk->next;
			chunkIndex++;
		}

		int pointInChunkIndex = pointIndex % POINTS_PER_CHUNK;
		Point point = chunk->points[pointInChunkIndex];

		drawPoint(point, node, framebuffer);

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
		Point point = chunk->points[pointInChunkIndex];

		int cx = int(point.x / c);
		int cy = int(point.y / c);
		int cz = int(point.z / c);
		if(cx >= 0 && cx < uniforms.gridX && cy >= 0 && cy < uniforms.gridY && cz >= 0 && cz < uniforms.gridZ){
			atomicAdd(&g[cx + uniforms.gridX * (cy + uniforms.gridY * cz)], 1u);
		}
	}
}

void refillVoxelGrid(Node* nodes){
	if(uniforms.gridRefill == 0) return;
	if(uniforms.voxelGridPtr == 0) return;

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	// zero first: a refill must not add to whatever was already there
	processRange(0, uniforms.gridX * uniforms.gridY * uniforms.gridZ, [&](int i){
		((uint32_t*)uniforms.voxelGridPtr)[i] = 0;
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
		if(sh_n >= uniforms.numNodesTotal) break;

		Node* node = &nodes[sh_n];
		// points only. Voxels are coarse summaries of the same surfaces, and
		// counting them as well would inflate the density of everything that
		// happens to be summarised, which is exactly what the cull compares.
		countNodeIntoGrid(node->points, node->numPoints);
		block.sync();
	}
	grid.sync();
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

		drawNode(node->points, node, node->numPoints, framebuffer);
		drawNode(node->voxelChunks, node, node->numVoxels, framebuffer);

		block.sync();
	}

	grid.sync();
}

void drawNodesHQS(Node* nodes, uint32_t numNodes, uint64_t* framebuffer){

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

	int numPixels = int(uniforms.width) * int(uniforms.height);
	uint32_t* fb_depth = allocator->alloc<uint32_t*>(1 * numPixels * sizeof(uint32_t));
	uint32_t* fb_color = allocator->alloc<uint32_t*>(4 * numPixels * sizeof(uint32_t));

	processRange(numPixels, [&](int pixelID){
		fb_depth[pixelID] = 0x7f800000;
	});
	processRange(numPixels, [&](int pixelID){
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

			for(int pointIndex = block.thread_rank(); pointIndex < node->numPoints; pointIndex += block.num_threads()){

				int targetChunkIndex = pointIndex / POINTS_PER_CHUNK;

				if(chunkIndex < targetChunkIndex){
					chunk = chunk->next;
					chunkIndex++;
				}

				int pointInChunkIndex = pointIndex % POINTS_PER_CHUNK;
				Point point = chunk->points[pointInChunkIndex];

				// Point density (apply_density_hqs.ps1). This was missing: density
				// lived only in drawPoint, which the HQS path never calls, so with
				// High-Quality-Shading ticked the slider did nothing whatsoever.
				// Same hash and the same place in all four loops, so the depth pass
				// and the colour pass always drop exactly the same points.
				if(uniforms.pointDensity < 0.999f){
					uint32_t dhx = *((uint32_t*)&point.x);
					uint32_t dhy = *((uint32_t*)&point.y);
					uint32_t dhz = *((uint32_t*)&point.z);
					uint32_t dh  = insectHash(dhx ^ (dhy * 0x9e3779b9u) ^ (dhz * 0x85ebca6bu));
					if(insectUnit(dh) > uniforms.pointDensity) continue;   // NOT return: plain for loop, not a lambda
				}

				// stray-point culling (apply_stray_cull.ps1). Same test in all four HQS
				// loops so the depth pass and the colour pass agree about which points
				// exist. Anchored on the two lines above, which are identical whether
				// apply_sweep_fixes.ps1 inserted this block or repaired an earlier one.
				{
					bool hqStray = isStrayPoint(point.x, point.y, point.z);
					if(uniforms.showCulledOnly && uniforms.minNeighbours > 0){ if(!hqStray) continue; }
					else if(hqStray) continue;
				}

				int x, y; float depth;
				if(!projectPoint(point.x, point.y, point.z, x, y, depth)) continue;   // apply_fisheye.ps1, HQS
				if(depth < uniforms.nearClip) continue;

				if(x > 1 && x < uniforms.width  - 2.0)
				if(y > 1 && y < uniforms.height - 2.0)
				if(depth > 0.0f)
				{
					// HQS depth pass, brought into line with drawPoint by
					// apply_test_source.ps1: centred, optionally round, and
					// clamped to width-1/height-1. Must match the colour pass
					// below exactly or the depth test and the colour disagree.
					int   R    = splatRadiusPx(depth);   // world-sized, HQS (apply_look.ps1)
					float half = float(R - 1) * 0.5f;
					float rad2 = (float(R) * 0.5f) * (float(R) * 0.5f);
					for(int ox = 0; ox < R; ox++)
					for(int oy = 0; oy < R; oy++)
					{
						if(uniforms.roundSplats && R > 2){
							float ddx = float(ox) - half;
							float ddy = float(oy) - half;
							if(ddx * ddx + ddy * ddy > rad2) continue;
						}
						uint32_t px = clamp(int(x + ox - half), 0, int(uniforms.width)  - 1);
						uint32_t py = clamp(int(y + oy - half), 0, int(uniforms.height) - 1);

						uint32_t pixelID = px + int(uniforms.width) * py;
						uint32_t udepth = *((uint32_t*)&depth);

						if(udepth < fb_depth[pixelID]){
							atomicMin(&fb_depth[pixelID], udepth);
						}
					}
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

		if(node->numVoxels > 0)
		{// draw voxels
			Chunk* chunk = node->voxelChunks;
			int chunkIndex = 0;

			for(int pointIndex = block.thread_rank(); pointIndex < node->numVoxels; pointIndex += block.num_threads()){

				int targetChunkIndex = pointIndex / POINTS_PER_CHUNK;

				if(chunkIndex < targetChunkIndex){
					chunk = chunk->next;
					chunkIndex++;
				}

				int pointInChunkIndex = pointIndex % POINTS_PER_CHUNK;
				Point point = chunk->points[pointInChunkIndex];

				// Point density (apply_density_hqs.ps1). This was missing: density
				// lived only in drawPoint, which the HQS path never calls, so with
				// High-Quality-Shading ticked the slider did nothing whatsoever.
				// Same hash and the same place in all four loops, so the depth pass
				// and the colour pass always drop exactly the same points.
				if(uniforms.pointDensity < 0.999f){
					uint32_t dhx = *((uint32_t*)&point.x);
					uint32_t dhy = *((uint32_t*)&point.y);
					uint32_t dhz = *((uint32_t*)&point.z);
					uint32_t dh  = insectHash(dhx ^ (dhy * 0x9e3779b9u) ^ (dhz * 0x85ebca6bu));
					if(insectUnit(dh) > uniforms.pointDensity) continue;   // NOT return: plain for loop, not a lambda
				}

				// stray-point culling (apply_stray_cull.ps1). Same test in all four HQS
				// loops so the depth pass and the colour pass agree about which points
				// exist. Anchored on the two lines above, which are identical whether
				// apply_sweep_fixes.ps1 inserted this block or repaired an earlier one.
				{
					bool hqStray = isStrayPoint(point.x, point.y, point.z);
					if(uniforms.showCulledOnly && uniforms.minNeighbours > 0){ if(!hqStray) continue; }
					else if(hqStray) continue;
				}

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

				int x, y; float depth;
				if(!projectPoint(point.x, point.y, point.z, x, y, depth)) continue;   // apply_fisheye.ps1, HQS
				if(depth < uniforms.nearClip) continue;

				if(x > 1 && x < uniforms.width  - 2.0)
				if(y > 1 && y < uniforms.height - 2.0)
				if(depth > 0.0f)
				{

					// HQS depth pass, brought into line with drawPoint by
					// apply_test_source.ps1: centred, optionally round, and
					// clamped to width-1/height-1. Must match the colour pass
					// below exactly or the depth test and the colour disagree.
					int   R    = splatRadiusPx(depth);   // world-sized, HQS (apply_look.ps1)
					float half = float(R - 1) * 0.5f;
					float rad2 = (float(R) * 0.5f) * (float(R) * 0.5f);
					for(int ox = 0; ox < R; ox++)
					for(int oy = 0; oy < R; oy++)
					{
						if(uniforms.roundSplats && R > 2){
							float ddx = float(ox) - half;
							float ddy = float(oy) - half;
							if(ddx * ddx + ddy * ddy > rad2) continue;
						}
						uint32_t px = clamp(int(x + ox - half), 0, int(uniforms.width)  - 1);
						uint32_t py = clamp(int(y + oy - half), 0, int(uniforms.height) - 1);

						uint32_t pixelID = px + int(uniforms.width) * py;
						uint32_t udepth = *((uint32_t*)&depth);

						if(udepth < fb_depth[pixelID]){
							atomicMin(&fb_depth[pixelID], udepth);
						}
					}

				}
			}
		}
	}

	grid.sync();

	if(grid.thread_rank() == 0){
		numNodesDrawn = 0;
	}

	grid.sync();

	auto inDebugZone = [&](int x, int y){
		if(x > uniforms.width / 2)  return true;
		else                        return false;
	};

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

			for(int pointIndex = block.thread_rank(); pointIndex < node->numPoints; pointIndex += block.num_threads()){

				int targetChunkIndex = pointIndex / POINTS_PER_CHUNK;

				if(chunkIndex < targetChunkIndex){
					chunk = chunk->next;
					chunkIndex++;
				}

				// if(chunkIndex > (int(10 * uniforms.time) % 50)) break;

				int pointInChunkIndex = pointIndex % POINTS_PER_CHUNK;
				Point point = chunk->points[pointInChunkIndex];

				// Point density (apply_density_hqs.ps1). This was missing: density
				// lived only in drawPoint, which the HQS path never calls, so with
				// High-Quality-Shading ticked the slider did nothing whatsoever.
				// Same hash and the same place in all four loops, so the depth pass
				// and the colour pass always drop exactly the same points.
				if(uniforms.pointDensity < 0.999f){
					uint32_t dhx = *((uint32_t*)&point.x);
					uint32_t dhy = *((uint32_t*)&point.y);
					uint32_t dhz = *((uint32_t*)&point.z);
					uint32_t dh  = insectHash(dhx ^ (dhy * 0x9e3779b9u) ^ (dhz * 0x85ebca6bu));
					if(insectUnit(dh) > uniforms.pointDensity) continue;   // NOT return: plain for loop, not a lambda
				}

				// stray-point culling (apply_stray_cull.ps1). Same test in all four HQS
				// loops so the depth pass and the colour pass agree about which points
				// exist. Anchored on the two lines above, which are identical whether
				// apply_sweep_fixes.ps1 inserted this block or repaired an earlier one.
				{
					bool hqStray = isStrayPoint(point.x, point.y, point.z);
					if(uniforms.showCulledOnly && uniforms.minNeighbours > 0){ if(!hqStray) continue; }
					else if(hqStray) continue;
				}

				int x, y; float depth;
				if(!projectPoint(point.x, point.y, point.z, x, y, depth)) continue;   // apply_fisheye.ps1, HQS
				if(depth < uniforms.nearClip) continue;

				if(x > 1 && x < uniforms.width  - 2.0)
				if(y > 1 && y < uniforms.height - 2.0)
				if(depth > 0.0f)
				{
					uint32_t color = point.color;
					if(uniforms.colorByNode){
						color = (node->getID() % 127) * 123456789;
					}else if(uniforms.colorByLOD){
						color = getLodColor(node->level);
					}
					// if(RIGHTSIDE_NODECOLORS && x > uniforms.width / 2){
					// 	color = (node->getID() % 127) * 123456789;
					// }

					if(RIGHTSIDE_NODECOLORS && inDebugZone(x, y)){
						color = (node->getID() % 127) * 123456789;
					}

					// color = chunkIndex * 1234567;
					
					// (ground band removed by apply_navigate.ps1)
					bool isTargetH = isSurveyTarget(color);
					if(isTargetH) countUp(&_numTargets);
					if(uniforms.findIsolate != 0 && uniforms.findTargets != 0 && !isTargetH) continue;
					if(uniforms.lightOn != 0) color = applyKeyLight(color, lightAt(point.x, point.y, point.z));   // gated, HQS
					if(isTargetH) color = 0x00ff00ffu;
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

					int   R    = splatRadiusPx(depth);   // world-sized, HQS (apply_look.ps1)
					float half = float(R - 1) * 0.5f;
					float rad2 = (float(R) * 0.5f) * (float(R) * 0.5f);
					for(int ox = 0; ox < R; ox++)
					for(int oy = 0; oy < R; oy++)
					{
						if(uniforms.roundSplats && R > 2){
							float ddx = float(ox) - half;
							float ddy = float(oy) - half;
							if(ddx * ddx + ddy * ddy > rad2) continue;
						}
						uint32_t px = clamp(int(x + ox - half), 0, int(uniforms.width)  - 1);
						uint32_t py = clamp(int(y + oy - half), 0, int(uniforms.height) - 1);

						uint32_t pixelID = px + int(uniforms.width) * py;

						if(emitting && _emission != nullptr){
							atomicAdd(&_emission[3 * pixelID + 0], emR);
							atomicAdd(&_emission[3 * pixelID + 1], emG);
							atomicAdd(&_emission[3 * pixelID + 2], emB);
						}

						float fbDepth = *((float*)&fb_depth[pixelID]);

						if(depth < fbDepth * 1.01f)
						{
							float sh = splatShade(ox, oy, R, half);   // apply_look.ps1
							atomicAdd(&fb_color[4 * pixelID + 0], uint32_t(float(rgba[0]) * sh));
							atomicAdd(&fb_color[4 * pixelID + 1], uint32_t(float(rgba[1]) * sh));
							atomicAdd(&fb_color[4 * pixelID + 2], uint32_t(float(rgba[2]) * sh));
							atomicAdd(&fb_color[4 * pixelID + 3], 1);
						}
					}
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

		if(node->numVoxels > 0)
		{// draw voxels
			Chunk* chunk = node->voxelChunks;
			int chunkIndex = 0;

			for(int pointIndex = block.thread_rank(); pointIndex < node->numVoxels; pointIndex += block.num_threads()){

				int targetChunkIndex = pointIndex / POINTS_PER_CHUNK;

				if(chunkIndex < targetChunkIndex){
					chunk = chunk->next;
					chunkIndex++;
				}

				// if(chunkIndex > (int(10 * uniforms.time) % 30)) break;

				int pointInChunkIndex = pointIndex % POINTS_PER_CHUNK;
				Point point = chunk->points[pointInChunkIndex];

				// Point density (apply_density_hqs.ps1). This was missing: density
				// lived only in drawPoint, which the HQS path never calls, so with
				// High-Quality-Shading ticked the slider did nothing whatsoever.
				// Same hash and the same place in all four loops, so the depth pass
				// and the colour pass always drop exactly the same points.
				if(uniforms.pointDensity < 0.999f){
					uint32_t dhx = *((uint32_t*)&point.x);
					uint32_t dhy = *((uint32_t*)&point.y);
					uint32_t dhz = *((uint32_t*)&point.z);
					uint32_t dh  = insectHash(dhx ^ (dhy * 0x9e3779b9u) ^ (dhz * 0x85ebca6bu));
					if(insectUnit(dh) > uniforms.pointDensity) continue;   // NOT return: plain for loop, not a lambda
				}

				// stray-point culling (apply_stray_cull.ps1). Same test in all four HQS
				// loops so the depth pass and the colour pass agree about which points
				// exist. Anchored on the two lines above, which are identical whether
				// apply_sweep_fixes.ps1 inserted this block or repaired an earlier one.
				{
					bool hqStray = isStrayPoint(point.x, point.y, point.z);
					if(uniforms.showCulledOnly && uniforms.minNeighbours > 0){ if(!hqStray) continue; }
					else if(hqStray) continue;
				}

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

				int x, y; float depth;
				if(!projectPoint(point.x, point.y, point.z, x, y, depth)) continue;   // apply_fisheye.ps1, HQS
				if(depth < uniforms.nearClip) continue;

				if(x > 1 && x < uniforms.width  - 2.0)
				if(y > 1 && y < uniforms.height - 2.0)
				if(depth > 0.0f)
				{

					uint32_t color = point.color;
					if(uniforms.colorByNode){
						color = (node->getID() % 127) * 123456789;
					}else if(uniforms.colorByLOD){
						color = getLodColor(node->level);
					}
					// if(RIGHTSIDE_NODECOLORS && x > uniforms.width / 2){
					// 	color = (node->getID() % 127) * 123456789;
					// }

					if(RIGHTSIDE_NODECOLORS && inDebugZone(x, y)){
						color = (node->getID() % 127) * 123456789;
					}

					// color = chunkIndex * 1234567;

					// (ground band removed by apply_navigate.ps1)
					bool isTargetH = isSurveyTarget(color);
					if(isTargetH) countUp(&_numTargets);
					if(uniforms.findIsolate != 0 && uniforms.findTargets != 0 && !isTargetH) continue;
					if(uniforms.lightOn != 0) color = applyKeyLight(color, lightAt(point.x, point.y, point.z));   // gated, HQS
					if(isTargetH) color = 0x00ff00ffu;
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

					int   R    = splatRadiusPx(depth);   // world-sized, HQS (apply_look.ps1)
					float half = float(R - 1) * 0.5f;
					float rad2 = (float(R) * 0.5f) * (float(R) * 0.5f);
					for(int ox = 0; ox < R; ox++)
					for(int oy = 0; oy < R; oy++)
					{
						if(uniforms.roundSplats && R > 2){
							float ddx = float(ox) - half;
							float ddy = float(oy) - half;
							if(ddx * ddx + ddy * ddy > rad2) continue;
						}
						uint32_t px = clamp(int(x + ox - half), 0, int(uniforms.width)  - 1);
						uint32_t py = clamp(int(y + oy - half), 0, int(uniforms.height) - 1);

						uint32_t pixelID = px + int(uniforms.width) * py;

						if(emitting && _emission != nullptr){
							atomicAdd(&_emission[3 * pixelID + 0], emR);
							atomicAdd(&_emission[3 * pixelID + 1], emG);
							atomicAdd(&_emission[3 * pixelID + 2], emB);
						}

						float fbDepth = *((float*)&fb_depth[pixelID]);

						if(depth < fbDepth * 1.01f)
						{
							float sh = splatShade(ox, oy, R, half);   // apply_look.ps1
							atomicAdd(&fb_color[4 * pixelID + 0], uint32_t(float(rgba[0]) * sh));
							atomicAdd(&fb_color[4 * pixelID + 1], uint32_t(float(rgba[1]) * sh));
							atomicAdd(&fb_color[4 * pixelID + 2], uint32_t(float(rgba[2]) * sh));
							atomicAdd(&fb_color[4 * pixelID + 3], 1);
						}
					}

				}
			}
		}
	}

	grid.sync();

	// RESOLVE
	processRange(numPixels, [&](int pixelID){
		
		uint32_t R = fb_color[4 * pixelID + 0];
		uint32_t G = fb_color[4 * pixelID + 1];
		uint32_t B = fb_color[4 * pixelID + 2];
		uint32_t C = fb_color[4 * pixelID + 3];

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

		drawBoundingBox(lines, pos, {scale, scale, scale}, color);
		int steps = 1;
		for(int si = 0; si < steps; si++){
			float s = 1.0 + 0.02 * float(si) / 5.0f;
			drawBoundingBox(lines, pos, float3{s * scale, scale, scale}, color);
			drawBoundingBox(lines, pos, float3{scale, s * scale, scale}, color);
			drawBoundingBox(lines, pos, float3{scale, s * scale, s * scale}, color);
		}
	});
}

// Capacity of the visibleNodes buffer allocated in kernel_render. Fisheye
// disables frustum culling, so wide lenses inside a large cloud can exceed it.
// Writes past the cap are dropped and the count is clamped after visibility.
constexpr uint32_t MAX_VISIBLE_NODES = 100'000;

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

	auto makeVisible = [&](Node* node){
		uint32_t visibleNodeIndex = atomicAdd(&numVisibleNodes, 1);
		if(visibleNodeIndex >= MAX_VISIBLE_NODES) return;
		visibleNodes[visibleNodeIndex] = *node;

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
		node->isLarge = dx > 2.0 * uniforms.minNodeSize || dy > 2.0 * uniforms.minNodeSize;

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

				makeVisible(child);
				// printf("test \n");
			}
		}else if(node->isLarge && node->isLeafFn() && node->visible){
			makeVisible(node);
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

extern "C" __global__
void kernel_render(
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

	uniforms = _uniforms;
	Allocator _allocator(buffer, 0);
	allocator = &_allocator;

	Node* visibleNodes                 = allocator->alloc<Node*>(100'000 * sizeof(Node));
	uint32_t& numVisibleNodes          = *allocator->alloc<uint32_t*>(4);
	uint32_t& numVisiblePoints         = *allocator->alloc<uint32_t*>(4);
	uint32_t& numVisibleVoxels         = *allocator->alloc<uint32_t*>(4);
	uint32_t& numVisibleInner          = *allocator->alloc<uint32_t*>(4);
	uint32_t& numVisibleLeaves         = *allocator->alloc<uint32_t*>(4);
	uint32_t& dbg_numSmallNodes        = *allocator->alloc<uint32_t*>(4);
	uint32_t& dbg_numSmallNodePoints   = *allocator->alloc<uint32_t*>(4);

	Lines* lines = allocator->alloc<Lines*>(sizeof(Lines));
	lines->count = 0;
	lines->vertices = allocator->alloc<Point*>(1'000'000 * sizeof(Point));
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
	// low frequency, so a quarter-res march upsampled bilinearly is
	// indistinguishable from full res and sixteen times cheaper.
	int volW = int(uniforms.width)  / 4; if(volW < 1) volW = 1;
	int volH = int(uniforms.height) / 4; if(volH < 1) volH = 1;
	float* volBuf = allocator->alloc<float*>(volW * volH * 3 * sizeof(float));
	if(grid.thread_rank() == 0){ _numEmitting = 0; _numDrawn = 0; _numTargets = 0; }

	// ---- build the light transmittance grid (apply_keylight.ps1) ----------
	// Only when the host says it is stale: after a load, or when a light
	// control moves. Otherwise this is skipped entirely and the light costs
	// one texture-like lookup per point.
	if(uniforms.lightOn != 0 && uniforms.lightDirty != 0 && uniforms.lightGridPtr != 0 && uniforms.voxelGridPtr != 0){
		grid.sync();
		int nCells = uniforms.gridX * uniforms.gridY * uniforms.gridZ;
		processRange(0, nCells, [&](int i){
			int cx = i % uniforms.gridX;
			int cy = (i / uniforms.gridX) % uniforms.gridY;
			int cz = i / (uniforms.gridX * uniforms.gridY);

			float trans = 1.0f;
			for(int s = 1; s <= uniforms.lightSteps; s++){
				int sx = cx + int(uniforms.lightDirX * float(s) + 0.5f);
				int sy = cy + int(uniforms.lightDirY * float(s) + 0.5f);
				int sz = cz + int(uniforms.lightDirZ * float(s) + 0.5f);
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
	refillVoxelGrid(nodes);   // apply_grid_refill.ps1: only when the host says stale

	// clear framebuffer
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){
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
		_emission[3 * pixelIndex + 0] = 0.0f;
		_emission[3 * pixelIndex + 1] = 0.0f;
		_emission[3 * pixelIndex + 2] = 0.0f;
	});

	grid.sync();

	float3 boxSize = uniforms.boxMax - uniforms.boxMin;
	float cubeSize = max(max(boxSize.x, boxSize.y), boxSize.z);
	float3 cubeMin = uniforms.boxMin;
	float3 cubeMax = cubeMin + cubeSize;
	float3 cubePosition = uniforms.boxMin + cubeSize * 0.5f;

	grid.sync();
	
	// if(uniforms.doUpdateVisibility)
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
			drawNodesHQS(visibleNodes, numVisibleNodes, framebuffer);
		}else{
			drawNodes(visibleNodes, numVisibleNodes, framebuffer);
		}

		grid.sync();

		// draw point cloud cube
		// if(uniforms.showBoundingBox)
		// if(grid.thread_rank() == 0)
		// {
		// 	float3 size = {cubeSize, cubeSize, cubeSize};
		// 	drawBoundingBox(lines, cubePosition, size, 0x000000ff);
		// }

		grid.sync();

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
		stats->numTargets       = _numTargets;
	}

	// if(false)
	if(grid.thread_rank() == 0) stats->stageNs[1] = nanotime() - tStart;   // apply_stage_timers.ps1
	// ---- voxel grid debug view (apply_voxel_grid.ps1) ------------------
	// Draws the centre of every occupied cell, coloured by count: blue for
	// sparse, through green, to red for dense. A verification view, like the
	// axes - it should sit exactly on the foliage and nowhere else.
	if(uniforms.useVoxelGrid && uniforms.showVoxelGrid && uniforms.voxelGridPtr != 0){
		int nCells = uniforms.gridX * uniforms.gridY * uniforms.gridZ;
		processRange(0, nCells, [&](int i){
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

	// ---- pick (apply_navigate.ps1) ----------------------------------------
	// After everything solid is drawn: read the depth under the requested
	// pixel, turn it back into a world position, hand it to the host via
	// Stats. One thread, once per frame, only when asked.
	if(grid.thread_rank() == 0){
		stats->pickValid = 0;
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
				stats->pickValid = 1;
			}
		}
	}
	grid.sync();

	// ---- placed markers (apply_navigate.ps1) ----------------------------
	// The checkerboard centres the user has picked. Drawn as a white cluster
	// with a small vertical spike so they read at any distance.
	// A solid cube of points, no spike. The spike was 18 points climbing by
	// 0.15 m each - fine in principle, but drawn as a line it made any bad
	// pick look like a beam running off to the horizon. A cube cannot do that.
	if(uniforms.numMarkers > 0 && uniforms.showMarkers != 0){   // apply_markers2.ps1
		processRange(0, uniforms.numMarkers, [&](int m){
			drawMarkerBall(uniforms.markerX[m], uniforms.markerY[m], uniforms.markerZ[m], framebuffer);
		});
		grid.sync();
	}

	// ---- insects (apply_insects.ps1) -------------------------------------
	// Sparse fliers in the empty volume between the trees. Position is a pure
	// function of index and time: a hash gives each one a home, a few sinusoids
	// give it a wander. Nothing is stored, initialised or updated, so changing
	// the count costs nothing but the drawing. Drawn after the forest, so they
	// depth-test against it correctly.
	if(uniforms.numInsects > 0){
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
					int cx = int(hx / c), cy = int(hy / c), cz = int(hz / c);
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

			float t = uniforms.time * uniforms.insectSpeed;
			float a = u3 * 6.2831853f;

			float px = hx + 2.2f * sin(t * 0.83f + a)        + 0.8f * sin(t * 2.7f + a * 3.1f);
			float py = hy + 2.2f * cos(t * 0.71f + a * 1.7f) + 0.8f * sin(t * 3.1f + a * 2.3f);
			float pz = hz + 1.1f * sin(t * 1.13f + a * 2.9f) + 0.4f * cos(t * 3.7f + a);

			drawInsect(px, py, pz, 0x00171c22u, uniforms.insectGlow, framebuffer);
		});
		grid.sync();
	}

	if(grid.thread_rank() == 0) stats->stageNs[2] = nanotime() - tStart;   // apply_stage_timers.ps1
	// Skip the whole pass when EDL is off or its strength is zero. It reads
	// four neighbouring pixels per pixel and was running regardless.
	if(uniforms.enableEDL && uniforms.edlStrength > 0.0f)
	{ // tile-based EDL

		struct Pixel{
			uint32_t color;
			float depth;
		};

		float* framebuffer_f = (float*)framebuffer;
		uint32_t* framebuffer_u32 = (uint32_t*)framebuffer;
		Pixel* framebuffer_p = (Pixel*)framebuffer;

		int width = uniforms.width;
		int height = uniforms.height;

		uint32_t tileSize = 16;
		uint32_t numTiles_x = width / tileSize;
		uint32_t numTiles_y = height / tileSize;
		uint32_t numTiles = numTiles_x * numTiles_y;
		uint32_t tilesPerBlock = numTiles / grid.num_blocks();

		for(int i = 0; i < tilesPerBlock; i++){

			block.sync();

			int tileID = i * grid.num_blocks() + block.group_index().x;
			int tileX = tileID % numTiles_x;
			int tileY = tileID / numTiles_x;
			int tileStart = tileX * tileSize + tileY * width * tileSize;

			int pixelID = tileStart + block.thread_rank() % tileSize + (block.thread_rank() / tileSize) * width;

			Pixel& pixel = framebuffer_p[pixelID];

			float PI = 3.1415;
			float numSamples = 50;
			float stepsize = 2.0 * PI / numSamples;
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
				index = min(index, width * height);

				Pixel neighbour = framebuffer_p[index];
				
				sum = sum + max(log2(pixel.depth) - log2(neighbour.depth), 0.0);
			}
			
			float response = sum / numSamples;
			float shade = __expf(-response * 300.0 * edlStrength);
			// shade = 1.0;

			uint32_t R = shade * ((pixel.color >>  0) & 0xff);
			uint32_t G = shade * ((pixel.color >>  8) & 0xff);
			uint32_t B = shade * ((pixel.color >> 16) & 0xff);
			uint32_t color = R | (G << 8) | (B << 16) | (255u << 24);

			pixel.color = color;

			block.sync();
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
	// ---- volumetric medium (apply_volumetric.ps1) -----------------------
	// The reason light only landed on leaves and trunks: those were the only
	// things in the scene. Real forest light scatters off the AIR - that is
	// why a lamp in mist has a body rather than only illuminating what it
	// touches. This marches a ray per quarter-res pixel through a participating
	// medium, accumulating light from every active event at each step.
	//
	// The voxel grid does two jobs here. Occupied cells add extinction, so
	// trunks and canopy cast shafts rather than glowing through. And the medium
	// is suppressed inside them, since solid wood is not air.
	if(uniforms.enableVolumetric && uniforms.numEvents > 0){

		processRange(0, volW * volH, [&](int i){
			int vx = i % volW;
			int vy = i / volW;

			// volBuf is scratch memory that also serves other kernels, and the
			// composite reads every entry, so clear it before any early return
			// (fisheye corners beyond the lens, or nothing to march through).
			volBuf[3*i + 0] = 0.0f;
			volBuf[3*i + 1] = 0.0f;
			volBuf[3*i + 2] = 0.0f;

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

			float tMax = uniforms.volMaxDist;
			if(viewDepth > 0.0f && viewDepth < 1.0e30f){
				float d = depthToDistance(viewDepth, dir);
				if(d < tMax) tMax = d;
			}
			if(tMax <= 0.0f) return;

			int   steps = uniforms.volSteps;
			float dt    = tMax / float(steps);

			// Dither the start point per pixel to trade banding for fine noise,
			// which the bilinear upsample then smooths away.
			uint32_t h = insectHash(uint32_t(i) * 2654435761u
					   + uint32_t(uniforms.frameCounter) * 40503u);
			float jitter = insectUnit(h);

			float accR = 0.0f, accG = 0.0f, accB = 0.0f;
			float trans = 1.0f;

			for(int s = 0; s < steps; s++){
				float t = (float(s) + jitter) * dt;
				float px = uniforms.camPos.x + dir.x * t;
				float py = uniforms.camPos.y + dir.y * t;
				float pz = uniforms.camPos.z + dir.z * t;

				// medium density: uniform, optionally pooled near the ground
				float dens = uniforms.volDensity;
				if(uniforms.volHeightFalloff > 0.0f){
					dens *= __expf(-fmaxf(pz, 0.0f) * uniforms.volHeightFalloff);   // apply_markers2.ps1: below z=0 this grew without bound
				}

				// occupied cells are solid: no medium, and they block
				float extinction = dens;
				if(uniforms.useVoxelGrid && gridCountAtPos(px, py, pz) > 0){
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
				if(trans < 0.003f) break;   // fully occluded, nothing more to add
			}

			volBuf[3*i + 0] = accR;
			volBuf[3*i + 1] = accG;
			volBuf[3*i + 2] = accB;
		});
		grid.sync();

	}else{
		processRange(0, volW * volH, [&](int i){
			volBuf[3*i + 0] = 0.0f;
			volBuf[3*i + 1] = 0.0f;
			volBuf[3*i + 2] = 0.0f;
		});
		grid.sync();
	}

	// ---- tier 1 shading composite (inserted by apply_tier1_shading.ps1) ----
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
		int numMips = 2 + int(uniforms.bloomRadius);
		if(numMips > 7) numMips = 7;
		if(numMips < 2) numMips = 2;
		{
			int off = 0;
			for(int L = 0; L < numMips; L++){
				mipW[L]   = max(W >> (L + 1), 1);
				mipH[L]   = max(H >> (L + 1), 1);
				mipOff[L] = off;
				off += mipW[L] * mipH[L] * 3;
			}
		}

		if(uniforms.enableBloom){

			// level 0: 2x2 box from the emission buffer. Reads emission, not a
			// luminance threshold, so only actual sources bloom.
			processRange(0, mipW[0] * mipH[0], [&](int i){
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
				processRange(0, mipW[L] * mipH[L], [&](int i){
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
				processRange(0, mipW[L] * mipH[L], [&](int i){
					int x = i % mipW[L];
					int y = i / mipW[L];

					float fx = (float(x) - 0.5f) * 0.5f;
					float fy = (float(y) - 0.5f) * 0.5f;
					int x0 = int(fx); x0 = x0 < 0 ? 0 : (x0 >= mipW[L+1] ? mipW[L+1]-1 : x0);
					int y0 = int(fy); y0 = y0 < 0 ? 0 : (y0 >= mipH[L+1] ? mipH[L+1]-1 : y0);
					int x1 = min(x0 + 1, mipW[L+1] - 1);
					int y1 = min(y0 + 1, mipH[L+1] - 1);
					float tx = fx - float(x0);
					float ty = fy - float(y0);

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
		// ---- composite: fog, bloom, exposure, tone map, encode --------
		processRange(0, W * H, [&](int pixelIndex){

			int x = pixelIndex % W;
			int y = pixelIndex / W;

			Pixel p = fb[pixelIndex];

			float r = srgbToLinear(float((p.color >>  0) & 0xff));
			float g = srgbToLinear(float((p.color >>  8) & 0xff));
			float b = srgbToLinear(float((p.color >> 16) & 0xff));

			// Background was cleared with depth = +infinity (0x7f800000).
			// It must not be fogged, or the sky fogs into itself.
			bool isGeometry = (p.depth > 0.0f) && (p.depth < 1.0e30f);

			// Below the horizon (apply_navigate.ps1): an empty pixel looking
			// downward is the void under the ground, not sky. Painted the
			// "below" colour, black by default, as a game engine would. A
			// narrow blend across the horizon so it does not cut.
			if(!isGeometry){
				float3 rd;
				if(pixelRayDir(float(x) + 0.5f, float(y) + 0.5f, rd)){
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
					r = r * (1.0f - f) + srgbToLinear(uniforms.belowR * 255.0f) * f;
					g = g * (1.0f - f) + srgbToLinear(uniforms.belowG * 255.0f) * f;
					b = b * (1.0f - f) + srgbToLinear(uniforms.belowB * 255.0f) * f;
				}
			}

			if(uniforms.enableFog && isGeometry){
				float f = 1.0f - __expf(-p.depth * uniforms.fogDensity);
				f = f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);
				r = r * (1.0f - f) + uniforms.fogR * f;
				g = g * (1.0f - f) + uniforms.fogG * f;
				b = b * (1.0f - f) + uniforms.fogB * f;
			}

			// unblurred emission core, before the blurred halo
			r += _emission[3 * pixelIndex + 0];
			g += _emission[3 * pixelIndex + 1];
			b += _emission[3 * pixelIndex + 2];

			// volumetric medium, bilinear from quarter resolution
			if(uniforms.enableVolumetric){
				float fvx = (float(x) + 0.5f) * 0.25f - 0.5f;
				float fvy = (float(y) + 0.5f) * 0.25f - 0.5f;
				int vx0 = int(fvx); vx0 = vx0 < 0 ? 0 : (vx0 >= volW ? volW-1 : vx0);
				int vy0 = int(fvy); vy0 = vy0 < 0 ? 0 : (vy0 >= volH ? volH-1 : vy0);
				int vx1 = min(vx0 + 1, volW - 1);
				int vy1 = min(vy0 + 1, volH - 1);
				float ux = fvx - float(vx0); ux = ux < 0.0f ? 0.0f : (ux > 1.0f ? 1.0f : ux);
				float uy = fvy - float(vy0); uy = uy < 0.0f ? 0.0f : (uy > 1.0f ? 1.0f : uy);

				int j00 = 3*(vx0 + volW*vy0), j10 = 3*(vx1 + volW*vy0);
				int j01 = 3*(vx0 + volW*vy1), j11 = 3*(vx1 + volW*vy1);
				float q00 = (1.0f-ux)*(1.0f-uy), q10 = ux*(1.0f-uy);
				float q01 = (1.0f-ux)*uy,        q11 = ux*uy;

				r += volBuf[j00+0]*q00 + volBuf[j10+0]*q10 + volBuf[j01+0]*q01 + volBuf[j11+0]*q11;
				g += volBuf[j00+1]*q00 + volBuf[j10+1]*q10 + volBuf[j01+1]*q01 + volBuf[j11+1]*q11;
				b += volBuf[j00+2]*q00 + volBuf[j10+2]*q10 + volBuf[j01+2]*q01 + volBuf[j11+2]*q11;
			}

			if(uniforms.enableBloom){
				// bilinear upsample from half resolution
				float fx = (float(x) - 0.5f) * 0.5f;
				float fy = (float(y) - 0.5f) * 0.5f;
				int x0 = int(fx); x0 = x0 < 0 ? 0 : (x0 >= bloomW ? bloomW - 1 : x0);
				int y0 = int(fy); y0 = y0 < 0 ? 0 : (y0 >= bloomH ? bloomH - 1 : y0);
				int x1 = (x0 + 1 >= bloomW) ? bloomW - 1 : x0 + 1;
				int y1 = (y0 + 1 >= bloomH) ? bloomH - 1 : y0 + 1;
				float tx = fx - float(x0);
				float ty = fy - float(y0);

				int i00 = mipOff[0] + 3*(x0 + bloomW*y0), i10 = mipOff[0] + 3*(x1 + bloomW*y0);
				int i01 = mipOff[0] + 3*(x0 + bloomW*y1), i11 = mipOff[0] + 3*(x1 + bloomW*y1);
				float w00 = (1.0f-tx)*(1.0f-ty), w10 = tx*(1.0f-ty);
				float w01 = (1.0f-tx)*ty,        w11 = tx*ty;

				float s = uniforms.bloomStrength;
				r += s * (bloomChain[i00+0]*w00 + bloomChain[i10+0]*w10 + bloomChain[i01+0]*w01 + bloomChain[i11+0]*w11);
				g += s * (bloomChain[i00+1]*w00 + bloomChain[i10+1]*w10 + bloomChain[i01+1]*w01 + bloomChain[i11+1]*w11);
				b += s * (bloomChain[i00+2]*w00 + bloomChain[i10+2]*w10 + bloomChain[i01+2]*w01 + bloomChain[i11+2]*w11);
			}

			// ---- colour grading (apply_features.ps1) ---------------------
			// Geometry only: the sky keeps the exact colour it was given, and
			// emission has already been added, so light sources are graded with
			// the surfaces they sit on rather than separately.
			if(isGeometry){
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

			r = acesToneMap(r);
			g = acesToneMap(g);
			b = acesToneMap(b);

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

