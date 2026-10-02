// modules/progressive_octree/main_progressive_octree.cpp


#include <iostream>
#include <filesystem>
#include <locale.h>
#include <string>
#include <queue>
#include <vector>
#include <mutex>
#include <thread>
#include <deque>
#include <atomic>
#include <optional>
#ifdef __cpp_lib_format
#include <format>
#else
#include "fmt/core.h"
using namespace fmt;
#endif

#include "CudaModularProgram.h"
#include "GLRenderer.h"
#include "cudaGL.h"
#include "../CudaPrint/CudaPrint.h"

#include "unsuck.hpp"
#include "laszip_api.h"

#include "HostDeviceInterface.h"
#include "CrashHandler.h"

// Uniforms is passed to the kernels by value, and CUDA caps kernel parameters
// at 4 KB. Growing past that fails at launch, not at compile time, without this.
static_assert(sizeof(Uniforms) <= 4096, "Uniforms exceeds the 4 KB kernel-parameter limit");
#include "SimlodLoader.h"
#include "LasLoader.h"

using namespace std;

constexpr uint64_t PINNED_MEM_POOL_SIZE = 200;         // pool of pointers to batches of pinned memory
constexpr uint64_t BATCH_STREAM_SIZE    = 50;          // ring buffer of batches that are async streamed to GPU
constexpr uint64_t MAX_BATCH_SIZE       = 1'000'000;   // each loaded batch comprises <size> points
constexpr int MAX_LOADQUEUE_SIZE        = 300;         // stop loading from disk if processing lags behind

CudaPrint cudaprint;


vector<string> paths = {
	"NONE",
	// "d:/dev/pointclouds/riegl/retz.las",
};

struct PinnedMemorySlot {
	void* memLocation;
	CUevent uploadEnd;

	// ReadFileEx with unbuffered flag has special alignment requirements.
	// memLocation + memOffset gives the correct beginning of the data
	uint64_t memOffset;
};

struct Point{
	float x;
	float y;
	float z;
	union{
		uint32_t color;
		uint8_t rgba[4];
	};
};

struct PointBatch{
	string file = "";
	int first = 0;
	int count = 0;
	shared_ptr<vector<Point>> points;
	PinnedMemorySlot pinnedMem;

	LasHeader lasHeader;
};

bool lazInBatches = false;
deque<PointBatch> batchesToProcess;
deque<PointBatch> batchesInPinnedMemory;
deque<PointBatch> batchesInPageableMemory;
deque<PinnedMemorySlot> pinnedMemoryInUpload;
atomic_bool resetInProgress;
mutex mtx_uploader;
vector<unique_ptr<mutex>> mtx_loader;
mutex mtx_batchesToProcess;
mutex mtx_batchesInPinnedMemory;
mutex mtx_batchesInPageableMemory;
mutex mtx_pinnedMemoryInUpload;

atomic_int batchStreamUploadIndex = 0;   // written by the uploader, read by the loaders and main thread
CUdeviceptr cptr_points_ring[BATCH_STREAM_SIZE];

CUdevice device;
CUcontext context;
int numSMs;

CUdeviceptr cptr_buffer;
CUdeviceptr cptr_buffer_persistent;
CUdeviceptr cptr_nodes;
CUdeviceptr cptr_renderbuffer;
CUdeviceptr cptr_stats;
CUdeviceptr cptr_numBatchesUploaded;
CUdeviceptr cptr_batchSizes;
CUdeviceptr cptr_batchTags;
CUdeviceptr cptr_frameStart;
CUgraphicsResource cugl_colorbuffer;
CUevent ce_render_start, ce_render_end;
CUevent ce_update_start, ce_update_end;
cudaStream_t stream_upload, stream_download;

CudaModularProgram* cuda_program_update = nullptr;
CudaModularProgram* cuda_program_render = nullptr;
// CudaModularProgram* cuda_program_filter = nullptr;
CudaModularProgram* cuda_program_reset  = nullptr;

glm::mat4 transform;
glm::mat4 transform_updatebound;

Stats stats;
void* h_stats_pinned = nullptr;

double t_drop_start = 0.0;

struct {
	bool useHighQualityShading       = true;
	bool showBoundingBox             = false;
	bool doUpdateVisibility          = true;
	bool showPoints                  = true;
	bool showMarkers                 = true;    // apply_markers2.ps1
	bool markersOnly                 = false;   // apply_markers2.ps1: hide forest and insects
	bool colorByNode                 = false;
	bool colorByLOD                  = false;
	bool colorWhite                  = false;
	bool autoFocusOnLoad             = true;
	bool benchmarkRendering          = false;
	float LOD                        = 0.2f;
	float minNodeSize                = 64.0f;
	bool  adaptiveDetail             = true;    // coarsen LOD when frames exceed the budget
	float frameBudgetMs              = 33.0f;   // render-kernel time to aim for (30 fps)
	bool  vsync                      = true;    // cap at display refresh: saves power and heat
	int pointSize                    = 1;
	float fovy                       = 60.0f;
	bool enableEDL                   = false;   // reference has no outlines (apply_look.ps1)
	float edlStrength                = 0.8f;

	// tier 1 shading (inserted by apply_tier1_shading.ps1)
	bool  roundSplats                = true;
	bool  enableFog                  = true;    // distance goes LIGHTER (apply_look.ps1)
	float fogDensity                 = 0.018f;
	float fogR                       = 0.72f;   // pale green haze
	float fogG                       = 0.82f;
	float fogB                       = 0.70f;
	bool  enableBloom                = false;
	float bloomThreshold             = 0.55f;
	float bloomStrength              = 1.2f;
	float bloomRadius                = 3.0f;   // now a mip level count, 0-5
	float exposure                   = 0.75f;   // low-key, as the reference
	float emissionGain               = 3.0f;
	int   numInsects                 = 6000;
	int   insectSize                 = 2;
	float insectGlow                 = 2.0f;
	float insectSpeed                = 0.5f;
	bool  showVoxelGrid              = false;
	int   insectPlacement            = 1;   // 0 anywhere, 1 in air near foliage

	// permanent test source, for tuning the light settings without waiting
	// for the schedule. File coordinates; the load offset is applied like any
	// other event. Default sits in a 17 m gap between two trees near the
	// centre of the test scene, with 10 m clearance to anything else.
	bool  testSourceOn               = false;
	float testX                      = -3.0f;
	float testY                      = -8.0f;
	float testZ                      = 14.0f;
	float testIntensity              = 1.0f;
	float testHz                     = 280.0f;
	bool  showEventMarkers           = true;
	bool  useVoxelGrid               = true;
	float pointDensity               = 1.0f;
	bool  enableVolumetric           = false;   // not in the reference; advanced only
	float volDensity                 = 0.035f;
	float volHeightFalloff           = 0.02f;
	float volOcclusion               = 0.6f;
	int   volSteps                   = 48;
	float volMaxDist                 = 160.0f;
	float volScatter                 = 1.5f;
	int   volShadowSteps             = 6;
	float falloffPower               = 2.5f;
	float coreBoost                  = 1.5f;
	int   minNeighbours              = 0;      // 0 = culling off
	int   cullReach                  = 0;
	bool  showCulledOnly             = false;

	// apply_features.ps1
	int64_t targetPoints             = 0;       // 0 = not yet set; filled from the cloud on load
	float saturation                 = 1.25f;
	float temperature                = -0.10f;  // blue-shifted blacks
	float tintColour[3]              = {0.55f, 0.75f, 0.60f};
	float tintMix                    = 0.0f;
	float skyColour[3]               = {0.72f, 0.82f, 0.70f};   // pale green, matching the haze

	// look refinement (apply_look.ps1)
	float pointWorldSize             = 0.06f;   // metres; 0 falls back to pointSize pixels
	float pointShading               = 0.6f;
	int   maxSplatPx                 = 24;
	float grain                      = 0.04f;
	float vignette                   = 0.35f;

	// directional key light (apply_keylight.ps1). Defaults: high and behind,
	// as in the reference, warm-green light through the canopy.
	float lightAzimuth               = 205.0f;  // degrees
	float lightElevation             = 62.0f;   // degrees above horizon
	float lightColour[3]             = {0.98f, 1.00f, 0.90f};   // near white, faintly warm
	float lightIntensity             = 1.2f;
	float ambient                    = 0.35f;   // shadowed points keep a third of their colour
	float lightAbsorb                = 0.40f;   // per metre: 5 m of canopy passes ~14%
	int   lightSteps                 = 40;

	// regroup (apply_regroup.ps1)
	bool  lightOn                    = false;   // off until you switch it on
	float nearClip                   = 0.5f;
	bool  groundOn                   = true;
	float groundHeight               = 0.35f;
	float groundColour[3]            = {0.0f, 0.0f, 0.0f};
	float mixColour[3]               = {1.0f, 1.0f, 1.0f};
	float eyeHeight                  = 1.8f;    // metres above the lowest point
	float lookUpDeg                  = 75.0f;
	int   lensMode                   = 0;       // 0 perspective, 1 fisheye
	float fisheyeFov                 = 180.0f;
	float belowColour[3]             = {0.0f, 0.0f, 0.0f};   // void under the ground
	bool  findTargets                = false;
	float findThreshold              = 170.0f;
	float markerScale                = 0.10f;   // apply_markers2.ps1: metres, ball diameter
	float findGrey                   = 30.0f;
	bool  findIsolate                = false;
	int   findBoost                  = 6;
	float horizonSoftness            = 12.0f;   // degrees
	float emissionColour[3]          = {1.0f, 0.45f, 0.85f};   // the reference pink
	bool  emissionByFrequency        = false;
	bool  fitToCloudDone             = false;
} settings;

struct PinnedMemPool{
	mutex mtx_pool;
	mutex mtx_register;

	deque<PinnedMemorySlot> pool;
	vector<PinnedMemorySlot> registered;

	PinnedMemPool(){
		
	}

	// put all registered batches back into the pool
	void refill(){
		pool.clear();
		pool.insert(
			pool.end(),
			make_move_iterator(registered.begin()),
			make_move_iterator(registered.end()));
	}

	void reserveSlots(int numSlots){
		for(int i = 0; i < numSlots; i++){
			this->reserveSlot();
		}
	}

	void reserveSlot(){
		uint64_t bytesPerSlot = MAX_BATCH_SIZE * sizeof(Point);
		// reserve some extra bytes in case of IO with special alignment
		uint64_t alignmentPadding = 1'048'576;

		void* pinnedMem = nullptr;
		CUevent slotEvent;

		// unchecked, a failure left pinnedMem uninitialised and the loaders
		// then wrote 16 MB through it
		CUresult res = cuMemAllocHost((void**)&pinnedMem, bytesPerSlot + alignmentPadding);
		if(res != CUDA_SUCCESS || pinnedMem == nullptr){
			fprintf(stderr, "[load] cuMemAllocHost of %.0f MB pinned memory failed (%d)\n",
				double(bytesPerSlot + alignmentPadding) / 1e6, int(res));
			throw std::bad_alloc();   // reaches the terminate handler, which logs a stack
		}
		cuEventCreate(&slotEvent, 0);

		PinnedMemorySlot slot = {
			.memLocation = pinnedMem, 
			.uploadEnd   = slotEvent,
		};
		pool.push_back(slot);

		lock_guard<mutex> lock(mtx_register);
		registered.push_back(slot);
	}

	PinnedMemorySlot acquire(){
		lock_guard<mutex> lock(mtx_pool);

		if(pool.size() == 0){
			printfmt("pool is empty, allocating additional pinned memory slots \n");
			reserveSlot();
		}

		PinnedMemorySlot slot = pool.front();
		slot.memOffset = 0;

		pool.pop_front();

		return slot;
	}

	void release(PinnedMemorySlot slot){
		lock_guard<mutex> lock(mtx_pool);

		pool.push_back(slot);
	}

	void release(vector<PinnedMemorySlot> slots){

		if (slots.size() == 0) return;

		lock_guard<mutex> lock(mtx_pool);

		pool.insert(
			pool.end(),
			make_move_iterator(slots.begin()),
			make_move_iterator(slots.end()));
	}

};


bool requestReset                  = false;
bool requestBenchmark              = false;
atomic_bool requestStepthrough     = false;
atomic_bool requestStep            = false;
bool requestColorFiltering         = false;
float renderingDuration            = 0.0f;
uint32_t numPointsUploaded         = 0;
float loadStart                    = 0.0f;

float kernelUpdateDuration         = 0.0f;
float totalUpdateDuration          = 0.0f;
double minKernelUpdateDuration     = Infinity;
double maxKernelUpdateDuration     = 0.0;
double avgKernelUpdateDuration     = 0.0;
double cntKernelUpdateDuration     = 0.0;

float kernelRenderDuration         = 0.0f;
float totalRenderDuration          = 0.0f;
double minKernelRenderDuration     = Infinity;
double maxKernelRenderDuration     = 0.0;
double avgKernelRenderDuration     = 0.0;
double cntKernelRenderDuration     = 0.0;

atomic_uint64_t numPointsTotal     = 0;
atomic_uint64_t numPointsLoaded    = 0;
atomic_uint64_t numBytesTotal      = 0;
atomic_uint64_t numBytesLoaded     = 0;
atomic_uint64_t numThreadsLoading  = 0;
atomic_int numBatchesTotal         = 0;
int numBatchesProcessed            = 0;
bool lastBatchFinishedDevice       = false;
bool loadStoppedAtCapacity         = false;

// How far the construct kernel has got, for the loader and uploader threads to
// gate on. Copied from stats after each readback: the main thread overwrites
// the stats struct itself every frame while those threads would be reading it.
atomic_uint32_t batchesProcessedHost = 0;
atomic_uint64_t pointsProcessedHost  = 0;
uint64_t momentaryBufferCapacity   = 0;
uint64_t persistentBufferCapacity  = 0;
vector<double> processFrameTimes; 

PinnedMemPool pinnedMemPool;

// ---- run log (apply_insects.ps1) -----------------------------------------
// Every [host] line is written to the console and to log/<timestamp>.log.
// The directory is created on first use, next to the working directory.
#include <cstdarg>
#include <ctime>
#include <cstring>
#include <algorithm>
#include <unordered_map>

static std::ofstream g_logFile;
static std::string   g_logPath;   // the crash handler appends here too
static bool          g_logOpened = false;
static size_t        g_logBytes  = 0;
static constexpr size_t LOG_MAX_BYTES = 20'000'000;   // one run; a 42 MB log was seen
static constexpr int    LOG_KEEP_RUNS = 10;           // older log files are deleted

// Delete all but the newest LOG_KEEP_RUNS - 1 logs, making room for this run's.
// Names carry a sortable timestamp, so name order is age order.
static void pruneOldLogs(){
	std::error_code ec;
	std::vector<std::filesystem::path> logs;
	for(auto& e : std::filesystem::directory_iterator("log", ec)){
		auto name = e.path().filename().string();
		if(e.is_regular_file() && name.rfind("simlod_", 0) == 0 && e.path().extension() == ".log"){
			logs.push_back(e.path());
		}
	}
	std::sort(logs.begin(), logs.end());
	int excess = int(logs.size()) - (LOG_KEEP_RUNS - 1);
	for(int i = 0; i < excess; i++) std::filesystem::remove(logs[i], ec);
}

static void logLine(const char* fmt, ...){
	char buf[2048];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);

	fputs(buf, stdout);

	if(!g_logOpened){
		g_logOpened = true;
		std::error_code ec;
		std::filesystem::create_directories("log", ec);
		pruneOldLogs();
		std::time_t tt = std::time(nullptr);
		std::tm tmv{};
	#ifdef _WIN32
		localtime_s(&tmv, &tt);
	#else
		localtime_r(&tt, &tmv);
	#endif
		char name[256];
		std::strftime(name, sizeof(name), "log/simlod_%Y-%m-%d_%H%M%S.log", &tmv);
		g_logPath = name;
		g_logFile.open(name);
		if(g_logFile.is_open()){
			printf("[log ] writing to %s\n", name);
		}
	}
	if(g_logFile.is_open()){
		size_t len = strlen(buf);
		if(g_logBytes + len > LOG_MAX_BYTES){
			g_logFile << "[log ] size cap reached; further lines go to the console only\n";
			g_logFile.close();
			return;
		}
		g_logBytes += len;
		g_logFile << buf;
		g_logFile.flush();   // flushed every line: a crash must not lose the log
	}
}

// Periodic status lines: written only when the text differs from the last
// copy written under the same key, and no more often than minInterval. The
// old once-a-second dump repeated identical lines and filled 42 MB in a run.
static void logIfChanged(const char* key, double minInterval, const char* fmt, ...){
	static std::unordered_map<std::string, std::pair<std::string, double>> last;
	char buf[2048];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	auto& entry = last[key];
	double t = now();
	if(entry.first == buf || t - entry.second < minInterval) return;
	entry.first = buf;
	entry.second = t;
	logLine("%s", buf);
}

// ---- settings tooltips (apply_tooltips.ps1) ------------------------------
// Attaches hover text to whichever ImGui widget was drawn immediately before.
// Wrapped so the text sits next to the widget it describes rather than in a
// table somewhere else.
static void tipFor(const char* text){
	if(ImGui::IsItemHovered()){
		ImGui::BeginTooltip();
		ImGui::PushTextWrapPos(460.0f);
		ImGui::TextUnformatted(text);
		ImGui::PopTextWrapPos();
		ImGui::EndTooltip();
	}
}

#include <csignal>

// ---- CUDA error reporting (fix_crash_instrument.ps1) -----------------------
// Kernel launches were checked, synchronisations never were. A kernel that
// faults on the device launches fine and the fault surfaces at the next
// cuCtxSynchronize - which nobody looked at - so a GPU memory error left no
// trace before the process died. Now every sync is checked; on error the name
// and the last known frame state go to the run log, then the process exits,
// because a CUDA context after an illegal-address fault is dead anyway.
static void cuCheck(CUresult res, const char* what){
	if(res == CUDA_SUCCESS) return;
	const char* name = nullptr; const char* desc = nullptr;
	cuGetErrorName(res, &name);
	cuGetErrorString(res, &desc);
	logLine("[CUDA ERROR] %s: %s (%d) - %s\n", what, name ? name : "?", int(res), desc ? desc : "");
	logLine("[CUDA ERROR] the context is unusable after this; exiting so the log is intact\n");
	fflush(stdout);
	exit(3);
}

float toggle = 1.0;

// ---- call schedule (inserted by apply_emission.ps1) -----------------------
// Reads the CSV the test kit generates. Columns:
//   call_id,source,freq_hz,channel,x,y,z,t_start_s,duration_s,t_end_s,phase
struct ScheduledCall {
	float  x, y, z;
	float  freqHz;
	double tStart, duration;
};
static std::vector<ScheduledCall> g_calls;
static bool   g_callsLoaded  = false;
static double g_showClock    = 0.0;
static bool   g_showPlaying  = false;
static float  g_visualTail   = 1.2f;    // seconds the glow outlasts the call
static float  g_eventRadius  = 4.0f;    // metres
static int    g_activeEvents = 0;        // for the GUI readout

#include <cstring>
#include <cmath>
#include <cstdlib>
// ---- survey frame (apply_markers2.ps1) ------------------------------------
// The mic survey gives each checkerboard a position in ITS frame (centre,
// north, east). The pick gives the same board a position in the FILE frame
// (the LAS coordinates). With three or more boards known in both, the rigid
// motion between the frames is fully determined:
//
//     survey = R * file + t
//
// This is the absolute orientation problem. Horn, B.K.P. (1987) "Closed-form
// solution of absolute orientation using unit quaternions", J. Opt. Soc. Am.
// A 4(4):629-642, gives R directly as the eigenvector of a 4x4 symmetric
// matrix built from the cross-covariance of the two point sets - no
// iteration, no initial guess, and the result is a proper rotation. Scale is
// fixed at 1: both frames are metric.
//
// The eigenvector is found by Jacobi rotations, which for a 4x4 symmetric
// matrix converge in a handful of sweeps to machine precision.
static bool   g_surveyValid   = false;
static double g_surveyR[9]    = {1,0,0, 0,1,0, 0,0,1};   // row-major
static double g_surveyT[3]    = {0,0,0};
static double g_surveyRms     = 0.0;
static int    g_surveyPairs   = 0;
static bool   g_callsInSurvey = false;   // calls.csv positions are survey coordinates

// survey -> file, the inverse of the fitted motion: file = R^T (survey - t)
static void surveyToFile(float& x, float& y, float& z){
	if(!g_surveyValid) return;
	double d0 = double(x) - g_surveyT[0], d1 = double(y) - g_surveyT[1], d2 = double(z) - g_surveyT[2];
	const double* R = g_surveyR;
	x = float(R[0]*d0 + R[3]*d1 + R[6]*d2);
	y = float(R[1]*d0 + R[4]*d1 + R[7]*d2);
	z = float(R[2]*d0 + R[5]*d1 + R[8]*d2);
}
static void fileToSurvey(double x, double y, double z, double& sx, double& sy, double& sz){
	const double* R = g_surveyR;
	sx = R[0]*x + R[1]*y + R[2]*z + g_surveyT[0];
	sy = R[3]*x + R[4]*y + R[5]*z + g_surveyT[1];
	sz = R[6]*x + R[7]*y + R[8]*z + g_surveyT[2];
}


// Strict numeric cell parse: the whole cell (bar surrounding whitespace) must
// be a finite number. std::stof would throw on "" or "north" and abort the app.
static bool parseNumber(const std::string& cell, double& out){
	const char* begin = cell.c_str();
	char* end = nullptr;
	out = std::strtod(begin, &end);
	if(end == begin) return false;
	while(*end == ' ' || *end == '\t' || *end == '\r') end++;
	return *end == '\0' && std::isfinite(out);
}

static std::vector<std::string> splitCsvLine(const std::string& line){
	std::vector<std::string> f;
	std::stringstream ss(line);
	std::string cell;
	while(std::getline(ss, cell, ',')) f.push_back(cell);
	return f;
}

static void loadCallSchedule(const std::string& path){
	g_calls.clear();
	std::ifstream in(path);
	if(!in){ std::cout << "[calls] cannot open " << path << std::endl; return; }
	std::string line;
	std::getline(in, line);                     // header
	int lineNo = 1, skipped = 0;
	while(std::getline(in, line)){
		lineNo++;
		if(line.empty() || line == "\r") continue;
		std::vector<std::string> f = splitCsvLine(line);
		double freq, x, y, z, t0, dur;
		if(f.size() < 10
			|| !parseNumber(f[2], freq) || !parseNumber(f[4], x) || !parseNumber(f[5], y)
			|| !parseNumber(f[6], z)    || !parseNumber(f[7], t0) || !parseNumber(f[8], dur)){
			logLine("[calls] %s line %d: malformed row skipped\n", path.c_str(), lineNo);
			skipped++;
			continue;
		}
		ScheduledCall c;
		c.freqHz   = float(freq);
		c.x        = float(x);
		c.y        = float(y);
		c.z        = float(z);
		c.tStart   = t0;
		c.duration = dur;
		if(g_callsInSurvey && g_surveyValid) surveyToFile(c.x, c.y, c.z);   // apply_markers2.ps1
		g_calls.push_back(c);
	}
	g_callsLoaded = !g_calls.empty();
	std::cout << "[calls] loaded " << g_calls.size() << " calls from " << path;
	if(skipped > 0) std::cout << " (" << skipped << " malformed rows skipped, see log)";
	std::cout << std::endl;
}

// Warm for low frequency, cool for high. Stretched across 240-310 Hz so the
// two test sources read as clearly different; widen this once real calls
// carry real frequencies.
static void freqToColour(float hz, float& r, float& g, float& b){
	// 220 Hz warm through 340 Hz cool, matching the four-source schedule
	float t = (hz - 220.0f) / 120.0f;
	t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
	r = 1.00f - 0.80f * t;
	g = 0.45f + 0.25f * t;
	b = 0.15f + 0.85f * t;
}
float lastFrameTime = static_cast<float>(now());
float timeSinceLastFrame = 0.0;

float3 boxMin  = float3{InfinityF, InfinityF, InfinityF};
float3 boxMax  = float3{-InfinityF, -InfinityF, -InfinityF};
float3 boxSize = float3{0.0, 0.0, 0.0};

// ---- voxel occupancy grid (apply_voxel_grid.ps1) --------------------------
// Sized from the cloud's bounding box on load: the longest side gets 256
// cells, the others proportionally, capped at 256 each. Zeroed on every load.
// Counts are accumulated on the GPU as points enter the octree.
static CUdeviceptr g_voxelGrid  = 0;
static int         g_gridResolution = 256;   // cells along the longest side
static int         g_gridBuiltAtRes = 0;     // what the CURRENT counts were built with
static CUdeviceptr g_lightGrid  = 0;         // apply_keylight.ps1: per-cell transmittance

// ---- markers (apply_navigate.ps1) -----------------------------------------
// Picked positions, renderer frame (translated by boxMin). Exported in FILE
// coordinates so they match the LAS/LAZ and the mic survey.
struct Marker {
	int   mic;
	float x, y, z;            // renderer frame (file minus boxMin)
	char  label[32];          // apply_markers2.ps1: editable, e.g. "north" or "mic 3"
	float sx, sy, sz;         // known survey coordinates, once entered
	bool  hasSurvey;
};
static std::vector<Marker> g_markers;
static int g_nextMic = 1;   // fix_nextmic.ps1: above the CSV helpers that use it
static Marker makeMarker(int mic, float x, float y, float z){
	Marker m{}; m.mic = mic; m.x = x; m.y = y; m.z = z; m.hasSurvey = false;
	snprintf(m.label, sizeof(m.label), "M%d", mic);
	return m;
}

// ---- on-screen notice (apply_markers2.ps1) --------------------------------
// One line at the top of the window for three seconds. Used when a marker
// lands and when the survey fit runs, so neither has to be found in the log.
static double g_noticeUntil = 0.0;
static char   g_noticeText[256] = {0};
static void notice(const char* fmt, ...){
	va_list ap; va_start(ap, fmt);
	vsnprintf(g_noticeText, sizeof(g_noticeText), fmt, ap);
	va_end(ap);
	g_noticeUntil = now() + 3.0;
}

// Largest-eigenvalue eigenvector of a symmetric 4x4 by cyclic Jacobi.
static void jacobiMaxEigenvector4(double N[4][4], double q[4]){
	double V[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
	for(int sweep = 0; sweep < 60; sweep++){
		double off = 0.0;
		for(int p = 0; p < 4; p++) for(int r = p + 1; r < 4; r++) off += N[p][r]*N[p][r];
		if(off < 1e-30) break;
		for(int p = 0; p < 3; p++){
			for(int r = p + 1; r < 4; r++){
				if(std::fabs(N[p][r]) < 1e-300) continue;
				double theta = (N[r][r] - N[p][p]) / (2.0 * N[p][r]);
				double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta*theta + 1.0));
				double c = 1.0 / std::sqrt(t*t + 1.0);
				double s = t * c;
				for(int k = 0; k < 4; k++){           // rotate columns p and r
					double kp = N[k][p], kr = N[k][r];
					N[k][p] = c*kp - s*kr;
					N[k][r] = s*kp + c*kr;
				}
				for(int k = 0; k < 4; k++){           // rotate rows p and r
					double pk = N[p][k], rk = N[r][k];
					N[p][k] = c*pk - s*rk;
					N[r][k] = s*pk + c*rk;
				}
				for(int k = 0; k < 4; k++){
					double kp = V[k][p], kr = V[k][r];
					V[k][p] = c*kp - s*kr;
					V[k][r] = s*kp + c*kr;
				}
			}
		}
	}
	int best = 0;
	for(int i = 1; i < 4; i++) if(N[i][i] > N[best][best]) best = i;
	for(int k = 0; k < 4; k++) q[k] = V[k][best];
}

// Fit from every marker that has survey coordinates. Fills g_surveyR/T and
// writes each marker's residual to the log. Returns false with fewer than
// three pairs, which is the minimum for a unique rigid motion in 3D.
static bool fitSurveyFrame(){
	std::vector<double> ax, ay, az, bx, by, bz;
	for(const Marker& m : g_markers){
		if(!m.hasSurvey) continue;
		ax.push_back(double(m.x) + double(boxMin.x));
		ay.push_back(double(m.y) + double(boxMin.y));
		az.push_back(double(m.z) + double(boxMin.z));
		bx.push_back(m.sx); by.push_back(m.sy); bz.push_back(m.sz);
	}
	int n = int(ax.size());
	if(n < 3){
		logLine("[survey] %d marker(s) with survey coordinates; three are needed\n", n);
		notice("Survey fit needs at least 3 markers with survey coordinates (have %d)", n);
		g_surveyValid = false;
		return false;
	}
	double ca[3] = {0,0,0}, cb[3] = {0,0,0};
	for(int i = 0; i < n; i++){
		ca[0] += ax[i]; ca[1] += ay[i]; ca[2] += az[i];
		cb[0] += bx[i]; cb[1] += by[i]; cb[2] += bz[i];
	}
	for(int k = 0; k < 3; k++){ ca[k] /= n; cb[k] /= n; }

	// cross-covariance, file (left) against survey (right): S_ab = sum a_i b_i
	double S[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
	for(int i = 0; i < n; i++){
		double a[3] = {ax[i]-ca[0], ay[i]-ca[1], az[i]-ca[2]};
		double b[3] = {bx[i]-cb[0], by[i]-cb[1], bz[i]-cb[2]};
		for(int r = 0; r < 3; r++) for(int c = 0; c < 3; c++) S[r][c] += a[r]*b[c];
	}
	double Sxx=S[0][0], Sxy=S[0][1], Sxz=S[0][2];
	double Syx=S[1][0], Syy=S[1][1], Syz=S[1][2];
	double Szx=S[2][0], Szy=S[2][1], Szz=S[2][2];
	// Horn 1987, eq. 27
	double N[4][4] = {
		{ Sxx+Syy+Szz,  Syz-Szy,       Szx-Sxz,       Sxy-Syx      },
		{ Syz-Szy,      Sxx-Syy-Szz,   Sxy+Syx,       Szx+Sxz      },
		{ Szx-Sxz,      Sxy+Syx,      -Sxx+Syy-Szz,   Syz+Szy      },
		{ Sxy-Syx,      Szx+Sxz,       Syz+Szy,      -Sxx-Syy+Szz  }
	};
	double q[4];
	jacobiMaxEigenvector4(N, q);
	double w = q[0], x = q[1], y = q[2], z = q[3];
	double R[9] = {
		w*w + x*x - y*y - z*z,  2.0*(x*y - w*z),        2.0*(x*z + w*y),
		2.0*(x*y + w*z),        w*w - x*x + y*y - z*z,  2.0*(y*z - w*x),
		2.0*(x*z - w*y),        2.0*(y*z + w*x),        w*w - x*x - y*y + z*z
	};
	for(int k = 0; k < 9; k++) g_surveyR[k] = R[k];
	g_surveyT[0] = cb[0] - (R[0]*ca[0] + R[1]*ca[1] + R[2]*ca[2]);
	g_surveyT[1] = cb[1] - (R[3]*ca[0] + R[4]*ca[1] + R[5]*ca[2]);
	g_surveyT[2] = cb[2] - (R[6]*ca[0] + R[7]*ca[1] + R[8]*ca[2]);
	g_surveyValid = true;
	g_surveyPairs = n;

	double sumSq = 0.0;
	int i = 0;
	for(const Marker& m : g_markers){
		if(!m.hasSurvey) continue;
		double px, py, pz;
		fileToSurvey(ax[i], ay[i], az[i], px, py, pz);
		double ex = px - bx[i], ey = py - by[i], ez = pz - bz[i];
		double e = std::sqrt(ex*ex + ey*ey + ez*ez);
		sumSq += e*e;
		logLine("[survey] %-10s file (%.3f %.3f %.3f) -> (%.3f %.3f %.3f)  survey (%.3f %.3f %.3f)  residual %.3f m\n",
			m.label, ax[i], ay[i], az[i], px, py, pz, bx[i], by[i], bz[i], e);
		i++;
	}
	g_surveyRms = std::sqrt(sumSq / n);
	double yawDeg = std::atan2(R[3], R[0]) * 180.0 / 3.14159265358979;
	logLine("[survey] fit from %d pairs: yaw %.2f deg, t = (%.3f %.3f %.3f), rms %.3f m\n",
		n, yawDeg, g_surveyT[0], g_surveyT[1], g_surveyT[2], g_surveyRms);
	logLine("[survey] R = [%.6f %.6f %.6f; %.6f %.6f %.6f; %.6f %.6f %.6f]\n",
		R[0],R[1],R[2],R[3],R[4],R[5],R[6],R[7],R[8]);
	{
		std::ofstream f("survey_transform.txt");
		f << "# survey = R * file + t   (file = LAS coordinates)\n";
		f << "pairs " << n << "\nrms_m " << g_surveyRms << "\nyaw_deg " << yawDeg << "\n";
		f << "R " << R[0]<<" "<<R[1]<<" "<<R[2]<<" "<<R[3]<<" "<<R[4]<<" "<<R[5]<<" "<<R[6]<<" "<<R[7]<<" "<<R[8] << "\n";
		f << "t " << g_surveyT[0] << " " << g_surveyT[1] << " " << g_surveyT[2] << "\n";
	}
	notice("Survey frame fitted from %d markers: yaw %.1f deg, rms %.3f m (see log)", n, yawDeg, g_surveyRms);
	return true;
}

// markers.csv (apply_markers2.ps1): file coordinates, label, survey coordinates
// Points are stored as float(file - boxMin) with boxMin itself a float, so
// double(boxMin) + double(rel) recovers file coordinates exactly. Write them
// with fixed decimals: the default stream precision (6 significant digits)
// rounds a georeferenced northing such as 5.9e6 to the nearest 10 m.
static void saveMarkersCsv(){
	std::ofstream f("markers.csv");
	if(!f.is_open()){
		logLine("[pick] cannot open markers.csv for writing\n");
		notice("Could not write markers.csv (is it open in another program?)");
		return;
	}
	f << "mic,label,x,y,z,sx,sy,sz,has_survey\n";
	for(const Marker& mk : g_markers){
		// commas or line breaks in a label would shift every later column
		char label[sizeof(mk.label)];
		snprintf(label, sizeof(label), "%s", mk.label);
		for(char* c = label; *c; c++) if(*c == ',' || *c == '"' || *c == '\n' || *c == '\r') *c = ' ';

		char row[512];
		snprintf(row, sizeof(row), "%d,%s,%.6f,%.6f,%.6f,%.9g,%.9g,%.9g,%d\n",
			mk.mic, label,
			double(mk.x) + double(boxMin.x), double(mk.y) + double(boxMin.y), double(mk.z) + double(boxMin.z),
			double(mk.sx), double(mk.sy), double(mk.sz), mk.hasSurvey ? 1 : 0);
		f << row;
	}
	f.close();
	if(f.fail()){
		logLine("[pick] write to markers.csv failed\n");
		notice("Writing markers.csv failed");
		return;
	}
	logLine("[pick] wrote %d markers to markers.csv (file coordinates)\n", int(g_markers.size()));
	notice("Saved %d markers to markers.csv", int(g_markers.size()));
}
static void loadMarkersCsv(){
	// markers are kept relative to boxMin, which is +infinity until a cloud loads
	if(!std::isfinite(boxMin.x) || !std::isfinite(boxMin.y) || !std::isfinite(boxMin.z)){
		notice("Load a point cloud before loading markers");
		return;
	}
	std::ifstream in("markers.csv");
	if(!in){ notice("markers.csv not found next to SimLOD.exe"); return; }
	std::vector<Marker> loaded;
	std::string line;
	std::getline(in, line);   // header
	int lineNo = 1, skipped = 0;
	while(std::getline(in, line)){
		lineNo++;
		if(line.empty() || line == "\r") continue;
		std::vector<std::string> f = splitCsvLine(line);
		double mic, x, y, z;
		if(f.size() < 5 || !parseNumber(f[0], mic)
			|| !parseNumber(f[2], x) || !parseNumber(f[3], y) || !parseNumber(f[4], z)){
			logLine("[pick] markers.csv line %d: malformed row skipped\n", lineNo);
			skipped++;
			continue;
		}
		Marker m{};
		m.mic = int(mic);
		snprintf(m.label, sizeof(m.label), "%s", f[1].c_str());
		m.x = float(x - double(boxMin.x));
		m.y = float(y - double(boxMin.y));
		m.z = float(z - double(boxMin.z));
		double sx, sy, sz, has;
		if(f.size() >= 9 && parseNumber(f[5], sx) && parseNumber(f[6], sy)
			&& parseNumber(f[7], sz) && parseNumber(f[8], has)){
			m.sx = float(sx); m.sy = float(sy); m.sz = float(sz);
			m.hasSurvey = has != 0.0;
		}
		loaded.push_back(m);
		if(loaded.size() >= 32) break;
	}
	g_markers = loaded;
	g_nextMic = 1;
	for(const Marker& m : g_markers) g_nextMic = std::max(g_nextMic, m.mic + 1);
	logLine("[pick] loaded %d markers from markers.csv (%d malformed rows skipped)\n", int(g_markers.size()), skipped);
	if(skipped > 0) notice("Loaded %d markers; skipped %d malformed rows (see log)", int(g_markers.size()), skipped);
	else            notice("Loaded %d markers from markers.csv", int(g_markers.size()));
}
static int    g_pickReqX = -1, g_pickReqY = -1;   // pixel of the outstanding request
static bool   g_pickWaiting = false;              // a request has not been answered yet
static int    g_pickSeq = 0;                      // id of the latest request
static int    g_pickSeenDone = 0;                 // consecutive Stats reads showing it answered
static bool   g_pKeyWasDown = false;          // P key state last frame (pick shortcut)
static bool   g_leftWasDown = false;
// g_nextMic moved up to the marker block (fix_nextmic.ps1)
static float  g_walkSpeed = 3.0f;                 // metres per second, W/A/S/D
static bool        g_lightDirty = true;
static bool        g_gridRefill = false;     // apply_grid_refill.ps1: counts need rebuilding
static int         g_gridX = 0, g_gridY = 0, g_gridZ = 0;
static float       g_cellSize   = 1.0f;
static size_t      g_gridBytes  = 0;

// Momentary buffer overrun check (apply_buffer_fix.ps1). The allocator does
// an atomicAdd and returns buffer + offset with no bounds test, so an overrun
// is silent until it corrupts something. Warn once per session.
static void checkMomentaryBudget(uint64_t used, uint64_t capacity){
	static bool warned = false;
	if(!warned && used > capacity){
		warned = true;
		printf("[WARN] momentary buffer overrun: %.1f MB used of %.1f MB. "
			   "Raise cptr_buffer_bytes. Expect crashes until you do.\n",
			   double(used) / 1'000'000.0, double(capacity) / 1'000'000.0);
	}
}

// Allocate the light transmittance grid only when the key light is on, at
// the occupancy grid's current dimensions. Freed by ensureVoxelGrid on resize
// and by the switch when the light is turned off.
static void ensureLightGrid(bool wanted){
	if(!wanted){
		if(g_lightGrid != 0){ cuMemFree(g_lightGrid); g_lightGrid = 0; }
		return;
	}
	if(g_lightGrid != 0) return;
	if(g_gridX <= 0 || g_gridY <= 0 || g_gridZ <= 0) return;
	// a failed size is not retried (and re-logged) every frame, only once the
	// occupancy grid changes dimensions
	static int failedX = -1, failedY = -1, failedZ = -1;
	if(g_gridX == failedX && g_gridY == failedY && g_gridZ == failedZ) return;
	size_t lbytes = size_t(g_gridX) * size_t(g_gridY) * size_t(g_gridZ) * sizeof(float);
	if(cuMemAlloc(&g_lightGrid, lbytes) != CUDA_SUCCESS){
		g_lightGrid = 0;
		failedX = g_gridX; failedY = g_gridY; failedZ = g_gridZ;
		logLine("[grid] light grid (%.0f MB) would not fit; key light disabled\n", double(lbytes) / 1e6);
		notice("Key light needs %.0f MB of GPU memory that is not free - lower the grid resolution", double(lbytes) / 1e6);
		return;
	}
	failedX = failedY = failedZ = -1;
	cuMemsetD8(g_lightGrid, 0, lbytes);
	g_lightDirty = true;
	logLine("[grid] light grid allocated, %.0f MB\n", double(lbytes) / 1e6);
}

static void ensureVoxelGrid(){
	float longest = std::max(boxSize.x, std::max(boxSize.y, boxSize.z));
	if(!(longest > 0.0f) || !std::isfinite(longest)) return;

	// Resolution is the number of cells along the cloud's LONGEST side; the
	// other two follow from the same cell size, so cells stay cubic.
	// Memory is 4 bytes a cell and grows with the cube of this, so 512 is the
	// practical ceiling on an 8 GB card - see the readout beside the slider.
	int R = g_gridResolution < 64 ? 64 : (g_gridResolution > 512 ? 512 : g_gridResolution);

	// After a failure at the minimum resolution, do not retry (and re-log)
	// every frame; wait until the cloud or the requested resolution changes.
	static int    failedRes = -1;
	static float3 failedBox = float3{0.0f, 0.0f, 0.0f};
	if(R == failedRes && boxSize.x == failedBox.x && boxSize.y == failedBox.y && boxSize.z == failedBox.z) return;

	float cell = longest / float(R);
	int gx = std::min(R, int(boxSize.x / cell) + 1);
	int gy = std::min(R, int(boxSize.y / cell) + 1);
	int gz = std::min(R, int(boxSize.z / cell) + 1);

	bool same = (gx == g_gridX && gy == g_gridY && gz == g_gridZ && cell == g_cellSize);
	if(same && g_voxelGrid != 0) return;

	size_t bytes = size_t(gx) * size_t(gy) * size_t(gz) * sizeof(uint32_t);
	if(g_voxelGrid != 0){
		cuMemFree(g_voxelGrid);   // always free first: the old grid is dead either
		g_voxelGrid = 0;          // way, and holding it makes the new one more
		g_gridBytes = 0;          // likely to fail for want of room
	}

	// cuMemAlloc was unchecked. On failure g_voxelGrid stayed 0, cuMemsetD8
	// then wrote through a null device pointer and the kernel read and wrote
	// through it after that - a crash at load, with nothing in the log to say
	// why. This reports the failure and steps the resolution down until it
	// fits, rather than dying.
	CUresult res = CUDA_ERROR_OUT_OF_MEMORY;
	while(true){
		size_t freeMem = 0, totalMem = 0;
		cuMemGetInfo(&freeMem, &totalMem);

		if(bytes + 64'000'000 < freeMem){
			res = cuMemAlloc(&g_voxelGrid, bytes);
		}else{
			res = CUDA_ERROR_OUT_OF_MEMORY;
			logLine("[grid] %zu MB needed but only %zu MB free on the card\n",
				bytes / 1'000'000, freeMem / 1'000'000);
		}

		if(res == CUDA_SUCCESS) break;

		g_voxelGrid = 0;
		if(R <= 64){
			logLine("[grid] FAILED even at resolution 64. The occupancy grid is off: "
					"insect placement, volumetric shadows and stray culling will do nothing.\n");
			g_gridX = g_gridY = g_gridZ = 0;
			g_gridBytes = 0;
			g_gridBuiltAtRes = 0;
			failedRes = R;
			failedBox = boxSize;
			return;
		}

		R /= 2;                       // halve and try again
		g_gridResolution = R;         // so the slider shows what was actually used
		cell = longest / float(R);
		gx = std::min(R, int(boxSize.x / cell) + 1);
		gy = std::min(R, int(boxSize.y / cell) + 1);
		gz = std::min(R, int(boxSize.z / cell) + 1);
		bytes = size_t(gx) * size_t(gy) * size_t(gz) * sizeof(uint32_t);
		logLine("[grid] stepping resolution down to %d (%.0f MB) and retrying\n",
			R, double(bytes) / 1e6);
	}

	g_gridBytes = bytes;
	failedRes = -1;

	// The light grid must match these dimensions, so drop it here; it is
	// allocated again on demand, and only while the key light is on. It used
	// to be allocated unconditionally - 20 to 156 MB for a feature that
	// defaults to off.
	if(g_lightGrid != 0){ cuMemFree(g_lightGrid); g_lightGrid = 0; }
	g_lightDirty = true;
	cuMemsetD8(g_voxelGrid, 0, bytes);

	g_gridX = gx; g_gridY = gy; g_gridZ = gz; g_cellSize = cell;
	g_gridBuiltAtRes = R;
	// The grid has just been zeroed, so whatever counts it held are gone. Say
	// so, rather than setting g_gridBuiltAtRes and reporting it as current -
	// which is what hid this: the resolution slider emptied the grid on every
	// move and the panel kept saying the grid was up to date.
	g_gridRefill = true;
	logLine("[grid] %d x %d x %d cells, %.3f m each, %.1f MB\n",
		gx, gy, gz, cell, double(bytes) / 1e6);
}

// ---- per-frame state --------------------------------------------------------
// getUniforms() is called by every kernel launch, several times a frame, so it
// must not allocate or consume one-shot flags: the first caller would take a
// flag the render kernel never sees. Grid allocation and the refill/light
// requests are settled here, once, at the start of each frame.
static bool g_frameGridRefill = false;   // render kernel recounts the occupancy grid
static bool g_frameLightDirty = false;   // render kernel rebuilds light transmittance
static float g_adaptReduce    = 0.0f;    // adaptive detail: 0 = as configured, 1 = cheapest

// Adaptive detail keeps the render kernel near the frame budget by lowering
// one reduction level that drives two costs:
//  - the LOD threshold (minNodeSize, up to 4x): fewer, coarser nodes far away
//  - the splat-size cap (down to a quarter): near the camera every point of
//    the nearest leaves is drawn regardless of LOD, as a splat of up to
//    maxSplatPx squared pixels, so on a dense scan seen up close this is the
//    cost that dominates (measured: LOD alone at 16x left a 54 M point scan
//    at 140 ms per frame).
// Reduces faster than it restores, which damps oscillation from the one- to
// two-frame lag of the asynchronous Stats copy.
static void updateAdaptiveDetail(){
	if(!settings.adaptiveDetail || settings.frameBudgetMs <= 0.0f){
		g_adaptReduce = 0.0f;
		return;
	}
	if(stats.stageNs[6] == 0) return;   // no measurement yet
	double ms = double(stats.stageNs[6]) / 1e6;
	if(ms > settings.frameBudgetMs * 1.10)      g_adaptReduce += 0.03f;
	else if(ms < settings.frameBudgetMs * 0.70) g_adaptReduce -= 0.015f;
	g_adaptReduce = std::clamp(g_adaptReduce, 0.0f, 1.0f);
}
static float effectiveMinNodeSize(){ return settings.minNodeSize * (1.0f + 3.0f * g_adaptReduce); }
static int   effectiveMaxSplatPx(){
	int px = int(float(settings.maxSplatPx) * (1.0f - 0.75f * g_adaptReduce) + 0.5f);
	return std::max(2, std::min(px, settings.maxSplatPx));
}

static void beginFrameState(){
	updateAdaptiveDetail();

	ensureVoxelGrid();
	ensureLightGrid(settings.lightOn);

	// Recount only once the device has processed every point. During a load
	// the construct kernel counts points as it inserts them, and a recount
	// part-way through would count the not-yet-inserted ones twice.
	g_frameGridRefill = g_gridRefill && lastBatchFinishedDevice && stats.numNodes > 0 && g_voxelGrid != 0;
	if(g_frameGridRefill){
		g_gridRefill = false;
		g_lightDirty = true;   // transmittance is derived from the counts
	}

	g_frameLightDirty = g_lightDirty && g_lightGrid != 0;
	if(g_frameLightDirty) g_lightDirty = false;
}

uint64_t frameCounter = 0;

void initCuda(){
	cuInit(0);
	cuDeviceGet(&device, 0);
	cuCtxCreate(&context, 0, device);
	cuStreamCreate(&stream_upload, CU_STREAM_NON_BLOCKING);
	cuStreamCreate(&stream_download, CU_STREAM_NON_BLOCKING);

	cuCtxGetDevice(&device);
	cuDeviceGetAttribute(&numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
}

Uniforms getUniforms(shared_ptr<GLRenderer> renderer){
	Uniforms uniforms;

	glm::mat4 world;
	glm::mat4 view = renderer->camera->view;
	glm::mat4 proj = renderer->camera->proj;
	glm::mat4 worldViewProj = proj * view * world;
	world = glm::transpose(world);
	view = glm::transpose(view);
	proj = glm::transpose(proj);
	worldViewProj = glm::transpose(worldViewProj);

	memcpy(&uniforms.world, &world, sizeof(world));
	memcpy(&uniforms.view, &view, sizeof(view));
	memcpy(&uniforms.proj, &proj, sizeof(proj));
	memcpy(&uniforms.transform, &worldViewProj, sizeof(worldViewProj));

	// volumetric medium (apply_volumetric.ps1): clip -> world for view rays,
	// plus the camera position and forward axis. worldViewProj was transposed
	// above for the device's row-major mat4; inverting it directly gives the inverse.
	{
		glm::mat4 wvpInv = glm::inverse(worldViewProj);   // fix_transform_inv
		memcpy(&uniforms.transformInv, &wvpInv, sizeof(wvpInv));

		glm::dvec3 cp = renderer->camera->position;
		// The camera already lives in the TRANSLATED frame: autofocus sets
	// controls->target from boxSize, not from file coordinates. So unlike
	// an event position, camPos must NOT have boxMin subtracted again.
	uniforms.camPos = float3{float(cp.x), float(cp.y), float(cp.z)};

		// camera forward in world space is -Z of the camera-to-world rotation
		glm::dmat4 cw = renderer->camera->world;
		glm::dvec3 fwd = -glm::normalize(glm::dvec3(cw[2]));
		uniforms.camDir = float3{float(fwd.x), float(fwd.y), float(fwd.z)};
		glm::dvec3 rgt = glm::normalize(glm::dvec3(cw[0]));
		glm::dvec3 upv = glm::normalize(glm::dvec3(cw[1]));
		uniforms.camRight   = float3{float(rgt.x), float(rgt.y), float(rgt.z)};
		uniforms.camUp      = float3{float(upv.x), float(upv.y), float(upv.z)};
		uniforms.lensMode   = settings.lensMode;
		uniforms.fisheyeFov = settings.fisheyeFov;
		uniforms.belowR = settings.belowColour[0];
		uniforms.belowG = settings.belowColour[1];
		uniforms.belowB = settings.belowColour[2];
		uniforms.findTargets   = settings.findTargets ? 1 : 0;
		uniforms.findThreshold = settings.findThreshold;
		uniforms.markerScale   = settings.markerScale;
		uniforms.findGrey      = settings.findGrey;
		uniforms.findIsolate   = settings.findIsolate ? 1 : 0;
		uniforms.findBoost     = settings.findBoost;
		uniforms.horizonSoftness = settings.horizonSoftness;
		uniforms.pickX = g_pickReqX;
		uniforms.pickY = g_pickReqY;
		uniforms.pickSeq = g_pickSeq;
		uniforms.numMarkers = int(std::min<size_t>(g_markers.size(), 32));   // apply_markers2.ps1
		uniforms.showMarkers = settings.showMarkers ? 1 : 0;
		for(int i = 0; i < uniforms.numMarkers; i++){
			uniforms.markerX[i] = g_markers[i].x;
			uniforms.markerY[i] = g_markers[i].y;
			uniforms.markerZ[i] = g_markers[i].z;
		}
	}

	if(settings.doUpdateVisibility){
		transform_updatebound = worldViewProj;
	}

	glm::mat4 transform_inv_updatebound = glm::inverse(transform_updatebound);
	memcpy(&uniforms.transform_updateBound, &transform_updatebound, sizeof(transform_updatebound));
	memcpy(&uniforms.transformInv_updateBound, &transform_inv_updatebound, sizeof(transform_inv_updatebound));
	
	uniforms.width                    = static_cast<float>(renderer->width);
	uniforms.height                   = static_cast<float>(renderer->height);
	uniforms.fovy_rad                 = 3.1415f * renderer->camera->fovy / 180.0;
	uniforms.time                     = static_cast<float>(now());
	uniforms.boxMin                   = float3{0.0f, 0.0f, 0.0f};
	uniforms.boxMax                   = boxSize;
	uniforms.frameCounter             = frameCounter;
	uniforms.showBoundingBox          = settings.showBoundingBox;
	uniforms.doUpdateVisibility       = settings.doUpdateVisibility;
	uniforms.showPoints               = settings.showPoints;
	uniforms.colorByNode              = settings.colorByNode;
	uniforms.colorByLOD               = settings.colorByLOD;
	uniforms.colorWhite               = settings.colorWhite;
	uniforms.LOD                      = settings.LOD;
	uniforms.minNodeSize              = effectiveMinNodeSize();
	uniforms.pointSize                = settings.pointSize;
	uniforms.useHighQualityShading    = settings.useHighQualityShading;
	uniforms.persistentBufferCapacity = persistentBufferCapacity;
	uniforms.momentaryBufferCapacity  = momentaryBufferCapacity;
	uniforms.enableEDL                = settings.enableEDL;
	uniforms.edlStrength              = settings.edlStrength;

	uniforms.roundSplats              = settings.roundSplats;
	uniforms.enableFog                = settings.enableFog;
	uniforms.fogDensity               = settings.fogDensity;
	uniforms.fogR                     = settings.fogR;
	uniforms.fogG                     = settings.fogG;
	uniforms.fogB                     = settings.fogB;
	uniforms.enableBloom              = settings.enableBloom;
	uniforms.bloomThreshold           = settings.bloomThreshold;
	uniforms.bloomStrength            = settings.bloomStrength;
	uniforms.bloomRadius              = settings.bloomRadius;
	uniforms.exposure                 = settings.exposure;

	// ---- excitation: fill the active-event uniforms from the schedule ----
	// The clock here is wall time. When the audio joins, replace it with the
	// playback position in seconds and picture and sound cannot drift.
	{
		// getUniforms is called for the update kernel, the render kernel and
		// on reset - two or three times a frame. Advance the clock only when
		// the frame number changes, so every launch in a frame sees one time.
		static double  lastT     = now();
		static int64_t lastFrame = -1;
		double nowT = now();
		if(renderer->frameCount != lastFrame){
			double dt = nowT - lastT;
			if(g_showPlaying) g_showClock += dt;
			lastT     = nowT;
			lastFrame = renderer->frameCount;
		}

		uniforms.numEvents    = 0;
		g_activeEvents        = 0;
		uniforms.emissionGain = settings.emissionGain;

		// permanent test source goes in slot 0, always on when enabled, so it
		// is ev0 in the log and independent of whether the show is playing
		{
			bool haveCloud0 = std::isfinite(boxMin.x) && std::isfinite(boxMin.y) && std::isfinite(boxMin.z);
			if(settings.testSourceOn && haveCloud0){
				float cr, cg, cb;
				if(settings.emissionByFrequency) freqToColour(settings.testHz, cr, cg, cb);
				else { cr = settings.emissionColour[0]; cg = settings.emissionColour[1]; cb = settings.emissionColour[2]; }
				int i = uniforms.numEvents++;
				uniforms.eventX[i]  = settings.testX - boxMin.x;
				uniforms.eventY[i]  = settings.testY - boxMin.y;
				uniforms.eventZ[i]  = settings.testZ - boxMin.z;
				uniforms.eventR[i]  = g_eventRadius;
				uniforms.eventI[i]  = settings.testIntensity;
				uniforms.eventCR[i] = cr;
				uniforms.eventCG[i] = cg;
				uniforms.eventCB[i] = cb;
				g_activeEvents = uniforms.numEvents;
			}
		}
		uniforms.numInsects   = settings.numInsects;
		uniforms.insectSize   = settings.insectSize;
		uniforms.insectGlow   = settings.insectGlow;
		uniforms.insectSpeed  = settings.insectSpeed;

	checkMomentaryBudget(stats.allocatedBytes_momentary, momentaryBufferCapacity);

	// ---- fit settings to the cloud (apply_features.ps1) --------------------
	// Everything below was hard-coded for the 100 m synthetic plot. Now it is
	// derived from the loaded cloud's extent, once per load, so a 20 m scan and
	// a 400 m scan both come up with sensible defaults. The sliders keep their
	// cloud-relative ranges afterwards so they can always reach the whole cloud.
	{
		static float3 fittedSize = float3{-1.0f, -1.0f, -1.0f};
		bool sizeChanged = (boxSize.x != fittedSize.x || boxSize.y != fittedSize.y || boxSize.z != fittedSize.z);
		bool haveBox = boxSize.x > 0.0f && std::isfinite(boxMin.x);
		if(haveBox && sizeChanged){
			fittedSize = boxSize;
			float longest = std::max(boxSize.x, std::max(boxSize.y, boxSize.z));
			float diag    = std::sqrt(boxSize.x*boxSize.x + boxSize.y*boxSize.y + boxSize.z*boxSize.z);

			// test source: centre of the cloud in plan, at 55% of its height
			settings.testX = boxMin.x + boxSize.x * 0.5f;
			settings.testY = boxMin.y + boxSize.y * 0.5f;
			settings.testZ = boxMin.z + boxSize.z * 0.55f;

			g_eventRadius        = longest * 0.12f;    // 12% of the longest side
			settings.volMaxDist  = diag * 1.5f;        // rays can cross the whole cloud
			settings.fitToCloudDone = true;
			logLine("[host] fitted to cloud: extent %.1f x %.1f x %.1f m, test source (%.1f, %.1f, %.1f), radius %.1f m, reach %.0f m\n",
				boxSize.x, boxSize.y, boxSize.z, settings.testX, settings.testY, settings.testZ, g_eventRadius, settings.volMaxDist);
		}
	}

	// ---- target point count <-> density (apply_features.ps1) --------------
	// The editable number is the target; density is derived from it against
	// the loaded total, so "35,000,000 / 54,522,507" means exactly that.
	{
		// The target used to fill from the point count on the FIRST frame after
		// load, when only a fraction of the cloud had arrived, then hold that
		// - so density drifted to 0.018 on its own as the rest streamed in.
		// Now it follows the total for as long as the user has not touched it:
		// untouched means "everything", whatever everything turns out to be.
		static int64_t lastTotal = 0;
		double total = double(stats.numPoints);
		int64_t totalI = int64_t(total);
		bool untouched = (settings.targetPoints <= 0) || (settings.targetPoints == lastTotal);
		if(untouched && totalI > 0){
			settings.targetPoints = totalI;
		}
		lastTotal = totalI;
		if(total > 0.0){
			double d = double(settings.targetPoints) / total;
			settings.pointDensity = float(d < 0.001 ? 0.001 : (d > 1.0 ? 1.0 : d));
		}
	}
	uniforms.voxelGridPtr    = (uint64_t)g_voxelGrid;
	uniforms.gridX           = g_gridX;
	uniforms.gridY           = g_gridY;
	uniforms.gridZ           = g_gridZ;
	uniforms.cellSize        = g_cellSize;
	uniforms.showVoxelGrid   = settings.showVoxelGrid;
	uniforms.insectPlacement = settings.insectPlacement;
	uniforms.numNodesTotal   = int(stats.numNodes);
	uniforms.gridRefill = g_frameGridRefill ? 1 : 0;   // decided once per frame in beginFrameState
	uniforms.showEventMarkers = settings.showEventMarkers;
	uniforms.useVoxelGrid     = settings.useVoxelGrid;
	uniforms.pointDensity     = settings.pointDensity;
	uniforms.enableVolumetric = settings.enableVolumetric;
	uniforms.volDensity       = settings.volDensity;
	uniforms.volHeightFalloff = settings.volHeightFalloff;
	uniforms.volOcclusion     = settings.volOcclusion;
	uniforms.volSteps         = settings.volSteps;
	uniforms.volMaxDist       = settings.volMaxDist;
	uniforms.volScatter       = settings.volScatter;
	uniforms.volShadowSteps   = settings.volShadowSteps;
	uniforms.falloffPower     = settings.falloffPower;
	uniforms.coreBoost        = settings.coreBoost;
	uniforms.minNeighbours    = settings.minNeighbours;
	uniforms.cullReach        = settings.cullReach;
	uniforms.showCulledOnly   = settings.showCulledOnly;
	uniforms.saturation       = settings.saturation;
	uniforms.temperature      = settings.temperature;
	uniforms.tintR            = settings.tintColour[0];
	uniforms.tintG            = settings.tintColour[1];
	uniforms.tintB            = settings.tintColour[2];
	uniforms.tintMix          = settings.tintMix;
	uniforms.skyR             = settings.skyColour[0];
	uniforms.skyG             = settings.skyColour[1];
	uniforms.skyB             = settings.skyColour[2];
	uniforms.pointWorldSize   = settings.pointWorldSize;
	uniforms.pointShading     = settings.pointShading;
	uniforms.maxSplatPx       = effectiveMaxSplatPx();
	uniforms.grain            = settings.grain;
	uniforms.vignette         = settings.vignette;
	{
		double az = settings.lightAzimuth   * 3.14159265358979 / 180.0;
		double el = settings.lightElevation * 3.14159265358979 / 180.0;
		uniforms.lightDirX = float(cos(el) * sin(az));
		uniforms.lightDirY = float(cos(el) * cos(az));
		uniforms.lightDirZ = float(sin(el));
		uniforms.lightGridPtr   = (uint64_t)g_lightGrid;
		uniforms.lightR         = settings.lightColour[0];
		uniforms.lightG         = settings.lightColour[1];
		uniforms.lightB         = settings.lightColour[2];
		uniforms.lightIntensity = settings.lightIntensity;
		uniforms.ambient        = settings.ambient;
		uniforms.lightAbsorb    = settings.lightAbsorb;
		uniforms.lightSteps     = settings.lightSteps;
		uniforms.lightDirty     = g_frameLightDirty ? 1 : 0;   // decided once per frame in beginFrameState
	}
	uniforms.lightOn        = settings.lightOn ? 1 : 0;
	uniforms.nearClip       = settings.nearClip;
	uniforms.groundOn       = settings.groundOn ? 1 : 0;
	uniforms.groundHeight   = settings.groundHeight;
	uniforms.groundR        = settings.groundColour[0];
	uniforms.groundG        = settings.groundColour[1];
	uniforms.groundB        = settings.groundColour[2];
	uniforms.mixR           = settings.mixColour[0];
	uniforms.mixG           = settings.mixColour[1];
	uniforms.mixB           = settings.mixColour[2];
	{
		double fovy = 3.14159265358979 * renderer->camera->fovy / 180.0;
		uniforms.focalPx = float(double(renderer->height) / (2.0 * tan(fovy * 0.5)));
	}

		// boxMin is +infinity until a cloud is loaded; without this the event
		// positions become -infinity and every distance test returns NaN.
		bool haveCloud = std::isfinite(boxMin.x) && std::isfinite(boxMin.y) && std::isfinite(boxMin.z);

		if(g_callsLoaded && g_showPlaying && haveCloud){
			for(const ScheduledCall& c : g_calls){
				if(uniforms.numEvents >= MAX_EVENTS) break;
				// Attack, sustain, exponential decay.
				//
				// The old envelope was 1 - age/(duration+tail), squared, which
				// began fading the instant the call started: a 3 s call with a
				// 4 s tail was down to 56% before the call had even finished.
				// Now the glow holds for the length of the call and only then
				// decays, exponentially, reaching about 5% at the end of the
				// tail. That is also what light and sound actually do.
				double age = g_showClock - c.tStart;
				if(age < 0.0) continue;
				double tail = double(g_visualTail) > 1e-3 ? double(g_visualTail) : 1e-3;
				if(age > c.duration + tail) continue;

				const double ATTACK = 0.06;   // 60 ms rise, so onsets read sharply
				float env;
				if(age < ATTACK){
					env = float(age / ATTACK);
				}else if(age < c.duration){
					env = 1.0f;
				}else{
					env = float(exp(-3.0 * (age - c.duration) / tail));
				}
				float cr, cg, cb;
				if(settings.emissionByFrequency) freqToColour(c.freqHz, cr, cg, cb);
				else { cr = settings.emissionColour[0]; cg = settings.emissionColour[1]; cb = settings.emissionColour[2]; }
				int i = uniforms.numEvents++;
				// SimLOD translates every point by the original LAS minimum on
				// load (main_progressive_octree.cpp ~line 901), so the renderer's
				// world starts at the origin and uniforms.boxMin reads as zero.
				// Any external coordinate - a call position here, a TDOA fix or a
				// microphone location later - must carry the same offset or it
				// lands outside the cloud entirely.
				uniforms.eventX[i]  = c.x - boxMin.x;
				uniforms.eventY[i]  = c.y - boxMin.y;
				uniforms.eventZ[i]  = c.z - boxMin.z;
				uniforms.eventR[i]  = g_eventRadius;
				uniforms.eventI[i]  = env;   // already shaped above; no extra squaring
				uniforms.eventCR[i] = cr;
				uniforms.eventCG[i] = cg;
				uniforms.eventCB[i] = cb;
			}
			g_activeEvents = uniforms.numEvents;
			if(g_calls.size() > 0){
				// the latest end, not the last row: calls.csv need not be sorted
				double last = 0.0;
				for(const ScheduledCall& c : g_calls) last = std::max(last, c.tStart + c.duration);
				if(g_showClock > last + 2.0) g_showClock = 0.0;   // loop
			}
		}
	}

	// ---- DEBUG LOG (apply_debug_log.ps1) --------------------------------
	// Each line is written only when its text changes, and the fast-moving
	// ones (timings, camera, events) at most every 5 seconds.
	{
		logIfChanged("stage", 5.0, "[host] stage ms: setup %.1f  points %.1f  extras %.1f  edl %.1f  volumetric %.1f  bloom %.1f  composite %.1f  total %.1f  | visible %.1f M pts %.1f M voxels, drawn %.1f M\n",
			double(stats.stageNs[0])/1e6, double(stats.stageNs[1]-stats.stageNs[0])/1e6, double(stats.stageNs[2]-stats.stageNs[1])/1e6,
			double(stats.stageNs[3]-stats.stageNs[2])/1e6, double(stats.stageNs[4]-stats.stageNs[3])/1e6, double(stats.stageNs[5]-stats.stageNs[4])/1e6,
			double(stats.stageNs[6]-stats.stageNs[5])/1e6, double(stats.stageNs[6])/1e6,
			stats.numVisiblePoints / 1e6, stats.numVisibleVoxels / 1e6, stats.numDrawn / 1e6);
		logIfChanged("camera", 5.0, "[host] lens=%d fov=%.0f cam=(%.1f %.1f %.1f) pitch=%.0f radius=%.3f detail=%.0f splatCap=%d\n",
			settings.lensMode, settings.lensMode ? settings.fisheyeFov : settings.fovy,
			uniforms.camPos.x, uniforms.camPos.y, uniforms.camPos.z,
			renderer->controls->pitch * 57.2958, renderer->controls->radius, effectiveMinNodeSize(), effectiveMaxSplatPx());
		logIfChanged("events", 5.0, "[host] numEvents=%d  gain=%.3f  calls=%d  density=%.3f\n",
			uniforms.numEvents, uniforms.emissionGain, (int)g_calls.size(), settings.pointDensity);
		logIfChanged("box", 0.0, "[host] box min=(%.2f %.2f %.2f) max=(%.2f %.2f %.2f), load offset (subtracted from every point) = (%.2f %.2f %.2f)\n",
			uniforms.boxMin.x, uniforms.boxMin.y, uniforms.boxMin.z,
			uniforms.boxMax.x, uniforms.boxMax.y, uniforms.boxMax.z,
			boxMin.x, boxMin.y, boxMin.z);
		logIfChanged("grid", 0.0, "[host] voxel grid %d x %d x %d @ %.3f m, insects=%d placement=%d\n",
			g_gridX, g_gridY, g_gridZ, g_cellSize, settings.numInsects, settings.insectPlacement);
	}

	if(settings.markersOnly){   // apply_markers2.ps1: the array on its own
		uniforms.showPoints = 0;
		uniforms.numInsects = 0;
	}
	return uniforms;
}

void resetCUDA(shared_ptr<GLRenderer> renderer){

	ensureVoxelGrid();   // apply_voxel_grid.ps1: size and zero before the first batch

	// ensureVoxelGrid returns early when the dimensions are unchanged, which
	// they are when the same cloud is reloaded. The construct kernel adds to
	// the counts as points arrive, so without this every reset doubled them.
	if(g_voxelGrid != 0) cuMemsetD8(g_voxelGrid, 0, g_gridBytes);
	g_gridRefill = false;
	g_lightDirty = true;

	Uniforms uniforms = getUniforms(renderer);

	void* args[] = {
		&uniforms, 
		&cptr_buffer_persistent,
		&cptr_nodes,
		&cptr_stats,
		&cudaprint.cptr,
		&cptr_numBatchesUploaded,
		&cptr_batchSizes,
	};

	// we only need one single thread to do the resetting on device
	uint32_t numGroups = 1;
	uint32_t workgroupSize = 1;

	auto res_launch = cuLaunchCooperativeKernel(cuda_program_reset->kernels["kernel"],
		numGroups, 1, 1,
		workgroupSize, 1, 1,
		0, 0, args);

	if(res_launch != CUDA_SUCCESS){
		printfmt("CUDA kernel 'reset' failed.\n");
	}

	cuCheck(cuCtxSynchronize(), "cuCtxSynchronize");
}

// incrementally updates the octree on the GPU
void updateOctree(shared_ptr<GLRenderer> renderer){

	// cuCtxSynchronize();
	
	Uniforms uniforms = getUniforms(renderer);

	int workgroupSize = 256;
	int numGroups     = 1 * numSMs;
	auto ptrPoints    = cptr_points_ring[0];

	void* args[] = {
		&uniforms, &ptrPoints, 
		&cptr_buffer, &cptr_buffer_persistent,
		&cptr_nodes,
		&cptr_stats, &cptr_frameStart,
		&cudaprint.cptr,
		&cptr_numBatchesUploaded,
		&cptr_batchSizes,
		&cptr_batchTags,
	};

	// printfmt("launching update!\n");
	// printfmt("    frame:                   {} \n", renderer->frameCount);
	// printfmt("    points:                  {} \n", batchStreamPointSize[processRingIndex]);
	// printfmt("    uploadIndex:             {} \n", batchStreamUploadIndex);
	// printfmt("    uploadRingIndex:         {} \n", batchStreamUploadIndex % BATCH_STREAM_SIZE);
	// printfmt("    processIndex:            {} \n", batchStreamProcessIndex);
	// printfmt("    processRingIndex:        {} \n", processRingIndex);
	// printfmt("    cptr_points_ringbuffer:  {} \n", cptr_points_ringbuffer);
	// printfmt("    ptrPoints:               {} \n", ptrPoints);

	cuEventRecord(ce_update_start, 0);
	// printfmt("launch update \n");
	auto res_launch = cuLaunchCooperativeKernel(cuda_program_update->kernels["kernel_construct"],
		numGroups, 1, 1,
		workgroupSize, 1, 1,
		0, 0, args);
	// printfmt("update done\n");

	if(res_launch != CUDA_SUCCESS){
		const char* str; 
		cuGetErrorString(res_launch, &str);
		printf("error: %s \n", str);
	}

	cuEventRecord(ce_update_end, 0);

	// benchmark kernel- slows down overall loading!
	if(requestBenchmark){ 
		cuCheck(cuCtxSynchronize(), "cuCtxSynchronize");

		float duration;
		cuEventElapsedTime(&duration, ce_update_start, ce_update_end);

		kernelUpdateDuration    += duration;
		minKernelUpdateDuration = std::min(minKernelUpdateDuration, double(duration));
		maxKernelUpdateDuration = std::max(maxKernelUpdateDuration, double(duration));
		avgKernelUpdateDuration = (cntKernelUpdateDuration * avgKernelUpdateDuration + duration) / (cntKernelUpdateDuration + 1.0);
		cntKernelUpdateDuration += 1.0;
	}

	requestStep = false;
	numBatchesProcessed++;

	// cuCtxSynchronize();
}

// post-process color-filtering.
// computes average color values for voxels.
// void doColorFiltering(shared_ptr<GLRenderer> renderer){

// 	if(!lastBatchFinishedDevice) return;

// 	Uniforms uniforms = getUniforms(renderer);

// 	int workgroupSize = 256;
// 	int numGroups = numSMs;

// 	void* args[] = {
// 		&uniforms,
// 		&cptr_buffer, 
// 		&cptr_nodes, 
// 		&cptr_stats
// 	};

// 	printfmt("launching color filter!\n");

// 	auto res_launch = cuLaunchCooperativeKernel(cuda_program_filter->kernels["kernel"],
// 		numGroups, 1, 1,
// 		workgroupSize, 1, 1,
// 		0, 0, args);

// 	if(res_launch != CUDA_SUCCESS){
// 		const char* str; 
// 		cuGetErrorString(res_launch, &str);
// 		printf("error: %s \n", str);
// 	}

// 	requestColorFiltering = false;
// }

// draw the octree with a CUDA kernel
void renderCUDA(shared_ptr<GLRenderer> renderer){

	Uniforms uniforms = getUniforms(renderer);

	static bool registered = false;
	static GLuint registeredHandle = -1;

	cuGraphicsGLRegisterImage(
		&cugl_colorbuffer, 
		renderer->view.framebuffer->colorAttachments[0]->handle, 
		GL_TEXTURE_2D, 
		CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

	// map OpenGL resources to CUDA
	vector<CUgraphicsResource> dynamic_resources = {cugl_colorbuffer};
	cuGraphicsMapResources(static_cast<int>(dynamic_resources.size()), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));

	CUDA_RESOURCE_DESC res_desc = {};
	res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
	cuGraphicsSubResourceGetMappedArray(&res_desc.res.array.hArray, cugl_colorbuffer, 0, 0);
	CUsurfObject output_surf;
	cuSurfObjectCreate(&output_surf, &res_desc);

	cuEventRecord(ce_render_start, 0);

	float time = static_cast<float>(now());
	int workgroupSize = 256;
	
	int maxActiveBlocksPerSM;
	cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, 
		cuda_program_render->kernels["kernel_render"], workgroupSize, 0);
	
	int numGroups = maxActiveBlocksPerSM * numSMs;
	
	void* args[] = {
		&cptr_renderbuffer,
		&uniforms, 
		&cptr_nodes, 
		&output_surf,
		&cptr_stats,
		&cptr_frameStart,
		& cudaprint.cptr
	};

	
	auto res_launch = cuLaunchCooperativeKernel(cuda_program_render->kernels["kernel_render"],
		numGroups, 1, 1,
		workgroupSize, 1, 1,
		0, 0, args);

	if(res_launch != CUDA_SUCCESS){
		const char* str; 
		cuGetErrorString(res_launch, &str);
		printf("error: %s \n", str);
	}

	cuEventRecord(ce_render_end, 0);

	// benchmark kernel- slows down overall loading!
	if(requestBenchmark){ 
		cuCheck(cuCtxSynchronize(), "cuCtxSynchronize");

		float duration;
		cuEventElapsedTime(&duration, ce_render_start, ce_render_end);

		kernelRenderDuration    += duration;
		minKernelRenderDuration = std::min(minKernelRenderDuration, double(duration));
		maxKernelRenderDuration = std::max(maxKernelRenderDuration, double(duration));
		avgKernelRenderDuration = (cntKernelRenderDuration * avgKernelRenderDuration + duration) / (cntKernelRenderDuration + 1.0);
		cntKernelRenderDuration += 1.0;
	}

	if(settings.benchmarkRendering){
		cuCheck(cuCtxSynchronize(), "cuCtxSynchronize");
		cuEventElapsedTime(&renderingDuration, ce_render_start, ce_render_end);
	}

	cuSurfObjectDestroy(output_surf);
	cuGraphicsUnmapResources(static_cast<int>(dynamic_resources.size()), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));

	cuGraphicsUnregisterResource(cugl_colorbuffer);
}

// compile kernels and allocate buffers
void initCudaProgram(shared_ptr<GLRenderer> renderer){

	// allocate most gpu buffers
	uint64_t nodesCapacity           = 200'000;
	uint64_t estimatedNodeSize       = 200;           // see struct Node in progressive_octree.cu, but some more just in case
	// Raised from 300 MB by apply_buffer_fix.ps1. The Stats panel reported
	// "momentary buffer 408.8 MB" against a 300 MB allocation while displaying
	// a 54.5 M point scan - writing 109 MB past the end of a CUDA allocation,
	// which is exactly the sort of fault that shows up as an intermittent
	// crash when something shifts the allocation pattern, such as moving a
	// slider. The allocator has no bounds check at all.
	//
	// 1.2 GB leaves room for the emission, bloom chain and volumetric buffers
	// at up to two abutted 1440p screens. See the overrun warning below, which
	// now reports the condition instead of letting it corrupt silently.
	uint64_t cptr_buffer_bytes       = 1'200'000'000;
	uint64_t cptr_nodes_bytes        = nodesCapacity * estimatedNodeSize;
	uint64_t cptr_renderbuffer_bytes = 200'000'000;

	momentaryBufferCapacity = cptr_buffer_bytes;

	// What the device had free before any of this. Everything else that is
	// already on the GPU (another SimLOD, a browser) shrinks the octree pool.
	size_t freeAtStart = 0;
	size_t totalMem = 0;
	cuMemGetInfo(&freeAtStart, &totalMem);

	// Unchecked, a failed allocation left a null pointer that the kernels then
	// wrote through. Fail at startup with the reason in the log instead.
	cuCheck(cuMemAlloc(&cptr_buffer                , cptr_buffer_bytes), "cuMemAlloc momentary buffer");
	cuCheck(cuMemAlloc(&cptr_nodes                 , cptr_nodes_bytes), "cuMemAlloc nodes");
	cuCheck(cuMemAlloc(&cptr_renderbuffer          , cptr_renderbuffer_bytes), "cuMemAlloc renderbuffer");
	cuCheck(cuMemAlloc(&cptr_stats                 , sizeof(Stats)), "cuMemAlloc stats");
	cuCheck(cuMemAlloc(&cptr_numBatchesUploaded    , 4), "cuMemAlloc numBatchesUploaded");
	cuCheck(cuMemAlloc(&cptr_batchSizes            , 4 * BATCH_STREAM_SIZE), "cuMemAlloc batchSizes");
	cuCheck(cuMemAlloc(&cptr_batchTags             , 4 * BATCH_STREAM_SIZE), "cuMemAlloc batchTags");
	cuCheck(cuMemsetD32(cptr_batchTags, 0xFFFFFFFFu, BATCH_STREAM_SIZE), "cuMemsetD32 batchTags");
	cuCheck(cuMemAlloc(&cptr_frameStart            , 8), "cuMemAlloc frameStart");
	cuCheck(cuMemAllocHost((void**)&h_stats_pinned , sizeof(Stats)), "cuMemAllocHost stats");

	// allocate ring buffer for point upload
	uint64_t cptr_points_bytes = MAX_BATCH_SIZE * sizeof(Point);
	CUdeviceptr devicemem = 0;

	cuCheck(cuMemAlloc(&devicemem, BATCH_STREAM_SIZE * cptr_points_bytes), "cuMemAlloc point ring buffer");

	for(uint64_t i = 0; i < BATCH_STREAM_SIZE; i++){
		cptr_points_ring[i] = devicemem + i * cptr_points_bytes;
	}

	// allocate persistent (over multiple frames) buffer with remaining GPU memory
	size_t availableMem = 0;
	cuMemGetInfo(&availableMem, &totalMem);

	size_t cptr_buffer_persistent_bytes = static_cast<size_t>(static_cast<double>(availableMem) * 0.80);
	persistentBufferCapacity = cptr_buffer_persistent_bytes;
	cuCheck(cuMemAlloc(&cptr_buffer_persistent, cptr_buffer_persistent_bytes), "cuMemAlloc persistent buffer");

	uint64_t total = cptr_buffer_bytes 
		+ cptr_nodes_bytes
		+ cptr_renderbuffer_bytes
		+ sizeof(Stats)
		+ cptr_buffer_persistent_bytes;

	printfmt("cuMemAlloc(&cptr_buffer,            {:8L} MB);\n", cptr_buffer_bytes / 1'000'000llu);
	printfmt("cuMemAlloc(&cptr_nodes,             {:8L} MB);\n", cptr_nodes_bytes / 1'000'000llu);
	printfmt("cuMemAlloc(&cptr_renderbuffer,      {:8L} MB);\n", cptr_renderbuffer_bytes / 1'000'000llu);
	printfmt("cuMemAlloc(&cptr_stats,             {:8L}   );\n", sizeof(Stats));
	printfmt("cuMemAlloc(&cptr_buffer_persistent, {:8L} MB);\n", cptr_buffer_persistent_bytes / 1'000'000llu);
	printfmt("==============================================\n");
	printfmt("                                    {:8L} MB  \n", total / 1'000'000llu);
	printfmt("\n");

	// The same plan in the run log: stdout is lost when the process is killed,
	// and the octree pool decides how many points fit.
	logLine("[vram] %.0f of %.0f MB free at start. Fixed: construct scratch %.0f, nodes %.0f, render %.0f, upload ring %.0f MB."
		" Octree pool %.0f MB (80%% of the %.0f MB then free)\n",
		freeAtStart / 1e6, totalMem / 1e6,
		cptr_buffer_bytes / 1e6, cptr_nodes_bytes / 1e6, cptr_renderbuffer_bytes / 1e6,
		(BATCH_STREAM_SIZE * cptr_points_bytes) / 1e6,
		cptr_buffer_persistent_bytes / 1e6, availableMem / 1e6);

	cuda_program_update = new CudaModularProgram({
		.modules = {
			"./modules/progressive_octree/progressive_octree_voxels.cu",
			  //"./modules/progressive_octree/progressive_octree_mno.cu",
			"./modules/progressive_octree/utils.cu",
		},
		.kernels = {"kernel_construct"}
	});

	cuda_program_reset = new CudaModularProgram({
		.modules = {
			"./modules/progressive_octree/reset.cu",
			"./modules/progressive_octree/utils.cu",
		},
		.kernels = {"kernel"}
	});

	cuda_program_render = new CudaModularProgram({
		.modules = {
			"./modules/progressive_octree/render.cu",
			"./modules/progressive_octree/utils.cu",
		},
		.kernels = {"kernel_render"}
	});

	// cuda_program_filter = new CudaModularProgram({
	// 	.modules = {
	// 		"./modules/progressive_octree/colorfilter.cu",
	// 		"./modules/progressive_octree/utils.cu",
	// 	},
	// 	.kernels = {"kernel"}
	// });

	cuEventCreate(&ce_render_start, 0);
	cuEventCreate(&ce_render_end, 0);
	cuEventCreate(&ce_update_start, 0);
	cuEventCreate(&ce_update_end, 0);
	
	cuGraphicsGLRegisterImage(&cugl_colorbuffer, renderer->view.framebuffer->colorAttachments[0]->handle, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

	size_t freeAfterInit = 0;
	cuMemGetInfo(&freeAfterInit, &totalMem);
	logLine("[vram] %.0f MB free after allocating and compiling the kernels\n", freeAfterInit / 1e6);
}

void reload(){

	printfmt("start loading \n");

	loadStart = static_cast<float>(now());
	totalUpdateDuration     = 0.0f;
	kernelUpdateDuration    = 0.0f;
	minKernelUpdateDuration = Infinity;
	maxKernelUpdateDuration = 0.0;
	avgKernelUpdateDuration = 0.0;
	cntKernelUpdateDuration = 0.0;

	totalRenderDuration     = 0.0f;
	kernelRenderDuration    = 0.0f;
	minKernelRenderDuration = Infinity;
	maxKernelRenderDuration = 0.0;
	avgKernelRenderDuration = 0.0;
	cntKernelRenderDuration = 0.0;

	lock_guard<mutex> lock_batchesToProcess(mtx_batchesToProcess);
	lock_guard<mutex> lock_batchesInPinnedMemory(mtx_batchesInPinnedMemory);
	lock_guard<mutex> lock_batchesInPageableMemory(mtx_batchesInPageableMemory);
	lock_guard<mutex> lock_pinnedMemoryInUpload(mtx_pinnedMemoryInUpload);

	// reset workload and loaded data
	batchesToProcess.clear();
	batchesInPinnedMemory.clear();
	batchesInPageableMemory.clear();
	pinnedMemoryInUpload.clear();
	pinnedMemPool.refill();

	boxMin = float3{InfinityF, InfinityF, InfinityF};
	boxMax = float3{-InfinityF, -InfinityF, -InfinityF};
	boxSize = float3{0.0, 0.0, 0.0};

	numPointsTotal  = 0;
	numPointsLoaded = 0;
	numBatchesTotal = 0;
	numBytesTotal   = 0;
	numBytesLoaded  = 0;
	lazInBatches = false;

	// query bounding box and number of points in input files
	// vector<PointBatch> batches;
	for(string path : paths){

		if(!fs::exists(path)) continue;

		numBytesTotal += fs::file_size(path);

		if(iEndsWith(path, "laz")){
			lazInBatches = true;
		}

		if(iEndsWith(path, "las") || iEndsWith(path, "laz")){

			LasHeader header = loadHeader(path);
			numPointsTotal += header.numPoints;

			boxMin.x = std::min(boxMin.x, float(header.min[0]));
			boxMin.y = std::min(boxMin.y, float(header.min[1]));
			boxMin.z = std::min(boxMin.z, float(header.min[2]));

			boxMax.x = std::max(boxMax.x, float(header.max[0]));
			boxMax.y = std::max(boxMax.y, float(header.max[1]));
			boxMax.z = std::max(boxMax.z, float(header.max[2]));

			for (uint64_t first = 0; first < header.numPoints; first += MAX_BATCH_SIZE) {
				PointBatch batch;
				batch.file = path;
				batch.first = static_cast<int>(first);
				batch.count = static_cast<int>(std::min(header.numPoints - first, MAX_BATCH_SIZE));
				batch.lasHeader = header;

				batchesToProcess.push_back(batch);
				numBatchesTotal++;
			}

		}else if(iEndsWith(path, "simlod")){

			auto buffer = readBinaryFile(path, 0, 24);
			uint64_t numPoints = (fs::file_size(path) - 24) / 16;

			boxMin.x = std::min(boxMin.x, buffer->get<float>( 0));
			boxMin.y = std::min(boxMin.y, buffer->get<float>( 4));
			boxMin.z = std::min(boxMin.z, buffer->get<float>( 8));

			boxMax.x = std::max(boxMax.x, buffer->get<float>(12));
			boxMax.y = std::max(boxMax.y, buffer->get<float>(16));
			boxMax.z = std::max(boxMax.z, buffer->get<float>(20));

			numPointsTotal += numPoints;

			for(uint64_t first = 0; first < numPoints; first += MAX_BATCH_SIZE){
				PointBatch batch;
				batch.file = path;
				batch.first = static_cast<int>(first);
				batch.count = static_cast<int>(std::min(numPoints - first, MAX_BATCH_SIZE));

				batchesToProcess.push_back(batch);
				numBatchesTotal++;
			}
		}

		

		
	}

	// std::random_device rd;
	// std::mt19937 g(rd());
	// std::shuffle(loadQueue.begin(), loadQueue.end(), g);
	// std::reverse(loadQueue.begin(), loadQueue.end());

	// if(batches.size() > 0){
	// 	loadQueue.push_back(batches[batches.size() - 1]);
	// 	loadQueue.push_back(batches[batches.size() - 2]);
	// }

	boxSize.x = boxMax.x - boxMin.x;
	boxSize.y = boxMax.y - boxMin.y;
	boxSize.z = boxMax.z - boxMin.z;

	stats                   = Stats();
	batchesProcessedHost    = 0;
	pointsProcessedHost     = 0;
	numPointsUploaded       = 0;
	numBatchesProcessed     = 0;
	lastBatchFinishedDevice = false;
	loadStoppedAtCapacity   = false;
	batchStreamUploadIndex  = 0;
	processFrameTimes.clear();

	if(numPointsTotal > 0){
		logLine("[load] start: %.2f M points in %d batches, %.2f GB on disk%s\n",
			numPointsTotal / 1e6, numBatchesTotal.load(), numBytesTotal / 1e9,
			lazInBatches ? " (LAZ)" : "");
	}
}

// The construct kernel stops inserting once the octree pool is nearly full.
// The loaders went on decoding up to BATCH_STREAM_SIZE more batches that would
// never be used, each holding a 16 MB pinned slot: about 50 M points, minutes
// of LAZ decoding. Drop the queued work instead and mark the load complete.
void stopLoadingAtCapacity(){
	{
		lock_guard<mutex> lock_batchesToProcess(mtx_batchesToProcess);
		batchesToProcess.clear();
	}

	// same lock order as the uploader: mtx_uploader, then the pinned queue
	lock_guard<mutex> lock_upload(mtx_uploader);
	lock_guard<mutex> lock_batchesInPinnedMemory(mtx_batchesInPinnedMemory);

	for(PointBatch& batch : batchesInPinnedMemory){
		pinnedMemPool.release(batch.pinnedMem);
	}
	batchesInPinnedMemory.clear();

	// Uploader and loaders idle once every batch counts as uploaded. Batches a
	// loader is still decoding land in batchesInPinnedMemory and stay there
	// until the next reload refills the pool.
	numBatchesTotal = batchStreamUploadIndex.load();
	loadStoppedAtCapacity = true;
}

// One-off, after the last batch: everything the octree build produced, so
// capacity can be planned from measurements rather than estimates.
void logLoadSummary(){
	// synchronous on the default stream, so the counters are the final ones
	Stats s;
	if(cuMemcpyDtoH(&s, cptr_stats, sizeof(Stats)) != CUDA_SUCCESS) return;

	double seconds = now() - double(loadStart);
	double inserted = double(s.numPointsProcessed);

	logLine("[load] done in %.1f s: %.2f M of %.2f M points inserted (%.1f M/s), %.2f M decoded%s\n",
		seconds, inserted / 1e6, numPointsTotal / 1e6, inserted / 1e6 / std::max(seconds, 1e-3),
		numPointsLoaded / 1e6,
		loadStoppedAtCapacity ? ". Stopped: the octree pool is full, the remaining points are not shown" : "");
	logLine("[load] octree: %u nodes (%u inner, %u leaves, %u non-empty), %.2f M points in leaves, %.2f M voxels (%.2f per point), %u point chunks, %u voxel chunks\n",
		s.numNodes, s.numInner, s.numLeaves, s.numNonemptyLeaves,
		s.numPoints / 1e6, s.numVoxels / 1e6, double(s.numVoxels) / std::max(double(s.numPoints), 1.0),
		s.numChunksPoints, s.numChunksVoxels);
	logLine("[load] memory: octree pool %.0f of %.0f MB used, %.1f bytes per point; construct scratch %.0f of %.0f MB\n",
		s.allocatedBytes_persistent / 1e6, persistentBufferCapacity / 1e6,
		double(s.allocatedBytes_persistent) / std::max(double(s.numPoints), 1.0),
		s.allocatedBytes_momentary / 1e6, momentaryBufferCapacity / 1e6);
	logLine("[load] integrity: %u points dropped (no chunk), %u upload-ring slots refilled while still in use\n",
		s.numDroppedPoints, s.numRingMismatches);
}

void reset(shared_ptr<GLRenderer> renderer){
	// before locking, notify threads that they should not try to acquire the lock (otherwise we might wait for a long time to lock the others out)
	resetInProgress = true;

	// lock out upload threads and loader threads
	for (size_t i = 0; i < mtx_loader.size(); ++i) {
		mtx_loader[i]->lock();
	}
	lock_guard<mutex> lock_upload(mtx_uploader);

	// finish pending upload
	cuCheck(cuCtxSynchronize(), "cuCtxSynchronize");

	// now reset octree-construction related state on device
	resetCUDA(renderer);
	cuCheck(cuCtxSynchronize(), "cuCtxSynchronize");

	// read stats just to make sure device and host are on the same page
	// (sync before the memcpy, or this copies the previous, stale contents)
	cuMemcpyDtoHAsync(h_stats_pinned, cptr_stats, sizeof(Stats), ((CUstream)CU_STREAM_DEFAULT));
	cuCheck(cuCtxSynchronize(), "cuCtxSynchronize");
	memcpy(&stats, h_stats_pinned, sizeof(Stats));

	// finally, reset host state to start reloading current data set from disk
	reload();

	requestReset = false;

	// notify other threads that they may now try to acquire the lock again
	resetInProgress = false;

	// unlock loader threads
	for (size_t i = 0; i < mtx_loader.size(); ++i) {
		mtx_loader[i]->unlock();
	}
}

void spawnLoader(size_t i, int maxLazThreads) {

	// Captures by value: the thread is detached and outlives this call. It used
	// to capture a local CpuData by reference, so the LAZ thread cap was read
	// from a dead stack slot and could be anything.
	thread t([i, maxLazThreads]() {
		installThreadTerminateHandler();
		cuCtxSetCurrent(context);

		while (true) {
			bool everythingIsDone = batchStreamUploadIndex == numBatchesTotal;
			bool processingLagsBehind = numPointsLoaded > pointsProcessedHost + BATCH_STREAM_SIZE * MAX_BATCH_SIZE;

			// if (processingLagsBehind) {
			// 	printfmt("processing lags behind\n");
			// }

			if (everythingIsDone || processingLagsBehind || resetInProgress.load()) {
				std::this_thread::sleep_for(1ms);
				continue;
			}

			// lock thread's own mutex
			lock_guard<mutex> lock_loader(*mtx_loader[i]);

			PointBatch batch;

			mtx_batchesToProcess.lock();
			// get batch to load, if load queue isn't full
			if (batchesToProcess.size() > 0) {
				const PointBatch& next = batchesToProcess.front();

				// LAZ decoding is CPU-bound, so cap the threads loading at once.
				// Tested and counted under this lock, so the cap is exact. Capped
				// threads used to retry without sleeping and kept this mutex busy
				// for the threads that were decoding.
				if(next.count > 0 && iEndsWith(next.file, "laz") && numThreadsLoading >= uint64_t(maxLazThreads)){
					mtx_batchesToProcess.unlock();
					std::this_thread::sleep_for(1ms);
					continue;
				}

				batch = next;
				batchesToProcess.pop_front();

				if(batch.count > 0) numThreadsLoading++;
			}
			mtx_batchesToProcess.unlock();

			if (batch.count > 0) {
				// load points in batch

				// Taken only with work in hand. Idle loaders took and returned a
				// slot every millisecond, and allocated a new one whenever the
				// pool happened to be empty.
				PinnedMemorySlot pinnedMemSlot = pinnedMemPool.acquire();
				Point* pinnedPoints = (Point*)pinnedMemSlot.memLocation;

				int batchID = batch.first / MAX_BATCH_SIZE;
				double t_start = now();
				// printfmt("start loading batch {} at {:.3f} \n", batchID, t_start);

				if(iEndsWith(batch.file, "las")){
					void* target = (void*)pinnedPoints;

					double translation[3] = {-boxMin.x, -boxMin.y, -boxMin.z};
					loadLasNative(batch.file, batch.lasHeader, batch.first, batch.count, target, translation);
					numBytesLoaded += batch.count * batch.lasHeader.bytesPerPoint;

					batch.pinnedMem = pinnedMemSlot;

					numPointsLoaded += batch.count;

					lock_guard<mutex> lock_batchesInPinnedMemory(mtx_batchesInPinnedMemory);
					batchesInPinnedMemory.push_back(batch);

				}else if(iEndsWith(batch.file, "laz")){
					laszip_POINTER laszip_reader = nullptr;
					laszip_header* header = nullptr;
					laszip_point* laz_point = nullptr;

					laszip_BOOL is_compressed;
					laszip_BOOL request_reader = true;

					// One reader per batch, destroyed at the end: it was created and
					// never destroyed, leaking a reader for every batch of every load.
					if(laszip_create(&laszip_reader) != 0 || laszip_reader == nullptr){
						logLine("[load] laszip_create failed for %s\n", batch.file.c_str());
						pinnedMemPool.release(pinnedMemSlot);
						numThreadsLoading--;
						continue;
					}
					laszip_request_compatibility_mode(laszip_reader, request_reader);
					if(laszip_open_reader(laszip_reader, batch.file.c_str(), &is_compressed) != 0){
						laszip_CHAR* err = nullptr;
						laszip_get_error(laszip_reader, &err);
						logLine("[load] cannot open %s: %s\n", batch.file.c_str(), err ? err : "unknown laszip error");
						laszip_destroy(laszip_reader);
						pinnedMemPool.release(pinnedMemSlot);
						numThreadsLoading--;
						continue;
					}

					laszip_get_header_pointer(laszip_reader, &header);
					laszip_get_point_pointer(laszip_reader, &laz_point);
					laszip_seek_point(laszip_reader, batch.first);

					for (int i = 0; i < batch.count; i++) {
						double XYZ[3];
						laszip_read_point(laszip_reader);
						laszip_get_coordinates(laszip_reader, XYZ);

						Point point;
						point.x = static_cast<float>(XYZ[0] - boxMin.x);
						point.y = static_cast<float>(XYZ[1] - boxMin.y);
						point.z = static_cast<float>(XYZ[2] - boxMin.z);

						auto rgb = laz_point->rgb;
						point.rgba[0] = rgb[0] > 255 ? rgb[0] / 256 : rgb[0];
						point.rgba[1] = rgb[1] > 255 ? rgb[1] / 256 : rgb[1];
						point.rgba[2] = rgb[2] > 255 ? rgb[2] / 256 : rgb[2];
						
						//int intensity = laz_point->intensity;
						//point.rgba[0] = intensity / 200;
						//point.rgba[1] = intensity / 200;
						//point.rgba[2] = intensity / 200;


						pinnedPoints[i] = point;
					}

					laszip_close_reader(laszip_reader);
					laszip_destroy(laszip_reader);

					batch.pinnedMem = pinnedMemSlot;

					numPointsLoaded += batch.count;

					lock_guard<mutex> lock_batchesInPinnedMemory(mtx_batchesInPinnedMemory);
					batchesInPinnedMemory.push_back(batch);
				}else if(iEndsWith(batch.file, "simlod")){
					
					batch.pinnedMem = pinnedMemSlot;

					// At least on windows, this uses winapi to do unbuffered loading to maximize SSD perf
					uint64_t padding;
					loadFileNative(batch.file, 24llu + 16llu * uint64_t(batch.first), 16llu * batch.count, pinnedMemSlot.memLocation, &padding);
					batch.pinnedMem.memOffset = padding;

					numBytesLoaded += 16llu * batch.count;
					numPointsLoaded += batch.count;

					lock_guard<mutex> lock_batchesInPinnedMemory(mtx_batchesInPinnedMemory);
					batchesInPinnedMemory.push_back(batch);
				}

				// double t_end = now();
				// double millies = (t_end - t_start) * 1000.0;
				// printfmt("finished loading batch {} at {:.3f}. duration: {:.3f} ms \n", batchID, t_end, millies);

				numThreadsLoading--;
			}

			std::this_thread::sleep_for(1ms);
		}

		});
	t.detach();
}

// Spawns a thread that uploads data to the GPU
// - Asynchronously schedules uploads in stream_upload
// - After every async copy, it also asynchronously updates the number of uploaded points
void spawnUploader(shared_ptr<GLRenderer> renderer) {
	double timestamp = now();

	thread t([&]() {
		installThreadTerminateHandler();

		vector<PinnedMemorySlot> availableSlots;

		while (true) {

			double t_start = now();
			double spintime = 0.0001;

			// actually seems to wait way, way longer than 100ns.
			// std::this_thread::sleep_for(100ns);

			// go to sleep if reset is in progress
			if (resetInProgress.load()) {
				std::this_thread::sleep_for(1ms);
				continue;
			}

			// this lock ensures that we don't reset and upload at the same time
			lock_guard<mutex> lock_uploader(mtx_uploader);

			// spin instead of sleep, because sleep takes too long
			while (now() < t_start + spintime) {
				// do some spinning to avoid attempting locks too often
			}

			if(requestStepthrough){
				if(requestStep){
					requestStep = false;
				}else{
					continue;
				}
			}

			// reclaim all pinned memory slots that are no longer needed and give them back to the pool
			mtx_pinnedMemoryInUpload.lock();
			while (!pinnedMemoryInUpload.empty() && cuEventQuery(pinnedMemoryInUpload.front().uploadEnd) == cudaSuccess) {
				availableSlots.push_back(pinnedMemoryInUpload.front());
				pinnedMemoryInUpload.pop_front();
			}
			mtx_pinnedMemoryInUpload.unlock();

			pinnedMemPool.release(availableSlots);
			availableSlots.clear();

			bool everythingIsDone = batchStreamUploadIndex == numBatchesTotal;

			// Batch k goes to ring slot k % BATCH_STREAM_SIZE, which last held
			// batch k - BATCH_STREAM_SIZE. That slot may be refilled only once the
			// kernel has finished its batch. batchletIndex counts finished batches
			// and the host copy lags, so it can only under-report; the margin of
			// two covers a copy read while it was being written. The old test
			// compared point counts with '>', which with full batches let batch
			// k + BATCH_STREAM_SIZE overwrite batch k before the kernel read it.
			bool ringFull = int64_t(batchStreamUploadIndex) >= int64_t(batchesProcessedHost) + int64_t(BATCH_STREAM_SIZE) - 2;

			if (everythingIsDone) continue;
			if (ringFull) continue;

			auto t_00 = now();

			// acquire work, or keep spinning if there is none
			PointBatch batch;
			{
				lock_guard<mutex> lock_batchesInPinnedMemory(mtx_batchesInPinnedMemory);

				if (batchesInPinnedMemory.size() > 0) {
					batch = batchesInPinnedMemory.front();
					batchesInPinnedMemory.pop_front();
				} else {
					continue;
				}
			}

			// UPLOAD
			uint32_t targetSlot = batchStreamUploadIndex;
			int uploadRingIndex = targetSlot % BATCH_STREAM_SIZE;

			auto source     = ((uint8_t*)batch.pinnedMem.memLocation) + batch.pinnedMem.memOffset;
			auto target     = cptr_points_ring[uploadRingIndex];
			size_t byteSize = batch.count * sizeof(Point);

			// Tag the slot as in transit, then with its batch index once the data
			// is in. kernel_construct counts a slot whose tag is not the batch it
			// expects (Stats::numRingMismatches): the tripwire for the gate above.
			cuMemsetD32Async(cptr_batchTags + 4 * uploadRingIndex, 0xFFFFFFFFu, 1, stream_upload);

			cuMemcpyHtoDAsync(target, source, byteSize, stream_upload);

			// record upload event
			cuEventRecord(batch.pinnedMem.uploadEnd, stream_upload);

			// since we process N batches per frame and batches may have varying amounts of points,
			// we need to let the kernel know the size of each individual batch
			cuMemsetD32Async(cptr_batchSizes + 4 * uploadRingIndex, batch.count, 1, stream_upload);
			cuMemsetD32Async(cptr_batchTags + 4 * uploadRingIndex, targetSlot, 1, stream_upload);

			// also let the kernel know how many batches we uploaded
			cuMemsetD32Async(cptr_numBatchesUploaded, batchStreamUploadIndex + 1, 1, stream_upload);

			batchStreamUploadIndex++;
			numPointsUploaded += batch.count;

			lock_guard<mutex> lock_pinnedMemoryInUpload(mtx_pinnedMemoryInUpload);
			pinnedMemoryInUpload.push_back(batch.pinnedMem);
		}

		});
	t.detach();

	setThreadPriorityHigh(t);
}

int main(int argc, char** argv){

	// ---- last-words handlers ---------------------------------------------
	// A hard crash on Windows kills the process silently. The structured-
	// exception filter (CrashHandler.cpp) catches access violations, divide
	// by zero and the like on every thread and writes a symbolised stack
	// trace to the run log. The old SIGSEGV handler only covered the main
	// thread and could not say where the fault was.
	logLine("[host] SimLOD starting\n");   // opens the run log, so its path is known
	installCrashHandler(g_logPath.c_str());
	// (abort() and uncaught exceptions are handled there too, with a stack)

	auto renderer = make_shared<GLRenderer>();
	auto cpu = getCpuData();
	// One loader per hardware thread (was two). LAZ decoding is capped lower,
	// leaving cores for the render loop, the uploader and the OS.
	int numThreads    = std::max(4, static_cast<int>(cpu.numProcessors));
	int maxLazThreads = std::max(1, static_cast<int>(cpu.numProcessors) - 4);
	logLine("[load] %d loader threads, at most %d decoding LAZ at once (%d hardware threads)\n",
		numThreads, maxLazThreads, static_cast<int>(cpu.numProcessors));


	renderer->controls->yaw    = -1.15;
	renderer->controls->pitch  = -0.57;
	renderer->controls->radius = sqrt(boxSize.x * boxSize.x + boxSize.y * boxSize.y + boxSize.z * boxSize.z);
	renderer->controls->target = {
		boxSize.x * 0.5f,
		boxSize.y * 0.5f,
		boxSize.z * 0.1f
	};

	// renderer->controls->yaw    = 0.982;
	// renderer->controls->pitch  = -0.875;
	// renderer->controls->radius = 449.807;
	// renderer->controls->target = { 1154.460, 218.177, -92.225, };

	// renderer->controls->yaw    = 7.670;
	// renderer->controls->pitch  = -0.677;
	// renderer->controls->radius = 929.239;
	// renderer->controls->target = { 606.560, 385.040, 13.848, };

	// position: 448.8209204653559, 768.7683535080489, 23.676426584479366 
	// renderer->controls->yaw    = -4.660;
	// renderer->controls->pitch  = -0.293;
	// renderer->controls->radius = 94.341;
	// renderer->controls->target = { 354.609, 764.038, 25.101, };


	initCuda();
	initCudaProgram(renderer);

	pinnedMemPool.reserveSlots(PINNED_MEM_POOL_SIZE);

	mtx_loader.reserve(numThreads);
	for(int i = 0; i < numThreads; i++){
		mtx_loader.push_back(make_unique<mutex>());
		spawnLoader(i, maxLazThreads);
	}

	cudaprint.init();
	
	spawnUploader(renderer);

	// Command line: SimLOD.exe [cloud.las|.laz|.simlod ...] [--view yaw pitch radius tx ty tz]
	// Files load as if dropped on the window; --view sets the camera afterwards
	// (target in the renderer frame, i.e. relative to the cloud's minimum corner),
	// which makes timings and screenshots reproducible.
	bool haveView = false;
	double view[6] = {0, 0, 0, 0, 0, 0};
	vector<string> argFiles;
	for(int i = 1; i < argc; i++){
		string arg = argv[i];
		if(arg == "--view" && i + 6 < argc){
			for(int k = 0; k < 6; k++) view[k] = std::atof(argv[i + 1 + k]);
			haveView = true;
			i += 6;
		}else if(arg == "--set" && i + 1 < argc){
			// --set name=value, for the settings that matter most on a small GPU
			string kv = argv[++i];
			size_t eq = kv.find('=');
			string key = kv.substr(0, eq);
			double v = (eq == string::npos) ? 1.0 : std::atof(kv.c_str() + eq + 1);
			if     (key == "hqs")      settings.useHighQualityShading = v != 0.0;
			else if(key == "splat")    settings.maxSplatPx            = int(v);
			else if(key == "detail")   settings.minNodeSize           = float(v);
			else if(key == "adaptive") settings.adaptiveDetail        = v != 0.0;
			else if(key == "budget")   settings.frameBudgetMs         = float(v);
			else if(key == "vsync")    settings.vsync                 = v != 0.0;
			else printfmt("unknown --set key: {} (hqs, splat, detail, adaptive, budget, vsync)\n", key);
		}else if(iEndsWith(arg, "las") || iEndsWith(arg, "laz") || iEndsWith(arg, "simlod")){
			std::error_code ec;
			auto absolute = std::filesystem::absolute(arg, ec);   // dropped files are absolute too
			argFiles.push_back(ec ? arg : absolute.string());
		}else{
			printfmt("ignoring command-line argument: {}\n", arg);
		}
	}
	if(argc > 1){
		logLine("[host] command line: %d file(s)%s\n", int(argFiles.size()), haveView ? ", --view" : "");
		for(auto& f : argFiles) logLine("[host]   %s\n", f.c_str());
	}

	reload();

	// Load command-line files the way a drop does: through reset() on the
	// first frame, which locks the loader threads out first. Calling reload()
	// here with real files deadlocked against the already-running loaders.
	bool startCameraPending = false;
	if(!argFiles.empty()){
		paths = argFiles;   // replaces the "NONE" placeholder
		requestReset = true;
		startCameraPending = true;
	}
	auto applyStartCamera = [&](){
		renderer->controls->yaw    = -1.15;
		renderer->controls->pitch  = -0.57;
		renderer->controls->radius = sqrt(boxSize.x * boxSize.x + boxSize.y * boxSize.y + boxSize.z * boxSize.z);
		renderer->controls->target = { boxSize.x * 0.5f, boxSize.y * 0.5f, boxSize.z * 0.1f };
		if(haveView){
			renderer->controls->yaw    = view[0];
			renderer->controls->pitch  = view[1];
			renderer->controls->radius = view[2];
			renderer->controls->target = { view[3], view[4], view[5] };
		}
	};
	if(argFiles.empty() && haveView) applyStartCamera();

	renderer->onFileDrop([&](vector<string> files){
		vector<string> pointCloudFiles;

		t_drop_start = now();
		printfmt("drop at {:.3f} \n", now());

		for(auto file : files){
			logLine("[load] dropped: %s\n", file.c_str());

			if(iEndsWith(file, "las") || iEndsWith(file, "laz")){
				pointCloudFiles.push_back(file);
			}else if(iEndsWith(file, "simlod")){
				pointCloudFiles.push_back(file);
			}
		}

		paths = pointCloudFiles;

		reset(renderer);

		if(settings.autoFocusOnLoad){
			renderer->controls->yaw = -1.15;
			renderer->controls->pitch = -0.57;
			renderer->controls->radius = sqrt(boxSize.x * boxSize.x + boxSize.y * boxSize.y + boxSize.z * boxSize.z);
			renderer->controls->target = {
				boxSize.x * 0.5f,
				boxSize.y * 0.5f,
				boxSize.z * 0.1f
			};
		}
	});

	auto update = [&](){
		cudaprint.update();

		renderer->camera->fovy = settings.fovy;
		renderer->camera->update();
	};

	auto render = [&](){

		timeSinceLastFrame = static_cast<float>(now()) - lastFrameTime;

		// ---- walking (apply_navigate.ps1) --------------------------------
		// W/S forward and back along the view, A/D sideways, Q/E down and up.
		// Only when the GUI does not have the keyboard. Speed in metres per
		// second, so it is the same at any frame rate.
		{
			ImGuiIO& io = ImGui::GetIO();
			if(!io.WantCaptureKeyboard){
				float dt = timeSinceLastFrame > 0.0f && timeSinceLastFrame < 0.5f ? timeSinceLastFrame : 0.016f;
				double d = double(g_walkSpeed) * double(dt);
				auto held = [](int key){ return Runtime::keyStates[key] > 0; };
				double fwd = 0.0, side = 0.0, up = 0.0;
				if(held(87)) fwd  += d;   // W
				if(held(83)) fwd  -= d;   // S
				if(held(68)) side += d;   // D
				if(held(65)) side -= d;   // A
				if(held(69)) up   += d;   // E
				if(held(81)) up   -= d;   // Q
				if(fwd != 0.0 || side != 0.0 || up != 0.0){
					renderer->controls->translate_local(side, up, fwd);   // fix_walk_axis
				if(renderer->controls->walkMode) renderer->controls->radius = 0.001;   // keep the camera at the target
				}
			}
		}

		// ---- picking (apply_navigate.ps1) --------------------------------
		// Alt + left click asks the kernel for the world position under the
		// cursor; the answer arrives in Stats next frame and becomes a marker.
		// Alt already suppresses orbiting in OrbitControls, so the click does
		// not also drag the view.
		{
			bool altDown  = glfwGetKey(renderer->window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS || glfwGetKey(renderer->window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;   // fix_pick_input
			bool leftDown = glfwGetMouseButton(renderer->window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;   // fix_pick_input
			ImGuiIO& io = ImGui::GetIO();
			bool pKey = glfwGetKey(renderer->window, GLFW_KEY_P) == GLFW_PRESS && !io.WantCaptureKeyboard;
			if(((altDown && leftDown && !g_leftWasDown) || (pKey && !g_pKeyWasDown)) && !io.WantCaptureMouse){
				double mx=0.0, my=0.0; glfwGetCursorPos(renderer->window,&mx,&my);
				g_pickReqX = int(mx);
				g_pickReqY = renderer->height - 1 - int(my);
				g_pickSeq++;
				g_pickSeenDone = 0;
				g_pickWaiting = true;
				logLine("[pick] requested at %d %d  alt=%d left=%d p=%d\n", g_pickReqX, g_pickReqY, int(altDown), int(leftDown), int(pKey));
			}
			g_leftWasDown = leftDown;
			g_pKeyWasDown = pKey;

			// Stats arrives by an unsynchronised async copy, so one read can be
			// stale or half-written. The kernel leaves an answered request in
			// place, so requiring the answer on two consecutive reads means the
			// second read is a complete copy of settled values.
			bool answered = false;
			if(g_pickWaiting && stats.pickSeqDone == g_pickSeq){
				answered = (++g_pickSeenDone >= 2);
			}else{
				g_pickSeenDone = 0;
			}
			if(answered){
				g_pickReqX = -1; g_pickReqY = -1;
			}
			if(answered && !stats.pickValid){
				g_pickWaiting = false;
				logLine("[pick] nothing under that pixel - sky, or beyond the drawn points\n");
				notice("Nothing under that pixel to place a marker on");
			}

			if(answered && stats.pickValid){
				if(g_markers.size() < 32){   // apply_markers2.ps1
					g_markers.push_back(makeMarker(g_nextMic++, stats.pickWX, stats.pickWY, stats.pickWZ));
					glm::dvec3 cpos = renderer->camera->position;
					double ddx = double(stats.pickWX) - cpos.x, ddy = double(stats.pickWY) - cpos.y, ddz = double(stats.pickWZ) - cpos.z;
					double dist = std::sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
					logLine("[pick] marker %d at renderer (%.3f %.3f %.3f) = file (%.3f %.3f %.3f)  %.2f m from the camera\n",
						g_markers.back().mic, stats.pickWX, stats.pickWY, stats.pickWZ,
						double(stats.pickWX) + boxMin.x, double(stats.pickWY) + boxMin.y, double(stats.pickWZ) + boxMin.z, dist);
					notice("Marker %d placed at file (%.2f, %.2f, %.2f), %.1f m away%s",
						g_markers.back().mic, double(stats.pickWX) + boxMin.x, double(stats.pickWY) + boxMin.y, double(stats.pickWZ) + boxMin.z, dist,
						dist < 1.0 ? "  - under 1 m: probably foliage at the near clip, not the board" : "");
				}else{
					notice("Marker limit of 32 reached");
				}
				g_pickWaiting = false;
			}
		}
		lastFrameTime = static_cast<float>(now());

		renderer->view.framebuffer->setSize(renderer->width, renderer->height);

		glBindFramebuffer(GL_FRAMEBUFFER, renderer->view.framebuffer->handle);

		if(!lastBatchFinishedDevice){
			processFrameTimes.push_back(timeSinceLastFrame);
		}

		if(requestReset){
			reset(renderer);
			if(startCameraPending){ applyStartCamera(); startCameraPending = false; }   // boxSize is known now
		}

		{   // GLRenderer starts with vsync off (glfwSwapInterval(0)); follow the setting
			static int appliedSwapInterval = -1;
			int wanted = settings.vsync ? 1 : 0;
			if(wanted != appliedSwapInterval){ glfwSwapInterval(wanted); appliedSwapInterval = wanted; }
		}

		beginFrameState();
		renderCUDA(renderer);

		if(!lastBatchFinishedDevice){
			updateOctree(renderer);
		}

		// if(requestColorFiltering){
		// 	doColorFiltering(renderer);
		// }

		if(!lastBatchFinishedDevice){
			totalUpdateDuration = 1000.0f * (static_cast<float>(now()) - loadStart);
		}else if(requestBenchmark && lastBatchFinishedDevice){
			requestBenchmark = false;
			printfmt("finished loading, disabling benchmarking. \n");
		}

		static int statsAge = 0;
		{
			// copy stats from gpu to cpu.
			// actually laggs behind because we do async copy.
			// lacks sync, but as long as bytes are updated atomically in multiples of 4 or 8 bytes, 
			// results should be fine.

			// seems to be fine to add the async copy to the main stream?
			cuMemcpyDtoHAsync(h_stats_pinned, cptr_stats, sizeof(Stats), ((CUstream)CU_STREAM_DEFAULT));
			memcpy(&stats, h_stats_pinned, sizeof(Stats));
			batchesProcessedHost = stats.batchletIndex;
			pointsProcessedHost  = stats.numPointsProcessed;

			statsAge = static_cast<int>(renderer->frameCount) - stats.frameID;

			if(!lastBatchFinishedDevice && numPointsTotal > 0){
				logIfChanged("load", 2.0, "[load] %.1f s: %.1f M decoded, %.1f M uploaded, %.1f M of %.1f M inserted; octree pool %.0f of %.0f MB\n",
					now() - double(loadStart), numPointsLoaded / 1e6, numPointsUploaded / 1e6,
					stats.numPointsProcessed / 1e6, numPointsTotal / 1e6,
					stats.allocatedBytes_persistent / 1e6, persistentBufferCapacity / 1e6);
			}

			if(stats.memCapacityReached && !loadStoppedAtCapacity && !lastBatchFinishedDevice){
				stopLoadingAtCapacity();
			}

			static uint64_t previousNumPointsProcessed = 0;
			uint64_t numPointsProcessed = stats.numPointsProcessed;

			// if(numPointsProcessed != previousNumPointsProcessed){
			// 	printfmt("processed {} at {:.3f}. since drop: {:.3f} \n", numPointsProcessed, now(), now() - t_drop_start);

			// 	previousNumPointsProcessed = numPointsProcessed;
			// }
			

			bool newLastBatchFinishedDevice = stats.numPointsProcessed == uint64_t(numPointsTotal);
			if(stats.memCapacityReached){
				newLastBatchFinishedDevice = true;
			}
			if(newLastBatchFinishedDevice != lastBatchFinishedDevice){
				lastBatchFinishedDevice = newLastBatchFinishedDevice;
				// the light was built (if at all) from partial counts during the load
				if(lastBatchFinishedDevice) g_lightDirty = true;
				printfmt("stats.numPointsProcessed = {} \n", stats.numPointsProcessed);
				printfmt("numPointsTotal = {} \n", uint64_t(numPointsTotal));
				printfmt("setting lastBatchFinishedDevice = {} \n", lastBatchFinishedDevice ? "true" : "false");

				if(lastBatchFinishedDevice && numPointsTotal > 0) logLoadSummary();
			}
		}

		if(Runtime::showGUI)
		{ // RENDER IMGUI SETTINGS WINDOW

			// ---- movable, collapsible windows (apply_windows.ps1) -------------
			// SetNextWindowPos/Size with no ImGuiCond default to Always, so the
			// windows were re-pinned every frame and could not be moved or
			// resized. FirstUseEver places them once and then leaves them alone.
			//
			// Collapsed, an ImGui window keeps its full width. This shrinks it
			// to the title text instead, and restores the size it had when the
			// user expands it again.
			auto beginPanel = [](const char* title, ImVec2 pos, ImVec2 size,
								 bool& collapsedState, ImVec2& savedSize) -> bool {
				ImGui::SetNextWindowPos(pos, ImGuiCond_FirstUseEver);
				if(collapsedState){
					float w = ImGui::CalcTextSize(title).x
							+ ImGui::GetStyle().WindowPadding.x * 2.0f
							+ ImGui::GetFrameHeight()            // collapse arrow
							+ ImGui::GetFontSize();              // close button
					ImGui::SetNextWindowSize(ImVec2(w, 0.0f), ImGuiCond_Always);
				}else{
					ImGui::SetNextWindowSize(savedSize.x > 0.0f ? savedSize : size,
											 ImGuiCond_FirstUseEver);
				}
				bool open = ImGui::Begin(title);
				bool nowCollapsed = ImGui::IsWindowCollapsed();
				if(!nowCollapsed){
					ImVec2 s = ImGui::GetWindowSize();
					if(s.x > 0.0f) savedSize = s;
				}
				collapsedState = nowCollapsed;
				return open;
			};

			auto windowSize = ImVec2(490, 280);
			static bool   settingsCollapsed = false;
			static ImVec2 settingsSaved     = ImVec2(0, 0);
			beginPanel("Settings", ImVec2(10, 300), windowSize, settingsCollapsed, settingsSaved);

			// =====================================================================
			// Settings, regrouped (apply_regroup.ps1). Eight sections. Look is open
			// by default; the rest collapse. Every control appears exactly once.
			// =====================================================================

			// ---- LOOK ---------------------------------------------------------
			if(ImGui::CollapsingHeader("Look", ImGuiTreeNodeFlags_DefaultOpen)){
				ImGui::SliderFloat("Point size (m)",   &settings.pointWorldSize, 0.0f, 0.30f, "%.3f");
				tipFor("Splat diameter in metres, projected by depth. Near points large, far ones small.");
				ImGui::SliderFloat("Point shading",    &settings.pointShading,   0.0f, 1.0f);
				tipFor("Shades each splat as a small sphere lit from the upper left. 0 is a flat disc.");
				ImGui::SliderFloat("Near clip (m)",    &settings.nearClip,       0.0f, 10.0f, "%.2f");
				tipFor("Points closer to the camera than this are not drawn, so foliage right at the lens does not fill the frame.");
				ImGui::Separator();
				ImGui::SliderFloat("Exposure",         &settings.exposure,       0.1f, 4.0f);
				ImGui::SliderFloat("Saturation",       &settings.saturation,     0.0f, 2.5f);
				ImGui::SliderFloat("Temperature",      &settings.temperature,   -1.0f, 1.0f);
				tipFor("Negative is cool, positive warm.");
				// A colour picker clamps to 0..1, so it could only ever REDUCE a
				// channel. A mixer must boost as well. Three sliders, 1.0 neutral.
				ImGui::SliderFloat("Mix red",          &settings.mixColour[0],   0.0f, 2.5f);
				ImGui::SliderFloat("Mix green",        &settings.mixColour[1],   0.0f, 2.5f);
				ImGui::SliderFloat("Mix blue",         &settings.mixColour[2],   0.0f, 2.5f);
				ImGui::SameLine(0.0f);
				if(ImGui::SmallButton("reset##mix")){ settings.mixColour[0] = settings.mixColour[1] = settings.mixColour[2] = 1.0f; }
				tipFor("Per-channel gain in linear light. 1.0 is neutral. Below removes a cast, above adds one.");
				ImGui::Separator();
				ImGui::SliderFloat("Depth haze",       &settings.fogDensity,     0.0f, 0.08f, "%.3f");
				tipFor("Distance blends toward the haze colour. The reference goes LIGHTER with distance, so keep the colour pale.");
				if(settings.fogDensity <= 0.0f) settings.enableFog = false; else settings.enableFog = true;
				{
					float hc[3] = {settings.fogR, settings.fogG, settings.fogB};
					if(ImGui::ColorEdit3("Haze colour", hc, ImGuiColorEditFlags_NoInputs)){
						settings.fogR = hc[0]; settings.fogG = hc[1]; settings.fogB = hc[2];
					}
				}
				ImGui::ColorEdit3 ("Sky colour",       settings.skyColour, ImGuiColorEditFlags_NoInputs);
				ImGui::Separator();
				ImGui::ColorEdit3 ("Below horizon",    settings.belowColour, ImGuiColorEditFlags_NoInputs);
				tipFor("Colour of empty space when the camera looks downward, past the ground. Black, like the void under a game level. Points themselves keep their scanned colour. Set it to the SKY colour to remove the horizon entirely.");
				ImGui::SameLine(0.0f);
				if(ImGui::SmallButton("= sky")){
					settings.belowColour[0] = settings.skyColour[0];
					settings.belowColour[1] = settings.skyColour[1];
					settings.belowColour[2] = settings.skyColour[2];
				}
				ImGui::SliderFloat("Horizon softness", &settings.horizonSoftness, 0.0f, 90.0f, "%.0f deg");
				tipFor("Degrees the sky-to-void gradient spans. It was effectively 1 degree, which drew a hard line across the frame. 12 reads as atmosphere; 90 is a full gradient from zenith to nadir; 0 is a deliberate hard edge.");
				ImGui::Separator();
				ImGui::SliderFloat("Grain",            &settings.grain,          0.0f, 0.15f, "%.3f");
				ImGui::SliderFloat("Vignette",         &settings.vignette,       0.0f, 1.0f);
			}

			// ---- LIGHT --------------------------------------------------------
			if(ImGui::CollapsingHeader("Light")){
				bool lm = false;
				lm |= ImGui::Checkbox   ("Key light on",     &settings.lightOn);
				tipFor("A directional light through the canopy, computed once per grid cell. Needs Use voxel grid on (under Advanced).");
				lm |= ImGui::SliderFloat("Azimuth",          &settings.lightAzimuth,   0.0f, 360.0f);
				lm |= ImGui::SliderFloat("Elevation",        &settings.lightElevation, 0.0f,  89.0f);
				tipFor("Where the light comes FROM. The reference is backlit and high.");
				lm |= ImGui::SliderFloat("Strength",         &settings.lightIntensity, 0.0f,   2.5f);
				ImGui::ColorEdit3 ("Light colour",     settings.lightColour, ImGuiColorEditFlags_NoInputs);
				ImGui::SliderFloat("Ambient",          &settings.ambient,        0.0f, 1.0f);
				tipFor("Floor under the key light, so shadowed points keep some of their own colour.");
				lm |= ImGui::SliderFloat("Canopy density",   &settings.lightAbsorb,    0.0f,   1.5f, "%.2f /m");
				tipFor("Light lost per metre of occupied foliage, so it means the same at every grid resolution. 0.4 lets about a seventh through five metres of canopy; 1.0 almost none.");
				if(lm) g_lightDirty = true;
			}

			// ---- CALLS --------------------------------------------------------
			if(ImGui::CollapsingHeader("Calls")){
				ImGui::ColorEdit3 ("Emission colour",  settings.emissionColour, ImGuiColorEditFlags_NoInputs);
				ImGui::SameLine(0.0f);
				ImGui::Checkbox   ("by frequency",     &settings.emissionByFrequency);
				ImGui::SliderFloat("Emission gain",    &settings.emissionGain,   0.0f, 12.0f);
				ImGui::SliderFloat("Event radius (m)", &g_eventRadius,           0.5f, std::max(40.0f, boxSize.x > 0.0f ? std::max(boxSize.x, boxSize.y) * 0.5f : 40.0f));
				tipFor("How far a call reaches. Nothing outside this is lit at all.");
				ImGui::SliderFloat("Falloff power",    &settings.falloffPower,   0.5f,  8.0f);
				tipFor("1 is a broad ramp, 6 concentrates light at the centre with a long feathered tail.");
				ImGui::SliderFloat("Core boost",       &settings.coreBoost,      0.0f,  8.0f);
				ImGui::SliderFloat("Visual tail (s)",  &g_visualTail,            0.0f,  4.0f);
				tipFor("Seconds the glow outlasts the call, decaying exponentially.");
				ImGui::Separator();
				ImGui::SliderFloat("Bloom strength",   &settings.bloomStrength,  0.0f, 4.0f);
				ImGui::SliderFloat("Bloom spread",     &settings.bloomRadius,    0.0f, 5.0f);
				settings.enableBloom = settings.bloomStrength > 0.0f;
				ImGui::Separator();
				if(ImGui::Button("Load calls.csv")){
					const char* candidates[] = {
						"calls.csv",
						"../../../rotokare_test_kit/kit/data/calls.csv",
						"../rotokare_test_kit/kit/data/calls.csv",
						"S:/SimLodMod/rotokare_test_kit/kit/data/calls.csv",
					};
					for(const char* p : candidates){
						if(std::filesystem::exists(p)){ loadCallSchedule(p); break; }
					}
					if(!g_callsLoaded) printf("[calls] no calls.csv found in any candidate location\n");
					g_showClock = 0.0;
				}
				ImGui::SameLine(0.0f);
				if(ImGui::Button(g_showPlaying ? "Pause" : "Play")){ g_showPlaying = !g_showPlaying; }
				ImGui::SameLine(0.0f);
				if(ImGui::Button("Restart")){ g_showClock = 0.0; }
				ImGui::Text("clock %.1f s   active %d", g_showClock, g_activeEvents);
				ImGui::Checkbox   ("Show event markers", &settings.showEventMarkers);
			}

			// ---- CAMERA -------------------------------------------------------
			if(ImGui::CollapsingHeader("Camera")){
				ImGui::TextDisabled("Lens");
				ImGui::SameLine(0.0f);
				if(ImGui::Button("Normal 60"))   { settings.lensMode = 0; settings.fovy = 60.0f; }
				ImGui::SameLine(0.0f);
				if(ImGui::Button("Wide 90"))     { settings.lensMode = 0; settings.fovy = 90.0f; }
				ImGui::SameLine(0.0f);
				if(ImGui::Button("Fisheye 180")) { settings.lensMode = 1; settings.fisheyeFov = 180.0f; }
				if(settings.lensMode == 0){
					ImGui::SliderFloat("Field of view", &settings.fovy, 20.0f, 120.0f);
					tipFor("Vertical field of view. Above about 100 a perspective lens stretches the edges; switch to Fisheye for more.");
				}else{
					ImGui::SliderFloat("Fisheye angle",  &settings.fisheyeFov, 90.0f, 250.0f);
					tipFor("Angle spanned top to bottom of the frame. 180 puts the horizon at the frame edge with the camera looking straight up; wider sees below it. The sides of a 16:9 frame reach further still.");
				}
				ImGui::Separator();
				ImGui::SliderFloat("Eye height (m)",   &settings.eyeHeight,      0.2f, 10.0f, "%.2f");
				// live in both directions: dragging the slider tilts the camera now,
				// and tilting the camera with the mouse moves the slider
				if(ImGui::SliderFloat("Look up (deg)",    &settings.lookUpDeg,    -89.0f, 89.0f)){
					renderer->controls->pitch = double(settings.lookUpDeg) * 3.14159265358979 / 180.0;
				}else{
					settings.lookUpDeg = float(renderer->controls->pitch * 180.0 / 3.14159265358979);
				}
				ImGui::SliderFloat("Walk speed (m/s)", &g_walkSpeed, 0.2f, 15.0f);
				tipFor("W/S walk along the view, A/D sideways, Q/E down and up, when the mouse is over the forest rather than the panel. The wheel walks too when the camera is standing in the forest.");
				ImGui::Checkbox("Walk mode", &renderer->controls->walkMode);
				tipFor("On: the wheel and W/S walk. Off: the wheel zooms the orbit.");
				if(ImGui::Button("Go to centre")){
					// Camera at the centre of the cloud in plan, eye height above the
					// lowest point, pitched up. Orbit controls put the camera at
					// target - radius * dir, so the target is pushed one tiny radius
					// along the view direction from where the camera should be.
					float cx = boxSize.x * 0.5f;
					float cyc = boxSize.y * 0.5f;
					float cz = settings.eyeHeight;
					double pitch = double(settings.lookUpDeg) * 3.14159265358979 / 180.0;
					double yaw   = renderer->controls->yaw;
					double cosPitch = cos(pitch), sinPitch = sin(pitch);
					double cosYaw   = cos(yaw),   sinYaw   = sin(yaw);
					double dx = -sinYaw * cosPitch, dy = cosYaw * cosPitch, dz = sinPitch;
					double r = 0.001;
					renderer->controls->pitch  = pitch;
					renderer->controls->radius = r;
					renderer->controls->target = glm::dvec3(double(cx) + dx * r, double(cyc) + dy * r, double(cz) + dz * r);
					renderer->controls->walkMode = true;   // fix_walk_targets.ps1
				}
				ImGui::SameLine(0.0f);
				if(ImGui::Button("Copy camera")){
					auto controls = renderer->controls;
					auto pos = controls->getPosition();
					auto target = controls->target;
					stringstream ss;
					ss << std::setprecision(2) << std::fixed;
					ss << format("// position: {}, {}, {} \n", pos.x, pos.y, pos.z);
					ss << format("renderer->controls->yaw    = {:.3f};\n", controls->yaw);
					ss << format("renderer->controls->pitch  = {:.3f};\n", controls->pitch);
					ss << format("renderer->controls->radius = {:.3f};\n", controls->radius);
					ss << format("renderer->controls->target = {{ {:.3f}, {:.3f}, {:.3f}, }};\n", target.x, target.y, target.z);
					string str = ss.str();
#ifdef _WIN32
					toClipboard(str);
#endif
				}
				ImGui::Checkbox   ("Auto-focus on load", &settings.autoFocusOnLoad);
			}

			// ---- DETAIL AND SPEED ---------------------------------------------
			if(ImGui::CollapsingHeader("Detail and speed")){
				{
					auto withCommas = [](int64_t v){
						std::string s = std::to_string(v < 0 ? -v : v);
						int n = int(s.size()) - 3;
						while(n > 0){ s.insert(n, ","); n -= 3; }
						return (v < 0 ? "-" : "") + s;
					};
					int64_t total = int64_t(stats.numPoints);
					ImGui::SetNextItemWidth(160.0f);
					bool edited = ImGui::InputScalar("##targetPoints", ImGuiDataType_S64, &settings.targetPoints,
													 nullptr, nullptr, "%lld", ImGuiInputTextFlags_EnterReturnsTrue);
					if(edited){
						if(settings.targetPoints < 1) settings.targetPoints = 1;
						if(total > 0 && settings.targetPoints > total) settings.targetPoints = total;
					}
					ImGui::SameLine(0.0f);
					ImGui::Text("/ %s loaded", withCommas(total).c_str());
				}
				if(ImGui::SliderFloat("Point density", &settings.pointDensity, 0.001f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic)){
					settings.targetPoints = int64_t(double(settings.pointDensity) * double(stats.numPoints) + 0.5);
					if(settings.targetPoints < 1) settings.targetPoints = 1;
				}
				tipFor("Fraction of the cloud traversed. The skipped points are never read, so this scales the main cost directly.");
				ImGui::Text("   %.2f M actually drawn this frame", stats.numDrawn / 1e6);
				ImGui::SliderFloat("Detail",           &settings.minNodeSize,    8.0f, 256.0f);
				tipFor("Level-of-detail threshold in pixels. Higher draws coarser summaries sooner and costs less.");
				ImGui::Checkbox("Adaptive detail", &settings.adaptiveDetail);
				tipFor("Coarsens the level of detail automatically while a frame takes longer than the budget, and refines it again when there is headroom. Keeps the view responsive on smaller GPUs and over Remote Desktop.");
				if(settings.adaptiveDetail){
					ImGui::SameLine(0.0f);
					ImGui::SetNextItemWidth(110.0f);
					ImGui::SliderFloat("Budget ms", &settings.frameBudgetMs, 10.0f, 100.0f, "%.0f");
					tipFor("Render time to aim for. 33 ms is 30 frames a second; raise it for more detail, lower it for smoother motion.");
					ImGui::Text("   reduced %.0f%%: detail %.0f px, splat cap %d px", 100.0f * g_adaptReduce, effectiveMinNodeSize(), effectiveMaxSplatPx());
				}
				ImGui::Checkbox("VSync", &settings.vsync);
				tipFor("Caps the frame rate at the display refresh. Saves GPU power and heat, which matters on a laptop GPU; turn it off only to measure raw speed.");
				ImGui::SliderInt  ("Max splat px",     &settings.maxSplatPx,     2, 64);
				tipFor("Cap on how large a near point may be drawn. Lower if the frame rate drops when the camera is close to foliage.");
			}

			// ---- CLEANUP ------------------------------------------------------
			if(ImGui::CollapsingHeader("Cleanup")){
				ImGui::SliderInt  ("Min neighbours",   &settings.minNeighbours, 0, 60);
				tipFor("Removes points whose grid cell holds fewer than this many. 0 is off. Scanner noise sits alone; leaves do not.");
				ImGui::SliderInt  ("Cull reach",       &settings.cullReach,     0, 1);
				tipFor("0 counts the point's own cell. 1 adds the six neighbours, which spares thin surfaces on a cell boundary.");
				ImGui::Checkbox   ("Show culled only", &settings.showCulledOnly);
				tipFor("Shows ONLY what would be removed. Raise Min neighbours until this is noise and nothing you want.");
				ImGui::Separator();
				// Edit a copy and apply it on release: every applied value frees and
				// reallocates the grid (an implicit device sync) and zeroes the counts,
				// which happened on every frame of a drag.
				static int pendingGridRes = g_gridResolution;
				if(!ImGui::IsAnyItemActive()) pendingGridRes = g_gridResolution;   // follow fallbacks
				ImGui::SliderInt  ("Grid resolution",  &pendingGridRes, 64, 512);
				if(ImGui::IsItemDeactivatedAfterEdit()) g_gridResolution = pendingGridRes;
				tipFor("Cells along the cloud's longest side. Applied when you release the slider. Memory grows with the cube: 256 is ~20 MB, 512 ~156 MB.");
				{
					float longest = std::max(boxSize.x, std::max(boxSize.y, boxSize.z));
					if(longest > 0.0f){
						int R = pendingGridRes;
						float cell = longest / float(R);
						int px = std::min(R, int(boxSize.x / cell) + 1);
						int py = std::min(R, int(boxSize.y / cell) + 1);
						int pz = std::min(R, int(boxSize.z / cell) + 1);
						double mb = double(px) * double(py) * double(pz) * 4.0 / 1e6;
						ImGui::Text("   %d x %d x %d @ %.3f m = %.0f MB", px, py, pz, cell, mb);
					}
					if(g_gridRefill) ImGui::Text("   counts are stale - rebuilding on the next frame");
				}
				if(ImGui::Button("Rebuild grid counts")){ g_gridRefill = true; }
			}

			// ---- MARKERS ------------------------------------------------------
			if(ImGui::CollapsingHeader("Markers")){
				ImGui::Checkbox("Find targets", &settings.findTargets);
				tipFor("Highlights points that are ACHROMATIC - red, green and blue nearly equal. A survey checkerboard is black, grey and white; foliage is strongly green and bark red-brown. This finds a board in shadow, which a brightness test cannot.");
				ImGui::SameLine(0.0f);
				ImGui::Checkbox("Isolate", &settings.findIsolate);
				tipFor("Hides everything that is NOT a match. The fastest way to find the boards: tick this, and whatever is left on screen is what the test is catching.");
				ImGui::SliderFloat("Greyness", &settings.findGrey, 2.0f, 90.0f, "%.0f");
				tipFor("How far apart the three channels may be and still count as grey. Start at 30. Raise it until the boards appear; lower it until the foliage goes.");
				ImGui::SliderFloat("Min brightness", &settings.findThreshold, 0.0f, 250.0f, "%.0f");
				tipFor("Floor, so near-black noise is not counted as grey. Drop it to about 40 to catch the dark squares of a board as well as the white ones.");
				ImGui::SliderInt("Boost size (px)", &settings.findBoost, 0, 20);
				tipFor("Draws matches larger. A 15 cm board at 30 m is under a pixel, so without this the targets are there but invisible.");
				ImGui::Text("   %u points match this frame", stats.numTargets);
				ImGui::SliderFloat("Marker size (m)", &settings.markerScale, 0.05f, 1.0f, "%.2f");
				ImGui::Separator();
				// ---- apply_markers2.ps1 ------------------------------------------
				ImGui::Checkbox("Show markers", &settings.showMarkers);
				ImGui::SameLine(0.0f);
				ImGui::Checkbox("Markers only", &settings.markersOnly);
				tipFor("Markers only hides the forest and the insects so the array can be seen on its own. Nothing is discarded.");
				ImGui::TextWrapped("Hover a checkerboard centre and press P. A magenta ball marks it. Coordinates below are FILE coordinates: the LAS frame, the one the mic survey must be matched to.");
				if(!g_markers.empty()) ImGui::Separator();
				int removeIdx = -1;
				for(size_t i = 0; i < g_markers.size(); i++){
					Marker& mk = g_markers[i];
					ImGui::PushID(int(i));
					ImGui::SetNextItemWidth(110.0f);
					ImGui::InputText("##label", mk.label, sizeof(mk.label));
					ImGui::SameLine(0.0f);
					ImGui::SetNextItemWidth(42.0f);
					ImGui::InputInt("##mic", &mk.mic, 0, 0);
					ImGui::SameLine(0.0f);
					if(ImGui::SmallButton("go")){
						renderer->controls->target = glm::dvec3(mk.x, mk.y, mk.z);
						renderer->controls->radius = 3.0;
						renderer->controls->walkMode = false;
					}
					ImGui::SameLine(0.0f);
					if(ImGui::SmallButton("x")) removeIdx = int(i);
					ImGui::Text("   file    %9.3f %9.3f %9.3f", double(mk.x) + boxMin.x, double(mk.y) + boxMin.y, double(mk.z) + boxMin.z);
					float sv[3] = { mk.sx, mk.sy, mk.sz };
					ImGui::Text("   survey"); ImGui::SameLine(0.0f);
					ImGui::SetNextItemWidth(230.0f);
					if(ImGui::InputFloat3("##survey", sv, "%.3f")){
						mk.sx = sv[0]; mk.sy = sv[1]; mk.sz = sv[2]; mk.hasSurvey = true;
					}
					if(!mk.hasSurvey){ ImGui::SameLine(0.0f); ImGui::TextDisabled("(not set)"); }
					if(g_surveyValid && mk.hasSurvey){
						double px, py, pz;
						fileToSurvey(double(mk.x) + boxMin.x, double(mk.y) + boxMin.y, double(mk.z) + boxMin.z, px, py, pz);
						double ex = px - mk.sx, ey = py - mk.sy, ez = pz - mk.sz;
						ImGui::Text("   fitted  %9.3f %9.3f %9.3f   residual %.3f m", px, py, pz, std::sqrt(ex*ex + ey*ey + ez*ez));
					}
					ImGui::PopID();
				}
				if(removeIdx >= 0) g_markers.erase(g_markers.begin() + removeIdx);
				ImGui::Separator();
				if(ImGui::Button("Save markers.csv")) saveMarkersCsv();
				ImGui::SameLine(0.0f);
				if(ImGui::Button("Load markers.csv")) loadMarkersCsv();
				ImGui::SameLine(0.0f);
				if(ImGui::Button("Clear all")){ g_markers.clear(); g_nextMic = 1; g_surveyValid = false; }
				ImGui::Separator();
				ImGui::TextUnformatted("Survey frame");
				tipFor("Type each board's known survey coordinates into its row, then fit. Three boards are the minimum; every extra one averages the pick error down. The result is written to survey_transform.txt and the log.");
				if(ImGui::Button("Fit survey frame")) fitSurveyFrame();
				ImGui::SameLine(0.0f);
				if(g_surveyValid){
					double yawDeg = std::atan2(g_surveyR[3], g_surveyR[0]) * 180.0 / 3.14159265358979;
					ImGui::Text("%d pairs, yaw %.2f deg, rms %.3f m", g_surveyPairs, yawDeg, g_surveyRms);
					ImGui::Text("   t = (%.3f, %.3f, %.3f)", g_surveyT[0], g_surveyT[1], g_surveyT[2]);
				}else{
					ImGui::TextDisabled("not fitted");
				}
				if(ImGui::Checkbox("calls.csv is in survey coordinates", &g_callsInSurvey)){
					if(g_callsLoaded) notice("Reload calls.csv for this to take effect");
				}
				tipFor("On: call positions are converted from the survey frame into the file frame when calls.csv is loaded, using the fit above. Off: calls.csv is taken to be in file (LAS) coordinates already.");
			}

			// ---- INSECTS ------------------------------------------------------
			if(ImGui::CollapsingHeader("Insects")){
				ImGui::SliderInt  ("Count",            &settings.numInsects,   0, 30000);
				ImGui::SliderInt  ("Size (px)",        &settings.insectSize,   1,     4);
				ImGui::SliderFloat("Glow",             &settings.insectGlow,   0.0f, 6.0f);
				ImGui::SliderFloat("Speed",            &settings.insectSpeed,  0.0f, 2.0f);
				ImGui::SliderInt  ("Placement",        &settings.insectPlacement, 0, 1);
				tipFor("1 keeps only insects in air near foliage, using the grid. 0 scatters them anywhere.");
			}

			// ---- ADVANCED -----------------------------------------------------
			if(ImGui::CollapsingHeader("Advanced")){
				ImGui::TextDisabled("Volumetric medium (expensive, not in the reference)");
				ImGui::Checkbox   ("Volumetric medium", &settings.enableVolumetric);
				ImGui::SliderFloat("Medium density",    &settings.volDensity,       0.0f,  0.12f);
				ImGui::SliderFloat("Medium scatter",    &settings.volScatter,       0.0f,  6.0f);
				ImGui::SliderFloat("Medium pools low",  &settings.volHeightFalloff, 0.0f,  0.05f);
				ImGui::SliderFloat("Medium occlusion",  &settings.volOcclusion,     0.0f,  3.0f);
				ImGui::SliderInt  ("Medium steps",      &settings.volSteps,         8,     128);
				ImGui::SliderInt  ("Medium shadow steps", &settings.volShadowSteps, 0,    16);
				ImGui::SliderFloat("Medium reach (m)",  &settings.volMaxDist,      20.0f, std::max(400.0f, boxSize.x > 0.0f ? 2.0f * std::sqrt(boxSize.x*boxSize.x + boxSize.y*boxSize.y + boxSize.z*boxSize.z) : 400.0f));
				ImGui::Separator();
				ImGui::TextDisabled("Test source");
				ImGui::Checkbox   ("Test source on",    &settings.testSourceOn);
				{
					float fxMin = std::isfinite(boxMin.x) ? boxMin.x : -50.0f, fxMax = std::isfinite(boxMin.x) ? boxMin.x + boxSize.x : 50.0f;
					float fyMin = std::isfinite(boxMin.y) ? boxMin.y : -50.0f, fyMax = std::isfinite(boxMin.y) ? boxMin.y + boxSize.y : 50.0f;
					float fzMin = std::isfinite(boxMin.z) ? boxMin.z :   0.0f, fzMax = std::isfinite(boxMin.z) ? boxMin.z + boxSize.z : 40.0f;
					ImGui::SliderFloat("Test X", &settings.testX, fxMin, fxMax);
					ImGui::SliderFloat("Test Y", &settings.testY, fyMin, fyMax);
					ImGui::SliderFloat("Test Z", &settings.testZ, fzMin, fzMax);
				}
				ImGui::SliderFloat("Test intensity",    &settings.testIntensity, 0.0f, 3.0f);
				ImGui::SliderFloat("Test Hz",           &settings.testHz, 200.0f, 400.0f);
				ImGui::Separator();
				ImGui::TextDisabled("Grading extras");
				ImGui::ColorEdit3 ("Tint colour",       settings.tintColour, ImGuiColorEditFlags_NoInputs);
				ImGui::SliderFloat("Tint mix",          &settings.tintMix,     0.0f,  1.0f);
				ImGui::Separator();
				ImGui::TextDisabled("Rendering");
				ImGui::Checkbox   ("Eye dome lighting", &settings.enableEDL);
				ImGui::SliderFloat("EDL strength",      &settings.edlStrength, 0.0f, 3.0f);
				ImGui::Checkbox   ("Round splats",      &settings.roundSplats);
				ImGui::Checkbox   ("High quality shading", &settings.useHighQualityShading);
				ImGui::Separator();
				ImGui::TextDisabled("Grid and debug");
				ImGui::Checkbox   ("Use voxel grid",    &settings.useVoxelGrid);
				ImGui::Checkbox   ("Show voxel grid",   &settings.showVoxelGrid);
				ImGui::Checkbox   ("Show bounding box", &settings.showBoundingBox);
				ImGui::Checkbox   ("Update visibility", &settings.doUpdateVisibility);
				ImGui::Checkbox   ("Show points",       &settings.showPoints);
				ImGui::Checkbox   ("Colour by node",    &settings.colorByNode);
				ImGui::Checkbox   ("Colour by LOD",     &settings.colorByLOD);
			}

			// if(ImGui::Button("Do Color Filtering!")){
			// 	requestColorFiltering = true;
			// }

			ImGui::End();
		}
		
		if(Runtime::showGUI)
		{ // RENDER IMGUI STATS WINDOW

			auto windowSize = ImVec2(490, 440);
			static bool   statsCollapsed = false;
			static ImVec2 statsSaved     = ImVec2(0, 0);
			ImGui::SetNextWindowPos(ImVec2(10, 590), ImGuiCond_FirstUseEver);
			if(statsCollapsed){
				float w = ImGui::CalcTextSize("Stats").x
						+ ImGui::GetStyle().WindowPadding.x * 2.0f
						+ ImGui::GetFrameHeight() + ImGui::GetFontSize();
				ImGui::SetNextWindowSize(ImVec2(w, 0.0f), ImGuiCond_Always);
			}else{
				ImGui::SetNextWindowSize(statsSaved.x > 0.0f ? statsSaved : windowSize,
										 ImGuiCond_FirstUseEver);
			}
			ImGui::Begin("Stats");
			{
				bool nc = ImGui::IsWindowCollapsed();
				if(!nc){ ImVec2 s = ImGui::GetWindowSize(); if(s.x > 0.0f) statsSaved = s; }
				statsCollapsed = nc;
			}

			{ // used/total mem progress
				size_t availableMem = 0;
				size_t totalMem = 0;
				cuMemGetInfo(&availableMem, &totalMem);
				size_t unavailableMem = totalMem - availableMem;

				string strProgress = format("{:3.1f} / {:3.1f}", 
					double(unavailableMem) / 1'000'000'000.0, 
					double(totalMem) / 1'000'000'000.0
				);
				float progress = static_cast<float>(static_cast<double>(unavailableMem) / static_cast<double>(totalMem));
				ImGui::ProgressBar(progress, ImVec2(0.f, 0.f), strProgress.c_str());
				ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
				ImGui::Text("Used GPU Memory");
			}

			{ // loaded/total points

				string strProgress = format("{} M / {} M", 
					numPointsLoaded / 1'000'000,
					numPointsTotal / 1'000'000
				);
				float progress = static_cast<float>(static_cast<double>(numPointsLoaded) / static_cast<double>(numPointsTotal));
				ImGui::ProgressBar(progress, ImVec2(0.f, 0.f), strProgress.c_str());
				ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
				ImGui::Text("Loaded Points");
			}

			{ // loaded/total points

				string strProgress = format("{} M / {} M", 
					stats.numPointsProcessed / 1'000'000, 
					numPointsTotal / 1'000'000
				);
				float progress = static_cast<float>(static_cast<double>(stats.numPointsProcessed) / static_cast<double>(numPointsTotal));
				ImGui::ProgressBar(progress, ImVec2(0.f, 0.f), strProgress.c_str());
				ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
				ImGui::Text("Processed Points");
			}

			auto locale = getSaneLocale();
			uint32_t numEmptyLeaves = stats.numLeaves - stats.numNonemptyLeaves;

			// apply_stage_timers.ps1: stage i is the interval from boundary i-1
			// to boundary i; stage 0 starts at kernel entry; -1 is the total.
			auto stageMs = [](int i) -> double {
				const uint64_t* s = stats.stageNs;
				if(i < 0) return double(s[6]) / 1e6;
				uint64_t a = (i == 0) ? 0ull : s[i - 1];
				uint64_t b = s[i];
				return b > a ? double(b - a) / 1e6 : 0.0;
			};
			auto toMS = [locale](double millies){
				string str = "-";

				if(millies > 0.0){
					str = format("{:.1Lf} ms", millies);
				}

				return leftPad(str, 15);
			};

			auto toM = [locale](double number){
				string str = format(locale, "{:.1Lf} M", number / 1'000'000.0);
				return leftPad(str, 14);
			};

			auto toB = [locale](double number) {
				string str = format(locale, "{:.1Lf} B", number / 1'000'000'000.0);
				return leftPad(str, 14);
			};

			auto toMB = [locale](double number){
				string str = format(locale, "{:.1Lf} MB", number / 1'000'000.0);
				return leftPad(str, 15);
			};
			auto toGB = [locale](double number){
				string str = format(locale, "{:.1Lf} GB", number / 1'000'000'000.0);
				return leftPad(str, 15);
			};

			auto toIntString = [locale](double number){
				string str = format(locale, "{:L}", number);
				return leftPad(str, 10);
			};

			double pointsSecUpdate = double(stats.numPointsProcessed) / (double(kernelUpdateDuration) / 1000.0);
			double pointsSecTotal = double(stats.numPointsProcessed) / (double(totalUpdateDuration) / 1000.0);
			double gbs_file = double(numBytesLoaded) / (totalUpdateDuration / 1000.0);
			double gbs_gpu = double(16llu * stats.numPointsProcessed) / (totalUpdateDuration / 1000.0);

			double millionPointsSecRendered = 1000.0 * float(stats.numVisiblePoints) / renderingDuration;
			double millionVoxelsSecRendered = 1000.0 * float(stats.numVisibleVoxels) / renderingDuration;
			double millionSamplesSecRendered = 1000.0 * (float(stats.numVisiblePoints + stats.numVisibleVoxels) / renderingDuration);

			if(!settings.benchmarkRendering){
				millionPointsSecRendered = 0.0;
				millionVoxelsSecRendered = 0.0;
				millionSamplesSecRendered = 0.0;
				renderingDuration = 0.0;
			}

			double M = 1'000'000.0;
			double B = 1'000'000'000.0;
			double MB = 1'000'000.0; // TIL: MB = 1'000'000 vs. MiB = 1024 * 1024
			double GB = 1'000'000'000.0;

			vector<vector<string>> table = {
				{"#update kernel duration  ", toMS(kernelUpdateDuration)                            , format("{:.1f}", kernelUpdateDuration)},
				{"    max                  ", toMS(maxKernelUpdateDuration)                         , format("{:.1f}", maxKernelUpdateDuration)},
				{"    avg                  ", toMS(avgKernelUpdateDuration)                         , format("{:.1f}", avgKernelUpdateDuration)},
				{"#update total duration   ", toMS(totalUpdateDuration)                             , format("{:.1f}", totalUpdateDuration)},
				{"points/sec update kernel ", toM(pointsSecUpdate)                                  , format("{:.1f}", totalUpdateDuration / M)},
				{"points/sec total         ", toM(pointsSecTotal)                                   , format("{:.1f}", pointsSecTotal / M)},
				{"GB/s (disk I/O)          ", toGB(gbs_file)                                        , format("{:.1f}", gbs_file / GB)},
				{"GB/s (gpu)               ", toGB(gbs_gpu)                                         , format("{:.1f}", gbs_gpu / GB)},
				{"=========================", " "                                                   , " "},
				{"#render kernel duration  ", toMS(kernelRenderDuration)                            , format("{:.1f}", kernelRenderDuration)},
				{"=========================", " "                                                   , " "},
				{"rendering duration       ", toMS(renderingDuration)                               , format("{:.1f}", renderingDuration)},
				{"    points / sec         ", toB(millionPointsSecRendered)                        , format("{:.1f}", millionPointsSecRendered / B)},
				{"    voxels / sec         ", toB(millionVoxelsSecRendered)                        , format("{:.1f}", millionVoxelsSecRendered / B)},
				{"    samples / sec        ", toB(millionSamplesSecRendered)                       , format("{:.1f}", millionSamplesSecRendered / B)},
				{"=========================", " "                                                   , " "},
				{"#points processed        ", toM(double(stats.numPointsProcessed))                 , format("{:.1f}", stats.numPointsProcessed / M)},
				{"#nodes                   ", toIntString(stats.numNodes)                           , format("{}", stats.numNodes)},
				{"    #inner               ", toIntString(stats.numInner)                           , format("{}", stats.numInner)},
				{"    #leaves (nonempty)   ", toIntString(stats.numNonemptyLeaves)                  , format("{}", stats.numNonemptyLeaves)},
				{"    #leaves (empty)      ", toIntString(numEmptyLeaves)                           , format("{}", numEmptyLeaves)},
				{"#chunks                  ", toIntString(stats.numNodes)                           , format("{}", stats.numNodes)},
				{"    #voxels              ", toIntString(stats.numChunksVoxels)                    , format("{}", stats.numChunksVoxels)},
				{"    #points              ", toIntString(stats.numChunksPoints)                    , format("{}", stats.numChunksPoints)},
				{"#samples                 ", toM(stats.numPoints + stats.numVoxels)                , format("{:.1f}", (stats.numPoints + stats.numVoxels) / M)},
				{"    #points              ", toM(stats.numPoints)                                  , format("{:.1f}", stats.numPoints / M)},
				{"    #voxels              ", toM(stats.numVoxels)                                  , format("{:.1f}", stats.numVoxels / M)},
				{"#drawn after density     ", toM(stats.numDrawn)                                   , format("{:.1f}", stats.numDrawn / M)},
				{"=== stage ms (GPU) ======", " "                                                   , " "},
				{"  setup + visibility     ", toMS(stageMs(0))                                      , format("{:.2f}", stageMs(0))},
				{"  points                 ", toMS(stageMs(1))                                      , format("{:.2f}", stageMs(1))},
				{"  grid/markers/insects   ", toMS(stageMs(2))                                      , format("{:.2f}", stageMs(2))},
				{"  EDL                    ", toMS(stageMs(3))                                      , format("{:.2f}", stageMs(3))},
				{"  volumetric             ", toMS(stageMs(4))                                      , format("{:.2f}", stageMs(4))},
				{"  bloom                  ", toMS(stageMs(5))                                      , format("{:.2f}", stageMs(5))},
				{"  composite              ", toMS(stageMs(6))                                      , format("{:.2f}", stageMs(6))},
				{"  kernel total           ", toMS(stageMs(-1))                                     , format("{:.2f}", stageMs(-1))},
				{"momentary buffer         ", toMB(stats.allocatedBytes_momentary)                  , format("{:.1f}", stats.allocatedBytes_momentary / MB)},
				{"    capacity             ", toMB(momentaryBufferCapacity)                         , format("{:.1f}", momentaryBufferCapacity / MB)},
				{"persistent buffer        ", toMB(stats.allocatedBytes_persistent)                 , format("{:.1f}", stats.allocatedBytes_persistent / MB)},
				{"=========================", " "                                                   , " "},
				{"#visible nodes           ", toIntString(stats.numVisibleNodes)                    , format("{}", stats.numVisibleNodes)},
				{"    #inner               ", toIntString(stats.numVisibleInner)                    , format("{}", stats.numVisibleInner)},
				{"    #leaves              ", toIntString(stats.numVisibleLeaves)                   , format("{}", stats.numVisibleLeaves)},
				{"#visible samples         ", toM(stats.numVisiblePoints + stats.numVisibleVoxels)  , format("{:.1f}", (stats.numVisiblePoints + stats.numVisibleVoxels) / 1'000'000.0f)},
				{"    #points              ", toM(stats.numVisiblePoints)                           , format("{:.1f}", stats.numVisiblePoints / 1'000'000.0f)},
				{"    #voxels              ", toM(stats.numVisibleVoxels)                           , format("{:.1f}", stats.numVisibleVoxels / 1'000'000.0f)},
			};

			if(ImGui::Button("Copy Stats")){
				stringstream ss;
				for (int row = 0; row < table.size(); row++){
					for (int column = 0; column < 2; column++){
						ss << table[row][column];
					}
					ss << "\n";
				}


				string str = ss.str();
				toClipboard(str);
			}
			
			
			auto flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV;
			if (ImGui::BeginTable("table1", 3, flags)){
				ImGui::TableSetupColumn("AAA", ImGuiTableColumnFlags_WidthStretch);
				ImGui::TableSetupColumn("BBB", ImGuiTableColumnFlags_WidthStretch);
				ImGui::TableSetupColumn("CCC", ImGuiTableColumnFlags_WidthFixed);
				for (int row = 0; row < table.size(); row++){
					ImGui::TableNextRow();
					for (int column = 0; column < 2; column++){
						ImGui::TableSetColumnIndex(column);
						
						ImGui::Text(table[row][column].c_str());
					}

					ImGui::PushID(row);

					ImGui::TableSetColumnIndex(2);
					if (ImGui::SmallButton("c")) {
						string str = table[row][2];
						toClipboard(str);
					}

					ImGui::PopID();
				}
				ImGui::EndTable();
			}
			

			ImGui::End();
		}

		// On-screen notice (markers, survey fit, file errors). It used to sit
		// inside the warning block below, so it only showed while a LAZ file
		// was loading or memory was full.
		if(now() < g_noticeUntil){
			ImGuiIO& io = ImGui::GetIO();
			ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, 40.0f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
			ImGui::SetNextWindowBgAlpha(0.80f);
			ImGui::Begin("##pickNotice", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs
				| ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing);
			ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 60, 255, 255));
			ImGui::TextUnformatted(g_noticeText);
			ImGui::PopStyleColor();
			ImGui::End();
		}

		if(stats.memCapacityReached || (lazInBatches && !lastBatchFinishedDevice))
		{
			ImGuiIO& io = ImGui::GetIO();
			ImGui::SetNextWindowSize(ImVec2(700, 100));
			// ImGui::SetNextWindowPos(ImVec2(300, 200));
			ImGui::SetNextWindowPos(
				ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y - 100),
				ImGuiCond_Always, ImVec2(0.5f,0.5f)
			);

			ImGui::Begin("Message");

			ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));

			if(stats.memCapacityReached){
#ifdef __cpp_lib_format
        string message = std::format("WARNING: Octree mem usage ({} MB) approaches total capacity ({} MB). Further points are ignored.",
					stats.allocatedBytes_persistent / 1'000'000llu,
					persistentBufferCapacity / 1'000'000llu
				);
#else
        string message = fmt::format("WARNING: Octree mem usage ({} MB) approaches total capacity ({} MB). Further points are ignored.",
                                     stats.allocatedBytes_persistent / 1'000'000llu,
                                     persistentBufferCapacity / 1'000'000llu
        );
#endif

				ImGui::Text(message.c_str());
			}

			if(lazInBatches){
				ImGui::Text("WARNING: Loading *.laz files. Load routines are optimized for *.las or *.simlod - laz may be slow.");
			}
			
			ImGui::Text(" ");

			ImGui::PopStyleColor();


			ImGui::End();
		}

		

		frameCounter++;
	};

	renderer->loop(update, render);

	return 0;
}
