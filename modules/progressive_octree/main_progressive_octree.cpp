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
#include <charconv>
#include <map>
#include <set>
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
#include "ProggyTinyFont.h"
#include "CrashHandler.h"
#include "UserFolders.h"

// Uniforms is passed to the kernels by value, and CUDA caps kernel parameters
// at 4 KB. Growing past that fails at launch, not at compile time, without this.
static_assert(sizeof(Uniforms) <= 4096, "Uniforms exceeds the 4 KB kernel-parameter limit");
#include "SimlodLoader.h"
#include "LasLoader.h"

using namespace std;

#ifdef _WIN32
// Hybrid-graphics laptops: ask the NVIDIA driver to render this program on the
// NVIDIA GPU, where CUDA can register the frame texture. NVIDIA, "Enabling High
// Performance Graphics Rendering on Optimus Systems" (driver release 302 on).
// Linking nvcuda.dll is a higher-priority method in the same document and
// should already select it; the log's "OpenGL:" line shows which GPU is used.
extern "C" { __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001; }
#endif

// BATCH_STREAM_SIZE, the device upload ring, is in HostDeviceInterface.h
constexpr uint64_t MAX_DECODED_AHEAD    = 50;          // batches the loaders may decode ahead of the kernel, in pinned host memory
constexpr uint64_t MAX_BATCH_SIZE       = 1'000'000;   // each loaded batch comprises <size> points
constexpr uint64_t VRAM_RESERVE_BYTES   = 512'000'000; // left free beside the octree pool: grids, GL framebuffers, headroom

CudaPrint cudaprint;


vector<string> paths;   // the point files of the current load

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
// the device Point (structures.cuh) is 16 bytes too; the upload ring and
// CONSTRUCT_SCRATCH_FIXED_BYTES are sized on that
static_assert(sizeof(Point) == 16, "Point must stay 16 bytes, as on the device");

struct PointBatch{
	string file = "";
	uint64_t first = 0;   // index of its first point in the file; an int wrapped past 2^31
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
GLuint g_registeredFrameTexture = 0;   // the texture cugl_colorbuffer is registered for; 0 = none

// What the frame in the frame texture was drawn from, so that renderCUDA can
// tell when the next frame would come out the same (sameAsLastDrawn).
static struct {
	bool        valid = false;
	Uniforms    key;                  // frameKey of its uniforms
	int         generation = -1;      // the render program's, so a hot reload draws again
	GLuint      texture = 0;
	CUdeviceptr renderBuffer = 0;
} g_lastDrawn;

// Called before a resize replaces the frame texture, so CUDA lets go of it
// before OpenGL deletes it. GL may give the new texture the same name.
static void releaseFrameTexture(){
	g_lastDrawn.valid = false;
	if(g_registeredFrameTexture == 0) return;
	cuGraphicsUnregisterResource(cugl_colorbuffer);
	g_registeredFrameTexture = 0;
}

cudaStream_t stream_upload, stream_download;

CudaModularProgram* cuda_program_update = nullptr;
CudaModularProgram* cuda_program_render = nullptr;
CudaModularProgram* cuda_program_reset  = nullptr;

glm::mat4 transform;
glm::mat4 transform_updatebound;

Stats stats;
void* h_stats_pinned = nullptr;

// Settings. Everything that changes how the cloud is represented starts at the
// value that changes nothing, the unedited baseline: each point in the colour
// stored in the file, with nothing added and nothing hidden. Every effect is
// built up from there. Reset to unedited (resetToUnedited) returns to these
// values, and Show unedited draws the cloud as they would.
struct {
	bool useHighQualityShading       = true;
	bool showBoundingBox             = false;
	bool doUpdateVisibility          = true;
	bool showPoints                  = true;
	bool showMarkers                 = true;
	bool markersOnly                 = false;   // hide the forest and insects
	bool colorByNode                 = false;
	bool colorByLOD                  = false;
	bool autoFocusOnLoad             = true;
	float minNodeSize                = 64.0f;
	bool  adaptiveDetail             = true;    // coarsen LOD when frames exceed the budget
	float frameBudgetMs              = 33.0f;   // render-kernel time to aim for (30 fps)
	bool  vsync                      = true;    // cap at display refresh: saves power and heat
	bool  skipUnchanged              = true;    // keep the last frame while nothing that changes it has changed
	int pointSize                    = 1;
	float fovy                       = 90.0f;
	bool enableEDL                   = false;   // reference has no outlines (apply_look.ps1)
	float edlStrength                = 0.8f;

	// tier 1 shading
	bool  roundSplats                = true;
	// Depth haze: distance goes LIGHTER. The slider is logarithmic: 0.001 is
	// already a visible haze, 0.018 washes out past ~30 m. The colour is sRGB,
	// like the other colour pickers (white is the same either way).
	bool  enableFog                  = false;   // follows fogDensity > 0
	float fogDensity                 = 0.0f;    // 0 = no haze
	float fogR                       = 1.0f;    // white haze
	float fogG                       = 1.0f;
	float fogB                       = 1.0f;
	float fogHeightFalloff           = 0.0f;    // per metre; 0 keeps the haze even with height
	float fogBase                    = 0.0f;    // metres above the lowest point
	bool  fogSky                     = false;   // haze between the camera and the sky as well
	bool  enableBloom                = false;
	float bloomStrength              = 0.0f;    // 0 = no bloom
	float bloomRadius                = 3.0f;   // now a mip level count, 0-5
	bool  showUnedited               = false;  // compare: draw the cloud as Reset to unedited leaves it
	float exposure                   = 1.0f;
	float toneCurve                  = 0.0f;   // 0 = none .. 1 = the ACES fit
	float emissionGain               = 3.0f;
	int   numInsects                 = 0;
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

	int64_t targetPoints             = 0;       // 0 = not yet set; filled from the cloud on load
	float saturation                 = 1.0f;
	float temperature                = 0.0f;   // negative cool, positive warm
	float tintColour[3]              = {0.55f, 0.75f, 0.60f};
	float tintMix                    = 0.0f;
	float skyColour[3]               = {0.16f, 0.17f, 0.17f};   // dark grey

	// look refinement
	float pointWorldSize             = 0.085f;   // metres; 0 falls back to pointSize pixels
	float pointShading               = 0.0f;     // 0 = flat disc
	int   maxSplatPx                 = 24;
	float grain                      = 0.0f;
	float vignette                   = 0.0f;

	// directional key light. Defaults: high and behind, as in the reference,
	// warm-green light through the canopy.
	float lightAzimuth               = 295.35f;  // degrees
	float lightElevation             = 56.1f;   // degrees above horizon
	float lightColour[3]             = {1.0f, 1.0f, 1.0f};      // white
	float lightIntensity             = 2.5f;
	float ambient                    = 0.109f;   // shadowed points keep a third of their colour
	float lightAbsorb                = 0.16f;   // per metre: 5 m of canopy passes about 45% (exp(-0.8))
	float sunbeams                   = 0.0f;    // key light in the haze; 0 = haze lit by ambient only
	float sunAnisotropy              = 0.5f;    // forward scattering of the haze (Henyey-Greenstein g)

	// key light switch, near clip, mixer, camera, lens, horizon, markers
	bool  lightOn                    = false;  // key light switch
	float nearClip                   = 0.81f;
	float mixColour[3]               = {1.0f, 1.0f, 1.0f};
	float eyeHeight                  = 3.37f;    // metres above the lowest point
	float lookUpDeg                  = 17.05f;
	int   lensMode                   = 0;       // 0 perspective, 1 fisheye
	float fisheyeFov                 = 180.0f;
	float belowColour[3]             = {0.0f, 0.0f, 0.0f};   // void under the ground
	float markerScale                = 0.20f;   // metres, ball diameter
	float horizonSoftness            = 0.0f;    // degrees

	// blown-out colour repair (render.cu): the scanner's camera whites out the
	// sunlit canopy; those points take the colour of their neighbourhood
	bool  whiteFix                   = false;
	float whiteLevel                 = 100.0f;  // darkest channel at least this ...
	float whiteSpread                = 120.0f;  // ... and channels this close = blown out
	int   whiteReach                 = 1;       // cells searched for colour
	float tintMargin                 = 0.0f;    // blue this far above green = sky cast

	// torch: spotlight from the camera, aimed with the mouse (T toggles)
	bool  torchOn                    = false;
	float torchIntensity             = 4.0f;
	float torchAngle                 = 18.0f;   // degrees, half-angle of the beam
	float torchRange                 = 12.0f;   // metres to half strength
	float emissionColour[3]          = {1.0f, 0.45f, 0.85f};   // the reference pink
	bool  emissionByFrequency        = false;
} settings;

static void logLine(const char* fmt, ...);   // the run log, defined below

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
			reserveSlot();
			logLine("[load] pinned memory pool empty: grown to %d slots\n", int(registered.size()));
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
atomic_uint64_t numPointsUploaded  = 0;   // written by the uploader, read by the main thread's progress log
double loadStart                   = 0.0;   // now() when the load began

float totalUpdateDuration          = 0.0f;   // ms since the load began, for the Stats window

atomic_uint64_t numPointsTotal     = 0;
atomic_uint64_t numPointsLoaded    = 0;
atomic_uint64_t numBytesTotal      = 0;
atomic_uint64_t numBytesLoaded     = 0;
atomic_uint64_t numThreadsLoading  = 0;
atomic_int numBatchesTotal         = 0;
bool lastBatchFinishedDevice       = false;
bool loadStoppedAtCapacity         = false;

// --set keep=f: load an evenly spaced fraction f of each LAS/LAZ file, for a
// cloud larger than the GPU holds (the 455 M-point scan on 8 GB) - instead of
// stopping at capacity with the end of the file missing. Point i of a file is
// kept when floor((i + 1) f) > floor(i f), so n points give exactly floor(n f)
// and numPointsTotal stays exact for the load-finished test.
uint32_t loadKeepPer10k            = 10000;
inline uint64_t keptBefore(uint64_t i){ return i * loadKeepPer10k / 10000; }
inline bool keepPoint(uint64_t i){ return keptBefore(i + 1) != keptBefore(i); }

// Points in batches a loader could not read: the file moved, locked or
// damaged during the load. skipUnreadableBatch takes such a batch off the
// totals, or the load never counted as finished (no refill, no colour grid,
// and --exit and --shots waited for ever). The main loop reports the count.
atomic_uint64_t g_pointsUnreadable = 0;
static void skipUnreadableBatch(const PointBatch& batch){
	uint64_t kept = keptBefore(batch.first + uint64_t(batch.count)) - keptBefore(batch.first);
	g_pointsUnreadable += kept;
	numPointsTotal     -= kept;
	numBatchesTotal--;
}

// How far the construct kernel has got, for the loader and uploader threads to
// gate on. Copied from stats after each readback: the main thread overwrites
// the stats struct itself every frame while those threads would be reading it.
atomic_uint32_t batchesProcessedHost = 0;
atomic_uint64_t pointsProcessedHost  = 0;
uint64_t momentaryBufferCapacity   = 0;
uint64_t persistentBufferCapacity  = 0;
uint64_t renderBufferCapacity      = 0;

PinnedMemPool pinnedMemPool;

// ---- run log ---------------------------------------------------------------
// Every [host] line is written to the console and to log/<timestamp>.log.
// The directory is created on first use, under the working directory; see
// logLine for the fallbacks.
#include <cstdarg>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <unordered_map>

static std::ofstream g_logFile;
static std::string   g_logPath;   // the crash handler appends here too
static bool          g_logOpened = false;
static size_t        g_logBytes  = 0;
static constexpr size_t LOG_MAX_BYTES = 20'000'000;   // one run; a 42 MB log was seen
static constexpr int    LOG_KEEP_RUNS = 10;           // older log files are deleted

static std::filesystem::path g_exeDir;   // set from argv[0] before the first log line

// Delete all but the newest LOG_KEEP_RUNS - 1 logs, making room for this run's.
// Names carry a sortable timestamp, so name order is age order.
static void pruneOldLogs(const std::filesystem::path& dir){
	std::error_code ec;
	std::vector<std::filesystem::path> logs;
	for(auto& e : std::filesystem::directory_iterator(dir, ec)){
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

	// loader threads log a file they cannot read, often several at once
	static std::mutex mtx_log;
	std::lock_guard<std::mutex> lock(mtx_log);

	fputs(buf, stdout);

	if(!g_logOpened){
		g_logOpened = true;
		auto wall = std::chrono::system_clock::now();
		std::time_t tt = std::chrono::system_clock::to_time_t(wall);
		int ms = int(std::chrono::duration_cast<std::chrono::milliseconds>(wall.time_since_epoch()).count() % 1000);
		std::tm tmv{};
	#ifdef _WIN32
		localtime_s(&tmv, &tt);
	#else
		localtime_r(&tt, &tmv);
	#endif
		// Milliseconds in the name: two runs started in the same second (a
		// scripted --shots run, a relaunch) shared one name, and the second
		// erased the first run's log.
		char stamp[64], name[96];
		std::strftime(stamp, sizeof(stamp), "%Y-%m-%d_%H%M%S", &tmv);
		snprintf(name, sizeof(name), "simlod_%s_%03d.log", stamp, ms);

		// log/ under the working directory, as before. When that cannot be
		// written (a read-only folder), next to the executable, then in the
		// temp folder. The full path goes to the console, the log itself and
		// the Display and debug panel, so the file can always be found:
		// started from Visual Studio, the working directory is build/Release.
		std::error_code ec;
		std::vector<std::filesystem::path> dirs = {std::filesystem::path("log")};
		if(!g_exeDir.empty()) dirs.push_back(g_exeDir / "log");
		auto tmp = std::filesystem::temp_directory_path(ec);
		if(!ec) dirs.push_back(tmp / "SimLOD" / "log");
		for(const auto& dir : dirs){
			std::filesystem::create_directories(dir, ec);
			pruneOldLogs(dir);
			auto path = std::filesystem::absolute(dir / name, ec);
			if(ec) path = dir / name;
			// Append, not truncate: the crash handler writes the stack trace
			// through its own handle, and a truncating stream would write its
			// later lines over the start of the trace.
			g_logFile.open(path, std::ios::out | std::ios::app);
			if(g_logFile.is_open()){
				g_logPath = path.string();
				break;
			}
		}
		if(g_logFile.is_open()){
			printf("[log ] writing to %s\n", g_logPath.c_str());
		}else{
			printf("[log ] cannot write a log file: tried ./log, next to the executable, and the temp folder\n");
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

// Show unedited skips every step these controls set (showUnedited() in
// render.cu), so while it is ticked they are drawn faint for the scope of one
// of these. They stay editable. Drawn as usual, they seemed broken: every
// setting but point size did nothing while the comparison was on.
struct UneditedDim{
	bool active;
	UneditedDim() : active(settings.showUnedited){
		if(active) ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.4f);
	}
	~UneditedDim(){ if(active) ImGui::PopStyleVar(); }
};

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

// ---- call schedule (inserted by apply_emission.ps1) -----------------------
// Reads the CSV the test kit generates. Columns:
//   call_id,source,freq_hz,channel,x,y,z,t_start_s,duration_s,t_end_s,phase
struct ScheduledCall {
	double x, y, z;     // as in calls.csv: file coordinates, or survey ones (g_callsInSurvey)
	float  freqHz;
	double tStart, duration;
};
static std::vector<ScheduledCall> g_calls;
static bool   g_callsLoaded  = false;
static double g_showClock    = 0.0;
static bool   g_showPlaying  = false;
static const float VISUAL_TAIL_START  = 1.2f;   // seconds
static const float EVENT_RADIUS_START = 4.0f;   // metres, until a cloud is loaded
static float  g_visualTail   = VISUAL_TAIL_START;    // seconds the glow outlasts the call
static float  g_eventRadius  = EVENT_RADIUS_START;   // metres
static float  g_eventRadiusArg = 0.0f;  // --set radius=: replaces the fit to the cloud; 0 = fit
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
static void surveyToFile(double& x, double& y, double& z){
	if(!g_surveyValid) return;
	double d0 = x - g_surveyT[0], d1 = y - g_surveyT[1], d2 = z - g_surveyT[2];
	const double* R = g_surveyR;
	x = R[0]*d0 + R[3]*d1 + R[6]*d2;
	y = R[1]*d0 + R[4]*d1 + R[7]*d2;
	z = R[2]*d0 + R[5]*d1 + R[8]*d2;
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

static void notice(const char* fmt, ...);   // one line on screen for three seconds, defined below

static void loadCallSchedule(const std::string& path){
	g_calls.clear();
	std::ifstream in(path);
	if(!in){
		logLine("[calls] cannot open %s\n", path.c_str());
		notice("Could not open %s", path.c_str());
		g_callsLoaded = false;
		return;
	}
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
		c.x        = x;   // converted from the survey frame, if need be, as each frame is built
		c.y        = y;
		c.z        = z;
		c.tStart   = t0;
		c.duration = dur;
		g_calls.push_back(c);
	}
	g_callsLoaded = !g_calls.empty();
	logLine("[calls] loaded %d calls from %s (%d malformed rows skipped)\n", int(g_calls.size()), path.c_str(), skipped);
	if(g_callsLoaded && g_callsInSurvey && !g_surveyValid){
		logLine("[calls] calls.csv is marked as survey coordinates, but no survey frame is fitted: positions used as file coordinates until one is\n");
		notice("calls.csv is marked as survey coordinates, but there is no survey fit: positions used as file (LAS) coordinates until there is");
	}else if(skipped > 0){
		notice("Loaded %d calls; skipped %d malformed rows (see log)", int(g_calls.size()), skipped);
	}else{
		notice("Loaded %d calls from %s", int(g_calls.size()), path.c_str());
	}
}

// Warm for low frequency, cool for high, stretched across 220-340 Hz so the
// test sources read as clearly different; widen this once real calls carry
// real frequencies.
static void freqToColour(float hz, float& r, float& g, float& b){
	// 220 Hz warm through 340 Hz cool, matching the four-source schedule
	float t = (hz - 220.0f) / 120.0f;
	t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
	r = 1.00f - 0.80f * t;
	g = 0.45f + 0.25f * t;
	b = 0.15f + 0.85f * t;
}
// in double, as now() gives them: in float the walking step varied from frame
// to frame once the session had run for a day or two
double lastFrameTime = now();
double timeSinceLastFrame = 0.0;

float3 boxMin  = float3{InfinityF, InfinityF, InfinityF};
float3 boxMax  = float3{-InfinityF, -InfinityF, -InfinityF};
float3 boxSize = float3{0.0, 0.0, 0.0};

// ---- voxel occupancy grid -----------------------------------------------------
// Sized from the cloud's bounding box on load: the longest side gets the
// chosen resolution, 64 to 512 cells, the others the same cell size. Zeroed on
// every load. Counts are accumulated on the GPU as points enter the octree.
static CUdeviceptr g_voxelGrid  = 0;
static int         g_gridResolution = 256;   // cells along the longest side
static int         g_gridBuiltAtRes = 0;     // what the CURRENT counts were built with
static CUdeviceptr g_lightGrid  = 0;         // per-cell transmittance toward the key light
static CUdeviceptr g_colourGrid = 0;         // per-cell repair colour, uint64; same dimensions; held while needed
static bool        g_colourGridFailed = false;   // no room at these dimensions: not retried every frame

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
static float3 g_markerFrameMin = {InfinityF, InfinityF, InfinityF};   // the boxMin the markers are relative to
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

// Eigenvalues of a symmetric 3x3 by cyclic Jacobi, as above, largest first.
static void jacobiEigenvalues3(double A[3][3], double ev[3]){
	for(int sweep = 0; sweep < 50; sweep++){
		double off  = A[0][1]*A[0][1] + A[0][2]*A[0][2] + A[1][2]*A[1][2];
		double diag = A[0][0]*A[0][0] + A[1][1]*A[1][1] + A[2][2]*A[2][2];
		if(off <= 1e-30 * diag) break;
		for(int p = 0; p < 2; p++){
			for(int r = p + 1; r < 3; r++){
				if(std::fabs(A[p][r]) < 1e-300) continue;
				double theta = (A[r][r] - A[p][p]) / (2.0 * A[p][r]);
				double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta*theta + 1.0));
				double c = 1.0 / std::sqrt(t*t + 1.0);
				double s = t * c;
				for(int k = 0; k < 3; k++){           // rotate columns p and r
					double kp = A[k][p], kr = A[k][r];
					A[k][p] = c*kp - s*kr;
					A[k][r] = s*kp + c*kr;
				}
				for(int k = 0; k < 3; k++){           // rotate rows p and r
					double pk = A[p][k], rk = A[r][k];
					A[p][k] = c*pk - s*rk;
					A[r][k] = s*pk + c*rk;
				}
			}
		}
	}
	ev[0] = A[0][0]; ev[1] = A[1][1]; ev[2] = A[2][2];
	for(int i = 0; i < 2; i++) for(int j = 0; j < 2 - i; j++) if(ev[j] < ev[j + 1]) std::swap(ev[j], ev[j + 1]);
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

	// The markers must not lie on one line: any rotation about that line
	// leaves them in place, so the fit cannot tell those rotations apart, and
	// still reports a small rms. Their spread off their best line is the root
	// of the second-largest eigenvalue of their scatter, per marker. A pick
	// error e turns the frame about the line by about e / spread radians:
	// with 1 cm and a 1 m spread that is 0.6 degrees, which moves a call 20 m
	// from the line by 0.2 m; with a 5 cm spread it is 11 degrees. Below 5 cm
	// the fit is refused, below 1 m it is made with a warning.
	double spread = 0.0;
	{
		double C[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
		for(int i = 0; i < n; i++){
			double a[3] = {ax[i]-ca[0], ay[i]-ca[1], az[i]-ca[2]};
			for(int r = 0; r < 3; r++) for(int c = 0; c < 3; c++) C[r][c] += a[r]*a[c];
		}
		double ev[3];
		jacobiEigenvalues3(C, ev);
		spread = std::sqrt(std::max(ev[1], 0.0) / n);
	}
	if(spread < 0.05){
		logLine("[survey] the %d markers lie on one line (%.3f m off it), so the rotation about it is undetermined; place a marker away from that line\n", n, spread);
		notice("Survey fit refused: the markers lie on one line (%.3f m off it). Add a marker away from that line.", spread);
		g_surveyValid = false;
		return false;
	}

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

	// Horn's method returns a proper rotation, so it cannot match a survey
	// frame of the other handedness, such as columns typed as northing,
	// easting against the file's easting, northing. With the markers close to
	// one plane (checkerboards at similar heights; three markers always are)
	// it matches one almost exactly by turning the vertical upside down. In a
	// test with N and E swapped the rms was 0.000-0.21 m, and a call 12 m above
	// the markers was placed 24 m away, below the ground. Both frames have z
	// up, so a fit whose vertical points down is refused.
	if(R[8] < 0.0){
		logLine("[survey] fit refused: it turns the vertical upside down (R[2][2] = %.3f), which swapped or mirrored survey axes produce\n", R[8]);
		notice("Survey fit refused: it would turn the vertical upside down. Check that the survey columns are in the file's order: x east, y north, z up.");
		g_surveyValid = false;
		return false;
	}
	// A level scan and a survey with measured heights share the vertical. A
	// survey without heights over sloping ground does not: 2 m of rise across
	// 30 m tilts the fit by 3.8 degrees, which moves a call 15 m up by 1 m.
	double tiltDeg = std::acos(std::clamp(R[8], -1.0, 1.0)) * 180.0 / 3.14159265358979;

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
	logLine("[survey] fit from %d pairs: yaw %.2f deg, tilt %.2f deg, t = (%.4f %.4f %.4f), rms %.3f m\n",
		n, yawDeg, tiltDeg, g_surveyT[0], g_surveyT[1], g_surveyT[2], g_surveyRms);
	// 12 decimals: applied to georeferenced coordinates (northings near 5.6e6)
	// six decimals of R put a point up to about 3 m out
	logLine("[survey] R = [%.12f %.12f %.12f; %.12f %.12f %.12f; %.12f %.12f %.12f]\n",
		R[0],R[1],R[2],R[3],R[4],R[5],R[6],R[7],R[8]);
	{
		// Full double precision. At the stream's default 6 significant digits
		// a translation of -5.63392e+06 was rounded to the nearest 10 m, the
		// problem saveMarkersCsv below already avoids.
		std::ofstream f("survey_transform.txt");
		f << std::setprecision(17);
		f << "# survey = R * file + t   (file = LAS coordinates)\n";
		f << "pairs " << n << "\nrms_m " << g_surveyRms << "\nyaw_deg " << yawDeg << "\n";
		f << "R " << R[0]<<" "<<R[1]<<" "<<R[2]<<" "<<R[3]<<" "<<R[4]<<" "<<R[5]<<" "<<R[6]<<" "<<R[7]<<" "<<R[8] << "\n";
		f << "t " << g_surveyT[0] << " " << g_surveyT[1] << " " << g_surveyT[2] << "\n";
	}
	logLine("[survey] the markers spread %.2f m off their best line\n", spread);
	if(spread < 1.0){
		notice("Survey frame fitted from %d markers, but they lie close to one line (%.2f m off it): the rotation about it is uncertain. Add a marker away from that line.", n, spread);
	}else if(tiltDeg > 2.0){
		logLine("[survey] the fit tilts the vertical by %.2f deg; level scans and surveyed heights agree on it\n", tiltDeg);
		notice("Survey frame fitted from %d markers, but it tilts the vertical by %.1f deg: check the survey heights (z) of the markers.", n, tiltDeg);
	}else{
		notice("Survey frame fitted from %d markers: yaw %.1f deg, rms %.3f m (see log)", n, yawDeg, g_surveyRms);
	}
	return true;
}

static std::string pathText(const std::filesystem::path& p);   // UTF-8, defined with the looks

// markers.csv, in the working directory: file coordinates, label, survey coordinates
static std::string markersPath(){
	std::error_code ec;
	std::filesystem::path p = std::filesystem::absolute("markers.csv", ec);
	return ec ? std::string("markers.csv") : pathText(p);
}
// Points are stored as float(file - boxMin) with boxMin itself a float, so
// double(boxMin) + double(rel) recovers file coordinates exactly. Write them
// with fixed decimals: the default stream precision (6 significant digits)
// rounds a georeferenced northing such as 5.9e6 to the nearest 10 m.
static void saveMarkersCsv(){
	// file coordinates need the cloud's corner: with none loaded they would be inf
	if(!std::isfinite(boxMin.x) || !std::isfinite(boxMin.y) || !std::isfinite(boxMin.z)){
		notice("Load a point cloud before saving markers");
		return;
	}
	std::ofstream f("markers.csv");
	if(!f.is_open()){
		logLine("[pick] cannot open %s for writing\n", markersPath().c_str());
		notice("Could not write %s (is it open in another program?)", markersPath().c_str());
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
	logLine("[pick] wrote %d markers to %s (file coordinates)\n", int(g_markers.size()), markersPath().c_str());
	notice("Saved %d markers to %s", int(g_markers.size()), markersPath().c_str());
}
static void loadMarkersCsv(){
	// markers are kept relative to boxMin, which is +infinity until a cloud loads
	if(!std::isfinite(boxMin.x) || !std::isfinite(boxMin.y) || !std::isfinite(boxMin.z)){
		notice("Load a point cloud before loading markers");
		return;
	}
	std::ifstream in("markers.csv");
	if(!in){
		logLine("[pick] %s not found\n", markersPath().c_str());
		notice("No markers.csv in the working directory: %s", markersPath().c_str());
		return;
	}
	std::vector<Marker> loaded;
	std::string line;
	std::getline(in, line);   // header
	int lineNo = 1, skipped = 0, beyondLimit = 0;
	while(std::getline(in, line)){
		lineNo++;
		if(line.empty() || line == "\r") continue;
		if(loaded.size() >= 32){ beyondLimit++; continue; }   // the marker arrays hold 32
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
	}
	// A file with no readable row, such as one re-saved with semicolons,
	// used to replace the markers with none.
	if(loaded.empty() && !g_markers.empty()){
		logLine("[pick] no marker read from %s (%d malformed rows); the %d markers in the scene are kept\n",
			markersPath().c_str(), skipped, int(g_markers.size()));
		notice("No marker could be read from markers.csv (see log); the current markers are kept");
		return;
	}
	g_markers = loaded;
	g_nextMic = 1;
	for(const Marker& m : g_markers) g_nextMic = std::max(g_nextMic, m.mic + 1);
	// the fit was made from the markers just replaced
	bool hadFit = g_surveyValid;
	g_surveyValid = false;
	logLine("[pick] loaded %d markers from %s (%d malformed rows skipped, %d rows beyond the limit of 32 left out)%s\n",
		int(g_markers.size()), markersPath().c_str(), skipped, beyondLimit, hadFit ? "; the survey fit is cleared" : "");
	if(beyondLimit > 0)   notice("Loaded the first 32 markers; %d more rows were left out (32 is the limit)", beyondLimit);
	else if(skipped > 0)  notice("Loaded %d markers; skipped %d malformed rows (see log)", int(g_markers.size()), skipped);
	else if(hadFit)       notice("Loaded %d markers from markers.csv; fit the survey frame again", int(g_markers.size()));
	else                  notice("Loaded %d markers from markers.csv", int(g_markers.size()));
}
static int    g_pickReqX = -1, g_pickReqY = -1;   // pixel of the outstanding request
static float  g_torchPx = -1.0f, g_torchPy = -1.0f;  // frame pixel the torch aims through; < 0 = centre
static bool   g_pickWaiting = false;              // a request has not been answered yet
static int    g_pickSeq = 0;                      // id of the latest request
static int    g_pickSeenDone = 0;                 // consecutive Stats reads showing it answered
static bool   g_pKeyWasDown = false;          // P key state last frame (pick shortcut)
static bool   g_leftWasDown = false;
// g_nextMic moved up to the marker block (fix_nextmic.ps1)
static float  g_walkSpeed = 3.0f;                 // metres per second, W/A/S/D
static ImFont* g_menuFont = nullptr;              // 10-pixel ProggyTiny for the Settings window
static bool   g_smallMenuText = true;             // Settings window in g_menuFont, with tighter spacing
static bool        g_lightDirty = true;
static bool        g_gridRefill = false;     // apply_grid_refill.ps1: counts need rebuilding
static int         g_gridX = 0, g_gridY = 0, g_gridZ = 0;
static float       g_cellSize   = 1.0f;
static size_t      g_gridBytes  = 0;

// Construct scratch (momentary buffer) check. The buffer is sized from the
// kernel's own capacities (CONSTRUCT_* in HostDeviceInterface.h); kernel_construct
// refuses to run, and reports its layout size here, if it would not fit with
// 1 KB to spare. Loading then stalls, so say why, once per session.
static void checkMomentaryBudget(uint64_t used, uint64_t capacity){
	static bool warned = false;
	if(!warned && used + 1024 > capacity){
		warned = true;
		logLine("[WARN] construct scratch overrun: the kernel lays out %.1f MB in a %.1f MB buffer and will not run. "
			"Keep CONSTRUCT_SCRATCH_FIXED_BYTES in HostDeviceInterface.h in step with kernel_construct's allocations.\n",
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
	// Checked against free memory first, as ensureVoxelGrid does: with the
	// driver's sysmem fallback on (the default), cuMemAlloc does not fail when
	// VRAM is short but quietly places the grid in system memory, which the
	// render kernel would then read over PCIe every frame.
	size_t freeMem = 0, totalMem = 0;
	cuMemGetInfo(&freeMem, &totalMem);
	if(lbytes + 64'000'000 >= freeMem || cuMemAlloc(&g_lightGrid, lbytes) != CUDA_SUCCESS){
		g_lightGrid = 0;
		failedX = g_gridX; failedY = g_gridY; failedZ = g_gridZ;
		logLine("[grid] light grid (%.0f MB) would not fit (%.0f MB free); key light disabled\n",
			double(lbytes) / 1e6, double(freeMem) / 1e6);
		notice("Key light needs %.0f MB of GPU memory that is not free - lower the grid resolution", double(lbytes) / 1e6);
		return;
	}
	failedX = failedY = failedZ = -1;
	cuMemsetD8(g_lightGrid, 0, lbytes);
	g_lightDirty = true;
	logLine("[grid] light grid allocated, %.0f MB\n", double(lbytes) / 1e6);
}

// The colour repair grid: 8 bytes a cell, twice the occupancy grid's size.
// Fix blown-out canopy reads it, and the bird's-eye view takes its colours
// from it. It used to be allocated with the occupancy grid and held for as
// long, whether or not either needed it; now it is held only while one of
// them does (beginFrameState), and the next grid rebuild fills it. The key
// light's grid, the occupancy grid's size, is allocated after this one, so
// room is left for it while the light is off: the optional grid must not
// cost the light.
static void ensureColourGrid(bool wanted){
	if(!wanted){
		if(g_colourGrid != 0){
			cuMemFree(g_colourGrid);
			g_colourGrid = 0;
			logLine("[grid] colour repair grid freed\n");
		}
		return;
	}
	if(g_colourGrid != 0 || g_colourGridFailed) return;
	if(g_voxelGrid == 0 || g_gridX <= 0 || g_gridY <= 0 || g_gridZ <= 0) return;
	size_t cbytes = size_t(g_gridX) * size_t(g_gridY) * size_t(g_gridZ) * sizeof(uint64_t);
	size_t lightRoom = g_lightGrid != 0 ? 0 : g_gridBytes;
	size_t freeMem = 0, totalMem = 0;
	cuMemGetInfo(&freeMem, &totalMem);
	if(cbytes + lightRoom + 64'000'000 < freeMem && cuMemAlloc(&g_colourGrid, cbytes) == CUDA_SUCCESS){
		cuMemsetD8(g_colourGrid, 0, cbytes);
		g_gridRefill = true;   // filled with the counts at the next rebuild
		logLine("[grid] colour repair grid allocated, %.1f MB\n", double(cbytes) / 1e6);
	}else{
		g_colourGrid = 0;
		g_colourGridFailed = true;
		logLine("[grid] colour repair grid (%.0f MB) would not fit (%.0f MB free); blown-out points stay white\n",
			double(cbytes) / 1e6, double(freeMem) / 1e6);
	}
}

// kernel_render's scratch, per frame: the visible-node list (16.8 MB, from
// HostDeviceInterface.h) and the 16 MB line list, then per pixel 45 B:
// framebuffer 8, emission 12, bloom chain 4, quarter-resolution volumetrics
// 1 (RGB and distance), and the high-quality pass's depth 4 and colour 16. The fixed 200 MB it
// used to get was overrun with no check by any window above 3.8 Mpx: a 4K
// frame needs 404 MB, a 1080p one 126 MB. Sized to the window instead,
// growing but never shrinking; growth after startup comes out of the VRAM
// reserve beside the octree pool.
static uint64_t renderBufferBytesFor(int width, int height){
	uint64_t pixels = uint64_t(std::max(width, 1)) * uint64_t(std::max(height, 1));
	uint64_t lists  = uint64_t(MAX_VISIBLE_NODES) * NODE_BYTES + 16'000'000;   // nodes, line vertices
	return lists + 45 * pixels + 1'000'000;   // + alignment and the bloom chain's padding
}

// Returns whether the buffer holds a frame of this size. On failure the old
// buffer is kept and renderCUDA skips its launch until the window shrinks.
static bool ensureRenderBuffer(int width, int height){
	static uint64_t failedNeed = 0;   // a size that would not fit, and how much was free then
	static size_t   failedFree = 0;
	uint64_t need = renderBufferBytesFor(width, height);
	if(need <= renderBufferCapacity){ failedNeed = 0; return true; }

	// do not retry (and re-log) a size that failed every frame, but do once
	// memory has been freed: a grid made smaller, the key light off
	if(need == failedNeed){
		size_t freeNow = 0, totalNow = 0;
		cuMemGetInfo(&freeNow, &totalNow);
		if(freeNow < failedFree + 32'000'000) return false;
	}

	// the previous frame's render may still be reading the old buffer
	cuCtxSynchronize();

	// Free first, then allocate, so only the growth needs to be free. If the
	// allocation fails anyway, put the old size back: it was free a moment ago.
	size_t freeMem = 0, totalMem = 0;
	cuMemGetInfo(&freeMem, &totalMem);
	bool grown = need - renderBufferCapacity + 64'000'000 < freeMem;
	if(grown){
		if(cptr_renderbuffer != 0){ cuMemFree(cptr_renderbuffer); cptr_renderbuffer = 0; }
		grown = cuMemAlloc(&cptr_renderbuffer, need) == CUDA_SUCCESS;
		if(!grown){
			cptr_renderbuffer = 0;
			if(renderBufferCapacity > 0 && cuMemAlloc(&cptr_renderbuffer, renderBufferCapacity) != CUDA_SUCCESS){
				cptr_renderbuffer = 0;
				renderBufferCapacity = 0;
			}
		}
	}
	if(!grown){
		failedNeed = need;
		cuMemGetInfo(&failedFree, &totalMem);   // after the restore
		logLine("[vram] a %d x %d frame needs a %.0f MB render buffer, which does not fit (%.0f MB free); not drawing at this size\n",
			width, height, need / 1e6, freeMem / 1e6);
		notice("Window too large for free GPU memory (needs %.0f MB) - make it smaller", need / 1e6);
		return false;
	}

	logLine("[vram] render buffer %.0f -> %.0f MB for a %d x %d frame\n",
		renderBufferCapacity / 1e6, need / 1e6, width, height);
	renderBufferCapacity = need;
	failedNeed = 0;
	return true;
}

static void ensureVoxelGrid(){
	float longest = std::max(boxSize.x, std::max(boxSize.y, boxSize.z));
	if(!(longest > 0.0f) || !std::isfinite(longest)) return;

	// Resolution is the number of cells along the cloud's LONGEST side; the
	// other two follow from the same cell size, so cells stay cubic.
	// The occupancy grid takes 4 bytes a cell, the light grid 4 more and the
	// colour grid 8 more, all growing with the cube of this, so 512 is the
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
	if(g_colourGrid != 0){ cuMemFree(g_colourGrid); g_colourGrid = 0; }

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

	// The colour repair grid, freed above with the old grid, is allocated
	// again by ensureColourGrid, and only while it is needed.
	g_colourGridFailed = false;   // new dimensions: worth another try
	// The grid has just been zeroed, so whatever counts it held are gone. Say
	// so, rather than setting g_gridBuiltAtRes and reporting it as current -
	// which is what hid this: the resolution slider emptied the grid on every
	// move and the panel kept saying the grid was up to date.
	g_gridRefill = true;
	logLine("[grid] %d x %d x %d cells, %.3f m each, %.1f MB\n",
		gx, gy, gz, cell, double(bytes) / 1e6);
}

// ---- bird's-eye view -------------------------------------------------------
// A plan view of the whole domain, built from the occupancy grid and the
// colour repair grid (held for it while the view is rebuilt, when there is
// room for it) each time the grid is rebuilt: per
// column, the colour of the highest coloured cell, or a green shade from the
// column's point count. The camera, its view direction and the markers are
// drawn over it every frame.
static int    g_refillGen  = 0;     // bumped when a grid rebuild is launched
static int    g_minimapGen = -1;    // the rebuild the minimap was made from
static GLuint g_minimapTex = 0;
static int    g_minimapW = 0, g_minimapH = 0;
static bool   g_minimapColoured = false;   // the view was built with the colour grid
static bool   g_birdsEyeDrawn   = false;   // drawn in the last settings window; read by beginFrameState

static void buildMinimap(){
	if(g_voxelGrid == 0 || g_gridX <= 0 || g_gridY <= 0 || g_gridZ <= 0) return;
	cuCtxSynchronize();   // the rebuild must have finished
	size_t n = size_t(g_gridX) * size_t(g_gridY) * size_t(g_gridZ);
	std::vector<uint32_t> occ(n);
	if(cuMemcpyDtoH(occ.data(), g_voxelGrid, n * sizeof(uint32_t)) != CUDA_SUCCESS) return;
	std::vector<uint64_t> col;
	if(g_colourGrid != 0){
		col.resize(n);
		if(cuMemcpyDtoH(col.data(), g_colourGrid, n * sizeof(uint64_t)) != CUDA_SUCCESS) col.clear();
	}

	int W = g_gridX, H = g_gridY;
	std::vector<uint32_t> img(size_t(W) * size_t(H), 0xff141414u);   // ABGR: near-black ground
	std::vector<float> dens(size_t(W) * size_t(H), 0.0f);
	float maxD = 1.0f;
	for(int y = 0; y < H; y++) for(int x = 0; x < W; x++){
		float s = 0.0f;
		for(int z = 0; z < g_gridZ; z++) s += float(occ[x + size_t(W) * (y + size_t(H) * z)]);
		float d = std::log1p(s);
		dens[x + size_t(W) * y] = d;
		maxD = std::max(maxD, d);
	}
	const uint64_t HAS_COLOUR = (1ull << 63) | (1ull << 62);
	for(int y = 0; y < H; y++) for(int x = 0; x < W; x++){
		float d = dens[x + size_t(W) * y] / maxD;
		if(d <= 0.0f) continue;
		uint32_t r = uint32_t(20 + 40 * d), g = uint32_t(40 + 150 * d), b = uint32_t(20 + 30 * d);
		if(!col.empty()){
			for(int z = g_gridZ - 1; z >= 0; z--){
				uint64_t q = col[x + size_t(W) * (y + size_t(H) * z)];
				if(q & HAS_COLOUR){
					float k = 0.55f + 0.45f * d;   // denser columns brighter
					r = uint32_t(std::min(255.0f, float((q      ) & 0xff) * k * 1.6f));
					g = uint32_t(std::min(255.0f, float((q >>  8) & 0xff) * k * 1.6f));
					b = uint32_t(std::min(255.0f, float((q >> 16) & 0xff) * k * 1.6f));
					break;
				}
			}
		}
		// row 0 of the image is the NORTH edge (largest y)
		img[x + size_t(W) * (H - 1 - y)] = 0xff000000u | (b << 16) | (g << 8) | r;
	}

	if(g_minimapTex == 0) glGenTextures(1, &g_minimapTex);
	glBindTexture(GL_TEXTURE_2D, g_minimapTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, img.data());
	glBindTexture(GL_TEXTURE_2D, 0);
	g_minimapW = W; g_minimapH = H;
	g_minimapColoured = !col.empty();
	logLine("[host] bird's-eye view rebuilt, %d x %d\n", W, H);
}

static void drawBirdsEye(shared_ptr<GLRenderer> renderer){
	g_birdsEyeDrawn = true;   // keeps the colour grid while this view needs it
	// Rebuilt after each grid rebuild. While the colour grid is on its way
	// (allocated next frame, then filled by a rebuild), wait for it rather
	// than build the view twice, once without colour.
	bool colourComing = g_colourGrid == 0 && !g_colourGridFailed;
	if(g_refillGen != g_minimapGen && !g_gridRefill && g_voxelGrid != 0 && !colourComing){
		buildMinimap();
		g_minimapGen = g_refillGen;
	}
	ImGui::TextDisabled("Bird's eye  (north up)");
	if(g_minimapTex == 0 || g_minimapW <= 0){
		ImGui::TextDisabled("   appears once the cloud has loaded");
		return;
	}
	float availW = ImGui::GetContentRegionAvail().x;
	float w = std::min(availW, 420.0f);
	float h = w * float(g_minimapH) / float(g_minimapW);
	if(h > 420.0f){ h = 420.0f; w = h * float(g_minimapW) / float(g_minimapH); }
	ImVec2 p0 = ImGui::GetCursorScreenPos();
	ImGui::Image((ImTextureID)(intptr_t)g_minimapTex, ImVec2(w, h));
	ImDrawList* dl = ImGui::GetWindowDrawList();

	// renderer frame (metres from the cloud's minimum corner) to minimap pixels
	float extX = float(g_gridX) * g_cellSize, extY = float(g_gridY) * g_cellSize;
	auto toMap = [&](double x, double y){
		return ImVec2(p0.x + float(x / extX) * w, p0.y + (1.0f - float(y / extY)) * h);
	};

	// The window's draw list clips to the whole window, so without this the
	// overlay was painted over the widgets around the map whenever the camera
	// stood outside the cloud - as the orbit camera usually does.
	ImVec2 p1(p0.x + w, p0.y + h);
	dl->PushClipRect(p0, p1, true);

	for(const Marker& m : g_markers){
		ImVec2 q = toMap(m.x, m.y);
		dl->AddCircleFilled(q, 3.0f, IM_COL32(255, 0, 255, 255));
	}

	// A camera off the map is drawn at the nearest edge as a ring, with a
	// tick pointing toward where it really is.
	const ImU32 yellow = IM_COL32(255, 220, 60, 255);
	glm::dvec3 cp = renderer->camera->position;
	glm::dvec3 fwd = -glm::normalize(glm::dvec3(renderer->camera->world[2]));
	ImVec2 cTrue = toMap(cp.x, cp.y);
	ImVec2 c(std::clamp(cTrue.x, p0.x + 6.0f, p1.x - 6.0f), std::clamp(cTrue.y, p0.y + 6.0f, p1.y - 6.0f));
	bool offMap = c.x != cTrue.x || c.y != cTrue.y;
	double fl = std::sqrt(fwd.x * fwd.x + fwd.y * fwd.y);
	if(fl > 1e-6){
		float len = 0.12f * w;
		ImVec2 tip(c.x + float(fwd.x / fl) * len, c.y - float(fwd.y / fl) * len);
		dl->AddLine(c, tip, yellow, 2.0f);
	}
	if(offMap){
		float dx = cTrue.x - c.x, dy = cTrue.y - c.y, dl2 = std::sqrt(dx * dx + dy * dy);
		dl->AddCircle(c, 5.0f, yellow, 0, 2.0f);
		dl->AddLine(c, ImVec2(c.x + dx / dl2 * 10.0f, c.y + dy / dl2 * 10.0f), yellow, 2.0f);
	}else{
		dl->AddCircleFilled(c, 5.0f, yellow);
		dl->AddCircle(c, 5.0f, IM_COL32(0, 0, 0, 255), 0, 1.5f);
	}
	dl->PopClipRect();
	ImGui::TextDisabled("   camera %.1f, %.1f m, %.1f m up%s  (yellow; magenta = markers)", cp.x, cp.y, cp.z,
		offMap ? ", off the map" : "");
	ImGui::Separator();
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
	// each measurement once: the stats readback can repeat a frame's values
	static uint32_t lastFrameID = ~0u;
	if(stats.frameID == lastFrameID) return;
	lastFrameID = stats.frameID;
	// Without the grid rebuilds at the start of the frame (stageNs[7]): a long
	// drag of a light control rebuilds the light grid on every frame, which
	// coarsened the points until well after the drag. A torn readback can
	// pair two frames' values, so a later [7] than [6] is ignored.
	uint64_t gridNs = stats.stageNs[7] <= stats.stageNs[6] ? stats.stageNs[7] : 0;
	double ms = double(stats.stageNs[6] - gridNs) / 1e6;
	if(ms > settings.frameBudgetMs * 1.10)      g_adaptReduce += 0.03f;
	else if(ms < settings.frameBudgetMs * 0.70) g_adaptReduce -= 0.015f;
	g_adaptReduce = std::clamp(g_adaptReduce, 0.0f, 1.0f);
}
static float effectiveMinNodeSize(){ return settings.minNodeSize * (1.0f + 3.0f * g_adaptReduce); }
static int   effectiveMaxSplatPx(){
	int px = int(float(settings.maxSplatPx) * (1.0f - 0.75f * g_adaptReduce) + 0.5f);
	return std::max(2, std::min(px, settings.maxSplatPx));
}

static bool g_frameRenderable = false;   // the render buffer holds this frame's size
static bool g_frameDrawn      = false;   // set by renderCUDA: the frame texture holds this frame
static bool g_drawEveryFrame  = false;   // --bench: every frame is drawn, to be timed
static float g_keptShare      = 0.0f;    // share of recent frames kept unchanged, for the settings window

static void beginFrameState(int width, int height){
	// A minimised window reports 0 x 0: nothing to draw (loading goes on).
	// Drawing anyway registered a zero-size GL texture with CUDA, and mapping
	// it crashed in the driver.
	g_frameRenderable = width > 0 && height > 0 && ensureRenderBuffer(width, height);
	ensureVoxelGrid();
	// The colour grid serves Fix blown-out canopy, and the bird's-eye view
	// (Camera) while that view waits to be rebuilt: held only while either
	// needs it. Turning the fix on tries again after a failure. For the view,
	// only once the load has finished: the rebuild that fills the grid waits
	// for that, and g_gridRefill is set throughout a load, so the grid was
	// held unused for the whole load (334 MB for the full scan at 512).
	bool birdsEyeShown = g_birdsEyeDrawn;   // drawn in last frame's settings window
	g_birdsEyeDrawn = false;
	static bool whiteFixWas = false;
	if(settings.whiteFix && !whiteFixWas) g_colourGridFailed = false;
	whiteFixWas = settings.whiteFix;
	ensureColourGrid(settings.whiteFix
		|| (birdsEyeShown && lastBatchFinishedDevice && (g_gridRefill || g_minimapGen != g_refillGen || !g_minimapColoured)));
	ensureLightGrid(settings.lightOn);

	// kernel_render is the only one to act on the requests below; on a frame
	// renderCUDA skips, leave them pending for the next frame that is drawn
	if(!g_frameRenderable){
		g_frameGridRefill = false;
		g_frameLightDirty = false;
		return;
	}

	// only on frames that are drawn: on the others the last render time is
	// stale, and the controller would drive detail to an extreme meanwhile
	updateAdaptiveDetail();

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

	// The GPU's own figures, so performance work can cite the machine it ran
	// on rather than a datasheet (the laptop part differs from the desktop
	// RTX A2000 in memory size and bus width).
	{
		char name[256] = {0};
		cuDeviceGetName(name, sizeof(name), device);
		int major = 0, minor = 0, memClockKHz = 0, busBits = 0, l2Bytes = 0, smClockKHz = 0, driver = 0;
		size_t totalBytes = 0;
		cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
		cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
		cuDeviceGetAttribute(&memClockKHz, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device);
		cuDeviceGetAttribute(&busBits, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device);
		cuDeviceGetAttribute(&l2Bytes, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device);
		cuDeviceGetAttribute(&smClockKHz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device);
		cuDeviceTotalMem(&totalBytes, device);
		cuDriverGetVersion(&driver);
		int nvrtcMajor = 0, nvrtcMinor = 0;
		nvrtcVersion(&nvrtcMajor, &nvrtcMinor);
		logLine("[host] GPU %s: compute %d.%d, %d SMs, %.0f MB, memory clock %d MHz, %d-bit bus, L2 %d KB, SM clock %d MHz; driver CUDA %d.%d, NVRTC %d.%d\n",
			name, major, minor, numSMs, double(totalBytes) / 1e6, memClockKHz / 1000, busBits, l2Bytes / 1024, smClockKHz / 1000,
			driver / 1000, (driver % 1000) / 10, nvrtcMajor, nvrtcMinor);
	}
}

// A colour picker value (sRGB, 0..1) in linear light, by the same curve as
// srgbToLinear in render.cu.
static float srgbToLinear01(float c){
	c = std::clamp(c, 0.0f, 1.0f);
	return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

// ---- Reset to unedited --------------------------------------------------------
// Every setting that changes how the cloud is represented goes back to its
// starting value: the switches and amounts to the unedited baseline (no light,
// torch, haze, sunbeams, bloom, insects, point shading or eye dome lighting,
// no canopy repair or cull, a neutral grade, no tone curve), and the settings
// that only shape an effect once it is on (light direction and colour, haze
// colour and floor, torch beam, insect size) to their starting values, so an
// effect turned on again starts from the same place each time. Point size,
// splat shape and high-quality shading go back to their starting values too,
// and the points are shown again in their stored colours: Show points on,
// Markers only and the debug colours off.
// Kept: the camera, lens and near clip, the sky, the horizon and the colour
// below it, markers and survey, calls and their playback, detail and speed,
// the bounding box, voxel grid and Update visibility views, and the grids.
static void resetToUnedited(){
	const decltype(settings) d{};   // the starting values

	// points
	settings.pointWorldSize        = d.pointWorldSize;
	settings.pointSize             = d.pointSize;
	settings.roundSplats           = d.roundSplats;
	settings.useHighQualityShading = d.useHighQualityShading;
	settings.pointShading          = d.pointShading;
	settings.enableEDL             = d.enableEDL;
	settings.edlStrength           = d.edlStrength;

	// key light and torch
	settings.lightOn        = d.lightOn;
	settings.lightAzimuth   = d.lightAzimuth;
	settings.lightElevation = d.lightElevation;
	settings.lightIntensity = d.lightIntensity;
	settings.ambient        = d.ambient;
	settings.lightAbsorb    = d.lightAbsorb;
	for(int i = 0; i < 3; i++) settings.lightColour[i] = d.lightColour[i];
	settings.torchOn        = d.torchOn;
	settings.torchIntensity = d.torchIntensity;
	settings.torchAngle     = d.torchAngle;
	settings.torchRange     = d.torchRange;

	// haze, sunbeams and the volume march. Volume reach follows the loaded
	// cloud, as it is set on load.
	settings.enableFog        = d.enableFog;
	settings.fogDensity       = d.fogDensity;
	settings.fogR             = d.fogR;
	settings.fogG             = d.fogG;
	settings.fogB             = d.fogB;
	settings.fogHeightFalloff = d.fogHeightFalloff;
	settings.fogBase          = d.fogBase;
	settings.fogSky           = d.fogSky;
	settings.sunbeams         = d.sunbeams;
	settings.sunAnisotropy    = d.sunAnisotropy;
	settings.volSteps         = d.volSteps;
	float diag = std::sqrt(boxSize.x*boxSize.x + boxSize.y*boxSize.y + boxSize.z*boxSize.z);
	settings.volMaxDist       = (diag > 0.0f && std::isfinite(diag)) ? diag * 1.5f : d.volMaxDist;

	// colour
	settings.exposure    = d.exposure;
	settings.toneCurve   = d.toneCurve;
	settings.saturation  = d.saturation;
	settings.temperature = d.temperature;
	for(int i = 0; i < 3; i++){
		settings.mixColour[i]  = d.mixColour[i];
		settings.tintColour[i] = d.tintColour[i];
	}
	settings.tintMix     = d.tintMix;
	settings.grain       = d.grain;
	settings.vignette    = d.vignette;

	// bloom (under Calls)
	settings.enableBloom   = d.enableBloom;
	settings.bloomStrength = d.bloomStrength;
	settings.bloomRadius   = d.bloomRadius;

	// insects
	settings.numInsects      = d.numInsects;
	settings.insectSize      = d.insectSize;
	settings.insectGlow      = d.insectGlow;
	settings.insectSpeed     = d.insectSpeed;
	settings.insectPlacement = d.insectPlacement;

	// scan clean-up: canopy repair and stray-point cull
	settings.whiteFix       = d.whiteFix;
	settings.whiteLevel     = d.whiteLevel;
	settings.whiteSpread    = d.whiteSpread;
	settings.whiteReach     = d.whiteReach;
	settings.tintMargin     = d.tintMargin;
	settings.minNeighbours  = d.minNeighbours;
	settings.cullReach      = d.cullReach;
	settings.showCulledOnly = d.showCulledOnly;

	// the points shown, in their stored colours
	settings.showPoints  = d.showPoints;
	settings.markersOnly = d.markersOnly;
	settings.colorByNode = d.colorByNode;
	settings.colorByLOD  = d.colorByLOD;

	settings.showUnedited = false;   // the settings are the baseline now; keep their controls live
	g_lightDirty = true;             // relit with the reset values when the light is next on
	logLine("[host] settings reset to unedited\n");
}

// ---- looks ----------------------------------------------------------------------
// A look is every setting that decides how the image looks, kept in a text
// file of name = value lines so it outlasts the session: Save look and Load
// look at the top of the settings panel, --look on the command line. Left
// out: the camera, lens and near clip, markers and survey, the call schedule
// and its playback, the test source, detail and speed (Max splat px among
// them), the grids, the debug views and Show culled only, and Volume reach,
// which is fitted to each cloud. The names are the file format: renaming one
// loses that value from files saved before. The settings type is written
// decltype(settings) throughout: MSVC takes an alias of the unnamed struct as
// its name and then refuses the struct's default values (error C7626).

// The event radius fitted to a cloud when it loads: 12% of its longest side.
static float fittedEventRadius(){
	float longest = std::max(boxSize.x, std::max(boxSize.y, boxSize.z));
	return (longest > 0.0f && std::isfinite(longest)) ? longest * 0.12f : EVENT_RADIUS_START;
}

// Calls section(title) for each panel, then setting(name, value, starting
// value, lowest, highest) for each setting a look holds, in panel order. The
// limits are the sliders' own, so a hand-edited file stays within what the
// panel allows. Colours are float[3]; switches take 0 and 1 as limits. The
// calls' event radius and visual tail are globals rather than members of s.
template<typename Section, typename Setting>
static void forEachLookSetting(decltype(settings)& s, const decltype(settings)& d, Section&& section, Setting&& setting){
	section("Points");
	setting("point_size_m",           s.pointWorldSize,        d.pointWorldSize,        0.0f, 0.3f);
	setting("point_shading",          s.pointShading,          d.pointShading,          0.0f, 1.0f);
	setting("round_splats",           s.roundSplats,           d.roundSplats,           0, 1);
	setting("high_quality_shading",   s.useHighQualityShading, d.useHighQualityShading, 0, 1);
	setting("eye_dome_lighting",      s.enableEDL,             d.enableEDL,             0, 1);
	setting("edl_strength",           s.edlStrength,           d.edlStrength,           0.0f, 3.0f);

	section("Light");
	setting("key_light_on",           s.lightOn,               d.lightOn,               0, 1);
	setting("light_azimuth_deg",      s.lightAzimuth,          d.lightAzimuth,          0.0f, 360.0f);
	setting("light_elevation_deg",    s.lightElevation,        d.lightElevation,        0.0f, 89.0f);
	setting("light_strength",         s.lightIntensity,        d.lightIntensity,        0.0f, 6.0f);
	setting("light_colour",           s.lightColour,           d.lightColour,           0.0f, 1.0f);
	setting("ambient",                s.ambient,               d.ambient,               0.0f, 1.0f);
	setting("canopy_density",         s.lightAbsorb,           d.lightAbsorb,           0.0f, 1.5f);
	setting("torch_on",               s.torchOn,               d.torchOn,               0, 1);
	setting("torch_strength",         s.torchIntensity,        d.torchIntensity,        0.0f, 20.0f);
	setting("torch_beam_deg",         s.torchAngle,            d.torchAngle,            2.0f, 60.0f);
	setting("torch_reach_m",          s.torchRange,            d.torchRange,            1.0f, 60.0f);

	section("Sky and haze");
	setting("sky_colour",             s.skyColour,             d.skyColour,             0.0f, 1.0f);
	setting("below_horizon_colour",   s.belowColour,           d.belowColour,           0.0f, 1.0f);
	setting("horizon_softness_deg",   s.horizonSoftness,       d.horizonSoftness,       0.0f, 90.0f);
	setting("depth_haze",             s.fogDensity,            d.fogDensity,            0.0f, 0.08f);
	setting("haze_thins_with_height", s.fogHeightFalloff,      d.fogHeightFalloff,      0.0f, 0.5f);
	setting("haze_floor_m",           s.fogBase,               d.fogBase,               0.0f, 1000.0f);
	setting("haze_over_sky",          s.fogSky,                d.fogSky,                0, 1);
	{
		// three settings in the struct, one colour in the panel and the file
		float haze[3] = {s.fogR, s.fogG, s.fogB};
		const float hazeStart[3] = {d.fogR, d.fogG, d.fogB};
		setting("haze_colour",        haze,                    hazeStart,               0.0f, 1.0f);
		s.fogR = haze[0]; s.fogG = haze[1]; s.fogB = haze[2];
	}
	setting("sunbeams",               s.sunbeams,              d.sunbeams,              0.0f, 4.0f);
	setting("forward_scatter",        s.sunAnisotropy,         d.sunAnisotropy,         0.0f, 0.9f);
	setting("volume_steps",           s.volSteps,              d.volSteps,              8, 128);

	section("Colour");
	setting("exposure",               s.exposure,              d.exposure,              0.1f, 16.0f);
	setting("tone_curve",             s.toneCurve,             d.toneCurve,             0.0f, 1.0f);
	setting("saturation",             s.saturation,            d.saturation,            0.0f, 2.5f);
	setting("temperature",            s.temperature,           d.temperature,          -1.0f, 1.0f);
	setting("mix",                    s.mixColour,             d.mixColour,             0.0f, 2.5f);
	setting("tint_colour",            s.tintColour,            d.tintColour,            0.0f, 1.0f);
	setting("tint_mix",               s.tintMix,               d.tintMix,               0.0f, 1.0f);
	setting("grain",                  s.grain,                 d.grain,                 0.0f, 0.15f);
	setting("vignette",               s.vignette,              d.vignette,              0.0f, 1.0f);

	section("Calls: glow, bloom and medium");
	setting("emission_colour",        s.emissionColour,        d.emissionColour,        0.0f, 1.0f);
	setting("emission_by_frequency",  s.emissionByFrequency,   d.emissionByFrequency,   0, 1);
	setting("emission_gain",          s.emissionGain,          d.emissionGain,          0.0f, 12.0f);
	setting("event_radius_m",         g_eventRadius,           fittedEventRadius(),     0.1f, 1000.0f);
	setting("falloff_power",          s.falloffPower,          d.falloffPower,          0.5f, 8.0f);
	setting("core_boost",             s.coreBoost,             d.coreBoost,             0.0f, 8.0f);
	setting("visual_tail_s",          g_visualTail,            VISUAL_TAIL_START,       0.0f, 4.0f);
	setting("bloom_strength",         s.bloomStrength,         d.bloomStrength,         0.0f, 4.0f);
	setting("bloom_spread",           s.bloomRadius,           d.bloomRadius,           0.0f, 5.0f);
	setting("volumetric_medium",      s.enableVolumetric,      d.enableVolumetric,      0, 1);
	setting("medium_density",         s.volDensity,            d.volDensity,            0.0f, 0.12f);
	setting("medium_scatter",         s.volScatter,            d.volScatter,            0.0f, 6.0f);
	setting("medium_pools_low",       s.volHeightFalloff,      d.volHeightFalloff,      0.0f, 0.05f);
	setting("medium_occlusion",       s.volOcclusion,          d.volOcclusion,          0.0f, 3.0f);
	setting("medium_shadow_steps",    s.volShadowSteps,        d.volShadowSteps,        0, 16);

	section("Insects");
	setting("insects",                s.numInsects,            d.numInsects,            0, 30000);
	setting("insect_size_px",         s.insectSize,            d.insectSize,            1, 4);
	setting("insect_glow",            s.insectGlow,            d.insectGlow,            0.0f, 6.0f);
	setting("insect_speed",           s.insectSpeed,           d.insectSpeed,           0.0f, 2.0f);
	setting("insect_placement",       s.insectPlacement,       d.insectPlacement,       0, 1);

	section("Scan clean-up");
	setting("min_neighbours",         s.minNeighbours,         d.minNeighbours,         0, 60);
	setting("cull_reach",             s.cullReach,             d.cullReach,             0, 1);
	setting("fix_blown_out_canopy",   s.whiteFix,              d.whiteFix,              0, 1);
	setting("white_level",            s.whiteLevel,            d.whiteLevel,            100.0f, 250.0f);
	setting("white_spread",           s.whiteSpread,           d.whiteSpread,           5.0f, 120.0f);
	setting("sky_cast",               s.tintMargin,            d.tintMargin,            0.0f, 60.0f);
	setting("colour_reach_cells",     s.whiteReach,            d.whiteReach,            1, 8);
}

static void lookAssign(float& a, float b){ a = b; }
static void lookAssign(int& a, int b){ a = b; }
static void lookAssign(bool& a, bool b){ a = b; }
static void lookAssign(float (&a)[3], const float (&b)[3]){ a[0] = b[0]; a[1] = b[1]; a[2] = b[2]; }

// Within the slider's range, as a value read from a look file is kept
static void lookClamp(float& v, float lo, float hi){ v = std::clamp(v, lo, hi); }
static void lookClamp(int& v, int lo, int hi){ v = std::clamp(v, lo, hi); }
static void lookClamp(bool&, int, int){}
static void lookClamp(float (&v)[3], float lo, float hi){ for(float& c : v) c = std::clamp(c, lo, hi); }

// The shortest text that reads back as the same float, with a point for the
// decimal whatever the system's region settings.
static std::string lookFormat(float v){
	char buf[32];
	auto r = std::to_chars(buf, buf + sizeof(buf), v);
	return std::string(buf, r.ptr);
}
static std::string lookFormat(int v){ return std::to_string(v); }
static std::string lookFormat(bool v){ return v ? "1" : "0"; }
static std::string lookFormat(const float (&v)[3]){
	return lookFormat(v[0]) + " " + lookFormat(v[1]) + " " + lookFormat(v[2]);
}

// n numbers separated by spaces or commas, and nothing else. from_chars
// ignores the locale, so "0,5" is refused rather than read as 0.
static bool lookNumbers(const std::string& text, double* out, int n){
	const char* p = text.data();
	const char* end = p + text.size();
	for(int i = 0; i < n; i++){
		while(p < end && (*p == ' ' || *p == '\t' || (i > 0 && *p == ','))) p++;
		if(p < end && *p == '+') p++;   // from_chars takes a minus but not a plus
		auto r = std::from_chars(p, end, out[i]);
		if(r.ec != std::errc() || !std::isfinite(out[i])) return false;
		p = r.ptr;
	}
	while(p < end && (*p == ' ' || *p == '\t')) p++;
	return p == end;
}
static bool lookParse(const std::string& text, float& v, float lo, float hi){
	double x;
	if(!lookNumbers(text, &x, 1)) return false;
	v = std::clamp(float(x), lo, hi);
	return true;
}
static bool lookParse(const std::string& text, int& v, int lo, int hi){
	double x;
	if(!lookNumbers(text, &x, 1)) return false;
	v = int(std::lround(std::clamp(x, double(lo), double(hi))));
	return true;
}
static bool lookParse(const std::string& text, bool& v, int, int){
	std::string t = text;
	for(char& c : t) c = char(std::tolower((unsigned char)c));
	if(t == "1" || t == "on"  || t == "true"  || t == "yes"){ v = true;  return true; }
	if(t == "0" || t == "off" || t == "false" || t == "no") { v = false; return true; }
	return false;
}
static bool lookParse(const std::string& text, float (&v)[3], float lo, float hi){
	double x[3];
	if(!lookNumbers(text, x, 3)) return false;
	for(int i = 0; i < 3; i++) v[i] = std::clamp(float(x[i]), lo, hi);
	return true;
}

// Paths as UTF-8 for the log and the panel, and back: ImGui works in UTF-8,
// and a std::string path on Windows would be read in the system code page,
// which turns a macron in a look's name into something else.
static std::string pathText(const std::filesystem::path& p){
	auto u = p.u8string();
	return std::string(reinterpret_cast<const char*>(u.data()), u.size());
}
static std::filesystem::path pathFromUtf8(const std::string& s){
	return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}

// Looks go in Documents\SimLOD\looks, so they outlast the build folder and,
// unlike the run log, do not depend on the working directory. If that cannot
// be made: next to the executable, then the working directory.
static std::filesystem::path looksFolder(){
	static std::filesystem::path chosen;
	if(!chosen.empty()) return chosen;
	std::vector<std::filesystem::path> dirs;
	auto docs = documentsFolder();
	if(!docs.empty()) dirs.push_back(docs / "SimLOD" / "looks");
	if(!g_exeDir.empty()) dirs.push_back(g_exeDir / "looks");
	std::error_code ec;
	auto here = std::filesystem::absolute("looks", ec);
	if(!ec) dirs.push_back(here);
	for(const auto& dir : dirs){
		std::filesystem::create_directories(dir, ec);
		if(std::filesystem::is_directory(dir, ec)){
			chosen = dir;
			logLine("[look] looks folder: %s\n", pathText(dir).c_str());
			break;
		}
	}
	return chosen;
}

// A file name from what was typed: characters Windows does not allow in a
// name become '_', spaces and dots at either end go, and the reserved device
// names (CON, NUL, COM1 and so on) get a '_' in front. Other characters,
// macrons among them, are kept as typed.
static std::string lookFileName(const std::string& typed){
	std::string s;
	for(unsigned char c : typed){
		s += (c < 32 || std::strchr("<>:\"/\\|?*", c) != nullptr) ? '_' : char(c);
	}
	size_t a = s.find_first_not_of(" .");
	if(a == std::string::npos) return "";
	size_t b = s.find_last_not_of(" .");
	s = s.substr(a, b - a + 1);
	std::string upper = s;
	for(char& c : upper) c = char(std::toupper((unsigned char)c));
	static const char* reserved[] = {"CON", "PRN", "AUX", "NUL",
		"COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
		"LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"};
	for(const char* r : reserved) if(upper == r) return "_" + s;
	return s;
}

static std::string g_lookName;   // the last look saved or loaded, offered as the next name

static bool saveLook(const std::filesystem::path& path, std::string& error){
	std::string text;
	{
		std::time_t tt = std::time(nullptr);
		std::tm tmv{};
	#ifdef _WIN32
		localtime_s(&tmv, &tt);
	#else
		localtime_r(&tt, &tmv);
	#endif
		char stamp[64];
		std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M", &tmv);
		text += "# SimLOD look, saved " + std::string(stamp) + "\n";
	}
	text += "# One setting per line, name = value. Colours are red, green and blue, from\n"
			"# 0 to 1; switches are 1 (on) or 0 (off). Loading a look first returns every\n"
			"# setting below to its starting value, so a setting left out of the file\n"
			"# takes its starting value. Values are kept within the sliders' ranges, and\n"
			"# unknown names are ignored. Text after # is a comment.\n";
	decltype(settings) start{};
	forEachLookSetting(settings, start,
		[&](const char* title){ text += "\n# "; text += title; text += "\n"; },
		[&](const char* name, auto& value, const auto&, auto, auto){
			text += name; text += " = "; text += lookFormat(value); text += "\n";
		});

	// Written under another name and renamed when complete, so an interrupted
	// save never leaves a short file in place of a good one.
	std::filesystem::path part = path;
	part += ".part";
	std::error_code ec;
	{
		std::ofstream out(part);
		out << text;
		out.close();
		if(!out){
			error = "could not write " + pathText(part);
			std::filesystem::remove(part, ec);
			return false;
		}
	}
	std::filesystem::rename(part, path, ec);
	if(ec){
		error = "could not write " + pathText(path) + ": " + ec.message();
		std::filesystem::remove(part, ec);
		return false;
	}
	return true;
}

static bool loadLook(const std::filesystem::path& path){
	std::ifstream in(path);
	if(!in){
		logLine("[look] cannot open %s\n", pathText(path).c_str());
		notice("Could not open the look %s", pathText(path.filename()).c_str());
		return false;
	}
	auto trim = [](const std::string& s){
		size_t a = s.find_first_not_of(" \t\r");
		if(a == std::string::npos) return std::string();
		size_t b = s.find_last_not_of(" \t\r");
		return s.substr(a, b - a + 1);
	};
	std::map<std::string, std::string> values;   // a name given twice: the last one counts
	std::vector<std::string> problems;
	std::string line;
	bool firstLine = true;
	while(std::getline(in, line)){
		// a byte order mark, which some editors write, would become part of the first name
		if(firstLine && line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);
		firstLine = false;
		size_t hash = line.find('#');
		if(hash != std::string::npos) line.erase(hash);
		line = trim(line);
		if(line.empty()) continue;
		size_t eq = line.find('=');
		if(eq == std::string::npos){ problems.push_back("no '=' in: " + line); continue; }
		values[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
	}

	auto before = settings;
	decltype(settings) start{};
	std::set<std::string> applied;
	int total = 0;
	forEachLookSetting(settings, start, [](const char*){},
		[&](const char* name, auto& value, const auto& startValue, auto lo, auto hi){
			total++;
			lookAssign(value, startValue);
			auto it = values.find(name);
			if(it == values.end()) return;
			if(lookParse(it->second, value, lo, hi)) applied.insert(name);
			else problems.push_back(std::string(name) + " = " + it->second + ": not a valid value, so its starting value is used");
			values.erase(it);
		});
	for(const auto& kv : values) problems.push_back("unknown setting " + kv.first + ", ignored");

	// A saved event radius is kept when a cloud loads, as --set radius= keeps
	// one; without one, the radius is fitted to each cloud.
	g_eventRadiusArg = applied.count("event_radius_m") ? g_eventRadius : 0.0f;
	// switches the panel sets from an amount
	settings.enableFog   = settings.fogDensity > 0.0f;
	settings.enableBloom = settings.bloomStrength > 0.0f;
	settings.showUnedited = false;   // show the look just loaded
	g_lightDirty = true;
	// the colour grid is built with these, so it must be built again
	// Fix blown-out canopy too: built with the fix off, the grid counts blown
	// points as colour sources, as the checkbox's rebuild already allows for.
	if(settings.whiteLevel != before.whiteLevel || settings.whiteSpread != before.whiteSpread
	|| settings.tintMargin != before.tintMargin || settings.whiteReach != before.whiteReach
	|| settings.whiteFix != before.whiteFix){
		g_gridRefill = true;
	}

	g_lookName = pathText(path.stem());
	logLine("[look] loaded %s: %d of its %d settings from the file\n", pathText(path).c_str(), int(applied.size()), total);
	for(const auto& p : problems) logLine("[look]   %s\n", p.c_str());
	if(problems.empty()) notice("Look %s loaded", g_lookName.c_str());
	else notice("Look %s loaded, with %d problem(s): see the run log", g_lookName.c_str(), int(problems.size()));
	return true;
}

// Keeps one look setting, by its name in look files, within its slider's range
static void clampLookSetting(const char* lookName){
	decltype(settings) start{};
	forEachLookSetting(settings, start, [](const char*){},
		[&](const char* name, auto& value, const auto&, auto lo, auto hi){
			if(std::strcmp(name, lookName) == 0) lookClamp(value, lo, hi);
		});
}

// --look: a file, or the name of a look in the looks folder
static bool loadLookArg(const std::string& arg){
	std::filesystem::path p(arg);   // argv is in the system code page, as a std::string path expects
	std::vector<std::filesystem::path> tries = {p};
	auto dir = looksFolder();
	if(!dir.empty()){
		tries.push_back(dir / p);
		auto withTxt = dir / p;
		withTxt += ".txt";
		tries.push_back(withTxt);
	}
	std::error_code ec;
	for(const auto& t : tries){
		if(std::filesystem::is_regular_file(t, ec)) return loadLook(t);
	}
	logLine("[look] --look %s: no such file here or in %s\n", arg.c_str(), pathText(dir).c_str());
	return false;
}

struct LookFile{
	std::filesystem::path path;
	std::string name;   // UTF-8, without .txt
	std::string age;    // "3 h ago"
};
static std::vector<LookFile> g_lookFiles;   // listed when Load look opens
static char g_lookNameBuf[128] = "";

// Age rather than date: file_time_type converts to a calendar date
// differently in MSVC's library and in GCC's.
static std::string lookAge(std::filesystem::file_time_type t){
	auto age = std::filesystem::file_time_type::clock::now() - t;
	long long minutes = std::chrono::duration_cast<std::chrono::minutes>(age).count();
	if(minutes < 1) return "just now";
	if(minutes < 60) return std::to_string(minutes) + " min ago";
	long long hours = minutes / 60;
	if(hours < 48) return std::to_string(hours) + " h ago";
	return std::to_string(hours / 24) + " days ago";
}

static void listLooks(){
	g_lookFiles.clear();
	auto dir = looksFolder();
	if(dir.empty()) return;
	std::error_code ec;
	std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>> found;
	for(const auto& e : std::filesystem::directory_iterator(dir, ec)){
		if(e.is_regular_file(ec) && e.path().extension() == ".txt") found.push_back({e.last_write_time(ec), e.path()});
	}
	std::sort(found.begin(), found.end(), [](const auto& a, const auto& b){ return a.first > b.first; });   // newest first
	for(const auto& f : found) g_lookFiles.push_back({f.second, pathText(f.second.stem()), lookAge(f.first)});
}

static void drawSaveLookPopup(){
	if(!ImGui::BeginPopupModal("Save look", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) return;
	if(ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
	ImGui::SetNextItemWidth(260.0f);
	bool enter = ImGui::InputText("Name", g_lookNameBuf, sizeof(g_lookNameBuf), ImGuiInputTextFlags_EnterReturnsTrue);
	std::string name = lookFileName(g_lookNameBuf);
	std::filesystem::path dir = looksFolder();
	std::filesystem::path path;
	if(!dir.empty() && !name.empty()) path = dir / pathFromUtf8(name + ".txt");
	std::error_code ec;
	bool exists = !path.empty() && std::filesystem::exists(path, ec);
	if(dir.empty()){
		ImGui::TextWrapped("No folder for looks could be made: Documents, beside SimLOD and the working folder were tried.");
	}else if(name.empty()){
		ImGui::TextDisabled("Type a name for the look.");
	}else{
		ImGui::TextDisabled("%s", pathText(path).c_str());
		if(exists) ImGui::Text("A look with this name exists; saving replaces it.");
	}
	bool save = ImGui::Button(exists ? "Replace" : "Save") || enter;
	ImGui::SameLine();
	bool cancel = ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Escape));
	if(save && !path.empty()){
		std::string error;
		if(saveLook(path, error)){
			g_lookName = name;
			logLine("[look] saved %s\n", pathText(path).c_str());
			notice("Look %s saved", name.c_str());
		}else{
			logLine("[look] not saved: %s\n", error.c_str());
			notice("The look was not saved: %s", error.c_str());
		}
		ImGui::CloseCurrentPopup();
	}else if(cancel){
		ImGui::CloseCurrentPopup();
	}
	ImGui::EndPopup();
}

static void drawLoadLookPopup(){
	if(!ImGui::BeginPopup("Load look")) return;
	std::filesystem::path dir = looksFolder();
	if(dir.empty()){
		ImGui::TextDisabled("No folder for looks could be made.");
	}else if(g_lookFiles.empty()){
		ImGui::TextDisabled("No saved looks yet. Save look puts them in this folder:");
	}
	for(size_t i = 0; i < g_lookFiles.size(); i++){
		const LookFile& f = g_lookFiles[i];
		ImGui::PushID(int(i));
		if(ImGui::Selectable(f.name.c_str(), false, 0, ImVec2(220.0f, 0.0f))) loadLook(f.path);   // closes the popup
		ImGui::SameLine();
		ImGui::TextDisabled("%s", f.age.c_str());
		ImGui::PopID();
	}
	if(!dir.empty()){
		ImGui::Separator();
		ImGui::TextDisabled("%s", pathText(dir).c_str());
		if(ImGui::SmallButton("Open folder")) openInFileBrowser(dir);
		ImGui::SameLine();
		if(ImGui::SmallButton("Copy path")) ImGui::SetClipboardText(pathText(dir).c_str());
	}
	ImGui::EndPopup();
}

Uniforms getUniforms(shared_ptr<GLRenderer> renderer){
	Uniforms uniforms;
	// Every byte, padding and unused event slots too: renderCUDA compares
	// frames' uniforms byte for byte.
	memset(&uniforms, 0, sizeof(uniforms));

	glm::mat4 world;
	glm::mat4 view = renderer->camera->view;
	glm::mat4 proj = renderer->camera->proj;
	glm::mat4 worldViewProj = proj * view * world;
	view = glm::transpose(view);
	worldViewProj = glm::transpose(worldViewProj);

	memcpy(&uniforms.view, &view, sizeof(view));
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
		uniforms.markerScale   = settings.markerScale;
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
	{
		// The insects move along a sum of sines at six fixed rates, in radians
		// a second at speed 1. Their phases are reduced here, in double: the
		// clock passed in float lost precision as the session ran on (a step
		// of 7.8 ms after 18 h, 15.6 ms after 36 h), so the insects moved in
		// jumps and, from about 3 days, not on every frame.
		static const double insectRate[6] = {0.83, 2.7, 0.71, 3.1, 1.13, 3.7};
		double t = now() * double(settings.insectSpeed);
		for(int k = 0; k < 6; k++) uniforms.insectPhase[k] = float(std::fmod(t * insectRate[k], 2.0 * 3.14159265358979));
	}
	uniforms.boxMin                   = float3{0.0f, 0.0f, 0.0f};
	uniforms.boxMax                   = boxSize;
	uniforms.frameCounter             = frameCounter;
	uniforms.showBoundingBox          = settings.showBoundingBox;
	// Show unedited draws the points as Reset to unedited leaves them: shown,
	// in their stored colours rather than the debug colours.
	uniforms.showPoints               = settings.showPoints || settings.showUnedited;
	uniforms.colorByNode              = settings.colorByNode && !settings.showUnedited;
	uniforms.colorByLOD               = settings.colorByLOD && !settings.showUnedited;
	uniforms.minNodeSize              = effectiveMinNodeSize();
	uniforms.pointSize                = settings.pointSize;
	uniforms.useHighQualityShading    = settings.useHighQualityShading;
	uniforms.persistentBufferCapacity = persistentBufferCapacity;
	uniforms.momentaryBufferCapacity  = momentaryBufferCapacity;
	uniforms.renderBufferCapacity     = renderBufferCapacity;
	uniforms.enableEDL                = settings.enableEDL;
	uniforms.edlStrength              = settings.edlStrength;

	uniforms.roundSplats              = settings.roundSplats;
	uniforms.enableFog                = settings.enableFog;
	uniforms.fogDensity               = settings.fogDensity;
	uniforms.fogR                     = srgbToLinear01(settings.fogR);   // the composite works in linear light
	uniforms.fogG                     = srgbToLinear01(settings.fogG);
	uniforms.fogB                     = srgbToLinear01(settings.fogB);
	uniforms.fogHeightFalloff         = settings.fogHeightFalloff;
	uniforms.fogBase                  = settings.fogBase;
	uniforms.fogSky                   = settings.fogSky ? 1 : 0;
	uniforms.enableBloom              = settings.enableBloom;
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
			float diag    = std::sqrt(boxSize.x*boxSize.x + boxSize.y*boxSize.y + boxSize.z*boxSize.z);

			// test source: centre of the cloud in plan, at 55% of its height
			settings.testX = boxMin.x + boxSize.x * 0.5f;
			settings.testY = boxMin.y + boxSize.y * 0.5f;
			settings.testZ = boxMin.z + boxSize.z * 0.55f;

			g_eventRadius        = g_eventRadiusArg > 0.0f ? g_eventRadiusArg : fittedEventRadius();
			settings.volMaxDist  = diag * 1.5f;        // rays can cross the whole cloud
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
		// A drop zeroes the stats. Remembering that 0 made the old cloud's total
		// look like a target the user had set, so a newly dropped cloud was
		// drawn at old total / new total of its points.
		if(totalI > 0) lastTotal = totalI;
		if(total > 0.0){
			double d = double(settings.targetPoints) / total;
			settings.pointDensity = float(d < 0.001 ? 0.001 : (d > 1.0 ? 1.0 : d));
		}
	}
	uniforms.voxelGridPtr    = (uint64_t)g_voxelGrid;
	uniforms.colourGridPtr   = (uint64_t)g_colourGrid;
	uniforms.whiteFix        = settings.whiteFix ? 1 : 0;
	uniforms.whiteLevel      = settings.whiteLevel;
	uniforms.whiteSpread     = settings.whiteSpread;
	uniforms.whiteReach      = settings.whiteReach;
	uniforms.tintMargin      = settings.tintMargin;
	uniforms.torchOn         = settings.torchOn ? 1 : 0;
	uniforms.torchX          = g_torchPx >= 0.0f ? g_torchPx : 0.5f * float(renderer->width);
	uniforms.torchY          = g_torchPy >= 0.0f ? g_torchPy : 0.5f * float(renderer->height);
	uniforms.torchIntensity  = settings.torchIntensity;
	{
		float outer = std::clamp(settings.torchAngle, 1.0f, 80.0f) * 0.017453292f;
		uniforms.torchCosOuter = std::cos(outer);
		uniforms.torchCosInner = std::cos(outer * 0.6f);   // soft edge over the outer 40%
	}
	uniforms.torchRange      = settings.torchRange;
	uniforms.gridX           = g_gridX;
	uniforms.gridY           = g_gridY;
	uniforms.gridZ           = g_gridZ;
	uniforms.cellSize        = g_cellSize;
	uniforms.showVoxelGrid   = settings.showVoxelGrid;
	uniforms.insectPlacement = settings.insectPlacement;
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
	uniforms.showUnedited     = settings.showUnedited ? 1 : 0;
	uniforms.toneCurve        = std::clamp(settings.toneCurve, 0.0f, 1.0f);
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
		// Enough one-cell steps for any ray to leave the grid, so canopy
		// anywhere above a cell can shade it. A fixed 40 steps reached 27.6 m
		// above a point at grid resolution 256 and 13.8 m at 512, so raising
		// the resolution took away the shade of the upper canopy. The march
		// stops on leaving the grid, or when almost no light is left.
		uniforms.lightSteps     = g_gridX + g_gridY + g_gridZ;
		uniforms.lightDirty     = g_frameLightDirty ? 1 : 0;   // decided once per frame in beginFrameState
		uniforms.sunbeams       = settings.sunbeams;
		uniforms.sunAnisotropy  = std::clamp(settings.sunAnisotropy, -0.95f, 0.95f);   // g = 1 is a spike
	}
	// Only with its grid. Without one (no room for it, or no occupancy grid to
	// build it from) lightAt reads full sun everywhere, so every point was lit
	// at k = ambient + (1 - ambient) * intensity, 2.34 at the defaults, with no
	// canopy shade, while the log and notice said the light was disabled.
	uniforms.lightOn        = (settings.lightOn && g_lightGrid != 0) ? 1 : 0;
	uniforms.nearClip       = settings.nearClip;
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
				// The loaders subtract boxMin from every point, so the
				// renderer's world starts at the origin and uniforms.boxMin
				// reads as zero. Any external coordinate - a call position
				// here, a TDOA fix or a microphone location later - must carry
				// the same offset or it lands outside the cloud entirely.
				// Survey coordinates are converted with the fit as it is now,
				// so a new fit, or none, applies at once. They were converted
				// once, when calls.csv was read.
				double fx = c.x, fy = c.y, fz = c.z;
				if(g_callsInSurvey) surveyToFile(fx, fy, fz);   // unchanged while no fit exists
				uniforms.eventX[i]  = float(fx - double(boxMin.x));
				uniforms.eventY[i]  = float(fy - double(boxMin.y));
				uniforms.eventZ[i]  = float(fz - double(boxMin.z));
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
		logIfChanged("haze", 5.0, "[host] haze=%.3f/m thins=%.3f/m floor=%.1fm sky=%d | light=%d strength=%.2f ambient=%.2f sunbeams=%.2f g=%.2f | medium steps=%d reach=%.0fm\n",
			settings.enableFog ? settings.fogDensity : 0.0f, settings.fogHeightFalloff, settings.fogBase, settings.fogSky ? 1 : 0,
			settings.lightOn ? 1 : 0, settings.lightIntensity, settings.ambient, settings.sunbeams, settings.sunAnisotropy,
			settings.volSteps, settings.volMaxDist);
		logIfChanged("box", 0.0, "[host] box min=(%.2f %.2f %.2f) max=(%.2f %.2f %.2f), load offset (subtracted from every point) = (%.2f %.2f %.2f)\n",
			uniforms.boxMin.x, uniforms.boxMin.y, uniforms.boxMin.z,
			uniforms.boxMax.x, uniforms.boxMax.y, uniforms.boxMax.z,
			boxMin.x, boxMin.y, boxMin.z);
		logIfChanged("grid", 0.0, "[host] voxel grid %d x %d x %d @ %.3f m, insects=%d placement=%d\n",
			g_gridX, g_gridY, g_gridZ, g_cellSize, settings.numInsects, settings.insectPlacement);
	}

	if(settings.markersOnly && !settings.showUnedited){   // the marker array on its own
		uniforms.showPoints = 0;
		uniforms.numInsects = 0;
	}
	return uniforms;
}

// A kernel launch that failed goes to the run log, once until that kernel
// launches again: a failure on every frame would fill the log. Failures were
// printed to the console only.
static void logLaunch(const char* kernel, CUresult res){
	static std::map<string, CUresult> last;
	CUresult& previous = last[kernel];
	if(res != CUDA_SUCCESS && res != previous){
		const char* name = nullptr;
		cuGetErrorName(res, &name);
		logLine("[host] kernel %s did not launch: %s\n", kernel, name ? name : "unknown error");
	}
	previous = res;
}

// False if the reset kernel did not run, and the device still holds the
// previous load's octree and counters.
bool resetCUDA(shared_ptr<GLRenderer> renderer){

	// The current cloud's grid: reload() reads the new box after this, and the
	// next beginFrameState sizes the grid for it, before the first batch.
	ensureVoxelGrid();

	// ensureVoxelGrid returns early when the dimensions are unchanged, which
	// they are when the same cloud is reloaded. The construct kernel adds to
	// the counts as points arrive, so without this every reset doubled them.
	if(g_voxelGrid != 0) cuMemsetD8(g_voxelGrid, 0, g_gridBytes);
	// The colour repair grid and the bird's-eye view are built only by a
	// refill. Without one, a cloud with the same bounds as the last (the same
	// scan, recoloured) kept the old cloud's colours. The refill waits for
	// the load to finish (beginFrameState), then runs once on the new cloud.
	if(g_colourGrid != 0) cuMemsetD8(g_colourGrid, 0, size_t(g_gridX) * size_t(g_gridY) * size_t(g_gridZ) * sizeof(uint64_t));
	g_gridRefill = true;
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
	logLaunch("reset", res_launch);

	cuCheck(cuCtxSynchronize(), "cuCtxSynchronize");

	return res_launch == CUDA_SUCCESS;
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

	auto res_launch = cuLaunchCooperativeKernel(cuda_program_update->kernels["kernel_construct"],
		numGroups, 1, 1,
		workgroupSize, 1, 1,
		0, 0, args);
	logLaunch("construct", res_launch);
}

// The uniforms of a frame, with the insects' phases and the frame number
// cleared where the kernel does not read them: the phases move only the
// insects, and the frame number changes only the grain. The unedited view
// draws neither. Insects at speed 0 keep their phases at 0.
static void frameKey(const Uniforms& uniforms, Uniforms* key){
	memcpy(key, &uniforms, sizeof(Uniforms));
	bool unedited = uniforms.showUnedited != 0;
	if(uniforms.numInsects <= 0 || unedited) memset(key->insectPhase, 0, sizeof(key->insectPhase));
	if(!(uniforms.grain > 0.0f) || unedited) key->frameCounter = 0;
}

// True when kernel_render would draw the frame already in the frame texture.
// Besides the uniforms, only these change what it draws: the octree, during a
// load; the grids, in a rebuild (beginFrameState asks for one when they are
// reallocated or emptied); the kernel, on a hot reload; and the texture and
// buffers, on a resize. Any doubt draws the frame.
static bool sameAsLastDrawn(const Uniforms& uniforms, GLuint texture){
	if(!settings.skipUnchanged || g_drawEveryFrame || !g_lastDrawn.valid) return false;
	if(!lastBatchFinishedDevice || g_frameGridRefill || g_frameLightDirty) return false;
	if(cuda_program_render->generation != g_lastDrawn.generation) return false;
	if(texture != g_lastDrawn.texture || cptr_renderbuffer != g_lastDrawn.renderBuffer) return false;
	Uniforms key;
	frameKey(uniforms, &key);
	return memcmp(&key, &g_lastDrawn.key, sizeof(Uniforms)) == 0;
}

// kernel_render did not run: put back the requests beginFrameState took for
// this frame, for the next frame that is drawn
static void keepFrameRequests(){
	if(g_frameGridRefill) g_gridRefill = true;
	if(g_frameLightDirty) g_lightDirty = true;
	g_frameGridRefill = false;
	g_frameLightDirty = false;
}

// draw the octree with a CUDA kernel
void renderCUDA(shared_ptr<GLRenderer> renderer){

	g_frameDrawn = false;

	// minimised, or the window outgrew free memory (ensureRenderBuffer said why, once)
	if(!g_frameRenderable) return;

	Uniforms uniforms = getUniforms(renderer);

	// kernel_render reads the uniforms from constant memory (render.cu).
	// Find that symbol in the current module; a hot reload replaces the
	// module, and with it the symbol's address. By generation, not handle:
	// the old module is unloaded, and the new one can get its handle.
	static int uniformsGeneration = -1;
	static CUdeviceptr uniformsSymbol = 0;
	if(uniformsGeneration != cuda_program_render->generation){
		size_t bytes = 0;
		uniformsGeneration = cuda_program_render->generation;
		CUmodule module = cuda_program_render->mod;   // none if the kernels never compiled, which they logged
		if(module == nullptr || cuModuleGetGlobal(&uniformsSymbol, &bytes, module, "uniforms") != CUDA_SUCCESS || bytes != sizeof(Uniforms)){
			uniformsSymbol = 0;
			if(module != nullptr) logLine("[host] the render kernel has no uniforms of %zu bytes (found %zu); frames are skipped\n", sizeof(Uniforms), bytes);
		}
	}
	if(uniformsSymbol == 0){
		keepFrameRequests();
		return;
	}

	// Registered once, and again only after a resize has replaced the
	// texture (releaseFrameTexture). "Registering a resource is potentially
	// high-overhead and therefore typically called only once per resource"
	// (CUDA C++ Programming Guide 12.4, 3.2.15); it was registered and
	// unregistered on every frame. Mapping stays per frame.
	GLuint frameTexture = renderer->view.framebuffer->colorAttachments[0]->handle;
	if(g_registeredFrameTexture != frameTexture){
		releaseFrameTexture();
		CUresult registered_res = cuGraphicsGLRegisterImage(&cugl_colorbuffer, frameTexture, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
		if(registered_res != CUDA_SUCCESS){
			// mapping a resource that did not register crashes in the driver
			static bool warned = false;
			if(!warned){ warned = true; logLine("[host] cannot register the frame texture with CUDA (%d); frame skipped\n", int(registered_res)); }
			keepFrameRequests();
			return;
		}
		g_registeredFrameTexture = frameTexture;
	}

	// Nothing that reaches the kernel has changed since the frame in the
	// texture was drawn, so that frame stays. A still view was drawn again
	// on every frame, keeping the GPU busy for most of each frame whenever
	// full detail cost more than the budget.
	CUfunction kernel = cuda_program_render->kernels["kernel_render"];
	bool kept = sameAsLastDrawn(uniforms, frameTexture);
	g_keptShare += 0.02f * ((kept ? 1.0f : 0.0f) - g_keptShare);   // about the last 50 frames
	if(kept){
		g_frameDrawn = true;   // the texture holds this frame
		return;
	}

	// map OpenGL resources to CUDA
	vector<CUgraphicsResource> dynamic_resources = {cugl_colorbuffer};
	cuGraphicsMapResources(static_cast<int>(dynamic_resources.size()), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));

	CUDA_RESOURCE_DESC res_desc = {};
	res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
	cuGraphicsSubResourceGetMappedArray(&res_desc.res.array.hArray, cugl_colorbuffer, 0, 0);
	CUsurfObject output_surf;
	cuSurfObjectCreate(&output_surf, &res_desc);

	int workgroupSize = 256;   // kernel_render's __launch_bounds__ in render.cu assumes 256
	
	int maxActiveBlocksPerSM = 0;   // on failure the launch below fails, and is logged
	cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, kernel, workgroupSize, 0);
	
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

	
	// On the stream the kernel runs on, so the copy follows the previous
	// frame's kernel and precedes this one. From pageable memory the call
	// returns once the bytes are staged, so the local may then go out of scope.
	cuMemcpyHtoDAsync(uniformsSymbol, &uniforms, sizeof(Uniforms), 0);

	auto res_launch = cuLaunchCooperativeKernel(kernel,
		numGroups, 1, 1,
		workgroupSize, 1, 1,
		0, 0, args);

	g_frameDrawn = res_launch == CUDA_SUCCESS;
	g_lastDrawn.valid = g_frameDrawn;
	if(g_frameDrawn){
		frameKey(uniforms, &g_lastDrawn.key);
		g_lastDrawn.generation   = cuda_program_render->generation;
		g_lastDrawn.texture      = frameTexture;
		g_lastDrawn.renderBuffer = cptr_renderbuffer;
	}
	if(g_frameDrawn && g_frameGridRefill) g_refillGen++;   // the bird's-eye view follows the grid
	logLaunch("render", res_launch);
	if(!g_frameDrawn) keepFrameRequests();

	cuSurfObjectDestroy(output_surf);
	cuGraphicsUnmapResources(static_cast<int>(dynamic_resources.size()), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));
}

// compile kernels and allocate buffers
void initCudaProgram(shared_ptr<GLRenderer> renderer){

	// allocate most gpu buffers
	uint64_t nodesCapacity           = 200'000;
	uint64_t estimatedNodeSize       = 200;           // see struct Node in progressive_octree.cu, but some more just in case
	// What the device had free before any of this. Everything else that is
	// already on the GPU (another SimLOD, a browser) shrinks the octree pool.
	size_t freeAtStart = 0;
	size_t totalMem = 0;
	cuMemGetInfo(&freeAtStart, &totalMem);

	// kernel_construct's scratch, and nothing else's: the render kernel's
	// emission, bloom and volumetric buffers live in cptr_renderbuffer. Sized
	// from the kernel's own capacities, 405 MB plus a chunk queue entry per
	// point chunk the pool could hold (total VRAM bounds the pool: 4 MB on an
	// 8 GB card). It was a fixed 1.2 GB, and before that 300 MB, which the
	// kernel overran by 109 MB with no check. The kernel now refuses to run if
	// its layout outgrows this.
	uint64_t cptr_buffer_bytes       = CONSTRUCT_SCRATCH_FIXED_BYTES + 8 * (totalMem / CHUNK_BYTES + 1);
	uint64_t cptr_nodes_bytes        = nodesCapacity * estimatedNodeSize;

	momentaryBufferCapacity = cptr_buffer_bytes;

	// Unchecked, a failed allocation left a null pointer that the kernels then
	// wrote through. Fail at startup with the reason in the log instead.
	cuCheck(cuMemAlloc(&cptr_buffer                , cptr_buffer_bytes), "cuMemAlloc momentary buffer");
	cuCheck(cuMemAlloc(&cptr_nodes                 , cptr_nodes_bytes), "cuMemAlloc nodes");
	// Sized before the octree pool takes the rest, for the largest frame
	// likely: a fixed --render size, or else the window (renderer->width is
	// not set until the render loop starts), the largest monitor it could be
	// maximised on, and at least 1080p. A 4K display gets its 405 MB here
	// rather than from the reserve on the first frame.
	int frameW = renderer->renderWidth, frameH = renderer->renderHeight;
	if(frameW <= 0 || frameH <= 0){
		glfwGetWindowSize(renderer->window, &frameW, &frameH);
		if(uint64_t(frameW) * uint64_t(frameH) < 1920ull * 1080ull){ frameW = 1920; frameH = 1080; }
		int numMonitors = 0;
		GLFWmonitor** monitors = glfwGetMonitors(&numMonitors);
		for(int i = 0; i < numMonitors; i++){
			const GLFWvidmode* mode = glfwGetVideoMode(monitors[i]);
			if(mode && uint64_t(mode->width) * uint64_t(mode->height) > uint64_t(frameW) * uint64_t(frameH)){
				frameW = mode->width;
				frameH = mode->height;
			}
		}
	}
	if(!ensureRenderBuffer(frameW, frameH)) ensureRenderBuffer(1920, 1080);
	uint64_t cptr_renderbuffer_bytes = renderBufferCapacity;
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

	cuda_program_update = new CudaModularProgram({
		.modules = {
			"./modules/progressive_octree/progressive_octree_voxels.cu",
		},
		.kernels = {"kernel_construct"}
	});

	cuda_program_reset = new CudaModularProgram({
		.modules = {
			"./modules/progressive_octree/reset.cu",
		},
		.kernels = {"kernel"}
	});

	cuda_program_render = new CudaModularProgram({
		.modules = {
			"./modules/progressive_octree/render.cu",
		},
		.kernels = {"kernel_render"}
	});

	// The octree pool gets the VRAM that is left, so it is sized only now that
	// the kernels are compiled. Their local memory (register spills, stack) is
	// claimed from the same VRAM when they are first launched, for every
	// thread the GPU can hold at once, and a launch fails if it is not there.
	size_t stackLimit = 0;
	int maxThreadsPerSM = 0;
	cuCtxGetLimit(&stackLimit, CU_LIMIT_STACK_SIZE);
	cuDeviceGetAttribute(&maxThreadsPerSM, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device);
	int maxLocalBytes = 0;
	for(CudaModularProgram* program : {cuda_program_update, cuda_program_reset, cuda_program_render}){
		for(auto& [kernelName, kernel] : program->kernels){
			int localBytes = 0;
			int numRegisters = 0;
			cuFuncGetAttribute(&localBytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);
			cuFuncGetAttribute(&numRegisters, CU_FUNC_ATTRIBUTE_NUM_REGS, kernel);
			logLine("[kernel] %s: %d registers, %d bytes local memory per thread\n",
				kernelName.c_str(), numRegisters, localBytes);
			maxLocalBytes = std::max(maxLocalBytes, localBytes);
		}
	}
	uint64_t localMemBytes = (uint64_t(maxLocalBytes) + stackLimit) * uint64_t(maxThreadsPerSM) * uint64_t(numSMs);

	// Everything but a fixed reserve, instead of 80% of what was free, which
	// left about 1 GB idle on an 8 GB card. The reserve is only what is left
	// over, not set aside for anything in particular, and it is shared by:
	// - the occupancy and light grids: about 21 MB each at the default
	//   resolution of 256; at 512, 156-166 MB each for the Rotokare scans but
	//   up to 537 MB for a cubic cloud. Both step down or switch off when
	//   memory is short.
	// - the render buffer growing with the window after startup (1080p to 4K
	//   adds 280 MB; a window that starts at 4K is sized before the pool)
	// - GL framebuffers, and the kernels' local memory.
	size_t availableMem = 0;
	cuMemGetInfo(&availableMem, &totalMem);

	// Keep back the reserve, but never more than half of what is free, so the
	// pool grows steadily with free memory (another GPU program may hold some).
	uint64_t reserveBytes = VRAM_RESERVE_BYTES + localMemBytes;
	size_t keepBack = size_t(std::min<uint64_t>(reserveBytes, availableMem / 2));
	size_t cptr_buffer_persistent_bytes = availableMem - keepBack;
	persistentBufferCapacity = cptr_buffer_persistent_bytes;
	cuCheck(cuMemAlloc(&cptr_buffer_persistent, cptr_buffer_persistent_bytes), "cuMemAlloc persistent buffer");
	if(cptr_buffer_persistent_bytes < 300'000'000){
		// kernel_construct stops 200 MB short of the end
		// 17.4 B a point measured for the DarkCanopy scan (commit 3962fbc);
		// 30 B, the estimate before points were stored in 12 bytes, gave
		// about half the real capacity
		logLine("[WARN] only %.0f MB of VRAM for the octree: room for about %.1f M points. Close other programs using the GPU.\n",
			cptr_buffer_persistent_bytes / 1e6,
			cptr_buffer_persistent_bytes > 200'000'000 ? (cptr_buffer_persistent_bytes - 200'000'000) / 17.4 / 1e6 : 0.0);
	}

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
		" Octree pool %.0f MB: the %.0f MB then free, less %.0f MB reserve (%.0f MB of it kernel local memory)\n",
		freeAtStart / 1e6, totalMem / 1e6,
		cptr_buffer_bytes / 1e6, cptr_nodes_bytes / 1e6, cptr_renderbuffer_bytes / 1e6,
		(BATCH_STREAM_SIZE * cptr_points_bytes) / 1e6,
		cptr_buffer_persistent_bytes / 1e6, availableMem / 1e6,
		double(availableMem - cptr_buffer_persistent_bytes) / 1e6, localMemBytes / 1e6);

	size_t freeAfterInit = 0;
	cuMemGetInfo(&freeAfterInit, &totalMem);
	logLine("[vram] %.0f MB free after allocating the octree pool\n", freeAfterInit / 1e6);
}

static bool isPointFile(const string& path){
	return iEndsWith(path, ".las") || iEndsWith(path, ".laz") || iEndsWith(path, ".simlod");
}

// Why a point file cannot be loaded, or "" when it can.
static string pointFileProblem(const string& path){
	std::error_code ec;
	if(!fs::is_regular_file(path, ec)) return "not found";
	if(iEndsWith(path, ".las") || iEndsWith(path, ".laz")) return lasFileProblem(path);
	if(iEndsWith(path, ".simlod")) return simlodFileProblem(path);
	return "not a .las, .laz or .simlod file";
}

// The files that can be loaded. Each one that cannot is logged, and *refusal
// says which, for the screen. A cut LAZ file used to crash the viewer inside
// laszip, a cut LAS file loaded copies of other points, and a missing file was
// left out without a message.
static vector<string> loadablePointFiles(const vector<string>& files, string* refusal){
	vector<string> loadable;
	int numRefused = 0;
	refusal->clear();
	for(const string& file : files){
		string problem = pointFileProblem(file);
		if(problem.empty()){
			loadable.push_back(file);
			continue;
		}
		logLine("[load] not loading %s: %s\n", file.c_str(), problem.c_str());
		if(numRefused++ == 0) *refusal = fs::path(file).filename().string() + ": " + problem;
	}
	if(numRefused > 1)       *refusal = std::to_string(numRefused) + " files not loaded (see the log), " + *refusal;
	else if(numRefused == 1) *refusal = "Not loaded - " + *refusal;
	return loadable;
}

void reload(){

	printfmt("start loading \n");

	loadStart = now();
	totalUpdateDuration     = 0.0f;

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
	g_pointsUnreadable = 0;
	numPointsLoaded = 0;
	numBatchesTotal = 0;
	numBytesTotal   = 0;
	numBytesLoaded  = 0;
	lazInBatches = false;
	uint64_t numPointsInFiles = 0;   // before --set keep thins them

	// query bounding box and number of points in input files
	// vector<PointBatch> batches;
	string refusal;
	for(string path : loadablePointFiles(paths, &refusal)){

		std::error_code ec;
		uint64_t fileSize = fs::file_size(path, ec);
		if(ec) continue;   // gone since the check
		numBytesTotal += fileSize;

		if(iEndsWith(path, ".laz")){
			lazInBatches = true;
		}

		if(iEndsWith(path, ".las") || iEndsWith(path, ".laz")){

			LasHeader header = loadHeader(path);
			numPointsTotal += keptBefore(header.numPoints);
			numPointsInFiles += header.numPoints;

			// Rounded outward, so the float box contains every point the
			// header bounds do: the octree stores points as fixed point
			// within this box and clamps anything outside onto its faces.
			// Rounding to nearest could cut off up to half a float spacing,
			// 0.25 m for georeferenced northings near 5.6e6.
			auto floorFloat = [](double v){ float f = float(v); return double(f) > v ? std::nextafter(f, -INFINITY) : f; };
			auto ceilFloat  = [](double v){ float f = float(v); return double(f) < v ? std::nextafter(f,  INFINITY) : f; };

			boxMin.x = std::min(boxMin.x, floorFloat(header.min[0]));
			boxMin.y = std::min(boxMin.y, floorFloat(header.min[1]));
			boxMin.z = std::min(boxMin.z, floorFloat(header.min[2]));

			boxMax.x = std::max(boxMax.x, ceilFloat(header.max[0]));
			boxMax.y = std::max(boxMax.y, ceilFloat(header.max[1]));
			boxMax.z = std::max(boxMax.z, ceilFloat(header.max[2]));

			for (uint64_t first = 0; first < header.numPoints; first += MAX_BATCH_SIZE) {
				PointBatch batch;
				batch.file = path;
				batch.first = first;
				batch.count = static_cast<int>(std::min(header.numPoints - first, MAX_BATCH_SIZE));
				batch.lasHeader = header;

				batchesToProcess.push_back(batch);
				numBatchesTotal++;
			}

		}else if(iEndsWith(path, ".simlod") && fileSize >= 24){

			auto buffer = readBinaryFile(path, 0, 24);
			uint64_t numPoints = (fileSize - 24) / 16;

			boxMin.x = std::min(boxMin.x, buffer->get<float>( 0));
			boxMin.y = std::min(boxMin.y, buffer->get<float>( 4));
			boxMin.z = std::min(boxMin.z, buffer->get<float>( 8));

			boxMax.x = std::max(boxMax.x, buffer->get<float>(12));
			boxMax.y = std::max(boxMax.y, buffer->get<float>(16));
			boxMax.z = std::max(boxMax.z, buffer->get<float>(20));

			numPointsTotal += numPoints;   // .simlod is not thinned
			numPointsInFiles += numPoints;

			for(uint64_t first = 0; first < numPoints; first += MAX_BATCH_SIZE){
				PointBatch batch;
				batch.file = path;
				batch.first = first;
				batch.count = static_cast<int>(std::min(numPoints - first, MAX_BATCH_SIZE));

				batchesToProcess.push_back(batch);
				numBatchesTotal++;
			}
		}

		

		
	}

	if(!refusal.empty()) notice("%s", refusal.c_str());

	// Markers stay put in FILE coordinates. Each cloud is shifted by its own
	// minimum corner, so a marker kept in the renderer frame used to land in a
	// different real place whenever a cloud with other bounds was dropped.
	// They are moved from the corner of the last cloud that loaded, not of the
	// last reload: a reload that found no cloud would otherwise break the chain.
	bool haveBox = std::isfinite(boxMin.x) && std::isfinite(boxMin.y) && std::isfinite(boxMin.z);
	if(haveBox && std::isfinite(g_markerFrameMin.x)){
		for(Marker& m : g_markers){
			m.x += g_markerFrameMin.x - boxMin.x;
			m.y += g_markerFrameMin.y - boxMin.y;
			m.z += g_markerFrameMin.z - boxMin.z;
		}
	}
	if(haveBox) g_markerFrameMin = boxMin;

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
	lastBatchFinishedDevice = false;
	loadStoppedAtCapacity   = false;
	batchStreamUploadIndex  = 0;

	if(numPointsTotal > 0){
		logLine("[load] start: %.2f M points in %d batches, %.2f GB on disk%s\n",
			numPointsTotal / 1e6, numBatchesTotal.load(), numBytesTotal / 1e9,
			lazInBatches ? " (LAZ)" : "");
		if(numPointsTotal != numPointsInFiles){
			logLine("[load] keeping %.1f%% of the %.2f M points in the file(s), evenly spaced (--set keep)\n",
				loadKeepPer10k / 100.0, numPointsInFiles / 1e6);
		}
	}
}

// The construct kernel stops inserting once the octree pool is nearly full.
// The loaders went on decoding up to MAX_DECODED_AHEAD more batches that would
// never be used, each holding a 16 MB pinned slot: about 50 M points, minutes
// of LAZ decoding. Drop the queued work instead and mark the load complete.
void stopLoadingAtCapacity(){
	{
		lock_guard<mutex> lock_batchesToProcess(mtx_batchesToProcess);
		batchesToProcess.clear();
	}

	// The uploader takes mtx_uploader on every pass while it works, and
	// std::mutex is not fair, so ask it to back off first, as reset() does.
	resetInProgress = true;
	{
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
	resetInProgress = false;
}

// One-off, after the last batch: everything the octree build produced, so
// capacity can be planned from measurements rather than estimates.
void logLoadSummary(){
	// synchronous on the default stream, so the counters are the final ones
	Stats s;
	if(cuMemcpyDtoH(&s, cptr_stats, sizeof(Stats)) != CUDA_SUCCESS) return;

	double seconds = now() - loadStart;
	double inserted = double(s.numPointsProcessed);

	logLine("[load] done in %.1f s: %.2f M of %.2f M points inserted (%.1f M/s), %.2f M decoded%s\n",
		seconds, inserted / 1e6, numPointsTotal / 1e6, inserted / 1e6 / std::max(seconds, 1e-3),
		numPointsLoaded / 1e6,
		loadStoppedAtCapacity ? ". Stopped: the octree pool is full, the remaining points are not shown" : "");
	logLine("[load] octree: %u nodes (%u inner, %u leaves, %u non-empty), %.2f M points in leaves, %.2f M voxels (%.2f per point), %u point chunks, %u voxel chunks\n",
		s.numNodes, s.numInner, s.numLeaves, s.numNonemptyLeaves,
		s.numPoints / 1e6, s.numVoxels / 1e6, double(s.numVoxels) / std::max(double(s.numPoints), 1.0),
		s.numChunksPoints, s.numChunksVoxels);
	size_t freeNow = 0;
	size_t totalNow = 0;
	cuMemGetInfo(&freeNow, &totalNow);
	logLine("[load] memory: octree pool %.0f of %.0f MB used, %.1f bytes per point; construct scratch %.0f of %.0f MB; %.0f MB of VRAM free\n",
		s.allocatedBytes_persistent / 1e6, persistentBufferCapacity / 1e6,
		double(s.allocatedBytes_persistent) / std::max(double(s.numPoints), 1.0),
		s.allocatedBytes_momentary / 1e6, momentaryBufferCapacity / 1e6,
		freeNow / 1e6);
	logLine("[load] integrity: %u points dropped (no chunk), %u upload-ring slots refilled while still in use, %u points outside the header box (stored on its faces)\n",
		s.numDroppedPoints, s.numRingMismatches, s.numClampedPoints);
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
	bool resetDone = resetCUDA(renderer);
	cuCheck(cuCtxSynchronize(), "cuCtxSynchronize");

	if(resetDone){
		// read stats just to make sure device and host are on the same page
		// (sync before the memcpy, or this copies the previous, stale contents)
		cuMemcpyDtoHAsync(h_stats_pinned, cptr_stats, sizeof(Stats), ((CUstream)CU_STREAM_DEFAULT));
		cuCheck(cuCtxSynchronize(), "cuCtxSynchronize");
		memcpy(&stats, h_stats_pinned, sizeof(Stats));

		// finally, reset host state to start reloading current data set from disk
		reload();
	}else{
		// The device kept the previous load's counters, from which the
		// construct kernel would have read stale upload slots. The cloud
		// already loaded stays.
		notice("The octree could not be reset (see the log) - keeping the current cloud");
	}

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
			// in kept points: with --set keep a batch holds that fraction of
			// MAX_BATCH_SIZE but still fills a whole pinned slot, and the
			// unscaled limit let the loaders pin 1/keep times as many batches
			bool processingLagsBehind = numPointsLoaded > pointsProcessedHost + keptBefore(MAX_DECODED_AHEAD * MAX_BATCH_SIZE);

			// if (processingLagsBehind) {
			// 	printfmt("processing lags behind\n");
			// }

			if (everythingIsDone || processingLagsBehind || resetInProgress.load()) {
				// Nothing left to load until the next drop: sleep longer, and
				// free the LAS read buffer. The loaders used to wake 1000
				// times a second each for the whole session.
				if(everythingIsDone && !resetInProgress.load()){
					releaseLasReadBuffer();
					std::this_thread::sleep_for(20ms);
				}else{
					std::this_thread::sleep_for(1ms);
				}
				continue;
			}

			// lock thread's own mutex
			unique_lock<mutex> lock_loader(*mtx_loader[i]);

			PointBatch batch;

			mtx_batchesToProcess.lock();
			// get batch to load, if load queue isn't full
			if (batchesToProcess.size() > 0) {
				const PointBatch& next = batchesToProcess.front();

				// LAZ decoding is CPU-bound, so cap the threads loading at once.
				// Tested and counted under this lock, so the cap is exact. Capped
				// threads used to retry without sleeping and kept this mutex busy
				// for the threads that were decoding.
				if(next.count > 0 && iEndsWith(next.file, ".laz") && numThreadsLoading >= uint64_t(maxLazThreads)){
					mtx_batchesToProcess.unlock();
					std::this_thread::sleep_for(1ms);
					continue;
				}

				batch = next;
				batchesToProcess.pop_front();

				if(batch.count > 0) numThreadsLoading++;
			}
			mtx_batchesToProcess.unlock();
			bool tookBatch = batch.count > 0;

			if (tookBatch) {
				// load points in batch

				// Taken only with work in hand. Idle loaders took and returned a
				// slot every millisecond, and allocated a new one whenever the
				// pool happened to be empty.
				PinnedMemorySlot pinnedMemSlot = pinnedMemPool.acquire();
				Point* pinnedPoints = (Point*)pinnedMemSlot.memLocation;

				if(iEndsWith(batch.file, ".las")){
					void* target = (void*)pinnedPoints;

					double translation[3] = {-boxMin.x, -boxMin.y, -boxMin.z};
					if(!loadLasNative(batch.file, batch.lasHeader, batch.first, batch.count, target, translation)){
						logLine("[load] cannot read points %llu to %llu of %s; skipped\n",
							(unsigned long long)batch.first, (unsigned long long)(batch.first + batch.count), batch.file.c_str());
						skipUnreadableBatch(batch);
						pinnedMemPool.release(pinnedMemSlot);
						numThreadsLoading--;
						continue;
					}
					numBytesLoaded += batch.count * batch.lasHeader.bytesPerPoint;
					if(loadKeepPer10k < 10000){   // --set keep: compact what is kept
						int kept = 0;
						for(int k = 0; k < batch.count; k++){
							if(keepPoint(uint64_t(batch.first) + k)) pinnedPoints[kept++] = pinnedPoints[k];
						}
						batch.count = kept;
					}

					batch.pinnedMem = pinnedMemSlot;

					numPointsLoaded += batch.count;

					lock_guard<mutex> lock_batchesInPinnedMemory(mtx_batchesInPinnedMemory);
					batchesInPinnedMemory.push_back(batch);

				}else if(iEndsWith(batch.file, ".laz")){
					laszip_POINTER laszip_reader = nullptr;
					laszip_header* header = nullptr;
					laszip_point* laz_point = nullptr;

					laszip_BOOL is_compressed;
					laszip_BOOL request_reader = true;

					// One reader per batch, destroyed at the end: it was created and
					// never destroyed, leaking a reader for every batch of every load.
					if(laszip_create(&laszip_reader) != 0 || laszip_reader == nullptr){
						logLine("[load] laszip_create failed for %s; %d points skipped\n", batch.file.c_str(), batch.count);
						skipUnreadableBatch(batch);
						pinnedMemPool.release(pinnedMemSlot);
						numThreadsLoading--;
						continue;
					}
					laszip_request_compatibility_mode(laszip_reader, request_reader);
					if(laszip_open_reader(laszip_reader, batch.file.c_str(), &is_compressed) != 0){
						laszip_CHAR* err = nullptr;
						laszip_get_error(laszip_reader, &err);
						logLine("[load] cannot open %s: %s; %d points skipped\n", batch.file.c_str(), err ? err : "unknown laszip error", batch.count);
						laszip_destroy(laszip_reader);
						skipUnreadableBatch(batch);
						pinnedMemPool.release(pinnedMemSlot);
						numThreadsLoading--;
						continue;
					}

					laszip_get_header_pointer(laszip_reader, &header);
					laszip_get_point_pointer(laszip_reader, &laz_point);

					// A failed seek or read leaves the whole batch out: the
					// points it gave would not be the file's.
					bool readFailed = laszip_seek_point(laszip_reader, batch.first) != 0;

					int kept = 0;
					for (int i = 0; i < batch.count && !readFailed; i++) {
						double XYZ[3];
						if(laszip_read_point(laszip_reader) != 0){
							readFailed = true;
							break;
						}
						if(!keepPoint(uint64_t(batch.first) + i)) continue;   // --set keep
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


						pinnedPoints[kept++] = point;
					}
					if(readFailed){
						laszip_CHAR* err = nullptr;
						laszip_get_error(laszip_reader, &err);
						logLine("[load] cannot read points %llu to %llu of %s: %s; skipped\n",
							(unsigned long long)batch.first, (unsigned long long)(batch.first + batch.count), batch.file.c_str(),
							err ? err : "unknown laszip error");
						laszip_close_reader(laszip_reader);
						laszip_destroy(laszip_reader);
						skipUnreadableBatch(batch);
						pinnedMemPool.release(pinnedMemSlot);
						numThreadsLoading--;
						continue;
					}
					batch.count = kept;

					laszip_close_reader(laszip_reader);
					laszip_destroy(laszip_reader);

					batch.pinnedMem = pinnedMemSlot;

					numPointsLoaded += batch.count;

					lock_guard<mutex> lock_batchesInPinnedMemory(mtx_batchesInPinnedMemory);
					batchesInPinnedMemory.push_back(batch);
				}else if(iEndsWith(batch.file, ".simlod")){
					
					batch.pinnedMem = pinnedMemSlot;

					// At least on windows, this uses winapi to do unbuffered loading to maximize SSD perf
					uint64_t padding = 0;
					if(!loadFileNative(batch.file, 24llu + 16llu * uint64_t(batch.first), 16llu * batch.count, pinnedMemSlot.memLocation, &padding)){
						logLine("[load] cannot read points %llu to %llu of %s; skipped\n",
							(unsigned long long)batch.first, (unsigned long long)(batch.first + batch.count), batch.file.c_str());
						skipUnreadableBatch(batch);
						pinnedMemPool.release(pinnedMemSlot);
						numThreadsLoading--;
						continue;
					}
					batch.pinnedMem.memOffset = padding;

					numBytesLoaded += 16llu * batch.count;
					numPointsLoaded += batch.count;

					lock_guard<mutex> lock_batchesInPinnedMemory(mtx_batchesInPinnedMemory);
					batchesInPinnedMemory.push_back(batch);
				}

				numThreadsLoading--;
			}

			// An empty queue stays empty until the next load. Not while
			// holding this loader's mutex, which reset() waits for.
			lock_loader.unlock();
			std::this_thread::sleep_for(tookBatch ? 1ms : 20ms);
		}

		});
	t.detach();
}

// Spawns a thread that uploads data to the GPU
// - Asynchronously schedules uploads in stream_upload
// - After every async copy, it also asynchronously updates the number of uploaded points
void spawnUploader() {

	thread t([]() {
		installThreadTerminateHandler();

		vector<PinnedMemorySlot> availableSlots;

		while (true) {

			// go to sleep if reset is in progress
			if (resetInProgress.load()) {
				std::this_thread::sleep_for(1ms);
				continue;
			}

			// Every batch uploaded and no upload in flight: nothing to do
			// until the next load. This thread used to spin here, keeping a
			// core busy for the whole session.
			if (batchStreamUploadIndex == numBatchesTotal) {
				bool uploadsInFlight;
				{
					lock_guard<mutex> lock_pinnedMemoryInUpload(mtx_pinnedMemoryInUpload);
					uploadsInFlight = !pinnedMemoryInUpload.empty();
				}
				if (!uploadsInFlight) {
					std::this_thread::sleep_for(20ms);
					continue;
				}
			}

			// Spin for 100 us between passes, to avoid attempting locks too
			// often: sleep_for(100us) seems to wait way, way longer. Outside
			// the lock, which reset() and stopLoadingAtCapacity() wait for.
			double t_start = now();
			while (now() < t_start + 0.0001) {
			}

			// this lock ensures that we don't reset and upload at the same time
			lock_guard<mutex> lock_uploader(mtx_uploader);

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

			if(byteSize > 0) cuMemcpyHtoDAsync(target, source, byteSize, stream_upload);   // --set keep can empty a short batch

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
}

// ---- screenshots from the command line ------------------------------------
// --shots <list>: once the cloud has loaded, visit each camera in the list,
// let the frame settle, and save it. --screenshot <file.bmp> saves the start
// camera's frame. With --exit the app quits after the last one. For comparing
// the look before and after a change, at a fixed size (--render).
struct ShotRequest{
	string path;                             // .bmp to write
	bool   hasView = false;
	double view[6] = {0, 0, 0, 0, 0, 0};     // yaw pitch radius tx ty tz, as --view
};

// The rendered frame, without the UI, as a 24-bit .bmp. GL rows run bottom to
// top, as .bmp rows do, so they are written in order.
static bool saveFrameBmp(GLRenderer& renderer, const string& path){
	auto& texture = renderer.view.framebuffer->colorAttachments[0];
	int w = texture->width;
	int h = texture->height;
	if(w <= 0 || h <= 0) return false;

	vector<uint8_t> rgba(size_t(w) * size_t(h) * 4);
	glGetTextureImage(texture->handle, 0, GL_RGBA, GL_UNSIGNED_BYTE, GLsizei(rgba.size()), rgba.data());

	int rowBytes = (w * 3 + 3) & ~3;
	uint32_t dataBytes = uint32_t(rowBytes) * uint32_t(h);
	uint8_t header[54] = {'B', 'M'};
	auto put32 = [&](int at, uint32_t v){ memcpy(header + at, &v, 4); };
	auto put16 = [&](int at, uint16_t v){ memcpy(header + at, &v, 2); };
	put32( 2, 54 + dataBytes);   // file size
	put32(10, 54);               // pixel data offset
	put32(14, 40);               // BITMAPINFOHEADER
	put32(18, uint32_t(w));
	put32(22, uint32_t(h));      // positive: rows bottom to top
	put16(26, 1);
	put16(28, 24);
	put32(34, dataBytes);

	std::ofstream out(path, std::ios::binary);
	if(!out) return false;
	out.write((const char*)header, 54);
	vector<uint8_t> row(rowBytes, 0);
	for(int y = 0; y < h; y++){
		const uint8_t* src = &rgba[size_t(y) * size_t(w) * 4];
		for(int x = 0; x < w; x++){
			row[3 * x + 0] = src[4 * x + 2];
			row[3 * x + 1] = src[4 * x + 1];
			row[3 * x + 2] = src[4 * x + 0];
		}
		out.write((const char*)row.data(), rowBytes);
	}
	return bool(out);
}

int main(int argc, char** argv){

	// ---- last-words handlers ---------------------------------------------
	// A hard crash on Windows kills the process silently. The structured-
	// exception filter (CrashHandler.cpp) catches access violations, divide
	// by zero and the like on every thread and writes a symbolised stack
	// trace to the run log. The old SIGSEGV handler only covered the main
	// thread and could not say where the fault was.
	{
		std::error_code ec;
		auto exe = std::filesystem::absolute(argc > 0 ? argv[0] : "", ec);
		if(!ec) g_exeDir = exe.parent_path();
	}
	logLine("[host] SimLOD starting\n");   // opens the run log, so its path is known
	installCrashHandler(g_logPath.c_str());
	{
		// Where things are read from and written to. Hot reload and the run
		// log both follow the working directory; started from Visual Studio
		// it is build/Release, whose modules/ is the copy made at build time.
		std::error_code ec;
		logLine("[host] working directory %s\n", std::filesystem::current_path(ec).string().c_str());
		logLine("[host] kernels from %s\n", std::filesystem::absolute("modules/progressive_octree", ec).string().c_str());
		logLine("[host] run log %s\n", g_logPath.empty() ? "(none)" : g_logPath.c_str());
	}
	// (abort() and uncaught exceptions are handled there too, with a stack)

	// --hidden: an invisible window for unattended runs; frames still render
	// offscreen, so --shots works, and nothing appears on the desktop
	for(int i = 1; i < argc; i++){
		if(string(argv[i]) == "--hidden") GLRenderer::startHidden = true;
	}

	// GLFW errors and window start-up failures go to the run log
	GLRenderer::log = [](const string& message){ logLine("[host] %s\n", message.c_str()); };
	auto renderer = make_shared<GLRenderer>();
	// Minimised, the window waits for events instead of drawing frames no one
	// sees, except while a load needs frames to insert its points.
	renderer->workPending = [](){ return !lastBatchFinishedDevice || requestReset; };
	{
		// Which GPU draws the window. On a hybrid-graphics laptop it must be
		// the NVIDIA one, or CUDA cannot register the frame texture and every
		// frame is skipped (see NvOptimusEnablement at the top of this file).
		const char* vendor   = (const char*)glGetString(GL_VENDOR);
		const char* glRender = (const char*)glGetString(GL_RENDERER);
		const char* version  = (const char*)glGetString(GL_VERSION);
		logLine("[host] OpenGL: %s, %s, %s\n", vendor ? vendor : "?", glRender ? glRender : "?", version ? version : "?");
	}

	// Fonts, before the first frame builds the atlas. The default font goes in
	// first so it stays the default for every other window. ProggyTiny gets
	// the settings AddFontDefault uses for ProggyClean, by the same designer:
	// no oversampling, glyphs snapped to whole pixels, one pixel lower.
	{
		ImGuiIO& io = ImGui::GetIO();
		io.Fonts->AddFontDefault();
		ImFontConfig cfg;
		cfg.OversampleH   = 1;
		cfg.OversampleV   = 1;
		cfg.PixelSnapH    = true;
		cfg.GlyphOffset.y = 1.0f;
		g_menuFont = io.Fonts->AddFontFromMemoryCompressedBase85TTF(ProggyTiny_compressed_data_base85, 10.0f, &cfg);
	}
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


	// --render W H is read before the GPU buffers are sized, so that the render
	// buffer for that frame size is in place before the octree pool
	for(int i = 1; i + 2 < argc; i++){
		if(string(argv[i]) == "--render"){
			renderer->renderWidth  = std::max(0, std::atoi(argv[i + 1]));
			renderer->renderHeight = std::max(0, std::atoi(argv[i + 2]));
			logLine("[host] rendering at %d x %d, scaled to the window\n", renderer->renderWidth, renderer->renderHeight);
		}
	}

	// Kernel compile and link messages, a line at a time: a compiler's log
	// can run past the 2,048 characters of one log line
	cudaProgramLog = [](const string& message){
		std::stringstream lines(message);
		string line;
		int numLines = 0;
		while(std::getline(lines, line)){
			if(++numLines > 60){ logLine("[kernels]   (the rest is on the console)\n"); break; }
			logLine("%s\n", line.c_str());
		}
	};
	initCuda();
	initCudaProgram(renderer);

	// Loaders stop taking batches once MAX_DECODED_AHEAD decoded batches wait
	// for the kernel, so at most that many slots, plus one per loader taking a
	// batch as the limit is reached, are in use. 200 slots (3.4 GB of RAM that
	// cannot be paged out) used to be reserved, of which about 70 were ever
	// used. acquire() adds a slot if the pool runs out.
	pinnedMemPool.reserveSlots(int(MAX_DECODED_AHEAD) + numThreads + 2);

	mtx_loader.reserve(numThreads);
	for(int i = 0; i < numThreads; i++){
		mtx_loader.push_back(make_unique<mutex>());
		spawnLoader(i, maxLazThreads);
	}

	cudaprint.init();
	
	spawnUploader();

	// Command line: SimLOD.exe [cloud.las|.laz|.simlod ...] [--view yaw pitch radius tx ty tz]
	// Files load as if dropped on the window; --view sets the camera afterwards
	// (target in the renderer frame, i.e. relative to the cloud's minimum corner),
	// which makes timings and screenshots reproducible.
	bool haveView = false;
	double view[6] = {0, 0, 0, 0, 0, 0};
	vector<string> argFiles;
	string argCalls;
	vector<ShotRequest> shots;
	double shotDelay = 3.0;        // seconds at each shot's camera before saving
	int benchFrames = 0;           // --bench N: time N frames at each shot's camera first
	bool exitAfterShots = false;
	for(int i = 1; i < argc; i++){
		string arg = argv[i];
		if(arg == "--view" && i + 6 < argc){
			for(int k = 0; k < 6; k++) view[k] = std::atof(argv[i + 1 + k]);
			haveView = true;
			i += 6;
		}else if(arg == "--render" && i + 2 < argc){
			i += 2;   // applied before the GPU buffers were sized
		}else if(arg == "--calls" && i + 1 < argc){
			argCalls = argv[++i];
		}else if(arg == "--screenshot" && i + 1 < argc){
			ShotRequest shot;
			shot.path = argv[++i];
			shots.push_back(shot);
		}else if(arg == "--shots" && i + 1 < argc){
			// one shot per line: name yaw pitch radius tx ty tz, written to
			// <name>.bmp beside the list; '#' starts a comment
			std::filesystem::path list = argv[++i];
			std::ifstream in(list);
			if(!in) logLine("[shot] cannot open %s\n", list.string().c_str());
			string line;
			bool firstLine = true;
			while(std::getline(in, line)){
				// a UTF-8 BOM (PowerShell's utf8) would become part of the first name
				if(firstLine && line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);
				firstLine = false;
				std::stringstream ss(line);
				ShotRequest shot;
				string name;
				if(!(ss >> name) || name[0] == '#') continue;
				for(int k = 0; k < 6; k++) ss >> shot.view[k];
				if(!ss){ logLine("[shot] %s: malformed line skipped: %s\n", list.string().c_str(), line.c_str()); continue; }
				shot.hasView = true;
				shot.path = (list.parent_path() / (name + ".bmp")).string();
				shots.push_back(shot);
			}
		}else if(arg == "--shot-delay" && i + 1 < argc){
			shotDelay = std::max(0.0, std::atof(argv[++i]));
		}else if(arg == "--exit"){
			exitAfterShots = true;
		}else if(arg == "--bench" && i + 1 < argc){
			benchFrames = std::max(0, std::atoi(argv[++i]));
		}else if(arg == "--hidden"){
			// applied before the window was created
		}else if(arg == "--look" && i + 1 < argc){
			// a saved look, as a file or by name in the looks folder; a --set
			// after it changes it further
			loadLookArg(argv[++i]);
		}else if(arg == "--set" && i + 1 < argc){
			// --set name=value, for the settings that matter most on a small GPU,
			// and for repeatable screenshots
			string kv = argv[++i];
			size_t eq = kv.find('=');
			string key = kv.substr(0, eq);
			double v = (eq == string::npos) ? 1.0 : std::atof(kv.c_str() + eq + 1);
			// Values are kept within the sliders' ranges, as a look file's
			// are: splat=1000 used to allow 1000-pixel splats against a slider
			// limit of 64.
			if     (key == "hqs")       settings.useHighQualityShading = v != 0.0;
			else if(key == "splat")     settings.maxSplatPx            = std::clamp(int(v), 2, 64);           // Max splat px
			else if(key == "detail")    settings.minNodeSize           = std::clamp(float(v), 8.0f, 256.0f);  // Detail
			else if(key == "adaptive")  settings.adaptiveDetail        = v != 0.0;
			else if(key == "budget")    settings.frameBudgetMs         = std::clamp(float(v), 10.0f, 100.0f); // Budget ms
			else if(key == "vsync")     settings.vsync                 = v != 0.0;
			else if(key == "insects")   settings.numInsects            = int(v);
			else if(key == "grain")     settings.grain                 = float(v);
			else if(key == "bloom")   { settings.bloomStrength         = float(v); settings.enableBloom = v > 0.0; }
			else if(key == "markers") { settings.showEventMarkers      = v != 0.0; settings.showMarkers = v != 0.0; }
			else if(key == "pointsize") settings.pointWorldSize        = float(v);
			else if(key == "gain")      settings.emissionGain          = float(v);
			else if(key == "haze")    { settings.fogDensity            = float(v); settings.enableFog = v > 0.0; }
			else if(key == "hazefall")  settings.fogHeightFalloff      = float(v);
			else if(key == "hazefloor") settings.fogBase               = float(v);
			else if(key == "hazesky")   settings.fogSky                = v != 0.0;
			else if(key == "light")     settings.lightOn               = v != 0.0;
			else if(key == "sunbeams")  settings.sunbeams              = float(v);
			else if(key == "whitefix")  settings.whiteFix              = v != 0.0;
			else if(key == "skycast")   settings.tintMargin            = float(v);   // Scan clean-up > Sky cast
			else if(key == "torch")     settings.torchOn               = v != 0.0;   // aims at the frame centre
			else if(key == "radius")  { g_eventRadiusArg = v > 0.0 ? std::clamp(float(v), 0.1f, 1000.0f) : 0.0f; if(v > 0.0) g_eventRadius = g_eventRadiusArg; }
			else if(key == "exposure")  settings.exposure              = float(v);
			else if(key == "unedited" || key == "native") settings.showUnedited = v != 0.0;   // 1 = the unedited view; "native" was its old name
			else if(key == "tone")      settings.toneCurve             = float(v);
			else if(key == "keep")      loadKeepPer10k = uint32_t(std::clamp(v, 0.01, 1.0) * 10000.0 + 0.5);   // fraction of each file loaded
			else logLine("[host] unknown --set key %s, ignored (hqs, splat, detail, adaptive, budget, vsync, insects, grain, bloom, markers, pointsize, gain, haze, hazefall, hazefloor, hazesky, light, sunbeams, whitefix, skycast, torch, radius, exposure, tone, unedited, keep)\n", key.c_str());
			// the keys that are look settings take the look's limits
			static const std::map<string, const char*> lookNameOf = {
				{"insects", "insects"}, {"grain", "grain"}, {"bloom", "bloom_strength"}, {"pointsize", "point_size_m"},
				{"gain", "emission_gain"}, {"haze", "depth_haze"}, {"hazefall", "haze_thins_with_height"},
				{"hazefloor", "haze_floor_m"}, {"sunbeams", "sunbeams"}, {"skycast", "sky_cast"},
				{"exposure", "exposure"}, {"tone", "tone_curve"},
			};
			auto look = lookNameOf.find(key);
			if(look != lookNameOf.end()) clampLookSetting(look->second);
		}else if(isPointFile(arg)){
			std::error_code ec;
			auto absolute = std::filesystem::absolute(arg, ec);   // dropped files are absolute too
			argFiles.push_back(ec ? arg : absolute.string());
		}else{
			logLine("[host] command-line argument not recognised, ignored: %s\n", arg.c_str());
		}
	}
	if(argc > 1){
		logLine("[host] command line: %d file(s)%s\n", int(argFiles.size()), haveView ? ", --view" : "");
		for(auto& f : argFiles) logLine("[host]   %s\n", f.c_str());
	}

	// --calls: load and play the schedule, as the Calls panel's Load and Play
	// buttons do. Bloom follows Bloom strength, as in that panel: off at the
	// unedited start, on with --set bloom=.
	if(!argCalls.empty()){
		loadCallSchedule(argCalls);
		g_showClock   = 0.0;
		g_showPlaying = g_callsLoaded;
		settings.enableBloom = settings.bloomStrength > 0.0f;
		logLine("[calls] %s: %d calls, %s\n", argCalls.c_str(), int(g_calls.size()), g_callsLoaded ? "playing" : "none loaded");
	}
	g_drawEveryFrame = benchFrames > 0;   // --bench times frames, so none is skipped
	if(benchFrames > 0 && shots.empty()){
		logLine("[shot] --bench times frames at each shot's camera, and no shot was given (--shots or --screenshot): nothing is timed\n");
	}
	if(!shots.empty()){
		logLine("[shot] %d screenshot(s) after loading, %.1f s each%s\n", int(shots.size()), shotDelay, exitAfterShots ? ", then exit" : "");
		GLRenderer::inputLocked = true;   // a stray click or scroll would move the shots' camera
	}else if(exitAfterShots){
		logLine("[shot] --exit given but there are no screenshots to take; exiting once the load finishes\n");
	}

	reload();

	// Load command-line files the way a drop does: through reset() on the
	// first frame, which locks the loader threads out first. Calling reload()
	// here with real files deadlocked against the already-running loaders.
	bool startCameraPending = false;
	if(!argFiles.empty()){
		paths = argFiles;
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

		for(auto file : files){
			logLine("[load] dropped: %s\n", file.c_str());

			if(isPointFile(file)){
				pointCloudFiles.push_back(file);
			}
		}

		// A stray CSV or folder must not clear the scene (and a reload of the
		// full scan takes minutes), nor a point file that cannot be loaded.
		bool anyPointFiles = !pointCloudFiles.empty();
		string refusal;
		pointCloudFiles = loadablePointFiles(pointCloudFiles, &refusal);
		if(pointCloudFiles.empty()){
			if(anyPointFiles) notice("%s - keeping the current cloud", refusal.c_str());
			else notice("No .las, .laz or .simlod in the drop - keeping the current cloud");
			return;
		}
		if(!refusal.empty()) notice("%s", refusal.c_str());

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

		timeSinceLastFrame = now() - lastFrameTime;

		// ---- walking (apply_navigate.ps1) --------------------------------
		// W/S forward and back along the view, A/D sideways, Q/E down and up.
		// Only when the GUI does not have the keyboard. Speed in metres per
		// second, so it is the same at any frame rate.
		{
			ImGuiIO& io = ImGui::GetIO();
			if(!io.WantCaptureKeyboard && !GLRenderer::inputLocked){
				float dt = timeSinceLastFrame > 0.0 && timeSinceLastFrame < 0.5 ? float(timeSinceLastFrame) : 0.016f;
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
			// window pixels to frame pixels through the rectangle the frame is
			// shown in: they differ when it is rendered at a fixed size (--render)
			double mx=0.0, my=0.0; glfwGetCursorPos(renderer->window,&mx,&my);
			double rectTop = double(renderer->windowHeight - (renderer->presentY + renderer->presentHeight));   // cursor y runs top down
			double fx = (mx - renderer->presentX) * double(renderer->width)  / double(std::max(renderer->presentWidth, 1));
			double fy = (my - rectTop)            * double(renderer->height) / double(std::max(renderer->presentHeight, 1));
			bool onFrame = fx >= 0.0 && fy >= 0.0 && fx < renderer->width && fy < renderer->height;
			// torch: aim through the cursor while it is over the frame; T toggles
			{
				// the key's edge is tracked even while shots lock the input, so a
				// T held as the lock lifts does not toggle either
				static bool tWasDown = false;
				bool tDown = glfwGetKey(renderer->window, GLFW_KEY_T) == GLFW_PRESS && !io.WantCaptureKeyboard;
				if(tDown && !tWasDown && !GLRenderer::inputLocked) settings.torchOn = !settings.torchOn;
				tWasDown = tDown;
				Runtime::torchMode = settings.torchOn;
				if(settings.torchOn && onFrame && !io.WantCaptureMouse && !GLRenderer::inputLocked){
					// clamped: fy can come within a pixel of height when the frame
					// is shown enlarged, and a negative y means "not aimed yet"
					g_torchPx = std::min(float(fx), float(renderer->width - 1));
					g_torchPy = std::max(0.0f, float(renderer->height - 1) - float(fy));
				}
			}
			if(((altDown && leftDown && !g_leftWasDown) || (pKey && !g_pKeyWasDown)) && !io.WantCaptureMouse && onFrame && !GLRenderer::inputLocked){
				g_pickReqX = int(fx);
				g_pickReqY = renderer->height - 1 - int(fy);
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
		lastFrameTime = now();

		if(renderer->width > 0 && renderer->height > 0){   // 0 x 0 while minimised
			auto& framebuffer = renderer->view.framebuffer;
			if(framebuffer->width != renderer->width || framebuffer->height != renderer->height){
				releaseFrameTexture();   // setSize deletes the texture and makes a new one
			}
			framebuffer->setSize(renderer->width, renderer->height);
		}

		glBindFramebuffer(GL_FRAMEBUFFER, renderer->view.framebuffer->handle);

		if(requestReset){
			reset(renderer);
			if(startCameraPending){
				applyStartCamera();   // boxSize is known now
				startCameraPending = false;
				if(numPointsTotal == 0){
					// Every file on the command line was refused (reload logged
					// why). Shots would be of empty sky, and --exit would end
					// as if all had gone well.
					if(!shots.empty()){
						logLine("[shot] nothing was loaded, so no screenshots are taken\n");
						shots.clear();
					}
					if(exitAfterShots){
						logLine("[host] --exit: nothing was loaded; exiting with code 1\n");
						GLRenderer::exitCode = 1;
						glfwSetWindowShouldClose(renderer->window, 1);
					}
				}
			}
		}

		{   // GLRenderer starts with vsync off (glfwSwapInterval(0)); follow the setting
			static int appliedSwapInterval = -1;
			int wanted = settings.vsync ? 1 : 0;
			if(wanted != appliedSwapInterval){ glfwSwapInterval(wanted); appliedSwapInterval = wanted; }
		}

		beginFrameState(renderer->width, renderer->height);
		renderCUDA(renderer);

		// ---- command-line screenshots (--shots, --screenshot) -------------
		// After the load: move to each shot's camera, wait shotDelay for the
		// occupancy grid, light and glow to settle, then save the frame just
		// rendered. A camera set here takes effect from the next frame. Mouse
		// and keys are locked out while shots run (inputLocked), but if the
		// camera still moves while a shot waits (anything but the controls'
		// own adjustment on the first frame, such as pinning the radius in
		// walk mode), the shot's camera is set again and the wait starts over.
		if(exitAfterShots && shots.empty() && lastBatchFinishedDevice){
			glfwSetWindowShouldClose(renderer->window, 1);
		}
		if(!shots.empty() && lastBatchFinishedDevice){
			static size_t shotIndex = 0;
			static int    shotPhase = 0;   // 0 set the camera, 1 note where it settled, 2 wait
			static double shotStart = 0.0;
			static double settled[6] = {0, 0, 0, 0, 0, 0};
			static double startView[6] = {0, 0, 0, 0, 0, 0};
			static bool   haveStartView = false;
			struct BenchSample{ double t[7]; uint32_t drawn; };   // ms at each stage boundary
			static vector<BenchSample> benchSamples;
			static uint32_t benchLastFrame = ~0u;
			auto& controls = renderer->controls;
			double current[6] = {controls->yaw, controls->pitch, controls->radius,
			                     controls->target.x, controls->target.y, controls->target.z};
			// the camera after the load (the start camera, or --view): shots
			// without a camera of their own (--screenshot) use it, even after
			// --shots has moved the camera
			if(!haveStartView){
				std::copy(current, current + 6, startView);
				haveStartView = true;
			}
			if(shotIndex < shots.size()){
				const ShotRequest& shot = shots[shotIndex];
				const double* view = shot.hasView ? shot.view : startView;
				if(shotPhase == 2 && !std::equal(current, current + 6, settled)){
					logLine("[shot] camera moved while waiting for %s; setting it again\n", shot.path.c_str());
					shotPhase = 0;
					benchSamples.clear();
				}
				if(shotPhase == 0){
					controls->yaw    = view[0];
					controls->pitch  = view[1];
					controls->radius = view[2];
					controls->target = { view[3], view[4], view[5] };
					shotStart = now();
					shotPhase = 1;
				}else if(shotPhase == 1){
					std::copy(current, current + 6, settled);
					shotPhase = 2;
				}else if(!g_frameDrawn){
					// minimised, or nothing could be drawn: the frame texture is
					// stale, so wait for shotDelay of frames that are drawn
					shotStart = now();
				}else if(now() - shotStart >= shotDelay && benchFrames > 0 && int(benchSamples.size()) < benchFrames){
					// --bench: time benchFrames frames at this camera before the
					// shot. The stage times come from kernel_render itself
					// (Stats::stageNs, ns since kernel start at each stage); the
					// readback lags, so each frame is taken once, by its frameID.
					if(stats.frameID != benchLastFrame && stats.stageNs[6] > 0){
						benchLastFrame = stats.frameID;
						BenchSample b;
						for(int k = 0; k < 7; k++) b.t[k] = double(stats.stageNs[k]) / 1e6;
						b.drawn = stats.numDrawn;
						benchSamples.push_back(b);
					}
				}else if(now() - shotStart >= shotDelay){
					if(!benchSamples.empty()){
						// medians, per stage: setup, points, extras, edl, volumetric, bloom, composite
						auto median = [&](auto stageOf){
							vector<double> v;
							for(const BenchSample& b : benchSamples) v.push_back(stageOf(b));
							std::sort(v.begin(), v.end());
							return v[v.size() / 2];
						};
						double total = median([](const BenchSample& b){ return b.t[6]; });
						double minTotal = 1e30;
						for(const BenchSample& b : benchSamples) minTotal = std::min(minTotal, b.t[6]);
						logLine("[bench] %s: render %.1f ms (min %.1f) | setup %.1f points %.1f extras %.1f edl %.1f volumetric %.1f bloom %.1f composite %.1f | drawn %.1f M | %d frames at %d x %d\n",
							std::filesystem::path(shot.path).stem().string().c_str(), total, minTotal,
							median([](const BenchSample& b){ return b.t[0]; }),
							median([](const BenchSample& b){ return b.t[1] - b.t[0]; }),
							median([](const BenchSample& b){ return b.t[2] - b.t[1]; }),
							median([](const BenchSample& b){ return b.t[3] - b.t[2]; }),
							median([](const BenchSample& b){ return b.t[4] - b.t[3]; }),
							median([](const BenchSample& b){ return b.t[5] - b.t[4]; }),
							median([](const BenchSample& b){ return b.t[6] - b.t[5]; }),
							benchSamples.back().drawn / 1e6, int(benchSamples.size()), renderer->width, renderer->height);
						benchSamples.clear();
						benchLastFrame = ~0u;
					}
					bool ok = saveFrameBmp(*renderer, shot.path);
					logLine("[shot] %s %s (%d x %d) camera %.3f %.3f %.3f  %.2f %.2f %.2f\n", ok ? "wrote" : "FAILED to write",
						shot.path.c_str(), renderer->width, renderer->height,
						current[0], current[1], current[2], current[3], current[4], current[5]);
					shotIndex++;
					shotPhase = 0;
					if(shotIndex == shots.size()){
						GLRenderer::inputLocked = false;   // shots done: the viewer is the user's again
						if(exitAfterShots) glfwSetWindowShouldClose(renderer->window, 1);
					}
				}
			}
		}

		if(!lastBatchFinishedDevice){
			updateOctree(renderer);
		}

		if(!lastBatchFinishedDevice){
			totalUpdateDuration = float(1000.0 * (now() - loadStart));
		}

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

			if(!lastBatchFinishedDevice && numPointsTotal > 0){
				logIfChanged("load", 2.0, "[load] %.1f s: %.1f M decoded, %.1f M uploaded, %.1f M of %.1f M inserted; octree pool %.0f of %.0f MB\n",
					now() - loadStart, numPointsLoaded / 1e6, numPointsUploaded / 1e6,
					stats.numPointsProcessed / 1e6, numPointsTotal / 1e6,
					stats.allocatedBytes_persistent / 1e6, persistentBufferCapacity / 1e6);
			}

			if(stats.memCapacityReached && !loadStoppedAtCapacity && !lastBatchFinishedDevice){
				stopLoadingAtCapacity();
			}


			{   // batches a loader could not read (skipUnreadableBatch)
				static uint64_t reportedUnreadable = 0;
				uint64_t unreadable = g_pointsUnreadable.load();
				if(unreadable != reportedUnreadable){
					reportedUnreadable = unreadable;
					if(unreadable > 0) notice("Could not read part of the input: %.2f M points skipped (see log)", double(unreadable) / 1e6);
				}
			}

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

			// ---- movable, collapsible windows ---------------------------------
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
					float w = ImGui::CalcTextSize(title, nullptr, true).x
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
				if(!nowCollapsed && collapsedState && savedSize.x > 0.0f){
					// just expanded: ImGui applied the narrow collapsed width
					// before expanding, so restore the saved size rather than
					// record the narrow one over it (as GLRenderer's Performance)
					ImGui::SetWindowSize(savedSize, ImGuiCond_Always);
				}else if(!nowCollapsed){
					ImVec2 s = ImGui::GetWindowSize();
					if(s.x > 0.0f) savedSize = s;
				}
				collapsedState = nowCollapsed;
				return open;
			};

			auto windowSize = ImVec2(490, 280);
			static bool   settingsCollapsed = false;
			static ImVec2 settingsSaved     = ImVec2(0, 0);
			// point count beside the title; ### keeps the window id fixed while it changes
			char settingsTitle[96];
			snprintf(settingsTitle, sizeof(settingsTitle), "Settings   %.1f M points loaded, %.1f M drawn###Settings",
				double(stats.numPointsProcessed) / 1e6, double(stats.numDrawn) / 1e6);
			// The 10-pixel menu font and tighter spacing give a slider row 16 pixels
			// rather than 23, so 44% more rows fit on screen. Small menu text
			// (Display and debug) turns it off.
			bool smallText = g_smallMenuText && g_menuFont != nullptr;
			if(smallText){
				ImGui::PushFont(g_menuFont);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,     ImVec2(4.0f, 2.0f));
				ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,      ImVec2(6.0f, 2.0f));
				ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, ImVec2(4.0f, 2.0f));
			}
			beginPanel(settingsTitle, ImVec2(10, 300), windowSize, settingsCollapsed, settingsSaved);

			// =====================================================================
			// Settings, in the order the image is made: the camera, how points are
			// drawn, the light on them, the air in front of them, then the grade.
			// After those come what the scene holds (calls, insects, markers), then
			// speed, clean-up of the scan, and debugging. Points, Sky and haze, and
			// Colour are open by default. Every control appears exactly once.
			// =====================================================================

			// ---- UNEDITED ---------------------------------------------------------
			// Every setting below starts at the value that changes nothing, so
			// the cloud starts as stored in the file. Reset returns there; Show
			// unedited compares the current settings with it.
			if(ImGui::Button("Reset to unedited")) resetToUnedited();
			tipFor("Returns every setting that changes how the cloud is represented to its starting value: the cloud as stored in the file, "
				   "with no light, torch, haze, sunbeams, bloom, insects, point shading, eye dome lighting, canopy repair or cull, "
				   "a neutral grade and no tone curve. Point size, round splats and high-quality shading go back to their starting values, "
				   "and the points are shown again in their stored colours (Show points on; Markers only, Colour by node and Colour by LOD off). "
				   "Kept: the camera, lens and near clip, the sky and horizon, markers and survey, calls, detail and speed, "
				   "and the bounding box, voxel grid and Update visibility.");
			ImGui::SameLine();
			ImGui::Checkbox("Show unedited", &settings.showUnedited);
			tipFor("Draws the cloud as Reset to unedited would leave it, without changing your settings: tick to compare, untick to return to them. "
				   "Point size, splat shape and high-quality shading stay as set, so the points keep their size and shape.");
			if(settings.showUnedited){
				ImGui::TextDisabled("Showing the unedited cloud: the faint settings have no effect until this is unticked.");
			}

			// ---- LOOKS ------------------------------------------------------------
			// A look is the settings that decide how the image looks, kept in a
			// text file so it outlasts the session (forEachLookSetting).
			if(ImGui::Button("Save look...")){
				snprintf(g_lookNameBuf, sizeof(g_lookNameBuf), "%s", g_lookName.empty() ? "look" : g_lookName.c_str());
				ImGui::OpenPopup("Save look");
			}
			tipFor("Saves the settings that decide how the image looks to a text file, so the look can be loaded in a later session: "
				   "Points, Light, Sky and haze, Colour, the calls' glow, bloom and medium, Insects, and Scan clean-up. "
				   "Not saved: the camera, lens and near clip, markers and survey, the call schedule, detail and speed, Volume reach, and the debug views. "
				   "The files go in Documents\\SimLOD\\looks; Load look shows the folder.");
			ImGui::SameLine();
			if(ImGui::Button("Load look...")){
				listLooks();
				ImGui::OpenPopup("Load look");
			}
			tipFor("Loads a saved look; the newest are listed first. Every setting a look holds first returns to its starting value, "
				   "so a look loads the same way whatever was set before.");
			drawSaveLookPopup();
			drawLoadLookPopup();

			// ---- CAMERA -------------------------------------------------------
			if(ImGui::CollapsingHeader("Camera")){
				drawBirdsEye(renderer);
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
				// It only set the height Go to centre used, so moving it changed
				// nothing on screen. Now the camera rises or drops to it, keeping
				// its place over the ground and where it looks.
				if(ImGui::SliderFloat("Eye height (m)",   &settings.eyeHeight,      0.2f, 10.0f, "%.2f")){
					glm::dvec3 pos = renderer->controls->getPosition();
					renderer->controls->target.z += double(settings.eyeHeight) - pos.z;
				}
				tipFor("Height of the camera above the lowest point of the cloud. Moving the slider moves the camera up or down; Go to centre uses this height too.");
				// live in both directions: dragging the slider tilts the camera now,
				// and tilting the camera with the mouse moves the slider
				if(ImGui::SliderFloat("Look up (deg)",    &settings.lookUpDeg,    -89.0f, 89.0f)){
					renderer->controls->pitch = double(settings.lookUpDeg) * 3.14159265358979 / 180.0;
				}else{
					settings.lookUpDeg = float(renderer->controls->pitch * 180.0 / 3.14159265358979);
				}
				ImGui::SliderFloat("Walk speed (m/s)", &g_walkSpeed, 0.2f, 15.0f);
				tipFor("W/S walk along the view, A/D sideways, Q/E down and up, when the mouse is over the forest rather than the panel. In Walk mode the wheel walks too.");
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
					renderer->controls->walkMode = true;
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

			// ---- POINTS -------------------------------------------------------
			if(ImGui::CollapsingHeader("Points", ImGuiTreeNodeFlags_DefaultOpen)){
				ImGui::SliderFloat("Point size (m)",   &settings.pointWorldSize, 0.0f, 0.30f, "%.3f");
				tipFor("Splat diameter in metres, projected by depth. Near points large, far ones small.");
				{
					UneditedDim dim;
					ImGui::SliderFloat("Point shading",    &settings.pointShading,   0.0f, 1.0f);
					tipFor("Shades each splat as a small sphere lit from the upper left. 0 is a flat disc.");
				}
				ImGui::SliderFloat("Near clip (m)",    &settings.nearClip,       0.0f, 10.0f, "%.2f");
				tipFor("Points closer to the camera than this are not drawn, so foliage right at the lens does not fill the frame.");
				ImGui::Checkbox   ("Round splats",      &settings.roundSplats);
				ImGui::Checkbox   ("High quality shading", &settings.useHighQualityShading);
				tipFor("Blends the colours of all points within 1% of the nearest depth at each pixel, which smooths the speckle where points overlap. Draws every point twice, a depth pass and a colour pass, so it costs more than the standard path.");
				ImGui::Separator();
				{
					UneditedDim dim;
					ImGui::Checkbox   ("Eye dome lighting", &settings.enableEDL);
					ImGui::SliderFloat("EDL strength",      &settings.edlStrength, 0.0f, 3.0f);
				}
			}

			// ---- LIGHT --------------------------------------------------------
			if(ImGui::CollapsingHeader("Light")){
				UneditedDim dim;
				bool lm = false;
				lm |= ImGui::Checkbox   ("Key light on",     &settings.lightOn);
				tipFor("A directional light through the canopy, computed once per cell of the occupancy grid (Grid resolution, under Scan clean-up).");
				lm |= ImGui::SliderFloat("Azimuth",          &settings.lightAzimuth,   0.0f, 360.0f);
				lm |= ImGui::SliderFloat("Elevation",        &settings.lightElevation, 0.0f,  89.0f);
				tipFor("Where the light comes FROM. The reference is backlit and high.");
				ImGui::SliderFloat("Strength",         &settings.lightIntensity, 0.0f,   6.0f);   // scales the light at draw time: no grid rebuild
				ImGui::ColorEdit3 ("Light colour",     settings.lightColour, ImGuiColorEditFlags_NoInputs);
				ImGui::SliderFloat("Ambient",          &settings.ambient,        0.0f, 1.0f);
				tipFor("Floor under the key light, so shadowed points keep some of their own colour. The haze in shade darkens to the same level.");
				lm |= ImGui::SliderFloat("Canopy density",   &settings.lightAbsorb,    0.0f,   1.5f, "%.2f /m");
				tipFor("Light lost per metre of occupied foliage, so it means the same at every grid resolution. 0.4 lets about a seventh through five metres of canopy; 1.0 almost none.");
				if(lm) g_lightDirty = true;
				ImGui::Separator();
				ImGui::Checkbox   ("Torch (T)",        &settings.torchOn);
				tipFor("A torch held at the camera. While it is on, moving the mouse over the forest aims the beam and does not turn the camera; walk with W/A/S/D and the wheel. T toggles it.");
				ImGui::SliderFloat("Torch strength",   &settings.torchIntensity, 0.0f, 20.0f, "%.1f");
				ImGui::SliderFloat("Torch beam (deg)", &settings.torchAngle,     2.0f, 60.0f, "%.0f");
				ImGui::SliderFloat("Torch reach (m)",  &settings.torchRange,     1.0f, 60.0f, "%.0f");
				tipFor("Distance at which the beam has fallen to half strength.");
			}

			// ---- SKY AND HAZE -------------------------------------------------
			if(ImGui::CollapsingHeader("Sky and haze", ImGuiTreeNodeFlags_DefaultOpen)){
				ImGui::ColorEdit3 ("Sky colour",       settings.skyColour, ImGuiColorEditFlags_NoInputs);
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
				UneditedDim dim;   // the haze and everything below it; the sky colours above still apply
				ImGui::SliderFloat("Depth haze",       &settings.fogDensity,     0.0f, 0.08f, "%.5f",
						ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
				tipFor("Distance blends toward the haze colour. The reference goes LIGHTER with distance, so keep the colour pale. The value is the haze's density per metre at and below the haze floor.");
				if(settings.fogDensity <= 0.0f) settings.enableFog = false; else settings.enableFog = true;
				ImGui::SliderFloat("Haze thins with height", &settings.fogHeightFalloff, 0.0f, 0.5f, "%.3f /m", ImGuiSliderFlags_Logarithmic);
				tipFor("0 keeps the haze even. Above 0 the haze thins with height above the haze floor, halving every 0.69 / value metres: at 0.1 every 7 m, so it lies low among the trunks and clears toward the canopy.");
				ImGui::SliderFloat("Haze floor (m)",   &settings.fogBase,        0.0f, boxSize.z > 0.0f ? boxSize.z : 40.0f, "%.1f");
				tipFor("Height above the lowest point up to which the haze keeps its full density. It only thins above this, so it only matters when Haze thins with height is above 0.");
				ImGui::Checkbox   ("Haze over sky",    &settings.fogSky);
				tipFor("The haze also lies between the camera and the sky, so gaps between the trunks and the horizon show haze rather than sky. Use it with Haze thins with height: even haze has no top, so it covers the whole sky.");
				{
					float hc[3] = {settings.fogR, settings.fogG, settings.fogB};
					if(ImGui::ColorEdit3("Haze colour", hc, ImGuiColorEditFlags_NoInputs)){
						settings.fogR = hc[0]; settings.fogG = hc[1]; settings.fogB = hc[2];
					}
					tipFor("Picked like the sky colour, so the same values in both match. With the key light on, haze in shade darkens to the Ambient level (Light), as points do, and Sunbeams, below, adds the light that reaches it.");
				}
				ImGui::Separator();
				ImGui::SliderFloat("Sunbeams",         &settings.sunbeams,       0.0f,   4.0f);
				tipFor("The key light (Light) scattered by the haze where the canopy lets it through, so the light shows in the air between the trunks. Needs Depth haze above 0 and the key light on; in front of the sky and below the horizon, Haze over sky as well. At 0 the haze gets only the ambient light. Marched at quarter resolution with Volume steps and Volume reach, below; more steps give smoother beams at more cost.");
				ImGui::SliderFloat("Forward scatter",  &settings.sunAnisotropy,  0.0f,   0.9f);
				tipFor("How much brighter the haze looks facing the light than with the light behind you (Henyey-Greenstein g). 0 scatters the same in every direction; at 0.5 the haze facing the light is 27 times brighter than facing away. Water droplets scatter strongly forward.");
				ImGui::SliderInt  ("Volume steps",      &settings.volSteps,         8,     128);
				tipFor("Samples along each quarter-resolution ray, for the sunbeams and for the calls' medium (Calls). More steps resolve narrower beams and cost proportionally more.");
				ImGui::SliderFloat("Volume reach (m)",  &settings.volMaxDist,      20.0f, std::max(400.0f, boxSize.x > 0.0f ? 2.0f * std::sqrt(boxSize.x*boxSize.x + boxSize.y*boxSize.y + boxSize.z*boxSize.z) : 400.0f));
				tipFor("How far each volumetric ray is marched. Haze beyond it is lit by the key light without canopy shade, so keep it past the far side of the cloud (it is set to 1.5 x the cloud's diagonal on load).");
			}

			// ---- COLOUR -------------------------------------------------------
			if(ImGui::CollapsingHeader("Colour", ImGuiTreeNodeFlags_DefaultOpen)){
				UneditedDim dim;
				ImGui::SliderFloat("Exposure",         &settings.exposure,       0.1f, 16.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
				tipFor("Overall brightness. 1 leaves it as stored. Logarithmic: up to 16x, for scans whose colour was darkened to tame the sunlit canopy.");
				ImGui::SliderFloat("Tone curve",       &settings.toneCurve,      0.0f, 1.0f);
				tipFor("0 leaves brightness as it is; a colour brighter than white is brought down to white whole, keeping its hue. "
					   "1 is the ACES fit, a film-like curve that darkens the deepest shadows, lifts the mid-tones and compresses the highlights. "
					   "It acts on each pixel's brightest channel and scales the other two with it, so hues are kept.");
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
				tipFor("Per-channel gain in linear light. 1.0 is neutral. Below removes a cast, above adds one. The scan's foliage is olive, red close to green, so red above green turns it yellow.");
				ImGui::Separator();
				ImGui::ColorEdit3 ("Tint colour",       settings.tintColour, ImGuiColorEditFlags_NoInputs);
				ImGui::SliderFloat("Tint mix",          &settings.tintMix,     0.0f,  1.0f);
				ImGui::Separator();
				ImGui::SliderFloat("Grain",            &settings.grain,          0.0f, 0.15f, "%.3f");
				ImGui::SliderFloat("Vignette",         &settings.vignette,       0.0f, 1.0f);
			}

			// ---- CALLS --------------------------------------------------------
			if(ImGui::CollapsingHeader("Calls")){
				if(ImGui::Button("Load calls.csv")){
					const char* candidates[] = {
						"calls.csv",
						"../../../rotokare_test_kit/kit/data/calls.csv",
						"../rotokare_test_kit/kit/data/calls.csv",
						"S:/SimLodMod/rotokare_test_kit/kit/data/calls.csv",
					};
					bool found = false;
					for(const char* p : candidates){
						if(std::filesystem::exists(p)){ loadCallSchedule(p); found = true; break; }
					}
					if(!found){
						g_calls.clear();
						g_callsLoaded = false;
						logLine("[calls] no calls.csv in the working directory or the test kit's folders\n");
						notice("No calls.csv found in the working directory or the test kit");
					}
					g_showClock = 0.0;
				}
				ImGui::SameLine(0.0f);
				if(ImGui::Button(g_showPlaying ? "Pause" : "Play")){ g_showPlaying = !g_showPlaying; }
				ImGui::SameLine(0.0f);
				if(ImGui::Button("Restart")){ g_showClock = 0.0; }
				ImGui::Text("clock %.1f s   active %d", g_showClock, g_activeEvents);
				ImGui::Checkbox   ("Show event markers", &settings.showEventMarkers);
				ImGui::Separator();
				// The unedited view draws each point's stored colour, without
				// the calls' glow, bloom or medium; the event markers still show.
				UneditedDim dim;
				ImGui::ColorEdit3 ("Emission colour",  settings.emissionColour, ImGuiColorEditFlags_NoInputs);
				ImGui::SameLine(0.0f);
				ImGui::Checkbox   ("by frequency",     &settings.emissionByFrequency);
				ImGui::SliderFloat("Emission gain",    &settings.emissionGain,   0.0f, 12.0f);
				ImGui::SliderFloat("Event radius (m)", &g_eventRadius,           0.1f, std::max(40.0f, boxSize.x > 0.0f ? std::max(boxSize.x, boxSize.y) * 0.5f : 40.0f));
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
				ImGui::TextDisabled("Volumetric medium (expensive, not in the reference)");
				ImGui::Checkbox   ("Volumetric medium", &settings.enableVolumetric);
				tipFor("Air that the calls light, so each call shows as a glow in the space around it. Trunks and canopy block it and cast shafts. Marched at quarter resolution with Volume steps and Volume reach (Sky and haze).");
				ImGui::SliderFloat("Medium density",    &settings.volDensity,       0.0f,  0.12f);
				ImGui::SliderFloat("Medium scatter",    &settings.volScatter,       0.0f,  6.0f);
				ImGui::SliderFloat("Medium pools low",  &settings.volHeightFalloff, 0.0f,  0.05f);
				ImGui::SliderFloat("Medium occlusion",  &settings.volOcclusion,     0.0f,  3.0f);
				ImGui::SliderInt  ("Medium shadow steps", &settings.volShadowSteps, 0,    16);
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
			}

			// ---- INSECTS ------------------------------------------------------
			if(ImGui::CollapsingHeader("Insects")){
				UneditedDim dim;   // no insects in the unedited view
				ImGui::SliderInt  ("Count",            &settings.numInsects,   0, 30000);
				ImGui::SliderInt  ("Size (px)",        &settings.insectSize,   1,     4);
				ImGui::SliderFloat("Glow",             &settings.insectGlow,   0.0f, 6.0f);
				ImGui::SliderFloat("Speed",            &settings.insectSpeed,  0.0f, 2.0f);
				ImGui::SliderInt  ("Placement",        &settings.insectPlacement, 0, 1);
				tipFor("1 keeps only insects in air near foliage, using the grid. 0 scatters them anywhere.");
			}

			// ---- MARKERS AND SURVEY -------------------------------------------
			if(ImGui::CollapsingHeader("Markers and survey")){
				ImGui::SliderFloat("Marker size (m)", &settings.markerScale, 0.05f, 1.0f, "%.2f");
				ImGui::Separator();
				ImGui::Checkbox("Show markers", &settings.showMarkers);
				ImGui::SameLine(0.0f);
				{
					UneditedDim dim;   // the unedited view shows the forest
					ImGui::Checkbox("Markers only", &settings.markersOnly);
					tipFor("Markers only hides the forest and the insects so the array can be seen on its own. Nothing is discarded.");
				}
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
					ImGui::SameLine(0.0f);
					if(!mk.hasSurvey){
						ImGui::TextDisabled("(not set)");
					}else if(ImGui::SmallButton("clear")){
						// a value typed by mistake left the marker in every fit
						mk.sx = mk.sy = mk.sz = 0.0f;
						mk.hasSurvey = false;
					}
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
					if(g_callsInSurvey && !g_surveyValid) notice("No survey fit yet: calls are placed as file (LAS) coordinates until there is one");
				}
				tipFor("On: call positions are converted from the survey frame into the file frame with the fit above, as it stands, so fitting again or clearing the markers moves them at once. Until there is a fit they are used as file coordinates. Off: calls.csv is taken to be in file (LAS) coordinates already.");
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
				ImGui::Checkbox("Skip unchanged frames", &settings.skipUnchanged);
				tipFor("Keeps the last frame while nothing that would change it has changed: the camera, a setting, a load, the calls or the insects. Saves GPU power on a still view. Turn it off if the view ever fails to update.");
				if(settings.skipUnchanged){
					ImGui::SameLine(0.0f);
					ImGui::Text("%.0f%% of recent frames kept", 100.0f * g_keptShare);
				}
				ImGui::SliderInt  ("Max splat px",     &settings.maxSplatPx,     2, 64);
				tipFor("Cap on how large a near point may be drawn. Lower if the frame rate drops when the camera is close to foliage.");
			}

			// ---- SCAN CLEAN-UP ------------------------------------------------
			if(ImGui::CollapsingHeader("Scan clean-up")){
				{
					UneditedDim dim;   // no cull in the unedited view
					ImGui::SliderInt  ("Min neighbours",   &settings.minNeighbours, 0, 60);
					tipFor("Removes points whose grid cell holds fewer than this many. 0 is off. Scanner noise sits alone; leaves do not.");
					ImGui::SliderInt  ("Cull reach",       &settings.cullReach,     0, 1);
					tipFor("0 counts the point's own cell. 1 adds the six neighbours, which spares thin surfaces on a cell boundary.");
					ImGui::Checkbox   ("Show culled only", &settings.showCulledOnly);
					tipFor("Shows ONLY what would be removed. Raise Min neighbours until this is noise and nothing you want.");
				}
				ImGui::Separator();
				{
					// The sliders rebuild the colour grid when let go, not on every
					// frame of a drag: a rebuild recounts every point and relights.
					UneditedDim dim;
					bool wf = ImGui::Checkbox("Fix blown-out canopy", &settings.whiteFix);
					tipFor("The scanner's camera saturates in the sunlit canopy, so leaves and branches there come out white. This recolours those points from the nearest well-exposed points around them: green from foliage, brown from bark.");
					ImGui::SliderFloat("White level",   &settings.whiteLevel,  100.0f, 250.0f, "%.0f");
					wf |= ImGui::IsItemDeactivatedAfterEdit();
					tipFor("A point counts as blown out when its darkest channel is at least this. Lower it to catch pale grey as well as white.");
					ImGui::SliderFloat("White spread",  &settings.whiteSpread,   5.0f, 120.0f, "%.0f");
					wf |= ImGui::IsItemDeactivatedAfterEdit();
					tipFor("... and its three channels are within this of each other. Raise it to catch pale tints (lavender, pale green).");
					ImGui::SliderFloat("Sky cast",      &settings.tintMargin,    0.0f, 60.0f, "%.0f");
					wf |= ImGui::IsItemDeactivatedAfterEdit();
					tipFor("Also repairs lilac, pink and blue points: blue higher than green by more than this. Bark and leaves never are. 0 turns it off; raise it if real blue things (signs, tape) get recoloured.");
					ImGui::SliderInt  ("Colour reach",  &settings.whiteReach,       1, 8, "%d cells");
					wf |= ImGui::IsItemDeactivatedAfterEdit();
					tipFor("How far a blown-out cell looks for a colour. Each cell is the occupancy grid's cell size.");
					if(wf) g_gridRefill = true;   // the colour grid is rebuilt with the counts
					if(settings.whiteFix && g_colourGridFailed) ImGui::Text("   not enough GPU memory for the colour grid");
				}
				ImGui::Separator();
				// Edit a copy and apply it on release: every applied value frees and
				// reallocates the grid (an implicit device sync) and zeroes the counts,
				// which happened on every frame of a drag.
				static int pendingGridRes = g_gridResolution;
				if(!ImGui::IsAnyItemActive()) pendingGridRes = g_gridResolution;   // follow fallbacks
				ImGui::SliderInt  ("Grid resolution",  &pendingGridRes, 64, 512);
				if(ImGui::IsItemDeactivatedAfterEdit()) g_gridResolution = pendingGridRes;
				tipFor("Cells along the cloud's longest side. Applied when you release the slider. Memory grows with the cube: 256 is ~20 MB, 512 ~156 MB. "
					   "The key light's grid takes as much again while the light is on, and the colour grid twice as much while Fix blown-out canopy is on "
					   "(and briefly while the bird's-eye view is rebuilt).");
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
						ImGui::Text("   + %.0f MB with the key light, + %.0f MB with Fix blown-out canopy", mb, 2.0 * mb);
					}
					if(g_gridRefill) ImGui::Text("   counts are stale - rebuilding on the next frame");
				}
				if(ImGui::Button("Rebuild grid counts")){ g_gridRefill = true; }
				ImGui::Checkbox   ("Use voxel grid",    &settings.useVoxelGrid);
				tipFor("Off stops the stray-point cull, insect placement near foliage, the grid view, the calls' medium and the canopy's dimming of the sunbeams along each view ray from using the occupancy grid. The grid is still built, and the key light's shade on the points and the canopy colour repair are not affected.");
			}

			// ---- DISPLAY AND DEBUG --------------------------------------------
			if(ImGui::CollapsingHeader("Display and debug")){
				ImGui::Checkbox   ("Small menu text",   &g_smallMenuText);
				tipFor("Draws this window in a 10-pixel font with tighter spacing, so more of it fits on screen. Off returns to the 13-pixel font.");
				ImGui::Separator();
				{
					UneditedDim dim;   // the unedited view shows the points
					ImGui::Checkbox   ("Show points",       &settings.showPoints);
				}
				ImGui::Checkbox   ("Show bounding box", &settings.showBoundingBox);
				ImGui::Checkbox   ("Show voxel grid",   &settings.showVoxelGrid);
				ImGui::Checkbox   ("Update visibility", &settings.doUpdateVisibility);
				tipFor("Off freezes the level of detail and the culling at the current view, so you can move away and see which parts of the octree that view selected.");
				{
					UneditedDim dim;   // the unedited view shows the stored colours
					ImGui::Checkbox   ("Colour by node",    &settings.colorByNode);
					ImGui::Checkbox   ("Colour by LOD",     &settings.colorByLOD);
				}
				ImGui::Separator();
				// The run log follows the working directory (build/Release when
				// started from Visual Studio), so say where this one is.
				ImGui::TextDisabled("Run log");
				ImGui::TextWrapped("%s", g_logPath.empty() ? "(no log file could be written)" : g_logPath.c_str());
				if(!g_logPath.empty() && ImGui::SmallButton("Copy log path")){
					ImGui::SetClipboardText(g_logPath.c_str());
				}
			}

			ImGui::End();
			if(smallText){
				ImGui::PopStyleVar(3);
				ImGui::PopFont();
			}
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
				if(!nc && statsCollapsed && statsSaved.x > 0.0f){
					ImGui::SetWindowSize(statsSaved, ImGuiCond_Always);   // just expanded, as beginPanel
				}else if(!nc){
					ImVec2 s = ImGui::GetWindowSize();
					if(s.x > 0.0f) statsSaved = s;
				}
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

			// Rates over the load so far. The rows for the kernels' own
			// durations and the rendering rates were filled only by a
			// benchmark that nothing switched on, so they read "-" or 0; the
			// stage rows below give the render kernel's times.
			double pointsSecTotal = double(stats.numPointsProcessed) / (double(totalUpdateDuration) / 1000.0);
			double gbs_file = double(numBytesLoaded) / (totalUpdateDuration / 1000.0);
			double gbs_gpu = double(16llu * stats.numPointsProcessed) / (totalUpdateDuration / 1000.0);

			double M = 1'000'000.0;
			double MB = 1'000'000.0; // TIL: MB = 1'000'000 vs. MiB = 1024 * 1024
			double GB = 1'000'000'000.0;

			vector<vector<string>> table = {
				{"#update total duration   ", toMS(totalUpdateDuration)                             , format("{:.1f}", totalUpdateDuration)},
				{"points/sec total         ", toM(pointsSecTotal)                                   , format("{:.1f}", pointsSecTotal / M)},
				// LAZ batches are not counted in numBytesLoaded
				{"GB/s (disk I/O)          ", lazInBatches ? leftPad("-", 15) : toGB(gbs_file)    , lazInBatches ? string("-") : format("{:.1f}", gbs_file / GB)},
				{"GB/s (gpu)               ", toGB(gbs_gpu)                                         , format("{:.1f}", gbs_gpu / GB)},
				{"=========================", " "                                                   , " "},
				{"#points processed        ", toM(double(stats.numPointsProcessed))                 , format("{:.1f}", stats.numPointsProcessed / M)},
				{"#nodes                   ", toIntString(stats.numNodes)                           , format("{}", stats.numNodes)},
				{"    #inner               ", toIntString(stats.numInner)                           , format("{}", stats.numInner)},
				{"    #leaves (nonempty)   ", toIntString(stats.numNonemptyLeaves)                  , format("{}", stats.numNonemptyLeaves)},
				{"    #leaves (empty)      ", toIntString(numEmptyLeaves)                           , format("{}", numEmptyLeaves)},
				{"#chunks                  ", toIntString(stats.numChunksPoints + stats.numChunksVoxels), format("{}", stats.numChunksPoints + stats.numChunksVoxels)},
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
