#pragma once

#include <string>
#include <unordered_map>
#include <functional>
#include <fstream>
#include <iterator>
#include "unsuck.hpp"
#include "Sha256.h"
#include "nvrtc.h"
#include <nvJitLink.h>
#include <cmath>
#include "cuda.h"

using std::string;

using namespace std;

// Where compile and link messages go: the application's run log, if it sets
// this before it builds a program; stderr if not. Compile errors used to go
// to stderr only, and a link error ended the program with nothing in the log.
inline std::function<void(const string&)> cudaProgramLog;

inline void kernelLog(const string& message){
	if(cudaProgramLog) cudaProgramLog(message);
	else std::cerr << message << std::endl;
}

// The kernel cache (CudaModularProgram::loadFromCache): on unless --no-kernel-cache
inline bool kernelCacheEnabled = true;

// Device 0's compute capability as one number, 86 for 8.6; 86 if it cannot be read
inline int deviceComputeCapability(){
	CUdevice dev;
	int major = 0, minor = 0;
	if(cuDeviceGet(&dev, 0) == CUDA_SUCCESS
		&& cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) == CUDA_SUCCESS
		&& cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) == CUDA_SUCCESS){
		return major * 10 + minor;
	}
	return 86;
}

struct CudaModule{

	string path = "";
	string name = "";
	bool success = false;   // the last compile succeeded, so ltoir is that compile's

	size_t ltoirSize = 0;
	char* ltoir = nullptr;

	CudaModule(string path, string name){
		this->path = path;
		this->name = name;
	}

	// Compiles the file to LTO-IR for link(). On failure success is false and
	// the program keeps the module it has.
	bool compile(){
		auto tStart = now();

		cout << "================================================================================" << endl;
		cout << "=== COMPILING: " << fs::path(path).filename().string() << endl;
		cout << "================================================================================" << endl;

		success = false;

		// NVRTC compiles against the toolkit's headers. A null CUDA_PATH was
		// turned into a std::string, an access violation that named nothing.
		const char* cudaPath = std::getenv("CUDA_PATH");
		if(cudaPath == nullptr || cudaPath[0] == '\0'){
			kernelLog("[kernels] CUDA_PATH is not set, so " + name + " cannot be compiled: it must point to a CUDA 12 toolkit");
			return false;
		}

		std::ifstream in(path);
		if(!in){
			kernelLog("[kernels] cannot read " + path);
			return false;
		}
		string source((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

		vector<string> options = this->options();
		std::vector<const char*> opts;
		for(const string& option : options) opts.push_back(option.c_str());

		for(auto opt : opts){
			cout << opt << endl;
		}
		cout << "====" << endl;

		nvrtcProgram prog;
		nvrtcResult res = nvrtcCreateProgram(&prog, source.c_str(), name.c_str(), 0, NULL, NULL);
		if(res != NVRTC_SUCCESS){
			kernelLog(std::format("[kernels] nvrtcCreateProgram failed for {}: {}", name, nvrtcGetErrorString(res)));
			return false;
		}

		res = nvrtcCompileProgram(prog, int(opts.size()), opts.data());
		if(res != NVRTC_SUCCESS){
			size_t logSize = 0;
			nvrtcGetProgramLogSize(prog, &logSize);
			string log(logSize, '\0');
			if(logSize > 0) nvrtcGetProgramLog(prog, log.data());
			std::cerr << log << std::endl;
			kernelLog(std::format("[kernels] {} did not compile ({}); the kernels in use are kept:\n{}", name, nvrtcGetErrorString(res), log));
			nvrtcDestroyProgram(&prog);   // it leaked on this path
			return false;
		}

		size_t size = 0;
		nvrtcGetLTOIRSize(prog, &size);
		char* ir = new char[size];
		nvrtcGetLTOIR(prog, ir);
		nvrtcDestroyProgram(&prog);

		delete[] ltoir;   // the previous compile's, which leaked on every reload
		ltoir = ir;
		ltoirSize = size;
		success = true;

		kernelLog(std::format("[kernels] compiled {} in {:.2f} s ({} B of LTO-IR)", name, now() - tStart, ltoirSize));
		return true;
	}

	// NVRTC's options for this module, which the kernel cache's key holds too
	vector<string> options() const {
		const char* cudaPath = std::getenv("CUDA_PATH");
		string dir = fs::path(path).parent_path().string();

		// Compile for the GPU that is actually present (the link step below
		// already targets it). Hardcoded compute_86 failed to link on older
		// cards and left newer ones (e.g. Ada, compute 8.9) on the 8.6 feature
		// set. Use the newest architecture this NVRTC supports that does not
		// exceed the device's.
		int deviceArch = deviceComputeCapability();
		int compileArch = 0;
		{
			int numArchs = 0;
			nvrtcGetNumSupportedArchs(&numArchs);
			std::vector<int> archs(numArchs);
			if(numArchs > 0) nvrtcGetSupportedArchs(archs.data());
			for(int a : archs) if(a <= deviceArch && a > compileArch) compileArch = a;
			if(compileArch == 0) compileArch = deviceArch;
		}

		return {
			std::format("--gpu-architecture=compute_{}", compileArch),
			"--use_fast_math",
			"--extra-device-vectorization",
			"-lineinfo",
			std::format("-I {}/include", cudaPath ? cudaPath : ""),
			std::format("-I {}", dir),
			"-I ./",
			"--relocatable-device-code=true",
			"-default-device",
			"-dlto",
			"--std=c++20",
			"--disable-warnings",
		};
	}

};


struct CudaModularProgram{

	struct CudaModularProgramArgs{
		vector<string> modules;
		vector<string> kernels;
	};

	vector<CudaModule*> modules;

	CUmodule mod = nullptr;      // where the kernels are from; replaced only by a link that succeeds
	void* cubin = nullptr;       // its image, kept while it is loaded
	size_t cubinSize = 0;
	// Counts the modules loaded. Compare this, not mod: a new module can be
	// given the handle of one just unloaded, and anything looked up in the
	// old one (a global's address, say) would then look current.
	int generation = 0;

	vector<std::function<void(void)>> compileCallbacks;

	vector<string> kernelNames;
	unordered_map<string, CUfunction> kernels;


	CudaModularProgram(CudaModularProgramArgs args){

		vector<string> modulePaths = args.modules;
		vector<string> kernelNames = args.kernels;

		this->kernelNames = kernelNames;

		for(auto modulePath : modulePaths){

			string moduleName = fs::path(modulePath).filename().string();
			auto module = new CudaModule(modulePath, moduleName);

			monitorFile(modulePath, [this, module]() {
				string key = cacheKey();   // before compiling: see link()
				if(!module->compile()) return;
				// a module loaded from the cache was never compiled
				for(auto other : modules){
					if(!other->success && !other->compile()) return;
				}
				link(key);
			});

			modules.push_back(module);
		}

		if(!loadFromCache()){
			string key = cacheKey();
			for(auto module : modules) module->compile();
			link(key);
		}
	}

	string label() const {
		return modules.empty() ? string("(no modules)") : modules[0]->name;
	}

	vector<string> linkOptions() const {
		// -lto, as nvJitLink documents it (Supported Link Options); nvJitLink
		// 12.0 refuses the -dlto used before, and its refusal went unchecked.
		return {"-lto", std::format("-arch=sm_{}", deviceComputeCapability())};
	}

	// ---- the kernel cache ------------------------------------------------------------
	// Every start compiled and linked all three programs again, about 7 s.
	// Each program's cubin is kept in a per-user folder under a key: a hash of
	// everything the cubin could depend on. Any change to a source or header
	// under ./modules, the options (and so the GPU), NVRTC's or nvJitLink's
	// version or the toolkit gives a new key and a compile. A stale entry
	// would run old code against new structs without any error, so the key
	// takes every file rather than trying to follow the includes, which
	// NVRTC cannot list.

	static std::filesystem::path cacheFolder(){
		static std::filesystem::path folder = []{
			std::filesystem::path base;
		#ifdef _WIN32
			// wide, so a user name outside the code page is kept as it is
			if(const wchar_t* local = _wgetenv(L"LOCALAPPDATA")) base = std::filesystem::path(local);
		#else
			if(const char* cache = std::getenv("XDG_CACHE_HOME")) base = cache;
			else if(const char* home = std::getenv("HOME")) base = std::filesystem::path(home) / ".cache";
		#endif
			if(base.empty()) return std::filesystem::path();
			std::filesystem::path dir = base / "SimLOD" / "kernels";
			std::error_code ec;
			std::filesystem::create_directories(dir, ec);
			return ec ? std::filesystem::path() : dir;
		}();
		return folder;
	}

	string cacheKey() const {
		Sha256 sha;
		auto add = [&](const string& text){
			uint64_t size = text.size();
			sha.update(&size, sizeof(size));
			sha.update(text.data(), text.size());
		};
		auto addFile = [&](const std::filesystem::path& file){
			std::ifstream in(file, std::ios::binary);
			add(string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));
		};

		add("SimLOD kernel cache, format 1");
		for(auto module : modules){
			add(module->path);
			for(const string& option : module->options()) add(option);
		}
		for(const string& kernelName : kernelNames) add(kernelName);
		for(const string& option : linkOptions()) add(option);

		int nvrtcMajor = 0, nvrtcMinor = 0;
		nvrtcVersion(&nvrtcMajor, &nvrtcMinor);
		unsigned int linkMajor = 0, linkMinor = 0;
		nvJitLinkVersion(&linkMajor, &linkMinor);
		add(std::format("nvrtc {}.{}, nvJitLink {}.{}", nvrtcMajor, nvrtcMinor, linkMajor, linkMinor));

		// the toolkit NVRTC compiles against: its folder, version.json (the
		// full version, patch included) and cuda.h
		const char* cudaPath = std::getenv("CUDA_PATH");
		if(cudaPath != nullptr){
			add(cudaPath);
			addFile(std::filesystem::path(cudaPath) / "version.json");
			addFile(std::filesystem::path(cudaPath) / "include" / "cuda.h");
		}

		// every source and header under ./modules, in a fixed order
		std::error_code ec;
		std::filesystem::path root = modules.empty() ? std::filesystem::path("./modules")
			: std::filesystem::path(modules[0]->path).parent_path().parent_path();
		vector<std::filesystem::path> files;
		for(auto it = std::filesystem::recursive_directory_iterator(root, ec); !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec)){
			string extension = it->path().extension().string();
			if(it->is_regular_file(ec) && (extension == ".cu" || extension == ".cuh" || extension == ".h")){
				files.push_back(it->path());
			}
		}
		std::sort(files.begin(), files.end());
		for(const auto& file : files){
			add(file.generic_string());
			addFile(file);
		}

		return sha.hex();
	}

	std::filesystem::path cacheFile() const {
		std::filesystem::path dir = cacheFolder();
		return dir.empty() ? dir : dir / (label() + ".cubin");
	}

	// The cached cubin, if its key is this program's current one
	bool loadFromCache(){
		std::filesystem::path file = cacheFile();
		if(!kernelCacheEnabled || file.empty()) return false;
		auto tStart = now();

		std::ifstream in(file, std::ios::binary);
		if(!in) return false;
		string header;
		std::getline(in, header);
		if(header != "SimLOD kernel cache " + cacheKey()){
			kernelLog("[kernels] " + label() + ": the cached kernels are out of date, compiling");
			return false;
		}
		string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
		if(bytes.empty()) return false;
		void* image = malloc(bytes.size());
		memcpy(image, bytes.data(), bytes.size());
		if(!load(image, bytes.size())) return false;   // load frees the image if it fails

		kernelLog(std::format("[kernels] {}: loaded from the cache in {:.0f} ms", label(), 1000.0 * (now() - tStart)));
		return true;
	}

	// Written under another name and renamed when complete, so an interrupted
	// write never leaves a short entry
	void storeInCache(const string& key){
		std::filesystem::path file = cacheFile();
		if(!kernelCacheEnabled || file.empty() || cubin == nullptr) return;
		std::filesystem::path part = file;
		part += ".part";
		std::error_code ec;
		{
			std::ofstream out(part, std::ios::binary);
			out << "SimLOD kernel cache " << key << "\n";
			out.write((const char*)cubin, std::streamsize(cubinSize));
			out.close();
			if(!out){
				std::filesystem::remove(part, ec);
				return;
			}
		}
		std::filesystem::rename(part, file, ec);
		if(ec) std::filesystem::remove(part, ec);
	}

	// Links the modules' LTO-IR for this GPU and loads the result. On any
	// failure the module and kernels in use are kept, so a hot reload with an
	// error leaves the viewer running on the code it had. key is the cache
	// key taken before the compile: a source saved again during the compile
	// then makes the entry look out of date, which only costs a compile.
	bool link(const string& key = ""){

		cout << "================================================================================" << endl;
		cout << "=== LINKING" << endl;
		cout << "================================================================================" << endl;

		auto tStart = now();

		for(auto module : modules){
			if(!module->success){
				return false;   // its compile failed, and said so
			}
		}

		vector<string> options = linkOptions();
		const char *lopts[] = {options[0].c_str(), options[1].c_str()};

		nvJitLinkHandle handle = nullptr;
		nvJitLinkResult res = nvJitLinkCreate(&handle, 2, lopts);
		if(res != NVJITLINK_SUCCESS){
			kernelLog(std::format("[kernels] nvJitLinkCreate failed ({}) for {}; the kernels in use are kept", int(res), label()));
			return false;
		}

		// A link error used to end the program (exit(1)), with the message on
		// stderr only.
		auto fail = [&](const char* step, nvJitLinkResult result){
			size_t logSize = 0;
			string log;
			if(nvJitLinkGetErrorLogSize(handle, &logSize) == NVJITLINK_SUCCESS && logSize > 0){
				log.resize(logSize);
				nvJitLinkGetErrorLog(handle, log.data());
			}
			kernelLog(std::format("[kernels] {} failed ({}) for {}; the kernels in use are kept:\n{}", step, int(result), label(), log));
			nvJitLinkDestroy(&handle);
			return false;
		};

		for(auto module : modules){
			res = nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, (void *)module->ltoir, module->ltoirSize, module->name.c_str());
			if(res != NVJITLINK_SUCCESS) return fail("nvJitLinkAddData", res);
		}

		res = nvJitLinkComplete(handle);
		if(res != NVJITLINK_SUCCESS) return fail("nvJitLinkComplete", res);

		size_t newSize = 0;
		res = nvJitLinkGetLinkedCubinSize(handle, &newSize);
		if(res != NVJITLINK_SUCCESS) return fail("nvJitLinkGetLinkedCubinSize", res);

		void* newCubin = malloc(newSize);
		res = nvJitLinkGetLinkedCubin(handle, newCubin);
		if(res != NVJITLINK_SUCCESS){
			free(newCubin);
			return fail("nvJitLinkGetLinkedCubin", res);
		}
		nvJitLinkDestroy(&handle);

		if(!load(newCubin, newSize)) return false;
		kernelLog(std::format("[kernels] linked {} in {:.2f} s ({} B)", label(), now() - tStart, newSize));
		if(!key.empty()) storeInCache(key);
		for(auto& callback : compileCallbacks){
			callback();
		}
		return true;
	}

	// Loads a cubin and takes its kernels. The module in use is replaced only
	// when every kernel resolves: an uninitialised kernel handle used to
	// replace a working one when a kernel was missing.
	bool load(void* newCubin, size_t newSize){
		CUmodule newMod = nullptr;
		CUresult cres = cuModuleLoadData(&newMod, newCubin);
		if(cres != CUDA_SUCCESS){
			kernelLog(std::format("[kernels] cuModuleLoadData failed ({}) for {}; the kernels in use are kept", int(cres), label()));
			free(newCubin);
			return false;
		}

		unordered_map<string, CUfunction> newKernels;
		for(const string& kernelName : kernelNames){
			CUfunction kernel = nullptr;
			cres = cuModuleGetFunction(&kernel, newMod, kernelName.c_str());
			if(cres != CUDA_SUCCESS){
				kernelLog(std::format("[kernels] {} has no kernel {} ({}); the kernels in use are kept", label(), kernelName, int(cres)));
				cuModuleUnload(newMod);
				free(newCubin);
				return false;
			}
			newKernels[kernelName] = kernel;
		}

		// The old module was never unloaded, so every reload kept its code and
		// data on the GPU. Its kernels may still be running.
		if(mod != nullptr){
			cuCtxSynchronize();
			cuModuleUnload(mod);
		}
		free(cubin);

		mod = newMod;
		cubin = newCubin;
		cubinSize = newSize;
		kernels = newKernels;
		generation++;
		return true;
	}

	void onCompile(std::function<void(void)> callback){
		compileCallbacks.push_back(callback);
	}

};
