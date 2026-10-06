#pragma once

#include <string>
#include <unordered_map>
#include <functional>
#include <fstream>
#include <iterator>
#include "unsuck.hpp"
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

		string dir = fs::path(path).parent_path().string();
		string optInclude = std::format("-I {}", dir);
		string cuda_include = std::format("-I {}/include", cudaPath);

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
		string optArch = std::format("--gpu-architecture=compute_{}", compileArch);

		std::vector<const char*> opts = {
			optArch.c_str(),
			"--use_fast_math",
			"--extra-device-vectorization",
			"-lineinfo",
			cuda_include.c_str(),
			optInclude.c_str(),
			"-I ./",
			"--relocatable-device-code=true",
			"-default-device",
			"-dlto",
			"--std=c++20",
			"--disable-warnings",
		};

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

			module->compile();

			monitorFile(modulePath, [this, module]() {
				if(module->compile()) link();
			});

			modules.push_back(module);
		}

		link();
	}

	string label() const {
		return modules.empty() ? string("(no modules)") : modules[0]->name;
	}

	// Links the modules' LTO-IR for this GPU and loads the result. On any
	// failure the module and kernels in use are kept, so a hot reload with an
	// error leaves the viewer running on the code it had.
	bool link(){

		cout << "================================================================================" << endl;
		cout << "=== LINKING" << endl;
		cout << "================================================================================" << endl;

		auto tStart = now();

		for(auto module : modules){
			if(!module->success){
				return false;   // its compile failed, and said so
			}
		}

		string strArch = std::format("-arch=sm_{}", deviceComputeCapability());

		// -lto, as nvJitLink documents it (Supported Link Options); nvJitLink
		// 12.0 refuses the -dlto used before, and its refusal went unchecked.
		const char *lopts[] = {"-lto", strArch.c_str()};

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
