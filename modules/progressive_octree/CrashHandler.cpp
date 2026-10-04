// Last-words handler for hard crashes on Windows.
//
// A structured exception (access violation, integer divide by zero, ...) on
// any thread reaches this filter, which appends the exception code and a
// symbolised stack trace to the run log, then lets the process die. Needs the
// .pdb next to the exe for function names and line numbers (Release builds
// generate one, see CMakeLists.txt).
//
// Kept in its own translation unit because <windows.h> defines 'near' and
// 'far' as macros, which breaks the Camera struct used elsewhere.

#include "CrashHandler.h"

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dbghelp.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <csignal>
#include <exception>

#pragma comment(lib, "dbghelp.lib")

static char g_crashLogPath[MAX_PATH] = {0};

static void writeOut(HANDLE file, const char* text){
	DWORD written = 0;
	if(file != INVALID_HANDLE_VALUE) WriteFile(file, text, DWORD(strlen(text)), &written, nullptr);
	fputs(text, stderr);
}

static const char* exceptionName(DWORD code){
	switch(code){
		case EXCEPTION_ACCESS_VIOLATION:      return "access violation";
		case EXCEPTION_INT_DIVIDE_BY_ZERO:    return "integer divide by zero";
		case EXCEPTION_STACK_OVERFLOW:        return "stack overflow";
		case EXCEPTION_ILLEGAL_INSTRUCTION:   return "illegal instruction";
		case EXCEPTION_ARRAY_BOUNDS_EXCEEDED: return "array bounds exceeded";
		case EXCEPTION_IN_PAGE_ERROR:         return "in-page error";
		case 0xC0000409:                      return "stack buffer overrun / fail-fast";
		default:                              return "unknown";
	}
}

static HANDLE openCrashLog(){
	if(g_crashLogPath[0] == 0) return INVALID_HANDLE_VALUE;
	return CreateFileA(g_crashLogPath, FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE,
	                   nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
}

// Walk and print the stack described by `context` (a copy; it is modified).
static void writeStack(HANDLE file, CONTEXT context){
	char line[1024];
	HANDLE process = GetCurrentProcess();
	HANDLE thread  = GetCurrentThread();
	SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
	SymInitialize(process, nullptr, TRUE);

	STACKFRAME64 frame = {};
	frame.AddrPC.Offset    = context.Rip;
	frame.AddrPC.Mode      = AddrModeFlat;
	frame.AddrFrame.Offset = context.Rbp;
	frame.AddrFrame.Mode   = AddrModeFlat;
	frame.AddrStack.Offset = context.Rsp;
	frame.AddrStack.Mode   = AddrModeFlat;

	alignas(SYMBOL_INFO) char symbolBuffer[sizeof(SYMBOL_INFO) + 256];
	SYMBOL_INFO* symbol = (SYMBOL_INFO*)symbolBuffer;

	for(int depth = 0; depth < 32; depth++){
		if(!StackWalk64(IMAGE_FILE_MACHINE_AMD64, process, thread, &frame, &context,
		                nullptr, SymFunctionTableAccess64, SymGetModuleBase64, nullptr)) break;
		DWORD64 address = frame.AddrPC.Offset;
		if(address == 0) break;

		memset(symbolBuffer, 0, sizeof(symbolBuffer));
		symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
		symbol->MaxNameLen   = 255;
		DWORD64 symOffset = 0;
		const char* name = SymFromAddr(process, address, &symOffset, symbol) ? symbol->Name : "?";

		IMAGEHLP_LINE64 src = {};
		src.SizeOfStruct = sizeof(src);
		DWORD lineOffset = 0;
		if(SymGetLineFromAddr64(process, address, &lineOffset, &src)){
			const char* base = strrchr(src.FileName, '\\');
			snprintf(line, sizeof(line), "[CRASH]   #%-2d %s  (%s:%lu)\n", depth, name, base ? base + 1 : src.FileName, src.LineNumber);
		}else{
			snprintf(line, sizeof(line), "[CRASH]   #%-2d %s  (0x%llX)\n", depth, name, (unsigned long long)address);
		}
		writeOut(file, line);
	}

	SymCleanup(process);
}

static LONG WINAPI onUnhandledException(EXCEPTION_POINTERS* ep){
	HANDLE file = openCrashLog();
	char line[256];
	DWORD code = ep->ExceptionRecord->ExceptionCode;
	snprintf(line, sizeof(line), "[CRASH] exception 0x%08lX (%s) on thread %lu\n",
	         code, exceptionName(code), GetCurrentThreadId());
	writeOut(file, line);
	writeStack(file, *ep->ContextRecord);
	if(file != INVALID_HANDLE_VALUE) CloseHandle(file);
	return EXCEPTION_EXECUTE_HANDLER;   // terminate
}

// abort() - from an assertion, a failed allocation, or an exception nobody
// caught - raises SIGABRT on the calling thread; log where that was.
static void onAbortSignal(int){
	HANDLE file = openCrashLog();
	char line[256];
	snprintf(line, sizeof(line), "[CRASH] abort() on thread %lu\n", GetCurrentThreadId());
	writeOut(file, line);
	CONTEXT context = {};
	RtlCaptureContext(&context);
	writeStack(file, context);
	if(file != INVALID_HANDLE_VALUE) CloseHandle(file);
	_exit(134);
}

static void onTerminate(){
	HANDLE file = openCrashLog();
	char line[1024];
	const char* what = "unknown exception";
	std::exception_ptr ep = std::current_exception();
	if(ep){
		try{ std::rethrow_exception(ep); }
		catch(const std::exception& e){ snprintf(line, sizeof(line), "[CRASH] uncaught exception on thread %lu: %s\n", GetCurrentThreadId(), e.what()); what = nullptr; }
		catch(...){}
	}
	if(what) snprintf(line, sizeof(line), "[CRASH] std::terminate on thread %lu (%s)\n", GetCurrentThreadId(), what);
	writeOut(file, line);
	// MSVC calls terminate before unwinding, so this stack still shows
	// where the exception was thrown.
	CONTEXT context = {};
	RtlCaptureContext(&context);
	writeStack(file, context);
	if(file != INVALID_HANDLE_VALUE) CloseHandle(file);
	_exit(134);
}

void installCrashHandler(const char* logPath){
	if(logPath) snprintf(g_crashLogPath, sizeof(g_crashLogPath), "%s", logPath);
	SetUnhandledExceptionFilter(onUnhandledException);
	// no "abort() has been called" dialog (Debug) or Windows error report:
	// go straight to the handler, which writes the stack to the log
	_set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
	signal(SIGABRT, onAbortSignal);
	installThreadTerminateHandler();
}

void installThreadTerminateHandler(){
	std::set_terminate(onTerminate);
}

#else

void installCrashHandler(const char*){}
void installThreadTerminateHandler(){}

#endif
