#pragma once

// Writes the exception and a symbolised stack trace to logPath when the
// process dies from an unhandled structured exception on any thread.
// Windows only; a no-op elsewhere.
void installCrashHandler(const char* logPath);

// Per thread (MSVC keeps terminate handlers per thread): an exception that
// escapes the thread's function is logged with its message and a stack trace
// before the process aborts. Call at the start of every std::thread body.
void installThreadTerminateHandler();
