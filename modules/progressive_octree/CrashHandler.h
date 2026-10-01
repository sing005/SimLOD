#pragma once

// Writes the exception and a symbolised stack trace to logPath when the
// process dies from an unhandled structured exception on any thread.
// Windows only; a no-op elsewhere.
void installCrashHandler(const char* logPath);
