#pragma once

#include <cstdint>
#include <string>

using std::string;

// Why a .simlod file cannot be loaded, or "" when it can.
string simlodFileProblem(const string& path);

// Reads bytes [firstByte, firstByte + numBytes) into target, *out_padding bytes
// in; false if they could not be read. See readFileRange.
bool loadFileNative(
	string file, 
	uint64_t firstByte, 
	uint64_t numBytes, 
	void* target,
	uint64_t* out_padding
);