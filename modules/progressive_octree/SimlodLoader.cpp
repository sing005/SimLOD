#define NOMINMAX

#include "SimlodLoader.h"
#include "LasLoader.h"

#include <filesystem>
#include <string>
#include <cmath>
#include <cstdio>

#include "unsuck.hpp"

// A .simlod file is a 24-byte box, six floats (min xyz, max xyz), then
// 16 bytes a point: three floats and RGBA.
string simlodFileProblem(const string& path){
	std::error_code ec;
	uint64_t size = fs::file_size(path, ec);
	if(ec) return "cannot be opened";
	if(size < 24 || (size - 24) % 16 != 0) return "not a .simlod file: its size is not a 24-byte box and 16 bytes a point";
	if(size == 24) return "no points";

	float box[6] = {};
	FILE* f = fopen(path.c_str(), "rb");
	if(f == nullptr) return "cannot be opened";
	bool read = fread(box, 1, sizeof(box), f) == sizeof(box);
	fclose(f);
	if(!read) return "cannot be read";
	for(int axis = 0; axis < 3; axis++){
		if(!std::isfinite(box[axis]) || !std::isfinite(box[axis + 3]) || box[axis] > box[axis + 3]) return "the box at its start is not valid";
	}

	return "";
}

bool loadFileNative(
	string file, 
	uint64_t firstByte, 
	uint64_t numBytes, 
	void* target,
	uint64_t* out_padding
){
	// target is a pinned slot, with room for the sectors around the range
	return readFileRange(file, getPhysicalSectorSize(file), firstByte, numBytes, target, out_padding);
}
