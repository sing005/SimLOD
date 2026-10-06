#pragma once

#include <cstdint>

#include "unsuck.hpp"

struct LasHeader{
	int versionMajor             = 0;
	int versionMinor             = 0;
	uint64_t headerSize          = 0;
	uint64_t offsetToPointData   = 0;
	uint64_t format              = 0;
	uint64_t bytesPerPoint       = 0;
	uint64_t numPoints           = 0;
	double scale[3]              = {0.0, 0.0, 0.0};
	double offset[3]             = {0.0, 0.0, 0.0};
	double min[3]                = {0.0, 0.0, 0.0};
	double max[3]                = {0.0, 0.0, 0.0};
};

inline LasHeader loadHeader(string file){
	auto headerBuffer = readBinaryFile(file, 0, 375);

	LasHeader header;
	header.versionMajor        = headerBuffer->get<uint8_t>(24);
	header.versionMinor        = headerBuffer->get<uint8_t>(25);
	header.headerSize          = headerBuffer->get<uint16_t>(94);
	header.offsetToPointData   = headerBuffer->get<uint32_t>(96);
	header.format              = headerBuffer->get<uint8_t>(104);
	header.bytesPerPoint       = headerBuffer->get<uint16_t>(105);
	if(header.versionMajor == 1 && header.versionMinor <= 3){
		header.numPoints       = headerBuffer->get<uint32_t>(107);
	}else{
		header.numPoints       = headerBuffer->get<uint64_t>(247);
	}

	
	header.scale[0] = headerBuffer->get<double>(131);
	header.scale[1] = headerBuffer->get<double>(139);
	header.scale[2] = headerBuffer->get<double>(147);

	header.offset[0] = headerBuffer->get<double>(155);
	header.offset[1] = headerBuffer->get<double>(163);
	header.offset[2] = headerBuffer->get<double>(171);

	header.min[0] = headerBuffer->get<double>(187);
	header.min[1] = headerBuffer->get<double>(203);
	header.min[2] = headerBuffer->get<double>(219);

	header.max[0] = headerBuffer->get<double>(179);
	header.max[1] = headerBuffer->get<double>(195);
	header.max[2] = headerBuffer->get<double>(211);

	return header;
}

// Why a .las or .laz file cannot be loaded, or "" when it can. Run before the
// file's batches are queued: a cut LAZ file crashed the viewer inside laszip,
// and a cut LAS file loaded copies of other points without a message.
string lasFileProblem(const string& path);

// Reads bytes [firstByte, firstByte + numBytes) of a file into buffer. On
// Windows the read bypasses the file cache, so it starts at the sector boundary
// at or before firstByte: the data begins *padding bytes into the buffer, which
// must hold numBytes + 2 x sectorSize. False if the file cannot be opened, or
// the read fails or ends before the last byte asked for.
bool readFileRange(const string& file, uint64_t sectorSize, uint64_t firstByte, uint64_t numBytes, void* buffer, uint64_t* padding);

// Reads and converts numPoints points from firstPoint on; false if they could not be read.
bool loadLasNative(string file, LasHeader header, uint64_t firstPoint, uint64_t numPoints, void* target, double translation[3]);

// Frees the calling thread's read buffer; the next LAS batch allocates it again.
void releaseLasReadBuffer();