#define NOMINMAX

#include "LasLoader.h"

#include <filesystem>
#include <mutex>
#include <deque>
#include <atomic>
#include <string>
#include <algorithm>
#include <locale>
#include <codecvt>
#include <cmath>
#include <cstdio>
#include <cstring>


using std::string;

// Each loader thread reads its LAS batches into a buffer of its own, about
// 36 MB for a batch of 1 M points. The thread frees it with
// releaseLasReadBuffer while there is nothing to load: 20 loader threads used
// to keep 720 MB for the rest of the session.
static thread_local void* thread_buffer = nullptr;
static thread_local uint64_t thread_buffer_size = 0;

static bool ensureThreadBuffer(uint64_t size){
	if(thread_buffer != nullptr && thread_buffer_size >= size) return true;
	free(thread_buffer);
	thread_buffer = malloc(size);
	thread_buffer_size = thread_buffer != nullptr ? size : 0;
	return thread_buffer != nullptr;
}

void releaseLasReadBuffer(){
	free(thread_buffer);
	thread_buffer = nullptr;
	thread_buffer_size = 0;
}


#ifdef _WIN32

#include "windows.h"

static thread_local bool  readPending = false;
static thread_local DWORD readError   = 0;
static thread_local DWORD readBytes   = 0;

// Runs on the thread that started the read, while that thread waits in SleepEx.
static VOID CALLBACK readCompleted(DWORD errorCode, DWORD numBytesTransfered, LPOVERLAPPED overlapped){
	readError   = errorCode;
	readBytes   = numBytesTransfered;
	readPending = false;
}

// - Standard file IO routines like fread or ifstream are fairly slow when it comes to PCIe 5 SSDs on windows.
// - Windows caches files when they are read, making the first read slower and surprisingly CPU-intensive,
//   but subsequent reads much faster because windows then serves the data from RAM.
// - However, we are interested in optimizing the first access to a point cloud file.
// - To do this, we use ReadFileEx with the unbuffered flag, which does not cache files.
// - Unbuffered reads add some additional complexity because they must be aligned to the disk's physical sector size.
bool readFileRange(const string& file, uint64_t sectorSize, uint64_t firstByte, uint64_t numBytes, void* buffer, uint64_t* padding){
	uint64_t endByte = firstByte + numBytes;
	uint64_t firstByte_aligned = firstByte - firstByte % sectorSize;
	uint64_t endByte_aligned = (endByte + sectorSize - 1) / sectorSize * sectorSize;
	uint64_t numBytes_aligned = endByte_aligned - firstByte_aligned;
	*padding = firstByte - firstByte_aligned;

	if(numBytes_aligned > 0xFFFFFFFFull) return false;   // one ReadFileEx reads less than 4 GB

	// was new TCHAR[1000], never freed (one leak per batch) and overflowed on
	// paths of 1000+ characters
	HANDLE hFile = CreateFileA(file.c_str(),
		GENERIC_READ,
		FILE_SHARE_READ,
		NULL,
		OPEN_EXISTING,
		FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING,
		NULL);
	if(hFile == INVALID_HANDLE_VALUE) return false;

	// The read must deliver every byte up to endByte; the sectors after it may
	// lie past the end of the file. A read that delivers nothing is tried again
	// a few times. It used to be tried for ever, and a read that came back short
	// was parsed whole, so the points past its end were the previous batch's.
	bool ok = false;
	for(int attempt = 0; attempt < 10; attempt++){
		OVERLAPPED ol = {};
		ol.Offset     = DWORD(firstByte_aligned & 0xffffffffull);
		ol.OffsetHigh = DWORD(firstByte_aligned >> 32);

		readPending = true;
		if(!ReadFileEx(hFile, buffer, DWORD(numBytes_aligned), &ol, readCompleted)){
			readPending = false;
			break;
		}
		while(readPending) SleepEx(INFINITE, TRUE);   // returns once readCompleted has run

		ok = readError == 0 && readBytes >= endByte - firstByte_aligned;
		if(ok || readError != 0 || readBytes != 0) break;
		SleepEx(1, TRUE);
	}

	CloseHandle(hFile);

	return ok;
}

#elif defined(__linux__)

bool readFileRange(const string& file, uint64_t /*sectorSize*/, uint64_t firstByte, uint64_t numBytes, void* buffer, uint64_t* padding){
	*padding = 0;
	FILE* f = fopen(file.c_str(), "rb");
	if(f == nullptr) return false;
	bool ok = fseek_64_all_platforms(f, int64_t(firstByte), SEEK_SET) == 0 && fread(buffer, 1, numBytes, f) == numBytes;
	fclose(f);
	return ok;
}

#endif


bool loadLasNative(string file, LasHeader header, uint64_t firstPoint, uint64_t numPoints, void* target, double translation[3]){

	// - figure out what buffer range to load based on firstPoint and numPoints
	// - load that buffer with winapi
	// - parse points and store them in <target>

	uint64_t sectorSize = getPhysicalSectorSize(file);
	uint64_t numBytes = header.bytesPerPoint * numPoints;
	uint64_t padding = 0;
	if(!ensureThreadBuffer(numBytes + 2 * sectorSize)) return false;
	if(!readFileRange(file, sectorSize, header.offsetToPointData + header.bytesPerPoint * firstPoint, numBytes, thread_buffer, &padding)){
		return false;
	}

	// now parse loaded points and write results to <target>
	uint64_t offset_rgb = 0;
	// byte offset of red in each point format that has colour (LAS 1.4)
	if(header.format == 2){
		offset_rgb = 20;
	}else if(header.format == 3 || header.format == 5){
		offset_rgb = 28;
	}else if(header.format == 7 || header.format == 8 || header.format == 10){
		offset_rgb = 30;
	}

	struct Point{
		float x, y, z;
		uint8_t r, g, b, a;
	};

	uint8_t* source = ((uint8_t*)thread_buffer) + padding;
	Point* pTarget = (Point*)target;
	Point point = {};   // formats without colour load black, as the LAZ path does
	int32_t XYZ[3];
	double scale_x = header.scale[0];
	double scale_y = header.scale[1];
	double scale_z = header.scale[2];
	double offset_x = header.offset[0] + translation[0];
	double offset_y = header.offset[1] + translation[1];
	double offset_z = header.offset[2] + translation[2];


	for(uint64_t i = 0; i < numPoints; i++){

		memcpy(XYZ, source + header.bytesPerPoint * i, 12);

		point.x = double(XYZ[0]) * scale_x + offset_x;
		point.y = double(XYZ[1]) * scale_y + offset_y;
		point.z = double(XYZ[2]) * scale_z + offset_z;

		if(offset_rgb > 0){
			uint16_t* rgb = (uint16_t*)(source + header.bytesPerPoint * i + offset_rgb);
			point.r = rgb[0] > 255 ? rgb[0] / 256 : rgb[0];
			point.g = rgb[1] > 255 ? rgb[1] / 256 : rgb[1];
			point.b = rgb[2] > 255 ? rgb[2] / 256 : rgb[2];
		}

		pTarget[i] = point;
	}

	return true;
}


// ---- checks before a file is loaded ------------------------------------------

namespace{

// A file opened for the checks, closed when it goes out of scope.
struct CheckedFile{
	FILE* f = nullptr;
	uint64_t size = 0;

	bool open(const string& path){
		std::error_code ec;
		size = fs::file_size(path, ec);
		if(ec) return false;
		f = fopen(path.c_str(), "rb");
		return f != nullptr;
	}

	~CheckedFile(){ if(f != nullptr) fclose(f); }

	// n bytes at offset; false where the file ends before them
	bool read(uint64_t offset, void* target, uint64_t n){
		return offset <= size && n <= size - offset
			&& fseek_64_all_platforms(f, int64_t(offset), SEEK_SET) == 0
			&& fread(target, 1, n, f) == n;
	}
};

template<typename T>
T fieldAt(const uint8_t* bytes, uint64_t offset){
	T value;
	memcpy(&value, bytes + offset, sizeof(T));
	return value;
}

string textf(const char* fmt, double a, double b = 0.0){
	char text[128];
	snprintf(text, sizeof(text), fmt, a, b);
	return text;
}

}

// Point record sizes of LAS point formats 0 to 10, as laszip checks them (laszip.cpp, LASzip::setup)
static const uint64_t minRecordSize[11] = {20, 28, 26, 34, 57, 63, 30, 36, 38, 59, 67};

string lasFileProblem(const string& path){
	CheckedFile file;
	if(!file.open(path)) return "cannot be opened";

	// the header: 227 bytes in LAS 1.0 to 1.2, 235 in 1.3, 375 in 1.4
	uint8_t h[375] = {};
	if(file.size < 227 || !file.read(0, h, std::min<uint64_t>(file.size, sizeof(h)))) return "too short to be a LAS file";
	if(memcmp(h, "LASF", 4) != 0) return "not a LAS file";

	int versionMajor           = h[24];
	int versionMinor           = h[25];
	uint64_t headerSize        = fieldAt<uint16_t>(h, 94);
	uint64_t offsetToPointData = fieldAt<uint32_t>(h, 96);
	uint32_t numVLRs           = fieldAt<uint32_t>(h, 100);
	int format                 = h[104];
	uint64_t bytesPerPoint     = fieldAt<uint16_t>(h, 105);
	uint64_t numPoints         = versionMinor <= 3 ? fieldAt<uint32_t>(h, 107) : fieldAt<uint64_t>(h, 247);   // as loadHeader reads them

	if(versionMajor != 1) return textf("LAS version %.0f.%.0f", versionMajor, versionMinor);
	if(headerSize < (versionMinor >= 4 ? 375u : 227u) || headerSize > file.size) return "damaged header";
	if(offsetToPointData < headerSize || offsetToPointData > file.size) return "damaged header";
	if(numPoints == 0) return "no points";
	for(int axis = 0; axis < 3; axis++){
		double max = fieldAt<double>(h, 179 + 16 * axis);
		double min = fieldAt<double>(h, 187 + 16 * axis);
		if(!std::isfinite(min) || !std::isfinite(max) || min > max) return "the bounds in the header are not valid";
	}

	// laszip marks compressed point data in the top two bits of the format (laszip_dll.cpp)
	bool compressed = (format & 0xC0) != 0;
	if(compressed && !iEndsWith(path, ".laz")) return "compressed (LAZ) but named .las";

	if(!compressed){
		if(format > 10) return textf("point format %.0f is not a LAS point format", format);
		if(bytesPerPoint < minRecordSize[format]) return textf("point records too short for point format %.0f", format);
		uint64_t pointsInFile = (file.size - offsetToPointData) / bytesPerPoint;
		if(pointsInFile < numPoints){
			return textf("cut short: %.2f M of %.2f M points are in the file", pointsInFile / 1e6, numPoints / 1e6);
		}
		return "";
	}

	// LAZ: laszip's own record says how the points are compressed
	uint16_t compressor = 0;
	uint32_t chunkSize = 0;
	bool haveLaszipRecord = false;
	uint64_t vlrStart = headerSize;
	for(uint32_t i = 0; i < numVLRs && vlrStart + 54 <= offsetToPointData; i++){
		uint8_t vlr[54];
		if(!file.read(vlrStart, vlr, sizeof(vlr))) break;
		uint64_t length = fieldAt<uint16_t>(vlr, 20);
		if(memcmp(vlr + 2, "laszip encoded", 15) == 0 && fieldAt<uint16_t>(vlr, 18) == 22204 && length >= 34){
			haveLaszipRecord = file.read(vlrStart + 54, &compressor, 2) && file.read(vlrStart + 54 + 12, &chunkSize, 4);
		}
		vlrStart += 54 + length;
	}
	if(!haveLaszipRecord) return "no laszip record in the header";
	if(compressor != 2 && compressor != 3) return "";   // not chunked: no chunk table to check

	// Chunked, laszip's default: the point data starts with the position of
	// the chunk table, which the writer fills in when it closes the file
	// (laswritepoint.cpp). Without the table, each batch's seek decodes the
	// file from its start, and decoding into the missing end of a cut file
	// crashed the viewer (a divide by zero in laszip's arithmetic decoder).
	int64_t tablePosition = 0;
	if(!file.read(offsetToPointData, &tablePosition, 8)) return "cut short: the points are missing";
	if(uint64_t(tablePosition) == offsetToPointData) return "no chunk table: the program writing the file stopped before finishing it";
	if(tablePosition == -1){
		// A writer that could not seek back puts the position in the last 8
		// bytes instead. In a cut file those are other bytes.
		if(!file.read(file.size - 8, &tablePosition, 8) || tablePosition < int64_t(offsetToPointData) + 8 || uint64_t(tablePosition) + 16 > file.size){
			return "cut short or damaged: the chunk table cannot be found";
		}
	}
	if(tablePosition < int64_t(offsetToPointData) + 8) return "damaged: the chunk table cannot be found";
	if(uint64_t(tablePosition) + 8 > file.size){
		return textf("cut short: at least %.1f MB missing from the end", double(uint64_t(tablePosition) + 8 - file.size) / 1e6);
	}

	uint32_t table[2] = {};   // version, number of chunks (lasreadpoint.cpp, read_chunk_table)
	if(!file.read(uint64_t(tablePosition), table, 8) || table[0] != 0) return "damaged chunk table";
	if(chunkSize != 0xFFFFFFFFu && uint64_t(table[1]) * chunkSize < numPoints){   // all chunks but the last hold chunkSize points
		return "cut short or damaged: the chunk table holds fewer points than the header";
	}

	return "";
}
