// Folders outside the build tree, for files that should outlast it (saved
// looks). The Windows calls stay in this file, as in CrashHandler.cpp, so the
// main source does not include windows.h. Elsewhere: HOME/Documents, and no
// file browser.

#include "UserFolders.h"

#ifdef _WIN32

#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#include <shlobj.h>

#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "ole32.lib")

std::filesystem::path documentsFolder(){
	PWSTR path = nullptr;
	std::filesystem::path result;
	// the folder's current location, following a redirection to OneDrive
	if(SUCCEEDED(SHGetKnownFolderPath(FOLDERID_Documents, KF_FLAG_DEFAULT, nullptr, &path))){
		result = path;
	}
	CoTaskMemFree(path);   // the caller frees it whether or not the call succeeded
	return result;
}

bool openInFileBrowser(const std::filesystem::path& dir){
	// ShellExecute can pass the request to a shell extension through COM, so
	// its documentation asks for COM to be initialised first: once, on the
	// thread that draws the panel, and left so for the life of the process.
	static HRESULT com = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
	(void)com;
	// ShellExecute returns a value greater than 32 on success
	HINSTANCE r = ShellExecuteW(nullptr, L"open", dir.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
	return reinterpret_cast<INT_PTR>(r) > 32;
}

#else

#include <cstdlib>

std::filesystem::path documentsFolder(){
	const char* home = std::getenv("HOME");
	return home != nullptr ? std::filesystem::path(home) / "Documents" : std::filesystem::path();
}

bool openInFileBrowser(const std::filesystem::path&){
	return false;
}

#endif
