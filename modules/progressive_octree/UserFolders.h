#pragma once

#include <filesystem>

// The user's Documents folder, the one File Explorer shows, which may be
// redirected to OneDrive. An empty path if it cannot be found.
std::filesystem::path documentsFolder();

// Opens a folder in the system's file browser. False if that failed.
bool openInFileBrowser(const std::filesystem::path& dir);
