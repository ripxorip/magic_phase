#if defined(_WIN32)

#include "PlatformSharedMemory.h"
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

PlatformSharedMemory::~PlatformSharedMemory()
{
    close();
}

bool PlatformSharedMemory::open (const char* name, size_t size, bool& createdNew)
{
    createdNew = false;

    if (ptr_ != nullptr)
        return true;  // Already open

    // Windows shared memory names use "Global\" or "Local\" prefix
    // For same-user DAW instances, Local is fine
    // Skip leading slash if present (POSIX convention)
    const char* cleanName = (name[0] == '/') ? name + 1 : name;

    char fullName[256];
    snprintf (fullName, sizeof (fullName), "Local\\%s", cleanName);

    // Calculate high/low size for CreateFileMapping
    DWORD sizeHigh = static_cast<DWORD> ((size >> 32) & 0xFFFFFFFF);
    DWORD sizeLow = static_cast<DWORD> (size & 0xFFFFFFFF);

    // Try to create or open existing
    fileHandle_ = CreateFileMappingA (
        INVALID_HANDLE_VALUE,  // Use paging file
        nullptr,               // Default security
        PAGE_READWRITE,
        sizeHigh,
        sizeLow,
        fullName
    );

    if (fileHandle_ == nullptr)
        return false;

    // Check if we created it or opened existing
    createdNew = (GetLastError() != ERROR_ALREADY_EXISTS);

    // Map into our address space
    ptr_ = MapViewOfFile (
        fileHandle_,
        FILE_MAP_ALL_ACCESS,
        0, 0,  // Offset
        size
    );

    if (ptr_ == nullptr)
    {
        CloseHandle (fileHandle_);
        fileHandle_ = nullptr;
        return false;
    }

    size_ = size;
    return true;
}

void PlatformSharedMemory::close()
{
    if (ptr_ != nullptr)
    {
        UnmapViewOfFile (ptr_);
        ptr_ = nullptr;
    }
    if (fileHandle_ != nullptr)
    {
        CloseHandle (fileHandle_);
        fileHandle_ = nullptr;
    }
    size_ = 0;
    // Note: The mapping object persists until all handles are closed
}

#endif // _WIN32
