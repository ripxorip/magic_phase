#pragma once
#include <cstddef>
#include <cstdint>

/**
 * Thin cross-platform shared memory abstraction.
 *
 * Creates or attaches to a named shared memory region that can be
 * accessed by multiple processes (plugin instances).
 *
 * Platform implementations:
 * - POSIX (macOS/Linux): shm_open + mmap
 * - Windows: CreateFileMapping + MapViewOfFile
 */
class PlatformSharedMemory
{
public:
    PlatformSharedMemory() = default;
    ~PlatformSharedMemory();

    // Non-copyable
    PlatformSharedMemory (const PlatformSharedMemory&) = delete;
    PlatformSharedMemory& operator= (const PlatformSharedMemory&) = delete;

    /**
     * Open or create shared memory region.
     * @param name Unique name for the shared memory (e.g., "/magic_phase")
     * @param size Size in bytes
     * @param createdNew Output: true if we created new memory (should initialize)
     * @return true on success
     */
    bool open (const char* name, size_t size, bool& createdNew);

    /**
     * Close and unmap the shared memory.
     * Does NOT destroy the shared memory - other instances may still use it.
     */
    void close();

    /**
     * Get pointer to mapped memory.
     * @return Pointer to shared memory, or nullptr if not open
     */
    void* data() const { return ptr_; }

    /**
     * Check if shared memory is currently open.
     */
    bool isOpen() const { return ptr_ != nullptr; }

    /**
     * Get the size of the mapped region.
     */
    size_t size() const { return size_; }

private:
    void* ptr_ = nullptr;
    size_t size_ = 0;

#if defined(_WIN32)
    void* fileHandle_ = nullptr;  // HANDLE
#else
    int fd_ = -1;
#endif
};
