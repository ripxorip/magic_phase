#if defined(__linux__) || defined(__APPLE__)

#include "PlatformSharedMemory.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>

PlatformSharedMemory::~PlatformSharedMemory()
{
    close();
}

bool PlatformSharedMemory::open (const char* name, size_t size, bool& createdNew)
{
    createdNew = false;

    if (ptr_ != nullptr)
        return true;  // Already open

    // Try to create exclusively first
    fd_ = shm_open (name, O_CREAT | O_EXCL | O_RDWR, 0666);
    if (fd_ >= 0)
    {
        // We created it - need to set size and initialize
        createdNew = true;
        if (ftruncate (fd_, static_cast<off_t> (size)) != 0)
        {
            ::close (fd_);
            shm_unlink (name);
            fd_ = -1;
            return false;
        }
    }
    else if (errno == EEXIST)
    {
        // Already exists - open and check size
        fd_ = shm_open (name, O_RDWR, 0666);
        if (fd_ < 0)
            return false;

        // Check if existing segment has correct size
        struct stat st;
        if (fstat (fd_, &st) == 0 && static_cast<size_t> (st.st_size) != size)
        {
            // Size mismatch - close and recreate
            ::close (fd_);
            shm_unlink (name);
            fd_ = shm_open (name, O_CREAT | O_EXCL | O_RDWR, 0666);
            if (fd_ < 0)
            {
                // Race condition - another process recreated it, try opening again
                fd_ = shm_open (name, O_RDWR, 0666);
                if (fd_ < 0)
                    return false;
            }
            else
            {
                createdNew = true;
                if (ftruncate (fd_, static_cast<off_t> (size)) != 0)
                {
                    ::close (fd_);
                    shm_unlink (name);
                    fd_ = -1;
                    return false;
                }
            }
        }
    }
    else
    {
        return false;
    }

    // Map into our address space
    ptr_ = mmap (nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (ptr_ == MAP_FAILED)
    {
        ptr_ = nullptr;
        ::close (fd_);
        fd_ = -1;
        return false;
    }

    size_ = size;
    return true;
}

void PlatformSharedMemory::close()
{
    if (ptr_ != nullptr)
    {
        munmap (ptr_, size_);
        ptr_ = nullptr;
    }
    if (fd_ >= 0)
    {
        ::close (fd_);
        fd_ = -1;
    }
    size_ = 0;
    // Note: We don't shm_unlink - other instances may still need it
}

#endif // POSIX
