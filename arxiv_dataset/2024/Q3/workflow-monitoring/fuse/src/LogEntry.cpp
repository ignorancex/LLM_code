#include <FileSystem.hpp>

#include <cstdint>

#include <ctime>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <shared_mutex>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

namespace LogFs
{
    LogFs::LogEntry LogEntry::GetOpen(int pid, uint64_t inode, int flags)
    {
        LogEntry le;
        le.start(pid, inode, 'O', -1);
        le.flags = flags;
        le.offset = le.size = 0;
        return le;
    }
    LogFs::LogEntry LogEntry::GetClose(int pid, uint64_t inode, uint64_t fh)
    {
        LogEntry le;
        le.start(pid, inode, 'C', fh);
        le.offset = le.size = le.flags = 0;
        return le;
    }
    LogFs::LogEntry LogEntry::GetRead(int pid, uint64_t inode, uint64_t fh, off_t off, size_t size)
    {
        LogEntry le;
        le.start(pid, inode, 'R', fh);
        le.size = size;
        le.offset = off;
        le.flags = 0;
        return le;
    }
    LogFs::LogEntry LogEntry::GetWrite(int pid, uint64_t inode, uint64_t fh, off_t off, size_t size)
    {
        LogEntry le;
        le.start(pid, inode, 'W', fh);
        le.size = size;
        le.offset = off;
        le.flags = 0;
        return le;
    }
    void LogEntry::end(int res)
    {
        GetRTime(&rTimeEnd);
        GetPidStatTimes(pid, &uTimeEnd, &sTimeEnd);
        if (event == 'O')
        {
            if (res >= 0)
            {
                filehandle = CurFh++;
                result = 0;
                {
                    std::unique_lock lk(MutexFhs);
                    Fhs[res] = filehandle;
                }
            }
            else
            {
                result = res;
                filehandle = -1;
            }
        }
        else
        {
            result = res;
        }
    }
    std::span<char> LogEntry::getBuf()
    {
        snprintf(buffer, sizeof(buffer),
            "%*ld.%0*ld,%*ld.%0*ld,%*d,%*ld.%0*ld,%*ld.%0*ld,%*ld.%0*ld,%*lu.%0*ld,%*lu,%c,%*d,%*ld,%*lu,%*lu,0x%0*x,%*s",
            SizeTimeSec, rTimeStart.tv_sec, SizeTimeNsec, rTimeStart.tv_nsec / 1000000,
            SizeTimeSec, rTimeEnd.tv_sec,   SizeTimeNsec, rTimeEnd.tv_nsec   / 1000000,
            SizePid, pid,
            SizeTimeSec, uTimeStart.tv_sec, SizeTimeNsec, uTimeStart.tv_nsec / 1000000,
            SizeTimeSec, uTimeEnd.tv_sec  , SizeTimeNsec, uTimeEnd.tv_nsec   / 1000000,
            SizeTimeSec, sTimeStart.tv_sec, SizeTimeNsec, sTimeStart.tv_nsec / 1000000,
            SizeTimeSec, sTimeEnd.tv_sec  , SizeTimeNsec, sTimeEnd.tv_nsec   / 1000000,
            SizeInode, inode,
            event,
            SizeResult, result,
            SizeFilehandle, filehandle,
            SizeOffset, offset,
            SizeSize, size,
            SizeFlags - 2, flags,
            SizePath, ""
        );
        buffer[SizeEntry-1] = '\n'; // replace null with newline
        return { buffer, SizeEntry };
    }
    bool LogEntry::unknownFh() const
    {
        return (filehandle == -2);
    }

    void LogEntry::start(int pid, uint64_t ino, char evt, uint64_t fh)
    {
        if (fh != -1)
        {
            std::shared_lock lk(MutexFhs);
            if (auto it = Fhs.find(fh); it == Fhs.end())
            {
                filehandle = -2;
            }
            else
            {
                filehandle = it->second;
            }
        }
        else
        {
            filehandle = fh;
        }
        if (ino != 0)
        {
            std::shared_lock lk(MutexInodes);
            auto it = Inodes.find(ino);
            inode = (it != Inodes.end()) ? it->second : 0;
        }
        event = evt;
        this->pid = pid;
        GetRTime(&rTimeStart);
        GetPidStatTimes(pid, &uTimeStart, &sTimeStart);
    }
    
    timespec GetRtMonDiff()
    {
        timespec rt, mon;
        if (::clock_gettime(CLOCK_REALTIME, &rt))
        {
            std::cerr << "Could not init realtime clock." << std::endl;
            std::exit(-1);
        }
        if (::clock_gettime(CLOCK_MONOTONIC_COARSE, &mon))
        {
            std::cerr << "Could not init monotonic clock." << std::endl;
            std::exit(-1);
        }
        // real - mon
        if (rt.tv_nsec < mon.tv_nsec)
        {
            rt.tv_nsec += 1000000000;
            rt.tv_sec -= 1;
        }
        rt.tv_sec -= mon.tv_sec;
        rt.tv_nsec -= mon.tv_nsec;
        return rt;
    }
    void LogEntry::GetRTime(timespec *rTime)
    {
        static timespec diff = GetRtMonDiff();
        if (::clock_gettime(CLOCK_MONOTONIC_COARSE, rTime) != 0)
        {
            *rTime = { 0, 0 };
        }
        else
        {
            rTime->tv_sec += diff.tv_sec;
            if ((rTime->tv_nsec += diff.tv_nsec) > 1000000000)
            {
                rTime->tv_nsec -= 1000000000;
                rTime->tv_sec += 1;
            }
        }
    }
    void LogEntry::GetPidStatTimes(int pid, timespec *utime, timespec *stime)
    {
        /// @todo: implement
        *utime = { 0, 0 };
        *stime = { 0, 0 };
    }

    void LogEntry::InformNewNode(uint64_t inode, bool created)
    {
        std::shared_lock lk(MutexInodes);
        if (auto it = Inodes.find(inode); it != Inodes.end())
        {
            if (created)
            {
                it->second = ++CurInode;
            }
        }
        else
        {
            lk.unlock();
            std::unique_lock ulk(MutexInodes);
            Inodes[inode] = ++CurInode;
        }
    }

    /// @todo: find way to eliminate old inodes from map
    /// @todo: find way to eliminate old fhs from map

    std::unordered_map<uint64_t, uint64_t> LogEntry::Fhs;
    std::shared_mutex LogEntry::MutexFhs;
    std::atomic<int64_t> LogEntry::CurFh = 0;
    std::unordered_map<uint64_t, uint64_t> LogEntry::Inodes;
    std::shared_mutex LogEntry::MutexInodes;
    std::atomic<int64_t> LogEntry::CurInode = 0;
}
