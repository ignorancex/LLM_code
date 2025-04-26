#include <FileSystem.hpp>

#include <fuse3/fuse_lowlevel.h>
#include <unistd.h>
#include <sys/stat.h>

namespace LogFs
{
    void Init(void *userdata, struct fuse_conn_info *conn)
    {
        // conn->proto_major; // minor version
        // conn->proto_minor; // major version
        // conn->max_write; // set by fuselib
        // conn->max_read = 0; // set by fuselib, answer currently ignored
        // conn->max_readahead; // set by fuselib
        // conn->capable; // set by fuselib, DO NOT WRITE
        // conn->want = 0; // see below
        // conn->max_background; // setby fuselib
        // conn->congestion_threshold; // setby fuselib
        // conn->time_gran = 1; // nano second resolution time stamps, currently ignored

        conn->want &= ~FUSE_CAP_AUTO_INVAL_DATA;
        conn->want &= ~FUSE_CAP_PARALLEL_DIROPS;    // @todo: i guess it doesn't help, still should try. Trouble otherwise
        conn->want |= FUSE_CAP_WRITEBACK_CACHE;
        conn->want |= FUSE_CAP_POSIX_ACL;
        conn->want |= FUSE_CAP_CACHE_SYMLINKS;
        conn->want |= FUSE_CAP_EXPLICIT_INVAL_DATA;

        // conn->want options:

        // set if kernel supports it:
        // FUSE_CAP_ASYNC_READ          // enables parallel read ops
    	// FUSE_CAP_AUTO_INVAL_DATA     // issue getattr not only when EOF (always set, ctrl with cache timeout and inval_entry)
    	// FUSE_CAP_ASYNC_DIO           // parallel directo io on same file descriptor
    	// FUSE_CAP_ATOMIC_O_TRUNC      // trunc is recognized by open (not explicitly done by kernel afterwards)

        // set if kernel supports it and cb functions implemented:
    	// FUSE_CAP_POSIX_LOCKS         // getlk/setlk(w)
    	// FUSE_CAP_FLOCK_LOCKS         // flock
    	// FUSE_CAP_READDIRPLUS         // readdirplus
    	// FUSE_CAP_READDIRPLUS_AUTO    // kernel is allowed to also use readdir if readdirplus is available

        // not set by default (only set this if part of conn->capable):
        // FUSE_CAP_EXPORT_SUPPORT      // lookup . and ..
        // FUSE_CAP_DONT_MASK           // umask to be not applied to create mode by kernel
        // FUSE_CAP_WRITEBACK_CACHE     // enable writeback cache
        // FUSE_CAP_POSIX_ACL           // ACL support (so kernel caches it), force-enables default-permissions
        // FUSE_CAP_CACHE_SYMLINKS      // Cache symlink in page cache, invalidate via fuse_lowlevel_notify_inval_inode()
        // FUSE_CAP_EXPLICIT_INVAL_DATA // only explicit data invalidation in page cache via fuse_lowlevel_notify_inval_inode() (AUTO_INVAL_DATA wins if both set)

        // only informed with 
    	// FUSE_CAP_PARALLEL_DIROPS     // parallel lookup and/or readdir on same directory. illegaly ignored by fuselib in want 
    	// FUSE_CAP_SPLICE_READ         // writebuf
    	// FUSE_CAP_IOCTL_DIR           // ioctl can be done on directories
    	// FUSE_CAP_HANDLE_KILLPRIV     // kill suid/sgid/cap on write/chown/trunc. illegaly ignored by fuselib in want 
        // FUSE_CAP_NO_OPEN_SUPPORT     // no need to implement open (nullptr in ops or reply ENOSYS)
        // FUSE_CAP_NO_OPENDIR_SUPPORT  // no need to implement opendir (nullptr in ops or reply ENOSYS)

        {
            std::unique_lock lock(Fs.createMutex);
            Fs.ProcFd = Fs.ProcFd != -1 ? Fs.ProcFd : open("/proc/self/fd", O_RDONLY | O_PATH, 0);
        }

        ::umask(0);

        LogEntry::InformNewNode(FUSE_ROOT_ID, false);
    }

    void Destroy(void *userdata)
    {
        Fs.killPollThread();
        ::close(Fs.pollpipeFd);
        {
            std::unique_lock cl(Fs.createMutex);
            std::unique_lock nl(Fs.nodesMutex);
            Fs.nodes.clear();
            ::close(Fs.ProcFd);
            Fs.ProcFd = -1;
        }
    }
}
