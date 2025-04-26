#include <FileSystem.hpp>

#include <bits/types/struct_timespec.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/xattr.h>
#include <sys/types.h>

namespace LogFs
{
    void ForgetMulti(fuse_req_t req, size_t count, struct fuse_forget_data *forgets)
    {
        Node **end = reinterpret_cast<Node**>(forgets);
        for (size_t i = 0; i < count; i++)
        {
            Node *cur = &Fs.getNode(forgets[i].ino);
            if (cur->lookup -= forgets[i].nlookup == 0)
            {
                *end++ = cur;
            }
        }

        if (Node **cur = reinterpret_cast<Node**>(forgets); cur != end)
        {
            for (std::unique_lock lock(Fs.nodesMutex); cur != end; cur++)
            {
                if ((*cur)->lookup == 0) // lookup is still zero
                {
                    Fs.nodes.erase((*cur)->ino);
                }
            }
        }

        ::fuse_reply_none(req);
    }
    void Forget(fuse_req_t req, fuse_ino_t ino, uint64_t nlookup)
    {
        fuse_forget_data forgets
        {
            .ino = ino,
            .nlookup = nlookup
        };
        ForgetMulti(req, 1, &forgets);
    }
    void Getattr(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi)
    {
        struct stat attr {};
        if (::fstat(Fs.getNode(ino).fd,  &attr) == 0)
        {
            ::fuse_reply_attr(req, &attr, Fs.AttrTimeout);
        }
        else
        {
            ::fuse_reply_err(req, errno);
        }
    }
    void Setattr(fuse_req_t req, fuse_ino_t ino, struct stat *attr, int toSet, struct fuse_file_info *fi)
    {
        int fd = Fs.getNode(ino).fd;
        char fdname[12];
        //char fdname[26];
        if (toSet & (FUSE_SET_ATTR_UID | FUSE_SET_ATTR_GID | FUSE_SET_ATTR_MODE | FUSE_SET_ATTR_ATIME | FUSE_SET_ATTR_MTIME | FUSE_SET_ATTR_ATIME_NOW | FUSE_SET_ATTR_MTIME_NOW))
        {
            snprintf(fdname, 12, "%d", fd);
            //snprintf(fdname, 26, "/proc/self/fd/%d", fd);
        }

        toSet &= ~FUSE_SET_ATTR_CTIME; /// @todo: handle ctime

        if ((toSet & FUSE_SET_ATTR_SIZE) != 0 && ::ftruncate(fd, attr->st_size) != 0 /* && (toSet & ~FUSE_SET_ATTR_SIZE) == 0 */)
        {
            ::fuse_reply_err(req, errno);
            return;
        }
        if ((toSet & (FUSE_SET_ATTR_UID | FUSE_SET_ATTR_GID)) != 0 && ::fchownat
            (
                Fs.ProcFd, fdname,
                //fd, "", would work for symlinks according to doc, but fails for other types.
                ((toSet & (FUSE_SET_ATTR_UID)) != 0) ? attr->st_uid : -1,
                ((toSet & (FUSE_SET_ATTR_GID)) != 0) ? attr->st_gid : -1,
                0
            ) != 0 /* && (toSet & ~(FUSE_SET_ATTR_UID | FUSE_SET_ATTR_GID)) == 0 */)
        {
            ::fuse_reply_err(req, errno);
            return;
        }
        if ((toSet & FUSE_SET_ATTR_MODE) != 0 && ::fchmodat(Fs.ProcFd, fdname, attr->st_mode, 0) != 0/* && (toSet & ~FUSE_SET_ATTR_MODE) == 0 */)
        {
            ::fuse_reply_err(req, errno);
            return;
        }
        if ((toSet & (FUSE_SET_ATTR_ATIME | FUSE_SET_ATTR_MTIME | FUSE_SET_ATTR_ATIME_NOW | FUSE_SET_ATTR_MTIME_NOW)) != 0) // utimensat
        {
            timespec times[2]
            {
                (toSet & FUSE_SET_ATTR_ATIME_NOW) ? timespec{ .tv_nsec = UTIME_NOW } : (toSet & FUSE_SET_ATTR_ATIME) != 0 ? attr->st_atim : timespec{ .tv_nsec = UTIME_OMIT },
                (toSet & FUSE_SET_ATTR_MTIME_NOW) ? timespec{ .tv_nsec = UTIME_NOW } : (toSet & FUSE_SET_ATTR_MTIME) != 0 ? attr->st_mtim : timespec{ .tv_nsec = UTIME_OMIT }
            };

            if (::utimensat(Fs.ProcFd, fdname, times, 0) != 0 /* && (toSet & ~(FUSE_SET_ATTR_ATIME | FUSE_SET_ATTR_MTIME)) == 0 */)
            {
                ::fuse_reply_err(req, errno);
                return;
            }
        }

        if (::fstat(fd, attr) != 0)
        {
            ::fuse_reply_err(req, errno);
        }
        else
        {
            ::fuse_reply_attr(req, attr, Fs.AttrTimeout);
        }
    }
    void Setxattr(fuse_req_t req, fuse_ino_t ino, const char *name, const char *value, size_t size, int flags)
    {
        ::fuse_reply_err(req, (::fsetxattr(Fs.getNode(ino).fd, name, value, size, flags) == 0) ? 0 : errno);
    }
    void Getxattr(fuse_req_t req, fuse_ino_t ino, const char *name, size_t size)
    {
        Fs.Buffer.resize(size);
        ssize_t res = ::fgetxattr(Fs.getNode(ino).fd, name, Fs.Buffer.data(), size);
        if (res > size)
        {
            res = 0;
            errno = ERANGE;
        }
        if (res == -1)
        {
            ::fuse_reply_err(req, errno);
        }
        else if (size == 0)
        {
            ::fuse_reply_xattr(req, res);
        }
        else
        {
            ::fuse_reply_buf(req, Fs.Buffer.data(), res);
        }
    }
    void Listxattr(fuse_req_t req, fuse_ino_t ino, size_t size)
    {
        Fs.Buffer.resize(size);
        ssize_t res = ::flistxattr(Fs.getNode(ino).fd, Fs.Buffer.data(), size);
        if (res > size)
        {
            res = 0;
            errno = ERANGE;
        }
        if (res == -1)
        {
            ::fuse_reply_err(req, errno);
        }
        else if (size == 0)
        {
            ::fuse_reply_xattr(req, res);
        }
        else
        {
            ::fuse_reply_buf(req, Fs.Buffer.data(), res);
        }
    }
    void Removexattr(fuse_req_t req, fuse_ino_t ino, const char *name)
    {
        ::fuse_reply_err(req, (::fremovexattr(Fs.getNode(ino).fd, name) == 0) ? 0 : errno);
    }
    void Statfs(fuse_req_t req, fuse_ino_t ino)
    {
        struct statvfs buf{};
        if (::fstatvfs(Fs.getNode(ino).fd, &buf) == 0)
        {
            ::fuse_reply_statfs(req, &buf);
        }
        else
        {
            ::fuse_reply_err(req, errno);
        }
    }
}
