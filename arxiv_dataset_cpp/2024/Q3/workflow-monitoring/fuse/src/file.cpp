#include <FileSystem.hpp>

namespace LogFs
{
    void Open(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi)
    {
        auto log = LogEntry::GetOpen(::fuse_req_ctx(req)->pid, ino, fi->flags);

        char fdname[12];
        int inofd = Fs.getNode(ino).fd;
        snprintf(fdname, 12, "%d", inofd);
        int res = ::openat(Fs.ProcFd, fdname, fi->flags, 0);
        res = (res == -1) ? -errno : res;

        log.end(res);

        if (res < 0)
        {
            ::fuse_reply_err(req, -res);
        }
        else
        {
            fi->direct_io = Fs.DirectIo;
            fi->keep_cache = Fs.KeepCache;
            fi->fh = static_cast<uint64_t>(res);
            ::fuse_reply_open(req, fi);
        }

        auto data = log.getBuf();
        auto size = readlinkat(Fs.ProcFd, fdname, &data.data()[LogEntry::OffPath], LogEntry::SizePath);
        Fs.writeLog(data);
    }
    void Read(fuse_req_t req, fuse_ino_t ino, size_t size, off_t offset, struct fuse_file_info *fi)
    {
        auto log = LogEntry::GetRead(::fuse_req_ctx(req)->pid, ino, fi->fh, offset, size);

        fuse_bufvec bv
        {
            .count = 1,
            .idx = 0,
            .off = 0,
            .buf{{
                .size = size,
                .flags = fuse_buf_flags(FUSE_BUF_IS_FD | FUSE_BUF_FD_SEEK),
                .mem = nullptr,
                .fd = static_cast<int>(fi->fh),
                .pos = offset
            }}
        };
        int res = ::fuse_reply_data(req, &bv, fuse_buf_copy_flags(0));
        res = (res == 0) ? bv.off : res;

        log.end(res);
        Fs.writeLog(log.getBuf());
    }
    void Write(fuse_req_t req, fuse_ino_t ino, const char *buf, size_t size, off_t off, struct fuse_file_info *fi)
    {
        auto log = LogEntry::GetWrite(::fuse_req_ctx(req)->pid, ino, fi->fh, off, size);
        
        int res = ::pwrite(static_cast<int>(fi->fh), buf, size, off);
        res = (res == -1) ? -errno : res;

        log.end(res);

        if (res < 0)
        {
            ::fuse_reply_err(req, -res);
        }
        else
        {
            ::fuse_reply_write(req, res);
        }

        Fs.writeLog(log.getBuf());
    }
    void WriteBuf(fuse_req_t req, fuse_ino_t ino, struct fuse_bufvec *bufv, off_t off, struct fuse_file_info *fi)
    {
        if ((bufv->buf->flags & FUSE_BUF_IS_FD) == 0)
        {
            Write(req, ino, reinterpret_cast<char*>(bufv->buf->mem), bufv->buf->size, off, fi);
            return;
        }

        auto log = LogEntry::GetWrite(::fuse_req_ctx(req)->pid, ino, fi->fh, off, bufv->buf->size);
        
        off64_t off64 = off;
        ssize_t res = ::splice(bufv->buf->fd, nullptr, static_cast<int>(fi->fh), &off, bufv->buf->size, SPLICE_F_MOVE);
        res = (res == -1) ? -errno : res;
        log.end(res);
        
        if (res < 0)
        {
            ::fuse_reply_err(req, -res);
        }
        else
        {
            bufv->buf->pos += res;
            ::fuse_reply_write(req, res);
        }
        
        Fs.writeLog(log.getBuf());
    }
    void Release(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi)
    {
        auto log = LogEntry::GetClose(::fuse_req_ctx(req)->pid, ino, fi->fh);

        int res = (::close(static_cast<int>(fi->fh)) == 0) ? 0 : errno;
        
        if (!log.unknownFh())
        {
            log.end(-res);
        }
        
        ::fuse_reply_err(req, res);

        if (!log.unknownFh())
        {
            Fs.writeLog(log.getBuf());
        }
    }
    void Fsync(fuse_req_t req, fuse_ino_t ino, int datasync, struct fuse_file_info *fi)
    {
        int fd = static_cast<int>(fi->fh);
        int res = (datasync != 0) ? ::fdatasync(fd) : ::fsync(fd);
        fuse_reply_err(req, (res == 0) ? 0 : errno);
    }
    void Fallocate(fuse_req_t req, fuse_ino_t ino, int mode, off_t offset, off_t length, struct fuse_file_info *fi)
    {
        ::fuse_reply_err(req, (::fallocate(static_cast<int>(fi->fh), mode, offset, length) == 0 ? 0 : errno));
    }

    void Poll(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi, struct fuse_pollhandle *pollhandle)
    {
        pollfd pfd
        {
            .fd = static_cast<int>(fi->fh),
            .events = static_cast<short>(fi->poll_events),
            .revents = 0
        };
        
        if (::poll(&pfd, 1, 0) == -1)
        {
            ::fuse_reply_err(req, errno);
            if (pollhandle != nullptr)
            {
                ::fuse_pollhandle_destroy(pollhandle);
            }
        }
        else
        {
            ::fuse_reply_poll(req, pfd.revents);
            if (pollhandle != nullptr)
            {
                pfd.revents = 0;
                FileSystem::PollMessage p
                {
                    .ph = pollhandle,
                    .pfd = pfd
                };
                Fs.poll(p);
            }
        }
    }
    void CopyFileRange(fuse_req_t req, fuse_ino_t srcIno, off_t srcOff, struct fuse_file_info *srcFi, fuse_ino_t dstIno, off_t dstOff, struct fuse_file_info *dstFi, size_t len, int flags)
    {
        off64_t srcOff64 = srcOff, dstOff64 = dstOff;
        ssize_t res = ::copy_file_range(static_cast<int>(srcFi->fh), &srcOff64, static_cast<int>(dstFi->fh), &dstOff64, len, flags);
        if (res != -1)
        {
            ::fuse_reply_write(req, res);
        }
        else
        {
            ::fuse_reply_err(req, errno);
        }
    }
    void Lseek(fuse_req_t req, fuse_ino_t ino, off_t off, int whence, struct fuse_file_info *fi)
    {
        off_t res = ::lseek(static_cast<int>(fi->fh), off, whence);
        if (res != ((off_t)-1))
        {
            ::fuse_reply_lseek(req, res);
        }
        else
        {
            ::fuse_reply_err(req, errno);
        }
    }
}
