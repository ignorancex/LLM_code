#include <FileSystem.hpp>

#include <cstring>
#include <algorithm>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <memory>
#include <cstdint>
#include <dirent.h>

namespace LogFs
{
    static_assert(sizeof(DIR*) <= sizeof(uint64_t), "DirFd is too big.");

    int ReplyEntry(fuse_req_t req, Node &node, const struct stat &attr)
    {
        const struct fuse_entry_param entry
        {
            .ino = reinterpret_cast<fuse_ino_t>(&node),
            .generation = 0,
            .attr = attr,
            .attr_timeout = Fs.AttrTimeout,
            .entry_timeout = Fs.EntryTimeout
        };
        auto res = ::fuse_reply_entry(req, &entry);
        return res;
    }

    Node *HandleCreation(fuse_req_t req, int parentFd, const char *name, int openFlags, struct stat *attr)
    {
        Node *node = nullptr;
        if (int fd = ::openat(parentFd, name, openFlags, 0); fd != -1 && ::fstat(fd, attr) == 0)
        {
            auto ctx = ::fuse_req_ctx(req);
            if ((openFlags & O_PATH) != 0)
            {
                ::fchownat(parentFd, name, ctx->uid, ctx->gid, AT_SYMLINK_NOFOLLOW);
            }
            else
            {
                ::fchown(fd, ctx->uid, ctx->gid);
            }
            bool dropFd = false;
            {
                std::unique_lock lock(Fs.nodesMutex);
                auto [value, inserted] = Fs.nodes.emplace(std::piecewise_construct, std::forward_as_tuple(attr->st_ino), std::forward_as_tuple(fd, attr->st_ino));
                node = &value->second;
                node->lookup++;
                dropFd = !inserted;
                if (inserted && S_ISREG(attr->st_mode))
                {
                    LogEntry::InformNewNode(reinterpret_cast<uint64_t>(node), true);
                }
            }
            if (dropFd)
            {
                ::close(fd); // this should be an error, since lookup and creation both should wait for current creation to complete.
            }
        }
        return node;
    }

    void Lookup(fuse_req_t req, fuse_ino_t parent, const char *name)
    {
        struct stat attr {};
        if (Node *node = Fs.findChild(Fs.getNode(parent), name, &attr); node != nullptr)
        {
            ReplyEntry(req, *node, attr);
        }
        else
        {
            ::fuse_reply_err(req, errno);
        }
    }
    void Mknod(fuse_req_t req, fuse_ino_t parent, const char *name, mode_t mode, dev_t rdev)
    {
        int parentfd = Fs.getNode(parent).fd;
        Node *node = nullptr;
        struct stat attr{};
        {
            std::unique_lock lock(Fs.createMutex);
            if (::mknodat(parentfd, name, mode, rdev) == 0)
            {
                node = HandleCreation(req, parentfd, name, (S_ISREG(mode) || S_ISLNK(mode) || S_ISDIR(mode)) ? O_RDWR : (O_RDONLY | O_PATH), &attr);
            }
        }
        if (node != nullptr)
        {
            ReplyEntry(req, *node, attr);
        }
        else
        {
            ::fuse_reply_err(req, errno);
        }
    }
    void Mkdir(fuse_req_t req, fuse_ino_t parent, const char *name, mode_t mode)
    {
        int parentfd = Fs.getNode(parent).fd;
        Node *node = nullptr;
        struct stat attr{};
        {
            std::unique_lock lock(Fs.createMutex);
            if (::mkdirat(parentfd, name, mode) == 0)
            {
                node = HandleCreation(req, parentfd, name, O_RDONLY, &attr);
            }
        }
        if (node != nullptr)
        {
            ReplyEntry(req, *node, attr);
        }
        else
        {
            ::fuse_reply_err(req, errno);
        }
    }
    void Unlink(fuse_req_t req, fuse_ino_t parent, const char *name)
    {
        int res = 0;
        {
            std::unique_lock lock(Fs.createMutex);
            res = ::unlinkat(Fs.getNode(parent).fd, name, 0);
        }
        ::fuse_reply_err(req, (res == 0) ? 0 : errno);
    }
    void Rmdir(fuse_req_t req, fuse_ino_t parent, const char *name)
    {
        int res = 0;
        {
            std::unique_lock lock(Fs.createMutex);
            res = ::unlinkat(Fs.getNode(parent).fd, name, AT_REMOVEDIR);
        }
        ::fuse_reply_err(req, (res == 0) ? 0 : errno);
    }
    void Rename(fuse_req_t req, fuse_ino_t parent, const char *name, fuse_ino_t newparent, const char *newname, unsigned int flags)
    {
        int res = 0;
        {
            std::unique_lock lock(Fs.createMutex);
            res = ::renameat2(Fs.getNode(parent).fd, name, Fs.getNode(newparent).fd, newname, static_cast<unsigned int>(flags));
        }
        ::fuse_reply_err(req, (res == 0)? 0 : errno);
    }
    void Opendir(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi)
    {
        if (int dirfd = ::openat(Fs.getNode(ino).fd, ".", O_RDONLY); dirfd != -1)
        {
            if (DIR* dir = ::fdopendir(dirfd); dir != nullptr)
            {
                const fuse_file_info fi
                {
                    .cache_readdir = true,
                    .fh = reinterpret_cast<uint64_t>(dir)
                };
                ::fuse_reply_open(req, &fi);
                return;
            }
        }
        ::fuse_reply_err(req, errno);
    }
    void Readdir(fuse_req_t req, fuse_ino_t ino, size_t size, off_t off, struct fuse_file_info *fi)
    {
        DIR* dir = reinterpret_cast<DIR*>(fi->fh);
        Fs.Buffer.resize(size);
        ::seekdir(dir, off);
    
        errno = 0;
        size_t needed = 0;
        for (char *cur = Fs.Buffer.data(); needed < size; cur += needed)
        {
            size -= needed;
            auto entry = ::readdir(dir);
            if (entry == nullptr)
            {
                break;
            }
            else if (entry->d_name[0] == '.' && ((entry->d_name[1] == '.' && entry->d_name[2] == 0) || entry->d_name[1] == 0))
            {
                needed = 0;
                continue;
            }
            struct stat attr
            {
                .st_ino = entry->d_ino,
                .st_mode = entry->d_type
            };
            needed = ::fuse_add_direntry(req, cur, size, entry->d_name, &attr, entry->d_off);
        }

        if (errno != 0 && size == Fs.Buffer.size())
        {
            ::fuse_reply_err(req, errno);
        }
        else
        {
            ::fuse_reply_buf(req, Fs.Buffer.data(), Fs.Buffer.size() - size);
        }
    }
    void Readdirplus(fuse_req_t req, fuse_ino_t ino, size_t size, off_t off, struct fuse_file_info *fi)
    {
        DIR* dir = reinterpret_cast<DIR*>(fi->fh);
        Fs.Buffer.resize(size);
        ::seekdir(dir, off);

        struct fuse_entry_param out
        {
            .generation = 0,
            .attr_timeout = Fs.AttrTimeout,
            .entry_timeout = Fs.EntryTimeout
        };
        errno = 0;
        size_t needed = 0;
        for (char *cur = Fs.Buffer.data(); needed < size; cur += needed)
        {
            size -= needed;
            auto entry = ::readdir(dir);
            if (entry == nullptr)
            {
                break;
            }
            else if (entry->d_name[0] == '.' && ((entry->d_name[1] == '.' && entry->d_name[2] == 0) || entry->d_name[1] == 0))
            {
                needed = 0;
                continue;
            }
            Node *res = Fs.findChild(Fs.getNode(ino), entry->d_name, &out.attr);
            if (res == nullptr)
            {
                break;
            }
            out.ino = reinterpret_cast<fuse_ino_t>(res);
            needed = ::fuse_add_direntry_plus(req, cur, size, entry->d_name, &out, entry->d_off);
        }

        if (errno != 0 && size == Fs.Buffer.size())
        {
            ::fuse_reply_err(req, errno);
        }
        else
        {
            ::fuse_reply_buf(req, Fs.Buffer.data(), Fs.Buffer.size() - size);
        }
    }
    void Releasedir(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi)
    {
        ::fuse_reply_err(req, (::closedir(reinterpret_cast<DIR*>(fi->fh)) == 0) ? 0 : errno);
    }
    void Fsyncdir(fuse_req_t req, fuse_ino_t ino, int datasync, struct fuse_file_info *fi)
    {
        int dirfd = ::dirfd(reinterpret_cast<DIR*>(fi->fh));
        ::fuse_reply_err(req, (dirfd != -1 && (((datasync != 0) ? ::fdatasync(dirfd) : ::fsync(dirfd)) == 0)) ? 0 : errno);
    }
    void Create(fuse_req_t req, fuse_ino_t parent, const char *name, mode_t mode, struct fuse_file_info *fi)
    {
        auto ctx = ::fuse_req_ctx(req);
        int parentFd = Fs.getNode(parent).fd;
        Node *node = nullptr;
        int fd = -1;
        struct fuse_entry_param entry
        {
            .generation = 0,
            .attr_timeout = Fs.AttrTimeout,
            .entry_timeout = Fs.EntryTimeout
        };
        
        LogEntry log = LogEntry::GetOpen(ctx->pid, 0, fi->flags | O_CREAT | O_EXCL);
        
        {
            std::unique_lock lock(Fs.createMutex);
            if (::mknodat(parentFd, name, (mode & ~S_IFMT) | S_IFREG, 0) == 0)  /// @todo: not sure about O_EXCL, but doc reads like it.
            {
                if (node = HandleCreation(req, parentFd, name, O_RDWR, &entry.attr); node != nullptr) /// @todo: we could get ridof atleast one open call here
                {
                    log = LogEntry::GetOpen(ctx->pid, reinterpret_cast<uint64_t>(node), fi->flags | O_CREAT | O_EXCL);
                    if (fd = ::openat(parentFd, name, fi->flags & ~(O_CREAT | O_EXCL), 0); fd == -1) /// @todo: again, not sure about O_EXCL
                    {
                        int err = errno;
                        std::unique_lock lock(Fs.nodesMutex); /// @todo: we could introduce a forgetmutex so a non 0 forget doesnt block nodes access
                        /// @todo: this is not nice, because file is already visible -> file creation side effect. deletion should happen before HandleCreation releases nodesMutex.
                        if (--node->lookup == 0) // no one else has seen the file (yet)
                        {
                            ::unlinkat(parentFd, name, 0);
                            Fs.nodes.erase(node->ino);
                            node = nullptr;
                        }
                        errno = err;
                    }
                }
            }
        }
        int res = (fd == -1) ? -errno : fd;
        log.end(res);

        if (node != nullptr)
        {
            entry.ino = reinterpret_cast<fuse_ino_t>(node);
            fi->direct_io = Fs.DirectIo;
            fi->keep_cache = Fs.KeepCache;
            fi->fh = static_cast<uint64_t>(fd);
            ::fuse_reply_create(req, &entry, fi);
        }
        else
        {
            ::fuse_reply_err(req, -res);
        }

        auto data = log.getBuf();
        char fdname[12];
        snprintf(fdname, 12, "%d", parentFd);
        auto size = readlinkat(Fs.ProcFd, fdname, &data.data()[LogEntry::OffPath], LogEntry::SizePath);
        if (size != -1 && size < LogEntry::SizePath)
        {
            data.data()[LogEntry::OffPath + size++] = '/';
            ::memcpy(&data.data()[LogEntry::OffPath + size], name, std::min(strlen(name), size_t(LogEntry::SizePath) - size));
        }
        Fs.writeLog(data);
    }
    void Symlink(fuse_req_t req, const char *link, fuse_ino_t parent, const char *name)
    {
        int parentfd = Fs.getNode(parent).fd;
        Node *node = nullptr;
        struct stat attr{};
        {
            std::unique_lock lock(Fs.createMutex);
            if (::symlinkat(link, parentfd, name) == 0)
            {
                node = HandleCreation(req, parentfd, name, O_PATH | O_NOFOLLOW, &attr);
            }
        }
        if (node != nullptr)
        {
            ReplyEntry(req, *node, attr);
        }
        else
        {
            ::fuse_reply_err(req, errno);
        }
    }
    void Link(fuse_req_t req, fuse_ino_t ino, fuse_ino_t newparent, const char *newname)
    {
        int parentfd = Fs.getNode(newparent).fd;
        Node *node = &Fs.getNode(ino);
        struct stat attr{};
        {
            std::unique_lock lock(Fs.createMutex);
            if (::linkat(node->fd, "", parentfd, newname, AT_EMPTY_PATH) != 0)
            {
                node = nullptr;
            }
        }
        struct fuse_entry_param entry
        {
            .ino = reinterpret_cast<fuse_ino_t>(node),
            .generation = 0,
            .attr_timeout = Fs.AttrTimeout,
            .entry_timeout = Fs.EntryTimeout
        };
        if (node != nullptr && fstat(node->fd, &entry.attr) == 0)
        {
            node->lookup++;
            ::fuse_reply_entry(req, &entry);
        }
        else
        {
            ::fuse_reply_err(req, errno);
        }
    }
}
