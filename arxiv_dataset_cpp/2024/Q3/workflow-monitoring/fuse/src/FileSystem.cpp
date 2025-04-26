#include <FileSystem.hpp>

#include <shared_mutex>
#include <unistd.h>
#include <thread>

namespace LogFs
{
    void Init           (void *userdata, struct fuse_conn_info *conn);
    void Destroy        (void *userdata);
    void Lookup         (fuse_req_t req, fuse_ino_t parent, const char *name);
    void Forget         (fuse_req_t req, fuse_ino_t ino, uint64_t nlookup);
    void Getattr        (fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi);
    void Setattr        (fuse_req_t req, fuse_ino_t ino, struct stat *attr, int to_set, struct fuse_file_info *fi);
    void Readlink       (fuse_req_t req, fuse_ino_t ino);
    void Mknod          (fuse_req_t req, fuse_ino_t parent, const char *name, mode_t mode, dev_t rdev);
    void Mkdir          (fuse_req_t req, fuse_ino_t parent, const char *name, mode_t mode);
    void Unlink         (fuse_req_t req, fuse_ino_t parent, const char *name);
    void Rmdir          (fuse_req_t req, fuse_ino_t parent, const char *name);
    void Symlink        (fuse_req_t req, const char *link, fuse_ino_t parent, const char *name);
    void Rename         (fuse_req_t req, fuse_ino_t parent, const char *name, fuse_ino_t newparent, const char *newname, unsigned int flags);
    void Link           (fuse_req_t req, fuse_ino_t ino, fuse_ino_t newparent, const char *newname);
    void Open           (fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi);
    void Read           (fuse_req_t req, fuse_ino_t ino, size_t size, off_t off, struct fuse_file_info *fi);
    void Write          (fuse_req_t req, fuse_ino_t ino, const char *buf, size_t size, off_t off, struct fuse_file_info *fi);
    //void Flush        (fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi);
    void Release        (fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi);
    void Fsync          (fuse_req_t req, fuse_ino_t ino, int datasync, struct fuse_file_info *fi);
    void Opendir        (fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi);
    void Readdir        (fuse_req_t req, fuse_ino_t ino, size_t size, off_t off, struct fuse_file_info *fi);
    void Releasedir     (fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi);
    void Fsyncdir       (fuse_req_t req, fuse_ino_t ino, int datasync, struct fuse_file_info *fi);
    void Statfs         (fuse_req_t req, fuse_ino_t ino);
    void Setxattr       (fuse_req_t req, fuse_ino_t ino, const char *name, const char *value, size_t size, int flags);
    void Getxattr       (fuse_req_t req, fuse_ino_t ino, const char *name, size_t size);
    void Listxattr      (fuse_req_t req, fuse_ino_t ino, size_t size);
    void Removexattr    (fuse_req_t req, fuse_ino_t ino, const char *name);
    //void Access       (fuse_req_t req, fuse_ino_t ino, int mask);
    void Create         (fuse_req_t req, fuse_ino_t parent, const char *name, mode_t mode, struct fuse_file_info *fi);
    //void Getlk        (fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi, struct flock *lock);
    //void Setlk        (fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi, struct flock *lock, int sleep);
    //void Bmap         (fuse_req_t req, fuse_ino_t ino, size_t blocksize, uint64_t idx);
    //void Ioctl          (fuse_req_t req, fuse_ino_t ino, unsigned int cmd, void *arg, struct fuse_file_info *fi, unsigned flags, const void *in_buf, size_t in_bufsz, size_t out_bufsz);
    void Poll           (fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi, struct fuse_pollhandle *ph);
    void WriteBuf       (fuse_req_t req, fuse_ino_t ino, struct fuse_bufvec *bufv, off_t off, struct fuse_file_info *fi);
    //void RetrieveReply(fuse_req_t req, void *cookie, fuse_ino_t ino, off_t offset, struct fuse_bufvec *bufv);
    void ForgetMulti    (fuse_req_t req, size_t count, struct fuse_forget_data *forgets);
    //void Flock        (fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi, int op);
    void Fallocate      (fuse_req_t req, fuse_ino_t ino, int mode, off_t offset, off_t length, struct fuse_file_info *fi);
    void Readdirplus    (fuse_req_t req, fuse_ino_t ino, size_t size, off_t off, struct fuse_file_info *fi);
    void CopyFileRange  (fuse_req_t req, fuse_ino_t ino_in, off_t off_in, struct fuse_file_info *fi_in, fuse_ino_t ino_out, off_t off_out, struct fuse_file_info *fi_out, size_t len, int flags);
    void Lseek          (fuse_req_t req, fuse_ino_t ino, off_t off, int whence, struct fuse_file_info *fi);

    const fuse_lowlevel_ops Ops
    {
        .init               = Init,
        .destroy            = Destroy,
        .lookup             = Lookup,
        .forget             = Forget,
        .getattr            = Getattr,
        .setattr            = Setattr,
        .readlink           = Readlink,
        .mknod              = Mknod,
        .mkdir              = Mkdir,
        .unlink             = Unlink,
        .rmdir              = Rmdir,
        .symlink            = Symlink,
        .rename             = Rename,
        .link               = Link,
        .open               = Open,
        .read               = Read,
        .write              = Write,
        .flush              = nullptr, // Only needed if every close of dup()'ed fds is needed or locks are implemented
        .release            = Release,
        .fsync              = Fsync,
        .opendir            = Opendir,
        .readdir            = Readdir,
        .releasedir         = Releasedir,
        .fsyncdir           = Fsyncdir,
        .statfs             = Statfs,
        .setxattr           = Setxattr,
        .getxattr           = Getxattr,
        .listxattr          = Listxattr,
        .removexattr        = Removexattr,
        .access             = nullptr, // Only if not default_permissions
        .create             = Create,
        .getlk              = nullptr, // Only if lock fcntl is supported
        .setlk              = nullptr, // Only if lock fcntl is supported
        .bmap               = nullptr, // Only for Block file systems
        .ioctl              = nullptr, // Not needed for now
        .poll               = Poll,
        .write_buf          = WriteBuf,
        .retrieve_reply     = nullptr, // Only if cache is retrieved by fs
        .forget_multi       = ForgetMulti,
        .flock              = nullptr, // Only if flock is supported
        .fallocate          = Fallocate,
        .readdirplus        = Readdirplus,
        .copy_file_range    = CopyFileRange,
        .lseek              = Lseek,
    };

    fuse_lowlevel_ops FileSystem::GetOps()
    {
        return Ops;
    }
    Node &FileSystem::getNode(fuse_ino_t ino) const
    {
        return (ino != FUSE_ROOT_ID) ? *reinterpret_cast<Node*>(ino) : *root;
    }
    Node *FileSystem::findChild(Node &parent, const char *name, struct stat *attr)
    {
        Node *res = nullptr;
        if (::fstatat(parent.fd, name, attr, AT_SYMLINK_NOFOLLOW) != -1)
        {
            {
                std::shared_lock creationLock(createMutex); // if this was just created, wait till it is openeed and inserted to nodes
                std::shared_lock sharedLock(nodesMutex);
                if (auto it = nodes.find(attr->st_ino); it != nodes.end())
                {
                    res = &it->second;
                    res->lookup++;
                    return res;
                }
            }
            {
                if (int fd = ::openat(parent.fd, name, O_PATH | O_NOFOLLOW); fd != -1)
                {
                    bool dropFd = false;
                    {
                        std::unique_lock uniqueLock(nodesMutex);
                        auto [it, inserted] = nodes.emplace(std::piecewise_construct, std::forward_as_tuple(attr->st_ino), std::forward_as_tuple(fd, attr->st_ino));
                        res = &it->second;
                        res->lookup++;
                        dropFd = !inserted;
                        if (inserted && S_ISREG(attr->st_mode))
                        {
                            LogEntry::InformNewNode(reinterpret_cast<uint64_t>(res), false); // must happen here so that inode is safely inserted before it could be used
                        }
                    }
                    if (dropFd) // needed since we can not upgrade a shared mutex to a unique one. The element could already be inserted.
                    {
                        ::close(fd);
                    }
                }
            }
        }
        return res;
    }
    Node::~Node()
    {
        ::close(fd);
    }
    int FileSystem::setupPollPipe()
    {
        if (pollpipeFd == -1)
        {
            int pipeFds[2];
            if (::pipe(pipeFds) != 0)
            {
                return -1;
            }
            std::thread([](int pipefd)
            {
                std::vector<fuse_pollhandle*> phs;
                std::vector<pollfd> fds;
                fds.push_back(
                    {
                        .fd = pipefd,
                        .events = POLLIN,
                        .revents = 0
                    });
                while(true)
                {
                    int res = ::poll(fds.data(), fds.size(), -1);
                    if (res == -1)
                    {
                        if (errno != EINTR && fds.size() > 1)
                        {
                            ::close(fds.back().fd);
                            ::fuse_lowlevel_notify_poll(phs.back());
                            ::fuse_pollhandle_destroy(phs.back());
                            phs.pop_back();
                            fds.pop_back();
                        }
                    }
                    else if (res != 0)
                    {
                        if (fds.front().revents & POLLIN)
                        {
                            PollMessage pm;
                            size_t bytesRead = 0;
                            while (bytesRead < sizeof(pm))
                            {
                                size_t res = ::read(pipefd, &pm, sizeof(pm) - bytesRead);
                                if (res <= 0)
                                {
                                    // ERROR
                                    for (auto i : phs)
                                    {
                                        ::fuse_pollhandle_destroy(i);
                                    }
                                    for (auto i : fds)
                                    {
                                        ::close(i.fd);
                                    }
                                    return;
                                }
                                bytesRead += res;
                            }
                            if (pm.ph == nullptr)
                            {
                                if (pm.pfd.fd != -1)
                                {
                                    for (auto i : phs)
                                    {
                                        ::fuse_lowlevel_notify_poll(i);
                                    }
                                }
                                for (auto i : phs)
                                {
                                    ::fuse_pollhandle_destroy(i);
                                }
                                for (auto i : fds)
                                {
                                    ::close(i.fd);
                                }
                                return;
                            }
                            phs.push_back(pm.ph);
                            fds.push_back(pm.pfd);
                            res--;
                        }
                        for (size_t cur = 0; cur < phs.size() && res != 0;)
                        {
                            if (fds[cur + 1].revents)
                            {
                                ::fuse_lowlevel_notify_poll(phs[cur]);
                                ::fuse_pollhandle_destroy(phs[cur]);
                                ::close(fds[cur + 1].fd);
                                std::swap(phs[cur], phs.back());
                                std::swap(fds[cur + 1], fds.back());
                                phs.resize(phs.size() - 1);
                                fds.resize(phs.size() - 1);
                                res--;
                            }
                            else
                            {
                                cur++;
                            }
                        }
                    }
                }
            }, pipeFds[0]).detach();
            pollpipeFd = pipeFds[1];
        }
        return pollpipeFd;
    }
    int FileSystem::killPollThread(bool notifyHandles)
    {
        return poll(PollMessage
        {
            .ph = nullptr,
            .pfd = { .fd = (notifyHandles ? 0 : -1) }
        });
    }
    int FileSystem::poll(const PollMessage &ph) const
    {
        size_t readBytes = 0;
        while (readBytes < sizeof(PollMessage))
        {
            size_t res = ::write(pollpipeFd, &ph, sizeof(PollMessage) - readBytes);
            if (res == -1)
            {
                return -1;
            }
            readBytes -= res;
        }
        return 0;
    }
    int FileSystem::writeLog(std::span<char> logData)
    {
        int tries = 0;
        size_t written = 0;
        while (written != logData.size())
        {
            tries++;
            int res = ::write(Fs.logFd, logData.data() + written, logData.size() - written);
            if (res == -1)
            {
                break;
            }
            written += res;
        }
        int res = (written == logData.size()) ? 0 : -errno;
        if (written != logData.size())
        {
            constexpr const char ErrMessage[] = "Writing log output failed.";
            ::write(STDERR_FILENO, ErrMessage, sizeof(ErrMessage));
        }
        else if (tries > 1)
        {
            constexpr const char ErrMessage[] = "Writing a single log record needed multiple attempts, log might be corrupted.";
            ::write(STDERR_FILENO, ErrMessage, sizeof(ErrMessage));
        }
        return res;
    }

    int LogFs::FileSystem::ProcFd = -1;
    thread_local std::vector<char> LogFs::FileSystem::Buffer;
}
