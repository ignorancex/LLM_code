#include <FileSystem.hpp>

#include <sys/stat.h>
#include <unistd.h>

namespace LogFs
{
    void Readlink(fuse_req_t req, fuse_ino_t ino)
    {
        int fd = Fs.getNode(ino).fd;

        ssize_t res = Fs.Buffer.empty() ? 0 : ::readlinkat(fd, "", Fs.Buffer.data(), Fs.Buffer.size());
        if (res >= Fs.Buffer.size())
        {
            struct stat attr;
            if (::fstat(fd, &attr) == 0)
            {
                Fs.Buffer.resize(attr.st_size + 1);
            }
            while (res >= 0)
            {
                res = ::readlinkat(fd, "", Fs.Buffer.data(), Fs.Buffer.size());
                if (res < Fs.Buffer.size())
                {
                    break;
                }
                Fs.Buffer.resize(Fs.Buffer.size() * 2);
            }
        }
        if (res >= 0 && res < Fs.Buffer.size())
        {
            Fs.Buffer[res] = 0;
            ::fuse_reply_readlink(req, Fs.Buffer.data());
        }
        else
        {
            ::fuse_reply_err(req, errno);
        }
    }
}
