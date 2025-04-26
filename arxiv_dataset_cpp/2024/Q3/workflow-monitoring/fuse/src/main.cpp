#include <FileSystem.hpp>

#include <fcntl.h>
#include <iostream>
#include <memory>
#include <sys/stat.h>

int main(int argc, char *argv[])
{
    fuse_args args = FUSE_ARGS_INIT(argc, argv);
    fuse_cmdline_opts opts;
    
    if (fuse_parse_cmdline(&args, &opts) != 0)
    {
        return -1;
    }
    if (opts.show_help || opts.mountpoint == nullptr)
    {
        std::cout << "Usage: " << argv[0] << "[options] <mountpoint>\n" << std::endl;
        fuse_cmdline_help();
        fuse_lowlevel_help();
        free(opts.mountpoint);
        fuse_opt_free_args(&args);
        return (opts.show_help) ? 0 : -1;
    }
    if (opts.show_version)
    {
        std::cout << "FUSE library version " << fuse_pkgversion() << std::endl;
        fuse_lowlevel_version();
        free(opts.mountpoint);
        fuse_opt_free_args(&args);
        return 0;
    }

    int res = LogFs::Fs.setupPollPipe();
    if (res == -1)
    {
        return errno;
    }
    struct stat buf{};
    int fd = ::open(opts.mountpoint, O_RDONLY /*| O_PATH*/);
    if (fd == -1 || ::fstat(fd, &buf) != 0)
    {
        std::cout << "Error opening mountpoint." << std::endl;
        return errno;
    }

    LogFs::Fs.root = std::make_unique<LogFs::Node>(fd, buf.st_ino);

    fuse_lowlevel_ops ops = LogFs::Fs.GetOps();
    fuse_session *session = ::fuse_session_new(&args, &ops, sizeof(ops), 0);
    if (session != nullptr && ::fuse_set_signal_handlers(session) == 0 && ::fuse_session_mount(session, opts.mountpoint) == 0 && ::fuse_daemonize(opts.foreground ? 1 : 0) == 0)
    {
        res = -1;
        {
            /* Block until ctrl+c or fusermount -u */
            if (opts.singlethread)
            {
                res = ::fuse_session_loop(session);
            }
            else
            {
                fuse_loop_config config
                {
                    .clone_fd = opts.clone_fd,
                    .max_idle_threads = 30//opts.max_idle_threads
                };
                res = ::fuse_session_loop_mt(session, &config);
            }
        }
    }

    return res;
}
