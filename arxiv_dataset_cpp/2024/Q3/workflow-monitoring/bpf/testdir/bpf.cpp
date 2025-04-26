// don't include in string to compile with BCC:
#include <helpers.h> // BPF stuff
#include <linux/err.h>

#define EDITOR_ONLY // stuff only defined for editing this file and having autocomplete, not relevant for the actual BPF program

#define PATH_DEPTH 10
#define FILENAME_BUFSIZE 64

#define FILTER_BY_PID 1
#define INCLUDE_CHILD_PROCESSES 1
#define FILTER_BY_UID 1
#define FILTER_BY_GID 1
#define FILTER_BY_PATH 1
#define PATHFILTER_PATHPART_LENGTH 16 // number of bytes / characters allocated per path part (determined by longest path part)
#define PATHFILTER_MAX_PARTS_PER_PATH 16 // maximum number of parts per path; excluding "/" parts as they are ignored; including a "\0" as last part
#define PATHFILTER_PATHES 3

#define LOG_DELETES 1

#define RB_PAGES_EVENT_MAIN 8
#define RB_PAGES_EVENT_PATH 8

// compile with BCC the code below:
//------BPF_START------
#include <linux/version.h>

#define randomized_struct_fields_start  struct {
#define randomized_struct_fields_end    };

#include <asm/ptrace.h> // pt_regs
#include <linux/fs.h> // file
#include <linux/path.h> // path
#include <linux/mount.h> // mount
#include <linux/uio.h> // iov_iter

#include <linux/sched.h>

// from fs/internal.h
struct open_flags {
	int open_flag;
	umode_t mode;
	int acc_mode;
	int intent;
	int lookup_flags;
};

#if FILTER_BY_PID
// Set of the PIDs to log.
BPF_HASH(log_pids, pid_t, u8);
#endif

#if FILTER_BY_UID
BPF_HASH(log_uids, u32, u8);
#endif

#if FILTER_BY_GID
BPF_HASH(log_gids, u32, u8);
#endif

#if FILTER_BY_PATH
struct path_part_t
{
    char str[PATHFILTER_PATHPART_LENGTH];
};

BPF_ARRAY(log_paths, struct path_part_t, PATHFILTER_MAX_PARTS_PER_PATH * PATHFILTER_PATHES);
#endif

// Since file* cannot be used directly (as it may be reused often), we have to generate a unique id for each open / opened handle.

BPF_ARRAY(global_handle_uid, u64, 1);
BPF_HASH(opened, struct file*, u64);

// Same has to be done for file inodes. Further we keep track of open handle count for the inodes. Both in closing a handle and unlinking the inode, handle count and if the inode is fully unlinked is checked and the inode is removed from the map if both are true.
struct saved_inode_t
{
    u64 inode_uid;
    u32 opened_handles;
};

BPF_ARRAY(global_inode_uid, u64, 1);
BPF_HASH(inodes, unsigned long, struct saved_inode_t);

// This struct is transmitted multiple times per open for each segment of the path (from back to front).
struct event_transmit_path_t
{
    u64 uid;
    u8 event_type;
    u8 path_type;
    u8 final; // last path for the event
    u8 last; // last path fragment for the event
    char filename[FILENAME_BUFSIZE];
};

BPF_RINGBUF_OUTPUT(event_transmit_path, RB_PAGES_EVENT_PATH);

enum event_type_t : u8
{
    TYPE_OPEN = 'O',
    TYPE_CLOSE = 'C',
    TYPE_READ = 'R',
    TYPE_WRITE = 'W',
    TYPE_DELETE = 'U'
};

// This struct is submitted for every event.
struct event_main_t
{
    u64 time_start;
    u64 time_end;
    u32 pid;
    u64 utime_start;
    u64 utime_end;
    u64 stime_start;
    u64 stime_end;
    u64 inode_uid;
    u8 type;
    s64 result;
    u64 handle_uid;
    u64 offset;
    u64 size;
    u64 flags;
};

BPF_RINGBUF_OUTPUT(event_main, RB_PAGES_EVENT_MAIN);

static void PrepareMainEvent(struct event_main_t* event)
{
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();

    event->time_start = bpf_ktime_get_ns();
    event->pid = bpf_get_current_pid_tgid() >> 32;
    event->utime_start = task->utime;
    event->stime_start = task->stime;
}

static void SetEventFileStuff(struct event_main_t* event, struct file* file)
{
    int zero = 0;
    unsigned long inode = file->f_inode->i_ino;
    struct saved_inode_t* saved_inode = inodes.lookup(&inode);

    if (saved_inode == NULL)
    {
        struct saved_inode_t new_saved_inode = {};
        u64* p_global_inode_uid = global_inode_uid.lookup(&zero);
        if (p_global_inode_uid != NULL) // will not fail, BPF wants the check though
            event->inode_uid = new_saved_inode.inode_uid = ++(*p_global_inode_uid);
        new_saved_inode.opened_handles = 1; // new inode also means new handle
        inodes.insert(&inode, &new_saved_inode);
    }
    else
        event->inode_uid = saved_inode->inode_uid;

    u64* saved_handle = opened.lookup(&file);
    if (saved_handle == NULL)
    { // new handle, generate a new uid, check inode
        u64 new_saved_handle;

        u64* p_global_handle_uid = global_handle_uid.lookup(&zero);
        if (p_global_handle_uid != NULL) // will not fail, BPF wants the check though
            event->handle_uid = new_saved_handle = ++(*p_global_handle_uid);

        opened.insert(&file, &new_saved_handle);

        if (saved_inode != NULL)
            saved_inode->opened_handles++;
    }
    else // handle known, just take saved information
        event->handle_uid = *saved_handle;

    event->offset = file->f_pos;
    event->flags = file->f_flags;
}

static void FinalizeAndSubmitMainEvent(struct pt_regs* ctx, struct event_main_t* event)
{
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();

    event->time_end = bpf_ktime_get_ns();
    event->utime_end = task->utime;
    event->stime_end = task->stime;

    event_main.ringbuf_output(event, sizeof(*event), 0);
}

static void OutputDentry(struct pt_regs* ctx, struct event_transmit_path_t* path_data, struct dentry* entry, bool finalize)
{
    path_data->last = 0;

    #pragma unroll (PATH_DEPTH)
    for (int i = 0; i < PATH_DEPTH; i++)
    {
        if (!entry) break;

        int copied_size = bpf_probe_read_kernel_str(path_data->filename, sizeof(path_data->filename), entry->d_name.name);

        if (copied_size > 0)
        {
            event_transmit_path.ringbuf_output(path_data, sizeof(*path_data), 0);
            entry = entry->d_parent;
        }
        else
        {
            entry = NULL;
        }
    }

    if (finalize)
    {
        path_data->last = 1; // communicate end
        event_transmit_path.ringbuf_output(path_data, sizeof(*path_data), 0);
    }
}

static void OutputFilePaths(struct pt_regs* ctx, struct event_transmit_path_t* path_data, struct file* fp)
{
    path_data->final = 0;
    path_data->path_type = 'F';
    OutputDentry(ctx, path_data, fp->f_path.dentry, true);

    path_data->path_type = 'M';
    OutputDentry(ctx, path_data, fp->f_path.mnt->mnt_root, true);

    path_data->final = 1;
    path_data->path_type = 'S';
    OutputDentry(ctx, path_data, fp->f_path.mnt->mnt_sb->s_root, true); // not sure what the functional difference is between this line and the one above
}

static bool CheckToLog()
{
#if FILTER_BY_PID
    pid_t pid = (bpf_get_current_pid_tgid() >> 32);

    if (log_pids.lookup(&pid) == NULL)
        return false;
#endif

#if FILTER_BY_UID
    u32 uid = (bpf_get_current_uid_gid() & 0xFFFFFFFF);

    if (log_uids.lookup(&uid) == NULL)
        return false;
#endif

#if FILTER_BY_GID
    u32 gid = (bpf_get_current_uid_gid() >> 32);

    if (log_gids.lookup(&gid) == NULL)
        return false;
#endif

    return true;
}

struct save_t
{
    union
    {
        const struct path* path; // open
        struct
        { // close
            unsigned long inode; // if -1, handle is not actually freed as there are other references to it
            bool unlinked_inode;
        };
    };

    struct file *fp;
    struct event_main_t event;
};

BPF_HASH(save, u64, struct save_t);

// internal function handling most logic for open kprobes to reduce redundancy
static void do_open(bool insert, struct file* fp)
{
    if (!CheckToLog())
        return;

    u64 id = bpf_get_current_pid_tgid();
    struct save_t saved = {};
    saved.fp = fp;
    PrepareMainEvent(&saved.event);
    saved.event.type = TYPE_OPEN;
    if (insert)
        save.insert(&id, &saved);
    else
        save.update(&id, &saved);
}

// internal function handling most logic for open kretprobes to reduce redundancy
// if fp is NULL, the FP saved in open is used
static void do_ret_open(struct pt_regs *ctx, s64 result, struct file* fp)
{
    u64 id = bpf_get_current_pid_tgid();
    struct save_t* saved = save.lookup(&id);

    if (saved != NULL)
    {
        if (fp != NULL)
            saved->fp = fp;

        if (saved->fp && !IS_ERR_VALUE(saved->fp)) // no error opening the file / valid file struct pointer
            SetEventFileStuff(&saved->event, saved->fp); // will not be executed on error -> therefor won't save handle
        else // error; todo: maybe save flags / filename from original call
            saved->fp = NULL;

        struct event_transmit_path_t path_data = {};
        path_data.event_type = TYPE_OPEN;
        path_data.uid = saved->event.handle_uid;

        saved->event.result = result;
        OutputFilePaths(ctx, &path_data, saved->fp);
        FinalizeAndSubmitMainEvent(ctx, &saved->event);
        save.delete(&id);
    }
}

// used for do_filp_open, do_file_open_root
// we need those to catch the cases that circumvent vfs_open
// however, a pointer to the file structure is returned instead of being passed as an argument
int open_without_file(struct pt_regs *ctx, int dfd, struct filename *pathname, const struct open_flags *op)
{
    do_open(true, NULL);
    return 0;
}

int ret_open_returning_file(struct pt_regs *ctx)
{
    struct file* fp = (struct file*)PT_REGS_RC(ctx);
    do_ret_open(ctx, IS_ERR_VALUE(fp) ? (s64)fp : 0, fp);
    return 0;
}

// used for vfs_open and the filesystem specific open functions
int open_with_file(struct pt_regs *ctx, void *unused, struct file *file)
{
    do_open(false, file);
    return 0;
    // we use update here since there might already be an entry if vfs_open is called through do_filp_open or do_file_open_root
    // then we basically only log vfs_open and drop do_filp_open / do_file_open_root as that is not needed anymore
    // because only one open log is needed and the other two functions are attached to catch any atomic_open case which on success skips vfs_open
}

int ret_open_without_file(struct pt_regs *ctx)
{
    do_ret_open(ctx, PT_REGS_RC(ctx), NULL);
    return 0;
}

int kprobe__filp_close(struct pt_regs *ctx, struct file *file)
{
    if (!CheckToLog())
        return 0;

    u64* saved_handle = opened.lookup(&file);

    if (saved_handle != NULL)
    {
        u64 id = bpf_get_current_pid_tgid();
        struct save_t saved = {};
        if (file->f_count.counter <= 1)
        {
            saved.inode = file->f_inode->i_ino;
            saved.unlinked_inode = (file->f_inode->i_nlink == 0);
        }
        else
            saved.inode = (unsigned long)-1;
        saved.fp = file;
        PrepareMainEvent(&saved.event);
        SetEventFileStuff(&saved.event, file);
        saved.event.type = TYPE_CLOSE;
        save.insert(&id, &saved);
    }

    return 0;
}

int kretprobe__filp_close(struct pt_regs *ctx)
{
    u64 id = bpf_get_current_pid_tgid();
    struct save_t* saved = save.lookup(&id);

    if (saved != NULL)
    {
        saved->event.result = PT_REGS_RC(ctx);

        if (saved->event.result == 0 && saved->inode != (unsigned long)-1)
        {
            opened.delete(&saved->fp);
            struct saved_inode_t* saved_inode = inodes.lookup(&saved->inode);

            if (saved_inode != NULL) // should always be the case
            {
                if (--saved_inode->opened_handles == 0 && saved->unlinked_inode)
                    inodes.delete(&saved->inode);
            }
        }

        FinalizeAndSubmitMainEvent(ctx, &saved->event);
        save.delete(&id);
    }

    return 0;
}

BPF_HASH(save_delete, u64, unsigned long);

#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 12, 0)
int kprobe__vfs_unlink(struct pt_regs *ctx, struct inode *dir, struct dentry *dentry, struct inode **delegated_inode)
#else
int kprobe__vfs_unlink(struct pt_regs *ctx, struct user_namespace *mnt_userns, struct inode *dir, struct dentry *dentry, struct inode **delegated_inode)
#endif
{
    u64 id = bpf_get_current_pid_tgid();
    unsigned long inode = dentry->d_inode->i_ino;
    struct saved_inode_t* saved_inode = NULL;

    if (dentry->d_inode->i_nlink <= 1)
    { // unlink potentially deletes file
        saved_inode = inodes.lookup(&inode);

        if (saved_inode != NULL && saved_inode->opened_handles == 0) // also it's one of "our" inodes that have no handles anymore, so we have to delete it, if unlink is successful
            save_delete.insert(&id, &inode);
    }

#if LOG_DELETES
    if (CheckToLog())
    {
        if (saved_inode == NULL)
            saved_inode = inodes.lookup(&inode);

        struct event_transmit_path_t path_data = {};
        path_data.event_type = TYPE_DELETE;
        path_data.last = 0;
        struct save_t saved = {};
        PrepareMainEvent(&saved.event);
        saved.event.type = TYPE_DELETE;

        if (saved_inode != NULL)
            path_data.uid = saved.event.inode_uid = saved_inode->inode_uid;
        else
            path_data.uid = saved.event.inode_uid = (u64)-inode;

        path_data.final = 0;
        path_data.path_type = 'F';
        OutputDentry(ctx, &path_data, dentry, true);
        path_data.final = 1;
        path_data.path_type = 'S';
        OutputDentry(ctx, &path_data, dentry->d_sb->s_root, true);

        save.insert(&id, &saved);
    }
#endif

    return 0;
}

int kretprobe__vfs_unlink(struct pt_regs *ctx)
{
    u64 id = bpf_get_current_pid_tgid();
    unsigned long* saved_inode = save_delete.lookup(&id);

    if (saved_inode != NULL)
    {
        if (PT_REGS_RC(ctx) == 0) // unlink succeeded
            inodes.delete(saved_inode);

        save_delete.delete(&id);
    }

#if LOG_DELETES
    struct save_t* saved = save.lookup(&id);

    if (saved != NULL)
    {
        saved->event.result = PT_REGS_RC(ctx);
        FinalizeAndSubmitMainEvent(ctx, &saved->event);
        save.delete(&id);
    }
#endif

    return 0;
}

static int do_readwrite(struct file* file, size_t size, loff_t* pos, u8 type)
{
    if (!CheckToLog())
        return 0;

    u64* saved_handle = opened.lookup(&file);

    if (saved_handle != NULL)
    {
        struct save_t saved = {};
        PrepareMainEvent(&saved.event);
        SetEventFileStuff(&saved.event, file);
        saved.event.type = type;
        saved.event.size = size;
        saved.event.offset = *pos;

        u64 id = bpf_get_current_pid_tgid();
        save.insert(&id, &saved);
    }

    return 0;
}

int probe__vfs_write(struct pt_regs *ctx, struct file *file, const char __user *buf, size_t count, loff_t *pos)
{
    return do_readwrite(file, count, pos, TYPE_WRITE);
}

int probe__vfs_read(struct pt_regs *ctx, struct file *file, char __user *buf, size_t count, loff_t *pos)
{
    return do_readwrite(file, count, pos, TYPE_READ);
}

int probe__do_iter_write(struct pt_regs *ctx, struct file *file, struct iov_iter *iter, loff_t *pos, rwf_t flags)
{ // covers vfs_iter_write, vfs_writev
    return do_readwrite(file, iter->count, pos, TYPE_WRITE);
}

int probe__do_iter_read(struct pt_regs *ctx, struct file *file, struct iov_iter *iter, loff_t *pos, rwf_t flags)
{ // covers vfs_iter_read, vfs_readv
    return do_readwrite(file, iter->count, pos, TYPE_READ);
}

int probe__vfs_iocb_iter_write(struct pt_regs *ctx, struct file *file, struct kiocb *iocb, struct iov_iter *iter)
{
    return do_readwrite(file, iter->count, &iocb->ki_pos, TYPE_WRITE);
}

int probe__vfs_iocb_iter_read(struct pt_regs *ctx, struct file *file, struct kiocb *iocb, struct iov_iter *iter)
{
    return do_readwrite(file, iter->count, &iocb->ki_pos, TYPE_READ);
}

int retprobe__readwrites(struct pt_regs *ctx)
{
    u64 id = bpf_get_current_pid_tgid();
    struct save_t* saved = save.lookup(&id);

    if (saved != NULL)
    {
        saved->event.result = PT_REGS_RC(ctx);
        FinalizeAndSubmitMainEvent(ctx, &saved->event);
        save.delete(&id);
    }

    return 0;
}

#if FILTER_BY_PID
TRACEPOINT_PROBE(sched, sched_process_exit)
{
    // args is defined as seen /sys/kernel/debug/tracing/events/sched/sched_process_exit/format
    #ifdef EDITOR_ONLY
    struct
    {
        u16 common_type;
        u8 common_flags;
        u8 common_preempt_count;
        s32 common_pid;

        char comm[16];
        pid_t pid;
        int prio;
    } *real_args;
    #define args real_args
    #endif

    pid_t pid = args->pid;
    log_pids.delete(&pid);

    #ifdef EDITOR_ONLY
    #undef args
    #endif
    return 0;
}

#if INCLUDE_CHILD_PROCESSES
TRACEPOINT_PROBE(sched, sched_process_fork)
{
    // args is defined as seen /sys/kernel/debug/tracing/events/sched/sched_process_fork/format
    #ifdef EDITOR_ONLY
    struct
    {
        u16 common_type;
        u8 common_flags;
        u8 common_preempt_count;
        s32 common_pid;

        char parent_comm[16];
        pid_t parent_pid;
        char child_comm[16];
        pid_t child_pid;
    } *real_args;
    #define args real_args
    #endif

    pid_t pid = args->parent_pid;

    if (log_pids.lookup(&pid) != NULL)
    {
        u8 one = 1;
        pid = args->child_pid;
        log_pids.insert(&pid, &one);
    }

    #ifdef EDITOR_ONLY
    #undef args
    #endif
    return 0;
}
#endif
#endif

//------BPF_END------
