// don't include in string to compile with BCC:
#include <helpers.h> // BPF stuff
#include <linux/err.h>

#define EDITOR_ONLY // stuff only defined for editing this file and having autocomplete, not relevant for the actual BPF program

#define RB_PAGES_EVENT_MAIN 8

// compile with BCC the code below:
//------BPF_START------
#include <linux/version.h>

#define randomized_struct_fields_start  struct {
#define randomized_struct_fields_end    };

#include <asm/ptrace.h> // pt_regs
#include <linux/sched.h>
#include <linux/cgroup.h>

struct event_main_t
{
    u64 time;
    u32 ppid;
    u32 pid;
    u64 cgroupid;
};

BPF_RINGBUF_OUTPUT(event_main, RB_PAGES_EVENT_MAIN);

// use internal function 'wake_up_new_task' instead of the fork tracepoint / syscall as here we get the fully prepared task struct of the child (which the fork tracepoint does not provide)
// source: https://research.nccgroup.com/2021/08/06/some-musings-on-common-ebpf-linux-tracing-bugs/
/*
int kprobe__wake_up_new_task(struct pt_regs *ctx, struct task_struct* child)
{
    struct event_main_t event;
    event.time = bpf_ktime_get_ns();
    event.ppid = bpf_get_current_pid_tgid() >> 32;
    event.pid = child->pid;
    //event.cgroupid = task_dfl_cgroup(child)->kn->id; // this is what bpf_get_current_cgroup_id does (on the current task)
    event.cgroupid = bpf_get_current_cgroup_id();
    event_main.ringbuf_output(&event, sizeof(event), 0);
}
*/

// alternative:
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
    #define ctx args // args can be used as ctx for perfbuf mode

    struct event_main_t event;
    event.time = bpf_ktime_get_ns();
    event.ppid = args->parent_pid;
    event.pid = args->child_pid;
    //event.cgroupid = task_dfl_cgroup(child)->kn->id; // this is what bpf_get_current_cgroup_id does (on the current task)
    event.cgroupid = bpf_get_current_cgroup_id();
    event_main.ringbuf_output(&event, sizeof(event), 0);

    #undef ctx
    #ifdef EDITOR_ONLY
    #undef args
    #endif
    return 0;
}
