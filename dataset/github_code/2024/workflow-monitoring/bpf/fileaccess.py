from time import time_ns
from bcc import BPF
import argparse
import ctypes
import os
import re
import sys

# read bpf file and extract the relevant code
bpf_text = ''
with open(os.path.dirname(os.path.realpath(__file__)) + '/testdir/bpf.cpp', 'r') as f:
    bpf_text = f.read()
bpf_text = bpf_text[(bpf_text.find('//------BPF_START------')+len('//------BPF_START------')):bpf_text.rfind('//------BPF_END------')]

parser = argparse.ArgumentParser(description="Trace file accesses", formatter_class=argparse.RawDescriptionHelpFormatter, epilog='')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--nofilter', action='store_true', help='apply no filter (warning: may lead to high output and degradation of overall performance; may want to use --time as well then)')
group.add_argument('-i', '--id', type=str, help='filter by comma separated list of PIDs')
group.add_argument('-p', '--program', type=str, help='commandline to call program that is going to be logged')
group.add_argument('-u', '--user', type=str, help='filter by comma separated list of UIDs or user names')
group.add_argument('-g', '--group', type=str, help='filter by comma separated list of GIDs or group names')

parser.add_argument('-P', '--path', type=str, help='filter by comma separated list of paths (provided by script, not BPF)')
parser.add_argument('-o', '--operations', type=str, help='specifies which operations to log (default: OCRWU)', default='OCRWU')
parser.add_argument('-c', '--includechildren', action='store_true', help='also include child processes')
parser.add_argument('-O', '--output', type=str, help='file to direct output to (will be overwritten) (default: stdout)')
#parser.add_argument('-v', '--verbose', action='store_true', help='verbose / human readable output')
parser.add_argument('-t', '--time', type=int, help='number of seconds to run the trace, 0 for unlimited (default: 0)', default=0)
parser.add_argument('-f', '--filesystems', type=str, help='comma seperated list of filesystems to filter by')

parser.add_argument('--depth', type=int, help='depth of loop unrolling, equals supported path depth (default: 10)', default=10)
parser.add_argument('--bufsize', type=int, help='size of the path buffer, maximum supported size per path part (default: 64)', default=64)
parser.add_argument('--delimiter', type=str, help='delimiter of output fields (default comma)', default=',')
parser.add_argument('--pathoutput', type=int, help='number of characters for path output (default 240)', default=240)
parser.add_argument('--evbufmain', type=int, help='page count for main event buffer (default: 8)', default=8)
parser.add_argument('--evbufpath', type=int, help='page count for path event buffer (default: 8)', default=8)
parser.add_argument('--perfbuf', action='store_true', help='use perf buf instead of ring buf output (might be needed on older kernels)')
parser.add_argument('--usermodepathfilter', action='store_true', help='filter by path (-P/--path) in usermode (Python) instead of BPF code (might be needed for older kernels)')

parser.add_argument('--printonly', action='store_true', help='only prints the final BPF program (after all options have been applied), does not execute it')
parser.add_argument('--compileonly', action='store_true', help='only compiles the final BPF program (after all options have been applied), does not execute it')

args = parser.parse_args()

args.operations = args.operations.upper()

if args.output:
    sys.stdout = open(args.output, 'w')

# apply settings to BPF code
pid_filter = (args.program is not None or args.id is not None)
bpf_text = bpf_text.replace('FILTER_BY_PID', '1' if pid_filter else '0')
bpf_text = bpf_text.replace('FILTER_BY_UID', '1' if args.user is not None else '0')
bpf_text = bpf_text.replace('FILTER_BY_GID', '1' if args.group is not None else '0')

bpf_text = bpf_text.replace('FILTER_BY_PATH', '1' if args.path is not None and not args.usermodepathfilter else '0')

bpf_text = bpf_text.replace('INCLUDE_CHILD_PROCESSES', '1' if args.includechildren else '0')
bpf_text = bpf_text.replace('LOG_DELETES', '1' if args.operations.find('U') >= 0 else '0')

if args.perfbuf:
    bpf_text = re.sub('BPF_RINGBUF_OUTPUT\(([^,]+)[^;]+', r'BPF_PERF_OUTPUT(\1)', bpf_text)
    bpf_text = re.sub('ringbuf_output\(([^,]+),([^,]+),[^;]+', r'perf_submit(ctx, \1, \2)', bpf_text)

bpf_text = bpf_text.replace('PATH_DEPTH', str(args.depth))
bpf_text = bpf_text.replace('FILENAME_BUFSIZE', str(args.bufsize))
bpf_text = bpf_text.replace('RB_PAGES_EVENT_MAIN', str(args.evbufmain))
bpf_text = bpf_text.replace('RB_PAGES_EVENT_PATH', str(args.evbufpath))

if args.printonly:
    print(bpf_text)
    exit(0)

# initialize BPF
b = BPF(text=bpf_text)

if args.compileonly:
    exit(0)

if args.filesystems is not None:
    fss = args.filesystems.split(',')
    for fs in fss:
        print('Attaching to fs ' + fs + '...', file=sys.stderr)
        try:
            b.attach_kprobe(event=fs + '_open', fn_name='open_with_file')
            print(fs + '_open attached', file=sys.stderr)
        except Exception as error:
            pass

        try:
            b.attach_kprobe(event=fs + '_file_open', fn_name='open_with_file')
            print(fs + '_file_open attached', file=sys.stderr)
        except Exception as error:
            pass
else:
    print('Attaching to all filesystems.', file=sys.stderr)
    b.attach_kprobe(event='vfs_open', fn_name='open_with_file')
    b.attach_kprobe(event='do_filp_open', fn_name='open_without_file')
    b.attach_kprobe(event='do_file_open_root', fn_name='open_without_file')

# generally attach to the generic functions returns as after the internal (fs specific) opens the file struct might not be fully populated
# also there is no drawback using the generic functions here since the internal ones pass through those anyway
b.attach_kretprobe(event='vfs_open', fn_name='ret_open_without_file')
b.attach_kretprobe(event='do_filp_open', fn_name='ret_open_returning_file')
b.attach_kretprobe(event='do_file_open_root', fn_name='ret_open_returning_file')

if args.operations.find('R') >= 0:
    b.attach_kretprobe(event='vfs_read', fn_name='retprobe__readwrites')
    b.attach_kprobe(event='vfs_read', fn_name='probe__vfs_read')
    b.attach_kretprobe(event='do_iter_read', fn_name='retprobe__readwrites')
    b.attach_kprobe(event='do_iter_read', fn_name='probe__do_iter_read')
    try: # on older kernels those functions don't exist and attaching will raise an exception; ignoring it is fine
        b.attach_kretprobe(event='vfs_iocb_iter_read', fn_name='retprobe__readwrites')
        b.attach_kprobe(event='vfs_iocb_iter_read', fn_name='probe__vfs_iocb_iter_read')
    except:
        pass

if args.operations.find('W') >= 0:
    b.attach_kretprobe(event='vfs_write', fn_name='retprobe__readwrites')
    b.attach_kprobe(event='vfs_write', fn_name='probe__vfs_write')
    b.attach_kretprobe(event='do_iter_write', fn_name='retprobe__readwrites')
    b.attach_kprobe(event='do_iter_write', fn_name='probe__do_iter_write')
    try: # on older kernels those functions don't exist and attaching will raise an exception; ignoring it is fine
        b.attach_kretprobe(event='vfs_iocb_iter_write', fn_name='retprobe__readwrites')
        b.attach_kprobe(event='vfs_iocb_iter_write', fn_name='probe__vfs_iocb_iter_write')
    except:
        pass

if args.id is not None:
    pids = [int(pid) for pid in args.id.split(',')]
    try:
        b['log_pids'].items_update_batch((ctypes.c_uint32 * len(pids))(*pids), (ctypes.c_uint32 * len(pids))(*([1]*len(pids))))
    except: # items_update_batch only supported since kernel 5.6, else update element by element
        for p in pids:
            b['log_pids'][ctypes.c_uint32(p)] = ctypes.c_uint32(1)

if args.program is not None:
    import signal
    from time import sleep
    pid = os.fork()

    if pid == 0:
        import shlex
        subprog = shlex.split(args.program)
        os.kill(os.getpid(), signal.SIGSTOP)
        os.execv(subprog[0], subprog)
        exit(1)

    b['log_pids'][ctypes.c_uint32(pid)] = ctypes.c_uint32(1)
    sleep(0.5)
    os.kill(pid, signal.SIGCONT)

    # following shit was tried but behaves weirdly and does not work:
    #proc = subprocess.Popen(shlex.split(args.program), preexec_fn=lambda: os.kill(os.getpid(), signal.SIGSTOP))

    #proc = subprocess.Popen(['bash', '-'], stdin=subprocess.PIPE, start_new_session=True)
    #proc.stdin.write(bytes("set -m\nsuspend\nexec " + args.program, 'utf-8'))
    #proc.stdin.flush()
    #try:
    #    proc.wait(0.5)
    #except subprocess.TimeoutExpired:
    #    pass
    #print(proc.pid)
    #b['log_pids'].items_update_batch((ctypes.c_uint32 * 1)(proc.pid), (ctypes.c_uint32 * 1)(1))
    #proc.send_signal(signal.SIGCONT)

if args.user is not None:
    from pwd import getpwnam
    uids = [(int(uid) if uid.isnumeric() else getpwnam(uid).pw_uid) for uid in args.user.split(',')]
    try:
        b['log_uids'].items_update_batch((ctypes.c_uint32 * len(uids))(*uids), (ctypes.c_uint32 * len(uids))(*([1]*len(uids))))
    except: # items_update_batch only supported since kernel 5.6, else update element by element
        for u in uids:
            b['log_uids'][ctypes.c_uint32(u)] = ctypes.c_uint32(1)

if args.group is not None:
    from grp import getgrnam
    gids = [(int(gid) if gid.isnumeric() else getgrnam(gid).gr_gid) for gid in args.group.split(',')]
    try:
        b['log_gids'].items_update_batch((ctypes.c_uint32 * len(gids))(*gids), (ctypes.c_uint32 * len(gids))(*([1]*len(gids))))
    except: # items_update_batch only supported since kernel 5.6, else update element by element
        for g in gids:
            b['log_gids'][ctypes.c_uint32(g)] = ctypes.c_uint32(1)

if args.path is not None:
    args.path = args.path.split(',')

# output configuration
SizeTimeSec    =  20                                # maybe a '-' followed by up to 19 digits, a '.' and 3 digits
SizeTimeNsec   =   3                                # 3 digits
SizeTime       =  SizeTimeSec + 1 + SizeTimeNsec    # maybe a '-' followed by up to 19 digits, a '.' and 3 digits
SizePid        =  11                                # maybe a '-' followed by up to 10 digits
SizeInode      =  20                                # up to 20 digits
SizeEvent      =   1                                # 1 byte
SizeResult     =  11                                # maybe a '-' followed by up to 10 digits
SizeFilehandle =  20                                # maybe a '-' followed by up to 19 digits
SizeOffset     =  20                                # maybe a '-' followed by up to 19 digits
SizeSize       =  20                                # up to 20 digits
SizeFlags      =  10                                # "0x" followed by 8 digits
SizePath       = args.pathoutput                    # default 240 characters
OffRTimeStart  =   0
OffRTimeEnd    = OffRTimeStart + SizeTime       + 1
OffPid         = OffRTimeEnd   + SizeTime       + 1
OffUTimeStart  = OffPid        + SizePid        + 1
OffUTimeEnd    = OffUTimeStart + SizeTime       + 1
OffSTimeStart  = OffUTimeEnd   + SizeTime       + 1
OffSTimeEnd    = OffSTimeStart + SizeTime       + 1
OffInode       = OffSTimeEnd   + SizeTime       + 1
OffEvent       = OffInode      + SizeInode      + 1
OffResult      = OffEvent      + SizeEvent      + 1
OffFilehandle  = OffResult     + SizeResult     + 1
OffOffset      = OffFilehandle + SizeFilehandle + 1
OffSize        = OffOffset     + SizeOffset     + 1
OffFlags       = OffSize       + SizeSize       + 1
OffPath        = OffFlags      + SizeFlags      + 1
SizeEntry      = OffPath       + SizePath       + 1

# dictionary of handle_uid to path to concat the different path parts coming from the transmit_path callback
open_pathes = dict()

# dictionary of handle_uid to array of events that are saved here until the path for the handle_uid is fully transmitted (only then we can correctly output the open event)
saved_events = dict()

# dictionary of inode to path to concat the different path parts coming from the transmit_path callback, same as the two dicts above
delete_pathes = dict()
saved_delete_event = dict()

# set of handles to filter for (according to -p / --path if used)
allowed_handles = set()

rtdelta = None

# output function
def output_event(event):
    global rtdelta

    if rtdelta is None:
        rtdelta = time_ns() - event.time_end

    path = ''

    time_start = event.time_start + rtdelta
    time_end = event.time_end + rtdelta

    unlink_ok = False

    if chr(event.type) == 'O':
        if event.handle_uid in open_pathes:
            path = open_pathes[event.handle_uid].pathes['F'] + ':' + open_pathes[event.handle_uid].pathes['M'] + ':' + open_pathes[event.handle_uid].pathes['S']
            del open_pathes[event.handle_uid]
        if args.usermodepathfilter and args.path is not None:
            for p in args.path:
                if path.startswith(p):
                    allowed_handles.add(event.handle_uid)
                    break
    elif chr(event.type) == 'U':
        path = delete_pathes[event.inode_uid].pathes['F'] + ':' + delete_pathes[event.inode_uid].pathes['M'] + ':' + delete_pathes[event.inode_uid].pathes['S']
        del delete_pathes[event.inode_uid]
        if args.usermodepathfilter and args.path is not None:
            for p in args.path:
                if path.startswith(p):
                    unlink_ok = True
                    break

    if len(path) > SizePath:
        path = path[:SizePath]

    if args.operations.find(chr(event.type)) >= 0 and (not args.usermodepathfilter or (args.path is None or event.handle_uid in allowed_handles or unlink_ok)):
        #print(time_start, time_end, event.pid, event.utime_start, event.utime_end, event.stime_start, event.stime_end, event.inode_uid, chr(event.type), event.result, event.handle_uid, event.offset, event.size, event.flags, path, sep=args.delimiter)
        time_start = time_start // 1000000
        time_end = time_end // 1000000
        utime_start = event.utime_start // 1000000
        utime_end = event.utime_end // 1000000
        stime_start = event.stime_start // 1000000
        stime_end = event.stime_end // 1000000
        print("%*ld.%0*ld,%*ld.%0*ld,%*d,%*ld.%0*ld,%*ld.%0*ld,%*ld.%0*ld,%*lu.%0*ld,%*lu,%c,%*d,%*ld,%*lu,%*lu,0x%0*x,%-*s" % (
            SizeTimeSec, time_start // 1000, SizeTimeNsec, time_start % 1000,
            SizeTimeSec, time_end // 1000,   SizeTimeNsec, time_end % 1000,
            SizePid, event.pid,
            SizeTimeSec, utime_start // 1000, SizeTimeNsec, utime_start % 1000,
            SizeTimeSec, utime_end // 1000,   SizeTimeNsec, utime_end % 1000,
            SizeTimeSec, stime_start // 1000, SizeTimeNsec, stime_start % 1000,
            SizeTimeSec, stime_end // 1000,   SizeTimeNsec, stime_end % 1000,
            SizeInode, event.inode_uid,
            event.type,
            SizeResult, event.result,
            SizeFilehandle, event.handle_uid,
            SizeOffset, event.offset,
            SizeSize, event.size,
            SizeFlags - 2, event.flags,
            SizePath, path))

    if chr(event.type) == 'C' and args.path:
        allowed_handles.discard(event.handle_uid)

class PathData:
    def __init__(self) -> None:
        self.pathes = {'F': '', 'M': '', 'S': ''}
        self.ready = 0

# BPF event callbacks
def handle_path(cpu, data, size):
    event = b['event_transmit_path'].event(data)
    pathes = open_pathes if chr(event.event_type) == 'O' else delete_pathes
    saved = saved_events if chr(event.event_type) == 'O' else saved_delete_event
    #print('transmit_path:', event.filename, event.uid, event.final, event.last, chr(event.event_type), chr(event.path_type), file=sys.stderr)

    if event.last == 0:
        if event.filename != b'/':
            if event.uid not in pathes:
                pathes[event.uid] = PathData()

            pathes[event.uid].pathes[chr(event.path_type)] = '/' + event.filename.decode('utf-8') + pathes[event.uid].pathes[chr(event.path_type)]
    else:
        if event.uid not in pathes:
            pathes[event.uid] = PathData()
        pathes[event.uid].ready |= event.final
        if event.final == 1 and event.uid in saved:
            #print('transmit_path:', event.filename, event.uid, event.last, chr(event.event_type), event.uid in pathes)#, file=sys.stderr)
            for e in saved[event.uid]:
                output_event(e)
            del saved[event.uid]

def handle_main(cpu, data, size):
    event = b['event_main'].event(data)
    #print('transmit_main:', chr(event.type), event.handle_uid, file=sys.stderr)

    if chr(event.type) == 'U' and (not event.inode_uid in delete_pathes or delete_pathes[event.inode_uid].ready == 0):
        saved_delete_event[event.inode_uid] = [event]
    elif chr(event.type) == 'O' and (not event.handle_uid in open_pathes or open_pathes[event.handle_uid].ready == 0):
        saved_events[event.handle_uid] = [event]
    elif event.handle_uid in saved_events:
        saved_events[event.handle_uid].append(event)
    else:
        output_event(event)

# attaching event callbacks
if args.perfbuf:
    b['event_transmit_path'].open_perf_buffer(handle_path, page_cnt=args.evbufpath)
    b['event_main'].open_perf_buffer(handle_main, page_cnt=args.evbufmain)
else:
    b['event_transmit_path'].open_ring_buffer(handle_path)
    b['event_main'].open_ring_buffer(handle_main)

running = True

def exitProg():
    global running
    running = False

if args.time > 0:
    import threading
    threading.Timer(args.time, exitProg).start()

print('time_start,time_end,pid,utime_start,utime_end,stime_start,stime_end,inode,type,result,handle,offset,size,flags,path')

while running:
    try:
        b.perf_buffer_poll(100) if args.perfbuf else b.ring_buffer_poll(100)
    except KeyboardInterrupt:
        exit()
