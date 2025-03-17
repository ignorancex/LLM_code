from time import time_ns
from bcc import BPF
import argparse
import ctypes
import os
import re
import sys

bpf_text = ""
with open(
    os.path.dirname(os.path.realpath(__file__)) + "/testdir/cgroup_informer.cpp", "r"
) as f:
    bpf_text = f.read()
bpf_text = bpf_text[
    (
        bpf_text.find("//------BPF_START------") + len("//------BPF_START------")
    ) : bpf_text.rfind("//------BPF_END------")
]

parser = argparse.ArgumentParser(
    description="Trace process starts",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="",
)
parser.add_argument(
    "-O",
    "--output",
    type=str,
    help="file to direct output to (will be overwritten) (default: stdout)",
)
parser.add_argument(
    "-H", "--header", action="store_true", help="print table header at start"
)
parser.add_argument(
    "--evbufmain",
    type=int,
    help="page count for main event buffer (default: 8)",
    default=8,
)
parser.add_argument(
    "--perfbuf",
    action="store_true",
    help="use perf buf instead of ring buf output (might be needed on older kernels)",
)
parser.add_argument(
    "--printonly",
    action="store_true",
    help="only prints the final BPF program (after all options have been applied), does not execute it",
)
parser.add_argument(
    "--compileonly",
    action="store_true",
    help="only compiles the final BPF program (after all options have been applied), does not execute it",
)
args = parser.parse_args()

if args.output:
    sys.stdout = open(args.output, "w")

if args.perfbuf:
    bpf_text = re.sub(
        "BPF_RINGBUF_OUTPUT\(([^,]+)[^;]+", r"BPF_PERF_OUTPUT(\1)", bpf_text
    )
    bpf_text = re.sub(
        "ringbuf_output\(([^,]+),([^,]+),[^;]+", r"perf_submit(ctx, \1, \2)", bpf_text
    )

bpf_text = bpf_text.replace("RB_PAGES_EVENT_MAIN", str(args.evbufmain))

if args.printonly:
    print(bpf_text)
    exit(0)

# initialize BPF
b = BPF(text=bpf_text)

if args.compileonly:
    exit(0)

SizeTimeSec = 20  # maybe a '-' followed by up to 19 digits, a '.' and 3 digits
SizeTimeNsec = 3  # 3 digits
SizeTime = (
    SizeTimeSec + 1 + SizeTimeNsec
)  # maybe a '-' followed by up to 19 digits, a '.' and 3 digits
SizePid = 11  # maybe a '-' followed by up to 10 digits
SizeCgroupId = 20

rtdelta = None


def handle_main(cpu, data, size):
    global rtdelta
    event = b["event_main"].event(data)

    if rtdelta is None:
        rtdelta = time_ns() - event.time

    time = event.time + rtdelta
    print(
        "%*ld.%0*ld,%*d,%*d,%*d"
        % (
            SizeTimeSec,
            time // 1000,
            SizeTimeNsec,
            time % 1000,
            SizePid,
            event.ppid,
            SizePid,
            event.pid,
            SizeCgroupId,
            event.cgroupid,
        )
    )


# attaching event callbacks
if args.perfbuf:
    b["event_main"].open_perf_buffer(handle_main, page_cnt=args.evbufmain)
else:
    b["event_main"].open_ring_buffer(handle_main)

if args.header:
    print("time, ppid, pid, cgroupid")

while True:
    try:
        b.perf_buffer_poll(100) if args.perfbuf else b.ring_buffer_poll(100)
    except KeyboardInterrupt:
        exit()
