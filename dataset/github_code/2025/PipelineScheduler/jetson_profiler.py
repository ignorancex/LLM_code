import sys
from jtop import jtop
from time import sleep

def get_jetson_stats(pid):
    with jtop(interval=0.05) as jetson:
        while True:
            for p in jetson.processes:
                if p[0] == pid:
                    print(f"{p[0]}|{p[6]}|{p[7]}|{p[8]}|{list(jetson.gpu.values())[0]['status']['load']}|{jetson.memory['RAM'].get('used', 0)}", flush=True)
            sleep(0.05)

def get_runtime_stats(pid):
    with jtop(interval=0.1) as jetson:
        while True:
            for p in jetson.processes:
                if len(p) > 8:
                    if p[0] >= pid:
                        print(f"{p[0]}|{p[6]}|{p[7]}|{p[8]}|{list(jetson.gpu.values())[0]['status']['load']}|{jetson.memory['RAM'].get('used', 0)}", flush=True)
            sleep(0.1)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python jetson_profiler.py <mode> <pid>")
        sys.exit(1)
    pid = int(sys.argv[2])
    if sys.argv[1] == 'runtime':
        get_runtime_stats(pid)
    else:
        get_jetson_stats(pid)
