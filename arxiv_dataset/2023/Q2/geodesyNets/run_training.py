import signal
import sys

import tomli

from gravann import training_runner

is_stop = False


def sigterm_handler(sig, frame):
    global is_stop
    is_stop = True
    print("Process requested to stop. Will stop after finishing the current training run!")


if sys.platform != "win32":
    signal.signal(signal.Signals.SIGTERM, sigterm_handler)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise IOError("No input provided!")
    with open(sys.argv[1], "rb") as config_file:
        cfg = tomli.load(config_file)
    print("INPUT LOADED")
    cuda_devices = sys.argv[2] if len(sys.argv) > 2 else cfg["cuda_devices"]
    training_runner.run(cfg, lambda: is_stop, cuda_devices)
