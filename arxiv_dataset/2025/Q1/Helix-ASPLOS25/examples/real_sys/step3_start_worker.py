# 2024.11.02 Yixuan Mei
import sys

from llm_sys.worker import run_worker


def main():
    # parse arguments
    if len(sys.argv) != 2:
        print("Usage: python step3_start_worker.py <scheduling_method>")
        print("  scheduling_method: maxflow | swarm | random")
        return
    scheduling_method = sys.argv[1]

    # check arguments
    assert scheduling_method in ["maxflow", "swarm", "random"], f"Unsupported scheduling method: {scheduling_method}!"
    print(f"Starting worker with scheduling method: {scheduling_method}.")

    # run worker
    run_worker(scheduling_method=scheduling_method, model_name="./model")


if __name__ == '__main__':
    main()
