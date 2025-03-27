# 2024.11.02 Yixuan Mei
import sys

from llm_sys.worker import run_worker


def main():
    # parse arguments
    if len(sys.argv) < 3:
        print("Usage: python step6_start_worker.py <llama30b/llama70b> <scheduling_method> <max_vram_usage>")
        print("  scheduling_method: maxflow | swarm | random")
        return
    model_name = sys.argv[1]
    scheduling_method = sys.argv[2]
    if len(sys.argv) == 4:
        max_vram_usage = float(sys.argv[3])
    else:
        max_vram_usage = 0.8

    # check arguments
    assert scheduling_method in ["maxflow", "swarm", "random"], f"Unsupported scheduling method: {scheduling_method}!"
    assert model_name in ["llama30b", "llama70b"], f"Invalid model name: {model_name}"
    print(f"Starting worker with scheduling method: {scheduling_method}.")
    print(f"Model: {model_name}")

    # run worker
    model_path = f"./models/{model_name}"
    run_worker(scheduling_method=scheduling_method, model_name=model_path, vram_usage=max_vram_usage)


if __name__ == '__main__':
    main()
