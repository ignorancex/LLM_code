# 2024.11.02 Yixuan Mei
import sys

from llm_sys.worker import run_worker


def main():
    run_worker(
        scheduling_method="maxflow",
        model_name="./models",
    )


if __name__ == '__main__':
    main()
