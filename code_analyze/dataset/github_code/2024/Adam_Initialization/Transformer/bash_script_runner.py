import subprocess
import os

def run_bash_script(data_path, model, problem, arch, output_path, num):
    try:
        # Ensure the output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Set the environment variables
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        # Define the bash command to run
        bash_command = [
            "python", "main.py", "--data", data_path,
            "--seed", "1",
            "--adam-eps", "1e-08",
            "--arch", arch, "--share-all-embeddings",
            "--optimizer", "adam", "--adam-betas", "(0.9, 0.98)", "--clip-norm", "0.0",
            "--dropout", "0.3", "--attention-dropout", "0.1", "--relu-dropout", "0.1",
            "--criterion", "label_smoothed_cross_entropy",
            "--lr-scheduler", "inverse_sqrt", "--warmup-init-lr", "1e-7", "--warmup-updates", "8000",
            "--lr", "0.0015", "--min-lr", "1e-9",
            "--label-smoothing", "0.1", "--weight-decay", "0.0001",
            "--max-tokens", "4096", "--save-dir", output_path,
            "--update-freq", "1", "--no-progress-bar", "--log-interval", "50",
            "--ddp-backend", "no_c10d",
            "--keep-last-epochs", str(num), "--max-epoch", "55",
            "--restore-file", f"{output_path}/checkpoint_best.pt"
        ]

        # Run the bash command and log output using tee
        with open(f"{output_path}/train_log.txt", "a") as log_file:
            process = subprocess.Popen(bash_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            for line in process.stdout:
                print(line.decode(), end="")  # print stdout to console
                log_file.write(line.decode())  # write stdout to log file

            process.wait()  # Wait for the process to complete

        print("Script finished with exit code:", process.returncode)
        return process.returncode
    except Exception as e:
        print(f"Error running bash script: {e}")
        return 1


if __name__ == "__main__":
    # Example usage
    DATA_PATH = "./data-bin/iwslt14.tokenized.de-en.joined"
    MODEL = "transformer"
    PROBLEM = "iwslt14_de_en"
    ARCH = "transformer_iwslt_de_en_v2"
    OUTPUT_PATH = "log/adam"
    NUM = 5

    # Run the bash script via the wrapper
    exit_code = run_bash_script(DATA_PATH, MODEL, PROBLEM, ARCH, OUTPUT_PATH, NUM)
#/home/abu/code/optimization/AdaM3/transformer/bash_script_runner.py