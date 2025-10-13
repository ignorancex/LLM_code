import argparse
import subprocess


def main(args):
    if args.language == "EN":
        subprocess.run(["python", "content_from_header_EN.py", args.content_dir])
        subprocess.run(["python", "process_content_raw_EN.py", "--content_dir", args.content_dir])
        subprocess.run(["python", "build_test_annt_EN.py", "--content_dir", args.content_dir])
    elif args.language == "TC":
        subprocess.run(["python", "content_from_header_TC.py", args.content_dir])
        subprocess.run(["python", "process_content_raw_ZH.py", "--content_dir", args.content_dir, "--language", "TC"])
        subprocess.run(["python", "build_test_annt_ZH.py", "--content_dir", args.content_dir])
    elif args.language == "SC":
        subprocess.run(["python", "content_from_header_SC.py", args.content_dir])
        subprocess.run(["python", "process_content_raw_ZH.py", "--content_dir", args.content_dir, "--language", "SC"])
        subprocess.run(["python", "build_test_annt_ZH.py", "--content_dir", args.content_dir])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", type=str, default="example")
    parser.add_argument("--language", type=str, default="EN", choices=["EN", "TC", "SC"])
    args = parser.parse_args()

    main(args)