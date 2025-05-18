import os
import csv
from collections import Counter

def scan_directory_for_py_files(directory):
    py_files = []
    for (root, _, files) in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    return py_files

def extract_base_name(files):
    return [os.path.splitext(os.path.basename(file))[0] for file in files]

def count_word_frequency(base_names):
    return Counter(base_names)

def count_comments(files):
    total_comments = 0
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                total_comments += sum((1 for line in f if line.strip().startswith('#')))
        except Exception as e:
    return total_comments

def write_word_frequency_to_csv(word_counter, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File Name', 'Frequency'])
        for (word, count) in word_counter.most_common():
            writer.writerow([word, count])

def main(directory, output_csv):
    py_files = scan_directory_for_py_files(directory)
    if not py_files:
        return
    base_names = extract_base_name(py_files)
    word_counter = count_word_frequency(base_names)
    total_comments = count_comments(py_files)
    write_word_frequency_to_csv(word_counter, output_csv)
if __name__ == '__main__':
    directory = './github_code/2024/'
    output_csv = './output_2024/file_name_frequency.csv'
    main(directory, output_csv)