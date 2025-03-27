import argparse


def init_args():
    parser = argparse.ArgumentParser(
        prog='Data to point cloud',
        description='Takes a space-separated-value file, drops the first row and the first element of every row, '
                    'and saves it.'
    )
    parser.add_argument('inputfile', type=str,
                        help='The input file')
    parser.add_argument('outputfile', type=str,
                        help='The output file')

    return parser.parse_args()


def load_vectors(file_name):
    with open(file_name, 'r') as file:
        n, d = map(int, file.readline().split())
        rows = []
        for line in file:
            split_row = line.rstrip().split(' ')
            rows.append(' '.join(split_row[1:]))
    return rows


def save_rows(file_name, rows):
    with open(file_name, 'w') as file:
        for row in rows:
            file.write(row)
            file.write('\n')


def main():
    args = init_args()
    rows = load_vectors(args.inputfile)
    save_rows(args.outputfile, rows)


if __name__ == '__main__':
    main()
