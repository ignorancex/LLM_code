#!/usr/bin/env python3

# command line program
import argparse
# csv management
import csv
# regular expression
import re
# internal modules
import libpost

def parse():
    parser = argparse.ArgumentParser(description='Extracts specific objects from data')
    parser.add_argument('objects', nargs='+', default=0, help='specify the objects to extract')
    return parser.parse_args()

def main(objects):
    # read file header
    header = libpost.get_file_header("objects.csv")
    indexs = []
    for object_name in objects + ["time"]:
        regex = "^{name}.*".format(name=object_name)
        indexs += [index for index, name in enumerate(header) if re.search(regex, name)]
    # extract objects
    with open('objects.csv', newline='') as fread:
        reader = csv.reader(fread, delimiter=',')
        next(reader, None) # skip header
        with open('objects_extracted.csv', 'w', newline='') as fwrite:
            writer = csv.writer(fwrite, delimiter=',')
            # write header
            header = [header[index] for index in indexs]
            header[0] = "#" + header[0]
            writer.writerow(header)
            # write the rest
            for row in reader:
                writer.writerow([row[index] for index in indexs])

if __name__ == '__main__':
    args = parse()
    main(args.objects)
