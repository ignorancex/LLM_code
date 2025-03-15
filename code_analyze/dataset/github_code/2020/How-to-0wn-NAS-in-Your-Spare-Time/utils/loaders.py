"""
    A module to load/store data from/to csv/pickle files.
"""
import csv
import pickle


# ------------------------------------------------------------------------------
#   CSV Reader/Writer
# ------------------------------------------------------------------------------
def load_from_csv(filename):
    csv_data = None
    with open(filename, 'r') as infile:
        csv_reader = csv.reader(infile)
        csv_data = [ \
            each for each in csv_reader]
    return csv_data

def store_to_csv(filename, data):
    with open(filename, 'w') as outfile:
        csv_writer = csv.writer(outfile)
        for each in data:
            csv_writer.writerow(each)
    # done.


# ------------------------------------------------------------------------------
#   Pickle Reader/Writer
# ------------------------------------------------------------------------------
def load_from_pickle(filename):
    pkl_data = []
    with open(filename, "rb") as infile:
        pkl_data = pickle.load(infile)
    return pkl_data

def store_to_pickle(filename, data):
    with open(filename, "wb") as outfile:
        pickle.dump(data, outfile)
    # done.
