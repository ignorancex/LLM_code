
import os
import numpy as np

class Logger:

    def __init__(self, path, log_name, names, clean=False):

        self.path = os.path.join(path, log_name)

        if clean and os.path.exists(self.path):
            os.remove(self.path)


        self.names = names


    def writeHeader(self, name):

        with open(self.path, 'a') as fp:

            fp.write('\n***********************************************************\n')
            fp.write('***********************************************************\n')
            fp.write(name.upper() + '\n')


    def writeLog(self, data):

        with open(self.path, 'a') as fp:

            fp.write('\n--------------------------------------\n')

            for name, d in zip(self.names, data):

                fp.write(name + ': ' + d + '\n')

            fp.write('\n')


    def writeException(self, name, exception):

        with open(self.path, 'a') as fp:

            fp.write('\n--------------------------------------\n')

            fp.write('!!! ' + self.names[0] + ': ' + name + '\n')
            fp.write('!!! ERROR: ' + '*******\n')

            fp.write('!!! EXCEPTION: ')
            fp.write(str(exception))
            fp.write('\n\n')


    def writeEnd(self):

        with open(self.path, 'a') as fp:

            fp.write('***********************************************************\n')



def saveFPS(path, set, subset, fps, types='detector'):

    if types == 'detector':

        path = os.path.join('outputs/detections', path, set, subset + '-fps.txt')


    elif types == 'tracker':

        path = os.path.join(path, subset + '-fps.txt')


    with open(path, 'w') as fp:

        fp.write('%0.2f'%fps)


def str2bool(v):

    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')