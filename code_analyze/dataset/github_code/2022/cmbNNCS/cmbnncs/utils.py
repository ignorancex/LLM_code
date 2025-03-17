import os
import sys
import shutil
import numpy as np


def mkdir(path):
    '''
    Make a directory in a particular location if it is not exists, otherwise, do nothing.
        
    Usage: mkdir('/home/jian/test'), mkdir('test/one') or mkdir('../test/one') 
    '''
    #remove the blank space in the before and after strings
    #path.strip() is used to remove the characters in the beginning and the end of the character string
#    path = path.strip()
    #remove all blank space in the strings, there is no need to use path.strip() when using this command
    path = path.replace(' ', '')
    #path.rstrip() is used to remove the characters in the right of the characters strings
    
    if path=='':
        raise ValueError('The path cannot be an empty string')
    path = path.rstrip("/")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print('The directory "%s" is successfully created !'%path)
        return True
    else:
#        print('The directory "%s" is already exists!'%path)
#        return False
        pass

def rmdir(path):
    '''
    remove a folder in a particular location if it is exists, otherwise, do nothing 
    '''
    isExists = os.path.exists(path)
    if isExists:
        shutil.rmtree(path)
        print('The folder "%s" is successfully removed !'%path)

def savetxt(path, FileName, File):
    '''
    save the .txt files using np.savetxt() funtion
    
    :param path: the path of the file to be saved
    :param FileName: the name of the file to be saved
    :param File: the file to be saved
    '''
    if path:
        mkdir(path)
        np.savetxt(path + '/' + FileName + '.txt', File)
    else:
        np.savetxt(FileName + '.txt', File)

def savedat(path, FileName, File):
    '''
    save the .dat files using np.savetxt() funtion
    
    :param path: the path of the file to be saved
    :param FileName: the name of the file to be saved
    :param File: the file to be saved
    '''
    if path:
        mkdir(path)
        np.savetxt(path + '/' + FileName + '.dat', File)
    else:
        np.savetxt(FileName + '.dat', File)

def savenpy(path, FileName, File, dtype=np.float64):
    '''
    save an array to a binary file in NumPy .npy format using np.save() function
    
    :param path: the path of the file to be saved
    :param FileName: the name of the file to be saved
    :param File: the file to be saved
    '''
    if type(File) is np.ndarray:
        File = File.astype(dtype)
    if path:
        mkdir(path)
        np.save(path + '/' + FileName + '.npy', File)
    else:
        np.save(FileName + '.npy', File)

class Logger(object):
    def __init__(self, path='logs', fileName="log", stream=sys.stdout):
        self.terminal = stream
        self.path = path
        self.fileName = fileName
        self._log()
    
    def _log(self):
        if self.path:
            mkdir(self.path)
            self.log = open(self.path+'/'+self.fileName+'.log', "w")
        else:
            self.log = open(self.fileName+'.log', "w")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def logger(path='logs', fileName='log'):
    sys.stdout = Logger(path=path, fileName=fileName, stream=sys.stdout)
    sys.stderr = Logger(path=path, fileName=fileName, stream=sys.stderr) # redirect std err, if necessary

