import os

class FilePath:
    def __init__(self, filedir='cnn', randn_num='', suffix='.pt', separator='__'):
        '''
        obtain the path of a specific file
        
        :param filedir: the relative path of a file
        :param randn_num: a random number that owned by a file
        :param suffix: the suffix of the file
        :param separator: symbol for splitting the random number in the file name
        
        Note: the random number should be at the end of the name of the file !
        '''
        self.filedir = filedir
        self.randn_num = randn_num
        self.separator = separator
        self.file_suffix = suffix
    
    def filePath(self):
        listdir = os.listdir(self.filedir)
        for File in listdir:
            if File.endswith(self.file_suffix):
                fileName = os.path.splitext(File)[0]
                randn = fileName.split(self.separator)[-1]
                if randn == self.randn_num:
                    target_file = self.filedir + '/' + File
        return target_file