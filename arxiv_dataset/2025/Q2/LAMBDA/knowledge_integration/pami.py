from .knw import knw


class pattern_mining(knw):
    def __init__(self):
        super().__init__()
        self.name = "pami"
        self.description = "The pami library is a Python library for pattern mining. User can choose many algorithms like FP-growth algorithm. For Transactional_T10I4D100K.csv, the separator should be '\\t'"
        self.core_function = "pami"
        self.mode = 'full'

    def pami(self):
        return """
        !pip install PAMI
        from PAMI.frequentPattern.basic import FPGrowth as alg
        fileURL = "https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/Transactional_T10I4D100K.csv"
        minSup = 300
        obj = alg.FPGrowth(iFile=fileURL, minSup=minSup, sep='\\t') #Note: You should strictly use separator '\\t'
        # obj.startMine()  #deprecated
        obj.mine()
        obj.save('frequentPatternsAtMinSupCount300.txt')
        frequentPatternsDF = obj.getPatternsAsDataFrame()
        print('Total No of patterns: ' + str(len(frequentPatternsDF)))  # print the total number of patterns
        print('Runtime: ' + str(obj.getRuntime()))  # measure the runtime
        print('Memory (RSS): ' + str(obj.getMemoryRSS()))
        print('Memory (USS): ' + str(obj.getMemoryUSS()))
        """


if __name__ == '__main__':
    pami = pattern_mining()
    print(pami.get_core_function())
