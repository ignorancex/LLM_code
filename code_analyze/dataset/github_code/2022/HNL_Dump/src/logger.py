class Logger(object):
    _instance = None
    DEBUG = True

    def __new__(cls, debug=True):
        if cls._instance is None:
            print('Creating Logger Object')
            cls._instance = super(Logger, cls).__new__(cls)
            # Put any initialization here.
            Logger.DEBUG = debug
        return cls._instance

    def log(self, string, important=False):
        if Logger.DEBUG or important:
            print(string)