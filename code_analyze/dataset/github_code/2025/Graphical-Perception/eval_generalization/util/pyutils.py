
import logging
import sys

def multiImport(modelName):
    names = modelName.split(".")
    className=names[-1]
    names.pop()
    s=""
    for name in names:
        if len(s)==0:
            s+=name
        else:
            s+="."+name
        logging.info("Try to import module %s"%s)
        __import__(s)
    return getattr(sys.modules[s],className)