import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os, sys
sys.path.append("../")
from crossasr import CrossASR
import utils

if __name__ == "__main__":

    config = utils.readJson(sys.argv[1]) # read json configuration file

    tts = utils.getTTS(config["tts"])
    asrs = utils.getASRS(config["asrs"])
    estimator = utils.getEstimator(config["estimator"]) if 'estimator' in config else None
    mutator = utils.getMutator(config["mutator"]) if 'mutator' in config else None

    crossasr = CrossASR(tts=tts, asrs=asrs, estimator=estimator, mutator=mutator, **utils.parseConfig(config))

    corpus_fpath = os.path.join(config["output_dir"], config["corpus_fpath"])
    texts = utils.readCorpus(corpus_fpath=corpus_fpath)
    crossasr.processCorpus(texts=texts)
    crossasr.printStatistic()


    
