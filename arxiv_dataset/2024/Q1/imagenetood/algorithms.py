from Detection.Baseline import MSP, max_logit, energy, react, ash, entropy, max_cosine
from Detection.ODIN import odin, odin_batched
from Detection.gradnorm import gradnorm
from Detection.Mahalanobis import mahalanobis
from Detection.ViM import vim, vlogits
from functools import partial
from Detection.Knn import knn
# run [pip install libmr] to use openmax
#from Detection.openmax import openmax

logit_only = {
    "Maximum Softmax Probability": MSP,
    "Maximum Logits": max_logit,
    "Energy": energy,
    "Maximum Cosine": max_cosine,
    "entropy": entropy
}

gradient_based = {
    "ODIN_0.0014_1000": partial(odin, epsilon=0.0014, temperature=1000),
    "ODIN": odin,
    "ODIN_batched": odin_batched,
    "GradNorm": gradnorm,
}

training_based = {"Mahalanobis": mahalanobis, 
                  "ReAct": react, 
                  "ViM": vim, 
                  "KNN": knn,
                  #"openmax": openmax,
                  "ash": ash,
                  }
