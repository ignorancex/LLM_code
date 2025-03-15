import sys
import pytest
import minvime.estimator_classification as esti # The file ../minvime/estimator_classification.py
import numpy as np

def test_estimate_model_requirements():
    """
       Test the ROI estimator for binary classification models.
    """

    auc, prec, recall, fprs, tprs  = esti.estimate_binary_model_requirements(
        tp=2000, fp=-100, tn=0, fn=0, cases=100000,
        baserate=0.001, minroi=10000
    )

    assert round(auc,3) == 0.723
    assert round(prec,4) == 0.064
    assert round(recall,4) == 0.2049


def test_estimate_intervention_requirements():
    tp, fp, tn, fn = esti.estimate_intervention_requirements(1000, 0.1, 10, 100, -100, 0.5, 0.01)
    assert round(tp,1) == 40.0
    assert round(fp,1) == -11.0


def test_auc():
    auc = esti.calculate_auc([0,0.1,0.5,1],[0,0.2,0.7,1])
    assert(auc) == 0.615

def test_simplicity_estimate():
    simp =  esti.simplicity_estimate(10,-2,30,0.33,200)
    assert(simp) == 0.000

    simp =  esti.simplicity_estimate(1,-1,10,0.5,1)
    assert(simp) == 0.4
