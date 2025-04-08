from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, recall_score, confusion_matrix,
    classification_report,
    r2_score, mean_squared_error
)


@dataclass
class ClassificationScores:
    precision: float
    recall: float
    f1: float
    support: float = None

    @classmethod
    def from_dict(cls, d:Dict[str, float]) -> "ClassificationScores":
        d = {k.split('-')[0]: v for k, v in d.items() if k.split('-')[0] in cls.__annotations__}
        return cls(**d)
    
@dataclass
class RegressionScores:
    r2: float
    mse: float
    rmse: float

    @classmethod
    def make(cls, true:np.ndarray, pred:np.ndarray) -> "RegressionScores":
        return cls(
            r2=r2_score(true, pred),
            mse=mean_squared_error(true, pred),
            rmse=mean_squared_error(true, pred, squared=False)
        )

    def __add__(self, other):
        return RegressionScores(
            r2=self.r2 + other.r2,
            mse=self.mse + other.mse,
            rmse=self.rmse + other.rmse
        )
    
    def __truediv__(self, other):
        return RegressionScores(
            r2=self.r2 / other,
            mse=self.mse / other,
            rmse=self.rmse / other
        )
        

@dataclass
class ClassificationReport:
    accuracy: float
    confusion: np.ndarray
    macro: ClassificationScores
    weighted: ClassificationScores
    labels: list
    label_scores: Dict[str, ClassificationScores] # label -> ClassificationScores

    f1: float = None # only for binary classification
    recall: float = None # only for binary classification

    @classmethod
    def make_report(cls, true:np.ndarray, pred:np.ndarray) -> "ClassificationReport":
        class_labels = np.unique(np.concatenate([true, pred]))
        report = classification_report(true, pred, labels=class_labels, output_dict=True, zero_division=0)
        rep = cls(
            accuracy=report.pop('accuracy'),
            confusion=confusion_matrix(true, pred, labels=class_labels),
            macro=ClassificationScores.from_dict(report.pop('macro avg')),
            weighted=ClassificationScores.from_dict(report.pop('weighted avg')),
            label_scores={label: ClassificationScores.from_dict(scores) for label, scores in report.items()},
            labels=list(class_labels)
        )
        if len(class_labels) == 2:
            rep.f1 = f1_score(true, pred)
            rep.recall = recall_score(true, pred)
        return rep

    @property
    def df(self):
        df_dict = {
            'Accuracy': self.accuracy,
            **{f'{score.title()} (macro)': getattr(self.macro, score) for score in self.macro.__annotations__ if score != 'support'},
        }
        df = pd.DataFrame([df_dict])
        return df

        

    
@dataclass
class RegressionReport:
    r2: float
    rmse: float
    labels: list = None # only for multivariate regression
    label_scores: Dict[str, float] = None # only for multivariate regression

    @classmethod
    def make_report(cls, true:np.ndarray, pred:np.ndarray, labels=None) -> "RegressionReport":
        report = cls(
            r2=r2_score(true, pred),
            rmse=mean_squared_error(true, pred, squared=False)
        )
        if len(true.shape) > 1 and true.shape[1] > 1:
            report.labels = labels or list(range(true.shape[1]))
            report.label_scores = {label: RegressionScores.make(true[:,i], pred[:,i]) for i,label in enumerate(report.labels)}
        return report
    
    @property
    def is_multivariate(self):
        return self.labels is not None
    
    @property
    def df(self):
        df_dict = {
            'R2 avg': self.r2,
            'RMSE avg': self.rmse,
        }
        if self.is_multivariate:
            df_dict.update({f'R2 {label}': scores.r2 for label, scores in self.label_scores.items()})
            df_dict.update({f'RMSE {label}': scores.rmse for label, scores in self.label_scores.items()})
        df = pd.DataFrame([df_dict])
        rmse_cols = ['RMSE avg'] 
        df = df.filter(items=['RMSE avg', 'Pearson avg'] + sorted(df.columns.difference(['Pearson avg', 'RMSE avg'])))
        df.columns = df.columns.str.replace('\s(a|b|c|d)_', ' ', regex=True)
        return df
    

@dataclass
class Results:
    taska: ClassificationReport
    taskb: RegressionReport
    taskc: ClassificationReport
    taskd: RegressionReport


def absolute_results(true_df:pd.DataFrame, pred_df:pd.DataFrame, tasks='abcd'):
    task_reports = {}
    for task in tasks:
        true=true_df.filter(regex=f'^{task}_').sort_index(axis=1)
        pred=pred_df.filter(regex=f'^{task}_').sort_index(axis=1)
        if len(true.columns) == 0 or len(pred.columns) == 0:
            task_reports['task'+task] = None
            continue
        if task in ['a', 'c']:
            task_reports['task'+task] = ClassificationReport.make_report(
                true=true.iloc[:,0].values,
                pred=pred.iloc[:,0].values
            )
        else:
            task_reports['task'+task] = RegressionReport.make_report(
                true=true.values,
                pred=pred.values,
                labels=true.columns.tolist() if task == 'd' else None
            )
    return Results(**task_reports)
    


def estimators_eval(estimators:List[Tuple[str,Any]], score_func:Callable[[np.ndarray, np.ndarray], float]):
    def fit_eval_estimators(X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray) -> dict:
        estimator_scores = {}
        for name, estimator in estimators:
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            score = score_func(y_test, y_pred)#*(1.4*((y_train>th).sum()/(len(y_train)-1))) # weighted for class imbalance
            print(f"\"{name}\" estimator score: {score:.4f}")
            estimator_scores[name] = score
        return estimator_scores
    return fit_eval_estimators


def label_metrics(score_fun, y_true, y_pred):
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        scores = []
        for i in range(y_true.shape[1]):
            scores.append(score_fun(y_true[:,i],y_pred[:,i]))
        return scores
    score = score_fun(y_true.ravel(), y_pred.ravel())
    if isinstance(score, list):
        return score
    elif isinstance(score, np.ndarray):
        return score.tolist()
    else:
        return [score]

def metrics_for_estimators(estimators, score_fun, X, y_true):
    metrics = {}
    for name, estimator in estimators:
        y_pred = estimator.predict(X)
        metrics[name] = label_metrics(score_fun, y_true, y_pred)
    return metrics