#This file has been developed by the SINAI research group for its usage in the MentalRiskES evaluation campaign at IberLEF 2023.

# Required libraries
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from scipy.stats import pearsonr

# Read Gold labels for BinaryClassification
def read_qrels(qrels_file):
    qrels={}
    df_golden_truth = pd.read_csv(qrels_file)
    for index, r in df_golden_truth.iterrows():
        qrels[ r['Subject'] ] = int(r['label'])
    print("\n"+str(len(qrels))+ " lines read in qrels file!\n\n")
    return(qrels)

# Read Gold labels for Simple Regression
def read_qrels_regression(qrels_file):
    qrels={}
    df_golden_truth = pd.read_csv(qrels_file)
    for index, r in df_golden_truth.iterrows():
        qrels[ r['Subject'] ] = float(r['label'])
    print("\n"+str(len(qrels))+ " lines read in qrels file!\n\n")
    return(qrels)

# Read Gold labels for Multiclass classification
def read_qrels_multiclass(qrels_file):
    qrels={}
    qrels1 = {}
    df_golden_truth = pd.read_csv(qrels_file)
    for index, r in df_golden_truth.iterrows():
        qrels1[ r['Subject'] ] = r['label']
        if "suffer" in r['label']:
            qrels[ r['Subject'] ] = 1
        else:
            qrels[ r['Subject'] ] = 0
    print("\n"+str(len(qrels))+ " lines read in qrels file!\n\n")
    return qrels, qrels1

# Read Gold labels for Multi-output regression
def read_qrels_multioutput(qrels_file):
    qrels={}
    df_golden_truth = pd.read_csv(qrels_file)
    for index, r in df_golden_truth.iterrows():
        qrels[ r['Subject'] ] = [r['suffer_in_favour'],r['suffer_against'],r['suffer_other'],r['control']]
    print("\n"+str(len(qrels))+ " lines read in qrels file!\n\n")
    return qrels

###########################################################################
# Calculation of Binary classification metrics for Binary classification tasks
class BinaryClassification():
    def __init__(self, task, data, qrels):
        self.run_results = data 
        self.qrels_b = read_qrels(qrels)
        self.task = task
    pass

    def penalty(self,delay):
        if self.task == "1": # TCA
            p = 0.0292 # trial
        elif self.task == "2": # Depression
            p = 0.0179 # trial
        pen = -1.0 + 2.0/(1+np.exp(-p*(delay-1)))
        return(pen)

    def n_pos(self):
        total_pos = 0
        for key in self.qrels_b:
            total_pos += self.qrels_b[key]
        return(total_pos)

    def eval_performance(self):
        print("===================================================")
        print("DECISION-BASED EVALUATION:") 
        self.run_results = self.run_results.sort_values(by=['nick'])
        total_pos=self.n_pos()
        erdes5 = np.zeros(len(self.run_results))
        erdes30 = np.zeros(len(self.run_results))
        erdes50 = np.zeros(len(self.run_results))
        ierdes = 0
        true_pos = 0
        false_pos = 0
        latency_tps = list()
        penalty_tps = list()

        # Latency-based metrics
        for index, r in self.run_results.iterrows():
            try:
                
                if ( self.qrels_b[ r['nick'] ] ==  r['pred'] ):
                    if ( r['pred'] == 1 ):
                        true_pos+=1
                        erdes5[ierdes]=1.0 - (1.0/(1.0+np.exp( (r["round"]+1) - 5.0)))
                        erdes30[ierdes]=1.0 - (1.0/(1.0+np.exp( (r["round"]+1) - 30.0)))
                        erdes50[ierdes]=1.0 - (1.0/(1.0+np.exp( (r["round"]+1) - 50.0)))
                        latency_tps.append(r["round"]+1)
                        penalty_tps.append(self.penalty(r["round"]+1))
                    else:
                        erdes5[ierdes]=0
                        erdes30[ierdes]=0
                        erdes50[ierdes]=0
                else:
                    if ( r['pred'] == 1 ):
                        false_pos+=1
                        erdes5[ierdes]=float(total_pos)/float(len(self.qrels_b))
                        erdes50[ierdes]=float(total_pos)/float(len(self.qrels_b))
                    else:
                        erdes5[ierdes]=1
                        erdes30[ierdes]=1
                        erdes50[ierdes]=1
            except KeyError:
                print("User does not appear in the qrels:"+r['nick'])
            ierdes+=1

        _speed = 1-np.median(np.array(penalty_tps))
        if true_pos != 0 :
            precision = float(true_pos) / float(true_pos+false_pos)    
            recall = float(true_pos) / float(total_pos)
            f1_erde = 2 * (precision * recall) / (precision + recall)
            _latencyweightedF1 = f1_erde*_speed
        else:
            _latencyweightedF1 = 0
            _speed = 0
            
        y_true = self.run_results['pred'].tolist() 
        y_pred_b = list(self.qrels_b.values()) 

        # Binary metrics
        accuracy = metrics.accuracy_score(y_true, y_pred_b)
        macro_precision = metrics.precision_score(y_true, y_pred_b, average='macro')
        macro_recall = metrics.recall_score(y_true, y_pred_b, average='macro')
        macro_f1 = metrics.f1_score(y_true, y_pred_b, average='macro') 
        micro_precision = metrics.precision_score(y_true, y_pred_b, average='micro')
        micro_recall = metrics.recall_score(y_true, y_pred_b, average='micro')
        micro_f1 = metrics.f1_score(y_true, y_pred_b, average='micro')

        print("BINARY METRICS: =============================")
        print("Accuracy:"+str(accuracy))
        print("Macro precision:"+str(macro_precision))
        print("Macro recall:"+str(macro_recall))
        print("Macro f1:"+str(macro_f1))
        print("Micro precision:"+str(micro_precision))
        print("Micro recall:"+str(micro_recall))
        print("Micro f1:"+str(micro_f1))

        print("LATENCY-BASED METRICS: =============================")
        print("ERDE_5:"+str(np.mean(erdes5))) 
        print("ERDE_50:"+str(np.mean(erdes50))) 
        print("Median latency:"+str(np.median(np.array(latency_tps)))) 
        print("Speed:"+str(_speed)) 
        print("latency-weightedF1:"+str(_latencyweightedF1)) 
        
        return {'Accuracy': accuracy, 'Macro_P': macro_precision, 'Macro_R': macro_recall,'Macro_F1': macro_f1,'Micro_P': micro_precision, 'Micro_R': micro_recall,
        'Micro_F1': micro_f1, 'ERDE5':np.mean(erdes5),'ERDE30': np.mean(erdes30),'ERDE50': np.mean(erdes50), 'latencyTP': np.median(np.array(latency_tps)), 
        'speed': _speed, 'latency-weightedF1': _latencyweightedF1}

    # Calculation of P@10, P@20, P@30, P@50
    def eval_performance_rank_based(self):
        print("===================================================")
        print("RANK-BASED EVALUATION:")
        ranks_at=[1,50,75] 
        rank_dit = {}
        for rank in ranks_at:
            print("Analizing ranking at round "+str(rank))
            rels_topk = [0,0,0,0] 
            self.run_results["label"] = self.qrels_b.values()
            self.run_results = self.run_results.sort_values(by=['pred'],ascending=False) 
            i = 0
            for index, r in self.run_results.iterrows():
                if i<10:
                    if r["pred"] == r['label']:
                        rels_topk[0] += 1
                        rels_topk[1] += 1
                        rels_topk[2] += 1
                        rels_topk[3] += 1
                elif i<20:
                    if r["pred"] == r['label']:
                        rels_topk[1] += 1
                        rels_topk[2] += 1
                        rels_topk[3] += 1
                elif i<30:
                    if r["pred"] == r['label']:
                        rels_topk[2] += 1
                        rels_topk[3] += 1
                elif i<50:
                    if r["pred"] == r['label']:
                        rels_topk[3] += 1
                else:
                    break
                i+=1
            p10 = float(rels_topk[0])/10.0
            p20 = float(rels_topk[1])/20.0
            p30 = float(rels_topk[2])/30.0
            p50 = float(rels_topk[3])/50.0

            print("PRECISION AT K: =============================")
            print("P@10:"+str(p10))
            print("P@20:"+str(p20))
            print("P@30:"+str(p30))
            print("P@50:"+str(p50))
            rank_dit[rank] = {"@10":p10,"@20":p20,"@30":p30,"@50":p50}
        return rank_dit


#############################################################################################
# Calculation of Regression metrics for Simple regression tasks
class ClassRegressionEvaluation():
    def __init__(self, task, data, qrels):
        self.run_results = data 
        self.qrels = read_qrels_regression(qrels)
        self.task = task

    def eval_performance(self):
        self.run_results = self.run_results.sort_values(by=['nick'])
        y_true = self.run_results['pred'].tolist() 
        
        y_pred_r = list(self.qrels.values()) 

        # Regression metrics
        _rmse = metrics.mean_squared_error(y_true, y_pred_r, sample_weight=None, multioutput='raw_values', squared=False)[0] 
        _pearson = np.corrcoef(y_true, y_pred_r)
        _pearson, _ = pearsonr(y_true, y_pred_r)

        print("REGRESSION METRICS: =============================")
        print("RMSE:"+str(_rmse))
        print("Pearson correlation coefficient:"+str(_pearson))

        return { 'RMSE:': _rmse, 'Pearson_coefficient': _pearson}

    # Calculation of P@10, P@20, P@30, P@50
    def eval_performance_rank_based(self):
        print("===================================================")
        print("RANK-BASED EVALUATION:")
        ranks_at=[1,25,50,75] 
        rank_dit = {}
        for rank in ranks_at:
            print("Analizing ranking at round "+str(rank))
            rels_topk = [0,0,0,0,0] 
            self.run_results_ = self.run_results[rank].sort_values(by=['nick'])
            self.run_results_["label"] = self.qrels.values()
            self.run_results_ = self.run_results_.sort_values(by=['pred'],ascending=False) 
            i = 0
            for index, r in self.run_results_.iterrows():
                if i<5:
                    if r["label"] == round(r["pred"],1):
                        rels_topk[0] += 1
                        rels_topk[1] += 1
                        rels_topk[2] += 1
                        rels_topk[3] += 1
                        rels_topk[4] += 1
                elif i<10:
                    if r['label'] == round(r["pred"],1):
                        rels_topk[1] += 1
                        rels_topk[2] += 1
                        rels_topk[3] += 1
                        rels_topk[4] += 1
                elif i<20:
                    if  r['label'] == round(r["pred"],1):
                        rels_topk[2] += 1
                        rels_topk[3] += 1
                        rels_topk[4] += 1
                elif i<30:
                    if  r['label'] == round(r["pred"],1):
                        rels_topk[3] += 1
                        rels_topk[4] += 1
                elif i<50:
                    if  r['label'] == round(r["pred"],1):
                        rels_topk[4] += 1
                else:
                    break
                i+=1
            p5 = float(rels_topk[0])/5.0
            p10 = float(rels_topk[1])/10.0
            p20 = float(rels_topk[2])/20.0
            p30 = float(rels_topk[3])/30.0
            p50 = float(rels_topk[4])/50.0

            print("PRECISION AT K: =============================")
            print("P@5:"+str(p5))
            print("P@10:"+str(p10))
            print("P@20:"+str(p20))
            print("P@30:"+str(p30))
            print("P@50:"+str(p50))
            rank_dit[rank] = {"@5":p5,"@10":p10,"@20":p20,"@30":p30,"@50":p50}
        return rank_dit


############################################################################
# Calculation of Binary metrics for Multiclass classification tasks
class BinaryMultiClassification():
    def __init__(self, task, data, qrels):
        self.run_results = data 
        self.qrels_b, self.qrels_multiclass  = read_qrels_multiclass(qrels)
        self.task = task
    pass

    def penalty(self,delay):
        if self.task == "1": # TCA
            p = 0.0411 # test
            p = 0.0292 # trial
        elif self.task == "2": # Depression
            p = 0.0326 # test
            p = 0.0179 # trial
        else: # Unkown
            p = 0.0308 # test
        pen = -1.0 + 2.0/(1+np.exp(-p*(delay-1)))
        return(pen)

    def n_pos(self):
        total_pos = 0
        for key in self.qrels_b:
            total_pos += self.qrels_b[key]
        return(total_pos)


    def eval_performance(self):
        print("===================================================")
        print("DECISION-BASED EVALUATION:") 
        self.run_results = self.run_results.sort_values(by=['nick'])
        total_pos=self.n_pos() # Total number of positive documents
        erdes5 = np.zeros(len(self.run_results))
        erdes30 = np.zeros(len(self.run_results))
        erdes50 = np.zeros(len(self.run_results))
        ierdes = 0
        true_pos = 0
        false_pos = 0
        latency_tps = list()
        penalty_tps = list()

        for index, r in self.run_results.iterrows():
            try:
                
                if ( self.qrels_b[ r['nick'] ] ==  r['pred_b'] ):
                    if ( r['pred_b'] == 1 ):
                        true_pos+=1
                        erdes5[ierdes]=1.0 - (1.0/(1.0+np.exp( (r["round"]+1) - 5.0)))
                        erdes30[ierdes]=1.0 - (1.0/(1.0+np.exp( (r["round"]+1) - 30.0)))
                        erdes50[ierdes]=1.0 - (1.0/(1.0+np.exp( (r["round"]+1) - 50.0)))
                        latency_tps.append(r["round"]+1)
                        penalty_tps.append(self.penalty(r["round"]+1))
                    else:
                        erdes5[ierdes]=0
                        erdes30[ierdes]=0
                        erdes50[ierdes]=0
                else:
                    if ( r['pred_b'] == 1 ):
                        false_pos+=1
                        erdes5[ierdes]=float(total_pos)/float(len(self.qrels_b))
                        erdes30[ierdes]=float(total_pos)/float(len(self.qrels_b))
                        erdes50[ierdes]=float(total_pos)/float(len(self.qrels_b))
                    else:
                        erdes5[ierdes]=1
                        erdes30[ierdes]=1
                        erdes50[ierdes]=1
            except KeyError:
                print("User does not appear in the qrels:"+r['nick'])
            ierdes+=1

        _speed = 1-np.median(np.array(penalty_tps))
        if true_pos != 0 :
            precision = float(true_pos) / float(true_pos+false_pos)    
            recall = float(true_pos) / float(total_pos)
            f1_erde = 2 * (precision * recall) / (precision + recall)
            _latencyweightedF1 = f1_erde*_speed
        else:
            _latencyweightedF1 = 0
            _speed = 0
            
        y_true = self.run_results['pred'].tolist()
        y_pred_b = list(self.qrels_multiclass.values())

        # Binary metrics
        accuracy = metrics.accuracy_score(y_true, y_pred_b)
        macro_precision = metrics.precision_score(y_true, y_pred_b, average='macro')
        macro_recall = metrics.recall_score(y_true, y_pred_b, average='macro')
        macro_f1 = metrics.f1_score(y_true, y_pred_b, average='macro')
        micro_precision = metrics.precision_score(y_true, y_pred_b, average='micro')
        micro_recall = metrics.recall_score(y_true, y_pred_b, average='micro')
        micro_f1 = metrics.f1_score(y_true, y_pred_b, average='micro')

        print("BINARY METRICS: =============================")
        print("Accuracy:"+str(accuracy))
        print("Macro precision:"+str(macro_precision))
        print("Macro recall:"+str(macro_recall))
        print("Macro f1:"+str(macro_f1))
        print("Micro precision:"+str(micro_precision))
        print("Micro recall:"+str(micro_recall))
        print("Micro f1:"+str(micro_f1))

        print("LATENCY-BASED METRICS: =============================")
        print("ERDE_5:"+str(np.mean(erdes5)))
        print("ERDE_50:"+str(np.mean(erdes50)))
        print("Median latency:"+str(np.median(np.array(latency_tps)))) 
        print("Speed:"+str(_speed)) 
        print("latency-weightedF1:"+str(_latencyweightedF1)) 
        
        return {'Accuracy': accuracy, 'Macro_P': macro_precision, 'Macro_R': macro_recall,'Macro_F1': macro_f1,'Micro_P': micro_precision, 'Micro_R': micro_recall,
        'Micro_F1': micro_f1, 'ERDE5':np.mean(erdes5),'ERDE30':np.mean(erdes30),'ERDE50': np.mean(erdes50), 'latencyTP': np.median(np.array(latency_tps)), 
        'speed': _speed, 'latency-weightedF1': _latencyweightedF1}
    
    # Calculation of P@10, P@20, P@30, P@50
    def eval_performance_rank_based(self):
        print("===================================================")
        print("PRECISION AT K - EVALUATION:")
        ranks_at=[1,50,75] 
        rank_dit = {}
        for rank in ranks_at:
            print("Analizing ranking at round "+str(rank))
            rels_topk = [0,0,0,0]
            self.run_results["label"] = self.qrels_b.values()
            self.run_results = self.run_results.sort_values(by=['pred_b'],ascending=False) 
            i = 0
            for index, r in self.run_results.iterrows():
                if i<10:
                    if r["pred_b"] == r['label']:
                        rels_topk[0] += 1
                        rels_topk[1] += 1
                        rels_topk[2] += 1
                        rels_topk[3] += 1
                elif i<20:
                    if r["pred_b"] == r['label']:
                        rels_topk[1] += 1
                        rels_topk[2] += 1
                        rels_topk[3] += 1
                elif i<30:
                    if r["pred_b"] == r['label']:
                        rels_topk[2] += 1
                        rels_topk[3] += 1
                elif i<50:
                    if r["pred_b"] == r['label']:
                        rels_topk[3] += 1
                else:
                    break
                i+=1
            p10 = float(rels_topk[0])/10.0
            p20 = float(rels_topk[1])/20.0
            p30 = float(rels_topk[2])/30.0
            p50 = float(rels_topk[3])/50.0

            print("PRECISION AT K: =============================")
            print("P@10:"+str(p10))
            print("P@20:"+str(p20))
            print("P@30:"+str(p30))
            print("P@50:"+str(p50))
            rank_dit[rank] = {"@10":p10,"@20":p20,"@30":p30,"@50":p50}
        return rank_dit


#######################################################################################
# Calculation of Regression metrics for Multi-output regression tasks
class ClassMultiRegressionEvaluation():

    def __init__(self, task, data, qrels):
        self.run_results = data 
        self.qrels = read_qrels_multioutput(qrels)
        self.task = task

    def eval_performance(self):
        self.run_results = self.run_results.sort_values(by=['nick'])
        y_true = self.run_results['pred'].tolist() 
        y_pred_r = list(self.qrels.values()) 

        # Regression metrics
        _rmse = metrics.mean_squared_error(y_true, y_pred_r, sample_weight=None, multioutput='raw_values', squared=False)[0]
        _pearson_sf, _ = pearsonr([item[0] for item in y_true] , [item[0] for item in y_pred_r])
        _pearson_sa, _ = pearsonr([item[1] for item in y_true] , [item[1] for item in y_pred_r])
        _pearson_so, _ = pearsonr([item[2] for item in y_true] , [item[2] for item in y_pred_r])
        _pearson_c, _ = pearsonr([item[3] for item in y_true] , [item[3] for item in y_pred_r])

        print("REGRESSION METRICS: =============================")
        print("RMSE:"+str(_rmse))
        print("Pearson correlation coefficient:")
        print("Pearson sf:"+str(_pearson_sf))
        print("Pearson sa:"+str(_pearson_sa))
        print("Pearson so:"+str(_pearson_so))
        print("Pearson c:"+str(_pearson_c))
        pearson = (_pearson_sf + _pearson_sa + _pearson_so + _pearson_c)/4
        return { 'RMSE:': _rmse, 'Pearson_mean': pearson,'Pearson_sf': _pearson_sf, 'Pearson_sa': _pearson_sa,'Pearson_so': _pearson_so,'Pearson_c': _pearson_c}
    
    # Calculation of P@10, P@20, P@30, P@50
    def eval_performance_rank_based(self):
        print("===================================================")
        print("PRECISION AT - EVALUATION:")
        ranks_at=[1,25,50,75] 
        rank_dit = {}
        for rank in ranks_at:
            print("Analizing ranking at round "+str(rank))
            self.run_results_ = self.run_results[rank].sort_values(by=['nick'])
            self.run_results_["label"] = self.qrels.values()
            self.run_results_ = self.run_results_.sort_values(by=['pred'],ascending=False) 
            p5 = 0
            p10 = 0
            p20 = 0
            p30 = 0
            p50 = 0 
            for j in range(0,4): 
                rels_topk = [0,0,0,0,0]
                i = 0
                for index, r in self.run_results_.iterrows():
                    if i<5:
                        if r['label'][j] == round(r["pred"][j],1):
                            rels_topk[0] += 1
                            rels_topk[1] += 1
                            rels_topk[2] += 1
                            rels_topk[3] += 1
                            rels_topk[4] += 1
                    elif i<10:
                        if r['label'][j] == round(r["pred"][j],1):
                            rels_topk[0] += 1
                            rels_topk[1] += 1
                            rels_topk[2] += 1
                            rels_topk[3] += 1
                    elif i<20:
                        if r['label'][j] == round(r["pred"][j],1):
                            rels_topk[1] += 1
                            rels_topk[2] += 1
                            rels_topk[3] += 1
                    elif i<30:
                        if r['label'][j] == round(r["pred"][j],1):
                            rels_topk[2] += 1
                            rels_topk[3] += 1
                    elif i<50:
                        if r['label'][j] == round(r["pred"][j],1):
                            rels_topk[3] += 1
                    else:
                        break
                    i+=1
                p5 += float(rels_topk[0])/5.0
                p10 += float(rels_topk[0])/10.0
                p20 += float(rels_topk[1])/20.0
                p30 += float(rels_topk[2])/30.0
                p50 += float(rels_topk[3])/50.0

            print("PRECISION AT K: =============================")
            print("P@5:"+str(p5/4))
            print("P@10:"+str(p10/4))
            print("P@20:"+str(p20/4))
            print("P@30:"+str(p30/4))
            print("P@50:"+str(p50/4))
            rank_dit[rank] = {"@5":p5/4,"@10":p10/4,"@20":p20/4,"@30":p30/4,"@50":p50/4}
        return rank_dit


# Class for calculating carbon emission values
class Emissions():
    def __init__(self, emissions_run) -> None:
        self.emissions_run = emissions_run
        self.aux = {}
        for key, value in emissions_run.items():
            self.aux[key] = 0
        pass

    # Update of values after a prediction has been made
    def update_emissions(self,emissions_round):
        # The values are accumulated in each round, so the difference is calculated to know the values for that round only
        for key, value in self.emissions_run.items():
            if key not in ["cpu_count","gpu_count","cpu_model","gpu_model", "ram_total_size"]:
                round_ = emissions_round[key] - self.aux[key]
            self.emissions_run[key].append(round_)
            self.aux[key] = emissions_round[key]

    # Calculation of final values after all predictions have been made
    def calculate_emissions(self):
        dict_ = {}
        for key, value in self.emissions_run.items():
            # Non-numerical values
            if key in ["cpu_count","gpu_count","cpu_model","gpu_model", "ram_total_size"]:
                dict_[key] = self.emissions_run[key][0]
            # Numerical values
            else:
                dict_[key+"_min"] = min(self.emissions_run[key])
                dict_[key+"_max"] = max(self.emissions_run[key])
                dict_[key+"_mean"] = sum(self.emissions_run[key])/len(self.emissions_run[key])
                dict_[key+"_var"] = np.var(self.emissions_run[key])
        return dict_
