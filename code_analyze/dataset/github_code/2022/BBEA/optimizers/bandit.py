from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import traceback
import copy

import numpy as np

from xoa.commons.logger import *
from xoa.commons.saver import *
from xoa.commons.converter import TimestringConverter

from xoa.spaces import *
from xoa.spaces.active import *

from optimizers.arms import SurrogateModelManager
from optimizers.repo import ResultsRepository

NUM_MAX_ITERATIONS = 1000 # XXX:reduce a maximum number of iterations for the NAS101 case 
MAX_ERROR = 10000


class HPOBanditMachine(object):
    ''' k-armed bandit machine of hyper-parameter optimization.'''
    def __init__(self, s_space, trainer, 
                 run_mode, target_val, time_expired, run_config, goal_metric,
                 num_resume=0, 
                 save_internal=False, 
                 calc_measure=False,
                 min_train_epoch=1,
                 id="HPOBanditMachine"):

        self.id = id

        self.search_space = s_space
        self.save_name = s_space.get_name()
        
        self.trainer = trainer
        
        self.calc_measure = calc_measure
        
        self.target_goal = target_val
        self.goal_metric = goal_metric
        self.time_expired = TimestringConverter().convert(time_expired)

        self.save_internal = save_internal
        if self.save_internal:
            info("Internal data will be stored.")
        self.num_resume = num_resume
        self.min_candidates = 100 # XXX:any number to be tested 
        self.cs = None # candidate sampler
        
        self.warm_up = { 'time': None, 'iters': 0, 'select': {}, 'revisit': 2 }        
        
        self.warm_up_revisit = 2 # XXX: number of best configurations from warm-up phase to be revisited
        
        self.run_config = run_config
        self.min_train_epoch = min_train_epoch
        self.max_train_epoch = None
        self.use_interim = True
        self.max_error = 100.0

        self.report_type = 'test'
        
        if self.run_config:
            if "min_train_epoch" in self.run_config:
                self.min_train_epoch = self.run_config["min_train_epoch"]

            if "max_train_epoch" in self.run_config:
                self.max_train_epoch = self.run_config["max_train_epoch"]

            if "warm_up" in self.run_config:
                warm_up_cfg = self.run_config["warm_up"]
                if "time" in warm_up_cfg:
                    self.warm_up['time'] = TimestringConverter().convert(warm_up_cfg["time"])
                
                if "iters" in warm_up_cfg:
                    self.warm_up['iters'] = warm_up_cfg["iters"]
                    
                if "revisit" in warm_up_cfg:
                    self.warm_up['revisit'] = warm_up_cfg['revisit']

            if "report_type" in self.run_config:
                self.report_type = self.run_config['report_type']

            if "use_interim" in self.run_config:
                self.use_interim = self.run_config['use_interim']

        self.run_mode = run_mode  # can be 'GOAL' or 'TIME'
        self.print_exception_trace = False

        self.stop_flag = False
                
        self.repo = ResultsRepository(self.goal_metric, self.report_type)
        self.cur_results = None
        self.incumbent = None
        self.steps_after_best = 0
        self.fail_time = 0.0

        self.show_run_condition()

    def get_search_space(self):
        return self.search_space

    def show_run_condition(self):
        if 'num_trials' in self.run_config:
            num_trials = self.run_config['num_trials']
        else:
            num_trials = 1
        
        mode = "{} performance of best configuration".format(self.report_type)
        criterion = ""
        if 'benchmark_mode' in self.run_config:
            mode += " experimented on benchmark mode"
            criterion = "when estimated wall clock time expires after {:.0f} sec".format(self.time_expired)
        else:
            term_time = time.localtime(time.time() + (self.time_expired * num_trials))
            criterion = "executing until {}".format(time.asctime(term_time))
        
        if self.run_mode == "GOAL":
            if criterion != "":
                criterion += " or "
            criterion += "when {} {} is achieved".format(self.target_goal, self.goal_metric)
        
        info("Termination condition: {}, it will return {} ".format(criterion, mode))

    def reset(self, run_config=None):
        if run_config is None:
            run_config = self.run_config                
        self.search_space.initialize()
        self.trainer.initialize()
        v = None
        if "search_space" in self.run_config:
            s_spec = self.run_config['search_space']
            
            if "verification" in s_spec and s_spec["verification"] == True:
                v = self.trainer.get_verifier()    

        self.mab = SurrogateModelManager(self.search_space, run_config, verifier=v)
        self.repo = ResultsRepository(self.goal_metric, self.report_type)
        # initialize candidate sampler
        if "search_space" in self.run_config:
            s_spec = self.run_config['search_space']
            if "resample" in s_spec:
                method = s_spec['resample']
                self.cs = CandidateSetResampler(self.search_space, self.mab, v) 
        
        self.incumbent = None
        self.steps_after_best = 0        
        self.fail_time = 0.0

        self.warm_up['select'] = {}


    def stop(self):
        self.stop_flag = True
    
    def get_cur_runtime(self):        
        return self.repo.get_elapsed_time()

    def choose_by_prior(self, n_rank):
        try:
            # returns n_rank th best performed configurations(which has high performance)
            for k in self.warm_up['select'].keys():                
                err = self.search_space.get_errors(k, error_type='test')
                if type(err) == list:
                    err = err[0]                
                s = self.warm_up['select'][k]
                
                if s['index'] != k: # validate key-value set
                    raise ValueError("Invalid key-value pair in dictionary.")
                if err == None or np.isnan(err):
                    err = MAX_ERROR
                s['test_error'] = err
                
            sorted_list = sorted(self.warm_up['select'].items(), key=lambda item: item[1]['test_error'] )
            r = sorted_list[n_rank][1]
            return r['index'], r['model'], r['acq_func']

        except Exception as ex:
            warn("Exception at choosing by prior: {}".format(ex))
            return None, None, None

    def is_warm_up_stage(self):
        
        if self.warm_up['time'] != None:
            if self.get_cur_runtime() <= self.warm_up['time']:
                return True
            else:
                return False        
        
        if self.warm_up['select'] != None:
            if self.warm_up['iters'] <= len(self.warm_up['select'].keys()):
                return False
            else:
                return True
        else:
            return False 

    def choose(self, model_name, acq_func, search_space=None):
        
        if search_space == None:
            search_space = self.search_space
                
        d = len(search_space.get_completions())
        c = len(search_space.get_candidates())
        debug("Choosing next candidate using {}-{} (obs.# {}, cand.# {})".format(model_name, acq_func, d, c)) 
        model = self.mab.get_model(model_name)

        if self.is_warm_up_stage():
            # low-fidelity optimization mode
            debug("Low-fidelity optimization will be performed")
            next_index, est_values = model.next(acq_func)
            self.warm_up['select'][next_index] = { "model": model_name, "acq_func": acq_func, "index": next_index }

            return next_index, est_values, self.min_train_epoch

        # high-fidelity optimization mode
        if self.warm_up['select'] == None:
            next_index, est_values = model.next(acq_func, self.min_train_epoch)
            return next_index, est_values, self.max_train_epoch

        n_warm_up = len(self.warm_up['select'].keys()) 
        if n_warm_up < self.warm_up_revisit:
            debug("The # of obs. in warm-up phase is less than {}: {}.".format(self.warm_up_revisit, len(self.warm_up['select'])))
            self.warm_up_revisit = n_warm_up
    
        # high-fidelity optimization from low-fidelity results
        n_compl = len(search_space.get_completions(self.min_train_epoch)) # get high fidelity results only
        if n_compl < self.warm_up_revisit:
            
            next_index, model_name, acq_func = self.choose_by_prior(search_space, n_compl)
            if next_index != None:                    
                info("High-fidelity optimization of {}: {}/{}".format(next_index, n_compl+1, self.warm_up_revisit))
                return next_index, None, self.max_train_epoch
        
        if self.warm_up['select'] != None:
            debug("Low-fidelity optimization ended completely.")
            self.warm_up['select'] = None # end of restoration
        
        next_index, est_values = model.next(acq_func, self.min_train_epoch)
        return next_index, est_values, self.max_train_epoch
    
    def evaluate(self, cand_index, train_epoch):
        
        eval_start_time = time.time()


        if self.use_interim:
        # set initial error for avoiding duplicate
            interim_error, cur_epoch = self.trainer.get_interim_error(cand_index)
            self.search_space.update_error(cand_index, interim_error, num_epochs=cur_epoch, error_type='valid')
        else:
            interim_error = self.max_error       
        try:
            train_result = self.trainer.train(cand_index, train_epoch)
        
        except SystemError as se:
            error("System error from trainer: {}".format(se))
            sys.exit(-1)

        except Exception as ex:
            warn("Exception from trainer: {}".format(ex))
            traceback.print_exc()
            train_result = None

        if train_result == None or not 'test_error' in train_result:
            train_result = {}
            # return interim error for avoiding stopping
            train_result['test_error'] = interim_error            
            train_result['early_terminated'] = True
            test_error = interim_error
        else:
            test_error = train_result['test_error']
        
        train_result['model_idx'] = cand_index

        if not 'exec_time' in train_result:
            train_result['exec_time'] = time.time() - eval_start_time

        if 'train_epoch' in train_result:
            train_epoch = train_result['train_epoch']
        else:
            train_result['train_epoch'] = train_epoch

        # Set final performance
        if test_error != None:
            self.search_space.update_error(cand_index, test_error, train_epoch, error_type='test')
        
        if 'valid_error' in train_result:
            valid_error = train_result['valid_error']
        else:
            valid_error = test_error # use of test error as validation error
        
        if valid_error != None:
            self.search_space.update_error(cand_index, valid_error, train_epoch, error_type='valid')

        return train_result

    def pull(self, n_steps):
        model = 'NONE'
        acq_func = 'RANDOM'
        s_spec = {}

        try:
            start_time = time.time()

            if self.incumbent != None and "known_best" in self.run_config:
                k_b = float(self.run_config["known_best"])
                #debug("Current best: {}, known best: {}".format(self.incumbent, k_b))
                # speedup remained iterations using random selection
                if self.goal_metric == "error" and \
                    self.incumbent <= k_b:
                    self.cs = None # XXX:stop to resample candidates
                    raise Exception('optimal error ({}) has been achieved.'.format(k_b))
                
                if self.goal_metric == "accuracy" and \
                    self.incumbent >= k_b:
                    self.cs = None # XXX:stop to resample candidates
                    raise Exception('optimal accuarcy ({}) has been achieved.'.format(k_b))            

            model, acq_func = self.mab.get_selected_model(n_steps)

            if "search_space" in self.run_config:
                s_spec = self.run_config['search_space']
                if "resample" in s_spec:
                    method = s_spec['resample']
                    vetoer = { 'model': model, 'acq_func': acq_func }
                    if 'allow_same_model' in s_spec and s_spec['allow_same_model'] == True:
                        vetoer = None
                if self.cs is not None:              
                    self.resample_candidates(method, n_steps, vetoer=vetoer)
            
            cand_index, est_values, train_epoch = self.choose(model, acq_func)

            if type(est_values) == dict:
                if s_spec and "evolve_div" in s_spec:
                    est_values['all_selection'] = self.mab.get_others_best()
                    est_values['all_selection'].append(cand_index)

        except KeyboardInterrupt:
            self.stop_flag = True
            return 0.0, None
        except Exception as ex:
            if not 'has been achieved.' in str(ex):
                error("The next candidate will be randomly chosen due to the following error: {}".format(ex))
                warn(traceback.format_exc())
            model = 'NONE'
            acq_func = 'RANDOM'
            cand_index, est_values, train_epoch = self.choose(model, acq_func)

        self.repo.update_trace(model, acq_func)
        opt_time = time.time() - start_time
        
        eval_result = self.evaluate(cand_index, train_epoch)
        eval_result['opt_time'] = opt_time
        
        # Check whether the evaluation has been failed
        if 'train_failed' in eval_result and eval_result['train_failed'] == True:
            warn("Training the configuration #{} failed!".format(cand_index))
        elif "add" in s_spec or "remove" in s_spec or \
            "intensify" in s_spec or "evolve" in s_spec or \
            "resample_steps" in s_spec:
            # update posterior distribution of the samples
            self.update_space(n_steps, s_spec, eval_result, est_values)
           
        # update evaluation result
        self.repo.append(eval_result)

        # reset candidate set
        self.search_space.restore_candidates()


        y = None
        if self.report_type == 'test' and 'test_{}'.format(self.goal_metric) in eval_result:
            y = eval_result['test_{}'.format(self.goal_metric)]
        elif self.report_type == 'validation' and 'valid_{}'.format(self.goal_metric) in eval_result:
            y = eval_result['valid_{}'.format(self.goal_metric)]
        elif self.goal_metric in eval_result:
            y = eval_result[self.goal_metric]
        elif self.goal_metric == 'accuracy' and 'test_error' in eval_result:
            y = 1.0 - eval_result['test_error']
        elif self.goal_metric == 'accuracy' and 'test_error' in eval_result:
            y = 1.0 - eval_result['test_error']
        else:
            raise ValueError("No {} in eval_result keys: {}".format(self.goal_metric, eval_result.keys()))
        if model != 'NONE':
            self.mab.feedback(n_steps, y, est_values, self.goal_metric)        

        return y, est_values

    def check_incumbent_updated(self, eval_result):
        cur_best = self.search_space.get_incumbent()
        if type(cur_best) == dict:
            if self.report_type == 'test':
                if 'test_error' in cur_best and 'test_error' in eval_result:
                    # Only when last evaluation updates the current best performance 
                    best_err = cur_best['test_error'] # None when it starts
                    new_err = eval_result['test_error']
                else:
                    raise ValueError("No attribute named test_error")
            elif self.report_type == 'validation':
                if 'valid_error' in cur_best and 'valid_error' in eval_result:
                    best_err = cur_best['valid_error'] # None when it starts
                    new_err = eval_result['valid_error']                        
                else:
                    raise ValueError("No attribute named valid_error")            
            else:
                raise ValueError("No supported report type: {}".format(self.report_type))
            if best_err == None or best_err > new_err:
                return True
            else:
                return False

    def update_incumbent(self, y):
        if y == None: # in case of error, skip belows
            self.steps_after_best += 1
            return                
        
        duration = self.get_cur_runtime()
        if self.incumbent == None:
            self.incumbent = y
            self.steps_after_best = 0
            
            info("Initial {} {} performance: {:.6f} ({:.0f} sec)".format(self.report_type, self.goal_metric, y, duration))
        elif self.goal_metric == "accuracy" and self.incumbent < y:
            info("Best {} accuracy change: {:.6f} -> {:.6f} ({:.0f} sec)".format(self.report_type, self.incumbent, y, duration))
            self.incumbent = y
            self.steps_after_best = 0            
        elif self.goal_metric == "error" and self.incumbent > y:
            info("Best {} error change:{:.6f} -> {:.6f} ({:.0f} sec)".format(self.report_type, self.incumbent, y, duration))
            self.incumbent = y
            self.steps_after_best = 0
        else:
            self.steps_after_best += 1
            
    def play(self, mode, spec, num_runs, save=True):
        temp_saver = TemporaryHistorySaver(self.save_name,
                                            mode, spec, num_runs, 
                                            self.run_config)
        saver = None

        if save == True:
            saver = HistorySaver(self.save_name, self.run_mode, self.target_goal,
                                    self.time_expired, self.run_config, 
                                    postfix=".{}".format(self.id))            
        
        bm_mode = False
        if 'benchmark_mode' in self.run_config and self.run_config['benchmark_mode']:
            bm_mode = True

        # For in-depth analysis
        opt_rec = None
        if self.save_internal:
            opt_rec = {}

        # restore prior history
        if self.num_resume > 0:
            self.cur_results, start_idx = saver.load(mode, spec, self.num_resume)
        else:
            self.cur_results, start_idx = temp_saver.restore()
        
        if start_idx > 0:
            info("HPO runs will be continued from temporary saved result(s).")

        for i in range(start_idx, num_runs): # loop for multiple HPO runs           
            start_time = time.time()
            self.reset()
            self.mab.reset(mode, spec)
            total_queries = NUM_MAX_ITERATIONS
            
            for j in range(NUM_MAX_ITERATIONS): # loop for each HPO run
                debug("Pulling the arm {} times at {:.1f}".format(j + 1, self.get_cur_runtime()))                 
                y, opt = self.pull(j)
                self.update_incumbent(y)

                if j > 0 and j % 100 == 0:
                    info("{} configurations have been evaluated ({:.0f} sec)".format(j, self.get_cur_runtime()))               

                if self.stop_flag == True:
                    return self.cur_results

                if num_runs == 1 and not bm_mode:
                    self.cur_results[i] = self.repo.get_current_status()
                    temp_saver.save(self.cur_results)
                    self.search_space.save_history(i) 

                    if self.save_internal:                        
                        if j % 10 == 0:
                            info("Storing internal data at step {}... ".format(j))
                            opt_rec['est_{}'.format(j)] = opt
                            dist = self.search_space.compute_all(self.trainer, self.max_train_epoch)
                            opt_rec['dist_{}'.format(j)] = dist

                # stopping criteria check
                if self.is_terminated(start_time):
                    if not bm_mode:
                        # update final search history
                        self.search_space.save_history(i, 'results/') 
                    values, counts = self.mab.get_stats()
                    self.repo.save_select_status(values, counts)
                    total_queries = j
                    break  

            wall_time = time.time() - start_time
            info("Best {} {} at run #{} is {:.6f} by {} queries. (wall-clock {:.0f} sec)".format(self.report_type, self.goal_metric, 
                 i, self.incumbent, total_queries, wall_time))

            self.cur_results[i] = self.repo.get_current_status()
            temp_saver.save(self.cur_results)
            self.search_space.archive(i)            

        if saver:
            saver.save(mode, spec, num_runs, self.cur_results, opt_rec)

        if start_idx == num_runs:
            warn("No more extra runs.")
        
        temp_saver.remove()

        return self.cur_results

    def is_terminated(self, start_time):
        
        if self.run_mode == 'GOAL':
            if self.goal_metric == "accuracy" and self.incumbent >= self.target_goal:
                return True
            elif self.goal_metric == "error" and self.incumbent <= self.target_goal:
                return True
        
        duration = self.get_cur_runtime()        
        if duration >= self.time_expired:
            info("Timeout: simulated run time is over ({:.0f} sec)".format(duration))
            return True
        elif time.time() - start_time >= self.time_expired:
            info("Timeout: actual run time is over ({:.0f} sec)".format(self.time_expired - duration))
            return True         
        else:
            #info("Termination check at {:.0f}".format(duration))
            return False

    def get_working_results(self):
        results = []
        if self.cur_results:
            results += self.cur_results

        results.append(self.get_repo().get_current_status())
        return results

    def get_repo(self):
        if self.repo == None:
            self.repo = ResultsRepository(self.goal_metric, self.report_type) 
        return self.repo

    def print_incumbent(self, results):
        for k in results.keys():
            try:
                result = results[k]
                
                error_min_index = 0
                cur_best_err = None
                for i in range(len(result['error'])):
                    err = result['error'][i]
                    if cur_best_err == None:
                        cur_best_err = err    
                    # if an error is None, ignore them
                    if err != None:
                        if cur_best_err > err:
                            cur_best_err = err
                            error_min_index = i
                
                best_model_index = result['model_idx'][error_min_index]
                best_error = result['test_error'][error_min_index]
                best_hpv = self.search_space.get_hpv_dict(best_model_index, int(k))
                info("[R{}H] {} -> {}.".format(k, best_hpv, best_error))
            except Exception as ex:
                #warn("[Run #{}] report failed".format(k))
                pass

    def update_space(self, n_done, s_spec, eval_result, est_values):
        v = None
        if "verification" in s_spec and s_spec["verification"] == True:
            v = self.trainer.get_verifier()

        an = 0
        if "resample_after" in s_spec:
            an = s_spec['resample_after']
        
        if n_done > an:
            csc = CandidateSetController(self.search_space, v)
            # for every step ss of HPO iteration
            if "update_rule" in s_spec:            
                pes_spec = s_spec["update_rule"]
                pess = 1  # default step size for adaptive sampling

                if "resample_steps" in s_spec:            
                    pess = s_spec["resample_steps"]
                else:
                    pess = 1

                if type(pess) == int:
                    if n_done > 0 and n_done % pess == 0:
                        if 'rule_name' in pes_spec:
                            update_rule = pes_spec['rule_name']
                            if update_rule == 'add_when_explore_or_evolve_when_exploit':
                                spec = copy.deepcopy(pes_spec)
                                if self.check_incumbent_updated(eval_result):
                                    spec['rule_name'] = 'exploit'
                                    spec.pop('add', None) # drop add condition
                                else:
                                    spec['rule_name'] = 'explore'
                                    spec.pop('evolve', None) # drop evolve condition
                                
                                debug("Current resampling spec: {}".format(spec))
                                pes_spec = spec
                            else:
                                raise NotImplementedError("Invalid update rule: {}".format(update_rule))                        

                        csc.resample(est_values, pes_spec)
                else:
                    warn("Invalid resampling step size!")
            else:
                # for global steps
                ss = 'new_record'  # default step size for adaptive sampling
                if "resample_steps" in s_spec:            
                    ss = s_spec["resample_steps"]

                if type(ss) == int:
                    if n_done > 0 and n_done % ss == 0:
                        csc.resample(est_values, s_spec)

                elif ss == 'new_record':
                    if self.check_incumbent_updated(eval_result):
                        info("Resampling will be performed due to new best found")
                        csc.resample(est_values, s_spec)

    def resample_candidates(self, method, n_steps, vetoer=None, verifier=None):
        
        s_method = 'UNIFORM' # default method
        n_cand = 100

        # parsing resampling method 
        if '[' in method and ']' in method:
            o_i = method.find('[')
            e_i = method.find(']')
            s_method = method[:o_i]
            n_cand = int(method[o_i+1:e_i])
        elif type(method) == int:
            n_cand = int(method)
        else:
            raise ValueError("Invalid resampling method: {}".format(method))

        # resampling using batched BO
        if 'BATCH-DIV' in s_method:
            # perform cross-over after batched BO diversification
            if n_steps <= 2:
                return            
            self.cs.ensemble(s_method, n_cand, n_steps, self.steps_after_best)            
        else:                      
            self.cs.resample(s_method, n_cand, n_steps, vetoer)


