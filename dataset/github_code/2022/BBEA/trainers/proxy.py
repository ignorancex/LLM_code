from xoa.commons.logger import *

try:
    from trainers.emul.arch_builder import NAS101Builder
    from trainers.emul.nas_trainer import NAS101Emulator
    from trainers.emul.nas_etr_trainer import NAS101ETREmulator

except ImportError as ie:
    warn("Import error for NAS-101-Benchmark: {}".format(ie))

try:
    from trainers.emul.nas_trainer import NAS201Emulator
    from trainers.emul.nas_etr_trainer import NAS201ETREmulator

except ImportError as ie:
    warn("Import error for NAS-201-Benchmark: {}".format(ie))



def create_verifier(surrogate, run_config):
    if 'NAS-Bench-101' in surrogate:
        builder = get_nas_builder(surrogate, run_config)
        return builder
    else:
        # Not supported other BM yet
        warn("Not supported surrogate BM to verify: {}".format(surrogate))
        return None


def get_nas_builder(surrogate, run_config, dataset='ImageNet16-120'):
    if surrogate == 'NAS-Bench-101':    
        if "data_path" in run_config:
            data_path = run_config["data_path"]
        else:
            data_path = 'lookup/nasbench_full.tfrecord'

        return NAS101Builder(data_path, surrogate)
    elif surrogate == 'NAS-Bench-201':
        from lookup.nas201bench.api import NAS201Bench
        return NAS201Bench(dataset) 
    else:
        raise ValueError('Not supported surrogate: {}'.format(surrogate))


def get_nas_emulator(bench_type, builder, space, run_config):
    
    if "min_train_epoch" in run_config:
        min_epoch = run_config["min_train_epoch"]
    else:
        min_epoch = 108

    if "worst_error" in run_config and type(run_config["worst_error"]) == float:
        worst_error = run_config["worst_error"]
    else:
        worst_error = None

    report_mean_test_acc = False
    if "report_mean_test_acc" in run_config:
        report_mean_test_acc = run_config["report_mean_test_acc"]

    if bench_type == 101:
        epoch_budgets = [4, 12, 36, 108]
        if "early_term_rule" in run_config:
            etr = run_config["early_term_rule"]
            if etr == "DecaTercet":
                return NAS101ETREmulator(builder, space, 0.1, epoch_budgets, worst_error, report_mean_test_acc)        
            elif etr == "PentaTercet":
                return NAS101ETREmulator(builder, space, 0.2, epoch_budgets, worst_error, report_mean_test_acc) 
            elif etr == "TetraTercet":
                return NAS101ETREmulator(builder, space, 0.25, epoch_budgets, worst_error, report_mean_test_acc)  
            elif etr != 'None':
                warn("ETR {} is not implemented for NAS101-benchmark".format(etr))

        return NAS101Emulator(builder, space, min_epoch, worst_error, report_mean_test_acc)
    
    elif bench_type == 201:
        epoch_budgets = [i * 10 for i in range(1, 21) ]
        if "early_term_rule" in run_config:
            etr = run_config["early_term_rule"]
            if etr == "DecaTercet":
                return NAS201ETREmulator(builder, space, 0.1, epoch_budgets, worst_error, report_mean_test_acc)        
            elif etr == "PentaTercet":
                return NAS201ETREmulator(builder, space, 0.2, epoch_budgets, worst_error, report_mean_test_acc) 
            elif etr == "TetraTercet":
                return NAS201ETREmulator(builder, space, 0.25, epoch_budgets, worst_error, report_mean_test_acc)  
            elif etr != 'None':
                warn("ETR {} is not implemented for NAS201-benchmark".format(etr))
        return NAS201Emulator(builder, space, min_epoch, worst_error, report_mean_test_acc)
    
    else:
        raise ValueError("Not supported benchmark type: {}".format(bench_type))


def get_fcnet_emulator(space, run_config, config_type):


    try:
        from trainers.emul.fcnet_trainer import TabularFCNetTrainEmulator
        from trainers.emul.fcnet_etr_trainer import TabularFCNetETREmulator    

    except ImportError as ie:
        raise ModuleNotFoundError("Import error for HPO-Benchmark: {}".format(ie))

    if "min_train_epoch" in run_config:
        min_epoch = run_config["min_train_epoch"]
    else:
        min_epoch = 100

    if "worst_error" in run_config and type(run_config["worst_error"]) == float:
        worst_error = run_config["worst_error"]
    else:
        worst_error = None

    if "data_path" in run_config:
        data_path = run_config["data_path"]
    else:
        data_path = 'lookup/fcnet_tabular_benchmarks/'

    if "early_term_rule" in run_config:
        etr = run_config["early_term_rule"]
        if etr == "DecaTercet":
            return TabularFCNetETREmulator(space, data_path, 0.1, worst_error, config_type)        
        elif etr == "PentaTercet":
            return TabularFCNetETREmulator(space, data_path, 0.2, worst_error, config_type) 
        elif etr == "TetraTercet":
            return TabularFCNetETREmulator(space, data_path, 0.25, worst_error, config_type)  
        elif etr != 'None':
            warn("ETR {} is not implemented for NAS benchmark".format(etr))

    return TabularFCNetTrainEmulator(space, data_path, min_epoch, worst_error, config_type)


def get_dnn_train_simulator(space, run_config):
    
    etr = None
    if run_config and "early_term_rule" in run_config:
        etr = run_config["early_term_rule"]

    expired_time = None
    if run_config and "warm_up_time" in run_config:
        expired_time = run_config["warm_up_time"]

    from trainers.emul.trainer import TrainEmulator
    from trainers.emul.threshold_etr import CompoundETRTrainer     

    if not hasattr(space, 'lookup'):
        raise ValueError("Invalid surrogate space")
    lookup = space.lookup
            
    if etr == None or etr == "None":
        return TrainEmulator(lookup)
    elif etr == "DecaTercet":
        return CompoundETRTrainer(lookup, 0.1)
    elif etr == "PentaTercet":
        return CompoundETRTrainer(lookup, 0.2) 
    elif etr == "TetraTercet":
        return CompoundETRTrainer(lookup, 0.25)
    else:
        debug("Invalid ETR: {}. Use no ETR instread".format(etr))
        return TrainEmulator(lookup)
