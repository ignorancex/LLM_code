from .bandit import HPOBanditMachine


def create_emulator(space,
                    run_mode, target_val, time_expired,
                    goal_metric="error", 
                    run_config=None,
                    save_internal=False,
                    num_resume=0,
                    id="Emulator"):

    from trainers import get_dnn_train_simulator
    t = get_dnn_train_simulator(space, run_config)

    if run_config != None and "early_term_rule" in run_config:
        id = "{}.ETR-{}".format(id, run_config["early_term_rule"]) 
    if not 'benchmark_mode' in run_config: 
        run_config['benchmark_mode'] = True # Set as benchmark mode
    
    machine = HPOBanditMachine(space, t, 
                               run_mode, target_val, time_expired, run_config, 
                               goal_metric=goal_metric,
                               num_resume=num_resume, 
                               save_internal=save_internal,
                               min_train_epoch=t.get_min_train_epoch(),
                               id=id)
    return machine


def create_runner(space, trainer, 
                  run_mode, target_val, time_expired, 
                  run_config, hp_config,
                  goal_metric="error",
                  save_internal=False,
                  num_resume=0,
                  early_term_rule="DecaTercet",
                  id="Runner"
                  ):
    if run_config and "early_term_rule" in run_config:
        early_term_rule = run_config["early_term_rule"]
        if early_term_rule != "None":
            id = "{}.ETR-{}".format(id, early_term_rule)        
    

    machine = HPOBanditMachine(space, trainer, 
                                run_mode, target_val, time_expired, run_config,
                                goal_metric=goal_metric,
                                num_resume=num_resume, 
                                save_internal=save_internal,
                                id=id)
    
    return machine


