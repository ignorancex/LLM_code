
import os
# import warnings
# import glob

import torch

# from configuration import config, coc
from configuration import config, save_config_info

from splogic.seq2seq import learning as seq2seq_learning
# from splogic.seq2seq.data_read import make_data_loader
# from splogic.seq2seq.validation import validate
from splogic.seq2seq import filemng
# from splogic.seq2seq.filemng import optim_measures, search_measures

from splogic.utility.acceleration import accelerator
from splogic.utility.tqdm import utqdm
from splogic.utility.logging import logger

# from . import learning
# from . import filemng
# from .validation import validate
# from .filemng import optim_measures, search_measures

# from .data_read import make_data_loader

from dhnamlib.pylib import filesys
from dhnamlib.pylib.iteration import not_none_valued_dict
from dhnamlib.pylib.mllib.learning import get_init_status, update_status
# from dhnamlib.pylib.function import identity
# from dhnamlib.pylib.lazy import LazyProxy
from dhnamlib.pylib.hflib.acceleration import broadcast_object


# make_data_loader = LazyProxy(lambda: coc.domain.data_read.make_data_loader)


MODEL_SYMLINK_NAME = 'model'
# COLLECTION_SYMLINK_NAME = 'collection'
# LEARNING_SYMLINK_NAME = 'learning'


@config
def run_train(
        pretrained_model_name_or_path=config.ph,
        grammar=config.ph,
        # compiler=config.ph,
        # test_executor=config.ph,
        # dynamic_binder=config.ph,
        # device=config.ph,
        # logger=config.ph,
        test_validator=config.ph,
        encoded_train_set=config.ph,
        encoded_val_set=config.ph,
        train_batch_size=config.ph,
        val_batch_size=config.ph,
        train_max_num_action_seqs=config.ph(None),
        learning_rate=config.ph,
        adam_epsilon=config.ph,
        weight_decay=config.ph,
        num_train_epochs=config.ph,
        using_scheduler=config.ph,
        num_warmup_epochs=config.ph,
        max_grad_norm=config.ph,
        patience=config.ph,
        softmax_masking=config.ph,
        constrained_decoding=config.ph,
        using_arg_candidate=config.ph,
        using_distinctive_union_types=config.ph,
        model_learning_dir_path=config.ph,
        restarting=False,
        # context_creator=config.ph,
        # denotation_equal=config.ph,
        # result_collector_cls=config.ph,
        optim_measures=config.ph,
        num_prediction_beams=config.ph,
        generation_max_length=config.ph,
        saving_optimizer=config.ph,
        num_epoch_repeats=config.ph(1),
        weaksup_learning=config.ph(False),
        make_data_loader_fn=config.ph,
        file_manager=config.ph,
        # save_analysis_fn=config.ph,
        # save_extra_performance_fn=config.ph(filemng.save_extra_performance),
):
    if restarting:
        assert filemng.is_finetuned(pretrained_model_name_or_path)

        if model_learning_dir_path is None:
            model_learning_dir_path = filesys.get_parent_path(pretrained_model_name_or_path)
    else:
        assert model_learning_dir_path is not None

    save_config_info(model_learning_dir_path)

    last_dir_path = os.path.join(model_learning_dir_path, 'last')
    # mkloc_unless_exist_wlmp(last_dir_path)
    # make_symlink_wlmp(seq2seq_learning.get_new_checkpoint_path(model_learning_dir_path), last_dir_path)

    best_dir_path = os.path.join(model_learning_dir_path, 'best')
    # mkloc_unless_exist_wlmp(best_dir_path)
    # copy_symlink_wlmp(last_dir_path, best_dir_path)

    cpm = filemng.AcceleratedCheckpointManager(
        checkpoint_loc_path=os.path.join(model_learning_dir_path, 'checkpoint'),
        symlink_glob_patterns=[os.path.join(model_learning_dir_path, 'last', MODEL_SYMLINK_NAME),
                               os.path.join(model_learning_dir_path, 'best', MODEL_SYMLINK_NAME)])

    model = filemng.load_model(
        pretrained_model_name_or_path,
        num_tokens=len(grammar.lf_tokenizer))
    # model.to(device)
    model.to(accelerator.device)

    if weaksup_learning:
        assert train_max_num_action_seqs is not None

    train_data_loader = make_data_loader_fn(
        encoded_dataset=encoded_train_set,
        decoder_start_token_id=grammar.model_config.decoder_start_token_id,
        pad_token_id=grammar.lf_tokenizer.pad_token_id,
        batch_size=train_batch_size,
        shuffle=True,
        num_epoch_repeats=num_epoch_repeats,
        max_num_action_seqs=train_max_num_action_seqs,
    )
    val_data_loader = make_data_loader_fn(
        encoded_dataset=encoded_val_set,
        decoder_start_token_id=grammar.model_config.decoder_start_token_id,
        pad_token_id=grammar.lf_tokenizer.pad_token_id,
        batch_size=val_batch_size,
        shuffle=False,
    )

    param_groups = seq2seq_learning.get_param_groups(model, learning_rate=learning_rate, weight_decay=weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, eps=adam_epsilon)

    scheduler = seq2seq_learning.make_scheduler(
        optimizer=optimizer,
        using_scheduler=using_scheduler,
        train_data_loader=train_data_loader,
        num_train_epochs=num_train_epochs,
        num_warmup_epochs=num_warmup_epochs,
    )

    if restarting:
        # Question: what do optimizer and scheduler save?
        filemng.load_and_update_optimizer(optimizer, pretrained_model_name_or_path)
        filemng.load_and_update_scheduler(scheduler, pretrained_model_name_or_path)

        status = dict(filemng.load_status(pretrained_model_name_or_path))
    else:
        status = get_init_status(measures=optim_measures, update_unit='epoch')

    # The model and optimizer should be passed together to the prepare method
    # https://huggingface.co/docs/accelerate/quicktour#enable-distributed-training-in-your-script
    model, train_data_loader, optimizer, scheduler = accelerator.prepare(
        model, train_data_loader, optimizer, scheduler)

    val_data_loader = accelerator.prepare_val_data_loader(val_data_loader)

    remaining_patience = patience

    forward_backward = seq2seq_learning.ws_forward_backward if weaksup_learning else seq2seq_learning.ss_forward_backward

    # for epoch in range(status['last_update_num'] + 1, 3):  # DEBUG
    for epoch in range(status['last_update_num'] + 1, num_train_epochs + 1):
        logger.info(f'Epoch {epoch} starts')
        model.train()

        # print('---- Remove debug code ----')  # COMMENT THIS
        # debug_batch_idx = -1                  # COMMENT THIS
        # loss = torch.tensor(0.)
        loss = 0
        # for batch in coc.xtqdm(train_data_loader, desc_fn=lambda: 'loss: {:7.4f}'.format(loss.item())):
        for batch in utqdm(train_data_loader, unit='loss', update_fn=lambda: loss, repr_format='{:7.4f}'):
            # if debug_batch_idx > 100:  # COMMENT THIS
            #     break                  # COMMENT THIS
            # else:                      # COMMENT THIS
            #     debug_batch_idx += 1   # COMMENT THIS

            # - `model.config.decoder_start_token_id` is the first id of output sequences.
            # - the order or decoder output tokens in a sequence: decoder_start_token_id, bos_token_id, others ...

            optimizer.zero_grad()

            loss = forward_backward(**not_none_valued_dict(
                grammar=grammar,
                model=model,
                batch=batch,
                softmax_masking=None if weaksup_learning else softmax_masking,
                batch_size=train_batch_size,
            ))

            assert accelerator.sync_gradients
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

        accelerator.wait_for_everyone()

        model.eval()
        # from dhnamlib.pylib.cProfiling import run_context
        # run_context('''validation = validate( grammar=grammar, compiler=compiler, model=model, context_creator=context_creator, data_loader=val_data_loader, batch_size=val_batch_size, num_beams=num_prediction_beams, generation_max_length=generation_max_length)''', sort='cumtime')
        # validation = validate(
        #     grammar=grammar,
        #     compiler=compiler,
        #     executor=test_executor,
        #     dynamic_binder=dynamic_binder,
        #     model=model,
        #     context_creator=context_creator,
        #     data_loader=val_data_loader,
        #     batch_size=val_batch_size,
        #     num_beams=num_prediction_beams,
        #     generation_max_length=generation_max_length,
        #     softmax_masking=softmax_masking,
        #     constrained_decoding=constrained_decoding,
        #     using_arg_candidate=using_arg_candidate,
        #     using_distinctive_union_types=using_distinctive_union_types,
        #     evaluating=True,
        #     denotation_equal=denotation_equal,
        #     result_collector_cls=result_collector_cls,
        # )
        validation = test_validator.validate(
            grammar=grammar,
            model=model,
            data_loader=val_data_loader,
            batch_size=val_batch_size,
            num_beams=num_prediction_beams,
            generation_max_length=generation_max_length,
            softmax_masking=softmax_masking,
            constrained_decoding=constrained_decoding,
            using_arg_candidate=using_arg_candidate,
            using_distinctive_union_types=using_distinctive_union_types,
            evaluating=True,
        )

        performance = validation['performance']

        logger.info(f'Epoch: {epoch} / Performance: {str(performance)}')

        updating_best = update_status(status, performance=performance)

        new_checkpoint_dir_path = cpm.get_new_checkpoint_path()
        with filemng.prepare_dir(new_checkpoint_dir_path) as temp_checkpoint_dir_path:
            filemng.skip_if_not_wlmp(temp_checkpoint_dir_path)

            if saving_optimizer:
                filemng.save_optimizer(optimizer, temp_checkpoint_dir_path)
            filemng.save_scheduler(scheduler, temp_checkpoint_dir_path)
            filemng.save_model(model, temp_checkpoint_dir_path)

        with cpm.cleaning(filemng.replace_dir(last_dir_path)) as temp_last_dir_path:
            filemng.skip_if_not_wlmp(temp_last_dir_path)

            filemng.save_status(status, temp_last_dir_path)
            filemng.save_performance(performance, temp_last_dir_path)
            file_manager.save_extra_performance(validation.get('extra_performance', None), temp_last_dir_path)
            file_manager.save_analysis(validation['analysis'], temp_last_dir_path)

            filemng.make_symlink(
                new_checkpoint_dir_path,
                os.path.join(temp_last_dir_path, MODEL_SYMLINK_NAME))

        if updating_best:
            with cpm.cleaning(filemng.replace_dir(best_dir_path)) as temp_best_dir_path:
                filemng.skip_if_not_wlmp(temp_best_dir_path)

                filemng.copy_dir(last_dir_path, temp_best_dir_path)

            remaining_patience = patience

            logger.info('Best model is updated')
        else:
            remaining_patience -= 1
            if remaining_patience < 0:
                break

        logger.info(f'Results are saved in "{model_learning_dir_path}"')


@config
def run_train_for_multiple_decoding_strategies(
        pretrained_model_name_or_path=config.ph,
        grammar=config.ph,
        # compiler=config.ph,
        # test_executor=config.ph,
        # dynamic_binder=config.ph,
        # device=config.ph,
        # logger=config.ph,
        test_validator=config.ph,
        encoded_train_set=config.ph,
        encoded_val_set=config.ph,
        train_batch_size=config.ph,
        val_batch_size=config.ph,
        learning_rate=config.ph,
        adam_epsilon=config.ph,
        weight_decay=config.ph,
        num_train_epochs=config.ph,
        using_scheduler=config.ph,
        num_warmup_epochs=config.ph,
        max_grad_norm=config.ph,
        patience=config.ph,
        softmax_masking=config.ph,
        # constrained_decoding=config.ph,
        # using_arg_candidate=config.ph,
        model_learning_dir_path=config.ph,
        restarting=False,
        # context_creator=config.ph,
        # denotation_equal=config.ph,
        # result_collector_cls=config.ph,
        optim_measures=config.ph,
        num_prediction_beams=config.ph,
        generation_max_length=config.ph,
        saving_optimizer=config.ph,

        # Argument for multiple decoding strategies
        decoding_strategy_configs=config.ph,

        # Argument for limited size of training data
        num_epoch_repeats=config.ph(1),

        make_data_loader_fn=config.ph,
        # save_analysis_fn=config.ph,
        # save_extra_performance_fn=config.ph(filemng.save_extra_performance),
        file_manager=config.ph,
):
    assert model_learning_dir_path is not None
    assert patience == float('inf'), 'the feature of patience is not implemented'

    save_config_info(model_learning_dir_path)

    common_last_dir_path = os.path.join(model_learning_dir_path, 'common:last')

    def get_best_dir_path(decoding_strategy_name):
        return os.path.join(model_learning_dir_path, f'{decoding_strategy_name}:best')

    def get_last_dir_path(decoding_strategy_name):
        return os.path.join(model_learning_dir_path, f'{decoding_strategy_name}:last')

    cpm = filemng.AcceleratedCheckpointManager(
        checkpoint_loc_path=os.path.join(model_learning_dir_path, 'checkpoint'),
        symlink_glob_patterns=[os.path.join(model_learning_dir_path, '*:last', MODEL_SYMLINK_NAME),
                               os.path.join(model_learning_dir_path, '*:best', MODEL_SYMLINK_NAME)])

    model = filemng.load_model(
        pretrained_model_name_or_path,
        num_tokens=len(grammar.lf_tokenizer))
    model.to(accelerator.device)

    train_data_loader = make_data_loader_fn(
        encoded_dataset=encoded_train_set,
        decoder_start_token_id=grammar.model_config.decoder_start_token_id,
        pad_token_id=grammar.lf_tokenizer.pad_token_id,
        batch_size=train_batch_size,
        shuffle=True,
        num_epoch_repeats=num_epoch_repeats)
    val_data_loader = make_data_loader_fn(
        encoded_dataset=encoded_val_set,
        decoder_start_token_id=grammar.model_config.decoder_start_token_id,
        pad_token_id=grammar.lf_tokenizer.pad_token_id,
        batch_size=val_batch_size,
        shuffle=False)

    param_groups = seq2seq_learning.get_param_groups(model, learning_rate=learning_rate, weight_decay=weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, eps=adam_epsilon)

    scheduler = seq2seq_learning.make_scheduler(
        optimizer=optimizer,
        using_scheduler=using_scheduler,
        train_data_loader=train_data_loader,
        num_train_epochs=num_train_epochs,
        num_warmup_epochs=num_warmup_epochs,
    )

    assert not restarting, 'The feature of restarting is not implemented'

    status_dict = dict(
        [decoding_strategy_config.decoding_strategy_name,
         get_init_status(measures=optim_measures, update_unit='epoch')]
        for decoding_strategy_config in decoding_strategy_configs)

    # The model and optimizer should be passed together to the prepare method
    # https://huggingface.co/docs/accelerate/quicktour#enable-distributed-training-in-your-script
    model, train_data_loader, optimizer, scheduler = accelerator.prepare(
        model, train_data_loader, optimizer, scheduler)

    val_data_loader = accelerator.prepare_val_data_loader(val_data_loader)

    assert not softmax_masking, 'The feature of softmax_masking is not implemented'

    for epoch in range(1, num_train_epochs + 1):
        logger.info(f'Epoch {epoch} starts')
        model.train()

        # debug_batch_cnt = -1    # COMMENT THIS

        # loss = torch.tensor(0.)
        loss = 0
        # for batch in coc.xtqdm(train_data_loader, desc_fn=lambda: 'loss: {:7.4f}'.format(loss.item())):
        for batch in utqdm(train_data_loader, unit='loss', update_fn=lambda: loss, repr_format='{:7.4f}'):
            # debug_batch_cnt += 1       # COMMENT THIS
            # if debug_batch_cnt > 100:  # COMMENT THIS
            #     break                  # COMMENT THIS

            optimizer.zero_grad()

            loss = seq2seq_learning.ss_forward_backward(
                grammar=grammar,
                model=model,
                batch=batch,
                softmax_masking=softmax_masking,
                batch_size=train_batch_size,
            )

            assert accelerator.sync_gradients
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

        accelerator.wait_for_everyone()

        model.eval()
        last_model_saved = False

        for decoding_strategy_config in decoding_strategy_configs:
            with config.let(decoding_strategy_config.items()):
                logger.info(f'Validation of "{config.decoding_strategy_name}" starts')

                if 'grammar_lazy_obj' in decoding_strategy_config:
                    _grammar = decoding_strategy_config.grammar_lazy_obj.get()
                else:
                    _grammar = grammar

                validation = test_validator.validate(
                    grammar=_grammar,
                    model=model,
                    data_loader=val_data_loader,
                    batch_size=val_batch_size,
                    num_beams=num_prediction_beams,
                    generation_max_length=generation_max_length,
                    softmax_masking=softmax_masking,
                    constrained_decoding=config.constrained_decoding,
                    using_arg_candidate=config.using_arg_candidate,
                    using_distinctive_union_types=config.using_distinctive_union_types,
                    evaluating=True,
                )

                performance = validation['performance']

                logger.info(f'Decoding strategy: {config.decoding_strategy_name} / Performance: {str(performance)}')

                status = status_dict[config.decoding_strategy_name]

                updating_best = update_status(status, performance=performance)

                if not last_model_saved:
                    last_model_saved = True

                    # save a model
                    new_checkpoint_dir_path = cpm.get_new_checkpoint_path()
                    with filemng.prepare_dir(new_checkpoint_dir_path) as temp_checkpoint_dir_path:
                        filemng.skip_if_not_wlmp(temp_checkpoint_dir_path)

                        if saving_optimizer:
                            filemng.save_optimizer(optimizer, temp_checkpoint_dir_path)
                        filemng.save_scheduler(scheduler, temp_checkpoint_dir_path)
                        filemng.save_model(model, temp_checkpoint_dir_path)

                    with cpm.cleaning(filemng.replace_dir(common_last_dir_path)) as temp_common_last_dir_path:
                        filemng.skip_if_not_wlmp(temp_common_last_dir_path)

                        filemng.make_symlink(
                            new_checkpoint_dir_path,
                            os.path.join(temp_common_last_dir_path, MODEL_SYMLINK_NAME))

                # save for a decoding strategy
                strategy_last_dir_path = os.path.join(model_learning_dir_path, f'{config.decoding_strategy_name}:last')
                with cpm.cleaning(filemng.replace_dir(strategy_last_dir_path)) as temp_strategy_last_dir_path:
                    filemng.skip_if_not_wlmp(temp_strategy_last_dir_path)

                    filemng.save_status(status, temp_strategy_last_dir_path)
                    filemng.save_performance(performance, temp_strategy_last_dir_path)
                    file_manager.save_extra_performance(validation.get('extra_performance', None), temp_strategy_last_dir_path)
                    file_manager.save_analysis(validation['analysis'], temp_strategy_last_dir_path)

                    filemng.copy_symlink(os.path.join(common_last_dir_path, MODEL_SYMLINK_NAME),
                                         os.path.join(temp_strategy_last_dir_path, MODEL_SYMLINK_NAME))

                if updating_best:
                    strategy_best_dir_path = os.path.join(model_learning_dir_path, f'{config.decoding_strategy_name}:best')
                    with cpm.cleaning(filemng.replace_dir(strategy_best_dir_path)) as temp_strategy_best_dir_path:
                        filemng.skip_if_not_wlmp(temp_strategy_best_dir_path)

                        filemng.copy_dir(strategy_last_dir_path, temp_strategy_best_dir_path)

                    logger.info('Best model is updated')

                logger.info(f'Results are saved in "{model_learning_dir_path}"')


@config
def run_test(
        *,
        # model_learning_dir_path=config.ph(None),
        model_path=config.ph(None),
        # model_dir_name=config.ph('best'),
        test_dir_path=config.ph,
        grammar=config.ph,
        # compiler=config.ph,
        # test_executor=config.ph,
        # dynamic_binder=config.ph,
        # device=config.ph,
        # logger=config.ph,
        test_validator=config.ph,
        encoded_test_set,
        test_batch_size=config.ph,
        softmax_masking=config.ph,
        constrained_decoding=config.ph,
        using_arg_candidate=config.ph,
        using_distinctive_union_types=config.ph,
        # context_creator=config.ph,
        # denotation_equal=config.ph,
        # result_collector_cls=config.ph,
        num_prediction_beams=config.ph,
        generation_max_length=config.ph,
        evaluating,
        analyzing=True,
        using_oracle=False,
        make_data_loader_fn=config.ph,
        # save_analysis_fn=config.ph,
        # save_extra_performance_fn=config.ph(filemng.save_extra_performance),
        file_manager=config.ph,
):
    # if model_path is None:
    #     # Check the default checkpoint path
    #     assert model_learning_dir_path is not None
    #     assert os.path.isdir(model_learning_dir_path), f'The learning directory {model_learning_dir_path} does not exist'
    #     model_path = os.path.join(model_learning_dir_path, model_dir_name, MODEL_SYMLINK_NAME)
    #     assert os.path.isdir(model_path), f'The checkpoint {model_path} does not exist. Specify `model_path`.'
    model = filemng.load_model(
        model_path,
        num_tokens=len(grammar.lf_tokenizer))
    model.to(accelerator.device)

    test_data_loader = make_data_loader_fn(
        encoded_dataset=encoded_test_set,
        decoder_start_token_id=grammar.model_config.decoder_start_token_id,
        pad_token_id=grammar.lf_tokenizer.pad_token_id,
        batch_size=test_batch_size,
        shuffle=False)

    model = accelerator.prepare(model)
    test_data_loader = accelerator.prepare_val_data_loader(test_data_loader)

    model.eval()
    validation = test_validator.validate(
        grammar=grammar,
        model=model,
        data_loader=test_data_loader,
        batch_size=test_batch_size,
        num_beams=num_prediction_beams,
        generation_max_length=generation_max_length,
        analyzing=analyzing,
        evaluating=evaluating,
        softmax_masking=softmax_masking,
        constrained_decoding=constrained_decoding,
        using_arg_candidate=using_arg_candidate,
        using_distinctive_union_types=using_distinctive_union_types,
        using_oracle=using_oracle,
    )

    if evaluating:
        logger.info('Performance: {}'.format(validation['performance']))

    filemng.mkloc_unless_exist(test_dir_path)
    save_config_info(test_dir_path)

    if analyzing:
        file_manager.save_analysis(validation['analysis'], test_dir_path)
    filemng.save_predictions(validation['predictions'], test_dir_path)
    if evaluating:
        filemng.save_performance(validation['performance'], test_dir_path)
        file_manager.save_extra_performance(validation.get('extra_performance', None), test_dir_path)
    filemng.save_time_info(validation['time_info'], test_dir_path)

    logger.info(f'Results are saved in "{test_dir_path}"')


# @config
def run_oracle_test(
        *,
        # model_learning_dir_path=config.ph(None),
        # model_path=config.ph(None),
        # test_dir_path=config.ph,
        # grammar=config.ph,
        # compiler=config.ph,
        # device=config.ph,
        # logger=config.ph,
        encoded_test_set,
        # test_batch_size=config.ph,
        # softmax_masking=config.ph,
        # constrained_decoding=config.ph,
        # using_arg_candidate=config.ph,
        # context_creator=config.ph,
        # num_prediction_beams=config.ph,
        # generation_max_length=config.ph,
        evaluating,
        analyzing=False,
):
    run_test(
        # model_learning_dir_path=model_learning_dir_path,
        # model_path=model_path,
        # test_dir_path=test_dir_path,
        # grammar=grammar,
        # compiler=compiler,
        # device=device,
        # logger=logger,
        encoded_test_set=encoded_test_set,
        # test_batch_size=test_batch_size,
        # softmax_masking=softmax_masking,
        # constrained_decoding=constrained_decoding,
        # using_arg_candidate=using_arg_candidate,
        # context_creator=context_creator,
        # num_prediction_beams=num_prediction_beams,
        # generation_max_length=generation_max_length,
        evaluating=evaluating,
        analyzing=analyzing,
        using_oracle=True,
    )


@config
def run_search_train(
        pretrained_model_path=config.ph,
        grammar=config.ph,
        # compiler=config.ph,
        # search_executor=config.ph,
        # dynamic_binder=config.ph,
        # # devices=config.ph,
        # logger=config.ph,
        # encoded_train_set=config.ph,
        search_validator=config.ph,
        test_validator=config.ph,
        encoded_val_set=config.ph,
        encoded_weaksup_search_set=config.ph,  # New
        train_batch_size=config.ph,
        val_batch_size=config.ph,
        search_batch_size=config.ph,  # New
        learning_rate=config.ph,
        adam_epsilon=config.ph,
        weight_decay=config.ph,
        num_train_epochs=config.ph,
        using_scheduler=config.ph,
        num_warmup_epochs=config.ph,
        max_grad_norm=config.ph,
        patience=config.ph,
        softmax_masking=config.ph,
        constrained_decoding=config.ph,
        using_arg_candidate=config.ph,
        using_distinctive_union_types=config.ph,
        extra_weaksup_collection_keys=config.ph([]),
        model_learning_dir_path=config.ph,
        resuming=config.ph(None),         # New
        # # max_num_iterations=None,  # New
        # context_creator=config.ph,
        # denotation_equal=config.ph,
        # result_collector_cls=config.ph,
        optim_measures=config.ph,
        search_measures=config.ph,
        num_prediction_beams=config.ph,
        num_search_beams=config.ph,  # New
        generation_max_length=config.ph,
        # # saving_optimizer=config.ph,
        max_search_optim_loops=config.ph(float('inf')),  # New
        make_data_loader_fn=config.ph,
        # save_extra_performance_fn=config.ph(filemng.save_extra_performance),
        file_manager=config.ph,
):
    if resuming:
        logger.info(f'Learning resumes with the directory "{model_learning_dir_path}"')

    search_dir_path = os.path.join(model_learning_dir_path, 'search')
    optim_dir_path = os.path.join(model_learning_dir_path, 'optim')

    filesys.asserts_conditional_exist(
        os.path.join(search_dir_path, 'last'), resuming,
        f'When resuming == True, {model_learning_dir_path} should contain previous result of learning',
        f'When resuming == False, {model_learning_dir_path} should not contain any previous result of learning',
    )

    search_cpm = filemng.AcceleratedCheckpointManager(
        checkpoint_loc_path=os.path.join(search_dir_path, 'checkpoint'),
        symlink_glob_patterns=[os.path.join(search_dir_path, 'last'),
                               os.path.join(search_dir_path, 'best')])
    optim_cpm = filemng.AcceleratedCheckpointManager(
        checkpoint_loc_path=os.path.join(optim_dir_path, 'checkpoint'),
        symlink_glob_patterns=[os.path.join(optim_dir_path, 'last'),
                               os.path.join(optim_dir_path, 'best')])

    def get_init_search_status():
        return get_init_status(measures=search_measures, update_unit='search')

    def get_init_optim_status():
        return get_init_status(measures=optim_measures, update_unit='optimization')

    if resuming:
        search_status = filemng.load_status(os.path.join(search_dir_path, 'last'), default=get_init_search_status())
        optim_status = filemng.load_status(os.path.join(optim_dir_path, 'last'), default=get_init_optim_status())

        # assert search_status['last_update_num'] == search_status['best_update_num']  # early stopping condition
        # assert optim_status['last_update_num'] == optim_status['best_update_num']  # early stopping condition

        assert search_status['last_update_num'] - 1 <= optim_status['last_update_num'] <= search_status['last_update_num']
        optim_first = optim_status['last_update_num'] + 1 == search_status['last_update_num']
    else:
        search_status = get_init_search_status()
        optim_status = get_init_optim_status()
        optim_first = False

    weaksup_search_data_loader = accelerator.prepare_val_data_loader(make_data_loader_fn(
        encoded_dataset=encoded_weaksup_search_set,
        decoder_start_token_id=grammar.model_config.decoder_start_token_id,
        pad_token_id=grammar.lf_tokenizer.pad_token_id,
        batch_size=search_batch_size,
        shuffle=False,
    ))

    def get_latest_model_path():
        if optim_status['last_update_num'] > 0:
            return os.path.join(optim_dir_path, 'last', 'best', MODEL_SYMLINK_NAME)
        else:
            return filesys.asserts_exist(pretrained_model_path)

    def load_latest_model():
        model = filemng.load_model(get_latest_model_path(), num_tokens=len(grammar.lf_tokenizer))
        model.to(accelerator.device)
        model = accelerator.prepare(model)
        return model

    def load_latest_encoded_weaksup_set():
        return filemng.load_weaksup_dataset(os.path.join(search_dir_path, 'last'))

    def run_search():
        accelerator.wait_for_everyone()  # before loading model
        model = load_latest_model()
        model.eval()

        logger.info('Search starts')

        validation = search_validator.validate(
            grammar=grammar,
            model=model,
            data_loader=weaksup_search_data_loader,
            batch_size=search_batch_size,
            num_beams=num_search_beams,
            generation_max_length=generation_max_length,
            analyzing=False,
            softmax_masking=softmax_masking,
            constrained_decoding=constrained_decoding,
            using_arg_candidate=using_arg_candidate,
            using_distinctive_union_types=using_distinctive_union_types,
            evaluating=True,
            using_oracle=True,
            collecting_weaksup_examples=True,
            extra_weaksup_collection_keys=extra_weaksup_collection_keys,
            # strict_postprocessing=True,
        )

        performance = validation['performance']
        updating_best = update_status(search_status, performance=performance)

        logger.info(f'Search update number: {search_status["last_update_num"]} / Performance: {str(performance)}')

        new_checkpoint_dir_path = search_cpm.get_new_checkpoint_path()
        with filemng.prepare_dir(new_checkpoint_dir_path) as temp_checkpoint_dir_path:
            filemng.skip_if_not_wlmp(temp_checkpoint_dir_path)

            filemng.save_status(search_status, temp_checkpoint_dir_path)
            filemng.save_performance(performance, temp_checkpoint_dir_path)
            file_manager.save_extra_performance(validation.get('extra_performance', None), temp_checkpoint_dir_path)
            filemng.save_weaksup_dataset(validation['weaksup_examples'], temp_checkpoint_dir_path)
            filemng.save_time_info(validation['time_info'], temp_checkpoint_dir_path)
            filemng.save_predictions(validation['predictions'], temp_checkpoint_dir_path)

        last_dir_path = os.path.join(search_dir_path, 'last')
        filemng.change_symlink(new_checkpoint_dir_path, last_dir_path)
        search_cpm.clean()

        if updating_best:
            best_dir_path = os.path.join(search_dir_path, 'best')
            filemng.copy_symlink(last_dir_path, best_dir_path)
            search_cpm.clean()

            logger.info('Best search result is updated')

        return updating_best

    def run_optim():
        accelerator.wait_for_everyone()  # before loading encoded_weaksup_set
        encoded_weaksup_set = load_latest_encoded_weaksup_set()

        new_checkpoint_dir_path = broadcast_object(optim_cpm.get_new_checkpoint_path())

        # checkpoint_loc_path = os.path.join(optim_dir_path, 'checkpoint')
        # filemng.mkloc_unless_exist(checkpoint_loc_path)
        # temp_checkpoint_dir_path = broadcast_object(filemng.mkdtemp(dir=checkpoint_loc_path))

        logger.info('Optimization starts')

        run_train(
            pretrained_model_name_or_path=get_latest_model_path(),
            grammar=grammar,
            # logger=logger,
            test_validator=test_validator,
            encoded_train_set=encoded_weaksup_set,
            encoded_val_set=encoded_val_set,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            learning_rate=learning_rate,
            adam_epsilon=adam_epsilon,
            weight_decay=weight_decay,
            num_train_epochs=num_train_epochs,
            using_scheduler=using_scheduler,
            num_warmup_epochs=num_warmup_epochs,
            max_grad_norm=max_grad_norm,
            patience=patience,
            softmax_masking=softmax_masking,
            constrained_decoding=constrained_decoding,
            using_arg_candidate=using_arg_candidate,
            using_distinctive_union_types=using_distinctive_union_types,
            model_learning_dir_path=new_checkpoint_dir_path,
            restarting=False,
            num_prediction_beams=num_prediction_beams,
            generation_max_length=generation_max_length,
            saving_optimizer=False,
            weaksup_learning=True,
        )

        # new_checkpoint_dir_path = optim_cpm.get_new_checkpoint_path()
        # filemng.rename_dir(temp_checkpoint_dir_path, new_checkpoint_dir_path)

        accelerator.wait_for_everyone()  # before loading performance

        performance = filemng.load_performance(os.path.join(new_checkpoint_dir_path, 'best'))
        updating_best = update_status(optim_status, performance=performance)

        filemng.save_status(optim_status, new_checkpoint_dir_path)  # save the status after update

        last_dir_path = os.path.join(optim_dir_path, 'last')
        filemng.change_symlink(new_checkpoint_dir_path, last_dir_path)
        optim_cpm.clean()

        logger.info(f'Optimization update number: {optim_status["last_update_num"]} / Performance: {str(performance)}')

        if updating_best:
            best_dir_path = os.path.join(optim_dir_path, 'best')
            filemng.copy_symlink(last_dir_path, best_dir_path)
            optim_cpm.clean()

            logger.info('Best optimization result is updated')

        return updating_best

    if optim_first:
        optim_updating_best = run_optim()
    else:
        optim_updating_best = True

    while optim_status['last_update_num'] < max_search_optim_loops:
        # Search
        search_updating_best = run_search()

        # if not search_updating_best:
        #     logger.info('Early stopping after search')
        #     break

        # Optimization
        optim_updating_best = run_optim()

        # if not optim_updating_best:
        #     logger.info('Early stopping after optimization')
        #     break
    else:
        logger.info('Maximum number of loops for search and optimization')


if __name__ == '__main__':
    if config.run_mode == 'train-default':
        run_train()
        # from dhnamlib.pylib.cProfiling import run_context
        # run_context('run_train(model_learning_dir_path=config.model_learning_dir_path)', sort='cumtime')
    elif config.run_mode == 'train-for-multiple-decoding-strategies':
        run_train_for_multiple_decoding_strategies()
    elif config.run_mode == 'test-on-val-set':
        run_test(encoded_test_set=config.encoded_val_set, evaluating=True)
        # from dhnamlib.pylib.cProfiling import run_context
        # run_context('run_test(encoded_test_set=config.encoded_val_set, evaluating=True)', sort='cumtime')
    elif config.run_mode == 'search-train':
        run_search_train()
    elif config.run_mode == 'test-on-test-set':
        run_test(encoded_test_set=config.encoded_test_set, evaluating=config.evaluating_test_set)
    elif config.run_mode == 'oracle-test-on-val-set':
        run_oracle_test(encoded_test_set=config.encoded_val_set, evaluating=True)
    else:
        raise Exception(f'Unknown execution type "{config.run_mode}"')

    # from dhnamlib.pylib.cProfiling import run_context
    # tuple(config.items(lazy=False))
    # run_context('run_train(model_learning_dir_path=config.model_learning_dir_path)', sort='cumtime')
