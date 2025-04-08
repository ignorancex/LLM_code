import logging
import os
import pickle
import shutil
import sys
from argparse import ArgumentError
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import MethodType
from typing import Optional

import evaluate.config
import torch
import transformers
from torch.distributed.elastic.multiprocessing.errors import record
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, CLIPFeatureExtractor, AutoTokenizer, \
    DataCollatorForLanguageModeling, BertConfig, BertModel, AutoModelForCausalLM, CLIPModel, CLIPVisionConfig
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers import set_seed
from transformers.integrations import WandbCallback
from transformers.trainer_utils import get_last_checkpoint, SchedulerType

from custom_trainer import CustomTrainer, CustomWandbCallback, CustomDeleteCheckpointsCallback, GlobalStepCallback
from dataset import create_dataset
from evaluation.cider.cider_huggingface import Cider
from models.custom_clip_vision_transformer.custom_clip_vision_transformer import CustomCLIPVisionTransformer
from models.memory_gpt2.configuration_memory_gpt2 import MemoryGPT2Config
from models.memory_gpt2.modeling_memory_gpt2 import MemoryGPT2LMHeadModel
from models.vision_encoder_decoder import VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from models.vision_encoder_encoder import VisionEncoderEncoderModel, VisionEncoderEncoderConfig
from utils import process_sample, collate_fn, compute_metrics, postprocess_text, _add_eos, \
    generate_with_backpropagation, is_main_process, load_model, save_predictions

logger = logging.getLogger(__name__)
is_debug = 'pydevd' in sys.modules


@dataclass
class VisionBackboneArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type." },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@record
def main():
    parser = HfArgumentParser((VisionBackboneArguments, Seq2SeqTrainingArguments))
    # model
    parser.add_argument('--encoder', action='store_true')
    parser.add_argument('--n_layer', type=int)
    parser.add_argument('--n_embd', type=int)
    parser.add_argument('--n_head', type=int)

    # training
    parser.add_argument('--custom_checkpoint_keeper', type=int, default=5)
    parser.add_argument('--scst', action='store_true')
    parser.add_argument('--force_arguments', action='store_true')

    # dataset
    parser.add_argument('--dataset_rate', type=float, default=1.0)
    parser.add_argument('--dataset_machine_rate', type=float, default=0.0)
    parser.add_argument('--train_datasets', type=str, nargs='+', default=['coco'])
    parser.add_argument('--validation_datasets', type=str, nargs='+', default=['coco_validation_dict'])
    parser.add_argument('--test_datasets', type=str, nargs='+', default=['coco_test_dict'])
    parser.add_argument('--scst_datasets', type=str, nargs='+', default=['coco_training_dict'])

    # scheduler
    parser.add_argument('--custom_lr_scheduler', type=str, choices=['CustomCosineScheduler', 'CustomScheduler', 'TransformerScheduler'],
                        default=None)
    parser.add_argument('--lr_multiplier', default=1.0, type=float)
    # CustomScheduler parameters
    parser.add_argument('--steps_min', default=None, type=int)
    parser.add_argument('--lr_min', default=0.0, type=float)
    parser.add_argument('--start_decreasing_steps', default=0, type=int)

    # memory
    parser.add_argument('--add_memory_slots_selfattn', action='store_true')
    parser.add_argument('--n_memory_slots', type=int, default=64)
    parser.add_argument('--freeze_memory', action='store_true')
    # kmeans
    parser.add_argument('--kmeans_memory', action='store_true')
    parser.add_argument('--deque_iters', type=int, default=10)
    parser.add_argument('--window', type=float, default=None)

    # parse model config
    default_config = transformers.GPT2Config()
    for k, v in default_config.to_dict().items():
        try:
            parser.add_argument(f'--{k}', default=v, type=type(v))
        except ArgumentError:  # this argument has been already added in the parser
            continue

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        vision_backbone_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        vision_backbone_args, training_args, _ = parser.parse_args_into_dataclasses()
    _, _, custom_args = parser.parse_args_into_dataclasses()
    custom_args_dict = {k: v for k, v in vars(custom_args).items() if k not in default_config.to_dict().keys()}

    # set default values for some arguments
    training_args.remove_unused_columns = False
    training_args.ignore_data_skip = True
    training_args.log_on_each_node = False
    training_args.dataloader_num_workers = 0
    training_args.fp16 = True
    training_args.fp16_full_eval = True

    # we do not need max_steps to be specified if:
    # - we are not training
    # - we are training indefinitely in XE phase with custom lr scheduler
    # - we are in SCST phase
    if not training_args.do_train or custom_args.custom_lr_scheduler or custom_args.scst:
        training_args.max_steps = sys.maxsize if not is_debug else 10

    # Setup logging
    logging.basicConfig(
        format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if custom_args.scst and training_args.lr_scheduler_type != SchedulerType.CONSTANT:
        logger.warning('`lr_scheduler_type` must be `constant` when training with SCST! Setting to `constant`...')
        training_args.lr_scheduler_type = SchedulerType.CONSTANT


    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Custom parameters {dict(sorted(custom_args_dict.items()))}")

    set_seed(training_args.seed)

    # register the models
    AutoConfig.register("clip_vision_model", CLIPVisionConfig)
    AutoModel.register(CLIPVisionConfig, CustomCLIPVisionTransformer)

    AutoConfig.register("vision-encoder-encoder", VisionEncoderEncoderConfig)
    AutoModel.register(VisionEncoderEncoderConfig, VisionEncoderEncoderModel)
    AutoConfig.register("memory_gpt2", MemoryGPT2Config)
    AutoModelForCausalLM.register(MemoryGPT2Config, MemoryGPT2LMHeadModel)

    # get the CLIP tokenizer and feature extractor
    clip_model_name = "openai/clip-vit-large-patch14"
    tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    feature_extractor = CLIPFeatureExtractor.from_pretrained(clip_model_name)

    text_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors='pt')

    eval_datasets = list()
    eval_datasets.extend(custom_args.validation_datasets)
    eval_datasets.extend(custom_args.test_datasets)
    
    no_gts = False
    # get vision encoder model
    dtype = torch.float16 if training_args.fp16 else torch.float32

    
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    clip_model.vision_model.main_input_name = "pixel_values"
    vision_encoder = clip_model.vision_model
        

    # create partial functions
    partial_process_sample = partial(process_sample, tokenizer=tokenizer, feature_extractor=feature_extractor)
    partial_collate_fn = partial(collate_fn, tokenizer=tokenizer, text_collator=text_collator)
    partial_compute_metrics = partial(compute_metrics, tokenizer=tokenizer, do_train=training_args.do_train, no_gts=no_gts)

    # remove files if `overwrite_output_dir` is True
    if training_args.overwrite_output_dir and Path(training_args.output_dir).is_dir() and \
            any(Path(training_args.output_dir).iterdir()):
        logger.warning(
            f"Output directory ({training_args.output_dir}) already exists and is not empty, deleting files..."
        )
        shutil.rmtree(training_args.output_dir, ignore_errors=True)

    # detect last checkpoint
    last_checkpoint = None
    if Path(training_args.output_dir).is_dir() and not training_args.overwrite_output_dir:  # and training_args.do_train
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and any(Path(training_args.output_dir).iterdir()):
            logger.warning(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # detect if it is the first SCST run
    is_first_scst_run = False
    if custom_args.scst and last_checkpoint is None:
        is_first_scst_run = True

    # get the checkpoint
    checkpoint = None
    if training_args.do_train:
        if custom_args.scst:
            if is_first_scst_run:
                if training_args.resume_from_checkpoint:
                    checkpoint = training_args.resume_from_checkpoint
                else:
                    raise ValueError('First SCST run, you must specify a checkpoint using `--resume_from_checkpoint`!')
            else:
                if last_checkpoint is not None:
                    checkpoint = last_checkpoint
                else:
                    raise ValueError('Last SCST checkpoint not found!')
        else:  # XE phase
            if training_args.resume_from_checkpoint:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            else:
                logger.info('No checkpoints found, training from scratch.')
    elif not training_args.do_train:  # evaluation
        if training_args.resume_from_checkpoint:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        else:
            raise ValueError('No checkpoints found for evaluation!')

    optimizer, scheduler = None, None

    # load model
    if checkpoint:
        model = load_model(checkpoint, custom_args_dict, dtype, force=is_first_scst_run or custom_args.force_arguments)
        optimizer = transformers.optimization.AdamW(model.decoder.parameters(), lr=training_args.learning_rate)
    else:  # create from scratch   
        logger.info('Using GPT-2 with memory.')
        gpt2_config = 'configs/memory_gpt2/config.json'
        custom_configuration = MemoryGPT2Config.from_pretrained(gpt2_config)
        custom_configuration.n_layer = custom_args.n_layer or custom_configuration.n_layer
        custom_configuration.n_embd = custom_args.n_embd or custom_configuration.n_embd
        custom_configuration.n_head = custom_args.n_head or custom_configuration.n_head
        encoder_common_keys = set(custom_configuration.to_dict().keys()) & set(custom_args_dict.keys())
        for key in encoder_common_keys:
            setattr(custom_configuration, key, custom_args_dict[key])
        gpt2_decoder = MemoryGPT2LMHeadModel(custom_configuration)
        
        if custom_args.encoder:
            config_encoder = BertConfig(hidden_size=custom_configuration.n_embd, d_embed=custom_configuration.n_embd,
                                        num_hidden_layers=custom_configuration.n_layer, intermediate_size=2048,
                                        num_attention_heads=custom_configuration.n_head)
            transformer_encoder = BertModel(config=config_encoder)

            encoder = VisionEncoderEncoderModel(vision_encoder=vision_encoder, encoder=transformer_encoder)

            model_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config=encoder.config,
                                                                                   decoder_config=gpt2_decoder.config)
            model = VisionEncoderDecoderModel(config=model_config, encoder=encoder, decoder=gpt2_decoder)
        else:
            model_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config=vision_encoder.config,
                                                                                   decoder_config=gpt2_decoder.config)
            model = VisionEncoderDecoderModel(config=model_config, encoder=vision_encoder, decoder=gpt2_decoder)

        model.config.add_cross_attention = True
        model.config.is_encoder_decoder = True
        model.config.decoder_start_token_id = model_config.decoder.decoder_start_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.hidden_size = model.decoder.config.n_embd  # for DeepSpeed compatibility

    # load the datasets
    train_dataset = None
    scst_dataset = None
    if training_args.do_train:
        if custom_args.scst:
            train_dataset = create_dataset(custom_args.scst_datasets, map_fn=partial_process_sample,
                                           batch_size=training_args.per_device_train_batch_size, shuffle=True, repeat=True)
            scst_dataset = create_dataset(custom_args.scst_datasets, map_fn=partial_process_sample, batch_size=100,
                                          shuffle=False, repeat=False)
        else:
            train_dataset = create_dataset(custom_args.train_datasets, map_fn=partial_process_sample,
                                           batch_size=training_args.per_device_train_batch_size, shuffle=True, repeat=True,
                                           dataset_rate=custom_args.dataset_rate,
                                           dataset_machine_rate=custom_args.dataset_machine_rate)
    eval_dataset = create_dataset(custom_args.validation_datasets, map_fn=partial_process_sample,
                                  batch_size=training_args.per_device_eval_batch_size, shuffle=False, repeat=False)
    test_dataset = create_dataset(custom_args.test_datasets, map_fn=partial_process_sample,
                                  batch_size=training_args.per_device_eval_batch_size, shuffle=False, repeat=False)

    # create the Trainer
    if training_args.deepspeed and (optimizer is not None or scheduler is not None):
        logger.warning("Passing `optimizers` is not allowed if Fairscale, Deepspeed or PyTorch FSDP is enabled. "
                       "Setting `optimizers` to (None, None)")
        optimizer, scheduler = None, None
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        custom_args=custom_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        data_collator=partial_collate_fn,
        compute_metrics=partial_compute_metrics,
        optimizers=(optimizer, scheduler),
        tokenizer=tokenizer,
        collate_fn=partial_collate_fn
    )
    if custom_args.scst and training_args.do_train:
        # Workaround to back propagate the gradient
        trainer.compute_loss = trainer.compute_SCST_reward

        cider_instance_file = Path('cider_instance.pickle')
        cache_dir = None
        try:
            logger.info('Loading Cider instance from file...')
            with open(cider_instance_file, 'rb') as f:
                unpickler = pickle.Unpickler(f)
                trainer.cider = unpickler.load()
        except (pickle.UnpicklingError, FileNotFoundError) as e:
            logger.warning(e)
            logger.info('Loading samples from dataset to initialize a Cider instance...')
            refs = list()
            # awful workaround to use only a slice of the whole dataset
            # because webdataset==0.1.103 does not have .withlength(n) and webdataset>0.1.103 is bugged
            len_refs = 500 if is_debug else 113300
            len_refs //= 100
            for i, examples in enumerate(tqdm(scst_dataset, total=len_refs)):
                if i >= len_refs:
                    break
                _, ref = postprocess_text(references=[example['gts'] for example in examples[1]])
                refs.extend(ref)
            refs = {k: v for k, v in enumerate(refs)}
            refs = _add_eos(refs, tokenizer)
            trainer.cider = Cider(refs)

            if is_main_process():
                logger.info('Saving Cider instance to file...')
                try:
                    with open(cider_instance_file, 'wb') as f:
                        pickler = pickle.Pickler(f)
                        pickler.dump(trainer.cider)
                except pickle.PicklingError as e:
                    logger.warning(e)

            torch.distributed.barrier()
        trainer.cider._data_dir_root = os.path.expanduser(cache_dir or evaluate.config.HF_METRICS_CACHE)

    # add generate_with_backpropagation method to the model
    if custom_args.scst or training_args.do_eval or training_args.do_predict:
        model.generate = MethodType(generate_with_backpropagation, model)
        model.generate_with_backpropagation = MethodType(generate_with_backpropagation, model)

    # add training callbacks
    trainer.add_callback(CustomDeleteCheckpointsCallback(custom_args.custom_checkpoint_keeper))
    trainer.add_callback(GlobalStepCallback())
    if 'wandb' in training_args.report_to:
        trainer.remove_callback(WandbCallback)
        trainer.add_callback(CustomWandbCallback(custom_args=custom_args_dict))

    # Training
    if training_args.do_train:
        if custom_args.scst and is_first_scst_run:
            resume_from_checkpoint = None
        else:
            resume_from_checkpoint = checkpoint
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint,
                                     ignore_keys_for_eval=['input_ids'])
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    def get_id(checkpoint_name):
        if isinstance(checkpoint_name, list):
            return "_".join(get_id(c) for c in checkpoint_name)
        idx = checkpoint_name.split('checkpoint-')[-1]
        if idx.endswith('/'):
            idx = idx[:-1]

        return idx

    try:
        checkpoint_id = get_id(checkpoint)
    except ValueError:
        logger.warning("Could not find checkpoint id, using None as default")
        checkpoint_id = None

    # Evaluation
    if training_args.do_eval:
        evaluation_output = trainer.evaluate(eval_dataset=eval_dataset, num_beams=training_args.generation_num_beams)
        save_predictions(evaluation_output, tokenizer, trainer, split='eval', checkpoint_id=checkpoint_id)

    # Prediction
    if training_args.do_predict:
        prediction_output = trainer.predict(test_dataset=test_dataset, num_beams=training_args.generation_num_beams)
        save_predictions(prediction_output, tokenizer, trainer, split='test', checkpoint_id=checkpoint_id)

    logger.info('END')


if __name__ == "__main__":
    main()
