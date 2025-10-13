# based on https://github.com/kuleshov-group/mdlm/blob/master/main.py
"""
Generate samples from a trained MDLM-style diffusion model with FK Steering
"""
import os
import json
from datetime import datetime

import hydra
import lightning as L

import sys

sys.path.append('mdlm')
import dataloader
import utils

# FKDiffusion wraps MDLM diffusion model
from fk_diffusion import FKDiffusion

from tqdm import tqdm

from mdlm.main import _print_config


def _load_from_checkpoint(config, tokenizer):
    """Load model from checkpoint"""
    if 'hf' in config.backbone:
        return FKDiffusion(config, tokenizer=tokenizer).to('cuda')

    return FKDiffusion.load_from_checkpoint(
        config.eval.checkpoint_path, tokenizer=tokenizer, config=config
    )


def generate_samples_with_prompt_file(config, logger, tokenizer):
    """Generate samples from a trained model using the config specified"""
    logger.info('Generating samples.')
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)

    model.gen_ppl_metric.reset()
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None

    # stride_length = config.sampling.stride_length
    # num_strides = config.sampling.num_strides

    prompts_from_file = [None]

    if config.sampling.prompt_file is not None:
        with open(config.sampling.prompt_file, 'r') as f:
            prompts_from_file = [json.loads(l) for l in f]
            prompts_from_file = [p["context_string"] for p in prompts_from_file]

    aggregated_text_samples = []
    aggregated_historic_means = []
    aggregated_best_r = []

    for prompt_text in prompts_from_file:
        for _ in tqdm(list(range(config.sampling.num_sample_batches))):
            if config.sampling.semi_ar:
                raise NotImplementedError(
                    "Zach: Semi-AR sampling on MDLM not supported"
                )
                # _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                #   stride_length=stride_length,
                #   num_strides=num_strides,
                #   dt=1 / config.sampling.steps)
                # text_samples = intermediate_samples[-1]
                # # Note: Samples generated using semi-ar method
                # # need to to be processed before computing generative perplexity
                # # since these samples contain numerous <|endoftext|> tokens
                # # and diffusion.compute_generative_perplexity() discards
                # # any text after the first EOS token.
            else:
                # samples, *aux_retval = model.restore_model_and_sample(
                #   num_steps=config.sampling.steps)

                results = model.restore_model_and_sample(
                    num_steps=config.sampling.steps, prompt_text=prompt_text
                )

                text_samples = model.tokenizer.batch_decode(results['best'])
                historical_means = results['historic_means']

            aggregated_text_samples.extend(text_samples)
            aggregated_historic_means.append(historical_means)
            aggregated_best_r.append(results['best_r'].item())

            print('Text samples:', text_samples)

    return dict(
        aggregated_text_samples=aggregated_text_samples,
        aggregated_historic_means=aggregated_historic_means,
        aggregated_best_r=aggregated_best_r,
    )


@hydra.main(version_base=None, config_path='configs', config_name='fk_steering_config')
def main(config):
    """Does the following:

    1. Load the model from the checkpoint
    2. For every prompt in the prompt file, generate samples
    3. Save the samples to a file along with final and intermediate rewards

    """
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)

    cur_date = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = f'fk_steering/sample_evaluation/{cur_date}'
    os.makedirs(path, exist_ok=False)

    # save info
    with open(f'{path}/info.json', 'w') as f:
        info = {'fk_steering': dict(config.fk_steering)}
        info['seed'] = config.seed
        info['checkpoint_path'] = config.eval.checkpoint_path
        info['prompt_file'] = config.sampling.prompt_file

        f.write(json.dumps(info))

    logger.info('Starting Sample Evaluation.')

    sample_results = generate_samples_with_prompt_file(config, logger, tokenizer)

    text_samples = sample_results['aggregated_text_samples']
    historic_means = sample_results['aggregated_historic_means']
    best_r = sample_results['aggregated_best_r']

    with open(f'{path}/text_samples.jsonl', 'w') as f:
        for text, r_means, r in zip(text_samples, historic_means, best_r):
            f.write(json.dumps({'text': text, 'r_means': r_means, 'r': r}) + '\n')

    # use matplotlib to plot the historic mean rewards per sample
    import matplotlib.pyplot as plt

    for m in historic_means:
        plt.plot(m)
    plt.savefig(f'{path}/historic_means.png')


if __name__ == '__main__':
    main()
