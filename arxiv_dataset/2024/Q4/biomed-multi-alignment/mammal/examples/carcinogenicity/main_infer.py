import os

import click
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp

from mammal.examples.carcinogenicity.task import CarcinogenicityTask
from mammal.keys import CLS_PRED, SCORES
from mammal.model import Mammal


@click.command()
@click.argument("finetune_output_dir")
@click.argument("drug_seq")
@click.option(
    "--device", default="cpu", help="Specify the device to use (default: 'cpu')."
)
def main(finetune_output_dir: str, drug_seq: str, device: str) -> None:
    click.echo(f"Using device: {device}")
    infer(finetune_output_dir=finetune_output_dir, drug_seq=drug_seq, device=device)


def infer(finetune_output_dir: str, drug_seq: str, device: str) -> dict:
    """
    :param finetune_output_dir: model_dir argument in fine-tuning
    :param drug_seq: smiles sequence of a drug
    """
    # Load tokenizer from the checkpoint dir.
    # NOTE It's important to load the tokenizer from the fine-tuning phase, since we've introduced a new token to it.
    tokenizer_op = ModularTokenizerOp.from_pretrained(
        os.path.join(finetune_output_dir, "tokenizer")
    )

    # Load model from the best checkpoint.
    # NOTE The total order of the checkpoints is induced by the monitored metric (see config)
    nn_model = Mammal.from_pretrained(
        pretrained_model_name_or_path=os.path.join(
            finetune_output_dir, "best_epoch.ckpt"
        )
    )
    nn_model.eval()
    nn_model.to(device=device)

    # Format the input drug sequence value into a prompt that fits MAMMAL's training paradigm.
    sample_dict = {"drug_seq": drug_seq}
    sample_dict = CarcinogenicityTask.data_preprocessing(
        sample_dict=sample_dict,
        sequence_key="drug_seq",
        tokenizer_op=tokenizer_op,
        device=nn_model.device,
    )

    # running in generate mode
    batch_dict = nn_model.generate(
        [sample_dict],
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=5,
    )

    # Post-process the model's output
    ans = CarcinogenicityTask.process_model_output(
        tokenizer_op=tokenizer_op,
        decoder_output=batch_dict[CLS_PRED][0],
        decoder_output_scores=batch_dict[SCORES][0],
    )

    # Print prediction
    print(ans)
    return ans


if __name__ == "__main__":
    main()
