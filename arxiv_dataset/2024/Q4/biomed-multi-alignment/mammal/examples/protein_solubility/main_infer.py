import os

import click
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp

from mammal.examples.protein_solubility.task import ProteinSolubilityTask
from mammal.keys import CLS_PRED, SCORES
from mammal.model import Mammal


@click.command()
@click.option(
    "--finetune_output_dir",
    default=None,
    help="Specify the model dir (default: None to download from huggingface).",
)
@click.argument(
    "protein_seq",
    default="NLMKRCTRGFRKLGKCTTLEEEKCKTLYPRGQCTCSDSKMNTHSCDCKSC",
)
@click.option(
    "--device", default="cpu", help="Specify the device to use (default: 'cpu')."
)
def main(finetune_output_dir: str, protein_seq: str, device: str):
    protein_solubility_infer(
        finetune_output_dir=finetune_output_dir, protein_seq=protein_seq, device=device
    )


def protein_solubility_infer(
    finetune_output_dir: str | None, protein_seq: str, device: str
):
    """
    :param finetune_output_dir: model_dir argument in finetuning or None for downloading from huggingface
    :param protein_seq: amino acid sequence of a protein
    """
    if finetune_output_dir is not None:
        # load tokenizer and model from finetune_output_dir
        tokenizer_op = ModularTokenizerOp.from_pretrained(
            os.path.join(finetune_output_dir, "tokenizer")
        )
        nn_model = Mammal.from_pretrained(
            pretrained_model_name_or_path=os.path.join(
                finetune_output_dir, "best_epoch.ckpt"
            )
        )
    else:
        # load tokenizer and model from huggingface
        tokenizer_op = ModularTokenizerOp.from_pretrained(
            "ibm/biomed.omics.bl.sm.ma-ted-458m.protein_solubility"
        )
        nn_model = Mammal.from_pretrained(
            "ibm/biomed.omics.bl.sm.ma-ted-458m.protein_solubility"
        )
    nn_model.eval()
    nn_model.to(device=device)

    # convert to MAMMAL style
    sample_dict = {"protein_seq": protein_seq}
    sample_dict = ProteinSolubilityTask.data_preprocessing(
        sample_dict=sample_dict,
        protein_sequence_key="protein_seq",
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
    ans = ProteinSolubilityTask.process_model_output(
        tokenizer_op=tokenizer_op,
        decoder_output=batch_dict[CLS_PRED][0],
        decoder_output_scores=batch_dict[SCORES][0],
    )

    # Print prediction
    print(ans)
    return ans


if __name__ == "__main__":
    main()
