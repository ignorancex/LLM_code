import os

import click
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp

from mammal.examples.dti_bindingdb_kd.task import DtiBindingdbKdTask
from mammal.model import Mammal


@click.command()
@click.option(
    "--finetune_output_dir",
    default=None,
    help="Specify the model dir (default: None to download from huggingface).",
)
@click.argument(
    "target_seq",
    default="NLMKRCTRGFRKLGKCTTLEEEKCKTLYPRGQCTCSDSKMNTHSCDCKSC",
)
@click.argument(
    "drug_seq",
    default="CC(=O)NCCC1=CNc2c1cc(OC)cc2",
)
@click.argument("norm_y_mean", default=5.79384684128215, type=float)
@click.argument("norm_y_std", default=1.33808027428196, type=float)
@click.option(
    "--device", default="cpu", help="Specify the device to use (default: 'cpu')."
)
def main(
    finetune_output_dir: str | None,
    target_seq: str,
    drug_seq: str,
    norm_y_mean: float,
    norm_y_std: float,
    device: str,
):
    dti_bindingdb_kd_infer(
        finetune_output_dir=finetune_output_dir,
        target_seq=target_seq,
        drug_seq=drug_seq,
        norm_y_mean=norm_y_mean,
        norm_y_std=norm_y_std,
        device=device,
    )


def dti_bindingdb_kd_infer(
    finetune_output_dir: str,
    target_seq: str,
    drug_seq: str,
    norm_y_mean: float,
    norm_y_std: float,
    device: str,
):
    """
    :param finetune_output_dir: model_dir argument in fine-tuning or None for downloading from huggingface
    :param target_seq: amino acid sequence of a target
    :param drug_seq: smiles representation of a drug
    :param norm_y_mean: specify the mean and std values used in fine-tuning
    :param norm_y_std: specify the mean and std values used in fine-tuning
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
            "ibm/biomed.omics.bl.sm.ma-ted-458m.dti_bindingdb_pkd"
        )
        nn_model = Mammal.from_pretrained(
            "ibm/biomed.omics.bl.sm.ma-ted-458m.dti_bindingdb_pkd"
        )
    nn_model.eval()
    nn_model.to(device=device)

    # convert to MAMMAL style
    sample_dict = {"target_seq": target_seq, "drug_seq": drug_seq}
    sample_dict = DtiBindingdbKdTask.data_preprocessing(
        sample_dict=sample_dict,
        tokenizer_op=tokenizer_op,
        target_sequence_key="target_seq",
        drug_sequence_key="drug_seq",
        norm_y_mean=None,
        norm_y_std=None,
        device=nn_model.device,
    )

    # forward pass - encoder_only mode which supports scalars predictions
    batch_dict = nn_model.forward_encoder_only([sample_dict])

    # Post-process the model's output
    batch_dict = DtiBindingdbKdTask.process_model_output(
        batch_dict,
        scalars_preds_processed_key="model.out.dti_bindingdb_kd",
        norm_y_mean=norm_y_mean,
        norm_y_std=norm_y_std,
    )
    ans = {
        "model.out.dti_bindingdb_kd": float(batch_dict["model.out.dti_bindingdb_kd"][0])
    }

    # Print prediction
    print(ans)
    return ans


if __name__ == "__main__":
    main()
