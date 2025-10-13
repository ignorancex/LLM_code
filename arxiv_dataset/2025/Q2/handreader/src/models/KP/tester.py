import rootutils
import torch
import torch.nn.functional as F
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.Atester import AbstractTester
from src.utils import LOGGER, compute_acc


class KPTester(AbstractTester):
    """
    A class for testing models using both greedy and beam search decoding strategies.

    This class uses a provided decoder to evaluate a model's performance on a dataset,
    computing predictions via greedy and beam search decoding. It tracks the Levenshtein
    accuracy of the predictions against ground truth labels.

    Parameters
    ----------
    decoder : object
        An object that provides methods for decoding (`beam_decode`, `greedy_decode`)
        and character conversion (`int_to_char`).

    Methods
    -------
    run_test(model, loader, device, beam_size=5, lm_beta=0.4, ins_gamma=1.2)
        Runs the test loop and computes accuracy using greedy and beam search decoding.
    """

    def __init__(self, decoder):
        """
        Initializes the KPTester instance.

        Parameters
        ----------
        decoder : object
            A decoder object used for decoding and mapping character indices to characters.
        """
        super().__init__()
        self.decoder = decoder

    @torch.no_grad()
    def run_test(
        self,
        model,
        loader,
        device: torch.device,
        beam_size: int = 5,
        lm_beta: float = 0.4,
        ins_gamma: float = 1.2,
    ):
        """
        Runs the test loop and computes accuracy using greedy and beam search decoding.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be tested.
        loader : torch.utils.data.DataLoader
            The data loader for the test dataset.
        device : torch.device
            The device on which computations are performed (e.g., "cuda" or "cpu").
        beam_size : int, optional
            The beam size for beam search decoding. Default is 5.
        lm_beta : float, optional
            The language model weight used for beam search decoding. Default is 0.4.
        ins_gamma : float, optional
            The insertion penalty used for beam search decoding. Default is 1.2.

        Returns
        -------
        lev_acc_greedy : float
            The Levenshtein accuracy of the greedy decoded predictions.
        lev_acc_beam : float
            The Levenshtein accuracy of the beam search decoded predictions.
        """
        model.eval()
        model = model.to(device)
        pbar_test = tqdm(loader)
        preds_greedy = []
        preds_beam = []
        gt_labels = []

        for i, batch in enumerate(pbar_test):
            inputs, targets, input_lengths, target_lengths = batch
            inputs, targets, input_lengths, target_lengths = (
                inputs.to(device),
                targets.to(device),
                input_lengths.to(device),
                target_lengths.to(device),
            )

            ctc_out = model(inputs)

            probs = F.softmax(ctc_out, dim=-1)
            for i in range(probs.shape[0]):
                current_preds_beam = self.decoder.beam_decode(
                    probs[i].detach().cpu().numpy(), beam_size, lm_beta, ins_gamma
                )

                current_preds_beam = "".join(current_preds_beam)
                preds_beam.append(current_preds_beam)

                current_preds_greedy = self.decoder.greedy_decode(probs[i].detach().cpu().numpy())

                current_preds_greedy = "".join(current_preds_greedy)

                preds_greedy.append(current_preds_greedy)
                cur_gt = "".join(
                    [
                        self.decoder.int_to_char[num]
                        for num in targets[i].detach().cpu().numpy()[: target_lengths[i]]
                    ]
                )
                gt_labels.append(cur_gt)

            pbar_test.set_description(
                f" Test lev {compute_acc(preds_greedy, gt_labels)} {compute_acc(preds_beam, gt_labels)}"
            )

        lev_acc_greedy = compute_acc(preds_greedy, gt_labels)
        lev_acc_beam = compute_acc(preds_beam, gt_labels)
        LOGGER.info(f"Final lev is : greedy: {lev_acc_greedy} beam: {lev_acc_beam}")
