import rootutils
import torch
import torch.nn.functional as F
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.Atester import AbstractTester
from src.utils import LOGGER, compute_acc


class KP_RGBTester(AbstractTester):
    """
    A class for testing models on RGB and keypoint data using beam search and greedy decoding.

    This class uses a given trained model to run tests on a dataset. It calculates the Levenshtein edit distance
    between the predicted sequences (both using beam search and greedy decoding) and the ground truth sequences.
    This allows for evaluating the performance of the model on recognizing sequences.
    """

    def __init__(
        self,
    ):
        """
        Initializes the tester.
        """
        super().__init__()

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
        Executes the testing loop to evaluate model performance using both greedy and beam search decoding.

        This function sets the model to evaluation mode, processes batches from the provided data loader,
        and performs decoding on the model's output to generate predictions. It computes the Levenshtein
        accuracy for both decoding methods and logs the results.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be evaluated.
        loader : torch.utils.data.DataLoader
            DataLoader providing batches of test data.
        device : torch.device
            The device (CPU or GPU) on which to perform computations.
        beam_size : int, optional
            The width of the beam for beam search decoding. Default is 5.
        lm_beta : float, optional
            Language model weight for beam search decoding. Default is 0.4.
        ins_gamma : float, optional
            Insertion penalty for beam search decoding. Default is 1.2.

        Returns
        -------
        lev_acc_greedy : float
            Levenshtein accuracy of greedy decoded predictions.
        lev_acc_beam : float
            Levenshtein accuracy of beam search decoded predictions.
        """
        model.eval()
        model = model.to(device)
        pbar_test = tqdm(loader)
        preds_beam = []
        preds_greedy = []
        gt_labels = []

        for i, batch in enumerate(pbar_test):
            inputs, landmarks, targets, input_lengths, target_lengths = batch
            inputs, landmarks, targets, input_lengths, target_lengths = (
                inputs.to(device),
                landmarks.to(device),
                targets.to(device),
                input_lengths.to(device),
                target_lengths.to(device),
            )
            ctc_out = model(inputs, landmarks, input_lengths)

            probs = F.softmax(ctc_out, dim=-1)
            for i in range(probs.shape[0]):
                current_preds_beam = model.decoder.beam_decode(
                    probs[i].detach().cpu().numpy(), beam_size, lm_beta, ins_gamma
                )
                current_preds_beam = "".join(current_preds_beam)
                preds_beam.append(current_preds_beam)

                current_preds_greedy = model.decoder.greedy_decode(probs[i].detach().cpu().numpy())

                current_preds_greedy = "".join(current_preds_greedy)

                preds_greedy.append(current_preds_greedy)
                cur_gt = "".join(
                    [
                        model.decoder.int_to_char[num]
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
