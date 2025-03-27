import copy
import operator
import os
import pickle
import random
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import numpy
import torch
from absl import flags

from src.reference_implementations.prompt_zoo.data_utility import augment_batch, white_space_fix
from src.reference_implementations.prompt_zoo.prompted_lm import LMPrompted

FLAGS = flags.FLAGS


@dataclass(order=True)
class PromptTemplate:
    """A dataclass to define the prompt template with two attributes:

    1 - tokens: a list of token indices from the vocabulary table.
        tokens have size prompt_length.
    2 - score: the log likelihood of the label given this prompt template
                at a training step over a training batch.
    This dataclass is used for implementing a PromptSearchMemory for the gradient-search
        discrete prompt optimization augmented with beam search for moving from one search step to another step
        while scoring the prompt tokens.
    """

    tokens: List[int] = field(compare=False)
    score: float


class PromptSearchMemory:
    """A search memory that keeps a sorted beam (list) which stores the top
    beam_size templates according to the their label likelihood computed over a
    batch of training data.

    This memory is used to easily store and read PromptTemplates at
    different stages of the gradient-search prompting augmented with
    beam search.
    """

    def __init__(self, instruction_ids: List[int]) -> None:
        """This initializes the search memory for the training and its beam
        will be dumped to disk while saving the model."""
        # allocate memory for the current beam of templates.
        FLAGS.prompt_length = len(instruction_ids)
        self.beam = []
        for beam_idx in range(FLAGS.g_beam_size):
            self.beam.append(PromptTemplate(tokens=instruction_ids, score=-float("inf")))

    def update_beam(self, beam_candidates: List[PromptTemplate]) -> None:
        """For the next training step, select the top beam_size prompt
        templates.

        out of beam_size * top_k template candidates inside the
        beam_candidates. The sorting is based on the score attribute of
        the PromptTemplate.
        """
        # sort the prompt templates by their score in descending order.
        sorted_pool = sorted(beam_candidates, key=operator.attrgetter("score"), reverse=True)

        # keep the top beam_size prompt templates.
        self.beam = sorted_pool[: FLAGS.g_beam_size]

    def get_beam_loss(self) -> float:
        """Return the list of template scores inside the beam.

        then consider the averaged negative label log-likelihood over a
        beam of templates as the loss.
        """
        searched_scores = [template.score for template in self.beam]
        return -sum(searched_scores) / len(searched_scores)

    def generate_beam_candidates(
        self, embedding_weight: torch.Tensor, log_likelihoods: torch.Tensor, prompt_step: int
    ) -> List[PromptTemplate]:
        """For each prompt template inside the beam at the current step,
        compute the gradient with respect to the embedding vector of the prompt
        token at step prompt_step of the template.

        Perform dot product with the embedding_weight table to compute a
        score for every possible word replacement. Then consider the
        top_k replacement tokens and generate the new beam_candidates.
        """
        beam_candidates = []
        embedding_grads = []

        for beam_idx, prompt_template in enumerate(self.beam):
            prompt_token_idx = prompt_template.tokens[prompt_step]
            log_likelihood = log_likelihoods[beam_idx].squeeze()
            if FLAGS.g_beam_size == 1:
                log_likelihood.backward()
                embedding_grad = embedding_weight.grad[prompt_token_idx]
                embedding_grads.append(embedding_grad)
            else:
                log_likelihood.backward(retain_graph=True)
                embedding_grad = embedding_weight.grad[prompt_token_idx].detach().clone()
                embedding_grads.append(embedding_grad)
                embedding_weight.grad.data.zero_()

        embedding_grads_tensor = torch.stack(embedding_grads, dim=1)
        vocab_scores = torch.matmul(embedding_weight - embedding_weight[prompt_token_idx], embedding_grads_tensor)

        top_scores, top_indices = torch.topk(
            torch.nn.functional.relu(vocab_scores), FLAGS.top_k, dim=0, largest=True, sorted=True
        )

        # memory is on RAM and not on GPU.
        for index, top_idx_per_beam in enumerate(top_indices.tolist()):
            for beam_idx, prompt_template in enumerate(self.beam):
                if top_scores[index][beam_idx] > 0:
                    candidate_template = copy.deepcopy(prompt_template)
                    candidate_template.tokens[prompt_step] = top_idx_per_beam[beam_idx]
                    candidate_template.score = -float("inf")
                    beam_candidates.append(candidate_template)

        return beam_candidates + self.beam


class SearchLM(LMPrompted):
    """Subclassing the RobertaPrompted class to introduce the roberta-large for
    gradient-search.

    We also define a search memory to keep templates as we are scoring
    them during training.
    """

    def __init__(
        self,
        seed: int,
        task_name: str,
        enable_data_augmentation: int,
        enable_paraphrase_training: int,
        load_paraphraser: int,
    ) -> None:
        super().__init__(seed, enable_data_augmentation, enable_paraphrase_training, load_paraphraser)

        if task_name == "sst2":
            instruction = "In this task, you are given sentences from movie reviews. \
                The task is to classify a sentence as 'great' if the sentiment of the \
                sentence is positive or as 'terrible' if the sentiment of the sentence is negative."
        elif task_name == "cr":
            instruction = "In this task, you are given sentences from customer \
                    reviews. The task is to classify a sentence as 'great' if \
                    the sentiment of the sentence is positive or as 'terrible' \
                    if the sentiment of the sentence is negative."
        elif task_name == "mr":
            instruction = "In this task, you are given sentences from movie \
                    reviews. The task is to classify a sentence as 'great' if \
                    the sentiment of the sentence is positive or as 'terrible' \
                    if the sentiment of the sentence is negative."
        elif task_name == "sst5":
            instruction = "In this task, you are given sentences from movie reviews. \
                Based on the given review, classify it to one of the five classes: \
                    (1) terrible, (2) bad, (3) okay, (4) good, and (5) great."
        elif task_name == "subj":
            instruction = "In this task, you are given sentences from reviews. \
                The task is to classify a sentence as 'subjective' if the \
                opinion of the sentence is subjective or as 'objective' \
                if the opinion of the sentence is objective."
        elif task_name == "trec":
            instruction = "You are given a question. You need to detect which \
                category better describes the question. Answer with \
                'Description', 'Entity', 'Expression', 'Human', 'Location', and 'Number'."
        elif task_name == "agnews":
            instruction = "In this task, you are given a news article. Your task is to classify \
                the article to one out of the four topics 'World', 'Sports', 'Business', 'Tech' \
                if the article's main topic is relevant to the world, sports, business, \
                and technology, correspondingly. If you are not sure about the topic, choose the closest option."

        instruct_ids = self.tokenizer(white_space_fix(instruction), add_special_tokens=False)["input_ids"]
        self.search_memory = PromptSearchMemory(instruct_ids)

        if FLAGS.mode in ["test", "inference", "eval"]:
            # load from the given checkpoint.
            # search memory is now defined, and it can be used.
            self.load_from_checkpoint()

    def load_from_checkpoint(self, model_path: Optional[str] = None) -> None:
        """Load the optimized prompt templates from the specified checkpoint
        name and update the internal beam."""
        m_path = FLAGS.model_path
        ckp_name = FLAGS.checkpoint
        try:
            with open(os.path.join(m_path, f"{ckp_name}.pkl"), "rb") as inp:
                self.search_memory.beam = pickle.load(inp)
                FLAGS.prompt_length = len(self.search_memory.beam[0].tokens)
        except Exception as e:
            raise Exception("Could not load the checkpoint due to error:{}".format(e))

    def save(self, checkpoint_name: str, model_path: Optional[str] = None) -> None:
        """Save the optimized prompt templates to the model_path for the
        specified checkpoint name."""
        m_path = FLAGS.model_path
        if not os.path.exists(m_path):
            os.makedirs(m_path)

        with open(os.path.join(m_path, f"{checkpoint_name}.pkl"), "wb") as outp:
            pickle.dump(self.search_memory.beam, outp, pickle.HIGHEST_PROTOCOL)

    def score_templates(
        self,
        batch: torch.utils.data.Dataset,
        prompt_templates: List[PromptTemplate],
        train: bool = False,
        for_augmentation: Optional[bool] = False,
    ) -> torch.Tensor:
        """Run a forward computation over the batch for each prompt templates
        and compute the log probability over the batch for that given prompt
        template.

        This function can be called for training or inference.
        """
        batch_size, _ = batch["input_ids"].size()
        template_scores_arr = []
        for template in prompt_templates:
            class_log_ps = self.lm_forward_pass(batch, train, [template.tokens])

            if train:
                if not for_augmentation:
                    template_scores = class_log_ps.view(-1, 1).reshape(batch_size, 1)
                else:
                    template_scores = class_log_ps.view(-1, 1, 1).reshape(batch_size, 1, 1)
            else:
                template_scores = class_log_ps.reshape(batch_size, 1, -1)

            if not for_augmentation:
                template_scores_arr.append(template_scores)
            else:
                # compute the correct objective for data augmentation.
                orig_batch_size = batch_size // (FLAGS.test_sample_size + 1)
                scores = template_scores.reshape(orig_batch_size, FLAGS.test_sample_size + 1, 1, -1)
                template_scores_arr.append(0.5 * scores[:, 0, :, :] + 0.5 * torch.mean(scores[:, 1:, :, :], dim=1))
        return torch.stack(template_scores_arr, dim=1)

    def train(self, batch: torch.utils.data.Dataset) -> Dict[str, float]:
        """The train loop for gradient-search method."""
        if self.enable_data_augmentation == 1:
            paraphrases = self.draw_samples_for_augmentation(batch)
            augment_batch(
                batch,
                paraphrases,
                self.tokenizer,
                num_return_seq=FLAGS.test_sample_size,
                remove_instruction=True,
            )

        prompt_index = random.randint(0, FLAGS.prompt_length - 1)
        self.train_mode_on()
        template_log_likelihood = self.score_templates(
            batch, self.search_memory.beam, train=True, for_augmentation=self.enable_data_augmentation == 1
        )
        template_log_likelihood = template_log_likelihood.mean(dim=0)  # mean across batch_size
        beam_candidates = self.search_memory.generate_beam_candidates(
            embedding_weight=self.model_pool["roberta_model"].roberta.embeddings.word_embeddings.weight,
            log_likelihoods=template_log_likelihood,
            prompt_step=prompt_index,
        )
        self.predict_mode_on()
        beam_candidate_scores = self.score_templates(
            batch, beam_candidates, train=True, for_augmentation=self.enable_data_augmentation == 1
        )
        beam_candidate_scores = beam_candidate_scores.mean(dim=0)  # mean across batch_size
        for index, score in enumerate(beam_candidate_scores.tolist()):
            beam_candidates[index].score = score[0]

        self.search_memory.update_beam(beam_candidates)
        return {"loss_value": self.search_memory.get_beam_loss()}

    def no_ensemble_predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, str]]:
        """The main prediction loop for a given potential class label using a
        beam of templates."""
        top_template = self.search_memory.beam[0]
        self.predict_mode_on()
        class_log_ps = self.score_templates(batch, [top_template], train=False, for_augmentation=False)
        class_log_ps = class_log_ps.squeeze()

        class_log_ps = class_log_ps.cpu().detach().numpy()

        # not efficient, but let's pair potential class along the prediction scores.
        # all transformer special tokens will be removed.
        # same labels have been repeated once per template in beam.
        inputs_str = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)
        prompt_str = self.tokenizer.batch_decode(top_template.tokens, skip_special_tokens=True)
        print("evaluating batch with prompt template:", prompt_str)
        for index, input_str in enumerate(inputs_str):
            for class_idx in range(FLAGS.num_classes):
                output_row = {
                    "potential_class": self.tokenizer.id_to_class[str(class_idx)],
                    "original_prediction_score": class_log_ps[index, class_idx],
                    "prompt_str": prompt_str,
                    "original_inputs": inputs_str[index],
                    "gold_class": batch["gold_outputs"][index],
                }
                yield output_row

    def paraphrase_and_predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, str]]:
        """The main prediction loop for a given potential class label using a
        beam of templates and the paraphraser."""
        inputs_str = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)
        paraphrases = self.draw_samples_for_augmentation(batch, for_train=False)
        augment_batch(
            batch,
            paraphrases,
            self.tokenizer,
            num_return_seq=FLAGS.test_sample_size,
            remove_instruction=True,
        )

        top_template = self.search_memory.beam[0]
        self.predict_mode_on()
        class_log_ps = self.score_templates(batch, [top_template], train=False, for_augmentation=False)
        class_log_ps = class_log_ps.squeeze()
        class_log_ps = class_log_ps.cpu().detach().numpy()

        prompt_str = self.tokenizer.batch_decode(top_template.tokens, skip_special_tokens=True)
        print("evaluating batch with prompt template:", prompt_str)
        for index, input_str in enumerate(inputs_str):
            for class_idx in range(FLAGS.num_classes):
                scores = class_log_ps[
                    index * (FLAGS.test_sample_size + 1) : (index + 1) * (FLAGS.test_sample_size + 1), class_idx
                ]
                avg_score = numpy.mean(scores[1:])
                output_row = {
                    "potential_class": self.tokenizer.id_to_class[str(class_idx)],
                    "prediction_score": avg_score,
                    "all_prediction_score": 0.5 * avg_score + 0.5 * scores[0],
                    "original_prediction_score": scores[0],
                    "gold_class": batch["gold_outputs"][index],
                    "paraphrases": paraphrases[index * FLAGS.test_sample_size : (index + 1) * FLAGS.test_sample_size],
                    "original_inputs": inputs_str[index],
                }
                yield output_row
