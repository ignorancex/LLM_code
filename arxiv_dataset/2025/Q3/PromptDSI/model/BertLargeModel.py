from torch import nn
from transformers import RobertaModel, RobertaConfig
# from adapters import SeqBnConfig, DoubleSeqBnConfig, BertAdapterModel, AutoAdapterModel

from .eprompt import (
    EPrompt,
    EPromptEnsemble,
    L2PEPrompt,
    EPromptWithTopicModelling,
    # EPromptWithTopicModellingShared,
)
from .codaprompt import CodaPrompt

from .PromptBertModel import PromptBertModel
from .CodaPromptBertModel import CodaPromptBertModel

from .OSPromptBertModel import OSPromptBertModel
from .OSCodaPromptBertModel import OSCodaPromptBertModel


# adapted from https://github.com/facebookresearch/DPR/blob/main/dpr/models/hf_models.py


##### IncDSI #####
class HFBertEncoder(RobertaModel):
    def __init__(self, config, **kwargs):
        RobertaModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.init_weights()

    @classmethod
    def init_encoder(cls, dropout: float = 0.1):
        cfg = RobertaModel.from_pretrained("roberta-base")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained("roberta-base", config=cfg)

    def forward(self, input_ids, attention_mask, layerwise_allocation=False, **kwargs):
        if layerwise_allocation:
            encoding = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )  # BaseModelOutputWithPoolingAndCrossAttentions (dict)
            return encoding
        else:
            hidden_states = None
            sequence_output, pooled_output = super().forward(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=False
            )
            pooled_output = sequence_output[:, 0, :]

            return sequence_output, pooled_output, hidden_states


class PromptBertEncoder(PromptBertModel):
    def __init__(self, config):
        if config.prompt_type == "l2peprompt":
            PromptBertModel.__init__(self, config, L2PEPrompt)
        elif config.prompt_type == "eprompt":
            PromptBertModel.__init__(self, config, EPrompt)
        elif config.prompt_type == "eprompt_ensemble":
            print("using eprompt ensemble")
            PromptBertModel.__init__(self, config, EPromptEnsemble)
        elif "eprompt_topic" in config.prompt_type:
            PromptBertModel.__init__(self, config, EPromptWithTopicModelling)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.init_weights()

    @classmethod
    def init_encoder(cls, args, dropout: float = 0.1):
        cfg = RobertaConfig.from_pretrained("roberta-base")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        # Adding prompt parameters to config
        cfg.update(vars(args))
        return cls.from_pretrained("roberta-base", config=cfg)

    def forward(
        self,
        input_ids,
        attention_mask,
        task_id,
        cls_features,
        train,
        f,
        previous_task_key_centroids,
        **kwargs,
    ):
        hidden_states = None
        output, res = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
            task_id=task_id,
            cls_features=cls_features,
            train=train,
            f=f,
            previous_task_key_centroids=previous_task_key_centroids,
        )
        sequence_output, pooled_output = output
        # Using pooled_output directly, so commented out the following line.
        # pooled_output = sequence_output[:, 0, :]

        return sequence_output, pooled_output, hidden_states, res


class CODAPromptBertEncoder(CodaPromptBertModel):
    def __init__(self, config):
        CodaPromptBertModel.__init__(self, config, CodaPrompt)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.init_weights()

    @classmethod
    def init_encoder(cls, args, dropout: float = 0.1):
        cfg = RobertaConfig.from_pretrained("roberta-base")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        # Adding prompt parameters to config
        cfg.update(vars(args))
        return cls.from_pretrained("roberta-base", config=cfg)

    def forward(
        self,
        input_ids,
        attention_mask,
        task_id,
        cls_features,
        train,
        f,
        previous_task_key_centroids,
        **kwargs,
    ):
        hidden_states = None
        output, res = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
            task_id=task_id,
            cls_features=cls_features,
            train=train,
            f=f,
            previous_task_key_centroids=previous_task_key_centroids,
        )
        sequence_output, pooled_output = output
        # Using pooled_output directly, so commented out the following line.
        # pooled_output = sequence_output[:, 0, :]

        return sequence_output, pooled_output, hidden_states, res


class OSPromptBertEncoder(OSPromptBertModel):
    def __init__(self, config):
        if config.prompt_type == "l2peprompt":
            OSPromptBertModel.__init__(self, config, L2PEPrompt)
        elif config.prompt_type == "eprompt":
            OSPromptBertModel.__init__(self, config, EPrompt)
        elif config.prompt_type == "eprompt_ensemble":
            print("using eprompt ensemble")
            OSPromptBertModel.__init__(self, config, EPromptEnsemble)
        # elif "eprompt_topic_shared" in config.prompt_type:
        #     OSPromptBertModel.__init__(self, config, EPromptWithTopicModellingShared)
        elif "eprompt_topic" in config.prompt_type:
            OSPromptBertModel.__init__(self, config, EPromptWithTopicModelling)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.init_weights()

    @classmethod
    def init_encoder(cls, args, dropout: float = 0.1):
        cfg = RobertaConfig.from_pretrained("roberta-base")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        # Adding prompt parameters to config
        cfg.update(vars(args))
        return cls.from_pretrained("roberta-base", config=cfg)

    def forward(
        self,
        input_ids,
        attention_mask,
        task_id,
        cls_features,
        train,
        f,
        previous_task_key_centroids,
        **kwargs,
    ):
        hidden_states = None
        output, res = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
            task_id=task_id,
            train=train,
            output_hidden_states=True,
            f=f,
            previous_task_key_centroids=previous_task_key_centroids,
        )
        sequence_output, pooled_output, all_hidden_states = output
        # Using pooled_output directly, so commented out the following line.
        # pooled_output = sequence_output[:, 0, :]

        return sequence_output, pooled_output, hidden_states, res


class OSCodaPromptBertEncoder(OSCodaPromptBertModel):
    def __init__(self, config):
        OSCodaPromptBertModel.__init__(self, config, CodaPrompt)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.init_weights()

    @classmethod
    def init_encoder(cls, args, dropout: float = 0.1):
        cfg = RobertaConfig.from_pretrained("roberta-base")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        # Adding prompt parameters to config
        cfg.update(vars(args))
        return cls.from_pretrained("roberta-base", config=cfg)

    def forward(
        self,
        input_ids,
        attention_mask,
        task_id,
        cls_features,
        train,
        f,
        previous_task_key_centroids,
        **kwargs,
    ):
        hidden_states = None
        output, res = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
            task_id=task_id,
            train=train,
            output_hidden_states=True,
            f=f,
            previous_task_key_centroids=previous_task_key_centroids,
        )
        sequence_output, pooled_output, all_hidden_states = output
        # Using pooled_output directly, so commented out the following line.
        # pooled_output = sequence_output[:, 0, :]

        return sequence_output, pooled_output, hidden_states, res


encoder_mapping = {
    "incdsi": HFBertEncoder,
    "prompt_bert": PromptBertEncoder,
    "osprompt_bert": OSPromptBertEncoder,
    "codaprompt_bert": CODAPromptBertEncoder,
    "oscodaprompt_bert": OSCodaPromptBertEncoder,
}


class QueryClassifier(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(self, class_num):
        super(QueryClassifier, self).__init__()
        # note here we only have question encoder
        self.question_model = HFBertEncoder.init_encoder()
        self.classifier = nn.Linear(
            self.question_model.config.hidden_size, class_num, bias=False
        )

    def query_emb(self, input_ids, attention_mask):
        sequence_output, pooled_output, hidden_states = self.question_model(
            input_ids, attention_mask
        )
        return pooled_output

    def forward(self, query_ids, attention_mask_q, return_hidden_emb=False, **kwargs):
        q_embs = self.query_emb(query_ids, attention_mask_q)
        if return_hidden_emb:
            return q_embs
        logits = self.classifier(q_embs)
        return logits


# A generalized version of the above class, the only difference is the question model is passed in as an argument
class GeneralQueryClassifier(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(self, args, question_model):
        super().__init__()
        if question_model == "incdsi":
            self.question_model = HFBertEncoder.init_encoder()
        else:
            self.question_model = encoder_mapping[question_model].init_encoder(args)
        # Experimental
        # NOTE: the BetterTransformer integration drops the mask support and can only be used for training that do not require a padding mask for batched training
        # self.question_model = BetterTransformer.transform(
        #     self.question_model, keep_original_model=False
        # )
        if args.dropout:
            self.dropout = nn.Dropout(args.dropout_rate)
        else:
            self.dropout = None
        self.classifier = nn.Linear(
            self.question_model.config.hidden_size, args.class_num, bias=False
        )

    def query_emb(
        self, query_ids, attention_mask_q, task_id=None, cls_features=None, train=False, layerwise_allocation=False, f=None, previous_task_key_centroids=None,
    ):
        output = self.question_model(
            query_ids,
            attention_mask_q,
            task_id=task_id,
            cls_features=cls_features,
            train=train,
            layerwise_allocation=layerwise_allocation,
            f=f,
            previous_task_key_centroids=previous_task_key_centroids,
        )
        if len(output) == 3:  # Original model (HFBertEncoder)
            # if layerwise_allocation:
            #     return output["hidden_states"][1:]
            # else:
            sequence_output, pooled_output, hidden_states = output
            return pooled_output
        else:
            sequence_output, pooled_output, hidden_states, res = output
            return pooled_output, res

    def forward(
        self,
        query_ids,
        attention_mask_q,
        return_hidden_emb=False,
        task_id=None,
        cls_features=None,
        train=False,
        layerwise_allocation=False,
        f=0,
        previous_task_key_centroids=None,
    ):
        output = self.query_emb(
            query_ids,
            attention_mask_q,
            task_id,
            cls_features,
            train,
            layerwise_allocation=layerwise_allocation,
            f=f,
            previous_task_key_centroids=previous_task_key_centroids, 
        )
        res = None
        # if layerwise_allocation:
        #     return output
        if len(output) == 2:
            q_embs, res = output
        else:  # Original model (HFBertEncoder)
            q_embs = output
        if return_hidden_emb:
            if train:
                return q_embs, res
            else:
                return q_embs#, res

        # Add dropout layer
        if self.dropout:
            q_embs = self.dropout(q_embs)
        logits = self.classifier(q_embs)
        # q_embs = q_embs / torch.norm(q_embs, dim=1, keepdim=True)
        # weight = self.classifier.weight.data / torch.norm(self.classifier.weight.data, dim=1, keepdim=True)
        # logits = torch.matmul(q_embs, weight.t())

        return logits, res#, q_embs


class CLSQueryClassifier(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(self, args, question_model):
        super().__init__()
        if question_model == "incdsi":
            self.question_model = HFBertEncoder.init_encoder()
        else:
            self.question_model = encoder_mapping[question_model].init_encoder(args)

        self.dropout = None
        self.classifier = nn.Linear(
            self.question_model.config.hidden_size, args.class_num, bias=False
        )

    def query_emb(
        self,
        query_ids,
        attention_mask_q,
        task_id=None,
        cls_features=None,
        train=False,
        layerwise_allocation=False,
        f=None,
        previous_task_key_centroids=None,
    ):
        output = self.question_model(
            query_ids,
            attention_mask_q,
            task_id=task_id,
            cls_features=cls_features,
            train=train,
            layerwise_allocation=True,
            f=f,
            previous_task_key_centroids=previous_task_key_centroids,
        )
        hidden_states = output[2]
        return hidden_states

    def forward(
        self,
        query_ids,
        attention_mask_q,
        return_hidden_emb=False,
        task_id=None,
        cls_features=None,
        train=False,
        layerwise_allocation=False,
        f=0,
        previous_task_key_centroids=None,
    ):
        q_embs = self.query_emb(
            query_ids,
            attention_mask_q,
            task_id,
            cls_features,
            train,
            layerwise_allocation=layerwise_allocation,
            f=f,
            previous_task_key_centroids=previous_task_key_centroids,
        )
        return q_embs


# class GeneralQueryClassifierWithAdapters(nn.Module):
#     """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

#     def __init__(self, args, question_model):
#         super().__init__()
#         self.question_model = BertAdapterModel.from_pretrained("roberta-base")
#         self.config = SeqBnConfig(
#             mh_adapter=True,
#             output_adapter=False,
#             reduction_factor=16, 
#             init_weights="bert",
#             non_linearity="linear",
#             leave_out=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # The IDs of the layers (starting at 0) where NO adapter modules should be added.
#         )
#         self.question_model.active_head = None
#         self.question_model.add_adapter("bottleneck_adapter", config=self.config)
#         self.question_model.set_active_adapters("bottleneck_adapter")
#         self.classifier = nn.Linear(
#             self.question_model.config.hidden_size, args.class_num, bias=False
#         )
#         pprint("Adapter config")
#         print(self.config)

#     def query_emb(
#         self,
#         query_ids,
#         attention_mask_q,
#         task_id=None,
#         cls_features=None,
#         train=False,
#         layerwise_allocation=False,
#         f=None,
#         previous_task_key_centroids=None,
#     ):
#         output = self.question_model(
#             query_ids,
#             attention_mask_q,
#             task_id=task_id,
#             cls_features=cls_features,
#             train=train,
#             layerwise_allocation=layerwise_allocation,
#             f=f,
#             previous_task_key_centroids=previous_task_key_centroids,
#         )
#         # output: <class 'transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions'>
#         # output: dict with two keys: odict_keys(['last_hidden_state', 'pooler_output'])
#         sequence_output, pooled_output = output.values()
#         pooled_output = sequence_output[:, 0, :]
#         return pooled_output

#     def forward(
#         self,
#         query_ids,
#         attention_mask_q,
#         return_hidden_emb=False,
#         task_id=None,
#         cls_features=None,
#         train=False,
#         layerwise_allocation=False,
#         f=0,
#         previous_task_key_centroids=None,
#     ):
#         output = self.query_emb(
#             query_ids,
#             attention_mask_q,
#             task_id,
#             cls_features,
#             train,
#             layerwise_allocation=layerwise_allocation,
#             f=f,
#             previous_task_key_centroids=previous_task_key_centroids,
#         )
#         res = None
#         q_embs = output
#         logits = self.classifier(q_embs)

#         return logits, res  # , q_embs
