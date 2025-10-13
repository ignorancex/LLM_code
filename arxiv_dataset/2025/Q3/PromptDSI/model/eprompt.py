import torch
import torch.nn as nn

topic_embeddings_dict = {
    "nq320k_bert": "/home/thuy0050/code/PromptDSI/bertopic/bert_old_topic_embeddings_80.pt",
    "nq320k_sbert": "/home/thuy0050/code/PromptDSI/bertopic/sbert_old_topic_embeddings_91.pt",
    "msmarco_bert": "/home/thuy0050/code/PromptDSI/bertopic/msmarco_bert_old_topic_embeddings_182.pt",
    "msmarco_sbert": "/home/thuy0050/code/PromptDSI/bertopic/msmarco_sbert_old_topic_embeddings_193.pt",
    "nq320k_roberta": "/home/thuy0050/code/PromptDSI/bertopic/nq320k_roberta_old_topic_embeddings_89.pt",
    "msmarco_roberta": "/home/thuy0050/code/PromptDSI/bertopic/msmarco_roberta_old_topic_embeddings_175.pt"
}

topic_pool_size_dict = {
    "nq320k_bert": 80,
    "nq320k_sbert": 91,
    "msmarco_bert": 182,
    "msmarco_sbert": 193,
    "nq320k_roberta": 89,
    "msmarco_roberta": 175,
}

class Prompt(nn.Module):
    def __init__(
        self,
        length=20,
        embed_dim=768,
        embedding_key="cls",
        prompt_init="uniform",
        prompt_pool=True,
        prompt_key=True,
        pool_size=10,
        top_k=5,
        batchwise_prompt=False,
        prompt_key_init="uniform",
        diverisify_prompt_freq=True,
        fasttext=False,
        **kwargs,
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.diversify_prompt_freq = diverisify_prompt_freq
        self.fasttext = fasttext
        self.optimized_prompt_indices = []

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == "zero":
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == "uniform":
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)

        # if using learnable prompt keys
        if prompt_key:
            if self.fasttext:
                key_shape = (pool_size, 300)
            else:
                key_shape = (pool_size, embed_dim)

            if prompt_key_init == "zero":
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == "uniform":
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean

        # if using prompt frequency table
        if self.diversify_prompt_freq:
            self.prompt_freq = torch.zeros(self.pool_size, device=self.device)
            self.norm_prompt_freq = torch.zeros(self.pool_size, device=self.device)

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x**2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(
            torch.maximum(square_sum, torch.tensor(epsilon, device=x.device))
        )
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None, cls_features=None, train=False):
        out = dict()
        if self.prompt_pool:
            x_embed_mean = cls_features

            prompt_norm = self.l2_normalize(
                self.prompt_key, dim=1
            )  # Shape: Pool_size, C. Sum of square of each prompt is 1
            x_embed_norm = self.l2_normalize(
                x_embed_mean, dim=1
            )  # Shape: B, C. Sum of square is 1

            similarity = torch.matmul(
                x_embed_norm, prompt_norm.t()
            )  # Shape: B, Pool_size. Each row is the similarity of the embedding to each prompt, with value in the range of [-1, 1]

            if train and self.diversify_prompt_freq:
                similarity += 1  # Shift all values to positive range [0, 2]
                # Apply frequency penalty
                similarity = (
                    similarity * (1.0 - self.norm_prompt_freq)
                    if sum(self.prompt_freq) > 0
                    else similarity
                )

            if prompt_mask is None:
                similarity_top_k, idx = torch.topk(
                    similarity, k=self.top_k, dim=1
                )  # B, top_k
                out["similarity"] = similarity_top_k
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(
                        idx, return_counts=True, sorted=True
                    )

                    # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                    # Unless dimension is specified, this will be flattend if it is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat(
                            [
                                prompt_id,
                                torch.full(
                                    (self.pool_size - prompt_id.shape[0],),
                                    torch.min(idx.flatten()),
                                    device=prompt_id.device,
                                ),
                            ]
                        )
                        id_counts = torch.cat(
                            [
                                id_counts,
                                torch.full(
                                    (self.pool_size - id_counts.shape[0],),
                                    0,
                                    device=id_counts.device,
                                ),
                            ]
                        )
                    _, major_idx = torch.topk(id_counts, k=self.top_k)  # top_k
                    major_prompt_id = prompt_id[major_idx]  # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1)  # B, top_k
            else:
                idx = prompt_mask  # B, top_k

            batched_prompt_raw = self.prompt[idx]  # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(
                batch_size, top_k * length, c
            )  # B, top_k * length, C

            out["prompt_idx"] = idx
            out["selected_prompt_indices"] = torch.unique(idx, return_counts=True)

            # if train and self.diversify_prompt_freq:
            if self.diversify_prompt_freq:
                # Update the prompt frequency table
                unique_values, counts = torch.unique(idx, return_counts=True)
                # print(f"idx: {unique_values}; count: {counts}")
                self.prompt_freq[unique_values] += counts

                # Calculate normalized frequency table
                self.norm_prompt_freq = self.prompt_freq / torch.sum(self.prompt_freq)
                out["norm_prompt_freq"] = self.norm_prompt_freq

            # Debugging, return sim as well
            # Modification: disable the following lines
            # out["prompt_norm"] = prompt_norm
            # out["x_embed_norm"] = x_embed_norm
            # out["similarity"] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx]  # B, top_k, C
            # out["selected_key"] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
            sim = batched_key_norm * x_embed_norm
            # B, top_k, C. Ideally, if the batched_key_norm and x_embed_norm are exactly the same, then the sum of sim should be B * top_k
            # st()
            reduce_sim = (
                torch.sum(sim) / x_embed.shape[0]
            )  # Scalar, with a maximum value of (batchsize * top_k) / batchsize = top_k

            reduce_sim = reduce_sim / self.top_k # Normalize by top_k

            out["reduce_sim"] = reduce_sim
        else:
            if self.prompt_init == "zero":
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == "uniform":
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)

        # The input with the prompt concatenated to the front. [B, prompt+token, C], CLS token is at index "prompt_length"
        out["total_prompt_len"] = batched_prompt.shape[1]
        out["prompted_embedding"] = torch.cat([batched_prompt, x_embed], dim=1)

        return out


class EPromptWithTopicModelling(Prompt):  # Cosine similarity
    def __init__(
        self,
        length=5,
        embed_dim=768,
        embedding_key="mean",
        prompt_init="uniform",
        prompt_pool=False,
        prompt_key=False,
        pool_size=None,
        top_k=None,
        batchwise_prompt=False,
        prompt_key_init="uniform",
        num_layers=1,
        num_heads=12,
        prompt_allocation=10,
        contrastive_loss=False,
        model_encoder=None,
        base_data_dir="",
        **kwargs,
    ):
        super().__init__(
            length=length,
            embed_dim=embed_dim,
            embedding_key=embedding_key,
            prompt_init=prompt_init,
            prompt_pool=prompt_pool,
            prompt_key=prompt_key,
            pool_size=pool_size,
            top_k=top_k,
            batchwise_prompt=batchwise_prompt,
            prompt_key_init=prompt_key_init,
            **kwargs,
        )
        
        dataset_name = "nq320k" if "nq320k" in base_data_dir else "msmarco"
        # model_name = "sbert" if "sbert" in model_encoder else "bert"
        if "roberta" in model_encoder:
            model_name = "roberta"
        elif "sbert" in model_encoder:
            model_name = "sbert"
        else:
            model_name = "bert"
        name = dataset_name + "_" + model_name
        topic_embeddings = torch.load(topic_embeddings_dict[name])
        self.pool_size = topic_pool_size_dict[name]

        # if "nq320k" in base_data_dir:
        #     if "sbert" in model_encoder:
        #         topic_embeddings = torch.load("/home/thuy0050/code/IncDSI/bertopic/sbert_old_topic_embeddings_91.pt")
        #         self.pool_size = 91
        #     else:
        #         topic_embeddings = torch.load("/home/thuy0050/code/IncDSI/bertopic/bert_old_topic_embeddings_80.pt")
        #         self.pool_size = 80
        # elif "msmarco" in base_data_dir:
        #     if "sbert" in model_encoder:
        #         topic_embeddings = torch.load(
        #             "/home/thuy0050/code/IncDSI/bertopic/msmarco_sbert_old_topic_embeddings_193.pt"
        #         )
        #         self.pool_size = 193
        #     else:
        #         topic_embeddings = torch.load(
        #             "/home/thuy0050/code/IncDSI/bertopic/msmarco_bert_old_topic_embeddings_182.pt"
        #         )
        #         self.pool_size = 182

        self.prompt_key = nn.Parameter(torch.from_numpy(topic_embeddings))
        self.prompt_allocation = prompt_allocation
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.contrastive_loss = contrastive_loss

        assert embed_dim % self.num_heads == 0
        prompt_pool_shape = (
            self.num_layers,
            2,
            self.pool_size,
            self.length,
            self.num_heads,
            embed_dim // self.num_heads,
        )
        if prompt_init == "zero":
            self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
        elif prompt_init == "uniform":
            self.prompt = nn.Parameter(
                torch.randn(prompt_pool_shape)
            )  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
            nn.init.uniform_(self.prompt, -1, 1)

    def forward(
        self,
        f=0,
        task_id=None,
        cls_features=None,
        train=False,
        previous_task_key_centroids=None,
        correct_assignment=False,
    ):
        out = dict()
        x_embed_mean = cls_features

        # s = self.prompt_allocation * (task_id - 1)  # Start from task 1
        # f = self.prompt_allocation * task_id
        # # s = self.prompt_allocation * task_id  # With training on task 0
        # # f = self.prompt_allocation * (task_id + 1)
        # if train or (correct_assignment and task_id > 0):
        #     prompt_pool = self.prompt[:, :, s:f]
        #     prompt_key = self.prompt_key[s:f]
        # else:
        prompt_pool = self.prompt
        prompt_key = self.prompt_key

        prompt_key_norm = self.l2_normalize(prompt_key, dim=-1)  # Pool_size, C
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1)  # B, C

        similarity = torch.matmul(
            prompt_key_norm, x_embed_norm.t()
        )  # pool_size, B or Pool_size, #class, B
        similarity = similarity.t()  # B, pool_size

        if len(similarity.shape) == 1:
            similarity = similarity.unsqueeze(1)

        (similarity_top_k, idx) = torch.topk(
            similarity, k=self.top_k, dim=1
        )  # B, top_k
        out["similarity"] = similarity_top_k
        out["idx"] = idx

        batched_prompt_raw = prompt_pool[:, :, idx]  # num_layers, B, top_k, length, C
        (
            num_layers,
            dual,
            batch_size,
            top_k,
            length,
            num_heads,
            heads_embed_dim,
        ) = batched_prompt_raw.shape
        batched_prompt = batched_prompt_raw.reshape(
            num_layers,
            batch_size,
            dual,
            top_k * length,
            num_heads,
            heads_embed_dim,
        )
        batched_key_norm = prompt_key_norm[idx]  # B, top_k, C
        x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
        # sim = batched_key_norm * x_embed_norm  # B, top_k, C
        # reduce_sim = torch.sum(sim) / x_embed_norm.shape[0]  # Scalar

        # out["reduce_sim"] = reduce_sim
        out["batched_prompt"] = batched_prompt

        return out


# class EPromptWithTopicModellingShared(Prompt):  # Cosine similarity
#     def __init__(
#         self,
#         length=5,
#         embed_dim=768,
#         embedding_key="mean",
#         prompt_init="uniform",
#         prompt_pool=False,
#         prompt_key=False,
#         pool_size=None,
#         top_k=None,
#         batchwise_prompt=False,
#         prompt_key_init="uniform",
#         num_layers=1,
#         num_heads=12,
#         prompt_allocation=10,
#         contrastive_loss=False,
#         model_encoder=None,
#         base_data_dir=None,
#         **kwargs,
#     ):
#         super().__init__(
#             length=length,
#             embed_dim=embed_dim,
#             embedding_key=embedding_key,
#             prompt_init=prompt_init,
#             prompt_pool=prompt_pool,
#             prompt_key=prompt_key,
#             pool_size=pool_size,
#             top_k=top_k,
#             batchwise_prompt=batchwise_prompt,
#             prompt_key_init=prompt_key_init,
#             **kwargs,
#         )
#         if "nq320k" in base_data_dir:
#             if "sbert" in model_encoder:
#                 topic_embeddings = torch.load(
#                     "/home/thuy0050/code/IncDSI/bertopic/sbert_old_topic_embeddings_91.pt"
#                 )
#                 self.pool_size = 91 + 1
#             else:
#                 topic_embeddings = torch.load(
#                     "/home/thuy0050/code/IncDSI/bertopic/bert_old_topic_embeddings_80.pt"
#                 )
#                 self.pool_size = 80 + 1
#         elif "msmarco" in base_data_dir:
#             if "sbert" in model_encoder:
#                 topic_embeddings = torch.load(
#                     "/home/thuy0050/code/IncDSI/bertopic/msmarco_sbert_old_topic_embeddings_193.pt"
#                 )
#                 self.pool_size = 193
#             else:
#                 topic_embeddings = torch.load(
#                     "/home/thuy0050/code/IncDSI/bertopic/msmarco_bert_old_topic_embeddings_182.pt"
#                 )
#                 self.pool_size = 182

#         self.prompt_key = nn.Parameter(torch.from_numpy(topic_embeddings))
#         self.prompt_allocation = prompt_allocation
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.contrastive_loss = contrastive_loss

#         assert embed_dim % self.num_heads == 0
#         prompt_pool_shape = (
#             self.num_layers,
#             2,
#             self.pool_size,
#             # self.length,
#             5,
#             self.num_heads,
#             embed_dim // self.num_heads,
#         )
#         if prompt_init == "zero":
#             self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
#         elif prompt_init == "uniform":
#             self.prompt = nn.Parameter(
#                 torch.randn(prompt_pool_shape)
#             )  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
#             nn.init.uniform_(self.prompt, -1, 1)

#     def forward(
#         self,
#         f=0,
#         task_id=None,
#         cls_features=None,
#         train=False,
#         previous_task_key_centroids=None,
#         correct_assignment=False,
#     ):
#         out = dict()
#         x_embed_mean = cls_features

#         prompt_pool = self.prompt
#         prompt_key = self.prompt_key

#         prompt_key_norm = self.l2_normalize(prompt_key, dim=-1)  # Pool_size, C
#         x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1)  # B, C

#         similarity = torch.matmul(
#             prompt_key_norm, x_embed_norm.t()
#         )  # pool_size, B or Pool_size, #class, B
#         similarity = similarity.t()  # B, pool_size

#         if len(similarity.shape) == 1:
#             similarity = similarity.unsqueeze(1)

#         (similarity_top_k, idx) = torch.topk(
#             similarity, k=self.top_k, dim=1
#         )  # B, top_k
#         out["similarity"] = similarity_top_k
#         out["idx"] = idx

#         batched_prompt_raw = prompt_pool[:, :, idx]  # num_layers, B, top_k, length, C
#         shared_prompt = prompt_pool[:,:,-1]
#         # shared_prompt = shared_prompt.unsqueeze(2).repeat(1, 1, batched_prompt_raw.shape[2], 1,1,1).unsqueeze(3)

#         shared_prompt = shared_prompt.unsqueeze(2)
#         shared_prompt = shared_prompt.expand(-1, -1, batched_prompt_raw.shape[2], -1, -1, -1).unsqueeze(3)
#         batched_prompt_raw = torch.cat([batched_prompt_raw, shared_prompt], dim=4)
#         (
#             num_layers,
#             dual,
#             batch_size,
#             top_k,
#             length,
#             num_heads,
#             heads_embed_dim,
#         ) = batched_prompt_raw.shape
#         batched_prompt = batched_prompt_raw.reshape(
#             num_layers,
#             batch_size,
#             dual,
#             top_k * length,
#             num_heads,
#             heads_embed_dim,
#         )
#         batched_key_norm = prompt_key_norm[idx]  # B, top_k, C
#         x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
#         # sim = batched_key_norm * x_embed_norm  # B, top_k, C
#         # reduce_sim = torch.sum(sim) / x_embed_norm.shape[0]  # Scalar

#         # out["reduce_sim"] = reduce_sim
#         out["batched_prompt"] = batched_prompt

#         return out


# class EPromptWithTopicModellingUpdateKeys(Prompt):  # Cosine similarity
#     def __init__(
#         self,
#         length=5,
#         embed_dim=768,
#         embedding_key="mean",
#         prompt_init="uniform",
#         prompt_pool=False,
#         prompt_key=False,
#         pool_size=None,
#         top_k=None,
#         batchwise_prompt=False,
#         prompt_key_init="uniform",
#         num_layers=1,
#         num_heads=12,
#         prompt_allocation=10,
#         contrastive_loss=False,
#         **kwargs,
#     ):
#         super().__init__(
#             length=length,
#             embed_dim=embed_dim,
#             embedding_key=embedding_key,
#             prompt_init=prompt_init,
#             prompt_pool=prompt_pool,
#             prompt_key=prompt_key,
#             pool_size=pool_size,
#             top_k=top_k,
#             batchwise_prompt=batchwise_prompt,
#             prompt_key_init=prompt_key_init,
#             **kwargs,
#         )
#         topic_embeddings = torch.load(
#             "/home/thuy0050/code/IncDSI/bertopic/old_topic_embeddings_natural_queries.pt"
#         )
#         self.prompt_key = nn.Parameter(torch.from_numpy(topic_embeddings))
#         self.prompt_allocation = prompt_allocation
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.contrastive_loss = contrastive_loss

#         assert embed_dim % self.num_heads == 0
#         prompt_pool_shape = (
#             self.num_layers,
#             2,
#             91, #self.pool_size,
#             self.length,
#             self.num_heads,
#             embed_dim // self.num_heads,
#         )
#         if prompt_init == "zero":
#             self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
#         elif prompt_init == "uniform":
#             self.prompt = nn.Parameter(
#                 torch.randn(prompt_pool_shape)
#             )  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
#             nn.init.uniform_(self.prompt, -1, 1)

#     def forward(
#         self,
#         f=0,
#         task_id=None,
#         cls_features=None,
#         train=False,
#         previous_task_key_centroids=None,
#         correct_assignment=False,
#     ):
#         out = dict()
#         x_embed_mean = cls_features

#         # s = self.prompt_allocation * (task_id - 1)  # Start from task 1
#         # f = self.prompt_allocation * task_id
#         # # s = self.prompt_allocation * task_id  # With training on task 0
#         # # f = self.prompt_allocation * (task_id + 1)
#         # if train or (correct_assignment and task_id > 0):
#         #     prompt_pool = self.prompt[:, :, s:f]
#         #     prompt_key = self.prompt_key[s:f]
#         # else:
#         prompt_pool = self.prompt
#         prompt_key = self.prompt_key

#         with torch.no_grad():
#             prompt_key_norm = self.l2_normalize(prompt_key, dim=-1)  # Pool_size, C
#             x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1)  # B, C

#             similarity = torch.matmul(
#                 prompt_key_norm, x_embed_norm.t()
#             )  # pool_size, B or Pool_size, #class, B
#             similarity = similarity.t()  # B, pool_size

#         if len(similarity.shape) == 1:
#             similarity = similarity.unsqueeze(1)

#         (similarity_top_k, idx) = torch.topk(
#             similarity, k=self.top_k, dim=1
#         )  # B, top_k
#         out["similarity"] = similarity_top_k
#         out["idx"] = idx
#         # # Key Contrastive loss
#         # if train and self.contrastive_loss and task_id > 1:
#         #     selected_prompts, _ = torch.unique(idx, return_counts=True)
#         #     pos_keys = prompt_key_norm[selected_prompts]
#         #     pos = torch.exp(1.0 - torch.matmul(pos_keys, pos_keys.T)).mean()

#         #     if previous_task_key_centroids is None:
#         #         neg_keys = torch.cat([self.prompt_key[:s], self.prompt_key[f:]], dim=0)
#         #     else:
#         #         neg_keys = torch.cat(
#         #             [previous_task_key_centroids, self.prompt_key[f:]], dim=0
#         #         )
#         #     neg_keys = self.l2_normalize(neg_keys, dim=-1)
#         #     neg = torch.exp(1.0 - torch.matmul(neg_keys, pos_keys.T)).mean()
#         #     key_contrastive_loss = -torch.log(
#         #         (neg / (pos + neg)) + 1e-12
#         #     )  # Add epsilon to avoid division by zero
#         #     out["key_contrastive_loss"] = key_contrastive_loss
#         # else:
#         #     out["key_contrastive_loss"] = 0

#         batched_prompt_raw = prompt_pool[:, :, idx]  # num_layers, B, top_k, length, C
#         (
#             num_layers,
#             dual,
#             batch_size,
#             top_k,
#             length,
#             num_heads,
#             heads_embed_dim,
#         ) = batched_prompt_raw.shape
#         batched_prompt = batched_prompt_raw.reshape(
#             num_layers,
#             batch_size,
#             dual,
#             top_k * length,
#             num_heads,
#             heads_embed_dim,
#         )
#         batched_key_norm = prompt_key_norm[idx]  # B, top_k, C
#         x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
#         sim = batched_key_norm * x_embed_norm  # B, top_k, C
#         reduce_sim = torch.sum(sim) / x_embed_norm.shape[0]  # Scalar

#         out["reduce_sim"] = reduce_sim
#         out["batched_prompt"] = batched_prompt

#         return out


class EPrompt(Prompt):  # Cosine similarity
    def __init__(
        self,
        length=5,
        embed_dim=768,
        embedding_key="mean",
        prompt_init="uniform",
        prompt_pool=False,
        prompt_key=False,
        pool_size=None,
        top_k=None,
        batchwise_prompt=False,
        prompt_key_init="uniform",
        num_layers=1,
        num_heads=12,
        prompt_allocation=10,
        contrastive_loss=False,
        **kwargs,
    ):
        super().__init__(
            length=length,
            embed_dim=embed_dim,
            embedding_key=embedding_key,
            prompt_init=prompt_init,
            prompt_pool=prompt_pool,
            prompt_key=prompt_key,
            pool_size=pool_size,
            top_k=top_k,
            batchwise_prompt=batchwise_prompt,
            prompt_key_init=prompt_key_init,
            **kwargs,
        )
        self.prompt_allocation = prompt_allocation
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.contrastive_loss = contrastive_loss

        assert embed_dim % self.num_heads == 0
        prompt_pool_shape = (
            self.num_layers,
            2,
            self.pool_size,
            self.length,
            self.num_heads,
            embed_dim // self.num_heads,
        )
        if prompt_init == "zero":
            self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
        elif prompt_init == "uniform":
            self.prompt = nn.Parameter(
                torch.randn(prompt_pool_shape)
            )  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
            nn.init.uniform_(self.prompt, -1, 1)

    def forward(
        self,
        f=0,
        task_id=None,
        cls_features=None,
        train=False,
        previous_task_key_centroids=None,
        correct_assignment=False,
    ):
        out = dict()
        x_embed_mean = cls_features

        s = self.prompt_allocation * (task_id - 1)  # Start from task 1
        f = self.prompt_allocation * task_id
        # s = self.prompt_allocation * task_id  # With training on task 0
        # f = self.prompt_allocation * (task_id + 1)
        # print("s: ", s, "f: ", f)
        if train or (correct_assignment and task_id > 0):
            prompt_pool = self.prompt[:, :, s:f]
            prompt_key = self.prompt_key[s:f]
        else:
            prompt_pool = self.prompt
            prompt_key = self.prompt_key

        prompt_key_norm = self.l2_normalize(prompt_key, dim=-1)  # Pool_size, C
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1)  # B, C

        # dot product
        # prompt_key_norm = prompt_key
        # x_embed_norm = x_embed_mean
        
        similarity = torch.matmul(
            prompt_key_norm, x_embed_norm.t()
        )  # pool_size, B or Pool_size, #class, B
        similarity = similarity.t()  # B, pool_size
        if len(similarity.shape) == 1:
            similarity = similarity.unsqueeze(1)

        (similarity_top_k, idx) = torch.topk(
            similarity, k=self.top_k, dim=1
        )  # B, top_k
        out["similarity"] = similarity_top_k
        out["idx"] = idx
        # # Key Contrastive loss
        # if train and self.contrastive_loss and task_id > 1:
        #     selected_prompts, _ = torch.unique(idx, return_counts=True)
        #     pos_keys = prompt_key_norm[selected_prompts]
        #     pos = torch.exp(1.0 - torch.matmul(pos_keys, pos_keys.T)).mean()

        #     if previous_task_key_centroids is None:
        #         neg_keys = torch.cat([self.prompt_key[:s], self.prompt_key[f:]], dim=0)
        #     else:
        #         neg_keys = torch.cat(
        #             [previous_task_key_centroids, self.prompt_key[f:]], dim=0
        #         )
        #     neg_keys = self.l2_normalize(neg_keys, dim=-1)
        #     neg = torch.exp(1.0 - torch.matmul(neg_keys, pos_keys.T)).mean()
        #     key_contrastive_loss = -torch.log(
        #         (neg / (pos + neg)) + 1e-12
        #     )  # Add epsilon to avoid division by zero
        #     out["key_contrastive_loss"] = key_contrastive_loss
        # else:
        #     out["key_contrastive_loss"] = 0

        batched_prompt_raw = prompt_pool[:, :, idx]  # num_layers, B, top_k, length, C
        (
            num_layers,
            dual,
            batch_size,
            top_k,
            length,
            num_heads,
            heads_embed_dim,
        ) = batched_prompt_raw.shape
        batched_prompt = batched_prompt_raw.reshape(
            num_layers,
            batch_size,
            dual,
            top_k * length,
            num_heads,
            heads_embed_dim,
        )
        batched_key_norm = prompt_key_norm[idx]  # B, top_k, C
        x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
        sim = batched_key_norm * x_embed_norm  # B, top_k, C
        reduce_sim = torch.sum(sim) / x_embed_norm.shape[0]  # Scalar

        out["reduce_sim"] = reduce_sim
        out["batched_prompt"] = batched_prompt

        return out


class L2PEPrompt(Prompt):  # Cosine similarity
    def __init__(
        self,
        length=5,
        embed_dim=768,
        embedding_key="mean",
        prompt_init="uniform",
        prompt_pool=False,
        prompt_key=False,
        pool_size=None,
        top_k=None,
        batchwise_prompt=False,
        prompt_key_init="uniform",
        num_layers=1,
        num_heads=12,
        prompt_allocation=10,
        contrastive_loss=False,
        **kwargs,
    ):
        super().__init__(
            length=length,
            embed_dim=embed_dim,
            embedding_key=embedding_key,
            prompt_init=prompt_init,
            prompt_pool=prompt_pool,
            prompt_key=prompt_key,
            pool_size=pool_size,
            top_k=top_k,
            batchwise_prompt=batchwise_prompt,
            prompt_key_init=prompt_key_init,
            **kwargs,
        )
        self.prompt_allocation = prompt_allocation
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.contrastive_loss = contrastive_loss

        assert embed_dim % self.num_heads == 0
        prompt_pool_shape = (
            self.num_layers,
            2,
            self.pool_size,
            self.length,
            self.num_heads,
            embed_dim // self.num_heads,
        )
        if prompt_init == "zero":
            self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
        elif prompt_init == "uniform":
            self.prompt = nn.Parameter(
                torch.randn(prompt_pool_shape)
            )  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
            nn.init.uniform_(self.prompt, -1, 1)

    def forward(
        self,
        f=0,
        task_id=None,
        cls_features=None,
        train=False,
        previous_task_key_centroids=None,
        correct_assignment=True,
    ):
        out = dict()
        x_embed_mean = cls_features

        prompt_pool = self.prompt
        prompt_key = self.prompt_key

        prompt_key_norm = self.l2_normalize(prompt_key, dim=-1)  # Pool_size, C
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1)  # B, C

        similarity = torch.matmul(
            prompt_key_norm, x_embed_norm.t()
        )  # pool_size, B or Pool_size, #class, B
        similarity = similarity.t()  # B, pool_size
        if len(similarity.shape) == 1:
            similarity = similarity.unsqueeze(1)

        (similarity_top_k, idx) = torch.topk(
            similarity, k=self.top_k, dim=1
        )  # B, top_k
        out["similarity"] = similarity_top_k
        out["idx"] = idx

        # # Key Contrastive loss
        # if train and self.contrastive_loss and task_id > 1:
        #     selected_prompts, _ = torch.unique(idx, return_counts=True)
        #     pos_keys = prompt_key_norm[selected_prompts]
        #     pos = torch.exp(1.0 - torch.matmul(pos_keys, pos_keys.T)).mean()

        #     if previous_task_key_centroids is None:
        #         neg_keys = torch.cat([self.prompt_key[:s], self.prompt_key[f:]], dim=0)
        #     else:
        #         neg_keys = torch.cat(
        #             [previous_task_key_centroids, self.prompt_key[f:]], dim=0
        #         )
        #     neg_keys = self.l2_normalize(neg_keys, dim=-1)
        #     neg = torch.exp(1.0 - torch.matmul(neg_keys, pos_keys.T)).mean()
        #     key_contrastive_loss = -torch.log(
        #         (neg / (pos + neg)) + 1e-12
        #     )  # Add epsilon to avoid division by zero
        #     out["key_contrastive_loss"] = key_contrastive_loss
        # else:
        #     out["key_contrastive_loss"] = 0

        batched_prompt_raw = prompt_pool[:, :, idx]  # num_layers, B, top_k, length, C
        (
            num_layers,
            dual,
            batch_size,
            top_k,
            length,
            num_heads,
            heads_embed_dim,
        ) = batched_prompt_raw.shape
        batched_prompt = batched_prompt_raw.reshape(
            num_layers,
            batch_size,
            dual,
            top_k * length,
            num_heads,
            heads_embed_dim,
        )
        batched_key_norm = prompt_key_norm[idx]  # B, top_k, C
        x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
        sim = batched_key_norm * x_embed_norm  # B, top_k, C
        reduce_sim = torch.sum(sim) / x_embed_norm.shape[0]  # Scalar

        out["reduce_sim"] = reduce_sim
        out["batched_prompt"] = batched_prompt

        return out


class EPromptEnsemble(Prompt):  # Cosine similarity
    def __init__(
        self,
        length=5,
        embed_dim=768,
        embedding_key="mean",
        prompt_init="uniform",
        prompt_pool=False,
        prompt_key=False,
        pool_size=None,
        top_k=None,
        batchwise_prompt=False,
        prompt_key_init="uniform",
        num_layers=1,
        num_heads=12,
        prompt_allocation=10,
        contrastive_loss=False,
        **kwargs,
    ):
        super().__init__(
            length=length,
            embed_dim=embed_dim,
            embedding_key=embedding_key,
            prompt_init=prompt_init,
            prompt_pool=prompt_pool,
            prompt_key=prompt_key,
            pool_size=pool_size,
            top_k=top_k,
            batchwise_prompt=batchwise_prompt,
            prompt_key_init=prompt_key_init,
            **kwargs,
        )
        self.prompt_allocation = prompt_allocation
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.contrastive_loss = contrastive_loss

        assert embed_dim % self.num_heads == 0
        prompt_pool_shape = (
            self.num_layers,
            2,
            self.pool_size,
            self.length,
            self.num_heads,
            embed_dim // self.num_heads,
        )
        if prompt_init == "zero":
            self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
        elif prompt_init == "uniform":
            self.prompt = nn.Parameter(
                torch.randn(prompt_pool_shape)
            )  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
            nn.init.uniform_(self.prompt, -1, 1)

    def forward(
        self,
        f=0,
        task_id=None,
        cls_features=None,
        train=False,
        previous_task_key_centroids=None,
    ):
        out = dict()
        x_embed_mean = cls_features

        s = self.prompt_allocation * (task_id - 1)  # Start from task 1
        f = self.prompt_allocation * task_id

        if train:
            prompt_pool = self.prompt[:, :, s:f]
            prompt_key = self.prompt_key[s:f]
        else:
            prompt_pool = self.prompt
            prompt_key = self.prompt_key

        prompt_key_norm = self.l2_normalize(prompt_key, dim=-1)  # Pool_size, C
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1)  # B, C

        similarity = torch.matmul(
            prompt_key_norm, x_embed_norm.t()
        )  # pool_size, B or Pool_size, #class, B
        similarity = similarity.t()  # B, pool_size
        if len(similarity.shape) == 1:
            similarity = similarity.unsqueeze(1)

        (similarity_top_k, idx) = torch.topk(
            similarity, k=self.top_k, dim=1
        )  # B, top_k
        out["similarity"] = similarity_top_k
        out["idx"] = idx

        # Contrastive loss
        if train and self.contrastive_loss and task_id > 1:
            selected_prompts, _ = torch.unique(idx, return_counts=True)
            pos_keys = prompt_key_norm[selected_prompts]
            pos = torch.exp(1.0 - torch.matmul(pos_keys, pos_keys.T)).mean()

            if previous_task_key_centroids is None:
                neg_keys = torch.cat([self.prompt_key[:s], self.prompt_key[f:]], dim=0)
            else:
                neg_keys = torch.cat(
                    [previous_task_key_centroids, self.prompt_key[f:]], dim=0
                )
            neg_keys = self.l2_normalize(neg_keys, dim=-1)
            neg = torch.exp(1.0 - torch.matmul(neg_keys, pos_keys.T)).mean()
            key_contrastive_loss = -torch.log(
                (neg / (pos + neg)) + 1e-12
            )  # Add epsilon to avoid division by zero
            out["key_contrastive_loss"] = key_contrastive_loss
        else:
            out["key_contrastive_loss"] = 0
            
        if train and f > 0 and task_id > 1:
            prompt_momentum = 0.1
            with torch.no_grad():
                batched_prompt_momentum = (
                    self.prompt[:, :, :f]
                    .detach()
                    .clone()
                    .mean(2, keepdim=True)
                    .unsqueeze(2)
                    .repeat(1, 1, idx.shape[0], 1, 1, 1, 1)
                )
            batched_prompt_raw = (1 - prompt_momentum) * prompt_pool[
                :, :, idx
            ] + prompt_momentum * batched_prompt_momentum
        else:
            batched_prompt_raw = prompt_pool[:, :, idx]  # num_layers, B, top_k, length, C
        (
            num_layers,
            dual,
            batch_size,
            top_k,
            length,
            num_heads,
            heads_embed_dim,
        ) = batched_prompt_raw.shape
        batched_prompt = batched_prompt_raw.reshape(
            num_layers,
            batch_size,
            dual,
            top_k * length,
            num_heads,
            heads_embed_dim,
        )
        batched_key_norm = prompt_key_norm[idx]  # B, top_k, C
        x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
        sim = batched_key_norm * x_embed_norm  # B, top_k, C
        reduce_sim = torch.sum(sim) / x_embed_norm.shape[0]  # Scalar

        out["reduce_sim"] = reduce_sim
        out["batched_prompt"] = batched_prompt

        return out
