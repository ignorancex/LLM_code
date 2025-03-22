import torch
import torch.nn as nn

from copy import deepcopy
from .BaseModel import BaseIncModel


class EPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False,
                 dtype=torch.float16):
        super().__init__()

        self.embed_dim = embed_dim
        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value
        self.dtype = dtype

        if self.prompt_pool:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length,
                                         self.num_heads, embed_dim // self.num_heads)

                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.pool_size, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(
                            prompt_pool_shape))  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
            else:
                prompt_pool_shape = (self.num_layers, self.pool_size, self.length, embed_dim)
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)

        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=[0, 2])
            self.prompt_key = prompt_mean

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_key_norm = self.l2_normalize(self.prompt_key, dim=-1)  # Pool_size, C
            prompt_key_norm = prompt_key_norm.to(x_embed.device)
            prompt_key_norm = prompt_key_norm.to(torch.float16)
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1)  # B, C

            similarity = torch.matmul(prompt_key_norm, x_embed_norm.t())  # pool_size, B or Pool_size, #class, B
            similarity = similarity.t()  # B, pool_size

            (similarity_top_k, idx) = torch.topk(similarity, k=self.top_k, dim=1)  # B, top_k
            out['similarity'] = similarity

            if self.batchwise_prompt:
                prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                # Unless dimension is specified, this will be flattend if it is not already 1D.
                if prompt_id.shape[0] < self.pool_size:
                    prompt_id = torch.cat([prompt_id,
                                           torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()),
                                                      device=prompt_id.device)])
                    id_counts = torch.cat(
                        [id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                _, major_idx = torch.topk(id_counts, k=self.top_k)  # top_k
                major_prompt_id = prompt_id[major_idx]  # top_k
                # expand to batch
                idx = major_prompt_id.expand(x_embed.shape[0], -1).contiguous()  # B, top_k

            if prompt_mask is not None:
                idx = prompt_mask  # B, top_k

            out['prompt_idx'] = idx
            if self.use_prefix_tune_for_e_prompt:
                batched_prompt_raw = self.prompt[:, :, idx]  # num_layers, B, top_k, length, C
                num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                )
            else:
                batched_prompt_raw = self.prompt[:, idx]
                num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers, batch_size, top_k * length, embed_dim
                )

            batched_key_norm = prompt_key_norm[idx]  # B, top_k, C

            out['selected_key'] = batched_key_norm
            out['prompt_key_norm'] = prompt_key_norm
            out['x_embed_norm'] = x_embed_norm

            # Put pull_constraint loss calculation inside
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
            sim = batched_key_norm * x_embed_norm  # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0]  # Scalar

            out['reduce_sim'] = reduce_sim
        else:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert self.embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.length,
                                         self.num_heads, self.embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.length,
                                         self.num_heads, self.embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(
                            torch.randn(prompt_pool_shape))  # num_layers, 2, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1, -1)
            else:
                prompt_pool_shape = (self.num_layers, self.length, self.embed_dim)
                if self.prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1)

        out['batched_prompt'] = batched_prompt

        return out


class vision_transformer(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.blocks = transformer.resblocks

    def forward(self, x: torch.Tensor, *args):
        return self.blocks(x)


class PromptBlock(nn.Module):
    def __init__(self, block):
        super(PromptBlock, self).__init__()
        self.d_model = block.attn.embed_dim
        self.num_heads = block.attn.num_heads

        self.qkv = nn.Linear(self.d_model, self.d_model * 3)
        self.qkv.weight.data = block.attn.in_proj_weight.data.clone()
        self.qkv.bias.data = block.attn.in_proj_bias.data.clone()

        self.out_proj = block.attn.out_proj

        self.ln_1 = block.ln_1
        self.mlp = block.mlp
        self.ln_2 = block.ln_2
        self.attn_mask = block.attn_mask
        if isinstance(self.d_model, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            self.head_dim = self.d_model.div(self.num_heads, rounding_mode='trunc')
        else:
            self.head_dim = self.d_model // self.num_heads
        self.scale = self.head_dim ** -0.5

    def attention(self, x: torch.Tensor, prompt):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if prompt is not None:
            # prefix key, value
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous().to(
                torch.float16)  # 2, B, num_heads, prompt_length, C // num_heads
            key_prefix = prompt[0]  # B, num_heads, prompt_length, embed_dim // num_heads
            value_prefix = prompt[1]  # B, num_heads, prompt_length, embed_dim // num_heads

            expected_shape = (B, self.num_heads, C // self.num_heads)

            assert (key_prefix.shape[0], key_prefix.shape[1], key_prefix.shape[
                3]) == expected_shape, f'key_prefix.shape: {key_prefix.shape} not match k.shape: {k.shape}'
            assert (value_prefix.shape[0], value_prefix.shape[1], value_prefix.shape[
                3]) == expected_shape, f'value_prefix.shape: {value_prefix.shape} not match v.shape: {v.shape}'

            k = torch.cat([key_prefix, k], dim=2)
            v = torch.cat([value_prefix, v], dim=2)

        # q_scaled = q / math.sqrt(C)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        return x

    def forward(self, x: torch.Tensor, prompt):
        x = x + self.attention(self.ln_1(x), prompt)
        x = x + self.mlp(self.ln_2(x))
        return x


class prompt_transformer(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        layers = transformer.layers
        self.blocks = nn.Sequential(*[PromptBlock(transformer.resblocks[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor, prompt):
        return self.blocks(x, prompt)


class CustomCLIP(BaseIncModel):
    def __init__(self, classes_names, clip_model):
        super().__init__(classes_names, clip_model)

        self.config = Configs

        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        embed_dim = clip_model.visual.transformer.width
        self.class_token = True

        self.positional_embedding = clip_model.visual.positional_embedding
        # self.transformer = deepcopy(clip_model.visual.transformer)
        self.ln_pre = clip_model.visual.ln_pre
        self.ln_post = clip_model.visual.ln_post
        self.proj = clip_model.visual.proj
        self.head = nn.Linear(embed_dim, len(classes_names)).half()
        self.use_prompt_mask = self.config.use_prompt_mask
        self.head_type = self.config.head_type
        self.global_pool = self.config.global_pool
        self.prompt_pool = self.config.prompt_pool
        num_heads = clip_model.visual.transformer.resblocks[0].attn.num_heads

        self.use_g_prompt = self.config.use_g_prompt
        self.g_prompt_length = self.config.g_prompt_length
        self.g_prompt_layer_idx = self.config.g_prompt_layer_idx
        # num_g_prompt : The actual number of layers to which g-prompt is attached.
        # In official code, create as many layers as the total number of layers and select them based on the index
        num_g_prompt = len(self.g_prompt_layer_idx) if self.g_prompt_layer_idx is not None else 0
        self.use_prefix_tune_for_g_prompt = self.config.use_prefix_tune_for_g_prompt

        self.use_e_prompt = self.config.use_e_prompt
        self.e_prompt_layer_idx = self.config.e_prompt_layer_idx
        num_e_prompt = len(self.e_prompt_layer_idx) if self.e_prompt_layer_idx is not None else 0
        self.use_prefix_tune_for_e_prompt = self.config.use_prefix_tune_for_e_prompt

        if not self.use_prefix_tune_for_g_prompt and not self.use_prefix_tune_for_g_prompt:
            self.use_g_prompt = False
            self.g_prompt_layer_idx = []

        if self.use_g_prompt and self.g_prompt_length is not None and len(self.g_prompt_layer_idx) != 0:
            if not self.use_prefix_tune_for_g_prompt:
                g_prompt_shape = (num_g_prompt, self.g_prompt_length, embed_dim)
                if self.config.prompt_init == 'zero':
                    self.g_prompt = nn.Parameter(torch.zeros(g_prompt_shape))
                elif self.config.prompt_init == 'uniform':
                    self.g_prompt = nn.Parameter(torch.randn(g_prompt_shape))
                    nn.init.uniform_(self.g_prompt, -1, 1)
            else:
                if self.config.same_key_value:
                    g_prompt_shape = (num_g_prompt, 1, self.g_prompt_length, num_heads, embed_dim // num_heads)
                    if self.config.prompt_init == 'zero':
                        self.g_prompt = nn.Parameter(torch.zeros(g_prompt_shape))
                    elif self.config.prompt_init == 'uniform':
                        self.g_prompt = nn.Parameter(torch.randn(g_prompt_shape))
                        nn.init.uniform_(self.g_prompt, -1, 1)
                    self.g_prompt = self.g_prompt.repeat(1, 2, 1, 1, 1)
                else:
                    g_prompt_shape = (num_g_prompt, 2, self.g_prompt_length, num_heads, embed_dim // num_heads)
                    if self.config.prompt_init == 'zero':
                        self.g_prompt = nn.Parameter(torch.zeros(g_prompt_shape))
                    elif self.config.prompt_init == 'uniform':
                        self.g_prompt = nn.Parameter(torch.randn(g_prompt_shape))
                        nn.init.uniform_(self.g_prompt, -1, 1)
        else:
            self.g_prompt = None

        if not (self.use_g_prompt or self.use_e_prompt):
            self.transformer = vision_transformer(deepcopy(clip_model.visual.transformer))
        elif not (self.use_prefix_tune_for_g_prompt or self.use_prefix_tune_for_e_prompt):
            # Prompt tunning
            self.transformer = vision_transformer(deepcopy(clip_model.visual.transformer))
        else:
            # Prefix tunning
            self.transformer = prompt_transformer(deepcopy(clip_model.visual.transformer))

        self.total_prompt_len = 0
        if self.prompt_pool:
            if not self.use_prefix_tune_for_g_prompt:
                self.total_prompt_len += self.config.g_prompt_length * len(self.g_prompt_layer_idx)
            if not self.use_prefix_tune_for_e_prompt:
                self.total_prompt_len += self.config.prompt_length * self.config.top_k * len(self.e_prompt_layer_idx)

        if self.config.use_e_prompt and self.config.e_prompt_layer_idx is not None:
            self.e_prompt = EPrompt(length=self.config.prompt_length, embed_dim=embed_dim,
                                    embedding_key=self.config.embedding_key,
                                    prompt_init=self.config.prompt_init, prompt_pool=self.config.prompt_pool,
                                    prompt_key=self.config.prompt_key,
                                    pool_size=self.config.pool_size, top_k=self.config.top_k,
                                    batchwise_prompt=self.config.batchwise_prompt,
                                    prompt_key_init=self.config.prompt_key_init, num_layers=num_e_prompt,
                                    use_prefix_tune_for_e_prompt=self.config.use_prefix_tune_for_e_prompt,
                                    num_heads=num_heads, same_key_value=self.config.same_key_value,
                                    dtype=self.dtype)

    def encode_image(self, images):
        return self.forward_features(images, -1, None, False)['x'][:, 0, :]

    def get_cls_features(self, images):
        x = self.conv1(images.type(self.dtype))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, prompt=None)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])
        return x

    def forward_features(self, x, task_id=-1, cls_features=None, train=False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        if self.use_g_prompt or self.use_e_prompt:
            if self.use_prompt_mask and train:
                start = task_id * self.e_prompt.top_k
                end = (task_id + 1) * self.e_prompt.top_k
                single_prompt_mask = torch.arange(start, end).to(x.device)
                prompt_mask = single_prompt_mask.unsqueeze(0).expand(x.shape[0], -1)
                if end > self.e_prompt.pool_size:
                    prompt_mask = None
            else:
                prompt_mask = None

            g_prompt_counter = -1
            e_prompt_counter = -1

            res = self.e_prompt(x, prompt_mask=prompt_mask, cls_features=cls_features)
            e_prompt = res['batched_prompt']

            for i, block in enumerate(self.transformer.blocks):
                if i in self.g_prompt_layer_idx:
                    if self.use_prefix_tune_for_g_prompt:
                        g_prompt_counter += 1
                        # Prefix tunning, [B, 2, g_prompt_length, num_heads, embed_dim // num_heads]
                        idx = torch.tensor([g_prompt_counter] * x.shape[0]).to(x.device)
                        g_prompt = self.g_prompt[idx]
                    else:
                        g_prompt = None
                    x = block(x, prompt=g_prompt)

                elif i in self.e_prompt_layer_idx:
                    e_prompt_counter += 1
                    if self.use_prefix_tune_for_e_prompt:
                        # Prefix tunning, [B, 2, top_k * e_prompt_length, num_heads, embed_dim // num_heads]
                        x = block(x, prompt=e_prompt[e_prompt_counter])
                    else:
                        # Pommpt tunning, [B, top_k * e_prompt_length, embed_dim]
                        prompt = e_prompt[e_prompt_counter]
                        x = torch.cat([prompt, x], dim=1)
                        x = block(x)
                else:
                    x = block(x, prompt=None)
        else:
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            res = dict()
        x = self.ln_post(x)
        res['x'] = x

        return res

    def forward_head(self, res, pre_logits: bool = False):
        x = res['x']
        if self.class_token and self.head_type == 'token':
            if self.prompt_pool:
                x = x[:, self.total_prompt_len]
            else:
                x = x[:, 0]
        elif self.head_type == 'gap' and self.global_pool == 'avg':
            x = x.mean(dim=1)
        elif self.head_type == 'prompt' and self.prompt_pool:
            x = x[:, 1:(1 + self.total_prompt_len)] if self.class_token else x[:, 0:self.total_prompt_len]
            x = x.mean(dim=1)
        elif self.head_type == 'token+prompt' and self.prompt_pool and self.class_token:
            x = x[:, 0:self.total_prompt_len + 1]
            x = x.mean(dim=1)
        else:
            raise ValueError(f'Invalid classifier={self.classifier}')

        res['pre_logits'] = x

        res['logits'] = self.head(x)

        return res

    def forward(self, x, task_id=-1, cls_features=None, train=False):
        res = self.forward_features(x, task_id=task_id, cls_features=cls_features, train=train)
        res = self.forward_head(res)
        return res


class OriginalCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.dtype = clip_model.dtype
        self.input_resolution = clip_model.visual.input_resolution
        self.output_dim = clip_model.visual.output_dim
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre

        self.transformer = clip_model.visual.transformer

        self.ln_post = clip_model.visual.ln_post

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        return x


class Configs:
    drop_rate = 0.0
    drop_path_rate = 0.0
    drop_block_rate = None
    global_pool = 'token'
    prompt_length = 5
    embedding_key = 'cls'
    prompt_init = 'uniform'
    prompt_pool = True
    prompt_key = True
    pool_size = 10
    top_k = 1
    batchwise_prompt = True
    prompt_key_init = 'uniform'
    head_type = 'token'
    use_prompt_mask = True
    use_g_prompt = True
    g_prompt_length = 5
    g_prompt_layer_idx = [0, 1]
    use_prefix_tune_for_g_prompt = True
    use_e_prompt = True
    e_prompt_layer_idx = [2, 3, 4]
    use_prefix_tune_for_e_prompt = True
    same_key_value = False
    shared_prompt_pool = True
    shared_prompt_key = False

