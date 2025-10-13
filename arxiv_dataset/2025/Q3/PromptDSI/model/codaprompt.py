import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    # if ortho:
    #     nn.init.orthogonal_(p)
    # else:
    #     nn.init.uniform_(p)
    return p  


class CodaPrompt(nn.Module):
    def __init__(
        self,
        length=8,
        pool_size=100,
        embed_dim=768,
        prompt_allocation=20,
        **kwargs,
    ):
        super().__init__()
        """
        prompt param: [pool size, prompt length, ortho mu]
        """
        self.task_count = 0
        self.num_head = 12
        self.s = -1
        self.f = -1
        self.key_d = embed_dim
        self.prompt_allocation = prompt_allocation
        self.pool_size = pool_size # prompt components
        self.length = length

        # e prompt init
        # for e in self.e_layers:
        self.prompt = tensor_prompt(self.pool_size, self.length, 768)
        self.prompt_key = tensor_prompt(self.pool_size, self.key_d)
        self.prompt_attention = tensor_prompt(self.pool_size, self.key_d)

        # self.prompt = self.gram_schmidt(self.prompt)
        # self.prompt_key = self.gram_schmidt(self.prompt_key)
        # self.prompt_attention = self.gram_schmidt(self.prompt_attention)
        # setattr(self, f"e_p_{e}", p)
        # setattr(self, f"e_k_{e}", k)
        # setattr(self, f"e_a_{e}", a)

    def _init_smart(self, prompt_param, e_layers=[0]):

        # prompt basic param
        self.pool_size = int(prompt_param[0])
        self.length = int(prompt_param[1])
        self.e_layers = e_layers # [0, 1, 2, 3, 4]

    def process_task_count(self):
        self.s = (self.task_count-1) * self.prompt_allocation # Start from task 1
        self.f = self.task_count * self.prompt_allocation

        # s = self.prompt_allocation * self.task_count  # With training on task 0
        # f = self.prompt_allocation * (self.task_count + 1)

        self.prompt = self.gram_schmidt(self.prompt)
        self.prompt_key = self.gram_schmidt(self.prompt_key)
        self.prompt_attention = self.gram_schmidt(self.prompt_attention)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = self.prompt_allocation
        s = int((self.task_count-1) * pt)
        f = int(self.task_count * pt)

        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)

        return torch.nn.Parameter(uu)

    def forward(
        self, 
        task_id=None,
        cls_features=None,
        train=False,
        previous_task_key_centroids=None,
    ):

        out = dict()
        x_embed_mean = cls_features

        s = self.s
        f = self.f 

        if train:
            prompt_pool = self.prompt[s:f]
            prompt_key = self.prompt_key[s:f]
            prompt_attention = self.prompt_attention[s:f]
        else:
            prompt_pool = self.prompt
            prompt_key = self.prompt_key
            prompt_attention = self.prompt_attention

        B, C = x_embed_mean.shape

        # with attention and cosine sim
        # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
        a_querry = torch.einsum("bd,kd->bkd", cls_features, prompt_attention)
        # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
        n_K = nn.functional.normalize(prompt_key, dim=1)
        q = nn.functional.normalize(a_querry, dim=2)

        aq_k = torch.einsum("bkd,kd->bk", q, n_K)
        # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
        prompt = torch.einsum("bk,kld->bld", aq_k, prompt_pool)

        prompt = prompt.reshape(
            prompt.shape[0],
            prompt.shape[1],
            self.num_head,
            -1
        )
        prompt = torch.stack(
            [prompt[:, : self.length // 2], prompt[:, self.length // 2 :]], dim=1 
        ).unsqueeze(0)

        out["batched_prompt"] = prompt

        return out


class CodaPromptNoAttention(nn.Module): # No attention
    def __init__(
        self,
        length=8,
        pool_size=100,
        embed_dim=768,
        prompt_allocation=20,
        **kwargs,
    ):
        super().__init__()
        """
        prompt param: [pool size, prompt length, ortho mu]
        """
        self.task_count = 0
        self.num_head = 12
        self.s = -1
        self.f = -1
        self.key_d = embed_dim
        self.prompt_allocation = prompt_allocation
        self.pool_size = pool_size  # prompt components
        self.length = length

        # e prompt init
        # for e in self.e_layers:
        self.prompt = tensor_prompt(self.pool_size, self.length, 768)
        self.prompt_key = tensor_prompt(self.pool_size, self.key_d)
        # self.prompt_attention = tensor_prompt(self.pool_size, self.key_d)

        # self.prompt = self.gram_schmidt(self.prompt)
        # self.prompt_key = self.gram_schmidt(self.prompt_key)
        # self.prompt_attention = self.gram_schmidt(self.prompt_attention)
        # setattr(self, f"e_p_{e}", p)
        # setattr(self, f"e_k_{e}", k)
        # setattr(self, f"e_a_{e}", a)

    def _init_smart(self, prompt_param, e_layers=[0]):

        # prompt basic param
        self.pool_size = int(prompt_param[0])
        self.length = int(prompt_param[1])
        self.e_layers = e_layers  # [0, 1, 2, 3, 4]

    def process_task_count(self):
        self.s = (self.task_count - 1) * self.prompt_allocation  # Start from task 1
        self.f = self.task_count * self.prompt_allocation

        # s = self.prompt_allocation * self.task_count  # With training on task 0
        # f = self.prompt_allocation * (self.task_count + 1)

        self.prompt = self.gram_schmidt(self.prompt)
        self.prompt_key = self.gram_schmidt(self.prompt_key)
        # self.prompt_attention = self.gram_schmidt(self.prompt_attention)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0], -1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = self.prompt_allocation
        s = int((self.task_count - 1) * pt)
        f = int(self.task_count * pt)

        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:, k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print("restarting!!!")
                        else:
                            uk = uk + proj
                if not redo:
                    uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)

        return torch.nn.Parameter(uu)

    def forward(
        self,
        task_id=None,
        cls_features=None,
        train=False,
        previous_task_key_centroids=None,
    ):

        out = dict()
        x_embed_mean = cls_features

        s = self.s
        f = self.f

        if train:
            prompt_pool = self.prompt[s:f]
            prompt_key = self.prompt_key[s:f]
            # prompt_attention = self.prompt_attention[s:f]
        else:
            prompt_pool = self.prompt
            prompt_key = self.prompt_key
            # prompt_attention = self.prompt_attention

        B, C = x_embed_mean.shape

        # with attention and cosine sim
        # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
        # a_querry = torch.einsum("bd,kd->bkd", cls_features, prompt_attention)
        a_querry = cls_features.unsqueeze(1).repeat(1, prompt_key.shape[0], 1)
        # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
        n_K = nn.functional.normalize(prompt_key, dim=1)
        q = nn.functional.normalize(a_querry, dim=2)

        aq_k = torch.einsum("bkd,kd->bk", q, n_K)
        # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
        prompt = torch.einsum("bk,kld->bld", aq_k, prompt_pool)

        prompt = prompt.reshape(prompt.shape[0], prompt.shape[1], self.num_head, -1)
        prompt = torch.stack(
            [prompt[:, : self.length // 2], prompt[:, self.length // 2 :]], dim=1
        ).unsqueeze(0)

        out["batched_prompt"] = prompt

        return out


def ortho_penalty(t):
    return ((t @ t.T - torch.eye(t.shape[0]).cuda()) ** 2).mean()
