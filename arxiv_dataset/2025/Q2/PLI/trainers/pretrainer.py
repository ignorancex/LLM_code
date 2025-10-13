from trainers.abc import AbstractBaseTrainer
from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils.metrics import AverageMeterSet
import clip
from clip.model import CLIP


def encode_with_pseudo_tokens(clip_model, text: torch.Tensor, pseudo_tokens: torch.Tensor,
                              num_tokens=1) -> torch.Tensor:
    """
    Use the CLIP model to encode a text with pseudo tokens.
    It replaces the word embedding of $ with the pseudo tokens for each element in the batch.
    Based on the original implementation of the CLIP model:
    https://github.com/openai/CLIP/blob/main/clip/model.py
    """
    x = clip_model.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]

    _, counts = torch.unique((text == 259).nonzero(as_tuple=True)[0], return_counts=True)  # 259 is the token of $
    cum_sum = torch.cat((torch.zeros(1, device=text.device).int(), torch.cumsum(counts, dim=0)[:-1]))
    first_tokens_indexes = (text == 259).nonzero()[cum_sum][:, 1]
    rep_idx = torch.cat([(first_tokens_indexes + n).unsqueeze(0) for n in range(num_tokens)])

    if pseudo_tokens.shape[0] == x.shape[0]:
        if len(pseudo_tokens.shape) == 2:
            pseudo_tokens = pseudo_tokens.unsqueeze(1)
        x[torch.arange(x.shape[0]).repeat_interleave(num_tokens).reshape(
            x.shape[0], num_tokens), rep_idx.T] = pseudo_tokens.to(x.dtype)
    else:
        first_tokens_indexes = (text == 259).nonzero()[torch.arange(0, x.shape[0] * num_tokens, num_tokens)][:, 1]
        rep_idx = torch.cat([(first_tokens_indexes + n).unsqueeze(0) for n in range(num_tokens)])
        x[torch.arange(x.shape[0]).repeat_interleave(num_tokens).reshape(
            x.shape[0], num_tokens), rep_idx.T] = pseudo_tokens.repeat(x.shape[0], 1, 1).to(x.dtype)

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x

class Pretrainer(AbstractBaseTrainer):
    def __init__(self, models: dict, val_loggers, evaluators, pretrained_dataloaders, criterions, optimizers, lr_schedulers,
                 num_epochs, train_loggers, train_evaluator, start_epoch=0, *args, **kwargs):
        super().__init__(models, val_loggers, evaluators, pretrained_dataloaders, criterions, optimizers, lr_schedulers,
                 num_epochs, train_loggers, train_evaluator, start_epoch)
        self.image_encoder = self.models['image_encoder']
        self.text_encoder = self.models['text_encoder']
        self.compositor = self.models['compositor']
        self.metric_loss = self.criterions['metric_loss']
        self.compositor_code = kwargs['compositor'] if 'compositor' in kwargs else "None"
        print("Compositor: ", self.compositor_code)
        # self.is_mask = kwargs['is_mask'] if 'is_mask' in kwargs else False
        self.is_mask = True
        self.mask_ratio = kwargs['mask_ratio'] if 'mask_ratio' in kwargs else 0.5
        print("Pretrainer mask ratio: ", self.mask_ratio)
        # which CLIP encoder to fine-tune, should be in ['both', 'text', 'image']
        if "tuning_module" in kwargs and kwargs['tuning_module'] in ['both', 'text', 'image', 'freeze']:
            if kwargs['tuning_module'] == 'text':
                print('Only the text encoder will be fine-tuned')
                for param in self.image_encoder.parameters():
                    param.requires_grad = False
            elif kwargs['tuning_module'] == 'image':
                print('Only the image encoder will be fine-tuned')
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
            elif kwargs['tuning_module'] == 'freeze':
                print('Both image and text encoders will be frozen')
                for param in self.image_encoder.parameters():
                    param.requires_grad = False
                for param in self.text_encoder.parameters():
                    param.requires_grad = False

        # show trainable parameters
        print("Trainable parameters: ")
        for key, value in models.items():
            print(key, sum(p.numel() for p in value.parameters() if p.requires_grad))

    def mask_forword(self, image, text, mask_ratio=0.5):

        # mask reference image
        ref_img_features = self.image_encoder(image, mask_ratio=mask_ratio) # (batch_size, proj_size)

        if self.compositor_code != 'phi':
            text_features = self.text_encoder(text) # (batch_size, proj_size)
        tar_features = self.image_encoder(image, proj=True)  # (batch_size, proj_size)

        if self.compositor_code == 'Combiner':
            composed_features = self.compositor.combine_features(ref_img_features, text_features)
            # tar_features = F.normalize(tar_features, dim=-1)
        elif self.compositor_code == 'phi':
            extimated_tokens = self.compositor(ref_img_features)
            prompt = "a photo of $"  # The $ is a special token that will be replaced with the pseudo tokens
            tokenized_prompt = clip.tokenize([prompt] * len(image)).to(self.device)
            composed_features = encode_with_pseudo_tokens(self.text_encoder, tokenized_prompt, extimated_tokens)
        else:
            composed_embeds = ref_img_features + text_features # uni-modal features has been normalized
            composed_features = self.compositor(composed_embeds)

        # contrastive loss
        # with torch.no_grad():
        #     temp = self.text_encoder.temp.clamp_(0.001, 0.5)
        loss = self.metric_loss( # contrastive loss without normalization
            # F.normalize(composed_features, dim=-1),
            # F.normalize(tar_features, dim=-1)
            composed_features,
            tar_features,
            # tau=temp
        )
        return loss
    def compositor_forward(self, images, captions=None, epoch=None):
        # Encode Target Images
        image_embeds = self.image_encoder(images, proj=False)  # (batch_size, n, hidden_size)
        tar_features = image_embeds[:, 0, :]  # [cls] token, (batch_size, hidden_size)
        # Encode and Fuse Reference Images with Texts
        # TODO: caption is None can be a prompt
        if captions is None:
            prompt = ["a photo of"] * len(images)
            captions = self.text_encoder.tokenizer(prompt, return_tensors="pt", padding='max_length',
                                                   max_length=10, truncation=True).input_ids.to(self.device)

        composed_ref_features = self.text_encoder(captions, image_embeds)  # (batch_size, hidden_size)
        # project to the same features space by shared compositor
        composed_ref_features = self.compositor(composed_ref_features)
        tar_features = self.compositor(tar_features)
        loss = self.metric_loss(composed_ref_features, tar_features)
        return loss
    def train_one_epoch(self, epoch) -> dict:
        average_meter_set = AverageMeterSet()
        train_dataloader = tqdm(self.train_dataloader, desc="Epoch {}/{}".format(epoch + 1, self.num_epochs))
        for batch_idx, batch in enumerate(train_dataloader):
            images = batch['image'].to(self.device)
            captions = batch.get('caption', None).to(self.device)

            self._reset_grad()
            if self.is_mask:
                self._to_train_mode()
                loss = self.mask_forword(images, captions, self.mask_ratio)
            else:
                loss = self.compositor_forward(images, captions, epoch)
            loss.backward()
            average_meter_set.update('loss', loss.item())
            self._update_grad()

        train_results = average_meter_set.averages()
        optimizers_dict = self._get_state_dicts(self.optimizers)
        for key in optimizers_dict.keys():
            train_results[key + '_lr'] = optimizers_dict[key]["param_groups"][0]["lr"]
        self._step_schedulers()
        return train_results

    def compute_loss(self, composed_ref_features, tar_features, epoch):
        loss = self.metric_loss(composed_ref_features, tar_features)
        return loss

    @classmethod
    def code(cls) -> str:
        return 'pretrainer'