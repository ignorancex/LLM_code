import clip
import torch
from clip import tokenize
import torch.nn as nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class  ClipTextEncoderPipeline(nn.Module):
    def __init__(self, clip_model_name='RN50.pt'):
        super().__init__()
        clip_model, _ = clip.load("./models/clip/" + clip_model_name, device=device)
        # TODO: Trainer not accept mix precision, RuntimeError: expected scalar type Float but found Half
        clip_model.float()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.token_embedding = clip_model.token_embedding
        self.dtype = clip_model.dtype


    def forward(self, text, attn_mask=None):
        text = tokenize(text).to(device)
        # text transform is in dataset getitem, so text: (batch_size, 1, n_ctx)
        # text = text.squeeze() # pipe line accept text from dataset, no batch
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]


        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x