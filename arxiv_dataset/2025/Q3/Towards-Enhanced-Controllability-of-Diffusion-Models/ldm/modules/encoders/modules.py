import torch
import torch.nn as nn
from functools import partial

from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test
from ldm.modules.diffusionmodules.openaimodel import StyleEncoderModel, ContentEncoderModel
# from ldm.modules.diffusionmodules.model import EqualLinear, EqualConvTranspose2d, PixelNorm, UpConvBlock, ConvBlock

# import segmentation_models_pytorch as smp



class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)



class StyleEncoder(nn.Module):
    def __init__(self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        use_time_condition=True,
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        # num_heads_upsample=-1, # 
        # use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
        *args,
        **kwargs):
        super().__init__()
        self.style_enc = StyleEncoderModel(
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            use_time_condition=use_time_condition,
            conv_resample=conv_resample,
            dims=dims,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            pool=pool,
        )

    def forward(self, x):
        return self.style_enc(x)

    def encode(self, x):
        return self(x)


class ContentEncoder(nn.Module):
    def __init__(self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        enc_channel_mult=(1, 2, 4, 8),
        dec_channel_mult=(1, 2, 4, 8),
        use_time_condition=True,
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        # num_heads_upsample=-1, # 
        # use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
        lowpass_filter=True,
        *args,
        **kwargs):
        super().__init__()
        self.content_enc = ContentEncoderModel(
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=dropout,
            enc_channel_mult=enc_channel_mult,
            dec_channel_mult=dec_channel_mult,
            use_time_condition=use_time_condition,
            conv_resample=conv_resample,
            dims=dims,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            pool=pool,
            lowpass_filter=lowpass_filter
        )

    def forward(self, x):
        return self.content_enc(x)

    def encode(self, x):
        return self(x)


# class DisentanglingEncoder(nn.Module):
#     def __init__(self,
#         in_channels=512,
#         n_mlp=8,
#         ):
#         super().__init__()
#         out_channels = in_channels
        
#         layers_style = []
#         for i in range(n_mlp):
#             layers_style.append(EqualLinear(in_channels, out_channels))
#             layers_style.append(nn.LeakyReLU(0.2))

#         self.style_enc = nn.Sequential(*layers_style)

#         layers_content = []
#         for i in range(n_mlp):
#             layers_content.append(EqualLinear(in_channels, out_channels))
#             layers_content.append(nn.LeakyReLU(0.2))

#         layers_content.append(nn.Unflatten(1, (out_channels, 1, 1)))
#         layers_content.append(EqualConvTranspose2d(in_channels, out_channels, 4, 1, 0)) # 512,1,1 => 512,4,4
#         layers_content.append(PixelNorm())
#         layers_content.append(nn.LeakyReLU(0.1))
#         layers_content.append(UpConvBlock(512, 256, 3, 1, upsample=True)) # 512,4,4 => 256,8,8
#         layers_content.append(UpConvBlock(256, 128, 3, 1, upsample=True)) # 256,8,8 => 128,16,16
#         layers_content.append(ConvBlock(128,64,3,1)) # 128,16,16 => 64,16,16
#         layers_content.append(ConvBlock(64,32,3,1)) # 64,16,16 => 32,16,16
#         layers_content.append(nn.Conv2d(32, 1, kernel_size=1)) # 32,16,16 => 1,16,16
#         layers_content.append(nn.Sigmoid())

#         self.content_enc = nn.Sequential(*layers_content)       
        

#     def forward(self, x):
#         return [self.style_enc(x), self.content_enc(x)]

#     def encode(self, x):
#         return self(x)


# class ContentEncoder(nn.Module):
#     def __init__(self,
#         encoder_name="resnet34",
#         encoder_weights=None,
#         in_channels=3,
#         classes=1,
#         ):
#         super().__init__()
#         self.content_enc = smp.UnetPlusPlus(        # https://github.com/qubvel/segmentation_models.pytorch#start
#             encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#             encoder_weights=encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
#             in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#             classes=classes,                      # model output channels (number of classes in your dataset)
#         )
#     def forward(self, x):
#         return self.content_enc(x)

#     def encode(self, x):
#         return self(x)