import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import copy
import numpy as np
import math

from eval import segment_bars_with_confidence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)

class AttentionHelper(nn.Module):
    def __init__(self):
        super(AttentionHelper, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        """
        Scalar dot attention (single-head-style) with robust causal masking.
        proj_query: (B, C, Lq)
        proj_key  : (B, C, Lk)
        proj_val  : (B, C, Lk)
        padding_mask (float/bool): broadcastable to (B, Lq, Lk); 1 for valid, 0 for masked
        return: out (B, C, Lq), attention (B, Lq, Lk)
        """
        m, c1, lq = proj_query.shape
        _, c2, lk = proj_key.shape

        assert c1 == c2, "query/key channel dims must match"

        # (B, Lq, Lk)
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key) / math.sqrt(c1)

        # Convert mask to float in case it's bool; avoid log(0)
        mask_f = padding_mask.to(energy.dtype)
        # Add log-mask => 0 -> large negative; keeps gradients stable
        attention = energy + torch.log(mask_f + 1e-9)
        attention = self.softmax(attention)
        # Hard zero masked positions after softmax (double safety)
        attention = attention * mask_f

        # (B, C, Lq) = (B, C, Lk) @ (B, Lk, Lq)
        out = torch.bmm(proj_val, attention.permute(0, 2, 1))
        return out, attention


class AttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, causal=True):  # r1 = r2
        super(AttLayer, self).__init__()

        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv   = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)
        self.conv_out   = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)

        self.bl = bl
        self.stage = stage
        self.att_type = att_type
        self.causal = causal

        assert self.att_type in ['normal_att', 'block_att', 'sliding_att']
        assert self.stage in ['encoder','decoder']

        self.att_helper = AttentionHelper()
        self.window_mask = self.construct_window_mask(causal=self.causal)

    def _make_causal_mask(self, base_key_mask):
        """
        base_key_mask: (B, 1, L) with 1 for valid frames, 0 for padding
        returns: (B, L, L) lower-triangular & padded mask for full-sequence attention
        """
        B, _, L = base_key_mask.shape
        tri = torch.tril(torch.ones(L, L, device=base_key_mask.device)).unsqueeze(0).expand(B, -1, -1)  # (B,L,L)
        key_valid = base_key_mask.repeat(1, L, 1)  # (B,L,L): broadcast valid keys along query axis
        return tri * key_valid

    def construct_window_mask(self, causal=True):
        """
        Construct sliding-window mask of shape (1, bl, bl + 2*(bl//2)).
        If causal=True, for each within-block query index i, allow only:
          - all left-half keys (previous-block overlap)
          - and up to i (inclusive) inside the current block
        => disallow any right-half keys and any 'future within the block'
        """
        L = self.bl
        h = L // 2
        wm = torch.zeros((1, L, L + 2 * h), device=device)
        if causal:
            for i in range(L):
                # allow [0 .. h+i] => left-half (0..h-1) + first i+1 keys of the current block
                wm[:, i, : h + i + 1] = 1
        else:
            # Bidirectional window (for reference): allow the whole window for each row
            wm[:] = 1
        return wm

    def forward(self, x1, x2, mask):
        # x1 from encoder (queries/keys), x2 from decoder (values) if stage == 'decoder'
        q = self.query_conv(x1)
        k = self.key_conv(x1)

        if self.stage == "decoder":
            assert x2 is not None
            v = self.value_conv(x2)
        else:
            v = self.value_conv(x1)

        if self.att_type == "normal_att":
            return self._normal_self_att(q, k, v, mask)
        elif self.att_type == "block_att":
            return self._block_wise_self_att(q, k, v, mask)
        elif self.att_type == "sliding_att":
            return self._sliding_window_self_att(q, k, v, mask)

    def _normal_self_att(self, q, k, v, mask):
        B, c1, L = q.size()
        base_key_mask = torch.ones((B, 1, L), device=device) * mask[:, 0:1, :]
        final_mask = self._make_causal_mask(base_key_mask)  # (B, L, L)

        out, _ = self.att_helper.scalar_dot_att(q, k, v, final_mask)
        out = self.conv_out(F.relu(out))
        out = out[:, :, 0:L]
        return out * mask[:, 0:1, :]

    def _block_wise_self_att(self, q, k, v, mask):
        B, c1, L = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()

        nb = L // self.bl
        if L % self.bl != 0:
            pad = self.bl - L % self.bl
            q = torch.cat([q, torch.zeros((B, c1, pad), device=device)], dim=-1)
            k = torch.cat([k, torch.zeros((B, c2, pad), device=device)], dim=-1)
            v = torch.cat([v, torch.zeros((B, c3, pad), device=device)], dim=-1)
            nb += 1

        base = torch.cat(
            [torch.ones((B, 1, L), device=device) * mask[:, 0:1, :],
             torch.zeros((B, 1, self.bl * nb - L), device=device)],
            dim=-1
        )  # (B,1,nb*bl)

        # reshape to blocks
        q = q.reshape(B, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(B * nb, c1, self.bl)
        k = k.reshape(B, c2, nb, self.bl).permute(0, 2, 1, 3).reshape(B * nb, c2, self.bl)
        v = v.reshape(B, c3, nb, self.bl).permute(0, 2, 1, 3).reshape(B * nb, c3, self.bl)

        base_blk = base.reshape(B, 1, nb, self.bl).permute(0, 2, 1, 3).reshape(B * nb, 1, self.bl)  # (B*nb,1,bl)
        tri = torch.tril(torch.ones(self.bl, self.bl, device=device)).unsqueeze(0).expand(B * nb, -1, -1)
        final_mask = tri * base_blk.repeat(1, self.bl, 1)  # (B*nb, bl, bl)

        out, _ = self.att_helper.scalar_dot_att(q, k, v, final_mask)
        out = self.conv_out(F.relu(out))

        out = out.reshape(B, c3, nb, self.bl).reshape(B, c3, nb * self.bl)
        out = out[:, :, 0:L]
        return out * mask[:, 0:1, :]

    def _sliding_window_self_att(self, q, k, v, mask):
        B, c1, L = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()

        assert B == 1, "sliding_att currently supports batch size 1"

        nb = L // self.bl
        if L % self.bl != 0:
            pad = self.bl - L % self.bl
            q = torch.cat([q, torch.zeros((B, c1, pad), device=device)], dim=-1)
            k = torch.cat([k, torch.zeros((B, c2, pad), device=device)], dim=-1)
            v = torch.cat([v, torch.zeros((B, c3, pad), device=device)], dim=-1)
            nb += 1
        base = torch.cat(
            [torch.ones((B, 1, L), device=device) * mask[:, 0:1, :],
            torch.zeros((B, 1, self.bl * nb - L), device=device)],
            dim=-1
        )  # (B,1,nb*bl)

        # query reshape -> (B*nb, c1, bl)
        q = q.reshape(B, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(B * nb, c1, self.bl)

        # key/value: add half-window paddings left/right
        half = self.bl // 2
        k = torch.cat([torch.zeros(B, c2, half, device=device), k, torch.zeros(B, c2, half, device=device)], dim=-1)
        v = torch.cat([torch.zeros(B, c3, half, device=device), v, torch.zeros(B, c3, half, device=device)], dim=-1)
        base = torch.cat([torch.zeros(B, 1, half, device=device), base, torch.zeros(B, 1, half, device=device)], dim=-1)

        # compose window slices for each block: (B*nb, C, bl+2*half)
        k = torch.cat([k[:, :, i * self.bl:(i + 1) * self.bl + 2 * half] for i in range(nb)], dim=0)
        v = torch.cat([v[:, :, i * self.bl:(i + 1) * self.bl + 2 * half] for i in range(nb)], dim=0)
        base = torch.cat([base[:, :, i * self.bl:(i + 1) * self.bl + 2 * half] for i in range(nb)], dim=0)  # (B*nb,1,W)

        # ---- CAUSAL mask inside the window ----
        # rows: 0..bl-1 (query idx within block), cols: 0..(bl+2*half-1) (windowed key idx)
        W = self.bl + 2 * half
        rows = torch.arange(self.bl, device=device).unsqueeze(1)           # (bl,1)
        cols = torch.arange(W, device=device).unsqueeze(0)                  # (1,W)
        # allow keys up to half + row (inclusive): left overlap + current position
        causal_wmask = (cols <= (half + rows)).float().unsqueeze(0)         # (1, bl, W)
        final_mask = causal_wmask.repeat(B * nb, 1, 1) * base               # (B*nb, bl, W)

        out, _ = self.att_helper.scalar_dot_att(q, k, v, final_mask)
        out = self.conv_out(F.relu(out))

        out = out.reshape(B, -1, nb, self.bl).reshape(B, -1, nb * self.bl)
        out = out[:, :, 0:L]
        return out * mask[:, 0:1, :]


class MultiHeadAttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, num_head):
        super(MultiHeadAttLayer, self).__init__()
        self.conv_out = nn.Conv1d(v_dim * num_head, v_dim, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(AttLayer(q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, causal=True))
             for _ in range(num_head)]
        )
        self.dropout = nn.Dropout(p=0.5)

        # NOTE: Enforce causal for all heads

    def forward(self, x1, x2, mask):
        out = torch.cat([layer(x1, x2, mask) for layer in self.layers], dim=1)
        out = self.conv_out(self.dropout(out))
        return out


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=0, dilation=dilation)

    def forward(self, x):  # x: (B, C, L)
        x = F.pad(x, (self.left_pad, 0))  # (left, right)
        return self.conv(x)


class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size=3, dilation=dilation),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class FCFeedForward(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),  # conv1d equals fc
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(out_channels, out_channels, 1),
        )

    def forward(self, x):
        return self.layer(x)


class ChannelLayerNorm1d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):  # x: (B, C, L)
        # LayerNorm은 마지막 축 기준이므로 (B, L, C)로 바꿔 적용 후 되돌림
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha):
        super(AttModule, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = ChannelLayerNorm1d(in_channels)
        # CAUSAL enforced (surgical videos: do not look ahead)
        self.att_layer = AttLayer(
            in_channels,
            in_channels,
            out_channels,
            r1,
            r1,
            r2,
            dilation,  # use dilation as window/block length
            att_type=att_type,
            stage=stage,
            causal=True,
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha

    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0, 2, 1)  # (1, d_model, L)
        self.pe = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return x + self.pe[:, :, 0 : x.shape[2]]


class Encoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, alpha):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)  # fc layer
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, "encoder", alpha) for i in range(num_layers)]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, None, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]
        return out, feature


class Decoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, att_type, alpha):
        super(Decoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, "decoder", alpha) for i in range(num_layers)]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fencoder, mask):
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)
        feature = F.relu(feature)
        out = self.conv_out(feature) * mask[:, 0:1, :]
        return out, feature


class MyTransformer(nn.Module):
    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate):
        super(MyTransformer, self).__init__()
        self.encoder = Encoder(
            num_layers,
            r1,
            r2,
            num_f_maps,
            input_dim,
            num_classes,
            channel_masking_rate,
            att_type="sliding_att",  # uses CAUSAL sliding window
            alpha=1.0,
        )
        self.decoders = nn.ModuleList(
            [
                copy.deepcopy(
                    Decoder(
                        num_layers,
                        r1,
                        r2,
                        num_f_maps,
                        num_classes,
                        num_classes,
                        att_type="sliding_att",  # uses CAUSAL sliding window
                        alpha=exponential_descrease(s),
                    )
                )
                for s in range(num_decoders)
            ]
        )

    def forward(self, x, mask):
        out, feature = self.encoder(x, mask)
        outputs = out.unsqueeze(0)
        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], feature * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs, feature


class Trainer:
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate):
        self.model = MyTransformer(3, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

        print("Model Size: ", sum(p.numel() for p in self.model.parameters()))
        self.mse = nn.MSELoss(reduction="none")
        self.num_classes = num_classes

        self.activations=[]
        self.contributions=[]

    @torch.no_grad()
    def _check_shapes(self, feat_out, logits, mask):
        assert feat_out.dim() == 3, f"feat_out shape expected (B,64,T), got {feat_out.shape}"
        assert logits.dim() == 3,  f"logits shape expected (B,C,T), got {logits.shape}"
        if mask is not None:
            assert mask.dim() == 3 and mask.size(1) == 1, f"mask shape expected (B,1,T), got {mask.shape}"
        assert feat_out.size(0) == logits.size(0) and feat_out.size(2) == logits.size(2), \
            "B/T mismatch between feat_out and logits"

    def compute_taylor_scores(
        self,
        feat_out: torch.Tensor,         # (B,64,T), requires grad 대상
        logits: torch.Tensor,           # (B,num_classes,T) - 마지막 stage 로짓
        num_classes: int,
        mask: torch.Tensor | None = None,   # (B,1,T) or None
        mode: str = "abs",              # "abs" | "relu" | "raw"
        retain_graph: bool = True,
    ) -> torch.Tensor:
        """
        반환: (T, 64, num_classes) = activation × (∂y_c/∂activation)
        """
        self._check_shapes(feat_out, logits, mask)
        B, Fdim, T = feat_out.shape
        assert B == 1, "현재 구현은 B=1을 가정합니다."

        # 필요 시 마스크를 로짓에 반영
        if mask is not None:
            logits = logits * mask  # (B,C,T)

        # per-class grad -> activation×grad
        per_class = []
        for c in range(num_classes):
            g_c = torch.autograd.grad(
                outputs=logits[:, c, :].sum(),  # 스칼라
                inputs=feat_out,                 # (B,64,T)
                retain_graph=retain_graph,
                create_graph=False,
                allow_unused=False,
            )[0]                                 # (B,64,T)

            contrib_c = (feat_out.detach() * g_c).squeeze(0).transpose(0, 1)  # (T,64)

            if mode == "abs":
                contrib_c = contrib_c.abs()
            elif mode == "relu":
                contrib_c = F.relu(contrib_c)
            # mode == "raw" 그대로 유지

            per_class.append(contrib_c)         # (T,64)

        return torch.stack(per_class, dim=-1)   # (T,64,C)
     
    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, batch_gen_tst=None):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        print("LR:{}".format(learning_rate))

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            # frame-level (micro) accumulators
            correct_frames = 0.0
            total_frames = 0.0

            # video-level (macro) accumulators
            sum_video_acc = 0.0
            num_videos = 0

            while batch_gen.has_next():
                batch_input, batch_target, mask, vids = batch_gen.next_batch(batch_size, False)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                ps, _ = self.model(batch_input, mask)

                loss = 0.0
                for p in ps:
                    # CE loss
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    # temporal smoothing loss (causal neighbor)
                    loss += 0.15 * torch.mean(
                        torch.clamp(
                            self.mse(
                                F.log_softmax(p[:, :, 1:], dim=1),
                                F.log_softmax(p.detach()[:, :, :-1], dim=1),
                            ),
                            min=0,
                            max=16,
                        )
                        * mask[:, :, 1:]
                    )

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                # last stage prediction
                _, predicted = torch.max(ps.data[-1], 1)  # (B, T)

                # ----- frame-level (micro) -----
                correct_frames += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total_frames += torch.sum(mask[:, 0, :]).item()

                # ----- video-level (macro): per-sample frame mean accuracy -----
                B = batch_input.size(0)
                for b in range(B):
                    valid = mask[b, 0, :].bool()
                    if valid.sum() == 0:
                        continue
                    acc_b = (predicted[b, valid] == batch_target[b, valid]).float().mean().item()
                    sum_video_acc += acc_b
                    num_videos += 1

            scheduler.step(epoch_loss)
            batch_gen.reset()

            frame_acc = float(correct_frames) / max(1.0, total_frames)
            video_acc = float(sum_video_acc) / max(1, num_videos)

            print(
                "[epoch %d]: epoch loss = %.6f,   frame_acc = %.6f,   video_acc = %.6f"
                % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples), frame_acc, video_acc)
            )

            if (epoch + 1) % 1 == 0 and batch_gen_tst is not None:
                self.test(batch_gen_tst, epoch)
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")

    def test(self, batch_gen_tst, epoch):
        self.model.eval()

        correct_frames = 0.0
        total_frames = 0.0
        sum_video_acc = 0.0
        num_videos = 0

        if_warp = False
        with torch.no_grad():
            while batch_gen_tst.has_next():
                batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1, if_warp)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                p, _ = self.model(batch_input, mask)
                _, predicted = torch.max(p.data[-1], 1)  # (B, T)

                # frame-level (micro)
                correct_frames += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total_frames   += torch.sum(mask[:, 0, :]).item()

                # video-level (macro)
                B = batch_input.size(0)
                for b in range(B):
                    valid = mask[b, 0, :].bool()
                    if valid.sum() == 0:
                        continue
                    acc_b = (predicted[b, valid] == batch_target[b, valid]).float().mean().item()
                    sum_video_acc += acc_b
                    num_videos += 1

        frame_acc = float(correct_frames) / max(1.0, total_frames)
        video_acc = float(sum_video_acc)   / max(1,   num_videos)

        print("---[epoch %d]---: tst frame_acc = %.6f,   tst video_acc = %.6f"
              % (epoch + 1, frame_acc, video_acc))

        self.model.train()
        batch_gen_tst.reset()

    def predict(self, model_dir, results_dir, features_path, batch_gen_tst, epoch, actions_dict, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))

            batch_gen_tst.reset()
            import time
            time_start = time.time()

            while batch_gen_tst.has_next():
                batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1)
                vid = vids[0]
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]

                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                # Mask all valid (1s); ensure shape is (B,1,T)
                pred_mask = torch.ones((input_x.size(0), 1, input_x.size(2)), device=device)

                predictions, _ = self.model(input_x, pred_mask)

                for i in range(len(predictions)):
                    confidence, predicted = torch.max(F.softmax(predictions[i], dim=1).data, 1)
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()

                    batch_target = batch_target.squeeze()
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()

                    segment_bars_with_confidence(results_dir + '/{}_stage{}.png'.format(vid, i),
                                                 confidence.tolist(),
                                                 batch_target.tolist(), predicted.tolist())

                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate(
                        (
                            recognition,
                            [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]
                            * sample_rate,
                        )
                    )
                f_name = vid.split("/")[-1].split(".")[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()

            time_end = time.time()
            

    def extract_activations(self, model_dir, results_dir, features_path, batch_gen_tst, epoch, actions_dict, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))

            batch_gen_tst.reset()

            while batch_gen_tst.has_next():
                batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1)
                vid = vids[0]
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]

                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions, features = self.model(input_x, torch.ones(input_x.size(), device=device))
                
                self.activations.append(features)
                    
            torch.save(self.activations, "ASFormer_train_activations.pkl")
            with open(f"ASFormer_train_activations.pkl", "wb") as f:
                pickle.dump(self.activations, f)


    # ------------------ 기존 함수: 호출부만 수정 ------------------
    def extract_contributions(self, model_dir, results_dir, features_path,
                              batch_gen_tst, epoch, actions_dict, sample_rate):
        """
        Per-class contribution: for each time step t and feature j,
        contrib[t, j, c] = feat_out[t, j] * d logits_c / d feat_out[t, j]
        결과 shape: (T, 64, num_classes)
        """
        self.model.eval()
        state = torch.load(f"{model_dir}/epoch-{epoch}.model", map_location=device)
        self.model.load_state_dict(state)
        self.model.to(device)

        self.activations = []
        self.contributions = []
        epoch_loss = 0.0

        batch_gen_tst.reset()

        while batch_gen_tst.has_next():
            # --------- data ---------
            batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1)
            batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
            vid = vids[0]

            feats_np = np.load(features_path + vid.split(".")[0] + ".npy")
            feats_np = feats_np[:, ::sample_rate]
            x = torch.tensor(feats_np, dtype=torch.float, device=device).unsqueeze(0)  # (1,C,T)

            # target: (B,T) long
            target = torch.as_tensor(batch_target, device=device).long()
            if target.dim() == 3 and target.size(1) == 1:
                target = target[:, 0, :]
            elif target.dim() == 1:
                target = target.unsqueeze(0)

            # mask: (B,1,T) float
            m = torch.as_tensor(mask, device=device)
            if m.dim() == 2:
                m = m.unsqueeze(1)
            elif m.dim() == 3 and m.size(1) != 1:
                m = m[:, :1, :]
            m = m.float()

            # --------- forward ---------
            B, C, T = x.shape
            model_mask = torch.ones((B, 1, T), device=device, dtype=x.dtype)
            predictions, feat_out = self.model(x, model_mask)  # feat_out: (B,64,T)

            # predictions를 stage list로 통일
            if isinstance(predictions, torch.Tensor):
                if predictions.dim() == 4:  # (S,B,C,T)
                    S = predictions.size(0)
                    stage_list = [predictions[s] for s in range(S)]  # 각 s: (B,C,T)
                elif predictions.dim() == 3:  # (B,C,T)
                    stage_list = [predictions]
                else:
                    raise RuntimeError(f"Unexpected predictions shape: {predictions.shape}")
            else:
                stage_list = list(predictions)

            # --------- per-class Taylor score 계산 ---------
            last_logits = stage_list[-1]  # (B,num_classes,T)

            predicted_classes = last_logits.argmax(dim=1).squeeze()
            gt_classes = target[0]

            contribution = self.compute_taylor_scores(
                feat_out=feat_out,
                logits=last_logits,
                num_classes=self.num_classes,
                mask=m,
                mode="abs",
                retain_graph=True,
            )  # (T,64,num_classes)

            # --------- 저장 리스트에 추가 ---------
            self.contributions.append([
                contribution.detach().cpu(),
                predicted_classes,
                gt_classes
            ])

        # --------- 저장 ---------
        with open("ASFormer_test_contributions.pkl", "wb") as f:
            pickle.dump(self.contributions, f)
            
if __name__ == '__main__':
    pass
