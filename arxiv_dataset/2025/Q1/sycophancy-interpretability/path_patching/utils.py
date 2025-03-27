#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (cy424151@alibaba-inc.com)
Date               : 2024-10-15 14:48
Last Modified By   : 陈蔚 (cy424151@alibaba-inc.com)
Last Modified Date : 2024-10-15 17:39
Description        : 
-------- 
Copyright (c) 2024 Alibaba Inc. 
'''

from typing import List, Dict

import einops
import plotly.express as px
import torch
from transformers import PreTrainedModel


def show_path_patching_results(
    m,
    xlabel="",
    ylabel="",
    title="",
    bartitle="",
    animate_axis=None,
    highlight_points=None,
    highlight_name="",
    return_fig=False,
    show_fig=True,
    **kwargs,
):
    """
    Plot a heatmap of the values in the matrix `m`
    """

    if animate_axis is None:
        fig = px.imshow(
            m,
            title=title if title else "",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            **kwargs,
        )

    else:
        fig = px.imshow(
            einops.rearrange(m, "a b c -> a c b"),
            title=title if title else "",
            animation_frame=animate_axis,
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            **kwargs,
        )

    fig.update_layout(
        coloraxis_colorbar=dict(
            title=bartitle,
            thicknessmode="pixels",
            thickness=50,
            lenmode="pixels",
            len=300,
            yanchor="top",
            y=1,
            ticks="outside",
        ),
    )

    if highlight_points is not None:
        fig.add_scatter(
            x=highlight_points[1],
            y=highlight_points[0],
            mode="markers",
            marker=dict(color="green", size=10, opacity=0.5),
            name=highlight_name,
        )

    fig.update_layout(
        yaxis_title=ylabel,
        xaxis_title=xlabel,
        xaxis_range=[-0.5, m.shape[1] - 0.5],
        showlegend=True,
        legend=dict(x=-0.1),
    )
    if highlight_points is not None:
        fig.update_yaxes(range=[m.shape[0] - 0.5, -0.5], autorange=False)
    if show_fig:
        fig.show()
    if return_fig:
        return fig


def create_batch(sliced_data: List[Dict], split: str = "xr_toks", pad_token_id: int = None):

    tokens = [item[split] for item in sliced_data]
    max_length = max([len(item) for item in tokens])
    
    tokens_tensor = torch.LongTensor([[pad_token_id] * (max_length - len(item)) + item for item in tokens])
    mask_tensor = torch.LongTensor([[0] * (max_length - len(item)) + [1] * len(item) for item in tokens])

    return tokens_tensor, mask_tensor


@torch.no_grad()
def compute_metric(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_token_ids: torch.Tensor,
    record_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute the metric used to measure the path patching effect.

    Parameters
    ----------
    model : PreTrainedModel
        Model to compute the metric.
    input_ids : torch.Tensor
        Input ids.
    attention_mask : torch.Tensor
        Attention mask.
    target_token_ids : torch.Tensor
        Target token ids.
    record_token_ids : torch.Tensor
        Record token ids, used as regularization.

    Returns
    -------
    torch.Tensor
        Metric value.
    """
    
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits.detach()
    
    batch_size = logits.size(0)

    logits_target = logits[torch.arange(batch_size), -1, target_token_ids]
    logits_sum = torch.zeros_like(logits_target).to(logits_target.device)
    
    for i in range(batch_size):
        logits_sum[i] = logits[i, -1, record_token_ids[i]].sum()

    return logits_target / logits_sum


