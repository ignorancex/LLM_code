#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from .gs_render import render_original_gs, render_gs_origin_wrapper
from .pvg_render import render_pvg, render_pvg_wrapper

EPS = 1e-5

rendererTypeCallbacks = {
    "gs": render_original_gs,
    "pvg": render_pvg
}

renderWrapperTypeCallbacks = {
    "gs": render_gs_origin_wrapper,
    "pvg": render_pvg_wrapper,
}


def get_renderer(render_type: str):
    return rendererTypeCallbacks[render_type], renderWrapperTypeCallbacks[render_type]
